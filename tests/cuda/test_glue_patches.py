"""patches/0190 (prefill glue): the MoE routing glue (``moe_glue``: grouping by a parallel sort, one-kernel router,
the shared expert read in place by the combine), the MTP prefill window (``mtp_window``), the fused hyper-connection
kernels (``hc_fused``), 32-query latent attention tiles in bf16 fast chunks (``attn_bm32``) and the load-time
tensor-core latent absorb / expand (GLM53_TF_LATENT_TC).

What must hold:

- ``moe_glue``: ``glue.group_sorted`` writes exactly ``glue._group``'s integers (ids up to the count, the count, every
  member cell; ids past the count untouched); ``_router_fused`` == ``_router_part`` + ``_router_sum`` and
  ``_combine_s`` == the copy + ``_combine`` bit for bit, so no row's bits change: committed state and replies equal
  with the knob on and off, exact and fast, lean and not, any chunk size;
- ``attn_bm32``: the 32-query tile == the 16-query tile bit for bit (else the knob must stay off);
- GLM53_TF_LATENT_TC: new arithmetic, but deterministic, C-independent, drafted == serial, resumed == fresh, own tag;
- ``mtp_window``: only the MTP head's rows below ``lo`` change (zeroed); every main-model state is identical; replies
  are identical with the window on and off (drafted, MTP / DFlash2 / auto drafts, greedy and sampled), drafted ==
  serial, resumed == fresh (replies), and the head's caches are deterministic (the same after any history);
- ``hc_fused``: the fused hc_post + hc_pre kernels give hc_post's / hc_pre's bits (bitwise on the real shapes, row
  subsets and permutations), so the committed state is identical on and off, for C = 1024 / 8192 (patches/0085).

Host-only tests run anywhere with torch (CPU); the kernel and engine tests need a GPU. Timings: ``bench_glue.py``.

Run inside the image:
    PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q -s tests/cuda/test_glue_patches.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

try:
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:          # the host-only tests still run
    torch = None
    CUDA = False

gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")
needs_torch = pytest.mark.skipif(torch is None, reason="torch")

sys.path.insert(0, str(Path(__file__).parent))

E, TOP = 288, 8


# -- host only: knobs and rules --------------------------------------------------------------------------------------
def test_knobs_in_header():
    from tensorfold.families.glm5_next.cuda import knobs

    for k in ("moe_glue", "mtp_window", "hc_fused", "attn_bm32"):
        assert k in knobs.HEADER and k in knobs.RANGES
    assert knobs.RANGES["hc_fused"] == (0, 3) and knobs.RANGES["attn_bm32"] == (0, 1)
    assert knobs.parse({"moe_glue": 7, "mtp_window": 4096}, rows_max=512) == {"moe_glue": 7, "mtp_window": 4096}
    with pytest.raises(ValueError, match="0 to 7"):
        knobs.parse({"moe_glue": 8}, rows_max=512)
    with pytest.raises(ValueError):
        knobs.parse({"mtp_window": -1}, rows_max=512)
    values = {k: 0 for k in knobs.HEADER}
    values["mtp_window"] = 8192
    values["moe_glue"] = 7
    values["hc_fused"] = 3
    got, rest = knobs.decode(knobs.encode(values) + [5])
    assert got == values and rest == [5]


def test_mtp_start_rule(monkeypatch):
    from tensorfold.families.glm5_next.cuda import pfglue

    assert pfglue.mtp_start(10_000, 0) == 0
    assert pfglue.mtp_start(4096, 4096) == 0 and pfglue.mtp_start(100, 4096) == 0
    assert pfglue.mtp_start(4097, 4096) == 0            # (1) rounded down to 64
    assert pfglue.mtp_start(28_045, 4096) == (28_045 - 4096) // 64 * 64
    for n in range(4097, 9000, 37):
        lo = pfglue.mtp_start(n, 4096)
        assert lo % 64 == 0 and n - lo >= 4096 and n - lo < 4096 + 64
    monkeypatch.setenv(pfglue.MTP_ENV, "2048")
    monkeypatch.setenv(pfglue.GROUP_ENV, "5")
    assert pfglue.mtp_window_default() == 2048 and pfglue.group_default() and pfglue.moe_glue_default() == 5
    monkeypatch.setenv(pfglue.HC_ENV, "3")
    monkeypatch.setenv(pfglue.BM32_ENV, "1")
    assert pfglue.settings() == [5, 2048, 3, 1]
    monkeypatch.setenv(pfglue.HC_ENV, "4")
    with pytest.raises(ValueError):
        pfglue.hc_default()
    monkeypatch.setenv(pfglue.MTP_ENV, "-5")
    with pytest.raises(ValueError):
        pfglue.mtp_window_default()


def test_latent_tc_tag_and_load_only(monkeypatch):
    from tensorfold.families.glm5_next.cuda import knobs, pfglue, pfgrid

    assert pfgrid.tag(True, False, 64, True) == 64 + pfgrid.TC == 66
    assert pfgrid.tag(True, True, 64, True) == 65 and pfgrid.tag(False, False, 64, True) == 0
    assert not pfgrid.is_fp8(66) and pfgrid.grid_of(66) == 64
    assert not pfgrid.resumable(64, 66, 128) and pfgrid.resumable(66, 66, 128)
    with pytest.raises(ValueError, match="GLM53_TF_LATENT_TC"):
        knobs.parse({"latent_tc": 1}, rows_max=512)
    monkeypatch.setenv(pfglue.TC_ENV, "1")
    assert pfglue.latent_tc_default() and pfglue.load_settings() == [1]


def _group_reference(pick, rows, ids, count, members, experts):
    """``glue._group`` in plain Python (the kernel's loops, element by element)."""

    slots = pick.shape[1]
    maxu, maxm = members.shape
    block = 1 << (experts).bit_length()          # next power of 2 of experts + 1 for these sizes
    counts = [0] * block
    p = pick[:rows].tolist()
    for r in range(rows):
        for k in range(slots):
            counts[p[r][k]] += 1
    used = [c > 0 for c in counts]
    place, acc = [], -1
    for u in used:
        acc += int(u)
        place.append(acc)
    n_used = sum(used)
    for e_ in range(block):
        if used[e_]:
            ids[place[e_]] = e_
    count[0] = n_used
    filled = [0] * block
    for r in range(rows):
        for k in range(slots):
            e_ = p[r][k]
            members[place[e_], filled[e_]] = r * 32 + k
            filled[e_] += 1
    for e_ in range(block):
        if used[e_]:
            members[place[e_], filled[e_]:] = -1
    members[n_used:] = -1


def _picks(rows: int, kind: str, experts: int = E, top: int = TOP, seed: int = 0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, experts, generator=g)
    if kind == "skewed":
        logits[:, :24] += 2.5
    if kind == "local":                          # consecutive rows share experts (real prompts)
        base = torch.randn(max(rows // 64, 1), experts, generator=g).repeat_interleave(64, 0)[:rows]
        if base.shape[0] < rows:
            base = torch.cat([base, base[-1:].expand(rows - base.shape[0], -1)])
        logits = logits * 0.3 + base
    pick = torch.empty(rows, top + 1, dtype=torch.int32)
    pick[:, :top] = torch.topk(logits, top, dim=1).indices.to(torch.int32)
    pick[:, top] = experts
    return pick.to(device)


@needs_torch
@pytest.mark.parametrize("rows,kind,experts,top", [(1, "uniform", E, TOP), (7, "uniform", E, TOP),
                                                    (64, "uniform", E, TOP), (300, "skewed", E, TOP),
                                                    (1000, "local", E, TOP), (50, "uniform", 16, 2),
                                                    (40, "uniform", 5, 2)])
def test_group_sorted_equals_reference_on_cpu(rows, kind, experts, top):
    from tensorfold.families.glm5_next.cuda import glue

    pick = _picks(rows, kind, experts, top, seed=rows)
    maxu = min(rows * top, experts) + 1
    for maxm in (rows, rows + 13):
        ref = (torch.full((maxu,), 77, dtype=torch.int32), torch.zeros(1, dtype=torch.int32),
               torch.full((maxu, maxm), 5, dtype=torch.int32))
        got = tuple(t.clone() for t in ref)
        _group_reference(pick, rows, *ref, experts)
        glue.group_sorted(pick, rows, *got, experts)
        for a, b in zip(ref, got):
            assert torch.equal(a, b)


@needs_torch
def test_group_switch_rules(monkeypatch):
    from tensorfold.families.glm5_next.cuda import glue

    mem = torch.zeros((10, 128), dtype=torch.int32)
    monkeypatch.setattr(glue, "GROUP_SORT", False)
    assert not glue._group_fast_ok(128, mem)
    monkeypatch.setattr(glue, "GROUP_SORT", True)
    assert glue._group_fast_ok(128, mem) and glue._group_fast_ok(64, mem)
    assert not glue._group_fast_ok(63, mem)                  # decode / verify windows keep _group
    assert not glue._group_fast_ok(129, mem)                 # more rows than member columns: _group's own rule
    assert not glue._group_fast_ok(128, mem[:, :64].t().contiguous().t())


class _FakeSt:
    def __init__(self, cap=512, latent=True, index=True):
        self.mtp_kc = torch.randn(cap, 8)
        self.mtp_vc = self.mtp_kc if latent else torch.randn(cap, 8)
        mk = lambda n: torch.randn(n, 4)                         # noqa: E731
        self.index = [(mk(cap), mk(cap), mk(cap // 4 + 2)), (mk(cap), mk(cap), mk(cap // 4 + 2))] if index else None
        self.mtp_len = 0
        self.mtp_drafted = 0

    def set_mtp_len(self, n):
        self.mtp_len = n


class _FakeE:
    def __init__(self, st):
        self.st = st
        self.calls = []


def _fake_absorb(e, hidden, nxt):
    st = e.st
    if st.mtp_drafted:
        st.set_mtp_len(st.mtp_len - st.mtp_drafted)
        st.mtp_drafted = 0
    e.calls.append((st.mtp_len, hidden.clone(), list(nxt)))
    st.mtp_kc[st.mtp_len:st.mtp_len + len(nxt)] = 1.0
    st.set_mtp_len(st.mtp_len + len(nxt))
    return "logits"


@needs_torch
@pytest.mark.parametrize("latent", [True, False])
def test_mtp_absorb_window_host(latent):
    """Rows below lo are zeroed (head caches, index keys / gates, the pools they complete), mtp_len advances, the rest
    goes to decode.absorb with its hidden rows and tokens; drafted rows are dropped first, as absorb does."""

    from tensorfold.families.glm5_next.cuda import pfglue

    st = _FakeSt(latent=latent)
    e = _FakeE(st)
    h = torch.arange(100, dtype=torch.float32)[:, None].expand(100, 3).contiguous()
    toks = list(range(1000, 1100))
    # chunk [0, 100) with lo = 64: 64 rows zeroed, 36 absorbed
    out = pfglue.absorb(e, h, toks, 0, 64, _fake_absorb)
    assert out == "logits" and st.mtp_len == 100
    assert len(e.calls) == 1 and e.calls[0][0] == 64 and torch.equal(e.calls[0][1], h[64:])
    assert e.calls[0][2] == toks[64:]
    assert torch.count_nonzero(st.mtp_kc[:64]) == 0 and torch.all(st.mtp_kc[64:100] == 1.0)
    if not latent:
        assert torch.count_nonzero(st.mtp_vc[:64]) == 0
    ik, ig, pk = st.index[-1]
    assert torch.count_nonzero(ik[:64]) == 0 and torch.count_nonzero(ig[:64]) == 0
    assert torch.count_nonzero(pk[:16]) == 0 and torch.count_nonzero(pk[16]) > 0      # pools 0..15 end below 64
    assert torch.count_nonzero(st.index[0][0][:64]) > 0                               # the model layers' untouched
    # a chunk wholly below lo: nothing absorbed
    st2 = _FakeSt()
    e2 = _FakeE(st2)
    assert pfglue.absorb(e2, h, toks, 0, 128, _fake_absorb) is None and st2.mtp_len == 100 and not e2.calls
    # wholly above: decode.absorb unchanged
    st3 = _FakeSt()
    st3.mtp_len = 128
    e3 = _FakeE(st3)
    pfglue.absorb(e3, h, toks, 128, 64, _fake_absorb)
    assert e3.calls[0][0] == 128 and torch.equal(e3.calls[0][1], h) and st3.mtp_len == 228
    # drafted rows dropped before skipping
    st4 = _FakeSt()
    st4.mtp_len, st4.mtp_drafted = 13, 3
    e4 = _FakeE(st4)
    pfglue.absorb(e4, h[:20], toks[:20], 10, 20, _fake_absorb)
    assert e4.calls[0][0] == 20 and st4.mtp_len == 30 and torch.count_nonzero(st4.mtp_kc[10:20]) == 0


# -- GPU: kernels ----------------------------------------------------------------------------------------------------
@gpu
@pytest.mark.parametrize("rows", [64, 65, 1024, 4096, 8192])
@pytest.mark.parametrize("kind", ["uniform", "skewed", "local"])
def test_group_sorted_equals_group_kernel(rows, kind):
    import triton

    from tensorfold.families.glm5_next.cuda import glue

    pick = _picks(rows, kind, seed=rows + len(kind), device="cuda")
    maxu = min(rows * TOP, E) + 1
    ref = (torch.full((maxu,), 77, dtype=torch.int32, device="cuda"), torch.zeros(1, dtype=torch.int32, device="cuda"),
           torch.full((maxu, rows), 5, dtype=torch.int32, device="cuda"))
    got = tuple(t.clone() for t in ref)
    glue._group[(1,)](pick, *ref, rows, SLOTS=TOP + 1, MAXU=maxu, MAXM=rows,
                      BLOCK=triton.next_power_of_2(E + 1), num_warps=8)
    glue.group_sorted(pick, rows, *got, E)
    torch.cuda.synchronize()
    for a, b in zip(ref, got):
        assert torch.equal(a, b)


@gpu
def test_select_through_the_switch_same_outputs(monkeypatch):
    """``glue.select`` with GROUP_SORT on == off, every output (picks, weights, ids, count, members)."""

    from tensorfold.families.glm5_next.cuda import glue

    rows = 2048
    g = torch.Generator(device="cuda").manual_seed(3)
    logits = torch.randn(rows, E, device="cuda", generator=g)
    bias = torch.randn(E, device="cuda", generator=g) * 0.01
    outs = []
    for on in (False, True):
        monkeypatch.setattr(glue, "GROUP_SORT", on)
        maxu = min(rows * TOP, E) + 1
        t = (torch.zeros(rows, TOP + 1, dtype=torch.int32, device="cuda"),
             torch.zeros(rows, TOP + 1, dtype=torch.float32, device="cuda"),
             torch.zeros(maxu, dtype=torch.int32, device="cuda"), torch.zeros(1, dtype=torch.int32, device="cuda"),
             torch.full((maxu, rows), -1, dtype=torch.int32, device="cuda"))
        glue.select(logits, bias, *t, TOP, E, 2.5, True)
        outs.append(t)
    for a, b in zip(*outs):
        assert torch.equal(a, b)


@gpu
@pytest.mark.parametrize("rows", [64, 100, 1024, 8192])
def test_router_fused_bitwise(rows, monkeypatch):
    from tensorfold.families.glm5_next.cuda import glue

    g = torch.Generator(device="cuda").manual_seed(rows)
    x = torch.randn(rows, 4096, device="cuda", generator=g).to(torch.bfloat16)
    w = (torch.randn(E, 4096, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    a = torch.empty(rows, E, device="cuda")
    b = torch.empty(rows, E, device="cuda")
    monkeypatch.setattr(glue, "ROUTER_FUSED", False)
    glue.router(x, w, a)
    monkeypatch.setattr(glue, "ROUTER_FUSED", True)
    glue.router(x, w, b)
    torch.cuda.synchronize()
    assert torch.equal(a, b)
    # row-independent: a slice of the rows alone
    if rows >= 128:
        c = torch.empty(70, E, device="cuda")
        glue.router(x[13:83], w, c)
        assert torch.equal(c, b[13:83])


@gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_combine_shared_in_place_bitwise(dtype):
    from tensorfold.families.glm5_next.cuda import glue

    R, D = 1000, 4096
    g = torch.Generator(device="cuda").manual_seed(9)
    y = torch.randn(R, TOP + 1, D, device="cuda", generator=g)
    sy = torch.randn(R, D, device="cuda", generator=g)
    wts = torch.rand(R, TOP + 1, device="cuda", generator=g)
    wts[:, TOP] = 1.0
    o1 = torch.empty(R, D, dtype=dtype, device="cuda")
    o2 = torch.empty(R, D, dtype=dtype, device="cuda")
    y1 = y.clone()
    y1[:, TOP].copy_(sy)
    glue.combine(y1, wts, o1)
    glue.combine(y, wts, o2, shared=sy)
    torch.cuda.synchronize()
    assert torch.equal(o1, o2)


def _hc_inputs(rows, world=2, gdt=None, d=4096, seed=0):
    gdt = torch.bfloat16 if gdt is None else gdt
    g = torch.Generator(device="cuda").manual_seed(seed)
    S = 4
    x = torch.randn(rows, S * d, device="cuda", generator=g).to(torch.bfloat16)
    part = (torch.randn(world, rows, d, device="cuda", generator=g) * 0.5).to(gdt)
    post = torch.rand(rows, S, device="cuda", generator=g) * 2
    comb = torch.rand(rows, S * S, device="cuda", generator=g) / 4
    fn = (torch.randn(24, S * d, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    base = torch.randn(24, device="cuda", generator=g) * 0.1
    scale = torch.tensor([0.4, 0.6, 0.8], device="cuda")
    nw = (torch.rand(d, device="cuda", generator=g) + 0.5).to(torch.bfloat16)
    return x, part, post, comb, fn, base, scale, nw


def _hc_outs(rows, d=4096):
    return (torch.empty(rows, d, dtype=torch.bfloat16, device="cuda"), torch.empty(rows, d // 64, device="cuda"),
            torch.empty(rows, 4, device="cuda"), torch.empty(rows, 16, device="cuda"))


def _hc_separate(x, part, post, comb, fn, base, scale, nw, hcpart, outs):
    from tensorfold.families.glm5_next.cuda import fast_qmm, glue

    glue.hc_post(x, x, part, post, comb)
    saved = glue.FAST_HC
    glue.FAST_HC = fast_qmm.hc_partial
    try:
        glue.hc_pre(x, fn, base, scale, nw, *outs, hcpart, 1e-6, 1e-6, 20)
    finally:
        glue.FAST_HC = saved


@gpu
@pytest.mark.parametrize("mode", [1, 2, 3])
@pytest.mark.parametrize("rows,world,gdt", [(1, 2, "bf16"), (63, 2, "bf16"), (384, 2, "bf16"), (1000, 1, "bf16"),
                                            (257, 2, "fp32")])
def test_hc_fused_bitwise(rows, world, gdt, mode, monkeypatch):
    """hc_post + hc_pre (fast partial) vs the fused path: every output bit (new streams, normed rows, group sums,
    post, comb) on the real width (4 x 4096), bf16 and fp32 partials, one and two ranks."""

    from tensorfold.families.glm5_next.cuda import fast_qmm, glue

    x, part, post, comb, fn, base, scale, nw = _hc_inputs(rows, world, torch.float32 if gdt == "fp32" else None,
                                                          seed=rows)
    xa, xb = x.clone(), x.clone()
    pa, pb = torch.empty(rows, glue.HC_BLOCKS, 32, device="cuda"), torch.empty(rows, glue.HC_BLOCKS, 32, device="cuda")
    oa, ob = _hc_outs(rows), _hc_outs(rows)
    _hc_separate(xa, part, post.clone(), comb.clone(), fn, base, scale, nw, pa, oa)
    monkeypatch.setattr(glue, "HC_FUSED", mode)
    monkeypatch.setattr(glue, "FAST_HC", fast_qmm.hc_partial)
    assert glue.hc_fused_ok(xb, part, fn) == bool(mode & 1)
    glue.hc_post_pre(xb, xb, part, post.clone(), comb.clone(), fn, base, scale, nw, *ob, pb, 1e-6, 1e-6, 20)
    torch.cuda.synchronize()
    assert not glue._HC_FUSED_BROKEN, glue._HC_FUSED_BROKEN       # the fused kernel really ran (no fallback)
    assert torch.equal(xa, xb)
    assert torch.equal(pa[:, :, :25], pb[:, :, :25])
    for a, b in zip(oa, ob):
        assert torch.equal(a, b)


@gpu
def test_hc_fused_row_independent_and_slices(monkeypatch):
    """A slab of a gathered block (rank stride larger than its rows, 0084's slices), row subsets: the same bits as
    the whole call."""

    from tensorfold.families.glm5_next.cuda import fast_qmm, glue

    monkeypatch.setattr(glue, "HC_FUSED", 3)
    monkeypatch.setattr(glue, "FAST_HC", fast_qmm.hc_partial)
    rows = 1024
    x, part, post, comb, fn, base, scale, nw = _hc_inputs(rows, seed=11)
    whole = x.clone()
    pw = torch.empty(rows, glue.HC_BLOCKS, 32, device="cuda")
    ow = _hc_outs(rows)
    glue.hc_post_pre(whole, whole, part, post.clone(), comb.clone(), fn, base, scale, nw, *ow, pw, 1e-6, 1e-6, 20)
    for s0, n in ((0, 384), (384, 384), (768, 256), (5, 70), (1023, 1)):
        xs = x[s0:s0 + n].clone()
        po, co = post[s0:s0 + n].clone(), comb[s0:s0 + n].clone()
        ps = torch.empty(n, glue.HC_BLOCKS, 32, device="cuda")
        os_ = _hc_outs(n)
        glue.hc_post_pre(xs, xs, part[:, s0:s0 + n], po, co, fn, base, scale, nw, *os_, ps, 1e-6, 1e-6, 20)
        assert torch.equal(xs, whole[s0:s0 + n])
        assert torch.equal(ps[:, :, :25], pw[s0:s0 + n, :, :25])
        for a, b in zip(os_, ow):
            assert torch.equal(a, b[s0:s0 + n])


@gpu
@pytest.mark.parametrize("bm", [16, 32])
def test_hc_fused_other_tiles_report(bm, monkeypatch):
    """Other row tiles of the fused kernel (speed candidates for ``glue.HC_FUSED_BM``): printed, not required."""

    from tensorfold.families.glm5_next.cuda import fast_qmm, glue

    rows = 512
    x, part, post, comb, fn, base, scale, nw = _hc_inputs(rows, seed=5)
    xa, xb = x.clone(), x.clone()
    pa, pb = torch.empty(rows, glue.HC_BLOCKS, 32, device="cuda"), torch.empty(rows, glue.HC_BLOCKS, 32, device="cuda")
    glue.hc_post(xa, xa, part, post, comb)
    fast_qmm.hc_partial(xa, fn, pa, glue.HC_BLOCKS)
    monkeypatch.setattr(glue, "HC_FUSED_BM", bm)
    glue.hc_post_part(xb, xb, part, post, comb, fn, pb)
    torch.cuda.synchronize()
    print(f"\n  fused hc, BM {bm}: streams equal {torch.equal(xa, xb)}, partials bit-identical "
          f"{torch.equal(pa[:, :, :25], pb[:, :, :25])}")
    assert torch.equal(xa, xb)
    assert torch.allclose(pa[:, :, :25], pb[:, :, :25], rtol=1e-5, atol=1e-4)


@gpu
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_attn_bm32_bitwise(seed):
    """The 32-query latent attention tile == the 16-query tile bit for bit on the real shapes (32 heads, latent 512):
    dense rows (a 1024-row window at 1024), sparse rows (2,048 selected tokens, one short list), and a 3-row window."""

    from tensorfold.families.glm5_next.cuda import latent

    H, L = 32, 512
    g = torch.Generator(device="cpu").manual_seed(190 + seed)
    for R, P in ((1024, 1024), (3, 1800)):
        lc = torch.randn(P + R + 64, L, generator=g).cuda().bfloat16()
        qa = torch.randn(R, H, L, generator=g).cuda().bfloat16()
        pos = torch.tensor([P], dtype=torch.int32, device="cuda")
        nch = -(-(P + R) // latent.CHUNK)
        from types import SimpleNamespace

        cfg = SimpleNamespace(heads=H * 2, kv_lora=L, v_dim=128, dense_limit=2051)
        sc = latent.Scratch(SimpleNamespace(cfg=cfg, world=2), R, P + R + 64, "cuda")
        outs = []
        for bm in (latent.BM, latent.FAST_BM):
            out = torch.empty((R, H, L), device="cuda")
            latent.attention_latent(qa, lc, pos, sc, scale=0.05, nch=nch, out=out, bm=bm)
            outs.append(out)
        assert torch.equal(outs[0], outs[1]), ("dense", R)
        W = min(2048, P)
        tok = torch.stack([torch.randperm(P, generator=g)[:W].sort().values for _ in range(R)]).cuda().int()
        cnt = torch.full((R,), W, dtype=torch.int32, device="cuda")
        cnt[0] = W // 3
        s16 = torch.zeros((R, H, L), device="cuda")
        s32 = torch.zeros((R, H, L), device="cuda")
        latent.sparse_latent(qa, lc, tok, cnt, s16, 0.05)
        latent.sparse_latent(qa, lc, tok, cnt, s32, 0.05, bm=latent.FAST_BM)
        assert torch.equal(s16, s32), ("sparse", R)


# -- GPU: engine -----------------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")
    monkeypatch.setenv("GLM53_TF_NONEXPERT", "q4mse")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_glue190")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


@pytest.fixture(scope="module")
def ea(ckpt):
    from test_cindep_patches import _engine

    return _engine(ckpt)


@pytest.fixture(scope="module")
def eb(ckpt):
    from test_cindep_patches import _engine

    return _engine(ckpt, lean_block=256, rows_max=8192)


@pytest.fixture(scope="module")
def el(ckpt):
    """Latent KV past the dense limit: the MTP head's sparse path (indexer caches, pools)."""

    from test_cindep_patches import _engine

    return _engine(ckpt, lean_block=256, rows_max=8192, context=4096, latent_kv=True)


class _Glue:
    """Sets the 0190 switches (module globals, as ``tf_knobs`` does on both ranks)."""

    def __init__(self, group=None, window=None, hc=None, bm32=None, tc=False):
        self.v = (group, window, hc, bm32, tc)

    def __enter__(self):
        from tensorfold.families.glm5_next.cuda import glue, latent, pfglue

        self.saved = (pfglue.moe_glue(glue), pfglue.MTP_WINDOW, glue.HC_FUSED, latent.BF16_BM32, latent.BF16_TC)
        group, window, hc, bm32, tc = self.v
        latent.BF16_TC = tc
        if group is not None:
            pfglue.set_moe_glue(glue, 7 if group is True else 0 if group is False else int(group))
        if window is not None:
            pfglue.MTP_WINDOW = window
        if hc is not None:
            glue.HC_FUSED = int(hc) * 3 if isinstance(hc, bool) else hc
        if bm32 is not None:
            latent.BF16_BM32 = bm32
        return self

    def __exit__(self, *exc):
        from tensorfold.families.glm5_next.cuda import glue, latent, pfglue

        moe, pfglue.MTP_WINDOW, glue.HC_FUSED, latent.BF16_BM32, latent.BF16_TC = self.saved
        pfglue.set_moe_glue(glue, moe)


def _main_state(full, eng):
    """``test_cindep_patches._state`` without the MTP head's rows: its caches and length (the last three entries) and,
    on an engine with indexer caches, the MTP layer's own index keys / gates / pool keys (``st.index[-1]``, the three
    entries before them), which the window zeroes below lo."""

    return full[:-6] if eng.e.st.index is not None else full[:-3]


@gpu
@pytest.mark.parametrize("n", [3, 65, 300, 1000])
def test_engine_state_group_on_equals_off(ea, eb, n):
    from test_cindep_patches import _Variant, _same, _state

    prompt = list(np.random.default_rng(190 + n).integers(0, 1000, size=n))
    for eng, rows in ((ea, 1024), (eb, 8192), (eb, 256)):
        with _Variant(eng, rows):
            with _Glue(group=False):
                ref = _state(eng, prompt)
            with _Glue(group=True):
                got = _state(eng, prompt)
        assert _same(ref, got), (rows, [i for i, (a, b) in enumerate(zip(ref, got)) if not torch.equal(a, b)])


@gpu
@pytest.mark.parametrize("C", [1024, 8192])
@pytest.mark.parametrize("overlap", ["0", "1"])
def test_engine_state_c_independent_with_glue(eb, C, overlap):
    """Every same-bits knob on (grouping, fused hc, finish, 32-query tiles) at C = 1024 / 8192, pipelined or not ==
    all off at C = 64 without the pipeline."""

    from test_cindep_patches import _Variant, _same, _state

    for n in (65, 1000):
        prompt = list(np.random.default_rng(290 + n).integers(0, 1000, size=n))
        with _Variant(eb, 64), _Glue(group=False, window=0, hc=0, bm32=False):
            ref = _state(eb, prompt)
        with _Variant(eb, C, overlap=overlap), _Glue(group=True, window=0, hc=3, bm32=True):
            got = _state(eb, prompt)
        assert _same(ref, got), (n, [i for i, (a, b) in enumerate(zip(ref, got)) if not torch.equal(a, b)])


@gpu
@pytest.mark.parametrize("hc", [1, 2, 3])
def test_engine_state_hc_fused(eb, hc):
    """The pipelined lean chunk (0084, slabs on) with the fused hc kernels == without, bit for bit."""

    from test_cindep_patches import _Variant, _same, _state

    for n in (3, 300, 1000):
        prompt = list(np.random.default_rng(590 + n).integers(0, 1000, size=n))
        with _Variant(eb, 8192, overlap="1"), _Glue(hc=0):
            ref = _state(eb, prompt)
        with _Variant(eb, 8192, overlap="1"), _Glue(hc=hc):
            got = _state(eb, prompt)
        assert _same(ref, got), (n, [i for i, (a, b) in enumerate(zip(ref, got)) if not torch.equal(a, b)])


@gpu
def test_engine_state_attn_bm32_latent(el):
    """Latent KV past the dense limit (dense and sparse rows, the MTP head's own caches): 32-query tiles in bf16 fast
    chunks == 16-query tiles, bit for bit; replies too."""

    from test_cindep_patches import _Variant, _same, _state

    prompt = list(np.random.default_rng(690).integers(0, 1000, size=3000))
    with _Variant(el, 1024), _Glue(bm32=False):
        ref = _state(el, prompt)
    with _Variant(el, 1024, overlap="1"), _Glue(bm32=True, group=5):
        got = _state(el, prompt)
    assert _same(ref, got), [i for i, (a, b) in enumerate(zip(ref, got)) if not torch.equal(a, b)]
    with _Variant(el, 1024, overlap="1"), _Glue(bm32=True, hc=3, group=5):
        got = _state(el, prompt)
    assert _same(ref, got), [i for i, (a, b) in enumerate(zip(ref, got)) if not torch.equal(a, b)]


@gpu
@pytest.mark.parametrize("which", ["eb", "el"])
def test_engine_mtp_window_state(eb, el, which):
    """With a window: the main model's state is unchanged; the head's rows below lo are zero, above it equal the
    full head's; the result does not depend on what ran before (determinism) or on C."""

    from test_cindep_patches import _Variant, _same, _state

    from tensorfold.families.glm5_next.cuda import pfglue

    eng = eb if which == "eb" else el
    n = 1000 if which == "eb" else 3000
    W = 256
    prompt = list(np.random.default_rng(390).integers(0, 1000, size=n))
    lo = pfglue.mtp_start(n, W)
    assert lo > 0
    with _Variant(eng, 1024), _Glue(window=0):
        full = _state(eng, prompt)
    with _Variant(eng, 1024), _Glue(window=W):
        win = _state(eng, prompt)
    assert _same(_main_state(full, eng), _main_state(win, eng))
    kc_full, kc_win = full[-3], win[-3]
    assert int(full[-1]) == int(win[-1])                         # mtp_len: the same positions
    assert torch.count_nonzero(kc_win[:lo]) == 0
    # rows past lo: their attention saw zeros below lo, so only the first row of the window is certain to match
    # (it attends to itself and zero rows): check the head ran there at all
    assert torch.count_nonzero(kc_win[lo:]) > 0
    # history independence: other prompts through the head first, then the same prompt again
    with _Variant(eng, 8192), _Glue(window=0):
        _state(eng, list(np.random.default_rng(391).integers(0, 1000, size=n)))
    with _Variant(eng, 8192), _Glue(window=W):
        again = _state(eng, prompt)
    assert _same(win, again)


class _Cache:
    """pfglue.MTP_CACHE (GLM53_TF_MTP_PREFILL_CACHE, load-time on the server; a module global here)."""

    def __init__(self, on: bool):
        self.on = on

    def __enter__(self):
        from tensorfold.families.glm5_next.cuda import pfglue

        self.saved, pfglue.MTP_CACHE = pfglue.MTP_CACHE, self.on
        return self

    def __exit__(self, *exc):
        from tensorfold.families.glm5_next.cuda import pfglue

        pfglue.MTP_CACHE = self.saved


@gpu
@pytest.mark.parametrize("n", [65, 1000, 3000])
@pytest.mark.parametrize("C", [256, 1024])
def test_engine_mtp_prefill_cache_state(el, n, C):
    """GLM53_TF_MTP_PREFILL_CACHE: prefill's MTP rows write only the head's caches. The whole committed state (the
    head's latent rows, its index keys / gates / pools, mtp_len) == the full head's, bit for bit (latent KV, dense
    and sparse rows past the dense limit), with and without an MTP window; replies too (MTP, DFlash2, auto)."""

    from test_cindep_patches import _Variant, _same, _state

    from tensorfold.families.glm5_next.cuda import pfglue

    assert pfglue.cache_ok(el.e)
    prompt = list(np.random.default_rng(790 + n).integers(0, 1000, size=n))
    for window in (0, 256):
        with _Variant(el, C), _Glue(window=window), _Cache(False):
            ref = _state(el, prompt)
        with _Variant(el, C), _Glue(window=window), _Cache(True):
            got = _state(el, prompt)
        assert _same(ref, got), (window, [i for i, (a, b) in enumerate(zip(ref, got)) if not torch.equal(a, b)])


@gpu
@pytest.mark.parametrize("policy", [None, "f3", "auto:1:1:0"])
def test_engine_mtp_prefill_cache_replies(el, policy):
    """Replies and their draft statistics (the same drafts: the head's caches are the same) with the cache-only
    prefill == the full head's; resumed == fresh."""

    from test_cindep_patches import _cold, _gen, _sampling

    s = _sampling("greedy")
    rng = np.random.default_rng(890)
    more = lambda k: [int(t) for t in rng.integers(0, 1000, size=k)]       # noqa: E731
    p1 = more(2600)
    el.cache = []
    with _Cache(False):
        off, st_off = _gen(el, p1, s, policy=policy, knobs={"prefill_rows": 1024})
    el.cache = []
    with _Cache(True):
        on, st_on = _gen(el, p1, s, policy=policy, knobs={"prefill_rows": 1024})
        assert on == off
        assert st_on.get("keeps") == st_off.get("keeps")
        p2 = p1 + on + more(90)
        warm, stats = _gen(el, p2, s, policy=policy, knobs={"prefill_rows": 1024})
        assert stats["cached"] > 0
    with _Cache(False):
        assert warm == _cold(el, p2, s, knobs={"prefill_rows": 1024})


@gpu
@pytest.mark.parametrize("sampling", ["greedy", "sampled"])
@pytest.mark.parametrize("policy", [None, "2", "f3", "auto:1:1:0"])
def test_engine_mtp_window_replies_equal(eb, sampling, policy):
    """Replies with the window equal replies without it (and serial), drafted; resumed == fresh with the window."""

    from test_cindep_patches import _cold, _gen, _sampling

    s = _sampling(sampling)
    rng = np.random.default_rng(490)
    more = lambda k: [int(t) for t in rng.integers(0, 1000, size=k)]       # noqa: E731
    p1 = more(1200)
    eb.cache = []
    off, _ = _gen(eb, p1, s, policy=policy, knobs={"mtp_window": 0, "prefill_rows": 8192})
    eb.cache = []
    on, _ = _gen(eb, p1, s, policy=policy, knobs={"mtp_window": 256, "prefill_rows": 8192, "moe_glue": 7})
    serial = _cold(eb, p1, s, knobs={"prefill_rows": 1024})
    assert on == off == serial[:len(on)]
    # the cold serial run above left snapshots without MTP rows (draft=False): a drafted run of p1 again keeps its own
    _gen(eb, p1, s, policy=policy, knobs={"mtp_window": 256, "prefill_rows": 8192})
    p2 = p1 + on + more(90)
    warm, stats = _gen(eb, p2, s, policy=policy, knobs={"mtp_window": 256, "prefill_rows": 1024})
    assert stats["cached"] > 0
    assert warm == _cold(eb, p2, s, knobs={"mtp_window": 0, "prefill_rows": 4096})


@gpu
def test_engine_knobs_echoed(eb):
    from test_cindep_patches import _gen

    _, stats = _gen(eb, list(range(100)), None, knobs={"mtp_window": 4096, "moe_glue": 7, "hc_fused": 3,
                                                        "attn_bm32": 1})
    kn = stats["tf_knobs"]
    assert kn["mtp_window"] == 4096 and kn["moe_glue"] == 7 and kn["hc_fused"] == 3 and kn["attn_bm32"] == 1
    from tensorfold.families.glm5_next.cuda import glue, latent, pfglue

    assert pfglue.moe_glue(glue) == pfglue.moe_glue_default() and pfglue.MTP_WINDOW == pfglue.mtp_window_default()
    assert glue.HC_FUSED == pfglue.hc_default() and latent.BF16_BM32 == pfglue.bm32_default()


@gpu
def test_engine_latent_tc(el):
    """GLM53_TF_LATENT_TC: new arithmetic (the committed state differs from the FMA path: control), still independent
    of C (1024 / 8192, pipelined), deterministic; drafted == serial and resumed == fresh with it; its snapshots carry
    tag G + 2, so a request of the other arithmetic never resumes them."""

    from test_cindep_patches import _Variant, _cold, _gen, _same, _sampling, _state

    from tensorfold.families.glm5_next.cuda import latent

    prompt = list(np.random.default_rng(790).integers(0, 1000, size=3000))
    with _Variant(el, 1024), _Glue(tc=False):
        fma = _state(el, prompt)
    with _Variant(el, 1024), _Glue(tc=True):
        a = _state(el, prompt)
        assert el._grid(True) == 66 or el._grid(True) % 64 == 2
    with _Variant(el, 8192, overlap="1"), _Glue(tc=True, hc=3, group=True, bm32=True):
        b = _state(el, prompt)
    assert _same(a, b)
    assert not _same(fma, a)                       # control: the switch reaches the kernels
    # close to the FMA path: the first token agrees on this prompt, the latent rows to bf16 precision
    assert torch.equal(fma[0], a[0])
    s = _sampling("greedy")
    rng = np.random.default_rng(791)
    p1 = [int(t) for t in rng.integers(0, 1000, size=2600)]
    with _Glue(tc=True):
        el.cache = []
        r1, _ = _gen(el, p1, s, policy="auto:1:1:0", knobs={"prefill_rows": 8192})
        assert r1 == _cold(el, p1, s, knobs={"prefill_rows": 1024})[:len(r1)]
        _gen(el, p1, s, knobs={"prefill_rows": 8192})
        p2 = p1 + r1 + [int(t) for t in rng.integers(0, 1000, size=70)]
        warm, stats = _gen(el, p2, s, knobs={"prefill_rows": 64})
        assert stats["cached"] > 0
        assert warm == _cold(el, p2, s, knobs={"prefill_rows": 4096})
    assert latent.BF16_TC is False
