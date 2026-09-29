"""patches/0240 (b12x-derived fast-prefill kernels, ``tf_knobs.b12x`` / GLM53_TF_B12X): bit 1 the hyper-connection
mixing dots on column blocks of all four streams fused with hc_post (``b12x_mhc``), bit 2 KDA in 16-row tiles with the
tile preparation and the recurrence in one program (``b12x_kda``), bit 4 one-pass sparse latent attention
(``b12x_attn``).

What must hold (every bit is new arithmetic in fast chunks, so bits are not compared with today's fast path, only
closeness is):

- each kernel is deterministic and row-independent: a row's bits do not depend on the other rows of the call (row
  subsets, permutations, rank-stride slices of 0084's slabs); KDA: a prompt split at any multiple of 64 gives the bits
  of one call, and a call's first k rows are a k-row call's;
- ``b12x_mhc``: the fused post + dots kernel == hc_post then the unfused dots, bit for bit (the pipelined and the plain
  lean / non-lean chunks must agree, patches/0084), and the dots equal a float64 reference to fp32 accuracy;
- ``b12x_kda``: within the error ``fast_kda`` has against the serial chain (``kda.chain``); ``b12x_attn``: as close to
  a float64 softmax as the chunked kernel;
- engine: with any bits the committed state is the same for every chunk size and with 0084's pipeline on or off
  (patches/0085), deterministic, drafted == serial, resumed == fresh; the state differs from bits 0 (control: the
  switch reaches the kernels) but stays close; a request with other bits never resumes these snapshots (tag
  G + 4 x bits); the knob is echoed and restored.

Host-only tests run anywhere with torch (CPU); kernel and engine tests need a GPU. Timings: ``bench_b12x.py``.

Run inside the image:
    PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q -s tests/cuda/test_b12x_patches.py
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


# -- host only --------------------------------------------------------------------------------------------------------
def test_knob_in_header():
    from tensorfold.families.glm5_next.cuda import knobs

    assert "b12x" in knobs.HEADER and knobs.HEADER[-1] == "b12x" and knobs.RANGES["b12x"] == (0, 7)
    assert knobs.parse({"b12x": 7}, rows_max=512) == {"b12x": 7}
    assert knobs.parse({"b12x": True}, rows_max=512) == {"b12x": 1}
    for bad in (8, -1, "2"):
        with pytest.raises(ValueError):
            knobs.parse({"b12x": bad}, rows_max=512)
    values = {k: 0 for k in knobs.HEADER}
    values["b12x"] = 5
    got, rest = knobs.decode(knobs.encode(values) + [9])
    assert got == values and rest == [9]


def test_tags_carry_the_bits(monkeypatch):
    from tensorfold.families.glm5_next.cuda import pfgrid

    monkeypatch.delenv(pfgrid.GRID_ENV, raising=False)
    assert pfgrid.tag(True) == 64 and pfgrid.tag(False, b12x=7) == 0
    seen = set()
    for fp8 in (False, True):
        for tc in (False, True):
            for bits in range(8):
                t = pfgrid.tag(True, fp8, 64, tc, bits)
                assert 64 <= t < 128 and pfgrid.grid_of(t) == 64
                assert pfgrid.is_fp8(t) == fp8
                assert t % 64 // pfgrid.B12X == bits
                seen.add((fp8, tc and not fp8, bits, t))
    tags = [t for *_, t in seen]
    assert len(set(tags)) == len(tags)                  # every (mode, bits) pair its own tag
    assert pfgrid.tag(True, grid=128, b12x=3) == 128 + 12
    assert not pfgrid.resumable(pfgrid.tag(True, b12x=2), pfgrid.tag(True), 128)
    assert not pfgrid.resumable(pfgrid.tag(True, b12x=2), pfgrid.tag(True, b12x=3), 128)
    assert pfgrid.resumable(pfgrid.tag(True, b12x=2), pfgrid.tag(True, b12x=2), 128)
    with pytest.raises(ValueError):
        pfgrid.tag(True, b12x=8)


def test_session_grid_of_b12x_tags():
    from tensorfold.families.glm5_next.cuda import pfgrid, sessions

    for bits in range(8):
        assert sessions.snapshot_grid(pfgrid.tag(True, grid=64, b12x=bits)) == 64


def test_switch_defaults(monkeypatch):
    from types import SimpleNamespace

    from tensorfold.families.glm5_next.cuda import b12xpf

    monkeypatch.delenv(b12xpf.ENV, raising=False)
    assert b12xpf.default() == 0
    monkeypatch.setenv(b12xpf.ENV, "6")
    assert b12xpf.default() == 6
    for bad in ("8", "x", "-1"):
        monkeypatch.setenv(b12xpf.ENV, bad)
        with pytest.raises(ValueError):
            b12xpf.default()
    assert b12xpf.effective(7, SimpleNamespace(meta={"latent_kv": False})) == 3
    assert b12xpf.effective(7, SimpleNamespace(meta={"latent_kv": True})) == 7
    assert b12xpf.effective(7, None) == 7
    assert b12xpf.describe(0) == "off" and b12xpf.describe(7) == "hc,kda,sparse"


@needs_torch
def test_set_bits_switches_the_modules():
    pytest.importorskip("triton")
    from tensorfold.families.glm5_next.cuda import b12xpf, glue, latent

    try:
        b12xpf.set_bits(5)
        assert glue.MHC and latent.SPARSE_ONE and not b12xpf.kda_on()
        b12xpf.set_bits(2)
        assert not glue.MHC and not latent.SPARSE_ONE and b12xpf.kda_on()
        with pytest.raises(ValueError):
            b12xpf.set_bits(9)
    finally:
        b12xpf.set_bits(0)
    assert not glue.MHC and not latent.SPARSE_ONE and not b12xpf.kda_on()


# -- GPU: KDA -----------------------------------------------------------------------------------------------------------
H_REAL = 32


def _kda_inputs(rows, heads=H_REAL, mode="mixed", seed=0, lower=-5.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    C = 3 * heads * 128
    b_off = C + 256
    p = torch.randn(rows, b_off + heads + 3, device=dev, generator=g)
    a = torch.randn(rows, heads * 128, device=dev, generator=g)
    if mode == "slow":
        a = a * 0.5 - 9.0
        p[:, b_off:b_off + heads] += 4.0
    elif mode == "fast":
        a = a * 0.5 + 12.0
    else:
        a = torch.where(torch.rand(rows, heads * 128, device=dev, generator=g) < 0.5, a * 0.5 - 9.0, a * 0.5 + 12.0)
    return dict(p=p.bfloat16(), b_off=b_off, a=a.bfloat16(),
                g=torch.randn(rows, heads * 128, device=dev, generator=g).bfloat16(),
                conv_state=torch.randn(3, C, device=dev, generator=g).bfloat16(),
                conv_w=(torch.randn(C, 4, device=dev, generator=g) * 0.5).bfloat16(),
                state_in=torch.randn(heads, 128, 128, device=dev, generator=g) * 0.1,
                a_log=torch.randn(heads, device=dev, generator=g) * 0.2,
                dt_bias=torch.randn(heads * 128, device=dev, generator=g) * 0.2,
                norm_w=(1 + 0.1 * torch.randn(128, device=dev, generator=g)).bfloat16(), eps=1e-6, lower=lower)


def _kargs(d, rows):
    return (d["p"], d["b_off"], d["a"], d["g"], d["conv_state"], d["conv_w"], d["state_in"], d["a_log"],
            d["dt_bias"], d["norm_w"], d["eps"], d["lower"], rows)


def _k16(d, rows, pos, **kw):
    from tensorfold.families.glm5_next.cuda import b12x_kda

    so = torch.empty_like(d["state_in"])
    o = b12x_kda.kda_prefill16(*_kargs(d, rows), None, so, pos=pos, **kw)
    torch.cuda.synchronize()
    return o.clone(), so


def _chain(d, rows):
    from tensorfold.families.glm5_next.cuda import kda

    so = torch.empty_like(d["state_in"])
    scr = kda.KDAScratch(rows, d["a_log"].numel(), "cuda")
    o = kda.chain(*_kargs(d, rows), scr, so).clone()
    return o, so


@gpu
@pytest.mark.parametrize("mode", ["slow", "fast", "mixed"])
@pytest.mark.parametrize("rows,pos", [(1024, 0), (1000, 4096), (37, 64), (1, 128)])
def test_kda16_close_to_the_serial_chain(mode, rows, pos):
    """Against ``kda.chain`` (the decode kernel, row by row): no worse than ``fast_kda`` (tf32) with a margin."""

    from tensorfold.families.glm5_next.cuda import fast_kda

    d = _kda_inputs(rows, mode=mode, seed=rows + pos)
    ref_o, ref_s = _chain(d, rows)
    o, s = _k16(d, rows, pos)
    so_old = torch.empty_like(s)
    o_old = fast_kda.kda_prefill_chunked(*_kargs(d, rows), None, so_old, pos=pos)
    assert torch.isfinite(o.float()).all() and torch.isfinite(s).all()
    rel = ((s - ref_s).norm() / ref_s.norm()).item()
    rel_old = ((so_old - ref_s).norm() / ref_s.norm()).item()
    diff = (o.float() - ref_o.float()).abs()
    diff_old = (o_old.float() - ref_o.float()).abs()
    print(f"\n[kda16 {mode} {rows}@{pos}] state rel {rel:.2e} (fast_kda {rel_old:.2e}), out mean {diff.mean():.2e} "
          f"(fast_kda {diff_old.mean():.2e}), max {diff.max():.3e}")
    assert rel < max(2e-3, 1.5 * rel_old), rel
    assert diff.mean() < max(2e-3, 1.5 * diff_old.mean().item()), diff.mean()
    assert (diff <= 0.03 + 0.02 * ref_o.float().abs()).all(), diff.max()


@gpu
def test_kda16_bf16_operands_report():
    """b12x's own policy (bf16 operands and state shadow): bounded, and printed next to tf32 (not the default)."""

    rows = 1024
    d = _kda_inputs(rows, seed=11)
    ref_o, ref_s = _chain(d, rows)
    for prec in ("tf32", "bf16"):
        o, s = _k16(d, rows, 0, prec=prec)
        rel = ((s - ref_s).norm() / ref_s.norm()).item()
        print(f"\n[kda16 {prec}] state rel {rel:.2e}, out mean {(o.float() - ref_o.float()).abs().mean():.2e}")
        assert rel < 2e-2


@gpu
def test_kda16_deterministic_split_and_prefix():
    """Two runs: the same bits. A prompt in calls cut at multiples of 64: the bits of one call. A call's first k rows
    (k not a multiple of 16): a k-row call's (a row never reads a later row, padded rows add exact zeros)."""

    rows = 1024
    d = _kda_inputs(rows, seed=21)
    C = d["conv_state"].shape[1]
    o1, s1 = _k16(d, rows, 8192)
    o2, s2 = _k16(d, rows, 8192)
    assert torch.equal(o1, o2) and torch.equal(s1, s2)
    for cuts in ((64,), (512,), (64, 128, 960), (320, 704)):
        outs, state, start = [], d["state_in"], 0
        for stop in list(cuts) + [rows]:
            part = dict(d)
            part.update(p=d["p"][start:], a=d["a"][start:], g=d["g"][start:], state_in=state,
                        conv_state=d["conv_state"] if start == 0 else d["p"][start - 3:start, :C].contiguous())
            o, state = _k16(part, stop - start, 8192 + start)
            outs.append(o)
            start = stop
        assert torch.equal(torch.cat(outs), o1), cuts
        assert torch.equal(state, s1), cuts
    for k in (1, 15, 17, 100, 1000):
        o, _ = _k16(d, k, 8192)
        assert torch.equal(o, o1[:k]), k


@gpu
def test_kda16_dispatch_and_fallback(monkeypatch):
    """``fastpf.kda_chain`` runs the 16-row kernel with bit 2 on, ``fast_kda`` without it and for |lower| > 5.3."""

    from tensorfold.families.glm5_next.cuda import b12xpf, fast_kda, fastpf

    rows = 200
    d = _kda_inputs(rows, heads=4, seed=31)
    ref, _ = _k16(d, rows, 64)
    so = torch.empty_like(d["state_in"])
    old = fast_kda.kda_prefill_chunked(*_kargs(d, rows), None, so, pos=64).clone()
    try:
        b12xpf.set_bits(2)
        got = fastpf.kda_chain(*_kargs(d, rows), None, torch.empty_like(d["state_in"]), pos=64)
        assert torch.equal(got, ref)
        wide = dict(d, lower=-6.0)
        so1, so2 = torch.empty_like(d["state_in"]), torch.empty_like(d["state_in"])
        a = fastpf.kda_chain(*_kargs(wide, rows), None, so1, pos=64).clone()
        b = fast_kda.kda_prefill_chunked(*_kargs(wide, rows), None, so2, pos=64)
        assert torch.equal(a, b) and torch.equal(so1, so2)
    finally:
        b12xpf.set_bits(0)
    got = fastpf.kda_chain(*_kargs(d, rows), None, torch.empty_like(d["state_in"]), pos=64)
    assert torch.equal(got, old)


# -- GPU: hyper-connections ---------------------------------------------------------------------------------------------
def _hc_inputs(rows, world=2, gdt=None, d=4096, seed=0):
    gdt = torch.bfloat16 if gdt is None else gdt
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, 4 * d, device="cuda", generator=g).to(torch.bfloat16)
    part = (torch.randn(world, rows, d, device="cuda", generator=g) * 0.5).to(gdt)
    post = torch.rand(rows, 4, device="cuda", generator=g) * 2
    comb = torch.rand(rows, 16, device="cuda", generator=g) / 4
    fn = (torch.randn(24, 4 * d, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    base = torch.randn(24, device="cuda", generator=g) * 0.1
    scale = torch.tensor([0.4, 0.6, 0.8], device="cuda")
    nw = (torch.rand(d, device="cuda", generator=g) + 0.5).to(torch.bfloat16)
    return x, part, post, comb, fn, base, scale, nw


def _hc_outs(rows, d=4096):
    return (torch.empty(rows, d, dtype=torch.bfloat16, device="cuda"), torch.empty(rows, d // 64, device="cuda"),
            torch.empty(rows, 4, device="cuda"), torch.empty(rows, 16, device="cuda"))


class _Mhc:
    def __init__(self, on: bool):
        self.on = on

    def __enter__(self):
        from tensorfold.families.glm5_next.cuda import fast_qmm, glue

        self.saved = (glue.MHC, glue.FAST_HC, glue.HC_FUSED)
        glue.MHC, glue.FAST_HC, glue.HC_FUSED = self.on, fast_qmm.hc_partial, 0
        return self

    def __exit__(self, *exc):
        from tensorfold.families.glm5_next.cuda import glue

        glue.MHC, glue.FAST_HC, glue.HC_FUSED = self.saved


def _site(x, part, post, comb, fn, base, scale, nw, hcpart, outs, fused):
    from tensorfold.families.glm5_next.cuda import glue

    if fused:
        glue.hc_post_pre(x, x, part, post, comb, fn, base, scale, nw, *outs, hcpart, 1e-6, 1e-6, 20)
    else:
        glue.hc_post(x, x, part, post, comb)
        glue.hc_pre(x, fn, base, scale, nw, *outs, hcpart, 1e-6, 1e-6, 20)


@gpu
@pytest.mark.parametrize("rows", [1, 63, 384, 1000])
def test_mhc_partials_vs_float64(rows):
    from tensorfold.families.glm5_next.cuda import b12x_mhc, glue

    x, *_ , fn, _, _, _ = _hc_inputs(rows, seed=rows)
    part = torch.empty(rows, glue.HC_BLOCKS, 32, device="cuda")
    b12x_mhc.partial(x, fn, part, glue.HC_BLOCKS)
    ref = b12x_mhc.partial_reference(x, fn, glue.HC_BLOCKS)
    err = (part[:, :, :25].double() - ref).abs()
    scale = ref.abs().amax(dim=0, keepdim=True).clamp_min(1e-3)
    assert (err / scale).max().item() < 1e-5


@gpu
@pytest.mark.parametrize("rows,world,gdt", [(1, 2, "bf16"), (63, 2, "bf16"), (384, 2, "bf16"), (1000, 1, "bf16"),
                                            (257, 2, "fp32")])
def test_mhc_fused_equals_unfused(rows, world, gdt):
    """hc_post + hc_pre (b12x partial) vs the fused kernel: every output bit (new streams, normed rows, group sums,
    post, comb), bf16 and fp32 partials, one and two ranks."""

    from tensorfold.families.glm5_next.cuda import b12x_mhc, glue

    x, part, post, comb, fn, base, scale, nw = _hc_inputs(rows, world, torch.float32 if gdt == "fp32" else None,
                                                          seed=rows)
    res = []
    for fused in (False, True):
        xx, pp, cc = x.clone(), post.clone(), comb.clone()
        hp = torch.empty(rows, glue.HC_BLOCKS, 32, device="cuda")
        outs = _hc_outs(rows)
        with _Mhc(True):
            assert glue.hc_fused_ok(xx, part, fn)
            _site(xx, part, pp, cc, fn, base, scale, nw, hp, outs, fused)
        torch.cuda.synchronize()
        res.append((xx, hp[:, :, :25].clone(), *outs))
    assert not b12x_mhc._BROKEN, b12x_mhc._BROKEN          # the fused kernel really ran
    for a, b in zip(*res):
        assert torch.equal(a, b)


@gpu
def test_mhc_row_independent_slices_and_permutations():
    """0084's slabs (a row slice of a gathered block: rank stride larger than its rows), row subsets and a permutation
    of the rows: the same bits as the whole call; two runs the same bits."""

    from tensorfold.families.glm5_next.cuda import glue

    rows = 1024
    x, part, post, comb, fn, base, scale, nw = _hc_inputs(rows, seed=41)
    whole = x.clone()
    pw = torch.empty(rows, glue.HC_BLOCKS, 32, device="cuda")
    ow = _hc_outs(rows)
    with _Mhc(True):
        _site(whole, part, post.clone(), comb.clone(), fn, base, scale, nw, pw, ow, True)
        again = x.clone()
        pa = torch.empty_like(pw)
        oa = _hc_outs(rows)
        _site(again, part, post.clone(), comb.clone(), fn, base, scale, nw, pa, oa, True)
        assert torch.equal(again, whole) and all(torch.equal(a, b) for a, b in zip(oa, ow))
        for s0, n in ((0, 384), (384, 384), (768, 256), (5, 70), (1023, 1)):
            for fused in (True, False):
                xs = x[s0:s0 + n].clone()
                ps = torch.empty(n, glue.HC_BLOCKS, 32, device="cuda")
                os_ = _hc_outs(n)
                _site(xs, part[:, s0:s0 + n], post[s0:s0 + n].clone(), comb[s0:s0 + n].clone(), fn, base, scale, nw,
                      ps, os_, fused)
                assert torch.equal(xs, whole[s0:s0 + n])
                assert torch.equal(ps[:, :, :25], pw[s0:s0 + n, :, :25])
                for a, b in zip(os_, ow):
                    assert torch.equal(a, b[s0:s0 + n])
        perm = torch.randperm(rows, device="cuda")
        xp = x[perm].clone()
        pp = torch.empty_like(pw)
        op = _hc_outs(rows)
        _site(xp, part[:, perm].contiguous(), post[perm].clone(), comb[perm].clone(), fn, base, scale, nw, pp, op, True)
        assert torch.equal(xp, whole[perm])
        for a, b in zip(op, ow):
            assert torch.equal(a, b[perm])


@gpu
def test_mhc_close_to_todays_fast_path():
    """New arithmetic, but only the mixes' summation order moved: new streams identical (hc_post is unchanged),
    post / comb within fp32 noise, normed rows within one bf16 step."""

    from tensorfold.families.glm5_next.cuda import glue

    rows = 512
    x, part, post, comb, fn, base, scale, nw = _hc_inputs(rows, seed=51)
    res = []
    for on in (False, True):
        xx = x.clone()
        hp = torch.empty(rows, glue.HC_BLOCKS, 32, device="cuda")
        outs = _hc_outs(rows)
        with _Mhc(on):
            _site(xx, part, post.clone(), comb.clone(), fn, base, scale, nw, hp, outs, on)
        res.append((xx, *outs))
    (x0, n0, _, p0, c0), (x1, n1, _, p1, c1) = res
    assert torch.equal(x0, x1)
    assert (p0 - p1).abs().max() < 1e-5 and (c0 - c1).abs().max() < 1e-5
    d = (n0.float() - n1.float()).abs()
    assert (d <= 1e-2 * n0.float().abs() + 1e-3).all(), d.max()
    assert not torch.equal(n0, n1) or not torch.equal(p0, p1)        # control: the switch reaches the kernels


# -- GPU: sparse attention ---------------------------------------------------------------------------------------------
def _attn_inputs(rows=96, H=32, L=512, P=6000, W=2051, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    lat = torch.randn(P, L, generator=g).bfloat16().cuda()
    qa = (torch.randn(rows, H, L, generator=g) * 0.3).bfloat16().cuda()
    tok = torch.full((rows, W), -1, dtype=torch.int32)
    cnt = torch.zeros(rows, dtype=torch.int32)
    for r in range(rows):
        n = [W, 0, 700, 33, W - 1][r % 5]
        cnt[r] = n
        tok[r, :n] = torch.randperm(P, generator=g)[:n].sort().values.int()
    return qa, lat, tok.cuda(), cnt.cuda()


@gpu
@pytest.mark.parametrize("cache", ["bf16", "fp8"])
def test_attn_one_pass_close_and_row_independent(cache):
    from tensorfold.families.glm5_next.cuda import b12x_attn, latent

    qa, lat, tok, cnt = _attn_inputs(seed=3)
    lc = lat if cache == "bf16" else latent.quantize_rows_reference(lat)
    deq = latent.dequantize_rows(lc)
    scale = 256 ** -0.5
    rows, H, L = qa.shape
    one = torch.full((rows, H, L), 7.0, device="cuda")
    chunked = torch.full((rows, H, L), 7.0, device="cuda")
    b12x_attn.sparse_latent_one(qa, lc, tok, cnt, one, scale)
    latent.sparse_latent(qa, lc, tok, cnt, chunked, scale, bm=latent.FAST_BM)
    ref = b12x_attn.reference(qa, deq, tok, cnt, scale)
    m = cnt > 0
    e1 = (one[m].double() - ref[m]).abs().max().item()
    e0 = (chunked[m].double() - ref[m]).abs().max().item()
    print(f"\n[attn {cache}] one pass vs f64 {e1:.3e}, chunked vs f64 {e0:.3e}")
    assert e1 <= 1.5 * e0 + 1e-4
    assert bool((one[~m] == 7.0).all())                                 # dense rows untouched
    again = torch.zeros_like(one)
    b12x_attn.sparse_latent_one(qa, lc, tok, cnt, again, scale)
    assert torch.equal(again[m], one[m])
    for s0, n in ((0, 1), (7, 40), (50, 46)):
        sub = torch.zeros((n, H, L), device="cuda")
        b12x_attn.sparse_latent_one(qa[s0:s0 + n].contiguous(), lc, tok[s0:s0 + n].contiguous(),
                                    cnt[s0:s0 + n].contiguous(), sub, scale)
        mm = cnt[s0:s0 + n] > 0
        assert torch.equal(sub[mm], one[s0:s0 + n][mm])
    perm = torch.randperm(rows, device="cuda")
    pout = torch.zeros_like(one)
    b12x_attn.sparse_latent_one(qa[perm].contiguous(), lc, tok[perm].contiguous(), cnt[perm].contiguous(), pout, scale)
    assert torch.equal(pout[m[perm]], one[perm][m[perm]])
    if cache == "fp8":
        viadeq = torch.zeros_like(one)
        b12x_attn.sparse_latent_one(qa, deq, tok, cnt, viadeq, scale)
        assert torch.equal(viadeq[m], one[m])                          # FP8 rows == their exact bf16 dequantization


# -- GPU: engine -----------------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")
    monkeypatch.setenv("GLM53_TF_NONEXPERT", "q4mse")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_b12x240")
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
    from test_cindep_patches import _engine

    return _engine(ckpt, lean_block=256, rows_max=8192, context=4096, latent_kv=True)


class _Bits:
    """``tf_knobs.b12x`` as both ranks set it (module globals)."""

    def __init__(self, bits: int):
        self.bits = bits

    def __enter__(self):
        from tensorfold.families.glm5_next.cuda import b12xpf

        self.saved = b12xpf.BITS
        b12xpf.set_bits(self.bits)
        return self

    def __exit__(self, *exc):
        from tensorfold.families.glm5_next.cuda import b12xpf

        b12xpf.set_bits(self.saved)


def _diff(a, b):
    return [i for i, (x, y) in enumerate(zip(a, b)) if not (x.shape == y.shape and torch.equal(x, y))]


@gpu
@pytest.mark.parametrize("bits", [1, 2, 3])
@pytest.mark.parametrize("n", [3, 65, 1000])
def test_engine_state_c_and_overlap_independent(ea, eb, bits, n):
    """With the bits on: the same committed state for C = 64 (non-lean), 1024, 8192 (lean, pipelined or not); the
    state differs from bits 0 (control) and the first token of a random prompt mostly agrees."""

    from test_cindep_patches import _Variant, _same, _state

    prompt = list(np.random.default_rng(2400 + n).integers(0, 1000, size=n))
    with _Variant(ea, 64), _Bits(bits):
        ref = _state(ea, prompt)
    for eng, C, ov in ((ea, 1024, "0"), (eb, 1024, "0"), (eb, 8192, "1"), (eb, 256, "1")):
        with _Variant(eng, C, overlap=ov), _Bits(bits):
            got = _state(eng, prompt)
        assert _same(ref, got), (bits, C, ov, _diff(ref, got))
    with _Variant(ea, 64), _Bits(bits):
        assert _same(ref, _state(ea, prompt))                    # deterministic
    with _Variant(ea, 64), _Bits(0):
        base = _state(ea, prompt)
    if n > 16 or bits & 1:
        assert not _same(ref, base), bits                         # the switch reaches the kernels
    # close to today's fast path: KDA state and the MTP input row within bf16-level noise
    for i in (1, 3):
        a, b = ref[i].float(), base[i].float()
        assert ((a - b).norm() / b.norm().clamp_min(1e-6)).item() < 5e-2, (i, bits)


@gpu
def test_engine_latent_sparse_bit(el):
    """Latent KV past the dense limit: bit 4 (one-pass sparse attention) and all bits: C-independent (pipelined or
    not), deterministic, differs from bits 0 (control)."""

    from test_cindep_patches import _Variant, _same, _state

    prompt = list(np.random.default_rng(2440).integers(0, 1000, size=3000))
    for bits in (4, 7):
        with _Variant(el, 1024), _Bits(bits):
            a = _state(el, prompt)
        with _Variant(el, 8192, overlap="1"), _Bits(bits):
            b = _state(el, prompt)
        assert _same(a, b), (bits, _diff(a, b))
        with _Variant(el, 1024), _Bits(bits):
            assert _same(a, _state(el, prompt))
    with _Variant(el, 1024), _Bits(0):
        base = _state(el, prompt)
    with _Variant(el, 1024), _Bits(4):
        four = _state(el, prompt)
    assert not _same(base, four)


@gpu
@pytest.mark.parametrize("sampling", ["greedy", "sampled"])
@pytest.mark.parametrize("policy", [None, "2", "f3", "auto:1:1:0"])
def test_engine_drafted_equals_serial_and_resumed_equals_fresh(eb, sampling, policy):
    """``tf_knobs.b12x`` = 3 (the latent-free bits on this engine): drafted replies == ``draft: false``; a follow-up
    resumed from the prompt's 64-grid snapshot with another C == a fresh prefill with a third C."""

    from test_cindep_patches import _cold, _gen, _sampling

    s = _sampling(sampling)
    rng = np.random.default_rng(2450 + [None, "2", "f3", "auto:1:1:0"].index(policy))
    p1 = [int(t) for t in rng.integers(0, 1000, size=700)]
    kn = {"b12x": 3}
    eb.cache = []
    r1, st1 = _gen(eb, p1, s, policy=policy, knobs=dict(kn, prefill_rows=8192))
    assert st1["tf_knobs"]["b12x"] == 3 and st1.get("b12x") == 3
    assert r1 == _cold(eb, p1, s, knobs=dict(kn, prefill_rows=1024))[:len(r1)]
    _gen(eb, p1, s, policy=policy, knobs=dict(kn, prefill_rows=8192))
    p2 = p1 + r1 + [int(t) for t in rng.integers(0, 1000, size=90)]
    warm, st2 = _gen(eb, p2, s, policy=policy, knobs=dict(kn, prefill_rows=64))
    assert st2["cached"] > 0
    assert warm == _cold(eb, p2, s, knobs=dict(kn, prefill_rows=4096))


@gpu
def test_engine_snapshots_never_cross_bits(eb):
    """A b12x snapshot is not resumed by a request with other bits, and the reverse: the tags differ. (A prefill
    drops the snapshots past its own resume point, so each direction starts from its own first request.)"""

    from test_cindep_patches import _gen

    assert eb._grid(True, b12x=3) == eb._grid(True, b12x=0) + 12
    assert eb._grid(True, b12x=4) == eb._grid(True, b12x=0)            # no latent cache: bit 4 changes nothing
    rng = np.random.default_rng(2460)
    p1 = [int(t) for t in rng.integers(0, 1000, size=300)]
    for first, second in ((3, 0), (0, 3), (3, 1)):
        eb.cache = []
        r1, _ = _gen(eb, p1, None, knobs={"b12x": first, "prefill_rows": 1024})
        p2 = p1 + r1 + [5, 6, 7]
        _, st = _gen(eb, p2, None, knobs={"b12x": second, "prefill_rows": 1024})
        assert st["cached"] == 0, (first, second)
    eb.cache = []
    r1, _ = _gen(eb, p1, None, knobs={"b12x": 3, "prefill_rows": 1024})
    _, st = _gen(eb, p1 + r1 + [5, 6, 7], None, knobs={"b12x": 3, "prefill_rows": 64})
    assert st["cached"] > 0                                              # control: same bits resume


@gpu
def test_engine_knob_echoed_and_restored(eb):
    from test_cindep_patches import _gen

    from tensorfold.families.glm5_next.cuda import b12xpf, glue, latent

    before = (b12xpf.BITS, glue.MHC, latent.SPARSE_ONE)
    _, stats = _gen(eb, list(range(100)), None, knobs={"b12x": 7})
    assert stats["tf_knobs"]["b12x"] == 7
    assert (b12xpf.BITS, glue.MHC, latent.SPARSE_ONE) == before == (b12xpf.default(), False, False)
