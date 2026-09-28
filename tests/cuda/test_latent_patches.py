"""patches/0060 (GLM53_TF_LATENT_KV=1) on TensorFold's synthetic GLM checkpoint (one GPU playing rank 0 of two).

Latent (absorbed) MLA: the DSA layers cache the kv_lora-wide latent (keys = values, one row a token) instead of
per-head keys and values, and attend in latent space (``latent.py``). The arithmetic is new, so replies differ
from the expanded engine's in the last bits; within the latent configuration every engine guarantee must hold.
Checked here:

- kernels on the real model's shapes (32 local heads, head dim 256, latent 512), 4-bit and BF16 kv_b: latent
  attention against the expanded kernel and an fp32 reference (bf16 tolerance); a window row gets the bits of the
  serial step at its position (dense and sparse); a captured-graph-style call (every chunk) equals exact chunks;
- the engine with the flag on: its caches are latent (and the bytes a token), drafted replies equal serial ones for
  every policy, a resumed prompt equals a fresh prefill, 64- and 512-row prefill chunks leave the same state bit for
  bit, past 2,051 tokens too (sparse top-k over the latent), and batched rounds (patches/0030) equal lone requests;
- latent against expanded on the same checkpoint: the prefill's hidden rows and logits agree to bf16 tolerance.

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda pytest -q tests/cuda/test_latent_patches.py
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from types import SimpleNamespace  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.glm5_next.cuda import latent, qmm, weights  # noqa: E402
from test_glm_engine import _TwoCopies, _checkpoint, _drafter, _generate  # noqa: E402
from test_patches import _index_heads_32  # noqa: E402

SAMPLINGS = [Sampling(1234, 1.0, 20, 0.95), None]
IDS = ["sampled", "greedy"]
DEV = "cuda"


@pytest.fixture(autouse=True)
def _no_lookup(monkeypatch):
    """patches/0020's lookup rounds would make the policies' windows depend on repeats; pin the MTP/DFlash2 arms."""

    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")


def _engine(path, *, latent_kv: bool = True, rows: int = 64, context: int = 0, batch: int = 1, nonexpert=None,
            long_graphs: bool = True):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GLM53_TF_LATENT_KV", "1" if latent_kv else "0")
        mp.setenv("GLM53_TF_LONGCTX_GRAPHS", "1" if long_graphs else "0")          # patches/0050
        mp.setenv("GLM53_TF_PREFILL_ROWS", str(rows))
        mp.setenv("GLM53_TF_BATCH", str(batch))
        if nonexpert is not None:
            mp.setattr(weights, "NONEXPERT", nonexpert)
        return GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", context=context,
                         comm=_TwoCopies())


# -- kernels on the real shapes -----------------------------------------------------------------------------------
H, DQ, L = 32, 256, 512


def _kvb(kind: str, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    wk = (torch.randn((H * DQ, L), generator=g) * 0.05).to(torch.bfloat16).to(DEV)
    wv = (torch.randn((H * DQ, L), generator=g) * 0.05).to(torch.bfloat16).to(DEV)
    if kind == "q4":
        return qmm.quantize4(wk), qmm.quantize4(wv)
    return qmm.make_b16(wk), qmm.make_b16(wv)


def _scratch(rows: int, capacity: int) -> latent.Scratch:
    cfg = SimpleNamespace(heads=2 * H, kv_lora=L, v_dim=DQ, dense_limit=2051)
    return latent.Scratch(SimpleNamespace(cfg=cfg, world=2), rows, capacity, DEV)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm())


@pytest.mark.parametrize("kind", ["q4", "b16"])
@torch.no_grad()
def test_latent_kernels_match_expanded(kind):
    """Latent attention (absorb, attend over the latent, expand) against the engine's expanded kernel on the same
    kv_b and latents, and both against an fp32 reference: all within bf16 rounding."""

    from tensorfold.families.glm5_next.cuda.attention import AttnScratch, attention, kv_write

    kv_k, kv_v = _kvb(kind, 1)
    cap, P, R = 3000, 1900, 6
    g = torch.Generator(device="cpu").manual_seed(2)
    lat = torch.randn((cap, L), generator=g).to(torch.bfloat16).to(DEV)
    q = (torch.randn((R, H, DQ), generator=g) * 2).to(torch.bfloat16).to(DEV)
    scale = DQ ** -0.5
    pos = torch.tensor([P], dtype=torch.int32, device=DEV)
    ref = latent.expanded_reference(q, lat[:P + R], kv_k, kv_v, scale, [P + r for r in range(R)])
    # expanded: keys and values through the engine's projections, cache, kernel
    kn = qmm.matmul(lat, kv_k).view(cap, H, DQ)
    vn = qmm.matmul(lat, kv_v).view(cap, H, DQ)
    kc, vc = torch.zeros_like(kn), torch.zeros_like(vn)
    kc[:P], vc[:P] = kn[:P], vn[:P]
    kv_write(kn[P:P + R].contiguous(), vn[P:P + R].contiguous(), kc, vc, pos)
    oe = attention(q, kc, vc, pos, AttnScratch(8, H, DQ, cap, DEV), scale=scale, nch=-(-(P + R) // 512))
    oe = oe.reshape(R, -1).float()
    # latent
    lc = torch.zeros((cap, L), dtype=torch.bfloat16, device=DEV)
    lc[:P] = lat[:P]
    latent.latent_write(lat[P:P + R].contiguous(), lc, pos)
    assert torch.equal(lc[:P + R], lat[:P + R])
    s = _scratch(8, cap)
    qa = latent.absorb(q, kv_k, s.qa[:R])
    u = latent.attention_latent(qa, lc, pos, s, scale=scale, nch=-(-(P + R) // 512))
    ol = latent.expand(u, kv_v, s.o[:R]).float()
    e_exp, e_lat, e_both = _rel(oe, ref), _rel(ol, ref), _rel(ol, oe)
    assert e_exp < 1e-2 and e_lat < 1e-2 and e_both < 1.5e-2, (e_exp, e_lat, e_both)
    assert e_lat < 2 * e_exp + 2e-3, (e_exp, e_lat)          # no worse than the expanded path's own rounding


@pytest.mark.parametrize("kind", ["q4", "b16"])
@torch.no_grad()
def test_latent_window_rows_equal_serial_rows(kind):
    """Row invariance at kernel level: each row of a window (dense chunks, and sparse top-k lists) has the bits of
    that row computed alone at its position, with the rest of the cache holding other values; a graph-style call
    (every chunk the scratch holds) has the bits of exact chunk counts."""

    kv_k, kv_v = _kvb(kind, 3)
    cap, P, R = 2600, 1500, 8
    g = torch.Generator(device="cpu").manual_seed(4)
    lat = torch.randn((cap, L), generator=g).to(torch.bfloat16).to(DEV)
    q = torch.randn((R, H, DQ), generator=g).to(torch.bfloat16).to(DEV)
    scale = DQ ** -0.5
    s = _scratch(16, cap)
    qa = latent.absorb(q, kv_k, s.qa[:R]).clone()
    for r in range(R):                                   # absorption and expansion: rows alone
        one = latent.absorb(q[r:r + 1].contiguous(), kv_k, s.qa[:1])
        assert torch.equal(one[0], qa[r])
    pos = torch.tensor([P], dtype=torch.int32, device=DEV)
    uw = latent.attention_latent(qa, lat, pos, s, scale=scale, nch=-(-(P + R) // 512)).clone()
    ug = latent.attention_latent(qa, lat, pos, s, scale=scale).clone()      # every chunk (a captured graph's call)
    assert torch.equal(uw, ug)
    ow = latent.expand(uw, kv_v, s.o[:R]).clone()
    for r in range(R):
        other = lat.clone()
        other[P + r + 1:] = torch.randn((cap - P - r - 1, L), generator=g).to(torch.bfloat16).to(DEV)
        pr = torch.tensor([P + r], dtype=torch.int32, device=DEV)
        u1 = latent.attention_latent(qa[r:r + 1].contiguous(), other, pr, s, scale=scale, nch=-(-(P + r + 1) // 512))
        assert torch.equal(u1[0], uw[r]), r
        o1 = latent.expand(u1.clone(), kv_v, s.o[:1])
        assert torch.equal(o1[0], ow[r]), r
    # sparse: per-row token lists (ascending, -1 padded), rows with count 0 left untouched
    W = 2051
    tokens = torch.full((R, W), -1, dtype=torch.int32)
    counts = torch.zeros((R,), dtype=torch.int32)
    for r in range(1, R, 2):
        n = 900 + 80 * r                                   # <= the P + r + 1 visible tokens
        tokens[r, :n] = torch.sort(torch.randperm(P + r + 1, generator=g)[:n]).values.to(torch.int32)
        counts[r] = n
    tokens, counts = tokens.to(DEV), counts.to(DEV)
    us = uw.clone()
    latent.sparse_latent(qa, lat, tokens, counts, us, scale)
    for r in range(R):
        if counts[r] == 0:
            assert torch.equal(us[r], uw[r])
            continue
        u1 = torch.zeros((1, H, L), dtype=torch.float32, device=DEV)
        latent.sparse_latent(qa[r:r + 1].contiguous(), lat, tokens[r:r + 1].contiguous(), counts[r:r + 1].contiguous(),
                             u1, scale)
        assert torch.equal(u1[0], us[r]), r
        idx = tokens[r, :counts[r]].long()
        p = torch.softmax(qa[r].float() @ lat[idx].float().t() * scale, dim=-1)
        assert _rel(us[r], p @ lat[idx].float()) < 5e-3


def test_scratch_does_not_grow_with_capacity():
    """Dense attention never runs past the dense limit, so a 1M-token cache does not size the chunk partials."""

    small, big = _scratch(64, 2560), _scratch(64, 1_000_000)
    assert small.nch == big.nch == -(-(2051 + 64) // 512)


# -- the engine with GLM53_TF_LATENT_KV=1 -------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm_latent")
    _checkpoint(path / "model")
    _drafter(path / "dflash2")
    return path


@pytest.fixture(scope="module")
def ckpt_x(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm_latent_x")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


@pytest.fixture(scope="module")
def el(ckpt):
    return _engine(ckpt)


@pytest.fixture(scope="module")
def el512(ckpt):
    return _engine(ckpt, rows=512)


@pytest.fixture(scope="module")
def ex(ckpt):
    """The same checkpoint on the expanded caches (the default)."""

    return _engine(ckpt, latent_kv=False)


def test_latent_off_by_default(ex):
    assert not latent.on(ex.w) and ex.e.st.kc[0].dim() == 3 and ex.e.st.vc is not ex.e.st.kc


def test_latent_caches_and_bytes(el, ex):
    c = el.w.cfg
    st = el.e.st
    assert latent.on(el.w) and isinstance(el.e.buf.attn, latent.Scratch)
    assert st.vc is st.kc and st.kc[0].shape == (st.capacity, c.kv_lora) and st.mtp_vc is st.mtp_kc
    assert st.mtp_kc.shape == (st.capacity, c.kv_lora)
    n = len(st.kc) + 1                                                      # DSA layers and the MTP head's
    assert latent.kv_bytes_per_token(st) == n * c.kv_lora * 2
    HL = c.heads // 2
    assert latent.kv_bytes_per_token(ex.e.st) == n * HL * (c.qk_dim + c.v_dim) * 2


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
def test_latent_drafted_equals_serial(el, sampling):
    prompt = list(np.random.default_rng(5).integers(0, 1000, size=37))
    serial, stats = _generate(el, prompt, sampling, draft=False, tokens=32)
    assert len(serial) == 32 and stats["drafts"] is False
    for policy in (None, "auto", "auto:1:1:0", "1", "2", "3", "7", "f3", "fc5:0.3", "c3:0.35", "a:0.6:0.85"):
        drafted, stats = _generate(el, prompt, sampling, policy=policy, tokens=32)
        assert drafted == serial, policy
        assert stats["rounds"] >= 1 and stats["min_rows"] >= 2, (policy, stats)


@pytest.mark.parametrize("sampling", [Sampling(7, 1.0, 20, 0.95), None], ids=IDS)
def test_latent_resumed_equals_fresh(el, sampling):
    rng = np.random.default_rng(9)
    first = list(rng.integers(0, 1000, size=70))              # more than one 64-row prefill chunk
    unrelated = list(rng.integers(0, 1000, size=12))
    reply, _ = _generate(el, first, sampling)
    after = first + reply + [5, 6, 7]
    warm, stats = _generate(el, after, sampling)
    assert stats["cached"] >= len(first) + len(reply) - 1
    _generate(el, unrelated, sampling)
    cold, stats = _generate(el, after, sampling)
    assert stats["cached"] == 0 and warm == cold
    serial, _ = _generate(el, after, sampling, draft=False)
    assert serial == cold


def _prefill_state(eng, prompt) -> list[torch.Tensor]:
    from tensorfold.families.glm5_next.cuda.decode import prefill

    e = eng.e
    first = prefill(e, list(prompt), None, mtp=True)
    st = e.st
    n = len(prompt)
    last_logits = qmm.matmul(e.last_hidden, eng.w.head).clone()          # the last prompt row's logits
    out = [torch.tensor([first]), st.rec[st.cur[0]].clone(), st.conv.clone(), e.last_hidden.clone(), last_logits]
    out += [k[:n].clone() for k in st.kc] + [st.mtp_kc[:st.mtp_len].clone()]
    eng.cache = []
    return out


@torch.no_grad()
def test_latent_prefill_chunk_sizes_same_bits(el, el512):
    """64- and 512-row prefill chunks leave the same committed state (KDA, conv, latent caches, MTP latent cache)."""

    prompt = list(np.random.default_rng(21).integers(0, 1000, size=700))
    a, b = _prefill_state(el, prompt), _prefill_state(el512, prompt)
    assert len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


@torch.no_grad()
def test_latent_close_to_expanded(el, ex):
    """Quality hook: on the same checkpoint the latent engine's prefill (the last row's final-normed hidden row and
    its logits) agrees with the expanded engine's to bf16 tolerance; only attention's arithmetic differs."""

    prompt = list(np.random.default_rng(23).integers(0, 1000, size=300))
    a, b = _prefill_state(el, prompt), _prefill_state(ex, prompt)
    hidden_rel, logit_rel = _rel(a[3], b[3]), _rel(a[4], b[4])
    assert hidden_rel < 3e-2 and logit_rel < 5e-2, (hidden_rel, logit_rel)
    top_a = torch.topk(a[4].float(), 5).indices
    top_b = torch.topk(b[4].float(), 5).indices
    assert int(top_a[0, 0]) in top_b[0].tolist()


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
def test_latent_exl3_bf16_kvb_drafted_equals_serial(ckpt_x, sampling):
    """EXL3 checkpoint: kv_b stays BF16 (GLM53_TF_NONEXPERT=bf16), the kernels' BF16 tile path."""

    e = _engine(ckpt_x, nonexpert="bf16")
    assert isinstance(e.w.layers[1].dsa.kv_k, qmm.B16)
    prompt = list(np.random.default_rng(8).integers(0, 1000, size=45))
    serial, _ = _generate(e, prompt, sampling, draft=False, tokens=24)
    for policy in (None, "auto:1:1:0", "f3", "2", "c3:0.35"):
        drafted, _ = _generate(e, prompt, sampling, policy=policy, tokens=24)
        assert drafted == serial, policy


# -- past 2,051 tokens: sparse top-k over the latent ----------------------------------------------------------------
@pytest.fixture(scope="module")
def long_ckpt(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm_latent_long")
    _checkpoint(path / "model", exl3=True)
    _index_heads_32(path / "model")
    _drafter(path / "dflash2")
    return path


@torch.no_grad()
def test_latent_long_block_paths(long_ckpt):
    """DSA block on the latent past the dense limit, on random caches: a window's rows equal serial rows (a window
    straddling 2,051 and one wholly past it), and patches/0050's capturable device-side selection (``npb``) equals
    the eager host-position path, window and single row."""

    from tensorfold.families.glm5_next.cuda import sparse
    from tensorfold.families.glm5_next.cuda.forward import State, dsa_block

    eng = _engine(long_ckpt, rows=64, context=4096, nonexpert="q4mse")
    w, b = eng.w, eng.e.buf
    assert latent.on(w) and b.lc is not None
    st = State(w, eng.e.st.capacity, eng.e.rows)
    layer = w.layers[1]
    lc, index = st.kc[0], st.index[0]
    g = torch.Generator(device="cpu").manual_seed(1)
    for t in (lc,) + tuple(index):
        t.copy_(torch.randn(t.shape, generator=g).to(t.dtype))
    snap = [t.clone() for t in (lc,) + tuple(index)]
    x = torch.randn((8, w.cfg.hidden), generator=g).to(torch.bfloat16).to(DEV)

    def restore():
        for t, s in zip((lc,) + tuple(index), snap):
            t.copy_(s)

    def run(P, rows, nch=None, host_pos=None, npb=None):
        R = len(rows)
        b.normed[:R] = x[rows]
        qmm.group_sums(b.normed[:R], b.xs[:R])
        st.set_pos(P)
        # rank 0's partial rows: dsa_block returns the gathered [world, R, D] (a reshape would mix ranks' rows)
        return dsa_block(layer, w, lc, lc, st.pos_dev, b, R, nch, index, host_pos, npb)[0].clone()

    pools = index[2].shape[0] - 2
    for P, R in ((2047, 8), (3000, 5)):
        restore()
        win = run(P, list(range(R)), -(-(P + R) // 512), P)
        restore()
        for r in range(R):
            one = run(P + r, [r], -(-(P + r + 1) // 512), P + r)
            assert torch.equal(one[0], win[r]), (P, r)
        if P >= w.cfg.dense_limit:
            restore()
            assert torch.equal(run(P, list(range(R)), npb=sparse.pool_bucket(P + R, pools)), win)
            restore()
            assert torch.equal(run(P, [0], npb=sparse.pool_bucket(P + 1, pools))[0], win[0])


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
def test_latent_sparse_long_context(long_ckpt, sampling):
    """A 3,000-token prompt crosses the dense limit: 64- and 512-row prefill chunks, serial and drafted decoding give
    the same reply, with patches/0050's long-context graphs and without; a follow-up resumed from the kept state
    equals a fresh prefill."""

    prompt = list(np.random.default_rng(31).integers(0, 1000, size=3000))
    e64 = _engine(long_ckpt, rows=64, context=4096, nonexpert="q4mse", long_graphs=False)
    a, _ = _generate(e64, prompt, sampling, draft=False, tokens=24)
    d, _ = _generate(e64, prompt, sampling, policy="2", tokens=24)
    assert a == d
    del e64
    torch.cuda.empty_cache()
    e = _engine(long_ckpt, rows=512, context=4096, nonexpert="q4mse")
    b, _ = _generate(e, prompt, sampling, draft=False, tokens=24)
    for policy in (None, "2", "f3"):
        c, _ = _generate(e, prompt, sampling, policy=policy, tokens=24)
        assert a == b == c, policy
    after = prompt + b + [3, 4]
    warm, stats = _generate(e, after, sampling, tokens=16)
    assert stats["cached"] >= len(prompt)
    _generate(e, [1, 2, 3], sampling, tokens=4)
    cold, stats = _generate(e, after, sampling, tokens=16)
    assert stats["cached"] == 0 and warm == cold


# -- batched rounds (patches/0030) ----------------------------------------------------------------------------------
@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
def test_latent_batched_equals_alone(ckpt, el, sampling):
    eb = _engine(ckpt, batch=2)
    assert eb.batch is not None and eb.batch.states[1].vc is eb.batch.states[1].kc
    rng = np.random.default_rng(41)
    pa, pb = [list(rng.integers(0, 1000, size=n)) for n in (37, 29)]
    want_a, _ = _generate(el, pa, sampling, draft=False, tokens=24)
    want_b, _ = _generate(el, pb, sampling, draft=False, tokens=24)
    for pol_a, pol_b in ((None, "2"), ("c3:0.35", "a:0.6:0.85"), ("0", "3")):
        (got_a, sa), (got_b, sb) = eb.batch.generate_batch([
            dict(prompt=pa, max_tokens=24, sampling=sampling, policy=pol_a),
            dict(prompt=pb, max_tokens=24, sampling=sampling, policy=pol_b)])
        assert got_a == want_a and got_b == want_b, (pol_a, pol_b)
        assert sa["batched_rounds"] >= 1 and sb["batched_rounds"] >= 1
