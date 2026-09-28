"""patches/0050 (long-context decode) on TensorFold's synthetic GLM checkpoint (one GPU playing rank 0 of two).

- Bounded scoring: scoring only the pools that exist (``sparse.pool_bucket``) selects the same tokens as scoring
  every pool the cache holds, and the device-side selection (``select_tokens_dev``: int64 keys + ``torch.topk``)
  selects the same tokens as the stable sort, ties included (duplicate pools, all-zero rows, -0.0 against +0.0).
- The device-side selection replays in a CUDA graph at other positions of its bucket.
- Past 2,051 tokens, a captured step (rows 1-8, both KDA parities, the MTP head) gives the logits of the eager
  upstream step (unbounded scoring), on its first (eager warm-up + capture) and later (replay) calls.
- Replies with GLM53_TF_LONGCTX_GRAPHS=1 equal upstream's (=0), serial and drafted, greedy and sampled, for prompts
  that start below the limit and cross it during decode, and prompts past it.

The long-context path runs here on the real model's 32 index heads (``_index_heads_32``). (Upstream's
``sparse._scores`` hardcoded H=32 and read past the synthetic checkpoint's 2-head rows; 0050 now takes the head
count from the tensors, which ``test_knob_patches`` relies on, and keeps upstream's code for 32 heads.)

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda pytest -q tests/cuda/test_longctx_patches.py
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.glm5_next.cuda import sparse, weights  # noqa: E402
from test_glm_engine import _TwoCopies, _checkpoint, _drafter, _generate  # noqa: E402
from test_patches import _index_heads_32  # noqa: E402

SAMPLINGS = [Sampling(1234, 1.0, 20, 0.95), None]
IDS = ["sampled", "greedy"]


# -- selection, kernel level ---------------------------------------------------------------------------------------

def _inputs(R: int, cap: int, seed: int, ties: bool):
    gen = torch.Generator().manual_seed(seed)
    np_cap = cap // 4
    pk = torch.randn((np_cap + 2, 128), generator=gen).to(torch.bfloat16)
    qi = torch.randn((R, 32 * 128), generator=gen).to(torch.bfloat16)
    wts = torch.randn((R, 160), generator=gen).to(torch.bfloat16)
    if ties:
        pk[1::3] = pk[0::3][:pk[1::3].shape[0]]          # every third pair of pools scores exactly alike
        qi[0] = 0                                          # row 0: every score 0 ...
        wts[0, 128:] = -wts[0, 128:].abs() - 0.01          # ... and -0.0 (all weights negative)
        if R > 1:
            qi[1] = 0                                      # row 1: every score +0.0 or -0.0 (mixed weight signs)
    return pk.cuda(), qi.cuda(), wts.cuda()[:, 128:]


@pytest.mark.parametrize("ties", [False, True], ids=["random", "ties"])
@pytest.mark.parametrize("pos,R,cap", [(2051, 1, 4104), (2500, 8, 4104), (4094, 8, 4104), (3500, 3, 8200),
                                       (5000, 8, 40000), (30000, 5, 40000)])
def test_bounded_and_device_selection_equal_unbounded(pos, R, cap, ties):
    pk, qi, wts = _inputs(R, cap, pos + R, ties)
    np_cap = cap // 4
    pos_dev = torch.tensor([pos], dtype=torch.int32, device="cuda")
    ref_t, ref_c = sparse.select_tokens(qi, wts, pk, pos, R, np_cap, pos_dev)        # upstream: every pool
    npb = sparse.pool_bucket(pos + R, np_cap)
    assert (pos + R) // 4 <= npb <= np_cap and npb >= sparse.TOPK_POOLS
    t, c = sparse.select_tokens(qi, wts, pk, pos, R, npb, pos_dev)                   # bounded, eager
    assert torch.equal(t, ref_t) and torch.equal(c, ref_c)
    sc = sparse.LongScratch(8, cap, 2, 256, "cuda")
    t, c = sparse.select_tokens_dev(qi, wts, pk, pos_dev, R, npb, sc)                # bounded, on the device
    assert torch.equal(t, ref_t) and torch.equal(c, ref_c)
    # the scores themselves: the same kernel over fewer pools gives the same bits for every pool it scores
    full = torch.empty((R, np_cap), dtype=torch.float32, device="cuda")
    import triton

    sparse._scores[(R, triton.cdiv(np_cap, 64))](qi, wts, wts.stride(0), pk, full, pos_dev, R, np_cap, 128 ** -0.5,
                                                H=32, D=128, BP=64, num_warps=4)
    assert torch.equal(sc.scores[:R * npb].view(R, npb), full[:, :npb])
    if ties:
        assert int(c[0]) > 0 and torch.equal(t[0, :2048].long() // 4, torch.arange(512, device="cuda")
                                            .repeat_interleave(4))                  # all ties: lowest pools


def test_device_selection_signed_zero_ties(monkeypatch):
    """Scores handed in directly: many exact ties, -0.0 among +0.0, -inf padding; keys order them as the stable
    descending sort does (-0.0 == +0.0, ties to the lower pool)."""

    R, pos, cap = 4, 6000, 12000
    np_cap = cap // 4
    npb = sparse.pool_bucket(pos + R, np_cap)
    g = torch.Generator(device="cuda").manual_seed(11)
    choice = torch.tensor([0.0, -0.0, 0.5, -0.5, 1.0, 1.0, 2.0, -3.0], device="cuda")
    scores = choice[torch.randint(0, choice.numel(), (R, npb), device="cuda", generator=g)]
    q = pos + torch.arange(R, device="cuda")
    scores = torch.where(torch.arange(npb, device="cuda")[None, :] < ((q + 1) // 4)[:, None], scores,
                         torch.full_like(scores, float("-inf")))

    class _K:
        def __getitem__(self, grid):
            return lambda qi, wts, ws, pk, out, *rest, **kw: out.copy_(scores)

    monkeypatch.setattr(sparse, "_scores", _K())
    qi = torch.zeros((R, 32 * 128), dtype=torch.bfloat16, device="cuda")
    wts = torch.zeros((R, 32), dtype=torch.bfloat16, device="cuda")
    pos_dev = torch.tensor([pos], dtype=torch.int32, device="cuda")
    ref_t, ref_c = sparse.select_tokens(qi, wts, None, pos, R, npb, pos_dev)
    t, c = sparse.select_tokens_dev(qi, wts, None, pos_dev, R, npb, sparse.LongScratch(8, cap, 2, 256, "cuda"))
    assert torch.equal(t, ref_t) and torch.equal(c, ref_c)


def test_device_selection_replays_in_a_graph():
    """Captured once, the selection follows the device position (and new queries) anywhere in its bucket."""

    R, cap, pos = 8, 40000, 9000
    np_cap = cap // 4
    npb = sparse.pool_bucket(pos + R, np_cap)
    pk, qi, wts = _inputs(R, cap, 5, False)
    pos_dev = torch.tensor([pos], dtype=torch.int32, device="cuda")
    sc = sparse.LongScratch(8, cap, 2, 256, "cuda")
    sparse.select_tokens_dev(qi, wts, pk, pos_dev, R, npb, sc)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        t, c = sparse.select_tokens_dev(qi, wts, pk, pos_dev, R, npb, sc)
    gen = torch.Generator().manual_seed(9)
    for p in (pos, pos + 37, 2 * npb + 3, 4 * npb - R):
        assert sparse.pool_bucket(p + R, np_cap) <= npb
        pos_dev.fill_(p)
        qi.copy_(torch.randn(qi.shape, generator=gen).to(torch.bfloat16))
        graph.replay()
        ref_t, ref_c = sparse.select_tokens(qi, wts, pk, p, R, np_cap, pos_dev)
        assert torch.equal(t, ref_t) and torch.equal(c, ref_c), p


def test_pool_buckets():
    for cap_pools in (640, 1026, 2050, 10002, 262146):
        seen = set()
        for end in range(2052, 4 * cap_pools + 1, 97):
            b = sparse.pool_bucket(end, cap_pools)
            assert end // 4 <= b <= cap_pools and b >= min(1024, cap_pools)
            seen.add(b)
        assert len(seen) <= 10                  # a handful of graphs a (rows, parity) up to 1M tokens


# -- engines ---------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def long_ckpt(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm_longctx")
    _checkpoint(path / "model", exl3=True)
    _index_heads_32(path / "model")
    _drafter(path / "dflash2")
    return path


@pytest.fixture(scope="module")
def engines(long_ckpt):
    """GlmEngine per (context, GLM53_TF_LONGCTX_GRAPHS), built on first use."""

    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    made: dict[tuple[int, bool], GlmEngine] = {}

    def get(context: int, on: bool):
        key = (context, on)
        if key not in made:
            m = pytest.MonkeyPatch()
            try:
                m.setattr(weights, "NONEXPERT", "q4mse")
                m.setenv("GLM53_TF_PREFILL_ROWS", "64")
                m.setenv("GLM53_TF_LONGCTX_GRAPHS", "1" if on else "0")
                made[key] = GlmEngine(long_ckpt / "model", rank=0, master="", port=0, drafter=long_ckpt / "dflash2",
                                      context=context, comm=_TwoCopies())
            finally:
                m.undo()
        return made[key]

    return get


@contextmanager
def _upstream(e):
    """Upstream's step on this engine: eager, every pool scored."""

    saved = e.graphs.long_rows, e.w.meta["longctx_bound"]
    e.graphs.long_rows, e.w.meta["longctx_bound"] = frozenset(), False
    try:
        yield
    finally:
        e.graphs.long_rows, e.w.meta["longctx_bound"] = saved


@pytest.mark.parametrize("context,n", [(4096, 2500), (4096, 4094), (8192, 3500), (8192, 7000)])
def test_long_graph_steps_equal_eager(engines, context, n):
    """Past 2,051 tokens: windows of 1-8 rows at both KDA parities, and MTP steps of 1-8 rows, through the
    long-context graphs (first call: eager warm-up + capture; second: replay) against the upstream eager step."""

    from tensorfold.families.glm5_next.cuda.decode import prefill
    from tensorfold.families.glm5_next.cuda.forward import commit

    eng = engines(context, True)
    e = eng.e
    rng = np.random.default_rng(n)
    prompt = [int(x) for x in rng.integers(0, 1000, size=n)]
    prefill(e, prompt, None, mtp=True)
    eng.cache = []                    # this prefill overwrote the caches the engine's snapshots point at
    toks = [int(x) for x in rng.integers(0, 1000, size=8)]
    hid = (torch.randn((8, e.w.cfg.hidden), generator=torch.Generator().manual_seed(n)) * 0.5).to(torch.bfloat16)
    hid = hid.cuda()
    cap = e.st.index[0][2].shape[0] - 2
    for _ in range(2):
        st = e.st
        for R in range(1, 9):
            with _upstream(e):
                ref = e.forward(toks[:R]).clone()
                ref_h = e.buf.fnormed[:R].clone()
            for _ in range(2):
                got = e.forward(toks[:R])
                assert torch.equal(got, ref) and torch.equal(e.buf.fnormed[:R], ref_h), (R, st.parity)
            assert ("main", R, st.parity, sparse.pool_bucket(st.pos + R, cap)) in e.graphs.long
        for k in range(1, 9):
            with _upstream(e):
                ref = e.mtp(toks[:k], hid[:k]).clone()
            for _ in range(2):
                assert torch.equal(e.mtp(toks[:k], hid[:k]), ref), k
            assert ("mtp", k, sparse.pool_bucket(st.mtp_len + k, cap)) in e.graphs.long
        e.forward(toks[:8])
        commit(e.w, st, e.buf, 8, 1)          # one more position, the other KDA parity


POLICIES = (None, "auto", "3", "c3:0.35", "a:0.6:0.85", "f7", "fc7:0.3", "l7")


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
@pytest.mark.parametrize("context,n", [(4096, 2040), (4096, 2500), (8192, 3500)])
def test_long_graph_replies_equal_upstream_and_serial(engines, context, n, sampling):
    """Serial replies with long-context graphs equal upstream's (GLM53_TF_LONGCTX_GRAPHS=0), and every drafted
    reply equals them (verify windows past 2,051 tokens give the bits of serial steps). n=2040 crosses the limit
    during decode (mixed windows run eagerly, later ones replay)."""

    off, on = engines(context, False), engines(context, True)
    assert off.e.graphs.long_rows == frozenset() and off.e.buf.lc is None
    prompt = list(np.random.default_rng(n + 1).integers(0, 1000, size=n))
    ref, _ = _generate(off, prompt, sampling, draft=False, tokens=32)
    got, _ = _generate(on, prompt, sampling, draft=False, tokens=32)
    assert len(ref) == 32 and got == ref
    for policy in POLICIES:
        drafted, _ = _generate(on, prompt, sampling, policy=policy, tokens=32)
        assert drafted == ref, policy
    keys = on.e.graphs.long
    assert any(k[0] == "main" for k in keys) and any(k[0] == "mtp" for k in keys)
    assert not off.e.graphs.long


def test_long_graph_resume_equals_fresh(engines):
    """A prompt that extends a reply decoded through the long-context graphs resumes like a fresh prefill."""

    on = engines(4096, True)
    prompt = list(np.random.default_rng(77).integers(0, 1000, size=2600))
    reply, _ = _generate(on, prompt, None, tokens=24)
    follow = prompt + reply + list(np.random.default_rng(78).integers(0, 1000, size=40))
    resumed, stats = _generate(on, follow, None, tokens=24)
    assert stats["cached"] >= len(prompt) + len(reply) - 1
    fresh, _ = _generate(on, follow, None, draft=False, tokens=24)       # a fresh prefill, serial decoding
    assert resumed == fresh
