"""patches/0065 (1M-token memory, fast prefill indexer) on TensorFold's synthetic GLM checkpoint (one GPU playing
rank 0 of two).

- Selection: prefill windows (more than 8 rows) select through ``sparse.select_pools_blocked`` (row blocks,
  tile-shared scoring, int32 keys, top-k threshold, no full sort); it must give exactly the tokens of the sorted
  path (``GLM53_TF_SELECT=sort``, upstream's stable sort over every pool of the bucket): random and tied scores,
  -0.0 against +0.0, NaN scores, rows crossing the dense limit, bucket boundaries, many rows, small row blocks.
  The row-block scoring kernel gives ``_scores``' fp32 bits (checked directly, beyond the load-time
  ``blocked_ok``).
- Index ring: the model layers' index keys and gates live in a ring (``State.index_ring``); pool keys written
  through a ring equal those written through full caches, and a snapshot restore far behind the ring's reach
  (the prompt snapshot after a long reply) resumes exactly like a fresh prefill.
- DFlash2 ring: the drafter's context in a ring of its sliding window gives the same block-pass bits as the linear
  cache (before and after the ring wraps, and after a rewind within reach).
- Engines: replies with every 0065 change on equal replies with all of them off (sort selection, full index
  caches, linear drafter cache), serial and drafted, greedy and sampled, past 2,051 tokens, 64- and 512-row
  prefill chunks; resumed == fresh.
- Memory: at a big capacity the capacity-scaled buffers are what the table in docs/MEMORY-1M.md says: rings do not
  grow with the capacity, the latent KV bytes a token drop the ring caches, and the eager selection's peak scratch
  at 1,024 rows and a 1M-token pool count stays within GLM53_TF_SELECT_MB (+ top-k temporaries).

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda pytest -q tests/cuda/test_1m_patches.py
"""

from __future__ import annotations

import json
import shutil
from contextlib import contextmanager

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import triton  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.glm5_next.cuda import dflash2, sparse, weights  # noqa: E402
from test_glm_engine import _TwoCopies, _checkpoint, _drafter, _generate  # noqa: E402
from test_patches import _index_heads_32  # noqa: E402

SAMPLINGS = [Sampling(1234, 1.0, 20, 0.95), None]
IDS = ["sampled", "greedy"]


@contextmanager
def _sorted_path():
    saved = sparse.SELECT
    sparse.SELECT = "sort"
    try:
        yield
    finally:
        sparse.SELECT = saved


def _inputs(R: int, np_: int, seed: int, ties: str, nh: int = 32):
    gen = torch.Generator().manual_seed(seed)
    pk = torch.randn((np_ + 2, 128), generator=gen).to(torch.bfloat16)
    qi = torch.randn((R, nh * 128), generator=gen).to(torch.bfloat16)
    wts = torch.randn((R, nh), generator=gen).to(torch.bfloat16)
    if ties == "ties":
        pk = pk[torch.randint(0, 7, (np_ + 2,), generator=gen)].contiguous()   # 7 distinct pools: huge ties
        qi[0] = 0                                                              # row 0: every score +-0.0
        wts[0] = -wts[0].abs() - 0.01
        if R > 1:
            qi[1] = 0
    elif ties == "nan":
        pk[5::97] = float("nan")                                               # some pools score NaN
    return pk.cuda(), qi.cuda(), wts.cuda()


# -- selection, kernel level ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("nh", [32, 4])
@pytest.mark.parametrize("pos,R,np_", [(2000, 100, 1024), (4000, 64, 1024), (9000, 300, 4096), (60000, 40, 16384)])
def test_row_block_scores_have_score_bits(pos, R, np_, nh):
    pk, qi, wts = _inputs(R, np_, pos, "random", nh)
    pos_dev = torch.tensor([pos], dtype=torch.int32, device="cuda")
    ref = torch.empty((R, np_), dtype=torch.float32, device="cuda")
    sparse._scores[(R, triton.cdiv(np_, 64))](qi, wts, wts.stride(0), pk, ref, pos_dev, R, np_, 128 ** -0.5, BP=64,
                                              num_warps=4, **sparse._heads(qi, wts))
    got = torch.empty_like(ref)
    for r0 in range(0, R, 48):                                  # row blocks, as select_pools_blocked runs them
        n = min(48, R - r0)
        sparse._launch_rows(qi, wts, pk, got[r0:], pos_dev, R, np_, r0, n, keys=False)
    assert torch.equal(ref.view(torch.int32), got.view(torch.int32))
    assert sparse.blocked_ok(qi.device, nh)


@pytest.mark.parametrize("ties", ["random", "ties", "nan"])
@pytest.mark.parametrize("pos,R,np_,rows", [
    (2020, 64, 1024, None),        # rows cross the dense limit (dense rows: count 0)
    (2045, 512, 1024, 16),         # 512 rows, 32 blocks of 16
    (3072, 1024, 1024, None),      # the last row sees exactly 1,024 pools: the bucket boundary
    (4090, 200, 2048, 48),
    (9000, 1024, 4096, 128),
    (70000, 300, 32768, None),
    (70000, 64, 262144, None),     # a 1M-token cache's pool count, scored up to the rows' own pools
])
def test_blocked_selection_equals_sorted(pos, R, np_, rows, ties):
    pk, qi, wts = _inputs(R, np_, pos + R, ties)
    pos_dev = torch.tensor([pos], dtype=torch.int32, device="cuda")
    with _sorted_path():
        ref_t, ref_c = sparse.select_tokens(qi, wts, pk, pos, R, np_, pos_dev)
    pools = sparse.select_pools_blocked(qi, wts, pk, pos, R, np_, pos_dev, rows=rows)
    t, c = sparse._tokens(pools, pos, R, qi.device)
    assert torch.equal(t, ref_t) and torch.equal(c, ref_c)
    t2, c2 = sparse.select_tokens(qi, wts, pk, pos, R, np_, pos_dev)          # the dispatch picks the blocked path
    assert torch.equal(t2, ref_t) and torch.equal(c2, ref_c)


def test_small_windows_keep_the_sorted_path(monkeypatch):
    calls = []
    real = sparse.select_pools_blocked
    monkeypatch.setattr(sparse, "select_pools_blocked", lambda *a, **k: calls.append(1) or real(*a, **k))
    pk, qi, wts = _inputs(8, 1024, 1, "random")
    pos_dev = torch.tensor([3000], dtype=torch.int32, device="cuda")
    sparse.select_tokens(qi, wts, pk, 3000, 8, 1024, pos_dev)
    assert not calls
    pk, qi, wts = _inputs(9, 1024, 1, "random")
    sparse.select_tokens(qi, wts, pk, 3000, 9, 1024, pos_dev)
    assert calls


def test_selection_scratch_is_bounded():
    """1,024 prefill rows against a 1M-token cache's pools: the sorted path would hold ~36 bytes an entry
    (1,024 x 250,000 x 36 = 9.2 GB); the blocked path its GLM53_TF_SELECT_MB of int32 keys and small top-k
    temporaries."""

    R, np_, pos = 1024, 250000, 998000
    pk, qi, wts = _inputs(R, np_, 3, "random")
    pos_dev = torch.tensor([pos], dtype=torch.int32, device="cuda")
    sparse.select_tokens(qi[:16], wts[:16], pk, pos, 16, np_, pos_dev)        # warm the kernels
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    sparse.select_tokens(qi, wts, pk, pos, R, np_, pos_dev)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    assert sparse.block_rows(R, np_) < R
    assert peak < (sparse.SELECT_MB << 20) * 2 + (64 << 20), peak / 2**20


# -- index ring ----------------------------------------------------------------------------------------------------

def test_index_ring_pool_keys_equal_full():
    """Windows of 1-60 rows, keeping a random prefix each time (verify-style), through a 128-row ring and through
    full caches: the committed pool keys are the same bits."""

    g = torch.Generator().manual_seed(5)
    cap = 3000
    full = [torch.zeros((cap, 128), dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    ring = [torch.zeros((128, 128), dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    pkf = torch.zeros((cap // 4 + 2, 128), dtype=torch.bfloat16, device="cuda")
    pkr = pkf.clone()
    lnw, lnb = (torch.randn(128, generator=g).to(torch.bfloat16).cuda() for _ in range(2))
    ape = torch.randn((4, 128), generator=g).to(torch.bfloat16).cuda()
    pos = 0
    while pos < 2900:
        R = int(torch.randint(1, 61, (1,), generator=g))
        kr = torch.randn((R, 160), generator=g).to(torch.bfloat16).cuda()
        gate = torch.randn((R, 128), generator=g).cuda()
        pd = torch.tensor([pos], dtype=torch.int32, device="cuda")
        sparse.index_update(kr[:, :128], gate, lnw, lnb, ape, full[0], full[1], pkf, pd)
        sparse.index_update(kr[:, :128], gate, lnw, lnb, ape, ring[0], ring[1], pkr, pd)
        pos += int(torch.randint(1, R + 1, (1,), generator=g))
    n = pos // 4
    assert torch.equal(pkf[:n].view(torch.int16), pkr[:n].view(torch.int16))


# -- DFlash2 ring --------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm_1m")
    _checkpoint(path / "model", exl3=True)
    _index_heads_32(path / "model")
    _drafter(path / "dflash2")
    small = path / "dflash2_w256"                 # the drafter with a 256-token window: its ring (512) wraps early
    shutil.copytree(path / "dflash2", small)
    cfg = json.loads((small / "config.json").read_text())
    cfg["sliding_window"] = 256
    (small / "config.json").write_text(json.dumps(cfg))
    return path


@pytest.fixture(scope="module")
def engines(ckpt):
    """GlmEngine per (context, prefill rows, 0065 on, drafter dir name), built on first use."""

    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    made: dict = {}

    def get(context: int, rows: int, on: bool, drafter: str = "dflash2"):
        key = (context, rows, on, drafter)
        if key not in made:
            m = pytest.MonkeyPatch()
            try:
                m.setattr(weights, "NONEXPERT", "q4mse")
                m.setenv("GLM53_TF_PREFILL_ROWS", str(rows))
                m.setattr(sparse, "INDEX_RING", on)
                m.setattr(dflash2, "RING_ON", on)
                made[key] = GlmEngine(ckpt / "model", rank=0, master="", port=0, drafter=ckpt / drafter,
                                      context=context, comm=_TwoCopies())
            finally:
                m.undo()
        return made[key]

    return get


def _feed(d, taps: torch.Tensor, pending: int):
    d.add_taps(taps)
    d._block_compute()
    return d.packed.clone(), d.proj.clone()


def test_drafter_ring_equals_linear(engines, ckpt, monkeypatch):
    """The same taps into a linear drafter cache and a 512-slot ring (256-token window): every block pass gives
    the same candidates and selector rows, before and after the ring wraps, and after a rewind within reach."""

    from tensorfold.families.glm5_next.cuda.dflash2 import Drafter

    w = engines(4096, 64, True).w
    monkeypatch.setattr(dflash2, "RING_ON", False)
    lin = Drafter(ckpt / "dflash2_w256", w, capacity=4104)
    monkeypatch.setattr(dflash2, "RING_ON", True)
    ring = Drafter(ckpt / "dflash2_w256", w, capacity=4104)
    assert not lin.ring and ring.ring and ring.cap == 512 and lin.cap == 4112
    g = torch.Generator().manual_seed(9)
    width = len(ring.tap_layers) * w.cfg.hidden
    pos = 0
    while pos < 1500:
        n = int(torch.randint(1, 65, (1,), generator=g))
        taps = (torch.randn((n, width), generator=g) * 0.5).to(torch.bfloat16).cuda()
        a, b = _feed(lin, taps, 7), _feed(ring, taps, 7)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), pos
        pos += n
        if pos > 700 and pos % 5 == 0:                  # a rewind of a few rows (within the ring's reach)
            for d in (lin, ring):
                d.pos_dev.sub_(3)
                d.context_end = d.context_end - 3
            pos -= 3
    assert int(ring.lo_dev) <= ring.context_end - ring.window and int(lin.lo_dev) == 0   # rewinds lost nothing
    for d in (lin, ring):                               # a rewind far behind: the ring masks what it lost
        d.pos_dev.fill_(100)
        d.context_end = 100
    assert int(ring.lo_dev) == 100 and int(lin.lo_dev) == 0
    taps = (torch.randn((40, width), generator=g) * 0.5).to(torch.bfloat16).cuda()
    got = _feed(ring, taps, 7)                          # a shorter context, but a well-defined one
    assert torch.isfinite(got[1]).all() and torch.isfinite(got[0][..., :ring.top_k]).all()


# -- engines -------------------------------------------------------------------------------------------------------

POLICIES = (None, "auto", "3", "c3:0.35", "f7", "fc7:0.3")


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
@pytest.mark.parametrize("context,rows,n", [(4096, 64, 2040), (4096, 64, 2600), (4096, 512, 3000),
                                            (8192, 512, 6000)])
def test_replies_equal_without_0065(engines, context, rows, n, sampling):
    """Serial replies with every 0065 change equal those without (sorted selection, full index caches, linear
    drafter cache), and every drafted reply equals them."""

    off, on = engines(context, rows, False), engines(context, rows, True)
    assert on.e.st.index_ring and not off.e.st.index_ring
    assert on.drafter.ring and not off.drafter.ring
    prompt = list(np.random.default_rng(n + rows).integers(0, 1000, size=n))
    with _sorted_path():
        ref, _ = _generate(off, prompt, sampling, draft=False, tokens=32)
    got, _ = _generate(on, prompt, sampling, draft=False, tokens=32)
    assert len(ref) == 32 and got == ref
    for policy in POLICIES:
        drafted, _ = _generate(on, prompt, sampling, policy=policy, tokens=32)
        assert drafted == ref, policy


@pytest.mark.parametrize("policy", ["3", "f7", "auto"])
def test_resume_behind_the_rings_equals_fresh(engines, policy):
    """The prompt snapshot resumed after a reply longer than the index ring (256 rows) and than the drafter's
    ring reach (small-window drafter): the committed tail comes back from ``conv`` and the reply equals a fresh
    prefill's."""

    eng = engines(4096, 64, True, "dflash2_w256")
    assert eng.e.st.index_ring == 256 and eng.drafter.ring
    rng = np.random.default_rng(31)
    first = [int(x) for x in rng.integers(0, 1000, size=2203)]        # 2,203: a snapshot inside a pool
    _generate(eng, first, None, policy=policy, tokens=700)            # reply: past both rings' reach
    after = first + [int(x) for x in rng.integers(0, 1000, size=37)]
    warm, stats = _generate(eng, after, None, policy=policy, tokens=24)
    assert stats["cached"] == len(first)
    _generate(eng, [int(x) for x in rng.integers(0, 1000, size=9)], None)
    cold, stats = _generate(eng, after, None, policy=policy, tokens=24)
    assert stats["cached"] == 0 and warm == cold
    serial, _ = _generate(eng, after, None, draft=False, tokens=24)
    assert serial == cold


# -- memory ----------------------------------------------------------------------------------------------------------

def test_memory_at_a_big_capacity(ckpt, monkeypatch):
    """A 1M-token engine state on the synthetic model: the rings do not grow with the capacity, the KV bytes a
    token are the latent caches, the pool keys and the MTP layer's index caches only, and the drafter holds its
    window, not the context."""

    from tensorfold.families.glm5_next.cuda import latent
    from tensorfold.families.glm5_next.cuda.dflash2 import Drafter
    from tensorfold.families.glm5_next.cuda.forward import State
    from tensorfold.families.glm5_next.cuda.weights import load

    monkeypatch.setattr(weights, "NONEXPERT", "q4mse")
    monkeypatch.setenv("GLM53_TF_LATENT_KV", "1")
    w = load(ckpt / "model", rank=0)
    w.meta.update(latent_kv=True, long_context=True)
    cap, rows = 1_000_008, 1024
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    st = State(w, cap, rows)
    after_state = torch.cuda.memory_allocated() - before
    c = w.cfg
    n_dsa = len(st.kc)
    ring = sparse.index_ring_rows(rows)
    assert st.index_ring == ring == 2048
    for ik, ig, pk in st.index[:n_dsa]:
        assert ik.shape[0] == ig.shape[0] == ring and pk.shape[0] == cap // 4 + 2
    mtp_ik = st.index[-1][0]
    assert mtp_ik.shape[0] == cap                                   # the MTP layer's stay full
    per = latent.kv_bytes_per_token(st)
    assert per == (n_dsa + 1) * c.kv_lora * 2 + 2 * c.index_dim * 2 + (n_dsa + 1) * c.index_dim * 2 // 4
    # everything the state holds that grows with the capacity is those bytes a token (+1% for the fixed parts)
    fixed = st.iring.numel() * 2 + st.conv.numel() * 2 + st.rec.numel() * 4 + st.proj.numel() * 2
    fixed += sum(t.numel() * t.element_size() for t in (st.scratch_set.out, st.scratch_set.k, st.scratch_set.v,
                                                        st.scratch_set.g, st.scratch_set.b))
    assert after_state - fixed <= per * cap * 1.01 + (8 << 20), (after_state, fixed, per * cap)
    monkeypatch.setattr(dflash2, "RING_ON", True)
    d = Drafter(ckpt / "dflash2", w, capacity=cap)
    assert d.ring and d.cap == dflash2.ring_slots(d.window, d.block) == 4096
    assert d.kc[0].shape[1] == 4096
