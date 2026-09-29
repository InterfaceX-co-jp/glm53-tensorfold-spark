"""patches/0550 on the GPU (docs/MEMORY-SAFETY.md): the selection scratch gives the same bits, and it removes the
caching allocator's growth that dipped MemAvailable at the end of long lone prefills (W15).

- Bits: ``select_pools_blocked`` with the scratch (``grow``, ``max``; junk-filled, grown across calls, too small ->
  fallback) == without it (``off``) == the sorted path, at production shapes (512-row sub-blocks, a 1M-token slot's
  262,148 pools, positions 70k / 300k / 600k / 1M, the 524,288 bucket boundary).
- The mechanism, measured: 512-row selections at the positions of a 100k -> 230k prefill (one call a 4,096-token
  piece) with ``off``: ``torch.cuda.memory_reserved`` grows by at least half the no-reuse bound
  (``memsafe.keys_growth``); with ``grow`` and ``reserve_for_prefill`` first: by at most the scratch + 32 MiB.
- ``torch.cuda.graph`` empties the allocator's cache before a capture (why MemFree came back when the needle's first
  MTP draft captured a new pool bucket); ``memsafe.trim_torch`` frees an unused cache over its limit.
- Engines: replies with the scratch on (``grow`` / ``max``) and the allocator trimmed before every chunk equal replies
  with ``off``, serial and drafted, past 2,051 tokens (TensorFold's synthetic checkpoint, test_1m_patches' fixtures).

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda pytest -q tests/cuda/test_memory_safety_patches.py
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.glm5_next.cuda import memsafe, sparse  # noqa: E402
from test_1m_patches import SAMPLINGS, IDS, _generate, _inputs, _sorted_path, ckpt, engines  # noqa: E402,F401

CAP = 262_148
MiB = 1 << 20


@pytest.fixture
def mode():
    saved = sparse.SCRATCH
    sparse._SCRATCH.clear()
    yield lambda m: setattr(sparse, "SCRATCH", m)
    sparse.SCRATCH = saved
    sparse._SCRATCH.clear()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("ties", ["random", "ties", "nan"])
@pytest.mark.parametrize("pos,R,np_", [(2020, 64, 1024), (70_000, 512, CAP), (300_000, 512, CAP),
                                       (524_288 - 512, 512, CAP), (524_288, 512, CAP), (1_048_000, 512, CAP)])
def test_scratch_same_bits(mode, pos, R, np_, ties):
    pk, qi, wts = _inputs(R, np_, pos + R, ties)
    pos_dev = torch.tensor([pos], dtype=torch.int32, device="cuda")
    npm = sparse.pool_bucket(pos + R, np_) if pos + R <= 4 * np_ else np_
    mode("off")
    ref = sparse.select_pools_blocked(qi, wts, pk, pos, R, npm, pos_dev)
    with _sorted_path():
        ref_t, ref_c = sparse.select_tokens(qi, wts, pk, pos, R, npm, pos_dev)
    t, c = sparse._tokens(ref, pos, R, qi.device)
    assert torch.equal(t, ref_t) and torch.equal(c, ref_c)
    for m in ("grow", "max"):
        mode(m)
        sparse._SCRATCH.clear()
        sparse.reserve_scratch(1, qi.device)                     # 16 MiB: too small at 300k+ (fallback growth)
        for buf in sparse._SCRATCH.values():
            buf.fill_(0x7F7F7F7F)
        assert torch.equal(sparse.select_pools_blocked(qi, wts, pk, pos, R, npm, pos_dev), ref), m
        sparse.reserve_for_prefill(pos, pos + R, R, pk, qi.device)
        for buf in sparse._SCRATCH.values():
            buf.fill_(-1)
        before = dict(sparse.SCRATCH_STATS)
        assert torch.equal(sparse.select_pools_blocked(qi, wts, pk, pos, R, npm, pos_dev), ref), m
        assert sparse.SCRATCH_STATS["fallback"] == before["fallback"]


def _sweep(p0: int, p1: int, step: int = 4096, R: int = 512):
    pk, qi, wts = _inputs(R, CAP, 7, "random")
    for p in range(p0, p1, step):
        pos_dev = torch.tensor([p], dtype=torch.int32, device="cuda")
        sparse.select_pools_blocked(qi, wts, pk, p, R, sparse.pool_bucket(p + R, CAP), pos_dev)
    torch.cuda.synchronize()
    return pk


def test_growth_without_scratch_and_none_with_it(mode):
    p0, p1 = 100_000, 230_000
    mode("off")
    _sweep(2048, 4096)                                   # kernels compiled, inputs allocated once
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_reserved()
    _sweep(p0, p1)
    grown_off = torch.cuda.memory_reserved() - base
    bound = sum(memsafe.keys_growth(p, p + 512, cached=())[0] for p in range(p0, p1, 4096))   # one call a piece
    assert bound > 0 and grown_off >= 0.5 * bound, (grown_off / MiB, bound / MiB)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    mode("grow")
    pk = _sweep(2048, 4096)
    got = sparse.reserve_for_prefill(p0, p1, 512, pk, pk.device)
    assert got >= memsafe.select_peak(p1, 512, CAP, sparse.SELECT_MB, p0)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_reserved()
    before = dict(sparse.SCRATCH_STATS)
    _sweep(p0, p1)
    grown_on = torch.cuda.memory_reserved() - base
    assert grown_on <= 32 * MiB + 128 * MiB, grown_on / MiB     # inputs' temporaries only, no key block
    assert sparse.SCRATCH_STATS == before
    print(f"reserved growth over {p0}-{p1}: off {grown_off / MiB:.0f} MiB (bound {bound / MiB:.0f}), "
          f"scratch {grown_on / MiB:.0f} MiB + {got / MiB:.0f} MiB held")


def test_graph_capture_empties_the_cache():
    torch.cuda.synchronize()
    x = torch.empty((256 * MiB,), dtype=torch.uint8, device="cuda")
    del x
    held = torch.cuda.memory_reserved()
    y = torch.zeros((4,), device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y.add_(1)
    assert torch.cuda.memory_reserved() <= held - 256 * MiB + 32 * MiB


def test_trim_frees_an_unused_cache():
    torch.cuda.synchronize()
    x = torch.empty((512 * MiB,), dtype=torch.uint8, device="cuda")
    del x
    t = memsafe.Trimmer(limit=256 * MiB, interval=0)
    assert memsafe.trim_torch(t) >= 512 * MiB and t.trims == 1
    assert memsafe.trim_torch(t) == 0


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
@pytest.mark.parametrize("context,rows,n", [(4096, 64, 2600), (8192, 512, 6000)])
def test_replies_equal_with_scratch_and_trims(engines, mode, monkeypatch, context, rows, n, sampling):
    eng = engines(context, rows, True)
    prompt = list(np.random.default_rng(n + 5).integers(0, 1000, size=n))
    mode("off")
    ref, _ = _generate(eng, prompt, sampling, draft=False, tokens=32)
    drafted_ref, _ = _generate(eng, prompt, sampling, policy="auto", tokens=32)
    assert drafted_ref == ref
    from tensorfold.families.glm5_next.cuda import decode

    real = decode.stage
    trimmer = memsafe.Trimmer(limit=1, interval=0)            # every chunk: empty the cache first

    def trimmed(*a, **k):
        memsafe.trim_torch(trimmer)
        return real(*a, **k)

    monkeypatch.setattr(decode, "stage", trimmed)
    for m in ("grow", "max"):
        mode(m)
        got, _ = _generate(eng, prompt, sampling, draft=False, tokens=32)
        assert got == ref, m
        drafted, _ = _generate(eng, prompt, sampling, policy="auto", tokens=32)
        assert drafted == ref, m
        assert sparse.scratch_bytes() > 0
