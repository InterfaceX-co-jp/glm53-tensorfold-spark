"""patches/0550 (docs/MEMORY-SAFETY.md): the prefill selection's scratch, the caching-allocator trim, and admission /
store decisions that count reclaimable page cache. Host only (the selection kernels in Triton's CPU interpreter).

- ``memsafe``'s size model: ``pool_bucket`` / ``block_rows`` equal ``sparse``'s; ``select_blocks`` equals the key blocks
  ``sparse.select_pools_blocked`` really allocates (recorded, interpreter); the no-reuse growth bound ``keys_growth``
  against W15's needle traces (0.75x of the bound at 298k and 314k), its quadratic shape and its 1M value; the scratch
  (``scratch_bytes``) bounded by GLM53_TF_SELECT_MB at every context up to 1,048,576, and admission's reservation;
- the scratch gives the same bits: ``select_pools_blocked`` with the scratch (``grow`` / ``max``, junk-filled, grown
  across calls, too small = fallback) == without it (``off``) == the sorted path, on random, tied, -0.0 and NaN scores,
  rows crossing 2,051 and pool buckets, several row blocks; no key block is allocated once reserved;
- page cache: ``parse_meminfo``, the credit (Dirty / Writeback / Mapped / keep), ``MemView``; admission: W15's copy
  (MemFree 2.1 GiB, 16.8 GiB cache) admits with ``available`` and waits with ``free`` (the old rule, reproduced exactly
  on a grid of values); no page cache and little free: both wait; the free-now floor; the wait / resume log lines;
  ``Batcher._mem_ok`` and ``SessionStore._can_grow`` on fakes;
- ``Trimmer``'s trigger and hysteresis; ``drop_file_cache`` on a real file.

Run: TRITON_INTERPRET=1 PYTHONPATH=<patched TensorFold>/src pytest -q tests/test_memory_safety.py
"""

from __future__ import annotations

import io
import os
import random
from types import SimpleNamespace

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402

from tensorfold.families.glm5_next.cuda import memsafe  # noqa: E402

GiB, MiB = memsafe.GiB, memsafe.MiB
CAP_1M = 1_048_592 // 4          # a 1,048,592-token slot's pools (the production pool's slots)


# -- the size model ------------------------------------------------------------------------------------------------
def test_model_matches_sparse_rules():
    sparse = pytest.importorskip("tensorfold.families.glm5_next.cuda.sparse")
    rng = random.Random(1)
    for _ in range(3000):
        cap = rng.choice([1024, 4096, 65536, 262_148, CAP_1M])
        end = rng.randrange(0, cap * 4 + 1)
        assert memsafe.pool_bucket(end, cap) == sparse.pool_bucket(end, cap)
        R = rng.choice([9, 16, 64, 512, 1024, 4096])
        np_max = rng.randrange(1, cap + 1)
        assert memsafe.block_rows(R, np_max, sparse.SELECT_MB) == sparse.block_rows(R, np_max)
    assert (memsafe.POOL, memsafe.TOPK_POOLS, memsafe.BRB, memsafe.MIN_BUCKET) == \
        (sparse.POOL, sparse.TOPK_POOLS, sparse.BRB, sparse.MIN_BUCKET)


def test_select_blocks_production_shapes():
    # 512-row sub-blocks (LEAN_BLOCK): one block of 512 x npc up to 524,288 tokens, then two of 256 rows
    assert memsafe.select_blocks(131_072 - 512, 512, CAP_1M) == [512 * 32768 * 4]
    b = memsafe.select_blocks(314_304, 512, CAP_1M)
    assert b == [512 * (-(-(314_304 + 512) // 4 // 64) * 64) * 4]
    assert b[0] / MiB == pytest.approx(153.75, abs=0.5)
    assert memsafe.select_blocks(524_288 - 512, 512, CAP_1M) == [256 * MiB]
    two = memsafe.select_blocks(1_048_576 - 512, 512, CAP_1M)
    assert len(two) == 2 and max(two) == 256 * MiB
    # every block of every sub-block to 1M stays within GLM53_TF_SELECT_MB
    assert memsafe.select_peak(1_048_576) == 256 * MiB
    assert memsafe.select_peak(1_048_576, select_mb=128) <= 128 * MiB


def test_select_peak_fast_equals_scan():
    # admission sizes a waiting prompt every round: only the last call of each pool bucket and the last (full) call
    rng = random.Random(3)
    for _ in range(4000):
        cap = rng.choice([1024, 4096, 65536, 262_148, CAP_1M])
        R = rng.choice([16, 64, 500, 512, 1024, 4096])
        p1 = rng.randrange(0, min(cap * 4, 1_048_576) + 1)
        p0 = rng.randrange(0, p1 + 1) if rng.random() < 0.5 else 0
        mb = rng.choice([64, 256])
        assert memsafe.select_peak(p1, R, cap, mb, p0) == memsafe.select_peak_scan(p1, R, cap, mb, p0), (cap, R, p0, p1)


def test_keys_growth_explains_w15_needles():
    # no scratch: the allocator's reserved growth (no-reuse bound) over the needles' prefills
    # B7b: flat for the first ~46% (~140k: the cache already held blocks that large), then the growth
    g_b7b = memsafe.keys_growth(0, 298_388)[0] - memsafe.keys_growth(0, 140_000)[0]
    g_gate = memsafe.keys_growth(0, 314_305)[0]
    # measured MemAvailable drops (results/W15): B7b 12.3 -> 9.3 (3.0 GiB), gated 10.9 -> 6.4 (4.5 GiB)
    assert 3.0 * GiB / g_b7b == pytest.approx(0.75, abs=0.05)
    assert 4.5 * GiB / g_gate == pytest.approx(0.78, abs=0.05)
    # quadratic: doubling the prompt ~4x the growth (up to 524k, then the 256 MiB segment serves every block)
    g128, g256, g512 = (memsafe.keys_growth(0, p)[0] for p in (131_072, 262_144, 524_288))
    assert 3.5 < g256 / g128 < 4.5 and 3.5 < g512 / g256 < 4.5
    assert memsafe.keys_growth(0, 1_048_576)[0] == g512
    assert g512 / GiB == pytest.approx(16.0, abs=0.2)
    # a cache that already holds a segment at least as large as every block: no growth
    assert memsafe.keys_growth(0, 300_000, cached=(256 * MiB,))[0] == 0
    # slope: MiB of new segments per 4,096 tokens grows with the position (W15: 0.6 -> 1.9-2.5 GiB a minute)
    s1 = memsafe.keys_growth(0, 200_000 + 4096)[0] - memsafe.keys_growth(0, 200_000)[0]
    s2 = memsafe.keys_growth(0, 300_000 + 4096)[0] - memsafe.keys_growth(0, 300_000)[0]
    assert s2 > s1 > 0


def test_scratch_bounded_every_context():
    most = 256 * MiB
    have = 0
    for p in range(4096, 1_048_577, 4096):          # a prefill in 4,096-token pieces: each piece reserves first
        before = have
        have = memsafe.scratch_bytes(p, have=have, p0=p - 4096)
        assert have <= most and have >= before
        assert have >= memsafe.select_peak(p, p0=p - 4096)
        assert have % (2 * MiB) == 0
    assert have == most
    assert memsafe.scratch_bytes(314_305) == 160 * MiB
    assert memsafe.scratch_bytes(100_000, mode="max") == most
    assert memsafe.scratch_bytes(1_048_576, mode="off") == 0
    assert memsafe.scratch_bytes(2000) == 0                          # dense rows only: nothing
    # admission's reservation: what is still to grow
    assert memsafe.scratch_growth(314_305, 0) == 160 * MiB
    assert memsafe.scratch_growth(314_305, 256 * MiB) == 0
    assert memsafe.scratch_growth(1_048_576, 160 * MiB) == 96 * MiB
    # the growth policy
    assert memsafe.scratch_grow_to(1, 0, most) == memsafe.SCRATCH_MIN
    assert memsafe.scratch_grow_to(100 * MiB, 64 * MiB, most) == 112 * MiB
    assert memsafe.scratch_grow_to(250 * MiB, 64 * MiB, most) == 256 * MiB
    assert memsafe.scratch_grow_to(300 * MiB, 0, most) == 300 * MiB    # never below the need
    assert memsafe.scratch_grow_to(10 * MiB, 64 * MiB, most) == 64 * MiB


# -- the scratch gives the same bits (interpreter) -----------------------------------------------------------------
@pytest.fixture
def sp():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    sparse = pytest.importorskip("tensorfold.families.glm5_next.cuda.sparse")
    if torch.cuda.is_available() or type(sparse._scores).__name__ != "InterpretedFunction":
        pytest.skip("Triton's CPU interpreter, no GPU (tests/cuda/test_memory_safety_patches.py on GPUs)")
    saved = (sparse.SCRATCH, dict(sparse._SCRATCH), dict(sparse.SCRATCH_STATS))
    sparse._SCRATCH.clear()
    yield sparse, torch
    sparse.SCRATCH = saved[0]
    sparse._SCRATCH.clear()
    sparse._SCRATCH.update(saved[1])
    sparse.SCRATCH_STATS.update(saved[2])


def _inputs(torch, R, NP, seed, kind="random", nh=4):
    g = torch.Generator().manual_seed(seed)
    qi = torch.randn((R, nh * 128), generator=g).to(torch.bfloat16)
    wts = torch.randn((R, nh), generator=g).to(torch.bfloat16)
    pk = torch.randn((NP + 2, 128), generator=g).to(torch.bfloat16)
    if kind == "ties":                                  # many equal scores: the tie rule decides
        pk = (pk * 0.0 + 1.0).to(torch.bfloat16)
        pk[::7] = 2.0
    elif kind == "zeros":                               # relu(0) and -0.0 scores
        wts = -wts.abs()
    elif kind == "nan":
        pk[5] = float("nan")
    return qi, wts, pk


@pytest.mark.parametrize("kind", ["random", "ties", "zeros", "nan"])
def test_scratch_same_bits_as_allocation(sp, kind):
    sparse, torch = sp
    NP = 2048
    cases = [(2040, 40, 1024), (3000, 48, 1024), (7000, 33, 2048)]
    for i, (pos, R, np_max) in enumerate(cases):
        qi, wts, pk = _inputs(torch, R, NP, 10 * i + len(kind), kind)
        pos_dev = torch.tensor([pos], dtype=torch.int32)
        npm = min(np_max, NP)
        sparse.SCRATCH = "off"
        ref = sparse.select_pools_blocked(qi, wts, pk, pos, R, npm, pos_dev, rows=16)
        for mode in ("grow", "max"):
            sparse.SCRATCH = mode
            sparse._SCRATCH.clear()
            sparse.reserve_scratch(1, qi.device)                           # small: grows inside (fallback path)
            for t in sparse._SCRATCH.values():
                t.fill_(0x7F7F7F7F)                                         # junk the kernel must overwrite
            got = sparse.select_pools_blocked(qi, wts, pk, pos, R, npm, pos_dev, rows=16)
            assert torch.equal(got, ref), (mode, pos)
            got2 = sparse.select_pools_blocked(qi, wts, pk, pos, R, npm, pos_dev)     # one block of R rows
            assert torch.equal(got2, ref)
        # == the sorted path (upstream's full sort)
        sparse.SCRATCH = "grow"
        scores = torch.empty((R, npm), dtype=torch.float32)
        sparse._scores[(R, -(-npm // 64))](qi, wts, wts.stride(0), pk, scores, pos_dev, R, npm, 128 ** -0.5, BP=64,
                                          num_warps=4, **sparse._heads(qi, wts))
        order = torch.sort(scores, dim=1, descending=True, stable=True).indices[:, :sparse.TOPK_POOLS]
        want = torch.sort(order, dim=1).values.to(torch.int32)
        dense = ((pos + torch.arange(R) + 1) // 4) <= sparse.TOPK_POOLS
        assert torch.equal(got[~dense], want[~dense])


def test_reserved_scratch_allocates_no_block(sp, monkeypatch):
    sparse, torch = sp
    sparse.SCRATCH = "grow"
    R, NP, pos = 32, 2048, 6000
    qi, wts, pk = _inputs(torch, R, NP, 3)
    pos_dev = torch.tensor([pos], dtype=torch.int32)
    got = sparse.reserve_for_prefill(pos, pos + R, R, pk, qi.device)
    assert got >= memsafe.select_peak(pos + R, R, NP, sparse.SELECT_MB, pos) and got == sparse.scratch_bytes(qi.device)
    real_empty = torch.empty
    shapes = []

    def spy(*a, **k):
        shapes.append(a[0] if a else k.get("size"))
        return real_empty(*a, **k)

    before = dict(sparse.SCRATCH_STATS)
    monkeypatch.setattr(sparse.torch, "empty", spy)
    sparse.select_pools_blocked(qi, wts, pk, pos, R, 2048, pos_dev, rows=16)
    monkeypatch.setattr(sparse.torch, "empty", real_empty)
    assert sparse.SCRATCH_STATS == before                               # no fallback, no growth
    assert all(tuple(s) == (R, sparse.TOPK_POOLS) for s in shapes if isinstance(s, tuple)), shapes   # the output only


def test_recorded_blocks_equal_model(sp, monkeypatch):
    sparse, torch = sp
    sparse.SCRATCH = "off"
    NP = 4096
    seen = []
    real = sparse._keys_block

    def rec(n, npc, dev):
        seen.append(n * npc * 4)
        return real(n, npc, dev)

    monkeypatch.setattr(sparse, "_keys_block", rec)
    for pos, R in ((4000, 64), (9000, 48), (12_000, 16)):
        qi, wts, pk = _inputs(torch, R, NP, pos)
        seen.clear()
        np_max = sparse.pool_bucket(pos + R, NP)                           # as latent._attend passes it
        sparse.select_pools_blocked(qi, wts, pk, pos, R, np_max, torch.tensor([pos], dtype=torch.int32))
        assert seen == memsafe.select_blocks(pos, R, NP, sparse.SELECT_MB)


# -- page cache and admission --------------------------------------------------------------------------------------
MEMINFO = """MemTotal:       127597344 kB
MemFree:         6531792 kB
MemAvailable:   11165080 kB
Buffers:          325728 kB
Cached:          7036196 kB
Dirty:             14104 kB
Writeback:             0 kB
AnonPages:       7931324 kB
Mapped:          2925296 kB
Shmem:           1962540 kB
HugePages_Total:       0
"""


def _mi(free, avail, dirty=0.0, wb=0.0, mapped=2.8):
    return {"MemFree": int(free * GiB), "MemAvailable": int(avail * GiB), "Dirty": int(dirty * GiB),
            "Writeback": int(wb * GiB), "Mapped": int(mapped * GiB)}


def test_parse_meminfo_and_credit():
    mi = memsafe.parse_meminfo(MEMINFO)
    assert mi["MemFree"] == 6531792 * 1024 and mi["Mapped"] == 2925296 * 1024 and "HugePages_Total" not in mi
    # head serving (2026-09-29): 4.6 GiB above MemFree, of which Mapped 2.8 is kept
    c = memsafe.page_cache_credit(mi, keep=2 * GiB)
    assert c == mi["MemAvailable"] - mi["MemFree"] - mi["Dirty"] - mi["Mapped"]
    assert c / GiB == pytest.approx(1.62, abs=0.02)
    # the W15 copy: MemFree 2.1, 16.8 GiB of page cache (~ MemAvailable 18.4), 1 GiB dirty
    assert memsafe.page_cache_credit(_mi(2.1, 18.4, dirty=1.0), keep=2 * GiB) / GiB == pytest.approx(12.5, abs=0.01)
    assert memsafe.page_cache_credit(_mi(2.1, 18.4, dirty=1.0, mapped=1.0), keep=2 * GiB) / GiB == \
        pytest.approx(13.3, abs=0.01)                                  # keep, not Mapped, when Mapped is smaller
    assert memsafe.page_cache_credit(_mi(5, 6, dirty=3)) == 0          # dirty data is not reclaimable at once
    assert memsafe.page_cache_credit({}) == 0
    assert memsafe.read_meminfo("/nonexistent/meminfo") == {}


def test_admission_measure(monkeypatch):
    copy = _mi(2.1, 18.4, dirty=1.0)
    hr = int(0.5 * GiB)                     # the store's unused budget
    need = 2 * GiB                          # GLM53_TF_BATCH_ADMIT_GB
    new = memsafe.view(int(2.1 * GiB), 0, "available", copy, keep=2 * GiB)
    old = memsafe.view(int(2.1 * GiB), 0, "free", copy)
    assert memsafe.admit_ok(new, hr, need, floor=GiB)        # the copy no longer serializes C4
    assert not memsafe.admit_ok(old, hr, need, floor=GiB)    # the old rule: 2.1 - 0.5 < 2
    # nothing reclaimable and little free: both wait (the OOM guard stays)
    tight = _mi(1.5, 3.0)
    assert not memsafe.admit_ok(memsafe.view(int(1.5 * GiB), 0, "available", tight), 0, need, floor=GiB)
    # plenty of page cache but almost nothing free now: the floor holds it
    assert not memsafe.admit_ok(memsafe.view(int(0.3 * GiB), 0, "available", copy, keep=2 * GiB), 0, need, GiB)
    # "free" is the old rule exactly: admit_free(free, headroom) >= admit_min
    batchplan = pytest.importorskip("tensorfold.families.glm5_next.cuda.batchplan")
    rng = random.Random(5)
    for _ in range(2000):
        free, h, m = rng.randrange(0, 20 * GiB), rng.randrange(0, 4 * GiB), rng.randrange(0, 4 * GiB)
        v = memsafe.view(free, 0, "free")
        assert memsafe.admit_ok(v, h, m, floor=GiB) == (batchplan.admit_free(free, h) >= m)
    monkeypatch.setenv("GLM53_TF_ADMIT_MEM", "sometimes")
    with pytest.raises(ValueError):
        memsafe.admit_policy()
    monkeypatch.setenv("GLM53_TF_ADMIT_MEM", "")
    assert memsafe.admit_policy() == "available"


def test_admit_log_lines():
    out = io.StringIO()
    log = memsafe.AdmitLog(every=30.0, out=out)
    v = memsafe.view(int(1.0 * GiB), 0, "available", _mi(1.0, 1.5))
    log.waiting(v, 0, 2 * GiB, 3, now=100.0)
    log.waiting(v, 0, 2 * GiB, 3, now=110.0)             # within 30 s: quiet
    log.waiting(v, 0, 2 * GiB, 2, now=131.0)
    log.admitted(now=140.0)
    log.admitted(now=150.0)                              # not waiting: quiet
    lines = out.getvalue().splitlines()
    assert len(lines) == 3 and "admission waits for memory (3 queued)" in lines[0]
    assert "page cache credit" in lines[0] and "waiting 31 s" in lines[1]
    assert lines[2] == "[tensorfold] admission resumed after 40.0 s waiting for memory"
    assert log.waits == 3 and log.episodes == 1


def test_batcher_mem_ok_on_a_fake(monkeypatch):
    batch = pytest.importorskip("tensorfold.families.glm5_next.cuda.batch")
    monkeypatch.setattr(memsafe, "read_meminfo", lambda path="/proc/meminfo": _mi(2.1, 18.4, dirty=1.0))
    import collections

    def fake(policy, free):
        f = SimpleNamespace(admit_min=2 * GiB, admit_policy=policy, admit_log=memsafe.AdmitLog(out=io.StringIO()),
                            counts=collections.Counter(), queue=[1, 2, 3], _free=lambda: free,
                            _headroom=lambda: int(0.5 * GiB), _scratch_need=lambda job: 0)
        return f
    job = SimpleNamespace(prompt=[1] * 100, stats={})
    f = fake("available", int(2.1 * GiB))
    assert batch.Batcher._mem_ok(f, job, [])
    f = fake("free", int(2.1 * GiB))
    assert not batch.Batcher._mem_ok(f, job, []) and f.counts["mem_deferred"] == 1
    assert "admission waits for memory" in f.admit_log.out.getvalue()
    # the prompt's scratch growth is part of the need (the largest of the round's admissions)
    f = fake("available", int(2.1 * GiB))
    f._scratch_need = lambda j: 13 * GiB if j is job else 0
    assert not batch.Batcher._mem_ok(f, job, [])
    f._scratch_need = lambda j: 0 if j is job else 13 * GiB
    assert not batch.Batcher._mem_ok(f, job, [(0, 0, SimpleNamespace(prompt=[1]))])
    # GLM53_TF_BATCH_ADMIT_GB=0 (tests, no CUDA: free 0): always admitted, as before
    f = fake("available", 0)
    f.admit_min = 0
    assert batch.Batcher._mem_ok(f, job, [])


def test_store_can_grow_counts_page_cache(monkeypatch):
    torch = pytest.importorskip("torch")
    sessions = pytest.importorskip("tensorfold.families.glm5_next.cuda.sessions")
    monkeypatch.setattr(memsafe, "read_meminfo", lambda path="/proc/meminfo": _mi(2.1, 18.4, dirty=1.0))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (int(2.1 * GiB), 128 * GiB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: 3 * GiB)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: 2 * GiB)
    store = SimpleNamespace(comps=[SimpleNamespace(live=SimpleNamespace(is_cuda=True))], reserve=6 * GiB)
    monkeypatch.setenv("GLM53_TF_ADMIT_MEM", "available")
    assert sessions.SessionStore._can_grow(store, int(0.5 * GiB))           # 2.1 + 1 + 12.5 - 0.5 >= 6
    monkeypatch.setenv("GLM53_TF_ADMIT_MEM", "free")
    assert not sessions.SessionStore._can_grow(store, int(0.5 * GiB))       # 2.1 + 1 - 0.5 < 6: as before


# -- the allocator trim ----------------------------------------------------------------------------------------------
def test_trimmer_trigger_and_hysteresis():
    state = {"reserved": 10 * GiB, "allocated": 7 * GiB, "stuck": 0}
    calls = []

    def empty():
        calls.append(1)
        state["reserved"] = state["allocated"] + state["stuck"]

    after = lambda: (state["reserved"], state["allocated"])  # noqa: E731
    t = memsafe.Trimmer(limit=2 * GiB, interval=0)
    assert t.maybe(8 * GiB, 7 * GiB, empty, after) == 0 and not calls          # 1 GiB unused: under the limit
    assert t.maybe(state["reserved"], state["allocated"], empty, after) == 3 * GiB   # 3 GiB unused: trimmed
    assert t.trigger == 2 * GiB and t.trims == 1
    # graph pools keep 2.5 GiB that empty_cache cannot free: the trigger moves above it, no trim every piece
    state.update(reserved=7 * GiB + int(2.6 * GiB), stuck=int(2.5 * GiB))
    gain = t.maybe(state["reserved"], state["allocated"], empty, after)
    assert gain < 256 * MiB and t.trigger == int(2.5 * GiB) + 2 * GiB
    n = len(calls)
    t.maybe(state["reserved"], state["allocated"], empty, after)
    assert len(calls) == n                                                          # no retrim at the same level
    state.update(reserved=7 * GiB + 5 * GiB, stuck=0)                              # the pools went: trims again
    assert t.maybe(state["reserved"], state["allocated"], empty, after) == 5 * GiB and t.trigger == 2 * GiB
    off = memsafe.Trimmer(limit=0, interval=0)
    assert off.maybe(100 * GiB, 0, empty, after) == 0
    # at most one trim every ``interval`` seconds (GLM53_TF_ALLOC_TRIM_S)
    slow = memsafe.Trimmer(limit=GiB, interval=10.0)
    n = len(calls)
    for now, want in ((100.0, True), (105.0, False), (109.9, False), (110.0, True)):
        state.update(reserved=7 * GiB + 3 * GiB, stuck=0)
        assert (slow.maybe(state["reserved"], state["allocated"], empty, after, now=now) > 0) == want, now
    assert len(calls) == n + 2


def test_trim_torch_without_cuda():
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("host test")
    assert memsafe.trim_torch(memsafe.Trimmer(limit=1)) == 0 and memsafe.trim_torch(None) == 0


def test_drop_file_cache(tmp_path, monkeypatch):
    p = tmp_path / "x.bin"
    fd = os.open(p, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        os.write(fd, b"\0" * 65536)
        assert memsafe.drop_file_cache(fd) == hasattr(os, "posix_fadvise")
        monkeypatch.setenv("GLM53_TF_DROP_OWN_CACHE", "0")
        assert memsafe.drop_file_cache(fd) is False
    finally:
        os.close(fd)


def test_settings(monkeypatch):
    monkeypatch.setenv("GLM53_TF_ALLOC_TRIM_GB", "0")
    assert memsafe.trim_bytes() == 0
    monkeypatch.setenv("GLM53_TF_ALLOC_TRIM_GB", "-1")
    with pytest.raises(ValueError):
        memsafe.trim_bytes()
    monkeypatch.setenv("GLM53_TF_ADMIT_CACHE_KEEP_GB", "3.5")
    assert memsafe.cache_keep_bytes() == int(3.5 * GiB)
    monkeypatch.setenv("GLM53_TF_ADMIT_FREE_FLOOR_GB", "x")
    with pytest.raises(ValueError):
        memsafe.free_floor_bytes()
    for k in ("GLM53_TF_ALLOC_TRIM_GB", "GLM53_TF_ADMIT_CACHE_KEEP_GB", "GLM53_TF_ADMIT_FREE_FLOOR_GB"):
        monkeypatch.delenv(k)
    assert (memsafe.trim_bytes(), memsafe.cache_keep_bytes(), memsafe.free_floor_bytes()) == (2 * GiB, 2 * GiB, GiB)
