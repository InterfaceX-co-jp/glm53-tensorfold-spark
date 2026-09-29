"""patches/0230 (``GLM53_TF_COMM_BACKEND=roce``): the one-shot RoCE all-gather kernel and its engine plumbing on ONE
GPU, without an RDMA device.

``roce.Runtime(..., loop_xor=K)`` replaces the RDMA proxy with a host thread (``roce.cpp``'s ``Loop``) that plays
the peer: it polls the same doorbell and "delivers" the peer's shard (this rank's staged shard XOR K) into the
receive slot, then the per-HCA flags, as the NIC would. Everything else is the production path: the pinned region,
the kernel (staging, doorbell, system-scope waits, rank-order copy, device epoch, arrival counters per grid size,
poison), the launcher, stream ordering, CUDA graph capture/replay, the health check and its diagnosis.

1. The kernel: every size up to the slot (odd, unaligned, every grid class), several dtypes, both rank positions;
   captured graphs replayed many times (the device epoch carries the sequence); collectives on two streams; a
   stalled peer times out within its bound, poisons the runtime (later launches do nothing), and the check raises
   the diagnosis (and writes the marker).
2. The engine: with ``RoceComm(_TwoCopies(), Runtime(loop_xor=0))`` (the loop's peer = this rank's partials, i.e.
   exactly ``_TwoCopies``) every model exchange runs through the kernel; logits and hidden rows of windows 1-8
   (graphs and eager) equal the ``_TwoCopies`` engine's bit for bit, drafted == serial for every policy (MTP,
   DFlash2), resumed == fresh; the device epoch shows every forward all-gather of a replayed step went over the
   kernel; the control exchanges (settings, headers) stayed on the base communicator.

patches/0350: the failure watch (``roce_watch.h``) in the loop thread: a stalled peer is "never", a peer that
delivers after the failure is "late" (not 0230's "delivered but not seen").

Needs the two Sparks (``tests/cuda/bench_roce.py``; docs in patches/0230's roce.py): the RDMA proxy itself (queue
pairs, striping over both CX7 functions, GID detection on the real ports), two-rank bits of a whole reply against
NCCL (serve both backends and diff greedy + sampled transcripts), latency, and the soak (b12x #313 / sparkring #278).

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda pytest -q tests/cuda/test_roce_patches.py
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.glm5_next.cuda import roce, weights  # noqa: E402
from tensorfold.families.glm5_next.cuda.forward import commit  # noqa: E402
from test_glm_engine import _TwoCopies, _checkpoint, _drafter, _generate  # noqa: E402

KEY = 0x5A
SAMPLINGS = [Sampling(1234, 1.0, 20, 0.95), None]
IDS = ["sampled", "greedy"]
POLICIES = (None, "auto", "auto:1:1:0", "2", "c3:0.35", "a:0.6:0.85", "f3", "fc5:0.3")


def _rt(rank=0, key=KEY, n_hca=2, max_kb=256, timeout_s=10.0, stall_after=-1, blocks=8, late=False):
    s = roce.Settings(max_bytes=max_kb * 1024, timeout_s=timeout_s, blocks=blocks)
    return roce.Runtime(rank=rank, world=2, hcas=[], n_hca=n_hca, s=s, loop_xor=key, loop_stall_after=stall_after,
                        loop_late=late)


def _expect(send: torch.Tensor, rank: int, key: int = KEY) -> torch.Tensor:
    mine = send.contiguous().view(-1).view(torch.uint8)
    peer = mine ^ key
    return torch.cat([mine, peer] if rank == 0 else [peer, mine])


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(-1).view(torch.uint8)


@pytest.fixture(scope="module")
def rts():
    out = {r: _rt(rank=r) for r in (0, 1)}
    yield out
    torch.cuda.synchronize()
    for rt in out.values():
        rt.close()


# -- 1. the kernel ----------------------------------------------------------------------------------------------
SIZES = [1, 3, 4, 7, 16, 100, 4096, 16 * 1024, 16 * 1024 + 4, 64 * 1024, 128 * 1024 + 12, 256 * 1024]


@pytest.mark.parametrize("rank", [0, 1])
def test_eager_sizes_and_alignment(rts, rank):
    rt = rts[rank]
    g = torch.Generator(device="cuda").manual_seed(7)
    for n in SIZES:
        for off in (0, 1, 4, 8):
            src = torch.randint(0, 256, (n + off,), dtype=torch.uint8, device="cuda", generator=g)
            send = src[off:]
            dst = torch.empty((2 * n + off,), dtype=torch.uint8, device="cuda")
            recv = dst[off:]
            rt.gather(send, recv)
            torch.cuda.synchronize()
            rt.check()
            assert torch.equal(recv, _expect(send, rank)), (n, off)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int32, torch.int64])
def test_eager_dtypes(rts, dtype):
    rt = rts[0]
    x = (torch.randn(3, 4096, device="cuda") * 100).to(dtype)
    y = torch.empty((2 * x.numel(),), dtype=dtype, device="cuda")
    rt.gather(x.view(-1), y)
    torch.cuda.synchronize()
    assert torch.equal(_bytes(y), _expect(x, 0))
    assert torch.equal(y[:x.numel()], x.view(-1))               # the local shard: the same values, rank 0 first


def test_epoch_and_completion_track_ops(rts):
    rt = rts[1]
    torch.cuda.synchronize()
    e0 = int(rt.counters[0].item())
    x = torch.ones(1000, device="cuda")
    y = torch.empty(2000, device="cuda")
    for _ in range(25):
        rt.gather(x, y)
    torch.cuda.synchronize()
    e1 = int(rt.counters[0].item())
    assert e1 - e0 == 25
    snap = rt.snapshot()
    assert snap["completed"] == e1 and snap["doorbell"] == e1 and snap["failed"] == 0


def test_graph_replay(rts):
    """Gathers of three grid classes and a kernel between them, captured once and replayed with new inputs."""

    rt = rts[0]
    sizes = [4 * 1024, 64 * 1024, 256 * 1024]                # 1, 4 and 8 blocks
    xs = [torch.empty((n // 4,), dtype=torch.float32, device="cuda") for n in sizes]
    ys = [torch.empty((2 * x.numel(),), dtype=torch.float32, device="cuda") for x in xs]
    zs = [torch.empty_like(y) for y in ys]
    for x in xs:
        x.normal_()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):                                  # warm-up, like the engine's graphs
            for x, y, z in zip(xs, ys, zs):
                rt.gather(x, y)
                z.copy_(y * 2)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for x, y, z in zip(xs, ys, zs):
            rt.gather(x, y)
            z.copy_(y * 2)
    e0 = int(rt.counters[0].item())
    for it in range(40):
        for x in xs:
            x.normal_()
        graph.replay()
        torch.cuda.synchronize()
        rt.check()
        for x, y, z in zip(xs, ys, zs):
            assert torch.equal(_bytes(y), _expect(x, 0)), it
            assert torch.equal(z, y * 2)
    assert int(rt.counters[0].item()) - e0 == 40 * len(sizes)


def test_two_streams_eager(rts):
    """Collectives issued on two streams run in issue order (the runtime orders the streams)."""

    rt = rts[0]
    a, b = torch.cuda.Stream(), torch.cuda.Stream()
    xs = [torch.full((4096,), float(i), device="cuda") for i in range(12)]
    ys = [torch.empty((8192,), device="cuda") for _ in xs]
    torch.cuda.synchronize()
    for i, (x, y) in enumerate(zip(xs, ys)):
        with torch.cuda.stream(a if i % 2 else b):
            rt.gather(x, y)
    torch.cuda.synchronize()
    rt.check()
    for x, y in zip(xs, ys):
        assert torch.equal(_bytes(y), _expect(x, 0))


def test_timeout_poisons_and_reports(monkeypatch, tmp_path):
    mark = tmp_path / "roce-failed"
    monkeypatch.setenv("GLM53_TF_ROCE_MARK", str(mark))
    rt = _rt(timeout_s=0.5, stall_after=3)
    try:
        x = torch.arange(2048, dtype=torch.float32, device="cuda")
        y = torch.zeros(4096, device="cuda")
        for _ in range(3):
            rt.gather(x, y)
        torch.cuda.synchronize()
        rt.check()
        y.fill_(-1)
        t = time.perf_counter()
        rt.gather(x, y)                       # the peer never answers
        torch.cuda.synchronize()
        waited = time.perf_counter() - t
        assert 0.4 < waited < 5.0, waited
        assert rt.failed
        with pytest.raises(roce.RoceError, match="timed out after 0.5 s waiting for rank 1") as info:
            rt.check()
        assert "never arrived" in str(info.value) and "sequence 4" in str(info.value)
        assert torch.equal(y, torch.full_like(y, -1))          # nothing copied from an unreliable slot
        assert mark.exists() and "rank 0" in mark.read_text()
        with pytest.raises(roce.RoceError):
            rt.gather(x, torch.empty_like(y))   # an eager launch checks first
        assert rt.snapshot()["completed"] == 3
    finally:
        torch.cuda.synchronize()
        rt.close()


def test_watch_records_the_failure_never(monkeypatch):
    """patches/0350: the loop thread's watch notices the failure word at once: the flag the kernel waited for as host
    memory held it then (the previous sequence of that slot), the kernel's last read (CTRL_ERR_SEEN), no arrival."""

    monkeypatch.setenv("GLM53_TF_ROCE_MARK", "")
    rt = _rt(timeout_s=0.3, stall_after=3)
    try:
        x = torch.arange(2048, dtype=torch.float32, device="cuda")
        y = torch.zeros(4096, device="cuda")
        for _ in range(4):
            rt.gather(x, y)
        torch.cuda.synchronize()
        time.sleep(0.05)
        snap = rt.snapshot()
        w = snap["watch"]
        assert (snap["err_seq"], snap["gpu_seen"]) == (4, 2)                  # slot 0 last held sequence 2
        assert (w["state"], w["seq"], w["host_at_fail"], w["gpu_seen"]) == (1, 4, 2, 2)
        kind, _ = roce.classify(4, snap["gpu_seen"], 2, w)
        assert kind == "never"
        with pytest.raises(roce.RoceError, match="never arrived"):
            rt.check()
    finally:
        torch.cuda.synchronize()
        rt.close()


def test_late_peer_is_late_not_unseen(monkeypatch):
    """patches/0350: a peer that delivers only after the wait failed (the W2 loopback's situation) is diagnosed as
    late -- 0230's check-time reading called it "delivered but not seen"."""

    monkeypatch.setenv("GLM53_TF_ROCE_MARK", "")
    rt = _rt(timeout_s=0.3, stall_after=3, late=True)
    try:
        x = torch.arange(2048, dtype=torch.float32, device="cuda")
        y = torch.zeros(4096, device="cuda")
        for _ in range(4):
            rt.gather(x, y)
        torch.cuda.synchronize()
        for _ in range(100):                       # the loop delivers sequence 4 once it saw the failure
            if rt.snapshot()["watch"]["state"] == 2:
                break
            time.sleep(0.01)
        snap = rt.snapshot()
        assert snap["watch"]["state"] == 2 and snap["watch"]["host_at_fail"] == 2
        assert int(rt.flags[rt._flag_word(1, 0, snap["err_hca"])]) == 4       # in host memory now
        with pytest.raises(roce.RoceError, match="AFTER the wait failed") as info:
            rt.check()
        assert "not a visibility problem" in str(info.value) and "sparkring" not in str(info.value)
        assert snap["completed"] == 3
    finally:
        torch.cuda.synchronize()
        rt.close()


def test_poisoned_graph_replay_is_a_noop(monkeypatch, tmp_path):
    """After a timeout inside a replayed graph, later replays do nothing (one timeout, not one per exchange)."""

    monkeypatch.setenv("GLM53_TF_ROCE_MARK", "")
    rt = _rt(timeout_s=0.3, stall_after=5)
    try:
        xs = [torch.randn(4096, device="cuda") for _ in range(3)]
        ys = [torch.empty(8192, device="cuda") for _ in xs]
        for x, y in zip(xs, ys):
            rt.gather(x, y)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for x, y in zip(xs, ys):
                rt.gather(x, y)
        g.replay()                           # ops 4, 5 delivered; op 6 times out
        torch.cuda.synchronize()
        assert rt.failed
        t = time.perf_counter()
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        assert time.perf_counter() - t < 0.25          # no further waits
        with pytest.raises(roce.RoceError, match="sequence 6"):
            rt.check()
    finally:
        torch.cuda.synchronize()
        rt.close()


def test_one_hca():
    rt = _rt(n_hca=1)
    try:
        x = torch.randn(5000, device="cuda")
        y = torch.empty(10000, device="cuda")
        rt.gather(x, y)
        torch.cuda.synchronize()
        assert torch.equal(_bytes(y), _expect(x, 0))
    finally:
        torch.cuda.synchronize()
        rt.close()


# -- 2. the engine -----------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    out = {}
    for kind in ("mlx", "exl3"):
        path = tmp_path_factory.mktemp(f"glm_roce_{kind}")
        _checkpoint(path / "model", exl3=kind == "exl3")
        _drafter(path / "dflash2")
        out[kind] = path
    return out


class _CountingTwoCopies(_TwoCopies):
    def __init__(self):
        self.sizes = []

    def all_gather(self, send, recv):
        self.sizes.append(send.numel() * send.element_size())
        super().all_gather(send, recv)


def _engine(path, rt=None):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    mp = pytest.MonkeyPatch()
    mp.setattr(weights, "NONEXPERT", "q4mse")
    mp.setenv("GLM53_TF_NONEXPERT", "q4mse")
    base = _CountingTwoCopies()
    comm = base if rt is None else roce.RoceComm(base, rt, rt.s.max_bytes)
    try:
        return GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", comm=comm)
    finally:
        mp.undo()


@pytest.fixture(scope="module", params=["mlx", "exl3"])
def pair(request, ckpts):
    """(the _TwoCopies engine, the same engine whose model exchanges run through the RoCE kernel)."""

    path = ckpts[request.param]
    base = _engine(path)
    rt = _rt(key=0, timeout_s=30.0)
    fast = _engine(path, rt)
    yield request.param, base, fast
    torch.cuda.synchronize()
    rt.close()


def test_control_exchanges_stay_on_base(pair):
    _, _, fast = pair
    comm = fast.comm
    assert isinstance(comm, roce.RoceComm) and fast.e.w.comm is comm
    assert comm.base.sizes, "the settings checks at load go through the base all_gather"
    before = len(comm.base.sizes)
    fast._share([1, 2, 3])                     # a header-like exchange
    assert len(comm.base.sizes) == before + 2


def test_window_logits_equal_twocopies(pair):
    """Every window width from the committed state: graphs (1-6 rows) and eager (7, 8), logits and hidden rows."""

    kind, base, fast = pair
    rt = fast.comm.rt
    rng = np.random.default_rng(31)
    prompt = [int(t) for t in rng.integers(0, 1000, size=19)]
    layers = len(fast.e.w.layers)
    for R in range(1, 9):
        rows = [int(t) for t in rng.integers(0, 1000, size=R)]
        got = []
        for eng in (base, fast):
            e = eng.e
            e.reset()
            e.forward(prompt)
            commit(e.w, e.st, e.buf, len(prompt), len(prompt))
            torch.cuda.synchronize()
            e0 = int(rt.counters[0].item())
            logits = e.forward(rows).clone()
            torch.cuda.synchronize()
            got.append((logits, e.buf.hidden[:R].clone(), int(rt.counters[0].item()) - e0))
        (lb, hb, _), (lf, hf, ops) = got
        assert torch.equal(lb, lf), (kind, R)
        assert torch.equal(hb, hf), (kind, R)
        assert ops == 2 * layers, (kind, R, ops)      # every block's exchange of the step went over the kernel
    rt.check()
    for eng in (base, fast):
        eng.e.reset()
        eng.cache.clear()


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
def test_replies_equal_twocopies(pair, sampling):
    """Serial and every drafting policy: the RoCE engine's reply equals the base engine's serial reply."""

    kind, base, fast = pair
    prompt = list(np.random.default_rng(32).integers(0, 1000, size=43))
    serial, _ = _generate(base, prompt, sampling, draft=False, tokens=32)
    got, _ = _generate(fast, prompt, sampling, draft=False, tokens=32)
    assert got == serial, (kind, "serial")
    for policy in POLICIES:
        drafted, _ = _generate(fast, prompt, sampling, policy=policy, tokens=32)
        assert drafted == serial, (kind, policy)
    fast.comm.rt.check()
    assert fast.comm.fast_ops > 0


def test_resume_equal_twocopies(pair):
    kind, base, fast = pair
    sampling = Sampling(41, 1.0, 20, 0.95)
    rng = np.random.default_rng(33)
    first = list(rng.integers(0, 1000, size=70))
    replies = []
    for eng in (base, fast):
        reply, _ = _generate(eng, first, sampling, policy="auto:1:1:0", tokens=20)
        warm, stats = _generate(eng, first + reply + [7, 8], sampling, tokens=20)
        assert stats["cached"] >= len(first) + len(reply) - 1
        replies.append((reply, warm))
    assert replies[0] == replies[1], kind


def test_threshold_sends_large_exchanges_to_base(ckpts):
    """GLM53_TF_ROCE_MAX_KB below one row's partial: every model exchange stays on the base communicator."""

    rt = _rt(key=0, max_kb=1)
    try:
        eng = _engine(ckpts["mlx"], rt)
        prompt = list(np.random.default_rng(34).integers(0, 1000, size=21))
        sizes_before = len(eng.comm.base.sizes)
        _generate(eng, prompt, None, draft=False, tokens=4)
        big = [n for n in eng.comm.base.sizes[sizes_before:] if n > 1024]
        assert big, "partials larger than the threshold went through the base all_gather"
    finally:
        torch.cuda.synchronize()
        rt.close()
