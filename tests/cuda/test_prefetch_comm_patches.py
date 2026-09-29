"""patches/0460 on ONE GPU (TensorFold's synthetic checkpoint, ``_TwoCopies`` as rank 1; the RoCE kernel with the
loop-back thread as the peer). docs/PREFETCH-COMM.md, "GPU test plan", step 1.

L2 prefetch (``GLM53_TF_L2PF``): nothing it does may change a bit.
- every mode (bulk / lines / touch) with every site (a, f, o, e): logits and hidden rows of windows 1-8 (graphs and
  eager) == the default engine's; the prefetcher launched at every site of an eager forward (a, f, o per layer, e per
  MoE layer); replies (serial, MTP, DFlash2, the per-round choice; greedy and sampled) == the default engine's serial
  reply; no weight byte changes;
- **the Batcher (the port)**: 4 slots with the graph knobs on, replies == serial, and the prefetcher's launch count
  grows during multi-slot rounds (0040 launched nothing there); batched rows == each slot's lone rows (eager,
  capture, replay).
RoCE (``GLM53_TF_ROCE_STRIPE_KB`` / ``_LEAN`` / ``_TRACE``; the loop thread plays the peer and honours the one-HCA
rule, so a kernel waiting for the wrong HCA times out here):
- every size and alignment, eager and in graphs, bits == the expected concatenation, for lean on / off and the
  one-HCA threshold on / off; the trace ring holds ordered stamps (start <= doorbell <= flags <= end).

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q tests/cuda/test_prefetch_comm_patches.py
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.glm5_next.cuda import l2pf, roce, weights  # noqa: E402
from tensorfold.families.glm5_next.cuda.forward import commit  # noqa: E402
from test_glm_engine import _TwoCopies, _checkpoint, _drafter, _generate  # noqa: E402

SAMPLINGS = [Sampling(1234, 1.0, 20, 0.95), None]
IDS = ["sampled", "greedy"]
POLICIES = (None, "auto", "auto:1:1:0", "2", "c3:0.35", "f3", "fc5:0.3")
ALL = "a,f,o,e"


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm_l2pf")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


def _engine(path, mode: str | None, sites: str = ALL, mb: str = "0.25", latent_kv: bool = True):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(weights, "NONEXPERT", "q4mse")
        mp.setenv("GLM53_TF_NONEXPERT", "q4mse")
        mp.setenv("GLM53_TF_LATENT_KV", "1" if latent_kv else "0")
        mp.delenv("GLM53_TF_COMM", raising=False)
        if mode is None:
            mp.delenv(l2pf.ENV, raising=False)
        else:
            mp.setenv(l2pf.ENV, mode)
            mp.setenv(l2pf.ENV + "_SITES", sites)
            mp.setenv(l2pf.ENV + "_MB", mb)
            mp.setenv(l2pf.ENV + "_EXPERT_KB", "16")
        return GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", comm=_TwoCopies())


@pytest.fixture(scope="module")
def base(ckpt):
    return _engine(ckpt, None)


@pytest.fixture(scope="module", params=["bulk", "lines", "touch"])
def pre(request, ckpt):
    e = _engine(ckpt, request.param)
    pf = e.e.w.meta["prefetch"]
    assert pf is not None and pf.L2PF and pf.s.mode == request.param
    return request.param, e


def test_plans_read_only_weights(pre):
    _, e = pre
    w = e.e.w
    pf = w.meta["prefetch"]
    kinds = {k for _, k in pf.plans}
    assert kinds == {"a", "f", "o"}
    for spans in pf.plans.values():
        assert sum(s.nbytes for s in spans) <= pf.budget
    if pf.s.mode != "touch":
        assert len(pf.eplans) == sum(1 for L in w.layers if L.moe is not None) + (1 if w.mtp is not None else 0)


def test_window_logits_equal_default(base, pre):
    mode, eng = pre
    rng = np.random.default_rng(31)
    prompt = [int(t) for t in rng.integers(0, 1000, size=19)]
    pf = eng.e.w.meta["prefetch"]
    w = eng.e.w
    per_forward = 3 * len(w.layers)
    moe = sum(1 for L in w.layers if L.moe is not None)
    for R in range(1, 9):
        rows = [int(t) for t in rng.integers(0, 1000, size=R)]
        got = []
        for x in (base, eng):
            e = x.e
            e.reset()
            e.forward(prompt)
            commit(e.w, e.st, e.buf, len(prompt), len(prompt))
            s0 = (pf.stats.launches, pf.stats.expert_launches)
            logits = e.forward(rows).clone()
            torch.cuda.synchronize()
            got.append((logits, e.buf.hidden[:R].clone(), pf.stats.launches - s0[0], pf.stats.expert_launches - s0[1]))
        (lb, hb, _, _), (lp, hp, n, ne) = got
        assert torch.equal(lb, lp) and torch.equal(hb, hp), (mode, R)
        if R > 6:                          # eager: every site of the forward
            assert n == per_forward, (mode, R, n)
            assert ne == (moe if mode != "touch" else 0), (mode, R, ne)
    for x in (base, eng):
        x.e.reset()
        x.cache.clear()


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
def test_replies_equal_default(base, pre, sampling):
    mode, eng = pre
    prompt = list(np.random.default_rng(32).integers(0, 1000, size=43))
    serial, _ = _generate(base, prompt, sampling, draft=False, tokens=32)
    assert _generate(eng, prompt, sampling, draft=False, tokens=32)[0] == serial, mode
    for policy in POLICIES:
        assert _generate(eng, prompt, sampling, policy=policy, tokens=32)[0] == serial, (mode, policy)


def test_prefetch_writes_no_weight(pre):
    _, eng = pre
    pf = eng.e.w.meta["prefetch"]
    spans = [s for plan in pf.plans.values() for s in plan]
    views = [s.tensor.reshape(-1).view(torch.uint8)[s.addr - s.tensor.data_ptr():][:s.nbytes] for s in spans]
    saved = [v.clone() for v in views]
    _generate(eng, [3, 4, 5, 6, 7], None, policy="auto", tokens=16)
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(views, saved))


# -- the Batcher (compute_multi / mtp_multi): the port --------------------------------------------------------------
def _batched(path, mode: str | None):
    from test_batch_parallel_patches import _engine as engine

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GLM53_TF_LOOKUP", "0")
        mp.delenv("GLM53_TF_DEPTH", raising=False)
        if mode is None:
            mp.delenv(l2pf.ENV, raising=False)
        else:
            mp.setenv(l2pf.ENV, mode)
            mp.setenv(l2pf.ENV + "_SITES", ALL)
            mp.setenv(l2pf.ENV + "_MB", "0.25")
            mp.setenv(l2pf.ENV + "_EXPERT_KB", "16")
        return engine(path, batch=4, pad="2,4,8", after=2, short=64, parity=True, mtp=True)


@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
def test_batched_replies_equal_serial(ckpt, base, greedy):
    from test_batch2_patches import _prompt, _sampling, _serial

    b4 = _batched(ckpt, "bulk")
    pf = b4.w.meta["prefetch"]
    sampling = _sampling(greedy)
    prompts = [_prompt(70 + i, 30 + 9 * i) for i in range(4)]
    ref = _batched(ckpt, None)
    want = [_serial(ref, p, sampling, 40) for p in prompts]
    before = pf.stats.launches
    got = b4.batch.generate_batch([dict(prompt=p, max_tokens=40, sampling=sampling, policy=pol)
                                   for p, pol in zip(prompts, (None, "f3", "auto", "2"))])
    assert [t for t, _ in got] == want
    c = b4.batch.counts
    assert c["graph"] + c["eager"] + c["capture"] >= 1                 # multi-slot rounds ran
    assert pf.stats.launches > before                                  # ... and prefetched (0040: nothing)


# -- RoCE (0460 knobs) over the loop-back peer ----------------------------------------------------------------------
KEY = 0x33


def _rt(rank: int, stripe_kb: int, lean: bool, trace: int = 256):
    s = roce.Settings(max_bytes=256 * 1024, timeout_s=10.0, stripe_kb=stripe_kb, lean=lean, trace=trace)
    return roce.Runtime(rank=rank, world=2, hcas=[], n_hca=2, s=s, loop_xor=KEY)


def _expect(send: torch.Tensor, rank: int) -> torch.Tensor:
    mine = send.contiguous().view(-1).view(torch.uint8)
    peer = mine ^ KEY
    return torch.cat([mine, peer] if rank == 0 else [peer, mine])


SIZES = [1, 7, 16, 100, 4096, 16 * 1024, 16 * 1024 + 4, 31 * 1024, 32 * 1024, 64 * 1024, 128 * 1024 + 12,
         256 * 1024]


@pytest.mark.parametrize("stripe_kb", [0, 32])
@pytest.mark.parametrize("lean", [False, True], ids=["classic", "lean"])
@pytest.mark.parametrize("rank", [0, 1])
def test_roce_sizes_eager_and_graph(stripe_kb, lean, rank):
    rt = _rt(rank, stripe_kb, lean)
    try:
        g = torch.Generator(device="cuda").manual_seed(5)
        for n in SIZES:
            for off in (0, 4):
                src = torch.randint(0, 256, (n + off,), dtype=torch.uint8, device="cuda", generator=g)
                send = src[off:]
                recv = torch.empty((2 * n,), dtype=torch.uint8, device="cuda")
                rt.gather(send, recv)
                torch.cuda.synchronize()
                rt.check()
                assert torch.equal(recv, _expect(send, rank)), (n, off)
        x = torch.randint(0, 256, (48 * 1024,), dtype=torch.uint8, device="cuda", generator=g)
        y = torch.empty((2 * x.numel(),), dtype=torch.uint8, device="cuda")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                rt.gather(x, y)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(20):
                rt.gather(x, y)
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        rt.check()
        assert torch.equal(y, _expect(x, rank))
        rows = rt.trace_rows()
        done = int(rt.ctrl[roce.CTRL_COMPLETED])
        assert done == 2 * len(SIZES) + 3 + 200 and len(rows) == min(256, done)
        for r in rows:
            assert 0 < r["start"] <= r["bell"] <= r["flag"] <= r["end"], r
        summ = roce.trace_summary(rows)
        assert summ["total"]["n"] == len(rows) and summ["wait"]["p50_us"] >= 0
    finally:
        torch.cuda.synchronize()
        rt.close()
