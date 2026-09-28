"""patches/0084 (GLM53_TF_PREFILL_OVERLAP, ``glm5_next/cuda/pfoverlap.py``) and 0093 (``tf_knobs.prefill_overlap``):
the pipelined lean prefill chunk. Same kernels on the same inputs as patches/0082's lean chunk, in another order
(each sub-block's all-gather on a comm stream while the next sub-block computes; hc_post and the next hc_pre per
L2-sized row slab) with the partials exchanged in the dtype they are written in. It must give 0082's bits.

Checked:

- host only, the HASH MODEL of test_lean_patches (every kernel exact integer arithmetic with the real row / position /
  KDA-state / cache / routing dependencies; the fake "other rank" sends different values): every variant (direct,
  gather, slab, all) equals 0082's lean chunk bit for bit (chunk lengths around the sub-blocks, two positions, blocks
  of 64 and 128, slabs of 64 and whole, EXL3 and MLX MoE, commit included). On the host a deferred exchange runs when
  it is waited for (the latest moment the comm stream could run it), so a slot reused too early or a post that reads
  before its exchange shows up. Controls: a post that skips its exchange, a missing drain for one-sub-block chunks and
  a missing drain before the MoE routing are all caught. The pipelined shape (pre(k) before post(k - 1)), the
  collectives' sizes and order equal 0082's (so ranks may even disagree on the knob), the knob parsing, the slab rows,
  the 0093 knob;
- GPU, kernels: hc_post on bf16 gathered partials (and on a row slice of them) == on their fp32 copy; combine into a
  bf16 partial == the fp32 combine rounded by ``copy_``; the fast-chunk matmul's bf16 store == its fp32 store rounded
  (o_proj / down shapes of the real model, 4-bit and BF16, 1-2048 rows, bf16 and patches/0083's FP8 kernels); a
  timing print (-s) of the hyper-connection work around one exchange on the real shapes: 0082's sequence vs direct vs
  direct + slabs;
- GPU, the synthetic EXL3 checkpoint (window buffers of 64 rows, lean chunks of 256 = 4 sub-blocks): one engine,
  the variants switched between prefills: every piece of committed state and the first token equal the unpipelined
  lean chunk's (3-700 tokens, bf16 and fp32 gathers, patches/0083's FP8 chunks), also with an all-gather stand-in
  that sleeps on the comm stream before it copies (a missing event wait would read stale partials); replies with
  MTP / DFlash2 / auto drafts equal the serial reply of the unpipelined engine (greedy and sampled); resumed == fresh
  (within, after a reply, across grid points); deterministic; per-request ``prefill_overlap``; 3,000 tokens on the
  latent cache past the dense limit.

What only the two Sparks can check (``_TwoCopies`` / the sleeping stand-in run the "exchange" as copies on one GPU):
NCCL on a second stream against the other rank's NCCL (no hang; ``bench/glmbench.py --suites exact`` with
GLM53_TF_PREFILL_OVERLAP=1 on both ranks, and with 1 on one rank and 0 on the other: same collectives, same bits),
that the NCCL kernels really run beside the compute kernels (nsys: ``ncclDevKernel_AllGather`` overlapping
``_fq4`` / ``kda`` / attention kernels), and the speed (GLM53_TF_PROFILE=1: the ``allgather`` row becomes the time the
compute stream still waited; prefill tok/s at 8k / 32k / 128k, 0 vs 1, and NCCL_MAX_NCHANNELS=1/2 against the default
for the SMs NCCL takes from the compute kernels).

Run inside the image:
    PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q -s tests/cuda/test_overlap_patches.py
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:          # pragma: no cover
    CUDA = False

from tensorfold.families.glm5_next.cuda import lean, pfoverlap as po  # noqa: E402

from test_lean_patches import _eq, _hash_run, _hash_weights, hashed  # noqa: E402,F401

gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")
SAMPLINGS = ["sampled", "greedy"]
GRID = 256
BLOCK = 64
VARIANTS = ["direct", "gather", "slab", "gather,slab"]


def _sampling(kind: str):
    from tensorfold.engine.exact_sampling import Sampling

    return None if kind == "greedy" else Sampling(1234, 1.0, 20, 0.95)


# -- knobs -------------------------------------------------------------------------------------------------------------
def test_parse():
    assert po.parse("0") == po.OFF and po.parse("") == po.OFF and po.parse("off") == po.OFF
    assert po.parse("1") == po.ALL == po.Mode(True, True, True)
    assert po.parse("gather") == po.Mode(True, True, False)
    assert po.parse("slab") == po.Mode(True, False, True)
    assert po.parse("direct") == po.Mode(True, False, False)
    assert po.parse(" slab , gather ") == po.ALL
    with pytest.raises(ValueError, match="comma list"):
        po.parse("fast")
    assert po.describe(po.OFF) == "off" and po.describe(po.ALL) == "gather,slab"
    assert po.describe(po.parse("direct")) == "direct"


def test_set_request(monkeypatch):
    monkeypatch.setattr(po, "DEFAULT", po.OFF)
    po.set_request(True)
    assert po.MODE == po.ALL                    # environment off: a request asking for it gets everything
    po.set_request(False)
    assert po.MODE == po.OFF
    monkeypatch.setattr(po, "DEFAULT", po.parse("gather"))
    po.set_request(True)
    assert po.MODE == po.parse("gather")        # environment on: the environment's variant
    po.set_request(False)
    assert po.MODE == po.OFF
    monkeypatch.setattr(po, "MODE", po.DEFAULT)


def test_slab_rows(monkeypatch):
    w = _hash_weights()                          # CPU: no L2 size, 256 rows
    monkeypatch.delenv(po.SLAB_ENV, raising=False)
    assert po.slab_rows(w, 1024) == 256 and po.slab_rows(w, 128) == 128
    monkeypatch.setenv(po.SLAB_ENV, "128")
    assert po.slab_rows(w, 1024) == 128 and po.slab_rows(w, 64) == 64
    monkeypatch.setenv(po.SLAB_ENV, "0")
    assert po.slab_rows(w, 1024) == 1024
    for bad in ("100", "-64"):
        monkeypatch.setenv(po.SLAB_ENV, bad)
        with pytest.raises(ValueError, match="multiple"):
            po.slab_rows(w, 1024)


def test_knob_0093():
    from tensorfold.families.glm5_next.cuda import knobs

    assert "prefill_overlap" in knobs.HEADER
    assert knobs.parse({"prefill_overlap": 1}, rows_max=64) == {"prefill_overlap": 1}
    assert knobs.parse({"prefill_overlap": False}, rows_max=64) == {"prefill_overlap": 0}
    with pytest.raises(ValueError, match="prefill_overlap"):
        knobs.parse({"prefill_overlap": 2}, rows_max=64)


@pytest.fixture
def strict(hashed, monkeypatch):
    """The hash model with the real kernels' shape contracts checked: hc_pre's partial-dot scratch holds the rows it
    is given (the window buffers hold a sub-block), every per-row operand of hc_pre / hc_post / combine has the rows of
    the streams it goes with, and the gathered partials are [world, rows, D] with contiguous rows."""

    from tensorfold.families.glm5_next.cuda import glue

    pre, post, comb = glue.hc_pre, glue.hc_post, glue.combine

    def hc_pre(x, fn, base, scale, norm_w, out, xs, post_, comb_, part, *a):
        R = x.shape[0]
        assert part.shape[0] >= R and out.shape[0] == xs.shape[0] == post_.shape[0] == comb_.shape[0] == R
        return pre(x, fn, base, scale, norm_w, out, xs, post_, comb_, part[:R], *a)

    def hc_post(x, xout, g, post_, comb_):
        assert g.dim() == 3 and g.shape[1] == x.shape[0] == post_.shape[0] == comb_.shape[0] == xout.shape[0]
        assert g.stride(2) == 1 and g.stride(1) == g.shape[2]
        return post(x, xout, g, post_, comb_)

    def combine(y, wts, out):
        assert out.shape[0] == y.shape[0] == wts.shape[0] and out.is_contiguous()
        return comb(y, wts, out)

    monkeypatch.setattr(glue, "hc_pre", hc_pre)
    monkeypatch.setattr(glue, "hc_post", hc_post)
    monkeypatch.setattr(glue, "combine", combine)
    return hashed


# -- the hash model ----------------------------------------------------------------------------------------------------
def _run(w, R, pos0, block, mode, monkeypatch, slab="64"):
    monkeypatch.setenv(po.SLAB_ENV, slab)
    monkeypatch.setattr(po, "MODE", po.parse(mode))
    try:
        return _hash_run(w, R, pos0, block=block)
    finally:
        monkeypatch.setattr(po, "MODE", po.OFF)


@pytest.mark.parametrize("pos0", [0, 128])
@pytest.mark.parametrize("R", [1, 2, 63, 64, 65, 128, 130, 200, 256])
def test_pipelined_equals_lean_on_a_hash_model(strict, monkeypatch, R, pos0):
    w = _hash_weights()
    for block in (64, 128):
        ref = _run(w, R, pos0, block, "0", monkeypatch)
        for mode in VARIANTS:
            for slab in ("64", "0"):
                got = _run(w, R, pos0, block, mode, monkeypatch, slab)
                assert _eq(ref, got), (R, pos0, block, mode, slab, [torch.equal(x, y) for x, y in zip(ref, got)])
    assert strict.calls["routed"] > 0 and strict.calls["kda"] > 0 and strict.calls["dsa"] > 0


def test_pipelined_equals_lean_non_exl3(strict, monkeypatch):
    """The MLX checkpoint's MoE (the shared expert a slot of qmm's grouped kernels) through the pipeline."""

    from test_lean_patches import _mod

    from tensorfold.families.glm5_next.cuda import qmm

    def gateup(x, xs, ex, group, act, axs, limit):
        R = x.shape[0]
        act[:R].copy_(_mod(x.double().sum(1)[:, None, None] + torch.arange(act.shape[1])[None, :, None]
                           + torch.arange(act.shape[2])))
        axs[:R].copy_(_mod(act[:R].double().reshape(R, act.shape[1], -1, 64).sum(-1)))

    def down(act, axs, ex, group, y):
        slots = act.shape[1]
        for u in range(int(group.count[0])):
            for code in group.members[u].tolist():
                if code < 0:
                    break
                r, k = code >> 5, code & 31
                y.view(-1, slots, y.shape[-1])[r, k] = _mod(act[r, k].double().sum() + axs[r, k].double().sum()
                                                            + int(group.ids[u]) + torch.arange(y.shape[-1]))

    monkeypatch.setattr(qmm, "moe_gateup", gateup)
    monkeypatch.setattr(qmm, "moe_down", down)
    w = _hash_weights(quant="mlx")
    for R in (70, 256):
        ref = _run(w, R, 64, 64, "0", monkeypatch)
        assert _eq(ref, _run(w, R, 64, 64, "1", monkeypatch)), R


def test_fp32_gathers_and_one_rank(strict, monkeypatch):
    """GLM53_TF_FAST_GATHER=fp32 (fp32 slots) and a single rank (no exchange: the slot is the gathered view)."""

    w = _hash_weights()
    w.meta["fast_gather16"] = False
    assert _eq(_run(w, 200, 64, 64, "0", monkeypatch), _run(w, 200, 64, 64, "1", monkeypatch))
    w = _hash_weights()
    w.comm = None                                  # (shapes stay those of rank 0 of 2)
    assert _eq(_run(w, 200, 64, 64, "0", monkeypatch), _run(w, 200, 64, 64, "1", monkeypatch))


def test_pipelined_shape_and_collectives(strict, monkeypatch):
    """pre(k) is issued before post(k - 1) except where a piece reads the previous post (one sub-block, the MoE
    routing); the exchanges are 0082's, same sizes, same order (so both ranks agree whatever their knob)."""

    from test_lean_patches import _Comm

    seen = []

    class Rec(_Comm):
        def all_gather(self, send, recv):
            seen.append((send.numel(), send.dtype))
            super().all_gather(send, recv)

    w = _hash_weights()
    w.comm = Rec()
    _run(w, 200, 0, 64, "0", monkeypatch)
    ref, seen[:] = list(seen), []
    _run(w, 200, 0, 64, "1", monkeypatch)
    assert seen == ref and len(ref) == 2 * len(w.layers) * 4
    order = po.compute.last.order
    # the first layer (KDA + dense MLP): a0 a64 [post a0] a128 [post a64] a192 [post a128] f0 [post a192] f64 ...
    assert order[:8] == ["pre 0a0", "pre 0a64", "post 0a0", "pre 0a128", "post 0a64", "pre 0a192", "post 0a128",
                         "pre 0f0"]
    # a MoE layer: every attention post before the routing, then the FFN pieces pipelined again
    i = order.index("pre 1f0")
    assert order[i - 1] == "post 1a192" and order[i + 1:i + 3] == ["pre 1f64", "post 1f0"]
    assert order.index("pre 2a0") < order.index("post 1f192")      # across layers too
    _run(w, 50, 0, 64, "1", monkeypatch)                            # one sub-block: every piece drained
    order = po.compute.last.order
    assert all(order[j].startswith("pre") != order[j + 1].startswith("pre") for j in range(len(order) - 1))


def test_hash_model_catches_scheduling_bugs(hashed, monkeypatch):
    """The controls: a post that does not wait for its exchange, a one-sub-block chunk without its drain, and the MoE
    routing without the last attention post are each seen by the hash model (or refused)."""

    w = _hash_weights()
    ref200, ref50 = _run(w, 200, 64, 64, "0", monkeypatch), _run(w, 50, 64, 64, "0", monkeypatch)

    def no_wait(self, h):                      # the exchange never runs before its post reads it
        if h.slot is not None:
            self.busy[h.slot] = False
            self.queue.clear()
        return h.out

    with monkeypatch.context() as m:
        m.setattr(po.Pipe, "wait", no_wait)
        assert not _eq(ref200, _run(w, 200, 64, 64, "gather", monkeypatch))

    piece = po._Chunk.piece
    with monkeypatch.context() as m:
        m.setattr(po._Chunk, "piece", lambda self, n, pre, post, drain: piece(self, n, pre, post, False))
        assert not _eq(ref50, _run(w, 50, 64, 64, "gather", monkeypatch))

    flush = po._Chunk.flush
    calls = []

    def lazy_flush(self):                      # skip the flush the MoE routing asks for (the one before any FFN piece)
        calls.append(1)
        if self.pending is not None and self.pending[0].endswith("a192") and "f" not in self.pending[0]:
            return
        flush(self)

    with monkeypatch.context() as m:
        m.setattr(po._Chunk, "flush", lazy_flush)
        try:
            got = _run(w, 256, 64, 64, "gather", monkeypatch)
        except RuntimeError:
            got = None                         # refused by the slot check: also caught
        assert got is None or not _eq(_run(w, 256, 64, 64, "0", monkeypatch), got)


def test_slot_reuse_refused(monkeypatch):
    """``Pipe`` refuses to hand out a slot whose exchange is still pending."""

    from types import SimpleNamespace as NS

    from test_lean_patches import _Comm

    w = NS(device=torch.device("cpu"), cfg=NS(hidden=64), world=2, comm=_Comm())
    p = po.Pipe(w, 8, torch.bfloat16)
    p.begin(True)
    b = NS(site=None)
    p.slot(8).fill_(1)
    h0 = p.issue(w, b, 8)
    p.slot(8).fill_(2)
    h1 = p.issue(w, b, 8)
    with pytest.raises(RuntimeError, match="pending"):
        p.slot(8)
    assert float(p.wait(h0)[1, 0, 0]) == 2.0            # rank "1" = send + 1
    p.slot(8)
    assert float(p.wait(h1)[0, 0, 0]) == 2.0 and p.idle()


# -- GPU: kernels ------------------------------------------------------------------------------------------------------
@gpu
@pytest.mark.parametrize("rows", [1, 7, 64, 333, 1024])
def test_hc_post_reads_bf16_partials(rows):
    from tensorfold.families.glm5_next.cuda import glue

    D, S = 4096, 4
    g = torch.Generator(device="cuda").manual_seed(rows)
    x = torch.randn(rows, S * D, generator=g, device="cuda").bfloat16()
    part16 = (torch.randn(2, rows, D, generator=g, device="cuda") * 3).bfloat16()
    post = torch.rand(rows, S, generator=g, device="cuda") * 2
    comb = torch.rand(rows, S * S, generator=g, device="cuda")
    want, got = x.clone(), x.clone()
    glue.hc_post(want, want, part16.float(), post, comb)           # 0082: the fp32 copy of the gathered bf16
    glue.hc_post(got, got, part16, post, comb)
    assert torch.equal(want, got)
    if rows >= 7:                                                   # a row slice (the rank stride stays rows * D)
        a, n = 3, rows - 5
        sl = x.clone()
        glue.hc_post(sl[a:a + n], sl[a:a + n], part16[:, a:a + n], post[a:a + n], comb[a:a + n])
        assert torch.equal(sl[a:a + n], want[a:a + n])


@gpu
@pytest.mark.parametrize("rows", [1, 100, 1024])
def test_combine_into_bf16(rows):
    from tensorfold.families.glm5_next.cuda import glue

    g = torch.Generator(device="cuda").manual_seed(rows)
    y = torch.randn(rows, 9, 4096, generator=g, device="cuda") * 4
    wts = torch.rand(rows, 9, generator=g, device="cuda")
    f = torch.empty(rows, 4096, device="cuda")
    h = torch.empty(rows, 4096, dtype=torch.bfloat16, device="cuda")
    glue.combine(y, wts, f)
    glue.combine(y, wts, h)
    assert torch.equal(f.bfloat16(), h)


# o_proj / down shapes a rank (N x K): KDA o 4096x4096, DSA o 4096x8192, dense down 4096x6144, shared down 4096x1024
OUT_SHAPES = [(4096, 4096), (4096, 8192), (4096, 6144), (4096, 1024)]


def _fp8():
    try:
        from tensorfold.families.glm5_next.cuda import fp8pf          # patches/0083

        return fp8pf
    except ImportError:
        return None


@gpu
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("m", [1, 17, 64, 100, 1024, 2048])
@pytest.mark.parametrize("n,k", OUT_SHAPES)
@pytest.mark.parametrize("kind", ["q4", "b16"])
def test_fast_matmul_bf16_store(kind, n, k, m, fp8, monkeypatch):
    """What a fast chunk's ``qmm.matmul`` (``fastpf.chunk``: fast_qmm's one-accumulator tiles from 64 rows, qmm's
    kernels below; with ``fp8``, patches/0083's FP8 kernels) stores as bf16 is its fp32 result rounded by ``copy_``."""

    from tensorfold.families.glm5_next.cuda import fastpf, qmm

    f8 = _fp8()
    if fp8:
        if f8 is None or not f8.available():
            pytest.skip("no patches/0083 FP8 kernels here")
        monkeypatch.setattr(f8, "ON", True)
    g = torch.Generator(device="cuda").manual_seed(n + k + m)
    wt = torch.randn(n, k, generator=g, device="cuda") * 0.02
    q = qmm.quantize4(wt) if kind == "q4" else qmm.make_b16(wt.bfloat16())
    x = torch.randn(m, k, generator=g, device="cuda").bfloat16()
    xs = qmm.group_sums(x)
    sk = torch.empty((8 * max(m, 64) * 16384,), device="cuda")
    b = type("B", (), {})()
    with fastpf.chunk(b):
        f = qmm.matmul(x, q, xs, out=torch.empty(m, n, device="cuda"), f32=True, part=sk)
        h = qmm.matmul(x, q, xs, out=torch.empty(m, n, dtype=torch.bfloat16, device="cuda"), f32=False, part=sk)
    assert torch.equal(f.bfloat16(), h)


def _time(fn, reps=10):
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


@gpu
def test_hc_timing_real_shapes():
    """One exchange's hyper-connection work on the real shapes (1024 rows, 4 x 4096 streams, 20 Sinkhorn iterations,
    the fast chunk's row-tiled mixing dots): 0082 (bf16 partial copied from fp32, gathered bf16 copied to fp32,
    hc_post, then the next hc_pre over the sub-block) vs direct (hc_post on the bf16 partials) vs direct in slabs
    (hc_post + hc_pre per slab, the rows still in L2). Same bits; times printed (-s)."""

    from tensorfold.families.glm5_next.cuda import fastpf, glue

    R, D, S = 1024, 4096, 4
    g = torch.Generator(device="cuda").manual_seed(0)
    x0 = torch.randn(R, S * D, generator=g, device="cuda").bfloat16()
    fn = (torch.randn(24, S * D, generator=g, device="cuda") * 0.01).bfloat16()
    base = torch.randn(24, generator=g, device="cuda") * 0.1
    scale = torch.tensor([0.5, 0.5, 0.5], device="cuda")
    norm = torch.ones(D, dtype=torch.bfloat16, device="cuda")
    part32 = torch.randn(R, D, generator=g, device="cuda")
    p16 = torch.empty(R, D, dtype=torch.bfloat16, device="cuda")
    g16 = torch.empty(2 * R * D, dtype=torch.bfloat16, device="cuda")
    g32 = torch.empty(2 * R * D, device="cuda")
    post = torch.rand(R, S, generator=g, device="cuda") * 2
    comb = torch.rand(R, S * S, generator=g, device="cuda")
    outs = {}

    def run(kind, slab=R):
        x = x0.clone()
        normed = torch.empty(R, D, dtype=torch.bfloat16, device="cuda")
        xs = torch.empty(R, D // 64, device="cuda")
        po_, co_ = post.clone(), comb.clone()
        hcpart = torch.empty(R, glue.HC_BLOCKS, 32, device="cuda")

        def step():
            x.copy_(x0)
            po_.copy_(post)
            co_.copy_(comb)
            if kind == "0082":
                p16.copy_(part32)
                g16[:R * D].copy_(p16.view(-1))
                g16[R * D:].copy_(p16.view(-1))
                g32.copy_(g16)
                gath = g32.view(2, R, D)
            else:
                p16.copy_(part32)                    # (the producer's own bf16 store: outside the timed difference)
                g16[:R * D].copy_(p16.view(-1))
                g16[R * D:].copy_(p16.view(-1))
                gath = g16.view(2, R, D)
            for s0 in range(0, R, slab):
                n = min(slab, R - s0)
                xx = x[s0:s0 + n]
                glue.hc_post(xx, xx, gath[:, s0:s0 + n], po_[s0:s0 + n], co_[s0:s0 + n])
                glue.hc_pre(xx, fn, base, scale, norm, normed[s0:s0 + n], xs[s0:s0 + n], po_[s0:s0 + n],
                            co_[s0:s0 + n], hcpart[:n], 1e-5, 1e-6, 20)

        b = type("B", (), {})()
        with fastpf.chunk(b):
            ms = _time(step)
            step()
        torch.cuda.synchronize()
        outs[(kind, slab)] = (x.clone(), normed.clone(), xs.clone(), po_.clone(), co_.clone())
        return ms

    t0 = run("0082")
    t1 = run("direct")
    ts = {s: run("direct", s) for s in (512, 384, 256, 128)}
    ref = outs[("0082", R)]
    for key, got in outs.items():
        assert all(torch.equal(a, b) for a, b in zip(ref, got)), key
    print(f"\n[hc around one exchange, 1024 rows] 0082 {t0:.3f} ms, direct {t1:.3f} ms, "
          + ", ".join(f"slab {s} {v:.3f} ms" for s, v in ts.items())
          + f" (includes the 2 x 16 MB stand-in 'gather' copies in both)")


# -- GPU: the engine ---------------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")
    monkeypatch.setenv("GLM53_TF_NONEXPERT", "q4mse")
    monkeypatch.setenv(po.SLAB_ENV, "64")
    yield
    po.MODE = po.OFF


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_overlap")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


@pytest.fixture(scope="module")
def el(ckpt):
    from test_lean_patches import _engine

    return _engine(ckpt, lean_on=True)


def _mode(mode: str):
    po.MODE = po.parse(mode)


class _Slow:
    """``_TwoCopies`` that first spins ~0.2 ms on the stream it is called on (the comm stream when pipelined): a
    consumer that does not wait for the exchange's event reads the slot before the copy lands."""

    rank, world = 0, 2

    def all_gather(self, send, recv):
        torch.cuda._sleep(300_000)
        n = send.numel()
        recv.view(-1)[:n].copy_(send.reshape(-1))
        recv.view(-1)[n:2 * n].copy_(send.reshape(-1))

    def barrier(self):
        torch.cuda.synchronize()


def _state(eng, prompt, mode):
    from test_patches import _prefill_state

    _mode(mode)
    try:
        return _prefill_state(eng, prompt)
    finally:
        _mode("0")


@gpu
@pytest.mark.parametrize("n", [3, 50, 64, 200, 256, 300, 700])
def test_overlap_state_equals_lean(el, n):
    from test_patches import _same

    prompt = list(np.random.default_rng(840 + n).integers(0, 1000, size=n))
    ref = _state(el, prompt, "0")
    for mode in VARIANTS:
        got = _state(el, prompt, mode)
        assert _same(ref, got), (mode, [torch.equal(x, y) for x, y in zip(ref, got)])


@gpu
def test_overlap_with_a_slow_exchange(el):
    """The stand-in exchange sleeps on the comm stream before it copies: still the unpipelined bits (the compute
    stream waits for each exchange's event, and no slot is rewritten before its exchange is done)."""

    from test_patches import _same

    prompt = list(np.random.default_rng(841).integers(0, 1000, size=700))
    ref = _state(el, prompt, "0")
    comm = el.w.comm
    el.w.comm = _Slow()
    try:
        for mode in ("gather", "gather,slab"):
            assert _same(ref, _state(el, prompt, mode)), mode
    finally:
        el.w.comm = comm


@gpu
def test_overlap_fp8_prefill(el):
    """patches/0083's FP8 fast chunks, pipelined: the unpipelined FP8 lean chunk's bits."""

    from test_patches import _same

    f8 = _fp8()
    if f8 is None or not f8.available():
        pytest.skip("no patches/0083 FP8 kernels here")
    prompt = list(np.random.default_rng(849).integers(0, 1000, size=700))
    el.e.fp8_prefill = True
    try:
        ref = _state(el, prompt, "0")
        for mode in ("gather", "gather,slab"):
            assert _same(ref, _state(el, prompt, mode)), mode
    finally:
        el.e.fp8_prefill = False
        f8.ON = False


@gpu
def test_overlap_fp32_gathers(el):
    from test_patches import _same

    prompt = list(np.random.default_rng(842).integers(0, 1000, size=500))
    el.w.meta["fast_gather16"] = False
    try:
        ref = _state(el, prompt, "0")
        assert _same(ref, _state(el, prompt, "1"))
    finally:
        el.w.meta["fast_gather16"] = True


def _gen(eng, prompt, sampling, *, draft=True, policy=None, tokens=24, knobs=None):
    out: list[int] = []
    eng.request.policy = policy
    eng.request.stop_eos = False
    eng.request.knobs = knobs
    try:
        stats = eng.generate(list(prompt), tokens, sampling, lambda new: out.extend(new), draft=draft)
    finally:
        eng.request.knobs = None
    return out, stats


def _cold(eng, prompt, sampling, **kw):
    eng.cache = []
    out, stats = _gen(eng, prompt, sampling, draft=False, **kw)
    assert stats["cached"] == 0
    return out


@gpu
@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_overlap_replies(el, sampling):
    """Pipelined: drafted (MTP, DFlash2, auto) == serial == the unpipelined engine's serial reply."""

    s = _sampling(sampling)
    prompt = list(np.random.default_rng(843).integers(0, 1000, size=600))
    _mode("0")
    ref = _cold(el, prompt, s, tokens=32)
    _mode("1")
    try:
        assert _cold(el, prompt, s, tokens=32) == ref
        for policy in (None, "auto:1:1:0", "f3", "2", "c3:0.35"):
            el.cache = []
            got, stats = _gen(el, prompt, s, policy=policy, tokens=32)
            assert got == ref, policy
            assert stats["fast_prefill"] == GRID
    finally:
        _mode("0")


@gpu
@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_overlap_resumed_equals_fresh(el, sampling):
    s = _sampling(sampling)
    rng = np.random.default_rng(844)
    more = lambda n: [int(t) for t in rng.integers(0, 1000, size=n)]       # noqa: E731
    p1 = more(600)
    _mode("1")
    try:
        el.cache = []
        r1, _ = _gen(el, p1, s, policy="auto:1:1:0")
        for name, prompt in (("within", p1 + more(3)), ("reply", p1 + r1 + more(5)), ("across", p1 + r1 + more(200))):
            _gen(el, p1, s, policy="auto:1:1:0")
            warm, stats = _gen(el, prompt, s, policy="2")
            assert stats["cached"] == 512, name
            cold = _cold(el, prompt, s)
            _mode("0")
            assert cold == _cold(el, prompt, s), name              # and the unpipelined fresh prefill's
            _mode("1")
            assert warm == cold, name
    finally:
        _mode("0")


@gpu
def test_overlap_deterministic(el):
    from test_patches import _same

    prompt = list(np.random.default_rng(845).integers(0, 1000, size=700))
    a = _state(el, prompt, "1")
    _gen(el, list(np.random.default_rng(846).integers(0, 1000, size=90)), None, tokens=4)
    assert _same(a, _state(el, prompt, "1"))


@gpu
def test_overlap_knob_per_request(el):
    """patches/0093: ``tf_knobs.prefill_overlap`` switches the pipeline for one request (same reply), is echoed, and
    the load-time default is back afterwards."""

    s = _sampling("sampled")
    prompt = list(np.random.default_rng(847).integers(0, 1000, size=500))
    ref = _cold(el, prompt, s)
    seen = []
    compute = po.compute

    def spy(*a, **k):
        seen.append(po.MODE)
        return compute(*a, **k)

    po.compute = spy
    try:
        el.cache = []
        got, stats = _gen(el, prompt, s, draft=False, knobs={"prefill_overlap": 1})
    finally:
        po.compute = compute
    assert got == ref and seen and all(m == po.ALL for m in seen)
    assert stats["tf_knobs"]["prefill_overlap"] == 1
    assert po.MODE == po.DEFAULT and el._knob_state()["prefill_overlap"] == int(po.DEFAULT.pipe)


@pytest.fixture(scope="module")
def long_ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter
    from test_patches import _index_heads_32

    path = tmp_path_factory.mktemp("glm_overlap_long")
    _checkpoint(path / "model", exl3=True)
    _index_heads_32(path / "model")
    _drafter(path / "dflash2")
    return path


@gpu
def test_overlap_long_context_latent(long_ckpt):
    """3,000 tokens across the dense limit on the latent cache (chunks of 512 in sub-blocks of 128: sparse rows, the
    index ring, the blocked selection per sub-block): pipelined == unpipelined, state and replies."""

    from test_lean_patches import _engine
    from test_patches import _same

    eng = _engine(long_ckpt, lean_on=True, block=128, rows=512, rows_max=512, context=4096, latent_kv=True)
    prompt = list(np.random.default_rng(848).integers(0, 1000, size=3000))
    ref = _state(eng, prompt, "0")
    got = _state(eng, prompt, "1")
    assert _same(ref, got), [torch.equal(x, y) for x, y in zip(ref, got)]
    _mode("0")
    r0 = _cold(eng, prompt, None)
    _mode("1")
    try:
        assert _cold(eng, prompt, None) == r0
        assert _gen(eng, prompt, None, policy="2")[0] == r0
    finally:
        _mode("0")
    del eng
    torch.cuda.empty_cache()
