"""patches/0320 (GLM53_TF_PREFILL_PP, ``glm5_next/cuda/pfpp.py``): the pipelined lean chunk (0084) with each rank doing
the hyper-connections, taps and final norm on ITS half of every sub-block's rows, the partials traded as row halves
(send/recv) and the results shared back in place. It must give 0084's bits on both ranks.

Checked:

- host only, TWO REAL PROCESSES over gloo (``torch.distributed``, CPU), each running test_lean_patches' HASH MODEL (every
  kernel exact integer arithmetic with the real row / position / KDA-state / cache / routing dependencies) as its rank,
  with rank-specific out-projection, down and expert matrices (so the two ranks' partials differ and the sums matter):
  on each rank the row-split chunk == 0084's pipelined chunk (the real all-gather) bit for bit: last-row logits,
  final-normed rows, DFlash2 taps, KDA states and conv windows, the attention caches, and the residual streams of the
  rank's own rows; both ranks agree on every replicated output; every 0084 variant (direct, gather, slab, both), chunk
  lengths around the sub-blocks (partial last sub-blocks, a rank owning 0 rows of one), two positions, EXL3 and MLX
  MoE, commit included; the swap fallback through a communicator without ``swap`` (padded all-gather) gives the same
  bits. On the host a deferred exchange runs when it is waited for (the latest moment the comm stream could run it), so
  a missing wait reads stale rows. Controls: a routing that does not wait for the shared rows, and a swap that sends
  the wrong half, are both caught; one-sub-block chunks keep 0084's path (no row swaps);
- host: the knob, the settings pair, the row split (64-aligned, disjoint, covering), the piece / share order;
- GPU, kernels (one GPU): every row-wise kernel the split calls on half a sub-block (hc_pre, hc_post on the [world,
  rows, D] slot view, 0190's fused post + partial when it launches, stream_mean, rmsnorm) gives the rows of the
  whole-sub-block call bit for bit, on the real shapes (D 4096, 4 streams) at 512 / 100 / 1024 rows, in a fast chunk;
- GPU, the engine, TWO PROCESSES ON ONE GPU (rank 0 and rank 1 of the synthetic EXL3 checkpoint, exchanges through a
  host-staged gloo communicator; ``PP_TWO_PROC=1``, slow): committed state and first token with the split == without
  it, on both ranks, 3-700 tokens.

What only the two Sparks can check: NCCL send/recv in a group on the comm stream against the other rank (no hang), the
speed (GLM53_TF_PROFILE=1: ``hc`` should halve, ``allgather`` stay ~0), and replies (``bench/glmbench.py --suites
exact`` and ab.py's reply hash with GLM53_TF_PREFILL_PP=1 on both ranks against 0). See docs/PREFILL-PP.md.

Run inside the image:
    PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q -s tests/cuda/test_prefill_pp_patches.py
Host part without the image: PYTHONPATH=<patched tree>/src:<patched tree>/tests/cuda:tests/cuda pytest -q ...
"""

from __future__ import annotations

import os
import tempfile
import traceback

import numpy as np
import pytest

try:
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:          # pragma: no cover
    CUDA = False

from tensorfold.families.glm5_next.cuda import fastpf, lean, pfoverlap as po, pfpp  # noqa: E402

gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")
BLOCK = 64
VARIANTS = ["direct", "gather", "slab", "gather,slab"]


# -- knob, settings, split ----------------------------------------------------------------------------------------------
def test_parse(monkeypatch):
    assert pfpp.parse("0") is False and pfpp.parse("") is False and pfpp.parse("off") is False
    assert pfpp.parse("1") is True and pfpp.parse(" on ") is True
    with pytest.raises(ValueError, match=pfpp.ENV):
        pfpp.parse("2")
    monkeypatch.setenv(pfpp.ENV, "1")
    assert pfpp.parse() is True


def test_settings(monkeypatch):
    monkeypatch.setattr(pfpp, "ON", False)
    assert pfpp.settings() == [0, 0]
    monkeypatch.setattr(pfpp, "ON", True)
    monkeypatch.setattr(po, "DEFAULT", po.ALL)
    assert pfpp.settings() == [1, 1]
    monkeypatch.setattr(po, "DEFAULT", po.OFF)
    assert pfpp.settings() == [1, 0]


def test_split():
    for r in range(1, 2049):
        h = pfpp.split(r)
        assert 0 < h <= r and (h == r or h % fastpf.GRID == 0), r
        (a0, n0), (a1, n1) = pfpp.own(r, 0), pfpp.own(r, 1)
        assert a0 == 0 and a1 == n0 and n0 + n1 == r and n1 >= 0, r
        assert n0 >= n1, r
    assert pfpp.split(512) == 256 and pfpp.split(2048) == 1024 and pfpp.split(100) == 64 and pfpp.split(40) == 40


def test_applies(monkeypatch):
    from types import SimpleNamespace as NS

    lb = NS(sub_blocks=lambda R: [(a, min(64, R - a)) for a in range(0, R, 64)])
    w = NS(comm=NS(world=2, rank=0), world=2)
    monkeypatch.setattr(pfpp, "ON", True)
    assert pfpp.applies(w, lb, 65) and pfpp.applies(w, lb, 2048)
    assert not pfpp.applies(w, lb, 64) and not pfpp.applies(w, lb, 1)          # one sub-block: 0084's path
    assert not pfpp.applies(NS(comm=None, world=2), lb, 256)
    assert not pfpp.applies(NS(comm=NS(world=1, rank=0), world=1), lb, 256)
    monkeypatch.setattr(pfpp, "ON", False)
    assert not pfpp.applies(w, lb, 256)


# -- host: two processes over gloo, the hash model -----------------------------------------------------------------------
class _Gloo:
    """A two-rank communicator over torch.distributed (gloo, CPU): byte copies, like NCCL's."""

    world = 2

    def __init__(self, rank: int, swap: bool = True) -> None:
        self.rank = rank
        self.swaps = 0
        if not swap:
            self.swap = None

    @staticmethod
    def _bytes(t):
        return t.contiguous().view(-1).view(torch.uint8)

    def all_gather(self, send, recv):
        import torch.distributed as dist

        s = self._bytes(send)
        parts = [torch.empty_like(s) for _ in range(2)]
        dist.all_gather(parts, s)
        recv.view(-1).view(torch.uint8).copy_(torch.cat(parts))

    def swap(self, sends, recvs):      # noqa: F811 - replaced by None for the fallback variant
        import torch.distributed as dist

        self.swaps += 1
        peer = 1 - self.rank
        reqs = [dist.isend(self._bytes(t), peer) for t in sends if t.numel()]
        reqs += [dist.irecv(t.view(-1).view(torch.uint8), peer) for t in recvs if t.numel()]
        for q in reqs:
            q.wait()

    def barrier(self):
        import torch.distributed as dist

        dist.barrier()


def _install_hash():
    """test_lean_patches' ``hashed`` fixture, without monkeypatch (a child process)."""

    from test_lean_patches import _Hash

    from tensorfold.families.glm5_next.cuda import exl3_mm, forward, glue, qmm

    k = _Hash(5)
    for name in ("embed", "hc_pre", "hc_post", "swiglu", "router", "select", "combine", "stream_mean", "rmsnorm"):
        setattr(glue, name, getattr(k, name))
    qmm.matmul = k.matmul
    qmm.group_sums = k.group_sums
    fastpf.kda_chain = k.kda_chain
    forward.dsa_block = k.dsa_block
    lean.dsa_block = k.dsa_block
    exl3_mm.routed = k.routed
    qmm.moe_gateup, qmm.moe_down = _mlx_moe()
    return k


def _mlx_moe(rank: int = 0):
    """test_overlap_patches' hash stand-ins for the MLX checkpoint's grouped MoE kernels (``rank`` shifts the down
    outputs: the ranks' expert halves differ)."""

    from test_lean_patches import _mod

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
                                                            + int(group.ids[u]) + rank + torch.arange(y.shape[-1]))

    return gateup, down


def _rank_weights(rank: int, quant: str, comm):
    """test_lean_patches' hash weights as ``rank``: the matrices TP splits by rank (the blocks' output projections,
    the dense / shared down, the routed experts, the head's vocabulary half) differ between the ranks; everything
    replicated (hyper-connection mixes, router, norms, embedding) is the same."""

    from test_lean_patches import _hash_weights

    w = _hash_weights(quant)
    w.rank, w.comm = rank, comm
    if rank:
        for L in w.layers:
            for q in [getattr(L.kda, "o", None), getattr(L.dsa, "o", None), getattr(L.mlp, "down", None)]:
                if q is not None:
                    q.W = q.W + 1
            if L.moe is not None:
                L.moe.experts.W = L.moe.experts.W + 1
                if L.moe.shared is not None:
                    L.moe.shared.down.W = L.moe.shared.down.W + 1
        w.head.W = w.head.W + 1
    return w


def _own_rows(R: int, rank: int) -> torch.Tensor:
    m = torch.zeros(R, dtype=torch.bool)
    for a in range(0, R, BLOCK):
        r = min(BLOCK, R - a)
        lo, n = pfpp.own(r, rank)
        m[a + lo:a + lo + n] = True
    return m


def _cases():
    out = []
    for quant in ("exl3", "mlx"):
        for R in (65, 100, 128, 130, 200, 256):
            for pos0 in (0, 128):
                for mode in VARIANTS:
                    if quant == "mlx" and (pos0 or mode not in ("gather,slab",)):
                        continue
                    out.append((quant, R, pos0, mode))
    return out


def _worker(rank: int, init: str, q, which: str) -> None:
    import torch.distributed as dist

    try:
        dist.init_process_group("gloo", init_method=f"file://{init}", rank=rank, world_size=2)
        os.environ[po.SLAB_ENV] = "64"
        _install_hash()
        from tensorfold.families.glm5_next.cuda import qmm

        qmm.moe_gateup, qmm.moe_down = _mlx_moe(rank)
        from test_lean_patches import _hash_run

        res = {}

        def run(w, R, pos0, mode, pp):
            po.MODE = po.parse(mode)
            pfpp.ON = pp
            try:
                return _hash_run(w, R, pos0, block=BLOCK)
            finally:
                po.MODE, pfpp.ON = po.OFF, False

        def compare(ref, got, R):
            # [last, fnormed, taps, x, conv, rec] + kcs; x: only this rank's rows are kept by the split
            own = _own_rows(R, rank)
            ok = [torch.equal(a, b) for i, (a, b) in enumerate(zip(ref, got)) if i != 3]
            ok.append(torch.equal(ref[3][own], got[3][own]))
            return all(ok) and len(ref) == len(got), ok

        if which == "equal":
            for quant, R, pos0, mode in _cases():
                comm = _Gloo(rank)
                w = _rank_weights(rank, quant, comm)
                ref = run(w, R, pos0, mode, False)
                got = run(w, R, pos0, mode, True)
                last = pfpp.compute.last
                same, detail = compare(ref, got, R)
                # replicated outputs must agree across the ranks (fnormed, taps, conv? no: KDA heads are split)
                fn = got[1].contiguous()
                both = torch.empty((2 * fn.numel(),), dtype=fn.dtype)
                comm.all_gather(fn.view(-1), both)
                agree = torch.equal(both[:fn.numel()], both[fn.numel():])
                shares = sum(1 for o in last.order if o.startswith("share")) if last is not None else -1
                res[(quant, R, pos0, mode)] = (same and agree, detail, agree, shares, comm.swaps)
        elif which == "layout":
            # hc_post adds [0] then [1]: after the swap, [0] must be rank 0's partial of this rank's rows, [1] rank 1's
            from types import SimpleNamespace as NS

            comm = _Gloo(rank)
            w = NS(device=torch.device("cpu"), cfg=NS(hidden=128), comm=comm, rank=rank, world=2, meta={})
            b = NS(site=None)
            ok = []
            for mode_overlap in (False, True):
                pipe = pfpp.SplitPipe(w, 256, torch.bfloat16)
                pipe.begin(mode_overlap)
                for r in (256, 200, 100, 40):
                    part = pipe.slot(r)
                    rows = torch.arange(r, dtype=torch.float64)[:, None] + torch.arange(128)[None]
                    part.copy_((rows * 3 + 1000 * rank).remainder(251))          # rank-specific values, exact in bf16
                    h = pipe.issue(w, b, r)
                    out = pipe.wait(h)
                    lo, n = pfpp.own(r, rank)
                    want = [(rows[lo:lo + n] * 3 + 1000 * k).remainder(251).to(torch.bfloat16) for k in (0, 1)]
                    ok.append(out.shape == (2, n, 128) and torch.equal(out[0], want[0]) and torch.equal(out[1], want[1]))
                assert pipe.idle()
            res["layout"] = all(ok), ok
        elif which == "fallback":
            comm = _Gloo(rank, swap=False)
            w = _rank_weights(rank, "exl3", comm)
            ref = run(w, 200, 64, "gather,slab", False)
            got = run(w, 200, 64, "gather,slab", True)
            res["fallback"] = compare(ref, got, 200)
        elif which == "controls":
            comm = _Gloo(rank)
            w = _rank_weights(rank, "exl3", comm)
            ref = run(w, 256, 0, "gather,slab", False)
            # 1) the routing reads every row before the shared rows arrived
            orig = pfpp._SplitChunk.need_all
            pfpp._SplitChunk.need_all = lambda self: None
            try:
                bad = run(w, 256, 0, "gather,slab", True)
            except RuntimeError:
                bad = None               # pending exchanges at the end: also a detection
            finally:
                pfpp._SplitChunk.need_all = orig
            # the pending host jobs of the broken run: drop them (both ranks the same way)
            res["no_wait"] = bad is None or not compare(ref, bad, 256)[0]
            # 2) the partial swap sends this rank's own rows instead of the other's
            orig_issue = pfpp.SplitPipe.issue

            def wrong(self, w, b, R):
                i = self.k % 2
                self.k += 1
                self.issued += 1
                b.site = None
                me, other = self.rank, 1 - self.rank
                lo, n = pfpp.own(R, me)
                olo, on_ = pfpp.own(R, other)
                S = self.buf[i]
                m = min(n, on_)
                return self._run(w, [S[me, lo:lo + m]], [S[other, lo:lo + m]], i, S[:, lo:lo + n])

            pfpp.SplitPipe.issue = wrong
            try:
                bad = run(w, 256, 0, "gather,slab", True)
            finally:
                pfpp.SplitPipe.issue = orig_issue
            res["wrong_half"] = not compare(ref, bad, 256)[0]
            # 3) one sub-block: 0084's path, no row swaps
            pfpp.compute.last = None
            run(w, 64, 0, "gather,slab", True)
            res["one_block"] = pfpp.compute.last is None
        q.put((rank, res, None))
    except Exception:                    # noqa: BLE001
        q.put((rank, None, traceback.format_exc()))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:                # noqa: BLE001
            pass


def _two(which: str):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    with tempfile.TemporaryDirectory() as d:
        init = os.path.join(d, "init")
        ps = [ctx.Process(target=_worker, args=(r, init, q, which)) for r in range(2)]
        for p in ps:
            p.start()
        out = {}
        for _ in range(2):
            rank, res, err = q.get(timeout=900)
            assert err is None, f"rank {rank}:\n{err}"
            out[rank] = res
        for p in ps:
            p.join(60)
    return out


def test_two_ranks_split_equals_pipelined():
    out = _two("equal")
    for rank in (0, 1):
        bad = {k: v for k, v in out[rank].items() if not v[0]}
        assert not bad, (rank, bad)
    for key, v in out[0].items():
        quant, R, pos0, mode = key
        blocks = -(-R // BLOCK)
        layers = 5
        # one share per sub-block per site (attention + FFN of every layer) and one for the embedding's hc_pre
        assert v[3] == blocks * (2 * layers + 1), (key, v[3])
        # the swaps: one per exchange (a sub-block's partial) and per share
        if "gather" in mode:
            assert v[4] == 2 * v[3] - blocks, (key, v[4])


def test_two_ranks_partial_swap_layout():
    out = _two("layout")
    for rank in (0, 1):
        assert out[rank]["layout"][0], (rank, out[rank]["layout"])


def test_two_ranks_swap_fallback_through_all_gather():
    out = _two("fallback")
    for rank in (0, 1):
        assert out[rank]["fallback"][0], (rank, out[rank]["fallback"])


def test_two_ranks_controls():
    out = _two("controls")
    for rank in (0, 1):
        assert out[rank]["no_wait"], rank
        assert out[rank]["wrong_half"], rank
        assert out[rank]["one_block"], rank


def test_piece_order_single_process(monkeypatch):
    """The split chunk's order with a fake two-rank communicator in one process (its swap returns zeros: only the
    schedule is checked): pre(k) before post(k - 1), a share after each post, the MoE routing after every share of the
    layer's attention was waited for, every share waited for by the end."""

    from test_lean_patches import _Hash, _hash_run, _hash_weights

    from tensorfold.families.glm5_next.cuda import exl3_mm, forward, glue, qmm

    k = _Hash(5)
    for name in ("embed", "hc_pre", "hc_post", "swiglu", "router", "select", "combine", "stream_mean", "rmsnorm"):
        monkeypatch.setattr(glue, name, getattr(k, name))
    monkeypatch.setattr(qmm, "matmul", k.matmul)
    monkeypatch.setattr(qmm, "group_sums", k.group_sums)
    monkeypatch.setattr(fastpf, "kda_chain", k.kda_chain)
    monkeypatch.setattr(forward, "dsa_block", k.dsa_block)
    monkeypatch.setattr(lean, "dsa_block", k.dsa_block)
    monkeypatch.setattr(exl3_mm, "routed", k.routed)
    monkeypatch.setenv(po.SLAB_ENV, "64")

    class Fake:
        rank, world = 0, 2

        def all_gather(self, send, recv):
            recv.view(-1)[:send.numel()].copy_(send.reshape(-1))

        def swap(self, sends, recvs):
            for r in recvs:
                r.zero_()

    w = _hash_weights()
    w.comm = Fake()
    monkeypatch.setattr(po, "MODE", po.ALL)
    monkeypatch.setattr(pfpp, "ON", True)
    _hash_run(w, 256, 0, block=BLOCK)
    order = pfpp.compute.last.order
    routing_at = [i for i, o in enumerate(order) if o == "pre 1f0"]
    assert routing_at
    # before layer 1's first FFN pre (after the routing), every sub-block's rows were waited for
    needs = [o for o in order[:routing_at[0]] if o.startswith("need")]
    assert {"need 0", "need 64", "need 128", "need 192"} <= set(needs)
    for li in range(5):
        for half in "af":
            for a in (64, 128, 192):
                i = order.index(f"pre {li}{half}{a}")
                j = order.index(f"post {li}{half}{a - 64}")
                assert i < j, (li, half, a)
    assert order.count("share 0") == 11
    assert not pfpp.compute.last.rows_pending


# -- GPU: the kernels on half a sub-block -------------------------------------------------------------------------------
@gpu
@pytest.mark.parametrize("fused", [0, 3])
@pytest.mark.parametrize("rows", [100, 512, 1024])
def test_row_split_kernels_real_shapes(rows, fused, monkeypatch):
    """hc_pre, hc_post on the [2, rows, D] slot's own-row view, 0190's hc_post + next hc_pre (fused when ``fused``
    and the kernel launches here, else the two kernels), stream_mean and rmsnorm: each rank's rows == the same rows of
    one call over the whole sub-block, in a fast chunk (``fastpf.chunk``: the row-tiled mixing dots)."""

    from tensorfold.families.glm5_next.cuda import glue

    monkeypatch.setattr(glue, "HC_FUSED", fused)
    D, S = 4096, 4
    g = torch.Generator(device="cuda").manual_seed(rows)
    x0 = torch.randn(rows, S * D, generator=g, device="cuda").bfloat16()
    fn = (torch.randn(24, S * D, generator=g, device="cuda") * 0.01).bfloat16()
    base = torch.randn(24, generator=g, device="cuda") * 0.1
    scale = torch.tensor([0.5, 0.5, 0.5], device="cuda")
    norm = (torch.rand(D, generator=g, device="cuda") + 0.5).bfloat16()
    slot = (torch.randn(2, rows, D, generator=g, device="cuda") * 3).bfloat16()
    post = torch.rand(rows, S, generator=g, device="cuda") * 2
    comb = torch.rand(rows, S * S, generator=g, device="cuda")
    h = pfpp.split(rows)
    parts = [(0, rows), (0, h), (h, rows - h)]

    def outs(n):
        return (torch.empty(n, D, dtype=torch.bfloat16, device="cuda"), torch.empty(n, D // 64, device="cuda"),
                torch.empty(n, S, device="cuda"), torch.empty(n, S * S, device="cuda"),
                torch.empty(n, glue.HC_BLOCKS, 32, device="cuda"))

    res = {}
    b = type("B", (), {})()
    with fastpf.chunk(b):
        for s0, n in parts:
            x = x0[s0:s0 + n].clone()
            o, xs, po_, co_, hp = outs(n)
            glue.hc_pre(x, fn, base, scale, norm, o, xs, po_, co_, hp, 1e-5, 1e-6, 20)
            y = x0[s0:s0 + n].clone()
            glue.hc_post(y, y, slot[:, s0:s0 + n], post[s0:s0 + n], comb[s0:s0 + n])
            z = x0[s0:s0 + n].clone()
            o2, xs2, po2, co2, hp2 = outs(n)
            glue.hc_post_pre(z, z, slot[:, s0:s0 + n], post[s0:s0 + n], comb[s0:s0 + n], fn, base, scale, norm, o2,
                             xs2, po2, co2, hp2, 1e-5, 1e-6, 20)
            hid = torch.empty(n, D, dtype=torch.bfloat16, device="cuda")
            glue.stream_mean(z, hid)
            fo, fx = torch.empty(n, D, dtype=torch.bfloat16, device="cuda"), torch.empty(n, D // 64, device="cuda")
            glue.rmsnorm(hid, norm, 1e-5, fo, fx)
            res[(s0, n)] = (o, xs, po_, co_, y, z, o2, xs2, po2, co2, hid, fo, fx)
    torch.cuda.synchronize()
    ref = res[(0, rows)]
    got = [torch.cat([u, v], 0) for u, v in zip(res[(0, h)], res[(h, rows - h)])]
    same = [torch.equal(u, v) for u, v in zip(ref, got)]
    assert all(same), same


# -- GPU: two engines (rank 0 and rank 1) on one GPU, host-staged exchanges ---------------------------------------------
class _HostGloo(_Gloo):
    """``_Gloo`` for CUDA tensors: the stream that calls it is synchronized, the bytes go through the host."""

    def all_gather(self, send, recv):
        torch.cuda.current_stream().synchronize()
        r = torch.empty(recv.numel(), dtype=recv.dtype)
        super().all_gather(send.cpu(), r)
        recv.view(-1).copy_(r.to(recv.device))

    def swap(self, sends, recvs):
        torch.cuda.current_stream().synchronize()
        hs = [t.cpu() for t in sends]
        hr = [torch.empty(t.shape, dtype=t.dtype) for t in recvs]
        super().swap(hs, hr)
        for t, h in zip(recvs, hr):
            t.copy_(h.to(t.device))

    def barrier(self):
        torch.cuda.synchronize()
        super().barrier()


def _engine_worker(rank: int, init: str, path: str, q) -> None:
    import torch.distributed as dist

    try:
        dist.init_process_group("gloo", init_method=f"file://{init}", rank=rank, world_size=2)
        from pathlib import Path

        from test_patches import _prefill_state, _same

        from tensorfold.families.glm5_next.cuda import weights
        from tensorfold.families.glm5_next.cuda.engine import GlmEngine

        weights.NONEXPERT = "q4mse"
        for k, v in {"GLM53_TF_NONEXPERT": "q4mse", "GLM53_TF_FAST_PREFILL": "1", "GLM53_TF_PREFILL_ROWS": "256",
                     "GLM53_TF_PREFILL_ROWS_MAX": "256", "GLM53_TF_SNAPSHOT_GRID": "256",
                     "GLM53_TF_LEAN_PREFILL": "1", "GLM53_TF_LEAN_BLOCK": "64", "GLM53_TF_LOOKUP": "0",
                     "GLM53_TF_PREFILL_OVERLAP": "1", po.SLAB_ENV: "64"}.items():
            os.environ[k] = v
        for k in ("GLM53_TF_BATCH", "GLM53_TF_CALIB_ONLINE", "GLM53_TF_PROFILE", "GLM53_TF_DEPTH", "GLM53_TF_FAST_GATHER",
                  "GLM53_TF_LATENT_KV", "GLM53_TF_KV_POOL_TOKENS"):
            os.environ.pop(k, None)
        po.DEFAULT = po.MODE = po.ALL
        # the host-staged communicator synchronizes its stream, which a CUDA graph capture forbids (W8: the engine's
        # decode-graph captures at load failed with cudaErrorStreamCaptureUnsupported): no decode graphs here, the
        # test compares eager prefills only
        from tensorfold.families.glm5_next.cuda import engine as _engine_mod

        _engine_mod.GRAPH_ROWS = ()
        eng = GlmEngine(Path(path) / "model", rank=rank, master="", port=0, drafter=None, comm=_HostGloo(rank))
        res = {}
        for n in (3, 64, 65, 200, 256, 300, 700):
            prompt = list(np.random.default_rng(3200 + n).integers(0, 1000, size=n))
            pfpp.ON = False
            ref = _prefill_state(eng, prompt)
            pfpp.ON = True
            got = _prefill_state(eng, prompt)
            pfpp.ON = False
            res[n] = _same(ref, got)
        q.put((rank, res, None))
    except Exception:                    # noqa: BLE001
        q.put((rank, None, traceback.format_exc()))


@gpu
@pytest.mark.skipif(os.environ.get("PP_TWO_PROC") != "1", reason="two engines on one GPU: set PP_TWO_PROC=1")
def test_two_engines_one_gpu(tmp_path):
    import torch.multiprocessing as mp
    from test_glm_engine import _checkpoint

    _checkpoint(tmp_path / "model", exl3=True)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    init = str(tmp_path / "init")
    ps = [ctx.Process(target=_engine_worker, args=(r, init, str(tmp_path), q)) for r in range(2)]
    for p in ps:
        p.start()
    for _ in range(2):
        rank, res, err = q.get(timeout=3600)
        assert err is None, f"rank {rank}:\n{err}"
        assert all(res.values()), (rank, res)
    for p in ps:
        p.join(60)
