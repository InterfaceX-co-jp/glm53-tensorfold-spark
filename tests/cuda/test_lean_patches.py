"""patches/0082 (GLM53_TF_LEAN_PREFILL=1, ``glm5_next/cuda/lean.py``): fast-prefill chunks of up to
GLM53_TF_PREFILL_ROWS_MAX rows with window buffers of GLM53_TF_LEAN_BLOCK rows.

A lean chunk runs every block in row sub-blocks on the (small) window buffers, except the MoE's routing and routed
experts, which run once over the whole chunk (each expert's weights read once per chunk); it must give the bits of
patches/0080's fast chunk of the same rows. Checked:

- host only: the lean orchestration on a HASH MODEL (every kernel replaced by exact integer arithmetic with the real
  kernels' row / position / state dependencies: a sequential KDA-like state and conv window, a causal cache, the
  expert grouping over the chunk) equals the non-lean fast chunk bit for bit, for chunk lengths around the sub-block
  and at two positions, commit included; a planted carry bug is caught (the control); with ``fast_qmm``'s dispatch
  (fewer than 64 rows -> qmm, other bits), a last sub-block of 1 / 17 / 63 rows still runs the fast kernels like the
  whole chunk (and the control without ``lean.chunk_rows`` differs); the fast rule
  (resumed == fresh, drafted == serial through the engine's real prefill / snapshot / resume code) with lean chunks
  that depend on their sub-blocks and a chunk grid larger than the window buffers; the lean set's bytes on the real
  model's shapes (meta device) against ``lean.budget`` and a per-row bound; the settings;
- GPU, the synthetic EXL3 checkpoint: lean engine (window buffers of 64 rows, lean chunks of 256) vs the non-lean fast
  engine at the same grid: identical committed state (KDA states, conv windows, attention and MTP caches, first
  token, the MTP head's input row) for prompts on, below and across the grid; identical replies with MTP and DFlash2
  drafts; drafted == serial; resumed == fresh; per-request prefill_rows up to the lean max (and exact requests clamped
  to the window buffers, same reply as the non-lean engine); memory: the lean engine's buffers against the non-lean
  one's; past 2,051 tokens on the latent cache.

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda pytest -q tests/cuda/test_lean_patches.py
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

try:
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:          # pragma: no cover
    CUDA = False

from tensorfold.families.glm5_next.cuda import fastpf, lean  # noqa: E402

gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")
SAMPLINGS = ["sampled", "greedy"]
GRID = 256                   # the engines' chunk grid (GLM53_TF_PREFILL_ROWS=256)
BLOCK = 64                   # the lean engine's window buffers / sub-blocks (GLM53_TF_LEAN_BLOCK=64)


def _sampling(kind: str):
    from tensorfold.engine.exact_sampling import Sampling

    return None if kind == "greedy" else Sampling(1234, 1.0, 20, 0.95)


# -- settings -------------------------------------------------------------------------------------------------------
def test_settings(monkeypatch):
    monkeypatch.delenv("GLM53_TF_LEAN_PREFILL", raising=False)
    monkeypatch.delenv("GLM53_TF_LEAN_BLOCK", raising=False)
    assert lean.settings() == [0, 1024]
    monkeypatch.setenv("GLM53_TF_LEAN_PREFILL", "1")
    monkeypatch.setenv("GLM53_TF_LEAN_BLOCK", "512")
    assert lean.settings() == [1, 512]
    for bad in ("100", "32", "0"):
        monkeypatch.setenv("GLM53_TF_LEAN_BLOCK", bad)
        with pytest.raises(ValueError, match="multiple"):
            lean.block()


# -- the hash model: the real orchestration, exact integer kernels ---------------------------------------------------
M = 251                      # every value is an integer < 256: exact in bf16, fp16 and fp32, sums exact in fp64


def _mod(t: torch.Tensor) -> torch.Tensor:
    return torch.remainder(t, M)


def _gs(x: torch.Tensor) -> torch.Tensor:
    R, K = x.shape
    return _mod(x.double().reshape(R, K // 64, 64).sum(-1))


class _Q:
    """A 'matrix': K x N small integers."""

    def __init__(self, k: int, n: int, g: torch.Generator):
        self.k, self.n = k, n
        self.W = torch.randint(0, 3, (k, n), generator=g).double()


class _Hash:
    """Stand-ins for the kernels ``forward`` and ``lean`` call, with the dependencies that matter: rows are
    independent except through the KDA-like state and conv window (sequential over absolute positions) and the
    attention cache (a row reads every position up to its own); the experts read the chunk's grouping."""

    def __init__(self, experts: int):
        self.E = experts
        self.calls: dict[str, int] = {}

    def _n(self, name):
        self.calls[name] = self.calls.get(name, 0) + 1

    # glue
    def embed(self, ids, table, dims, copies, out):
        out.copy_(_mod(ids.double()[:, None] * 7 + torch.arange(dims * copies, dtype=torch.float64)[None]))
        return out

    def hc_pre(self, x, fn, base, scale, norm_w, out, xs, post, comb, part, eps, hc_eps, iters):
        D = out.shape[1]
        S = x.shape[1] // D
        acc = float(fn[0]) + sum(x[:, s * D:(s + 1) * D].double() * (s + 1) for s in range(S))
        acc = _mod(acc)
        out.copy_(acc)
        xs.copy_(_gs(acc))
        post.copy_(_mod(acc[:, :S] + 1))
        comb.copy_(_mod(acc[:, S:S + S * S] + 2))

    def hc_post(self, x, xout, g, post, comb):
        D = g.shape[2]
        S = x.shape[1] // D
        gs = g.double().sum(0)
        new = torch.cat([_mod(x[:, s * D:(s + 1) * D].double() + gs * post[:, s:s + 1].double()
                              + comb[:, s:s + 1].double()) for s in range(S)], 1)
        xout.copy_(new)

    def swiglu(self, gu, out, xs, limit):
        W = out.shape[1]
        y = _mod(gu[:, :W].double() * 3 + gu[:, W:2 * W].double())
        out.copy_(y)
        xs.copy_(_gs(y))

    def router(self, x, w, out):
        out.copy_(_mod(x.double() @ w.W))
        return out

    def select(self, logits, bias, pick, wts, ids, count, members, top_k, experts, scale, norm):
        R = logits.shape[0]
        lg = logits.double()
        for k in range(top_k):
            pick[:R, k] = torch.remainder(lg[:, 0].long() + 2 * k, experts).int()
            wts[:R, k] = torch.remainder(lg[:, 1] + k + 1, 7).float()
        pick[:R, top_k] = experts
        wts[:R, top_k] = 1.0
        used = sorted(set(pick[:R].flatten().tolist()))
        ids.zero_()
        ids[:len(used)] = torch.tensor(used, dtype=ids.dtype)
        count.fill_(len(used))
        members.fill_(-1)
        for u, e in enumerate(used):
            codes = [r * 32 + k for r in range(R) for k in range(top_k + 1) if int(pick[r, k]) == e]
            members[u, :len(codes)] = torch.tensor(codes, dtype=members.dtype)

    def combine(self, y, wts, out):
        out.copy_(_mod((y.double() * wts.double()[:, :, None]).sum(1)))

    def stream_mean(self, x, out):
        R, D = out.shape
        out.copy_(_mod(x.double().view(R, -1, D).sum(1)))
        return out

    def rmsnorm(self, x, w, eps, out, xs=None):
        y = _mod(x.double() * 3 + 1)
        out.copy_(y)
        if xs is not None:
            xs.copy_(_gs(y))
        return out

    # qmm
    def matmul(self, x, q, xs=None, *, out=None, f32=False, part=None):
        self._n("matmul")
        y = x.double() @ q.W
        if xs is not None:
            y = y + xs.double().sum(1, keepdim=True)
        y = _mod(y)
        if out is None:
            out = torch.empty((x.shape[0], q.n), dtype=torch.float32 if f32 else torch.bfloat16)
        out.copy_(y)
        return out

    def group_sums(self, x, out=None):
        y = _gs(x)
        if out is None:
            return y.float()
        out.copy_(y)
        return out

    # KDA: sequential over absolute positions, reads the conv window [3 rows before; the rows] and the state
    def kda_chain(self, p, b_off, a, g, conv_state, conv_w, state_in, a_log, dt_bias, norm_w, eps, lower, rows,
                  scratch, state_out, *, pos):
        self._n("kda")
        assert pos % fastpf.GRID == 0
        C = conv_state.shape[1]
        win = torch.cat([conv_state.double(), p[:rows, :C].double()], 0)
        wch = torch.arange(C, dtype=torch.float64) % 5 + 1
        s = float(state_in.reshape(-1)[0])
        t = 0.0                           # a second state channel, updated per row like the first
        outs = []
        for i in range(rows):
            f = float((win[i:i + 4] * wch).sum()) + float(p[i, b_off]) + float(a[i, :8].double().sum()) \
                + float(g[i, :8].double().sum())
            s = (s * 31 + f + (pos + i) * 7) % M
            t = (t + s) % M
            outs.append(s)
        o = scratch.out[:rows]
        o.copy_(_mod(torch.tensor(outs, dtype=torch.float64)[:, None] + torch.arange(o.shape[1])))
        new = state_in.clone()
        new.view(-1)[0] = s
        new.view(-1)[1:] = _mod(state_in.reshape(-1)[1:].double() + t)
        state_out.copy_(new)
        return o

    # DSA: writes the window's keys at pos_dev.., then each row reads every key up to its own position
    def dsa_block(self, layer, w, kc, vc, pos_dev, b, R, nch, index=None, host_pos=None, npb=None):
        from tensorfold.families.glm5_next.cuda import forward
        from tensorfold.families.glm5_next.cuda.attention import CHUNK

        self._n("dsa")
        pos = int(pos_dev[0])
        assert host_pos == pos and nch == -(-(pos + R) // CHUNK) and npb is None
        feat = _mod(b.normed[:R].double() @ layer.dsa.w)
        kc[pos:pos + R, 0, 0] = feat
        cum = kc[:pos + R, 0, 0].double().cumsum(0)[pos:pos + R]
        row = _mod(cum * 3 + feat)
        width = b.xs_ao.shape[1] * 64
        o = b.attn.out.view(b.attn.out.shape[0], -1)[:R, :width]
        o.copy_(_mod(row[:, None] + torch.arange(width)))
        return forward.out_proj(w, b, o, layer.dsa.o, self.group_sums(o, b.xs_ao[:R]), R, "dsa.o_proj")

    # the routed experts: through the chunk's grouping (only the listed members are written)
    def routed(self, x, pick, group, ex, s, y, R, limit, fast=False):
        self._n("routed")
        assert fast
        slots = s.slots
        n = int(group.count[0])
        for u in range(n):
            e = int(group.ids[u])
            if e >= self.E:
                continue
            for code in group.members[u].tolist():
                if code < 0:
                    break
                r, k = code >> 5, code & 31
                assert r < R and int(pick[r, k]) == e
                y[r * slots + k] = _mod((x[r].double() * ex.W[e]).sum() + torch.arange(y.shape[1]))


class _Comm:
    rank, world = 0, 2

    def all_gather(self, send, recv):
        n = send.numel()
        recv.view(-1)[:n].copy_(send.reshape(-1))
        recv.view(-1)[n:2 * n].copy_(send.reshape(-1) + 1)     # the "other rank" differs


def _hash_weights(quant="exl3"):
    from tensorfold.families.glm5_next.cuda import weights

    g = torch.Generator().manual_seed(5)
    D, S, E, K = 128, 2, 5, 2
    kinds = ["kda", "kda", "dsa", "kda", "dsa"]
    mlps = ["dense", "moe", "moe", "moe", "moe"]
    cfg = weights.Config(hidden=D, layers=len(kinds), vocab=96, eps=1e-5, heads=4, q_lora=64, kv_lora=64, qk_dim=64,
                         v_dim=64, lin_heads=2, lin_dim=128, conv=4, lower=-5.0, experts=E, top_k=K, moe_width=128,
                         shared_width=128, dense_width=128, routed_scale=1.0, norm_topk=True, streams=S, hc_iters=1,
                         hc_eps=1e-6, index_heads=2, index_dim=128, index_topk=16, kpool=4, limit=10.0, kinds=kinds,
                         mlp_kinds=mlps, eos=(0,), mtp_layers=0, group_size=64, bits=4, quant=quant)
    LL, HL = 1, 2
    C = 3 * LL * 128
    tag = iter(range(1, 1000))
    hc = lambda: NS(fn=torch.tensor([float(next(tag))]), base=None, scale=None)      # noqa: E731
    layers = []
    for i, (kind, mk) in enumerate(zip(kinds, mlps)):
        L = NS(kind=kind, index=i, attn_hc=hc(), ffn_hc=hc(), in_norm=None, post_norm=None, kda=None, dsa=None,
               mlp=None, moe=None)
        if kind == "kda":
            L.kda = NS(proj=_Q(D, C + 256 + 64, g), fa_off=C, ga_off=C + 128, b_off=C + 256, fb=_Q(128, LL * 128, g),
                       gb=_Q(128, LL * 128, g), conv=None, a_log=None, dt_bias=None, norm=None, o=_Q(LL * 128, D, g))
        else:
            L.dsa = NS(w=torch.randint(0, 3, (D,), generator=g).double(), o=_Q(HL * 64, D, g))
        if mk == "dense":
            L.mlp = NS(gu=_Q(D, 128, g), down=_Q(64, D, g))
        else:
            L.moe = NS(router=_Q(D, E, g), bias=None, experts=NS(W=torch.randint(0, 3, (E, D), generator=g).double()),
                       shared=NS(gu=_Q(D, 128, g), down=_Q(64, D, g)) if quant == "exl3" else None)
        layers.append(L)
    return NS(cfg=cfg, device=torch.device("cpu"), world=2, rank=0, head=_Q(D, 96, g), embed=None, norm=None,
              layers=layers, mtp=None, comm=_Comm(), draft_head=None,
              meta={"latent_kv": False, "long_context": False, "fast_gather16": True, "prefetch": None})


@pytest.fixture
def hashed(monkeypatch):
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
    return k


def _hash_run(w, R: int, pos0: int, *, block: int | None, rows: int = 256, seed: int = 0):
    """One fast chunk of R rows at pos0 from a random committed state: through ``lean`` (window buffers of ``block``
    rows) or ``forward.compute(fast=True)`` (block None, window buffers of ``rows``). Everything it produced."""

    from tensorfold.families.glm5_next.cuda import forward

    g = torch.Generator().manual_seed(seed)
    cap = 1024
    brows = block or rows
    b = forward.Buffers(w, brows, cap)
    b.set_taps((1, 3), w.cfg.hidden)
    st = forward.State(w, cap, brows)
    st.rec.copy_(torch.randint(0, M, st.rec.shape, generator=g).float())
    st.conv.copy_(torch.randint(0, M, st.conv.shape, generator=g).to(st.conv.dtype))
    for kc in st.kc:
        kc[:pos0, 0, 0] = torch.randint(0, M, (pos0,), generator=g).to(kc.dtype)
    st.set_pos(pos0)
    ids = torch.randint(0, 96, (R,), generator=torch.Generator().manual_seed(seed + 1)).int()
    conv0 = st.conv.clone()
    cur = st.cur[0]
    if block is None:
        b.ids[:R].copy_(ids)
        logits = forward.compute(w, st, b, R, nch=forward.chunks_for(st, R), host_pos=st.pos, fast=True, head=True)
        last = logits[R - 1:R].clone()
        fn, taps, x = b.fnormed[:R].clone(), torch.cat([t[:R] for t in b.taps], 1), b.x[:R].clone()
        n = len(st.cur)
        C = st.conv.shape[2]
        # forward.commit's conv shift (a Triton kernel) in torch: the last 3 rows of [window; projection rows]
        conv = st.conv.clone()
        conv[:n] = torch.stack([torch.cat([conv0[i], st.proj[i, :R, :C]], 0)[-3:] for i in range(n)])
        rec = st.rec[1 - cur].clone()
    else:
        lb = lean.LeanBuffers(w, rows, block, taps=2)
        lb.ids[:R].copy_(ids)
        last = lean.compute(w, st, b, lb, R, head=True).clone()
        fn, taps, x = lb.fnormed[:R].clone(), lb.tapcat[:R].clone(), lb.x[:R].clone()
        lean.commit(w, st, lb, R)
        assert st.cur[0] == 1 - cur and st.pos == pos0 + R
        conv, rec = st.conv.clone(), st.rec[st.cur[0]].clone()
    kcs = [kc[:pos0 + R].clone() for kc in st.kc]
    return [last, fn, taps, x, conv, rec] + kcs


def _eq(a, b):
    return len(a) == len(b) and all(x.shape == y.shape and torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("pos0", [0, 128])
@pytest.mark.parametrize("R", [1, 2, 3, 5, 63, 64, 65, 128, 130, 200, 256])
def test_lean_equals_fast_chunk_on_a_hash_model(hashed, R, pos0):
    w = _hash_weights()
    ref = _hash_run(w, R, pos0, block=None)
    for block in (64, 128):
        got = _hash_run(w, R, pos0, block=block)
        assert _eq(ref, got), (R, pos0, block, [torch.equal(x, y) for x, y in zip(ref, got)])
    assert hashed.calls["routed"] > 0 and hashed.calls["kda"] > 0 and hashed.calls["dsa"] > 0


class _FastMM:
    """A stand-in for ``fast_qmm`` with its dispatch rule: calls of fewer than ``MIN_ROWS`` rows go to ``qmm.matmul``
    (the hash model's plain product), the others to a 'fast kernel' whose bits differ (+1)."""

    MIN_ROWS = 64

    def __init__(self):
        self.small = []                   # row counts that went to qmm

    def matmul_fast(self, x, q, xs=None, *, out=None, f32=False, part=None):
        from tensorfold.families.glm5_next.cuda import qmm

        if x.shape[0] < self.MIN_ROWS:
            self.small.append(x.shape[0])
            return qmm.matmul(x, q, xs, out=out, f32=f32, part=part)
        y = qmm.matmul(x, q, xs, out=out, f32=f32, part=part)
        return y.copy_(_mod(y.double() + 1))


@pytest.fixture
def fast_dispatch(hashed, monkeypatch):
    """The hash model with ``qmm.matmul``'s FAST_MM hook (as in qmm) and ``_FastMM`` installed as fast_qmm."""

    from tensorfold.families.glm5_next.cuda import qmm

    def matmul(x, q, xs=None, *, out=None, f32=False, part=None):
        if qmm.FAST_MM is not None:
            fn, qmm.FAST_MM = qmm.FAST_MM, None
            try:
                return fn(x, q, xs, out=out, f32=f32, part=part)
            finally:
                qmm.FAST_MM = fn
        return hashed.matmul(x, q, xs, out=out, f32=f32, part=part)

    monkeypatch.setattr(qmm, "matmul", matmul)
    fq = _FastMM()
    monkeypatch.setattr(fastpf, "fast_qmm", fq)
    return fq


@pytest.mark.parametrize("tail", [1, 17, 63])
@pytest.mark.parametrize("full", [1, 3])
def test_lean_tail_sub_block_uses_the_chunk_kernels(fast_dispatch, tail, full):
    """A chunk of 64 k + tail rows in 64-row sub-blocks: its last sub-block (1-63 rows) runs the fast kernels, as the
    whole chunk does, and only the head's single row goes to qmm (in both paths)."""

    w = _hash_weights()
    R = 64 * full + tail
    ref = _hash_run(w, R, 64, block=None)
    assert fast_dispatch.small == [1]                        # the whole-chunk path: only the head
    fast_dispatch.small.clear()
    got = _hash_run(w, R, 64, block=64)
    assert fast_dispatch.small == [1], fast_dispatch.small
    assert _eq(ref, got), [torch.equal(x, y) for x, y in zip(ref, got)]
    fq = fastpf.fast_qmm                                                                       # restored
    assert fq.MIN_ROWS == 64 and getattr(fq.matmul_fast, "__func__", None) is _FastMM.matmul_fast


def test_lean_short_chunk_stays_on_qmm(fast_dispatch):
    """A chunk of fewer than 64 rows is one sub-block and runs qmm in both paths (no override)."""

    w = _hash_weights()
    ref = _hash_run(w, 40, 64, block=None)
    fast_dispatch.small.clear()
    got = _hash_run(w, 40, 64, block=64)
    assert _eq(ref, got) and set(fast_dispatch.small) == {40, 1}


def test_lean_tail_control(fast_dispatch, monkeypatch):
    """The control: without ``chunk_rows`` the 17-row tail goes to qmm and the bits differ."""

    import contextlib

    w = _hash_weights()
    ref = _hash_run(w, 64 + 17, 64, block=None)
    monkeypatch.setattr(lean, "chunk_rows", lambda w, R: contextlib.nullcontext())
    assert not _eq(ref, _hash_run(w, 64 + 17, 64, block=64))


def test_lean_routes_the_whole_chunk_once(hashed):
    """The point of the lean set: the routed experts run once a MoE layer over all rows, whatever the sub-blocks."""

    w = _hash_weights()
    moe = sum(1 for l in w.layers if l.moe is not None)
    hashed.calls.clear()
    _hash_run(w, 256, 0, block=64)
    assert hashed.calls["routed"] == moe
    assert hashed.calls["kda"] == 4 * sum(1 for l in w.layers if l.kind == "kda")        # 4 sub-blocks a layer


def test_lean_hash_model_catches_a_carry_bug(hashed, monkeypatch):
    """The control: without the conv-window carry between sub-blocks the hash model sees the difference."""

    w = _hash_weights()
    ref = _hash_run(w, 200, 0, block=None)
    monkeypatch.setattr(lean, "_carry", lambda *a: None)
    assert not _eq(ref, _hash_run(w, 200, 0, block=64))


def test_lean_equals_fast_chunk_non_exl3(hashed, monkeypatch):
    """The MLX checkpoint's MoE (qmm's grouped kernels, the shared expert a slot of them) through the lean set."""

    from tensorfold.families.glm5_next.cuda import qmm

    def gateup(x, xs, ex, group, act, axs, limit):
        R = x.shape[0]
        act[:R].copy_(_mod(x.double().sum(1)[:, None, None] + torch.arange(act.shape[1])[None, :, None]
                           + torch.arange(act.shape[2])))
        axs[:R].copy_(_mod(act[:R].double().reshape(R, act.shape[1], -1, 64).sum(-1)))

    def down(act, axs, ex, group, y):
        n = int(group.count[0])
        slots = act.shape[1]
        for u in range(n):
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
        assert _eq(_hash_run(w, R, 64, block=None), _hash_run(w, R, 64, block=64)), R


# -- the rule with lean chunks, through the engine's real prefill / snapshot / resume code ---------------------------
@pytest.mark.parametrize("policy", ["0", "2"])
def test_lean_rule_on_a_hostile_fake_model(monkeypatch, policy):
    """patches/0080's hostile fake model, with lean chunks whose bits depend on their sub-blocks too and a chunk grid
    (256) four times the window buffers (64): resumed == fresh and drafted == serial, the whole state compared."""

    from test_fastpf_patches import _fake_engine, _fake_request

    from tensorfold.families.glm5_next.cuda import decode

    rows, C = 64, 256
    rng = np.random.default_rng(82)

    def use(g):
        f = g.fake

        def compute(w, st, b, lb, R, *, head=True):
            assert R <= lb.rows and st.pos % C == 0
            f.compute(w, st, b, R, fast=True)
            f.hidden[:R] = torch.tensor([f.mix(int(h), R, st.pos, lb.block) for h in f.hidden[:R, 0]])[:, None]
            st.rec[1 - st.cur[0], 0] = f.hidden[R - 1, 0]
            return f.hidden[:R].clone()

        fake_lean = NS(stage=f.stage, compute=compute, commit=lambda w, st, lb, R: f.commit(w, st, None, R, R))
        for name in ("stage", "compute", "commit"):
            monkeypatch.setattr(decode, name, getattr(f, name))
        monkeypatch.setattr(decode, "lean_mod", fake_lean)

    def engine():
        g = _fake_engine(monkeypatch, rows, True)
        g.e.lean = NS(rows=C, block=rows)
        g.e.prefill_rows = C
        return g

    g, ref = engine(), engine()
    more = lambda n: [int(t) for t in rng.integers(0, 1000, size=n)]        # noqa: E731
    long = more(2 * C + 17)
    script = [long, more(5), long + more(3), long[:C + 1] + more(C), long + more(2 * C)]
    last, reply, resumed = [], [], 0
    for turn in range(30):
        if turn < len(script):
            prompt = script[turn]
        else:
            kind = rng.integers(0, 4)
            grow = more(int(rng.choice([1, 3, 30, C - 1, C, C + 5, 2 * C])))
            prompt = (last + reply + grow if kind == 0 else last + grow if kind == 1 else
                      last[:int(rng.integers(0, len(last) + 1))] + grow if kind == 2 else grow)[:3000]
        use(g)
        got, cached, state = _fake_request(g, prompt, policy=policy)
        assert cached % C == 0 and all(s.grid == C for s in g.cache)
        resumed += cached > 0
        use(ref)
        ref.cache = []
        want, c0, want_state = _fake_request(ref, prompt, policy=policy)
        assert c0 == 0 and got == want and state == want_state, (turn, len(prompt), cached)
        last, reply = prompt, got
    assert resumed >= 8, resumed


# -- memory on the real model's shapes (meta device) -----------------------------------------------------------------
def _real_weights():
    from tensorfold.families.glm5_next.cuda import weights

    kinds = ["dsa" if i % 4 == 3 else "kda" for i in range(45)]
    cfg = weights.Config(hidden=4096, layers=45, vocab=154880, eps=1e-5, heads=64, q_lora=1536, kv_lora=512,
                         qk_dim=256, v_dim=256, lin_heads=64, lin_dim=128, conv=4, lower=-5.0, experts=288, top_k=8,
                         moe_width=2048, shared_width=2048, dense_width=12288, routed_scale=2.5, norm_topk=True,
                         streams=4, hc_iters=20, hc_eps=1e-6, index_heads=32, index_dim=128, index_topk=2048, kpool=4,
                         limit=10.0, kinds=kinds, mlp_kinds=["dense"] * 3 + ["moe"] * 42, eos=(0,), mtp_layers=1,
                         group_size=64, bits=4, quant="exl3")
    layers = [NS(kind=k, index=i, kda=NS(proj=NS(n=12576))) for i, k in enumerate(kinds)]
    return NS(cfg=cfg, device=torch.device("meta"), world=2, head=NS(n=154880 // 2), layers=layers, mtp=object(),
              meta={"latent_kv": True, "long_context": True})


def _tensor_bytes(obj, seen=None) -> int:
    """Device bytes of the tensors an object holds (nested tensorfold objects too), each storage once (by id on the
    meta device, where views are counted again: a few small ones)."""

    seen = set() if seen is None else seen
    total = 0
    for v in vars(obj).values():
        for x in (v if isinstance(v, (list, tuple)) else [v]):
            if isinstance(x, torch.Tensor):
                if x.device.type == "cpu":
                    continue
                meta = x.device.type == "meta"
                key = id(x) if meta else x.untyped_storage().data_ptr()
                if key in seen:
                    continue
                seen.add(key)
                total += x.numel() * x.element_size() if meta else x.untyped_storage().nbytes()
            elif hasattr(x, "__dict__") and type(x).__module__.startswith("tensorfold"):
                total += _tensor_bytes(x, seen)
    return total


def test_lean_memory_real_shapes():
    """The lean set a rank holds at P = 2048 / 4096 / 8192 rows (EXL3, 5 DFlash2 taps): what ``budget`` says, linear in
    P (no quadratic scratch), ~397 KiB a row (3.10 GiB at 8192), against ~3 MiB a row of one window buffer set."""

    from tensorfold.families.glm5_next.cuda import forward

    w = _real_weights()
    got = {}
    for P in (2048, 4096, 8192):
        lb = lean.LeanBuffers(w, P, 1024, taps=5)
        got[P] = lb.nbytes()
        assert got[P] == sum(lean.budget(w, P, 1024, taps=5).values())
    per_row = (got[8192] - got[4096]) / 4096
    assert got[8192] - 2 * got[4096] == got[4096] - 2 * got[2048]          # linear up to the constant carry
    assert per_row < 420 * 1024 and got[8192] < 3.3 * 2**30, (per_row, got[8192] / 2**30)
    b = forward.Buffers(w, 1024, capacity=262152)
    per_buf_row = _tensor_bytes(b) / 1024
    assert per_buf_row > 5 * per_row, (per_buf_row, per_row)               # what the lean set avoids per row


# -- GPU: the synthetic EXL3 checkpoint --------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_lookup(monkeypatch):
    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")
    monkeypatch.setenv("GLM53_TF_NONEXPERT", "q4mse")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_lean")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


def _engine(path, *, lean_on: bool, fast: bool = True, rows: int = GRID, rows_max: int = GRID,
            block: int = BLOCK, context: int = 0, latent_kv: bool = False):
    from test_glm_engine import _TwoCopies

    from tensorfold.families.glm5_next.cuda import weights
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    with pytest.MonkeyPatch.context() as m:
        m.setattr(weights, "NONEXPERT", "q4mse")
        m.setenv("GLM53_TF_NONEXPERT", "q4mse")
        m.setenv("GLM53_TF_FAST_PREFILL", "1" if fast else "0")
        m.setenv("GLM53_TF_PREFILL_ROWS", str(rows))
        m.setenv("GLM53_TF_PREFILL_ROWS_MAX", str(rows_max))
        # patches/0085: snapshots on the chunk grid (patches/0080's rule, which these tests check; the 64-token
        # grid is tests/cuda/test_cindep_patches.py's)
        m.setenv("GLM53_TF_SNAPSHOT_GRID", str(max(64, rows // 64 * 64)))
        m.setenv("GLM53_TF_LEAN_PREFILL", "1" if lean_on else "0")
        m.setenv("GLM53_TF_LEAN_BLOCK", str(block))
        m.setenv("GLM53_TF_LATENT_KV", "1" if latent_kv else "0")
        m.setenv("GLM53_TF_LOOKUP", "0")
        for k in ("GLM53_TF_BATCH", "GLM53_TF_CALIB_ONLINE", "GLM53_TF_PROFILE", "GLM53_TF_DEPTH",
                  "GLM53_TF_FAST_GATHER"):
            m.delenv(k, raising=False)
        return GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", context=context,
                         comm=_TwoCopies())


@pytest.fixture(scope="module")
def el(ckpt):
    return _engine(ckpt, lean_on=True)


@pytest.fixture(scope="module")
def ef(ckpt):
    return _engine(ckpt, lean_on=False)


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
def test_lean_engine_is_lean(el, ef):
    e = el.e
    assert e.lean is not None and e.buf.rows == BLOCK and e.mbuf.rows == BLOCK and e.st.proj.shape[1] == BLOCK
    assert e.lean.rows == GRID and e.lean.block == BLOCK and e.prefill_max == GRID and el._grid() == GRID
    assert ef.e.lean is None and ef.e.buf.rows == GRID
    assert e.lean.nbytes() == sum(lean.budget(el.w, GRID, BLOCK, len(el.drafter.tap_layers)).values())
    def rows_bytes(eng):              # everything sized by rows: both window buffer sets, the lean set, State's rows
        ss = eng.e.st.scratch_set
        kda = sum(t.numel() * t.element_size() for t in (ss.out, ss.k, ss.v, ss.g, ss.b))
        lb = eng.e.lean.nbytes() if eng.e.lean is not None else 0
        return _tensor_bytes(eng.e.buf) + _tensor_bytes(eng.e.mbuf) + lb + kda + eng.e.st.proj.numel() * 2

    mine, theirs = rows_bytes(el), rows_bytes(ef)
    assert mine < 0.75 * theirs, (mine, theirs)


@gpu
@pytest.mark.parametrize("n", [3, 50, 256, 300, 321, 337, 383, 700])
def test_lean_state_equals_fast(el, ef, n):
    """Same bits as patches/0080's fast chunks of the same grid: every piece of committed state and the first token,
    for prompts below, on and across the grid (chunks of 256 = 4 sub-blocks, and partial ones; 321 / 337 / 383 / 700:
    a last chunk whose last sub-block has 1 / 17 / 63 / 60 rows, which must still run the fast kernels)."""

    from test_patches import _prefill_state, _same

    prompt = list(np.random.default_rng(820 + n).integers(0, 1000, size=n))
    a = _prefill_state(ef, prompt)
    b = _prefill_state(el, prompt)
    assert _same(a, b), [torch.equal(x, y) for x, y in zip(a, b)]
    assert torch.equal(el.e.st.conv, ef.e.st.conv)


@gpu
@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_lean_replies_equal_fast(el, ef, sampling):
    """Replies (MTP, DFlash2, auto, serial) equal the non-lean fast engine's, and drafted == serial on the lean one."""

    s = _sampling(sampling)
    prompt = list(np.random.default_rng(83).integers(0, 1000, size=600))
    serial = _cold(el, prompt, s, tokens=32)
    assert serial == _cold(ef, prompt, s, tokens=32)
    for policy in (None, "auto:1:1:0", "f3", "fc5:0.3", "2", "c3:0.35", "a:0.6:0.85"):
        for eng in (el, ef):
            eng.cache = []
            drafted, stats = _gen(eng, prompt, s, policy=policy, tokens=32)
            assert drafted == serial, policy
            assert stats["fast_prefill"] == GRID


@gpu
@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_lean_resumed_equals_fresh(el, sampling):
    s = _sampling(sampling)
    rng = np.random.default_rng(84)
    more = lambda n: [int(t) for t in rng.integers(0, 1000, size=n)]       # noqa: E731
    p1 = more(600)                                   # grid points 256, 512; the snapshot at 512
    el.cache = []
    r1, _ = _gen(el, p1, s, policy="auto:1:1:0")
    assert len(el.cache[-1].ids) == 512 and el.cache[-1].grid == GRID
    cases = [("within", p1 + more(3)),              # the new prompt's last grid point is still 512
             ("reply", p1 + r1 + more(5)),          # re-prefills 512.. (tail, reply, new) in one lean chunk
             ("across", p1 + r1 + more(200))]       # 824 tokens: a new snapshot at 768
    for name, prompt in cases:
        for policy in ("auto:1:1:0", "2", "f3"):
            _gen(el, p1, s, policy="auto:1:1:0")     # back to the state after p1 (its snapshot at 512)
            warm, stats = _gen(el, prompt, s, policy=policy)
            assert stats["cached"] == 512, (name, policy, stats["cached"])
            assert len(el.cache[-1].ids) == (len(prompt) // GRID) * GRID
            assert warm == _cold(el, prompt, s), (name, policy)
    p2 = p1 + r1 + more(200)                         # a chain: p3 resumes from the snapshot p2's resumed prefill made
    _gen(el, p1, s)
    _gen(el, p2, s)
    p3 = p2 + more(300)
    warm, stats = _gen(el, p3, s)
    assert stats["cached"] == 768 and warm == _cold(el, p3, s)


@gpu
def test_lean_knobs(el, ef):
    """Per-request prefill_rows up to the lean max (another grid: equal to the non-lean engine with the same knob);
    exact requests on the lean engine run chunks of at most the window buffers' rows (same reply as the non-lean
    engine's 256-row exact chunks); past the max: refused."""

    s = _sampling("sampled")
    prompt = list(np.random.default_rng(85).integers(0, 1000, size=500))
    for knobs in ({"prefill_rows": 192}, {"prefill_rows": 128}, {"fast_prefill": 0, "prefill_rows": 256}):
        a = _cold(el, prompt, s, knobs=knobs)
        b = _cold(ef, prompt, s, knobs=knobs)
        assert a == b, knobs
    with pytest.raises(ValueError, match="prefill_rows"):
        el.parse_knobs({"prefill_rows": GRID + 1})
    assert el.parse_knobs({"prefill_rows": GRID}) == {"prefill_rows": GRID}


@gpu
def test_lean_deterministic(el):
    from test_patches import _prefill_state, _same

    prompt = list(np.random.default_rng(86).integers(0, 1000, size=700))
    a = _prefill_state(el, prompt)
    _gen(el, list(np.random.default_rng(87).integers(0, 1000, size=90)), None, tokens=4)
    assert _same(a, _prefill_state(el, prompt))


@pytest.fixture(scope="module")
def long_ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter
    from test_patches import _index_heads_32

    path = tmp_path_factory.mktemp("glm_lean_long")
    _checkpoint(path / "model", exl3=True)
    _index_heads_32(path / "model")
    _drafter(path / "dflash2")
    return path


@gpu
def test_lean_long_context_latent(long_ckpt):
    """3,000 tokens across the dense limit on the latent cache, chunks of 512 in sub-blocks of 128 (sparse rows,
    the index ring, the blocked selection per sub-block): the non-lean fast engine's state and replies."""

    from test_patches import _prefill_state, _same

    kw = dict(rows=512, rows_max=512, context=4096, latent_kv=True)
    a_eng = _engine(long_ckpt, lean_on=True, block=128, **kw)
    prompt = list(np.random.default_rng(88).integers(0, 1000, size=3000))
    a = _prefill_state(a_eng, prompt)
    ra = _cold(a_eng, prompt, None)
    da, _ = _gen(a_eng, prompt, None, policy="2")
    del a_eng
    torch.cuda.empty_cache()
    b_eng = _engine(long_ckpt, lean_on=False, **kw)
    b = _prefill_state(b_eng, prompt)
    assert _same(a, b), [torch.equal(x, y) for x, y in zip(a, b)]
    assert ra == _cold(b_eng, prompt, None) == da
