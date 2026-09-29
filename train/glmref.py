"""PyTorch reference of the two drafters TensorFold's GLM-5.3-Flash engine runs: the MTP head (the checkpoint's nextn
layer, ``cuda/mtp.py``) and the DFlash2 block drafter (``cuda/dflash2.py``), written to follow the engine's arithmetic
step by step (docs/DRAFTER-TRAINING.md §3) and differentiable, so the same code is the trainer's model, the offline
acceptance evaluator and the parity check against the engine (tests/cuda/test_drafter_ref_patches.py).

What "the engine's arithmetic" means here, per op (the engine's kernel in brackets):

- RMSNorm [glue.rmsnorm]: y = bf16(w * bf16(x * rsqrt(mean(x^2) + eps))), fp32 inside.
- Non-expert projections [qmm.matmul]: the engine stores every BF16 non-expert weight of an EXL3 checkpoint as
  affine 4-bit groups of 64 (GLM53_TF_NONEXPERT=q4mse in production: ``quantize4(mse=True)``). ``Quant`` reproduces
  that quantization bit for bit (the same torch ops as ``qmm.quantize4`` / ``_clip_search``) and the matmul is the
  dequantized weight in fp32, accumulated in fp32, rounded once to bf16 (or kept fp32 for row-parallel partials).
  The engine's kernel sums groups in a fixed order, so outputs agree to fp32 rounding, not bitwise.
- Latent MLA [latent.py]: q' = bf16(W_k,h^T q_h), scores against the cached latent rows (FP8 rows with
  GLM53_TF_KV_DTYPE=fp8: ``fp8_latent`` is patches/0220's row rule), softmax in fp32, u = sum(bf16(p) c) / sum(p),
  o_h = bf16(W_v,h u_h). DSA's top-2,048 selection past 2,051 keys follows ``sparse.py``'s definition
  (``Indexer``): dense attention below it, exactly as the engine.
- MoE [glue.router / select / combine, exl3_mm]: sigmoid scores, top-8 by score + bias, weights score / sum x 2.5;
  routed experts from the EXL3 trellis (``exl3_weight``: the engine's reference decoder, float64, then the storage
  dtype); gate / up rounded to bf16, act = bf16(bf16(silu(min(g, L))) * clip(u, L)); the shared expert as a q4mse
  MLP; fp32 combine, one bf16 rounding into the residual.
- Heads [qmm.matmul on w.head]: the target's head as the engine stores it (q4mse of the checkpoint's BF16 head).

Tensor parallelism: the engine splits heads, expert width and vocabulary over two ranks and adds fp32 partials rank 0
first; the reference computes the whole layer on one device (the partial sums differ by fp32 rounding only).
``shard=(rank, 2)`` instead builds rank ``rank``'s half only, which is what the one-GPU engine tests run (rank 0 with
no communicator: its own partial alone).

Nothing here imports the engine's package (no triton): the EXL3 decoder is loaded from its file
(``vendor/TensorFold/.../cuda/exl3.py``, numpy + torch only) when routed experts are read.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BF = torch.bfloat16
PREFIX = "model.language_model."
REPO = Path(__file__).resolve().parents[1]
EXL3_PY = REPO / "vendor/TensorFold/src/tensorfold/families/glm5_next/cuda/exl3.py"


# -- checkpoint reading (no safetensors dependency) ------------------------------------------------------------------
_DT = {"BF16": (torch.bfloat16, 2), "F16": (torch.float16, 2), "F32": (torch.float32, 4), "I16": (torch.int16, 2),
       "I32": (torch.int32, 4), "U32": (torch.int32, 4), "I64": (torch.int64, 8), "U8": (torch.uint8, 1),
       "I8": (torch.int8, 1), "U16": (torch.int16, 2), "F64": (torch.float64, 8)}


def read_header(path: str | Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


class Checkpoint:
    """Full (unsplit) tensors of a safetensors checkpoint folder by name, as CPU torch tensors."""

    def __init__(self, model_dir: str | Path) -> None:
        self.dir = Path(model_dir)
        idx = self.dir / "model.safetensors.index.json"
        if idx.exists():
            self.where = {k: self.dir / v for k, v in json.loads(idx.read_text())["weight_map"].items()}
        else:
            self.where = {}
            for p in sorted(self.dir.glob("*.safetensors")):
                for k in read_header(p)[0]:
                    if k != "__metadata__":
                        self.where[k] = p
        self._hdr: dict = {}
        self._map: dict = {}

    def __contains__(self, name: str) -> bool:
        return name in self.where

    def names(self) -> list[str]:
        return list(self.where)

    def get(self, name: str) -> torch.Tensor:
        p = self.where[name]
        if p not in self._hdr:
            self._hdr[p] = read_header(p)
            self._map[p] = np.memmap(p, dtype=np.uint8, mode="r")
        hdr, base = self._hdr[p]
        info = hdr[name]
        a, b = info["data_offsets"]
        dt, _ = _DT[info["dtype"]]
        raw = np.array(self._map[p][base + a:base + b], copy=True)
        return torch.from_numpy(raw).view(dt).reshape(info["shape"])

    def config(self) -> dict:
        return json.loads((self.dir / "config.json").read_text())


def write_safetensors(path: str | Path, tensors: dict[str, torch.Tensor], metadata: dict | None = None) -> None:
    """Minimal safetensors writer (BF16 / F16 / F32 / I32 / U8 ...), sorted names, 8-byte aligned header."""

    rev = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.int32: "I32",
           torch.int16: "I16", torch.uint8: "U8", torch.int64: "I64", torch.float64: "F64", torch.int8: "I8"}
    header: dict = {"__metadata__": {k: str(v) for k, v in (metadata or {}).items()}}
    blobs, off = [], 0
    for name in sorted(tensors):
        t = tensors[name].detach().contiguous().cpu()
        raw = t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.bfloat16 else \
            t.view(torch.int16).numpy().tobytes()
        header[name] = {"dtype": rev[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


# -- the engine's number formats -----------------------------------------------------------------------------------
def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (w.float() * (xf * r).to(BF).float()).to(BF)


def _clip_search(g, lo, hi, factors=(1.0, 0.97, 0.94, 0.91, 0.88, 0.85, 0.82, 0.79, 0.76)):
    """``qmm._clip_search`` (the same ops, so the same ranges)."""

    best_err = best_lo = best_hi = None
    mid, half = (hi + lo) / 2, (hi - lo) / 2
    for f in factors:
        l, h = mid - half * f, mid + half * f
        s = ((h - l) / 15).clamp_min(1e-8).to(BF).float()
        b = l.to(BF).float()
        q = torch.round((g - b[..., None]) / s[..., None]).clamp(0, 15)
        err = ((q * s[..., None] + b[..., None] - g) ** 2).sum(-1)
        if best_err is None:
            best_err, best_lo, best_hi = err, l, h
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_lo, best_hi = torch.where(better, l, best_lo), torch.where(better, h, best_hi)
    return best_lo, best_hi


@torch.no_grad()
def quantize4(w: torch.Tensor, mse: bool = False, chunk: int = 8192) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``qmm.quantize4`` without the packing: (codes uint8 [N, K], scales bf16 [N, K/64], biases bf16 [N, K/64])."""

    n, k = w.shape
    codes = torch.empty((n, k), dtype=torch.uint8, device=w.device)
    scales = torch.empty((n, k // 64), dtype=BF, device=w.device)
    biases = torch.empty_like(scales)
    for r in range(0, n, chunk):
        g = w[r:r + chunk].float().view(-1, k // 64, 64)
        lo, hi = g.amin(-1), g.amax(-1)
        if mse:
            lo, hi = _clip_search(g, lo, hi)
        scale = ((hi - lo) / 15).clamp_min(1e-8).to(BF)
        bias = lo.to(BF)
        q = torch.round((g - bias.float()[..., None]) / scale.float()[..., None]).clamp(0, 15)
        codes[r:r + chunk] = q.view(-1, k).to(torch.uint8)
        scales[r:r + chunk], biases[r:r + chunk] = scale, bias
    return codes, scales, biases


def dequant4(codes: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
    """fp32 values s * q + b (exact in fp32), [N, K]."""

    n, k = codes.shape
    return (codes.float().view(n, k // 64, 64) * scales.float()[..., None] + biases.float()[..., None]).view(n, k)


def fake_quant(w: torch.Tensor, mode: str) -> torch.Tensor:
    """The weight the engine multiplies by, for a BF16 weight stored as ``mode`` (bf16 | q4 | q4mse), in fp32.
    Differentiable with a straight-through estimator (quantization-aware training)."""

    if mode == "bf16":
        return w.float() + (w.to(BF).float() - w.float()).detach()
    q = dequant4(*quantize4(w.detach().to(BF), mse=mode == "q4mse"))      # the engine quantizes the BF16 export
    return w.float() + (q - w.float()).detach()


class Quant(nn.Module):
    """A linear weight [N, K] (BF16 master) and the storage format the engine gives it at load. ``cache``: frozen
    weights keep their dequantized fp32 copy (no re-quantization a step)."""

    def __init__(self, weight: torch.Tensor, mode: str, trainable: bool = False) -> None:
        super().__init__()
        self.mode = mode
        self.weight = nn.Parameter(weight.to(BF).clone(), requires_grad=trainable)
        self._frozen: torch.Tensor | None = None

    def w(self) -> torch.Tensor:
        if not self.weight.requires_grad:
            if self._frozen is None or self._frozen.device != self.weight.device:
                self._frozen = fake_quant(self.weight.detach(), self.mode)
            return self._frozen
        self._frozen = None
        return fake_quant(self.weight, self.mode)

    def forward(self, x: torch.Tensor, f32: bool = False) -> torch.Tensor:
        y = x.float() @ self.w().t()
        return y if f32 else y.to(BF)


def fp8_latent(x: torch.Tensor) -> torch.Tensor:
    """patches/0220's FP8 latent row (``latent.quantize_rows_reference``) read back as bf16: per row the smallest
    power of two s with amax / s <= 448, e4m3 values (nearest even), then e4m3 x s (exact in bf16)."""

    xf = x.float()
    amax = xf.abs().amax(-1)
    bits = amax.contiguous().view(torch.int32)
    E = (bits >> 23) & 0xFF
    e = E - 135 + ((bits & 0x7FFFFF) > 0x600000).to(torch.int32)
    e = torch.where(amax > 0, e, torch.zeros_like(e)).clamp(-126, 126)
    scale = ((e + 127) << 23).view(torch.float32)
    inv = ((127 - e) << 23).view(torch.float32)
    q = (xf * inv[..., None]).to(torch.float8_e4m3fn).float()
    out = (q * scale[..., None]).to(BF)
    return x + (out.float() - x.float()).detach().to(x.dtype) if x.requires_grad else out


# -- EXL3 routed experts ---------------------------------------------------------------------------------------------
_EXL3 = None


def exl3_module():
    """The engine's EXL3 reference decoder (``cuda/exl3.py``), loaded from its file (no triton import)."""

    global _EXL3
    if _EXL3 is None:
        path = Path(os.environ.get("GLM53_EXL3_PY", str(EXL3_PY)))
        spec = importlib.util.spec_from_file_location("tf_exl3_ref", path)
        _EXL3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_EXL3)
    return _EXL3


def _states_torch(trellis: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """``exl3.states`` in torch (any device): the 16-bit state of every value, [K/16, N/16, 256] int64."""

    t = trellis.to(torch.int64) & 0xFFFF
    words = t[..., 0::2] | (t[..., 1::2] << 16)
    nw = 8 * bits
    p = torch.arange(256, device=t.device)
    first = p * bits + bits - 16 + 256 * bits
    last = first + 16
    i0, i1 = (first // 32) % nw, ((last - 1) // 32) % nw
    shift = ((last - 1) // 32 + 1) * 32 - last
    a, b = words[..., i0], words[..., i1]
    return (((a << 32) | b) >> shift) & 0xFFFF


def exl3_weight(trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, dtype=BF,
                device: str | torch.device | None = None) -> torch.Tensor:
    """W [K, N] with y = x @ W: diag(suh) H_K W_q H_N diag(svh) (``exl3.dequantize``), on ``device`` in float64
    arithmetic (the codebook and tile order from the engine's decoder), stored as ``dtype``."""

    ex = exl3_module()
    dev = torch.device(device) if device is not None else trellis.device
    s = _states_torch(trellis.to(dev))
    kt, nt = s.shape[0], s.shape[1]
    book = torch.from_numpy(ex.mcg_values().astype(np.float64)).to(dev)
    vals = book[s]                                               # [kt, nt, 256]
    rows, cols = (torch.from_numpy(a).to(dev) for a in ex.tile_positions())
    wq = torch.zeros((kt, 16, nt, 16), dtype=torch.float64, device=dev)
    wq[:, rows, :, cols] = vals.permute(2, 0, 1)
    wq = wq.reshape(kt * 16, nt * 16)
    h = torch.from_numpy(ex.hadamard()).to(dev) / math.sqrt(ex.HAD)
    K, N = wq.shape
    w = (wq.T.reshape(N, K // 128, 128) @ h).reshape(N, K).T          # H_K on the rows (inputs)
    w = w * suh.to(dev).double()[:, None]
    w = (w.reshape(K, N // 128, 128) @ h).reshape(K, N)               # H_N on the columns (outputs)
    w = w * svh.to(dev).double()[None, :]
    return w.to(dtype)


# -- configuration ------------------------------------------------------------------------------------------------
@dataclass
class Cfg:
    hidden: int
    layers: int
    vocab: int
    eps: float
    heads: int
    q_lora: int
    kv_lora: int
    qk_dim: int
    v_dim: int
    experts: int
    top_k: int
    moe_width: int
    shared_width: int
    routed_scale: float
    norm_topk: bool
    limit: float
    index_heads: int
    index_dim: int
    index_topk: int
    kpool: int
    eos: tuple

    @classmethod
    def read(cls, raw: dict) -> "Cfg":
        t = dict(raw.get("text_config") or raw)
        eos = t.get("eos_token_id", raw.get("eos_token_id"))
        return cls(hidden=int(t["hidden_size"]), layers=int(t["num_hidden_layers"]), vocab=int(t["vocab_size"]),
                   eps=float(t["rms_norm_eps"]), heads=int(t["num_attention_heads"]), q_lora=int(t["q_lora_rank"]),
                   kv_lora=int(t["kv_lora_rank"]),
                   qk_dim=int(t["qk_nope_head_dim"]) + int(t.get("qk_rope_head_dim", 0)), v_dim=int(t["v_head_dim"]),
                   experts=int(t["n_routed_experts"]), top_k=int(t["num_experts_per_tok"]),
                   moe_width=int(t["moe_intermediate_size"]),
                   shared_width=int(t["moe_intermediate_size"]) * int(t.get("n_shared_experts", 1)),
                   routed_scale=float(t["routed_scaling_factor"]), norm_topk=bool(t.get("norm_topk_prob", True)),
                   limit=float(t.get("swiglu_limit", 10.0)), index_heads=int(t.get("index_n_heads", 32)),
                   index_dim=int(t.get("index_head_dim", 128)), index_topk=int(t.get("index_topk", 2048)),
                   kpool=int(t.get("index_kpool", 4)),
                   eos=tuple(eos) if isinstance(eos, list) else (int(eos),))

    @property
    def dense_limit(self) -> int:
        return self.index_topk + self.kpool - 1


def is_exl3(raw: dict) -> bool:
    q = raw.get("quantization") or raw.get("quantization_config") or {}
    return str(q.get("quant_method") or "").lower() == "exl3"


# -- DSA's indexer (token selection past the dense limit) -------------------------------------------------------------
class Indexer(nn.Module):
    """``sparse.py``'s selection: per token k = LayerNorm(wk x), gate g = x . compress_gate; pools of 4 with key
    bf16(sum_j bf16(bf16(softmax_j(g_j + ape_j)) * k_j)); query qi = wq_b(q residual) (heads of 128) with head weights
    weights_proj(x) / sqrt(heads); pool score sum_h w_h relu(qi_h . pool / sqrt(128)); the best 512 complete pools
    plus the visible tokens of the incomplete last pool. Frozen (it only chooses keys)."""

    def __init__(self, cfg: Cfg, wk, wproj, wq_b, ln_w, ln_b, gate, ape, mode: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.kw = Quant(torch.cat([wk, wproj]), mode)
        self.qb = Quant(wq_b, mode)
        self.register_buffer("ln_w", ln_w.to(BF))
        self.register_buffer("ln_b", ln_b.to(BF))
        self.register_buffer("gate", gate.to(BF))
        self.register_buffer("ape", ape.to(BF))

    @torch.no_grad()
    def keys(self, normed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> (index keys bf16 [T, 128], head weights fp32 [T, H], gates fp32 [T, 128])."""

        c = self.cfg
        kr = self.kw(normed)
        k = F.layer_norm(kr[:, :c.index_dim].float(), (c.index_dim,), self.ln_w.float(), self.ln_b.float(),
                         1e-6).to(BF)
        wts = kr[:, c.index_dim:].float() / math.sqrt(c.index_heads)
        g = (normed.float() @ self.gate.float().t()).to(BF)          # the gate cache holds bf16 rows
        return k, wts, g

    @torch.no_grad()
    def pools(self, k: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Pool keys of every complete pool [T // 4, 128] bf16."""

        P = self.cfg.kpool
        n = k.shape[0] // P
        kk = k[:n * P].view(n, P, -1).float()
        gg = g[:n * P].view(n, P, -1).float() + self.ape.float()[None]
        wgt = torch.softmax(gg, dim=1).to(BF).float()
        return (wgt * kk).to(BF).float().sum(1).to(BF)

    @torch.no_grad()
    def select(self, qr: torch.Tensor, wts: torch.Tensor, pools: torch.Tensor, qpos: torch.Tensor,
               chunk: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokens each query row (at positions ``qpos``) attends to: (idx [T, M] int64 ascending, ok [T, M] bool),
        M = index_topk + kpool - 1 = 2,051. Rows before the dense limit keep every token 0..q."""

        c = self.cfg
        H, Dh, P = c.index_heads, c.index_dim, c.kpool
        M = c.dense_limit
        top = c.index_topk // P
        dev = qr.device
        qi = self.qb(qr).float().view(-1, H, Dh)
        T = qi.shape[0]
        idx = torch.zeros((T, M), dtype=torch.long, device=dev)
        ok = torch.zeros((T, M), dtype=torch.bool, device=dev)
        lane = torch.arange(M, device=dev)
        pk = pools.float()
        for a in range(0, T, chunk):
            q = qpos[a:a + chunk].long()
            n = q.shape[0]
            dense = q < c.dense_limit
            idx[a:a + n] = torch.where(dense[:, None], lane[None].expand(n, M), idx[a:a + n])
            ok[a:a + n] = torch.where(dense[:, None], lane[None] <= q[:, None], ok[a:a + n])
            if bool(dense.all()):
                continue
            npool = (q + 1) // P
            NPm = int(npool.max())
            s = torch.relu(torch.einsum("thd,pd->thp", qi[a:a + n], pk[:NPm]) / math.sqrt(Dh))
            score = (wts[a:a + n].float()[:, :, None] * s).sum(1)
            score = score.masked_fill(torch.arange(NPm, device=dev)[None] >= npool[:, None], float("-inf"))
            keep = torch.topk(score, top, dim=-1).indices.sort(dim=-1).values          # [n, 512]
            body = (keep[:, :, None] * P + torch.arange(P, device=dev)).reshape(n, -1)
            tl_ = torch.arange(P - 1, device=dev)
            tail = npool[:, None] * P + tl_
            tail_ok = tl_[None] < (q + 1 - npool * P)[:, None]
            sp_idx = torch.cat([body, tail], dim=1)
            sp_ok = torch.cat([torch.ones_like(body, dtype=torch.bool), tail_ok], dim=1)
            idx[a:a + n] = torch.where(dense[:, None], idx[a:a + n], sp_idx)
            ok[a:a + n] = torch.where(dense[:, None], ok[a:a + n], sp_ok)
        return torch.where(ok, idx, torch.zeros_like(idx)), ok


# -- the MTP head ------------------------------------------------------------------------------------------------------
TRAINABLE_DEFAULT = ("eh", "enorm", "hnorm", "in_norm", "post_norm", "q_a_kv_a", "q_norm", "kv_norm", "q_b", "kv_b",
                     "o", "shared", "router", "norm")


class MTP(nn.Module):
    """GLM-5.3-Flash's MTP layer as the engine runs it (``mtp.mtp_compute``):

        x = eh_proj([enorm(embed(t_{i+1})) | hnorm(h_i)])          embedding zeroed at the head's first position
        x = x + attn(input_layernorm(x))                           latent MLA over the head's own cache, plain residual
        x = x + moe(post_attention_layernorm(x))
        out = shared_head.norm(x);  logits = head(out)             ``out`` is the next chained step's h

    ``embed`` [V, D] bf16 and ``head`` (the target's, as the engine stores it) are frozen and shared with the main
    model. ``mode``: the non-expert storage (q4mse in production; bf16 for a BF16-head engine)."""

    def __init__(self, cfg: Cfg, t: dict[str, torch.Tensor], experts: dict[str, torch.Tensor], embed: torch.Tensor,
                 head: torch.Tensor, *, mode: str = "q4mse", kv_fp8: bool = True,
                 trainable: Iterable[str] = (), expert_dtype=BF, shard: tuple[int, int] | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.kv_fp8 = kv_fp8
        tr = set(trainable)
        D, H = cfg.hidden, cfg.heads
        self.shard = shard
        r, world = shard if shard is not None else (0, 1)
        HL = H // world
        hs = slice(r * HL, (r + 1) * HL)

        def P(x):
            return nn.Parameter(x.to(BF).clone(), requires_grad=False)

        self.enorm, self.hnorm = P(t["enorm"]), P(t["hnorm"])
        self.in_norm, self.post_norm, self.norm = P(t["in_norm"]), P(t["post_norm"]), P(t["norm"])
        self.q_norm, self.kv_norm = P(t["q_norm"]), P(t["kv_norm"])
        for name in ("enorm", "hnorm", "in_norm", "post_norm", "norm", "q_norm", "kv_norm"):
            getattr(self, name).requires_grad_(name in tr)
        self.eh = Quant(t["eh"], mode, "eh" in tr)
        self.q_a_kv_a = Quant(torch.cat([t["q_a"], t["kv_a"]]), mode, "q_a_kv_a" in tr)
        qk, vd = cfg.qk_dim, cfg.v_dim
        self.q_b = Quant(t["q_b"].view(H, qk, -1)[hs].reshape(HL * qk, -1), mode, "q_b" in tr)
        kvb = t["kv_b"].view(H, qk + vd, -1)[hs]
        self.kv_k = Quant(kvb[:, :qk].reshape(HL * qk, -1), mode, "kv_b" in tr)
        self.kv_v = Quant(kvb[:, qk:].reshape(HL * vd, -1), mode, "kv_b" in tr)
        self.o = Quant(t["o"].view(D, H, vd)[:, hs].reshape(D, HL * vd), mode, "o" in tr)
        self.heads = HL
        self.index = Indexer(cfg, t["ix_wk"], t["ix_wproj"], t["ix_wq_b"], t["ix_ln_w"], t["ix_ln_b"], t["ix_gate"],
                             t["ix_ape"], mode) if "ix_wk" in t else None
        self.router = nn.Parameter(t["router"].to(BF).clone(), requires_grad="router" in tr)
        self.register_buffer("bias", t["router_bias"].float())
        sw = cfg.shared_width // world
        ss = slice(r * sw, (r + 1) * sw)
        self.s_gu = Quant(torch.cat([t["s_gate"][ss], t["s_up"][ss]]), mode, "shared" in tr)
        self.s_down = Quant(t["s_down"][:, ss], mode, "shared" in tr)
        # routed experts, frozen: gate / up [E, K=D, N=NI], down [E, K=NI, N=D] (y = x @ W)
        ni = cfg.moe_width // world
        es = slice(r * ni, (r + 1) * ni)
        self.register_buffer("e_gate", experts["gate"][:, :, es].to(expert_dtype).contiguous(), persistent=False)
        self.register_buffer("e_up", experts["up"][:, :, es].to(expert_dtype).contiguous(), persistent=False)
        self.register_buffer("e_down", experts["down"][:, es, :].to(expert_dtype).contiguous(), persistent=False)
        self.register_buffer("embed", embed.to(BF), persistent=False)
        V = head.shape[0]
        vs = slice(r * (V // world), (r + 1) * (V // world))
        self.register_buffer("head", head[vs].to(BF).contiguous(), persistent=False)

    # -- pieces ------------------------------------------------------------------------------------------------------
    def inputs(self, hidden: torch.Tensor, tokens: torch.Tensor, zero_first: torch.Tensor | None = None) -> torch.Tensor:
        """x = eh_proj([enorm(e) | hnorm(h)]) for rows (h, next token); ``zero_first``: rows whose embedding is zero
        (the head's first cache position)."""

        c = self.cfg
        e = self.embed[tokens.long()]
        if zero_first is not None:
            e = torch.where(zero_first[:, None], torch.zeros_like(e), e)
        cat = torch.cat([rmsnorm(e, self.enorm, c.eps), rmsnorm(hidden.to(BF), self.hnorm, c.eps)], dim=-1)
        return self.eh(cat)

    def project(self, x: torch.Tensor):
        """Attention inputs of rows x: (normed, q residual, absorbed queries q' [T, H, L] bf16, latent rows c [T, L])."""

        c = self.cfg
        normed = rmsnorm(x, self.in_norm, c.eps)
        dp = self.q_a_kv_a(normed)
        qr = rmsnorm(dp[:, :c.q_lora], self.q_norm, c.eps)
        lat = rmsnorm(dp[:, c.q_lora:], self.kv_norm, c.eps)
        if self.kv_fp8:
            lat = fp8_latent(lat)
        q = self.q_b(qr).view(-1, self.heads, c.qk_dim)
        wk = self.kv_k.w().view(self.heads, c.qk_dim, c.kv_lora)
        qa = torch.einsum("thd,hdl->thl", q.float(), wk).to(BF)
        return normed, qr, qa, lat

    def expand(self, u: torch.Tensor) -> torch.Tensor:
        """u [T, H, L] fp32 -> o [T, H * v_dim] bf16 (latent.expand), then o_proj as an fp32 partial."""

        c = self.cfg
        wv = self.kv_v.w().view(self.heads, c.v_dim, c.kv_lora)
        o = torch.einsum("thl,hvl->thv", u, wv).to(BF).reshape(u.shape[0], -1)
        return self.o(o, f32=True)

    def moe(self, x: torch.Tensor) -> torch.Tensor:
        """The MoE branch of rows x (the residual stream, bf16) -> fp32 [T, D] (this shard's partial)."""

        c = self.cfg
        n2 = rmsnorm(x, self.post_norm, c.eps)
        logits = n2.float() @ self.router.float().t()
        score = torch.sigmoid(logits)
        pick = torch.topk(score + self.bias, c.top_k, dim=-1).indices
        wts = torch.gather(score, 1, pick)
        if c.norm_topk:
            wts = wts / (wts.sum(-1, keepdim=True) + 1e-20)
        wts = wts * c.routed_scale
        out = torch.zeros((x.shape[0], c.hidden), dtype=torch.float32, device=x.device)
        L = c.limit
        flat = pick.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=c.experts).tolist()
        rows_all = (torch.arange(flat.numel(), device=x.device) // c.top_k)[order]
        slot_all = order
        at = 0
        for e, n in enumerate(counts):
            if n == 0:
                continue
            rows = rows_all[at:at + n]
            slots = slot_all[at:at + n]
            at += n
            xe = n2[rows].to(self.e_gate.dtype)
            g = (xe @ self.e_gate[e]).to(BF).float().clamp(max=L)
            u = (xe @ self.e_up[e]).to(BF).float().clamp(-L, L)
            act = ((g * torch.sigmoid(g)).to(BF).float() * u).to(BF).float()
            y = act @ self.e_down[e].float()
            out.index_add_(0, rows, y * wts.reshape(-1)[slots][:, None])
        gu = self.s_gu(n2)
        sw = gu.shape[1] // 2
        g = gu[:, :sw].float().clamp(max=L)
        u = gu[:, sw:].float().clamp(-L, L)
        act = ((g * torch.sigmoid(g)).to(BF).float() * u).to(BF)
        return out + self.s_down(act, f32=True)

    def logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(shared_head.norm(x): the next step's h, logits [T, V / world] bf16)."""

        out = rmsnorm(x, self.norm, self.cfg.eps)
        return out, out @ self.head.t()          # bf16 inputs, fp32 accumulation, one bf16 rounding (as qmm)

    # -- attention with the chained-step keys -----------------------------------------------------------------------
    def attend(self, qa: torch.Tensor, keys: torch.Tensor, allow: torch.Tensor | list, extra: list[torch.Tensor],
               chunk: int = 256) -> torch.Tensor:
        """Latent attention for T query rows. ``keys`` [N, L] (the head's cache: its first-step rows), ``allow``: a
        bool mask [T, N] or (idx [T, M], ok [T, M]) (DSA selection, ``Indexer.select``), ``extra``: each [T, L], one more key per row
        (the rows' own chained-step keys, in order; the last is the row's own). -> u [T, H, L] fp32."""

        c = self.cfg
        scale = c.qk_dim ** -0.5
        outs = []
        for a in range(0, qa.shape[0], chunk):
            b = min(a + chunk, qa.shape[0])
            q = qa[a:b].float()                                             # [t, H, L]
            if isinstance(allow, torch.Tensor):
                s = torch.einsum("thl,nl->thn", q, keys.float()) * scale
                s = s.masked_fill(~allow[a:b, None, :], float("-inf"))
                vals = keys.float()[None].expand(b - a, -1, -1)
            else:                                                           # DSA selection: (idx, ok) per row
                idx, ok = allow[0][a:b], allow[1][a:b]
                vals = keys.float()[idx]                                    # [t, m, L]
                s = torch.einsum("thl,tml->thm", q, vals) * scale
                s = s.masked_fill(~ok[:, None, :], float("-inf"))
            if extra:
                ex = torch.stack([k[a:b].float() for k in extra], dim=1)    # [t, E, L]
                s = torch.cat([s, torch.einsum("thl,tel->the", q, ex) * scale], dim=-1)
                vals = torch.cat([vals, ex], dim=1)
            m = s.amax(-1, keepdim=True)
            p = torch.exp(s - m)
            den = p.sum(-1, keepdim=True)
            u = torch.einsum("thn,tnl->thl", p.to(BF).float(), vals) / den
            outs.append(u)
        return torch.cat(outs, dim=0)

    # -- whole passes -------------------------------------------------------------------------------------------
    def chain(self, hidden: torch.Tensor, tokens: torch.Tensor, steps: int = 3, *, first: int = 0, ctx: int = 0,
              sparse: bool = True, reduce=None):
        """Teacher-forced training-time recurrence over one document (FastMTP's shared-weight chain).

        ``hidden`` [N, D]: the main model's final-normed rows at local positions 0..N-1 (absolute ``first`` + i),
        ``tokens`` [N + 1]: the token ids at local positions 0..N (tokens[i + 1] follows row i). Step 1 absorbs every
        position, as the engine builds the head's cache: entry i reads (h_i, t_{i+1}) and predicts t_{i+2}. For an
        anchor entry a, step k >= 2 sits at entry e = a + k - 1: it reads step k-1's output of the same anchor and
        t_{e+1}, and attends to step-1 entries 0..a plus the anchor's own entries of steps 2..k (the keys a drafted
        chain sees at inference, where drafted tokens equal the true ones up to the first miss). The head's first
        cache entry (absolute position 0) has its embedding zeroed. Rows before ``ctx`` are context only (step-1
        keys, no gradient, no outputs).

        Attention: dense over the rows given while they number at most 2,051 (the engine's rule: exact). Longer:
        with ``sparse`` and a document from position 0 (``first`` 0), DSA's top-2,048 selection (``Indexer``) as the
        engine; ``sparse=False`` keeps dense attention over the rows given (a training approximation for a window
        cut out of a long document: the head sees the last ``N`` positions instead of the selected ones).

        -> [logits of step k: [N - ctx - (k - 1), V / world] bf16]; row j of step k predicts the token at local
        position ctx + j + k + 1, i.e. the target's distribution at main row ctx + j + k. ``reduce(k, logits)``: what
        to keep of each step's logits instead (evaluation: a few numbers a row, not [rows, 154,880])."""

        N = hidden.shape[0]
        dev = hidden.device
        zf = (torch.arange(N, device=dev) + first) == 0
        if ctx:
            with torch.no_grad():
                x0c = self.inputs(hidden[:ctx], tokens[1:ctx + 1], zf[:ctx])
                n0c, _, _, lat0c = self.project(x0c)
                ixc = self.index.keys(n0c) if self.index is not None else None
        x = self.inputs(hidden[ctx:], tokens[ctx + 1:N + 1], zf[ctx:])
        normed, qr, qa, lat = self.project(x)
        keys = torch.cat([lat0c, lat]) if ctx else lat
        sel = None
        if sparse and self.index is not None and first + N > self.cfg.dense_limit:
            if first:
                raise ValueError("past the dense limit the chain needs the whole document from position 0 (ctx=...)")
            k_i, w_i, g_i = self.index.keys(normed)
            if ctx:
                k_i, w_i, g_i = torch.cat([ixc[0], k_i]), torch.cat([ixc[1], w_i]), torch.cat([ixc[2], g_i])
            sel = (self.index.pools(k_i, g_i), w_i)
        T = N - ctx
        outs = []
        prev = None
        own: list[torch.Tensor] = []                  # the anchors' chain keys of steps 2.. (indexed by anchor)
        cols = torch.arange(N, device=dev)
        for k in range(1, steps + 1):
            n = T - (k - 1)
            if n <= 0:
                break
            anchor = torch.arange(ctx, ctx + n, device=dev)
            if k == 1:
                xk, qrk, qak = x, qr, qa
            else:
                xk = self.inputs(prev[:n], tokens[anchor + k], None)
                _, qrk, qak, latk = self.project(xk)
                own = [o[:n] for o in own] + [latk]
            if sel is not None:
                pools, wts = sel
                idx, ok = self.index.select(qrk, wts[anchor], pools, anchor)
                allow = (idx, ok & (idx <= anchor[:, None]))
            else:
                allow = cols[None, :] <= anchor[:, None]
            u = self.attend(qak, keys, allow, own if k > 1 else [])
            xk = (xk.float() + self.expand(u).to(BF).float()).to(BF)
            xk = (xk.float() + self.moe(xk).to(BF).float()).to(BF)
            out, lg = self.logits(xk)
            outs.append(lg if reduce is None else reduce(k, lg))
            del lg
            prev = out
        return outs

    def step(self, hidden: torch.Tensor, tokens: torch.Tensor, keys: torch.Tensor, pos0: int,
             zero_first: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The engine's ``mtp_forward`` on n rows appended to a cache ``keys`` [pos0, L] (causal among the new
        rows): -> (logits [n, V / world], the rows' shared_head.norm outputs, the cache with the rows' keys)."""

        n = hidden.shape[0]
        dev = hidden.device
        zf = torch.zeros(n, dtype=torch.bool, device=dev)
        if zero_first:
            zf[0] = True
        x = self.inputs(hidden, tokens, zf)
        _, _, qa, lat = self.project(x)
        cache = torch.cat([keys, lat]) if keys is not None and keys.numel() else lat
        cols = torch.arange(cache.shape[0], device=dev)
        allow = cols[None, :] <= (pos0 + torch.arange(n, device=dev))[:, None]
        u = self.attend(qa, cache, allow, [])
        x = (x.float() + self.expand(u).to(BF).float()).to(BF)
        x = (x.float() + self.moe(x).to(BF).float()).to(BF)
        out, lg = self.logits(x)
        return lg, out, cache




# -- loading the MTP head from a checkpoint --------------------------------------------------------------------------
MTP_NAMES = {   # reference name -> checkpoint name under layers.<L>.
    "enorm": "enorm.weight", "hnorm": "hnorm.weight", "eh": "eh_proj.weight", "norm": "shared_head.norm.weight",
    "in_norm": "input_layernorm.weight", "post_norm": "post_attention_layernorm.weight",
    "q_a": "self_attn.q_a_proj.weight", "kv_a": "self_attn.kv_a_proj_with_mqa.weight",
    "q_norm": "self_attn.q_a_layernorm.weight", "kv_norm": "self_attn.kv_a_layernorm.weight",
    "q_b": "self_attn.q_b_proj.weight", "kv_b": "self_attn.kv_b_proj.weight", "o": "self_attn.o_proj.weight",
    "ix_wk": "self_attn.indexer.wk.weight", "ix_wproj": "self_attn.indexer.weights_proj.weight",
    "ix_wq_b": "self_attn.indexer.wq_b.weight", "ix_ln_w": "self_attn.indexer.k_norm.weight",
    "ix_ln_b": "self_attn.indexer.k_norm.bias", "ix_gate": "self_attn.indexer.index_kpool_compress_gate",
    "ix_ape": "self_attn.indexer.index_kpool_compress_ape",
    "router": "mlp.gate.weight", "router_bias": "mlp.gate.e_score_correction_bias",
    "s_gate": "mlp.shared_experts.gate_proj.weight", "s_up": "mlp.shared_experts.up_proj.weight",
    "s_down": "mlp.shared_experts.down_proj.weight",
}
# reference trainable group -> the reference names it exports
EXPORT_GROUPS = {"eh": ["eh"], "enorm": ["enorm"], "hnorm": ["hnorm"], "in_norm": ["in_norm"],
                 "post_norm": ["post_norm"], "norm": ["norm"], "q_a_kv_a": ["q_a", "kv_a"], "q_norm": ["q_norm"],
                 "kv_norm": ["kv_norm"], "q_b": ["q_b"], "kv_b": ["kv_b"], "o": ["o"], "router": ["router"],
                 "shared": ["s_gate", "s_up", "s_down"]}


def mtp_prefix(cfg: Cfg) -> str:
    return PREFIX + f"layers.{cfg.layers}."


def load_mtp_tensors(ck: Checkpoint, cfg: Cfg, override: str | Path | None = None) -> dict[str, torch.Tensor]:
    """The MTP layer's non-expert tensors (full), from the checkpoint or, per tensor, from an override folder
    (GLM53_TF_MTP_WEIGHTS's format: an earlier export)."""

    ov = Checkpoint(override) if override else None
    pre = mtp_prefix(cfg)
    out = {}
    for ref, name in MTP_NAMES.items():
        full = pre + name
        src = ov if ov is not None and full in ov else ck
        if full in src:
            out[ref] = src.get(full)
    return out


def load_experts(ck: Checkpoint, cfg: Cfg, device="cpu", dtype=BF, cache: str | Path | None = None,
                 layer: int | None = None) -> dict[str, torch.Tensor]:
    """Routed experts of the MTP layer as dense W (y = x @ W): gate / up [E, D, NI], down [E, NI, D]. EXL3
    checkpoints go through the engine's decoder (a few minutes on a GPU; ``cache``: a .pt written once)."""

    if cache and Path(cache).exists():
        got = torch.load(cache, map_location=device)
        return {k: v.to(dtype) for k, v in got.items()}
    raw = ck.config()
    L = cfg.layers if layer is None else layer
    pre = PREFIX + f"layers.{L}.mlp.experts."
    out = {}
    for proj, key in (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")):
        mats = []
        for e in range(cfg.experts):
            n = pre + f"{e}.{proj}."
            if is_exl3(raw):
                mats.append(exl3_weight(ck.get(n + "trellis"), ck.get(n + "suh"), ck.get(n + "svh"), dtype, device))
            else:
                mats.append(ck.get(n + "weight").to(device, dtype).t().contiguous())
        out[key] = torch.stack(mats)
    if cache:
        torch.save({k: v.cpu() for k, v in out.items()}, cache)
    return out


def load_head(ck: Checkpoint, mode: str, device="cpu") -> torch.Tensor:
    """The target's head as the engine multiplies by it (bf16 values of the q4 / q4mse copy, or the BF16 head)."""

    w = ck.get("lm_head.weight").to(device)
    if mode == "bf16":
        return w.to(BF)
    out = torch.empty_like(w, dtype=BF)
    for a in range(0, w.shape[0], 16384):
        out[a:a + 16384] = dequant4(*quantize4(w[a:a + 16384], mse=mode == "q4mse")).to(BF)
    return out


def build_mtp(model_dir: str | Path, *, device="cuda", mode: str = "q4mse", kv_fp8: bool = True,
              trainable: Iterable[str] = (), override: str | Path | None = None,
              expert_cache: str | Path | None = None, expert_dtype=BF,
              shard: tuple[int, int] | None = None) -> MTP:
    ck = Checkpoint(model_dir)
    raw = ck.config()
    cfg = Cfg.read(raw)
    t = load_mtp_tensors(ck, cfg, override)
    experts = load_experts(ck, cfg, device=device, dtype=expert_dtype, cache=expert_cache)
    embed = ck.get(PREFIX + "embed_tokens.weight")
    if embed.dtype != BF:
        raise ValueError("the reference reads a BF16 embedding (EXL3 checkpoints)")
    head = load_head(ck, mode, device)
    m = MTP(cfg, {k: v.to(device) for k, v in t.items()}, experts, embed.to(device), head, mode=mode, kv_fp8=kv_fp8,
            trainable=trainable, expert_dtype=expert_dtype, shard=shard)
    return m.to(device)


def export_mtp(m: MTP, out_dir: str | Path, groups: Iterable[str], meta: dict | None = None) -> list[str]:
    """The trained groups as BF16 tensors under the checkpoint's names, for GLM53_TF_MTP_WEIGHTS (the engine
    quantizes them at load exactly as it quantizes the checkpoint's own). -> names written."""

    if m.shard is not None:
        raise ValueError("export a full (unsharded) head")
    c = m.cfg
    pre = mtp_prefix(c)
    H, qk, vd = c.heads, c.qk_dim, c.v_dim
    vals: dict[str, torch.Tensor] = {}
    for g in groups:
        for ref in EXPORT_GROUPS[g]:
            if ref in ("enorm", "hnorm", "in_norm", "post_norm", "norm", "q_norm", "kv_norm"):
                w = getattr(m, ref)
            elif ref == "eh":
                w = m.eh.weight
            elif ref in ("q_a", "kv_a"):
                w = m.q_a_kv_a.weight[:c.q_lora] if ref == "q_a" else m.q_a_kv_a.weight[c.q_lora:]
            elif ref == "q_b":
                w = m.q_b.weight
            elif ref == "kv_b":
                w = torch.cat([m.kv_k.weight.view(H, qk, -1), m.kv_v.weight.view(H, vd, -1)], dim=1).reshape(
                    H * (qk + vd), -1)
            elif ref == "o":
                w = m.o.weight
            elif ref == "router":
                w = m.router
            elif ref == "s_gate":
                w = m.s_gu.weight[:c.shared_width]
            elif ref == "s_up":
                w = m.s_gu.weight[c.shared_width:]
            elif ref == "s_down":
                w = m.s_down.weight
            vals[pre + MTP_NAMES[ref]] = w.detach().to(BF).cpu()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_safetensors(out / "mtp.safetensors", vals, {"format": "pt"})
    (out / "mtp.json").write_text(json.dumps(dict(meta or {}, tensors=sorted(vals)), indent=1))
    return sorted(vals)


# -- the DFlash2 block drafter -----------------------------------------------------------------------------------------
def _rope(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    """dflash2._prep_kernel's rotary step on normed bf16 heads [..., rows, hd] (half-split pairs)."""

    hd = x.shape[-1]
    inv = 1.0 / theta ** (torch.arange(hd // 2, device=x.device, dtype=torch.float32) * 2 / hd)
    ph = pos.float()[:, None] * inv[None, :]
    cos, sin = ph.cos(), ph.sin()
    a, b = x[..., :hd // 2].float(), x[..., hd // 2:].float()
    return torch.cat([(a * cos - b * sin).to(BF), (b * cos + a * sin).to(BF)], dim=-1)


def _head_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * r * w.float()).to(BF)


def dconv(x: torch.Tensor, dyn: torch.Tensor, base: torch.Tensor, branch: int, gs: int, block: int,
          residual: torch.Tensor | None = None) -> torch.Tensor:
    """dflash2._dconv_kernel: two-tap grouped dynamic convolution inside each block of ``block`` rows (row r mixes
    rows r and r - 1 of its block), kernels bf16(base + dyn), then the residual. x [R, D] bf16, dyn [R, 2*2*G]."""

    R, D = x.shape
    G = D // gs
    d = dyn.float().view(R, 2, 2, G)[:, branch]                   # [R, tap, G]
    k0 = (base[branch, 0].float()[None] + d[:, 0].repeat_interleave(gs, dim=-1)).to(BF).float()
    k1 = (base[branch, 1].float()[None] + d[:, 1].repeat_interleave(gs, dim=-1)).to(BF).float()
    xf = x.float().view(R // block, block, D)
    prev = torch.cat([torch.zeros_like(xf[:, :1]), xf[:, :-1]], dim=1).view(R, D)
    y = (x.float() * k0 + prev * k1).to(BF)
    if residual is not None:
        y = (residual.float() + y.float()).to(BF)
    return y


class DFlash(nn.Module):
    """The DFlash2 drafter (``cuda/dflash2.py``): context = hidden_norm(fc(taps)) per committed position; a block of
    [pending, mask x (B - 1)] at positions p .. p + B - 1 through Qwen3-style layers whose attention and MLP inputs
    and outputs pass the grouped dynamic convolutions; the block attends to the context keys within the sliding
    window (all of them when the config has none) and to its own rows (bidirectional unless ``is_causal``); rows
    1.. give candidates for p + 1, ... through the target's head, and the selector's projection for the chain.
    Handles incoai's layout (5 layers, 5 taps, window 2,048) and canada-quant's DFlash2-G (8 layers, 9 taps, full
    attention, a learnt mask embedding and its own embed / head), which the engine does not load yet."""

    def __init__(self, cfg: dict, t: dict[str, torch.Tensor], embed: torch.Tensor, head: torch.Tensor, *,
                 mode: str = "q4", trainable: bool = False, mask_embedding: torch.Tensor | None = None,
                 shard: tuple[int, int] | None = None) -> None:
        super().__init__()
        dc = cfg["dflash_config"]
        self.cfg = cfg
        self.D = int(cfg["hidden_size"])
        self.hd = int(cfg["head_dim"])
        self.H, self.KV = int(cfg["num_attention_heads"]), int(cfg["num_key_value_heads"])
        # ``shard`` (rank, world): rank's heads, MLP width and vocabulary only (the engine's split, its partials)
        r, world = shard if shard is not None else (0, 1)
        self.shard = shard
        hd = self.hd
        self.H, self.KV = self.H // world, self.KV // world
        inter = int(cfg["intermediate_size"]) // world
        qs, ks = slice(r * self.H * hd, (r + 1) * self.H * hd), slice(r * self.KV * hd, (r + 1) * self.KV * hd)
        ms = slice(r * inter, (r + 1) * inter)
        self.eps = float(cfg["rms_norm_eps"])
        self.theta = float(cfg["rope_parameters"]["rope_theta"])
        self.block = int(dc["block_size"])
        self.gs = int(dc["conv_group_size"])
        self.mask_id = int(dc["mask_token_id"])
        self.taps = tuple(int(i) for i in dc["target_layer_ids"])
        sw = cfg.get("sliding_window")
        self.window = int(sw) - 1 if sw else None
        self.causal = bool(cfg.get("is_causal", True))
        self.top_k = int(dc.get("selector_top_k", 16))
        self.mode = mode
        P = lambda x: nn.Parameter(x.to(BF).clone(), requires_grad=trainable)       # noqa: E731
        self.fc = Quant(t["fc.weight"], mode, trainable)
        self.hidden_norm, self.norm = P(t["hidden_norm.weight"]), P(t["norm.weight"])
        self.hproj = P(t["candidate_selector.hidden_projection.weight"])
        self.register_buffer("pred", t["candidate_selector.predecessor_codebook"].float())
        self.register_buffer("succ", t["candidate_selector.successor_codebook"].float())
        self.layers = nn.ModuleList()
        for i in range(int(cfg["num_hidden_layers"])):
            p = f"layers.{i}."
            L = nn.Module()
            L.in_norm, L.post_norm = P(t[p + "input_layernorm.weight"]), P(t[p + "post_attention_layernorm.weight"])
            L.a_base, L.m_base = P(t[p + "attention_conv.base_kernel"]), P(t[p + "mlp_conv.base_kernel"])
            L.a_kp = Quant(t[p + "attention_conv.kernel_projection.weight"], mode, trainable)
            L.m_kp = Quant(t[p + "mlp_conv.kernel_projection.weight"], mode, trainable)
            q, k, v = (t[p + f"self_attn.{x}_proj.weight"] for x in "qkv")
            L.q, L.k, L.v = Quant(q[qs], mode, trainable), Quant(k[ks], mode, trainable), Quant(v[ks], mode, trainable)
            L.q_norm, L.k_norm = P(t[p + "self_attn.q_norm.weight"]), P(t[p + "self_attn.k_norm.weight"])
            L.o = Quant(t[p + "self_attn.o_proj.weight"][:, qs], mode, trainable)
            L.gu = Quant(torch.cat([t[p + "mlp.gate_proj.weight"][ms], t[p + "mlp.up_proj.weight"][ms]]), mode,
                         trainable)
            L.down = Quant(t[p + "mlp.down_proj.weight"][:, ms], mode, trainable)
            self.layers.append(L)
        self.register_buffer("embed", embed.to(BF), persistent=False)
        V = head.shape[0]
        self.register_buffer("head", head[r * (V // world):(r + 1) * (V // world)].to(BF).contiguous(),
                             persistent=False)
        self.mask_emb = None if mask_embedding is None else nn.Parameter(mask_embedding.to(BF).reshape(-1).clone(),
                                                                          requires_grad=trainable)

    def context(self, taps: torch.Tensor) -> torch.Tensor:
        """Committed positions' taps [N, T * D] -> the context rows hidden_norm(fc(taps)) [N, D] bf16."""

        return rmsnorm(self.fc(taps.to(BF)), self.hidden_norm, self.eps)

    def ctx_kv(self, L, ctx: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = L.k(ctx).view(-1, self.KV, self.hd).transpose(0, 1)                       # [KV, N, hd]
        v = L.v(ctx).view(-1, self.KV, self.hd).transpose(0, 1)
        k = _rope(_head_norm(k, L.k_norm, self.eps), pos, self.theta)
        return k, v

    def blocks(self, ctx: torch.Tensor, anchors: torch.Tensor, pending: torch.Tensor, first: int = 0):
        """Blocks at anchors (local positions a: the pending token t_a at a, context rows 0..a-1) over one
        document's context rows ``ctx`` [N, D]. -> (normed rows h [A, B - 1, D] bf16, logits [A, B - 1, V] bf16)."""

        A, B, D = anchors.shape[0], self.block, self.D
        dev = ctx.device
        ids = torch.full((A, B), self.mask_id, dtype=torch.long, device=dev)
        ids[:, 0] = pending.long()
        x = self.embed[ids.view(-1)].view(A, B, D)
        if self.mask_emb is not None:
            x = torch.cat([x[:, :1], self.mask_emb.to(BF)[None, None].expand(A, B - 1, D)], dim=1)
        x = x.reshape(A * B, D)
        qpos = (anchors[:, None] + torch.arange(B, device=dev)[None]).reshape(-1)       # block rows' positions
        cpos = torch.arange(ctx.shape[0], device=dev)
        # masks: context key j visible to a block row at q (anchor a) when j < a and q - j <= window
        a_rows = anchors.repeat_interleave(B)
        m_ctx = cpos[None, :] < a_rows[:, None]
        if self.window is not None:
            m_ctx &= (qpos[:, None] - cpos[None, :]) <= self.window
        blk = torch.arange(A, device=dev).repeat_interleave(B)
        m_blk = blk[:, None] == blk[None, :]
        if self.causal:
            m_blk &= qpos[None, :] <= qpos[:, None]
        mask = torch.cat([m_ctx, m_blk], dim=1)
        for L in self.layers:
            ck, cv = self.ctx_kv(L, ctx, cpos + first)
            normed = rmsnorm(x, L.in_norm, self.eps)
            dyn = L.a_kp(normed)
            h = dconv(normed, dyn, L.a_base, 0, self.gs, B)
            q = L.q(h).view(-1, self.H, self.hd).transpose(0, 1)
            k = L.k(h).view(-1, self.KV, self.hd).transpose(0, 1)
            v = L.v(h).view(-1, self.KV, self.hd).transpose(0, 1)
            q = _rope(_head_norm(q, L.q_norm, self.eps), qpos + first, self.theta)
            k = _rope(_head_norm(k, L.k_norm, self.eps), qpos + first, self.theta)
            K = torch.cat([ck, k], dim=1).repeat_interleave(self.H // self.KV, dim=0)
            V = torch.cat([cv, v], dim=1).repeat_interleave(self.H // self.KV, dim=0)
            o = F.scaled_dot_product_attention(q.float()[None], K.float()[None], V.float()[None],
                                               attn_mask=mask[None, None], scale=self.hd ** -0.5)[0]
            o = o.to(BF).transpose(0, 1).reshape(-1, self.H * self.hd)
            x = dconv(L.o(o), dyn, L.a_base, 1, self.gs, B, residual=x)
            normed = rmsnorm(x, L.post_norm, self.eps)
            dyn = L.m_kp(normed)
            gu = L.gu(dconv(normed, dyn, L.m_base, 0, self.gs, B))
            w = gu.shape[1] // 2
            act = ((gu[:, :w].float() * torch.sigmoid(gu[:, :w].float())).to(BF).float() * gu[:, w:].float()).to(BF)
            x = dconv(L.down(act), dyn, L.m_base, 1, self.gs, B, residual=x)
        x = x.view(A, B, D)[:, 1:]
        h = rmsnorm(x.reshape(-1, D), self.norm, self.eps)
        logits = h @ self.head.t()
        return h.view(A, B - 1, D), logits.view(A, B - 1, -1)

    @torch.no_grad()
    def chain(self, h: torch.Tensor, logits: torch.Tensor, anchor_token: int, depth: int, edge: float = 0.6):
        """Greedy chain of one block (dflash2.Drafter.chain without sampling noise): top-k candidates a row, pick by
        logit + edge x selector(prev, cand)."""

        proj = (h.float() @ self.hproj.float().t())
        vals, cand = torch.topk(logits.float(), self.top_k, dim=-1)
        out, prev = [], int(anchor_token)
        for d in range(min(depth, logits.shape[0])):
            e = self.succ[cand[d]] @ (self.pred[prev] * proj[d])
            j = int(torch.argmax(vals[d] + edge * e))
            prev = int(cand[d, j])
            out.append(prev)
        return out


def load_dflash(draft_dir: str | Path, target_dir: str | Path, *, device="cuda", mode: str = "q4",
                trainable: bool = False, head_mode: str = "q4mse", shard: tuple[int, int] | None = None) -> DFlash:
    """A DFlash2 checkpoint folder (config.json + model.safetensors [+ mask_embedding.pt]) with the target's embedding
    and head (the engine drafts with the target's head: ``w.draft_head`` or ``w.head``)."""

    d = Path(draft_dir)
    cfg = json.loads((d / "config.json").read_text())
    ck = Checkpoint(d)
    t = {k: ck.get(k) for k in ck.names()}
    tgt = Checkpoint(target_dir)
    embed = tgt.get(PREFIX + "embed_tokens.weight").to(BF)
    head = load_head(tgt, head_mode, device)
    mask = None
    if (d / "mask_embedding.pt").exists():
        got = torch.load(d / "mask_embedding.pt", map_location="cpu")
        mask = got if isinstance(got, torch.Tensor) else next(iter(got.values()))
    return DFlash(cfg, {k: v.to(device) for k, v in t.items()}, embed.to(device), head, mode=mode,
                  trainable=trainable, mask_embedding=mask, shard=shard).to(device)


def export_dflash(m: DFlash, src_dir: str | Path, out_dir: str | Path, meta: dict | None = None) -> None:
    """The drafter in its checkpoint's own layout (config.json copied, BF16 model.safetensors): what DRAFTER= loads
    (the engine quantizes it to 4 bits at load)."""

    if m.shard is not None:
        raise ValueError("export a full (unsharded) drafter")
    src, out = Path(src_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck = Checkpoint(src)
    t = {k: ck.get(k) for k in ck.names()}
    t["fc.weight"] = m.fc.weight.detach().cpu()
    t["hidden_norm.weight"], t["norm.weight"] = m.hidden_norm.detach().cpu(), m.norm.detach().cpu()
    t["candidate_selector.hidden_projection.weight"] = m.hproj.detach().cpu()
    for i, L in enumerate(m.layers):
        p = f"layers.{i}."
        t[p + "input_layernorm.weight"], t[p + "post_attention_layernorm.weight"] = L.in_norm.detach().cpu(), \
            L.post_norm.detach().cpu()
        t[p + "attention_conv.base_kernel"], t[p + "mlp_conv.base_kernel"] = L.a_base.detach().cpu(), \
            L.m_base.detach().cpu()
        t[p + "attention_conv.kernel_projection.weight"] = L.a_kp.weight.detach().cpu()
        t[p + "mlp_conv.kernel_projection.weight"] = L.m_kp.weight.detach().cpu()
        for x in "qkvo":
            t[p + f"self_attn.{x}_proj.weight"] = getattr(L, x).weight.detach().cpu()
        t[p + "self_attn.q_norm.weight"], t[p + "self_attn.k_norm.weight"] = L.q_norm.detach().cpu(), \
            L.k_norm.detach().cpu()
        w = L.gu.weight.shape[0] // 2
        t[p + "mlp.gate_proj.weight"], t[p + "mlp.up_proj.weight"] = L.gu.weight[:w].detach().cpu(), \
            L.gu.weight[w:].detach().cpu()
        t[p + "mlp.down_proj.weight"] = L.down.weight.detach().cpu()
    write_safetensors(out / "model.safetensors", {k: v.to(BF) if v.is_floating_point() else v for k, v in t.items()},
                      {"format": "pt"})
    (out / "config.json").write_text((src / "config.json").read_text())
    if m.mask_emb is not None:
        torch.save(m.mask_emb.detach().cpu(), out / "mask_embedding.pt")
    (out / "TRAINING.json").write_text(json.dumps(meta or {}, indent=1))


def master_fp32(m: nn.Module) -> list[nn.Parameter]:
    """Trainable parameters kept in fp32 (AdamW on bf16 weights loses the small updates); every use casts them the
    way the engine stores them (bf16, or the 4-bit copy through ``fake_quant``). -> the trainable parameters."""

    out = []
    for p in m.parameters():
        if p.requires_grad:
            p.data = p.data.float()
            out.append(p)
    return out
