"""CPU tests of the drafter reference (train/glmref.py) and patches/0430's dump pieces that need no GPU:

- ``quantize4`` / ``_clip_search`` are the engine's own (``qmm.quantize4``'s source, run here without triton);
- ``exl3_weight`` (torch, any device) equals the engine's float64 reference decoder (``exl3.dequantize``);
- ``fp8_latent`` equals patches/0220's ``latent.quantize_rows_reference`` read back;
- the MTP training-time chain (all anchors and steps at once) equals drafting step by step through ``MTP.step``
  (the engine's order: absorb the kept rows, then one chained step at a time), dense and with DSA selection;
- DFlash2 blocks computed together equal blocks computed one at a time (anchors never see each other);
- the MTP export reads back under the checkpoint's names (what GLM53_TF_MTP_WEIGHTS loads).

    TF_SRC=<patched tree>/src python -m pytest -q tests/test_drafter_ref.py
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train"))
import glmref as R  # noqa: E402

# a PATCHED source tree (the image's /src/TensorFold/src, or a checkout with patches/ applied): the engine functions
# compared against come from patches (0001's q4mse, 0220's FP8 rows)
TF_SRC = Path(os.environ.get("TF_SRC", "/src/TensorFold/src"))
CUDA_DIR = TF_SRC / "tensorfold/families/glm5_next/cuda"


def _engine_funcs(path: Path, names: tuple[str, ...], ns: dict) -> dict:
    """Functions of an engine module by source (the module itself imports triton)."""

    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    mod = ast.Module(body=body, type_ignores=[])
    exec(compile(mod, str(path), "exec"), ns)
    return ns


@pytest.mark.parametrize("mse", [False, True])
def test_quantize4_is_the_engines(mse):
    if not (CUDA_DIR / "qmm.py").exists() or "mse" not in (CUDA_DIR / "qmm.py").read_text():
        pytest.skip("TF_SRC: a patched TensorFold source tree")
    got = {}
    ns = _engine_funcs(CUDA_DIR / "qmm.py", ("quantize4", "_clip_search"),
                       {"torch": torch, "make_q4": lambda w, s, b: got.update(w=w, s=s, b=b), "Q4": object})
    g = torch.Generator().manual_seed(0)
    w = (torch.randn(96, 256, generator=g) * 0.05).to(torch.bfloat16)
    w[3, :64] = 0.0                                          # a flat group
    ns["quantize4"](w, mse=mse)
    codes, s, b = R.quantize4(w, mse=mse)
    words = got["w"].to(torch.int64) & 0xFFFFFFFF
    unpacked = ((words[..., None] >> (torch.arange(8) * 4)) & 0xF).reshape(96, 256)
    assert torch.equal(unpacked.to(torch.uint8), codes)
    assert torch.equal(got["s"], s) and torch.equal(got["b"], b)


def test_exl3_weight_matches_reference_decoder():
    if not R.EXL3_PY.exists():
        pytest.skip("vendor/TensorFold not checked out")
    ex = R.exl3_module()
    rng = np.random.default_rng(1)
    K, N = 256, 128
    trellis = torch.from_numpy(rng.integers(-2**15, 2**15, size=(K // 16, N // 16, 64)).astype(np.int16))
    suh = torch.from_numpy((rng.standard_normal(K) * 0.03).astype(np.float16))
    svh = torch.from_numpy((rng.standard_normal(N) * 0.03).astype(np.float16))
    ref = ex.dequantize(trellis, suh, svh)
    got = R.exl3_weight(trellis, suh, svh, dtype=torch.float64, device="cpu")
    assert torch.allclose(got, ref, rtol=0, atol=1e-12)


def test_fp8_latent_matches_patch_0220():
    if not (CUDA_DIR / "latent.py").exists():
        pytest.skip("TF_SRC: a patched TensorFold source tree")
    ns = _engine_funcs(CUDA_DIR / "latent.py", ("quantize_rows_reference",),
                       {"torch": torch, "ROW8": 528, "SCALE_AT": 512})
    g = torch.Generator().manual_seed(2)
    x = (torch.randn(33, 512, generator=g) * torch.logspace(-3, 1, 33)[:, None]).to(torch.bfloat16)
    x[5] = 0
    rows = ns["quantize_rows_reference"](x)
    vals = rows[:, :512].contiguous().view(torch.float8_e4m3fn).float()
    scale = rows[:, 512:516].contiguous().view(torch.float32)
    assert torch.equal((vals * scale).to(torch.bfloat16), R.fp8_latent(x))


# -- a tiny MTP head --------------------------------------------------------------------------------------------------
TINY = {"text_config": {
    "hidden_size": 128, "num_hidden_layers": 2, "vocab_size": 96, "rms_norm_eps": 1e-5, "num_attention_heads": 2,
    "q_lora_rank": 64, "kv_lora_rank": 64, "qk_nope_head_dim": 64, "qk_rope_head_dim": 0, "v_head_dim": 64,
    "n_routed_experts": 6, "num_experts_per_tok": 2, "moe_intermediate_size": 64, "n_shared_experts": 1,
    "routed_scaling_factor": 2.5, "norm_topk_prob": True, "swiglu_limit": 10.0, "index_n_heads": 2,
    "index_head_dim": 128, "index_topk": 16, "index_kpool": 4, "eos_token_id": [95], "intermediate_size": 64}}


def _tiny_mtp(seed=0, mode="q4mse"):
    cfg = R.Cfg.read(TINY)
    g = torch.Generator().manual_seed(seed)
    D, H, V = cfg.hidden, cfg.heads, cfg.vocab

    def rn(*shape, s=0.05, o=0.0):
        return (torch.randn(*shape, generator=g) * s + o).to(torch.bfloat16)

    t = {"enorm": rn(D, o=1), "hnorm": rn(D, o=1), "eh": rn(D, 2 * D, s=0.08), "norm": rn(D, o=1),
         "in_norm": rn(D, o=1), "post_norm": rn(D, o=1), "q_a": rn(cfg.q_lora, D, s=0.08),
         "kv_a": rn(cfg.kv_lora, D, s=0.08), "q_norm": rn(cfg.q_lora, o=1), "kv_norm": rn(cfg.kv_lora, o=1),
         "q_b": rn(H * cfg.qk_dim, cfg.q_lora, s=0.1), "kv_b": rn(H * (cfg.qk_dim + cfg.v_dim), cfg.kv_lora, s=0.1),
         "o": rn(D, H * cfg.v_dim, s=0.08), "ix_wk": rn(128, D, s=0.08), "ix_wproj": rn(2, D, s=0.3),
         "ix_wq_b": rn(2 * 128, cfg.q_lora, s=0.1), "ix_ln_w": rn(128, o=1), "ix_ln_b": rn(128),
         "ix_gate": rn(128, D), "ix_ape": rn(4, 128), "router": rn(cfg.experts, D, s=0.2),
         "router_bias": torch.randn(cfg.experts, generator=g) * 0.01,
         "s_gate": rn(cfg.moe_width, D, s=0.08), "s_up": rn(cfg.moe_width, D, s=0.08),
         "s_down": rn(D, cfg.moe_width, s=0.08)}
    ex = {"gate": rn(cfg.experts, D, cfg.moe_width, s=0.08), "up": rn(cfg.experts, D, cfg.moe_width, s=0.08),
          "down": rn(cfg.experts, cfg.moe_width, D, s=0.08)}
    return R.MTP(cfg, t, ex, rn(V, D, s=0.5), rn(V, D, s=0.08), mode=mode, kv_fp8=True, trainable={"eh", "o"})


def _sequential(m, hidden, tokens, steps):
    """Drafting as the engine does it: the cache holds entries 0..a (every true row), then chained steps append the
    chain's own entries; teacher-forced tokens."""

    N = hidden.shape[0]
    with torch.no_grad():
        lg1, out1, keys = m.step(hidden, tokens[1:N + 1], None, 0, zero_first=True)
        res = [[lg1[a] for a in range(N)]] + [[] for _ in range(steps - 1)]
        for a in range(N):
            cache, prev = keys[:a + 1], out1[a:a + 1]
            for k in range(2, steps + 1):
                e = a + k - 1
                if e >= N:
                    break
                lg, prev, cache = m.step(prev, tokens[e + 1:e + 2], cache, e)
                res[k - 1].append(lg[0])
    return res


def test_mtp_chain_equals_step_by_step_drafting():
    m = _tiny_mtp()
    g = torch.Generator().manual_seed(5)
    N = 12
    hidden = (torch.randn(N, 128, generator=g)).to(torch.bfloat16)
    tokens = torch.randint(0, 96, (N + 1,), generator=g)
    with torch.no_grad():
        got = m.chain(hidden, tokens, steps=3, sparse=False)
    want = _sequential(m, hidden, tokens, 3)
    for k in range(3):
        assert got[k].shape[0] == len(want[k]) == N - k
        w = torch.stack(want[k]).float()
        assert torch.allclose(got[k].float(), w, atol=2e-2, rtol=0), (k, (got[k].float() - w).abs().max())
        assert (got[k].float().argmax(-1) == w.argmax(-1)).float().mean() > 0.9


def test_mtp_chain_context_rows_do_not_change_outputs():
    m = _tiny_mtp(1)
    g = torch.Generator().manual_seed(6)
    N = 14
    hidden = torch.randn(N, 128, generator=g).to(torch.bfloat16)
    tokens = torch.randint(0, 96, (N + 1,), generator=g)
    with torch.no_grad():
        full = m.chain(hidden, tokens, steps=2, sparse=False)
        part = m.chain(hidden, tokens, steps=2, ctx=5, sparse=False)
    assert torch.equal(full[0][5:], part[0])
    assert torch.equal(full[1][5:], part[1])


def test_mtp_chain_sparse_selection_runs_and_is_dense_below_the_limit():
    m = _tiny_mtp(2)
    g = torch.Generator().manual_seed(7)
    N = 40                                          # dense limit 16 + 4 - 1 = 19: rows past it select 4 pools
    hidden = torch.randn(N, 128, generator=g).to(torch.bfloat16)
    tokens = torch.randint(0, 96, (N + 1,), generator=g)
    with torch.no_grad():
        sp = m.chain(hidden, tokens, steps=2, sparse=True)
        de = m.chain(hidden, tokens, steps=2, sparse=False)
    lim = m.cfg.dense_limit
    assert torch.equal(sp[0][:lim], de[0][:lim])
    assert not torch.equal(sp[0][lim + 4:], de[0][lim + 4:])
    idx, ok = m.index.select(torch.zeros(1, 64, dtype=torch.bfloat16), torch.ones(1, 2),
                             torch.zeros(10, 128, dtype=torch.bfloat16), torch.tensor([39]))
    assert int(ok.sum()) == 16                      # 4 pools of 4 (+ an empty tail at q = 39)


def test_mtp_chain_backward_reaches_trainable_weights():
    m = _tiny_mtp(3)
    g = torch.Generator().manual_seed(8)
    hidden = torch.randn(10, 128, generator=g).to(torch.bfloat16)
    tokens = torch.randint(0, 96, (11,), generator=g)
    out = m.chain(hidden, tokens, steps=3, sparse=False)
    loss = sum(o.float().logsumexp(-1).mean() for o in out)
    loss.backward()
    assert m.eh.weight.grad is not None and m.eh.weight.grad.abs().sum() > 0
    assert m.o.weight.grad is not None and m.o.weight.grad.abs().sum() > 0
    assert m.q_b.weight.grad is None


def test_mtp_export_reads_back_under_checkpoint_names(tmp_path):
    m = _tiny_mtp(4)
    names = R.export_mtp(m, tmp_path, ["eh", "o", "kv_b", "shared", "q_a_kv_a"])
    ck = R.Checkpoint(tmp_path)
    pre = R.mtp_prefix(m.cfg)
    assert set(ck.names()) == set(names)
    assert all(n.startswith(pre) for n in names)
    assert torch.equal(ck.get(pre + "eh_proj.weight"), m.eh.weight.detach())
    t = R.load_mtp_tensors(ck, m.cfg)
    m2 = R.MTP(m.cfg, {**{k: v for k, v in _tiny_tensors_of(m).items()}, **t},
               {"gate": m.e_gate, "up": m.e_up, "down": m.e_down}, m.embed, m.head)
    assert torch.equal(m2.kv_k.weight, m.kv_k.weight) and torch.equal(m2.kv_v.weight, m.kv_v.weight)
    assert torch.equal(m2.s_gu.weight, m.s_gu.weight)


def _tiny_tensors_of(m):
    """The reference names of a built head (for rebuilding one with some tensors replaced)."""

    c = m.cfg
    H = c.heads
    return {"enorm": m.enorm, "hnorm": m.hnorm, "eh": m.eh.weight, "norm": m.norm, "in_norm": m.in_norm,
            "post_norm": m.post_norm, "q_a": m.q_a_kv_a.weight[:c.q_lora], "kv_a": m.q_a_kv_a.weight[c.q_lora:],
            "q_norm": m.q_norm, "kv_norm": m.kv_norm, "q_b": m.q_b.weight,
            "kv_b": torch.cat([m.kv_k.weight.view(H, c.qk_dim, -1), m.kv_v.weight.view(H, c.v_dim, -1)], 1).reshape(
                H * (c.qk_dim + c.v_dim), -1), "o": m.o.weight, "router": m.router, "router_bias": m.bias,
            "s_gate": m.s_gu.weight[:c.shared_width], "s_up": m.s_gu.weight[c.shared_width:],
            "s_down": m.s_down.weight}


def test_mtp_shard_halves_add_up():
    """Rank 0's and rank 1's partial branches (the engine's row-parallel outputs) add up to the whole layer's."""

    m = _tiny_mtp(5, mode="bf16")
    t = _tiny_tensors_of(m)
    ex = {"gate": m.e_gate, "up": m.e_up, "down": m.e_down}
    halves = [R.MTP(m.cfg, t, ex, m.embed, m.head, mode="bf16", shard=(r, 2)) for r in (0, 1)]
    x = torch.randn(7, 128, generator=torch.Generator().manual_seed(9)).to(torch.bfloat16)
    whole = m.moe(x)
    parts = sum(h.moe(x) for h in halves)
    assert torch.allclose(whole, parts, atol=1e-4)
    lg = torch.cat([h.logits(x)[1] for h in halves], dim=1)
    assert torch.equal(lg, m.logits(x)[1])


# -- DFlash2 ------------------------------------------------------------------------------------------------------------
def _tiny_dflash(window=6, mask=False):
    D, H, KV, hd, inter, V = 128, 4, 2, 32, 128, 96
    cfg = {"hidden_size": D, "head_dim": hd, "num_attention_heads": H, "num_key_value_heads": KV,
           "rms_norm_eps": 1e-5, "rope_parameters": {"rope_theta": 10000.0}, "sliding_window": window,
           "is_causal": False, "intermediate_size": inter, "num_hidden_layers": 2,
           "dflash_config": {"mask_token_id": 90, "conv_group_size": 16, "conv_kernel_size": 2, "block_size": 4,
                             "selector_rank": 8, "selector_top_k": 4, "target_layer_ids": [0, 1]}}
    g = torch.Generator().manual_seed(11)
    rn = lambda *s, sc=0.05, o=0.0: (torch.randn(*s, generator=g) * sc + o).to(torch.bfloat16)     # noqa: E731
    t = {"fc.weight": rn(D, 2 * D, sc=0.05), "hidden_norm.weight": rn(D, o=1), "norm.weight": rn(D, o=1),
         "candidate_selector.hidden_projection.weight": rn(8, D), "candidate_selector.predecessor_codebook": rn(V, 8),
         "candidate_selector.successor_codebook": rn(V, 8)}
    for i in range(2):
        p = f"layers.{i}."
        t.update({p + "self_attn.q_proj.weight": rn(H * hd, D), p + "self_attn.k_proj.weight": rn(KV * hd, D),
                  p + "self_attn.v_proj.weight": rn(KV * hd, D), p + "self_attn.o_proj.weight": rn(D, H * hd),
                  p + "self_attn.q_norm.weight": rn(hd, o=1), p + "self_attn.k_norm.weight": rn(hd, o=1),
                  p + "mlp.gate_proj.weight": rn(inter, D), p + "mlp.up_proj.weight": rn(inter, D),
                  p + "mlp.down_proj.weight": rn(D, inter), p + "input_layernorm.weight": rn(D, o=1),
                  p + "post_attention_layernorm.weight": rn(D, o=1),
                  p + "attention_conv.base_kernel": rn(2, 2, D, sc=0.1, o=0.5),
                  p + "attention_conv.kernel_projection.weight": rn(4 * D // 16, D, sc=0.02),
                  p + "mlp_conv.base_kernel": rn(2, 2, D, sc=0.1, o=0.5),
                  p + "mlp_conv.kernel_projection.weight": rn(4 * D // 16, D, sc=0.02)})
    return R.DFlash(cfg, t, rn(V, D, sc=0.5), rn(V, D, sc=0.08), mode="q4",
                    mask_embedding=rn(D) if mask else None)


@pytest.mark.parametrize("window,mask", [(6, False), (None, True)])
def test_dflash_blocks_are_independent(window, mask):
    m = _tiny_dflash(window, mask)
    g = torch.Generator().manual_seed(12)
    N = 20
    ctx = m.context(torch.randn(N, 256, generator=g).to(torch.bfloat16))
    anchors = torch.tensor([3, 9, 10, 17])
    pending = torch.randint(0, 96, (4,), generator=g)
    with torch.no_grad():
        h, lg = m.blocks(ctx, anchors, pending)
        for i in range(4):
            h1, lg1 = m.blocks(ctx, anchors[i:i + 1], pending[i:i + 1])
            # the same rows; only the fp32 sums over masked-out keys run in another order (bf16 last-bit noise)
            assert torch.allclose(lg[i].float(), lg1[0].float(), atol=5e-2)
            assert (lg[i].float().argmax(-1) == lg1[0].float().argmax(-1)).float().mean() >= 0.66
        # the sliding window: context rows older than the window do not matter
        if window is not None:
            ctx2 = ctx.clone()
            ctx2[:17 - m.window] = 0                  # older than any block row can see (q - j > window)
            h2, lg2 = m.blocks(ctx2, anchors[3:], pending[3:])
            assert torch.allclose(lg2[0].float(), lg[3].float(), atol=5e-2)
    drafts = m.chain(h[0], lg[0], int(pending[0]), 3)
    assert len(drafts) == 3


# -- the offline acceptance evaluator on a synthetic document -----------------------------------------------------------
class _FakeDoc:
    """A dump document stand-in whose target IS the reference head (so the stock head's greedy acceptance of its
    own first step is 1 where the dump's top-1 is its own argmax)."""

    def __init__(self, n, lp, ids, hidden, taps, tokens):
        self.n, self._lp, self._ids, self._h, self._t, self._tok = n, lp, ids, hidden, taps, tokens

    def tokens(self):
        return self._tok.numpy().astype(np.int32)

    def kinds(self):
        return np.r_[np.zeros(self.n // 2, np.uint8), np.ones(self.n - self.n // 2, np.uint8)]

    def get(self, name, a, b):
        return {"lp": self._lp, "ids": self._ids, "hidden": self._h}[name][a:b]

    def taps(self, a, b):
        return self._t[a:b]


def test_eval_mtp_counts_a_self_consistent_first_step():
    import eval_accept as E

    m = _tiny_mtp(6)
    g = torch.Generator().manual_seed(13)
    n = 30
    hidden = torch.randn(n, 128, generator=g).to(torch.bfloat16)
    tokens = torch.randint(0, 96, (n,), generator=g)
    with torch.no_grad():
        step1 = m.chain(hidden, torch.cat([tokens, tokens[:1]]), steps=1, sparse=False)[0]
    # the dump's row r describes position r + 1; make row j + 1 the head's own step-1 prediction from row j
    lq = torch.log_softmax(step1.float(), -1)
    lp, ids = torch.topk(lq, 32, dim=-1)
    lp = torch.cat([lp[:1], lp[:-1]]).to(torch.float16)
    ids = torch.cat([ids[:1], ids[:-1]]).to(torch.int32)
    doc = _FakeDoc(n, lp, ids, hidden, None, tokens)
    rep = E.eval_mtp(m, [doc], steps=2, window=64, ctx=0, samples=16)      # one span from 0: the dump's view
    assert rep["all"]["greedy"]["a"][0] >= 0.95        # (CPU matmul blocking differs with the row count: near-ties)
    assert rep["all"]["T=1.0"]["a"][0] > 0.7          # the same distribution (up to the top-32 cut), coupled
    assert 0 <= rep["all"]["greedy"]["a"][1] <= 1


def test_eval_dflash_runs():
    import eval_accept as E

    m = _tiny_dflash(6)
    g = torch.Generator().manual_seed(14)
    n = 40
    ids = torch.randint(0, 96, (n, 8), generator=g).to(torch.int32)
    doc = _FakeDoc(n, torch.zeros(n, 8, dtype=torch.float16), ids, None,
                   torch.randn(n, 256, generator=g).to(torch.bfloat16), torch.randint(0, 96, (n,), generator=g))
    rep = E.eval_dflash(m, [doc], stride=4, batch=4)
    assert len(rep["all"]["greedy"]["a"]) == m.block - 1
