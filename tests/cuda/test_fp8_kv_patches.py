"""patches/0220 (GLM53_TF_KV_DTYPE=fp8, load-time): the latent caches of the DSA layers and the MTP head hold each
token's 512-wide latent as e4m3 values with one power-of-two fp32 scale a row (528 B a row instead of 1,024 B).

``latent_write`` quantizes every row on its own; the latent attention kernels (dense ``_lchunks``, sparse
``_lsparse_chunks``, 16- and 32-query tiles) dequantize the rows they read (e4m3 x 2^k, exact in bf16) and then run
the bf16 arithmetic unchanged. Checked here (TensorFold's synthetic checkpoint, one GPU playing rank 0 of two):

- kernels on the real shapes (32 local heads, head dim 256, latent 512): the writer's bytes == torch's float8_e4m3fn
  conversion (``latent.quantize_rows_reference``) on every finite bf16 value <= 448 and on random rows over 11
  decades, a row written alone == the same row in a block, the error bound (half an e4m3 step); FP8 attention (dense
  and sparse, BM 16 and 32) == the bf16 kernels over the dequantized rows, bit for bit; absorb -> attend -> expand
  against the expanded fp32 reference within a few % (FP8 KV against bf16 KV: the quality hook); a window row ==
  the serial row at its position (dense and sparse), a graph-style call == exact chunk counts;
- the engine with fp8 KV: cache layout and bytes a token; drafted == serial for every policy; resumed == fresh;
  64- and 512-row prefill chunks leave the same state bit for bit; past 2,051 tokens (sparse top-k over the FP8
  latent) 64 / 512 rows, serial / drafted, with and without the 0050 graphs, resumed == fresh; batched rounds
  (0120, 2 and 4 slots with 0200's on-set) == each request alone; the session store (0110) resumes FP8 rows and
  its keys differ from bf16 keys; sessions in batch slots (0180) with FP8 == alone;
- fp8 against bf16 KV on the same checkpoint: the last prompt row's hidden row and logits within tolerance, the
  greedy first token among bf16's top 5; a snapshot never resumes on an engine with the other row format;
  GLM53_TF_KV_DTYPE validation (fp8 needs the latent cache).

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q tests/cuda/test_fp8_kv_patches.py
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from types import SimpleNamespace  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.glm5_next.cuda import latent, qmm, weights  # noqa: E402
from test_glm_engine import _TwoCopies, _checkpoint, _drafter, _generate  # noqa: E402

SAMPLINGS = [Sampling(1234, 1.0, 20, 0.95), None]
IDS = ["sampled", "greedy"]
DEV = "cuda"
H, DQ, L = 32, 256, 512


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")
    monkeypatch.delenv("GLM53_TF_DEPTH", raising=False)


def _engine(path, *, fp8: bool = True, rows: int = 64, rows_max: int | None = None, context: int = 0, batch: int = 1,
            nonexpert: str | None = None, long_graphs: bool = True, gib: float = 0.0, env: dict | None = None):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GLM53_TF_LATENT_KV", "1")
        mp.setenv("GLM53_TF_KV_DTYPE", "fp8" if fp8 else "bf16")
        mp.setenv("GLM53_TF_LONGCTX_GRAPHS", "1" if long_graphs else "0")
        mp.setenv("GLM53_TF_PREFILL_ROWS", str(rows))
        mp.setenv("GLM53_TF_PREFILL_ROWS_MAX", str(rows_max or rows))
        mp.setenv("GLM53_TF_BATCH", str(batch))
        mp.setenv("GLM53_TF_BATCH_ADMIT_GB", "0")
        mp.setenv("GLM53_TF_BATCH_RESERVE_GB", "0.25")
        mp.setenv("GLM53_TF_SESSION_GIB", str(gib))
        mp.setenv("GLM53_TF_SESSION_RESERVE_GIB", "0")
        mp.setenv("GLM53_TF_SESSION_EVERY", "0")
        mp.setenv("GLM53_TF_SESSION_FORK_MIN", "256")
        for k in ("GLM53_TF_FAST_PREFILL", "GLM53_TF_LEAN_PREFILL", "GLM53_TF_FP8_PREFILL", "GLM53_TF_CALIB_ONLINE",
                  "GLM53_TF_PROFILE", "GLM53_TF_BATCH_SESSIONS", "GLM53_TF_BATCH_PIECE"):
            mp.delenv(k, raising=False)
        for k, v in (env or {}).items():
            mp.setenv(k, str(v))
        if nonexpert is not None:
            mp.setattr(weights, "NONEXPERT", nonexpert)
            mp.setenv("GLM53_TF_NONEXPERT", nonexpert)
        return GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", context=context,
                         comm=_TwoCopies())


def _free(*engines) -> None:
    for e in engines:
        del e
    torch.cuda.empty_cache()


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm())


def _q8(x: torch.Tensor) -> torch.Tensor:
    """bf16 rows [n, 512] through the GPU writer -> FP8 rows [n, 528]."""

    lc = torch.zeros((x.shape[0], latent.ROW8), dtype=torch.uint8, device=DEV)
    latent.latent_write(x.contiguous(), lc, torch.zeros((1,), dtype=torch.int32, device=DEV))
    return lc


# -- the writer ---------------------------------------------------------------------------------------------------
@torch.no_grad()
def test_writer_bytes_equal_torch_conversion():
    """Every finite bf16 value <= 448 (a 448 anchor per row: scale 1) and random rows over 11 decades: the kernel's
    bytes (values and scale) == torch's round-to-nearest-even float8_e4m3fn conversion of x / 2^k."""

    pat = torch.arange(0, 1 << 16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float()
    pat = pat[torch.isfinite(pat) & (pat.abs() <= 448)]
    pat = pat[: (pat.numel() // 511) * 511].view(-1, 511)
    rows = torch.cat([pat, torch.full((pat.shape[0], 1), 448.0)], 1)
    g = torch.Generator(device="cpu").manual_seed(1)
    rnd = torch.randn((400, L), generator=g) * torch.logspace(-8, 3, 400)[:, None]
    rnd[7] = 0.0
    rnd[9, :] = -0.0
    rnd[11, 3] = 1e-30
    x = torch.cat([rows, rnd]).to(torch.bfloat16).to(DEV)
    got = _q8(x)
    want = latent.quantize_rows_reference(x)
    bad = (got != want).any(1).nonzero().flatten().tolist()
    assert not bad, f"{len(bad)} rows differ from torch's e4m3 conversion (first: {bad[:5]})"
    scale = got[:, 512:516].contiguous().view(torch.float32).flatten()
    assert torch.all(torch.log2(scale) == torch.round(torch.log2(scale)))           # powers of two
    assert float(scale[rows.shape[0] - 1]) == 1.0 and float(scale[rows.shape[0] + 7]) == 1.0


@torch.no_grad()
def test_writer_rows_are_independent_and_bounded():
    g = torch.Generator(device="cpu").manual_seed(2)
    x = (torch.randn((300, L), generator=g) * torch.logspace(-3, 2, 300)[:, None]).to(torch.bfloat16).to(DEV)
    block = torch.zeros((400, latent.ROW8), dtype=torch.uint8, device=DEV)
    latent.latent_write(x, block, torch.tensor([50], dtype=torch.int32, device=DEV))
    assert int(block[:50].sum()) == 0 and int(block[350:].sum()) == 0
    one = torch.zeros_like(block)
    for r in range(0, 300, 7):
        latent.latent_write(x[r:r + 1].contiguous(), one, torch.tensor([50 + r], dtype=torch.int32, device=DEV))
        assert torch.equal(one[50 + r], block[50 + r]), r
    dq = latent.dequantize_rows(block[50:350]).float()
    xf = x.float()
    s = block[50:350, 512:516].contiguous().view(torch.float32)
    err = (dq - xf).abs()
    # half an e4m3 step: 2^-4 of |x| in the normal range, 2^-10 s below 2^-6 s (the subnormal step is 2^-9 s)
    assert torch.all(err <= torch.maximum(xf.abs() * 2.0 ** -4, s * 2.0 ** -10)), float(err.max())
    assert torch.all(xf.abs().amax(1) / s.flatten() <= 448) and torch.all(xf.abs().amax(1) / s.flatten() > 224)


# -- attention kernels --------------------------------------------------------------------------------------------
def _kvb(kind: str, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    wk = (torch.randn((H * DQ, L), generator=g) * 0.05).to(torch.bfloat16).to(DEV)
    wv = (torch.randn((H * DQ, L), generator=g) * 0.05).to(torch.bfloat16).to(DEV)
    if kind == "q4":
        return qmm.quantize4(wk), qmm.quantize4(wv)
    return qmm.make_b16(wk), qmm.make_b16(wv)


def _scratch(rows: int, capacity: int) -> latent.Scratch:
    cfg = SimpleNamespace(heads=2 * H, kv_lora=L, v_dim=DQ, dense_limit=2051)
    return latent.Scratch(SimpleNamespace(cfg=cfg, world=2), rows, capacity, DEV)


def _latents(cap: int, seed: int) -> torch.Tensor:
    """kv_a_layernorm-like rows: unit RMS times a per-channel gain with a few large channels (outliers stress the
    per-row scale)."""

    g = torch.Generator(device="cpu").manual_seed(seed)
    gain = torch.exp(torch.randn((L,), generator=g) * 0.5)
    gain[torch.randperm(L, generator=g)[:4]] *= 8
    return (torch.randn((cap, L), generator=g) * gain).to(torch.bfloat16).to(DEV)


@pytest.mark.parametrize("bm", [16, 32])
@torch.no_grad()
def test_fp8_attention_is_bf16_attention_on_dequantized_rows(bm):
    """Dense and sparse latent attention over FP8 rows == the bf16 kernels over ``dequantize_rows`` of them, bit for
    bit (the dequantization is exact and the rest of the arithmetic is shared)."""

    cap, P, R = 3000, 1900, 8
    lat = _latents(cap, 3)
    lc8 = _q8(lat)
    lcb = latent.dequantize_rows(lc8).contiguous()
    g = torch.Generator(device="cpu").manual_seed(4)
    qa = torch.randn((R, H, L), generator=g).to(torch.bfloat16).to(DEV)
    pos = torch.tensor([P], dtype=torch.int32, device=DEV)
    s = _scratch(16, cap)
    kw = dict(bm=bm, stages=latent.FAST_STAGES) if bm == 32 else {}
    for nch in (-(-(P + R) // 512), None):
        a = latent.attention_latent(qa, lc8, pos, s, scale=DQ ** -0.5, nch=nch, **kw).clone()
        b = latent.attention_latent(qa, lcb, pos, s, scale=DQ ** -0.5, nch=nch, **kw).clone()
        assert torch.equal(a, b), nch
    W = 2051
    tokens = torch.full((R, W), -1, dtype=torch.int32)
    counts = torch.zeros((R,), dtype=torch.int32)
    for r in range(0, R, 2):
        n = 700 + 150 * r
        tokens[r, :n] = torch.sort(torch.randperm(P + r + 1, generator=g)[:n]).values.to(torch.int32)
        counts[r] = n
    tokens, counts = tokens.to(DEV), counts.to(DEV)
    ua = torch.zeros((R, H, L), dtype=torch.float32, device=DEV)
    ub = torch.zeros_like(ua)
    latent.sparse_latent(qa, lc8, tokens, counts, ua, DQ ** -0.5, s.sparse_part(R), **kw)
    latent.sparse_latent(qa, lcb, tokens, counts, ub, DQ ** -0.5, s.sparse_part(R), **kw)
    assert torch.equal(ua, ub)


@pytest.mark.parametrize("kind", ["q4", "b16"])
@torch.no_grad()
def test_fp8_latent_close_to_expanded_reference(kind):
    """Quality hook at kernel level: absorb -> attention over FP8 rows -> expand against the fp32 expanded reference
    on the original (bf16) latents, next to the bf16 cache's own error."""

    kv_k, kv_v = _kvb(kind, 1)
    cap, P, R = 3000, 1900, 6
    lat = _latents(cap, 5)
    g = torch.Generator(device="cpu").manual_seed(2)
    q = (torch.randn((R, H, DQ), generator=g) * 2).to(torch.bfloat16).to(DEV)
    scale = DQ ** -0.5
    pos = torch.tensor([P], dtype=torch.int32, device=DEV)
    ref = latent.expanded_reference(q, lat[:P + R], kv_k, kv_v, scale, [P + r for r in range(R)])
    outs = {}
    for name, lc in (("bf16", lat.clone()), ("fp8", _q8(lat))):
        s = _scratch(8, cap)
        qa = latent.absorb(q, kv_k, s.qa[:R])
        u = latent.attention_latent(qa, lc, pos, s, scale=scale, nch=-(-(P + R) // 512))
        outs[name] = latent.expand(u, kv_v, s.o[:R]).float().clone()
    e16, e8 = _rel(outs["bf16"], ref), _rel(outs["fp8"], ref)
    print(f"[fp8 kv] {kind}: rel err vs fp32 expanded: bf16 {e16:.2e}, fp8 {e8:.2e}")
    assert e16 < 1e-2 and e8 < 4e-2, (e16, e8)


@torch.no_grad()
def test_fp8_window_rows_equal_serial_rows():
    """Row invariance on FP8 rows: each row of a window (dense chunks, sparse lists) has the bits of that row alone at
    its position, whatever the cache holds past it; a graph-style call (every chunk) == exact chunk counts."""

    cap, P, R = 2600, 1500, 8
    lat = _latents(cap, 6)
    lc = _q8(lat)
    g = torch.Generator(device="cpu").manual_seed(7)
    qa = torch.randn((R, H, L), generator=g).to(torch.bfloat16).to(DEV)
    scale = DQ ** -0.5
    s = _scratch(16, cap)
    pos = torch.tensor([P], dtype=torch.int32, device=DEV)
    uw = latent.attention_latent(qa, lc, pos, s, scale=scale, nch=-(-(P + R) // 512)).clone()
    assert torch.equal(uw, latent.attention_latent(qa, lc, pos, s, scale=scale).clone())
    for r in range(R):
        other = lc.clone()
        other[P + r + 1:] = _q8(_latents(cap - P - r - 1, 100 + r))
        pr = torch.tensor([P + r], dtype=torch.int32, device=DEV)
        u1 = latent.attention_latent(qa[r:r + 1].contiguous(), other, pr, s, scale=scale, nch=-(-(P + r + 1) // 512))
        assert torch.equal(u1[0], uw[r]), r
    tokens = torch.full((R, 2051), -1, dtype=torch.int32)
    counts = torch.zeros((R,), dtype=torch.int32)
    for r in range(1, R, 2):
        n = 900 + 80 * r
        tokens[r, :n] = torch.sort(torch.randperm(P + r + 1, generator=g)[:n]).values.to(torch.int32)
        counts[r] = n
    tokens, counts = tokens.to(DEV), counts.to(DEV)
    us = uw.clone()
    latent.sparse_latent(qa, lc, tokens, counts, us, scale)
    for r in range(R):
        if counts[r] == 0:
            assert torch.equal(us[r], uw[r])
            continue
        u1 = torch.zeros((1, H, L), dtype=torch.float32, device=DEV)
        latent.sparse_latent(qa[r:r + 1].contiguous(), lc, tokens[r:r + 1].contiguous(), counts[r:r + 1].contiguous(),
                             u1, scale)
        assert torch.equal(u1[0], us[r]), r


def test_kv_dtype_setting(monkeypatch):
    monkeypatch.setenv("GLM53_TF_LATENT_KV", "1")
    for v, want in (("", "bf16"), ("bf16", "bf16"), ("FP8", "fp8"), (" fp8 ", "fp8")):
        monkeypatch.setenv("GLM53_TF_KV_DTYPE", v)
        assert latent.kv_dtype() == want
    monkeypatch.setenv("GLM53_TF_KV_DTYPE", "e5m2")
    with pytest.raises(ValueError):
        latent.kv_dtype()
    monkeypatch.setenv("GLM53_TF_KV_DTYPE", "fp8")
    monkeypatch.setenv("GLM53_TF_LATENT_KV", "0")
    with pytest.raises(ValueError, match="LATENT_KV"):
        latent.kv_dtype()


# -- the engine ----------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm_fp8kv")
    _checkpoint(path / "model")
    _drafter(path / "dflash2")
    return path


@pytest.fixture(scope="module")
def e8(ckpt):
    return _engine(ckpt)


@pytest.fixture(scope="module")
def e8_512(ckpt):
    return _engine(ckpt, rows=512)


@pytest.fixture(scope="module")
def e16(ckpt):
    return _engine(ckpt, fp8=False)


def test_fp8_caches_and_bytes(e8, e16):
    st, c = e8.e.st, e8.w.cfg
    assert e8.w.meta["kv_fp8"] and not e16.w.meta["kv_fp8"]
    assert st.vc is st.kc and st.mtp_vc is st.mtp_kc
    assert all(t.dtype == torch.uint8 and t.shape == (st.capacity, latent.ROW8) for t in st.kc)
    assert st.mtp_kc.dtype == torch.uint8 and st.mtp_kc.shape == (st.capacity, latent.ROW8)
    n = len(st.kc) + 1
    assert latent.kv_bytes_per_token(st) == n * latent.ROW8
    assert latent.kv_bytes_per_token(e16.e.st) == n * c.kv_lora * 2
    # every per-position component the session store pages (layout) keeps its row format
    from tensorfold.families.glm5_next.cuda import sessions

    comps = sessions.layout(st)
    assert any(x.live.dtype == torch.uint8 for x in comps)


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
def test_fp8_drafted_equals_serial(e8, sampling):
    prompt = list(np.random.default_rng(5).integers(0, 1000, size=37))
    serial, stats = _generate(e8, prompt, sampling, draft=False, tokens=32)
    assert len(serial) == 32 and stats["drafts"] is False
    for policy in (None, "auto", "auto:1:1:0", "1", "2", "3", "7", "f3", "fc5:0.3", "c3:0.35", "a:0.6:0.85"):
        drafted, stats = _generate(e8, prompt, sampling, policy=policy, tokens=32)
        assert drafted == serial, policy
        assert stats["rounds"] >= 1 and stats["min_rows"] >= 2, (policy, stats)


@pytest.mark.parametrize("sampling", [Sampling(7, 1.0, 20, 0.95), None], ids=IDS)
def test_fp8_resumed_equals_fresh(e8, sampling):
    rng = np.random.default_rng(9)
    first = list(rng.integers(0, 1000, size=70))
    unrelated = list(rng.integers(0, 1000, size=12))
    reply, _ = _generate(e8, first, sampling)
    after = first + reply + [5, 6, 7]
    warm, stats = _generate(e8, after, sampling)
    assert stats["cached"] >= len(first) + len(reply) - 1
    _generate(e8, unrelated, sampling)
    cold, stats = _generate(e8, after, sampling)
    assert stats["cached"] == 0 and warm == cold
    serial, _ = _generate(e8, after, sampling, draft=False)
    assert serial == cold


def _prefill_state(eng, prompt) -> list[torch.Tensor]:
    from tensorfold.families.glm5_next.cuda.decode import prefill

    e = eng.e
    first = prefill(e, list(prompt), None, mtp=True)
    st = e.st
    n = len(prompt)
    last_logits = qmm.matmul(e.last_hidden, eng.w.head).clone()
    out = [torch.tensor([first]), st.rec[st.cur[0]].clone(), st.conv.clone(), e.last_hidden.clone(), last_logits]
    out += [k[:n].clone() for k in st.kc] + [st.mtp_kc[:st.mtp_len].clone()]
    eng.cache = []
    return out


@torch.no_grad()
def test_fp8_prefill_chunk_sizes_same_bits(e8, e8_512):
    prompt = list(np.random.default_rng(21).integers(0, 1000, size=700))
    a, b = _prefill_state(e8, prompt), _prefill_state(e8_512, prompt)
    assert len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


@torch.no_grad()
def test_fp8_close_to_bf16_kv(e8, e16):
    """Quality hook: the same checkpoint with fp8 and bf16 latent rows: the last prompt row's final-normed hidden row
    and logits agree within tolerance, and fp8's greedy first token is among bf16's top 5. The cached latents
    themselves: fp8 rows == bf16 rows within half an e4m3 step."""

    prompt = list(np.random.default_rng(23).integers(0, 1000, size=300))
    a, b = _prefill_state(e8, prompt), _prefill_state(e16, prompt)
    hidden_rel, logit_rel = _rel(a[3], b[3]), _rel(a[4], b[4])
    print(f"[fp8 kv] engine: hidden rel {hidden_rel:.2e}, logits rel {logit_rel:.2e}")
    # loose: the synthetic checkpoint's random weights are no model of GLM's sensitivity (the real check is the
    # user's quality pass, docs/GPU-PLAN-4x256k.md); this catches a broken dequantization (errors of order 1)
    assert hidden_rel < 0.15 and logit_rel < 0.2, (hidden_rel, logit_rel)
    assert int(torch.topk(a[4].float(), 1).indices[0, 0]) in torch.topk(b[4].float(), 5).indices[0].tolist()
    n16 = b[5].float()
    n8 = latent.dequantize_rows(a[5]).float()
    assert _rel(n8[:10], n16[:10]) < 5e-2                        # the first layer's rows (same inputs up to there)


def test_snapshot_never_crosses_row_formats(e8, e16):
    from tensorfold.families.glm5_next.cuda.decode import restore, take_snapshot

    snap = take_snapshot(e16.e, [1, 2, 3], None, mtp=False)
    assert snap.kv == 0 and take_snapshot(e8.e, [1, 2, 3], None, mtp=False).kv == 1
    with pytest.raises(ValueError, match="latent rows"):
        restore(e8.e, snap)


# -- past 2,051 tokens: sparse top-k over FP8 rows --------------------------------------------------------------------
@pytest.fixture(scope="module")
def long_ckpt(tmp_path_factory):
    from test_patches import _index_heads_32

    path = tmp_path_factory.mktemp("glm_fp8kv_long")
    _checkpoint(path / "model", exl3=True)
    _index_heads_32(path / "model")
    _drafter(path / "dflash2")
    return path


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
def test_fp8_sparse_long_context(long_ckpt, sampling):
    """A 3,000-token prompt crosses the dense limit: 64- and 512-row chunks, serial and drafted, with and without
    patches/0050's graphs, give the same reply; a follow-up resumed from the kept state equals a fresh prefill."""

    prompt = list(np.random.default_rng(31).integers(0, 1000, size=3000))
    e64 = _engine(long_ckpt, rows=64, context=4096, nonexpert="q4mse", long_graphs=False)
    a, _ = _generate(e64, prompt, sampling, draft=False, tokens=24)
    d, _ = _generate(e64, prompt, sampling, policy="2", tokens=24)
    assert a == d
    _free(e64)
    e = _engine(long_ckpt, rows=512, context=4096, nonexpert="q4mse")
    b, _ = _generate(e, prompt, sampling, draft=False, tokens=24)
    for policy in (None, "2", "f3"):
        c, _ = _generate(e, prompt, sampling, policy=policy, tokens=24)
        assert a == b == c, policy
    after = prompt + b + [3, 4]
    warm, stats = _generate(e, after, sampling, tokens=16)
    assert stats["cached"] >= len(prompt)
    _generate(e, [1, 2, 3], sampling, tokens=4)
    cold, stats = _generate(e, after, sampling, tokens=16)
    assert stats["cached"] == 0 and warm == cold
    _free(e)


# -- batched rounds (0120 / 0200) --------------------------------------------------------------------------------------
BATCH_ON_SET = {"GLM53_TF_BATCH_CAPTURE_AFTER": 3, "GLM53_TF_BATCH_PARITY_KEY": 1, "GLM53_TF_BATCH_MTP": 1,
                "GLM53_TF_BATCH_ROW_MS": 6.5, "GLM53_TF_BATCH_SHORT": 64}


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
@pytest.mark.parametrize("slots", [2, 4])
def test_fp8_batched_equals_alone(ckpt, e8, slots, sampling):
    """2 and 4 slots with 0200's production on-set: every reply == the request served alone on the fp8 engine."""

    eb = _engine(ckpt, batch=slots, env=BATCH_ON_SET if slots == 4 else None)
    assert eb.batch is not None and all(s.kc[0].dtype == torch.uint8 for s in eb.batch.states)
    rng = np.random.default_rng(41)
    prompts = [list(rng.integers(0, 1000, size=n)) for n in (37, 29, 80, 150)[:slots]]
    want = [_generate(e8, p, sampling, draft=False, tokens=24)[0] for p in prompts]
    pols = [None, "2", "c3:0.35", "a:0.6:0.85"]
    for shift in range(2):
        got = eb.batch.generate_batch([dict(prompt=p, max_tokens=24, sampling=sampling, policy=pols[(i + shift) % 4])
                                       for i, p in enumerate(prompts)])
        assert [t for t, _ in got] == want, shift
        assert all(s["batched_rounds"] >= 1 for _, s in got)
    _free(eb)


# -- sessions (0110; in batch slots, 0180) ------------------------------------------------------------------------------
@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
def test_fp8_sessions_resume_and_keys(ckpt, e8, sampling):
    """The session store pages FP8 rows: interleaved sessions resume (cached > 0) and every reply == a fresh prefill
    on the fp8 engine without the store; entry / page keys carry the row format."""

    from tensorfold.families.glm5_next.cuda import sessions

    es = _engine(ckpt, gib=1.0)
    assert es.store is not None and sessions.KV_TAG == b"kv:fp8"
    rng = np.random.default_rng(12)
    sys_prompt = list(rng.integers(0, 1000, size=600))
    turns = {k: sys_prompt + list(rng.integers(0, 1000, size=n)) for k, n in (("A", 150), ("B", 170), ("C", 130))}
    for k in "ABCBA":
        e8.cache = []
        want, _ = _generate(e8, turns[k], sampling, draft=False, tokens=16)
        got, stats = _generate(es, turns[k], sampling, policy="auto:1:1:0", tokens=16)
        assert got == want, (k, stats)
        turns[k] = turns[k] + got + [3, 4]
    assert es.store.stats["restores"] >= 1
    k8 = sessions.ids_key(0, [1, 2, 3], True, True)
    sessions.KV_TAG = b""
    try:
        assert sessions.ids_key(0, [1, 2, 3], True, True) != k8
    finally:
        sessions.KV_TAG = b"kv:fp8"
    _free(es)


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
def test_fp8_batch_sessions_equal_alone(ckpt, e8, sampling):
    """patches/0180 with FP8 rows: 4 sessions on 3 slots over 3 waves (sessions move between slots and resume from
    the store): every reply == the request alone; follow-ups resume."""

    eng = _engine(ckpt, batch=3, gib=1.0, env={"GLM53_TF_BATCH_SESSIONS": 1, "GLM53_TF_BATCH_PREFILL_SHARE": 1.0})
    assert eng.store is not None and eng.store is eng.batch.store
    rng = np.random.default_rng(13)
    sys_prompt = list(rng.integers(0, 1000, size=600))
    turns = {k: sys_prompt + list(rng.integers(0, 1000, size=n)) for k, n in (("A", 150), ("B", 170), ("C", 130),
                                                                               ("D", 110))}
    for w, order in enumerate(("ABCD", "DCBA", "BDAC")):
        got = eng.batch.generate_batch([dict(prompt=turns[k], max_tokens=16, sampling=sampling, policy="2")
                                        for k in order])
        for k, (reply, stats) in zip(order, got):
            e8.cache = []
            want, _ = _generate(e8, turns[k], sampling, draft=False, tokens=16)
            assert reply == want, (w, k, stats)
            if w:
                assert stats["cached"] > 0, (w, k, stats)
            turns[k] = turns[k] + reply + [7 + w, 11 + w]
    _free(eng)
