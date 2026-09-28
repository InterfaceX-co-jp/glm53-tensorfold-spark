"""patches/0083 (FP8 fast prefill, ``glm5_next/cuda/fp8pf.py``, GLM53_TF_FP8_PREFILL) and 0092 (``tf_knobs.fp8_prefill``).

FP8 prefill changes only the main model's FAST prefill chunks: the large non-expert 4-bit / BF16 matmuls run on e4m3
tensor cores with per-row activation scales (``fast_qmm.matmul_fp8``), and on the latent cache (patches/0060) the
absorb / expand projections run on bf16 tensor cores (``latent.absorb_tc`` / ``expand_tc``) and latent attention
uses 32-query tiles. Checked:

- host only: the switch, the snapshot tag (C + 1), the knob (parse, header), and the fast rule with FP8 and bf16 fast
  requests mixed in one conversation on a hostile fake model (the real prefill / snapshot / resume code): every
  resumed request equals a fresh prefill of the same mode, and the two modes never resume from each other;
- kernels (GPU on the real per-rank shapes at M = 1024 / 2048 / 8192; Triton's CPU interpreter on small shapes when
  no GPU is visible, with its e4m3 rounding and fp8 dots replaced by torch's): ``matmul_fp8`` against a host model
  of the same arithmetic (x rounded to e4m3 per row, exact 4-bit weights), against the fp32 product (the FP8 error
  bound) and against the bf16 fast kernel; deterministic; row-independent (a row's bits do not depend on the other
  rows or on M: what lean sub-blocks and any chunking need); the "cvt" and "bits" weight encodings give the same
  bits (the latter needs e4m3 subnormals in the tensor cores); every tile config gives the same bits; the tiny
  shapes and short calls fall back to the bf16 fast kernel; BF16 weights; the latent tensor-core absorb / expand
  against the fp32 kernels and a float64 reference; 32-query latent attention against 16-query;
- the engine with FP8 fast prefill (synthetic checkpoint, ``FP8_MIN_NK = 0`` so every matmul runs in fp8): it runs
  and differs from the bf16 fast prefill (the control), deterministic, drafted == serial (7 policies, greedy and
  sampled), resumed == fresh (within the last chunk, across grid points, after a reply, a chain), FP8 and bf16
  snapshots never mix (per-request knob), lean (0082) == non-lean within FP8, the latent cache past 2,051 tokens;
- quality: FP8 vs bf16 fast prefill of the same prompt: final-normed rows, logits and KDA states within a bound;
  most rows keep their argmax (the values are printed with -s, to tighten the bounds on hardware).
- timing (GPU, -s): fp8 vs the bf16 fast kernel on every large shape and M, the tile sweep, absorb / expand.

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda python -m pytest -q -s tests/cuda/test_fp8_patches.py
Host / interpreter part: PYTHONPATH=<patched TensorFold>/src:<TensorFold>/tests/cuda:tests/cuda pytest -q tests/cuda/test_fp8_patches.py
(run it in its own pytest process: the interpreter must be switched on before Triton kernels are defined).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

try:
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:          # the host-only tests still run
    torch = None
    CUDA = False

if not CUDA:
    os.environ.setdefault("TRITON_INTERPRET", "1")

from tensorfold.families.glm5_next.cuda import fastpf, fp8pf, knobs  # noqa: E402

gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")
INTERP = os.environ.get("TRITON_INTERPRET") == "1" and not CUDA
kernels_here = pytest.mark.skipif(not (CUDA or INTERP) or torch is None, reason="needs a GPU or Triton's interpreter")
DEV = "cuda" if CUDA else "cpu"
SAMPLINGS = ["sampled", "greedy"]
GRID = 128


def _sampling(kind: str):
    from tensorfold.engine.exact_sampling import Sampling

    return None if kind == "greedy" else Sampling(1234, 1.0, 20, 0.95)


def _rel(a, b) -> float:
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


# -- host only ------------------------------------------------------------------------------------------------------
def test_switch_tag_and_settings(monkeypatch):
    monkeypatch.delenv("GLM53_TF_FP8_PREFILL", raising=False)
    assert not fp8pf.enabled() and not fp8pf.default()
    monkeypatch.setenv("GLM53_TF_FP8_PREFILL", "1")
    assert fp8pf.enabled() and fp8pf.settings()[0] == 1
    monkeypatch.setenv("GLM53_TF_FP8_PREFILL", "fp8")
    with pytest.raises(ValueError, match="0 or 1"):
        fp8pf.enabled()
    for c in (64, 128, 960, 2048, 8192):
        assert fp8pf.tag(c, False) == c and fp8pf.tag(c, True) == c + 1
        assert fp8pf.grid_of(c) == fp8pf.grid_of(c + 1) == c
    assert fp8pf.tag(0, True) == 0 and fp8pf.grid_of(0) == 0         # exact prefills: no tag


def test_knob():
    if "fp8_prefill" not in knobs.HEADER:
        pytest.skip("patches/0092 not installed")
    assert knobs.parse({"fp8_prefill": 1}, rows_max=512) == {"fp8_prefill": 1}
    assert knobs.parse({"fp8_prefill": True, "fast_prefill": 1}, rows_max=512) == {"fp8_prefill": 1,
                                                                                    "fast_prefill": 1}
    with pytest.raises(ValueError, match="FP8"):
        knobs.parse({"fp8_prefill": 1}, rows_max=512, fp8_ok=False)
    assert knobs.parse({"fp8_prefill": 0}, rows_max=512, fp8_ok=False) == {"fp8_prefill": 0}
    with pytest.raises(ValueError, match="0 to 1"):
        knobs.parse({"fp8_prefill": 2}, rows_max=512)
    values = {k: 0 for k in knobs.HEADER}
    values.update(prefill_rows=256, fast_prefill=1, fp8_prefill=1)
    got, rest = knobs.decode(knobs.encode(values) + [9])
    assert got == values and rest == [9]


def _fake8_engine(monkeypatch, rows: int):
    """test_fastpf_patches' GlmEngine-shaped fake, with a model whose FAST chunks also depend on the FP8 switch (as
    the real kernels do): a stale or cross-mode resume changes the result."""

    import types
    from types import SimpleNamespace

    from test_fastpf_patches import _Fake, _FakeEngine

    from tensorfold.families.glm5_next.cuda import decode
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    class Fake8(_Fake):
        def compute(self, w, st, b, R, *, logits=True, nch=None, host_pos=None, npb=None, fast=False, head=True):
            s = int(st.rec[st.cur[0], 0])
            mode = (2 if fp8pf.ON else 1) if fast else 0
            for i, t in enumerate(self.ids[:R]):
                p = st.pos + i
                s = self.mix(s, t, p, self.dig[p], st.pos if fast else -1, R if fast else -1, mode)
                self.kv[p] = self.mix(s, 7)
                self.dig[p + 1] = self.mix(self.dig[p], int(self.kv[p]))
                self.hidden[i, 0] = s
            st.rec[1 - st.cur[0], 0] = s
            return self.hidden[:R].clone()

    fake = Fake8()
    monkeypatch.setattr(decode, "_sync", lambda w: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    e = _FakeEngine(fake, rows, True)
    e.fp8_prefill = False
    g = SimpleNamespace(e=e, drafter=None, cache=[], eos=(), online=None, calib_on=False, costs={}, f_most=7,
                        depth_cost=0, fake=fake)
    for name in ("_run", "_resume", "_remember", "_grid", "_drafters"):
        setattr(g, name, types.MethodType(getattr(GlmEngine, name), g))
    return g


@pytest.mark.skipif(torch is None, reason="torch")
@pytest.mark.parametrize("policy", ["0", "2"])
@pytest.mark.parametrize("rows", [64, 200])
def test_rule_with_fp8_and_bf16_requests_mixed(monkeypatch, rows, policy):
    from test_fastpf_patches import _fake_request

    from tensorfold.families.glm5_next.cuda import decode

    rng = np.random.default_rng(830 + rows)
    C = fastpf.grid(rows)

    def use(g):
        for name in ("stage", "compute", "commit"):
            monkeypatch.setattr(decode, name, getattr(g.fake, name))

    g = _fake8_engine(monkeypatch, rows)
    ref = _fake8_engine(monkeypatch, rows)
    more = lambda n: [int(t) for t in rng.integers(0, 1000, size=n)]         # noqa: E731
    long = more(2 * C + 17)
    last, reply = long, []
    resumed = crossed = 0
    for turn in range(48):
        fp8 = bool(rng.integers(0, 2)) if turn >= 2 else bool(turn)
        kind = rng.integers(0, 4)
        grow = more(int(rng.choice([1, 3, 30, C - 1, C, C + 5, 2 * C])))
        prompt = (last + reply + grow if kind == 0 else last + grow if kind == 1 else
                  last[:int(rng.integers(0, len(last) + 1))] + grow if kind == 2 else grow)[:3000]
        # a snapshot of the other mode that the prompt extends: must not be used
        crossed += any(s.grid == fp8pf.tag(C, not fp8) and prompt[:len(s.ids)] == s.ids and len(s.ids) < len(prompt)
                       for s in g.cache)
        use(g)
        g.e.fp8_prefill = fp8
        got, cached, state = _fake_request(g, prompt, policy=policy)
        assert all(s.grid in (C, C + 1) and len(s.ids) % C == 0 for s in g.cache)
        use(ref)
        ref.e.fp8_prefill = fp8
        ref.cache = []
        want, c0, want_state = _fake_request(ref, prompt, policy=policy)
        assert c0 == 0 and got == want and state == want_state, (fp8, turn, len(prompt), cached)
        resumed += cached > 0
        last, reply = prompt, got
    assert resumed >= 6 and crossed >= 3, (resumed, crossed)


# -- kernels ----------------------------------------------------------------------------------------------------------
@pytest.fixture
def interp_fp8(monkeypatch):
    """In Triton's CPU interpreter, the conversions and dots these kernels use done by torch, as the GPU does them:
    to e4m3 and to bf16 with RTNE (the interpreter's e4m3 rounding drops the carry into the exponent and rounds ties
    up; its bf16 rounding truncates, and it casts integers to bf16 by reinterpreting bits), and fp8 / bf16 dots on
    the values (it multiplies bf16's raw uint16 bits)."""

    if not INTERP:
        return
    import triton.language as tl
    from triton.runtime import interpreter as interp

    cast = interp.InterpreterBuilder.cast_impl
    to_fp = interp.InterpreterBuilder.create_fp_to_fp
    dot = interp.InterpreterBuilder.create_dot

    def f32(h):
        d = np.ascontiguousarray(h.data)
        if h.dtype.scalar == tl.float8e4nv:
            return torch.from_numpy(d).view(torch.float8_e4m3fn).float().numpy()
        if h.dtype.scalar == tl.bfloat16:
            return torch.from_numpy(d.view(np.int16)).view(torch.bfloat16).float().numpy()
        return d.astype(np.float32)

    def cast_impl(self, src, dst_type):
        dst = dst_type.scalar
        if dst == tl.bfloat16 and src.dtype.scalar != tl.bfloat16:
            t = torch.from_numpy(f32(src)).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
            return interp.TensorHandle(t, tl.bfloat16)
        if src.dtype.scalar == tl.bfloat16 and dst != tl.bfloat16:
            return interp.TensorHandle(f32(src).astype(interp._get_np_dtype(dst_type)), dst)
        return cast(self, src, dst_type)

    def create_fp_to_fp(self, src, dst_type, rounding_mode):
        if dst_type.scalar == tl.float8e4nv and src.dtype.scalar in (tl.float32, tl.float16, tl.bfloat16):
            t = torch.from_numpy(f32(src))
            return interp.TensorHandle(t.to(torch.float8_e4m3fn).view(torch.uint8).numpy(), tl.float8e4nv)
        return to_fp(self, src, dst_type, rounding_mode)

    def create_dot(self, a, b, d, input_precision, max_num_imprecise_acc):
        low = (tl.float8e4nv, tl.bfloat16)
        if a.dtype.scalar in low or b.dtype.scalar in low:
            return interp.TensorHandle(np.matmul(f32(a), f32(b)).astype(np.float32) + d.data, d.dtype.scalar)
        return dot(self, a, b, d, input_precision, max_num_imprecise_acc)

    monkeypatch.setattr(interp.InterpreterBuilder, "cast_impl", cast_impl)
    for name in ("create_si_to_fp", "create_ui_to_fp", "create_fp_ext", "create_fp_trunc"):
        monkeypatch.setattr(interp.InterpreterBuilder, name, lambda self, src, dst_type: cast_impl(self, src, dst_type))
    monkeypatch.setattr(interp.InterpreterBuilder, "create_fp_to_fp", create_fp_to_fp)
    monkeypatch.setattr(interp.InterpreterBuilder, "create_dot", create_dot)


# per-rank GLM-5.3-Flash shapes (N x K) that run in fp8; the interpreter gets small stand-ins
REAL = [(12576, 4096), (4096, 4096), (2048, 4096), (8192, 1536), (4096, 8192), (12288, 4096), (4096, 6144),
        (4096, 1024), (4096, 1536)]
SHAPES = REAL if CUDA else [(200, 256), (128, 384)]
ROWS = [1024, 2048, 8192] if CUDA else [64, 130]


def _x(m: int, k: int, seed: int) -> "torch.Tensor":
    """Activations like a norm's output: N(0, 1), a few outlier channels (x30), one all-zero row, one tiny row."""

    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(m, k, generator=g)
    x[:, torch.randperm(k, generator=g)[:4]] *= 30.0
    x[1] = 0.0
    x[2] *= 1e-6
    return x.to(DEV).bfloat16()


_Q = {}


def _q4(n: int, k: int):
    from tensorfold.families.glm5_next.cuda import qmm

    if (n, k) not in _Q:
        g = torch.Generator(device="cpu").manual_seed(n * 7 + k)
        _Q[(n, k)] = qmm.quantize4((torch.randn(n, k, generator=g) * 0.02).to(DEV))
    return _Q[(n, k)]


@kernels_here
@pytest.mark.parametrize("m", ROWS)
@pytest.mark.parametrize("n,k", SHAPES)
def test_fp8_q4_close_deterministic_row_independent(interp_fp8, monkeypatch, n, k, m):
    from tensorfold.families.glm5_next.cuda import fast_qmm, qmm

    if CUDA and m == 8192 and (n, k) not in ((12576, 4096), (4096, 8192), (4096, 4096)):
        pytest.skip("8192 rows: the three shapes that dominate")
    monkeypatch.setattr(fast_qmm, "FP8_MIN_NK", 0)
    q = _q4(n, k)
    x = _x(m, k, n + k + m)
    got = fast_qmm.matmul_fp8(x, q, f32=True)
    assert torch.equal(got, fast_qmm.matmul_fp8(x, q, f32=True))                           # deterministic
    model = fast_qmm.fp8_reference(x, q)
    ref = x.float() @ qmm.dequantize_q4(q).T
    loose = fast_qmm.matmul_prefill(x, q, f32=True, exact=False)
    r_model, r_ref, r_loose = _rel(got, model), _rel(got, ref), _rel(loose, ref)
    print(f"\n  fp8 {n}x{k} M={m}: vs host model {r_model:.2e}, vs fp32 {r_ref:.2e} (bf16 fast kernel {r_loose:.2e})")
    # the host model rounds to the same e4m3 values (power-of-two row scales: exact); only fp32 summation order differs
    # (the tensor cores' per-group sums vs torch's matmul): measured on GB10 2.6e-5 to 3.2e-5 over every shape and M
    assert r_model < 6e-5, r_model
    x8, sx, xs8 = fast_qmm.fp8_rows(x)
    v8, vsx = fast_qmm.fp8_row_values(x.float())
    assert torch.equal(x8.float(), v8) and torch.equal(sx, vsx[:, 0])
    assert r_ref < 4e-2, r_ref                     # e4m3: 3 mantissa bits, ~2-2.5% rms on random inputs
    assert torch.all(got[1] == 0)                  # the all-zero row: exactly zero, no NaN
    assert torch.isfinite(got).all()
    # row independence: a slice, a permutation, the first rows alone give the same bits
    s0 = min(m // 3, m - 64)                       # 64+ rows: fewer run qmm's bf16 kernels (``MIN_ROWS``)
    sub = slice(s0, s0 + 64)
    assert torch.equal(fast_qmm.matmul_fp8(x[sub], q, f32=True), got[sub])
    perm = torch.randperm(m, generator=torch.Generator().manual_seed(m)).to(DEV)
    assert torch.equal(fast_qmm.matmul_fp8(x[perm].contiguous(), q, f32=True), got[perm])
    if m > 1024:
        assert torch.equal(fast_qmm.matmul_fp8(x[:1024], q, f32=True), got[:1024])      # lean sub-blocks
    # bf16 output = the fp32 sums rounded once
    assert torch.equal(fast_qmm.matmul_fp8(x, q), got.bfloat16())
    # strided rows (qmm.matmul's contract: e.g. the KDA f_a / g_a slices)
    wide = torch.zeros((m, k + 128), dtype=torch.bfloat16, device=DEV)
    wide[:, 64:64 + k] = x
    assert torch.equal(fast_qmm.matmul_fp8(wide[:, 64:64 + k], q, f32=True), got)


@kernels_here
def test_fp8_weight_encodings_and_tiles_same_bits(interp_fp8, monkeypatch):
    """"cvt" (q as a normal e4m3 number) and "bits" (the nibble read as e4m3, q * 2^-9, subnormal for q < 8) give the
    same bits, and so does every tile config: the tile only picks which program runs a row's fixed operations. If
    "bits" differs on a GPU, its tensor cores flush e4m3 subnormals: keep "cvt"."""

    from tensorfold.families.glm5_next.cuda import fast_qmm

    monkeypatch.setattr(fast_qmm, "FP8_MIN_NK", 0)
    n, k, m = (4096, 4096, 1024) if CUDA else (192, 256, 100)
    q = _q4(n, k)
    x = _x(m, k, 5)
    monkeypatch.setattr(fast_qmm, "FP8_WENC", "cvt")
    want = fast_qmm.matmul_fp8(x, q, f32=True)
    monkeypatch.setattr(fast_qmm, "FP8_WENC", "bits")
    assert torch.equal(fast_qmm.matmul_fp8(x, q, f32=True), want)
    monkeypatch.setattr(fast_qmm, "FP8_WENC", "cvt")
    tiles = ["128,128,8,3", "128,64,4,3", "64,128,4,2", "128,256,8,3", "256,128,8,3"] if CUDA else ["64,64,4,2",
                                                                                                    "32,128,4,2"]
    for t in tiles:
        monkeypatch.setenv("GLM53_TF_FP8_TILE", t)
        assert torch.equal(fast_qmm.matmul_fp8(x, q, f32=True), want), t


@kernels_here
def test_fp8_fallbacks_and_b16(interp_fp8, monkeypatch):
    from tensorfold.families.glm5_next.cuda import fast_qmm, qmm

    # tiny shapes and calls under 64 rows: the bf16 fast kernel's (and qmm's) bits
    small = (4096, 128) if CUDA else (128, 128)
    q = _q4(*small)
    x = _x(300, small[1], 9)
    assert small[0] * small[1] < fast_qmm.FP8_MIN_NK
    assert torch.equal(fast_qmm.matmul_fp8(x, q, f32=True), fast_qmm.matmul_prefill(x, q, f32=True, exact=False))
    monkeypatch.setattr(fast_qmm, "FP8_MIN_NK", 0)
    x = _x(40, small[1], 10)
    assert torch.equal(fast_qmm.matmul_fp8(x, q, f32=True), qmm.matmul(x, q, f32=True))
    # BF16 weights: per-output-channel e4m3 scales
    n, k, m = (4096, 4096, 1024) if CUDA else (128, 256, 160)
    g = torch.Generator(device="cpu").manual_seed(11)
    w = (torch.randn(n, k, generator=g) * 0.02)
    w[3] = 0.0
    b = qmm.make_b16(w.to(DEV).bfloat16())
    x = _x(m, k, 12)
    got = fast_qmm.matmul_fp8(x, b, f32=True)
    assert torch.equal(got, fast_qmm.matmul_fp8(x, b, f32=True))
    ref = x.float() @ b.weight.float().T
    r_model, r_ref = _rel(got, fast_qmm.fp8_reference(x, b)), _rel(got, ref)
    print(f"\n  fp8 BF16 {n}x{k} M={m}: vs host model {r_model:.2e}, vs fp32 {r_ref:.2e}")
    # BF16 weights: e4m3 weights too, summed in 64-input steps; the summation-order deviation from the host model was
    # measured at 1.9e-4 on GB10 (4096 x 4096, M = 1024)
    assert r_model < 5e-4 and r_ref < 6e-2, (r_model, r_ref)
    assert torch.all(got[:, 3] == 0) and torch.isfinite(got).all()
    assert torch.equal(fast_qmm.matmul_fp8(x[64:128], b, f32=True), got[64:128])


def _latent_shapes():
    return (32, 256, 512, 256) if CUDA else (2, 128, 128, 128)          # heads, qk dim, latent, v dim


@kernels_here
@pytest.mark.parametrize("m", [1024, 2048] if CUDA else [70])
def test_latent_tensor_core_absorb_expand(interp_fp8, m):
    from tensorfold.families.glm5_next.cuda import latent, qmm

    H, DQ, L, DV = _latent_shapes()
    g = torch.Generator(device="cpu").manual_seed(m)
    kk = qmm.quantize4((torch.randn(H * DQ, L, generator=g) * 0.05).to(DEV))
    kv = qmm.quantize4((torch.randn(H * DV, L, generator=g) * 0.05).to(DEV))
    q = torch.randn(m, H, DQ, generator=g).to(DEV).bfloat16()
    u = torch.randn(m, H, L, generator=g).to(DEV)
    a0 = latent.absorb(q, kk, torch.empty((m, H, L), dtype=torch.bfloat16, device=DEV))
    a1 = latent.absorb_tc(q, kk, torch.empty((m, H, L), dtype=torch.bfloat16, device=DEV))
    wk = qmm.dequantize_q4(kk).double().view(H, DQ, L)
    ra = torch.einsum("rhi,hij->rhj", q.double(), wk)
    e0 = latent.expand(u, kv, torch.empty((m, H * DV), dtype=torch.bfloat16, device=DEV))
    e1 = latent.expand_tc(u, kv, torch.empty((m, H * DV), dtype=torch.bfloat16, device=DEV))
    wv = qmm.dequantize_q4(kv).double().view(H, DV, L)
    re = torch.einsum("rhj,hnj->rhn", u.double(), wv).reshape(m, H * DV)
    print(f"\n  absorb: fp32 kernel {_rel(a0, ra):.2e}, tensor cores {_rel(a1, ra):.2e}; "
          f"expand: fp32 kernel {_rel(e0, re):.2e}, tensor cores {_rel(e1, re):.2e}")
    assert _rel(a1, ra) < 6e-3 and _rel(e1, re) < 6e-3          # bf16 output rounding is ~2e-3
    assert torch.equal(a1, latent.absorb_tc(q, kk, torch.empty_like(a1)))
    assert torch.equal(e1, latent.expand_tc(u, kv, torch.empty_like(e1)))
    s = slice(m // 2, m // 2 + 64)                               # row-independent
    assert torch.equal(latent.absorb_tc(q[s].contiguous(), kk, torch.empty_like(a1[s])), a1[s])
    assert torch.equal(latent.expand_tc(u[s].contiguous(), kv, torch.empty_like(e1[s])), e1[s])


@kernels_here
def test_latent_attention_32_query_tiles(interp_fp8):
    """FAST_BM (32 queries a tile) vs BM (16): same attention to fp32 rounding (0060 measured identical bits on the
    real shapes), deterministic, dense and sparse."""

    from types import SimpleNamespace

    from tensorfold.families.glm5_next.cuda import latent

    H, L = (32, 512) if CUDA else (4, 128)
    R, P = (1024, 1024) if CUDA else (6, 40)
    g = torch.Generator(device="cpu").manual_seed(3)
    lc = torch.randn(P + R + 64, L, generator=g).to(DEV).bfloat16()
    qa = torch.randn(R, H, L, generator=g).to(DEV).bfloat16()
    pos = torch.tensor([P], dtype=torch.int32, device=DEV)
    cfg = SimpleNamespace(heads=H * 2, kv_lora=L, v_dim=128, dense_limit=2051)
    sc = latent.Scratch(SimpleNamespace(cfg=cfg, world=2), R, P + R + 64, DEV)
    nch = -(-(P + R) // latent.CHUNK)
    out16 = latent.attention_latent(qa, lc, pos, sc, scale=0.05, nch=nch, out=torch.empty((R, H, L), device=DEV))
    out32 = latent.attention_latent(qa, lc, pos, sc, scale=0.05, nch=nch, out=torch.empty((R, H, L), device=DEV),
                                    bm=latent.FAST_BM, stages=latent.FAST_STAGES)
    again = latent.attention_latent(qa, lc, pos, sc, scale=0.05, nch=nch, out=torch.empty((R, H, L), device=DEV),
                                    bm=latent.FAST_BM, stages=latent.FAST_STAGES)
    print(f"\n  dense latent attention, 32- vs 16-query tiles: {_rel(out32, out16):.2e}, "
          f"bit-identical {torch.equal(out32, out16)}")
    assert torch.equal(out32, again) and _rel(out32, out16) < 1e-5
    W = 2048 if CUDA else 24
    tok = torch.stack([torch.randperm(P, generator=g)[:W].sort().values for _ in range(R)]).to(DEV).int()
    cnt = torch.full((R,), W, dtype=torch.int32, device=DEV)
    cnt[0] = W // 3
    s16 = torch.zeros((R, H, L), device=DEV)
    s32 = torch.zeros((R, H, L), device=DEV)
    latent.sparse_latent(qa, lc, tok, cnt, s16, 0.05)
    latent.sparse_latent(qa, lc, tok, cnt, s32, 0.05, bm=latent.FAST_BM, stages=latent.FAST_STAGES)
    print(f"  sparse: {_rel(s32, s16):.2e}, bit-identical {torch.equal(s32, s16)}")
    assert _rel(s32, s16) < 1e-5


# -- engines (GPU) ----------------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_lookup(monkeypatch):
    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")
    monkeypatch.setenv("GLM53_TF_NONEXPERT", "q4mse")


@pytest.fixture(scope="module")
def every_matmul_fp8():
    """The synthetic checkpoint's matrices are far below FP8_MIN_NK: run all of them in fp8 so the engine tests see
    the FP8 kernels (the switch is read at call time)."""

    from tensorfold.families.glm5_next.cuda import fast_qmm

    old = fast_qmm.FP8_MIN_NK
    fast_qmm.FP8_MIN_NK = 0
    yield
    fast_qmm.FP8_MIN_NK = old


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_fp8")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


def _engine(path, *, fp8: bool = True, lean_block: int = 0, **kw):
    from test_fastpf_patches import _engine as fast_engine

    with pytest.MonkeyPatch.context() as m:
        m.setenv("GLM53_TF_FP8_PREFILL", "1" if fp8 else "0")
        m.setenv("GLM53_TF_LEAN_PREFILL", "1" if lean_block else "0")
        if lean_block:
            m.setenv("GLM53_TF_LEAN_BLOCK", str(lean_block))
        m.delenv("GLM53_TF_FP8_TILE", raising=False)
        return fast_engine(path, **kw)


@pytest.fixture(scope="module")
def e8(ckpt, every_matmul_fp8):
    return _engine(ckpt)


@pytest.fixture(scope="module")
def e16(ckpt, every_matmul_fp8):
    return _engine(ckpt, fp8=False)


def _gen(eng, prompt, sampling, **kw):
    from test_fastpf_patches import _gen as gen

    return gen(eng, prompt, sampling, **kw)


def _cold(eng, prompt, sampling, **kw):
    from test_fastpf_patches import _cold as cold

    return cold(eng, prompt, sampling, **kw)


def _tagged_only(eng, n: int, tag: int = GRID + 1):
    assert eng.cache and all(s.grid == tag and len(s.ids) % GRID == 0 and 0 < len(s.ids) <= n for s in eng.cache)
    assert len(eng.cache[-1].ids) == (n // GRID) * GRID


@gpu
def test_fp8_engine_runs_fp8_and_is_deterministic(e8, e16):
    from test_patches import _prefill_state, _same

    assert e8.e.fp8_prefill and not e16.e.fp8_prefill and e8._grid() == GRID + 1 and e16._grid() == GRID
    prompt = list(np.random.default_rng(831).integers(0, 1000, size=700))
    a = _prefill_state(e8, prompt)
    _gen(e8, list(np.random.default_rng(832).integers(0, 1000, size=90)), None, tokens=4)
    b = _prefill_state(e8, prompt)
    assert _same(a, b)
    c = _prefill_state(e16, prompt)
    assert not _same(a, c), "the FP8 kernels did not run"                  # the control
    e8.cache = []
    _, stats = _gen(e8, prompt, None, tokens=4)
    assert stats["fast_prefill"] == GRID and stats.get("fp8_prefill") == 1
    _tagged_only(e8, len(prompt))


@gpu
@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_fp8_drafted_equals_serial(e8, sampling):
    s = _sampling(sampling)
    prompt = list(np.random.default_rng(833).integers(0, 1000, size=300))
    serial = _cold(e8, prompt, s, tokens=32)
    for policy in (None, "auto:1:1:0", "f3", "fc5:0.3", "2", "c3:0.35", "a:0.6:0.85"):
        e8.cache = []
        drafted, stats = _gen(e8, prompt, s, policy=policy, tokens=32)
        assert drafted == serial, policy
        _tagged_only(e8, len(prompt))


@gpu
@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_fp8_resumed_equals_fresh(e8, sampling):
    s = _sampling(sampling)
    rng = np.random.default_rng(834)
    more = lambda n: [int(t) for t in rng.integers(0, 1000, size=n)]       # noqa: E731
    p1 = more(300)
    e8.cache = []
    r1, _ = _gen(e8, p1, s, policy="auto:1:1:0")
    _tagged_only(e8, len(p1))
    for name, prompt, cached in (("within", p1 + more(3), 256), ("reply", p1 + r1 + more(5), 256),
                                 ("across", p1 + r1 + more(120), 256)):
        for policy in ("auto:1:1:0", "2", "f3"):
            _gen(e8, p1, s, policy="auto:1:1:0")
            warm, stats = _gen(e8, prompt, s, policy=policy)
            assert stats["cached"] == cached, (name, policy, stats["cached"])
            assert warm == _cold(e8, prompt, s), (name, policy)
    _gen(e8, p1, s)
    p2 = p1 + r1 + more(120)
    _gen(e8, p2, s)
    p3 = p2 + more(200)
    warm, stats = _gen(e8, p3, s)
    assert stats["cached"] == 384 and warm == _cold(e8, p3, s)


@gpu
def test_fp8_and_bf16_snapshots_never_mix(e8):
    if "fp8_prefill" not in knobs.HEADER:
        pytest.skip("patches/0092 not installed")
    prompt = list(np.random.default_rng(835).integers(0, 1000, size=300))
    longer = prompt + [7, 8, 9]
    e8.cache = []
    _gen(e8, prompt, None)                                                # an FP8 snapshot at 256 (tag C + 1 = 129)
    bf, stats = _gen(e8, longer, None, knobs={"fp8_prefill": 0})
    assert stats["cached"] == 0 and "fp8_prefill" not in stats and stats["tf_knobs"]["fp8_prefill"] == 0
    assert all(c.grid == GRID for c in e8.cache)
    assert bf == _cold(e8, longer, None, knobs={"fp8_prefill": 0})
    _gen(e8, prompt, None, knobs={"fp8_prefill": 0})                     # a bf16 fast snapshot at 256
    f8, stats = _gen(e8, longer, None)
    assert stats["cached"] == 0 and stats["fp8_prefill"] == 1
    assert f8 == _cold(e8, longer, None)
    assert e8._knob_state()["fp8_prefill"] == 1                           # the load-time default is back
    # an exact request on the FP8 engine: no fast chunks, exact snapshots
    ex, stats = _gen(e8, longer, None, knobs={"fast_prefill": 0})
    assert "fast_prefill" not in stats and all(c.grid == 0 for c in e8.cache)


@gpu
def test_fp8_resume_from_other_mode_refused(e8):
    from tensorfold.families.glm5_next.cuda.decode import prefill, take_snapshot

    e = e8.e
    prompt = list(np.random.default_rng(836).integers(0, 1000, size=300))
    e8.cache = []
    prefill(e, prompt, None, mtp=True)
    bf = take_snapshot(e, prompt[:256], e.last_hidden, mtp=True, grid=GRID)      # a bf16 fast tag
    with pytest.raises(ValueError, match="grid"):
        prefill(e, prompt + [1, 2], None, mtp=True, resume=bf)
    e8.cache = []


@gpu
@pytest.mark.parametrize("n", [50, 256, 300, 704])
def test_fp8_lean_equals_fp8_fast(ckpt, every_matmul_fp8, n):
    """Lean prefill (0082) keeps "same bits as the fast chunk" within FP8: the kernels are row-independent. Lengths
    whose chunks are multiples of 64 rows or shorter than 64 (a chunk that leaves a sub-block of 1-63 rows behind
    another sub-block runs that sub-block on qmm, the whole chunk on the fast kernels, in bf16 and FP8 alike)."""

    from test_patches import _prefill_state, _same

    el = _engine(ckpt, lean_block=64, rows=256, rows_max=256)
    ef = _engine(ckpt, rows=256, rows_max=256)
    assert el.e.lean is not None and el.e.fp8_prefill
    prompt = list(np.random.default_rng(837 + n).integers(0, 1000, size=n))
    assert _same(_prefill_state(ef, prompt), _prefill_state(el, prompt))


@gpu
def test_fp8_quality_close_to_bf16_fast(e8, e16):
    """Quality bound: the same 300-token prompt through FP8 and bf16 fast prefill (synthetic random weights, every
    matmul in fp8). The last chunk's final-normed rows, their logits and the KDA states stay close; most rows keep
    their argmax. The real model's check is MMLU (bench) with fp8_prefill 0 vs 1."""

    from tensorfold.families.glm5_next.cuda import qmm
    from tensorfold.families.glm5_next.cuda.decode import prefill

    prompt = list(np.random.default_rng(838).integers(0, 1000, size=300))
    got = {}
    for name, eng in (("fp8", e8), ("bf16", e16)):
        eng.cache = []
        prefill(eng.e, prompt, None, mtp=True)
        rows = eng.e.buf.fnormed[:len(prompt) - 256].clone()
        got[name] = (rows, qmm.matmul(rows, eng.w.head).float(), eng.e.st.rec[eng.e.st.cur[0]].clone())
        eng.cache = []
    (h8, l8, s8), (h6, l6, s6) = got["fp8"], got["bf16"]
    same = (l8.argmax(-1) == l6.argmax(-1)).float().mean().item()
    print(f"\n  FP8 vs bf16 fast prefill: hidden {_rel(h8, h6):.3e}, logits {_rel(l8, l6):.3e}, "
          f"KDA state {_rel(s8, s6):.3e}, argmax kept {same:.3f}")
    assert _rel(h8, h6) < 8e-2 and _rel(l8, l6) < 1.2e-1 and _rel(s8, s6) < 1.2e-1
    assert same >= 0.6, same


@pytest.fixture(scope="module")
def long_ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter
    from test_patches import _index_heads_32

    path = tmp_path_factory.mktemp("glm_fp8_long")
    _checkpoint(path / "model", exl3=True)
    _index_heads_32(path / "model")
    _drafter(path / "dflash2")
    return path


@gpu
@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_fp8_long_context_latent(long_ckpt, every_matmul_fp8, sampling):
    """3,000 tokens in FP8 fast chunks of 256 on the latent cache (tensor-core absorb / expand, 32-query tiles, dense
    and sparse): drafted == serial, and a follow-up resumed from the snapshot at 2,816 equals a fresh prefill."""

    s = _sampling(sampling)
    e = _engine(long_ckpt, rows=256, rows_max=256, context=4096, latent_kv=True)
    assert e.e.fp8_prefill
    prompt = list(np.random.default_rng(839).integers(0, 1000, size=3000))
    serial = _cold(e, prompt, s)
    for policy in (None, "2", "f3"):
        e.cache = []
        drafted, _ = _gen(e, prompt, s, policy=policy)
        assert drafted == serial, policy
    assert len(e.cache[-1].ids) == 2816 and e.cache[-1].grid == 257
    after = prompt + serial + [3, 4]
    cold = _cold(e, after, s, tokens=16)
    for policy in ("f3", "2"):
        e.cache = []
        _gen(e, prompt, s, policy=policy)
        warm, stats = _gen(e, after, s, tokens=16, policy=policy)
        assert stats["cached"] == 2816 and warm == cold, policy


# -- timing (GPU, -s) --------------------------------------------------------------------------------------------------
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
@pytest.mark.parametrize("m", [1024, 2048, 8192])
def test_fp8_timing(m, monkeypatch):
    from tensorfold.families.glm5_next.cuda import fast_qmm, qmm

    print(f"\n  M={m}: shape, bf16 fast kernel ms (TFLOP/s) -> fp8 ms incl. row quantization (TFLOP/s), speedup")
    for n, k in REAL:
        q = _q4(n, k)
        x = _x(m, k, 1)
        xs = qmm.group_sums(x)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=DEV)
        t16 = _time(lambda: fast_qmm.matmul_prefill(x, q, xs, out=out, exact=False))
        t8 = _time(lambda: fast_qmm.matmul_fp8(x, q, xs, out=out))
        tq = _time(lambda: fast_qmm.fp8_rows(x))
        f = fast_qmm.flops(m, n, k) / 1e9
        print(f"  {n}x{k}: {t16:.3f} ({f / t16:.0f}) -> {t8:.3f} ({f / t8:.0f}; quantize {tq:.3f}), {t16 / t8:.2f}x")
    if m == 1024:
        n, k = 12576, 4096
        q, x = _q4(n, k), _x(m, k, 2)
        for enc in ("cvt", "bits"):
            monkeypatch.setattr(fast_qmm, "FP8_WENC", enc)
            for t in ("128,128,8,3", "128,128,4,3", "128,64,4,3", "64,128,4,3", "128,256,8,3", "256,128,8,3",
                      "128,128,8,4", "128,128,8,2"):
                monkeypatch.setenv("GLM53_TF_FP8_TILE", t)
                try:
                    ms = _time(lambda: fast_qmm.matmul_fp8(x, q))
                    print(f"  tile {t} {enc}: {ms:.3f} ms")
                except Exception as exc:          # noqa: BLE001 - e.g. out of shared memory: report, go on
                    print(f"  tile {t} {enc}: {type(exc).__name__}")


@gpu
def test_latent_tc_timing():
    from tensorfold.families.glm5_next.cuda import latent, qmm

    H, DQ, L, DV = _latent_shapes()
    m = 1024
    kk = qmm.quantize4((torch.randn(H * DQ, L, device=DEV) * 0.05))
    kv = qmm.quantize4((torch.randn(H * DV, L, device=DEV) * 0.05))
    q = torch.randn(m, H, DQ, device=DEV).bfloat16()
    u = torch.randn(m, H, L, device=DEV)
    oa = torch.empty((m, H, L), dtype=torch.bfloat16, device=DEV)
    oe = torch.empty((m, H * DV), dtype=torch.bfloat16, device=DEV)
    ta0 = _time(lambda: latent.absorb(q, kk, oa))
    ta1 = _time(lambda: latent.absorb_tc(q, kk, oa))
    te0 = _time(lambda: latent.expand(u, kv, oe))
    te1 = _time(lambda: latent.expand_tc(u, kv, oe))
    print(f"\n  1024 rows, one layer: absorb {ta0:.3f} -> {ta1:.3f} ms, expand {te0:.3f} -> {te1:.3f} ms")
