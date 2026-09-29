"""patches/0470 (GLM53_TF_NONEXPERT=q8 / GLM53_TF_NONEXPERT_MAP) in Triton's CPU interpreter, no GPU: the 8-bit format
(``qmm.Q8``, ``qmm.quantize8``) and every kernel that reads it.

- ``quantize8``: q in -127..127, the scale the smallest bf16 >= amax / 127 (nothing clips, every error <= s / 2), an
  all-zero group scale 1, ``dequantize_q8`` exact; groups of 128 (64 when K is not a multiple of 128).
- decode (``qmm.matmul`` -> ``_q8mm``, split-K slices + ``_reduce``) and prefill (``fast_qmm.matmul_prefill`` ->
  ``_fq8``, exact and one-accumulator) against a float64 reference of the dequantized weights; ``exact`` gives
  qmm's bits; a row's bits never depend on the call's row count (1-40 rows, every row bucket) or the tile table.
- the latent MLA absorb / expand (v1 and 0390's v2) with an 8-bit kv_b against the expanded reference, v1 == v2.
- ``fastpf.attach_kda_bf16``'s generic dequantization and ``overlap._mat``.

The interpreter's own casts and dots are not the GPU's, so the fixture models them (as tests/test_b12x_interpreter.py
does): int -> bf16 converts the value (the interpreter reinterprets the bits), fp32 -> bf16 rounds to nearest even
(``cvt.rn.bf16.f32``), and a ``tl.dot`` is the EXACT sum of its products rounded once to fp32, plus the accumulator
operand. The exact sum is what makes bit-level claims testable here: a bf16 x int8 product is exact in float64 and
so is a sum of 128 of them over these magnitudes, so the modelled dot does not depend on the tile's shape. On the GPU
the per-element mma.sync k16 chain is not that sum, but it is one fixed instruction sequence for every tile and row
count, which tests/cuda/test_q8_patches.py checks bit for bit on hardware.

Run: TRITON_INTERPRET=1 PYTHONPATH=<patched TensorFold>/src pytest -q tests/test_q8_interpreter.py
"""

from __future__ import annotations

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
qmm = pytest.importorskip("tensorfold.families.glm5_next.cuda.qmm")
fast_qmm = pytest.importorskip("tensorfold.families.glm5_next.cuda.fast_qmm")
latent = pytest.importorskip("tensorfold.families.glm5_next.cuda.latent")

if not hasattr(qmm, "Q8"):
    pytest.skip("needs patches/0470", allow_module_level=True)

INTERP = os.environ.get("TRITON_INTERPRET") == "1" and type(qmm._q8mm).__name__ == "InterpretedFunction"
pytestmark = pytest.mark.skipif(not INTERP or torch.cuda.is_available(),
                                reason="Triton's CPU interpreter, no GPU (tests/cuda/test_q8_patches.py on GPUs)")


def _bf16_to_f32(u16):
    return (np.asarray(u16).astype(np.uint32) << 16).view(np.float32)


def _f32_to_bf16(f32):
    t = torch.from_numpy(np.ascontiguousarray(f32, dtype=np.float32)).to(torch.bfloat16)
    return t.view(torch.int16).numpy().view(np.uint16)


@pytest.fixture(autouse=True)
def _gpu_model(monkeypatch):
    from triton.runtime import interpreter as interp
    import triton.language as tl

    cast = interp.InterpreterBuilder.cast_impl

    def cast_impl(self, src, dst_type):
        s, d = src.dtype.scalar, dst_type.scalar
        if s == tl.float32 and d == tl.bfloat16:
            return interp.TensorHandle(_f32_to_bf16(src.data), d)
        if s == tl.bfloat16 and d == tl.float32:
            return interp.TensorHandle(_bf16_to_f32(src.data), d)
        if s.is_int() and d == tl.bfloat16:
            return interp.TensorHandle(_f32_to_bf16(src.data.astype(np.float32)), d)
        return cast(self, src, dst_type)

    def wide(h):
        if h.dtype.scalar == tl.bfloat16:
            return _bf16_to_f32(h.data).astype(np.float64)
        return np.asarray(h.data, dtype=np.float64)

    def create_dot(self, a, b, d, input_precision, max_num_imprecise_acc):
        out = (wide(a) @ wide(b)).astype(np.float32) + np.asarray(d.data, dtype=np.float32)
        return interp.TensorHandle(out.astype(np.float32), tl.float32)

    B = interp.InterpreterBuilder
    monkeypatch.setattr(B, "cast_impl", cast_impl)
    for name in ("create_fp_trunc", "create_fp_ext", "create_si_to_fp", "create_ui_to_fp"):
        monkeypatch.setattr(B, name, lambda self, s, t: cast_impl(self, s, t))
    monkeypatch.setattr(B, "create_dot", create_dot)
    yield


def _w(n, k, seed=0, scale=0.02):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, k, generator=g) * scale
    w[:, : k // 8] *= 6.0                          # some groups with a larger range (eh_proj's embedding half)
    return w.to(torch.bfloat16)


def _x(m, k, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, k, generator=g).to(torch.bfloat16)


# -- the format ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("k", [128, 384, 192])
def test_quantize8_format(k):
    w = _w(70, k)
    w[3] = 0                                        # an all-zero row: every group scale 1, q 0
    w[5, :64] = 0
    q = qmm.quantize8(w)
    gs = 128 if k % 128 == 0 else 64
    assert q.gs == gs and q.weight.dtype == torch.int8 and q.scales.dtype == torch.bfloat16
    assert q.weight.shape == (70, k) and q.scales.shape == (k // gs, 70)
    assert int(q.weight.abs().max()) <= 127 and int(q.weight.min()) >= -127
    wf = w.float().view(70, k // gs, gs)
    amax = wf.abs().amax(-1)
    s = q.scales.t().float()
    nz = amax > 0
    assert (s[nz] * 127 >= amax[nz]).all(), "a value clips"
    # the smallest such bf16: one step down is below amax / 127
    down = (q.scales.t().view(torch.int16) - 1).view(torch.bfloat16).float()
    assert (down[nz] * 127 < amax[nz]).all()
    assert (s[~nz] == 1).all() and (q.weight[3] == 0).all()
    dq = qmm.dequantize_q8(q)
    err = (dq - w.float()).abs().view(70, k // gs, gs)
    assert (err <= s[..., None] / 2 + 1e-12).all()
    # exact dequantization: s * q in fp32 equals the float64 product
    assert torch.equal(dq.double(), q.weight.double() * s.double().repeat_interleave(gs, -1))
    assert torch.equal(qmm.dequantize_any(q), dq)


def test_quantize8_deterministic_and_chunked():
    w = _w(300, 256, seed=3)
    a, b = qmm.quantize8(w), qmm.quantize8(w, chunk=64)
    assert torch.equal(a.weight, b.weight) and torch.equal(a.scales, b.scales)


def test_quantize8_error_vs_q4():
    """8 bits: ~13x smaller relative error than q4mse on Gaussian weights (0.65% vs 8.6% on GLM-5.3-Flash's own)."""

    w = _w(256, 512, seed=4)
    e8 = (qmm.dequantize_q8(qmm.quantize8(w)) - w.float()).norm() / w.float().norm()
    e4 = (qmm.dequantize_q4(qmm.quantize4(w, mse=True)) - w.float()).norm() / w.float().norm()
    assert e8 < 0.012 and e4 / e8 > 8, (float(e8), float(e4))


def test_split_matches_q4_buffers():
    """The K slices a Q8 matmul uses never need more partial rows than a Q4 one of the same shape (the forward's
    ``b.sk`` buffer is sized for Q4), for every per-rank GLM-5.3-Flash shape."""

    shapes = [(12576, 4096), (4096, 128), (4096, 4096), (2048, 4096), (8192, 1536), (8192, 512), (4096, 8192),
              (12288, 4096), (4096, 6144), (2048, 4096), (4096, 1024), (160, 4096), (4096, 1536), (77440, 4096)]
    for n, k in shapes:
        sk8, sk4 = qmm.q8_split_k(n, k), qmm.split_k(n, k)
        assert sk8 <= sk4 and (k // qmm.q8_group(k)) % sk8 == 0, (n, k, sk8, sk4)


# -- matmuls -------------------------------------------------------------------------------------------------------
SHAPES = [(96, 512), (64, 384), (130, 256), (64, 192)]


def _ref(x, q):
    return x.double() @ qmm.dequantize_q8(q).double().T


@pytest.mark.parametrize("n,k", SHAPES)
def test_decode_matches_reference(n, k, monkeypatch):
    q = qmm.quantize8(_w(n, k))
    for sk in (1, 2, 4):
        if (k // q.gs) % sk:
            continue
        monkeypatch.setattr(qmm, "q8_split_k", lambda n_, k_, sk=sk: sk)
        for m in (1, 3, 16, 17, 40):
            x = _x(m, k, seed=m)
            y = qmm.matmul(x, q, f32=True)
            ref = _ref(x, q)
            assert ((y.double() - ref).abs() <= 1e-5 * ref.abs().max() + 1e-6).all(), (sk, m)
            yb = qmm.matmul(x, q)
            assert yb.dtype == torch.bfloat16 and torch.equal(yb, y.to(torch.bfloat16))


@pytest.mark.parametrize("n,k", SHAPES)
def test_rows_independent(n, k, monkeypatch):
    """A row's bits never depend on the other rows or on how many there are (drafted == serial)."""

    monkeypatch.setattr(qmm, "q8_split_k", lambda n_, k_: 2 if (k_ // qmm.q8_group(k_)) % 2 == 0 else 1)
    q = qmm.quantize8(_w(n, k, seed=2))
    x = _x(40, k, seed=9)
    full = qmm.matmul(x, q, f32=True)
    for m in (1, 2, 7, 16, 17, 33):
        part = qmm.matmul(x[:m].contiguous(), q, f32=True)
        assert torch.equal(part, full[:m]), m
    alone = qmm.matmul(x[23:24].contiguous(), q, f32=True)
    assert torch.equal(alone, full[23:24])
    # strided rows (a view of a wider buffer) give the same bits
    wide = torch.zeros(40, k + 64, dtype=torch.bfloat16)
    wide[:, :k] = x
    assert torch.equal(qmm.matmul(wide[:, :k], q, f32=True), full)


@pytest.mark.parametrize("n,k", SHAPES)
def test_prefill_exact_is_qmm_bits(n, k, monkeypatch):
    monkeypatch.setattr(qmm, "q8_split_k", lambda n_, k_: 2 if (k_ // qmm.q8_group(k_)) % 2 == 0 else 1)
    q = qmm.quantize8(_w(n, k, seed=5))
    x = _x(70, k, seed=6)
    dec = qmm.matmul(x, q, f32=True)
    for tile in ("64,64,4,3", "128,64,4,2", "32,128,4,2"):
        monkeypatch.setenv("GLM53_TF_Q8_TILE", tile)
        got = fast_qmm.matmul_prefill(x, q, f32=True, min_rows=1)
        assert torch.equal(got, dec), tile
        loose = fast_qmm.matmul_prefill(x, q, f32=True, exact=False, min_rows=1)
        ref = _ref(x, q)
        assert ((loose.double() - ref).abs() <= 1e-5 * ref.abs().max() + 1e-6).all()
        # one accumulator: row-independent too
        assert torch.equal(fast_qmm.matmul_prefill(x[5:6].contiguous(), q, f32=True, exact=False, min_rows=1),
                           loose[5:6])
    monkeypatch.delenv("GLM53_TF_Q8_TILE")
    # below MIN_ROWS the helper routes to qmm; FP8 prefill runs 8-bit weights on the bf16 fast kernel
    assert torch.equal(fast_qmm.matmul(x[:8].contiguous(), q, f32=True), dec[:8])
    fp8 = fast_qmm.matmul_fp8(x, q, f32=True, min_rows=1)
    assert torch.equal(fp8, fast_qmm.matmul_prefill(x, q, f32=True, exact=False, min_rows=1))


def test_decode_cfg_override_same_bits(monkeypatch):
    q = qmm.quantize8(_w(128, 512, seed=7))
    x = _x(9, 512)
    base = qmm.matmul(x, q, f32=True)
    for cfg in ("1,4,2", "4,8,3", "2,2,1"):
        monkeypatch.setenv("GLM53_TF_Q8_DECODE_CFG", cfg)
        assert torch.equal(qmm.matmul(x, q, f32=True), base), cfg


# -- latent MLA with an 8-bit kv_b ------------------------------------------------------------------------------------
def test_latent_absorb_expand_q8(monkeypatch):
    H, DQ, L = 2, 128, 128
    kv_k = qmm.quantize8(_w(H * DQ, L, seed=11, scale=0.05))
    kv_v = qmm.quantize8(_w(H * DQ, L, seed=12, scale=0.05))
    wk, wv = qmm.dequantize_q8(kv_k).double(), qmm.dequantize_q8(kv_v).double()
    for R in (1, 5, 17):
        q = _x(R * H, DQ, seed=R).view(R, H, DQ).contiguous()
        outs = {}
        for v2 in (False, True):
            monkeypatch.setattr(latent, "EXPAND_V2", v2)
            a = torch.empty(R, H, L, dtype=torch.bfloat16)
            latent.absorb(q, kv_k, a)
            u = torch.randn(R, H, L, generator=torch.Generator().manual_seed(R))
            e = torch.empty(R, H * DQ, dtype=torch.bfloat16)
            latent.expand(u, kv_v, e)
            outs[v2] = (a, e)
            ref_a = torch.einsum("rhi,hij->rhj", q.double(), wk.view(H, DQ, L))
            ref_e = torch.einsum("rhj,hnj->rhn", u.double(), wv.view(H, DQ, L)).reshape(R, H * DQ)
            assert ((a.double() - ref_a).abs() <= 0.01 * ref_a.abs().max()).all()
            assert ((e.double() - ref_e).abs() <= 0.01 * ref_e.abs().max()).all()
        assert torch.equal(outs[False][0], outs[True][0]) and torch.equal(outs[False][1], outs[True][1])
    assert latent._parts(kv_k)[3] == 2 and latent._parts(kv_k)[2] == 128


# -- consumers -----------------------------------------------------------------------------------------------------
def test_consumers():
    from tensorfold.families.glm5_next.cuda import overlap

    q = qmm.quantize8(_w(64, 256))
    assert [t.data_ptr() for t in overlap._mat(q)] == [q.scales.data_ptr(), q.weight.data_ptr()]
    q4 = qmm.quantize4(_w(64, 256))
    assert len(overlap._mat(q4)) == 3
    b = qmm.make_b16(_w(64, 256))
    assert torch.equal(qmm.dequantize_any(b), b.weight.float())
