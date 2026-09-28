"""patches/0081 (fast prefill kernels) in Triton's CPU interpreter: numerics on small shapes, no GPU.

- ``fast_qmm.matmul_prefill`` against ``qmm.matmul`` (4-bit and BF16, with and without qmm's K split, strided rows):
  equal bit for bit in the interpreter (both run the same per-group arithmetic in the same order);
- ``fast_kda.kda_prefill_chunked`` against the serial chain's arithmetic (``fast_kda.serial_reference``, fp32 row by
  row, the ``kda.cu`` recipe) with fp32 dots (the algorithm's own error) and with tf32-rounded dot operands (what
  the GPU default does), for slow, fast and mixed decays, aligned and unaligned ``pos``; determinism; a prompt split
  at a multiple of 64 gives the bits of one call; ``save_replay`` fills the chain's replay scratch.

Run: TRITON_INTERPRET=1 PYTHONPATH=<patched TensorFold>/src pytest -q tests/test_fastk_interpreter.py
(the module sets TRITON_INTERPRET itself when it is imported first). ~1-2 minutes on a laptop CPU.
"""

from __future__ import annotations

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

fast_kda = pytest.importorskip("tensorfold.families.glm5_next.cuda.fast_kda")
from tensorfold.families.glm5_next.cuda import fast_qmm, kda, qmm  # noqa: E402

if os.environ.get("TRITON_INTERPRET") != "1" or type(fast_kda._kda_prep).__name__ != "InterpretedFunction":
    pytest.skip("needs Triton's interpreter (TRITON_INTERPRET=1 before triton is imported)", allow_module_level=True)
if torch.cuda.is_available():
    # On a GPU host (the image) the interpreter's results are not trustworthy (qmm's kernels misread their inputs:
    # errors of 1e7 in F2), and tests/cuda/test_fastk_patches.py checks the same things on the hardware.
    pytest.skip("CPU-interpreter checks: run where no GPU is visible", allow_module_level=True)

H = 2
C = 3 * H * 128


# -- tf32 operand rounding in the interpreter (the GPU's default for fast_kda's big products) -------------------
@pytest.fixture
def tf32_dots(monkeypatch):
    from triton.runtime import interpreter as interp
    import triton.language as tl

    orig = interp.InterpreterBuilder.create_dot

    def rnd(x):
        u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
        u = (u + 0xFFF + ((u >> 13) & 1)) & 0xFFFFE000
        return u.astype(np.uint32).view(np.float32)

    def create_dot(self, a, b, d, input_precision, max_num_imprecise_acc):
        name = str(input_precision).upper()
        if "TF32" in name and "X3" not in name and a.dtype.scalar == tl.float32:
            a = interp.TensorHandle(rnd(a.data), a.dtype.scalar)
            b = interp.TensorHandle(rnd(b.data), b.dtype.scalar)
        return orig(self, a, b, d, input_precision, max_num_imprecise_acc)

    monkeypatch.setattr(interp.InterpreterBuilder, "create_dot", create_dot)


# -- fast_qmm ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n,k", [(200, 256), (128, 2048), (256, 128)])
@pytest.mark.parametrize("m", [70, 130])
def test_fast_qmm_q4_equals_qmm(n, k, m):
    g = torch.Generator().manual_seed(n + k + m)
    q = qmm.quantize4(torch.randn(n, k, generator=g))
    x = torch.randn(m, k, generator=g).bfloat16()
    for f32 in (True, False):
        want = qmm.matmul(x, q, f32=f32)
        got = fast_qmm.matmul_prefill(x, q, f32=f32)
        assert torch.equal(got, want), (n, k, m, f32, qmm.split_k(n, k))
        again = fast_qmm.matmul_prefill(x, q, f32=f32)
        assert torch.equal(got, again)
    # exact=False: one accumulator; same values to fp32 rounding
    loose = fast_qmm.matmul_prefill(x, q, f32=True, exact=False)
    ref = x.float() @ qmm.dequantize_q4(q).T
    assert (loose - ref).abs().max() <= 1e-4 * ref.abs().max() + 1e-4


def test_fast_qmm_strided_rows_and_out():
    g = torch.Generator().manual_seed(7)
    q = qmm.quantize4(torch.randn(128, 1024, generator=g))
    wide = torch.randn(80, 1100, generator=g).bfloat16()
    x = wide[:, 30:30 + 1024]
    xs = qmm.group_sums(x)
    out = torch.empty((80, 128), dtype=torch.float32)
    got = fast_qmm.matmul_prefill(x, q, xs, out=out, f32=True, part=None)
    assert got is out
    assert torch.equal(got, qmm.matmul(x, q, xs, f32=True))


@pytest.mark.parametrize("n,k", [(200, 256), (128, 512)])
def test_fast_qmm_b16_equals_qmm(n, k):
    g = torch.Generator().manual_seed(n * k)
    b = qmm.make_b16(torch.randn(n, k, generator=g))
    x = torch.randn(70, k, generator=g).bfloat16()
    for f32 in (True, False):
        assert torch.equal(fast_qmm.matmul_prefill(x, b, f32=f32), qmm.matmul(x, b, f32=f32)), qmm.b16_split_k(n, k)


# -- fast_kda ----------------------------------------------------------------------------------------------------
def _inputs(rows: int, mode: str, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    b_off = C + 256
    p = torch.randn(rows, b_off + H + 3, generator=g)
    a = torch.randn(rows, H * 128, generator=g)
    if mode == "slow":            # decay ~ 1 - 2e-4 a step, beta ~ 1: long memory, the solve does real work
        a = a * 0.5 - 9.0
        p[:, b_off:b_off + H] += 4.0
    elif mode == "fast":          # decay ~ e^-5 a step: exercises the exp ranges of the 16-row blocks
        a = a * 0.5 + 12.0
    elif mode == "mixed":
        a = torch.where(torch.rand(rows, H * 128, generator=g) < 0.5, a * 0.5 - 9.0, a * 0.5 + 12.0)
    return dict(p=p.bfloat16(), b_off=b_off, a=a.bfloat16(), g=torch.randn(rows, H * 128, generator=g).bfloat16(),
                conv_state=torch.randn(3, C, generator=g).bfloat16(),
                conv_w=(torch.randn(C, 4, generator=g) * 0.5).bfloat16(),
                state_in=torch.randn(H, 128, 128, generator=g) * 0.1,
                a_log=torch.randn(H, generator=g) * 0.2, dt_bias=torch.randn(H * 128, generator=g) * 0.2,
                norm_w=(1 + 0.1 * torch.randn(128, generator=g)).bfloat16(), eps=1e-6, lower=-5.0)


def _args(d, rows):
    return (d["p"], d["b_off"], d["a"], d["g"], d["conv_state"], d["conv_w"], d["state_in"], d["a_log"],
            d["dt_bias"], d["norm_w"], d["eps"], d["lower"], rows)


def _fast(d, rows, pos, precision, **kw):
    so = torch.empty_like(d["state_in"])
    o = fast_kda.kda_prefill_chunked(*_args(d, rows), None, so, pos=pos, precision=precision, **kw)
    return o, so


@pytest.mark.parametrize("mode", ["slow", "fast", "mixed"])
@pytest.mark.parametrize("rows,pos", [(150, 0), (150, 37), (1, 63), (64, 64)])
def test_fast_kda_fp32_matches_serial(mode, rows, pos):
    d = _inputs(rows, mode)
    ref_o, ref_s = fast_kda.serial_reference(*_args(d, rows))
    o, s = _fast(d, rows, pos, "ieee")
    assert torch.isfinite(o.float()).all() and torch.isfinite(s).all()
    assert (s - ref_s).abs().max() <= 1e-4 * max(1.0, ref_s.abs().max().item())
    diff = (o.float() - ref_o.float()).abs()
    assert (diff <= 1e-2 + 4e-3 * ref_o.float().abs()).all(), diff.max()          # one bf16 step at most


@pytest.mark.parametrize("mode", ["slow", "mixed"])
def test_fast_kda_tf32_error_bound(mode, tf32_dots):
    rows, pos = 200, 17
    d = _inputs(rows, mode, seed=3)
    ref_o, ref_s = fast_kda.serial_reference(*_args(d, rows))
    o, s = _fast(d, rows, pos, "tf32")
    rel = ((s - ref_s).norm() / ref_s.norm()).item()
    assert rel < 2e-3, rel
    diff = (o.float() - ref_o.float()).abs()
    assert diff.mean() < 2e-3, diff.mean()
    assert (diff <= 0.03 + 0.02 * ref_o.float().abs()).all(), diff.max()


def test_fast_kda_deterministic_and_split_invariant():
    rows = 192
    d = _inputs(rows, "mixed", seed=5)
    o1, s1 = _fast(d, rows, 128, "tf32")
    o2, s2 = _fast(d, rows, 128, "tf32")
    assert torch.equal(o1, o2) and torch.equal(s1, s2)
    # the same prompt in two calls cut at a multiple of 64 (rows 0..127, then 128..191): the bits of one call
    cut = 128
    first = dict(d)
    o_a, s_a = _fast(first, cut, 128, "tf32")
    second = dict(d)
    second["p"], second["a"], second["g"] = d["p"][cut:], d["a"][cut:], d["g"][cut:]
    second["conv_state"] = d["p"][cut - 3:cut, :C].contiguous()
    second["state_in"] = s_a
    o_b, s_b = _fast(second, rows - cut, 128 + cut, "tf32")
    assert torch.equal(torch.cat([o_a, o_b]), o1)
    assert torch.equal(s_b, s1)


def test_fast_kda_save_replay_and_in_place_state():
    rows = 70
    d = _inputs(rows, "mixed", seed=9)
    scratch = kda.KDAScratch(rows, H, "cpu")
    so = torch.empty_like(d["state_in"])
    o = fast_kda.kda_prefill_chunked(*_args(d, rows), scratch, so, pos=5, save_replay=True, precision="ieee")
    assert o.data_ptr() == scratch.out.data_ptr()
    # replaying the saved rows with the serial update from state_in gives the chunked state (fp32 tolerance)
    s = d["state_in"].clone()
    for r in range(rows):
        s = s * scratch.g[r][:, None, :]
        kv = (s * scratch.k[r][:, None, :]).sum(-1)
        s = s + ((scratch.v[r].float() - kv) * scratch.b[r][:, None])[:, :, None] * scratch.k[r][:, None, :]
    assert (s - so).abs().max() < 1e-4
    # state_out may be state_in itself
    inplace = d["state_in"].clone()
    fast_kda.kda_prefill_chunked(*_args(dict(d, state_in=inplace), rows), None, inplace, pos=5, precision="ieee")
    assert torch.equal(inplace, so)


def test_fast_kda_supported_lower():
    """GLM-5.3's lower bound is -5; past 80/15 the wrapper hands the rows to the serial chain (a CUDA extension)."""

    assert fast_kda.supported(-5.0) and not fast_kda.supported(-8.0)


def test_fast_qmm_small_m_delegates(monkeypatch):
    calls = []
    real = qmm.matmul
    monkeypatch.setattr(qmm, "matmul", lambda *a, **k: calls.append(1) or real(*a, **k))
    q = qmm.quantize4(torch.randn(128, 256))
    x = torch.randn(3, 256).bfloat16()
    assert torch.equal(fast_qmm.matmul_prefill(x, q), real(x, q)) and calls
