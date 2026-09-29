"""patches/0240 (b12x-derived fast-prefill kernels) in Triton's CPU interpreter: numerics on small shapes, no GPU.

- ``b12x_kda.kda_prefill16`` against the serial chain's arithmetic (``fast_kda.serial_reference``) with fp32 dots (the
  algorithm's own error: ~1e-7) and with tf32-rounded operands (no worse than ``fast_kda``); its float64 tile model;
  determinism, a prompt split at multiples of 64 == one call, a call's first k rows == a k-row call;
- ``b12x_mhc``: the partials against float64, the fused post + dots == hc_post then the dots (bitwise);
- ``b12x_attn``: one pass against a float64 softmax and against the chunked kernel, FP8 rows == their dequantization.

The interpreter of Triton 3.8 truncates fp32 -> bf16 casts and multiplies bf16 dot operands as their uint16 bits;
``_fix_interpreter`` makes both behave like the GPU (round to nearest even; bf16 widened to fp32 before the product).
Bitwise row-slice checks are left to the GPU tests (numpy's matmul blocking depends on the row count).

Run: TRITON_INTERPRET=1 PYTHONPATH=<patched TensorFold>/src pytest -q tests/test_b12x_interpreter.py (~1 minute).
"""

from __future__ import annotations

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

b12x_kda = pytest.importorskip("tensorfold.families.glm5_next.cuda.b12x_kda")
from tensorfold.families.glm5_next.cuda import b12x_attn, b12x_mhc, fast_kda, glue, latent  # noqa: E402

if os.environ.get("TRITON_INTERPRET") != "1" or type(b12x_kda._k16).__name__ != "InterpretedFunction":
    pytest.skip("needs Triton's interpreter (TRITON_INTERPRET=1 before triton is imported)", allow_module_level=True)
if torch.cuda.is_available():
    pytest.skip("CPU-interpreter checks: run where no GPU is visible", allow_module_level=True)

H = 2
C = 3 * H * 128


def _bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def _f32_to_bf16(f32):
    t = torch.from_numpy(np.ascontiguousarray(f32, dtype=np.float32)).to(torch.bfloat16)
    return t.view(torch.int16).numpy().view(np.uint16)


def _rnd_tf32(x):
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0xFFF + ((u >> 13) & 1)) & 0xFFFFE000
    return u.astype(np.uint32).view(np.float32)


@pytest.fixture(autouse=True)
def _fix_interpreter(monkeypatch):
    from triton.runtime import interpreter as interp
    import triton.language as tl

    cast, dot = interp.InterpreterBuilder.cast_impl, interp.InterpreterBuilder.create_dot

    def cast_impl(self, src, dst_type):
        s, d = src.dtype.scalar, dst_type.scalar
        if s == tl.float32 and d == tl.bfloat16:
            return interp.TensorHandle(_f32_to_bf16(src.data), d)
        if s == tl.bfloat16 and d == tl.float32:
            return interp.TensorHandle(_bf16_to_f32(src.data), d)
        return cast(self, src, dst_type)

    def create_dot(self, a, b, d, input_precision, max_num_imprecise_acc):
        if a.dtype.scalar == tl.bfloat16:
            a = interp.TensorHandle(_bf16_to_f32(a.data), tl.float32)
        if b.dtype.scalar == tl.bfloat16:
            b = interp.TensorHandle(_bf16_to_f32(b.data), tl.float32)
        if getattr(self, "_tf32", False) and "TF32" in str(input_precision).upper() \
                and "X3" not in str(input_precision).upper():
            a = interp.TensorHandle(_rnd_tf32(a.data), tl.float32)
            b = interp.TensorHandle(_rnd_tf32(b.data), tl.float32)
        return dot(self, a, b, d, input_precision, max_num_imprecise_acc)

    monkeypatch.setattr(interp.InterpreterBuilder, "cast_impl", cast_impl)
    monkeypatch.setattr(interp.InterpreterBuilder, "create_fp_trunc", lambda self, s, t: cast_impl(self, s, t))
    monkeypatch.setattr(interp.InterpreterBuilder, "create_fp_ext", lambda self, s, t: cast_impl(self, s, t))
    monkeypatch.setattr(interp.InterpreterBuilder, "create_dot", create_dot)
    monkeypatch.setattr(interp.InterpreterBuilder, "_tf32", False, raising=False)


@pytest.fixture
def tf32_dots(monkeypatch):
    from triton.runtime import interpreter as interp

    monkeypatch.setattr(interp.InterpreterBuilder, "_tf32", True, raising=False)


# -- KDA ---------------------------------------------------------------------------------------------------------------
def _inputs(rows: int, mode: str, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    b_off = C + 256
    p = torch.randn(rows, b_off + H + 3, generator=g)
    a = torch.randn(rows, H * 128, generator=g)
    if mode == "slow":
        a = a * 0.5 - 9.0
        p[:, b_off:b_off + H] += 4.0
    elif mode == "fast":
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


def _k16(d, rows, pos, **kw):
    so = torch.empty_like(d["state_in"])
    return b12x_kda.kda_prefill16(*_args(d, rows), None, so, pos=pos, **kw), so


@pytest.mark.parametrize("mode", ["slow", "fast", "mixed"])
@pytest.mark.parametrize("rows,pos", [(40, 0), (40, 6), (1, 15), (16, 64)])
def test_kda16_fp32_matches_serial(mode, rows, pos):
    d = _inputs(rows, mode)
    ref_o, ref_s = fast_kda.serial_reference(*_args(d, rows))
    o, s = _k16(d, rows, pos, prec="ieee")
    assert torch.isfinite(o.float()).all() and torch.isfinite(s).all()
    assert (s - ref_s).abs().max() <= 1e-5 * max(1.0, ref_s.abs().max().item())
    diff = (o.float() - ref_o.float()).abs()
    assert (diff <= 1e-2 + 4e-3 * ref_o.float().abs()).all(), diff.max()        # one bf16 step at most
    to, ts = b12x_kda.tile_reference(*_args(d, rows), pos)
    assert (s - ts).abs().max() <= 1e-5 * max(1.0, ts.abs().max().item())


@pytest.mark.parametrize("mode", ["slow", "mixed"])
def test_kda16_tf32_no_worse_than_fast_kda(mode, tf32_dots):
    rows, pos = 200, 64
    d = _inputs(rows, mode, seed=3)
    ref_o, ref_s = fast_kda.serial_reference(*_args(d, rows))
    o, s = _k16(d, rows, pos)
    so = torch.empty_like(d["state_in"])
    oo = fast_kda.kda_prefill_chunked(*_args(d, rows), None, so, pos=pos, precision="tf32")
    rel = ((s - ref_s).norm() / ref_s.norm()).item()
    rel_old = ((so - ref_s).norm() / ref_s.norm()).item()
    assert rel < max(2e-3, 1.5 * rel_old), (rel, rel_old)
    diff = (o.float() - ref_o.float()).abs()
    assert diff.mean() < max(2e-3, 1.5 * (oo.float() - ref_o.float()).abs().mean().item())


def test_kda16_deterministic_split_and_prefix():
    rows = 192
    d = _inputs(rows, "mixed", seed=5)
    o1, s1 = _k16(d, rows, 128)
    o2, s2 = _k16(d, rows, 128)
    assert torch.equal(o1, o2) and torch.equal(s1, s2)
    cut = 64
    oa, sa = _k16(d, cut, 128)
    second = dict(d)
    second.update(p=d["p"][cut:], a=d["a"][cut:], g=d["g"][cut:], conv_state=d["p"][cut - 3:cut, :C].contiguous(),
                  state_in=sa)
    ob, sb = _k16(second, rows - cut, 128 + cut)
    assert torch.equal(torch.cat([oa, ob]), o1) and torch.equal(sb, s1)


def test_kda16_prefix_rows_close():
    """A call's first k rows vs a k-row call: equal on the GPU (tested there); here within fp32 noise (numpy's matmul
    blocking depends on the row count)."""

    d = _inputs(192, "mixed", seed=6)
    o1, _ = _k16(d, 192, 0, prec="ieee")
    for k in (1, 17, 100):
        o, _ = _k16(d, k, 0, prec="ieee")
        assert (o.float() - o1[:k].float()).abs().max() <= 1e-2


# -- hyper-connections -----------------------------------------------------------------------------------------------
def test_mhc_partials_and_fused_equals_unfused():
    torch.manual_seed(0)
    D, R, NB = 1024, 70, 4
    x = (torch.randn(R, 4 * D) * 2).bfloat16()
    fn = (torch.randn(24, 4 * D) * 0.02).bfloat16()
    part = torch.zeros(R, NB, 32)
    b12x_mhc.partial(x, fn, part, NB)
    ref = b12x_mhc.partial_reference(x, fn, NB)
    scale = ref.abs().amax(dim=0, keepdim=True).clamp_min(1e-3)
    assert ((part[:, :, :25].double() - ref).abs() / scale).max() < 1e-5
    for gdt in (torch.float32, torch.bfloat16):
        g = torch.randn(2, R, D).to(gdt)
        post, comb = torch.rand(R, 4) * 2, torch.rand(R, 4, 4)
        xo1 = torch.empty_like(x)
        glue.hc_post(x, xo1, g, post, comb)
        p1 = torch.zeros(R, NB, 32)
        b12x_mhc.partial(xo1, fn, p1, NB)
        xo2, p2 = torch.empty_like(x), torch.zeros(R, NB, 32)
        assert b12x_mhc.post_partial(x, xo2, g, post, comb, fn, p2, NB)
        assert torch.equal(xo1, xo2) and torch.equal(p1[:, :, :25], p2[:, :, :25])


def test_mhc_step_width_rule():
    assert b12x_mhc.fits(4096, 16) and b12x_mhc.step(4096, 16) == b12x_mhc.BK
    assert b12x_mhc.fits(512, 16) and b12x_mhc.step(512, 16) == 32         # the synthetic test model's width
    assert not b12x_mhc.fits(4096 + 8, 16)


# -- sparse attention --------------------------------------------------------------------------------------------------
def test_attn_one_pass():
    torch.manual_seed(0)
    R, Hh, L, T, W = 5, 32, 512, 3000, 2051
    qa = (torch.randn(R, Hh, L) * 0.5).bfloat16()
    lat = torch.randn(T, L).bfloat16()
    tokens = torch.full((R, W), -1, dtype=torch.int32)
    counts = torch.tensor([2051, 0, 700, 2051, 33], dtype=torch.int32)
    for r in range(R):
        n = int(counts[r])
        tokens[r, :n] = torch.sort(torch.randperm(T)[:n]).values.int()
    scale = 256 ** -0.5
    one, chunked = torch.full((R, Hh, L), 7.0), torch.full((R, Hh, L), 7.0)
    b12x_attn.sparse_latent_one(qa, lat, tokens, counts, one, scale)
    latent.sparse_latent(qa, lat, tokens, counts, chunked, scale, None, bm=32)
    ref = b12x_attn.reference(qa, lat, tokens, counts, scale)
    m = counts > 0
    e1 = (one[m].double() - ref[m]).abs().max().item()
    e0 = (chunked[m].double() - ref[m]).abs().max().item()
    assert e1 <= 1.5 * e0 + 1e-4
    assert bool((one[1] == 7.0).all())
    lc8 = latent.quantize_rows_reference(lat)
    o8, ob = torch.full((R, Hh, L), 7.0), torch.full((R, Hh, L), 7.0)
    b12x_attn.sparse_latent_one(qa, lc8, tokens, counts, o8, scale)
    b12x_attn.sparse_latent_one(qa, latent.dequantize_rows(lc8), tokens, counts, ob, scale)
    assert torch.equal(o8, ob)
