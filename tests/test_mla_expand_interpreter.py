"""patches/0390 (GLM53_TF_MLA_EXPAND=v2) in Triton's CPU interpreter: the retiled latent absorb / expand give v1's bits
for every row, whatever the launch's row count and the tile table, no GPU.

What v1 computes on the GPU (checked from the compiled IR in ``test_mla_expand_compile.py``, from Triton's source in
docs/MLA-EXPAND.md): every output element is ONE fp32 FMA chain, acc = fma(a_k, b_k, acc) for k = 0 .. K - 1 in
order, from +0.0 (the ``acc + tl.dot(x, w)`` of each 64-wide group is folded into ``tl.dot(x, w, acc)`` by Triton's
Combine pass, and an ``ieee`` fp32 dot is lowered to exactly that chain by FMADotUtility.cpp), with the weights
dequantized exactly (q * s + b) and the sum rounded once to bf16. Triton's interpreter computes ``tl.dot`` with
``np.matmul`` (no defined order) and never runs the Combine pass, and it converts fp32 to bf16 by truncation, so here:

- ``tl.dot`` is replaced by an exact model of the GPU lowering: the per-element fp32 FMA chain over k in order, from
  the dot's accumulator operand (fp32 FMA emulated exactly: the product is exact in float64, the sum is rounded to
  odd in float64, then to nearest-even fp32; exact since 53 >= 24 + 2), and fp32 -> bf16 rounds to nearest-even
  (the GPU's ``cvt.rn.bf16.f32``);
- v1 runs as Triton compiles it: its source with ``acc = acc + tl.dot(x, w, ...)`` rewritten to
  ``acc = tl.dot(x, w, acc, ...)`` (the Combine fold; nothing else changes);
- a plain numpy reference computes the same chain directly (the specification).

Checked bit for bit (int16 views of the bf16 outputs): v2 == v1 == reference, 4-bit (``qmm.Q4``) and BF16 kv_b, for
launches of 1, 7, 16, 17, 64, 513, 2,048 and 8,192 rows (the real shapes -- 32 heads, head dim 256, latent 512 -- up to
64 rows; 2 heads with head dim 64 / latent 128 past that, to keep the emulation fast); every row of a launch equals
the row launched alone; every entry of a forced tile table (16-128 rows a program, 16-64 k a dot step) gives the
same bits.

Run: TRITON_INTERPRET=1 PYTHONPATH=<patched TensorFold>/src pytest -q tests/test_mla_expand_interpreter.py
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
latent = pytest.importorskip("tensorfold.families.glm5_next.cuda.latent")
qmm = pytest.importorskip("tensorfold.families.glm5_next.cuda.qmm")

if not hasattr(latent, "_expand2"):
    pytest.skip("needs patches/0390", allow_module_level=True)

INTERP = os.environ.get("TRITON_INTERPRET") == "1" and type(latent._expand2).__name__ == "InterpretedFunction"
pytestmark = pytest.mark.skipif(not INTERP or torch.cuda.is_available(),
                                reason="Triton's CPU interpreter, no GPU (tests/cuda/test_mla_expand_patches.py on GPUs)")

from triton.runtime import interpreter as _interp  # noqa: E402


# -- exact fp32 FMA ---------------------------------------------------------------------------------------------------
def fma32(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """fp32 fma(a, b, c) rounded once (nearest-even), elementwise with broadcasting: the product of two fp32 values
    is exact in float64; the float64 sum is rounded to odd (TwoSum error term), which makes the final rounding to
    fp32 correct."""

    p = a.astype(np.float64) * b.astype(np.float64)
    c = np.broadcast_to(c.astype(np.float64), p.shape)
    s = p + c
    bb = s - p
    err = (p - (s - bb)) + (c - bb)
    even = (s.view(np.int64) & 1) == 0
    fix = (err != 0) & even & np.isfinite(s)
    if fix.any():
        s = np.where(fix, np.nextafter(s, np.where(err > 0, np.inf, -np.inf)), s)
    return s.astype(np.float32)


def test_fma32_is_exact():
    """The emulation against exact rational arithmetic (fractions) on hard cases: ties, cancellation, subnormals."""

    from fractions import Fraction

    rng = np.random.default_rng(0)
    a = rng.standard_normal(4000).astype(np.float32)
    b = rng.standard_normal(4000).astype(np.float32)
    c = (-(a.astype(np.float64) * b) + rng.standard_normal(4000) * 1e-7).astype(np.float32)   # heavy cancellation
    tiny = np.float32(1e-38)
    a = np.concatenate([a, np.float32([1 + 2 ** -23, 1 + 2 ** -12, tiny, 3.0, -0.0])])
    b = np.concatenate([b, np.float32([1 - 2 ** -23, 1 + 2 ** -12, tiny, 1 / 3, 1.0])])
    c = np.concatenate([c, np.float32([-1.0, 2 ** -24, 0.0, -1.0, 0.0])])
    got = fma32(a, b, c)

    def exact(x, y, z):
        v = Fraction(float(x)) * Fraction(float(y)) + Fraction(float(z))
        f = np.float32(float(v))                     # float(v) rounds once to float64 ...
        lo, hi = sorted((f, np.nextafter(f, np.float32(np.inf if Fraction(float(f)) < v else -np.inf))))
        dl, dh = abs(Fraction(float(lo)) - v), abs(Fraction(float(hi)) - v)   # ... so pick the nearest fp32 exactly
        if dl != dh:
            return lo if dl < dh else hi
        return lo if (lo.view(np.int32) & 1) == 0 else hi

    want = np.array([exact(x, y, z) for x, y, z in zip(a, b, c)], dtype=np.float32)
    nz = want != 0
    assert np.array_equal(got[nz].view(np.int32), want[nz].view(np.int32))
    assert np.array_equal(got[~nz], want[~nz])


def chain(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """a [M, K] @ b [K, N] + c as the GPU's ``ieee`` fp32 dot: acc = c; acc = fma(a[:, k], b[k, :], acc), k ascending."""

    acc = c.astype(np.float32)
    for k in range(a.shape[1]):
        acc = fma32(a[:, k:k + 1], b[k:k + 1, :], acc)
    return acc


@pytest.fixture(autouse=True)
def exact_dot(monkeypatch):
    """Triton's interpreter with ``tl.dot`` = the GPU's FMA chain (fp32 x fp32 -> fp32 dots; others unchanged)."""

    orig = _interp.InterpreterBuilder.create_dot

    def create_dot(self, a, b, d, input_precision, max_num_imprecise_acc):
        if a.data.dtype == np.float32 and b.data.dtype == np.float32 and d.data.dtype == np.float32 \
                and a.data.ndim == 2:
            return _interp.TensorHandle(chain(a.data, b.data, d.data), d.dtype.scalar)
        return orig(self, a, b, d, input_precision, max_num_imprecise_acc)

    monkeypatch.setattr(_interp.InterpreterBuilder, "create_dot", create_dot)

    # the GPU's fp32 -> bf16 is cvt.rn.bf16.f32 (nearest-even); the interpreter truncates (``_convert_float`` with no
    # rounding mode): model the GPU
    orig_cast = _interp.InterpreterBuilder.cast_impl

    def cast_impl(self, src, dst_type):
        if src.dtype.scalar == triton.language.float32 and dst_type.scalar == triton.language.bfloat16:
            data = torch.from_numpy(np.ascontiguousarray(src.data)).to(torch.bfloat16).view(torch.int16).numpy()
            return _interp.TensorHandle(data.view(np.uint16), dst_type.scalar)
        return orig_cast(self, src, dst_type)

    monkeypatch.setattr(_interp.InterpreterBuilder, "cast_impl", cast_impl)
    monkeypatch.setattr(_interp.InterpreterBuilder, "create_fp_trunc", cast_impl)
    yield


# -- v1 as Triton compiles it (the Combine fold), in the interpreter --------------------------------------------------
def _v1_folded():
    """v1's ``_absorb`` / ``_expand`` with ``acc = acc + tl.dot(x, w, ...)`` rewritten to ``acc = tl.dot(x, w, acc,
    ...)``: what Triton's CombineDotAddPattern makes of them before lowering (asserted on the compiled IR in
    ``test_mla_expand_compile.py``)."""

    import tempfile

    src = ["import triton\nimport triton.language as tl\n"
           "from tensorfold.families.glm5_next.cuda.latent import _wtile\n"]
    for fn in (latent._absorb, latent._expand):
        s = inspect.getsource(fn.fn)
        for old, new in (("acc = acc + tl.dot(x.to(tl.float32), wt, input_precision=\"ieee\")",
                          "acc = tl.dot(x.to(tl.float32), wt, acc, input_precision=\"ieee\")"),
                         ("acc = acc + tl.dot(x, tl.trans(wt), input_precision=\"ieee\")",
                          "acc = tl.dot(x, tl.trans(wt), acc, input_precision=\"ieee\")")):
            s = s.replace(old, new)
        assert "acc = acc +" not in s and s.count("tl.dot(") == 1, s
        src.append(s)
    d = tempfile.mkdtemp(prefix="v1fold")
    path = os.path.join(d, "v1fold.py")
    with open(path, "w") as f:
        f.write("\n\n".join(src))
    spec = importlib.util.spec_from_file_location("v1fold", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v1fold"] = mod
    spec.loader.exec_module(mod)
    return mod


V1 = None


def v1_absorb(q, kv_k, out):
    global V1
    V1 = V1 or _v1_folded()
    R, H, DQ = q.shape
    L = out.shape[2]
    W, S, B, is_q4 = latent._parts(kv_k)
    V1._absorb[(triton.cdiv(R, latent.BM), H, L // 64)](q, W, S, B, out, R, H=H, DQ=DQ, L=L, N=kv_k.n, IS_Q4=is_q4,
                                                        BMR=latent.BM, num_warps=4)
    return out


def v1_expand(u, kv_v, out):
    global V1
    V1 = V1 or _v1_folded()
    R, H, L = u.shape
    W, S, B, is_q4 = latent._parts(kv_v)
    DV = kv_v.n // H
    V1._expand[(triton.cdiv(R, latent.BM), kv_v.n // 64)](u, W, S, B, out, R, H=H, DV=DV, L=L, N=kv_v.n,
                                                          IS_Q4=is_q4, BMR=latent.BM, num_warps=4)
    return out


# -- the specification, in numpy --------------------------------------------------------------------------------------
def dequant(kv) -> np.ndarray:
    """kv_b rows [N, K] as fp32, exactly as ``_wtile``: s * q + b (q * s is exact: q < 16, s bf16) or the bf16 value."""

    if isinstance(kv, qmm.Q4):
        words = qmm.untile_words(kv.weight, kv.n).numpy().astype(np.int64)            # [N, K / 8]
        q = ((words[:, :, None] >> (4 * np.arange(8))[None, None, :]) & 0xF).reshape(kv.n, kv.k).astype(np.float32)
        s = kv.scales.t().float().numpy().repeat(64, axis=1)
        b = kv.biases.t().float().numpy().repeat(64, axis=1)
        return (q * s + b).astype(np.float32)
    return kv.weight.float().numpy()


def bf16(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).to(torch.bfloat16)                # round to nearest-even


def ref_absorb(q: torch.Tensor, kv_k) -> torch.Tensor:
    R, H, DQ = q.shape
    w = dequant(kv_k)                                                                   # [H DQ, L]
    qf = q.float().numpy()
    return torch.stack([bf16(chain(qf[:, h, :], w[h * DQ:(h + 1) * DQ], np.zeros((R, w.shape[1]), np.float32)))
                        for h in range(H)], 1)


def ref_expand(u: torch.Tensor, kv_v) -> torch.Tensor:
    R, H, L = u.shape
    w = dequant(kv_v)                                                                   # [H DV, L]
    DV = kv_v.n // H
    uf = u.numpy()
    return torch.cat([bf16(chain(uf[:, h, :], w[h * DV:(h + 1) * DV].T, np.zeros((R, DV), np.float32)))
                      for h in range(H)], 1)


# -- inputs -----------------------------------------------------------------------------------------------------------
REAL = dict(H=32, D=256, L=512)
SMALL = dict(H=2, D=64, L=128)


def _kv(kind: str, n: int, k: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn((n, k), generator=g) * 0.05).to(torch.bfloat16)
    return qmm.quantize4(w, mse=(seed % 2 == 0)) if kind == "q4" else qmm.make_b16(w)


def _inputs(kind: str, R: int, H: int, D: int, L: int, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn((R, H, D), generator=g)).to(torch.bfloat16).contiguous()
    u = (torch.randn((R, H, L), generator=g) * 0.3).float().contiguous()
    u[:, :, ::7] *= 1e-3                                         # spread magnitudes: more rounding sensitivity
    return q, u, _kv(kind, H * D, L, seed + 10), _kv(kind, H * D, L, seed + 20)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _run(fn, *args):
    out = args[-1]
    fn(*args)
    return out


def _v2(monkeypatch, on=True):
    monkeypatch.setattr(latent, "EXPAND_V2", on)


ROWS = [1, 7, 16, 17, 64, 513, 2048, 8192]


@pytest.mark.parametrize("kind", ["q4", "b16"])
@pytest.mark.parametrize("R", ROWS)
def test_v2_equals_v1_and_reference(kind, R, monkeypatch):
    shp = REAL if R <= 64 else SMALL
    H, D, L = shp["H"], shp["D"], shp["L"]
    q, u, kv_k, kv_v = _inputs(kind, R, H, D, L, seed=R)
    _v2(monkeypatch)
    a2 = _run(latent.absorb, q, kv_k, torch.empty((R, H, L), dtype=torch.bfloat16))
    e2 = _run(latent.expand, u, kv_v, torch.empty((R, H * D), dtype=torch.bfloat16))
    a1 = _run(v1_absorb, q, kv_k, torch.empty((R, H, L), dtype=torch.bfloat16))
    e1 = _run(v1_expand, u, kv_v, torch.empty((R, H * D), dtype=torch.bfloat16))
    assert torch.equal(_bits(a2), _bits(a1)), "absorb v2 != v1"
    assert torch.equal(_bits(e2), _bits(e1)), "expand v2 != v1"
    if R <= 513:                                            # the numpy chain over everything: minutes past that
        assert torch.equal(_bits(a2), _bits(ref_absorb(q, kv_k))), "absorb != reference"
        assert torch.equal(_bits(e2), _bits(ref_expand(u, kv_v))), "expand != reference"
    else:                                                   # a sample of rows
        idx = torch.tensor([0, 1, 63, 64, 127, 128, R // 2, R - 2, R - 1])
        assert torch.equal(_bits(a2[idx]), _bits(ref_absorb(q[idx].contiguous(), kv_k)))
        assert torch.equal(_bits(e2[idx]), _bits(ref_expand(u[idx].contiguous(), kv_v)))


@pytest.mark.parametrize("kind", ["q4", "b16"])
def test_v2_rows_independent_of_launch(kind, monkeypatch):
    """Each row of a 300-row launch == the row launched alone, in a 7-row window, in a 64-row window (every tile
    table entry is met: 16-, 32-, 64- and 128-row programs)."""

    H, D, L = SMALL["H"], SMALL["D"], SMALL["L"]
    R = 300
    q, u, kv_k, kv_v = _inputs(kind, R, H, D, L, seed=5)
    _v2(monkeypatch)
    a = _run(latent.absorb, q, kv_k, torch.empty((R, H, L), dtype=torch.bfloat16))
    e = _run(latent.expand, u, kv_v, torch.empty((R, H * D), dtype=torch.bfloat16))
    for lo, n in ((0, 1), (1, 1), (150, 1), (299, 1), (40, 7), (100, 17), (200, 33), (236, 64)):
        a_w = _run(latent.absorb, q[lo:lo + n].contiguous(), kv_k, torch.empty((n, H, L), dtype=torch.bfloat16))
        e_w = _run(latent.expand, u[lo:lo + n].contiguous(), kv_v, torch.empty((n, H * D), dtype=torch.bfloat16))
        assert torch.equal(_bits(a_w), _bits(a[lo:lo + n])), (lo, n)
        assert torch.equal(_bits(e_w), _bits(e[lo:lo + n])), (lo, n)


@pytest.mark.parametrize("kind", ["q4", "b16"])
def test_v2_tile_table_does_not_change_bits(kind, monkeypatch):
    """Any (rows a program, k a dot step, warps) gives the same bits: the tile table is a speed knob only."""

    H, D, L = SMALL["H"], SMALL["D"], SMALL["L"]
    R = 200
    q, u, kv_k, kv_v = _inputs(kind, R, H, D, L, seed=9)
    _v2(monkeypatch)
    base = None
    for bm in (16, 32, 64, 128):
        for bk in (16, 32, 64):
            monkeypatch.setattr(latent, "V2_TILES", {"expand": ((1 << 62, bm, bk, 4),),
                                                     "absorb": ((1 << 62, bm, bk, 8),)})
            a = _run(latent.absorb, q, kv_k, torch.empty((R, H, L), dtype=torch.bfloat16))
            e = _run(latent.expand, u, kv_v, torch.empty((R, H * D), dtype=torch.bfloat16))
            if base is None:
                base = (_bits(a).clone(), _bits(e).clone())
            assert torch.equal(_bits(a), base[0]) and torch.equal(_bits(e), base[1]), (bm, bk)


def test_expand_bn_does_not_change_bits(monkeypatch):
    """expand's output rows a program (V2_BN 32 / 64 / 128 / 256 within a 256-wide head): same bits."""

    H, D, L = 2, 256, 128
    R = 40
    _, u, _, kv_v = _inputs("q4", R, H, D, L, seed=11)
    _v2(monkeypatch)
    outs = []
    for bn in (32, 64, 128, 256):
        monkeypatch.setattr(latent, "V2_BN", bn)
        outs.append(_bits(_run(latent.expand, u, kv_v, torch.empty((R, H * D), dtype=torch.bfloat16))).clone())
    assert all(torch.equal(o, outs[0]) for o in outs)
    assert torch.equal(outs[0], _bits(ref_expand(u, kv_v)))


def test_v1_is_the_default_and_knob_parses(monkeypatch):
    monkeypatch.delenv("GLM53_TF_MLA_EXPAND", raising=False)
    assert latent.expand_mode() == "v1"
    for v, want in (("v2", "v2"), ("V2", "v2"), ("1", "v2"), ("v1", "v1"), ("0", "v1"), ("off", "v1"), ("", "v1")):
        monkeypatch.setenv("GLM53_TF_MLA_EXPAND", v)
        assert latent.expand_mode() == want
    monkeypatch.setenv("GLM53_TF_MLA_EXPAND", "v3")
    with pytest.raises(ValueError):
        latent.expand_mode()


def test_v1_folded_differs_from_unfolded_control(monkeypatch):
    """Control: the Combine fold matters. v1 run WITHOUT the fold (each 64-wide group's chain from 0, then added)
    gives different bits from the folded chain on these inputs, so the equalities above are not vacuous."""

    H, D, L = REAL["H"], REAL["D"], REAL["L"]
    R = 16
    _, u, _, kv_v = _inputs("q4", R, H, D, L, seed=3)
    _v2(monkeypatch, False)
    unfolded = _run(latent.expand, u, kv_v, torch.empty((R, H * D), dtype=torch.bfloat16))  # v1 as written: unfolded
    folded = _run(v1_expand, u, kv_v, torch.empty((R, H * D), dtype=torch.bfloat16))
    assert not torch.equal(_bits(unfolded), _bits(folded))
    assert torch.equal(_bits(folded), _bits(ref_expand(u, kv_v)))
