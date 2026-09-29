"""patches/0390: the latent absorb / expand kernels, v1 (``_absorb`` / ``_expand``) and v2 (``_absorb2`` / ``_expand2``),
compiled for sm_121 (GB10) WITHOUT a GPU, with the Triton the tree runs under (the image's 3.7.1, or 3.8), and the
compiled IR checked for the property v2's bit-identity rests on: every output element is one fp32 FMA chain over k
in order, from the accumulator the loop carries.

- v1, TTIR (after Triton's Combine pass): the group loop's one ``tt.dot`` takes the loop-carried accumulator as its
  ``c`` operand and is yielded directly: ``acc + tl.dot(x, w)`` was folded into ``tl.dot(x, w, acc)``, so there is
  no separate fp32 add between groups.
- v2, TTIR: the 64 / BK dots of a group form one chain: the first takes the loop-carried accumulator, each next one
  the previous dot's result, the last is yielded.
- both, TTGIR: every dot's operands are ``dot_op`` encodings of a ``blocked`` parent (Triton's FMA lowering,
  FMADotUtility.cpp: acc = fmuladd(a_k, b_k, acc), k = 0 .. K - 1, per element) -- never ``mma`` / tf32.
- both, PTX: the only fp32 arithmetic is ``fma.rn.f32`` / ``fma.rn.f32x2`` (the dots and the dequantization
  q * s + b) plus conversions: no ``add.f32`` / ``mul.f32`` (no unfused step anywhere), no ``mma``; the fp32 -> bf16
  store conversion is ``cvt.rn.bf16.f32``; FMAs a thread = accumulator elements x K + dequantization.
- v2 fits: no spills, <= 128 registers for 8-warp programs at prefill tiles (2 programs an SM), shared memory within
  the 99 KB of a GB10 block (ptxas from Triton's own toolchain, when present; asserted under the image's Triton
  3.7.x only -- 3.8 allocates a little differently: 146 registers for expand's 64-row tile, 8 B of spill for
  absorb's 128-row one).

Not the interpreter: run in its own process without TRITON_INTERPRET.
    PYTHONPATH=<patched tree>/src pytest -q tests/test_mla_expand_compile.py      (~1-2 minutes)
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile

import pytest

if os.environ.get("TRITON_INTERPRET") == "1":
    pytest.skip("compiles kernels: run without TRITON_INTERPRET", allow_module_level=True)

triton = pytest.importorskip("triton")
torch = pytest.importorskip("torch")
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

from tensorfold.families.glm5_next.cuda import latent  # noqa: E402

if not hasattr(latent, "_expand2"):
    pytest.skip("needs patches/0390", allow_module_level=True)

TGT = GPUTarget("cuda", 121, 32)
# the register / spill fit is the image's compiler's (3.7.1); under another Triton it is reported, not asserted
FIT = triton.__version__.startswith("3.7.")
H, D, L = 32, 256, 512
N = H * D


def _compile(kind: str, v2: bool, q4: bool, bm: int = 16, bk: int = 16, warps: int = 4):
    wt = "*i32" if q4 else "*bf16"
    if kind == "expand":
        sig = {"U": "*fp32", "W": wt, "S": "*bf16", "B": "*bf16", "OUT": "*bf16", "M": "i32"}
        cst = {"H": H, "DV": D, "L": L, "N": N, "IS_Q4": q4, "BMR": bm}
        fn = latent._expand
        if v2:
            fn = latent._expand2
            cst.update(BN=latent.V2_BN, BK=bk)
    else:
        sig = {"Q": "*bf16", "W": wt, "S": "*bf16", "B": "*bf16", "QA": "*bf16", "M": "i32"}
        cst = {"H": H, "DQ": D, "L": L, "N": N, "IS_Q4": q4, "BMR": bm}
        fn = latent._absorb
        if v2:
            fn = latent._absorb2
            cst.update(BK=bk)
    for k in cst:
        sig[k] = "constexpr"
    return triton.compile(ASTSource(fn=fn, signature=sig, constexprs=cst), target=TGT,
                          options={"num_warps": warps, "num_stages": 1 if v2 else 3})


def _ptxas(ptx: str):
    """(registers, spill bytes) from Triton's bundled ptxas, or None when it is missing."""

    d = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin")
    exe = next((os.path.join(d, x) for x in ("ptxas-blackwell", "ptxas") if os.path.exists(os.path.join(d, x))), None)
    if exe is None:
        return None
    arch = re.search(r"^\.target\s+(\S+)", ptx, re.M).group(1)
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "k.ptx")
        open(p, "w").write(ptx)
        r = subprocess.run([exe, f"-arch={arch}", "-v", p, "-o", os.path.join(t, "k.cubin")], capture_output=True,
                           text=True)
    if r.returncode:
        return None
    regs = int(re.search(r"Used (\d+) registers", r.stderr).group(1))
    spill = int(re.search(r"(\d+) bytes spill stores", r.stderr).group(1))
    return regs, spill


def _dot_chain(ttir: str) -> list[tuple[str, str]]:
    """(c operand, result) of every tt.dot, in program order."""

    return re.findall(r"(%[\w.]+) = tt\.dot %[\w.]+, %[\w.]+, (%[\w.]+)", ttir)


def _loop_iter_arg(ttir: str) -> str:
    m = re.search(r"scf\.for .*iter_args\((%[\w.]+) = ", ttir)
    assert m, "no loop-carried accumulator"
    return m.group(1)


def _yielded(ttir: str) -> str:
    return re.search(r"scf\.yield (%[\w.]+)", ttir).group(1)


def _float_ops(ptx: str) -> dict[str, int]:
    ops = re.findall(r"^\s*(?:@%p\d+\s+)?((?:fma|add|sub|mul|div|mma|cvt|neg|abs|min|max)[\w.]*)", ptx, re.M)
    out: dict[str, int] = {}
    for o in ops:
        out[o] = out.get(o, 0) + 1
    return out


@pytest.mark.parametrize("q4", [True, False], ids=["q4", "b16"])
@pytest.mark.parametrize("kind", ["expand", "absorb"])
def test_v1_is_one_fma_chain(kind, q4):
    k = _compile(kind, False, q4)
    ttir, ttgir, ptx = k.asm["ttir"], k.asm["ttgir"], k.asm["ptx"]
    dots = _dot_chain(ttir)
    assert len(dots) == 1
    res, c = dots[0]
    assert c == _loop_iter_arg(ttir), "the Combine fold did not happen: acc + dot is a separate fp32 add"
    assert _yielded(ttir) == res
    assert "arith.addf" not in ttir.split("scf.for")[1].split("scf.yield")[0].split("tt.dot")[1], \
        "an fp32 add after the dot"
    _fma_path(ttgir, ptx)


@pytest.mark.parametrize("q4", [True, False], ids=["q4", "b16"])
@pytest.mark.parametrize("kind", ["expand", "absorb"])
@pytest.mark.parametrize("R", [1, 64, 512, 8192])
def test_v2_is_one_fma_chain(kind, q4, R):
    bm, bk, warps = latent.v2_tiles(R, kind)
    k = _compile(kind, True, q4, bm, bk, warps)
    ttir, ttgir, ptx = k.asm["ttir"], k.asm["ttgir"], k.asm["ptx"]
    dots = _dot_chain(ttir)
    assert len(dots) == 64 // bk
    prev = _loop_iter_arg(ttir)
    for res, c in dots:
        assert c == prev, "a dot does not continue the previous one's accumulator"
        prev = res
    assert _yielded(ttir) == prev
    ops = _fma_path(ttgir, ptx)
    # FMAs a thread: (bm x 64 accumulators / threads) x 64 k a group, + the dequantization of the weight slabs
    per_thread = bm * 64 // (32 * warps) * 64
    fmas = ops.get("fma.rn.f32", 0) + 2 * ops.get("fma.rn.f32x2", 0)
    assert per_thread <= fmas <= per_thread + (64 * 64 // (32 * warps) + 64 if q4 else 0), (fmas, per_thread)
    fit = _ptxas(ptx)
    print(f"{kind} R={R} tile {bm}x{bk} w{warps}: registers / spill bytes {fit}, shared {k.metadata.shared} B")
    if fit is not None and FIT:
        regs, spill = fit
        assert spill == 0, f"{kind} {bm}x{bk} w{warps} spills {spill} B"
        if R >= 512:
            assert regs * 32 * warps * 2 <= 65536, f"{kind}: {regs} registers: fewer than 2 programs an SM"
    assert k.metadata.shared <= 99 * 1024


def _fma_path(ttgir: str, ptx: str) -> dict[str, int]:
    for enc in re.findall(r"tt\.dot [^\n]*", ttgir):
        assert "#ttg.dot_op<{opIdx = 0, parent = #blocked" in enc and "mma" not in enc, enc
        assert "inputPrecision = tf32" not in enc
    ops = _float_ops(ptx)
    bad = {o: n for o, n in ops.items() if o.startswith(("add.f32", "add.rn.f32", "sub.f32", "mul.f32", "mul.rn.f32",
                                                          "div", "mma", "add.ftz", "mul.ftz", "fma.rz", "fma.rm",
                                                          "fma.rp", "fma.ftz", "add.f32x2", "mul.f32x2"))}
    assert not bad, bad
    assert ops.get("cvt.rn.bf16.f32", 0) + ops.get("cvt.rn.bf16x2.f32", 0) > 0
    assert not any(o.startswith("cvt.rz.bf16") or o.startswith("cvt.rn.relu") for o in ops)
    return ops


def test_v2_small_tiles_fit():
    """Decode / verify / MTP windows (16-row programs): no spills either."""

    for kind in ("expand", "absorb"):
        bm, bk, warps = latent.v2_tiles(8, kind)
        fit = _ptxas(_compile(kind, True, True, bm, bk, warps).asm["ptx"])
        if fit is not None and FIT:
            assert fit[1] == 0, (kind, fit)
