"""patches/0440: the streaming decode kernels compiled for sm_121 (GB10) WITHOUT a GPU, and the compiled reference
(``qmm._qmm``) checked for the instruction sequence ``q4_stream.cu`` reproduces.

- ``exl3_stream.cu`` / ``q4_stream.cu`` (device code, every instantiation the host dispatch can launch, nvcc
  ``-arch=sm_121 -O3`` with torch's half / bf16 defines): no spills, registers within the launch bounds, dynamic +
  static shared memory within a GB10 block (101,376 B), resident CTAs an SM (printed); the PTX has the instructions
  the design names -- mma.m16n8k16 f16 (experts) / bf16 (dense), 16-byte ``cp.async.cg`` with a source size,
  ``griddepcontrol.wait`` / ``launch_dependents`` (PDL exists on sm_121), ``sub.rn.bf16x2`` -- and the dense kernel
  has no unfused fp32 multiply (every product-add is an fma, as ``_qmm``'s).
- The device helpers ``exl3_stream.cu`` copies (codebook, tile decode, mma, Hadamard transform, bf16 rounding, the
  gate/up and down epilogue bodies) are character for character exl3.cu's / exl3_dec.cu's.
- ``_qmm`` as Triton compiles it (the image's 3.7.x, or 3.8) for every per-rank decode shape and row bucket: in TTIR
  each group is ``p = tt.dot(x, q^T, zeros)``, then ``acc + p * s``, then ``+ xs * b`` (acc carried by the loop); in
  PTX the dot is m16n8k16 bf16 -> f32 and the epilogue is 2 fmas an element and group (``fma.rn.f32`` or the packed
  ``fma.rn.f32x2``) with no ``add.f32`` / ``mul.f32``: acc = fma(xs, b, fma(p, s, acc)) -- what q4_stream.cu does
  with __fmaf_rn. (Triton's MMAv2 lowering chains the k repetitions of one dot in ascending k,
  third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp ``for k < repK``, from the dot's accumulator,
  here zero.)

    NVCC=/usr/local/cuda/bin/nvcc PYTHONPATH=<patched tree>/src pytest -q -s tests/test_decode_kernels_compile.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

if os.environ.get("TRITON_INTERPRET") == "1":
    pytest.skip("compiles kernels: run without TRITON_INTERPRET", allow_module_level=True)

SMEM_MAX = 101376
SM_SMEM = 102400
REG_FILE = 65536
FLAGS = ["-arch=sm_121", "-O3", "-cubin", "-include", "cstdint", "-D__CUDA_NO_HALF_OPERATORS__",
         "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__",
         "--expt-relaxed-constexpr", "-Xptxas", "-v"]


def _src_dir() -> Path:
    try:
        from tensorfold.families.glm5_next.cuda import __file__ as f
    except ImportError:
        pytest.skip("needs the patched tree on PYTHONPATH")
    d = Path(f).parent
    if not (d / "exl3_stream.cu").exists():
        pytest.skip("needs patches/0440")
    return d


def _nvcc() -> str:
    for c in (os.environ.get("NVCC"), shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"):
        if c and Path(c).exists():
            return c
    pytest.skip("no nvcc (set NVCC=...)")


def _device_only(src: str, cut: str, inst: list[str]) -> str:
    for h in ("#include <ATen/ATen.h>\n", "#include <ATen/cuda/CUDAContext.h>\n", "#include <c10/cuda/CUDAGuard.h>\n"):
        src = src.replace(h, "")
    src = src[:src.index(cut)].replace("namespace {", "", 1)
    return src + "\n" + "\n".join(inst) + "\n"


def _compile(src: str, want_ptx: bool = True):
    nvcc = _nvcc()
    with tempfile.TemporaryDirectory() as td:
        cu = Path(td) / "k.cu"
        cu.write_text(src)
        r = subprocess.run([nvcc, *FLAGS, "-o", str(Path(td) / "k.cubin"), str(cu)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        ptx = ""
        if want_ptx:
            p = subprocess.run([nvcc, "-arch=sm_121", "-O3", "-ptx", "-include", "cstdint",
                                "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
                                "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__",
                                "--expt-relaxed-constexpr", "-o", str(Path(td) / "k.ptx"), str(cu)],
                               capture_output=True, text=True)
            assert p.returncode == 0, p.stderr[-3000:]
            ptx = (Path(td) / "k.ptx").read_text()
    stats = {}
    name = None
    for line in r.stderr.splitlines():
        m = re.search(r"Compiling entry function '(\S+)'", line)
        if m:
            name = m.group(1)
            stats[name] = {}
        m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m and name:
            stats[name]["spill"] = int(m.group(1)) + int(m.group(2))
        m = re.search(r"Used (\d+) registers.*?(?:(\d+) bytes smem)?$", line)
        if m and name:
            stats[name]["regs"] = int(m.group(1))
            s = re.search(r"(\d+) bytes smem", line)
            stats[name]["smem"] = int(s.group(1)) if s else 0
    return stats, ptx


def _ctas(regs: int, threads: int, smem: int) -> int:
    by_regs = REG_FILE // (((regs + 7) // 8) * 8 * threads)
    by_smem = SM_SMEM // (smem + 1024)
    return min(by_regs, by_smem, 48 * 32 // threads)


# -- E1 ---------------------------------------------------------------------------------------------------------------
E1_CFGS = [(4, 4), (4, 6), (2, 4), (2, 6), (2, 8), (8, 3)]


def _e1_smem(nt: int, stages: int) -> int:
    return (4 * stages * (nt * 32 + 128) + 3 * 16 * nt * 16) * 4


def test_exl3_stream_compiles_and_fits():
    d = _src_dir()
    inst = [f"template __global__ void stream_kernel<{nt},{s},{m},0>(Launch, int);" for nt, s in E1_CFGS for m in (0, 1)]
    inst += [f"template __global__ void stream_kernel<4,4,{m},{p}>(Launch, int);" for m in (0, 1) for p in (1, 2)]
    src = _device_only((d / "exl3_stream.cu").read_text(), "template <int NT, int STAGES>\nconstexpr int smem_bytes", inst)
    stats, ptx = _compile(src)
    assert len(stats) == len(inst)
    for name, st in sorted(stats.items()):
        m = re.search(r"stream_kernelILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E", name)
        nt, stages, mode, probe = map(int, m.groups())
        total = _e1_smem(nt, stages) + st["smem"]
        ctas = _ctas(st["regs"], 128, total)
        print(f"exl3_stream NT={nt} STAGES={stages} {'gate/up' if mode == 0 else 'down'} probe={probe}: "
              f"{st['regs']} registers, {st['spill']} B spilled, {total} B shared -> {ctas} CTAs an SM")
        assert st["spill"] == 0
        assert st["regs"] <= 170                            # __launch_bounds__(128, 3)
        assert total <= SMEM_MAX
        assert ctas >= 2
    assert "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32" in ptx
    assert re.search(r"cp\.async\.cg\.shared\.global \[[^]]+\], \[[^]]+\], 16;", ptx)
    assert "griddepcontrol.wait" in ptx and "griddepcontrol.launch_dependents" in ptx
    assert "cp.async.wait_group" in ptx and "prefetch.global.L2" in ptx
    _check_pdl_prologue(ptx, allowed_loads=("ld.global.cg",))


def _check_pdl_prologue(ptx: str, allowed_loads=()):
    """Every kernel: before its first griddepcontrol.wait (in program text; the PDL prologue comes first), no
    cp.async, no store, and no global load but the allowed forms (E1: the speculative ld.global.cg of the grouping,
    whose values only address L2 prefetches) -- the sfxnz kit's PDL race lesson: nothing the previous kernel wrote may
    be consumed before the wait."""

    for body in re.split(r"\.entry ", ptx)[1:]:
        name = body.split("(")[0]
        i = body.find("griddepcontrol.wait")
        assert i > 0, name
        pre = body[:i]
        assert "cp.async" not in pre, name
        assert not re.search(r"\bst\.global", pre), name
        for ld in re.findall(r"\bld\.global[.\w]*", pre):
            assert ld.startswith(allowed_loads) if allowed_loads else False, (name, ld)


def _functions(src: str, names: list[str]) -> dict[str, str]:
    """The text of each named __device__ function (signature to the matching closing brace)."""

    out = {}
    for n in names:
        m = re.search(r"__device__ __forceinline__ [^\n]*\b" + n + r"\(", src)
        assert m, n
        i = src.index("{", m.start())
        depth = 0
        for j in range(i, len(src)):
            depth += src[j] == "{"
            depth -= src[j] == "}"
            if depth == 0:
                out[n] = src[m.start():j + 1]
                break
    return out


def test_exl3_stream_helpers_are_verbatim():
    d = _src_dir()
    mine = (d / "exl3_stream.cu").read_text()
    base = (d / "exl3.cu").read_text()
    dec = (d / "exl3_dec.cu").read_text()
    names = ["mcg2", "decode_tile", "mma16816", "fwht128", "bf16r"]
    a, b = _functions(mine, names), _functions(base, names)
    for n in names:
        assert a[n] == b[n], n
    epi = ["gateup_epilogue", "down_epilogue"]
    a, b = _functions(mine, epi), _functions(dec, epi)
    for n in epi:
        assert a[n] == b[n], n
    assert "constexpr float HAD_SCALE = 0.08838834764831845f;" in mine


# -- E2 ---------------------------------------------------------------------------------------------------------------
E2_CFGS = [(1, 1, 4), (1, 1, 6), (1, 1, 8), (1, 2, 4), (2, 1, 4), (4, 1, 4)]


def _e2_smem(rt: int, gps: int, stages: int) -> int:
    return stages * (gps * 64 * 8 * 4 + 2 * gps * 64 * 2 + gps * rt * 16 * 72 * 2 + gps * rt * 16 * 4)


def test_q4_stream_compiles_and_fits():
    d = _src_dir()
    inst = [f"template __global__ void q4_kernel<{r},{g},{s}>(Args, int);" for r, g, s in E2_CFGS]
    src = _device_only((d / "q4_stream.cu").read_text(), "int sm_count()", inst)
    stats, ptx = _compile(src)
    assert len(stats) == len(inst)
    for name, st in sorted(stats.items()):
        r, g, s = map(int, re.search(r"q4_kernelILi(\d+)ELi(\d+)ELi(\d+)E", name).groups())
        total = _e2_smem(r, g, s) + st["smem"]
        ctas = _ctas(st["regs"], 128, total)
        print(f"q4_stream RT={r} GPS={g} STAGES={s}: {st['regs']} registers, {st['spill']} B spilled, {total} B shared "
              f"-> {ctas} CTAs an SM")
        assert st["spill"] == 0 and total <= SMEM_MAX and ctas >= 2
    assert "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32" in ptx
    assert "sub.rn.bf16x2" in ptx
    assert re.search(r"cp\.async\.cg\.shared\.global \[[^]]+\], \[[^]]+\], 16, %r\d+;", ptx)     # zero-fill form
    assert "cp.async.ca.shared.global" in ptx                                                   # the group sums
    assert "griddepcontrol.wait" in ptx
    _check_pdl_prologue(ptx)
    assert "fma.rn.f32" in ptx
    assert not re.search(r"\bmul(\.rn)?\.f32\b", ptx)            # no unfused product anywhere
    assert "cvt.rn.bf16.f32" in ptx


# -- the compiled reference: qmm._qmm ---------------------------------------------------------------------------------
SHAPES = [(12576, 4096), (4096, 4096), (4096, 128), (2048, 4096), (8192, 1536), (8192, 512), (4096, 8192),
          (12288, 4096), (4096, 6144), (4096, 1024), (77440, 4096)]


def _qmm_compile(n, k, bm):
    triton = pytest.importorskip("triton")
    pytest.importorskip("torch")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from tensorfold.families.glm5_next.cuda import qmm

    sk = qmm.split_k(n, k)
    c_gpi, warps, stages = qmm.SHAPE_CFG.get(f"{n}x{k}", qmm.CONFIG[bm]) if bm == 16 else qmm.CONFIG[bm]
    gpi = qmm.gpi_for((k // 64) // sk, c_gpi)
    fn = qmm._qmm
    sig = {"X": "*bf16", "XS": "*fp32", "W": "*i32", "S": "*bf16", "B": "*bf16", "OUT": "*bf16", "PART": "*fp32",
           "M": "i32", "x_stride": "i32"}
    cst = dict(N=n, K=k, SK=sk, BM=bm, BLOCK_N=64, GPI=gpi, F32=False)
    for kk in cst:
        sig[kk] = "constexpr"
    sig = {a: sig[a] for a in fn.arg_names}
    attrs = {(fn.arg_names.index(a),): [["tt.divisibility", 16]] for a, t in sig.items() if t.startswith("*")}
    c = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=cst, attrs=attrs), target=GPUTarget("cuda", 121, 32),
                       options={"num_warps": warps, "num_stages": stages})
    return c, gpi, warps


def _unfused(ptx: str) -> int:
    return len(re.findall(r"\b(add|mul|sub)(\.rn)?\.f32(x2)?\b", ptx))


@pytest.mark.parametrize("bm", [16, 32, 64])
def test_qmm_reference_sequence(bm):
    """``_qmm``'s per-group arithmetic for every per-rank shape at this row bucket. The fusion is asserted under the
    image's Triton (3.7.x); under another one the unfused slots are counted and printed, and
    ``decode_stream.qmm_reference_fused`` keeps the streaming dense kernel off (Triton 3.8.0: 16-24 unfused mul / add
    pairs a thread for 12288x4096 and 77440x4096, a different set at each row bucket)."""

    triton = pytest.importorskip("triton")
    fit = triton.__version__.startswith("3.7.")
    bad = []
    for n, k in SHAPES:
        c, gpi, warps = _qmm_compile(n, k, bm)
        ttir = c.asm["ttir"]
        fn = ttir[ttir.index("tt.func"):]
        body = fn[fn.index("scf.for"):] if "scf.for" in fn else fn        # K = 128: the loop is gone (one step)
        body = body[:body.index("scf.yield")] if "scf.yield" in body else body[:body.index("tt.return")]
        dots = re.findall(r"= tt\.dot (%[\w.]+), (%[\w.]+), (%[\w.]+)", body)
        assert len(dots) == gpi, (n, k)
        zeros = set(re.findall(r"(%[\w.]+) = arith\.constant dense<0\.000000e\+00> : tensor<\d+x64xf32>", ttir))
        assert all(d[2] in zeros for d in dots), (n, k)            # each group's dot starts from +0.0
        ops = re.findall(r"= arith\.(mulf|addf) ", body)
        assert ops == ["mulf", "addf", "mulf", "addf"] * gpi, (n, k, ops)     # acc + p * s, then + xs * b
        ptx = c.asm["ptx"]
        assert "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32" in ptx
        fmas = len(re.findall(r"\bfma\.rn\.f32\b", ptx)) + 2 * len(re.findall(r"\bfma\.rn\.f32x2\b", ptx))
        per_thread = bm * 64 // (warps * 32)
        if _unfused(ptx):
            bad.append((n, k, _unfused(ptx)))
        else:
            assert fmas == 2 * per_thread * gpi, (n, k, fmas)          # two fmas an element and group, nothing else
    print(f"Triton {triton.__version__}, BM {bm}: unfused _qmm epilogues {bad or 'none'}")
    if fit:
        assert not bad
