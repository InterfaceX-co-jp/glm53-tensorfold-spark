"""patches/0520: the PTX-level evidence for ``hc_fused.cu`` (GLM53_TF_HC_CUDA), WITHOUT a GPU.

The reference: glue._hc_post -> _hc_partial -> _hc_finish compiled for sm_121 by the Triton on the path (the image's
3.7.1: asserted; another version: reported), at the production constexprs (D 4096, 4 streams, 16 blocks, SUB 128,
20 Sinkhorn iterations, 2 ranks; 4 / 4 / 8 warps; fn bf16).

- TTGIR: the layouts the specification's reduction trees come from (``tests/hc_fused_emu.py``).
- PTX: the instruction inventory (div.full / sqrt.approx.ftz / ex2.approx, mixed-precision bf16 ops, packed f32x2).
- **The Triton kernels' PTX, run in the interpreter, equals the specification** bit for bit (every output, seven input
  kinds) -- the finish kernel through ``ptxas_view`` (Triton's non-rounding mul / add may be contracted by ptxas, and
  are). Control: the finish kernel's plain PTX semantics differ, so the view matters.
- **ptxas_view == SASS**: Triton's own cubin (its ptxas) disassembled: FFMA / FADD / MUFU.RCP / EX2 / SQRT counts are
  the view's (the finish kernel's Sinkhorn loop counted as often as SASS unrolls it).
- ``hc_fused.cu`` (nvcc for sm_121 when available): no spills, registers / shared memory printed; every fp32 operation
  in its PTX is round-to-nearest (``.rn``: nothing for ptxas to contract) or one of the unit instructions
  (rcp / sqrt / ex2 .approx.ftz, one MUFU each in its SASS); **its PTX, run in the interpreter, equals the
  specification** for 1-16 rows, every row-group size, random step-CTA orders, seven input kinds, and leaves the
  tickets at zero. Mutations of the source (one reduction or rounding detail each) are caught.

    NVCC=.../nvcc PYTHONPATH=<patched tree>/src pytest -q -s tests/test_hc_fused_compile.py      (~2-4 minutes)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

if os.environ.get("TRITON_INTERPRET") == "1":
    pytest.skip("compiles kernels: run without TRITON_INTERPRET", allow_module_level=True)

sys.path.insert(0, os.path.dirname(__file__))
import hc_fused_emu as E  # noqa: E402

D = 4096
W = 4 * D


def _triton():
    triton = pytest.importorskip("triton")
    pytest.importorskip("torch")
    return triton


def _fit() -> bool:
    return _triton().__version__.startswith("3.7.")


@lru_cache(maxsize=1)
def _compiled():
    triton = _triton()
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    try:
        from tensorfold.families.glm5_next.cuda import glue
    except ImportError:
        pytest.skip("needs the patched tree on PYTHONPATH")
    tgt = GPUTarget("cuda", 121, 32)

    def comp(fn, sig, cst, warps):
        sig = dict(sig)
        for k in cst:
            sig[k] = "constexpr"
        sig = {a: sig[a] for a in fn.arg_names}
        attrs = {(fn.arg_names.index(a),): [["tt.divisibility", 16]] for a, t in sig.items() if t.startswith("*")}
        return triton.compile(ASTSource(fn=fn, signature=sig, constexprs=cst, attrs=attrs), target=tgt,
                              options={"num_warps": warps})

    return {
        "post": comp(glue._hc_post, {"X": "*bf16", "XOUT": "*bf16", "G": "*fp32", "POST": "*fp32", "COMB": "*fp32",
                                     "RS": "i32"}, dict(D=D, S=4, WORLD=2, BLOCK=1024), 4),
        "partial": comp(glue._hc_partial, {"X": "*bf16", "FN": "*bf16", "PART": "*fp32"},
                        dict(WIDE=W, NB=16, SUB=128), 4),
        "finish": comp(glue._hc_finish, {"X": "*bf16", "PART": "*fp32", "BASE": "*fp32", "SCALE": "*fp32",
                                         "NW": "*bf16", "OUT": "*bf16", "XS": "*fp32", "POST": "*fp32",
                                         "COMB": "*fp32", "eps_norm": "fp32", "hc_eps": "fp32"},
                       dict(D=D, S=4, NB=16, ITERS=20, BLOCK=D), 8),
    }


def _kernels():
    c = _compiled()
    kp = E.parse(c["post"].asm["ptx"])
    kq = E.parse(c["partial"].asm["ptx"])
    kf = E.parse(c["finish"].asm["ptx"])
    return kp, kq, kf


def _report_or_assert(ok: bool, msg: str):
    if _fit():
        assert ok, msg
    elif not ok:
        print(f"[Triton {_triton().__version__}, not the image's 3.7.x] {msg}")


# -- the Triton reference ---------------------------------------------------------------------------------------------
LAYOUTS = {
    "partial": ["sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]",
                "sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]"],
    "finish": ["sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]",
               "sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]",
               "sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [2, 4, 1], order = [2, 1, 0]",
               "sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]",
               "sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]",
               "sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]"],
    "post": ["sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]"],
}


def test_triton_layouts():
    """The blocked layouts whose in-thread chains, lane butterflies and warp combinations the specification spells out:
    _hc_partial's [32, 128] product tile (8 columns a thread, 16 lanes along a row, 4 warps along rows) and 128-vector
    of squares (one a thread); _hc_finish's 32-vectors (a lane each, replicated by the 8 warps), [4, 4] Sinkhorn tile
    (j on lanes 0-1, i on lanes 2-3), 4096-row (8 columns a thread, two repetitions) and [64, 64] group sums."""

    c = _compiled()
    for name, want in LAYOUTS.items():
        ttgir = c[name].asm["ttgir"]
        for lay in want:
            _report_or_assert(lay in ttgir, f"{name}: layout {lay} not found")


def test_triton_ptx_inventory():
    c = _compiled()
    ptx = {k: v.asm["ptx"] for k, v in c.items()}

    def n(k, pat):
        return len(re.findall(pat, ptx[k]))

    got = {
        "post fma.rn.f32": n("post", r"\bfma\.rn\.f32\b"), "post mul.f32": n("post", r"\bmul\.f32\b"),
        "post add.f32": n("post", r"\badd\.f32\b"),
        "partial fma.rn.f32.bf16": n("partial", r"\bfma\.rn\.f32\.bf16\b"),
        "partial shfl": n("partial", r"shfl\.sync\.bfly"),
        "finish div.full.f32": n("finish", r"\bdiv\.full\.f32\b"),
        "finish sqrt.approx.ftz.f32": n("finish", r"\bsqrt\.approx\.ftz\.f32\b"),
        "finish ex2.approx.f32": n("finish", r"\bex2\.approx\.f32\b"),
        "finish mul.bf16x2": n("finish", r"\bmul\.bf16x2\b"),
        "finish add.rn.f32.bf16": n("finish", r"\badd\.rn\.f32\.bf16\b"),
        "finish max.f32": n("finish", r"\bmax\.f32\b"),
    }
    want = {"post fma.rn.f32": 128, "post mul.f32": 32, "post add.f32": 8, "partial fma.rn.f32.bf16": 1,
            "partial shfl": 23, "finish div.full.f32": 10, "finish sqrt.approx.ftz.f32": 2,
            "finish ex2.approx.f32": 3, "finish mul.bf16x2": 8, "finish add.rn.f32.bf16": 14, "finish max.f32": 2}
    print("Triton", _triton().__version__, got)
    _report_or_assert(got == want, f"PTX inventory {got} != {want}")


@pytest.mark.parametrize("kind", E.KINDS)
def test_triton_ptx_equals_spec(kind):
    """The three kernels' own PTX (finish through ptxas's view) on the interpreter == the specification."""

    kp, kq, kf = _kernels()
    kfv, stats = E.ptxas_view(kf)
    R = 2 if kind != "real" else 3
    inp = E.make_inputs(R, seed=21, kind=kind)
    got = E.run_triton(kp, kq, kfv, inp)
    bad = E.diff(got, E.ref_of(inp))
    _report_or_assert(bad == [], f"{kind}: {bad} differ from the specification (ptxas view {stats})")


def test_ptxas_view_is_needed():
    """Control: the finish kernel's plain PTX semantics (no contraction) give other bits, and the view makes the
    contractions ptxas makes (Triton 3.7.1: 11, and the two constant divisions by 2^14 / 2^12)."""

    kp, kq, kf = _kernels()
    kfv, stats = E.ptxas_view(kf)
    print("ptxas view:", stats)
    differs = []
    for kind in ("random", "real"):
        inp = E.make_inputs(2, seed=5, kind=kind)
        differs.append(E.diff(E.run_triton(kp, kq, kf, inp), E.ref_of(inp)))
    _report_or_assert(any(differs), "plain PTX semantics equal the specification: the control is vacuous")
    _report_or_assert(stats["contracted"] == 11 and stats["div_pow2"] == 2, f"ptxas view stats {stats}")


def _nvdisasm() -> str | None:
    cands = [os.environ.get("NVDISASM")]
    try:
        import triton

        cands.append(str(Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "nvdisasm"))
    except ImportError:
        pass
    nv = os.environ.get("NVCC") or shutil.which("nvcc")
    if nv:
        cands.append(str(Path(nv).parent / "nvdisasm"))
    cands.append(shutil.which("nvdisasm"))
    for c in cands:
        if c and Path(c).exists():
            return c
    return None


def _sass(cubin: bytes) -> str:
    nd = _nvdisasm()
    if nd is None:
        pytest.skip("no nvdisasm")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "k.cubin"
        p.write_bytes(cubin)
        r = subprocess.run([nd, "-c", str(p)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        return r.stdout


@pytest.mark.parametrize("name", ["post", "partial", "finish"])
def test_ptxas_view_matches_sass(name):
    """What the view says ptxas executes is what Triton's ptxas emitted: FFMA, FADD and MUFU counts (the loop body
    counted as often as the SASS unrolls it, read from the shuffle count)."""

    c = _compiled()[name]
    k = E.parse(c.asm["ptx"])
    view, stats = E.ptxas_view(k)
    out, ins, sh_out, sh_in = E.loop_census(view)
    s = E.sass_census(_sass(c.asm["cubin"]))
    unroll = (s["SHFL"] - sh_out) // sh_in if sh_in else 1
    want = {key: out[key] + unroll * ins[key] for key in ("FFMA", "FADD", "RCP", "EX2", "SQRT")}
    got = {key: s[key] for key in want}
    print(f"{name}: SASS {got}, view {want} (loop x{unroll}), view stats {stats}")
    _report_or_assert(got == want, f"{name}: SASS {got} != ptxas view {want}")


# -- hc_fused.cu ------------------------------------------------------------------------------------------------------
def _nvcc() -> str:
    for c in (os.environ.get("NVCC"), shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"):
        if c and Path(c).exists():
            return c
    pytest.skip("no nvcc (set NVCC=...)")


def _src() -> str:
    try:
        from tensorfold.families.glm5_next.cuda import __file__ as f
    except ImportError:
        pytest.skip("needs the patched tree on PYTHONPATH")
    p = Path(f).parent / "hc_fused.cu"
    if not p.exists():
        pytest.skip("needs patches/0520")
    return p.read_text()


def _device_only(src: str) -> str:
    return src[:src.index("// ---- host")]


@lru_cache(maxsize=8)
def _cuda_build(src: str):
    nvcc = _nvcc()
    with tempfile.TemporaryDirectory() as td:
        cu = Path(td) / "k.cu"
        cu.write_text(src)
        r = subprocess.run([nvcc, "-arch=sm_121", "-O3", "-cubin", "-Xptxas", "-v", "-o", str(Path(td) / "k.cubin"),
                            str(cu)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        p = subprocess.run([nvcc, "-arch=sm_121", "-O3", "-ptx", "-o", str(Path(td) / "k.ptx"), str(cu)],
                           capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-3000:]
        return r.stderr, (Path(td) / "k.ptx").read_text(), (Path(td) / "k.cubin").read_bytes()


def test_cuda_compiles_for_sm121():
    log, ptx, _ = _cuda_build(_device_only(_src()))
    regs = int(re.search(r"Used (\d+) registers", log).group(1))
    spill = sum(int(v) for v in re.findall(r"(\d+) bytes spill", log))
    smem = int(re.search(r"(\d+) bytes smem", log).group(1))
    ctas = min(65536 // (((regs + 7) // 8) * 8 * 256), 102400 // (smem + 1024), 1536 // 256)
    print(f"hc_fused_kernel sm_121: {regs} registers, {spill} B spilled, {smem} B shared, {ctas} CTAs an SM "
          f"({ctas * 48} on GB10)")
    assert spill == 0
    assert regs <= 80 and smem <= 24 * 1024
    assert ctas >= 3                          # 16 rows at 4 a group: 4 x 32 + 16 = 144 CTAs in one wave


def test_cuda_ptx_is_strict():
    """Every fp32 add / sub / mul / fma rounds to nearest (.rn): ptxas contracts none of them; the approximate units
    are the .approx.ftz forms (one MUFU each); no division instruction; the mixed-precision bf16 instructions Triton's
    kernels use appear the same number of times."""

    _, ptx, _ = _cuda_build(_device_only(_src()))
    body = "\n".join(ln.split("//")[0] for ln in ptx.splitlines())
    loose = re.findall(r"\b(?:add|sub|mul|fma|mad)\.(?:ftz\.)?(?:sat\.)?f32(?:x2)?\b", body)
    assert not loose, f"non-rounding fp32 arithmetic: {sorted(set(loose))}"
    assert not re.search(r"\bdiv\.\w*\.?f32\b|\bdiv\.full\b|\bdiv\.approx\b", body)
    assert not re.search(r"\.ftz\.f32\b", re.sub(r"(rcp|sqrt|ex2)\.approx\.ftz\.f32", "", body)), "other ftz ops"
    counts = {"rcp.approx.ftz.f32": 16, "sqrt.approx.ftz.f32": 2, "ex2.approx.ftz.f32": 3, "mul.bf16x2": 8,
              "fma.rn.f32.bf16": 64, "add.rn.f32.bf16": 14, "max.f32": 2}
    for ins, n in counts.items():
        assert len(re.findall(r"\b" + re.escape(ins) + r"\b", body)) == n, ins
    for ins in ("ld.acquire.gpu.global", "atom.global.add.u32", "membar.gl", "nanosleep", "cvt.rn.bf16.f32",
                "griddepcontrol.wait"):
        assert ins in body, ins
    # vector shared-memory accesses need their array 16-byte aligned (the interpreter does not model alignment)
    if re.search(r"\b(ld|st)\.shared\.v4", body):
        assert re.search(r"\.shared \.align 16 \.b8 \w*xsm", body), "xsm must be __align__(16)"


def test_cuda_sass_units():
    """In the SASS every unit instruction is one MUFU (plus the one MUFU.RCP of the integer division r / rg), and the
    FFMA / FMUL counts are the PTX's fma.rn / mul.rn counts (ptxas changed no rounding)."""

    _, ptx, cubin = _cuda_build(_device_only(_src()))
    s = E.sass_census(_sass(cubin))
    k = E.parse(ptx)
    p = E.fp_census(k)
    print("hc_fused SASS", s, "PTX", p)
    assert s["EX2"] == 3 and s["SQRT"] == 2
    assert s["RCP"] == p["RCP"] + s["I2F.RP"]
    assert s["FFMA"] == p["FFMA"] and s["FMUL"] == p["FMUL"]


CUDA_CASES = [(1, 4, "random", 0), (2, 4, "zeros", 0), (3, 2, "extreme", 0), (2, 1, "tiny", 0),
              (3, 4, "nonfinite", 0), (5, 4, "real", 0), (4, 2, "collapse", 0), (9, 8, "random", 0),
              (16, 4, "real", 0), (3, 4, "real", 1)]


@pytest.mark.parametrize("R,rg,kind,pdl", CUDA_CASES)
def test_cuda_ptx_equals_spec(R, rg, kind, pdl):
    """hc_fused.cu's own PTX on the interpreter (step CTAs in a random order, finishers after their group's tickets)
    == the specification, every output bit; the tickets are back at zero after the launch (pdl: the
    griddepcontrol.wait path, a no-op here)."""

    _, ptx, _ = _cuda_build(_device_only(_src()))
    k = E.parse(ptx)
    inp = E.make_inputs(R, seed=31 + R, kind=kind)
    groups = (R + min(R, rg) - 1) // min(R, rg)
    order = np.random.default_rng(R * rg).permutation(32 * groups)
    got = E.run_cuda(k, inp, rg=rg, order=order, pdl=bool(pdl))
    assert E.diff(got, E.ref_of(inp)) == []
    assert not got["tickets"].any()


MUTATIONS = {
    # the collapse's first step on the other slot parity
    "collapse parity": ("float v = (j & 1) ? __fmaf_rn(pk[0], a0, __fmul_rn(pk[1], a1)) : "
                        "__fmaf_rn(pk[1], a1, __fmul_rn(pk[0], a0));",
                        "float v = (j & 1) ? __fmaf_rn(pk[1], a1, __fmul_rn(pk[0], a0)) : "
                        "__fmaf_rn(pk[0], a0, __fmul_rn(pk[1], a1));"),
    # the dots' butterfly in the other order
    "dot butterfly": ("            v = __fadd_rn(v, shx(v, 8));\n            v = __fadd_rn(v, shx(v, 4));\n"
                      "            v = __fadd_rn(v, shx(v, 2));\n            v = __fadd_rn(v, shx(v, 1));\n"
                      "            if (l == 0)",
                      "            v = __fadd_rn(v, shx(v, 1));\n            v = __fadd_rn(v, shx(v, 2));\n"
                      "            v = __fadd_rn(v, shx(v, 4));\n            v = __fadd_rn(v, shx(v, 8));\n"
                      "            if (l == 0)"),
    # the square sums' warps combined (W0 + W1) + (W2 + W3)
    "square tree": ("__fadd_rn(__fadd_rn(wsm[lt][0], wsm[lt][2]), __fadd_rn(wsm[lt][1], wsm[lt][3]))",
                    "__fadd_rn(__fadd_rn(wsm[lt][0], wsm[lt][1]), __fadd_rn(wsm[lt][2], wsm[lt][3]))"),
    # div.full + hc_eps not fused (Triton's PTX taken literally)
    "unfused eps": ("float cm = div_full_add(ce, rsum, A.hc_eps);",
                    "float cm = __fadd_rn(div_full(ce, rsum), A.hc_eps);"),
    # an exact division instead of the unit sequence
    "exact division": ("    return __fmul_rn(p.r, p.a);\n}",
                       "    return __fdiv_rn(a, b);\n}"),
    # hc_post's first product on the other stream
    "post order": ("            float v = __fmul_rn(x1, c1);\n            v = __fmaf_rn(x0, c0, v);",
                   "            float v = __fmul_rn(x0, c0);\n            v = __fmaf_rn(x1, c1, v);"),
}


@pytest.mark.parametrize("name", list(MUTATIONS))
def test_cuda_mutations_are_caught(name):
    """Negative controls at the kernel level: each one-detail mutation of hc_fused.cu changes some output bit on
    one of three input kinds (so the PTX comparison above would see such a mistake)."""

    src = _device_only(_src())
    old, new = MUTATIONS[name]
    assert src.count(old) == 1, name
    _, ptx, _ = _cuda_build(src.replace(old, new))
    k = E.parse(ptx)
    caught = []
    for kind in ("random", "real", "collapse"):
        inp = E.make_inputs(2, seed=77, kind=kind)
        bad = E.diff(E.run_cuda(k, inp, rg=2), E.ref_of(inp))
        if bad:
            caught.append((kind, bad))
            break
    assert caught, f"mutation {name} not caught"


def test_extension_sources_compile():
    """The binding (hc_fused.cpp, g++ syntax check) and the whole hc_fused.cu, host part included (nvcc for sm_121),
    against torch's headers. A CPU-only torch lacks c10/cuda's generated cuda_cmake_macros.h (an empty stub stands in)
    and the pip toolkit lacks the cuBLAS / cuSPARSE headers ATen/cuda/CUDAContext.h pulls in (the stream header, which
    declares the one function used, stands in)."""

    pytest.importorskip("torch")
    import sysconfig

    from torch.utils.cpp_extension import include_paths

    nvcc = _nvcc()
    src = _src()
    from tensorfold.families.glm5_next.cuda import __file__ as f

    d = Path(f).parent
    cxx = shutil.which("g++") or shutil.which("c++")
    inc = include_paths()
    if cxx is None or not any((Path(p) / "torch" / "extension.h").exists() for p in inc):
        pytest.skip("no host compiler or torch headers")
    pyinc = sysconfig.get_paths()["include"]
    cuinc = str(Path(nvcc).parent.parent / "include")
    with tempfile.TemporaryDirectory() as td:
        stub = Path(td) / "stub" / "c10" / "cuda" / "impl"
        stub.mkdir(parents=True)
        (stub / "cuda_cmake_macros.h").write_text("#pragma once\n")
        incs = ["-I", str(Path(td) / "stub")] + sum([["-I", p] for p in inc], []) + ["-I", pyinc]
        r = subprocess.run([cxx, *incs, "-I", cuinc, "-std=c++20", "-fsyntax-only", "-DTORCH_EXTENSION_NAME=hc_test",
                            "-DTORCH_API_INCLUDE_EXTENSION_H", str(d / "hc_fused.cpp")], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        full = src.replace("#include <ATen/cuda/CUDAContext.h>", "#include <c10/cuda/CUDAStream.h>\n"
                           "namespace at { namespace cuda { using c10::cuda::getCurrentCUDAStream; } }")
        cu = Path(td) / "k.cu"
        cu.write_text(full)
        r = subprocess.run([nvcc, "-arch=sm_121", "-O3", "-std=c++20", "-c", "-o", str(Path(td) / "k.o"), *incs,
                            "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
                            "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__",
                            "--expt-relaxed-constexpr", str(cu)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
