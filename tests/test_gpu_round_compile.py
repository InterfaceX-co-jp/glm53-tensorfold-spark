"""patches/0450: the GPU round's kernels compiled for sm_121 (GB10) without a GPU.

Triton (``triton.compile`` for target cuda 121, the production specializations):

- the sampler (``gpusample._choose_kernel``, every block size and with / without probabilities) and the libm entry
  (``_libm_kernel``) compile with ``enable_fp_fusion=False``; every float64 add / sub / mul / div in their PTX carries
  an explicit ``.rn`` (ptxas never contracts those) and the only fused operations are the ``fma.rn.f64`` the port
  writes: 15 + 8 per ``_log`` (the near-1 and the table paths, both computed) and 8 per ``_exp`` evaluation, times the
  elements a thread holds;
- the resident glue (``gpuround._stage_kernel``, ``_accept_kernel``, ``_finish_kernel``, ``_backlog_kernel``,
  ``_conv_shift_dev``, ``_mtp_next_kernel``, ``_f_depth_kernel``) and DFlash2's device chain (``dflash2._chain_kernel``)
  compile.

CUDA (``kda.cu``, when ``nvcc`` can target sm_121: GLM53_TF_NVCC or ``nvcc`` on the PATH; the device code only, the ATen
glue needs the image's headers): ``chain_slots_kernel``'s float instructions are exactly ``chain_kernel``'s plus
``replay_layers_kernel``'s (per opcode), ``replay_slots_kernel``'s are ``replay_layers_kernel``'s, and both new kernels
fit 64 registers a thread (1,024-thread blocks).

Run without TRITON_INTERPRET: PYTHONPATH=<patched TensorFold>/src pytest -q tests/test_gpu_round_compile.py (~2 min)
"""

from __future__ import annotations

import collections
import os
import re
import shutil
import subprocess
import tempfile

import pytest

if os.environ.get("TRITON_INTERPRET") == "1":
    pytest.skip("compiles kernels: run without TRITON_INTERPRET", allow_module_level=True)

triton = pytest.importorskip("triton")
gs = pytest.importorskip("tensorfold.families.glm5_next.cuda.gpusample")
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

from tensorfold.families.glm5_next.cuda import dflash2, gpuround as gr  # noqa: E402

TGT = GPUTarget("cuda", 121, 32)


def _compile(fn, sig: dict, cst: dict, warps: int = 4, fusion: bool = False):
    sig = dict(sig)
    for k in cst:
        sig[k] = "constexpr"
    attrs = {(fn.arg_names.index(n),): [["tt.divisibility", 16]] for n, t in sig.items() if t.startswith("*")}
    try:
        return triton.compile(ASTSource(fn=fn, signature=sig, constexprs=cst, attrs=attrs), target=TGT,
                              options={"num_warps": warps, "enable_fp_fusion": fusion})
    except Exception as exc:  # noqa: BLE001
        if "ptxas" in str(exc).lower() or "not found" in str(exc).lower():
            pytest.skip(f"cannot compile for sm_121 here: {exc}")
        raise


def _f64_ops(ptx: str) -> collections.Counter:
    return collections.Counter(re.findall(r"\b((?:add|sub|mul|div|fma|mad)(?:\.[a-z0-9]+)*\.f64)\b", ptx))


CHOOSE = {"G": "*fp32", "GI": "*i32", "P": "i64", "INT": "*i64", "FLT": "*fp64", "TOK": "*i32", "PROB": "*fp64",
          "SCR": "*fp64", "CONST": "*fp64", "LOGT": "*fp64", "EXPT": "*i64"}


@pytest.mark.parametrize("block", [16, 32, 64, 128, 256])
@pytest.mark.parametrize("want", [False, True])
def test_choose_kernel_float64_is_as_written(block, want):
    k = _compile(gs._choose_kernel, CHOOSE, {"WORLD": 2, "BLOCK": block, "CH": min(32, block), "WANT": want})
    ops = _f64_ops(k.asm["ptx"])
    bad = {op: n for op, n in ops.items() if not (op.endswith(".rn.f64") and op.split(".")[0] in
                                                  ("add", "sub", "mul", "div", "fma"))}
    assert not bad, bad                     # no contractible float64 op, no mad
    assert ops["fma.rn.f64"] > 0


def test_libm_fma_counts_are_the_ports():
    sig = {"X": "*fp64", "Y": "*fp64", "n": "i32", "CONST": "*fp64", "LOGT": "*fp64", "EXPT": "*i64"}
    per_thread = 256 // (4 * 32)            # BLOCK 256 over 4 warps: 2 elements a thread
    want = {0: 23, 1: 8, 2: 46}             # log: 15 (near 1) + 8 (table); exp: 8; gumbel: two logs
    for which, fmas in want.items():
        k = _compile(gs._libm_kernel, sig, {"WHICH": which, "BLOCK": 256})
        ops = _f64_ops(k.asm["ptx"])
        # Triton 3.7.1 (the image's) recomputes one fma with its multiplicands swapped instead of reusing it (the same
        # value: an fma rounds once whatever the operand order) and lowers a negation as 0 - x (a zero's sign only;
        # the port never negates a zero that matters: log(+-0) = -inf); 3.8 emits exactly the port's
        assert fmas * per_thread <= ops["fma.rn.f64"] <= fmas * per_thread + per_thread, (which, ops)
        assert set(op for op in ops if not op.endswith(".rn.f64")) == set(), ops


def test_resident_glue_compiles():
    NI, NH, NF, NP, MAXD = gr.NI, gr.NH, gr.NF, gr.NP, gr.MAXD
    _compile(gr._stage_kernel, {"I": "*i32", "H": "*i32", "DRAFT": "*i32", "ROWSLOT": "*i32", "ROWIDX": "*i32",
                                "IDS": "*i32", "SRC": "*i64", "INT": "*i64"},
             {"NI_": NI, "NH_": NH, "MAXD_": MAXD}, warps=1)
    _compile(gr._accept_kernel, {"I": "*i32", "H": "*i32", "F": "*fp64", "DRAFT": "*i32", "TOK": "*i32", "EOS": "*i32",
                                 "n_eos": "i32", "KEEPV": "*i32", "MTOK": "*i32", "REC": "*i32", "SLOTS": "*i32"},
             {"BACKLOG_": 32, "NI_": NI, "NH_": NH, "NF_": NF, "MAXD_": MAXD, "REC_W": 16 + 64 + MAXD, "MAXR": 64},
             warps=1)
    _compile(gr._finish_kernel, {"I": "*i32", "NEXT": "*i32", "NPROB": "*fp64", "DRAFT": "*i32", "DPROB": "*fp64",
                                 "REC": "*i32", "RECF": "*fp64", "SLOTS": "*i32"},
             {"NI_": NI, "MAXD_": MAXD, "REC_W": 16 + 64 + MAXD, "RECF_W": MAXD, "MAXR": 64}, warps=1)
    _compile(gr._backlog_kernel, {"SRC": "*bf16", "src_row": "i32", "DST": "*bf16", "dst_slot": "i32",
                                  "dst_row": "i32", "I": "*i32", "H": "*i32", "SLOTS": "*i32", "width": "i32",
                                  "col": "i32", "backlog": "i32"},
             {"NI_": NI, "NH_": NH, "WHICH": 0, "BLOCK": 1024})
    _compile(gr._conv_shift_dev, {"CONV": "*bf16", "PROJ": "*bf16", "I": "*i32", "slot": "i32", "conv_layer": "i32",
                                  "proj_layer": "i32", "proj_row": "i32"},
             {"C": 12288, "TAPS": 3, "NI_": NI, "BLOCK": 1024})
    _compile(gr._mtp_next_kernel, {"I": "*i32", "F": "*fp64", "P": "*fp64", "NEXT": "*i32", "NPROB": "*fp64",
                                   "TOKS": "*i32", "PROBS": "*fp64", "SLOTS": "*i32", "j": "i32"},
             {"NI_": NI, "NF_": NF, "NP_": NP, "MAXD_": MAXD, "QMAX": gr.Q_MAX}, warps=1)
    _compile(gr._f_depth_kernel, {"I": "*i32", "P": "*fp64", "NPROB": "*fp64", "SLOTS": "*i32"},
             {"NI_": NI, "NP_": NP, "MAXD_": MAXD, "QMAX": gr.Q_MAX}, warps=1)


def test_dflash_chain_kernel_compiles():
    sig = {"PACKED": "*fp32", "PACKED_I": "*i32", "pw": "i32", "PROJ": "*fp32", "pr": "i32", "PRED": "*fp32",
           "SUCC": "*fp32", "cb": "i32", "I": "*i32", "NEXT": "*i32", "NPROB": "*fp64", "slot": "i32", "seed": "i64",
           "sampled": "i32", "depth": "i32", "DC": "*fp64", "CONST": "*fp64", "LOGT": "*fp64", "EXPT": "*i64"}
    for K, rsel in ((8, 64), (16, 128)):
        _compile(dflash2._chain_kernel, sig, {"K": K, "WORLD": 2, "BLOCK": max(32, 2 * K), "RSEL": rsel,
                                              "RB": triton.next_power_of_2(rsel), "NI_": gr.NI, "MAXD_": gr.MAXD})


# -- CUDA: kda.cu ------------------------------------------------------------------------------------------------------
def _nvcc() -> str | None:
    p = os.environ.get("GLM53_TF_NVCC") or shutil.which("nvcc")
    return p if p and os.path.exists(p) else None


def _kda_ptx(tmp: str) -> tuple[str, str]:
    here = os.path.dirname(gr.__file__)
    src = open(os.path.join(here, "kda.cu")).read()
    body = src[src.index("namespace {"):src.index("}  // namespace") + len("}  // namespace")]
    cu = os.path.join(tmp, "kda_dev.cu")
    with open(cu, "w") as f:
        head = "#include <cuda_bf16.h>\n#include <cuda_runtime.h>\n"
        f.write(head + body.replace("namespace {", "namespace kd {", 1))
    nvcc = _nvcc()
    inc = os.path.join(os.path.dirname(os.path.dirname(nvcc)), "include")
    flags = ["-std=c++17", "-arch=sm_121", "-O3", "--fmad=false", f"-I{inc}", "-allow-unsupported-compiler"]
    ptx = os.path.join(tmp, "kda.ptx")
    r = subprocess.run([nvcc, *flags, "-ptx", "-o", ptx, cu], capture_output=True, text=True)
    if r.returncode:
        pytest.skip(f"nvcc cannot build sm_121 here: {r.stderr[-400:]}")
    r2 = subprocess.run([nvcc, *flags, "-cubin", "-Xptxas", "-v", "-o", os.path.join(tmp, "kda.cubin"), cu],
                        capture_output=True, text=True)
    return open(ptx).read(), r2.stderr


@pytest.mark.skipif(_nvcc() is None, reason="nvcc (GLM53_TF_NVCC)")
def test_kda_slot_kernels_are_the_same_float_ops():
    with tempfile.TemporaryDirectory() as tmp:
        ptx, info = _kda_ptx(tmp)
    funcs = {}
    for part in re.split(r"\.visible \.entry |\.entry ", ptx)[1:]:
        name = re.sub(r"^_ZN2kd\d+", "", part.split("(")[0])
        funcs[name] = collections.Counter(re.findall(
            r"\b((?:fma|mul|add|sub|div|rcp|sqrt|rsqrt|ex2|lg2|neg|max|min)\.[\w.]*f32)\b", part))
    get = lambda stem: next(v for k, v in funcs.items() if k.startswith(stem))           # noqa: E731
    chain, replay = get("chain_kernel"), get("replay_layers_kernel")
    assert get("chain_slots_kernel") == chain + replay
    assert get("replay_slots_kernel") == replay
    regs = {m.group(1): int(m.group(2)) for m in re.finditer(
        r"Compiling entry function '_ZN2kd\d+(\w+?)E.*?\n.*?Used (\d+) registers", info, re.S)}
    for stem in ("chain_slots_kernel", "replay_slots_kernel"):
        n = next((v for k, v in regs.items() if k.startswith(stem)), None)
        assert n is not None and n <= 64, (stem, regs)
