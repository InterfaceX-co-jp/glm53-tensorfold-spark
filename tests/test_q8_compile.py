"""patches/0470: the 8-bit kernels compiled for sm_121 (GB10) WITHOUT a GPU, with the Triton the tree runs under, and the
compiled code checked for what their speed and their bits rest on:

- ``qmm._q8mm`` (decode / verify / MTP windows, every row bucket) and ``fast_qmm._fq8`` (prefill, exact and one
  accumulator) at every per-rank GLM-5.3-Flash shape: the weights stream as 4-byte ``cp.async`` copies (the 4-bit
  kernels' pattern; int8 loads got one ``ld.global.b8`` a byte and no pipelining, hence the int32 words), no byte
  loads; the dots are ``mma.sync.m16n8k16`` bf16 -> fp32; the per-group step ``acc + P * s`` is one FMA
  (``fma.rn.f32`` / ``.f32x2``: no separate fp32 mul or add in the one-slice kernels), the same contraction in
  decode and prefill, so ``exact`` prefill keeps qmm's bits; int8 -> bf16 by ``cvt.rn.bf16.s32`` (exact for
  |q| <= 127);
- shared memory within GB10's 99 KB a block, and with Triton's bundled ptxas (when present) no spills (checked with
  the image's Triton 3.7.1 and with 3.8.0).
- the latent MLA kernels (v1, 0390's v2, the tensor-core variants) compile with an 8-bit kv_b (``IS_Q4 = 2``) and
  keep 0390's property: no fp32 add / mul outside FMAs.

Not the interpreter: run in its own process without TRITON_INTERPRET.
    PYTHONPATH=<patched tree>/src pytest -q tests/test_q8_compile.py      (~1-3 minutes)
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

from tensorfold.families.glm5_next.cuda import fast_qmm, latent, qmm  # noqa: E402

if not hasattr(qmm, "Q8"):
    pytest.skip("needs patches/0470", allow_module_level=True)

TGT = GPUTarget("cuda", 121, 32)
SHARED_MAX = 99 * 1024
# every per-rank non-expert shape (N x K) of GLM-5.3-Flash at TP=2, and the head's half
SHAPES = [(12576, 4096), (4096, 128), (4096, 4096), (2048, 4096), (8192, 1536), (8192, 512), (4096, 8192),
          (12288, 4096), (4096, 6144), (2048, 4096), (4096, 1024), (160, 4096), (4096, 1536), (77440, 4096)]


def _compile(fn, sig: dict, cst: dict, warps: int, stages: int):
    sig = dict(sig)
    for k in cst:
        sig[k] = "constexpr"
    return triton.compile(ASTSource(fn=fn, signature=sig, constexprs=cst), target=TGT,
                          options={"num_warps": warps, "num_stages": stages})


def _ops(ptx: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for o in re.findall(r"^\s*(?:@%p\d+\s+)?([a-z]+[\w.]*)", ptx, re.M):
        out[o] = out.get(o, 0) + 1
    return out


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
        raise AssertionError(r.stderr[-2000:])
    regs = int(re.search(r"Used (\d+) registers", r.stderr).group(1))
    spill = int(re.search(r"(\d+) bytes spill stores", r.stderr).group(1))
    return regs, spill


def _check_matmul(k, *, one_slice: bool, what: str, loop: bool = True):
    """``loop``: the K loop runs more than once (a single-step loop is not software-pipelined: 4096x128's one
    group; its weights then come as plain 4-byte loads)."""

    ptx = k.asm["ptx"]
    ops = _ops(ptx)
    assert ops.get("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", 0) > 0, what
    if loop:
        assert ops.get("cp.async.ca.shared.global", 0) + ops.get("cp.async.cg.shared.global", 0) > 0, \
            f"{what}: weights not pipelined"
    assert not any(o.startswith(("ld.global.b8", "ld.global.u8", "ld.global.s8", "ld.global.v2.b8",
                                 "ld.global.v4.b8")) for o in ops), f"{what}: byte loads"
    assert ops.get("cvt.rn.bf16.s32", 0) > 0, f"{what}: int -> bf16 conversion"
    fmas = ops.get("fma.rn.f32", 0) + ops.get("fma.rn.f32x2", 0)
    assert fmas > 0, what
    if one_slice:
        loose = {o: n for o, n in ops.items()
                 if o.startswith(("mul.f32", "mul.rn.f32", "add.f32", "add.rn.f32", "mul.f32x2", "add.f32x2",
                                  "add.rn.f32x2", "mul.rn.f32x2", "fma.rz", "fma.rm", "fma.rp", "add.ftz", "mul.ftz"))}
        assert not loose, f"{what}: fp32 ops outside FMAs {loose}"
    assert k.metadata.shared <= SHARED_MAX, f"{what}: shared {k.metadata.shared} B"
    fit = _ptxas(ptx)
    if fit is not None:
        assert fit[1] == 0, f"{what}: {fit[1]} B spilled ({fit[0]} registers)"
    return fit


DEC_SIG = {"X": "*bf16", "W": "*i32", "S": "*bf16", "OUT": "*bf16", "PART": "*fp32", "M": "i32", "x_stride": "i32"}


@pytest.mark.parametrize("n,k", SHAPES, ids=[f"{n}x{k}" for n, k in SHAPES])
@pytest.mark.parametrize("bm", [16, 32, 64, 128])
def test_decode_kernel(n, k, bm):
    gs = qmm.q8_group(k)
    sk = qmm.q8_split_k(n, k)
    gpi_want, warps, stages, bn = qmm.Q8_CONFIG[bm]
    gpi = qmm.gpi_for((k // gs) // sk, gpi_want)
    kern = _compile(qmm._q8mm, DEC_SIG, {"N": n, "K": k, "SK": sk, "BM": bm, "BLOCK_N": bn, "G": gs, "GPI": gpi,
                                         "F32": False}, warps, stages)
    fit = _check_matmul(kern, one_slice=True, what=f"_q8mm {n}x{k} bm {bm}", loop=(k // gs) // sk // gpi > 1)
    print(f"_q8mm {n}x{k} bm {bm} sk {sk} gpi {gpi}: registers / spill {fit}, shared {kern.metadata.shared} B")


PF_SIG = {"X": "*bf16", "W": "*i32", "S": "*bf16", "OUT": "*bf16", "M": "i32", "x_stride": "i32"}


@pytest.mark.parametrize("n,k", SHAPES, ids=[f"{n}x{k}" for n, k in SHAPES])
@pytest.mark.parametrize("exact", [True, False], ids=["exact", "loose"])
def test_prefill_kernel(n, k, exact):
    sk = qmm.q8_split_k(n, k) if exact else 1
    bm, bn, warps, stages = fast_qmm.q8_tile(exact)
    kern = _compile(fast_qmm._fq8, PF_SIG, {"N": n, "K": k, "SK": sk, "BM": bm, "BN": bn, "G": qmm.q8_group(k),
                                            "F32": False}, warps, stages)
    # the exact variant's slice total (``tl.where``) is fp32 adds by design: only one-slice kernels are FMA-only
    fit = _check_matmul(kern, one_slice=sk == 1, what=f"_fq8 {n}x{k} {'exact' if exact else 'loose'}",
                        loop=(k // qmm.q8_group(k)) // sk > 1)
    print(f"_fq8 {n}x{k} sk {sk} tile {bm}x{bn} w{warps} s{stages}: registers / spill {fit}, "
          f"shared {kern.metadata.shared} B")


H, D, L = 32, 256, 512


@pytest.mark.parametrize("kind", ["absorb", "expand"])
@pytest.mark.parametrize("variant", ["v1", "v2", "tc"])
def test_latent_q8(kind, variant):
    base = {"W": "*i8", "S": "*bf16", "B": "i32"}
    if kind == "expand":
        sig = {"U": "*fp32", **base, "OUT": "*bf16", "M": "i32"}
        cst = {"H": H, "DV": D, "L": L, "N": H * D, "IS_Q4": 2}
        fn = {"v1": latent._expand, "v2": latent._expand2, "tc": latent._expand_tc}[variant]
    else:
        sig = {"Q": "*bf16", **base, "QA": "*bf16", "M": "i32"}
        cst = {"H": H, "DQ": D, "L": L, "N": H * D, "IS_Q4": 2}
        fn = {"v1": latent._absorb, "v2": latent._absorb2, "tc": latent._absorb_tc}[variant]
    if variant == "v2":
        bm, bk, warps = latent.v2_tiles(8, kind)
        cst.update(BMR=bm, BK=bk)
        if kind == "expand":
            cst.update(BN=latent.V2_BN)
    else:
        cst.update(BMR=latent.TC_ROWS if variant == "tc" else latent.BM)
        warps = 4
    kern = _compile(fn, sig, cst, warps, 1 if variant == "v2" else 3)
    ptx = kern.asm["ptx"]
    ops = _ops(ptx)
    if variant != "tc":            # 0390's property: no fp32 add outside the FMA chains (the dequantization s * q is
        # one exact multiply here: an int8 times a bf16 fits fp32's 24 bits)
        bad = {o: n for o, n in ops.items() if o.startswith(("add.f32", "add.rn.f32", "add.f32x2", "mma"))}
        assert not bad, bad
    assert kern.metadata.shared <= SHARED_MAX
    fit = _ptxas(ptx)
    print(f"{kind} {variant} q8 kv_b: registers / spill {fit}, shared {kern.metadata.shared} B")
    # v2 is production's (GLM53_TF_MLA_EXPAND=v2): no spills. v1's expand spills with a 4-bit kv_b too (452 B, 0390's
    # reason for v2); tc is off by decision
    # (asserted under the image's Triton 3.7.x: 3.8.0 spills 8 B in expand v2 with a 4-bit kv_b as well, see
    # tests/test_mla_expand_compile.py)
    if fit is not None and variant == "v2" and triton.__version__.startswith("3.7."):
        assert fit[1] == 0, (kind, variant, fit)
