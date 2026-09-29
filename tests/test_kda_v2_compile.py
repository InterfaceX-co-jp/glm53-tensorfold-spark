"""patches/0400: ``kda_v2``'s kernels compiled for sm_121 (GB10) WITHOUT a GPU, checked against ``fast_kda``'s
compiled kernels op by op -- the part of "same bits" the interpreter cannot see (layouts, Triton's rewrites, LLVM's
FMA contraction). Compiled with the production launch's specializations (16-divisible pointers, row strides,
``b_off``, ``R`` = 512 and ``off`` = 0), 32 heads.

- ``_prep_ref`` (``_prep_item`` on ``_kda_prep``'s grid) compiles to ``_kda_prep``'s PTX, instruction for instruction
  (the kernel name aside): the helper is today's prep;
- ``_state2`` with 64 value rows, 8 warps, no K split, compiles to ``_kda_state``'s float instructions (the grid
  axes swapped: only ``ctaid`` reads differ);
- every other ``_state2`` setting (value rows 16 / 32 / 64 / 128, 2 / 4 / 8 warps, K split) and both ``_fused``
  branches: the same float ops in the same order as the reference, every ``tt.dot`` an mma v2 m16n8 chain with kWidth 1
  operands at tf32 (ieee in the prep's solve), each chain's accumulator from the same source (zero / the previous dot /
  S * e^{G_C}) over the same total K, and every ``tt.reduce`` / ``tt.scan`` (the only layout-ordered ops) with the
  reference's operand layout;
- a register cap (``maxnreg``) changes only the PTX's ``.maxnreg`` line (ptxas' allocation, not the program);
- the PTX float-instruction counts (fma / ex2 / rcp / sqrt / cvt / mma) of ``_fused`` == the prep's + one scan
  step's: no FMA contraction gained or lost by the fusion (mul / add counts may differ where a layout holds a tensor
  twice: the mma layout of a 64-row tile on 8 warps along M computes each element on two warps).
Also prints registers / spills / shared memory of each (``-s``).

Run without TRITON_INTERPRET, on the image's Triton (3.7.1) and on 3.8:
    PYTHONPATH=<patched tree>/src pytest -q -s tests/test_kda_v2_compile.py            (~5 minutes)
"""

from __future__ import annotations

import collections
import glob
import os
import re
import subprocess
import tempfile

import pytest

if os.environ.get("TRITON_INTERPRET") == "1":
    pytest.skip("compiles kernels: run without TRITON_INTERPRET", allow_module_level=True)

triton = pytest.importorskip("triton")
kda_v2 = pytest.importorskip("tensorfold.families.glm5_next.cuda.kda_v2")
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

from tensorfold.families.glm5_next.cuda import fast_kda as fk  # noqa: E402

TGT = GPUTarget("cuda", 121, 32)
H = 32
DIV = {"p_stride", "b_off", "a_stride", "R", "off"}          # 16-divisible in production (R = 512, off = 0)
PREP_SIG = {"P": "*bf16", "p_stride": "i32", "b_off": "i32", "A": "*bf16", "a_stride": "i32", "CS": "*bf16",
            "CW": "*bf16", "A_LOG": "*fp32", "DT_BIAS": "*fp32", "WORKBUF": "*fp32", "SW": "*fp32", "SU": "*fp32",
            "SQ": "*fp32", "SK": "*fp32", "KSAVE": "*fp32", "VSAVE": "*fp32", "GSAVE": "*fp32", "BSAVE": "*fp32",
            "lower": "fp32", "R": "i32", "off": "i32"}
STATE_SIG = {"WORKBUF": "*fp32", "SW": "*fp32", "SU": "*fp32", "SQ": "*fp32", "SK": "*fp32", "S_IN": "*fp32",
             "S_OUT": "*fp32", "OUT": "*bf16", "R": "i32", "off": "i32", "NC": "i32"}
FUSED_SIG = {"P": "*bf16", "p_stride": "i32", "b_off": "i32", "A": "*bf16", "a_stride": "i32", "CS": "*bf16",
             "CW": "*bf16", "A_LOG": "*fp32", "DT_BIAS": "*fp32", "RINGBUF": "*fp32", "TMPBUF": "*fp32", "SW": "*fp32",
             "SU": "*fp32", "SQ": "*fp32", "SK": "*fp32", "S_IN": "*fp32", "S_OUT": "*fp32", "OUT": "*bf16",
             "CTL": "*i32", "lower": "fp32", "R": "i32", "off": "i32", "NC": "i32", "LAG": "i32"}
PREP_CST = {"H": H, "SAVE": False, "BCC": 16, "PREC": "tf32"}
_CACHE: dict = {}


def _compile(fn, sig, cst, warps, stages=1, maxnreg=None):
    key = (fn.__name__, tuple(sorted(cst.items())), warps, stages, maxnreg)
    if key in _CACHE:
        return _CACHE[key]
    sig = dict(sig)
    for k in cst:
        sig[k] = "constexpr"
    attrs = {(fn.arg_names.index(n),): [["tt.divisibility", 16]] for n, t in sig.items()
             if t.startswith("*") or n in DIV}
    try:
        k = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=cst, attrs=attrs), target=TGT,
                           options={"num_warps": warps, "num_stages": stages,
                                    **({"maxnreg": maxnreg} if maxnreg else {})})
    except Exception as exc:  # noqa: BLE001
        if "ptxas" in str(exc).lower() or "not found" in str(exc).lower():
            pytest.skip(f"cannot compile for sm_121 here: {exc}")
        raise
    _CACHE[key] = k
    return k


def _resources(k) -> str:
    ptxas = (glob.glob(os.path.join(os.path.dirname(triton.__file__), "backends/nvidia/bin/ptxas-blackwell"))
             + glob.glob(os.path.join(os.path.dirname(triton.__file__), "backends/nvidia/bin/ptxas")))
    if not ptxas:
        return f"smem {k.metadata.shared}"
    with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as f:
        f.write(k.asm["ptx"])
    r = subprocess.run([ptxas[0], "-arch=sm_121a", "-v", f.name, "-o", os.devnull], capture_output=True, text=True)
    os.unlink(f.name)
    regs = re.search(r"Used (\d+) registers", r.stderr)
    sp = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", r.stderr)
    return (f"regs {regs.group(1) if regs else '?'} spill {sp.group(1) if sp else '?'}/{sp.group(2) if sp else '?'} "
            f"smem {k.metadata.shared}")


# -- TTGIR --------------------------------------------------------------------------------------------------------------
FLOAT_OPS = ("arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.negf", "arith.maxnumf", "arith.minnumf",
             "arith.maximumf", "arith.minimumf", "math.exp", "math.exp2", "math.sqrt", "math.rsqrt", "tt.precise_sqrt",
             "tt.precise_divf", "arith.truncf", "arith.extf", "tt.fp_to_fp", "tt.dot", "tt.reduce", "tt.scan",
             "math.fma")


def _encodings(ttgir: str) -> dict:
    return {m.group(1): m.group(2) for m in re.finditer(r"^(#[\w]+) = (.*)$", ttgir, re.M)}


def _canon_mma(s: str) -> str:
    m = re.search(r"versionMajor = (\d+), versionMinor = (\d+).*instrShape = \[([\d, ]+)\]", s)
    return f"mma(v{m.group(1)}.{m.group(2)},{m.group(3)})" if m else s


def _resolve(t: str, enc: dict, full: bool) -> str:
    """A type with its encodings spelled out (``full``) or reduced to what decides per-element bits (dots: mma
    version / instruction shape and the operands' kWidth; elementwise: none)."""

    def rep(m):
        name = m.group(0)
        if name not in enc:
            return name
        d = enc[name]
        if "nvidia_mma" in d:
            return _canon_mma(d)
        return d if full else "L"

    t = re.sub(r"#[\w]+(?![\w.])", rep, t)
    t = re.sub(r"#ttg\.dot_op<\{opIdx = (\d), parent = ([^,]+), kWidth = (\d+)\}>",
               lambda m: f"dot{m.group(1)}(k{m.group(3)},{m.group(2)})", t)
    if not full:
        t = re.sub(r"#ttg\.slice<\{dim = \d, parent = L\}>", "L", t)
    return t


def _float_ops(ttgir: str, *, shapes: bool = True) -> list[str]:
    """The float ops in program order. Reduce / scan: operand type with its FULL layout (their order depends on
    it), axis, combine op. Dot: operand / result encodings reduced to mma version + kWidth, input precision, what
    the accumulator is (zero, another dot, a mulf, ...), K. Elementwise: op, result element type (and shape)."""

    enc = _encodings(ttgir)
    lines = ttgir.splitlines()
    defs = {}
    for ln in lines:
        m = re.match(r"\s*(%[\w#]+)(?::\d+)? = ([\w.]+|\"[\w.]+\")", ln)
        if m:
            defs[m.group(1)] = (m.group(2).strip('"'), ln)
    out = []
    for n, ln in enumerate(lines):
        m = re.match(r"\s*(?:%[\w#:]+ = )?(\"?[\w.]+\"?)", ln)
        if not m:
            continue
        op = m.group(1).strip('"')
        if op not in FLOAT_OPS:
            continue
        if op in ("tt.reduce", "tt.scan"):
            body = []
            for j in range(n + 1, min(n + 8, len(lines))):
                bm = re.match(r"\s*(?:%[\w#:]+ = )?([\w.]+)", lines[j])
                if bm and bm.group(1) in FLOAT_OPS:
                    body.append(bm.group(1))
                if "}) :" in lines[j] or lines[j].strip().startswith("})"):
                    typ = lines[j].split(" : ", 1)[1] if " : " in lines[j] else lines[j]
                    break
            axis = re.search(r"axis = (\d+)", ln)
            rev = "rev" if "reverse = true" in ln else ""
            out.append(f"{op} axis{axis.group(1) if axis else '?'}{rev} {body} "
                       f"{_resolve(re.sub(r' loc\(.*$', '', typ), enc, True)}")
            continue
        if op == "tt.dot":
            ops = re.search(r"tt\.dot (%[\w#]+), (%[\w#]+), (%[\w#]+)(?:, inputPrecision = (\w+))?", ln)
            acc = ops.group(3)
            src = defs.get(acc, ("?", ""))
            while src[0] == "ttg.convert_layout":                  # a layout move of the accumulator: its source
                acc = re.search(r"convert_layout (%[\w#]+)", src[1]).group(1)
                src = defs.get(acc, ("?", ""))
            kind = src[0]
            if kind == "arith.constant":
                kind = "zero" if re.search(r"dense<0\.0+e\+00>", src[1]) else "const"
            typ = re.sub(r" loc\(.*$", "", ln.split(" : ", 1)[1])
            kk = re.search(r"tensor<\d+x(\d+)x", typ)
            sig = _resolve(typ, enc, False)
            if not shapes:
                sig = re.sub(r"tensor<[\dx]+", "tensor<", sig)
            out.append(f"tt.dot {ops.group(4) or 'tf32?'} acc={kind} K={kk.group(1) if kk else '?'} {sig}")
            continue
        typ = re.sub(r" loc\(.*$", "", ln.split(" : ")[-1])
        el = re.findall(r"x?(f32|bf16|f16|f64)", typ)
        out.append(f"{op} {el[-1] if el else typ}")
    return out


def _ptx_float_counts(ptx: str) -> collections.Counter:
    c = collections.Counter()
    for ln in ptx.splitlines():
        s = ln.strip()
        m = re.match(r"(?:@%p\d+ )?((?:fma|mul|add|sub|neg|ex2|rcp|sqrt|rsqrt|div|mma|max|min|cvt)[.\w]*)", s)
        if m and any(t in m.group(1) for t in ("f32", "bf16", "tf32", "f16")):
            c[m.group(1)] += 1
    return c


def _prep(warps=8):
    return _compile(fk._kda_prep, PREP_SIG, PREP_CST, warps)


def _state_ref():
    return _compile(fk._kda_state, STATE_SIG, {"H": H, "BV": 64, "PREC": "tf32"}, 8)


def _state2(bv, warps, ks, maxnreg=None):
    return _compile(kda_v2._state2, STATE_SIG, {"H": H, "BV": bv, "PREC": "tf32", "KSPLIT": ks}, warps,
                    maxnreg=maxnreg)


def _fused(nvb, ks, ring=3, maxnreg=None):
    return _compile(kda_v2._fused, FUSED_SIG, {"H": H, "NVB": nvb, "RING": ring, "BCC": 16, "PREC": "tf32",
                                               "KSPLIT": ks}, 8, maxnreg=maxnreg)


def _strip(ptx: str, name: str) -> str:
    """The PTX without debug lines / labels, the kernel's name, and virtual registers renamed in order of first
    appearance (inlining shifts their numbers, not the instructions)."""

    keep = []
    names: dict[str, str] = {}

    def ren(m):
        r = m.group(0)
        if r not in names:
            names[r] = f"%{m.group(1)}{len(names)}"
        return names[r]

    for ln in ptx.split("\n\t.section")[0].splitlines():            # the DWARF sections come last
        s = ln.strip()
        if s.startswith((".loc", ".file", "//", "$L__tmp")) or not s or "debug" in s:
            continue
        s = s.replace(name, "KERNEL")
        s = re.sub(r"%(rd|r|fd|f|p|rs|h|hh|b)(\d+)", ren, s)
        keep.append(s)
    return "\n".join(keep)


def _chains(ops: list[str]) -> list[tuple]:
    """Dots grouped into accumulator chains: (precision, first acc source, total K)."""

    out = []
    for o in ops:
        if not o.startswith("tt.dot"):
            continue
        prec, acc, k = re.match(r"tt\.dot (\S+) acc=(\S+) K=(\d+)", o).groups()
        if acc == "tt.dot" and out and out[-1][3]:
            p, a, kk, _ = out[-1]
            out[-1] = (p, a, kk + int(k), True)
        else:
            out.append((prec, acc, int(k), True))
    return [x[:3] for x in out]


# -- tests --------------------------------------------------------------------------------------------------------------
def test_prep_helper_is_kda_prep():
    a = _prep()
    b = _compile(kda_v2._prep_ref, PREP_SIG, dict(PREP_CST, CG=""), 8)
    print(f"\n  _kda_prep  {_resources(a)}\n  _prep_ref  {_resources(b)}")
    assert _strip(b.asm["ptx"], "_prep_ref") == _strip(a.asm["ptx"], "_kda_prep")


def test_state2_bv64_is_kda_state():
    a, b = _state_ref(), _state2(64, 8, 0)
    print(f"\n  _kda_state bv64 w8 {_resources(a)}\n  _state2    bv64 w8 {_resources(b)}")
    assert _float_ops(b.asm["ttgir"]) == _float_ops(a.asm["ttgir"])
    assert _ptx_float_counts(b.asm["ptx"]) == _ptx_float_counts(a.asm["ptx"])
    pa = re.sub(r"%ctaid\.[xy]", "%ctaid", _strip(a.asm["ptx"], "_kda_state"))
    pb = re.sub(r"%ctaid\.[xy]", "%ctaid", _strip(b.asm["ptx"], "_state2"))
    fa = [x for x in pa.splitlines() if re.search(r"\b(fma|mul|add|sub|ex2|mma|cvt)\.[\w.]*f32", x)]
    fb = [x for x in pb.splitlines() if re.search(r"\b(fma|mul|add|sub|ex2|mma|cvt)\.[\w.]*f32", x)]
    assert len(fa) == len(fb)


def _collapse(xs):
    """Consecutive repeats as one: a value-block or key split runs an op once per part (each element once)."""

    return [x for n, x in enumerate(xs) if n == 0 or x != xs[n - 1]]


def _check_scan(ops_new, ops_ref):
    """Same elementwise ops in order, same dot chains (per element: acc source, total K, precision), mma v2 m16n8
    with kWidth 1 everywhere, no reduce / scan. A split into parts repeats an op or a chain once per part."""

    ew = lambda ops: [o for o in ops if not o.startswith("tt.dot")]            # noqa: E731
    assert _collapse(ew(ops_new)) == _collapse(ew(ops_ref))
    assert _collapse(_chains(ops_new)) == _collapse(_chains(ops_ref))
    for o in ops_new:
        if o.startswith("tt.dot"):
            assert "mma(v2.0,16, 8)" in o and re.findall(r"kWidth = \d+", o) == ["kWidth = 1"] * 2, o
    assert not any(o.startswith(("tt.reduce", "tt.scan")) for o in ops_new)


@pytest.mark.parametrize("bv,warps,ks", [(16, 4, 1), (32, 4, 1), (32, 8, 1), (32, 4, 0), (64, 4, 1), (64, 8, 1),
                                         (64, 8, 0), (32, 2, 1)])
def test_state2_settings_same_arithmetic(bv, warps, ks):
    k = _state2(bv, warps, ks)
    print(f"\n  _state2 bv{bv} w{warps} ks{ks}: {_resources(k)}")
    _check_scan(_float_ops(k.asm["ttgir"]), _float_ops(_state_ref().asm["ttgir"]))


@pytest.mark.parametrize("nvb,ks", [(2, 0), (2, 1), (4, 1), (4, 0)])
def test_fused_is_prep_then_scan_step(nvb, ks):
    f = _fused(nvb, ks)
    print(f"\n  _fused nvb{nvb} ks{ks}: {_resources(f)}")
    ops = _float_ops(f.asm["ttgir"])
    prep = _float_ops(_prep().asm["ttgir"])
    n = len(prep)
    red = lambda ops: [o for o in ops if o.startswith(("tt.reduce", "tt.scan"))]   # noqa: E731
    assert red(ops[:n]) == red(prep) and len(red(prep)) == 33          # 2 norms, the cumsum, 2 x 15 solve steps
    assert [o for o in ops[:n] if not o.startswith("tt.dot")] == [o for o in prep if not o.startswith("tt.dot")]
    assert _chains(ops[:n]) == _chains(prep)
    _check_scan(ops[n:], _float_ops(_state_ref().asm["ttgir"]))
    # PTX: fused == prep + one step of the same scan (8 warps): no FMA contraction gained or lost
    step = _state2(128 // nvb, 8, ks)
    want = _ptx_float_counts(_prep().asm["ptx"]) + _ptx_float_counts(step.asm["ptx"])
    got = _ptx_float_counts(f.asm["ptx"])
    print(f"  PTX float instructions fused - (prep + step): +{dict(got - want)} -{dict(want - got)}")
    assert set(got) == set(want)
    for key in got:            # contraction would move counts between fma and mul / add; a layout that holds a
        if key.startswith(("fma", "mma", "ex2", "rcp", "sqrt", "rsqrt", "div", "cvt")):      # tensor twice (mma
            assert got[key] == want[key], key                   # replication) only repeats a mul, never an fma


@pytest.mark.parametrize("which,cap", [("state2-32-4", 168), ("state2-16-4", 128), ("fused-4-1", 128)])
def test_register_cap_changes_only_the_directive(which, cap):
    kind, a, b = which.split("-")
    mk = (lambda m: _state2(int(a), int(b), 1, m)) if kind == "state2" else (lambda m: _fused(int(a), int(b), 3, m))
    free, capped = mk(None), mk(cap)
    print(f"\n  {which}: free {_resources(free)}; maxnreg {cap}: {_resources(capped)}")
    name = "_state2" if kind == "state2" else "_fused"
    lines = [x for x in _strip(capped.asm["ptx"], name).splitlines() if not x.startswith(".maxnreg")]
    assert "\n".join(lines) == _strip(free.asm["ptx"], name)
    assert f".maxnreg {cap}" in capped.asm["ptx"]
