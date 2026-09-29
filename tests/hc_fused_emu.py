"""patches/0520: CPU tools for the fused hyper-connection boundary kernel (``hc_fused.cu``), used by
``tests/test_hc_fused.py`` and ``tests/test_hc_fused_compile.py``.

Three pieces:

1. **fp32 helpers** (exact fma, bf16 rounding) and **stand-ins for the approximate hardware units** (MUFU.RCP,
   MUFU.SQRT, MUFU.EX2). The stand-ins are deterministic functions that are deliberately NOT the exact functions
   (the last bit is flipped for about half of the inputs, and they flush subnormals like the units do). Their values
   do not matter: two programs agree under them only if they feed the SAME operand bits into the SAME unit, and a
   program that used an exact division or square root where the other used the approximate unit disagrees.

2. **A PTX interpreter** (SIMT, all threads of a CTA as numpy lanes, min-PC scheduling for divergence, shuffles,
   barriers, shared memory, stmatrix, atomics), with the approximate PTX instructions defined by the SASS ptxas makes
   of them on sm_121 (read from Triton 3.7.1's own ptxas):

   - ``div.full.f32 d, a, b``: p0 = |b| > 2^126, p1 = |b| >= 2^-126 (or NaN); p0: a, b *= 0.25; !p1: a, b *= 2^24;
     d = RCP(b) * a (an FMUL);
   - ``ex2.approx.f32 d, x``: p = x >= -126 (or NaN); !p: x *= 0.5; d = EX2(x); !p: d = d * d;
   - ``ex2.approx.ftz.f32`` = EX2, ``rcp.approx.ftz.f32`` = RCP, ``sqrt.approx.ftz.f32`` = SQRT (one unit op each).

3. **``ptxas_view(ptx)``**: Triton's PTX rewritten to what ptxas actually executes. PTX ``mul`` / ``add`` /
   ``sub`` without a rounding modifier may be contracted by ptxas, and ptxas does contract them (Triton 3.7.1 +
   its ptxas 13.1 for sm_121a; seen in the SASS of ``_hc_finish``). The rules, checked against the SASS by
   ``test_hc_fused_compile.py`` (FFMA / FADD / MUFU counts): packed ``.f32x2`` operations are split into lanes (the
   ``mov.b64`` packs followed), dead lanes are dropped, ``div.full.f32`` is expanded as above (the divisor 2^k by a
   constant becomes an exact multiply), and a non-rounding ``mul`` whose only use is a non-rounding ``add`` / ``sub``
   is fused into one ``fma.rn``.

Nothing here imports torch or triton.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# ------------------------------------------------------------------------------------------------------------------
# fp32 helpers
# ------------------------------------------------------------------------------------------------------------------
F32 = np.float32
U32 = np.uint32
U64 = np.uint64


def f2u(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def u2f(u) -> np.ndarray:
    return np.asarray(u, dtype=np.uint64).astype(np.uint32).view(np.float32)


def fma32(a, b, c) -> np.ndarray:
    """fp32 fma(a, b, c), one rounding (nearest-even): exact float64 product, float64 sum rounded to odd, then fp32."""

    a, b, c = np.broadcast_arrays(*(np.asarray(v, dtype=np.float32) for v in (a, b, c)))
    with np.errstate(invalid="ignore", over="ignore"):
        p = a.astype(np.float64) * b.astype(np.float64)
    c = c.astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        s = p + c
        bb = s - p
        err = (p - (s - bb)) + (c - bb)
    even = (s.view(np.int64) & 1) == 0
    fix = (err != 0) & even & np.isfinite(s)
    if fix.any():
        s = np.where(fix, np.nextafter(s, np.where(err > 0, np.inf, -np.inf)), s)
    with np.errstate(over="ignore"):
        return s.astype(np.float32)


def add32(a, b):
    with np.errstate(invalid="ignore", over="ignore"):
        return (np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)).astype(np.float32)


def sub32(a, b):
    with np.errstate(invalid="ignore", over="ignore"):
        return (np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)).astype(np.float32)


def mul32(a, b):
    with np.errstate(invalid="ignore", over="ignore"):
        return (np.asarray(a, dtype=np.float32) * np.asarray(b, dtype=np.float32)).astype(np.float32)


def max32(a, b):
    """PTX max.f32 / fmaxf: a NaN operand gives the other one; +0 beats -0."""

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    out = np.where(a > b, a, b)
    out = np.where(np.isnan(a), b, np.where(np.isnan(b), a, out))
    zero = (a == 0) & (b == 0)
    out = np.where(zero, np.where(np.signbit(a) & np.signbit(b), np.float32(-0.0), np.float32(0.0)), out)
    return out.astype(np.float32)


def bf16_to_f32(u16) -> np.ndarray:
    return (np.asarray(u16, dtype=np.uint32) << 16).view(np.float32)


def f32_to_bf16(x) -> np.ndarray:
    """fp32 -> bf16 bits, nearest-even (cvt.rn.bf16.f32); NaN -> 0x7FFF (the hardware's canonical NaN)."""

    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + np.uint64(0x7FFF) + ((u >> np.uint64(16)) & np.uint64(1))) >> np.uint64(16)).astype(np.uint16)
    nan = np.isnan(np.asarray(x, dtype=np.float32))
    return np.where(nan, np.uint16(0x7FFF), r).astype(np.uint16)


def bf16r(x) -> np.ndarray:
    return bf16_to_f32(f32_to_bf16(x))


def _round_f64(x: np.ndarray, p: int, emin: int) -> np.ndarray:
    """float64 values rounded to p significant bits (nearest-even), gradual underflow below 2^emin: the value, as
    float64 (exact)."""

    x = np.asarray(x, dtype=np.float64)
    out = x.copy()
    fin = np.isfinite(x) & (x != 0)
    if fin.any():
        m, e = np.frexp(np.abs(x[fin]))                   # |x| = m 2^e, m in [0.5, 1)
        e = e - 1                                         # |x| = (2m) 2^e
        q = np.maximum(e, emin) - (p - 1)                 # quantum
        scaled = np.abs(x[fin]) / np.exp2(q.astype(np.float64))
        r = np.round(scaled)                              # numpy rounds half to even
        out[fin] = np.sign(x[fin]) * r * np.exp2(q.astype(np.float64))
    return out


def bf16_mul_bits(a16, b16) -> np.ndarray:
    """mul.rn.bf16 (mul.bf16x2 lane): the exact product rounded once to bf16 (nearest-even, gradual underflow)."""

    p = bf16_to_f32(a16).astype(np.float64) * bf16_to_f32(b16).astype(np.float64)
    r = _round_f64(p, 8, -126)
    with np.errstate(over="ignore"):
        r32 = r.astype(np.float32)                        # exact (8 bits) unless it overflows the range
    big = np.abs(r) >= 2.0 ** 128
    r32 = np.where(big, np.copysign(np.float32(np.inf), r32), r32)
    return f32_to_bf16(r32)


# -- the approximate units (stand-ins) -------------------------------------------------------------------------------
TINY = np.float32(2.0 ** -126)


def _flip(r: np.ndarray, x: np.ndarray, salt: int) -> np.ndarray:
    """Flip the last bit of finite nonzero results for inputs whose mantissa hash bit is set (inputs with a zero
    mantissa -- powers of two -- are left exact, as the units are)."""

    xb = f2u(x).astype(np.uint32)
    h = ((xb * np.uint32(0x9E3779B1)) >> np.uint32(salt)) & np.uint32(1)
    man = (xb & np.uint32(0x7FFFFF)) != 0
    ok = np.isfinite(r) & (r != 0) & man & (h == 1)
    rb = f2u(r).astype(np.uint32)
    return np.where(ok, rb ^ np.uint32(1), rb).view(np.float32)


def mufu_rcp(x) -> np.ndarray:
    """MUFU.RCP stand-in: subnormal inputs flushed (-> inf), subnormal results flushed to zero."""

    x = np.asarray(x, dtype=np.float32)
    xf = np.where(np.abs(x) < TINY, np.copysign(np.float32(0), x), x)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        r = (1.0 / xf.astype(np.float64)).astype(np.float32)
    r = np.where(np.abs(r) < TINY, np.copysign(np.float32(0), r), r)
    return _flip(r, xf, 7)


def mufu_sqrt(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    xf = np.where(np.abs(x) < TINY, np.copysign(np.float32(0), x), x)
    with np.errstate(invalid="ignore"):
        r = np.sqrt(xf.astype(np.float64)).astype(np.float32)
    r = np.where(xf < 0, np.float32(np.nan), r)
    return _flip(r, xf, 11)


def mufu_ex2(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    xf = np.where(np.abs(x) < TINY, np.copysign(np.float32(0), x), x)
    with np.errstate(over="ignore", under="ignore"):
        r = np.exp2(xf.astype(np.float64)).astype(np.float32)
    r = np.where(np.abs(r) < TINY, np.float32(0), r)
    return _flip(r, xf, 13)


def div_full(a, b) -> np.ndarray:
    """div.full.f32 as ptxas lowers it (sm_121): scale both operands by 1/4 above 2^126 or by 2^24 below 2^-126,
    then RCP(b') * a' (one rounding)."""

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if VARIANT == "exact_div":
        with np.errstate(all="ignore"):
            return (a.astype(np.float64) / b.astype(np.float64)).astype(np.float32)
    a2, b2, r = div_full_parts(a, b)
    return mul32(r, a2)


def div_full_parts(a, b):
    """(a', b', RCP(b')) of the div.full lowering (the last step is r * a', or fma(r, a', c) when ptxas fuses it)."""

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    p0 = np.abs(b) > np.float32(2.0 ** 126)
    p1 = ~(np.abs(b) < TINY)                              # GEU: NaN counts as >=
    a2 = np.where(p0, mul32(a, np.float32(0.25)), a)
    b2 = np.where(p0, mul32(b, np.float32(0.25)), b)
    a2 = np.where(~p1, mul32(a2, np.float32(2.0 ** 24)), a2)
    b2 = np.where(~p1, mul32(b2, np.float32(2.0 ** 24)), b2)
    return a2.astype(np.float32), b2.astype(np.float32), mufu_rcp(b2)


def ex2_approx(x) -> np.ndarray:
    """ex2.approx.f32 (not ftz) as ptxas lowers it: halve inputs below -126, square the result back."""

    x = np.asarray(x, dtype=np.float32)
    p = ~(x < np.float32(-126.0))
    xh = np.where(p, x, mul32(x, np.float32(0.5)))
    r = mufu_ex2(xh)
    return np.where(p, r, mul32(r, r)).astype(np.float32)


LOG2E = np.uint32(0x3FB8AA3B).view(np.float32)            # the constant Triton multiplies by before ex2


# ------------------------------------------------------------------------------------------------------------------
# PTX parsing
# ------------------------------------------------------------------------------------------------------------------
@dataclass
class Ins:
    op: str                     # full opcode, e.g. "fma.rn.f32x2"
    args: list                  # operand strings (vector groups as lists, addresses as ("addr", base, off))
    guard: str | None = None    # predicate register
    neg: bool = False           # @!p
    label: str | None = None    # for bra: target
    text: str = ""


@dataclass
class Kernel:
    name: str
    params: list[str]
    code: list[Ins]
    labels: dict[str, int]
    shared: dict[str, tuple[int, int]] = field(default_factory=dict)    # symbol -> (offset, size)
    regtypes: dict[str, str] = field(default_factory=dict)
    shared_bytes: int = 0


def _strip(ptx: str) -> str:
    out = []
    skip = 0
    for ln in ptx.splitlines():
        s = ln.split("//", 1)[0].rstrip()
        st = s.strip()
        if st.startswith(".section"):
            skip = 1
            continue
        if skip:
            if st == "}":
                skip = 0
            continue
        if not st or st.startswith((".loc", ".file", ".pragma")):
            continue
        out.append(s)
    return "\n".join(out)


def _split_ops(s: str) -> list:
    """Operands: commas at depth 0; {a, b} -> list; [x + 4] -> ("addr", x, 4)."""

    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    out = []
    for p in parts:
        if p.startswith("{"):
            out.append([q.strip() for q in p[1:-1].split(",")])
        elif p.startswith("["):
            inner = p[1:-1].strip()
            m = re.match(r"^(\S+?)\s*\+\s*(-?\w+)$", inner)
            if m:
                out.append(("addr", m.group(1), int(m.group(2), 0)))
            else:
                out.append(("addr", inner, 0))
        else:
            out.append(p)
    return out


_SHARED_RE = re.compile(r"(?:\.extern\s+)?\.shared\s+\.align\s+(\d+)\s+\.(\w+)\s+([\w$]+)(?:\[(\d*)\])?\s*;")


def _shared_decl(m):
    al, ty, sym, n = int(m.group(1)), m.group(2), m.group(3), m.group(4)
    esz = max(_width(ty) // 8, 1) if ty != "b8" else 1
    count = int(n) if n else (0 if m.group(4) is not None else 1)
    return al, sym, esz * count


def parse(ptx: str, name: str | None = None) -> Kernel:
    """The (first, or ``name``d) .entry of a PTX module."""

    src = _strip(ptx)
    shared: dict[str, tuple[int, int]] = {}
    off = 0
    for m in _SHARED_RE.finditer(src):
        al, sym, n = _shared_decl(m)
        off = (off + al - 1) // al * al
        if al < 16 and off % 16 == 0:
            off += al                  # adversarial placement: aligned to what the array declares, no more
        shared[sym] = (off, n)
        off += n
    ents = list(re.finditer(r"\.entry\s+(\w+)\s*\((.*?)\)", src, re.S))
    if name is not None:
        ents = [e for e in ents if e.group(1) == name]
    if not ents:
        raise ValueError("no .entry")
    e = ents[0]
    params = [p for p in re.findall(r"(\w+_param_\d+)", e.group(2))]
    body_start = src.index("{", e.end())
    depth, i = 0, body_start
    while True:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    body = src[body_start + 1:i]
    # local shared declarations inside the body
    regtypes: dict[str, str] = {}
    code: list[Ins] = []
    labels: dict[str, int] = {}
    # scope braces (inline asm blocks) on their own lines are dropped; vector-operand braces stay in their statement
    text = re.sub(r"\n\s*\{\s*\n", "\n", "\n" + body + "\n")        # scope braces on their own lines
    text = re.sub(r"\n\s*\}\s*\n", "\n", text)
    text = re.sub(r"\n\s*\}\s*\n", "\n", text)
    for raw in text.split(";"):
        s = raw.strip()
        while True:
            m = re.match(r"^(\$?[\w$]+):\s*(.*)$", s, re.S)
            if not m:
                break
            labels[m.group(1)] = len(code)
            s = m.group(2).strip()
        s = s.strip()
        while s[:1] in ("{", "}") and not re.match(r"^\{\s*%", s):
            s = s[1:].strip()
        if not s:
            continue
        s = " ".join(s.split())
        if s.startswith(".reg"):
            m = re.match(r"\.reg\s+\.(\w+)\s+(.*)$", s)
            if m:
                for nm in m.group(2).split(","):
                    nm = nm.strip()
                    mm = re.match(r"^(%?\w+)<(\d+)>$", nm)
                    if mm:
                        regtypes[mm.group(1)] = m.group(1)
                    else:
                        regtypes[nm] = m.group(1)
            continue
        if s.startswith((".shared", ".local", ".maxntid", ".reqntid", ".minnctapersm", ".maxnreg")):
            continue
        guard, neg = None, False
        m = re.match(r"^@(!?)(%\w+)\s+(.*)$", s)
        if m:
            neg, guard, s = m.group(1) == "!", m.group(2), m.group(3)
        m = re.match(r"^(\S+)\s*(.*)$", s)
        op, rest = m.group(1), m.group(2)
        args = _split_ops(rest) if rest else []
        ins = Ins(op, args, guard, neg, text=s)
        if op.startswith("bra"):
            ins.label = args[-1]
        code.append(ins)
    shared_bytes = off
    return Kernel(e.group(1), params, code, labels, shared, regtypes, shared_bytes)


# ------------------------------------------------------------------------------------------------------------------
# ptxas's view of a PTX program (Triton's): lanes split, div.full expanded, mul + add contracted
# ------------------------------------------------------------------------------------------------------------------
def _is_reg(x) -> bool:
    return isinstance(x, str) and x.startswith("%")


def _fconst(s: str):
    if isinstance(s, str) and re.match(r"^0f[0-9A-Fa-f]{8}$", s):
        return np.uint32(int(s[2:], 16)).view(np.float32)
    return None


def ptxas_view(k: Kernel) -> tuple[Kernel, dict]:
    """(the rewritten kernel, statistics). Only straight-line register dataflow is rewritten; loops are left as they
    are (their loop-carried registers have more than one definition and are never contracted)."""

    code = [Ins(i.op, [list(a) if isinstance(a, list) else a for a in i.args], i.guard, i.neg, i.label, i.text)
            for i in k.code]
    lab_at = {}
    for nm, idx in k.labels.items():
        lab_at.setdefault(idx, []).append(nm)
    out: list[Ins] = []
    newlabels: dict[str, int] = {}
    stats = {"lanes_split": 0, "div_expanded": 0, "div_pow2": 0, "contracted": 0, "dead": 0}
    pair: dict[str, tuple[str, str]] = {}                      # b64 register used as an f32 pair -> lanes

    def lanes(r):
        if r not in pair:
            pair[r] = (r + ".lo", r + ".hi")
        return pair[r]

    # which b64 registers are f32 pairs: operands / results of .f32x2 ops and of mov.b64 packs of 32-bit parts
    f32x2_regs = set()
    for i in code:
        if i.op.endswith(".f32x2"):
            for a in i.args:
                if _is_reg(a):
                    f32x2_regs.add(a)
    changed = True
    while changed:                                              # close over mov.b64 pack / unpack
        changed = False
        for i in code:
            if i.op == "mov.b64" and len(i.args) == 2:
                d, s = i.args
                if isinstance(s, list) and _is_reg(d) and d not in f32x2_regs:
                    f32x2_regs.add(d)
                    changed = True
                if isinstance(d, list) and _is_reg(s) and s not in f32x2_regs:
                    f32x2_regs.add(s)
                    changed = True
    # registers defined once, by a move of an fp32 immediate (Triton loads div.full's constant divisors this way)
    nd0: dict[str, int] = {}
    for i in code:
        for d in _dests(i):
            nd0[d] = nd0.get(d, 0) + 1
    fimm = {i.args[0]: i.args[1] for i in code if i.op in ("mov.b32", "mov.f32") and isinstance(i.args[0], str)
            and nd0.get(i.args[0]) == 1 and _fconst(i.args[1]) is not None}
    tmp = [0]

    def fresh():
        tmp[0] += 1
        return f"%_t{tmp[0]}"

    for idx, i in enumerate(code):
        for nm in lab_at.get(idx, []):
            newlabels[nm] = len(out)
        if i.op.endswith(".f32x2"):
            base = i.op[:-6] + ".f32"
            stats["lanes_split"] += 1
            for li in (0, 1):
                args = [lanes(a)[li] if _is_reg(a) and a in f32x2_regs else a for a in i.args]
                out.append(Ins(base, args, i.guard, i.neg, text=f"{base} (lane {li} of {i.text})"))
            continue
        if i.op == "mov.b64" and len(i.args) == 2 and (isinstance(i.args[0], list) or isinstance(i.args[1], list)):
            d, s = i.args
            if isinstance(s, list):                              # pack
                for li in (0, 1):
                    if s[li] != "_":
                        out.append(Ins("mov.b32", [lanes(d)[li], s[li]], i.guard, i.neg, text="pack"))
            else:                                                # unpack
                for li in (0, 1):
                    if d[li] != "_":
                        out.append(Ins("mov.b32", [d[li], lanes(s)[li]], i.guard, i.neg, text="unpack"))
            continue
        if i.op == "mov.b64" and _is_reg(i.args[0]) and i.args[0] in f32x2_regs and _is_reg(i.args[1]) \
                and i.args[1] in f32x2_regs:
            for li in (0, 1):
                out.append(Ins("mov.b32", [lanes(i.args[0])[li], lanes(i.args[1])[li]], i.guard, i.neg, text="mov"))
            continue
        if i.op == "div.full.f32":
            d, a, b = i.args
            bc = _fconst(fimm.get(b, b))
            if bc is not None and bc > 0 and np.frexp(np.float64(bc))[0] == 0.5 and abs(np.log2(bc)) < 126:
                inv = np.float32(1.0) / bc
                out.append(Ins("mul.f32", [d, a, "0f%08X" % int(np.float32(inv).view(np.uint32))], i.guard, i.neg,
                               text=f"div.full by 2^k: {i.text}"))
                stats["div_pow2"] += 1
                continue
            stats["div_expanded"] += 1
            a2, b2, r = fresh(), fresh(), fresh()
            out.append(Ins("hc.divscale", [a2, b2, a, b], i.guard, i.neg, text=f"div.full scale: {i.text}"))
            out.append(Ins("rcp.approx.ftz.f32", [r, b2], i.guard, i.neg, text="div.full rcp"))
            out.append(Ins("mul.f32", [d, r, a2], i.guard, i.neg, text="div.full mul"))
            continue
        out.append(i)
    for nm, idx in k.labels.items():
        if idx >= len(code):
            newlabels[nm] = len(out)
    # branch targets
    for i in out:
        if i.op.startswith("bra"):
            i.label = i.args[-1]

    # copy propagation for plain 32-bit movs of registers (single definition), then dead code, then contraction
    defs: dict[str, int] = {}
    ndefs: dict[str, int] = {}
    for j, i in enumerate(out):
        for d in _dests(i):
            ndefs[d] = ndefs.get(d, 0) + 1
            defs[d] = j
    alias: dict[str, str] = {}
    for j, i in enumerate(out):
        if i.op in ("mov.b32", "mov.f32") and i.guard is None and isinstance(i.args[0], str) and _is_reg(i.args[1]) \
                and ndefs.get(i.args[0]) == 1 \
                and ndefs.get(i.args[1], 0) <= 1:
            alias[i.args[0]] = i.args[1]

    def res(r):
        seen = 0
        while r in alias and seen < 64:
            r = alias[r]
            seen += 1
        return r

    for i in out:
        n = len(_dests(i))
        i.args = [a if (x < n and not i.op.startswith(("st.", "stmatrix", "atom", "red"))) else _res_arg(a, res)
                  for x, a in enumerate(i.args)]
    out = _dce(out, newlabels, stats)
    # uses
    uses: dict[str, list[int]] = {}
    for j, i in enumerate(out):
        for s in _srcs(i):
            uses.setdefault(s, []).append(j)
    ndefs = {}
    for j, i in enumerate(out):
        for d in _dests(i):
            ndefs[d] = ndefs.get(d, 0) + 1
    kill = set()
    for j, i in enumerate(out):
        if i.op in ("mul.f32",) and i.guard is None:
            d = i.args[0]
            u = uses.get(d, [])
            if ndefs.get(d) != 1 or len(u) != 1:
                continue
            k2 = out[u[0]]
            if k2.op not in ("add.f32", "sub.f32") or k2.guard is not None:
                continue
            others = [a for a in k2.args[1:]]
            if others.count(d) != 1:
                continue
            a, b = i.args[1], i.args[2]
            if k2.op == "add.f32":
                c = others[1] if others[0] == d else others[0]
                out[u[0]] = Ins("fma.rn.f32", [k2.args[0], a, b, c], text=f"contracted: {i.text} + {k2.text}")
            elif others[0] == d:                    # (a b) - c = fma(a, b, -c)
                out[u[0]] = Ins("hc.fmsub", [k2.args[0], a, b, others[1]], text=f"contracted: {k2.text}")
            else:                                   # c - (a b) = fma(-a, b, c)
                out[u[0]] = Ins("hc.fnmadd", [k2.args[0], a, b, others[0]], text=f"contracted: {k2.text}")
            kill.add(j)
            stats["contracted"] += 1
    if kill:
        remap = {}
        new = []
        for j, i in enumerate(out):
            remap[j] = len(new)
            if j not in kill:
                new.append(i)
        remap[len(out)] = len(new)
        newlabels = {nm: remap[idx] for nm, idx in newlabels.items()}
        out = new
    kk = Kernel(k.name, k.params, out, newlabels, k.shared, k.regtypes, k.shared_bytes)
    return kk, stats


def _res_arg(a, res):
    if isinstance(a, list):
        return [res(x) if _is_reg(x) else x for x in a]
    if isinstance(a, tuple):
        return ("addr", res(a[1]) if _is_reg(a[1]) else a[1], a[2])
    return res(a) if _is_reg(a) else a


NOSIDE = ("mov", "add", "sub", "mul", "fma", "cvt", "selp", "setp", "and", "or", "xor", "shl", "shr", "bfe", "neg",
          "max", "min", "div", "sqrt", "ex2", "rcp", "mad", "hc.divscale", "hc.fmsub", "hc.fnmadd", "abs", "not", "prmt",
          "cvta",
          "rsqrt", "lg2", "sin", "cos", "tanh", "copysign")


def _dests(i: Ins) -> list[str]:
    op = i.op
    if not i.args:
        return []
    if op.startswith(("st.", "stmatrix", "bar", "bra", "ret", "exit", "red.", "membar", "fence", "nanosleep",
                      "griddepcontrol", "prefetch", "trap")):
        return []
    if op == "hc.divscale":
        return [i.args[0], i.args[1]]
    d = i.args[0]
    if isinstance(d, list):
        return [x for x in d if _is_reg(x)]
    if isinstance(d, str) and "|" in d:
        return [x for x in d.split("|") if _is_reg(x)]
    return [d] if _is_reg(d) else []


def _srcs(i: Ins) -> list[str]:
    nd = 2 if i.op == "hc.divscale" else (0 if not _dests(i) else 1)
    out = []
    for a in i.args[nd:]:
        if isinstance(a, list):
            out += [x for x in a if _is_reg(x)]
        elif isinstance(a, tuple):
            if _is_reg(a[1]):
                out.append(a[1])
        elif _is_reg(a):
            out.append(a)
    if i.op.startswith(("st.", "stmatrix", "red.")) or (i.op.startswith("atom") and False):
        pass
    if i.guard:
        out.append(i.guard)
    return out


def _dce(code: list[Ins], labels: dict, stats: dict) -> list[Ins]:
    while True:
        used = set()
        for i in code:
            used.update(_srcs(i))
            # store / atomic sources: everything they read
            if i.op.startswith(("st.", "stmatrix", "red.")):
                for a in i.args:
                    if isinstance(a, list):
                        used.update(x for x in a if _is_reg(x))
                    elif isinstance(a, tuple):
                        used.add(a[1])
                    elif _is_reg(a):
                        used.add(a)
        dead = set()
        for j, i in enumerate(code):
            if not i.op.split(".")[0] in [n.split(".")[0] for n in NOSIDE] and not i.op.startswith("hc."):
                continue
            if i.op.startswith(("ld.", "atom", "shfl", "vote", "bar")):
                continue
            ds = _dests(i)
            if ds and all(d not in used for d in ds):
                dead.add(j)
        if not dead:
            return code
        stats["dead"] += len(dead)
        remap = {}
        new = []
        for j, i in enumerate(code):
            remap[j] = len(new)
            if j not in dead:
                new.append(i)
        remap[len(code)] = len(new)
        for nm in list(labels):
            labels[nm] = remap[labels[nm]]
        code[:] = new


def loop_census(k: Kernel) -> tuple[dict, dict, int, int]:
    """(fp census outside the loop, inside it, shuffles outside, inside) of a kernel with at most one loop (a label and
    a backward branch to it). SASS unrolls such loops; its shuffle count gives the unroll factor."""

    body = set()
    for j, i in enumerate(k.code):
        if i.op.startswith("bra") and k.labels.get(i.label, j + 1) <= j:
            body.update(range(k.labels[i.label], j + 1))
    ins = Kernel(k.name, k.params, [i for j, i in enumerate(k.code) if j in body], {})
    out = Kernel(k.name, k.params, [i for j, i in enumerate(k.code) if j not in body], {})
    sh = lambda kk: sum(1 for i in kk.code if i.op.startswith("shfl"))            # noqa: E731
    return fp_census(out), fp_census(ins), sh(out), sh(ins)


def sass_census(sass: str) -> dict[str, int]:
    """FFMA / FADD / FMUL / MUFU / SHFL counts of an nvdisasm listing."""

    ops = re.findall(r"^\s+/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z0-9_.]+)", sass, re.M)
    c = {"FFMA": 0, "FADD": 0, "FMUL": 0, "RCP": 0, "EX2": 0, "SQRT": 0, "SHFL": 0, "I2F.RP": 0}
    for o in ops:
        base = o.split(".")[0]
        if base in ("FFMA", "FADD", "FMUL"):
            c[base] += 1
        elif o.startswith("MUFU."):
            k = o.split(".")[1]
            if k in c:
                c[k] += 1
        elif base == "SHFL":
            c["SHFL"] += 1
        elif o.startswith("I2F.RP"):
            c["I2F.RP"] += 1
    return c


def fp_census(k: Kernel) -> dict[str, int]:
    """fp32 arithmetic of a (ptxas-view) kernel, in SASS terms: FFMA, FADD (add / sub), FMUL, MUFU units."""

    c = {"FFMA": 0, "FADD": 0, "FMUL": 0, "RCP": 0, "EX2": 0, "SQRT": 0}
    for i in k.code:
        op = i.op
        if op in ("fma.rn.f32", "hc.fmsub", "hc.fnmadd"):
            c["FFMA"] += 1
        elif op in ("add.f32", "add.rn.f32", "sub.f32", "sub.rn.f32"):
            c["FADD"] += 1
        elif op in ("mul.f32", "mul.rn.f32"):
            c["FMUL"] += 1
        elif op == "rcp.approx.ftz.f32":
            c["RCP"] += 1
        elif op.startswith("ex2.approx"):
            c["EX2"] += 1
        elif op.startswith("sqrt.approx"):
            c["SQRT"] += 1
    return c


# ------------------------------------------------------------------------------------------------------------------
# SIMT interpreter
# ------------------------------------------------------------------------------------------------------------------
class Memory:
    """Global memory: numpy buffers at fake device addresses."""

    def __init__(self):
        self.bufs: list[tuple[int, np.ndarray]] = []
        self.next = 1 << 32

    def alloc(self, arr: np.ndarray) -> int:
        b = np.ascontiguousarray(arr).view(np.uint8).reshape(-1).copy()
        base = self.next
        self.bufs.append((base, b))
        self.next += (len(b) + 4096 + 255) // 256 * 256 + (1 << 20)
        return base

    def get(self, base: int, dtype, shape) -> np.ndarray:
        for b, arr in self.bufs:
            if b == base:
                return arr.view(dtype).reshape(shape).copy()
        raise KeyError(base)

    def _locate(self, addr: np.ndarray):
        bases = np.array([b for b, _ in self.bufs], dtype=np.int64)
        idx = np.searchsorted(bases, addr.astype(np.int64), side="right") - 1
        return idx

    def read(self, addr: np.ndarray, nbytes: int) -> np.ndarray:
        out = np.zeros(addr.shape, dtype=np.uint64)
        if addr.size == 0:
            return out
        idx = self._locate(addr)
        for bi in np.unique(idx):
            if bi < 0:
                raise IndexError(f"global read at {hex(int(addr[idx == bi][0]))}: no buffer")
            base, arr = self.bufs[bi]
            sel = idx == bi
            off = addr[sel].astype(np.int64) - base
            if (off < 0).any() or (off + nbytes > len(arr)).any():
                raise IndexError(f"global read out of bounds: buffer {bi} offsets {off.min()}..{off.max()} "
                                 f"size {len(arr)}")
            v = np.zeros(off.shape, dtype=np.uint64)
            for k in range(nbytes):
                v |= arr[off + k].astype(np.uint64) << np.uint64(8 * k)
            out[sel] = v
        return out

    def write(self, addr: np.ndarray, vals: np.ndarray, nbytes: int) -> None:
        if addr.size == 0:
            return
        idx = self._locate(addr)
        for bi in np.unique(idx):
            if bi < 0:
                raise IndexError("global write: no buffer")
            base, arr = self.bufs[bi]
            sel = idx == bi
            off = addr[sel].astype(np.int64) - base
            if (off < 0).any() or (off + nbytes > len(arr)).any():
                raise IndexError("global write out of bounds")
            v = vals[sel].astype(np.uint64)
            # sequential lane order (a later lane wins on conflicts)
            for k in range(nbytes):
                arr[off + k] = ((v >> np.uint64(8 * k)) & np.uint64(0xFF)).astype(np.uint8)


def _mask_bits(n: int) -> np.uint64:
    return np.uint64((1 << n) - 1) if n < 64 else np.uint64(0xFFFFFFFFFFFFFFFF)


def _width(ty: str) -> int:
    m = re.match(r"^[a-z]+(\d+)", ty)
    return int(m.group(1)) if m else 32


class CTA:
    def __init__(self, k: Kernel, mem: Memory, params: dict, ctaid, nctaid, ntid: int, dyn_shared: int = 0):
        self.k, self.mem, self.params = k, mem, params
        self.n = ntid
        self.ctaid, self.nctaid = ctaid, nctaid
        self.regs: dict[str, np.ndarray] = {}
        self.shared = np.zeros(max(k.shared_bytes, 0) + dyn_shared + 16, dtype=np.uint8)
        self.tid = np.arange(ntid, dtype=np.uint64)
        self.pbase = {nm: (0x7F00 << 32) + 65536 * i for i, nm in enumerate(k.params)}
        self.stats = {"steps": 0}

    # -- operands ----------------------------------------------------------------------------------------------
    def val(self, a, lanes=None) -> np.ndarray:
        n = self.n
        if isinstance(a, str):
            if a.startswith("%"):
                if a in ("%tid.x",):
                    return self.tid.copy()
                if a in ("%tid.y", "%tid.z"):
                    return np.zeros(n, dtype=np.uint64)
                if a == "%ntid.x":
                    return np.full(n, self.n, dtype=np.uint64)
                if a == "%laneid":
                    return self.tid & np.uint64(31)
                if a == "%warpid":
                    return self.tid >> np.uint64(5)
                if a.startswith("%ctaid."):
                    return np.full(n, self.ctaid["xyz".index(a[-1])], dtype=np.uint64)
                if a.startswith("%nctaid."):
                    return np.full(n, self.nctaid["xyz".index(a[-1])], dtype=np.uint64)
                if a.startswith("%envreg") or a.startswith("%clock") or a == "%globaltimer":
                    return np.zeros(n, dtype=np.uint64)
                r = self.regs.get(a)
                if r is None:
                    return np.zeros(n, dtype=np.uint64)       # undefined register (Triton's dead lanes)
                return r
            if a in self.k.shared:
                return np.full(n, self.k.shared[a][0], dtype=np.uint64)
            if a in self.params:
                if isinstance(self.params[a], bytes):
                    return np.full(n, np.uint64(self.pbase[a]), dtype=np.uint64)
                return np.full(n, np.uint64(self.params[a]), dtype=np.uint64)
            f = _fconst(a)
            if f is not None:
                return np.full(n, np.uint64(int(np.float32(f).view(np.uint32))), dtype=np.uint64)
            if re.match(r"^0d[0-9A-Fa-f]{16}$", a):
                return np.full(n, np.uint64(int(a[2:], 16)), dtype=np.uint64)
            v = int(a, 0)
            return np.full(n, np.uint64(v & 0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
        raise ValueError(a)

    def setr(self, d: str, v: np.ndarray, m: np.ndarray, bits: int = 64):
        if d == "_":
            return
        v = np.asarray(v).astype(np.uint64) & _mask_bits(bits)
        cur = self.regs.get(d)
        if cur is None:
            cur = np.zeros(self.n, dtype=np.uint64)
        self.regs[d] = np.where(m, v, cur)

    def addr(self, a) -> np.ndarray:
        _, base, off = a
        if base in self.params:
            return np.full(self.n, np.uint64(self.params[base] + off), dtype=np.uint64)
        if base in self.k.shared:
            return np.full(self.n, np.uint64(self.k.shared[base][0] + off), dtype=np.uint64)
        return (self.val(base).astype(np.int64) + off).astype(np.uint64)

    # -- shared memory -----------------------------------------------------------------------------------------
    def sread(self, addr, nbytes, m):
        a = addr[m].astype(np.int64)
        out = np.zeros(self.n, dtype=np.uint64)
        if a.size and ((a < 0).any() or (a + nbytes > len(self.shared)).any()):
            raise IndexError(f"shared read out of bounds {a.min()}..{a.max()}")
        v = np.zeros(a.shape, dtype=np.uint64)
        for k in range(nbytes):
            v |= self.shared[a + k].astype(np.uint64) << np.uint64(8 * k)
        out[m] = v
        return out

    def swrite(self, addr, vals, nbytes, m):
        a = addr[m].astype(np.int64)
        if a.size and ((a < 0).any() or (a + nbytes > len(self.shared)).any()):
            raise IndexError("shared write out of bounds")
        v = vals[m].astype(np.uint64)
        for k in range(nbytes):
            self.shared[a + k] = ((v >> np.uint64(8 * k)) & np.uint64(0xFF)).astype(np.uint8)

    # -- run -----------------------------------------------------------------------------------------------------
    def run(self, max_steps: int = 2_000_000):
        code = self.k.code
        n = self.n
        pc = np.zeros(n, dtype=np.int64)
        done = np.zeros(n, dtype=bool)
        steps = 0
        while not done.all():
            steps += 1
            if steps > max_steps:
                raise RuntimeError("step limit")
            live = ~done
            mn = pc[live].min()
            if mn >= len(code):
                done[live & (pc >= len(code))] = True
                continue
            m = live & (pc == mn)
            ins = code[mn]
            gm = m
            if ins.guard is not None:
                p = self.val(ins.guard).astype(bool)
                gm = m & (~p if ins.neg else p)
            nxt = self.exec(ins, m, gm)
            if nxt is None:
                pc[m] = mn + 1
            elif isinstance(nxt, tuple) and nxt[0] == "exit":
                done[gm] = True
                pc[m & ~gm] = mn + 1
            else:
                tgt = self.k.labels[nxt]
                pc[gm] = tgt
                pc[m & ~gm] = mn + 1
        self.stats["steps"] = steps

    # -- instructions ------------------------------------------------------------------------------------------
    def exec(self, ins: Ins, m: np.ndarray, gm: np.ndarray):
        op = ins.op
        a = ins.args
        parts = op.split(".")
        base = parts[0]
        V = self.val

        if base in ("bra",):
            return ins.label
        if base in ("ret", "exit"):
            return ("exit",)
        if base == "bar" or op.startswith("barrier"):
            return None                             # min-PC scheduling: every live thread is here
        if base in ("membar", "fence", "nanosleep", "griddepcontrol", "prefetch", "prefetchu", "discard"):
            return None
        if not gm.any() and not base.startswith("shfl"):
            return None

        if base == "ld" and parts[1] == "param":
            ty = parts[-1]
            name, off = a[1][1], a[1][2]
            if name not in self.params:               # an address of the param space in a register
                ad = int(self.val(name)[np.argmax(gm)]) + off
                for nm, bs in self.pbase.items():
                    if bs <= ad < bs + 65536:
                        name, off = nm, ad - bs
                        break
                else:
                    raise IndexError("ld.param: not a parameter address")
            pv = self.params[name]
            w = _width(ty) // 8
            dsts = a[0] if isinstance(a[0], list) else [a[0]]
            for k, dd in enumerate(dsts):
                if isinstance(pv, (bytes, bytearray)):
                    v = int.from_bytes(pv[off + k * w: off + (k + 1) * w], "little")
                else:
                    v = (int(pv) >> (8 * (off + k * w))) & ((1 << (8 * w)) - 1)
                self.setr(dd, np.full(self.n, np.uint64(v), dtype=np.uint64), gm, 8 * w)
            return None
        if base == "cvta":
            self.setr(a[0], V(a[1]), gm, 64)
            return None
        if base == "mov":
            ty = parts[-1]
            d, s = a
            if isinstance(s, list):                     # pack
                w = _width(ty) // len(s)
                v = np.zeros(self.n, dtype=np.uint64)
                for k, x in enumerate(s):
                    v |= (V(x) & _mask_bits(w)) << np.uint64(w * k)
                self.setr(d, v, gm, _width(ty))
            elif isinstance(d, list):                   # unpack
                w = _width(ty) // len(d)
                sv = V(s)
                for k, x in enumerate(d):
                    if x != "_":
                        self.setr(x, (sv >> np.uint64(w * k)) & _mask_bits(w), gm, w)
            else:
                self.setr(d, V(s), gm, _width(ty) if ty != "pred" else 1)
            return None
        if base in ("ld", "ldu"):
            space = parts[1]
            ty = parts[-1]
            w = _width(ty) // 8
            ad = self.addr(a[1])
            dsts = a[0] if isinstance(a[0], list) else [a[0]]
            if (ad[gm] % np.uint64(w * len(dsts))).any():
                raise RuntimeError(f"misaligned {w * len(dsts)}-byte access: {ins.text}")
            for k, d in enumerate(dsts):
                if space.startswith("shared"):
                    v = self.sread(ad + np.uint64(k * w), w, gm)
                else:
                    v = np.zeros(self.n, dtype=np.uint64)
                    v[gm] = self.mem.read(ad[gm] + np.uint64(k * w), w)
                if ty.startswith("s") and w < 8:
                    sh = np.uint64(64 - 8 * w)
                    v = ((v << sh).astype(np.int64) >> np.int64(64 - 8 * w)).astype(np.uint64)
                self.setr(d, v, gm, 64 if w == 8 else 8 * w if not ty.startswith("s") else 64)
            return None
        if base == "st":
            space = parts[1]
            ty = parts[-1]
            w = _width(ty) // 8
            ad = self.addr(a[0])
            srcs = a[1] if isinstance(a[1], list) else [a[1]]
            if (ad[gm] % np.uint64(w * len(srcs))).any():
                raise RuntimeError(f"misaligned {w * len(srcs)}-byte access: {ins.text}")
            for k, s in enumerate(srcs):
                v = V(s) & _mask_bits(8 * w)
                if space.startswith("shared"):
                    self.swrite(ad + np.uint64(k * w), v, w, gm)
                else:
                    self.mem.write(ad[gm] + np.uint64(k * w), v[gm], w)
            return None
        if base == "stmatrix":
            # stmatrix.sync.aligned.m8n8.x1.shared.b16 [addr], {r}: lanes 0-7 give the row addresses; lane L holds
            # row L / 4, columns 2 (L % 4), 2 (L % 4) + 1
            if "x1" not in parts or "trans" in parts:
                raise NotImplementedError(op)
            ad = self.addr(a[0])
            val = V(a[1][0])
            for w0 in range(0, self.n, 32):
                if not m[w0:w0 + 32].all():
                    if m[w0:w0 + 32].any():
                        raise RuntimeError("stmatrix with a partial warp")
                    continue
                for L in range(32):
                    row, col = L // 4, 2 * (L % 4)
                    dst = int(ad[w0 + row]) + 2 * col
                    v = int(val[w0 + L]) & 0xFFFFFFFF
                    self.shared[dst:dst + 4] = np.frombuffer(np.uint32(v).tobytes(), dtype=np.uint8)
            return None
        if base == "shfl":
            # shfl.sync.bfly.b32 d, a, b, c, mask
            mode = parts[2]
            d = a[0]
            pdst = None
            if isinstance(d, str) and "|" in d:
                d, pdst = d.split("|")
            src = V(a[1])
            bval = V(a[2]).astype(np.int64)
            lane = (self.tid & np.uint64(31)).astype(np.int64)
            if mode == "bfly":
                partner = lane ^ bval
            elif mode == "down":
                partner = lane + bval
            elif mode == "up":
                partner = lane - bval
            elif mode == "idx":
                partner = bval
            else:
                raise NotImplementedError(op)
            ok = (partner >= 0) & (partner <= 31) if mode != "idx" else np.ones(self.n, dtype=bool)
            partner = np.where(ok, partner, lane)
            idx = (self.tid.astype(np.int64) & ~31) + partner
            if not gm[idx[gm]].all():
                raise RuntimeError(f"shfl partner lane not active: {ins.text}")
            self.setr(d, src[idx], gm, 32)
            if pdst:
                self.setr(pdst, ok.astype(np.uint64), gm, 1)
            return None
        if base == "atom" or base == "red":
            # atom.global.add.u32 d, [a], b ; red.global.add.u32 [a], b
            space = parts[1]
            kind = [p for p in parts if p in ("add", "exch", "cas", "max", "min", "inc", "and", "or")][0]
            ty = parts[-1]
            w = _width(ty) // 8
            if base == "atom":
                d, ad_, b_ = a[0], a[1], a[2]
            else:
                d, ad_, b_ = None, a[0], a[1]
            ad = self.addr(ad_)
            bv = V(b_)
            old = np.zeros(self.n, dtype=np.uint64)
            for L in np.nonzero(gm)[0]:                   # lanes in order
                o = self.mem.read(ad[L:L + 1], w)[0] if not space.startswith("shared") else self.sread(ad, w, np.eye(self.n, dtype=bool)[L])[L]
                if kind == "add":
                    nv = (int(o) + int(bv[L])) & ((1 << (8 * w)) - 1)
                elif kind == "exch":
                    nv = int(bv[L])
                else:
                    raise NotImplementedError(op)
                old[L] = o
                if not space.startswith("shared"):
                    self.mem.write(ad[L:L + 1], np.array([nv], dtype=np.uint64), w)
                else:
                    mm = np.zeros(self.n, dtype=bool)
                    mm[L] = True
                    self.swrite(ad, np.full(self.n, nv, dtype=np.uint64), w, mm)
            if d is not None:
                self.setr(d, old, gm, 8 * w)
            return None
        return self.alu(ins, op, parts, a, gm)

    def alu(self, ins, op, parts, a, gm):
        V = self.val
        base = parts[0]
        ty = parts[-1]
        d = a[0] if a else None

        def F(x):
            return u2f(V(x) & np.uint64(0xFFFFFFFF))

        def setf(dst, f):
            self.setr(dst, f2u(np.asarray(f, dtype=np.float32)).astype(np.uint64), gm, 32)

        if op == "hc.divscale":
            a2, b2, _ = div_full_parts(F(a[2]), F(a[3]))
            setf(a[0], a2)
            setf(a[1], b2)
            return None
        if op == "hc.fmsub":
            setf(d, fma32(F(a[1]), F(a[2]), -F(a[3])))
            return None
        if op == "hc.fnmadd":
            setf(d, fma32(-F(a[1]), F(a[2]), F(a[3])))
            return None
        if ty == "f32x2" and base in ("add", "sub", "mul", "fma"):
            # packed pair: each 32-bit lane is the scalar operation (IEEE, nearest-even)
            def lane(x, k):
                return u2f((V(x) >> np.uint64(32 * k)) & np.uint64(0xFFFFFFFF))

            res = []
            for k in (0, 1):
                if base == "add":
                    r = add32(lane(a[1], k), lane(a[2], k))
                elif base == "sub":
                    r = sub32(lane(a[1], k), lane(a[2], k))
                elif base == "mul":
                    r = mul32(lane(a[1], k), lane(a[2], k))
                else:
                    if "rn" not in parts:
                        raise NotImplementedError(op)
                    r = fma32(lane(a[1], k), lane(a[2], k), lane(a[3], k))
                res.append(f2u(r).astype(np.uint64))
            self.setr(d, res[0] | (res[1] << np.uint64(32)), gm, 64)
            return None
        if ty == "f32" and base in ("add", "sub", "mul", "fma", "div", "sqrt", "ex2", "rcp", "max", "min", "neg",
                                     "abs", "mad"):
            if "sat" in parts:
                raise NotImplementedError(op)
            if base == "add":
                r = add32(F(a[1]), F(a[2]))
            elif base == "sub":
                r = sub32(F(a[1]), F(a[2]))
            elif base == "mul":
                r = mul32(F(a[1]), F(a[2]))
            elif base in ("fma", "mad"):
                if "rn" not in parts:
                    raise NotImplementedError(op)
                r = fma32(F(a[1]), F(a[2]), F(a[3]))
            elif base == "div":
                if "full" in parts:
                    r = div_full(F(a[1]), F(a[2]))
                elif "rn" in parts:
                    with np.errstate(all="ignore"):
                        r = (F(a[1]).astype(np.float64) / F(a[2]).astype(np.float64)).astype(np.float32)
                else:
                    raise NotImplementedError(op)
            elif base == "sqrt":
                if "approx" in parts and "ftz" in parts:
                    r = mufu_sqrt(F(a[1]))
                elif "rn" in parts:
                    with np.errstate(invalid="ignore"):
                        r = np.sqrt(F(a[1]).astype(np.float64)).astype(np.float32)
                else:
                    raise NotImplementedError(op)
            elif base == "ex2":
                if "ftz" in parts:
                    r = mufu_ex2(F(a[1]))
                else:
                    r = ex2_approx(F(a[1]))
            elif base == "rcp":
                if "approx" in parts and "ftz" in parts:
                    r = mufu_rcp(F(a[1]))
                else:
                    raise NotImplementedError(op)
            elif base == "max":
                r = max32(F(a[1]), F(a[2]))
            elif base == "min":
                r = -max32(-F(a[1]), -F(a[2]))
            elif base == "neg":
                r = -F(a[1])
            elif base == "abs":
                r = np.abs(F(a[1]))
            if "ftz" in parts and base in ("add", "sub", "mul", "fma", "max", "min"):
                raise NotImplementedError(op)       # not used by these kernels
            setf(d, r)
            return None
        if op == "fma.rn.f32.bf16":
            r = fma32(bf16_to_f32(V(a[1]) & np.uint64(0xFFFF)), bf16_to_f32(V(a[2]) & np.uint64(0xFFFF)), F(a[3]))
            setf(d, r)
            return None
        if op == "add.rn.f32.bf16":
            r = add32(bf16_to_f32(V(a[1]) & np.uint64(0xFFFF)), F(a[2]))
            setf(d, r)
            return None
        if op in ("mul.bf16x2", "mul.rn.bf16x2"):
            x, y = V(a[1]), V(a[2])
            lo = bf16_mul_bits(x & np.uint64(0xFFFF), y & np.uint64(0xFFFF)).astype(np.uint64)
            hi = bf16_mul_bits((x >> np.uint64(16)) & np.uint64(0xFFFF),
                               (y >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.uint64)
            self.setr(d, lo | (hi << np.uint64(16)), gm, 32)
            return None
        if base == "cvt":
            return self.cvt(ins, parts, a, gm)
        if base == "selp":
            p = V(a[3]).astype(bool)
            self.setr(d, np.where(p, V(a[1]), V(a[2])), gm, _width(ty))
            return None
        if base == "setp":
            return self.setp(parts, a, gm)
        if base in ("and", "or", "xor", "not") and ty == "pred":
            if base == "not":
                r = ~V(a[1]).astype(bool)
            else:
                x, y = V(a[1]).astype(bool), V(a[2]).astype(bool)
                r = x & y if base == "and" else x | y if base == "or" else x ^ y
            self.setr(d, r.astype(np.uint64), gm, 1)
            return None
        w = _width(ty)
        M = _mask_bits(w)
        signed = ty.startswith("s")

        def S(x):                                   # signed view
            v = V(x) & M
            if w == 64:
                return v.astype(np.int64)
            return ((v << np.uint64(64 - w)).astype(np.int64) >> np.int64(64 - w))

        if base in ("add", "sub", "mul", "mad", "and", "or", "xor", "shl", "shr", "neg", "not", "min", "max", "bfe",
                    "abs", "div", "rem", "prmt", "popc", "clz", "brev", "bfi", "shf", "sad"):
            if base == "add":
                r = V(a[1]) + V(a[2])
            elif base == "sub":
                r = V(a[1]) - V(a[2])
            elif base == "mul" or base == "mad":
                if "wide" in parts:
                    if signed:
                        r = (S(a[1]) * S(a[2])).astype(np.uint64)
                    else:
                        r = (V(a[1]) & M) * (V(a[2]) & M)
                    if base == "mad":
                        r = r + V(a[3])
                    self.setr(d, r, gm, 2 * w)
                    return None
                if "hi" in parts:
                    if signed:
                        r = ((S(a[1]).astype(object) * S(a[2]).astype(object)) >> w)
                        r = np.array([int(v) & ((1 << w) - 1) for v in r], dtype=np.uint64)
                    else:
                        r = ((V(a[1]) & M).astype(object) * (V(a[2]) & M).astype(object)) >> w
                        r = np.array([int(v) for v in r], dtype=np.uint64)
                else:
                    r = V(a[1]) * V(a[2])
                if base == "mad":
                    r = r + V(a[3])
            elif base == "and":
                r = V(a[1]) & V(a[2])
            elif base == "or":
                r = V(a[1]) | V(a[2])
            elif base == "xor":
                r = V(a[1]) ^ V(a[2])
            elif base == "not":
                r = ~V(a[1])
            elif base == "neg":
                r = (-S(a[1])).astype(np.uint64)
            elif base == "abs":
                r = np.abs(S(a[1])).astype(np.uint64)
            elif base == "shl":
                sh = V(a[2]) & np.uint64(0xFFFFFFFF)
                r = np.where(sh >= w, np.uint64(0), (V(a[1]) & M) << np.minimum(sh, np.uint64(63)))
            elif base == "shr":
                sh = V(a[2]) & np.uint64(0xFFFFFFFF)
                if signed:
                    r = (S(a[1]) >> np.minimum(sh, np.uint64(w - 1)).astype(np.int64)).astype(np.uint64)
                else:
                    r = np.where(sh >= w, np.uint64(0), (V(a[1]) & M) >> np.minimum(sh, np.uint64(63)))
            elif base in ("min", "max"):
                if signed:
                    x, y = S(a[1]), S(a[2])
                else:
                    x, y = V(a[1]) & M, V(a[2]) & M
                r = (np.minimum(x, y) if base == "min" else np.maximum(x, y)).astype(np.uint64)
            elif base == "bfe":
                pos = (V(a[2]) & np.uint64(0xFF)).astype(np.int64)
                ln = (V(a[3]) & np.uint64(0xFF)).astype(np.int64)
                x = (V(a[1]) & M).astype(object)
                out = []
                for xv, p, l_ in zip(x, pos, ln):
                    xv = int(xv)
                    if l_ == 0:
                        out.append(0)
                        continue
                    f = (xv >> min(int(p), w - 1)) & ((1 << int(l_)) - 1) if p < w else 0
                    if signed:
                        sb = min(int(p) + int(l_) - 1, w - 1)
                        sign = (xv >> sb) & 1
                        if sign and l_ < w:
                            f |= ((1 << w) - 1) ^ ((1 << int(l_)) - 1)
                    out.append(f)
                r = np.array(out, dtype=np.uint64)
            elif base == "div" or base == "rem":
                x, y = (S(a[1]), S(a[2])) if signed else (V(a[1]) & M, V(a[2]) & M)
                y = np.where(y == 0, 1, y)
                if signed:
                    q = np.trunc(x / y).astype(np.int64)
                    r = (q if base == "div" else x - q * y).astype(np.uint64)
                else:
                    r = (x // y) if base == "div" else (x % y)
            elif base == "prmt":
                x, y, s = V(a[1]) & np.uint64(0xFFFFFFFF), V(a[2]) & np.uint64(0xFFFFFFFF), V(a[3])
                if len(parts) > 2 and parts[1] not in ("b32",):
                    raise NotImplementedError(op)
                xy = x | (y << np.uint64(32))
                r = np.zeros(self.n, dtype=np.uint64)
                for k in range(4):
                    sel = (s >> np.uint64(4 * k)) & np.uint64(0xF)
                    byte = (xy >> ((sel & np.uint64(7)) * np.uint64(8))) & np.uint64(0xFF)
                    msb = (byte >> np.uint64(7)) & np.uint64(1)
                    byte = np.where((sel & np.uint64(8)) != 0, np.where(msb == 1, np.uint64(0xFF), np.uint64(0)), byte)
                    r |= byte << np.uint64(8 * k)
            elif base == "popc":
                x = V(a[1]) & M
                r = np.array([bin(int(v)).count("1") for v in x], dtype=np.uint64)
                self.setr(d, r, gm, 32)
                return None
            elif base == "shf":
                # shf.l/r.wrap/clamp.b32 d, a, b, c: funnel of b:a
                lo, hi, sh = V(a[1]) & np.uint64(0xFFFFFFFF), V(a[2]) & np.uint64(0xFFFFFFFF), V(a[3]) & np.uint64(
                    0xFFFFFFFF)
                sh = np.minimum(sh, np.uint64(32)) if "clamp" in parts else sh & np.uint64(31)
                cat = lo | (hi << np.uint64(32))
                if parts[1] == "l":
                    r = ((cat << sh) >> np.uint64(32)) & np.uint64(0xFFFFFFFF)
                else:
                    r = (cat >> sh) & np.uint64(0xFFFFFFFF)
            else:
                raise NotImplementedError(op)
            self.setr(d, r, gm, w)
            return None
        raise NotImplementedError(f"{op}: {ins.text}")

    def cvt(self, ins, parts, a, gm):
        V = self.val
        d, s = a[0], a[1]
        tys = [p for p in parts[1:] if re.match(r"^(bf16|[usfb]\d+)(x2)?$", p)]
        dt, st = tys[0], tys[1] if len(tys) > 1 else tys[0]
        if dt == "bf16x2" and st == "f32":
            hi = f32_to_bf16(u2f(V(a[1]) & np.uint64(0xFFFFFFFF))).astype(np.uint64)
            lo = f32_to_bf16(u2f(V(a[2]) & np.uint64(0xFFFFFFFF))).astype(np.uint64)
            self.setr(d, lo | (hi << np.uint64(16)), gm, 32)
            return None
        if dt == "bf16" and st == "f32":
            if "rn" not in parts:
                raise NotImplementedError(ins.text)
            self.setr(d, f32_to_bf16(u2f(V(s) & np.uint64(0xFFFFFFFF))).astype(np.uint64), gm, 16)
            return None
        if dt == "f32" and st == "bf16":
            self.setr(d, f2u(bf16_to_f32(V(s) & np.uint64(0xFFFF))).astype(np.uint64), gm, 32)
            return None
        if dt[0] in "usb" and st[0] in "usb":
            wd, ws = _width(dt), _width(st)
            v = V(s) & _mask_bits(ws)
            if st[0] == "s" and wd > ws:
                v = ((v << np.uint64(64 - ws)).astype(np.int64) >> np.int64(64 - ws)).astype(np.uint64)
            self.setr(d, v, gm, wd)
            return None
        if dt == "f32" and st[0] in "us":
            ws = _width(st)
            v = V(s) & _mask_bits(ws)
            if st[0] == "s":
                v = ((v << np.uint64(64 - ws)).astype(np.int64) >> np.int64(64 - ws))
            if "rn" not in parts:
                raise NotImplementedError(ins.text)
            self.setr(d, f2u(np.asarray(v, dtype=np.float64).astype(np.float32)).astype(np.uint64), gm, 32)
            return None
        if dt[0] in "us" and st == "f32":
            f = u2f(V(s) & np.uint64(0xFFFFFFFF)).astype(np.float64)
            if "rzi" in parts:
                f = np.trunc(f)
            elif "rni" in parts:
                f = np.round(f)
            elif "rmi" in parts:
                f = np.floor(f)
            elif "rpi" in parts:
                f = np.ceil(f)
            self.setr(d, np.nan_to_num(f).astype(np.int64).astype(np.uint64), gm, _width(dt))
            return None
        raise NotImplementedError(ins.text)

    def setp(self, parts, a, gm):
        V = self.val
        cmp, ty = parts[1], parts[-1]
        comb = parts[2] if len(parts) > 3 and parts[2] in ("and", "or", "xor") else None
        d = a[0]
        pd, qd = (d.split("|") + [None])[:2]
        if ty == "f32":
            x, y = u2f(V(a[1]) & np.uint64(0xFFFFFFFF)), u2f(V(a[2]) & np.uint64(0xFFFFFFFF))
            un = np.isnan(x) | np.isnan(y)
            base = {"eq": x == y, "ne": x != y, "lt": x < y, "le": x <= y, "gt": x > y, "ge": x >= y,
                    "equ": (x == y) | un, "neu": (x != y) | un, "ltu": (x < y) | un, "leu": (x <= y) | un,
                    "gtu": (x > y) | un, "geu": (x >= y) | un, "num": ~un, "nan": un}[cmp]
            if cmp == "ne":
                base = (x != y) & ~un
        else:
            w = _width(ty)
            M = _mask_bits(w)
            if ty.startswith("s"):
                sh = np.uint64(64 - w)
                x = ((V(a[1]) & M) << sh).astype(np.int64) >> np.int64(64 - w)
                y = ((V(a[2]) & M) << sh).astype(np.int64) >> np.int64(64 - w)
            else:
                x, y = V(a[1]) & M, V(a[2]) & M
            base = {"eq": x == y, "ne": x != y, "lt": x < y, "le": x <= y, "gt": x > y, "ge": x >= y,
                    "lo": x < y, "ls": x <= y, "hi": x > y, "hs": x >= y}[cmp]
        if comb:
            c = V(a[3]).astype(bool)
            p = base & c if comb == "and" else base | c if comb == "or" else base ^ c
            q = ~base & c if comb == "and" else ~base | c if comb == "or" else ~base ^ c
        else:
            p, q = base, ~base
        self.setr(pd, p.astype(np.uint64), gm, 1)
        if qd:
            self.setr(qd, q.astype(np.uint64), gm, 1)
        return None


def launch(k: Kernel, mem: Memory, args: list, grid, block: int, dyn_shared: int = 65536, order=None) -> list[CTA]:
    """Run every CTA of a grid, one after the other (in ``order``, default x-major index order). ``args``: values
    of the kernel parameters in order (addresses from ``mem.alloc``, ints, or np.float32 for f32 parameters)."""

    params = {}
    for name, v in zip(k.params, args):
        if isinstance(v, (bytes, bytearray)):
            params[name] = bytes(v)
        elif isinstance(v, np.floating) or isinstance(v, float):
            params[name] = int(np.float32(v).view(np.uint32))
        else:
            params[name] = int(v) & 0xFFFFFFFFFFFFFFFF
    gx, gy, gz = (list(grid) + [1, 1, 1])[:3]
    ids = [(x, y, z) for z in range(gz) for y in range(gy) for x in range(gx)]
    if order is not None:
        ids = [ids[i] for i in order]
    ctas = []
    for cid in ids:
        c = CTA(k, mem, params, cid, (gx, gy, gz), block, dyn_shared)
        c.run()
        ctas.append(c)
    return ctas


# ------------------------------------------------------------------------------------------------------------------
# The specification: glue._hc_post -> glue._hc_partial -> glue._hc_finish, as Triton 3.7.1 + its ptxas execute them
# on sm_121 (D = 4096, S = 4, NB = 16, SUB = 128, 4 / 4 / 8 warps), per element, per lane, in order
# ------------------------------------------------------------------------------------------------------------------
# negative controls (tests): a variant changes one detail of the specification, and must change bits somewhere
VARIANTS = ("no_contract", "collapse_a", "bfly_rev", "ss_tree", "exact_div", "post_order", "mix_tree")
# not variants, because they change no value: _hc_partial's chain starting from column 0 instead of 1, or unfused
# instead of fused square sums (products of two bf16 values are exact in fp32)
VARIANT = ""

TWO_M14 = np.float32(2.0 ** -14)          # 1 / (S D): ptxas turns div.full(x, 16384) into x * 2^-14 (exact)
TWO_M12 = np.float32(2.0 ** -12)          # 1 / D


def bfly(v: np.ndarray, levels, op=add32) -> np.ndarray:
    """A shfl.bfly reduction over the last axis: for each level L in order, v[l] = op(v[l], v[l ^ L]); every lane
    ends with the same value (op is commutative); lane 0's is returned."""

    v = np.asarray(v, dtype=np.float32)
    n = v.shape[-1]
    lane = np.arange(n)
    for L in levels:
        v = op(v, v[..., lane ^ L])
    return v[..., 0]


def hc_post_ref(x: np.ndarray, g: np.ndarray, post: np.ndarray, comb: np.ndarray) -> np.ndarray:
    """x [R, 4D] bf16 bits, g [W, R, D] fp32, post [R, 4], comb [R, 16] -> x' [R, 4D] bf16 bits.
    X_s = bf16(fma(branch, post_s, fma(x3, c3s, fma(x2, c2s, fma(x0, c0s, x1 * c1s))))), branch = bf16(g0 + g1 ...)."""

    D = g.shape[2]
    xs = [bf16_to_f32(x[:, s * D:(s + 1) * D]) for s in range(4)]
    acc = g[0].astype(np.float32)
    for k in range(1, g.shape[0]):
        acc = add32(acc, g[k])
    br = bf16r(acc)
    out = np.empty_like(x)
    for s in range(4):
        c = [comb[:, j * 4 + s][:, None].astype(np.float32) for j in range(4)]
        ps = post[:, s][:, None].astype(np.float32)
        if VARIANT == "post_order":
            v = mul32(xs[0], c[0])
            v = fma32(xs[1], c[1], v)
        else:
            v = mul32(xs[1], c[1])
            v = fma32(xs[0], c[0], v)
        v = fma32(xs[2], c[2], v)
        v = fma32(xs[3], c[3], v)
        v = fma32(br, ps, v)
        out[:, s * D:(s + 1) * D] = f32_to_bf16(v)
    return out


def chain8(xv: np.ndarray, wv: np.ndarray) -> np.ndarray:
    """_hc_partial's in-thread product sum over 8 columns (last axis): mul of column 1, then fma of 0, 2, 3, .. 7."""

    v = mul32(xv[..., 1], wv[..., 1])
    v = fma32(xv[..., 0], wv[..., 0], v)
    for j in range(2, 8):
        v = fma32(xv[..., j], wv[..., j], v)
    return v


def hc_partial_ref(xp: np.ndarray, fn: np.ndarray, NB: int = 16) -> np.ndarray:
    """xp [R, 4D] bf16 bits (the new streams), fn [24, 4D] bf16 bits -> part [R, NB, 32] fp32 (slots 25.. = NaN,
    never written). Block b: acc_m = ((+0 + T_0) + T_1) .. + T_7, T_t = butterfly(8, 4, 2, 1) over 16 lanes of
    chain8 over the lane's 8 columns of the t-th 128-column step; squares: per column k of a step, fma(x, x, s) over
    the 8 steps from +0, then butterfly(16, 8, 4, 2, 1) in each warp of 32 columns and (W0 + W2) + (W1 + W3)."""

    R, wide = xp.shape
    KB = wide // NB
    steps = KB // 128
    x = bf16_to_f32(xp).reshape(R, NB, steps, 16, 8)
    w = bf16_to_f32(fn).reshape(24, NB, steps, 16, 8)
    part = np.full((R, NB, 32), np.nan, dtype=np.float32)
    acc = np.zeros((R, NB, 24), dtype=np.float32)
    for t in range(steps):
        v = chain8(x[:, None, :, t], w[None, :, :, t])        # [R, 24, NB, 16]
        T = bfly(v, (1, 2, 4, 8) if VARIANT == "bfly_rev" else (8, 4, 2, 1))    # [R, 24, NB]
        acc = add32(acc, np.moveaxis(T, 1, 2))
    part[:, :, :24] = acc
    xk = bf16_to_f32(xp).reshape(R, NB, steps, 128)
    s = np.zeros((R, NB, 128), dtype=np.float32)
    for t in range(steps):
        s = fma32(xk[:, :, t], xk[:, :, t], s)
    W = bfly(s.reshape(R, NB, 4, 32), (16, 8, 4, 2, 1))       # [R, NB, 4]
    if VARIANT == "ss_tree":
        part[:, :, 24] = add32(add32(W[..., 0], W[..., 1]), add32(W[..., 2], W[..., 3]))
    else:
        part[:, :, 24] = add32(add32(W[..., 0], W[..., 2]), add32(W[..., 1], W[..., 3]))
    return part


# which element slots of a thread's 8 (and the next 8) start the collapse with fma(p1, x1, p0 * x0) ("A") rather
# than fma(p0, x0, p1 * x1) ("B"): LLVM's packed-fma formation in _hc_finish (Triton 3.7.1), even slots A, odd B
COLLAPSE_A = (True, False) * 8


def hc_finish_ref(xp: np.ndarray, part: np.ndarray, base: np.ndarray, scale: np.ndarray, nw: np.ndarray, eps: float,
                  hc_eps: float, iters: int, collapse_a=COLLAPSE_A):
    """-> out [R, D] bf16 bits, xs [R, D/64] fp32, post [R, 4], comb [R, 16]."""

    R, wide = xp.shape
    D = nw.shape[0]
    NB = part.shape[1]
    eps = np.float32(eps)
    hce = np.float32(hc_eps)
    mix = np.zeros((R, 32), dtype=np.float32)
    ss = np.zeros((R,), dtype=np.float32)
    if VARIANT == "mix_tree":                                  # pairwise instead of in block order
        pm = part.copy()
        while pm.shape[1] > 1:
            pm = add32(pm[:, 0::2], pm[:, 1::2])
        mix, ss = pm[:, 0, :], pm[:, 0, 24]
    else:
        for b in range(NB):
            mix = add32(mix, part[:, b, :])
            ss = add32(ss, part[:, b, 24])
    nc = VARIANT == "no_contract"
    q = add32(mul32(ss, TWO_M14), eps) if nc else fma32(ss, TWO_M14, eps)
    rinv = div_full(np.float32(1.0), mufu_sqrt(q))
    mixs = mul32(mix, rinv[:, None])
    s_pre, s_post, s_comb = (np.float32(v) for v in scale[:3])
    bse = np.zeros(32, dtype=np.float32)
    bse[:24] = base[:24]
    lane = np.arange(32)

    def pick(sc, m):
        v = fma32(mixs, sc, bse[None, :])                       # [R, 32]
        return bfly(np.where(lane[None, :] == m, v, np.float32(0.0)), (16, 8, 4, 2, 1))

    pre_l = np.stack([pick(s_pre, k) for k in range(4)], axis=1)
    post_l = np.stack([pick(s_post, 4 + k) for k in range(4)], axis=1)
    cl = np.stack([np.stack([pick(s_comb, 8 + 4 * i + j) for j in range(4)], axis=1) for i in range(4)], axis=1)
    cmax = bfly(cl, (2, 1), op=max32)                          # [R, 4] (rows i)
    ce = ex2_approx(mul32(sub32(cl, cmax[:, :, None]), LOG2E))
    rs = bfly(ce, (2, 1))                                      # [R, 4]: (c0 + c2) + (c1 + c3) over j
    a2, _, r = div_full_parts(ce, np.broadcast_to(rs[:, :, None], ce.shape))
    comb = add32(mul32(r, a2), hce) if nc else fma32(r, a2, hce)   # ptxas fuses div.full's multiply with + hc_eps

    def colsum(cm):
        return bfly(np.swapaxes(cm, 1, 2), (2, 1))             # over i: (c0 + c2) + (c1 + c3) -> [R, 4] (cols j)

    comb = div_full(comb, np.broadcast_to(add32(colsum(comb), hce)[:, None, :], comb.shape))
    for _ in range(iters - 1):
        comb = div_full(comb, np.broadcast_to(add32(bfly(comb, (2, 1)), hce)[:, :, None], comb.shape))
        comb = div_full(comb, np.broadcast_to(add32(colsum(comb), hce)[:, None, :], comb.shape))
    t = div_full(np.float32(1.0), add32(ex2_approx(mul32(sub32(np.float32(0.0), post_l), LOG2E)), np.float32(1.0)))
    post = add32(t, t)
    a2, _, r = div_full_parts(np.float32(1.0), add32(ex2_approx(mul32(sub32(np.float32(0.0), pre_l), LOG2E)),
                                                     np.float32(1.0)))
    pre = add32(mul32(r, a2), hce) if nc else fma32(r, a2, hce)
    sv = np.arange(4)
    p = [bfly(np.where(sv[None, :] == k, pre, np.float32(0.0)), (2, 1)) for k in range(4)]
    x = [bf16_to_f32(xp[:, s * D:(s + 1) * D]) for s in range(4)]
    P = [pk[:, None] for pk in p]
    va = fma32(P[1], x[1], mul32(P[0], x[0]))
    vb = fma32(P[0], x[0], mul32(P[1], x[1]))
    # element d of a row: thread d // 8 (first half) or (d - D/2) // 8, slot d % 8 (+ 8 in the second half)
    d = np.arange(D)
    slot = (d % 8) + 8 * (d >= D // 2)
    isa = np.array((True,) * 16 if VARIANT == "collapse_a" else collapse_a, dtype=bool)[slot]
    v = np.where(isa[None, :], va, vb)
    v = fma32(P[2], x[2], v)
    v = fma32(P[3], x[3], v)
    c = bf16r(v)                                               # [R, D]
    nthr = D // 16
    e = c.reshape(R, 2, nthr, 8)                               # [R, half, thread, slot]
    order = [(0, 0)] + [(0, j) for j in range(2, 8)] + [(1, j) for j in range(8)]
    s2 = mul32(e[:, 0, :, 1], e[:, 0, :, 1])
    for n_, (h, j) in enumerate(order):
        if nc and n_ >= 2 and j % 2 == 1 or nc and h == 1 and j % 2 == 1:    # the PTX's unfused odd squares
            s2 = add32(mul32(e[:, h, :, j], e[:, h, :, j]), s2)
        else:
            s2 = fma32(e[:, h, :, j], e[:, h, :, j], s2)        # (ptxas contracted the unfused odd squares too)
    Wv = bfly(s2.reshape(R, nthr // 32, 32), (16, 8, 4, 2, 1))  # [R, warps]
    tot = bfly(Wv, (4, 2, 1)) if Wv.shape[-1] == 8 else None
    q2 = add32(mul32(tot, TWO_M12), eps) if nc else fma32(tot, TWO_M12, eps)
    rinv2 = div_full(np.float32(1.0), mufu_sqrt(q2))
    yb = bf16_mul_bits(nw[None, :], f32_to_bf16(mul32(rinv2[:, None], c)))
    y = bf16_to_f32(yb).reshape(R, D // 64, 8, 8)              # [R, group, thread, slot]
    ch = add32(y[..., 0], y[..., 1])
    for j in range(2, 8):
        ch = add32(ch, y[..., j])
    xsum = bfly(ch, (4, 2, 1))                                 # [R, groups]
    return yb, xsum, post, comb.reshape(R, 16)


def hc_boundary_ref(x, g, post, comb, fn, base, scale, nw, eps, hc_eps, iters, variant: str = ""):
    """The whole boundary: hc_post then hc_pre (partial + finish): -> dict of every output. ``variant``: one of
    ``VARIANTS`` (negative controls), "" for the specification."""

    global VARIANT
    if variant and variant not in VARIANTS:
        raise ValueError(variant)
    saved, VARIANT = VARIANT, variant
    try:
        with np.errstate(invalid="ignore", over="ignore", divide="ignore", under="ignore"):
            xp = hc_post_ref(x, g, post, comb)
            part = hc_partial_ref(xp, fn)
            out, xs, npost, ncomb = hc_finish_ref(xp, part, base, scale, nw, eps, hc_eps, iters)
    finally:
        VARIANT = saved
    return {"x": xp, "part": part, "out": out, "xs": xs, "post": npost, "comb": ncomb}


# ------------------------------------------------------------------------------------------------------------------
# A lane-level port of hc_fused.cu (its CTAs, thread mapping, shuffles, smem exchanges and tickets), numpy over the
# 256 threads of a CTA: the CPU model of the CUDA kernel that tests/test_hc_fused.py compares with the specification
# above. (tests/test_hc_fused_compile.py runs the kernel's own PTX in the interpreter as well.)
# ------------------------------------------------------------------------------------------------------------------
def _shx(v: np.ndarray, m: int) -> np.ndarray:
    """__shfl_xor_sync over the CTA's lanes (arrays indexed by thread id; partners stay inside the warp)."""

    lt = np.arange(v.shape[-1])
    return v[..., lt ^ m]


def hc_cuda_port(x, g, post, comb, fn, base, scale, nw, eps, hc_eps, iters, rg: int = 4, step_order=None):
    """hc_fused.cu on numpy: -> the same dict as ``hc_boundary_ref``. ``step_order``: the order the step CTAs run in
    (a permutation; the finishers always run after their group's 32 step CTAs, as their tickets make them)."""

    x = np.array(x, dtype=np.uint16, copy=True)
    post = np.array(post, dtype=np.float32, copy=True)
    comb = np.array(comb, dtype=np.float32, copy=True)
    R = x.shape[0]
    Dm, Sm, WIDEm, NBm, KBm, STEPSm, NM = 4096, 4, 16384, 16, 1024, 8, 24
    rg = min(R, rg)
    groups = (R + rg - 1) // rg
    xp = np.zeros((R, WIDEm), dtype=np.uint16)
    ts = np.full((R, NBm, STEPSm, NM), np.nan, dtype=np.float32)
    part = np.full((R, NBm, 32), np.nan, dtype=np.float32)
    out = np.zeros((R, Dm), dtype=np.uint16)
    xs = np.zeros((R, Dm // 64), dtype=np.float32)
    cnt = np.zeros(groups, dtype=np.int64)
    lt = np.arange(256)
    fnf = bf16_to_f32(fn)

    def step_cta(grp, q, t):
        r0 = grp * rg
        nr = min(rg, R - r0)
        col0 = q * KBm + t * 128
        l, gi = lt & 15, lt >> 4
        pairs = [gi + 16 * i for i in range(6)]
        wv = [fnf[(p % NM)[:, None], ((p // NM) * Dm + col0 + l * 8)[:, None] + np.arange(8)[None, :]] for p in pairs]
        xsm = np.zeros((nr, Sm, 128), dtype=np.float32)
        c, h = lt & 127, lt >> 7
        for rr in range(nr):
            r = r0 + rr
            xr = lambda s_: bf16_to_f32(x[r, s_ * Dm + col0 + c])           # noqa: E731
            x0, x1, x2, x3 = xr(0), xr(1), xr(2), xr(3)
            acc = g[0, r, col0 + c].astype(np.float32)
            for k in range(1, g.shape[0]):
                acc = add32(acc, g[k, r, col0 + c])
            branch = bf16r(acc)
            for e in range(2):
                s = 2 * h + e
                c0, c1, c2, c3 = (comb[r, j * 4 + s] for j in range(4))
                ps = post[r, s]
                v = mul32(x1, c1)
                v = fma32(x0, c0, v)
                v = fma32(x2, c2, v)
                v = fma32(x3, c3, v)
                v = fma32(branch, ps, v)
                vb = f32_to_bf16(v)
                xsm[rr, s, c] = bf16_to_f32(vb)
                xp[r, s * Dm + col0 + c] = vb
        for rr in range(nr):
            r = r0 + rr
            for i, p in enumerate(pairs):
                s, m = p // NM, p % NM
                xv = xsm[rr, s[:, None], (l * 8)[:, None] + np.arange(8)[None, :]]
                v = mul32(xv[:, 1], wv[i][:, 1])
                v = fma32(xv[:, 0], wv[i][:, 0], v)
                for j in range(2, 8):
                    v = fma32(xv[:, j], wv[i][:, j], v)
                for mm in (8, 4, 2, 1):
                    v = add32(v, _shx(v, mm))
                sel = l == 0
                ts[r, (4 * s + q)[sel], t, m[sel]] = v[sel]
        cnt[grp] += 1

    def finish_cta(r):
        assert cnt[r // rg] == 32
        lane, warp = lt & 31, lt >> 5
        psm = np.zeros((NBm, 32), dtype=np.float32)
        wsm = np.zeros((NBm, 4), dtype=np.float32)
        xv = [[xp[r, s * Dm + 8 * lt[:, None] + np.arange(8)[None, :]],
               xp[r, s * Dm + Dm // 2 + 8 * lt[:, None] + np.arange(8)[None, :]]] for s in range(Sm)]
        for e in range(2):
            j = lt + 256 * e
            ok = j < NBm * NM
            b_, m_ = j[ok] // NM, j[ok] % NM
            acc = np.zeros(ok.sum(), dtype=np.float32)
            for u in range(STEPSm):
                acc = add32(acc, ts[r, b_, u, m_])
            psm[b_, m_] = acc
            part[r, b_, m_] = acc
        k, bh = lt & 127, lt >> 7
        for i in range(8):
            b_ = bh + 2 * i
            s = np.zeros(256, dtype=np.float32)
            for u in range(STEPSm):
                xk = bf16_to_f32(xp[r, b_ * KBm + u * 128 + k])
                s = fma32(xk, xk, s)
            for mm in (16, 8, 4, 2, 1):
                s = add32(s, _shx(s, mm))
            sel = lane == 0
            wsm[b_[sel], (k >> 5)[sel]] = s[sel]
        for bb in range(NBm):
            v = add32(add32(wsm[bb, 0], wsm[bb, 2]), add32(wsm[bb, 1], wsm[bb, 3]))
            psm[bb, NM] = v
            part[r, bb, NM] = v
        for s in range(Sm):
            x[r, s * Dm + 8 * lt[:, None] + np.arange(8)[None, :]] = xv[s][0]
            x[r, s * Dm + Dm // 2 + 8 * lt[:, None] + np.arange(8)[None, :]] = xv[s][1]
        mix = np.zeros(256, dtype=np.float32)
        ss = np.zeros(256, dtype=np.float32)
        for bb in range(NBm):
            mix = add32(mix, psm[bb, lane])
            ss = add32(ss, psm[bb, NM])
        rinv = div_full(np.float32(1.0), mufu_sqrt(fma32(ss, TWO_M14, np.float32(eps))))
        mixs = mul32(mix, rinv)
        bse = np.where(lane < NM, np.concatenate([base[:NM], np.zeros(8, np.float32)])[np.minimum(lane, 31)], 0)
        bse = bse.astype(np.float32)

        def pick(sc, target):
            v = np.where(lane == target, fma32(mixs, np.float32(sc), bse), np.float32(0.0))
            for mm in (16, 8, 4, 2, 1):
                v = add32(v, _shx(v, mm))
            return v

        pl = pick(scale[0], warp & 3)
        ql = pick(scale[1], (warp & 3) + 4)
        c1 = pick(scale[2], 8 + warp)
        c2 = pick(scale[2], 16 + warp)
        lsm = np.zeros((3, 4, 4), dtype=np.float32)
        for wv_ in range(8):
            t0 = wv_ * 32
            if wv_ < 4:
                lsm[0, 0, wv_] = pl[t0]
                lsm[1, 0, wv_] = ql[t0]
            lsm[2, wv_ >> 2, wv_ & 3] = c1[t0]
            lsm[2, (wv_ >> 2) + 2, wv_ & 3] = c2[t0]
        ci, cj = (lane >> 2) & 3, lane & 3
        pre_l, post_l = lsm[0, 0, lt & 3], lsm[1, 0, lt & 3]
        cl = lsm[2, ci, cj]
        cmax = max32(cl, _shx(cl, 2))
        cmax = max32(cmax, _shx(cmax, 1))
        ce = ex2_approx(mul32(sub32(cl, cmax), LOG2E))
        rs = add32(ce, _shx(ce, 2))
        rs = add32(rs, _shx(rs, 1))
        a2, _, rr_ = div_full_parts(ce, rs)
        cm = fma32(rr_, a2, np.float32(hc_eps))
        cs = add32(cm, _shx(cm, 8))
        cs = add32(cs, _shx(cs, 4))
        cm = div_full(cm, add32(cs, np.float32(hc_eps)))
        for _ in range(1, iters):
            rs = add32(cm, _shx(cm, 2))
            rs = add32(rs, _shx(rs, 1))
            cm = div_full(cm, add32(rs, np.float32(hc_eps)))
            cs = add32(cm, _shx(cm, 8))
            cs = add32(cs, _shx(cs, 4))
            cm = div_full(cm, add32(cs, np.float32(hc_eps)))
        tp = div_full(np.float32(1.0), add32(ex2_approx(mul32(sub32(np.float32(0.0), post_l), LOG2E)),
                                             np.float32(1.0)))
        pst = add32(tp, tp)
        a2, _, rr_ = div_full_parts(np.float32(1.0), add32(ex2_approx(mul32(sub32(np.float32(0.0), pre_l), LOG2E)),
                                                           np.float32(1.0)))
        pre = fma32(rr_, a2, np.float32(hc_eps))
        post[r, :] = pst[:4]
        comb[r, :] = cm[:16]
        pk = []
        for kk in range(4):
            v = np.where((lt & 3) == kk, pre, np.float32(0.0))
            v = add32(v, _shx(v, 2))
            v = add32(v, _shx(v, 1))
            pk.append(v)
        cv = np.zeros((256, 16), dtype=np.float32)
        for hf in range(2):
            for j in range(8):
                a = [bf16_to_f32(xv[s][hf][:, j]) for s in range(Sm)]
                if j & 1:
                    v = fma32(pk[0], a[0], mul32(pk[1], a[1]))
                else:
                    v = fma32(pk[1], a[1], mul32(pk[0], a[0]))
                v = fma32(pk[2], a[2], v)
                v = fma32(pk[3], a[3], v)
                cv[:, 8 * hf + j] = bf16r(v)
        s2 = mul32(cv[:, 1], cv[:, 1])
        s2 = fma32(cv[:, 0], cv[:, 0], s2)
        for j in range(2, 16):
            s2 = fma32(cv[:, j], cv[:, j], s2)
        for mm in (16, 8, 4, 2, 1):
            s2 = add32(s2, _shx(s2, mm))
        wq = s2[::32].copy()
        v = np.zeros(256, dtype=np.float32)
        v[:8] = wq
        for mm in (4, 2, 1):
            v = add32(v, _shx(v, mm))
        rinv2 = div_full(np.float32(1.0), mufu_sqrt(fma32(v[0], TWO_M12, np.float32(eps))))
        yb = np.zeros((256, 16), dtype=np.uint16)
        for hf in range(2):
            wbits = nw[hf * (Dm // 2) + 8 * lt[:, None] + np.arange(8)[None, :]]
            yb[:, 8 * hf:8 * hf + 8] = bf16_mul_bits(wbits, f32_to_bf16(mul32(rinv2, cv[:, 8 * hf:8 * hf + 8])))
        out[r, 8 * lt[:, None] + np.arange(8)[None, :]] = yb[:, :8]
        out[r, Dm // 2 + 8 * lt[:, None] + np.arange(8)[None, :]] = yb[:, 8:]
        for hf in range(2):
            y = bf16_to_f32(yb[:, 8 * hf:8 * hf + 8])
            s = add32(y[:, 0], y[:, 1])
            for j in range(2, 8):
                s = add32(s, y[:, j])
            for mm in (4, 2, 1):
                s = add32(s, _shx(s, mm))
            sel = (lt & 7) == 0
            xs[r, hf * (Dm // 128) + (lt[sel] >> 3)] = s[sel]

    steps = [(grp, q, t) for grp in range(groups) for q in range(4) for t in range(8)]
    if step_order is not None:
        steps = [steps[i] for i in step_order]
    done_groups = set()
    for grp, q, t in steps:
        step_cta(grp, q, t)
        if cnt[grp] == 32 and grp not in done_groups:
            done_groups.add(grp)
            for r in range(grp * rg, min(R, (grp + 1) * rg)):
                finish_cta(r)
    return {"x": x, "part": part, "out": out, "xs": xs, "post": post, "comb": comb}


# ------------------------------------------------------------------------------------------------------------------
# inputs and comparison
# ------------------------------------------------------------------------------------------------------------------
KINDS = ("random", "zeros", "extreme", "tiny", "nonfinite", "real", "collapse")


def make_inputs(R: int, seed: int = 0, kind: str = "random", world: int = 2) -> dict:
    """A boundary's inputs (numpy; bf16 as uint16 bits). Kinds:
    random     normal streams / partials, small fn, post in [0, 2), comb in [0, 0.5)
    zeros      signed zeros everywhere they can be (streams, partials, comb, post, fn rows), a few ones
    extreme    large fn / base / scale: logits far past +-87 (ex2's halving path, div.full's 1/4 scaling), comb
               logits spread so that Sinkhorn entries underflow, huge partials (bf16 rounding to inf)
    tiny       subnormal streams and partials (bf16 and fp32 subnormals, square sums below 2^-126)
    nonfinite  a few inf / NaN streams and partials
    real       the model's scales: streams ~ N(0, 1) with a few large channels, fn ~ N(0, 0.03), base ~ N(0, 1),
               scale ~ (1, 1, 1), norm weights ~ 1 +- 0.2, post ~ 2 sigmoid, comb a Sinkhorn-like 4 x 4
    """

    rng = np.random.default_rng(seed + 1000 * KINDS.index(kind))
    D_, W_ = 4096, 16384

    def bf(a):
        return f32_to_bf16(np.asarray(a, dtype=np.float32))

    x = rng.standard_normal((R, W_)).astype(np.float32)
    g = (rng.standard_normal((world, R, D_)) * 0.5).astype(np.float32)
    post = (rng.random((R, 4)) * 2).astype(np.float32)
    comb = (rng.random((R, 16)) * 0.5).astype(np.float32)
    fn = rng.standard_normal((24, W_)) * 0.02
    base = (rng.standard_normal(24) * 0.5).astype(np.float32)
    scale = np.array([0.7, 1.3, 2.1], dtype=np.float32)
    nw = 1 + 0.1 * rng.standard_normal(D_)
    if kind == "zeros":
        x = np.where(rng.random(x.shape) < 0.5, -0.0, 0.0).astype(np.float32)
        x[:, ::97] = 1.0
        g = np.where(rng.random(g.shape) < 0.5, -0.0, 0.0).astype(np.float32)
        g[..., ::31] = -1.0
        comb[:, ::3] = -0.0
        comb[:, 1::5] = 0.0
        post[:, 0] = -0.0
        fn[::4] = 0.0
        fn[1::4] *= -0.0
        base[::5] = -0.0
    elif kind == "extreme":
        # logits around the thresholds of the unit sequences: ex2 inputs just past -126 (halving) and 1 + ex2(.) past
        # 2^126 (div.full's 1/4 scaling); Sinkhorn entries that underflow; large streams and partials
        x *= 3.0
        x[:, ::41] = 3.0e4
        g *= 50.0
        g[..., ::53] = 1.0e30
        fn = rng.standard_normal((24, W_)) * 1e-4
        base = np.array([-88.0, 87.5, -87.9, 100.0, -88.2, -150.0, 88.5, -87.6]
                        + list(rng.standard_normal(16) * 60.0), dtype=np.float32)
        scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        post = (rng.standard_normal((R, 4)) * 40).astype(np.float32)
        comb = (rng.standard_normal((R, 16)) * 30).astype(np.float32)
    elif kind == "tiny":
        x = (rng.standard_normal((R, W_)) * 1e-39).astype(np.float32)
        x[:, ::7] *= 1e-3
        g = (rng.standard_normal((world, R, D_)) * 1e-40).astype(np.float32)
        comb = (rng.random((R, 16)) * 1e-20).astype(np.float32)
        post = (rng.random((R, 4)) * 1e-20).astype(np.float32)
        fn = rng.standard_normal((24, W_)) * 1e-30
        nw = 1e-30 * (1 + rng.random(D_))
    elif kind == "nonfinite":
        x[:, 5] = np.inf
        x[:, 4100] = -np.inf
        x[0, 9000] = np.nan
        g[0, :, 17] = np.inf
        g[-1, :, 18] = np.nan
    elif kind == "real":
        x = rng.standard_normal((R, W_)).astype(np.float32)
        x[:, rng.choice(W_, 40, replace=False)] *= 40.0
        g = (rng.standard_normal((world, R, D_)) * 0.3).astype(np.float32)
        fn = rng.standard_normal((24, W_)) * 0.03
        base = rng.standard_normal(24).astype(np.float32)
        scale = (1.0 + 0.1 * rng.standard_normal(3)).astype(np.float32)
        nw = 1 + 0.2 * rng.standard_normal(D_)
        post = (2.0 / (1.0 + np.exp(-rng.standard_normal((R, 4))))).astype(np.float32)
        m = rng.random((R, 4, 4)) + 0.05
        for _ in range(5):
            m /= m.sum(axis=2, keepdims=True)
            m /= m.sum(axis=1, keepdims=True)
        comb = m.reshape(R, 16).astype(np.float32)
    if kind == "collapse":
        # fn = 0, so the mixes are the bases and the collapse weights p_k = sigmoid(base_k) + eps do not depend on
        # the streams; then every column gets stream values (x0, x1, 0, 0) whose collapse p0 x0 + p1 x1 rounds to
        # different bf16 values depending on which product is rounded first (fma(p1, x1, p0 x0) vs fma(p0, x0,
        # p1 x1)): the streams must reproduce _hc_finish's per-slot choice to get its bits
        fn = np.zeros((24, W_))
        g = np.zeros((world, R, D_), dtype=np.float32)
        comb = np.zeros((R, 16), dtype=np.float32)
        for s_ in range(4):
            comb[:, s_ * 4 + s_] = 1.0
        post = np.zeros((R, 4), dtype=np.float32)
        p = _pre_weights(base, scale)
        x0s, x1s, x2s = _tie_pairs(p, rng, D_)
        x = np.zeros((R, W_), dtype=np.float32)
        x[:, 0:D_] = x0s
        x[:, D_:2 * D_] = x1s
        x[:, 2 * D_:3 * D_] = x2s
        x = x.astype(np.float32)
    return {"x": bf(x), "g": g.astype(np.float32), "post": post.astype(np.float32), "comb": comb.astype(np.float32),
            "fn": bf(fn), "base": base.astype(np.float32), "scale": scale.astype(np.float32), "nw": bf(nw),
            "eps": np.float32(1e-5), "hc_eps": np.float32(1e-6), "iters": 20}


def _pre_weights(base, scale) -> list:
    """The collapse weights p_0..3 of a row whose mixing dots are all zero (fn = 0): _hc_finish's pre."""

    pre_l = add32(fma32(np.float32(0.0), np.float32(scale[0]), base[:4]), np.float32(0.0))
    a2, _, r = div_full_parts(np.float32(1.0), add32(ex2_approx(mul32(sub32(np.float32(0.0), pre_l), LOG2E)),
                                                     np.float32(1.0)))
    return list(fma32(r, a2, np.float32(1e-6)))


def _tie_pairs(p, rng, n: int):
    """n stream triples (x0, x1, x2) (bf16 values; x3 = 0) for which the collapse rounds to different bf16 values
    with fma(p1, x1, p0 x0) and with fma(p0, x0, p1 x1) as its first step (then + p2 x2 + p3 * 0)."""

    p0, p1, p2, p3 = (np.float32(v) for v in p)
    allv = bf16_to_f32(np.arange(0x3C00, 0x4300, dtype=np.uint16))     # 2^-7 .. 128, positive
    xs0 = rng.choice(allv, 256)
    xs1 = rng.choice(allv, 256)
    va = fma32(p1, xs1, mul32(p0, xs0))
    vb = fma32(p0, xs0, mul32(p1, xs1))
    keep = va != vb
    trip = []
    x2all = np.concatenate([allv, -allv])
    for x0, x1, a_, b_ in zip(xs0[keep], xs1[keep], va[keep], vb[keep]):
        ca = fma32(p3, np.float32(0.0), fma32(p2, x2all, a_))
        cb = fma32(p3, np.float32(0.0), fma32(p2, x2all, b_))
        hit = f32_to_bf16(ca) != f32_to_bf16(cb)
        trip += [(x0, x1, x2) for x2 in x2all[hit][:4]]
        if len(trip) >= 64:
            break
    if not trip:
        raise RuntimeError("no tie triples")
    t = np.array(trip, dtype=np.float32)[rng.integers(0, len(trip), n)]
    return t[:, 0], t[:, 1], t[:, 2]


def same(a, b) -> bool:
    """Bitwise equality, all NaNs equal (the hardware makes canonical NaNs; the payload is not an output)."""

    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype == np.uint16:
        fa, fb = bf16_to_f32(a), bf16_to_f32(b)
        na, nb = np.isnan(fa), np.isnan(fb)
        return bool(np.array_equal(na, nb) and np.array_equal(a[~na], b[~nb]))
    fa, fb = a.astype(np.float32), b.astype(np.float32)
    na, nb = np.isnan(fa), np.isnan(fb)
    return bool(np.array_equal(na, nb) and np.array_equal(fa[~na].view(np.uint32), fb[~nb].view(np.uint32)))


OUTPUTS = ("x", "part", "out", "xs", "post", "comb")


def diff(got: dict, ref: dict) -> list[str]:
    """The outputs that differ (the partial buffer's slots 0-24 only: 25-31 are never written)."""

    bad = []
    for k in OUTPUTS:
        a, b = got[k], ref[k]
        if k == "part":
            a, b = np.asarray(a)[:, :, :25], np.asarray(b)[:, :, :25]
        if not same(a, b):
            bad.append(k)
    return bad


def ref_of(inp: dict, **kw) -> dict:
    return hc_boundary_ref(inp["x"], inp["g"], inp["post"], inp["comb"], inp["fn"], inp["base"], inp["scale"],
                           inp["nw"], inp["eps"], inp["hc_eps"], inp["iters"], **kw)


def port_of(inp: dict, rg: int = 4, step_order=None) -> dict:
    return hc_cuda_port(inp["x"], inp["g"], inp["post"], inp["comb"], inp["fn"], inp["base"], inp["scale"], inp["nw"],
                        inp["eps"], inp["hc_eps"], inp["iters"], rg=rg, step_order=step_order)


# ------------------------------------------------------------------------------------------------------------------
# running the compiled kernels' PTX on an input set
# ------------------------------------------------------------------------------------------------------------------
def _outputs(mem: Memory, a: dict, R: int) -> dict:
    return {"x": mem.get(a["x"], np.uint16, (R, 16384)), "part": mem.get(a["part"], np.float32, (R, 16, 32)),
            "out": mem.get(a["out"], np.uint16, (R, 4096)), "xs": mem.get(a["xs"], np.float32, (R, 64)),
            "post": mem.get(a["post"], np.float32, (R, 4)), "comb": mem.get(a["comb"], np.float32, (R, 16))}


def _alloc_io(mem: Memory, inp: dict) -> dict:
    R = inp["x"].shape[0]
    return {"x": mem.alloc(inp["x"]), "g": mem.alloc(inp["g"]), "post": mem.alloc(inp["post"]),
            "comb": mem.alloc(inp["comb"]), "fn": mem.alloc(inp["fn"]),
            "part": mem.alloc(np.full((R, 16, 32), np.nan, np.float32)), "base": mem.alloc(inp["base"]),
            "scale": mem.alloc(inp["scale"]), "nw": mem.alloc(inp["nw"]),
            "out": mem.alloc(np.zeros((R, 4096), np.uint16)), "xs": mem.alloc(np.zeros((R, 64), np.float32)),
            "dummy": mem.alloc(np.zeros(4096, np.uint8))}


def run_triton(post_k: Kernel, partial_k: Kernel, finish_k: Kernel, inp: dict) -> dict:
    """glue._hc_post (grid (R, D / 1024), 4 warps) -> _hc_partial ((R, 16), 4 warps) -> _hc_finish ((R,), 8 warps), in
    place on x, as glue.hc_post + glue.hc_pre launch them. Pass ``ptxas_view`` kernels for ptxas's semantics."""

    R = inp["x"].shape[0]
    mem = Memory()
    a = _alloc_io(mem, inp)
    z = a["dummy"]
    launch(post_k, mem, [a["x"], a["x"], a["g"], a["post"], a["comb"], R * 4096, z, z][:len(post_k.params)], (R, 4),
           128)
    launch(partial_k, mem, [a["x"], a["fn"], a["part"], z, z][:len(partial_k.params)], (R, 16), 128)
    launch(finish_k, mem, [a["x"], a["part"], a["base"], a["scale"], a["nw"], a["out"], a["xs"], a["post"], a["comb"],
                           np.float32(inp["eps"]), np.float32(inp["hc_eps"]), z, z][:len(finish_k.params)], (R,), 256)
    return _outputs(mem, a, R)


def run_cuda(k: Kernel, inp: dict, rg: int = 4, order=None, pdl: bool = False) -> dict:
    """hc_fused.cu's kernel (its PTX), grid 32 G + R CTAs of 256 threads (``order``: a permutation of the step CTAs;
    the finishers run last, which their tickets allow)."""

    import struct

    R = inp["x"].shape[0]
    mem = Memory()
    a = _alloc_io(mem, inp)
    xp = mem.alloc(np.zeros((64, 16384), np.uint16))
    ts = mem.alloc(np.full((64, 16, 8, 24), np.nan, np.float32))
    cnt = mem.alloc(np.zeros(65, np.int32))
    rg = min(R, rg)
    groups = (R + rg - 1) // rg
    world = inp["g"].shape[0]
    args = struct.pack("<QQqi4xQQQQQQQQQQQQiiiiffi4x", a["x"], a["g"], R * 4096, world, a["post"], a["comb"],
                       a["fn"], a["base"], a["scale"], a["nw"], a["out"], a["xs"], a["part"], xp, ts, cnt, R, rg,
                       groups, int(inp["iters"]), float(inp["eps"]), float(inp["hc_eps"]), int(pdl))
    nstep = 32 * groups
    ids = list(range(nstep)) if order is None else [int(i) for i in order]
    launch(k, mem, [args], (nstep + R,), 256, order=ids + list(range(nstep, nstep + R)))
    o = _outputs(mem, a, R)
    o["tickets"] = mem.get(cnt, np.int32, (65,))
    return o
