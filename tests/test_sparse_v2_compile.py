"""patches/0410: the v2 sparse latent attention kernel (``sparse_v2._lsparse_v2``, Gluon) and the one-pass kernel it
replaces (``b12x_attn._lsparse_one``) compiled for sm_121 (GB10) WITHOUT a GPU, under the Triton the tree runs on (the
image's 3.7.1, or 3.8), and the compiled IR compared where the bits are decided:

- layouts: v2's softmax / PV layout is the reference's ``#mma`` exactly (as compiled by this Triton), so the two
  ``tt.reduce`` ops (max, and the non-associative sum) lower to the same tree; the QK dot is ``nvidia_mma`` v2
  ``instrShape [16, 8]`` ([1, 8] or [2, 4]); every dot operand ``kWidth = 2``;
- the key loop's arithmetic, op for op in program order, with result types resolved (layout aliases expanded):
  identical to the reference's, except the QK dot's parent layout when QKL = 1;
- the chains: the QK dot's accumulator is a zero constant, the PV dot's is ``arith.mulf`` of the loop-carried o (the
  Combine fold of ``o * alpha + dot``), in both;
- PTX: the same count of every floating-point instruction class (fma / mul / add / sub / div / ex2 / max / cvt / selp
  on floats / setp on floats / packed x2 forms), the mma count 96 (QKL 0) or 64 (QKL 1) vs the reference's 96; no
  unpaired arithmetic, gathers as ``cp.async.cg`` 16-byte copies with a src-size operand (zero-fill of masked rows);
- fit (asserted under 3.7.x, printed otherwise): the default configurations use <= 255 registers with no spills and
  <= 99 KB of shared memory, one 8-warp CTA an SM like the reference.

Not the interpreter: run in its own process without TRITON_INTERPRET.
    PYTHONPATH=<patched tree>/src pytest -q -s tests/test_sparse_v2_compile.py      (~1-2 minutes)
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
torch = pytest.importorskip("torch")
pytest.importorskip("triton.experimental.gluon")
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402
from triton.experimental.gluon._runtime import GluonASTSource  # noqa: E402

from tensorfold.families.glm5_next.cuda import b12x_attn, sparse_v2  # noqa: E402

TGT = GPUTarget("cuda", 121, 32)
FIT = triton.__version__.startswith("3.7.")
SMEM_MAX = 101376


def _ordered(fn, sig):
    return {n: sig[n] for n in fn.arg_names}


def _attrs(fn, sig):
    return {(fn.arg_names.index(n),): [["tt.divisibility", 16]] for n, t in sig.items() if t.startswith("*")}


def _sig(fp8, paged):
    sig = {"QA": "*bf16", "LC": "*u8" if fp8 else "*bf16", "LCS": "*fp32" if fp8 else "*bf16", "TOK": "*i32",
           "CNT": "*i32", "OUT": "*fp32"}
    cst = {"H": 32, "L": 512, "KTS": 32, "SCALE": 0.0625, "FP8": fp8, "RB": 528 if fp8 else 512,
           "SB": 132 if fp8 else 0, "SA": 128, "PSH": 8 if paged else 0}
    if paged:
        sig["PT"] = "*i32"
    else:
        cst["PT"] = None
    return sig, cst


def _compile_ref(fp8, paged=False):
    sig, cst = _sig(fp8, paged)
    cst.update(BMQ=32, OPQ=fp8, W=2051)
    fn = b12x_attn._lsparse_one
    for k in cst:
        sig[k] = "constexpr"
    sig = _ordered(fn, sig)
    return triton.compile(ASTSource(fn=fn, signature=sig, constexprs=cst, attrs=_attrs(fn, sig)), target=TGT,
                          options={"num_warps": b12x_attn.WARPS, "num_stages": b12x_attn.STAGES})


def _compile_v2(fp8, paged=False, stages=2, qkl=1, qreg=0):
    sig, cst = _sig(fp8, paged)
    sig["W"] = "i32"
    cst.update(STAGES=stages, QKL=qkl if fp8 else 0, QREG=qreg if fp8 and qkl else 0)
    fn = sparse_v2._lsparse_v2
    for k in cst:
        sig[k] = "constexpr"
    sig = _ordered(fn, sig)
    return triton.compile(GluonASTSource(fn=fn, signature=sig, constexprs=cst, attrs=_attrs(fn, sig)), target=TGT,
                          options={"num_warps": sparse_v2.WARPS})


def _ptxas(ptx):
    d = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin")
    exe = next((os.path.join(d, x) for x in ("ptxas-blackwell", "ptxas") if os.path.exists(os.path.join(d, x))), None)
    if exe is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".ptx", mode="w", delete=False) as f:
        f.write(ptx)
    r = subprocess.run([exe, "-arch=sm_121a", "-v", f.name, "-o", os.devnull], capture_output=True, text=True)
    os.unlink(f.name)
    regs = int(re.search(r"Used (\d+) registers", r.stderr).group(1))
    sp = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", r.stderr)
    return regs, int(sp.group(1)), int(sp.group(2))


# -- TTGIR ------------------------------------------------------------------------------------------------------------
def _aliases(ttgir):
    out = {}
    for line in ttgir.splitlines():
        m = re.match(r"^(#[\w]+) = (.*)$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _resolve(text, al):
    return re.sub(r"#[A-Za-z_][\w]*", lambda m: al.get(m.group(0), m.group(0)), text)


def _mma_defs(ttgir):
    return sorted(v for v in _aliases(ttgir).values() if "nvidia_mma" in v)


FLOAT_OPS = ("tt.dot", "arith.mulf", "arith.addf", "arith.subf", "arith.divf", "arith.maxnumf", "arith.cmpf",
             "arith.truncf", "arith.extf", "math.exp", "math.exp2", "tt.fp_to_fp", "tt.reduce.return")


def _arith(ttgir, qk_generic=False):
    """The float arithmetic from the key loop to the end of the kernel, in order: (op, resolved result / operand types)
    -- ``arith.select`` on float values and ``tt.reduce`` headers included. ``qk_generic``: the first dot (QK) keeps
    only its operand kWidths and shapes, not its parent layout."""

    al = _aliases(ttgir)
    lines = ttgir.splitlines()
    i0 = next(i for i, x in enumerate(lines) if "scf.for" in x and "iter_args" in x and "f32" in x)
    out, dots = [], 0
    for x in lines[i0:]:
        s = re.sub(r" loc\(.*?\)$", "", x.strip())
        if s.startswith("}) : ("):                            # a tt.reduce's operand / result types
            out.append(("tt.reduce", _resolve(s[len("}) : "):], al)))
            continue
        m = re.match(r"^(?:%[\w#:]+ = )?\"?([a-z_]+\.[a-z_.]+)\"?", s)
        if not m:
            continue
        op = m.group(1)
        if op == "tt.reduce":
            continue
        if op == "arith.select":
            ty = s.split(" : ", 1)[1]
            if "f32" in ty or "bf16" in ty:
                out.append((op, _resolve(ty, al)))
            continue
        if op not in FLOAT_OPS:
            continue
        ty = _resolve(s.split(" : ", 1)[1], al) if " : " in s else ""
        if op in ("arith.maxnumf", "arith.addf") and "tensor" not in ty:
            ty = "scalar " + ty
        if op == "tt.dot" and "-> tensor<32x32xf32" in ty:     # QK: one chain over K, maybe cut in K halves
            if qk_generic:
                ty = re.sub(r"nvidia_mma<\{[^}]*\}>", "MMA", ty)
            k = int(re.search(r"tensor<32x(\d+)xbf16", ty).group(1))
            ty = re.sub(r"tensor<32x\d+xbf16", "tensor<32xKxbf16", re.sub(r"tensor<\d+x32xbf16", "tensor<Kx32xbf16", ty))
            if out and out[-1][0] == "tt.dot QK" and out[-1][1][0] == ty:
                out[-1] = ("tt.dot QK", (ty, out[-1][1][1] + k))
            else:
                out.append(("tt.dot QK", (ty, k)))
            continue
        out.append((op, ty))
    return out


def _dot_lines(ttgir):
    return [x for x in ttgir.splitlines() if " tt.dot " in x]


def _defining(ttgir, name):
    return next(x for x in ttgir.splitlines() if re.match(rf"\s*{re.escape(name)} = ", x))


# -- PTX --------------------------------------------------------------------------------------------------------------
def _fclasses(ptx):
    c = collections.Counter()
    for line in ptx.splitlines():
        s = line.strip()
        m = re.match(r"(?:@!?%p\d+\s+)?([a-z][\w.]*)", s)
        if not m:
            continue
        op = m.group(1)
        head = op.split(".")[0]
        if head in ("fma", "div", "rcp", "ex2", "lg2", "sqrt", "mma", "shfl"):
            c[op] += 1
        elif head in ("add", "mul", "sub", "max", "min", "neg", "abs", "selp", "setp") and re.search(r"f32|f16|bf16", op):
            c[op] += 1
        elif head == "cvt" and re.search(r"f32|f16|bf16|e4m3", op):
            c[op] += 1
    return c


# (fp8, paged, stages, qkl, qreg)
CFGS = [(True, False, 3, 1, 1), (True, True, 3, 1, 1), (True, False, 2, 1, 1), (True, False, 2, 1, 0),
        (True, True, 2, 1, 0), (True, False, 2, 0, 0), (False, False, 2, 0, 0), (False, True, 2, 0, 0)]
IDS = lambda c: f"{'fp8' if c[0] else 'bf16'}{'-paged' if c[1] else ''}-s{c[2]}q{c[3]}r{c[4]}"  # noqa: E731


@pytest.fixture(scope="module")
def k():
    ks = {("ref", fp8, paged): _compile_ref(fp8, paged) for fp8 in (True, False) for paged in (False, True)}
    for fp8, paged, st, qkl, qreg in CFGS:
        ks[("v2", fp8, paged, st, qkl, qreg)] = _compile_v2(fp8, paged, st, qkl, qreg)
    return ks


def _ref(k, key):
    return k[("ref", key[1], key[2])]


@pytest.mark.parametrize("cfg", CFGS, ids=IDS)
def test_v2_softmax_layout_is_the_reference_mma(k, cfg):
    key = ("v2",) + cfg
    ref, v2 = _ref(k, key).asm["ttgir"], k[key].asm["ttgir"]
    rm = _mma_defs(ref)
    assert len(rm) == 1 and "warpsPerCTA = [1, 8]" in rm[0], rm
    assert rm[0] in _mma_defs(v2)
    for t in (ref, v2):
        assert set(re.findall(r"kWidth = \d+", t)) == {"kWidth = 2"}
        assert all("instrShape = [16, 8]" in d and "versionMajor = 2" in d for d in _mma_defs(t))
    if cfg[3]:
        assert any("warpsPerCTA = [2, 4]" in d for d in _mma_defs(v2))


@pytest.mark.parametrize("cfg", CFGS, ids=IDS)
def test_v2_loop_arithmetic_is_the_reference(k, cfg):
    key = ("v2",) + cfg
    ref, v2 = _ref(k, key).asm["ttgir"], k[key].asm["ttgir"]
    a, b = _arith(ref, qk_generic=True), _arith(v2, qk_generic=True)
    assert len(a) >= 25
    assert a == b, next((i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y) if len(a) == len(b) else (len(a), len(b))


@pytest.mark.parametrize("cfg", CFGS, ids=IDS)
def test_chains_start_where_the_reference_starts(k, cfg):
    """QK: one chain from a zero constant (with QREG two dots, the second continuing the first's result: MMAv2's
    lowering walks K in ascending k16 steps from the dot's c, so the chain is the single dot's); PV: from o * alpha."""

    key = ("v2",) + cfg
    for t in (_ref(k, key).asm["ttgir"], k[key].asm["ttgir"]):
        dl = _dot_lines(t)
        *qks, pv = dl
        assert len(qks) == (2 if t is not _ref(k, key).asm["ttgir"] and cfg[4] else 1)
        c_qk = re.search(r"tt\.dot %[\w#]+, %[\w#]+, (%[\w#]+)", qks[0]).group(1)
        for a, b in zip(qks, qks[1:]):
            res = re.match(r"\s*(%[\w#]+) = tt\.dot", a).group(1)
            assert re.search(r"tt\.dot %[\w#]+, %[\w#]+, (%[\w#]+)", b).group(1) == res
        c_pv = re.search(r"tt\.dot %[\w#]+, %[\w#]+, (%[\w#]+)", pv).group(1)
        assert "arith.constant dense<0.000000e+00>" in _defining(t, c_qk)
        d = _defining(t, c_pv)
        assert "arith.mulf" in d                              # c = o * alpha: the Combine fold of o * alpha + dot
        assert "x512xf32" in d


@pytest.mark.parametrize("cfg", CFGS, ids=IDS)
def test_ptx_float_instructions_match(k, cfg):
    key = ("v2",) + cfg
    a, b = _fclasses(_ref(k, key).asm["ptx"]), _fclasses(k[key].asm["ptx"])
    mma = [x for x in a if x.startswith("mma")]
    assert mma == ["mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"]
    assert a[mma[0]] == 96 and b[mma[0]] == (64 if cfg[3] else 96)
    for op in set(a) | set(b):
        if not op.startswith("mma"):
            assert a[op] == b[op], (op, a[op], b[op])
    for op in b:                                              # IEEE round-to-nearest only (no rz / ftz variants)
        assert ".rz" not in op and ".ftz" not in op, op


def test_v2_gathers_are_zero_filling_cp_async(k):
    ptx = k[("v2", True, False, 3, 1, 1)].asm["ptx"]
    rows = re.findall(r"cp\.async\.cg\.shared\.global \[[^]]+\], \[[^]]+\], 0x10, %r\d+;", ptx)
    assert len(rows) == 12                                    # 4 16-byte chunks a thread a tile: 2 prologue + loop
    assert "cp.async.wait_group" in ptx and "cp.async.commit_group" in ptx


def test_fit(k, capsys):
    rows = []
    for key, kern in k.items():
        r = _ptxas(kern.asm["ptx"])
        rows.append((key, r, kern.metadata.shared))
    with capsys.disabled():
        print(f"\n[sparse v2] Triton {triton.__version__}")
        for key, r, sh in rows:
            print(f"  {str(key):36s} regs/spill st/ld {r}  smem {sh}")
    for key, r, sh in rows:
        if key[0] == "v2":
            assert sh == sparse_v2.smem_need(key[1], key[3], key[4], key[5]), (key, sh)
    if not FIT:
        return
    for key, r, sh in rows:
        if key[0] == "v2" and tuple(key[3:]) == sparse_v2.config(key[1]):     # the default configurations
            assert r is None or (r[0] <= 255 and r[1] == 0 and r[2] == 0), (key, r)
            assert sh <= SMEM_MAX, (key, sh)


def test_config_and_mode(monkeypatch):
    monkeypatch.delenv(sparse_v2.ENV, raising=False)
    monkeypatch.delenv(sparse_v2.CFG_ENV, raising=False)
    assert sparse_v2.mode() is False
    assert sparse_v2.config(True) == (3, 1, 1) and sparse_v2.config(False) == (2, 0, 0)
    monkeypatch.setenv(sparse_v2.ENV, "1")
    assert sparse_v2.mode() is True
    monkeypatch.setenv(sparse_v2.ENV, "2")
    with pytest.raises(ValueError):
        sparse_v2.mode()
    monkeypatch.setenv(sparse_v2.CFG_ENV, "2,0")
    assert sparse_v2.config(True) == (2, 0, 0) and sparse_v2.config(False) == (2, 0, 0)
    monkeypatch.setenv(sparse_v2.CFG_ENV, "2,1,1")
    assert sparse_v2.config(True) == (2, 1, 1) and sparse_v2.config(False) == (2, 0, 0)
    for bad in ("1,1", "3,1", "3,1,0", "4,1,1", "2,0,1", "2,2", "x"):   # 3 stages with q in smem: 115 KB
        monkeypatch.setenv(sparse_v2.CFG_ENV, bad)
        with pytest.raises(ValueError):
            sparse_v2.config(True)


def test_sparse_latent_one_dispatch(monkeypatch):
    """b12x_attn: v2 only when on (``V2`` / ``v2=True``) at the production shape with the default tile; ``bm`` /
    ``stages`` / ``v2=False`` keep the reference (benches compare the two)."""

    seen = []
    monkeypatch.setattr(sparse_v2, "sparse_latent_v2", lambda *a, **k: seen.append("v2"))

    class Fake:
        def __getitem__(self, grid):
            return lambda *a, **k: seen.append("one")

    monkeypatch.setattr(b12x_attn, "_lsparse_one", Fake())
    qa = torch.zeros(3, 32, 512, dtype=torch.bfloat16)
    args = (qa, torch.zeros(64, 512, dtype=torch.bfloat16), torch.zeros(3, 8, dtype=torch.int32),
            torch.ones(3, dtype=torch.int32), torch.zeros(3, 32, 512), 0.06)
    monkeypatch.setattr(b12x_attn, "V2", None)
    b12x_attn.sparse_latent_one(*args)
    b12x_attn.sparse_latent_one(*args, v2=True)
    monkeypatch.setattr(b12x_attn, "V2", sparse_v2)
    b12x_attn.sparse_latent_one(*args)
    b12x_attn.sparse_latent_one(*args, bm=32)
    b12x_attn.sparse_latent_one(*args, v2=False)
    b12x_attn.sparse_latent_one(qa[:, :16].contiguous(), *args[1:4], torch.zeros(3, 16, 512), 0.06)
    assert seen == ["one", "v2", "v2", "one", "one", "one"]
