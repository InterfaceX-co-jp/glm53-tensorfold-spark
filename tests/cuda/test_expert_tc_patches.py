"""patches/0330: the ``tc`` routed-expert kernels (``GLM53_TF_FAST_EXPERTS=tc``, ``exl3_tc.cu``): fat's arithmetic in a
warp-specialized persistent kernel. One producer warp a CTA claims items by ticket, gathers member rows with cp.async and
copies trellis words with cp.async.bulk into an NSA-stage ring guarded by mbarriers (full: 32 cp.async arrivals + the
bulk bytes; empty: one arrival a consumer warp); 16 consumer warps (8 in configuration 3) run fat's inner loop; the
epilogue has its own shared buffer; item info goes through a 2-deep mbarrier queue. Items are (expert, pass of 64 MH
members, NBC column blocks).

It must equal fast2 / fat BIT FOR BIT, be row-independent (patches/0085) and deterministic.

Checked:

- host only (no GPU, no torch needed for the model):
  * the helpers copied from exl3_fast.cu (mcg2, decode_tile, mma16816, bf16r, fwht_row, swz) are the same text, and
    the epilogue's arithmetic lines are fat's;
  * the env switch (``tc`` is a fat-family mode; TC_CFG / TC_TICKET / TC_CTAS parsed and refused) and
    ``exl3_mm.routed``'s dispatch with fake extensions (tc on the shared input, fat when gate / up differ, fast2 inside
    auto's window, fast2 when the per-request knob says 0);
  * a Python model of the kernel's index arithmetic, for every configuration and odd shapes: the items cover every
    (expert, pass, column-block group) once (static stride and ticket walks); every (expert, member, matrix, column) is
    computed by exactly one (warp, lane, fragment) and its B values are the member's row at the right k (the producer's
    swizzled cp.async destinations read back through the consumers' ldmatrix addresses, ldmatrix and m16n8k16
    fragment semantics); the A operand is fat's trellis word for that (matrix, k tile, column tile); the epilogue
    writes every output (row, column) of every present block once, from the right accumulator; zero-filled rows are
    never stored;
  * the shared-memory and register budgets (<= 99 KB a CTA, 2 CTAs an SM for the 8-warp configuration, the
    __maxnreg__ cap times the threads within the register file);
  * the mbarrier protocol, simulated with random interleavings of the producer and consumer warps and random
    cp.async / bulk completion times (phases, parities, arrival counts, tx bytes): no deadlock, a consumer reads a
    stage only after every byte of it landed, the producer refills a slot only after every consumer released it, item
    info is never overwritten while in use, every warp terminates; and the control: a protocol with one arrival too few
    on "empty" is caught;
- GPU kernels: tc == fast2 == fat bit for bit (Xd and Y; configurations 0-3; ticket on / off; a CTA cap; 64-8192
  rows, uniform and skewed; real per-rank shapes and a small shape with 3 column blocks and many passes); row
  independence (row subsets and permutations); repeatable; the timing probes launch;
- GPU engine (synthetic EXL3 checkpoint, gate / up sign vectors equal as in the real one): committed state tc == fast2
  (lean and not); drafted == serial and resumed == fresh with tc; tc and fast2 resume each other's snapshots.

Run inside the image:
    PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q -s tests/cuda/test_expert_tc_patches.py
Host-only parts anywhere with numpy + pytest (TF_SRC=<patched tree>/src for the source-text checks).
"""

from __future__ import annotations

import heapq
import importlib
import os
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

try:
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:          # the host-only model tests still run
    torch = None
    CUDA = False

gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")
needs_torch = pytest.mark.skipif(torch is None, reason="torch")

D, NI, E, TOP, LIMIT = 4096, 1024, 288, 8, 10.0
NB, MTL, NG, WPM = 8, 2, 8, 4


def _src_dir() -> Path | None:
    cands = [os.environ.get("TF_SRC", ""), "/src/TensorFold/src"]
    try:
        import tensorfold

        cands.insert(0, str(Path(tensorfold.__file__).parent.parent))
    except ImportError:
        pass
    for c in cands:
        p = Path(c) / "tensorfold/families/glm5_next/cuda"
        if c and (p / "exl3_tc.cu").exists():
            return p
    return None


SRC = _src_dir()
needs_src = pytest.mark.skipif(SRC is None, reason="the patched TensorFold source (TF_SRC)")


# -- host only: source text ------------------------------------------------------------------------------------------
def _body(text: str, signature: str) -> str:
    """The text of the function whose definition starts with ``signature``, up to its matching brace."""
    i = text.index(signature)
    j = text.index("{", i)
    depth = 0
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[i:k + 1]
    raise AssertionError(signature)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


@needs_src
@pytest.mark.parametrize("sig", [
    "__device__ __forceinline__ uint32_t mcg2(",
    "__device__ __forceinline__ void decode_tile(",
    "__device__ __forceinline__ void mma16816(",
    "__device__ __forceinline__ float bf16r(",
    "__device__ __forceinline__ void fwht_row(",
    "__device__ __forceinline__ int swz(",
])
def test_helpers_verbatim(sig):
    fast = (SRC / "exl3_fast.cu").read_text()
    tcs = (SRC / "exl3_tc.cu").read_text()
    assert _norm(_body(tcs, sig)) == _norm(_body(fast, sig)), sig


def _arith(text: str) -> list[str]:
    """The epilogue's arithmetic lines (every line using HAD_SCALE, expf or bf16r(...) in an expert kernel)."""
    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        if ("HAD_SCALE" in ln or "expf(" in ln or "bf16r(bf16r" in ln or "sdd[" in ln) and "constexpr" not in ln \
                and not ln.startswith("//"):
            ln = re.sub(r"\bjj\b", "j", ln)
            out.append(_norm(ln))
    return out


@needs_src
def test_epilogue_arithmetic_is_fats():
    fast = (SRC / "exl3_fast.cu").read_text()
    tcs = (SRC / "exl3_tc.cu").read_text()
    fat = fast[fast.index("namespace fat {"):fast.index("// (MTL, NG, KS, NSA, NGR) as fast2's configurations")]
    fat = _body(fat, "__global__ void __launch_bounds__")
    mine = _body(tcs, "__global__ void __maxnreg__")
    a, b = _arith(fat), _arith(mine)
    assert a and a == b, (a, b)


# -- host only: plumbing ---------------------------------------------------------------------------------------------
_ENVS = ("GLM53_TF_FAST_EXPERTS", "GLM53_TF_TC_CFG", "GLM53_TF_TC_TICKET", "GLM53_TF_TC_CTAS", "GLM53_TF_ONCE_PAIR",
         "GLM53_TF_FAT_STAGES")


@needs_torch
def test_env_modes(monkeypatch):
    from tensorfold.families.glm5_next.cuda import exl3_mm

    try:
        for mode, fat, tc in (("tc", True, True), ("TC", True, True), ("fat", True, False), ("once", True, False),
                              ("fast2", False, False), (None, False, False)):
            for k in _ENVS:
                monkeypatch.delenv(k, raising=False)
            if mode is not None:
                monkeypatch.setenv("GLM53_TF_FAST_EXPERTS", mode)
            importlib.reload(exl3_mm)
            assert (exl3_mm.FAT, exl3_mm.TC, exl3_mm.TC_CFG, exl3_mm.TC_TICKET, exl3_mm.TC_CTAS) == \
                (fat, tc, (0, 0), True, 0), mode
            assert exl3_mm.family() == int(fat)
        monkeypatch.setenv("GLM53_TF_FAST_EXPERTS", "tc")
        monkeypatch.setenv("GLM53_TF_TC_CFG", "2,3")
        monkeypatch.setenv("GLM53_TF_TC_TICKET", "0")
        monkeypatch.setenv("GLM53_TF_TC_CTAS", "40")
        importlib.reload(exl3_mm)
        assert (exl3_mm.TC_CFG, exl3_mm.TC_TICKET, exl3_mm.TC_CTAS) == ((2, 3), False, 40)
        for bad in (("GLM53_TF_TC_CFG", "4,0"), ("GLM53_TF_TC_CFG", "1"), ("GLM53_TF_TC_CTAS", "-1")):
            monkeypatch.setenv(*bad)
            with pytest.raises(ValueError):
                importlib.reload(exl3_mm)
            monkeypatch.delenv(bad[0])
    finally:
        for k in _ENVS:
            monkeypatch.delenv(k, raising=False)
        importlib.reload(exl3_mm)


class _FakeExt:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def f(*a):
            self.calls.append((name, a))
        return f


@needs_torch
@pytest.mark.parametrize("fat,tc,auto,shared", [(True, True, False, True), (True, True, False, False),
                                                (False, True, False, True), (True, True, True, True),
                                                (True, False, False, True)])
def test_routed_dispatch(monkeypatch, fat, tc, auto, shared):
    """TC runs inside the fat family (tf_knobs.fat_experts = 1), on the one rotated input; fat when gate and up have
    different sign vectors; fast2 when the knob says 0 or auto's window holds the chunk."""

    from tensorfold.families.glm5_next.cuda import exl3_mm

    ext, fx, tx = _FakeExt(), _FakeExt(), _FakeExt()
    monkeypatch.setattr(exl3_mm, "_ext", lambda: ext)
    monkeypatch.setattr(exl3_mm, "_fast_ext", lambda: fx)
    monkeypatch.setattr(exl3_mm, "_tc_ext", lambda: tx)
    monkeypatch.setattr(exl3_mm, "V2", None)
    monkeypatch.setattr(exl3_mm, "FAT", fat)
    monkeypatch.setattr(exl3_mm, "TC", tc)
    monkeypatch.setattr(exl3_mm, "ONCE", False)
    monkeypatch.setattr(exl3_mm, "AUTO", auto)
    monkeypatch.setattr(exl3_mm, "FAST2_ROWS", (0, 4096))
    monkeypatch.setattr(exl3_mm, "FAT_SHARED_X", True)
    monkeypatch.setattr(exl3_mm, "TC_CFG", (1, 2))
    monkeypatch.setattr(exl3_mm, "TC_TICKET", True)
    monkeypatch.setattr(exl3_mm, "TC_CTAS", 7)
    suh = torch.ones((2, 256), dtype=torch.float16)
    z = lambda *s: torch.zeros(s, dtype=torch.float16)            # noqa: E731
    ex = exl3_mm.Exl3Experts(torch.zeros(1), torch.zeros(1), torch.zeros(1), suh, suh.clone() if shared else -suh,
                             z(2, 128), z(2, 128), z(2, 128), z(2, 256), 2, 128, 256)
    s = NS(slots=3, rows=4, xg=z(12, 256), xu=z(12, 256), xd=z(12, 128))
    grp = NS(ids=torch.zeros(3, dtype=torch.int32), count=torch.ones(1, dtype=torch.int32),
             members=torch.full((3, 4), -1, dtype=torch.int32))
    x = torch.zeros((4, 256), dtype=torch.bfloat16)
    exl3_mm.routed(x, torch.zeros((4, 3), dtype=torch.int32), grp, ex, s, torch.zeros((12, 256)), 4, LIMIT, fast=True)
    names = [c[0] for c in ext.calls + fx.calls + tx.calls]
    if not fat:
        assert names == ["rot_in", "gateup", "down"]
    elif auto:
        assert names == ["rot_in1", "gateup", "down"]
    elif not tc:
        assert names == ["rot_in1", "gateup_fat", "down_fat"]
    elif not shared:
        assert names == ["rot_in", "gateup_fat", "down_fat"]
    else:
        assert names == ["rot_in1", "gateup_tc", "down_tc"]
        gu, dn = tx.calls[0][1], tx.calls[1][1]
        assert gu[0] is s.xg and gu[-3:] == (1, True, 7)
        assert dn[0] is s.xd and dn[-3:] == (2, True, 7)


# -- host only: a Python model of exl3_tc.cu ---------------------------------------------------------------------------
# (MH, NBC, KS, NSA, NGR) as exl3_tc.cu's TC_GU* / TC_DN* macros
CFGS = {
    "gu1": (2, 1, 2, 4, 3, 1), "gu2": (2, 2, 1, 4, 3, 1), "gu3": (2, 1, 1, 2, 4, 1),
    "dn1": (1, 1, 4, 4, 3, 1), "dn2": (1, 2, 2, 4, 3, 1), "dn3": (1, 1, 2, 2, 4, 1),
}


@needs_src
def test_model_configs_match_the_source():
    txt = (SRC / "exl3_tc.cu").read_text()
    for name, (mats, *rest) in CFGS.items():
        macro = f"#define TC_{name[:2].upper()}{name[2]} " + ", ".join(str(v) for v in rest)
        assert macro in txt, macro


def _cfg(mats, mh, nbc, ks, nsa, ngr):
    c = NS(MATS=mats, MH=mh, NBC=nbc, KS=ks, NSA=nsa, NGR=ngr)
    c.BM = 64 * mh
    c.CW = mats * WPM * nbc * mh
    c.THREADS = (c.CW + 1) * 32
    c.MINB = 2 if c.CW <= 8 else 1
    c.REGS = (65536 // (c.THREADS * c.MINB)) // 8 * 8
    c.CPR, c.LDA = ks * 2, ks * 16
    c.A_BYTES = c.BM * c.LDA * 2
    c.W_CHUNK = NB * 32 * 4
    c.W_CHUNKS = mats * nbc * ks
    c.W_BYTES = c.W_CHUNKS * c.W_CHUNK
    c.STAGE = c.A_BYTES + c.W_BYTES
    c.RING = nsa * c.STAGE
    c.ER, c.LDE, c.ROUNDS = ngr * 8, 132, NG // ngr
    c.EPI = mh * mats * nbc * c.ER * c.LDE * 4
    c.SMEM = c.RING + c.EPI
    c.STATIC = 8 * (2 * nsa + 4) + 4 * 2 * c.BM + 4 * 2 * 4        # barriers, rows_sh[2][BM], info[2][4]
    return c


def _swz(cpr, r, c):
    return (c ^ (r & 7)) if cpr == 8 else (c ^ ((r >> 1) & 3))


@pytest.mark.parametrize("name", list(CFGS))
def test_model_budgets(name):
    c = _cfg(*CFGS[name])
    assert c.SMEM + c.STATIC <= 101376, (name, c.SMEM + c.STATIC)          # 99 KB a CTA on sm_120 / sm_121
    assert c.MINB * (c.SMEM + c.STATIC + 1024) <= 102400, name             # 100 KB an SM, 1 KB reserved a CTA
    assert c.REGS >= 104 and c.REGS * c.THREADS * c.MINB <= 65536, (name, c.REGS)
    assert c.W_CHUNKS <= 32 and c.A_BYTES % 128 == 0 and c.STAGE % 128 == 0
    assert (c.BM * c.CPR) % 32 == 0 and c.BM % 32 == 0                     # every producer lane the same trip count


def _plan(counts, BM):
    passes = [-(-n // BM) for n in counts]
    first = np.concatenate([[0], np.cumsum(passes)]).astype(int)
    return int(first[-1]), first                        # plan[0], plan[1 + u] (plan[2 + u] = first[u + 1])


def _walk(items, first, maxu):
    """The producer's u walk over its ascending items: (u, pass) a pass index tt."""
    u = 0
    out = []
    for tt in items:
        while u + 1 < maxu and first[u + 1] <= tt:
            u += 1
        out.append((u, tt - first[u]))
    return out


def _counts(rows, kind, rng, e=E, top=TOP):
    w = np.ones(e) if kind == "uniform" else 1.0 / np.arange(1, e + 1) ** 0.8
    w = w / w.sum()
    picks = np.stack([rng.choice(e, size=top, replace=False, p=w) for _ in range(rows)])
    counts = np.bincount(picks.flatten(), minlength=e)
    return picks, counts


@pytest.mark.parametrize("name", list(CFGS))
@pytest.mark.parametrize("nblk", [1, 3, 8, 32])
def test_model_items_cover_every_pass_once(name, nblk):
    c = _cfg(*CFGS[name])
    rng = np.random.default_rng(3)
    counts = [0, 5, 64, 65, 128, 129, 300, 0, 1]
    total_p, first = _plan(counts, c.BM)
    NBI = -(-nblk // c.NBC)
    total = total_p * NBI
    maxu = len(counts)
    want = {(u, p, nbi) for u, n in enumerate(counts) for p in range(-(-n // c.BM)) for nbi in range(NBI)}
    for grid in (1, 5, 48):
        # static stride: CTA b runs b, b + grid, ...; ticket: items handed out in order to random CTAs
        per_cta = {b: list(range(b, total, grid)) for b in range(grid)}
        order = list(range(total))
        tick = {b: [] for b in range(grid)}
        for it in order:
            tick[int(rng.integers(grid))].append(it)
        for sched in (per_cta, tick):
            got = []
            for b, its in sched.items():
                tts = [it // NBI for it in its]
                for (u, p), it in zip(_walk(tts, first, maxu), its):
                    assert 0 <= p < -(-counts[u] // c.BM), (u, p)
                    got.append((u, p, it % NBI))
            assert len(got) == len(set(got)) and set(got) == want


def _ldsm4(lane_addr, smem_read):
    """ldmatrix .x4 .b16: lanes 8i..8i+7 give the row addresses of matrix i; register i of lane L holds matrix i's row
    L / 4, elements 2 (L % 4) and + 1. -> regs[lane][i] = (value0, value1)."""
    regs = [[None] * 4 for _ in range(32)]
    for i in range(4):
        rows = [smem_read(lane_addr[8 * i + j], 8) for j in range(8)]          # 8 halves (16 bytes) a row
        for L in range(32):
            r = rows[L // 4]
            regs[L][i] = (r[2 * (L % 4)], r[2 * (L % 4) + 1])
    return regs


def _mma_b(b0, b1, lane):
    """m16n8k16 B fragment: b0 = (k 2t, 2t + 1; column n = g), b1 = (k 2t + 8, 2t + 9; column g)."""
    g, t = lane >> 2, lane & 3
    return {(2 * t, g): b0[0], (2 * t + 1, g): b0[1], (2 * t + 8, g): b1[0], (2 * t + 9, g): b1[1]}


@pytest.mark.parametrize("name", list(CFGS))
def test_model_b_operand_is_the_members_rows(name):
    """Producer: row r's 16-byte chunk c of stage s -> ring half offset r LDA + swz(r, c) 8 (row = rows_sh[r]; zero
    past the count). Consumer (mh, np, kk, lane): ldmatrix at row mh 64 + 16 np + lr, chunk swz(lr, 2 kk + lc). Every
    B value an mma of n group ng of a warp of group mh gets must be X[row of member mh 64 + 8 ng + n][s KS 16 + kk 16 +
    k]; symbolic values ('x', row, k)."""
    c = _cfg(*CFGS[name])
    rng = np.random.default_rng(11)
    for cnt in (c.BM, c.BM - 3, 17, 1):
        rows_sh = [int(v) for v in rng.permutation(10 * c.BM)[:c.BM]]
        rows_sh = [r if i < cnt else -1 for i, r in enumerate(rows_sh)]
        for s in (0, 1, 5):
            ring = {}
            k0 = s * c.KS * 16
            for i in range(c.BM * c.CPR):                                    # every producer lane's cp.async
                r, ch = i // c.CPR, i % c.CPR
                row = rows_sh[r]
                for h in range(8):
                    ring[r * c.LDA + _swz(c.CPR, r, ch) * 8 + h] = ("x", row, k0 + ch * 8 + h) if row >= 0 else 0.0
            read = lambda a, n: [ring[a + h] for h in range(n)]              # noqa: E731
            for mh in range(c.MH):
                cw = cnt - mh * 64
                for kk in range(c.KS):
                    for np_ in range(NG // 2):
                        if not 16 * np_ < cw:
                            continue
                        addr = []
                        for lane in range(32):
                            lr = (lane & 7) + ((lane >> 4) << 3)
                            lc = (lane >> 3) & 1
                            chunk = _swz(c.CPR, lr, kk * 2 + lc)
                            addr.append(mh * 64 * c.LDA + (16 * np_ + lr) * c.LDA + chunk * 8)
                        regs = _ldsm4(addr, read)
                        for half_, (i0, i1) in enumerate(((0, 1), (2, 3))):  # n groups 2 np and 2 np + 1
                            ng = 2 * np_ + half_
                            for lane in range(32):
                                for (k, n), v in _mma_b(regs[lane][i0], regs[lane][i1], lane).items():
                                    m = mh * 64 + 8 * ng + n
                                    want = ("x", rows_sh[m], k0 + kk * 16 + k) if rows_sh[m] >= 0 else 0.0
                                    assert v == want, (name, cnt, s, mh, kk, ng, lane, k, n, v, want)


@pytest.mark.parametrize("name", list(CFGS))
@pytest.mark.parametrize("nblk", [1, 3, 8, 32])
def test_model_a_operand_is_fats_word(name, nblk):
    """Bulk chunk (m, b, kk) of stage s: words of k tile s KS + kk, column tiles (nbi NBC + b) NB .. + NB - 1 of matrix
    m, at stage word ((m NBC + b) KS + kk) NB 32. The consumer (mat, b, slice) reads word ((mat NBC + b) KS) NB 32 +
    slice MTL 32 + lane + (kk NB + l) 32: it must be fat's word for column tile nb NB + slice MTL + l, k tile s KS + kk
    (tw + ((kt NTILES + l) 32) in fat), for every present block, and absent blocks are never read."""
    c = _cfg(*CFGS[name])
    NTILES = nblk * NB
    NBI = -(-nblk // c.NBC)
    for nbi in range(NBI):
        nbn = min(c.NBC, nblk - nbi * c.NBC)
        for s in (0, 3):
            stage = {}
            for lane in range(32):
                if lane < c.W_CHUNKS and (lane // c.KS) % c.NBC < nbn:
                    kk, b, m = lane % c.KS, (lane // c.KS) % c.NBC, lane // (c.KS * c.NBC)
                    src = ((s * c.KS + kk) * NTILES + (nbi * c.NBC + b) * NB) * 32
                    dst = ((m * c.NBC + b) * c.KS + kk) * NB * 32
                    for w in range(NB * 32):
                        stage[dst + w] = (m, src + w)
            assert len(stage) * 4 == c.MATS * nbn * c.KS * c.W_CHUNK         # the expect_tx bytes
            for warp in range(c.CW):
                slice_, b = warp % WPM, (warp // WPM) % c.NBC
                mat = (warp // (WPM * c.NBC)) % c.MATS
                if nbi * c.NBC + b >= nblk:
                    continue                                                  # cw = 0: never reads
                nb = nbi * c.NBC + b
                for lane in range(32):
                    base = (mat * c.NBC + b) * c.KS * NB * 32 + slice_ * MTL * 32 + lane
                    for kk in range(c.KS):
                        for l in range(MTL):
                            got = stage[base + (kk * NB + l) * 32]
                            kt = s * c.KS + kk
                            want = (mat, (kt * NTILES + nb * NB + slice_ * MTL + l) * 32 + lane)
                            assert got == want, (name, nbi, s, warp, lane, kk, l)


@pytest.mark.parametrize("name", list(CFGS))
@pytest.mark.parametrize("nblk", [1, 3, 8])
def test_model_every_output_once(name, nblk):
    """Accumulator acc[l][ng][c] of consumer (mh, mat, b, slice), lane (g, t): column slice 32 + l 16 + g (+ 8 for
    c >= 2) of block nb0 + b, member mh 64 + 8 ng + 2 t + (c & 1) (m16n8 C fragment). The epilogue writes it at
    ep[((mh MATS + mat) NBC + b) ER + 8 n + 2 t (+ 1)][slice 32 + l 16 + g (+ 8)] in round ng / NGR; row task
    (th, tb, r) reads ep[((th MATS + m) NBC + tb) ER + r][:] and stores output row rows_sh[th 64 + ER rd + r], block
    nb0 + tb. Every (member < cnt, matrix, column of a present block) must reach its own row once."""
    c = _cfg(*CFGS[name])
    NBI = -(-nblk // c.NBC)
    for cnt in (c.BM, c.BM - 5, 9, 1):
        rows_sh = [1000 + i if i < cnt else -1 for i in range(c.BM)]
        for nbi in range(NBI):
            nb0 = nbi * c.NBC
            written = {}
            for rd in range(c.ROUNDS):
                ep = {}
                for warp in range(c.CW):
                    slice_, b = warp % WPM, (warp // WPM) % c.NBC
                    mat, mh = (warp // (WPM * c.NBC)) % c.MATS, warp // (WPM * c.NBC * c.MATS)
                    cw = cnt - mh * 64 if nb0 + b < nblk else 0
                    if not c.ER * rd < cw:
                        continue
                    for lane in range(32):
                        g, t = lane >> 2, lane & 3
                        for n in range(c.NGR):
                            ng = rd * c.NGR + n
                            for l in range(MTL):
                                for cc in range(4):
                                    col = slice_ * 16 * MTL + l * 16 + g + (8 if cc >= 2 else 0)
                                    member = mh * 64 + 8 * ng + 2 * t + (cc & 1)
                                    erow = ((mh * c.MATS + mat) * c.NBC + b) * c.ER + 8 * n + 2 * t + (cc & 1)
                                    key = (erow, col)
                                    assert key not in ep
                                    ep[key] = (mat, nb0 + b, member, col)
                for warp in range(c.CW):
                    for task in range(warp, c.MH * c.NBC * c.ER, c.CW):
                        th, tb, r = task // (c.NBC * c.ER), (task // c.ER) % c.NBC, task % c.ER
                        if c.ER * rd + r >= cnt - th * 64 or nb0 + tb >= nblk:
                            continue
                        row = rows_sh[th * 64 + c.ER * rd + r]
                        assert row >= 0
                        for m in range(c.MATS):
                            erow = ((th * c.MATS + m) * c.NBC + tb) * c.ER + r
                            for col in range(128):
                                mat, nb, member, cl = ep[(erow, col)]
                                assert (mat, nb, cl) == (m, nb0 + tb, col) and rows_sh[member] == row
                                key = (row, m, nb, col)
                                assert key not in written
                                written[key] = True
            present = [nb for nb in range(nb0, min(nb0 + c.NBC, nblk))]
            assert len(written) == cnt * c.MATS * len(present) * 128, (name, cnt, nbi)


# -- host only: the mbarrier protocol, simulated ---------------------------------------------------------------------
class _Bar:
    """An mbarrier: expected arrivals a phase, pending arrivals, tx count, the phase number."""

    def __init__(self, count):
        self.count, self.pending, self.tx, self.phase = count, count, 0, 0

    def done(self, parity):             # try_wait.parity: the phase of that parity has completed
        return (self.phase & 1) != parity

    def _maybe_flip(self):
        if self.pending == 0 and self.tx == 0:
            self.phase += 1
            self.pending = self.count

    def arrive(self, tx=0):
        assert self.pending > 0, "arrival beyond the expected count"
        self.tx += tx
        self.pending -= 1
        self._maybe_flip()

    def complete_tx(self, n):
        self.tx -= n
        self._maybe_flip()


def _simulate(c, S, items_total, seed, empty_count=None, max_steps=400_000):
    """One CTA: the producer warp and CW consumer warps as generators, scheduled at random; async copies land after
    random delays. Returns the number of items the consumers finished. Asserts the hazards."""
    rnd = random.Random(seed)
    IQ = 2
    full = [_Bar(33) for _ in range(c.NSA)]
    empty = [_Bar(c.CW if empty_count is None else empty_count) for _ in range(c.NSA)]
    ifull = [_Bar(32) for _ in range(IQ)]
    iempty = [_Bar(c.CW) for _ in range(IQ)]
    landed = [None] * c.NSA               # (item, stage) whose bytes are all in the slot
    writing = [None] * c.NSA              # (item, stage) being written
    readers = [set() for _ in range(c.NSA)]
    info = [None] * IQ
    info_users = [set() for _ in range(IQ)]
    events = []                           # (time, seq, fn) async completions
    clock = [0]
    seq = [0]

    def later(fn):
        seq[0] += 1
        heapq.heappush(events, (clock[0] + rnd.randint(1, 40), seq[0], fn))

    def producer():
        g = 0
        for j in range(items_total + 1):
            q = j % IQ
            while not iempty[q].done(((j // IQ) & 1) ^ 1):
                yield
            assert not info_users[q], "item info overwritten while in use"
            if j == items_total:
                info[q] = None
                for _ in range(32):
                    ifull[q].arrive()
                return
            info[q] = j
            for _ in range(32):
                ifull[q].arrive()
            for s in range(S):
                slot = g % c.NSA
                while not empty[slot].done(((g // c.NSA) & 1) ^ 1):
                    yield
                assert not readers[slot], "slot refilled while a consumer reads it"
                writing[slot] = (j, s)
                landed[slot] = None
                parts = [0]
                need = 32 + c.W_CHUNKS

                def land(slot=slot, js=(j, s), parts=parts, need=need):
                    parts[0] += 1
                    if parts[0] == need:
                        landed[slot] = js
                full[slot].arrive(tx=c.W_BYTES)                     # lane 0's arrive.expect_tx
                for _ in range(c.W_CHUNKS):                          # the bulk copies
                    later(lambda slot=slot, land=land: (land(), full[slot].complete_tx(c.W_CHUNK)))
                for _ in range(32):                                  # each lane's cp.async, then its noinc arrival
                    later(lambda slot=slot, land=land: (land(), full[slot].arrive()))
                g += 1
                yield

    done_items = [0]

    def consumer(w):
        g = 0
        j = 0
        while True:
            q = j % IQ
            while not ifull[q].done((j // IQ) & 1):
                yield
            if info[q] is None:
                return
            item = info[q]
            info_users[q].add(w)
            for s in range(S):
                slot = g % c.NSA
                while not full[slot].done((g // c.NSA) & 1):
                    yield
                assert landed[slot] == (item, s), ("read before the bytes landed", landed[slot], (item, s))
                readers[slot].add(w)
                yield                                                # the warp computes on the stage
                assert landed[slot] == (item, s), "stage changed under a reader"
                readers[slot].discard(w)
                empty[slot].arrive()
                g += 1
            yield                                                    # epilogue
            info_users[q].discard(w)
            iempty[q].arrive()
            if w == 0:
                done_items[0] += 1
            j += 1

    warps = [producer()] + [consumer(w) for w in range(c.CW)]
    alive = list(range(len(warps)))
    steps = 0
    while alive:
        steps += 1
        assert steps < max_steps, "deadlock (no progress)"
        if events and (rnd.random() < 0.3 or len(alive) == 0):
            _, _, fn = heapq.heappop(events)
            clock[0] += 1
            fn()
            continue
        i = rnd.choice(alive)
        try:
            next(warps[i])
        except StopIteration:
            alive.remove(i)
        clock[0] += 1
    while events:
        _, _, fn = heapq.heappop(events)
        fn()
    return done_items[0]


@pytest.mark.parametrize("name", ["gu1", "gu3", "dn2"])
@pytest.mark.parametrize("seed", range(6))
def test_model_protocol_random_interleavings(name, seed):
    c = _cfg(*CFGS[name])
    c.CW = min(c.CW, 6)                   # fewer consumer warps keep the simulation quick; the protocol is the same
    for S, items in ((c.NSA, 3), (5, 4), (16, 3)):
        assert _simulate(c, S, items, seed * 100 + S) == items


def test_model_protocol_control_is_caught():
    """One arrival too few expected on "empty": the producer may refill a slot a consumer still reads."""
    c = _cfg(*CFGS["gu1"])
    c.CW = 4
    caught = 0
    for seed in range(40):
        try:
            _simulate(c, 7, 3, seed, empty_count=c.CW - 1)
        except AssertionError:
            caught += 1
    assert caught > 0


# -- GPU: kernels ----------------------------------------------------------------------------------------------------
def _layer(shared: bool = True, d=D, ni=NI, e=E, seed=5):
    sys.path.insert(0, str(Path(__file__).parent))
    from test_patches import _exl3_layer

    ex = _exl3_layer(d, ni, e, seed=seed)
    if shared:
        ex.suh_u = ex.suh_g.clone()
    ex.__dict__.pop("_shared_suh", None)
    return ex


def _picks(rows: int, kind: str, d=D, e=E, top=TOP):
    x = (torch.randn((rows, d), generator=torch.Generator().manual_seed(6)) * 0.5).to(torch.bfloat16)
    g = torch.Generator().manual_seed(7)
    w = torch.ones(e) if kind == "uniform" else 1.0 / torch.arange(1, e + 1).float() ** 0.8
    picks = torch.full((rows, top + 1), e, dtype=torch.int32)
    picks[:, :top] = torch.multinomial(w.expand(rows, e), top, replacement=False, generator=g).int()
    return x, picks


def _run(ex, x, picks, mode, *, cfg=(0, 0), ticket=True, ctas=0):
    """One MoE layer's routed experts (fast=True) with ``mode`` in fast2 / fat / tc; (Y, Xd) of the routed slots."""
    sys.path.insert(0, str(Path(__file__).parent))
    from test_fastpf_patches import _fast_group

    from tensorfold.families.glm5_next.cuda import exl3_mm

    names = ("FAT", "ONCE", "AUTO", "TC", "TC_CFG", "TC_TICKET", "TC_CTAS", "FAT_SHARED_X", "FAT_STAGES")
    saved = {k: getattr(exl3_mm, k) for k in names}
    exl3_mm.FAT, exl3_mm.ONCE, exl3_mm.AUTO, exl3_mm.TC = mode in ("fat", "tc"), False, False, mode == "tc"
    exl3_mm.TC_CFG, exl3_mm.TC_TICKET, exl3_mm.TC_CTAS = cfg, ticket, ctas
    exl3_mm.FAT_SHARED_X, exl3_mm.FAT_STAGES = True, 3
    try:
        n, slots = x.shape[0], picks.shape[1]
        scratch = exl3_mm.Scratch(n, slots, ex.dims, ex.width, "cuda")
        y = torch.zeros((n * slots, ex.dims), dtype=torch.float32, device="cuda")
        exl3_mm.routed(x.cuda(), picks.cuda(), _fast_group(picks, ex.count), ex, scratch, y, n, LIMIT, fast=True)
        torch.cuda.synchronize()
        top = slots - 1
        return (y.view(n, slots, ex.dims)[:, :top].cpu(), scratch.xd.view(n, slots, ex.width)[:, :top].cpu())
    finally:
        for k, v in saved.items():
            setattr(exl3_mm, k, v)


@gpu
@pytest.mark.parametrize("rows,kind", [(64, "uniform"), (300, "skewed"), (1024, "uniform"), (2048, "skewed"),
                                       (2048, "uniform"), (4160, "skewed"), (8192, "uniform"), (8192, "skewed")])
def test_tc_equals_fast2_bitwise(rows, kind):
    ex = _layer()
    x, picks = _picks(rows, kind)
    y2, xd2 = _run(ex, x, picks, "fast2")
    assert torch.isfinite(y2).all() and y2.abs().max() > 0
    yf, xdf = _run(ex, x, picks, "fat")
    assert torch.equal(yf, y2) and torch.equal(xdf, xd2), "fat != fast2 (0170 broken?)"
    for cfg in ((0, 0), (1, 1), (2, 2), (3, 3), (1, 2), (2, 1)):
        for ticket, ctas in ((True, 0), (False, 0), (True, 7)):
            yt, xdt = _run(ex, x, picks, "tc", cfg=cfg, ticket=ticket, ctas=ctas)
            assert torch.equal(xdt, xd2), ("Xd", cfg, ticket, ctas)
            assert torch.equal(yt, y2), ("Y", cfg, ticket, ctas)


@gpu
@pytest.mark.parametrize("rows", [5, 200, 700])
def test_tc_odd_shapes(rows):
    """d = ni = 384: 3 column blocks (a column-block group of 2 or 4 holds an absent block); 20 experts, top 4: many
    passes of 64 / 128 members."""
    ex = _layer(d=384, ni=384, e=20, seed=9)
    x, picks = _picks(rows, "skewed", d=384, e=20, top=4)
    y2, xd2 = _run(ex, x, picks, "fast2")
    for cfg in ((1, 1), (2, 2), (3, 3)):
        yt, xdt = _run(ex, x, picks, "tc", cfg=cfg)
        assert torch.equal(xdt, xd2) and torch.equal(yt, y2), cfg


@gpu
@pytest.mark.parametrize("rows", [700, 4160])
def test_tc_row_independent_and_repeatable(rows):
    ex = _layer()
    x, picks = _picks(rows, "skewed")
    y, xd = _run(ex, x, picks, "tc")
    y2, xd2 = _run(ex, x, picks, "tc")
    assert torch.equal(y, y2) and torch.equal(xd, xd2)                       # ticket order varies between runs
    rng = np.random.default_rng(2)
    for sub in ([5], list(range(64, 200)), list(range(rows - 1, -1, -3)), list(rng.permutation(rows)),
                list(range(0, rows, 2))):
        s = torch.tensor(sub)
        ys, xds = _run(ex, x[s], picks[s], "tc")
        assert torch.equal(ys, y[s]) and torch.equal(xds, xd[s]), sub[:3]


@gpu
def test_tc_probes_run():
    """The timing probes (no decode / no mma) launch on every configuration; their outputs are not checked."""
    sys.path.insert(0, str(Path(__file__).parent))
    from test_fastpf_patches import _fast_group

    from tensorfold.families.glm5_next.cuda import exl3_mm

    ex = _layer()
    tx = exl3_mm._tc_ext()
    for rows in (1024, 4160):
        x, picks = _picks(rows, "uniform")
        grp = _fast_group(picks, E)
        s = exl3_mm.Scratch(rows, TOP + 1, D, NI, "cuda")
        y = torch.zeros((rows * (TOP + 1), D), dtype=torch.float32, device="cuda")
        for cfg in (1, 2, 3):
            for probe in (1, 2):
                tx.gateup_tc(s.xg, ex.gt, ex.ut, grp.ids, grp.count, grp.members, ex.svh_g, ex.svh_u, ex.suh_d, s.xd,
                             D, NI, TOP + 1, LIMIT, cfg, True, 0, probe)
                tx.down_tc(s.xd, ex.dt, grp.ids, grp.count, grp.members, ex.svh_d, y, NI, D, TOP + 1, cfg, True, 0,
                           probe)
        torch.cuda.synchronize()


# -- GPU: engine -----------------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_lookup(monkeypatch):
    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")
    monkeypatch.setenv("GLM53_TF_NONEXPERT", "q4mse")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    if not CUDA:
        pytest.skip("CUDA only")
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_tc330")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


def _engine(path, **kw):
    sys.path.insert(0, str(Path(__file__).parent))
    from test_mia_prefill_patches import _engine as mia_engine

    return mia_engine(path, **kw)


@pytest.fixture(scope="module")
def ea(ckpt):
    return _engine(ckpt)


@pytest.fixture(scope="module")
def eb(ckpt):
    return _engine(ckpt, lean_block=256, rows_max=8192)


class _Mode:
    """Switch the process-wide expert kernels (what GLM53_TF_FAST_EXPERTS sets at import) for a block."""

    def __init__(self, mode: str):
        self.mode = mode

    def __enter__(self):
        from tensorfold.families.glm5_next.cuda import exl3_mm

        self.saved = (exl3_mm.FAT, exl3_mm.TC, exl3_mm.ONCE, exl3_mm.AUTO)
        exl3_mm.FAT, exl3_mm.TC = self.mode in ("fat", "tc"), self.mode == "tc"
        exl3_mm.ONCE = exl3_mm.AUTO = False

    def __exit__(self, *exc):
        from tensorfold.families.glm5_next.cuda import exl3_mm

        exl3_mm.FAT, exl3_mm.TC, exl3_mm.ONCE, exl3_mm.AUTO = self.saved


@gpu
@pytest.mark.parametrize("n", [3, 65, 300, 1000])
def test_engine_state_tc_equals_fast2(ea, eb, n):
    from test_cindep_patches import _Variant, _same, _state

    prompt = list(np.random.default_rng(330 + n).integers(0, 1000, size=n))
    for eng, rows in ((ea, 1024), (eb, 8192), (eb, 256)):
        with _Variant(eng, rows):
            with _Mode("fast2"):
                ref = _state(eng, prompt)
            with _Mode("tc"):
                got = _state(eng, prompt)
            assert _same(ref, got), (rows, [i for i, (a, b) in enumerate(zip(ref, got)) if not torch.equal(a, b)])


@gpu
def test_engine_drafted_serial_and_resumed_fresh(eb):
    """With tc on (the knob fat_experts = 1 selects it while TC is set): drafted == serial, resumed == fresh with other
    chunk sizes."""
    from test_cindep_patches import _cold, _gen, _sampling

    with _Mode("tc"):
        knobs = {"fat_experts": 1}
        rng = np.random.default_rng(430)
        more = lambda n: [int(t) for t in rng.integers(0, 1000, size=n)]       # noqa: E731
        for sampling in ("sampled", "greedy"):
            s = _sampling(sampling)
            p1 = more(900)
            eb.cache = []
            r1, _ = _gen(eb, p1, s, policy="auto:1:1:0", knobs=dict(knobs, prefill_rows=8192))
            serial = _cold(eb, p1, s, knobs=dict(knobs, prefill_rows=1024))
            assert r1 == serial[:len(r1)], sampling                               # drafted == serial
            _gen(eb, p1, s, knobs=dict(knobs, prefill_rows=8192))
            p2 = p1 + r1 + more(80)
            warm, stats = _gen(eb, p2, s, knobs=dict(knobs, prefill_rows=64))
            assert stats["cached"] > 0
            assert warm == _cold(eb, p2, s, knobs=dict(knobs, prefill_rows=4096)), sampling   # resumed == fresh


@gpu
@pytest.mark.parametrize("first", [0, 1])
def test_tc_shares_snapshots_with_fast2(eb, first):
    from test_cindep_patches import _cold, _gen, _sampling

    with _Mode("tc"):
        s = _sampling("greedy")
        rng = np.random.default_rng(530 + first)
        p1 = [int(t) for t in rng.integers(0, 1000, size=700)]
        eb.cache = []
        r1, _ = _gen(eb, p1, s, knobs={"fat_experts": first, "prefill_rows": 1024})
        p2 = p1 + r1 + [int(t) for t in rng.integers(0, 1000, size=50)]
        warm, stats = _gen(eb, p2, s, knobs={"fat_experts": 1 - first, "prefill_rows": 1024})
        assert stats["cached"] > 0
        assert warm == _cold(eb, p2, s, knobs={"fat_experts": first, "prefill_rows": 4096})
