"""patches/0260: the ``once`` routed-expert kernels (``GLM53_TF_FAST_EXPERTS=once``): fat with two work items a CTA
that share their trellis decode.

fat (0170) decodes a weight tile once per item = (expert, pass of 64 members, 128-column block): ceil(n / 64) times a
chunk. At 1024-row chunks that is already once (~28 members an expert); at 8192 rows gate/up decodes every tile ~4
times and down ~2. ``once`` pairs two passes of an expert in one CTA of two fat "halves": each warp decodes half of its
column tiles and swaps them with the partner warp of the other half (a 64-thread named barrier a k tile), so a tile is
decoded ceil(P / 2) times. An odd last pass (and every pass with ``GLM53_TF_ONCE_PAIR=0``) runs as a "split" unit: one
pass, two adjacent column blocks, the member rows shared.

It must equal fast2 (and fat) BIT FOR BIT, be row-independent (patches/0085) and deterministic.

Checked:

- host only: the env switch (``once`` is a fat-family mode: ``tf_knobs.fat_experts`` still picks fat-family vs
  fast2), the dispatch in ``exl3_mm.routed`` (with fake extensions), and a Python model of the kernel's index
  arithmetic: the plan's units cover every (expert, pass, column block) item exactly once (pair on / off, odd column
  block counts, stride and ticket walks); every warp reads the same member-row bytes and trellis words as fat's warp
  for that item (both unit kinds, all four tilings); the fragment exchange hands each warp the tile fat's warp decodes
  and never overwrites a slot before the partner read it; the shared-memory budget (<= 99 KB, 1 CTA an SM); the
  decode-count table of PATCHES 0260;
- GPU kernels: once == fast2 == fat bit for bit (Xd and Y; shared and own inputs; pair on / off; ticket on / off;
  300-8192 rows, uniform and skewed; real per-rank shapes and a small shape with odd column-block counts); row
  independence (row subsets and permutations); repeatable; the timing probes run;
- GPU engine (synthetic EXL3 checkpoint, gate / up sign vectors made equal as in the real one): committed state
  once == fast2 (lean and not); drafted == serial and resumed == fresh with once; a fast2 request resumes a once
  request's snapshot and the reverse (the same bits share snapshots).

Run inside the image:
    PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q -s tests/cuda/test_expert_once_patches.py
"""

from __future__ import annotations

import importlib
import math
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
NB = 8


# -- host only: plumbing ---------------------------------------------------------------------------------------------
_ENVS = ("GLM53_TF_FAST_EXPERTS", "GLM53_TF_ONCE_PAIR", "GLM53_TF_ONCE_MIN_ROWS", "GLM53_TF_FAT_STAGES",
         "GLM53_TF_FAT_TICKET")


@needs_torch
def test_env_modes(monkeypatch):
    from tensorfold.families.glm5_next.cuda import exl3_mm

    try:
        for mode, fat, once in (("once", True, True), ("fat", True, False), ("fast2", False, False),
                                ("v1", False, False), (None, False, False), ("ONCE", True, True)):
            for k in _ENVS:
                monkeypatch.delenv(k, raising=False)
            if mode is not None:
                monkeypatch.setenv("GLM53_TF_FAST_EXPERTS", mode)
            importlib.reload(exl3_mm)
            assert (exl3_mm.FAT, exl3_mm.ONCE, exl3_mm.ONCE_PAIR, exl3_mm.ONCE_MIN_ROWS) == (fat, once, True, 0), mode
        monkeypatch.setenv("GLM53_TF_ONCE_PAIR", "0")
        monkeypatch.setenv("GLM53_TF_ONCE_MIN_ROWS", "4096")
        importlib.reload(exl3_mm)
        assert exl3_mm.ONCE and not exl3_mm.ONCE_PAIR and exl3_mm.ONCE_MIN_ROWS == 4096
        monkeypatch.setenv("GLM53_TF_FAST_EXPERTS", "twice")
        with pytest.raises(ValueError, match="once"):
            importlib.reload(exl3_mm)
    finally:
        for k in _ENVS:
            monkeypatch.delenv(k, raising=False)
        importlib.reload(exl3_mm)


class _FakeExt:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: self.calls.append((name, a, k))


@needs_torch
@pytest.mark.parametrize("fat,once,pair,min_rows", [(True, True, True, 0), (True, True, False, 4), (True, False, True, 0),
                                                    (False, True, True, 0), (True, True, True, 5)])
def test_routed_dispatch(monkeypatch, fat, once, pair, min_rows):
    """FAT (the per-request fat_experts knob) picks the fat family, ONCE picks once within it (for windows of at least
    ONCE_MIN_ROWS rows: fat below); once reads one rotated input when gate and up share their sign vector, and passes
    the ticket and pair switches."""

    from tensorfold.families.glm5_next.cuda import exl3_mm

    ext, fx = _FakeExt(), _FakeExt()
    monkeypatch.setattr(exl3_mm, "_ext", lambda: ext)
    monkeypatch.setattr(exl3_mm, "_fast_ext", lambda: fx)
    monkeypatch.setattr(exl3_mm, "V2", None)
    monkeypatch.setattr(exl3_mm, "FAT", fat)
    monkeypatch.setattr(exl3_mm, "ONCE", once)
    monkeypatch.setattr(exl3_mm, "ONCE_PAIR", pair)
    monkeypatch.setattr(exl3_mm, "ONCE_MIN_ROWS", min_rows)
    monkeypatch.setattr(exl3_mm, "FAT_TICKET", True)
    monkeypatch.setattr(exl3_mm, "FAT_SHARED_X", True)
    suh = torch.ones((2, 256), dtype=torch.float16)
    z = lambda *s: torch.zeros(s, dtype=torch.float16)            # noqa: E731
    ex = exl3_mm.Exl3Experts(torch.zeros(1), torch.zeros(1), torch.zeros(1), suh, suh.clone(), z(2, 128), z(2, 128),
                             z(2, 128), z(2, 256), 2, 128, 256)
    s = NS(slots=3, rows=4, xg=z(12, 256), xu=z(12, 256), xd=z(12, 128))
    grp = NS(ids=torch.zeros(3, dtype=torch.int32), count=torch.ones(1, dtype=torch.int32),
             members=torch.full((3, 4), -1, dtype=torch.int32))
    x = torch.zeros((4, 256), dtype=torch.bfloat16)
    exl3_mm.routed(x, torch.zeros((4, 3), dtype=torch.int32), grp, ex, s, torch.zeros((12, 256)), 4, LIMIT, fast=True)
    names = [c[0] for c in ext.calls + fx.calls]
    if not fat:
        assert names == ["rot_in", "gateup", "down"]
        return
    if not once or min_rows > 4:                                    # the window has 4 rows (members [3, 4])
        assert names == ["rot_in1", "gateup_fat", "down_fat"]
        return
    assert names == ["rot_in1", "gateup_once", "down_once"]
    gu, dn = fx.calls[1][1], fx.calls[2][1]
    assert gu[1] is gu[0] and gu[-3:] == (True, True, pair)       # XU = XG (shared input), shared_x, ticket, pair
    assert dn[-2:] == (True, pair)


# -- host only: a Python model of exl3_fast.cu's ``once`` namespace ---------------------------------------------------
def _cfg(mats, mtl, ng, ks, nsa, ngr, shx):
    bm, wpm = ng * 8, NB // mtl
    hw = mats * wpm
    xm = 1 if shx else mats
    lda, lde, er = ks * 16, 132, ngr * 8
    rows = xm * bm * lda * 2
    wset = mats * ks * NB * 32 * 4
    stage = max(2 * rows + wset, rows + 2 * wset)
    xf = mtl // 2
    xbuf = 2 * (2 * hw) * xf * 32 * 16
    epi_h = mats * er * lde * 4
    return NS(MATS=mats, MTL=mtl, NG=ng, KS=ks, NSA=nsa, NGR=ngr, SHX=shx, BM=bm, WPM=wpm, HW=hw, W=2 * hw,
              THREADS=2 * hw * 32, XM=xm, CPR=ks * 2, LDA=lda, LDE=lde, ER=er, XF=xf, ROWS=rows, WSET=wset,
              STAGE=stage, RING=nsa * stage, XBUF=xbuf, EPI_H=epi_h, SMEM=max(nsa * stage + xbuf, 2 * epi_h))


CFGS = {                                      # exl3_fast.cu: ONCE_GU_SHX, ONCE_GU_OWN, ONCE_DN, ONCE_DN_LARGE
    "gu_shx": _cfg(2, 2, 8, 4, 3, 4, True),
    "gu_own": _cfg(2, 2, 8, 2, 3, 4, False),
    "dn": _cfg(1, 2, 8, 4, 3, 8, False),
    "dn_large": _cfg(1, 2, 16, 2, 3, 8, False),
}
STATIC = {"gu_shx": 528, "gu_own": 528, "dn": 528, "dn_large": 1040}    # ptxas -v (sm_121): rows_sh + scalars


def _swz(cpr, r, c):
    return c ^ (r & 7) if cpr == 8 else c ^ ((r >> 1) & 3)


def _plan(passes, nblk, pair):
    """plan_kernel: first unit of each distinct expert, and the total."""
    hb = (nblk + 1) // 2
    units = []
    for p in passes:
        pairs = p // 2 if pair else 0
        units.append(pairs * nblk + (p - 2 * pairs) * hb)
    first = np.concatenate([[0], np.cumsum(units)]).astype(int).tolist()
    return first, first[-1]


def _unit(j, p, nblk, pair):
    """The kernel's unit decode: (paired, (pass0, nb0), (pass1, nb1) or None for an absent second column block)."""
    hb = (nblk + 1) // 2
    pairs = p // 2 if pair else 0
    if j < pairs * nblk:
        q, nb = j // nblk, j % nblk
        return True, (2 * q, nb), (2 * q + 1, nb)
    j2 = j - pairs * nblk
    ps, nb0 = 2 * pairs + j2 // hb, 2 * (j2 % hb)
    return False, (ps, nb0), ((ps, nb0 + 1) if nb0 + 1 < nblk else None)


@pytest.mark.parametrize("pair", [True, False])
def test_model_units_cover_every_item_once(pair):
    rng = np.random.default_rng(260)
    for trial in range(60):
        maxu = int(rng.integers(1, 300))
        passes = [int(v) for v in rng.integers(0, 7, size=maxu)]
        passes[-1] = 0                                                        # the shared expert: nothing
        nblk = int(rng.choice([1, 2, 3, 8, 32]))
        first, total = _plan(passes, nblk, pair)
        grid = int(rng.integers(1, 97))
        seen = []
        decodes = {}
        order = [list(range(c, total, grid)) for c in range(grid)]           # static stride (ticket: any ascending
        if trial % 2:                                                         # split of 0..total-1 over the CTAs)
            cut = sorted(rng.integers(0, total + 1, size=grid - 1).tolist()) if total else [0] * (grid - 1)
            bounds = [0] + cut + [total]
            order = [list(range(bounds[c], bounds[c + 1])) for c in range(grid)]
        for its in order:
            u = 0
            for it in its:
                while u + 1 < maxu and first[u + 1] <= it:                   # plan[2 + u] = first unit of u + 1
                    u += 1
                assert first[u] <= it < first[u + 1]
                paired, a, b = _unit(it - first[u], passes[u], nblk, pair)
                assert 0 <= a[0] < passes[u] and 0 <= a[1] < nblk
                seen.append((u,) + a)
                if b is not None:
                    assert 0 <= b[0] < passes[u] and 0 <= b[1] < nblk
                    seen.append((u,) + b)
                if paired:
                    assert b[1] == a[1] and b[0] == a[0] + 1                  # two passes, one column block
                    decodes[(u, a[1])] = decodes.get((u, a[1]), 0) + 1
                else:
                    assert b is None or (b[0] == a[0] and b[1] == a[1] + 1)   # one pass, two column blocks
                    for x in (a, b):
                        if x is not None:
                            decodes[(u, x[1])] = decodes.get((u, x[1]), 0) + 1
        want = [(u, p, nb) for u in range(maxu) for p in range(passes[u]) for nb in range(nblk)]
        assert sorted(seen) == sorted(want)                                    # every item exactly once
        for u in range(maxu):
            for nb in range(nblk):
                if passes[u]:
                    expect = math.ceil(passes[u] / 2) if pair else passes[u]
                    assert decodes[(u, nb)] == expect                          # a tile's decodes a chunk


def _stage_reads(c, paired, h, hw, s, e, nb0, nb1, rows_sh, KT, NTILES):
    """What warp (h, hw) of a once CTA reads in stage s (a dict of (kind, kk, ...) -> source), through the ring layout
    the kernel writes (load_stage) and the addresses it reads (a_off / w_off / ldmatrix / Wd)."""
    nrs = 2 if paired else 1
    RR = nrs * c.BM
    rows_bytes = nrs * c.ROWS
    ring = {}
    for i in range(c.XM * RR * c.CPR):                                          # rows: halves index
        m, r, ch = i // (RR * c.CPR), (i // c.CPR) % RR, i % c.CPR
        ring[("a", (m * RR + r) * c.LDA + _swz(c.CPR, r, ch) * 8)] = (m, rows_sh[r], s * c.KS * 16 + ch * 8)
    for i in range((3 - nrs) * c.MATS * c.KS * NB * 8):                         # words: uint32 index
        q, n, kk, m = i & 7, (i >> 3) % NB, (i // (8 * NB)) % c.KS, (i // (8 * NB * c.KS)) % c.MATS
        js = i // (8 * NB * c.KS * c.MATS)
        src = (m, ((e * KT + s * c.KS + kk) * NTILES + (nb1 if js else nb0) * NB + n) * 32 + q * 4)
        for w in range(4):
            ring[("w", rows_bytes // 4 + (((js * c.MATS + m) * c.KS + kk) * NB + n) * 32 + q * 4 + w)] = (src[0], src[1] + w)
    mat, sl = hw // c.WPM, hw % c.WPM
    xmat = 0 if c.SHX else mat
    a_off = (xmat * nrs * c.BM + (h * c.BM if paired else 0)) * c.LDA
    w_off = ((0 if paired else h) * c.MATS + mat) * c.KS * NB * 32 + sl * c.MTL * 32
    got = {}
    for kk in range(c.KS):
        for lane in range(32):
            lr, lc = (lane & 7) + ((lane >> 4) << 3), (lane >> 3) & 1
            chunk = _swz(c.CPR, lr, kk * 2 + lc)
            for np_ in range(c.NG // 2):
                got[("a", kk, np_, lane)] = ring[("a", a_off + (16 * np_ + lr) * c.LDA + chunk * 8)]
            for l in range(c.MTL):
                got[("w", kk, l, lane)] = ring[("w", rows_bytes // 4 + w_off + lane + (kk * NB + l) * 32)]
    return got


def _fat_reads(c, h_rows, mat, sl, s, e, nb, KT, NTILES):
    """What fat's warp (mat, slice) reads for the item (e, rows h_rows, nb) in stage s (fast2's addresses)."""
    xmat = 0 if c.SHX else mat
    got = {}
    for kk in range(c.KS):
        for lane in range(32):
            lr, lc = (lane & 7) + ((lane >> 4) << 3), (lane >> 3) & 1
            for np_ in range(c.NG // 2):
                got[("a", kk, np_, lane)] = (xmat, h_rows[16 * np_ + lr], s * c.KS * 16 + kk * 16 + lc * 8)
            for l in range(c.MTL):
                # fast2: tw = T + (e KT NTILES + nb NB + slice MTL) 32 + lane, then + (kt NTILES + l) 32
                got[("w", kk, l, lane)] = (mat, (e * KT * NTILES + nb * NB + sl * c.MTL) * 32 + lane +
                                           ((s * c.KS + kk) * NTILES + l) * 32)
    return got


@pytest.mark.parametrize("name", list(CFGS))
@pytest.mark.parametrize("paired", [True, False])
def test_model_warps_read_fats_bytes(name, paired):
    c = CFGS[name]
    rng = np.random.default_rng(7)
    KT, NTILES, e = 4 * c.KS, 4 * NB, 3
    nb0, nb1 = (2, 2) if paired else (2, 3)
    pass_rows = [rng.integers(-1, 5000, size=c.BM).tolist() for _ in range(2)]
    rows_sh = pass_rows[0] + (pass_rows[1] if paired else pass_rows[0])        # the kernel's rows_sh [2][BM]
    for s in range(2):
        for h in range(2):
            for hw in range(c.HW):
                got = _stage_reads(c, paired, h, hw, s, e, nb0, nb1, rows_sh, KT, NTILES)
                want = _fat_reads(c, pass_rows[h if paired else 0], hw // c.WPM, hw % c.WPM, s, e,
                                  nb1 if h else nb0, KT, NTILES)
                assert got == want, (name, paired, s, h, hw)


def test_model_fragment_exchange():
    """Paired units: warp (h, hw) decodes tiles l = 2i + h and reads slot (partner, i) of the k tile's parity, which the
    partner filled with ITS tile 2i + 1 - h of the same k tile: the word fat's warp decodes for l (both halves read the
    same word set). Slot (w, i) of parity p is rewritten at k tile kt + 2 only after the barrier of kt + 1, which the
    reader reaches after reading it at kt: checked on a schedule where a warp runs ahead as far as the barriers allow."""
    for c in CFGS.values():
        for hw in range(c.HW):
            for h in range(2):
                warp, partner = h * c.HW + hw, (1 - h) * c.HW + hw
                assert partner // c.HW != h and partner % c.HW == hw
                have = {}
                for i in range(c.XF):
                    have[2 * i + h] = ("mine", i)
                    have[2 * i + 1 - h] = ("partner", (partner * c.XF + i))   # slot the partner wrote: its own tile
                    assert (partner * c.XF + i) // c.XF == partner
                assert sorted(have) == list(range(c.MTL))
    # the race check on two warps: events (write kt), bar kt, (read kt); bar kt blocks until both arrive
    KT = 12
    for lead in range(2):
        slots = {}                           # (parity, writer) -> kt of the value in it
        pos = [0, 0]                         # next k tile of each warp
        stage = [0, 0]                       # 0: write, 1: at barrier, 2: read
        steps = 0
        while min(pos) < KT:
            steps += 1
            assert steps < 10000
            order = (lead, 1 - lead)
            moved = False
            for w in order:
                kt = pos[w]
                if kt >= KT:
                    continue
                if stage[w] == 0:
                    slots[(kt & 1, w)] = kt
                    stage[w] = 1
                    moved = True
                elif stage[w] == 1:
                    other = 1 - w
                    if pos[other] > kt or (pos[other] == kt and stage[other] >= 1):
                        stage[w] = 2
                        moved = True
                elif stage[w] == 2:
                    assert slots[(kt & 1, 1 - w)] == kt, (w, kt)                # the partner's tile of THIS k tile
                    stage[w], pos[w] = 0, kt + 1
                    moved = True
                if moved:
                    break
            assert moved


@pytest.mark.parametrize("name", list(CFGS))
def test_model_shared_memory_budget(name):
    c = CFGS[name]
    assert c.SMEM + STATIC[name] <= 101376                     # sm_120/121: 99 KB a CTA (opt-in)
    assert 2 * c.EPI_H <= c.SMEM and c.RING + c.XBUF <= c.SMEM
    for paired in (True, False):
        nrs = 2 if paired else 1
        assert nrs * c.ROWS + (3 - nrs) * c.WSET <= c.STAGE
    assert c.ROWS % 16 == 0 and c.WSET % 16 == 0 and c.STAGE % 16 == 0
    assert c.HW <= 15 and c.MTL % 2 == 0


def _counts(rows, kind, rng):
    w = np.ones(E) if kind == "uniform" else 1.0 / np.arange(1, E + 1) ** 0.8
    g = np.log(w / w.sum())[None, :] - np.log(-np.log(rng.random((rows, E))))
    return np.bincount(np.argpartition(-g, TOP, axis=1)[:, :TOP].ravel(), minlength=E)


def decode_table(rows_list=(1024, 2048, 4096, 8192), seed=0):
    """Decodes of a weight tile per chunk (mean over present experts, gate/up tiles weighted 2:1 against down's)."""
    rng = np.random.default_rng(seed)
    out = {}
    for kind in ("uniform", "skewed"):
        for r in rows_list:
            n = _counts(r, kind, rng)
            n = n[n > 0]
            gu = np.ceil(n / 64)
            dn = np.ceil(n / (128 if r >= 4096 else 64))
            fat = (2 * gu + dn).sum() / (3 * n.size)
            once = (2 * np.ceil(gu / 2) + np.ceil(dn / 2)).sum() / (3 * n.size)
            out[(kind, r)] = (float(n.mean()), float(gu.mean()), float(dn.mean()), float(fat), float(once))
    return out


def test_decode_count_table():
    t = decode_table()
    for (kind, r), (mean, gu, dn, fat, once) in t.items():
        print(f"[0260] {kind:7s} {r:5d} rows: {mean:6.1f} members an expert; fat decodes a tile {fat:.2f}x "
              f"(gate/up {gu:.2f}, down {dn:.2f}), once {once:.2f}x")
    assert t[("uniform", 1024)][3] == pytest.approx(1.0, abs=0.01)            # fat already decodes once at 1024
    assert t[("uniform", 8192)][1] == pytest.approx(4.0, abs=0.1)
    assert t[("uniform", 8192)][4] == pytest.approx(1.7, abs=0.1)


# -- GPU: kernels ----------------------------------------------------------------------------------------------------
def _layer(shared: bool, d=D, ni=NI, e=E, seed=5):
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


def _run(ex, x, picks, mode, *, ticket=True, pair=True, shared_x=True):
    """One MoE layer's routed experts (fast=True) with ``mode`` in fast2 / fat / once; returns (Y, Xd) of the routed
    slots on the host."""
    sys.path.insert(0, str(Path(__file__).parent))
    from test_fastpf_patches import _fast_group

    from tensorfold.families.glm5_next.cuda import exl3_mm

    names = ("FAT", "ONCE", "ONCE_PAIR", "ONCE_MIN_ROWS", "FAT_TICKET", "FAT_SHARED_X", "FAT_STAGES")
    saved = {k: getattr(exl3_mm, k) for k in names}
    exl3_mm.FAT, exl3_mm.ONCE = mode in ("fat", "once"), mode == "once"
    exl3_mm.ONCE_PAIR, exl3_mm.FAT_TICKET, exl3_mm.FAT_SHARED_X, exl3_mm.FAT_STAGES = pair, ticket, shared_x, 3
    exl3_mm.ONCE_MIN_ROWS = 0
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
def test_fast2_is_the_reference_here():
    import os

    assert os.environ.get("GLM53_TF_FAST_EXPERTS", "fast2") != "v1", "fast2 must be on in the C++ switch"


@gpu
@pytest.mark.parametrize("shared", [True, False], ids=["shared_x", "own_x"])
@pytest.mark.parametrize("rows,kind", [(300, "skewed"), (1024, "uniform"), (2048, "skewed"), (4160, "skewed"),
                                       (8192, "uniform")])
def test_once_equals_fast2_bitwise(rows, kind, shared):
    ex = _layer(shared)
    x, picks = _picks(rows, kind)
    y2, xd2 = _run(ex, x, picks, "fast2")
    assert torch.isfinite(y2).all() and y2.abs().max() > 0
    yf, xdf = _run(ex, x, picks, "fat")
    assert torch.equal(yf, y2) and torch.equal(xdf, xd2), "fat != fast2 (0170 broken?)"
    for pair in (True, False):
        for ticket in (True, False):
            yo, xdo = _run(ex, x, picks, "once", pair=pair, ticket=ticket)
            assert torch.equal(xdo, xd2), ("Xd", pair, ticket)
            assert torch.equal(yo, y2), ("Y", pair, ticket)


@gpu
@pytest.mark.parametrize("rows", [5, 200, 700])
def test_once_odd_column_blocks(rows):
    """d = ni = 384: 3 column blocks for gate/up and down, so split units carry an absent second block; 20 experts, top
    4, so experts have many passes (pair units, an odd last pass)."""
    ex = _layer(True, d=384, ni=384, e=20, seed=9)
    x, picks = _picks(rows, "skewed", d=384, e=20, top=4)
    y2, xd2 = _run(ex, x, picks, "fast2")
    for pair in (True, False):
        yo, xdo = _run(ex, x, picks, "once", pair=pair)
        assert torch.equal(xdo, xd2) and torch.equal(yo, y2), pair


@gpu
@pytest.mark.parametrize("shared", [True, False], ids=["shared_x", "own_x"])
@pytest.mark.parametrize("rows", [700, 4160])
def test_once_row_independent_and_repeatable(shared, rows):
    ex = _layer(shared)
    x, picks = _picks(rows, "skewed")
    y, xd = _run(ex, x, picks, "once")
    y2, xd2 = _run(ex, x, picks, "once")
    assert torch.equal(y, y2) and torch.equal(xd, xd2)                       # ticket order varies between runs
    rng = np.random.default_rng(2)
    for sub in ([5], list(range(64, 200)), list(range(rows - 1, -1, -3)), list(rng.permutation(rows)),
                list(range(0, rows, 2))):
        s = torch.tensor(sub)
        ys, xds = _run(ex, x[s], picks[s], "once")
        assert torch.equal(ys, y[s]) and torch.equal(xds, xd[s]), sub[:3]


@gpu
def test_once_probes_run():
    """The timing probes (no decode / no mma) launch on every configuration; their outputs are not checked."""
    sys.path.insert(0, str(Path(__file__).parent))
    from test_fastpf_patches import _fast_group

    from tensorfold.families.glm5_next.cuda import exl3_mm

    ex = _layer(True)
    fx = exl3_mm._fast_ext()
    for rows in (1024, 4160):
        x, picks = _picks(rows, "uniform")
        grp = _fast_group(picks, E)
        s = exl3_mm.Scratch(rows, TOP + 1, D, NI, "cuda")
        y = torch.zeros((rows * (TOP + 1), D), dtype=torch.float32, device="cuda")
        for probe in (1, 2):
            fx.gateup_once(s.xg, s.xg, ex.gt, ex.ut, grp.ids, grp.count, grp.members, ex.svh_g, ex.svh_u, ex.suh_d,
                           s.xd, D, NI, TOP + 1, LIMIT, True, True, True, probe)
            fx.down_once(s.xd, ex.dt, grp.ids, grp.count, grp.members, ex.svh_d, y, NI, D, TOP + 1, True, True, probe)
        torch.cuda.synchronize()


# -- GPU: engine -----------------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_lookup(monkeypatch):
    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")
    monkeypatch.setenv("GLM53_TF_NONEXPERT", "q4mse")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_once260")
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

    def __init__(self, mode: str, pair: bool = True):
        self.mode, self.pair = mode, pair

    def __enter__(self):
        from tensorfold.families.glm5_next.cuda import exl3_mm

        self.saved = (exl3_mm.FAT, exl3_mm.ONCE, exl3_mm.ONCE_PAIR)
        exl3_mm.FAT, exl3_mm.ONCE = self.mode in ("fat", "once"), self.mode == "once"
        exl3_mm.ONCE_PAIR = self.pair

    def __exit__(self, *exc):
        from tensorfold.families.glm5_next.cuda import exl3_mm

        exl3_mm.FAT, exl3_mm.ONCE, exl3_mm.ONCE_PAIR = self.saved


@gpu
@pytest.mark.parametrize("n", [3, 65, 300, 1000])
def test_engine_state_once_equals_fast2(ea, eb, n):
    from test_cindep_patches import _Variant, _same, _state

    prompt = list(np.random.default_rng(260 + n).integers(0, 1000, size=n))
    for eng, rows in ((ea, 1024), (eb, 8192), (eb, 256)):
        with _Variant(eng, rows):
            with _Mode("fast2"):
                ref = _state(eng, prompt)
            for pair in (True, False):
                with _Mode("once", pair):
                    got = _state(eng, prompt)
                assert _same(ref, got), (rows, pair, [i for i, (a, b) in enumerate(zip(ref, got))
                                                      if not torch.equal(a, b)])


@gpu
def test_engine_drafted_serial_and_resumed_fresh(eb):
    """With once on (the knob fat_experts = 1 selects it while ONCE is set): drafted == serial, resumed == fresh with
    other chunk sizes."""
    from test_cindep_patches import _cold, _gen, _sampling

    from tensorfold.families.glm5_next.cuda import exl3_mm

    saved = exl3_mm.ONCE
    exl3_mm.ONCE = True
    try:
        knobs = {"fat_experts": 1}
        rng = np.random.default_rng(460)
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
    finally:
        exl3_mm.ONCE = saved


@gpu
@pytest.mark.parametrize("first", [0, 1])
def test_once_shares_snapshots_with_fast2(eb, first):
    """once == fast2, so a fast2 request resumes a once request's snapshot (and the reverse) with the same reply."""
    from test_cindep_patches import _cold, _gen, _sampling

    from tensorfold.families.glm5_next.cuda import exl3_mm

    saved = exl3_mm.ONCE
    exl3_mm.ONCE = True
    try:
        s = _sampling("greedy")
        rng = np.random.default_rng(560 + first)
        p1 = [int(t) for t in rng.integers(0, 1000, size=700)]
        eb.cache = []
        r1, _ = _gen(eb, p1, s, knobs={"fat_experts": first, "prefill_rows": 1024})
        p2 = p1 + r1 + [int(t) for t in rng.integers(0, 1000, size=50)]
        warm, stats = _gen(eb, p2, s, knobs={"fat_experts": 1 - first, "prefill_rows": 1024})
        assert stats["cached"] > 0
        assert warm == _cold(eb, p2, s, knobs={"fat_experts": first, "prefill_rows": 4096})
    finally:
        exl3_mm.ONCE = saved
