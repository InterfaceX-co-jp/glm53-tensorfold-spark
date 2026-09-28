"""patches/0170 (MiaAI-Lab / Reederey87 prefill wins, ported): the ``fat`` routed-expert kernels
(``GLM53_TF_FAST_EXPERTS=fat``, ``tf_knobs.fat_experts``) and the KDA input projection's bf16 copy
(``GLM53_TF_KDA_PROJ_BF16=1``).

- fat: fast2's arithmetic (weights as the mma A operand, one ascending k chain per element, the same transforms and
  epilogue) with the E2/E3 data movement (trellis words through the cp.async ring, one rotated input for gate and up
  when their sign vectors are equal, swizzled 48 KB stages at 2 CTAs an SM, ticket scheduling). It must equal fast2
  BIT FOR BIT, be row-independent (patches/0085) and deterministic.
- KDA bf16: fast chunks multiply a bf16 copy of the SAME 4-bit weights (every row count: no Mia-style M > 512 switch,
  which would break 0085's row independence); new arithmetic, close to the 4-bit path.

Checked:

- host only: the knob / env plumbing (header block, load-only refusal, ``fastpf.settings``, parsing), the memory
  estimate, the copy dispatch (fast chunks only, FP8 off), and a Python model of the fat kernel's index arithmetic
  (swizzled row stages read back what was written, the weight ring hands each warp fast2's words, the expert walk ==
  fast2's binary search, the ticket covers every item once);
- GPU kernels: fat == fast2 bit for bit (Xd and Y; shared and separate inputs; 3 / 4 stages; ticket on / off; uniform
  and skewed routing; 4096+ rows for down's large configuration); ``rot_in1`` == ``rot_in``'s first output; fat is
  row-independent (row subsets and permutations) and repeatable; the KDA copy's matmul is row-independent (1-63-row
  slices, unaligned slices, permutations) and within bf16 rounding of the 4-bit fast kernel; timings (``-s``);
- GPU engine (synthetic EXL3 checkpoint, the gate/up sign vectors made equal so the shared-input path runs): the
  committed state after a fast prefill with fat == fast2 (bit for bit, several prompt lengths, lean and not); with the
  KDA copies the state does not depend on the chunk size (0085); drafted == serial; resumed == fresh with another C.

Run inside the image:
    PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q -s tests/cuda/test_mia_prefill_patches.py
"""

from __future__ import annotations

import bisect
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:          # the host-only tests still run
    torch = None
    CUDA = False

gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")
needs_torch = pytest.mark.skipif(torch is None, reason="torch")

D, NI, E, TOP, LIMIT = 4096, 1024, 288, 8, 10.0


# -- host only: plumbing ---------------------------------------------------------------------------------------------
def test_knob_in_header_and_load_only_bf16():
    from tensorfold.families.glm5_next.cuda import knobs

    assert "fat_experts" in knobs.HEADER and knobs.RANGES["fat_experts"] == (0, 1)       # patches/0190 appends after it
    assert knobs.parse({"fat_experts": True}, rows_max=512) == {"fat_experts": 1}
    with pytest.raises(ValueError, match="0 to 1"):
        knobs.parse({"fat_experts": 2}, rows_max=512)
    with pytest.raises(ValueError, match="GLM53_TF_KDA_PROJ_BF16"):
        knobs.parse({"kda_proj_bf16": 1}, rows_max=512)
    values = {k: 0 for k in knobs.HEADER}
    values["fat_experts"] = 1
    got, rest = knobs.decode(knobs.encode(values) + [7])
    assert got == values and rest == [7]


def test_fastpf_kda_bf16_settings(monkeypatch):
    from tensorfold.families.glm5_next.cuda import fastpf

    monkeypatch.delenv("GLM53_TF_KDA_PROJ_BF16", raising=False)
    assert not fastpf.kda_bf16() and fastpf.settings()[-1] == 0
    monkeypatch.setenv("GLM53_TF_KDA_PROJ_BF16", "1")
    assert fastpf.kda_bf16() and fastpf.settings()[-1] == 1
    monkeypatch.setenv("GLM53_TF_KDA_BF16_TILE", "128,64,4,3")
    assert fastpf.kda_bf16_tile() == (128, 64, 4, 3)
    monkeypatch.setenv("GLM53_TF_KDA_BF16_TILE", "100,64,4,3")
    with pytest.raises(ValueError):
        fastpf.kda_bf16_tile()
    # GLM-5.3-Flash per rank at TP 2: [q|k|v|f_a|g_a|b] 12576 x 4096 in each KDA layer (Mia's 98.25 MiB a layer)
    assert fastpf.kda_bf16_bytes(12576, 4096, 1) / 2**20 == pytest.approx(98.25)
    assert fastpf.kda_bf16_bytes(12576, 4096, 34) / 2**30 == pytest.approx(3.26, abs=0.01)


@needs_torch
def test_kda_proj_dispatch(monkeypatch):
    from types import SimpleNamespace as NS

    from tensorfold.families.glm5_next.cuda import fastpf, fp8pf, qmm

    k = NS(proj="q4", proj16="bf16")
    monkeypatch.setattr(qmm, "FAST_MM", None)
    assert fastpf.kda_proj(k) == "q4"                            # decode / exact prefill
    monkeypatch.setattr(qmm, "FAST_MM", object())
    assert fastpf.kda_proj(k) == "bf16"                          # a fast chunk
    monkeypatch.setattr(fp8pf, "ON", True)
    assert fastpf.kda_proj(k) == "q4"                            # FP8 prefill keeps its own kernels
    monkeypatch.setattr(fp8pf, "ON", False)
    assert fastpf.kda_proj(NS(proj="q4")) == "q4"                # no copy (GLM53_TF_KDA_PROJ_BF16=0)


@needs_torch
def test_exl3_mm_env(monkeypatch):
    import importlib

    from tensorfold.families.glm5_next.cuda import exl3_mm

    try:
        for env, fat in (({"GLM53_TF_FAST_EXPERTS": "fat"}, True), ({}, False), ({"GLM53_TF_FAST_EXPERTS": "v1"}, False)):
            for k in ("GLM53_TF_FAST_EXPERTS", "GLM53_TF_FAT_STAGES", "GLM53_TF_FAT_TICKET"):
                monkeypatch.delenv(k, raising=False)
            for k, v in env.items():
                monkeypatch.setenv(k, v)
            importlib.reload(exl3_mm)
            assert exl3_mm.FAT == fat and exl3_mm.FAT_STAGES == 3 and exl3_mm.FAT_TICKET and exl3_mm.FAT_SHARED_X
        monkeypatch.setenv("GLM53_TF_FAT_STAGES", "5")
        with pytest.raises(ValueError):
            importlib.reload(exl3_mm)
        monkeypatch.setenv("GLM53_TF_FAT_STAGES", "4")
        monkeypatch.setenv("GLM53_TF_FAST_EXPERTS", "fatt")
        with pytest.raises(ValueError):
            importlib.reload(exl3_mm)
    finally:
        for k in ("GLM53_TF_FAST_EXPERTS", "GLM53_TF_FAT_STAGES", "GLM53_TF_FAT_TICKET"):
            monkeypatch.delenv(k, raising=False)
        importlib.reload(exl3_mm)


# -- host only: the fat kernel's index arithmetic (a Python model of exl3_fast.cu's ``fat`` namespace) ---------------
def _swz(cpr: int, r: int, c: int) -> int:
    return c ^ (r & 7) if cpr == 8 else c ^ ((r >> 1) & 3)


@pytest.mark.parametrize("ks,bm", [(4, 64), (2, 128)])
def test_model_swizzled_rows_read_back(ks, bm):
    """load_stage writes 16-byte chunk c of row r at swz(r, c); ldmatrix lane l of n pair np reads row 16 np + lr,
    logical chunk 2 kk + lc at swz(lr, .): the same bytes as fast2's padded layout, and each 8-lane phase hits 8
    distinct chunk columns (no bank conflicts)."""

    cpr = ks * 2
    where = {}
    for r in range(bm):
        cols = {_swz(cpr, r, c) for c in range(cpr)}
        assert cols == set(range(cpr))                                        # a permutation of the row's chunks
        for c in range(cpr):
            where[(r, _swz(cpr, r, c))] = c                                   # physical -> logical chunk
    for np_ in range(bm // 16):
        for kk in range(ks):
            phases = {}
            for lane in range(32):
                lr = (lane & 7) + ((lane >> 4) << 3)
                lc = (lane >> 3) & 1
                row = 16 * np_ + lr
                phys = _swz(cpr, lr, kk * 2 + lc)
                assert where[(row, phys)] == kk * 2 + lc                      # fast2 reads k offset kk 16 + lc 8
                phases.setdefault(lane // 8, set()).add((row * cpr + phys) % 8)   # 16-byte bank group
            assert all(len(v) == 8 for v in phases.values())


def test_model_weight_ring_equals_fast2_words():
    MATS, KS, NB, MTL, KT, NTILES, e, nb = 2, 4, 8, 2, 256, 64, 5, 3
    WPM = NB // MTL
    for s in range(3):
        ring = {}
        for i in range(MATS * KS * NB * 8):                                   # load_stage's chunk loop
            q, n, kk, m = i & 7, (i >> 3) % NB, (i // (8 * NB)) % KS, i // (8 * NB * KS)
            for w in range(4):
                src = ((e * KT + s * KS + kk) * NTILES + nb * NB + n) * 32 + q * 4 + w
                ring[((m * KS + kk) * NB + n) * 32 + q * 4 + w] = (m, src)
        for warp in range(MATS * WPM):
            mat, sl = warp // WPM, warp % WPM
            for kk in range(KS):
                for l in range(MTL):
                    for lane in range(32):
                        got = ring[mat * KS * NB * 32 + sl * MTL * 32 + lane + (kk * NB + l) * 32]
                        # fast2: tw + (kt * NTILES + l) * 32, tw = T + (e KT NTILES + nb NB + slice MTL) 32 + lane
                        want = ((e * KT * NTILES + nb * NB + sl * MTL) * 32 + lane) + ((s * KS + kk) * NTILES + l) * 32
                        assert got == (mat, want)


def test_model_expert_walk_and_ticket():
    rng = np.random.default_rng(1)
    for trial in range(50):
        maxu = int(rng.integers(1, 300))
        items = [int(v) for v in rng.integers(0, 4, size=maxu)]
        items[-1] = 0                                                         # the shared expert: no items
        first = np.concatenate([[0], np.cumsum(items)]).tolist()             # plan[1 + u]
        total_passes = first[-1]
        nblk = 8
        total = total_passes * nblk
        grid = int(rng.integers(1, 97))
        claimed = []
        for cta in range(grid):                                               # static stride: ascending per CTA
            u = 0
            for it in range(cta, total, grid):
                tt = it // nblk
                while u + 1 < maxu and first[u + 1] <= tt:                   # plan[2 + u] = first pass of u + 1
                    u += 1
                ref = bisect.bisect_right(first[:maxu], tt) - 1               # fast2's binary search over offs
                assert u == ref and first[u] <= tt < first[u] + items[u]
                claimed.append(it)
        assert sorted(claimed) == list(range(total))


# -- GPU: kernels ----------------------------------------------------------------------------------------------------
def _layer(shared: bool):
    sys.path.insert(0, str(Path(__file__).parent))
    from test_patches import _exl3_layer

    ex = _exl3_layer(D, NI, E, seed=5)
    if shared:
        ex.suh_u = ex.suh_g.clone()               # equal values, another tensor: exercises the torch.equal check
    ex.__dict__.pop("_shared_suh", None)
    return ex


def _picks(rows: int, kind: str):
    x = (torch.randn((rows, D), generator=torch.Generator().manual_seed(6)) * 0.5).to(torch.bfloat16)
    g = torch.Generator().manual_seed(7)
    w = torch.ones(E) if kind == "uniform" else 1.0 / torch.arange(1, E + 1).float() ** 0.8
    picks = torch.full((rows, TOP + 1), E, dtype=torch.int32)
    picks[:, :TOP] = torch.multinomial(w.expand(rows, E), TOP, replacement=False, generator=g).int()
    return x, picks


def _run(ex, x, picks, *, fat: bool, stages: int = 3, ticket: bool = True, shared_x: bool = True):
    from test_fastpf_patches import _fast_group

    from tensorfold.families.glm5_next.cuda import exl3_mm

    saved = (exl3_mm.FAT, exl3_mm.FAT_STAGES, exl3_mm.FAT_TICKET, exl3_mm.FAT_SHARED_X)
    exl3_mm.FAT, exl3_mm.FAT_STAGES, exl3_mm.FAT_TICKET, exl3_mm.FAT_SHARED_X = fat, stages, ticket, shared_x
    try:
        n = x.shape[0]
        scratch = exl3_mm.Scratch(n, TOP + 1, D, NI, "cuda")
        y = torch.zeros((n * (TOP + 1), D), dtype=torch.float32, device="cuda")
        exl3_mm.routed(x.cuda(), picks.cuda(), _fast_group(picks, E), ex, scratch, y, n, LIMIT, fast=True)
        torch.cuda.synchronize()
        return y.view(n, TOP + 1, D)[:, :TOP].cpu(), scratch.xd.view(n, TOP + 1, NI)[:, :TOP].cpu()
    finally:
        exl3_mm.FAT, exl3_mm.FAT_STAGES, exl3_mm.FAT_TICKET, exl3_mm.FAT_SHARED_X = saved


@gpu
def test_fast2_is_the_default_here():
    assert os.environ.get("GLM53_TF_FAST_EXPERTS", "fast2") != "v1", "run with fast2 (the reference) in the C++ switch"


@gpu
@pytest.mark.parametrize("shared", [True, False], ids=["shared_x", "own_x"])
@pytest.mark.parametrize("rows,kind", [(300, "skewed"), (1024, "uniform"), (4160, "skewed")])
def test_fat_equals_fast2_bitwise(rows, kind, shared):
    ex = _layer(shared)
    x, picks = _picks(rows, kind)
    y2, xd2 = _run(ex, x, picks, fat=False)
    assert torch.isfinite(y2).all() and y2.abs().max() > 0
    from tensorfold.families.glm5_next.cuda import exl3_mm

    assert exl3_mm.shared_suh(ex) == shared
    for stages in (3, 4):
        for ticket in (True, False):
            yf, xdf = _run(ex, x, picks, fat=True, stages=stages, ticket=ticket)
            assert torch.equal(xdf, xd2), ("Xd", stages, ticket)
            assert torch.equal(yf, y2), ("Y", stages, ticket)


@gpu
def test_rot_in1_equals_rot_in():
    from tensorfold.families.glm5_next.cuda import exl3_mm

    ex = _layer(False)
    x, picks = _picks(333, "skewed")
    x, picks = x.cuda(), picks.cuda()
    P = 333 * (TOP + 1)
    a, b = (torch.zeros((P, D), dtype=torch.float16, device="cuda") for _ in range(2))
    c = torch.full((P, D), 7.0, dtype=torch.float16, device="cuda")
    exl3_mm._ext().rot_in(x, x.stride(0), picks, ex.suh_g, ex.suh_u, a, b, 333, D, TOP + 1)
    exl3_mm._fast_ext().rot_in1(x, x.stride(0), picks, ex.suh_g, c, 333, D, TOP + 1)
    torch.cuda.synchronize()
    routed = torch.ones(333, TOP + 1, dtype=torch.bool)
    routed[:, TOP] = False                                            # the shared expert's slot is not written
    routed = routed.flatten().cuda()
    assert torch.equal(c[routed], a[routed])
    assert (c[~routed] == 7.0).all()


@gpu
@pytest.mark.parametrize("shared", [True, False], ids=["shared_x", "own_x"])
def test_fat_row_independent_and_repeatable(shared):
    ex = _layer(shared)
    x, picks = _picks(700, "skewed")
    y, xd = _run(ex, x, picks, fat=True)
    y2, xd2 = _run(ex, x, picks, fat=True, ticket=True)
    assert torch.equal(y, y2) and torch.equal(xd, xd2)                 # ticket order varies between runs
    for sub in ([5], list(range(64, 200)), list(range(699, -1, -3)), list(np.random.default_rng(2).permutation(700))):
        s = torch.tensor(sub)
        ys, xds = _run(ex, x[s], picks[s], fat=True)
        assert torch.equal(ys, y[s]) and torch.equal(xds, xd[s]), sub[:3]


@gpu
@pytest.mark.parametrize("rows", [1024, 4096, 8192])
def test_fat_timing(rows):
    """Prints ms of one MoE layer's rot_in + gate/up + down per rank: fast2, fat (3 / 4 stages, ticket on / off)."""

    ex = _layer(True)
    x, picks = _picks(rows, "skewed")

    def ms(**kw):
        _run(ex, x, picks, **kw)
        t = []
        for _ in range(5):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            _run(ex, x, picks, **kw)
            b.record()
            torch.cuda.synchronize()
            t.append(a.elapsed_time(b))
        return sorted(t)[2]

    base = ms(fat=False)
    line = [f"rows {rows}: fast2 {base:.2f} ms"]
    for stages in (3, 4):
        for ticket in (True, False):
            v = ms(fat=True, stages=stages, ticket=ticket)
            line.append(f"fat s{stages} t{int(ticket)} {v:.2f} ({base / v:.2f}x)")
    print("[0170] " + "; ".join(line), "(includes the host-side grouping and a Scratch allocation: compare ratios)")


def _kda_q4(n=12576, k=4096, seed=0):
    from tensorfold.families.glm5_next.cuda import qmm

    g = torch.Generator().manual_seed(seed)
    w = (torch.randn((n, k), generator=g) * 0.02).to(torch.bfloat16).cuda()
    return qmm.quantize4(w, mse=True)


@gpu
def test_kda_copy_row_independent_and_close():
    from tensorfold.families.glm5_next.cuda import fast_qmm, qmm

    q = _kda_q4()
    c = qmm.make_b16(qmm.dequantize_q4(q))
    x = (torch.randn((1000, q.k), generator=torch.Generator().manual_seed(1)) * 0.5).to(torch.bfloat16).cuda()
    full = fast_qmm.matmul_prefill(x, c, exact=False, min_rows=1)
    ref = fast_qmm.matmul_prefill(x, q, qmm.group_sums(x), exact=False, min_rows=1, f32=True)
    rel = ((full.float() - ref).norm() / ref.norm()).item()
    assert rel < 5e-3, rel                                             # weights rounded to bf16 once, bf16 output
    for a, b in ((0, 1), (3, 66), (64, 128), (17, 999), (0, 63), (500, 1000)):
        part = fast_qmm.matmul_prefill(x[a:b], c, exact=False, min_rows=1)
        assert torch.equal(part, full[a:b]), (a, b)
    perm = torch.randperm(1000, generator=torch.Generator().manual_seed(3)).cuda()
    assert torch.equal(fast_qmm.matmul_prefill(x[perm], c, exact=False, min_rows=1), full[perm])


@gpu
@pytest.mark.parametrize("rows", [1024, 8192])
def test_kda_copy_timing(rows):
    """Prints ms of the 12576 x 4096 KDA projection in a fast chunk: 4-bit (``matmul_fast``) vs the bf16 copy, for the
    default bf16 tile and a few others (GLM53_TF_KDA_BF16_TILE)."""

    from tensorfold.families.glm5_next.cuda import fast_qmm, qmm

    q = _kda_q4()
    c = qmm.make_b16(qmm.dequantize_q4(q))
    x = (torch.randn((rows, q.k)) * 0.5).to(torch.bfloat16).cuda()
    xs = qmm.group_sums(x)
    out = torch.empty((rows, q.n), dtype=torch.bfloat16, device="cuda")

    def ms(fn):
        fn()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(10):
            fn()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) / 10

    q4 = ms(lambda: fast_qmm.matmul_prefill(x, q, xs, out=out, exact=False, min_rows=1))
    line = [f"rows {rows}: q4 {q4:.2f} ms"]
    for tile in (None, (128, 64, 4, 3), (128, 128, 8, 3), (128, 128, 4, 3), (64, 128, 4, 3), (256, 128, 8, 3)):
        c.tile = tile
        try:
            v = ms(lambda: fast_qmm.matmul_prefill(x, c, out=out, exact=False, min_rows=1))
            line.append(f"bf16 {tile or 'default'} {v:.2f} ({q4 / v:.2f}x)")
        except Exception as exc:                                       # noqa: BLE001 - e.g. out of shared memory
            line.append(f"bf16 {tile}: {type(exc).__name__}")
    print("[0170] " + "; ".join(line))


# -- GPU: engine -----------------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_lookup(monkeypatch):
    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")
    monkeypatch.setenv("GLM53_TF_NONEXPERT", "q4mse")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_mia170")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


def _share_suh(eng):
    """Make every EXL3 expert layer's up sign vector the gate's (the real checkpoint's property), for both paths."""

    from tensorfold.families.glm5_next.cuda import exl3_mm

    n = 0
    for layer in list(eng.w.layers) + ([eng.w.mtp] if getattr(eng.w, "mtp", None) is not None else []):
        for obj in (layer, getattr(layer, "moe", None), getattr(getattr(layer, "layer", None), "moe", None)):
            ex = getattr(obj, "experts", None)
            if isinstance(ex, exl3_mm.Exl3Experts):
                ex.suh_u = ex.suh_g
                ex.__dict__.pop("_shared_suh", None)
                n += 1
    assert n > 0
    eng.cache = []


def _engine(path, *, lean_block=0, rows_max=1024, kda_bf16=False):
    sys.path.insert(0, str(Path(__file__).parent))
    from test_cindep_patches import _engine as cindep_engine

    with pytest.MonkeyPatch.context() as m:
        m.setenv("GLM53_TF_KDA_PROJ_BF16", "1" if kda_bf16 else "0")
        eng = cindep_engine(path, lean_block=lean_block, rows_max=rows_max)
    _share_suh(eng)
    return eng


@pytest.fixture(scope="module")
def ea(ckpt):
    return _engine(ckpt)


@pytest.fixture(scope="module")
def eb(ckpt):
    return _engine(ckpt, lean_block=256, rows_max=8192)


@pytest.fixture(scope="module")
def ek(ckpt):
    return _engine(ckpt, lean_block=256, rows_max=8192, kda_bf16=True)


class _Fat:
    def __init__(self, on: bool):
        self.on = on

    def __enter__(self):
        from tensorfold.families.glm5_next.cuda import exl3_mm

        self.saved = exl3_mm.FAT
        exl3_mm.FAT = self.on

    def __exit__(self, *exc):
        from tensorfold.families.glm5_next.cuda import exl3_mm

        exl3_mm.FAT = self.saved


@gpu
@pytest.mark.parametrize("n", [3, 65, 300, 1000])
def test_engine_state_fat_equals_fast2(ea, eb, n):
    from test_cindep_patches import _Variant, _same, _state

    prompt = list(np.random.default_rng(170 + n).integers(0, 1000, size=n))
    for eng, rows in ((ea, 1024), (eb, 8192), (eb, 256)):
        with _Variant(eng, rows):
            with _Fat(False):
                ref = _state(eng, prompt)
            with _Fat(True):
                got = _state(eng, prompt)
        assert _same(ref, got), (rows, [i for i, (a, b) in enumerate(zip(ref, got)) if not torch.equal(a, b)])


@gpu
@pytest.mark.parametrize("n", [3, 64, 65, 300, 1000])
def test_engine_kda_bf16_state_independent_of_chunk_size(ek, n):
    from test_cindep_patches import _Variant, _same, _state

    from tensorfold.families.glm5_next.cuda import fastpf

    kda = [l.kda for l in ek.w.layers if l.kind == "kda"]
    assert kda and all(getattr(k, "proj16", None) is not None for k in kda)
    prompt = list(np.random.default_rng(270 + n).integers(0, 1000, size=n))
    got = {}
    for rows in (64, 1024, 8192):
        for fat in (False, True):
            with _Variant(ek, rows), _Fat(fat):
                got[(rows, fat)] = _state(ek, prompt)
    ref = got[(64, False)]
    bad = [k for k, v in got.items() if not _same(ref, v)]
    assert not bad, bad


@gpu
def test_engine_kda_bf16_close_to_q4(eb, ek):
    """The copies change a fast prefill's arithmetic a little: the first tokens agree with the 4-bit path on most
    prompts (a sanity bound, not an exactness claim)."""

    from test_cindep_patches import _Variant, _state

    agree = 0
    for seed in range(8):
        prompt = list(np.random.default_rng(370 + seed).integers(0, 1000, size=300))
        with _Variant(eb, 1024):
            a = _state(eb, prompt)[0]
        with _Variant(ek, 1024):
            b = _state(ek, prompt)[0]
        agree += int(torch.equal(a, b))
    assert agree >= 5, agree


@gpu
@pytest.mark.parametrize("which", ["fat", "kda_bf16"])
def test_engine_drafted_serial_and_resumed_fresh(eb, ek, which):
    from test_cindep_patches import _cold, _gen, _sampling

    eng = ek if which == "kda_bf16" else eb
    knobs = {"fat_experts": 1} if which == "fat" else {"fat_experts": 0}
    rng = np.random.default_rng(470)
    more = lambda n: [int(t) for t in rng.integers(0, 1000, size=n)]       # noqa: E731
    for sampling in ("sampled", "greedy"):
        s = _sampling(sampling)
        p1 = more(900)
        eng.cache = []
        r1, _ = _gen(eng, p1, s, policy="auto:1:1:0", knobs=dict(knobs, prefill_rows=8192))
        serial = _cold(eng, p1, s, knobs=dict(knobs, prefill_rows=1024))
        assert r1 == serial[:len(r1)], sampling                               # drafted == serial
        _gen(eng, p1, s, knobs=dict(knobs, prefill_rows=8192))
        p2 = p1 + r1 + more(80)
        warm, stats = _gen(eng, p2, s, knobs=dict(knobs, prefill_rows=64))
        assert stats["cached"] > 0
        assert warm == _cold(eng, p2, s, knobs=dict(knobs, prefill_rows=4096)), sampling   # resumed == fresh


@gpu
def test_fat_knob_shares_snapshots_with_fast2(eb):
    """fat == fast2, so a fast2 request resumes from a fat request's snapshot (and the reverse) with the same reply."""

    from test_cindep_patches import _cold, _gen, _sampling

    s = _sampling("greedy")
    rng = np.random.default_rng(570)
    p1 = [int(t) for t in rng.integers(0, 1000, size=700)]
    eb.cache = []
    r1, _ = _gen(eb, p1, s, knobs={"fat_experts": 1, "prefill_rows": 1024})
    p2 = p1 + r1 + [int(t) for t in rng.integers(0, 1000, size=50)]
    warm, stats = _gen(eb, p2, s, knobs={"fat_experts": 0, "prefill_rows": 1024})
    assert stats["cached"] > 0
    assert warm == _cold(eb, p2, s, knobs={"fat_experts": 1, "prefill_rows": 4096})
