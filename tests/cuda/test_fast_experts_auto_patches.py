"""patches/0270: ``GLM53_TF_FAST_EXPERTS=auto`` -- the fast chunks' routed-expert kernels picked per chunk by its rows.

W2's bench (``bench_experts.py``): fast2 is 1.13-1.24x faster than fat at 1,024-2,048 rows, fat is faster from ~4,096
rows. ``auto`` runs fast2 for chunks of ``lo <= R < hi`` rows (``GLM53_TF_FAST2_ROWS=lo,hi``, default ``0,4096``) and
the fat family (fat, or once when configured) otherwise. fast2 there reads the one rotated input (``rot_in1``) when
gate and up share their sign vector, as fat does. fast2 == fat == once bit for bit, so which kernel a chunk gets never
changes a bit (0085's row independence, shared snapshots).

Checked:

- host only: the env switch (``auto`` is a fat-family mode; ``GLM53_TF_FAST2_ROWS`` parsed and checked), the
  per-request knob (``fat_experts`` 0 / 1 / 2 <-> ``family`` / ``set_family``, the knob range), and the dispatch in
  ``exl3_mm.routed`` with fake extensions (fast2 inside the window on rot_in1's shared input, fat / once outside,
  both rotations when the sign vectors differ);
- GPU kernels: auto == fast2 == fat bit for bit on both sides of the threshold (shared and own inputs, uniform and
  skewed routing); rows of a chunk run by fat equal the same rows in a chunk run by fast2 (row subsets across the
  threshold);
- GPU engine (synthetic EXL3 checkpoint, gate / up sign vectors equal as in the real one): committed state auto ==
  fast2 with a threshold inside the chunk sizes used; drafted == serial and resumed == fresh with auto; an auto
  request resumes a fast2 request's snapshot and the reverse.

Run inside the image:
    PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q -s tests/cuda/test_fast_experts_auto_patches.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace as NS

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

LIMIT = 10.0
_ENVS = ("GLM53_TF_FAST_EXPERTS", "GLM53_TF_FAST2_ROWS", "GLM53_TF_ONCE_MIN_ROWS")


# -- host only ---------------------------------------------------------------------------------------------------------
@needs_torch
def test_env_modes(monkeypatch):
    from tensorfold.families.glm5_next.cuda import exl3_mm

    try:
        for mode, fat, auto, once in (("auto", True, True, False), ("AUTO", True, True, False),
                                      ("fat", True, False, False), ("once", True, False, True),
                                      ("fast2", False, False, False), (None, False, False, False)):
            for k in _ENVS:
                monkeypatch.delenv(k, raising=False)
            if mode is not None:
                monkeypatch.setenv("GLM53_TF_FAST_EXPERTS", mode)
            importlib.reload(exl3_mm)
            assert (exl3_mm.FAT, exl3_mm.AUTO, exl3_mm.ONCE) == (fat, auto, once), mode
            assert exl3_mm.FAST2_ROWS == (0, 4096)
            assert exl3_mm.family() == (2 if auto else int(fat))
        monkeypatch.setenv("GLM53_TF_FAST_EXPERTS", "auto")
        monkeypatch.setenv("GLM53_TF_FAST2_ROWS", "256,2049")
        importlib.reload(exl3_mm)
        assert exl3_mm.FAST2_ROWS == (256, 2049)
        assert [exl3_mm.auto_fast2(r) for r in (64, 255, 256, 2048, 2049, 8192)] == [False, False, True, True, False,
                                                                                     False]
        for bad in ("4096", "10,5", "-1,5", "a,b"):
            monkeypatch.setenv("GLM53_TF_FAST2_ROWS", bad)
            with pytest.raises(ValueError):
                importlib.reload(exl3_mm)
        monkeypatch.setenv("GLM53_TF_FAST2_ROWS", "0,4096")
        monkeypatch.setenv("GLM53_TF_FAST_EXPERTS", "auto2")
        with pytest.raises(ValueError, match="auto"):
            importlib.reload(exl3_mm)
    finally:
        for k in _ENVS:
            monkeypatch.delenv(k, raising=False)
        importlib.reload(exl3_mm)


@needs_torch
def test_family_knob(monkeypatch):
    from tensorfold.families.glm5_next.cuda import exl3_mm, knobs

    monkeypatch.setattr(exl3_mm, "FAT", False)
    monkeypatch.setattr(exl3_mm, "AUTO", False)
    for v, fat, auto in ((2, True, True), (1, True, False), (0, False, False), (2, True, True)):
        exl3_mm.set_family(v)
        assert (exl3_mm.FAT, exl3_mm.AUTO, exl3_mm.family()) == (fat, auto, v)
    with pytest.raises(ValueError):
        exl3_mm.set_family(3)
    assert knobs.RANGES["fat_experts"] == (0, 2)
    assert knobs.parse({"fat_experts": 2}, rows_max=512) == {"fat_experts": 2}
    with pytest.raises(ValueError, match="0 to 2"):
        knobs.parse({"fat_experts": 3}, rows_max=512)
    values = {k: 0 for k in knobs.HEADER}
    values["fat_experts"] = 2
    got, rest = knobs.decode(knobs.encode(values) + [5])
    assert got == values and rest == [5]
    monkeypatch.setattr(exl3_mm, "AUTO", False)
    monkeypatch.setattr(exl3_mm, "FAST2_ROWS", (0, 4096))
    assert not exl3_mm.auto_fast2(100)                        # only while auto is the family


class _FakeExt:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: self.calls.append((name, a, k))


@needs_torch
@pytest.mark.parametrize("fat,auto,once,window,shared,want", [
    (True, True, False, (0, 4096), True, "fast2"),
    (True, True, False, (0, 4), True, "fat"),              # R = 4 is outside [0, 4)
    (True, True, False, (5, 99), True, "fat"),
    (True, True, True, (8, 99), True, "once"),
    (True, True, True, (0, 99), True, "fast2"),
    (True, True, False, (0, 4096), False, "fast2"),
    (True, False, False, (0, 4096), True, "fat"),
    (False, False, False, (0, 4096), True, "plain"),
])
def test_routed_dispatch(monkeypatch, fat, auto, once, window, shared, want):
    from tensorfold.families.glm5_next.cuda import exl3_mm

    ext, fx = _FakeExt(), _FakeExt()
    monkeypatch.setattr(exl3_mm, "_ext", lambda: ext)
    monkeypatch.setattr(exl3_mm, "_fast_ext", lambda: fx)
    monkeypatch.setattr(exl3_mm, "V2", None)
    monkeypatch.setattr(exl3_mm, "FAT", fat)
    monkeypatch.setattr(exl3_mm, "AUTO", auto)
    monkeypatch.setattr(exl3_mm, "ONCE", once)
    monkeypatch.setattr(exl3_mm, "ONCE_MIN_ROWS", 0)
    monkeypatch.setattr(exl3_mm, "FAST2_ROWS", window)
    monkeypatch.setattr(exl3_mm, "FAT_TICKET", True)
    monkeypatch.setattr(exl3_mm, "FAT_SHARED_X", True)
    suh = torch.ones((2, 256), dtype=torch.float16)
    z = lambda *s: torch.zeros(s, dtype=torch.float16)            # noqa: E731
    ex = exl3_mm.Exl3Experts(torch.zeros(1), torch.zeros(1), torch.zeros(1), suh, suh.clone() if shared else -suh,
                             z(2, 128), z(2, 128), z(2, 128), z(2, 256), 2, 128, 256)
    s = NS(slots=3, rows=4, xg=z(12, 256), xu=z(12, 256), xd=z(12, 128))
    grp = NS(ids=torch.zeros(3, dtype=torch.int32), count=torch.ones(1, dtype=torch.int32),
             members=torch.full((3, 4), -1, dtype=torch.int32))
    x = torch.zeros((4, 256), dtype=torch.bfloat16)
    exl3_mm.routed(x, torch.zeros((4, 3), dtype=torch.int32), grp, ex, s, torch.zeros((12, 256)), 4, LIMIT, fast=True)
    names = [c[0] for c in ext.calls + fx.calls]
    rot = "rot_in1" if shared else "rot_in"
    if want == "plain":
        assert names == ["rot_in", "gateup", "down"]
        return
    assert names == {"fast2": [rot, "gateup", "down"], "fat": [rot, "gateup_fat", "down_fat"],
                     "once": [rot, "gateup_once", "down_once"]}[want]
    gu = fx.calls[1][1] if shared else fx.calls[0][1]
    assert (gu[1] is gu[0]) == shared                              # XU = XG exactly when the input is shared


# -- GPU: kernels --------------------------------------------------------------------------------------------------------
def _layer(shared: bool):
    sys.path.insert(0, str(Path(__file__).parent))
    from test_expert_once_patches import _layer as layer

    return layer(shared)


def _run(ex, x, picks, mode, window=(0, 4096)):
    """One MoE layer's routed experts (fast=True) with ``mode`` in fast2 / fat / auto -> (Y, Xd) of the routed slots."""
    sys.path.insert(0, str(Path(__file__).parent))
    from test_fastpf_patches import _fast_group

    from tensorfold.families.glm5_next.cuda import exl3_mm

    names = ("FAT", "AUTO", "ONCE", "FAST2_ROWS", "FAT_SHARED_X", "FAT_STAGES", "FAT_TICKET")
    saved = {k: getattr(exl3_mm, k) for k in names}
    exl3_mm.FAT, exl3_mm.AUTO, exl3_mm.ONCE = mode in ("fat", "auto"), mode == "auto", False
    exl3_mm.FAST2_ROWS, exl3_mm.FAT_SHARED_X, exl3_mm.FAT_STAGES, exl3_mm.FAT_TICKET = window, True, 3, True
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


def _picks(rows, kind):
    sys.path.insert(0, str(Path(__file__).parent))
    from test_expert_once_patches import _picks as picks

    return picks(rows, kind)


@gpu
@pytest.mark.parametrize("shared", [True, False], ids=["shared_x", "own_x"])
@pytest.mark.parametrize("rows,kind", [(64, "uniform"), (700, "skewed"), (2048, "uniform"), (4096, "skewed"),
                                       (4160, "uniform")])
def test_auto_equals_fast2_and_fat_bitwise(rows, kind, shared):
    ex = _layer(shared)
    x, picks = _picks(rows, kind)
    y2, xd2 = _run(ex, x, picks, "fast2")
    assert torch.isfinite(y2).all() and y2.abs().max() > 0
    yf, xdf = _run(ex, x, picks, "fat")
    assert torch.equal(yf, y2) and torch.equal(xdf, xd2), "fat != fast2 (0170 broken?)"
    for window in ((0, 4096), (0, 1 << 30), (0, 0), (rows, rows + 1)):
        ya, xda = _run(ex, x, picks, "auto", window)
        assert torch.equal(xda, xd2) and torch.equal(ya, y2), window


@gpu
@pytest.mark.parametrize("shared", [True, False], ids=["shared_x", "own_x"])
def test_rows_across_the_threshold(shared):
    """A 4,160-row chunk runs fat under auto (0, 4096); its row subsets of < 4,096 rows run fast2: the same rows."""
    ex = _layer(shared)
    rows = 4160
    x, picks = _picks(rows, "skewed")
    y, xd = _run(ex, x, picks, "auto")
    rng = np.random.default_rng(7)
    for sub in ([3], list(range(64, 1088)), list(rng.permutation(rows)[:4095]), list(range(0, rows, 2))):
        s = torch.tensor(sub)
        ys, xds = _run(ex, x[s], picks[s], "auto")
        assert torch.equal(ys, y[s]) and torch.equal(xds, xd[s]), len(sub)


# -- GPU: engine ---------------------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_lookup(monkeypatch):
    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")
    monkeypatch.setenv("GLM53_TF_NONEXPERT", "q4mse")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_auto270")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


@pytest.fixture(scope="module")
def eb(ckpt):
    sys.path.insert(0, str(Path(__file__).parent))
    from test_mia_prefill_patches import _engine as mia_engine

    return mia_engine(ckpt, lean_block=256, rows_max=8192)


class _Auto:
    """auto with a threshold inside the chunk sizes the tests use (chunks of >= 512 rows run fat, smaller ones fast2)."""

    def __init__(self, window=(0, 512)):
        self.window = window

    def __enter__(self):
        from tensorfold.families.glm5_next.cuda import exl3_mm

        self.saved = (exl3_mm.FAT, exl3_mm.AUTO, exl3_mm.FAST2_ROWS)
        exl3_mm.FAST2_ROWS = self.window
        return self

    def __exit__(self, *exc):
        from tensorfold.families.glm5_next.cuda import exl3_mm

        exl3_mm.FAT, exl3_mm.AUTO, exl3_mm.FAST2_ROWS = self.saved


@gpu
@pytest.mark.parametrize("n", [65, 300, 1000])
def test_engine_state_auto_equals_fast2(eb, n):
    from test_cindep_patches import _Variant, _same, _state

    from tensorfold.families.glm5_next.cuda import exl3_mm

    prompt = list(np.random.default_rng(270 + n).integers(0, 1000, size=n))
    with _Auto():
        for rows in (8192, 1024, 256):
            with _Variant(eb, rows):
                exl3_mm.set_family(0)
                ref = _state(eb, prompt)
                exl3_mm.set_family(2)
                got = _state(eb, prompt)
                assert _same(ref, got), (rows, [i for i, (a, b) in enumerate(zip(ref, got)) if not torch.equal(a, b)])


@gpu
def test_engine_drafted_serial_and_resumed_fresh(eb):
    from test_cindep_patches import _cold, _gen, _sampling

    with _Auto():
        knobs = {"fat_experts": 2}
        rng = np.random.default_rng(470)
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
@pytest.mark.parametrize("first,second", [(2, 0), (0, 2), (2, 1)])
def test_auto_shares_snapshots(eb, first, second):
    from test_cindep_patches import _cold, _gen, _sampling

    with _Auto():
        s = _sampling("greedy")
        rng = np.random.default_rng(570 + first)
        p1 = [int(t) for t in rng.integers(0, 1000, size=700)]
        eb.cache = []
        r1, _ = _gen(eb, p1, s, knobs={"fat_experts": first, "prefill_rows": 1024})
        p2 = p1 + r1 + [int(t) for t in rng.integers(0, 1000, size=50)]
        warm, stats = _gen(eb, p2, s, knobs={"fat_experts": second, "prefill_rows": 1024})
        assert stats["cached"] > 0
        assert warm == _cold(eb, p2, s, knobs={"fat_experts": first, "prefill_rows": 4096})
