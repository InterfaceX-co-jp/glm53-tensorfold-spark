"""patches/0335: solo prefill pieces (``GLM53_TF_SOLO_PIECE=N``, default 0 = off) and the lean set's lazy Xu
(``GLM53_TF_LEAN_LAZY_XU=1``).

In batch mode a prompt prefills in pieces of ``GLM53_TF_BATCH_PIECE`` tokens (2,048 in production) so that other
requests' rounds run between them. When the request is ALONE in the batch (no other slot holds a request when its next
piece is cut), 0335 cuts the piece at N tokens instead (fast prefills only), and at the batch piece again as soon as
another request is admitted. With ``GLM53_TF_PREFILL_ROWS_MAX`` >= N the piece runs as one N-row fast chunk, so the
routed experts read their weights once per N rows. Same bits: pieces end on the 64-token snapshot grid, and a fast
prefill resumed from any grid snapshot equals a fresh one (patches/0085); the choice is made from the slots, which
both ranks hold alike.

Checked:

- host only: ``batchplan.solo_piece`` (solo only for fast prefills, only alone); piece bounds under random schedules of
  solo and normal pieces (every end on the grid, increasing, the last one the prompt's end, the same as the cuts
  0180's session resumes need); ``Batcher._piece`` with a fake engine: the prefill it runs ends at the solo bound when
  the slot is alone and at the batch bound when another slot is busy, counts solo pieces into the stats, and
  GLM53_TF_SOLO_PIECE is refused below 64; the lazy Xu of the lean set (not allocated, then allocated on first use,
  counted in ``nbytes`` only then);
- GPU (synthetic EXL3 checkpoint, one GPU as rank 0 of two): a fast lean prompt alone in a 2-slot batch prefills in
  solo pieces and its reply equals the lone engine's; a second request admitted mid-prefill turns the first one's
  next pieces into normal ones, both replies exact; the follower (rank 1) replays the same rounds and pieces; the
  lazy Xu engine gives the same replies.

Run inside the image:
    PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q tests/cuda/test_solo_piece_patches.py
"""

from __future__ import annotations

import random
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tensorfold.families.glm5_next.cuda import batchplan

try:
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:          # the host-only tests still run
    torch = None
    CUDA = False

gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")
needs_torch = pytest.mark.skipif(torch is None, reason="torch")


# -- host only -----------------------------------------------------------------------------------------------------
def test_solo_piece_rule():
    assert batchplan.solo_piece(8192, 2048, 64, []) == 8192
    assert batchplan.solo_piece(8192, 2048, 64, [2]) == 2048            # another request holds a slot
    assert batchplan.solo_piece(8192, 2048, 0, []) == 2048              # exact prefill: unchanged
    assert batchplan.solo_piece(0, 2048, 64, []) == 2048                # off


def test_piece_bounds_with_solo_and_normal_pieces():
    """Any mix of solo and normal pieces (and 0180's resume points between grid bounds) cuts a prompt into increasing
    pieces that end on the grid, the last at the prompt's end; with solo pieces there are never more pieces."""
    rng = random.Random(335)
    for _ in range(3000):
        grid = rng.choice([64, 128, 1024])
        piece = rng.choice([64, 256, 2048])
        solo = rng.choice([4096, 8192, 700])
        n = rng.randint(1, 60000)
        done = rng.choice([0, 0, 0, rng.randrange(0, n, 64)]) if n > 64 else 0
        ends, normal = [], []
        d = done
        while d < n:
            alone = rng.random() < 0.6
            step = batchplan.solo_piece(solo, piece, grid, [] if alone else [1])
            e = batchplan.piece_end(d, n, step, grid)
            assert d < e <= n and (e == n or e % 64 == 0), (d, e, n)
            if e < n and d % grid == 0:
                assert e % grid == 0
            ends.append(e)
            d = e
        d = done
        while d < n:
            d = batchplan.piece_end(d, n, piece, grid)
            normal.append(d)
        assert ends[-1] == n and ends == sorted(set(ends))
        if solo >= piece:
            d, alone_ends = done, []
            while d < n:
                d = batchplan.piece_end(d, n, batchplan.solo_piece(solo, piece, grid, []), grid)
                alone_ends.append(d)
            assert len(alone_ends) <= len(normal)


class _Ctx:
    def __init__(self, v=None):
        self.v = v

    def __enter__(self):
        return self.v

    def __exit__(self, *a):
        return False


def _fake_batcher(monkeypatch, *, solo: int, piece: int, busy: list[int], n: int, done: int = 0):
    """A Batcher without __init__ around a fake engine: ``_piece(0)`` runs one piece of an ``n``-token fast prompt;
    the fake prefill records the prompt length it was given and leaves the grid snapshot there."""
    from tensorfold.families.glm5_next.cuda import batch, decode

    calls = []

    def prefill(e, ids, sampling, *, mtp, drafter, resume):
        calls.append((len(ids), resume))
        e.fast_snap = NS(ids=list(ids))
        return 0

    monkeypatch.setattr(decode, "prefill", prefill)
    b = batch.Batcher.__new__(batch.Batcher)
    b.piece, b.solo = piece, solo
    e = NS(prefill_rows=0, snap_grid=64, fast_prefill=True, fast_snap=None, checkpoints=(), mark_snaps=[],
           last_hidden=None)
    g = NS(_drafters=lambda code: (None, False, False), _knobs=lambda values: _Ctx(), _grid=lambda: 64)
    b.g = g
    b.drafters = [None] * 4
    b.counts = __import__("collections").Counter()
    b._on = lambda slot: _Ctx(e)
    b._borrow = lambda slot: _Ctx()
    b._save = lambda slot, saves: None
    b._remember = lambda slot, snap: None
    b.store = None
    job = NS(prompt=list(range(n)), sampling=None, code=0, values={}, draft=False, stats={}, out=None)
    b.seqs = [None] * 4
    b.seqs[0] = batch.Seq(job, 0, 0, done=done, grid=64)
    for s in busy:
        b.seqs[s] = batch.Seq(NS(prompt=[1], stats={}), s, 0)
    return b, calls, job


@needs_torch
@pytest.mark.parametrize("busy", [[], [2]])
def test_piece_uses_the_solo_bound_only_alone(monkeypatch, busy):
    b, calls, job = _fake_batcher(monkeypatch, solo=8192, piece=2048, busy=busy, n=20000)
    b._piece(0)
    want = 8192 if not busy else 2048
    assert calls[-1][0] == want and b.seqs[0].done == want
    assert b.seqs[0].solo == (0 if busy else 1)
    # the next piece: a request arrives (or leaves) in between -> the rule is re-evaluated at the boundary
    if busy:
        b.seqs[2] = None
    else:
        b.seqs[3] = b.seqs[0].__class__(NS(prompt=[1], stats={}), 3, 1)
    b._piece(0)
    assert calls[-1][0] == want + (8192 if busy else 2048)


@needs_torch
def test_last_piece_stats(monkeypatch):
    """A 9,000-token prompt alone: pieces end at 4,096, 8,192 and 9,000; the stats count 3 solo pieces (the fake
    stops at the first token, after the stats)."""
    from tensorfold.families.glm5_next.cuda import batch

    class _Stop(Exception):
        pass

    def emit(self, seq, toks):
        raise _Stop

    b, calls, job = _fake_batcher(monkeypatch, solo=4096, piece=2048, busy=[], n=9000)
    monkeypatch.setattr(batch.Batcher, "_emit", emit)
    with pytest.raises(_Stop):
        for _ in range(3):
            b._piece(0)
    assert [c[0] for c in calls] == [4096, 8192, 9000]
    assert job.stats["pieces"] == 3 and job.stats["solo_pieces"] == 3


@needs_torch
def test_solo_refused_below_the_grid(monkeypatch):
    """The env is parsed with the other batch settings; values 1-63 are refused (0 = off)."""
    from tensorfold.families.glm5_next.cuda import batch

    src = Path(batch.__file__).read_text()
    assert 'GLM53_TF_SOLO_PIECE' in src and "self.solo" in src
    for v, ok in (("0", True), ("64", True), ("8192", True), ("63", False), ("-1", False)):
        monkeypatch.setenv("GLM53_TF_SOLO_PIECE", v)
        val = batch._env_int("GLM53_TF_SOLO_PIECE", 0)
        assert ok == (val == 0 or val >= 64), v


@needs_torch
def test_lazy_xu(monkeypatch):
    from tensorfold.families.glm5_next.cuda import lean

    eager = lean._Exl3Rows(16, 9, 256, 128, "cpu")
    lazy = lean._Exl3Rows(16, 9, 256, 128, "cpu", lazy_xu=True)
    assert eager._xu is not None and lazy._xu is None
    x = lazy.xu
    assert x.shape == eager.xu.shape and x.dtype == torch.float16 and lazy.xu is x
    monkeypatch.setenv("GLM53_TF_LEAN_LAZY_XU", "1")
    assert lean.lazy_xu()
    monkeypatch.setenv("GLM53_TF_LEAN_LAZY_XU", "0")
    assert not lean.lazy_xu()


def test_memory_table():
    """docs/EXPERT-TC.md's 0335 memory numbers: the lean set is 397 KiB a row (72 of it Xu)."""
    per_row, xu = 397 * 1024, 9 * 4096 * 2
    gib = lambda rows, lazy=False: rows * (per_row - (xu if lazy else 0)) / 2**30          # noqa: E731
    assert round(gib(2048), 2) == 0.78 and round(gib(4096), 2) == 1.55 and round(gib(8192), 2) == 3.10
    assert round(gib(8192) - gib(2048), 2) == 2.33
    assert round(gib(8192, True) - gib(2048), 2) == 1.76
    assert round(gib(4096, True) - gib(2048), 2) == 0.49


# -- GPU --------------------------------------------------------------------------------------------------------------
def _b2():
    sys.path.insert(0, str(Path(__file__).parent))
    import test_batch2_patches as b2

    return b2


def _solo_engine(ckpt, solo, **kw):
    b2 = _b2()
    with pytest.MonkeyPatch.context() as m:
        m.setenv("GLM53_TF_SOLO_PIECE", str(solo))
        return b2._engine(ckpt, **kw)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    if not CUDA:
        pytest.skip("CUDA only")
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_solo335")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


@gpu
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
def test_alone_prefills_in_solo_pieces(ckpt, greedy):
    b2 = _b2()
    lone = b2._engine(ckpt, fast=True, lean=True, block=64, rows=256, rows_max=512)
    eng = _solo_engine(ckpt, 512, batch=2, fast=True, lean=True, block=64, rows=256, rows_max=512, piece=128)
    sampling = b2._sampling(greedy)
    p = b2._prompt(335, 1500)
    want, _ = b2._run(lone, p, sampling, policy="auto", tokens=24)
    got, st = b2._run(eng, p, sampling, policy="auto", tokens=24)
    assert got == want
    assert st["pieces"] == -(-1500 // 512) and st.get("solo_pieces") == st["pieces"], st
    b2._free(lone, eng)


@gpu
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
def test_second_request_turns_pieces_normal(ckpt, greedy):
    """B arrives while A (alone) prefills: A's later pieces are 128-token ones, both replies exact."""
    b2 = _b2()
    # a 3,000-token prompt needs more than the default 2,051-token context (W8)
    lone = b2._engine(ckpt, fast=True, lean=True, block=64, rows=256, rows_max=512, context=4096)
    eng = _solo_engine(ckpt, 512, batch=2, fast=True, lean=True, block=64, rows=256, rows_max=512, piece=128,
                       share=0.5, context=4096)
    sampling = b2._sampling(greedy)
    pa, pb = b2._prompt(336, 3000), b2._prompt(337, 40)
    want_a, _ = b2._run(lone, pa, sampling, policy="auto", tokens=16)
    want_b, _ = b2._run(lone, pb, sampling, policy="2", tokens=40)
    got = {}
    t = threading.Thread(target=lambda: got.__setitem__("a", b2._run(eng, pa, sampling, policy="auto", tokens=16)))
    t.start()
    while not any(s is not None and s.pieces >= 1 for s in eng.batch.seqs):
        time.sleep(0.005)
    got["b"] = b2._run(eng, pb, sampling, policy="2", tokens=40)
    t.join(600)
    assert got["a"][0] == want_a and got["b"][0] == want_b
    sa = got["a"][1]
    assert 1 <= sa.get("solo_pieces", 0) < sa["pieces"], sa                  # solo first, normal once B came
    b2._free(lone, eng)


@gpu
def test_follower_replays_solo_rounds(ckpt):
    b2 = _b2()
    r0 = _solo_engine(ckpt, 512, batch=2, fast=True, lean=True, block=64, rows=256, rows_max=512, piece=128)
    r1 = _solo_engine(ckpt, 512, batch=2, fast=True, lean=True, block=64, rows=256, rows_max=512, piece=128)
    r1.batch.costs = r0.batch.costs
    r1.batch.round_costs = batchplan.RoundCosts(r0.batch.costs)
    sent: list[list[int]] = []
    share = r0._share

    class _Done(Exception):
        pass

    def record(values):
        out = share(values)
        sent.append(list(out))
        return out

    def replay(values):
        assert values is None
        if not sent:
            raise _Done
        return sent.pop(0)

    r0._share, r1._share = record, replay
    sampling = b2._sampling(True)
    r0.batch.generate_batch([dict(prompt=b2._prompt(338, 1400), max_tokens=20, sampling=sampling, policy="auto")])
    r0.batch.generate_batch([dict(prompt=b2._prompt(339, 900), max_tokens=20, sampling=sampling, policy="auto"),
                             dict(prompt=b2._prompt(340, 700), max_tokens=20, sampling=sampling, policy="2")])
    time.sleep(0.5)
    with pytest.raises(_Done):
        r1.batch.follow()
    key = lambda d: (d["sha256"], tuple(d["keeps"]), d["arms"], d["cancelled"])      # noqa: E731
    assert [key(d) for d in r1.batch.log] == [key(d) for d in r0.batch.log]
    assert [t[1:] for t in r1.batch.trace] == [t[1:] for t in r0.batch.trace]
    del r0._share, r1._share
    b2._free(r0, r1)


@gpu
def test_lazy_xu_engine_same_reply(ckpt):
    b2 = _b2()
    ref = b2._engine(ckpt, fast=True, lean=True, block=64, rows=256, rows_max=512)
    with pytest.MonkeyPatch.context() as m:
        m.setenv("GLM53_TF_LEAN_LAZY_XU", "1")
        eng = b2._engine(ckpt, fast=True, lean=True, block=64, rows=256, rows_max=512)
    assert eng.e.lean.exl3._xu is None and eng.e.lean.nbytes() < ref.e.lean.nbytes()
    sampling = b2._sampling(True)
    p = b2._prompt(341, 900)
    assert b2._run(eng, p, sampling, tokens=16)[0] == b2._run(ref, p, sampling, tokens=16)[0]
    b2._free(ref, eng)
