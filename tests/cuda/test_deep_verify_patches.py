"""patches/0380 (``GLM53_TF_MAX_DRAFT_ROWS``, 8 to 16, default 8; ``GLM53_TF_DFLASH_BLOCK``; ``glm5_next/cuda/deep.py``):
verify windows of up to 16 rows (a pending token and up to 15 drafts), used when the drafts are confident: the cost
depths of patches/0071 and the lookup gate of patches/0020 decide, with their caps raised.

The knob is read at import (as every load-time knob), so this file runs twice: in the calling process (the default,
8 rows: everything is upstream's) and, from ``test_at_16_rows_in_a_subprocess``, once more in a child process with
``GLM53_TF_MAX_DRAFT_ROWS=16`` where the ``at16`` tests run.

Checked:

- host only: the knobs (range, refusals, the DFlash2 block bound by the window); at 8 rows every cap is upstream's
  (``engine.MAX_ROWS``, ``depth`` / ``lookup`` ``MAX_DRAFTS``, the ``auto_fdrafts`` knob, ``latent.SPARSE_ROWS``) and
  ``calib.window_costs`` is the upstream fit; a 16-row calibration keeps the 1..8 table of the same times and extends
  it on its own line; at 16 rows the policy specs, knobs and caps accept 15 drafts and refuse 16; the cost depths
  verify past 7 drafts only when confident (DFlash2 picks, lookup bands, MTP chains) and the lookup copies 15 tokens;
- torch on the CPU, patches/0180's hostile fake model and the REAL ``batch.Stepper`` / ``Batcher._plan`` /
  ``_execute`` / ``_verify`` / ``follow`` (fake drafters that draft the true continuation or a wrong token,
  deterministically; a DFlash2 stand-in with a 16-row block): ``auto`` with cost depths and gated lookup, forced
  lookup ``l15:3``, ``of15``, ``om15``, requests alone and in shared rounds: every reply equals a fresh prefill +
  serial decoding, windows of 9-16 rows occur (kept and cut), and a follower batcher decides the same windows;
  ``decode.auto_decode`` (the lone engine's loop) likewise;
- GPU (TensorFold's synthetic checkpoints, one GPU playing rank 0 of two): past 2,051 tokens windows of 1-16 rows
  and MTP steps of 1-16 rows through the long-context graphs == the eager upstream step bit for bit; dense windows of
  9-16 rows == the rows of R serial steps; drafted replies (lone and 4 batched requests, greedy and sampled, with a
  16-row DFlash2 block) == serial.

Run: PYTHONPATH=<tree>/src:<tree>/tests/cuda:tests/cuda pytest -q tests/cuda/test_deep_verify_patches.py (inside the
image: PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda).
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold.families.glm5_next.cuda import batchplan, calib, deep, depth, knobs, lookup

try:
    import torch
except ImportError:          # the host-only tests still run
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="torch")
at8 = pytest.mark.skipif(deep.ROWS != 8, reason="the default (8-row) process")
at16 = pytest.mark.skipif(deep.ROWS != 16, reason="runs in the GLM53_TF_MAX_DRAFT_ROWS=16 child process")
CUDA = torch is not None and torch.cuda.is_available()
gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")

V8 = [31.0, 39.0, 45.0, 51.0, 56.0, 62.0, 68.0, 74.0]


def _costs(rows: int) -> dict:
    """The two-Spark window table (docs/PATCHES.md 0200) extended at ~5.75 ms a row, and the drafter costs."""

    verify = V8 + [V8[-1] + 5.75 * r for r in range(1, rows - 7)]
    return {"verify": verify[:rows], "mtp": 2.04, "mtp_step": 1.68, "mtp_row": 0.1, "block": 3.88, "taps_row": 0.05}


COSTS = _costs(deep.ROWS)


# -- the child process -------------------------------------------------------------------------------------------------
@at8
def test_at_16_rows_in_a_subprocess():
    """This file again with GLM53_TF_MAX_DRAFT_ROWS=16 (the knob is read at import): every ``at16`` test must run."""

    env = dict(os.environ, GLM53_TF_MAX_DRAFT_ROWS="16", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs", str(Path(__file__))],
                       env=env, capture_output=True, text=True, timeout=3600)
    tail = r.stdout[-6000:] + r.stderr[-2000:]
    assert r.returncode == 0, tail
    assert "passed" in r.stdout and "GLM53_TF_MAX_DRAFT_ROWS=16 child" not in r.stdout, tail


# -- host only -------------------------------------------------------------------------------------------------------
def test_knobs():
    assert deep.max_rows("") == 8 and deep.max_rows("16") == 16 and deep.max_rows(" 12 ") == 12
    for bad in ("7", "17", "0", "x", "8.5"):
        with pytest.raises(ValueError, match="GLM53_TF_MAX_DRAFT_ROWS"):
            deep.max_rows(bad)
    assert deep.dflash_block(16, "") == 0 and deep.dflash_block(16, "16") == 16 and deep.dflash_block(8, "8") == 8
    for rows, bad in ((8, "16"), (16, "17"), (16, "1"), (16, "y")):
        with pytest.raises(ValueError, match="GLM53_TF_DFLASH_BLOCK"):
            deep.dflash_block(rows, bad)
    assert deep.DRAFTS == deep.ROWS - 1


@at8
def test_default_is_upstream():
    from tensorfold.families.glm5_next.cuda import engine

    assert engine.MAX_ROWS == 8 and depth.MAX_DRAFTS == 7 and lookup.MAX_DRAFTS == 7
    assert knobs.RANGES["auto_fdrafts"] == (1, 7)
    assert depth.Calibration().kept["f"] == [0.0] * 7
    with pytest.raises(ValueError):
        depth.parse_policy("of8")
    with pytest.raises(ValueError):
        engine.encode_policy("c8:0.3")
    # upstream's fit: the windows of 2 rows and more on one line
    rng = random.Random(3)
    both = {f"v{r}_{s}": 30 + 6.2 * r + rng.random() for r in range(1, 9) for s in range(3)}
    import statistics

    v = {r: statistics.fmean(both[f"v{r}_{s}"] for s in range(3)) for r in range(1, 9)}
    base, slope = calib.fit_line(list(range(2, 9)), [v[r] for r in range(2, 9)])
    assert calib.window_costs(both, 8, 3) == [v[1]] + [base + slope * r for r in range(2, 9)]


@needs_torch
@at8
def test_default_sparse_rows():
    pytest.importorskip("triton")
    from tensorfold.families.glm5_next.cuda import latent

    assert latent.SPARSE_ROWS == 8


def test_calibration_table_past_8_rows():
    """16 windows timed: the first 8 are the 8-row table of the same times; 9..16 continue from the 8-row window's
    fitted cost along the slope of the measured 8..16 windows (a flatter or steeper top than the first 8)."""

    rng = random.Random(5)
    for top in (4.0, 6.5, 9.0):
        both = {}
        for s in range(3):
            for r in range(1, 17):
                base = 31.0 if r == 1 else 33.0 + 6.0 * r if r <= 8 else 33.0 + 48.0 + top * (r - 8)
                both[f"v{r}_{s}"] = base + rng.random() * 0.2
        t8, t16 = calib.window_costs(both, 8, 3), calib.window_costs(both, 16, 3)
        assert t16[:8] == t8 and len(t16) == 16
        steps = [t16[r] - t16[r - 1] for r in range(8, 16)]
        assert all(abs(x - top) < 0.3 for x in steps), (top, steps)
    for n in (9, 12):
        assert len(calib.window_costs({f"v{r}": 30.0 + 5 * r for r in range(1, n + 1)}, n, 0)) == n


@at16
def test_caps_follow_the_knob(monkeypatch):
    from tensorfold.families.glm5_next.cuda import engine

    assert engine.MAX_ROWS == 16 and depth.MAX_DRAFTS == 15 and lookup.MAX_DRAFTS == 15
    assert knobs.RANGES["auto_fdrafts"] == (1, 15)
    assert knobs.parse({"auto_fdrafts": 15}, rows_max=64) == {"auto_fdrafts": 15}
    with pytest.raises(ValueError, match="auto_fdrafts"):
        knobs.parse({"auto_fdrafts": 16}, rows_max=64)
    assert depth.parse_policy("of15") == [depth.OPT_KIND, 15, 1, 0] and depth.parse_policy("om") == [6, 15, 0, 0]
    with pytest.raises(ValueError):
        depth.parse_policy("of16")
    assert engine.encode_policy("c15:0.3")[1] == 15 and engine.encode_policy("fc15:0.3")[0] == 13
    assert lookup.parse_policy("l15:3") == [lookup.LOOKUP_KIND, 15, 3, 0]
    for bad in ("c16:0.3", "16", "l16"):
        with pytest.raises(ValueError):
            engine.encode_policy(bad)
    monkeypatch.setenv("GLM53_TF_AUTO_FDRAFTS", "15")
    assert engine.auto_f_most() == 15
    assert depth.Calibration().kept["m"] == [0.0] * 15
    assert depth.as_cost([13, 15, 300000, 0]) == [depth.OPT_KIND, 15, 1, 0]


@needs_torch
@at16
def test_sparse_rows_follow_the_knob():
    pytest.importorskip("triton")
    from tensorfold.families.glm5_next.cuda import latent

    assert latent.SPARSE_ROWS == 16


@at16
def test_cost_depths_go_past_7_only_when_confident():
    """DFlash2 picks, lookup bands and MTP chains: deep windows for confident drafts, short ones otherwise."""

    o = depth.DepthOptimizer(COSTS, most_m=15, most_f=15)
    assert o.most_f == 15 and o.most_m == 15
    for _ in range(12):                          # a stream running at ~0.1 tokens / ms
        o.rounds.append((6, 60.0))
    assert o.f_depth([0.99] * 15) == 15          # 0.99 x the prior correction: every row pays
    assert o.f_depth([0.99] * 6 + [0.2] * 9) <= 7
    k_hi, s_hi = o.lookup_depth(0.97, 15, 0.1)
    k_lo, _ = o.lookup_depth(0.45, 15, 0.1)
    assert k_hi > 7 and s_hi > 0 and k_lo <= 3
    o.mtp_begin()
    j = 0
    while True:
        take, more = o.mtp_next(j, 0.999, 15)
        assert take
        j += 1
        if not more:
            break
    assert j > 7                                  # a sure chain runs past the old cap
    o.mtp_begin()
    assert o.mtp_next(0, 0.3, 15) == (True, False)
    # the calibration keeps every position's evidence apart
    o.cal.record("f", [0.9] * 15, 16)
    assert o.cal.kept["f"][14] > 0 and o.cal.tried["f"][14] == 1.0


@at16
def test_lookup_copies_15_tokens():
    rng = random.Random(1)
    block = [rng.randrange(5000) for _ in range(40)]
    prompt = block * 2 + block[:6]
    lk = lookup.Lookup(prompt, COSTS, most=15, min_match=3, gated=False)
    assert lk.most == 15
    got = lk.plan([block[6]], 64)
    assert got == block[7:22]
    lk8 = lookup.Lookup(prompt, _costs(8), most=15, min_match=3, gated=False)
    assert lk8.most == 7                           # a request's table bounds it too


# -- torch on the CPU: the real Stepper on patches/0180's hostile fake model -------------------------------------------
def _harness():
    import test_batch_sessions_patches as h

    return h


def _truth(h, st, pending: int, k: int) -> list[int]:
    """The next k tokens serial decoding commits after ``pending`` from the slot's committed state (a copy)."""

    import copy

    st = copy.deepcopy(st)
    out, t = [], pending
    for _ in range(k):
        h.MODEL.stage(None, st, None, [t])
        hid = h.MODEL.compute(None, st, None, 1)
        h.MODEL.commit(None, st, None, 1, 1)
        t = int(hid[0, 0]) % 5
        out.append(t)
    return out


def _good(h, pos: int, j: int, salt: int, rate: int = 90) -> bool:
    return h.MODEL.mix(pos, j, salt) % 100 < rate


class _FakeDFlash:
    """A DFlash2 stand-in with a 16-row block (``GLM53_TF_DFLASH_BLOCK=16``): up to 15 picks, the true continuation
    with a wrong token where ``_good`` says so, and a probability that tells them apart (a function of the committed
    state only: both ranks alike)."""

    block = 16

    def __init__(self, h, state) -> None:
        self.h, self.state = h, state
        self.context_end = 0
        self.pos_dev = torch.zeros(1, dtype=torch.int64)

    def reset(self) -> None:
        self.context_end = 0

    def add_taps(self, taps) -> None:
        self.context_end += int(taps.shape[0])

    def propose(self, last, k, sampling, confidence, probs=None):
        k = min(k, self.block - 1)
        st = self.state()
        true = _truth(self.h, st, int(last), k)
        out = []
        for j, t in enumerate(true):
            ok = _good(self.h, st.pos, j, 11)
            out.append(t if ok else (t + 1) % 5)
            if probs is not None:
                probs.append(0.97 if ok else 0.2)
        return out


def _fake_mtp_draft(h):
    def draft(e, hidden, next_tokens, position, count, sampling, confidence=0.0, opt=None):
        st = e.st
        true = _truth(h, st, int(next_tokens[-1]), count)
        drafts, chained = [], 0
        if opt is not None:
            opt.mtp_begin()
        for j in range(count):
            ok = _good(h, st.pos, j, 29, 85)
            d = true[j] if ok else (true[j] + 2) % 5
            if opt is not None:
                take, more = opt.mtp_next(j, 0.98 if ok else 0.3, count)
                if not take:
                    break
                drafts.append(d)
                if not more:
                    break
            else:
                drafts.append(d)
            if j + 1 < count:
                chained += 1
        st.mtp_drafted = chained
        return drafts

    return draft


def _batcher(monkeypatch, *, rank: int = 0):
    from tensorfold.families.glm5_next.cuda import batch, decode

    h = _harness()
    real = batch.Stepper
    bat = h._fake_batcher(monkeypatch, n=3, rows=64, fast=False, piece=256, budget_pages=4000.0, rank=rank)
    monkeypatch.setattr(batch, "Stepper", real)             # the real decode loop, cut at the forward
    monkeypatch.setattr(batch, "commit", h.MODEL.commit)
    monkeypatch.setattr(decode, "draft", _fake_mtp_draft(h))
    seen: list[int] = []                                    # every verified window's rows
    window = real.window

    def spy(self):
        win = window(self)
        seen.append(len(win))
        return win

    monkeypatch.setattr(real, "window", spy)
    bat.store = None
    bat.g.store = None
    bat._remember = lambda slot, snap: None
    bat.g.drafter = object()
    bat.g.f_most = 15
    bat.max_rows = deep.ROWS
    bat.drafters = [_FakeDFlash(h, (lambda s=s: bat.states[s])) for s in range(bat.n)]
    bat.m_rows = [torch.zeros((batch.BACKLOG, 1), dtype=torch.int64) for _ in range(bat.n)]
    bat.f_taps = [torch.zeros((batch.BACKLOG, 1), dtype=torch.int64) for _ in range(bat.n)]
    e = bat.g.e
    e.tap_rows = lambda R: torch.zeros((R, 1), dtype=torch.int64)
    # the fake forward runs each slot alone: its rows start at 0 of its own ``hid``, not at the round's offset
    e.main_hidden = lambda rows: e.st.hid[0:rows.stop - rows.start]
    e.buf = SimpleNamespace(taps=[torch.zeros((bat.n * 16 + 16, 1), dtype=torch.int64)])
    bat.costs = COSTS
    bat.round_costs = batchplan.RoundCosts(COSTS, 6.5)
    bat.seen = seen
    return h, bat


CODES = {
    "auto+lookup": [4, 2, 8, 30000, 1, 4],       # auto with cost depths (job.cost), gated lookup (min match 4)
    "l15:3": [lookup.LOOKUP_KIND, 15, 3, 0],     # forced lookup: every match of 3+ verifies up to 15 tokens
    "of15": [depth.OPT_KIND, 15, 1, 0],
    "om15": [depth.OPT_KIND, 15, 0, 0],
}


def _job(h, prompt, tokens, code):
    import queue

    from tensorfold.families.glm5_next.cuda.batch import Job

    values = dict(h._values(False, 64), auto_fdrafts=15)
    return Job(list(prompt), tokens, None, False, True, list(code), "deep", 1, values, out=queue.SimpleQueue())


def _serve(h, bat, prompts, arrive, tokens, codes):
    jobs = [None] * len(prompts)
    rnd = 0
    while True:
        for i, (p, at) in enumerate(zip(prompts, arrive)):
            if jobs[i] is None and at <= rnd:
                jobs[i] = _job(h, p, tokens, codes[i % len(codes)])
                bat.queue.append(jobs[i])
        if bat.queue or any(s is not None for s in bat.seqs):
            cancels, admits, pieces = bat._plan()
            bat._execute(cancels, admits, pieces)
        elif all(j is not None for j in jobs):
            break
        rnd += 1
        assert rnd < 5000
    out = []
    for j in jobs:
        reply, done = h._drain(j)
        assert done
        out.append((reply, j.stats))
    return out


def _prompts(seed: int, n: int = 5):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        p = [rng.randrange(1000) for _ in range(rng.randint(40, 300))]
        if i % 2:                                 # repeats: long lookup matches
            p = p[:60] * 3
        out.append(p)
    return out


@needs_torch
@at16
@pytest.mark.parametrize("which", list(CODES) + ["mixed"])
def test_real_stepper_deep_windows_equal_serial(monkeypatch, which):
    h, bat = _batcher(monkeypatch)
    codes = list(CODES.values()) if which == "mixed" else [CODES[which]]
    tokens = 64
    for seed in (1, 2):
        prompts = _prompts(seed)
        got = _serve(h, bat, prompts, [0, 0, 3, 12, 30], tokens, codes)
        for p, (reply, stats) in zip(prompts, got):
            want, _ = h._reference(p, tokens, 64, False)
            assert reply == want, stats
            assert stats["rounds"] == len(stats["keeps"]) and sum(stats["keeps"]) + 1 >= tokens
            assert max(stats["keeps"]) <= 16
            # patches/0380's stats: the drafts each round verified (per-position acceptance on the GPU)
            assert len(stats["depths"]) == len(stats["keeps"]) and max(stats["depths"]) <= 15
            assert all(k <= d + 1 for k, d in zip(stats["keeps"], stats["depths"]))
    log = list(bat.log)
    keeps = [k for d in log for k in d["keeps"]]
    arms = "".join(d["arms"] for d in log)
    assert 9 <= max(bat.seen) <= 16, (which, max(bat.seen))  # windows past the old cap were verified
    if which == "l15:3":            # the fake's copies are mostly wrong (5 symbols): deep windows cut early
        assert "l" in arms
    else:                           # confident drafters: deep windows kept whole or in part
        assert max(keeps) >= 9, (which, max(keeps))
    if which in ("of15", "om15"):
        assert set(arms) == {which[1]}


@needs_torch
@at16
def test_follower_decides_the_same_deep_windows(monkeypatch):
    """Rank 1 runs rank 0's plans and makes every window decision itself: the same (no clock, no exchange)."""

    h, r0 = _batcher(monkeypatch)
    _, r1 = _batcher(monkeypatch, rank=1)
    sent: list[list[int]] = []

    def record(values):
        sent.append([int(v) for v in values])
        return list(values)

    class Done(Exception):
        pass

    def replay(values):
        if not sent:
            raise Done
        return sent.pop(0)

    r0.g._share, r1.g._share = record, replay
    prompts = _prompts(9, 4)
    _serve(h, r0, prompts, [0, 0, 2, 9], 48, list(CODES.values()))
    with pytest.raises(Done):
        r1.follow()
    key = lambda d: (d["slot"], d["sha256"], tuple(d["keeps"]), d["arms"])      # noqa: E731
    assert [key(d) for d in r1.log] == [key(d) for d in r0.log]
    assert max(k for d in r0.log for k in d["keeps"]) >= 9


@needs_torch
@at16
@pytest.mark.parametrize("which", ["auto+lookup", "l15:3", "om15"])
def test_lone_auto_decode_deep_windows_equal_serial(monkeypatch, which):
    """``decode.auto_decode`` (GLM53_TF_BATCH=1) on the fake model with 16-row windows == serial decoding."""

    from tensorfold.families.glm5_next.cuda import decode
    from tensorfold.families.glm5_next.cuda.decode import DepthPolicy, DrafterChoice, auto_decode, prefill

    h = _harness()
    for name in ("stage", "compute", "commit"):
        monkeypatch.setattr(decode, name, getattr(h.MODEL, name))
    monkeypatch.setattr(decode, "_sync", lambda w: None)
    monkeypatch.setattr(decode, "draft", _fake_mtp_draft(h))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    code = CODES[which]
    tokens = 80
    for seed in (3, 4, 5):
        prompt = _prompts(seed, 2)[seed % 2]
        want, _ = h._reference(prompt, tokens, 64, False)
        st = h._St()
        e = h._E(st, 64, False)
        e.w = SimpleNamespace(mtp=object(), cfg=SimpleNamespace(hidden=1, eos=()), device="cpu")
        e.buf = SimpleNamespace(taps=[torch.zeros((64, 1), dtype=torch.int64)])
        e.tap_rows = lambda R: torch.zeros((R, 1), dtype=torch.int64)

        def forward(win, e=e):
            h.MODEL.stage(None, e.st, None, win)
            return h.MODEL.compute(None, e.st, None, len(win))

        e.forward = forward
        dr = _FakeDFlash(h, lambda e=e: e.st) if which != "om15" else None
        first = prefill(e, list(prompt), None, mtp=True, drafter=dr)
        e.lookup = lookup.lookup_for(code, prompt, COSTS, stop_eos=False)
        e.calib = None
        e.depth = depth.DepthOptimizer(COSTS, most_m=15, most_f=15)
        if e.lookup is not None:
            e.lookup.opt = e.depth
        choice = DrafterChoice(COSTS, first="f") if dr is not None and code[0] == 4 else None
        pol = DepthPolicy(3, fixed=True, confidence=0.35)
        res = auto_decode(e, dr, first, tokens, None, choice=choice, m_policy=pol,
                          f_policy=DepthPolicy(15, fixed=True, confidence=0.3))
        assert res.tokens == want, (which, seed, res.arms)
        assert max(res.keeps) <= 16
        if which == "l15:3":
            assert "l" in res.arms


# -- GPU: TensorFold's synthetic checkpoints (one GPU playing rank 0 of two) ---------------------------------------------
@pytest.fixture(scope="module")
def dckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_deep")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


def _gpu_engine(path, *, batch: int = 1, block: int = 16, **kw):
    from test_batch2_patches import _engine as engine

    with pytest.MonkeyPatch.context() as m:
        m.setenv("GLM53_TF_NONEXPERT", "q4mse")
        m.setenv("GLM53_TF_DFLASH_BLOCK", str(block))
        m.setenv("GLM53_TF_AUTO_FDRAFTS", str(block - 1))
        m.setenv("GLM53_TF_DEPTH", "cost")
        m.setenv("GLM53_TF_BATCH_MTP", "1")
        m.setenv("GLM53_TF_BATCH_ROW_MS", "6.5")
        return engine(path, batch=batch, **kw)


@gpu
@at16
def test_gpu_dense_windows_equal_serial_rows(dckpt):
    """Dense windows of 1-16 rows: every row's logits == the logits of serial steps over the same tokens."""

    import numpy as np

    from tensorfold.families.glm5_next.cuda.decode import prefill

    eng = _gpu_engine(dckpt)
    e = eng.e
    assert e.graphs is not None and max(e.long_rows or (16,)) == 16 and eng.drafter.block == 16
    prompt = [int(t) for t in np.random.default_rng(3).integers(0, 1000, size=300)]
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, size=16)]
    with torch.no_grad():
        prefill(e, prompt, None, mtp=True)
        eng.cache = []
        serial = []
        base = e.st.pos
        from tensorfold.families.glm5_next.cuda.forward import commit

        for r in range(16):
            serial.append(e.forward(toks[r:r + 1])[:1].clone())
            commit(e.w, e.st, e.buf, 1, 1)
        prefill(e, prompt, None, mtp=True)
        assert e.st.pos == base
        for R in range(1, 17):
            got = e.forward(toks[:R]).clone()
            for r in range(R):
                assert torch.equal(got[r:r + 1], serial[r]), (R, r)


@gpu
@at16
def test_gpu_long_windows_equal_eager_upstream():
    """Past 2,051 tokens: windows of 1-16 rows (both parities) and MTP steps of 1-16 rows through the long-context
    graphs == upstream's eager step (the 0050 test at 16 rows)."""

    import tempfile

    import numpy as np
    import test_longctx_patches as lc

    from tensorfold.families.glm5_next.cuda import sparse, weights
    from tensorfold.families.glm5_next.cuda.decode import prefill
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine
    from tensorfold.families.glm5_next.cuda.forward import commit
    from test_glm_engine import _TwoCopies, _checkpoint, _drafter

    path = Path(tempfile.mkdtemp(prefix="glm_deep_long"))
    _checkpoint(path / "model", exl3=True)
    lc._index_heads_32(path / "model")
    _drafter(path / "dflash2")
    with pytest.MonkeyPatch.context() as m:
        m.setattr(weights, "NONEXPERT", "q4mse")
        m.setenv("GLM53_TF_PREFILL_ROWS", "64")
        m.setenv("GLM53_TF_LONGCTX_GRAPHS", "1")
        eng = GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", context=4096,
                        comm=_TwoCopies())
    e = eng.e
    assert e.buf.lc.rows == 16 and e.long_rows == tuple(range(1, 17))
    rng = np.random.default_rng(2500)
    prefill(e, [int(x) for x in rng.integers(0, 1000, size=2500)], None, mtp=True)
    eng.cache = []
    toks = [int(x) for x in rng.integers(0, 1000, size=16)]
    hid = (torch.randn((16, e.w.cfg.hidden), generator=torch.Generator().manual_seed(1)) * 0.5).to(torch.bfloat16)
    hid = hid.cuda()
    cap = e.st.index[0][2].shape[0] - 2
    for _ in range(2):
        st = e.st
        for R in range(1, 17):
            with lc._upstream(e):
                ref = e.forward(toks[:R]).clone()
            for _ in range(2):
                assert torch.equal(e.forward(toks[:R]), ref), (R, st.parity)
            assert ("main", R, st.parity, sparse.pool_bucket(st.pos + R, cap)) in e.graphs.long
        for k in range(1, 17):
            with lc._upstream(e):
                ref = e.mtp(toks[:k], hid[:k]).clone()
            for _ in range(2):
                assert torch.equal(e.mtp(toks[:k], hid[:k]), ref), k
        e.forward(toks[:16])
        commit(e.w, st, e.buf, 16, 1)


@gpu
@at16
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
def test_gpu_drafted_replies_equal_serial(dckpt, greedy):
    """Lone and 4 batched requests (o, l15:3, of15, om15; a 16-row DFlash2 block) on repeated prompts == serial."""

    from test_batch2_patches import _free, _repeated, _run, _sampling, _serial

    sampling = _sampling(greedy)
    lone = _gpu_engine(dckpt)
    prompts = [_repeated(300 + i, block=40, times=3, tail=8 + i) for i in range(4)]
    want = [_serial(lone, p, sampling, 64) for p in prompts]
    deepest = 0
    for p, w_ in zip(prompts, want):
        for pol in ("o", "l15:3", "of15", "om15"):
            got, stats = _run(lone, p, sampling, policy=pol, tokens=64)
            assert got == w_, pol
            deepest = max(deepest, max(stats.get("keeps") or [1]))
    b4 = _gpu_engine(dckpt, batch=4)
    try:
        got = b4.batch.generate_batch([dict(prompt=p, max_tokens=64, sampling=sampling, policy=pol)
                                       for p, pol in zip(prompts, ("o", "l15:3", "of15", "om15"))])
        assert [t for t, _ in got] == want
        deepest = max([deepest] + [max(s.get("keeps") or [1]) for _, s in got])
    finally:
        _free(b4)
    assert deepest >= 9
