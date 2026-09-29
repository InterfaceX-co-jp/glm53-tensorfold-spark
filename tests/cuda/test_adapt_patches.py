"""patches/0340 (``GLM53_TF_BATCH_ADAPT=1``, off by default; ``glm5_next/cuda/adapt.py`` + ``batch.py``): in shared
batched rounds each slot of a request with cost-derived depths picks its round's drafter (MTP, DFlash2, or with
``GLM53_TF_BATCH_ADAPT_SERIAL=1`` none: a one-row serial round) by the drafts' surplus at the shared rate and the
round's marginal row costs; alone, its own ``DrafterChoice`` decides as before. Drafts only: replies unchanged.

Checked:

- host only: ``extra_ms`` / ``drafter_ms`` against ``lookup.round_ms`` and ``RoundCosts.table``; ``SlotChoice``: every
  arm once, then the best surplus, serial only when allowed and no arm pays, probes after ``EVERY`` rounds, alone =
  the base choice's pick (untouched), only shared rounds build the surplus while the base still records every drafted
  round, two instances fed the same rounds decide alike (no clock); the env knobs;
- ``bench/draftsim.py``'s copy of ``decode.DrafterChoice`` picks exactly as the real one (torch needed to import it);
- torch on the CPU, patches/0180's hostile fake model and the REAL ``batch.Stepper`` / ``Batcher._plan`` /
  ``_execute`` / ``_verify`` (fake drafters that draft the true continuation or a wrong token, deterministically):
  requests of ``auto`` with cost depths arriving at different times (alone and shared rounds), with 0340 off, on,
  on with serial rounds, and with every arm (m / f / serial) forced at random per slot and round: every reply equals
  a fresh prefill + serial decoding; with 0340 on some rounds are serial and some slots change drafter mid-request;
  a follower batcher (rank 1) that replays rank 0's plans decides the same arms and keeps.

Run: PYTHONPATH=<tree>/src:<tree>/tests/cuda:tests/cuda pytest -q tests/cuda/test_adapt_patches.py (inside the image:
PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda).
"""

from __future__ import annotations

import copy
import importlib.util
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold.families.glm5_next.cuda import adapt, batchplan, depth, lookup

try:
    import torch
except ImportError:          # the host-only tests still run
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="torch")

COSTS = {"verify": [31.0, 39.0, 45.0, 51.0, 56.0, 62.0, 68.0, 74.0], "mtp": 2.04, "mtp_step": 1.68, "mtp_row": 0.1,
         "block": 3.88, "taps_row": 0.05}


# -- host only ------------------------------------------------------------------------------------------------------
def test_extra_ms_is_the_round_past_a_serial_row():
    rc = batchplan.RoundCosts(COSTS, 6.5)
    for others in (0, 3, 9):
        table = rc.table(others)
        for arm in "mf":
            for rows, steps, backlog in ((1, 1, 1), (3, 2, 4), (8, 7, 1), (5, 1, 0)):
                want = (table[rows - 1] - table[0]) + (lookup.round_ms(COSTS, arm, rows, steps, backlog)
                                                       - lookup.round_ms(COSTS, "l", rows, 0, 0))
                assert adapt.extra_ms(COSTS, table, arm, rows, steps, backlog) == pytest.approx(want)
    assert adapt.drafter_ms(COSTS, "s", 1, 0, 0) == 0.0 and adapt.drafter_ms(COSTS, "l", 4, 0, 0) == 0.0
    # past the table a row costs the ROW_MS floor
    t = rc.table(9)
    assert adapt.extra_ms(COSTS, t, "f", 3, 0, 0) == pytest.approx(2 * 6.5 + COSTS["block"])


class _Base:
    def __init__(self) -> None:
        self.picks = 0
        self.rec = []

    def pick(self) -> str:
        self.picks += 1
        return "f"

    def record(self, *a) -> None:
        self.rec.append(a)


def test_slot_choice_rules():
    table = batchplan.RoundCosts(COSTS, 6.5).table(6)
    base = _Base()
    sc = adapt.SlotChoice(COSTS, "mf", base, every=4, window=4, serial=True)
    # alone: the base decides (and nothing else changes)
    assert sc.pick(None) == "f" and base.picks == 1
    sc.record("f", 4, 0, 2, 3, table, shared=False)
    assert base.rec and not sc.hist["f"] and sc.since == {"m": 0, "f": 0}
    # shared: each arm once first
    rate = 0.07
    assert sc.pick(rate) == "m"
    sc.record("m", 3, 2, 1, 3, table, shared=True)
    assert sc.pick(rate) == "f"
    sc.record("f", 8, 0, 3, 8, table, shared=True)
    assert len(base.rec) == 3                             # the base records every drafted round
    # f: 7 tokens for ~7 rows + a block; m: 2 tokens for 2 rows + 2 steps -> f
    vf, vm = sc.surplus("f", rate), sc.surplus("m", rate)
    assert vf > vm > 0
    assert sc.pick(rate) == "f"
    # m goes stale: probed after ``every`` shared rounds without it
    for _ in range(3):
        sc.record("f", 8, 0, 1, 8, table, shared=True)
    assert sc.since["m"] == 4 and sc.pick(rate) == "m" and sc.probes == 1
    sc.record("m", 3, 2, 1, 3, table, shared=True)
    assert sc.since == {"m": 0, "f": 1}
    # both arms worthless at a high rate: serial, never when not allowed
    assert sc.pick(10.0) == adapt.SERIAL
    sc.record(adapt.SERIAL, 1, 0, 0, 1, table, shared=True)
    assert sc.serial == 1 and len(base.rec) == 7          # serial rounds never reach the base
    no = adapt.SlotChoice(COSTS, "mf", None, every=0, serial=False)
    no.hist = copy.deepcopy(sc.hist)
    assert no.pick(10.0) in "mf"
    # a margin keeps drafting until serial is clearly better
    m = adapt.SlotChoice(COSTS, "f", None, every=0, serial=True, margin=5.0)
    m.record("f", 3, 0, 1, 1, table, shared=True)          # nothing kept: negative surplus
    assert m.surplus("f", 0.07) < 0 and m.pick(0.07) == "f"
    # one arm only (of / om requests): never the other
    one = adapt.SlotChoice(COSTS, "m", None)
    assert one.pick(0.07) == "m" and one.pick(None) == "m"


def test_slot_choice_is_a_function_of_its_rounds():
    rng = random.Random(3)
    rounds = []
    for _ in range(300):
        arm = rng.choice("mfs")
        rows = 1 if arm == "s" else rng.randint(2, 8)
        rounds.append((arm, rows, rng.randint(1, 4), rng.randint(0, 9), rng.randint(1, rows), rng.randint(0, 20),
                       rng.random() < 0.8, rng.uniform(0.03, 0.12)))
    picks = []
    for _ in range(2):
        rc = batchplan.RoundCosts(COSTS, 6.5)
        sc = adapt.SlotChoice(COSTS, "mf", None, serial=True)
        got = []
        for arm, rows, steps, backlog, keep, others, shared, rate in rounds:
            got.append(sc.pick(rate if shared else None))
            sc.record(arm, rows, steps, backlog, keep, rc.table(others), shared)
        picks.append(got)
    assert picks[0] == picks[1]
    assert set(picks[0]) >= {"m", "f"}


def test_env(monkeypatch):
    for k in ("GLM53_TF_BATCH_ADAPT", "GLM53_TF_BATCH_ADAPT_SERIAL", "GLM53_TF_BATCH_ADAPT_EVERY",
              "GLM53_TF_BATCH_ADAPT_WINDOW"):
        monkeypatch.delenv(k, raising=False)
    assert not adapt.env_on() and not adapt.env_serial()
    assert adapt.env_every() == adapt.EVERY and adapt.env_window() == adapt.WINDOW
    monkeypatch.setenv("GLM53_TF_BATCH_ADAPT", "1")
    monkeypatch.setenv("GLM53_TF_BATCH_ADAPT_SERIAL", "1")
    monkeypatch.setenv("GLM53_TF_BATCH_ADAPT_EVERY", "16")
    monkeypatch.setenv("GLM53_TF_BATCH_ADAPT_WINDOW", "4")
    assert adapt.env_on() and adapt.env_serial() and adapt.env_every() == 16 and adapt.env_window() == 4
    monkeypatch.setenv("GLM53_TF_BATCH_ADAPT", "yes")
    with pytest.raises(ValueError, match="0 or 1"):
        adapt.env_on()


def _draftsim():
    path = Path(__file__).resolve().parents[2] / "bench" / "draftsim.py"
    spec = importlib.util.spec_from_file_location("draftsim", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@needs_torch
def test_draftsim_drafter_choice_is_decodes():
    from tensorfold.families.glm5_next.cuda import decode

    sim = _draftsim()
    rng = random.Random(7)
    for first, every in (("f", 8), ("m", 3), ("f", 0)):
        a = decode.DrafterChoice(COSTS, first=first, every=every)
        b = sim.DrafterChoice(COSTS, first=first, every=every)
        for _ in range(400):
            x, y = a.pick(), b.pick()
            assert x == y
            rows = rng.randint(1, 8)
            args = (x, rows, rng.randint(1, rows), rng.randint(0, 9), rng.randint(1, rows))
            a.record(*args)
            b.record(*args)


def test_draftsim_runs_the_engine_code():
    """The simulator's baseline drives ``depth`` / ``batchplan`` / ``adapt`` from this tree, deterministically."""

    sim = _draftsim()
    mods = (depth, batchplan, adapt)
    surv = {"m": [0.8, 0.4, 0.2, 0.1, 0.05, 0.0, 0.0], "f": [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65]}
    streams = [dict(surv=dict(surv), raw=dict(surv), beta={"m": 1.0, "f": 1.0}, tpr=2.0, rkeep={}) for _ in range(3)]
    kw = dict(tokens=128, signal="info", true_row=6.0, slot_ms=7.0)
    for v in ("base", "adapt", "adapt:serial=0", "oracle"):
        r1 = sim.Sim(mods, streams, sim.parse_variant(v), seed=1, **kw).run()
        r2 = sim.Sim(mods, streams, sim.parse_variant(v), seed=1, **kw).run()
        assert repr(r1) == repr(r2) and r1["agg"] > 0


# -- torch on the CPU: the real Stepper on patches/0180's hostile fake model ------------------------------------------
def _harness():
    import test_batch_sessions_patches as h

    return h


def _truth(h, st, pending: int, k: int) -> list[int]:
    """The next k tokens serial decoding commits after ``pending`` from the slot's committed state (a copy)."""

    st = copy.deepcopy(st)
    out, t = [], pending
    for _ in range(k):
        h.MODEL.stage(None, st, None, [t])
        hid = h.MODEL.compute(None, st, None, 1)
        h.MODEL.commit(None, st, None, 1, 1)
        t = int(hid[0, 0]) % 5
        out.append(t)
    return out


def _good(h, pos: int, j: int, salt: int) -> bool:
    return h.MODEL.mix(pos, j, salt) % 100 < 70


class _FakeDFlash:
    """A DFlash2 stand-in for one slot: proposes the true continuation with a wrong token where ``_good`` says so,
    with a probability that tells them apart (a function of the committed state only: both ranks alike)."""

    def __init__(self, h, bat, slot: int) -> None:
        self.h, self.bat, self.slot = h, bat, slot
        self.context_end = 0
        self.pos_dev = torch.zeros(1, dtype=torch.int64)

    def reset(self) -> None:
        self.context_end = 0

    def add_taps(self, taps) -> None:
        self.context_end += int(taps.shape[0])

    def propose(self, last, k, sampling, confidence, probs=None):
        st = self.bat.states[self.slot]
        true = _truth(self.h, st, int(last), k)
        out = []
        for j, t in enumerate(true):
            ok = _good(self.h, st.pos, j, 11)
            out.append(t if ok else (t + 1) % 5)
            if probs is not None:
                probs.append(0.9 if ok else 0.25)
        return out


def _fake_mtp_draft(h):
    def draft(e, hidden, next_tokens, position, count, sampling, confidence=0.0, opt=None):
        st = e.st
        true = _truth(h, st, int(next_tokens[-1]), count)
        drafts, chained = [], 0
        if opt is not None:
            opt.mtp_begin()
        for j in range(count):
            ok = _good(h, st.pos, j, 29)
            d = true[j] if ok else (true[j] + 2) % 5
            if opt is not None:
                take, more = opt.mtp_next(j, 0.85 if ok else 0.3, count)
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


def _batcher(monkeypatch, *, adapt_on: bool, serial: bool, rank: int = 0, forced: bool = False):
    from tensorfold.families.glm5_next.cuda import batch, decode

    h = _harness()
    real = batch.Stepper
    rows = 64
    bat = h._fake_batcher(monkeypatch, n=3, rows=rows, fast=False, piece=256, budget_pages=4000.0, rank=rank)
    monkeypatch.setattr(batch, "Stepper", real)             # the real decode loop, cut at the forward
    monkeypatch.setattr(batch, "commit", h.MODEL.commit)
    monkeypatch.setattr(decode, "draft", _fake_mtp_draft(h))
    bat.store = None
    bat.g.store = None
    bat._remember = lambda slot, snap: None
    bat.g.drafter = object()
    bat.drafters = [_FakeDFlash(h, bat, s) for s in range(bat.n)]
    bat.m_rows = [torch.zeros((batch.BACKLOG, 1), dtype=torch.int64) for _ in range(bat.n)]
    bat.f_taps = [torch.zeros((batch.BACKLOG, 1), dtype=torch.int64) for _ in range(bat.n)]
    e = bat.g.e
    e.tap_rows = lambda R: torch.zeros((R, 1), dtype=torch.int64)
    # the fake forward runs each slot alone: its rows start at 0 of its own ``hid``, not at the round's offset
    e.main_hidden = lambda rows: e.st.hid[0:rows.stop - rows.start]
    e.buf = SimpleNamespace(taps=[torch.zeros((bat.n * 8 + 8, 1), dtype=torch.int64)])
    bat.costs = COSTS
    bat.round_costs = batchplan.RoundCosts(COSTS, 6.5)
    bat.adapt, bat.adapt_serial = adapt_on, serial
    if forced:                                              # any arm, any round: the invariant cannot depend on it
        def pick(self, rate):
            if rate is None:
                return self.base.pick() if self.base is not None else "m"
            st = bat.states[[i for i, q in enumerate(bat.seqs) if q is not None and q.stepper is not None
                             and q.stepper.adapt is self][0]]
            return "mfs"[h.MODEL.mix(st.pos, 5) % 3]

        monkeypatch.setattr(adapt.SlotChoice, "pick", pick)
    return h, bat


def _auto_job(h, prompt, tokens):
    import queue

    from tensorfold.families.glm5_next.cuda.batch import Job

    return Job(list(prompt), tokens, None, False, True, [4, 2, 8, 30000], "auto", 1, h._values(False, 64),
               out=queue.SimpleQueue())


def _serve(h, bat, prompts, arrive, tokens):
    jobs = [None] * len(prompts)
    rnd = 0
    while True:
        for i, (p, at) in enumerate(zip(prompts, arrive)):
            if jobs[i] is None and at <= rnd:
                jobs[i] = _auto_job(h, p, tokens)
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
    return [[rng.randrange(1000) for _ in range(rng.randint(40, 300))] for _ in range(n)]


@needs_torch
@pytest.mark.parametrize("mode", ["off", "on", "serial", "forced"])
def test_real_stepper_replies_equal_serial(monkeypatch, mode):
    h, bat = _batcher(monkeypatch, adapt_on=mode != "off", serial=mode in ("serial", "forced"), forced=mode == "forced")
    tokens = 40
    for seed in (1, 2):
        prompts = _prompts(seed)
        got = _serve(h, bat, prompts, [0, 0, 3, 12, 30], tokens)
        arms = ""
        for p, (reply, stats) in zip(prompts, got):
            want, _ = h._reference(p, tokens, 64, False)
            assert reply == want, stats
            arms += stats["drafters"]
            assert stats["rounds"] == len(stats["keeps"]) and sum(stats["keeps"]) + 1 >= tokens
            if mode != "off":
                assert "adapt" in stats
        assert set(arms) >= set("mf")
        if mode == "forced":
            assert "s" in arms and any(len(set(s["drafters"])) == 3 for _, s in got)
        if mode == "off":
            assert "s" not in arms


@needs_torch
def test_follower_decides_the_same_arms(monkeypatch):
    """Rank 1 runs rank 0's plans and makes every depth / arm decision itself: the same (no clock, no exchange)."""

    h, r0 = _batcher(monkeypatch, adapt_on=True, serial=True)
    _, r1 = _batcher(monkeypatch, adapt_on=True, serial=True, rank=1)
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
    _serve(h, r0, prompts, [0, 0, 2, 9], 40)
    with pytest.raises(Done):
        r1.follow()
    key = lambda d: (d["slot"], d["sha256"], tuple(d["keeps"]), d["arms"])      # noqa: E731
    assert [key(d) for d in r1.log] == [key(d) for d in r0.log]
    assert any("m" in d["arms"] and "f" in d["arms"] for d in r0.log)


# -- GPU: TensorFold's synthetic EXL3 checkpoint with the DFlash2 drafter (one GPU playing rank 0 of two) -------------
CUDA = torch is not None and torch.cuda.is_available()
gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")


def _gpu_engine(path, *, on: bool, serial: bool, **kw):
    from test_batch2_patches import _engine as engine

    with pytest.MonkeyPatch.context() as m:
        m.setenv("GLM53_TF_LOOKUP", "0")
        m.setenv("GLM53_TF_NONEXPERT", "q4mse")
        m.setenv("GLM53_TF_BATCH_MTP", "1")
        m.setenv("GLM53_TF_BATCH_ROW_MS", "6.5")
        m.setenv("GLM53_TF_BATCH_ADAPT", "1" if on else "0")
        m.setenv("GLM53_TF_BATCH_ADAPT_SERIAL", "1" if serial else "0")
        m.setenv("GLM53_TF_BATCH_ADAPT_EVERY", "3")
        return engine(path, **kw)


@pytest.fixture(scope="module")
def gckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_adapt")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


@pytest.fixture(scope="module")
def gref(gckpt):
    return _gpu_engine(gckpt, on=False, serial=False)


@pytest.fixture(scope="module")
def gb4(gckpt):
    return _gpu_engine(gckpt, on=True, serial=True, batch=4)


@gpu
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
@pytest.mark.parametrize("forced", [False, True], ids=["chosen", "forced"])
def test_four_requests_equal_serial_with_adapt(gref, gb4, greedy, forced, monkeypatch):
    """4 requests (auto with cost depths, of, om) in shared rounds with 0340 on, serial rounds allowed; ``forced``:
    every slot's arm (MTP / DFlash2 / serial) drawn at random each round. Every reply == serial decoding."""

    from test_batch2_patches import _prompt, _sampling
    from test_batch_parallel_patches import _serial

    if forced:
        rng = random.Random(5)
        monkeypatch.setattr(adapt.SlotChoice, "pick", lambda self, rate: (
            (self.base.pick() if self.base is not None else self.arms[:1]) if rate is None
            else rng.choice(self.arms + adapt.SERIAL)))
    sampling = _sampling(greedy)
    prompts = [_prompt(170 + i, 30 + 11 * i) for i in range(4)]
    want = [_serial(gref, p, sampling, 48) for p in prompts]
    for quad in (("o", "o", "of", "om"), ("o", "om3", "o", "of5")):
        got = gb4.batch.generate_batch([dict(prompt=p, max_tokens=48, sampling=sampling, policy=pol)
                                        for p, pol in zip(prompts, quad)])
        assert [t for t, _ in got] == want, quad
        assert all("adapt" in s for _, s in got)
        if forced:
            assert any("s" in s["drafters"] for _, s in got)
