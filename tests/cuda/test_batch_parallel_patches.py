"""patches/0200 (batched parallelism on top of patches/0120 / 0180; every item opt-in, off = 0120's behaviour):

- ``GLM53_TF_BATCH_CAPTURE_AFTER=N``: a multi-sequence graph key (slots, rows per slot, context modes) is captured on
  its N-th sighting; before that its rounds run eagerly through the same capturable code (same bits);
- ``GLM53_TF_BATCH_PAD=2,4,8``: each slot's window padded up to the next listed size with its last token (fewer graph
  keys); padded rows are verified but never kept, so every committed bit is the unpadded window's;
- ``GLM53_TF_BATCH_SHORT=N``: prompts with at most N tokens left prefill in the round they are admitted (several a
  round), without waiting for the fair share; long prompts keep ``GLM53_TF_BATCH_PREFILL_SHARE``;
- ``GLM53_TF_BATCH_PARITY_KEY=1``: KDA parities in the graph key instead of 71 MB state copies a slot and round;
- ``GLM53_TF_BATCH_MTP=1``: the MTP chains of all slots drafted together (one head pass a chained step);
- ``GLM53_TF_BATCH_ROW_MS=F``: batch-aware cost depths price a row past the verify table at >= F ms;
- per request ``round_kinds`` in the stats (graph / eager / capture / alone rounds, padded rows; wall ms of its
  verify rounds, its drafting and the other requests' pieces it waited through).

Checked:

- host only: ``pick_pieces``, ``pad_mask`` / ``padded``, ``Sightings``, plans with several pieces round trip, the
  row-cost floor in ``verify_ms`` / ``RoundCosts``;
- torch on any device (CPU is enough), patches/0180's hostile fake model and real ``Batcher._plan`` / ``_execute`` /
  ``_piece`` / ``_verify`` / ``follow``: short prompts prefill together (several pieces a round) and every reply and
  slot state still equals a fresh prefill + serial decode; a follower replays the same rounds; padded windows (a fake
  forward that pads) keep every reply and state exact, i.e. ``_verify`` reads each slot's rows at the padded offsets
  and commits the padded forward; ``MtpChains`` on a fake head gives each slot the drafts, head-cache length and
  optimizer calls of ``decode.draft`` alone (confidence stops, cost-depth stand-ins, several rounds); the one-gather
  draft sampler == ``decode.sample_rows`` row by row (greedy / sampled, with / without the draft's probability);
- GPU (TensorFold's synthetic EXL3 checkpoint with the DFlash2 drafter, one GPU playing rank 0 of two): padded
  batched rows == each sequence's lone rows (graphs: first round and replay); a key captured on its N-th sighting;
  4 requests with the graph / pad / short / parity knobs on == serial for mixed policies, sampled and greedy; short
  prompts admitted while a long one prefills in pieces get pieces in the same round, replies exact; one MTP head
  pass over 2-4 slots' rows == each slot's own head pass (logits, output rows); batched MTP drafting: replies ==
  serial and the same keeps as per-slot drafting.

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q
tests/cuda/test_batch_parallel_patches.py (host part anywhere: PYTHONPATH=<tree>/src:<tree>/tests/cuda).
"""

from __future__ import annotations

from types import SimpleNamespace

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


# -- host only ------------------------------------------------------------------------------------------------------
def test_pick_pieces():
    p = [(0, 5000, 3), (1, 200, 5), (3, 200, 4), (2, 90, 6)]
    for allow in (True, False):                               # short = 0: patches/0120's one piece
        want = [batchplan.pick_piece(p)] if allow else []
        assert batchplan.pick_pieces(p, allow, 0) == want
    assert batchplan.pick_pieces(p, False, 256) == [2, 3, 1]  # every short one, fewest left first, even unfair
    assert batchplan.pick_pieces(p, True, 256) == [2, 3, 1, 0]
    assert batchplan.pick_pieces(p, True, 100) == [2, 3]      # the long rest: pick_piece's choice
    assert batchplan.pick_pieces(p, False, 100) == [2]
    assert batchplan.pick_pieces([], True, 100) == []
    assert batchplan.pick_pieces([(0, 5000, 1)], True, 100) == [0]


def test_plan_with_several_pieces_round_trips():
    admits = [(2, 0, [1, 2, 3]), (3, 64, [4])]
    plan = batchplan.encode_plan([1], admits, [2, 3, 0])
    assert batchplan.parse_plan(plan) == ([1], admits, [2, 3, 0])


def test_pad_mask_and_padded():
    assert batchplan.pad_mask("", 8) == 0 and batchplan.pad_mask("0", 8) == 0
    m = batchplan.pad_mask("2,4,8", 8)
    assert m == 0b10001010
    assert [batchplan.padded(r, m) for r in range(1, 10)] == [2, 2, 4, 4, 8, 8, 8, 8, 9]
    assert [batchplan.padded(r, 0) for r in range(1, 9)] == list(range(1, 9))
    one = batchplan.pad_mask("1, 3", 8)
    assert [batchplan.padded(r, one) for r in range(1, 6)] == [1, 3, 3, 4, 5]
    for bad in ("9", "-1", "x"):
        with pytest.raises(ValueError):
            batchplan.pad_mask(bad, 8)


def test_sightings():
    s = batchplan.Sightings(1)
    assert s.capture("a") and s.capture("a")                 # patches/0120: the first sighting
    s = batchplan.Sightings(3)
    assert [s.capture("k") for _ in range(4)] == [False, False, True, True]
    assert not s.capture("other")
    with pytest.raises(ValueError):
        batchplan.Sightings(0)


def test_row_cost_floor_past_the_table():
    v = [31.0, 39.0, 45.0, 51.0, 56.0, 62.0, 68.0, 74.0]      # two Sparks, 1..8 rows
    top = (74.0 - 51.0) / 4                                  # 0120's extension: the upper half's slope (5.75)
    assert batchplan.verify_ms(v, 12) == pytest.approx(74 + 4 * top)
    assert batchplan.verify_ms(v, 12, 6.5) == pytest.approx(74 + 4 * 6.5)
    assert batchplan.verify_ms(v, 12, 2.0) == pytest.approx(74 + 4 * top)     # a floor, never lower
    assert batchplan.verify_ms(v, 5, 9.0) == 56.0                            # inside the table: as measured
    costs = {"verify": v, "mtp": 1.7, "mtp_step": 1.5, "mtp_row": 0.1, "block": 3.0, "taps_row": 0.05}
    old, new = batchplan.RoundCosts(costs), batchplan.RoundCosts(costs, 6.5)
    assert old.table(9) == [batchplan.verify_ms(v, 10 + k) for k in range(8)]
    t = new.table(9)
    assert all(b - a == pytest.approx(6.5) for a, b in zip(t, t[1:]))
    assert new.round_ms([("s", 4, 0, 0)] * 4) - old.round_ms([("s", 4, 0, 0)] * 4) == pytest.approx(8 * (6.5 - top))


# -- torch on any device: patches/0180's hostile fake model ------------------------------------------------------------
def _harness():
    import test_batch_sessions_patches as h

    return h


@needs_torch
@pytest.mark.parametrize("fast", [False, True], ids=["exact", "fast"])
def test_short_prompts_prefill_together_on_a_fake_model(monkeypatch, fast):
    """GLM53_TF_BATCH_SHORT on the real round loop with a fair share that holds long pieces back: rounds run several
    pieces, and every reply and every slot state at the end equals a fresh prefill + serial decode."""

    import numpy as np

    h = _harness()
    rows = 64
    bat = h._fake_batcher(monkeypatch, n=3, rows=rows, fast=fast, piece=256, budget_pages=4000.0)
    bat.fair = batchplan.Fairness(0.3)
    bat.short = 700
    h._conversations(bat, rng=np.random.default_rng(11 + int(fast)), fast=fast, rows=rows, turns=8)
    many = [pieces for _, pieces, _ in bat.trace if len(pieces) > 1]
    assert many, "no round ran more than one piece"
    assert all(len(set(p)) == len(p) for _, p, _ in bat.trace)


@needs_torch
def test_short_prompts_follower_replays_rank0(monkeypatch):
    import numpy as np

    h = _harness()
    rows = 64
    r0 = h._fake_batcher(monkeypatch, n=3, rows=rows, fast=False, piece=256, budget_pages=20.0)
    r1 = h._fake_batcher(monkeypatch, n=3, rows=rows, fast=False, piece=256, budget_pages=20.0, rank=1)
    r0.short = 700                      # rank 0 alone decides the pieces; rank 1 runs what the plans say
    r0.fair = batchplan.Fairness(0.3)
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
    h._conversations(r0, rng=np.random.default_rng(5), fast=False, rows=rows, turns=6, check=False)
    with pytest.raises(Done):
        r1.follow()
    key = lambda d: (d["slot"], d["sha256"], tuple(d["keeps"]), d["cancelled"])      # noqa: E731
    assert [key(d) for d in r1.log] == [key(d) for d in r0.log]
    assert [t[1:] for t in r1.trace] == [t[1:] for t in r0.trace]
    assert any(len(p) > 1 for _, p, _ in r0.trace)
    assert [st.state() for st in r0.states] == [st.state() for st in r1.states]


@needs_torch
@pytest.mark.parametrize("spec", ["3", "2,4,8"])
def test_padded_windows_on_a_fake_model(monkeypatch, spec):
    """A forward that pads each window (as GLM53_TF_BATCH_PAD does) through the real ``_verify``: sampling reads each
    slot's rows at the padded offsets and the commit gets the padded row count; replies and states stay exact."""

    import numpy as np

    from tensorfold.families.glm5_next.cuda import batch

    h = _harness()
    rows = 64
    mask = batchplan.pad_mask(spec, 8)
    bat = h._fake_batcher(monkeypatch, n=3, rows=rows, fast=False, piece=256, budget_pages=4000.0)
    seen = []

    class _PadStepper(h._Stepper):
        def accept(self, e, sampled, off, shared, rows=None):
            seen.append(rows)
            return super().accept(e, sampled, off, shared)

    monkeypatch.setattr(batch, "Stepper", _PadStepper)

    def forward(active, windows):
        out, ran = [], []
        for s, win in zip(active, windows):
            Rp = batchplan.padded(len(win), mask)
            win = list(win) + [win[-1]] * (Rp - len(win))
            with bat._on(s) as ee:
                h.MODEL.stage(None, ee.st, None, win)
                out.append(h.MODEL.compute(None, ee.st, None, Rp))
            ran.append(Rp)
        bat.last_rows = ran
        return torch.cat(out)

    bat._forward = forward
    h._conversations(bat, rng=np.random.default_rng(21), fast=False, rows=rows, turns=6)
    assert seen and all(r == batchplan.padded(1, mask) for r in seen)


class _Head:
    """A fake MTP head on the CPU: each slot's head cache holds a hash chain of (hidden, token) rows; a pass returns,
    per sequence, the hash after its last row (the "logits": token = h % 50, probability = (h // 50 % 100) / 100) and
    that row's output (the chained drafts' hidden)."""

    M = (1 << 61) - 1

    def mix(self, *v) -> int:
        h = 1469598103934665603
        for x in v:
            h = (h * 1099511628211 + int(x) + 12345) % self.M
        return h

    def rows(self, st, tokens, hidden) -> tuple[int, int]:
        for i, t in enumerate(tokens):
            p = st.mtp_len + i
            prev = st.mc[p - 1] if p else (7 if st.mtp_len == 0 else 0)
            st.mc[p] = self.mix(prev, t, int(hidden[i, 0]), p)
        h = st.mc[st.mtp_len + len(tokens) - 1]
        return h, self.mix(h, 3)


class _HeadSt:
    def __init__(self, pos: int) -> None:
        self.pos, self.mtp_len, self.mtp_drafted = pos, 0, 0
        self.mc = {}

    def set_mtp_len(self, n: int) -> None:
        self.mtp_len = n


class _HeadEngine:
    def __init__(self, head) -> None:
        self.head = head
        self.st = None
        self.w = None
        self.mbuf = SimpleNamespace(rows=64, fnormed=torch.zeros((64, 1), dtype=torch.int64))
        self.draft_n = 1

    def mtp(self, tokens, hidden):
        h, out = self.head.rows(self.st, tokens, hidden)
        self.mbuf.fnormed[0, 0] = out
        return torch.tensor([[h]], dtype=torch.int64)

    def draft_hidden(self, row):
        return self.mbuf.fnormed[0:1].clone()

    def sample(self, logits, positions, sampling, *, draft=False, probs=None):
        h = int(logits[0, 0])
        if probs is not None:
            probs.append((h // 50 % 100) / 100)
        return [h % 50 + positions[0] % 3]


class _Opt:
    """A cost-depth optimizer stand-in (``depth.DepthOptimizer``'s mtp_begin / mtp_next contract)."""

    def __init__(self, cut: float) -> None:
        self.cut = cut
        self.calls = []

    def mtp_begin(self) -> None:
        self.calls.append("begin")

    def mtp_next(self, j, p, count):
        self.calls.append((j, round(p, 2)))
        take = j == 0 or p >= self.cut
        return take, take and j + 1 < count and p >= self.cut + 0.2


@needs_torch
@pytest.mark.parametrize("seed", range(6))
def test_mtp_chains_follow_decode_draft(monkeypatch, seed):
    """``MtpChains`` (GLM53_TF_BATCH_MTP) on a fake head: for random backlogs, depths, confidence stops and cost-depth
    stand-ins, every slot gets the drafts, head-cache length, chained-entry count and optimizer calls that
    ``decode.draft`` gives it alone, over several rounds (so the trim of the last round's chained entries too)."""

    import contextlib

    import numpy as np

    from tensorfold.families.glm5_next.cuda import batch
    from tensorfold.families.glm5_next.cuda.decode import draft

    rng = np.random.default_rng(seed)
    head = _Head()
    n = 4
    kinds = [("conf", 0.35), ("conf", 0.0), ("opt", 0.4), ("conf", 0.6)]

    def fake_multi(w, sts, b, tokens, hidden):
        out = []
        for i, (st, t, h) in enumerate(zip(sts, tokens, hidden)):
            lg, o = head.rows(st, t, h)
            b.fnormed[i, 0] = o
            out.append(lg)
        return torch.tensor(out, dtype=torch.int64).view(-1, 1)

    monkeypatch.setattr(batch, "mtp_multi", fake_multi)

    def world():
        sts = [_HeadSt(int(rng.integers(5, 50))) for _ in range(n)]
        steppers = []
        for s in range(n):
            kind, v = kinds[s]
            steppers.append(SimpleNamespace(
                slot=s, job=SimpleNamespace(sampling=None), m_policy=SimpleNamespace(confidence=v if kind == "conf"
                                                                                       else 0.0),
                opt=_Opt(v) if kind == "opt" else None, m_rows=torch.zeros((32, 1), dtype=torch.int64),
                m_next=[], arm="s", drafts=[], steps=0, backlog=0))
        return sts, steppers

    state = rng.bit_generator.state
    ref_sts, ref_steps = world()
    rng.bit_generator.state = state
    got_sts, got_steps = world()
    e_ref, e_got = _HeadEngine(head), _HeadEngine(head)

    @contextlib.contextmanager
    def on(slot):
        saved = e_got.st
        e_got.st = got_sts[slot]
        try:
            yield e_got
        finally:
            e_got.st = saved

    bat = SimpleNamespace(g=SimpleNamespace(e=e_got), _on=on)
    for rnd in range(5):
        plan = []
        for s in range(n):
            k = int(rng.integers(1, 9))
            toks = [int(t) for t in rng.integers(0, 1000, size=k)]
            rows = torch.tensor(rng.integers(0, 1000, size=(k, 1)), dtype=torch.int64)
            depth = int(rng.integers(1, 6))
            plan.append((toks, rows, depth))
        alone = rnd % 3 == 2 and seed % 2                 # some rounds: one slot only (its own path)
        chains = batch.MtpChains()
        for s, (toks, rows, depth) in enumerate(plan):
            if alone and s:
                continue
            # alone: ``Stepper.propose``'s call of decode.draft on the slot
            e_ref.st = ref_sts[s]
            q = ref_steps[s]
            want = draft(e_ref, rows, toks, ref_sts[s].pos + 1, depth, None, q.m_policy.confidence, opt=q.opt)
            q.drafts, q.steps = want, 1 + ref_sts[s].mtp_drafted
            # together
            g = got_steps[s]
            g.m_rows[:len(toks)] = rows
            g.m_next = list(toks)
            chains.add(g, got_sts[s], len(toks), depth)
        chains.run(bat)
        for s in range(n):
            if alone and s:
                continue
            r, g = ref_steps[s], got_steps[s]
            assert g.drafts == r.drafts and g.steps == r.steps and g.arm == "m" and g.m_next == [], (rnd, s)
            a, b_ = ref_sts[s], got_sts[s]
            assert (a.mtp_len, a.mtp_drafted) == (b_.mtp_len, b_.mtp_drafted), (rnd, s)
            assert {k: v for k, v in a.mc.items() if k < a.mtp_len} == {k: v for k, v in b_.mc.items()
                                                                       if k < b_.mtp_len}
            if r.opt is not None:
                assert g.opt.calls == r.opt.calls
            steps = len(r.drafts)
            ref_sts[s].pos += steps + 1                   # a round's commit moves the position on
            got_sts[s].pos += steps + 1


class _Gather:
    """A two-rank all-gather on the CPU: rank 1 holds the same candidates (every value tied across the ranks)."""

    def all_gather(self, src, dst):
        n = src.numel()
        dst[:n].copy_(src)
        dst[n:].copy_(src)


@needs_torch
@pytest.mark.parametrize("world", [1, 2])
def test_sample_drafts_equals_sample_rows(world):
    """``sample_drafts`` (one all-gather for every slot's draft row) == ``decode.sample_rows`` row by row: greedy and
    sampled, with and without the draft's probability (greedy with a probability takes 20 + MARGIN candidates)."""

    import numpy as np

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.glm5_next.cuda.batch import sample_drafts
    from tensorfold.families.glm5_next.cuda.decode import sample_rows

    rng = np.random.default_rng(world)
    V = 300
    w = SimpleNamespace(comm=None if world == 1 else _Gather(), world=world, vocab_offset=17)
    for _ in range(20):
        logits = torch.tensor(rng.normal(size=(5, V)) * 3, dtype=torch.float32).to(torch.bfloat16)
        logits[2, 7] = logits[2, 9] = logits[2].max() + 1            # a tie: the lower id wins
        specs = []
        for r in range(5):
            samp = None if rng.random() < 0.4 else Sampling(int(rng.integers(0, 1 << 40)), float(rng.uniform(0.3, 1.5)),
                                                            int(rng.integers(1, 60)), float(rng.uniform(0.5, 1.0)))
            specs.append((r, int(rng.integers(0, 5000)), samp, bool(rng.random() < 0.6)))
        got = sample_drafts(w, logits, specs)
        for (r, pos, samp, want), (tok, p) in zip(specs, got):
            probs: list[float] = []
            want_tok = sample_rows(w, logits[r:r + 1], [pos], samp, None, probs if want else None)[0]
            assert tok == want_tok
            assert (p is None) == (not want)
            if want:
                assert p == probs[0]


# -- GPU ------------------------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")
    monkeypatch.setenv("GLM53_TF_NONEXPERT", "q4mse")
    monkeypatch.delenv("GLM53_TF_DEPTH", raising=False)


def _engine(path, *, pad: str = "", after: int = 1, short: int = 0, parity: bool = False, mtp: bool = False, **kw):
    from test_batch2_patches import _engine as engine

    with pytest.MonkeyPatch.context() as m:
        m.setenv("GLM53_TF_BATCH_MTP", "1" if mtp else "0")
        m.setenv("GLM53_TF_BATCH_PARITY_KEY", "1" if parity else "0")
        m.setenv("GLM53_TF_BATCH_PAD", pad)
        m.setenv("GLM53_TF_BATCH_CAPTURE_AFTER", str(after))
        m.setenv("GLM53_TF_BATCH_SHORT", str(short))
        return engine(path, **kw)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_batch_parallel")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


@pytest.fixture(scope="module")
def ref(ckpt):
    return _engine(ckpt)


@pytest.fixture(scope="module")
def b4p(ckpt):
    """4 slots, every 0200 knob on."""

    return _engine(ckpt, batch=4, pad="2,4,8", after=2, short=64, parity=True)


def _serial(e, prompt, sampling, tokens):
    from test_batch2_patches import _serial as serial

    return serial(e, prompt, sampling, tokens)


@gpu
def test_padded_rows_are_each_sequences_bits(b4p):
    """Windows padded to 2 / 4 / 8 rows: each real row has the bits of its sequence's lone forward (logits, MTP rows,
    DFlash2 taps) at the padded offsets; the key's first round, its capture on the 2nd sighting and the replay."""

    from test_batch2_patches import _prompt, _reset

    from tensorfold.families.glm5_next.cuda.decode import prefill
    from tensorfold.families.glm5_next.cuda.forward import forward

    bat, e, w = b4p.batch, b4p.e, b4p.w
    b = e.buf
    assert bat.pad == batchplan.pad_mask("2,4,8", 8) and bat.sightings.after == 2 and bat.short == 64
    assert bat.parity_key
    with torch.no_grad():
        for s in range(bat.n):
            with bat._on(s) as es, bat._borrow(s):
                prefill(es, _prompt(61 + s, 40 + 13 * s), None, mtp=True)
        sets = [([0, 1, 2, 3], [[5, 6, 7], [8], [9, 10, 11, 12, 13], [17, 18]]), ([1, 3], [[40, 41, 42], [44]])]
        for active, windows in sets:
            alone = []
            for s, win in zip(active, windows):
                logits = forward(w, bat.states[s], b, win)
                taps = torch.cat([t[:len(win)] for t in b.taps], dim=1)
                alone.append((logits.clone(), b.fnormed[:len(win)].clone(), taps.clone()))
            before = bat.counts["capture"]
            for it in range(3):                 # eager (1st sighting), capture (2nd), replay (3rd)
                got = bat._forward(active, windows).clone()
                ran = list(bat.last_rows)
                assert ran == [batchplan.padded(len(x), bat.pad) for x in windows]
                T = sum(ran)
                normed = b.fnormed[:T].clone()
                taps = torch.cat([t[:T] for t in b.taps], dim=1).clone()
                off = 0
                for (lg, nm, tp), win, n in zip(alone, windows, ran):
                    R = len(win)
                    assert torch.equal(got[off:off + R], lg), (active, windows, it)
                    assert torch.equal(normed[off:off + R], nm) and torch.equal(taps[off:off + R], tp)
                    off += n
                if it == 0:
                    assert bat.counts["capture"] == before
                if it == 1:
                    assert bat.counts["capture"] == before + 1
            assert bat.last_kind == "graph"
    assert bat.counts["pad_rows"] > 0
    _reset(bat)


QUADS = [(None, "f3", "o", "2"), ("auto:1:1:0", "fc5:0.3", "0", "a:0.6:0.85"), ("of", "c3:0.35", "7", "om2")]


@gpu
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
def test_four_requests_equal_serial_with_every_knob(ref, b4p, greedy):
    from test_batch2_patches import _prompt, _sampling

    sampling = _sampling(greedy)
    prompts = [_prompt(70 + i, 30 + 9 * i) for i in range(4)]
    want = [_serial(ref, p, sampling, 40) for p in prompts]
    for quad in QUADS:
        got = b4p.batch.generate_batch([dict(prompt=p, max_tokens=40, sampling=sampling, policy=pol)
                                        for p, pol in zip(prompts, quad)])
        assert [t for t, _ in got] == want, quad
        kinds = [s["round_kinds"] for _, s in got]
        kind = ("alone", "graph", "eager", "capture")
        assert all(sum(d.get(k, 0) for k in kind) == s["rounds"] for d, (_, s) in zip(kinds, got))
        assert all(d["verify_ms"] > 0 for d in kinds)
    c = b4p.batch.counts
    assert c["graph"] >= 1 and c["pad_rows"] > 0
    assert b4p.batch.multi and all(len(k) == 4 for k in b4p.batch.multi)     # parities keyed, no copies


@gpu
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
def test_short_prompts_prefill_in_one_round(ckpt, ref, greedy):
    """A long prompt prefilling in pieces (share 0.3) while short prompts arrive: the short ones get their pieces in
    the round they are admitted, several in one round; every reply equals serial."""

    from test_batch2_patches import _free, _prompt, _sampling

    eng = _engine(ckpt, batch=4, piece=64, share=0.3, short=48)
    try:
        sampling = _sampling(greedy)
        prompts = [_prompt(90, 400), _prompt(91, 20), _prompt(92, 33), _prompt(93, 41)]
        want = [_serial(ref, p, sampling, 16) for p in prompts]
        got = eng.batch.generate_batch([dict(prompt=p, max_tokens=16, sampling=sampling, policy="c3:0.35")
                                        for p in prompts])
        assert [t for t, _ in got] == want
        trace = list(eng.batch.trace)
        assert any(len(p) >= 2 for _, p, _ in trace), trace[:8]
        slots = [s["slot"] for _, s in got]
        first = {s: min(r for r, p, _ in trace if s in p) for s in slots[1:]}
        assert len(set(first.values())) < 3              # at least two short prompts shared their first round
    finally:
        _free(eng)


@pytest.fixture(scope="module")
def b4(ckpt):
    """4 slots, 0120 as it is (every 0200 knob off)."""

    return _engine(ckpt, batch=4)


@pytest.fixture(scope="module")
def b4m(ckpt):
    """4 slots, MTP drafts batched across slots."""

    return _engine(ckpt, batch=4, mtp=True)


@gpu
def test_mtp_multi_rows_are_each_slots_bits(b4m):
    """One head pass over 2-4 slots' MTP rows (absorb backlogs of different lengths, a slot whose head cache is empty,
    single chained rows) == each slot's own ``Engine.mtp``: logits of its last row and the head's output row."""

    from test_batch2_patches import _prompt, _reset

    from tensorfold.families.glm5_next.cuda.batch import mtp_multi
    from tensorfold.families.glm5_next.cuda.decode import prefill

    bat, e, w = b4m.batch, b4m.e, b4m.w
    assert bat.batch_mtp
    g = torch.Generator(device="cpu").manual_seed(3)
    with torch.no_grad():
        for s in range(bat.n):
            with bat._on(s) as es, bat._borrow(s):
                prefill(es, _prompt(81 + s, 30 + 11 * s), None, mtp=True)
        bat.states[3].set_mtp_len(0)                       # an empty head cache: its first embedding is zeroed
        for sizes in ([3, 1, 5, 2], [1, 1, 1], [2, 7]):
            slots = list(range(len(sizes)))
            toks = [[int(t) for t in torch.randint(0, 1000, (k,), generator=g)] for k in sizes]
            hid = [(torch.randn((k, w.cfg.hidden), generator=g) * 0.5).to(torch.bfloat16).to(w.device)
                   for k in sizes]
            alone = []
            for s, t, h in zip(slots, toks, hid):
                with bat._on(s) as es:
                    lg = es.mtp(t, h)
                    alone.append((lg.clone(), es.mbuf.fnormed[0:1].clone()))
            got = mtp_multi(w, [bat.states[s] for s in slots], e.mbuf, toks, hid)[:, :e.draft_n].clone()
            for i, (lg, fn) in enumerate(alone):
                assert torch.equal(got[i:i + 1], lg), (sizes, i)
                assert torch.equal(e.mbuf.fnormed[i:i + 1], fn), (sizes, i)
    _reset(bat)


FIXED = ("c3:0.35", "2", "3", "1")            # policies whose drafts depend on nothing but the request's rows
MTP_QUADS = [("c3:0.35", "2", "om", "c3:0.35"), ("a:0.6:0.85", "3", "om3", "auto:1:1:0"),
             ("o", "c3:0.35", "f3", "1")]


@gpu
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
def test_batched_mtp_drafts_equal_per_slot_drafts(ref, b4, b4m, greedy):
    """4 requests with MTP-heavy policies: replies == serial, and every fixed-policy request's rounds (keeps,
    drafters) are those of the same batch with per-slot drafting: the chains drafted together are the slots' own."""

    from test_batch2_patches import _prompt, _sampling

    sampling = _sampling(greedy)
    prompts = [_prompt(110 + i, 26 + 7 * i) for i in range(4)]
    want = [_serial(ref, p, sampling, 40) for p in prompts]
    batched = 0
    for quad in MTP_QUADS:
        reqs = [dict(prompt=p, max_tokens=40, sampling=sampling, policy=pol) for p, pol in zip(prompts, quad)]
        per_slot = b4.batch.generate_batch(reqs)
        together = b4m.batch.generate_batch(reqs)
        assert [t for t, _ in together] == want, quad
        assert [t for t, _ in per_slot] == want, quad
        for pol, (_, a), (_, b) in zip(quad, per_slot, together):
            if pol in FIXED:                    # (cost-priced choices read the engines' own load-time costs)
                assert a["keeps"] == b["keeps"] and a["drafters"] == b["drafters"], quad
        batched += sum(s["round_kinds"].get("mtp_batched", 0) for _, s in together)
    assert batched > 0
