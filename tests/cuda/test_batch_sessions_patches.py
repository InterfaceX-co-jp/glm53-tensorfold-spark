"""patches/0180 (GLM53_TF_BATCH > 1 with GLM53_TF_SESSION_GIB > 0 and GLM53_TF_BATCH_SESSIONS=1): patches/0110's
session store behind patches/0120's batch slots.

Host only (no torch; runnable anywhere: PYTHONPATH=<tree>/src:<tree>/tests/cuda pytest -q <this file>):

- piece bounds for a fast resume on the 64-token snapshot grid between two bounds of the chunk grid; the piece grid
  (chunk grid and snapshot grid); where a stored session is restored (the free slot holding most of its pages, then
  the least recently admitted); admission memory with the store's headroom set aside; the opt-in switch.

With torch on any device (CPU is enough):

- the store bound to several slots: a save from one slot restores into another (rows, drafter window), each slot's
  live map is its own (pages skipped only where THAT slot holds them), evictions never touch a slot's copy, slots of
  another layout are refused;
- the real ``Batcher._plan`` / ``_execute`` / ``_admit`` / ``_piece`` / ``_finish`` / ``follow`` with the real store on
  a hostile fake model whose rows, KV entries and states depend on everything they may depend on (fast chunks on
  their start and length): 4 sessions over a shared system prompt on 3 slots, interleaved turns (after the reply,
  after the prompt, forks), exact and fast, roomy and tiny budgets: every reply and every slot's whole state at the
  end of a request equals a fresh prefill + serial decode of the same prompt; resumes come from the store across
  slots; a follower replaying rank 0's messages makes the same decisions and ends with the same store;
- Fixes (2026-09-28): the follower's side of a save decision is its role (``follow``), not ``g.rank`` (a second
  batcher that says rank 0, as the GPU test's, replays instead of sending); sessions admitted together over a shared
  system prompt (as many slots as sessions: nothing stored yet at admission) still leave a fork mark, because the
  prompts in flight count as fork partners.

On the GPU (TensorFold's synthetic EXL3 checkpoint with the DFlash2 drafter, one GPU playing rank 0 of two):

- the store is off in batch mode unless GLM53_TF_BATCH_SESSIONS=1;
- 4 sessions (shared 600-token system prompt, mixed policies) on 3 and 4 slots, submitted concurrently in waves,
  sampled and greedy: every reply equals the same request served alone with a fresh prefill (serial decoding), later
  turns resume from the store (restored into whichever slot is free), the fork mark serves a new session;
- the shared system prompt's pages are stored once (reference counts), restores skip pages a slot already holds;
- a tiny budget: entries (also ones a running slot restored) are evicted while slots run, replies stay exact;
- fast prefill (chunks of 128, snapshots on the 64-grid): resumes between two chunk-grid bounds, exact vs lone fast;
- a second batch engine replaying rank 0's plans, prompts, session plans and save decisions (as rank 1 does) makes
  the same restores, saves and evictions and the same replies.

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q
tests/cuda/test_batch_sessions_patches.py
"""

from __future__ import annotations

import collections
import contextlib
import threading
import time
from types import SimpleNamespace

import pytest

from tensorfold.families.glm5_next.cuda import batchplan, sessions

try:
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:          # the host-only tests still run
    torch = None
    CUDA = False

gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")
needs_torch = pytest.mark.skipif(torch is None, reason="torch")

COSTS = {"verify": [29.2, 38.9, 45.2, 55.2, 59.5, 63.2, 65.6, 69.2], "mtp": 1.7, "mtp_step": 1.5, "mtp_row": 0.1,
         "block": 3.0, "taps_row": 0.05}


# -- host only ------------------------------------------------------------------------------------------------------
def test_piece_bounds_resume_between_chunk_bounds():
    # a stored fast snapshot at 64 with 1024-token chunks: the first piece runs to the next chunk bound, then on grid
    ends, done = [], 64
    while done < 5000:
        done = batchplan.piece_end(done, 5000, 1500, 1024)
        ends.append(done)
    assert ends == [1024, 2048, 3072, 4096, 5000]
    assert batchplan.piece_end(960, 5000, 1024, 1024) == 1024        # the next bound, however close
    assert batchplan.piece_end(4160, 4500, 1024, 1024) == 4500       # the prompt ends first
    assert batchplan.piece_end(0, 5000, 2048, 1024) == 2048          # on the grid: as patches/0120
    with pytest.raises(ValueError, match="off the chunk grid"):
        batchplan.piece_end(100, 5000, 2048, 1024)                  # a fast resume is always on 64
    assert batchplan.piece_end(100, 5000, 2048, 0) == 2148           # exact: anywhere


def test_piece_grid():
    assert batchplan.piece_grid(1024, 64) == 1024 and batchplan.piece_grid(64, 64) == 64
    assert batchplan.piece_grid(128, 256) == 256 and batchplan.piece_grid(384, 256) == 768
    assert batchplan.piece_grid(0, 64) == 0 and batchplan.piece_grid(128, 0) == 128


def test_place_entry_and_admission_memory():
    # most pages already held wins; then the least recently admitted; then the lowest slot
    assert batchplan.place_entry([0, 2, 3], {2: 5, 3: 5}, {0: 1, 2: 9, 3: 4}) == 3
    assert batchplan.place_entry([0, 2, 3], {0: 1}, {0: 7, 2: 1, 3: 1}) == 0
    assert batchplan.place_entry([1, 2], {}, {1: 3, 2: 3}) == 1
    assert batchplan.admit_free(10 << 30, 4 << 30) == 6 << 30
    assert batchplan.admit_free(10 << 30, -5) == 10 << 30


def test_opt_in_switch(monkeypatch):
    monkeypatch.delenv("GLM53_TF_BATCH_SESSIONS", raising=False)
    assert not sessions.batch_enabled()
    monkeypatch.setenv("GLM53_TF_BATCH_SESSIONS", "1")
    assert sessions.batch_enabled()
    monkeypatch.setenv("GLM53_TF_BATCH_SESSIONS", "yes")
    with pytest.raises(ValueError, match="0 or 1"):
        sessions.batch_enabled()


# -- torch on any device: a fake model with per-slot caches -----------------------------------------------------------
class _Model:
    """A stand-in for the model on the CPU, hostile as the rules allow (integer hashes: equal means equal): row p's
    state reads the state before it, its token, its position and a digest of every KV entry before p (its attention),
    and in a fast chunk also the chunk's start and length; a KV entry and the MTP row are functions of that. Every
    cache lives on the State, so slots are independent."""

    M = (1 << 61) - 1

    def mix(self, *v: int) -> int:
        h = 1469598103934665603
        for x in v:
            h = (h * 1099511628211 + int(x) + 12345) % self.M
        return h

    def stage(self, w, st, b, tokens):
        st.win = [int(t) for t in tokens]
        return len(st.win)

    def compute(self, w, st, b, R, *, logits=True, nch=None, host_pos=None, npb=None, fast=False, head=True):
        s = int(st.rec[st.cur[0], 0])
        hid = []
        for i, t in enumerate(st.win[:R]):
            p = st.pos + i
            s = self.mix(s, t, p, int(st.dig[p]), st.pos if fast else -1, R if fast else -1)
            st.kv[p] = self.mix(s, 7)
            st.dig[p + 1] = self.mix(int(st.dig[p]), int(st.kv[p]))
            hid.append(s)
        st.hid = torch.tensor(hid, dtype=torch.int64).view(-1, 1)
        st.rec[1 - st.cur[0], 0] = s
        return st.hid.clone()

    def commit(self, w, st, b, R, keep):
        st.rec[1 - st.cur[0], 0] = st.hid[keep - 1, 0]
        st.cur = [1 - st.cur[0]]
        st.conv[0] = torch.tensor(([0, 0, 0] + st.win[:keep])[-3:], dtype=torch.int64)
        st.set_pos(st.pos + keep)


MODEL = _Model()


class _St:
    """A slot's State: KV entries and their prefix digests (``kc``), the MTP head's rows, a KDA-like state."""

    def __init__(self, cap: int = 4096) -> None:
        self.capacity = cap
        self.kv = torch.zeros(cap, dtype=torch.int64)
        self.dig = torch.zeros(cap + 1, dtype=torch.int64)        # dig[p]: digest of kv[0:p]
        self.mkv = torch.zeros(cap, dtype=torch.int64)
        self.kc = [self.kv, self.dig[1:]]
        self.vc = self.kc
        self.index, self.dsa_index = None, {}
        self.mtp_kc = self.mtp_vc = self.mkv
        self.rec = torch.zeros((2, 1), dtype=torch.int64)
        self.conv = torch.zeros((1, 3), dtype=torch.int64)
        self.proj = torch.zeros((1, 8))
        self.cur = [0]
        self.pos = self.mtp_len = self.mtp_drafted = 0
        self.win: list[int] = []
        self.hid = torch.zeros((1, 1), dtype=torch.int64)

    def reset(self) -> None:                  # as ``State.reset``: positions and KDA state; the caches keep junk
        self.rec.zero_()
        self.conv.zero_()
        self.cur = [0]
        self.pos = self.mtp_len = self.mtp_drafted = 0

    def set_pos(self, n: int) -> None:
        self.pos = n

    def set_mtp_len(self, n: int) -> None:
        self.mtp_len = n

    def state(self) -> tuple:
        return (self.pos, int(self.rec[self.cur[0], 0]), self.conv.tolist(), self.kv[:self.pos].tolist(),
                self.dig[:self.pos + 1].tolist(), self.mtp_len, self.mkv[:self.mtp_len].tolist())


class _E:
    """The engine ``decode._prefill`` / ``take_snapshot`` / ``restore`` / ``absorb`` run on (``st`` swapped per slot
    by ``Batcher._on``)."""

    def __init__(self, st: _St, rows: int, fast: bool) -> None:
        from tensorfold.families.glm5_next.cuda import fastpf

        self.st, self.graphs = st, None
        self.w = SimpleNamespace(mtp=object())
        self.buf, self.lean = None, None
        self.mbuf = SimpleNamespace(rows=rows)
        self.rows = self.prefill_rows = rows
        self.fast_prefill, self.fp8_prefill = fast, False
        self.snap_grid = fastpf.grid(rows)       # the fake's fast rows depend on their chunk: grid = C (0080's rule)
        self.fast_snap, self.last_hidden = None, None
        self.mark_snaps, self.checkpoints = [], ()

    def reset(self) -> None:
        self.st.reset()

    def main_hidden(self, rows):
        return self.st.hid[rows]

    def mtp(self, next_tokens, hidden):
        for i, t in enumerate(next_tokens):
            self.st.mkv[self.st.mtp_len + i] = MODEL.mix(int(hidden[i, 0]), t)
        return hidden[-1:]

    def draft_hidden(self, row):
        return self.st.hid[:1]

    def sample(self, logits, positions, sampling, **kw):
        return [int(logits[i, 0]) % 5 for i in range(logits.shape[0])]


class _Stepper:
    """``batch.Stepper`` for the fake: serial rounds (one row a round), the MTP input rows of every committed position
    kept as the reply snapshot's pending rows (as the real Stepper's backlog)."""

    def __init__(self, bat, slot, job, first, last_hidden) -> None:
        self.job, self.out = job, [first]
        self.policy, self.use_mtp, self.drafter, self.opt = "fake", True, None, None
        self.rows = [last_hidden.clone()]
        self.keeps, self.arms, self.depths = [], [], []
        self.rounds = self.shared = 0
        self.start = time.perf_counter()

    def done(self, eos) -> bool:
        return len(self.out) >= self.job.max_tokens

    def propose(self, e) -> None:
        pass

    def window(self) -> list[int]:
        return [self.out[-1]]

    def part(self):
        return "s", 1, 0, 0

    def accept(self, e, sampled, off, shared) -> int:
        MODEL.commit(e.w, e.st, None, 1, 1)
        self.rows.append(e.st.hid[0:1].clone())
        self.out.append(int(sampled[0]))
        self.keeps.append(1)
        self.arms.append("s")
        self.depths.append(0)
        self.rounds += 1
        self.shared += int(shared)
        return 1

    def close(self):
        return torch.cat(self.rows[:len(self.out)])


def _values(fast: bool, rows: int) -> dict:
    from tensorfold.families.glm5_next.cuda import knobs

    v = {k: 0 for k in knobs.HEADER}
    v.update(prefill_rows=rows, fast_prefill=int(fast), auto_fdrafts=7)
    return v


def _fake_batcher(monkeypatch, *, n: int, rows: int, fast: bool, piece: int, budget_pages: float, rank: int = 0,
                  fork: int = 64):
    """A real ``batch.Batcher`` (its ``_plan`` / ``_execute`` / ``_admit`` / ``_piece`` / ``_finish`` / ``follow``) over
    the fake model, with a real ``SessionStore`` bound to its slots; ``_forward``, sampling and the Stepper are the
    fake's. Rank 0 by default (``_share`` returns its input)."""

    import types

    from tensorfold.families.glm5_next.cuda import batch, decode
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    for name in ("stage", "compute", "commit"):
        monkeypatch.setattr(decode, name, getattr(MODEL, name))
    monkeypatch.setattr(decode, "_sync", lambda w: None)
    monkeypatch.setattr(batch, "Stepper", _Stepper)
    monkeypatch.setattr(batch.Batcher, "_free", staticmethod(lambda: 1 << 50))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)

    def sample_multi(w, logits, specs):
        return [[int(logits[off + i, 0]) % 5 for i in range(R)] for off, R, _, _ in specs]

    monkeypatch.setattr(batch, "sample_multi", sample_multi)
    states = [_St() for _ in range(n)]
    e = _E(states[0], rows, fast)
    g = SimpleNamespace(e=e, w=e.w, drafter=None, rank=rank, f_most=7, costs=COSTS, store=None, serial_only=False)
    for name in ("_drafters", "_grid", "_store_save"):
        setattr(g, name, types.MethodType(getattr(GlmEngine, name), g))
    g._share = lambda values: list(values)
    g._knobs = lambda values: contextlib.nullcontext()
    bat = object.__new__(batch.Batcher)
    bat.__dict__.update(
        g=g, n=n, states=states, graphs=[None] * n, drafters=[None] * n, max_rows=8, piece=piece,
        fair=batchplan.Fairness(1.0), reserve=0.0, admit_min=0.0, last_verify=False, store=None, store_budget=0,
        counts=collections.Counter(), caches=[[] for _ in range(n)], used=[0] * n, seqs=[None] * n, round=0,
        eos=(), costs=COSTS, round_costs=batchplan.RoundCosts(COSTS), defaults=_values(fast, rows),
        log=collections.deque(maxlen=1000), trace=collections.deque(maxlen=10000), last_piece_s=0.0,
        cv=threading.Condition(), stopping=False, queue=collections.deque(), error=None, thread=None,
        use_graphs=False, multi={}, pool=None, following=False)

    def forward(active, windows):
        out = []
        for s, win in zip(active, windows):
            with bat._on(s) as ee:
                MODEL.stage(None, ee.st, None, win)
                out.append(MODEL.compute(None, ee.st, None, len(win)))
        return torch.cat(out)

    bat._forward = forward
    store = sessions.SessionStore(e, None, 0, every_tokens=0, fork_tokens=fork)
    store.index.budget = int(budget_pages * store.index.page_bytes)
    g.store = store
    bat.attach_store(store)
    # each request's whole slot state when it ends (before anything else can touch the slot)
    bat.ended = {}
    finish = bat._finish

    def spy(slot, *, cancelled=False):
        q = bat.seqs[slot]
        bat.ended[id(q.job)] = (bat.states[slot].state(), q.job.stats.get("restored"),
                                q.job.stats.get("restored") in store.index.entries)
        return finish(slot, cancelled=cancelled)

    bat._finish = spy
    return bat


def _reference(prompt, tokens: int, rows: int, fast: bool):
    """A fresh prefill of ``prompt`` on a fresh state and serial decoding: (reply, state)."""

    from tensorfold.families.glm5_next.cuda import decode

    st = _St()
    st.kv.fill_(-99)                          # junk where nothing was written
    e = _E(st, rows, fast)
    out = [decode.prefill(e, list(prompt), None, mtp=True, drafter=None)]
    while len(out) < tokens:
        MODEL.stage(None, st, None, [out[-1]])
        h = MODEL.compute(None, st, None, 1)
        MODEL.commit(None, st, None, 1, 1)
        out.append(int(h[0, 0]) % 5)
    return out, st.state()


def _job(prompt, tokens, fast, rows):
    import queue

    from tensorfold.families.glm5_next.cuda.batch import Job

    return Job(list(prompt), tokens, None, False, True, [1, 2, 0, 0], "2", 0, _values(fast, rows),
               out=queue.SimpleQueue())


def _drain(job) -> tuple[list[int], bool]:
    got, done = [], False
    while not job.out.empty():
        item = job.out.get()
        if item is None:
            done = True
        elif isinstance(item, BaseException):
            raise item
        else:
            got.extend(item)
    return got, done


def _conversations(bat, *, rng, fast: bool, rows: int, turns: int = 14, tokens: int = 12, check: bool = True):
    """4 sessions over one shared system prompt, driven round by round on rank 0: each next turn is queued a few
    rounds after the last one ended (the others keep running meanwhile). Every reply and the slot's whole state at
    the end equal a fresh prefill + serial decode. -> (stats of every request, in order)."""

    system = [int(t) for t in rng.integers(0, 1000, size=600)]
    other = [int(t) for t in rng.integers(0, 1000, size=300)]
    base = {"A": system + [int(t) for t in rng.integers(0, 1000, size=150)],
            "B": system + [int(t) for t in rng.integers(0, 1000, size=170)],
            "C": system[:450] + [int(t) for t in rng.integers(0, 1000, size=130)],
            "D": other + [int(t) for t in rng.integers(0, 1000, size=90)]}
    last = {k: None for k in base}             # (prompt, reply) of the session's last turn
    wait = {k: i for i, k in enumerate(base)}  # rounds until the session queues its next turn
    running: dict[str, tuple] = {}
    got_stats = []
    left = {k: turns for k in base}
    rnd = 0
    while any(left.values()) or running:
        for k in base:
            if k in running or not left[k]:
                continue
            if wait[k] > 0:
                wait[k] -= 1
                continue
            if last[k] is None:
                prompt = base[k]
            else:
                p, r = last[k]
                kind = int(rng.integers(0, 4))
                grow = [int(t) for t in rng.integers(0, 1000, size=int(rng.choice([1, 5, 63, 64, 130])))]
                prompt = (p + r + grow if kind < 2 else p + grow if kind == 2 else
                          p[:int(rng.integers(300, len(p) + 1))] + grow)[:3500]
            job = _job(prompt, tokens, fast, rows)
            bat.queue.append(job)
            running[k] = (prompt, job)
            left[k] -= 1
        if bat.queue or any(s is not None for s in bat.seqs):
            cancels, admits, pieces = bat._plan()
            bat._execute(cancels, admits, pieces)
        rnd += 1
        assert rnd < 20000
        for k, (prompt, job) in list(running.items()):
            reply, done = _drain(job)
            job.reply = getattr(job, "reply", []) + reply
            if not done:
                continue
            del running[k]
            last[k] = (prompt, job.reply)
            wait[k] = int(rng.integers(0, 6))
            got_stats.append(job.stats)
            if check:
                want, state = _reference(prompt, tokens, rows, fast)
                assert job.reply == want, (k, job.stats)
                assert bat.ended[id(job)][0] == state, (k, job.stats)
            assert bat.store.index.used <= bat.store.index.budget
    return got_stats


@needs_torch
def test_store_bound_to_slots_copies_across_slots():
    """Save from slot 0, restore into slot 2 (slot 2's caches then hold the entry's rows); the live maps are per slot
    (a restore into slot 1 copies pages slot 2 holds); an eviction after a restore leaves the slot's copy alone; a
    slot of another layout is refused."""

    e = _E(_St(2048), 64, False)
    slots = [e.st, _St(2048), _St(2048)]
    store = sessions.SessionStore(e, None, 0, every_tokens=0, fork_tokens=64)
    store.index.budget = 1000 * store.index.page_bytes
    store.bind_slots(slots)
    ids = list(range(3, 900))
    st0 = slots[0]
    for t in range(len(ids)):                         # what a prefill of ids on slot 0 leaves
        st0.kv[t] = t * 7 + 1
        st0.dig[t + 1] = t * 11 + 2
        st0.mkv[t] = t * 13 + 3
    snap = SimpleNamespace(ids=ids, rec=torch.tensor([5]), conv=torch.zeros(3), pending=torch.zeros((1, 1)),
                           mtp_len=len(ids) - 1, drafter_end=-1, grid=0, window=None)
    with store.on(0):
        store.begin(None, 0)
        assert store.save(snap).status == sessions.SAVED
    entry = store.entry_of(snap)
    assert len(entry.pages) == 3 and store.live_of(0) == dict(enumerate(entry.pages))
    for s in (2, 1):
        slots[s].kv.fill_(-1)
        slots[s].mkv.fill_(-1)
        with store.on(s):
            assert store.restore(entry) is snap
        assert slots[s].kv[:len(ids)].tolist() == st0.kv[:len(ids)].tolist()
        assert slots[s].dig[1:len(ids) + 1].tolist() == st0.dig[1:len(ids) + 1].tolist()
        assert slots[s].mkv[:len(ids) - 1].tolist() == st0.mkv[:len(ids) - 1].tolist()
    assert store.stats["copied"] == 6 and store.stats["skipped"] == 0         # each slot got its own copy
    assert store.overlap(entry, 1) == store.overlap(entry, 2) == 3 and store.cur == 0
    with store.on(2):                                  # slot 2 again: it holds every page already
        store.restore(entry)
    assert store.stats["copied"] == 6 and store.stats["skipped"] == 3
    with store.on(1):                                  # slot 1 runs another prompt from 100 on
        store.begin(None, 100)
    assert store.live_of(1) == {} and len(store.live_of(2)) == 3
    before = slots[2].kv[:len(ids)].clone()
    store.index.evict(entry.id)                        # evicted while slot 2 runs the session it restored
    store._sync_slabs()
    assert not store.slabs and torch.equal(slots[2].kv[:len(ids)], before)
    bad = _St(1024)
    with pytest.raises(RuntimeError, match="layout"):
        store.bind_slots([e.st, bad])
    with pytest.raises(RuntimeError, match="slot 0"):
        store.bind_slots([slots[1], slots[2]])


@needs_torch
@pytest.mark.parametrize("budget", [4000.0, 20.0], ids=["roomy", "tiny"])
@pytest.mark.parametrize("fast", [False, True], ids=["exact", "fast"])
def test_batched_sessions_on_a_hostile_fake_model(monkeypatch, fast, budget):
    import numpy as np

    rows = 64
    bat = _fake_batcher(monkeypatch, n=3, rows=rows, fast=fast, piece=256, budget_pages=budget)
    stats = _conversations(bat, rng=np.random.default_rng(7 + int(fast)), fast=fast, rows=rows)
    ix, store = bat.store.index, bat.store
    restored = [s for s in stats if s.get("restored")]
    assert len(stats) == 56 and len(restored) >= 8, [(s.get("cached"), s.get("slot")) for s in stats]
    assert store.stats["restores"] == len(restored)
    # restores into a slot other than the one the session last ran on happen (sessions move between slots)
    assert len({s["slot"] for s in restored}) >= 2
    assert ix.counters["saved"] > 10 and bat.counts["pieces"] > len(stats)
    if fast:
        assert all(x.tag == 64 and len(x.ids) % 64 == 0 for x in ix.entries.values())
    if budget < 100:
        assert ix.counters["evicted"] > 0
        # a session restored into a slot and evicted from the store while that slot still ran it (this seed's
        # exact run; ``test_store_bound_to_slots_copies_across_slots`` checks the copy survives deterministically)
        if not fast:
            assert any(not still for _, rid, still in bat.ended.values() if rid)
    else:
        assert ix.counters["evicted"] == 0
        first = {x.pages[0] for x in ix.entries.values() if x.pages}
        assert any(ix.pages[k].refs >= 3 for k in first)           # the system prompt's first page, shared


@needs_torch
@pytest.mark.parametrize("fast", [False, True], ids=["exact", "fast"])
def test_fake_follower_replays_rank0(monkeypatch, fast):
    """Rank 0's messages (plans, prompts, session plans with digests, save decisions) replayed through ``follow`` on
    a second batcher playing rank 1: the same requests end the same way, and both stores end identical."""

    import numpy as np

    rows = 64
    r0 = _fake_batcher(monkeypatch, n=3, rows=rows, fast=fast, piece=256, budget_pages=20.0)
    r1 = _fake_batcher(monkeypatch, n=3, rows=rows, fast=fast, piece=256, budget_pages=20.0, rank=1)
    sent: list[list[int]] = []

    def record(values):
        sent.append([int(v) for v in values])
        return list(values)

    class Done(Exception):
        pass

    def replay(values):
        assert values is None
        if not sent:
            raise Done
        return sent.pop(0)

    r0.g._share, r1.g._share = record, replay
    _conversations(r0, rng=np.random.default_rng(3), fast=fast, rows=rows, turns=8, check=False)
    with pytest.raises(Done):
        r1.follow()
    key = lambda d: (d["slot"], d["sha256"], tuple(d["keeps"]), d["cancelled"])      # noqa: E731
    assert [key(d) for d in r1.log] == [key(d) for d in r0.log] and len(r0.log) == 32
    assert [t[1:] for t in r1.trace] == [t[1:] for t in r0.trace]
    a, b = r0.store.index, r1.store.index
    assert a.digest() == b.digest() and sorted(a.entries) == sorted(b.entries) and a.counters == b.counters
    assert a.counters["evicted"] > 0 and r0.store.stats == r1.store.stats
    assert [st.state() for st in r0.states] == [st.state() for st in r1.states]
    # a store that diverged is refused at the next admission's plan
    job = _job(list(range(700)), 4, fast, rows)
    r0.queue.append(job)
    while r0.queue or any(s is not None for s in r0.seqs):
        r0._execute(*r0._plan())
    b.evict(next(iter(b.entries)))
    with pytest.raises(RuntimeError, match="diverged"):
        r1.follow()


def _run_idle(bat) -> None:
    """Rank 0's rounds until nothing waits or runs."""

    rounds = 0
    while bat.queue or any(s is not None for s in bat.seqs):
        bat._execute(*bat._plan())
        rounds += 1
        assert rounds < 20000


@needs_torch
@pytest.mark.parametrize("fast", [False, True], ids=["exact", "fast"])
def test_fake_follower_role_is_follow_not_rank(monkeypatch, fast):
    """Fixes (2026-09-28), the GPU follower test's failure: the follower is whoever runs ``follow`` (as for every
    other message of a round), not ``g.rank``. A second batcher that says rank 0 (as the GPU test's second engine,
    built on one GPU) replays rank 0's save decisions through ``follow`` instead of sending its own."""

    import numpy as np

    rows = 64
    r0 = _fake_batcher(monkeypatch, n=3, rows=rows, fast=fast, piece=256, budget_pages=20.0)
    r1 = _fake_batcher(monkeypatch, n=3, rows=rows, fast=fast, piece=256, budget_pages=20.0, rank=0)
    sent: list[list[int]] = []

    def record(values):
        sent.append([int(v) for v in values])
        return list(values)

    class Done(Exception):
        pass

    def replay(values):
        assert values is None, "the follower sent its own values"
        if not sent:
            raise Done
        return sent.pop(0)

    r0.g._share, r1.g._share = record, replay
    _conversations(r0, rng=np.random.default_rng(5), fast=fast, rows=rows, turns=6, check=False)
    with pytest.raises(Done):
        r1.follow()
    a, b = r0.store.index, r1.store.index
    assert a.counters["saved"] > 0 and a.digest() == b.digest() and a.counters == b.counters
    assert r0.store.stats == r1.store.stats
    assert [st.state() for st in r0.states] == [st.state() for st in r1.states]
    assert not r0.following and r1.following


@needs_torch
@pytest.mark.parametrize("fast", [False, True], ids=["exact", "fast"])
@pytest.mark.parametrize("slots", [3, 4])
def test_fake_concurrent_forks_leave_a_fork_mark(monkeypatch, fast, slots):
    """Fixes (2026-09-28), the GPU test's ``cached == 0`` on 4 slots: 4 sessions sharing a 600-token system prompt,
    submitted at once. Rank 0 plans each admission's marks when it admits it; with as many slots as sessions they are
    all admitted in one round, before anything is stored, so a fork point against the stored entries alone never
    existed (and the later turns resume past it). The in-flight prompts count as fork partners too: a new session
    forking from the system prompt then resumes at the fork mark, whatever the slot count; every reply is exact."""

    import numpy as np

    rows, tokens = 64, 8
    bat = _fake_batcher(monkeypatch, n=slots, rows=rows, fast=fast, piece=256, budget_pages=4000.0, fork=256)
    rng = np.random.default_rng(11)
    system = [int(t) for t in rng.integers(0, 1000, size=600)]
    prompts = [system + [int(t) for t in rng.integers(0, 1000, size=k)] for k in (150, 170, 130, 110)]
    jobs = [_job(p, tokens, fast, rows) for p in prompts]
    bat.queue.extend(jobs)
    _run_idle(bat)
    want = 512 if not fast else 576                     # the page / 64-grid point at or before the shared 600
    for i, (p, job) in enumerate(zip(prompts, jobs)):
        reply, done = _drain(job)
        assert done and reply == _reference(p, tokens, rows, fast)[0]
        # admitted together: fresh; on 3 slots the 4th waits for a slot and resumes at the mark the 2nd took
        assert job.stats["cached"] == (0 if i < slots else want), (i, job.stats)
    assert bat.store.index.has(64 if fast else 0, system[:want], True, False) is not None
    fork = system + [5] * 90
    job = _job(fork, tokens, fast, rows)
    bat.queue.append(job)
    _run_idle(bat)
    reply, done = _drain(job)
    assert done and job.stats["cached"] == want and job.stats.get("restored")
    assert reply == _reference(fork, tokens, rows, fast)[0]


# -- GPU ------------------------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")
    monkeypatch.setenv("GLM53_TF_NONEXPERT", "q4mse")
    monkeypatch.delenv("GLM53_TF_DEPTH", raising=False)


def _engine(path, *, batch: int = 4, gib: float = 1.0, on: bool = True, rows: int = 64, rows_max: int | None = None,
            fast: bool = False, piece: int | None = None, fork: int = 256, every: int = 0, reserve: float = 0.25):
    from test_glm_engine import _TwoCopies

    from tensorfold.families.glm5_next.cuda import weights
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    with pytest.MonkeyPatch.context() as m:
        m.setattr(weights, "NONEXPERT", "q4mse")
        m.setenv("GLM53_TF_NONEXPERT", "q4mse")
        m.setenv("GLM53_TF_BATCH", str(batch))
        m.setenv("GLM53_TF_BATCH_SESSIONS", "1" if on else "0")
        m.setenv("GLM53_TF_SESSION_GIB", str(gib))
        m.setenv("GLM53_TF_SESSION_EVERY", str(every))
        m.setenv("GLM53_TF_SESSION_FORK_MIN", str(fork))
        m.setenv("GLM53_TF_SESSION_RESERVE_GIB", "0")
        m.setenv("GLM53_TF_PREFILL_ROWS", str(rows))
        m.setenv("GLM53_TF_PREFILL_ROWS_MAX", str(rows_max or rows))
        m.setenv("GLM53_TF_FAST_PREFILL", "1" if fast else "0")
        m.setenv("GLM53_TF_BATCH_PREFILL_SHARE", "1.0")
        m.setenv("GLM53_TF_BATCH_RESERVE_GB", str(reserve))
        m.setenv("GLM53_TF_BATCH_ADMIT_GB", "0")
        if piece is not None:
            m.setenv("GLM53_TF_BATCH_PIECE", str(piece))
        else:
            m.delenv("GLM53_TF_BATCH_PIECE", raising=False)
        for k in ("GLM53_TF_CALIB_ONLINE", "GLM53_TF_PROFILE", "GLM53_TF_DEPTH", "GLM53_TF_FAST_GATHER",
                  "GLM53_TF_BATCH_GRAPH_ROWS", "GLM53_TF_BATCH_MAX_GRAPHS", "GLM53_TF_LATENT_KV",
                  "GLM53_TF_LEAN_PREFILL", "GLM53_TF_FP8_PREFILL", "GLM53_TF_SNAPSHOT_GRID"):
            m.delenv(k, raising=False)
        return GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", comm=_TwoCopies())


def _free(*engines) -> None:
    for e in engines:
        if e.batch is not None:
            e.batch.stop()
    torch.cuda.empty_cache()


def _sampling(greedy: bool):
    from tensorfold.engine.exact_sampling import Sampling

    return None if greedy else Sampling(1234, 1.0, 20, 0.95)


def _serial(ref, prompt, sampling, tokens, knobs_=None):
    """The same request served alone: a fresh prefill and serial decoding on a lone engine without a store."""

    out: list[int] = []
    ref.request.policy = None
    ref.request.stop_eos = False
    ref.request.knobs = knobs_
    ref.cache = []
    try:
        stats = ref.generate(list(prompt), tokens, sampling, lambda new: out.extend(new), draft=False)
    finally:
        ref.request.knobs = None
    assert stats["cached"] == 0 and len(out) == tokens
    return out


GREEDY = [True, False]
GIDS = ["greedy", "sampled"]
POLICIES = {"A": "auto", "B": "f3", "C": "2", "D": "o"}


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_batch_sessions")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


@pytest.fixture(scope="module")
def ref(ckpt):
    return _engine(ckpt, batch=1, gib=0, on=False)


def _gpu_sessions(seed: int, system: int = 600):
    import numpy as np

    rng = np.random.default_rng(seed)
    sys_prompt = [int(t) for t in rng.integers(0, 1000, size=system)]
    own = {"A": 150, "B": 170, "C": 130, "D": 110}
    return {k: sys_prompt + [int(t) for t in rng.integers(0, 1000, size=n)] for k, n in own.items()}


def _waves(eng, ref, sampling, prompts, *, waves: int = 3, tokens: int = 20, orders=None):
    """Each wave queues one turn of every session at once (4 concurrent requests); a session's next prompt is its
    prompt + reply + 2 tokens. Every reply against the lone fresh reference. -> stats per wave, by session."""

    turns = {k: list(p) for k, p in prompts.items()}
    out = []
    for w in range(waves):
        order = orders[w] if orders else list(turns)
        reqs = [dict(prompt=turns[k], max_tokens=tokens, sampling=sampling, policy=POLICIES[k]) for k in order]
        got = eng.batch.generate_batch(reqs)
        wave = {}
        for k, (reply, stats) in zip(order, got):
            assert reply == _serial(ref, turns[k], sampling, tokens), (w, k, stats)
            wave[k] = (len(turns[k]), stats)
            turns[k] = turns[k] + reply + [7 + w, 11 + w]
        out.append(wave)
    return out


@gpu
def test_store_off_in_batch_mode_by_default(ckpt):
    e = _engine(ckpt, batch=2, on=False)
    assert e.batch is not None and e.store is None and e.batch.store is None
    _free(e)


@gpu
@pytest.mark.parametrize("greedy", GREEDY, ids=GIDS)
@pytest.mark.parametrize("slots", [3, 4])
def test_concurrent_interleaved_sessions_equal_alone(ckpt, ref, slots, greedy):
    """4 sessions (shared system prompt, one policy each), 3 waves of 4 concurrent requests on 3 / 4 slots, the order
    shuffled per wave so sessions change slots: every reply equals the request served alone; turns 2 and 3 resume
    from what the session saved (from the store when it moved to another slot); the fork mark serves a new session."""

    eng = _engine(ckpt, batch=slots)
    assert eng.batch.n == slots and eng.store is eng.batch.store
    sampling = _sampling(greedy)
    prompts = _gpu_sessions(1 if greedy else 2)
    waves = _waves(eng, ref, sampling, prompts, orders=[list("ABCD"), list("DCBA"), list("BDAC")])
    for w in (1, 2):                    # every follow-up resumes after the reply (a slot's own or the store's)
        for k, (n, stats) in waves[w].items():
            before, _ = waves[w - 1][k]
            assert stats["cached"] >= before + 20 - 1, (w, k, stats)
    moved = [s for wave in waves[1:] for _, s in wave.values() if s.get("restored")]
    if slots == 3:                      # 4 sessions on 3 slots: some session resumes on another slot
        assert moved
    assert all(s["cached"] > 0 for s in moved)
    # a new session forking from the system prompt resumes at the fork mark (a page point <= 600)
    fork = prompts["A"][:600] + [5] * 90
    (reply, stats), = eng.batch.generate_batch([dict(prompt=fork, max_tokens=12, sampling=sampling)])
    assert stats["cached"] == 512 and reply == _serial(ref, fork, sampling, 12), stats
    ix = eng.store.index
    assert sum(len(x.pages) for x in ix.entries.values()) > len(ix.pages)
    first = {x.pages[0] for x in ix.entries.values() if x.pages}
    assert any(ix.pages[k].refs >= 3 for k in first)                  # the system prompt's pages stored once
    assert eng.store.stats["skipped"] > 0                              # pages a slot already held are not copied
    _free(eng)


@gpu
@pytest.mark.parametrize("greedy", GREEDY, ids=GIDS)
def test_eviction_under_a_tiny_budget_while_slots_run(ckpt, ref, greedy):
    eng = _engine(ckpt, batch=3)
    sampling = _sampling(greedy)
    prompts = _gpu_sessions(3)
    _waves(eng, ref, sampling, prompts, waves=1)
    ix = eng.store.index
    biggest = max(x.private for x in ix.entries.values())
    ix.budget = 2 * biggest + 2 * ix.extent_bytes          # about two entries and two slabs of pages
    _waves(eng, ref, sampling, _gpu_sessions(4), waves=3, orders=[list("ABCD"), list("CADB"), list("DBCA")])
    assert ix.counters["saved"] > ix.counters["evicted"] > 0 and ix.used <= ix.budget
    _free(eng)


@gpu
@pytest.mark.parametrize("greedy", GREEDY, ids=GIDS)
def test_fast_prefill_sessions_batched(ckpt, greedy):
    """Fast prefill with 128-row chunks and 64-token snapshots: stored grid snapshots / marks between two chunk
    bounds resume through a first piece that ends on the next bound; replies equal the lone fast engine's fresh
    ones; entries are fast-tagged only and exact requests never restore them."""

    lone = _engine(ckpt, batch=1, gib=0, on=False, fast=True, rows=128, rows_max=256)
    eng = _engine(ckpt, batch=3, fast=True, rows=128, rows_max=256, piece=256, fork=64)
    sampling = _sampling(greedy)
    prompts = _gpu_sessions(5, system=640)
    turns = {k: list(p) for k, p in prompts.items()}
    for w in range(3):
        order = list("ABCD") if w % 2 == 0 else list("DBCA")
        got = eng.batch.generate_batch([dict(prompt=turns[k], max_tokens=16, sampling=sampling, policy="2")
                                        for k in order])
        for k, (reply, stats) in zip(order, got):
            assert reply == _serial(lone, turns[k], sampling, 16), (w, k, stats)
            if w:
                assert stats["cached"] % 64 == 0 and stats["cached"] > 0, (w, k, stats)
            turns[k] = turns[k] + reply + [3, 4, 5]
    assert all(x.tag == 64 for x in eng.store.index.entries.values())
    (exact, se), = eng.batch.generate_batch([dict(prompt=turns["A"], max_tokens=12, sampling=sampling,
                                                  knobs={"fast_prefill": 0})])
    assert se["cached"] == 0 and not se.get("restored")
    assert exact == _serial(lone, turns["A"], sampling, 12, knobs_={"fast_prefill": 0})
    _free(lone, eng)


class _Done(Exception):
    pass


@gpu
@pytest.mark.parametrize("greedy", GREEDY, ids=GIDS)
def test_follower_replays_rank0_sessions(ckpt, greedy):
    """A second batch engine with the store replays rank 0's round plans, prompts, session plans and save decisions
    through ``follow``, as rank 1 does: the same replies, keeps and drafters, the same store at the end (digest,
    entries, counters, restores), with evictions; a diverged store is refused."""

    r0 = _engine(ckpt, batch=3, piece=128)
    r1 = _engine(ckpt, batch=3, piece=128)
    r1.batch.costs = r0.batch.costs
    from tensorfold.families.glm5_next.cuda import batchplan as bp

    r1.batch.round_costs = bp.RoundCosts(r0.batch.costs)
    sent: list[list[int]] = []
    share = r0._share

    def record(values):
        got = share(values)
        sent.append(list(got))
        return got

    def replay(values):
        assert values is None
        if not sent:
            raise _Done
        return sent.pop(0)

    r0._share, r1._share = record, replay
    sampling = _sampling(greedy)
    ix = r0.store.index
    prompts = _gpu_sessions(6)
    turns = {k: list(p) for k, p in prompts.items()}
    for w in range(3):
        order = [list("ABCD"), list("DACB"), list("CBDA")][w]
        got = r0.batch.generate_batch([dict(prompt=turns[k], max_tokens=16, sampling=sampling, policy=POLICIES[k])
                                       for k in order])
        for k, (reply, _) in zip(order, got):
            turns[k] = turns[k] + reply + [w]
        time.sleep(0.5)                                       # rank 0's loop is back waiting for work
        with pytest.raises(_Done):
            r1.batch.follow()
        assert r1.store.index.digest() == ix.digest()
        if w == 0:                                            # evictions from here on, on both ranks
            ix.budget = r1.store.index.budget = 3 * max(x.private for x in ix.entries.values()) + 2 * ix.extent_bytes
    key = lambda d: (d["sha256"], tuple(d["keeps"]), d["arms"], d["cancelled"])      # noqa: E731
    assert [key(d) for d in r1.batch.log] == [key(d) for d in r0.batch.log] and len(r0.batch.log) == 12
    fx = r1.store.index
    assert fx.digest() == ix.digest() and sorted(fx.entries) == sorted(ix.entries) and fx.counters == ix.counters
    assert r1.store.stats == r0.store.stats and r0.store.stats["restores"] > 0 and ix.counters["evicted"] > 0
    r0.batch.generate_batch([dict(prompt=turns["A"], max_tokens=8, sampling=sampling)])
    time.sleep(0.5)
    fx.evict(next(iter(fx.entries)))
    with pytest.raises(RuntimeError, match="diverged"):
        r1.batch.follow()
    del r0._share, r1._share
    _free(r0, r1)
