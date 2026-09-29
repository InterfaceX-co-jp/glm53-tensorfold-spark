"""patches/0370: host work off the decode round's critical path (GLM53_TF_DECODE_OVERLAP) and host threads on the fast
cores (GLM53_TF_CPU_PIN). Both off by default; scheduling only.

Checked (host only / torch on the CPU; no GPU needed):

- the knob's parsing, the plan rider's encoding, ``Emitter`` (order, the callback's exception at ``close``) and
  ``ForwardTimer``'s CPU fallback;
- patches/0180's hostile fake model under real ``Batcher._plan`` / ``_execute`` / ``_verify`` / ``_finish`` / ``follow``
  with every part of the knob, alone and together: every reply and every slot's state at its end equal a fresh
  prefill + serial decoding (the same assertion as without the knob);
- lockstep: rank 0's messages (plan shares AND the sampler exchanges with their riders) replayed in ONE ordered stream
  through ``follow`` on a batcher playing rank 1: every message is of the kind rank 1 expects at that point, rank 1
  sends zero riders, both ranks end the same requests the same way (sha, keeps, cancels) through the same rounds, with
  identical stores and slot states; plans did ride (fewer plan shares than without the knob), cancels included;
- ``emit``: a round's tokens reach the caller only after the next round's forward was launched (or at the request's
  end), in order, before its end marker; nothing is lost;
- ``_plan_ahead`` plans only cancels-only rounds (not with a queue, a prompt prefilling, or while stopping);
- ``auto_decode`` on a fake engine: the same tokens, keeps and online-timer rows with ``sync`` as without;
- ``cpupin``: GB10's cores read from a fake sysfs (X925 / A725 by part, capacities, L3), the auto / fast / explicit
  plans, a container that allows fewer cpus, errors; on this Linux host: ``serving`` pins the calling thread, a thread
  it starts inherits its core and ``sweep`` moves it to ``rest``, NCCL-named threads go to ``comm``, a thread sitting
  on the RoCE cpu is left there;
- ``scripts/serve.sh``: ``CPUSET`` -> ``docker run --cpuset-cpus`` (tests/test_serve_ops.py).

Run: PYTHONPATH=<tree>/src:<tree>/tests/cuda:tests/cuda pytest -q tests/cuda/test_decode_overlap_patches.py
"""

from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace

import pytest

from tensorfold.families.glm5_next.cuda import cpupin, decode_overlap as dover

try:
    import torch
except ImportError:          # the host-only tests still run
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="torch")
linux = pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="Linux affinity calls")


# -- the knob ---------------------------------------------------------------------------------------------------------
def test_parse():
    assert not dover.parse(None).on and not dover.parse("0").on and not dover.parse("").on
    all_ = dover.parse("1")
    assert all_.sync and all_.emit and all_.plan and all_.gil and all_.switch_us == 500
    assert all_.code() == 15 and all_.describe() == "sync,emit,plan,gil"
    two = dover.parse("sync, plan")
    assert two.sync and two.plan and not two.emit and not two.gil and two.code() == 0b101
    assert dover.parse("gil", "250").switch_us == 250
    with pytest.raises(ValueError, match="unknown part"):
        dover.parse("sync,fast")
    with pytest.raises(ValueError, match="100..5000"):
        dover.parse("gil", "50")


def test_env_is_read_once(monkeypatch):
    dover.env.cache_clear()
    monkeypatch.setenv("GLM53_TF_DECODE_OVERLAP", "emit")
    try:
        assert dover.env().emit and not dover.env().sync
        monkeypatch.setenv("GLM53_TF_DECODE_OVERLAP", "0")
        assert dover.env().emit                                  # cached for the process
    finally:
        dover.env.cache_clear()


def test_rider_round_trip():
    from tensorfold.families.glm5_next.cuda import batchplan

    n = dover.rider_size(4)
    assert n == 12
    for cancels in ([], [2], [0, 1, 2, 3]):
        plan = batchplan.encode_plan(cancels, [], [])
        words = dover.rider_encode(plan, n)
        assert len(words) == n and dover.rider_decode(words) == plan
        assert batchplan.parse_plan(dover.rider_decode(words)) == (cancels, [], [])
    assert dover.rider_decode(dover.rider_encode(None, n)) is None
    assert dover.rider_decode([0] * n) is None                   # rank 1's words
    with pytest.raises(ValueError, match="does not fit"):
        dover.rider_encode(list(range(20)), n)
    with pytest.raises(RuntimeError, match="same patched"):
        dover.rider_decode([7] + [0] * (n - 1))


def test_emitter_order_and_errors():
    got = []
    em = dover.Emitter(lambda toks: got.append(list(toks)) or False)
    for i in range(200):
        assert em([i, i + 1]) is False
    em.close()
    assert got == [[i, i + 1] for i in range(200)]

    def bad(toks):
        if toks[0] == 3:
            raise KeyError("client")
        got.append(toks)

    got.clear()
    em = dover.Emitter(bad)
    for i in range(6):
        em([i])
    with pytest.raises(KeyError):
        em.close()
    assert got == [[0], [1], [2]]                                # nothing after the failure
    em = dover.Emitter(bad)
    em([3])
    em.close(raise_error=False)                                  # while another exception propagates


@needs_torch
def test_forward_timer_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    t = dover.ForwardTimer()
    t.start()
    time.sleep(0.01)
    t.stop()
    assert 5.0 < t.ms() < 1000.0


# -- the batcher on patches/0180's hostile fake model -----------------------------------------------------------------
PARTS = {"off": "0", "sync": "sync", "emit": "emit", "plan": "plan", "all": "sync,emit,plan"}


def _overlap_batcher(monkeypatch, spec: str, *, n: int = 3, rank: int = 0, fast: bool = False):
    import test_batch_sessions_patches as tb

    bat = tb._fake_batcher(monkeypatch, n=n, rows=64, fast=fast, piece=256, budget_pages=20.0, rank=rank)
    bat.overlap = dover.parse(spec)
    if bat.overlap.plan:
        bat.rider_n = dover.rider_size(n)
        bat.rider_host = torch.zeros((bat.rider_n,), dtype=torch.int32)
        bat.rider_dev = torch.zeros((bat.rider_n,), dtype=torch.int32)
    return bat


class _Wire:
    """One ordered stream of rank 0's messages: ``_share`` values and sampler exchanges (with the rider rank 0 put in
    them). Rank 1 replays it: each message must be of the kind rank 1 asks for next (the two ranks' collectives in
    the same order), and rank 1's own rider must be zeros."""

    class Done(Exception):
        pass

    def __init__(self, monkeypatch) -> None:
        from tensorfold.families.glm5_next.cuda import batch

        self.msgs: list[tuple[str, list[int]]] = []
        self.replaying = False
        self.samples = 0
        wire = self

        def sample_multi(w, logits, specs, rider=None):
            toks = [[int(logits[off + i, 0]) % 5 for i in range(R)] for off, R, _, _ in specs]
            if rider is None:
                return toks
            if not wire.replaying:
                words = [int(v) for v in rider.tolist()]
                wire.msgs.append(("rider", words))
                return toks, words
            assert all(int(v) == 0 for v in rider.tolist()), "rank 1 sent a plan"
            kind, words = wire._pop()
            assert kind == "rider", f"rank 1 exchanged a sampler rider where rank 0 sent a {kind}"
            return toks, words

        monkeypatch.setattr(batch, "sample_multi", sample_multi)

    def _pop(self):
        if not self.msgs:
            raise self.Done
        return self.msgs.pop(0)

    def attach(self, r0, r1) -> None:
        def record(values):
            self.msgs.append(("share", [int(v) for v in values]))
            return list(values)

        def replay(values):
            assert values is None
            kind, got = self._pop()
            assert kind == "share", f"rank 1 read a plan share where rank 0 sent a {kind}"
            return got

        r0.g._share, r1.g._share = record, replay
        if r0.overlap.plan:                     # the driver takes a plan that rode in, as ``_serve`` does
            plan = r0._plan
            r0._plan = lambda: r0._take_ahead() or plan()

    def rode(self) -> int:
        return sum(1 for k, w in self.msgs if k == "rider" and w[0] == 1)


@needs_torch
@pytest.mark.parametrize("part", list(PARTS))
@pytest.mark.parametrize("fast", [False, True], ids=["exact", "fast"])
def test_replies_exact_with_every_part(monkeypatch, part, fast):
    """0180's 4 sessions x 14 turns over shared prefixes: every reply and end state == fresh prefill + serial."""

    import numpy as np
    import test_batch_sessions_patches as tb

    bat = _overlap_batcher(monkeypatch, PARTS[part], fast=fast)
    wire = _Wire(monkeypatch)
    if bat.overlap.plan:
        plan = bat._plan
        bat._plan = lambda: bat._take_ahead() or plan()
    stats = tb._conversations(bat, rng=np.random.default_rng(11 + int(fast)), fast=fast, rows=64)
    assert len(stats) == 56
    if bat.overlap.plan:
        riders = sum(1 for k, _ in wire.msgs if k == "rider")
        print(f"[0370] {part}: {wire.rode()} of {riders} verify rounds planned the next one")
        assert wire.rode() > 50                                   # busy mix: arrivals and pieces plan at the top
    assert bat._held is None                                      # nothing left undelivered


@needs_torch
@pytest.mark.parametrize("part", ["plan", "all"])
@pytest.mark.parametrize("fast", [False, True], ids=["exact", "fast"])
def test_follower_replays_rank0_in_lockstep(monkeypatch, part, fast):
    import numpy as np
    import test_batch_sessions_patches as tb

    r0 = _overlap_batcher(monkeypatch, PARTS[part], fast=fast)
    r1 = _overlap_batcher(monkeypatch, PARTS[part], fast=fast, rank=1)
    wire = _Wire(monkeypatch)
    wire.attach(r0, r1)
    tb._conversations(r0, rng=np.random.default_rng(3), fast=fast, rows=64, turns=8, check=False)
    _settle(r0)
    rode = wire.rode()
    shares = sum(1 for k, _ in wire.msgs if k == "share")
    wire.replaying = True
    with pytest.raises(_Wire.Done):
        r1.follow()
    key = lambda d: (d["slot"], d["sha256"], tuple(d["keeps"]), d["cancelled"])      # noqa: E731
    assert [key(d) for d in r1.log] == [key(d) for d in r0.log] and len(r0.log) == 32
    assert [t[1:] for t in r1.trace] == [t[1:] for t in r0.trace]
    a, b = r0.store.index, r1.store.index
    assert a.digest() == b.digest() and a.counters == b.counters and r0.store.stats == r1.store.stats
    assert [st.state() for st in r0.states] == [st.state() for st in r1.states]
    assert rode > 50
    # the same conversations without the knob share a plan at the top of every round
    off0 = _overlap_batcher(monkeypatch, "0", fast=fast)
    off1 = _overlap_batcher(monkeypatch, "0", fast=fast, rank=1)
    wire2 = _Wire(monkeypatch)
    wire2.attach(off0, off1)
    tb._conversations(off0, rng=np.random.default_rng(3), fast=fast, rows=64, turns=8, check=False)
    shares_off = sum(1 for k, _ in wire2.msgs if k == "share")
    # (the schedules differ by the rounds an arrival waits, so the counts compare loosely)
    assert shares < shares_off, (shares, shares_off, rode)


def _settle(bat) -> None:
    """The round a plan that rode in describes (``_serve`` runs it at once; the tests' drivers stop when nothing is in
    flight, while rank 1's ``follow`` runs it)."""

    while bat._ahead is not None:
        bat._execute(*bat._take_ahead())


def _run_rounds(bat, jobs, *, cancel_at: dict | None = None, check=None):
    """Queue ``jobs`` and run rank 0's rounds (the ``_serve`` loop's order) until all ended."""

    bat.queue.extend(jobs)
    rounds = 0
    while bat.queue or any(s is not None for s in bat.seqs) or bat._ahead is not None:
        for idx, at in (cancel_at or {}).items():
            if rounds == at:
                jobs[idx].cancel = True
        plan = bat._take_ahead() or bat._plan()
        bat._execute(*plan)
        if check is not None:
            check(bat, rounds)
        rounds += 1
        assert rounds < 5000
    return rounds


@needs_torch
@pytest.mark.parametrize("part", ["plan", "all"])
def test_cancels_ride_and_replay(monkeypatch, part):
    """A request cancelled mid-decode (the client went away) ends at the same round on both ranks, through a plan that
    rode in the sampler exchange; the others' replies are unchanged."""

    import numpy as np
    import test_batch_sessions_patches as tb

    rng = np.random.default_rng(5)
    prompts = [[int(t) for t in rng.integers(0, 1000, size=s)] for s in (300, 200, 250)]
    r0 = _overlap_batcher(monkeypatch, PARTS[part])
    r1 = _overlap_batcher(monkeypatch, PARTS[part], rank=1)
    wire = _Wire(monkeypatch)
    wire.attach(r0, r1)
    jobs = [tb._job(p, 40, False, 64) for p in prompts]
    _run_rounds(r0, jobs, cancel_at={1: 12})
    assert jobs[1].stats.get("cancelled") is True
    cancel_riders = [w for k, w in wire.msgs if k == "rider" and w[0] == 1 and w[2] > 0]
    assert cancel_riders, "the cancel did not ride"
    for i in (0, 2):
        got, done = tb._drain(jobs[i])
        assert done and got == tb._reference(prompts[i], 40, 64, False)[0]
    wire.replaying = True
    with pytest.raises(_Wire.Done):
        r1.follow()
    key = lambda d: (d["slot"], d["sha256"], tuple(d["keeps"]), d["cancelled"])      # noqa: E731
    assert [key(d) for d in r1.log] == [key(d) for d in r0.log]
    assert [t[1:] for t in r1.trace] == [t[1:] for t in r0.trace]
    assert [st.state() for st in r0.states] == [st.state() for st in r1.states]


@needs_torch
@pytest.mark.parametrize("first_now", [False, True], ids=["first-held", "first-now"])
def test_emit_after_the_next_forward_launch(monkeypatch, first_now):
    """``emit``: tokens a round accepted are handed out when the next round's forward has been launched (or when the
    request ends), in order, before the end marker, none lost. patches/0540 (GLM53_TF_EMIT_FIRST, default on): a
    prompt's first token goes out when its piece ends instead (``first-now``), so no token waits for a launch here."""

    import numpy as np
    import test_batch_sessions_patches as tb

    bat = _overlap_batcher(monkeypatch, "emit")
    bat.emit_first = first_now
    rng = np.random.default_rng(9)
    prompts = [[int(t) for t in rng.integers(0, 1000, size=s)] for s in (130, 190)]
    jobs = [tb._job(p, 25, False, 64) for p in prompts]
    seen = {id(j): [] for j in jobs}
    forward = bat._forward
    launches = []

    def spy(active, windows):
        launches.append(sum(len(t) for _, t in (bat._held or [])))
        out = forward(active, windows)
        return out

    bat._forward = spy

    def check(b, rnd):
        active = set(b.trace[-1][2]) if b.trace and b.trace[-1][0] == b.round else set()
        held = {id(job): sum(len(t) for jj, t in (b._held or []) if jj is job) for job in jobs}
        ended = False
        for job in jobs:
            got, done = tb._drain(job)
            seen[id(job)].extend(got)
            if done:
                assert held[id(job)] == 0                         # its end flushed everything first
                seen[id(job)].append(None)
                ended = True
        for job in jobs:
            slot = next((i for i, s in enumerate(b.seqs) if s is not None and s.job is job), None)
            if slot is not None and slot in active:
                # this round's token waits for the next launch (a request's end hands out everyone's)
                assert held[id(job)] == (0 if ended else 1)

    _run_rounds(bat, jobs, check=check)
    for p, job in zip(prompts, jobs):
        toks = seen[id(job)]
        assert toks[-1] is None and None not in toks[:-1]
        assert toks[:-1] == tb._reference(p, 25, 64, False)[0]
    assert bat._held is None and len(launches) > 20
    if first_now:
        assert max(launches) == 0                                 # patches/0540: first tokens never wait for a launch
    else:
        assert max(launches) >= 1                                 # tokens were waiting when forwards launched


@needs_torch
def test_plan_ahead_only_for_cancel_only_rounds(monkeypatch):
    import test_batch_sessions_patches as tb

    bat = _overlap_batcher(monkeypatch, "plan")
    _Wire(monkeypatch)
    job = tb._job(list(range(10, 200)), 30, False, 64)
    bat.queue.append(job)
    while not any(s is not None and s.stepper is not None for s in bat.seqs):
        bat._execute(*bat._plan())
    assert bat._plan_ahead() == [0, 0, 0]                         # decoding, nothing waits: nothing to do
    job.cancel = True
    slot = next(i for i, s in enumerate(bat.seqs) if s is not None)
    assert bat._plan_ahead() == [1, slot, 0, 0]
    job.cancel = False
    bat.queue.append(tb._job(list(range(5, 90)), 5, False, 64))
    assert bat._plan_ahead() is None                              # an admission is the loop's decision
    bat.queue.clear()
    bat.stopping = True
    assert bat._plan_ahead() is None
    bat.stopping = False
    other = SimpleNamespace(job=SimpleNamespace(cancel=False), stepper=None)
    free = next(i for i, s in enumerate(bat.seqs) if s is None)
    bat.seqs[free] = other                                        # a prompt still prefilling: its piece
    assert bat._plan_ahead() is None
    bat.seqs[free] = None
    bat.following = True
    bat.rider_host.zero_()
    words = bat._rider().tolist()
    assert words == [0] * bat.rider_n                             # rank 1 never plans


# -- single-engine decode loops: ``sync`` ------------------------------------------------------------------------------
class _FakeDecode:
    """An ``auto_decode`` engine on the CPU: a row's logits are a hash of (position, token, previous state); MTP drafts
    are right 3 times in 4."""

    def __init__(self) -> None:
        self.w = SimpleNamespace(cfg=SimpleNamespace(eos=(), hidden=4), device="cpu")
        self.st = SimpleNamespace(pos=100, mtp_drafted=0)
        self.buf = SimpleNamespace(taps=())
        self.rows = 8
        self.last_hidden = torch.zeros((1, 4), dtype=torch.bfloat16)
        self.state = 7
        self.win: list[tuple[int, int]] = []
        self.calib_rows: list[int] = []
        self.lookup = self.depth = None
        self.calib = lambda R, ms: self.calib_rows.append(R)

    def _h(self, *v):
        h = 1469598103934665603
        for x in v:
            h = (h * 1099511628211 + int(x) + 1) % ((1 << 61) - 1)
        return h

    def forward(self, tokens):
        s, rows = self.state, []
        for i, t in enumerate(tokens):
            s = self._h(s, t, self.st.pos + i)
            rows.append(s)
        self.win = rows
        return torch.tensor(rows, dtype=torch.int64).view(-1, 1)

    def sample(self, logits, positions, sampling):
        return [int(logits[i, 0]) % 97 for i in range(logits.shape[0])]

    def main_hidden(self, rows):
        return torch.zeros((rows.stop - rows.start, 4), dtype=torch.bfloat16)

    def tap_rows(self, n):
        return torch.zeros((n, 0), dtype=torch.bfloat16)


@needs_torch
@pytest.mark.parametrize("spec", ["0", "sync"])
def test_auto_decode_same_tokens_with_sync(monkeypatch, spec):
    from tensorfold.families.glm5_next.cuda import decode

    results = {}
    for s in ("0", spec):
        dover.env.cache_clear()
        monkeypatch.setenv("GLM53_TF_DECODE_OVERLAP", s)
        e = _FakeDecode()
        syncs = []
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: syncs.append(1))
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        def commit(w, st, b, R, keep, e=e):
            e.state = e.win[keep - 1]
            st.pos += keep

        def draft(e_, hidden, next_tokens, pos, depth, sampling, conf, opt=None, e=e):
            s, out = e.state, []
            for i in range(depth):          # the serial chain from the committed state, wrong at every 4th guess
                s = e._h(s, next_tokens[-1] if i == 0 else out[-1], e.st.pos + i)
                tok = s % 97
                out.append(tok if (pos + i) % 4 else (tok + 1) % 97)
            return out

        monkeypatch.setattr(decode, "commit", commit)
        monkeypatch.setattr(decode, "draft", draft)
        monkeypatch.setattr(decode, "absorb", lambda *a, **k: None)
        res = decode.auto_decode(e, None, 5, 120, None, choice=None, m_policy=decode.DepthPolicy(3, fixed=True),
                                 f_policy=decode.DepthPolicy(3, fixed=True))
        results[s] = (res.tokens, res.keeps, e.calib_rows, len(syncs))
    dover.env.cache_clear()
    base = results["0"]
    got = results[spec]
    assert got[0] == base[0] and got[1] == base[1] and got[2] == base[2] and len(base[0]) == 120
    assert any(k > 1 for k in base[1])                            # drafts were kept
    if spec == "sync":
        assert got[3] == base[3] - len(base[1])                   # one host wait less a round (the two _sync stay)


# -- cpupin -----------------------------------------------------------------------------------------------------------
def _gb10(tmp_path):
    """A fake /sys/devices/system/cpu and /proc/cpuinfo as on both Sparks (read 2026-09-28)."""

    root = tmp_path / "cpu"
    lines = []
    for n in range(20):
        fast = n in range(5, 10) or n in range(15, 20)
        d = root / f"cpu{n}"
        (d / "cpufreq").mkdir(parents=True)
        cap = (997 if n < 10 else (1024 if n == 19 else 1017)) if fast else (718 if n < 10 else 731)
        (d / "cpu_capacity").write_text(f"{cap}\n")
        (d / "cpufreq" / "cpuinfo_max_freq").write_text("3900000\n" if fast else "2808000\n")
        for i, (lvl, size) in enumerate(((1, "64K"), (2, "2048K" if fast else "512K"),
                                         (3, "8192K" if n < 10 else "16384K"))):
            idx = d / "cache" / f"index{i}"
            idx.mkdir(parents=True)
            (idx / "level").write_text(f"{lvl}\n")
            (idx / "size").write_text(size + "\n")
        lines += [f"processor\t: {n}", "CPU implementer\t: 0x41",
                  f"CPU part\t: {'0xd85' if fast else '0xd87'}", ""]
    info = tmp_path / "cpuinfo"
    info.write_text("\n".join(lines))
    return cpupin.read_cpus(root, info)


def test_gb10_cores(tmp_path):
    cpus = _gb10(tmp_path)
    assert cpus[5].part == cpupin.X925 and cpus[0].part == cpupin.A725
    assert cpus[19].capacity == 1024 and cpus[15].l3_kb == 16384 and cpus[5].l3_kb == 8192
    assert cpupin.fast_cpus(cpus, range(20)) == [19, 18, 17, 16, 15, 9, 8, 7, 6, 5]
    # a container limited to the A725s: the best of what it allows
    assert cpupin.fast_cpus(cpus, range(5)) == [4, 3, 2, 1, 0]
    assert cpupin.fast_cpus(cpus, [0, 1, 12, 13]) == [13, 12, 1, 0]      # one class: best first


def test_plans(tmp_path):
    cpus = _gb10(tmp_path)
    allow = frozenset(range(20))
    assert cpupin.plan("0", cpus, allow) is None and cpupin.plan(None, cpus, allow) is None
    p = cpupin.plan("auto", cpus, allow, roce=True)
    assert p.serve == {19} and p.roce == 18 and p.comm == {16, 17} and p.rest == frozenset(range(16))
    p = cpupin.plan("1", cpus, allow)
    assert p.serve == {19} and p.roce is None and p.comm == {17, 18} and p.rest == allow - {17, 18, 19}
    f = cpupin.plan("fast", cpus, allow, roce=True)
    assert f.serve == f.comm == f.rest == frozenset([5, 6, 7, 8, 9, 15, 16, 17, 18, 19]) and f.roce is None
    e = cpupin.plan("serve=9;rest=0-8,10-14;nice=5", cpus, allow, roce=True)
    assert e.serve == {9} and e.rest == frozenset(range(9)) | frozenset(range(10, 15)) and e.nice == 5
    assert e.roce == 18 and e.comm == {16, 17}                    # unlisted roles from auto
    assert cpupin.plan("roce=none", cpus, allow, roce=True).roce is None
    assert "serve 19 roce 18 comm 16-17 rest 0-15" in cpupin.plan("auto", cpus, allow, roce=True).describe()
    small = cpupin.plan("auto", cpus, frozenset([5, 6, 0, 1]), roce=True)    # too few cores to dedicate them all
    assert small.serve == {6} and small.roce is None and small.rest == {0, 1, 5} and small.comm == small.rest
    for bad in ("serve=25", "serve=", "sever=3", "nice=40", "serve=5-3"):
        with pytest.raises(ValueError):
            cpupin.plan(bad, cpus, allow)
    assert cpupin.cpulist([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"
    assert cpupin.parse_cpulist("0-2, 5,7-8") == {0, 1, 2, 5, 7, 8}


@linux
def test_serving_sweep_on_this_host(monkeypatch):
    """The real affinity calls on this machine's threads (restored afterwards)."""

    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 4:
        pytest.skip("needs 4 cpus")
    before = {t: os.sched_getaffinity(t) for t in map(int, os.listdir("/proc/self/task"))}
    serve, roce, comm = allowed[-1], allowed[-2], allowed[-3]
    rest = frozenset(allowed[:-3])
    p = cpupin.Plan("roles", frozenset([serve]), frozenset([comm]), rest, roce)
    monkeypatch.setattr(cpupin, "_S", cpupin._State(plan=p))
    out: dict[str, object] = {}
    go = threading.Event()
    stop = threading.Event()

    def child():
        out["child_tid"] = threading.get_native_id()
        out["child_before"] = os.sched_getaffinity(0)
        go.set()
        stop.wait(10)

    def proxy():                                   # a RoCE-like proxy pinned to its cpu by itself
        os.sched_setaffinity(0, {roce})
        out["proxy_tid"] = threading.get_native_id()
        stop.wait(10)

    def nccl():
        cpupin.set_thread_name("NCCL Progress 0")
        out["nccl_tid"] = threading.get_native_id()
        stop.wait(10)

    def loop():
        cpupin.serving()
        out["serve_tid"] = threading.get_native_id()
        out["serve_aff"] = os.sched_getaffinity(0)
        th = threading.Thread(target=child, daemon=True)
        th.start()
        go.wait(10)
        stop.wait(10)

    ths = [threading.Thread(target=f, daemon=True) for f in (proxy, nccl)]
    for th in ths:
        th.start()
    time.sleep(0.2)
    lt = threading.Thread(target=loop, daemon=True)
    lt.start()
    go.wait(10)
    try:
        assert out["serve_aff"] == {serve}
        assert out["child_before"] == {serve}                     # inherited from the round loop's thread
        cpupin.sweep()
        assert os.sched_getaffinity(out["child_tid"]) == rest     # moved by the sweep
        assert os.sched_getaffinity(out["serve_tid"]) == {serve}
        assert os.sched_getaffinity(out["proxy_tid"]) == {roce}   # left alone
        assert os.sched_getaffinity(out["nccl_tid"]) == {comm}
        assert os.sched_getaffinity(threading.get_native_id()) == rest
        assert cpupin._comm(out["serve_tid"]) == "tf-serve"
    finally:
        stop.set()
        for th in ths + [lt]:
            th.join(10)
        for t, aff in before.items():
            try:
                os.sched_setaffinity(t, aff)
            except OSError:
                pass


@linux
def test_early_off_and_on(monkeypatch):
    monkeypatch.setattr(cpupin, "_S", cpupin._State())
    monkeypatch.delenv("GLM53_TF_CPU_PIN", raising=False)
    assert cpupin.early(0) is None
    cpupin.serving()                                              # no plan: nothing happens
    cpupin.tick()
    with cpupin.serving_request():
        pass
    before = os.sched_getaffinity(0)
    monkeypatch.setenv("GLM53_TF_CPU_PIN", "fast")
    monkeypatch.delenv("NCCL_SET_THREAD_NAME", raising=False)
    monkeypatch.setenv("GLM53_TF_COMM_BACKEND", "nccl")
    try:
        p = cpupin.early(0)
        assert p is not None and p.mode == "fast" and os.environ.get("NCCL_SET_THREAD_NAME") == "1"
        assert os.sched_getaffinity(0) == set(p.rest) <= before
        with cpupin.serving_request():                           # fast: nothing dedicated
            assert os.sched_getaffinity(0) == set(p.rest)
    finally:
        os.sched_setaffinity(0, before)
