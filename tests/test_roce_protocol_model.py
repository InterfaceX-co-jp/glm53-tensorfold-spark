"""patches/0350: an executable model of patches/0230's RoCE all-gather protocol (kernel, doorbell, proxy, RC queue
pairs striped over the HCAs, per-HCA sequence flags, two slots) and of ``tests/cuda/bench_roce.py``'s single-node
``loopback`` harness. Pure Python, no torch, no GPU, no RDMA.

What it shows:

1. The W2 loopback failure (``results/W2/roce-loopback.log``: rank 0 timed out at sequence 312 on the first HCA,
   "doorbell 312, completed 311, this rank's proxy posted up to 312", the flag in host memory at the check) is the
   HARNESS, not the memory model: the old ``loopback`` warms each rank's CUDA graph with ``2 x ops`` eager
   collectives of THAT RANK ALONE (``_graph`` then ``synchronize``), in one process that drives both ranks. Rank 0's
   first warm-up collective (1 + (10 + iters) + 1 = 312 at the default ``--iters 300``) needs rank 1's, which the
   host only launches after ``synchronize`` returns: rank 0 waits out its timeout, poisons; rank 1's warm-up then
   runs sequence 312 (rank 0 had rung its doorbell, so rank 0's proxy had sent it) and the flag lands in rank 0's host
   memory AFTER rank 0 gave up; ``check`` reads it there and 0230's diagnosis calls that "delivered but not seen".
   The model reproduces every number of the log line, and predicts ``--iters N`` -> sequence N + 12 on either HCA
   count (the GPU re-run is the check).
2. With the fixed harness (both ranks' warm-ups interleaved, graphs captured without executing, replays of both
   ranks enqueued together) nothing times out.
3. 0230's protocol keeps its invariants under random schedules (kernels, proxies and NIC queues interleaved at
   random, proxies lagging, up to two HCAs): a send slot is never restaged before the NIC has read it for the peer
   that needs it, a receive slot is never read before its payload of that sequence landed, the doorbell is never
   more than two sequences ahead of the proxy. Controls: a flag posted on another queue pair than its payload, or a
   third slot of lag, are caught.
4. 0350's failure watch classifies the W2 failure as "arrived after the timeout", not "delivered but not seen".
5. patches/0460: shards below ``GLM53_TF_ROCE_STRIPE_KB`` travel on ONE HCA (seq % HCAs; the kernel waits for that
   HCA's flag only) and stripes up to ``GLM53_TF_ROCE_INLINE`` bytes are posted inline (the proxy copies the staged
   bytes at post time; the NIC never reads the send slot for them). Random schedules with mixed sizes keep every
   invariant, and the payload a proxy copies is always the op's own. Controls: a kernel that waits for another HCA
   than the sender used, or ranks with different thresholds, time out (hence the setting is agreed at load).

Run: pytest -q tests/test_roce_protocol_model.py  (no dependencies; ~2 s)
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

import pytest

SLOTS = 2


class Violation(AssertionError):
    pass


# -- the model -------------------------------------------------------------------------------------------------------
@dataclass
class Kernel:
    """One all-gather launch on one rank (roce.cu's gather_kernel as a state machine)."""

    rank: int
    payload: int
    state: str = "start"          # start -> wait -> done
    seq: int = 0
    nbytes: int = 16384           # patches/0460: the padded shard size (decides striped / one HCA)


@dataclass
class Rank:
    r: int
    world: int
    n_hca: int
    epoch: int = 0                # completed sequence (device)
    doorbell: int = 0             # ctrl[CTRL_SEQ]
    completed: int = 0            # ctrl[CTRL_COMPLETED]
    poison: bool = False
    failed: bool = False
    err: dict | None = None
    proxy_last: int = 0
    send: list = field(default_factory=lambda: [None] * SLOTS)       # (rank, seq, payload) staged in send[slot]
    recv: dict = field(default_factory=dict)                         # (src, slot) -> (src, seq, payload)
    flag: dict = field(default_factory=dict)                         # (src, slot, hca) -> seq
    queue: deque = field(default_factory=deque)                      # launched kernels, in stream order
    watch: dict | None = None                                        # 0350: the proxy's view at the failure
    slot_nbytes: list = field(default_factory=lambda: [0] * SLOTS)   # 0460: ctrl[CTRL_SLOT_NBYTES + slot]
    stripe_min: int = 0                                              # 0460: this rank's GLM53_TF_ROCE_STRIPE_KB


class Model:
    """Two (or more) ranks in one box. ``flag_qp_skew``: control -- the flag of HCA h is posted on HCA h + 1's queue
    pair (not ordered after its payload). ``lag_slots``: the proxy's catch-up bound (0230: SLOTS)."""

    def __init__(self, world=2, n_hca=2, seed=None, flag_qp_skew=False, lag_slots=SLOTS, proxy_delay=0.0,
                 stripe_min=0, inline_max=0, wait_rule_skew=False, stripe_mins=None):
        self.ranks = [Rank(r, world, n_hca) for r in range(world)]
        # patches/0460: the one-HCA threshold (per rank for the mismatch control), inline posts, and a control whose
        # kernel waits for another HCA than the one the sender used
        for rk in self.ranks:
            rk.stripe_min = stripe_mins[rk.r] if stripe_mins is not None else stripe_min
        self.inline_max = inline_max
        self.wait_rule_skew = wait_rule_skew
        self.world, self.n_hca = world, n_hca
        self.nic: dict[tuple[int, int, int], deque] = {}     # (src, dst, hca) -> RC queue pair's work requests
        self.rng = random.Random(seed) if seed is not None else None
        self.flag_qp_skew = flag_qp_skew
        self.lag_slots = lag_slots
        self.proxy_delay = proxy_delay
        self.log: list[str] = []

    # host side
    def launch(self, r: int, payload: int, nbytes: int = 16384) -> None:
        self.ranks[r].queue.append(Kernel(r, payload, nbytes=nbytes))

    def hcas(self, rk: Rank, seq: int, nbytes: int, waiting: bool = False) -> list[int]:
        """patches/0460 (roce_common.h ``one_hca`` / ``single_hca``): the HCAs an op uses."""

        if rk.stripe_min and nbytes < rk.stripe_min and self.n_hca > 1:
            h = seq % self.n_hca
            if waiting and self.wait_rule_skew:
                h = (seq + 1) % self.n_hca
            return [h]
        return list(range(self.n_hca))

    # one step of each actor; each returns True when it changed something
    def _kernel_step(self, rk: Rank) -> bool:
        if not rk.queue:
            return False
        k = rk.queue[0]
        if k.state == "start":
            if rk.poison:                         # a poisoned runtime: the launch does nothing
                rk.queue.popleft()
                return True
            k.seq = rk.epoch + 1
            slot = k.seq % SLOTS
            # the NIC must have read what it still owes a peer from this slot
            for (src, dst, h), q in self.nic.items():
                for wr in q:
                    if src == rk.r and wr[0] == "data" and wr[2] == slot and wr[3] != k.seq and len(wr) < 6:
                        raise Violation(f"rank {rk.r} restaged send slot {slot} for seq {k.seq} while the NIC still "
                                        f"owes seq {wr[3]} from it")
            rk.send[slot] = (rk.r, k.seq, k.payload)
            rk.slot_nbytes[slot] = k.nbytes
            rk.doorbell = k.seq
            k.state = "wait"
            return True
        if k.state == "wait":
            slot = k.seq % SLOTS
            for p in range(self.world):
                if p == rk.r:
                    continue
                for h in self.hcas(rk, k.seq, k.nbytes, waiting=True):
                    if rk.flag.get((p, slot, h), 0) != k.seq:
                        return False
            for p in range(self.world):
                if p == rk.r:
                    continue
                got = rk.recv.get((p, slot))
                if got is None or got[:2] != (p, k.seq):
                    raise Violation(f"rank {rk.r} seq {k.seq}: receive slot of rank {p} holds {got}")
            rk.epoch = rk.completed = k.seq
            rk.queue.popleft()
            return True
        return False

    def _proxy_step(self, rk: Rank) -> bool:
        if rk.doorbell == rk.proxy_last:
            return False
        pending = rk.doorbell - rk.proxy_last
        if pending > self.lag_slots:
            raise Violation(f"rank {rk.r}: doorbell {rk.doorbell} is {pending} ahead of the proxy")
        for s in range(rk.proxy_last + 1, rk.doorbell + 1):
            nbytes = rk.slot_nbytes[s % SLOTS]                  # the catch-up reads the slot's byte count
            used = self.hcas(rk, s, nbytes)
            for p in range(self.world):
                if p == rk.r:
                    continue
                for h in used:                              # per HCA: its stripe, then its flag (0230's order)
                    stripe = nbytes if len(used) == 1 else -(-nbytes // len(used))
                    if self.inline_max and stripe <= self.inline_max:
                        staged = rk.send[s % SLOTS]             # 0460 inline: the proxy copies the slot NOW
                        if staged is None or staged[1] != s:
                            raise Violation(f"rank {rk.r} proxy copied slot {s % SLOTS} for seq {s} inline but it "
                                            f"holds {staged}")
                        self.nic.setdefault((rk.r, p, h), deque()).append(("data", p, s % SLOTS, s, h, staged))
                    else:
                        self.nic.setdefault((rk.r, p, h), deque()).append(("data", p, s % SLOTS, s, h))
                    hq = (h + 1) % self.n_hca if self.flag_qp_skew else h
                    self.nic.setdefault((rk.r, p, hq), deque()).append(("flag", p, s % SLOTS, s, h))
            rk.proxy_last = s
        return True

    def _nic_step(self, key) -> bool:
        q = self.nic.get(key)
        if not q:
            return False
        wr = q.popleft()
        kind, dst, slot, seq, h = wr[:5]
        src = key[0]
        d = self.ranks[dst]
        if kind == "data":
            # read from the sender's slot when transmitted; 0460 inline: the copy the proxy made at post time
            staged = wr[5] if len(wr) > 5 else self.ranks[src].send[slot]
            if staged is None or staged[1] != seq:
                raise Violation(f"NIC of rank {src} sent slot {slot} for seq {seq} but it holds {staged}")
            d.recv[(src, slot)] = staged
        else:
            d.flag[(src, slot, h)] = seq
        return True

    def _watch_step(self, rk: Rank) -> None:
        """0350's failure watch (roce_watch.h), run by the proxy thread: the host's view right when it notices the
        failure word, then whether the flag arrives later."""

        if rk.failed and rk.watch is None:
            e = rk.err
            v = rk.flag.get((e["peer"], e["seq"] % SLOTS, e["hca"]), 0)
            rk.watch = {"host_at_fail": v, "late": False}
        elif rk.watch is not None and not rk.watch["late"] and rk.watch["host_at_fail"] != rk.err["seq"]:
            e = rk.err
            if rk.flag.get((e["peer"], e["seq"] % SLOTS, e["hca"]), 0) == e["seq"]:
                rk.watch["late"] = True

    def _actors(self):
        acts = []
        for rk in self.ranks:
            acts.append(("gpu", lambda rk=rk: self._kernel_step(rk)))
            acts.append(("proxy", lambda rk=rk: self._proxy_step(rk)))
        for key in list(self.nic):
            acts.append(("nic", lambda key=key: self._nic_step(key)))
        return acts

    def run(self) -> None:
        """Run until nothing can move (the GPU work launched so far, the proxies, the NIC)."""

        while True:
            acts = self._actors()
            if self.rng is not None:
                self.rng.shuffle(acts)
                if self.proxy_delay and self.rng.random() < self.proxy_delay:      # proxies late: GPU/NIC first
                    acts.sort(key=lambda a: a[0] == "proxy")
            moved = False
            for _, a in acts:
                if a():
                    moved = True
                    for rk in self.ranks:
                        self._watch_step(rk)
                    if self.rng is not None:
                        break
            if not moved:
                return

    def synchronize(self) -> None:
        """``torch.cuda.synchronize()``: returns when every launched kernel ended. A kernel whose flags never come
        times out (in the model: when nothing else can move), records the failure and poisons its rank."""

        while True:
            self.run()
            waiting = [rk for rk in self.ranks if rk.queue]
            if not waiting:
                return
            rk = waiting[0]
            k = rk.queue[0]
            slot = k.seq % SLOTS
            miss = [(p, h) for p in range(self.world) if p != rk.r
                    for h in self.hcas(rk, k.seq, k.nbytes, waiting=True)
                    if rk.flag.get((p, slot, h), 0) != k.seq]
            p, h = miss[0]
            if not rk.failed:
                rk.err = {"seq": k.seq, "peer": p, "hca": h, "gpu_seen": rk.flag.get((p, slot, h), 0)}
                rk.failed = True
            rk.poison = True
            rk.queue.popleft()                    # the timed-out kernel ends (no copy, no epoch)
            for x in self.ranks:
                self._watch_step(x)

    def check(self, r: int) -> dict | None:
        """``Runtime.check``'s view: the failure record and the host flag word NOW (0230's diagnosis)."""

        rk = self.ranks[r]
        if not rk.failed:
            return None
        e = rk.err
        now = rk.flag.get((e["peer"], e["seq"] % SLOTS, e["hca"]), 0)
        return {"seq": e["seq"], "hca": e["hca"], "doorbell": rk.doorbell, "completed": rk.completed,
                "posted": rk.proxy_last, "flag_now": now, "watch": dict(rk.watch or {}), "gpu_seen": e["gpu_seen"]}


# -- the loopback harness (bench_roce.py), old and fixed -------------------------------------------------------------
def old_loopback(m: Model, iters: int = 300, ops: int = 90, reps: int = 50) -> None:
    """``loopback`` as of 0230, first size only (it failed there): one collective of both ranks, the eager timing
    (10 warm-up + iters), then ``_graph`` for rank 0 (2 x ops eager warm-ups of rank 0 ALONE, synchronize, capture),
    ``_graph`` for rank 1, three and ``reps`` replays of both graphs, checks."""

    def both():
        for r in (0, 1):
            m.launch(r, 0)

    both()
    m.synchronize()
    for _ in range(10 + iters):
        both()
    m.synchronize()
    graphs = {}
    for r in (0, 1):
        for _ in range(2 * ops):                   # warm-up on the side stream: rank r only
            m.launch(r, 0)
        m.synchronize()
        graphs[r] = ops                           # capture: recorded, not executed
    for _ in range(3 + reps):
        for r in (0, 1):
            for _ in range(graphs[r]):
                m.launch(r, 0)
    m.synchronize()


def new_loopback(m: Model, iters: int = 300, ops: int = 90, reps: int = 50) -> None:
    """0350's ``loopback``: the warm-ups of both ranks interleaved op by op (``_graph_pair``), captures that execute
    nothing, replays of both ranks' graphs enqueued together."""

    def both():
        for r in (0, 1):
            m.launch(r, 0)

    both()
    m.synchronize()
    for _ in range(10 + iters):
        both()
    m.synchronize()
    for _ in range(2 * ops):
        both()
    m.synchronize()
    for _ in range(3 + reps):
        for r in (0, 1):
            for _ in range(ops):
                m.launch(r, 0)
    m.synchronize()


# -- 1. the W2 failure, reproduced -----------------------------------------------------------------------------------
@pytest.mark.parametrize("n_hca", [1, 2])
@pytest.mark.parametrize("iters", [300, 100, 7])
def test_old_loopback_reproduces_w2(iters, n_hca):
    m = Model(n_hca=n_hca)
    old_loopback(m, iters=iters)
    got = m.check(0)
    assert got is not None
    seq = iters + 12                              # 1 + (10 + iters) + the first warm-up of rank 0's graph
    # results/W2/roce-loopback.log: "at sequence 312 ... the flag HAS reached this host's memory ...; doorbell 312,
    # completed 311, this rank's proxy posted up to 312"
    assert (got["seq"], got["doorbell"], got["completed"], got["posted"], got["flag_now"]) == \
        (seq, seq, seq - 1, seq, seq)
    # ... but the flag was NOT there when rank 0 timed out: rank 1 sent it during its own warm-up, afterwards
    assert got["gpu_seen"] != seq and got["watch"]["host_at_fail"] != seq and got["watch"]["late"]
    r1 = m.check(1)                               # rank 1 then waits for 313, which poisoned rank 0 never sends
    assert r1 is not None and r1["seq"] == seq + 1 and r1["flag_now"] != seq + 1


def test_new_loopback_completes():
    for n_hca in (1, 2):
        m = Model(n_hca=n_hca)
        new_loopback(m)
        assert m.check(0) is None and m.check(1) is None
        assert m.ranks[0].completed == m.ranks[1].completed == 1 + 310 + 180 + 53 * 90


# -- 3. the protocol's invariants under random schedules -------------------------------------------------------------
def _random_program(m: Model, rng: random.Random, n_ops: int) -> None:
    """Both ranks launch the same number of collectives, in bursts the host enqueues in random order, with
    synchronizes in between only where every rank has launched the same count (as any correct program does)."""

    done = [0, 0]
    while min(done) < n_ops:
        burst = rng.randint(1, 12)
        order = [0, 1] if rng.random() < 0.5 else [1, 0]
        for r in order:
            for _ in range(min(burst, n_ops - done[r])):
                m.launch(r, rng.randint(0, 1 << 30))
                done[r] += 1
            if rng.random() < 0.5:
                m.run()                           # the GPU moves while the host is still enqueueing
        if done[0] == done[1] and rng.random() < 0.3:
            m.synchronize()
    m.synchronize()


@pytest.mark.parametrize("n_hca", [1, 2])
@pytest.mark.parametrize("seed", range(40))
def test_random_schedules_keep_the_invariants(seed, n_hca):
    rng = random.Random(1000 * n_hca + seed)
    m = Model(n_hca=n_hca, seed=seed, proxy_delay=0.3)
    _random_program(m, rng, 60)
    assert m.check(0) is None and m.check(1) is None
    assert m.ranks[0].completed == m.ranks[1].completed == 60


def test_three_ranks():
    for seed in range(10):
        m = Model(world=3, n_hca=2, seed=seed, proxy_delay=0.3)
        for i in range(40):
            for r in random.Random(seed + i).sample(range(3), 3):
                m.launch(r, i)
        m.synchronize()
        assert all(m.check(r) is None for r in range(3))


def test_control_flag_on_another_queue_pair_is_caught():
    """A flag not ordered after its payload (posted on the other HCA's queue pair) lets a kernel read a receive slot
    before its payload landed: the model finds it."""

    caught = 0
    for seed in range(60):
        m = Model(n_hca=2, seed=seed, flag_qp_skew=True)
        try:
            _random_program(m, random.Random(seed), 30)
        except Violation:
            caught += 1
    assert caught > 0


def test_control_proxy_lag_bound_is_reached():
    """0230's bound (a doorbell at most SLOTS ahead of the proxy) is tight: with a bound of one, random schedules
    exceed it."""

    caught = 0
    for seed in range(60):
        m = Model(n_hca=2, seed=seed, lag_slots=1, proxy_delay=0.9)
        try:
            _random_program(m, random.Random(seed), 30)
        except Violation as exc:
            assert "ahead of the proxy" in str(exc)
            caught += 1
    assert caught > 0


# -- 5. patches/0460: one-HCA small ops and inline posts -------------------------------------------------------------
SIZES = [16, 112, 1024, 16384, 32768, 65536, 131072]


def _mixed_program(m: Model, rng: random.Random, n_ops: int) -> None:
    """``_random_program`` with a size per op (the same on every rank: an all-gather's shards are equal)."""

    sizes = [rng.choice(SIZES) for _ in range(n_ops)]
    done = [0] * m.world
    while min(done) < n_ops:
        burst = rng.randint(1, 12)
        for r in rng.sample(range(m.world), m.world):
            for _ in range(min(burst, n_ops - done[r])):
                m.launch(r, rng.randint(0, 1 << 30), sizes[done[r]])
                done[r] += 1
            if rng.random() < 0.5:
                m.run()
        if len(set(done)) == 1 and rng.random() < 0.3:
            m.synchronize()
    m.synchronize()


@pytest.mark.parametrize("inline_max", [0, 1024])
@pytest.mark.parametrize("stripe_min", [0, 32768, 1 << 20])
@pytest.mark.parametrize("seed", range(12))
def test_one_hca_and_inline_keep_the_invariants(seed, stripe_min, inline_max):
    rng = random.Random(7000 + seed)
    m = Model(n_hca=2, seed=seed, proxy_delay=0.3, stripe_min=stripe_min, inline_max=inline_max)
    _mixed_program(m, rng, 60)
    assert m.check(0) is None and m.check(1) is None
    assert m.ranks[0].completed == m.ranks[1].completed == 60


def test_one_hca_three_ranks():
    for seed in range(6):
        m = Model(world=3, n_hca=2, seed=seed, proxy_delay=0.3, stripe_min=32768, inline_max=512)
        _mixed_program(m, random.Random(seed), 40)
        assert all(m.check(r) is None for r in range(3))


def test_one_hca_ops_use_one_queue_pair():
    """A small op posts its payload and flag on HCA seq % 2 only; a large one on both (the proxy's rule)."""

    m = Model(n_hca=2, stripe_min=32768)
    for size, used in ((16384, 1), (65536, 2), (16, 1), (32768, 2)):
        for r in (0, 1):
            m.launch(r, 0, size)
        m.run()
        assert m.check(0) is None and m.check(1) is None
        seq = m.ranks[0].completed
        hcas = {h for (src, slot, h), v in m.ranks[1].flag.items() if src == 0 and v == seq}
        assert len(hcas) == used and (used == 2 or hcas == {seq % 2})


def test_control_wait_rule_mismatch_times_out():
    """A kernel waiting for another HCA's flag than the sender set (a rule not shared by kernel and proxy) never
    completes: the model's timeout fires on the first small op."""

    m = Model(n_hca=2, seed=1, stripe_min=32768, wait_rule_skew=True)
    for r in (0, 1):
        m.launch(r, 0, 16384)
    m.synchronize()
    assert m.check(0) is not None and m.check(0)["seq"] == 1


def test_control_threshold_mismatch_times_out():
    """Ranks started with different thresholds disagree on the HCA of a small op: a timeout (why
    ``Settings.agreed`` carries GLM53_TF_ROCE_STRIPE_KB)."""

    hung = 0
    for seq_first in range(4):
        m = Model(n_hca=2, stripe_mins=[32768, 0])
        for _ in range(seq_first):
            for r in (0, 1):
                m.launch(r, 0, 65536)                 # striped on both ranks: fine
        for r in (0, 1):
            m.launch(r, 0, 16384)                     # rank 0: one HCA; rank 1: striped
        m.synchronize()
        hung += int(m.check(0) is not None or m.check(1) is not None)
    assert hung == 4
