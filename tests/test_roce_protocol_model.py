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


class Model:
    """Two (or more) ranks in one box. ``flag_qp_skew``: control -- the flag of HCA h is posted on HCA h + 1's queue
    pair (not ordered after its payload). ``lag_slots``: the proxy's catch-up bound (0230: SLOTS)."""

    def __init__(self, world=2, n_hca=2, seed=None, flag_qp_skew=False, lag_slots=SLOTS, proxy_delay=0.0):
        self.ranks = [Rank(r, world, n_hca) for r in range(world)]
        self.world, self.n_hca = world, n_hca
        self.nic: dict[tuple[int, int, int], deque] = {}     # (src, dst, hca) -> RC queue pair's work requests
        self.rng = random.Random(seed) if seed is not None else None
        self.flag_qp_skew = flag_qp_skew
        self.lag_slots = lag_slots
        self.proxy_delay = proxy_delay
        self.log: list[str] = []

    # host side
    def launch(self, r: int, payload: int) -> None:
        self.ranks[r].queue.append(Kernel(r, payload))

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
                    if src == rk.r and wr[0] == "data" and wr[2] == slot and wr[3] != k.seq:
                        raise Violation(f"rank {rk.r} restaged send slot {slot} for seq {k.seq} while the NIC still "
                                        f"owes seq {wr[3]} from it")
            rk.send[slot] = (rk.r, k.seq, k.payload)
            rk.doorbell = k.seq
            k.state = "wait"
            return True
        if k.state == "wait":
            slot = k.seq % SLOTS
            for p in range(self.world):
                if p == rk.r:
                    continue
                for h in range(self.n_hca):
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
            for p in range(self.world):
                if p == rk.r:
                    continue
                for h in range(self.n_hca):
                    self.nic.setdefault((rk.r, p, h), deque()).append(("data", p, s % SLOTS, s, h))
                    hq = (h + 1) % self.n_hca if self.flag_qp_skew else h
                    self.nic.setdefault((rk.r, p, hq), deque()).append(("flag", p, s % SLOTS, s, h))
            rk.proxy_last = s
        return True

    def _nic_step(self, key) -> bool:
        q = self.nic.get(key)
        if not q:
            return False
        kind, dst, slot, seq, h = q.popleft()
        src = key[0]
        d = self.ranks[dst]
        if kind == "data":
            staged = self.ranks[src].send[slot]              # read from the sender's slot when transmitted
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
            miss = [(p, h) for p in range(self.world) if p != rk.r for h in range(self.n_hca)
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
