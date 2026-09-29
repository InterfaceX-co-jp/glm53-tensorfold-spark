"""NCCL vs the one-shot RoCE all-gather (patches/0230) between the two Sparks: latency (eager and CUDA graph), bits,
the fail-stop path, and a soak. Standalone: needs only the patched ``tensorfold`` (the image, or a patched tree on
PYTHONPATH), one GPU a node, /dev/infiniband.

Both nodes, same arguments but ``--rank``:

  python tests/cuda/bench_roce.py --rank 0 --master $HEAD_IP            # head node
  python tests/cuda/bench_roce.py --rank 1 --master $HEAD_IP            # worker node

Modes (default ``bench``):
  bench     for each size (bytes ONE rank sends; default 16k,64k,128k,1m): the RoCE result equals NCCL's and the
            expected bytes; eager latency (a launch + wait, what an eager sampler exchange costs); graph latency
            (``--ops`` all-gathers captured in one graph, like a decode step's 90, replayed; per op); and a
            decode-like graph (each all-gather followed by a small kernel that reads its output)
  fault     rank 1 skips one collective: rank 0 must raise the diagnosis within ~``--timeout`` s (both ranks
            print what they saw)
  soak      the decode-like graph at 16 KiB (and 128 KiB) for ``--minutes``, bits checked every replay; reports
            the health snapshot every minute (b12x #313 / sparkring #278 signatures)
  loopback  ONE node, one process: two runtimes (rank 0 and 1) connected through the same HCA(s) by the NIC's
            loopback; bits and latency, no second node needed (a first smoke test of the proxy and queue pairs).
            patches/0350: one process drives BOTH ranks, so every collective of one rank must be launched together
            with the other's (a rank's collective completes only when its peer's of the same sequence runs). 0230's
            version warmed each rank's CUDA graph with that rank's collectives alone and synchronized: rank 0 waited
            for a rank 1 the host had not launched yet and timed out at sequence iters + 12 (312: results/W2), and
            its diagnosis mistook the flag rank 1 sent later for "delivered but not seen"
            (tests/test_roce_protocol_model.py). The warm-ups are now interleaved (``_graph_pair``)
  stress    patches/0350: ``--iters`` all-gathers (default 100,000 a size) whose payload changes EVERY op (x += 1 on
            every rank; a stale receive slot or a flag overtaking its payload would show as a mismatch), eager and
            as replayed graphs of ``--ops``, sizes 16k and 128k, mismatches counted on the device and checked every
            ``--check`` ops; with ``--loop`` on ONE node (both ranks in this process, over the NIC loopback), else
            two nodes like ``bench``. Run it per CX7 function (GLM53_TF_ROCE_HCA=<name>) and striped over both

Knobs: GLM53_TF_ROCE_* as in the engine (GLM53_TF_ROCE_HCAS=1 to compare with one CX7 function; the largest size
sets GLM53_TF_ROCE_MAX_KB unless it is set). NCCL: NCCL_SOCKET_IFNAME / NCCL_IB_HCA / NCCL_PROTO as for serving.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import torch


def _size(tok: str) -> int:
    tok = tok.strip().lower()
    mult = {"k": 1024, "m": 1024 * 1024}.get(tok[-1:], 1)
    return int(float(tok[:-1] if tok[-1:] in "km" else tok) * mult)


def _pattern(rank: int, n: int, salt: int) -> torch.Tensor:
    from tensorfold.families.glm5_next.cuda.roce import pattern

    return pattern(rank, n, salt)


def _expected(world: int, n: int, salt: int) -> torch.Tensor:
    return torch.cat([_pattern(r, n, salt) for r in range(world)])


def _time_eager(fn, iters: int) -> float:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e6


def _graph(fn, ops: int, tail=None) -> torch.cuda.CUDAGraph:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            for _ in range(ops):
                fn()
                if tail is not None:
                    tail()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(ops):
            fn()
            if tail is not None:
                tail()
    return g


def _graph_pair(fns, ops: int, tails=None) -> list[torch.cuda.CUDAGraph]:
    """patches/0350: CUDA graphs of ``ops`` collectives for each of several runtimes driven by ONE process (the NIC
    loopback): the eager warm-ups of every rank interleaved op by op, each rank on its own side stream (a rank's
    collective only completes once its peers' of the same sequence run: warming one rank alone waits out the timeout,
    0230's W2 failure), then each rank's graph captured (a capture executes nothing, so one at a time is fine)."""

    tails = tails or [None] * len(fns)
    streams = [torch.cuda.Stream() for _ in fns]
    for st in streams:
        st.wait_stream(torch.cuda.current_stream())
    for _ in range(2):
        for _ in range(ops):
            for fn, tail, st in zip(fns, tails, streams):
                with torch.cuda.stream(st):
                    fn()
                    if tail is not None:
                        tail()
    for st in streams:
        torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    graphs = []
    for fn, tail in zip(fns, tails):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(ops):
                fn()
                if tail is not None:
                    tail()
        graphs.append(g)
    return graphs


def _time_graph(g: torch.cuda.CUDAGraph, reps: int, ops: int) -> float:
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / (reps * ops)


def _setup(args):
    from tensorfold.families.glm5_next.cuda import roce
    from tensorfold.families.glm5_next.cuda.comm import NCCL

    torch.cuda.set_device(0)
    base = NCCL(args.rank, 2, args.master, args.port)
    base.barrier()
    s = roce.settings()
    if not os.environ.get("GLM53_TF_ROCE_MAX_KB"):
        s = dataclasses.replace(s, max_bytes=max(args.sizes))
    if args.timeout is not None:
        s = dataclasses.replace(s, timeout_s=args.timeout)
    comm = roce.connect(base, s)            # detects + pairs the HCAs, connects, probes against NCCL (raises)
    rt = comm.rt
    if args.rank == 0:
        print(json.dumps({"hcas": [dataclasses.asdict(h) for h in rt.hcas], "slot_bytes": rt.slot_bytes,
                          "timeout_s": s.timeout_s, "blocks": s.blocks, "threads": s.threads,
                          "nccl_proto": os.environ.get("NCCL_PROTO", "")}), flush=True)
    return roce, base, rt


def bench(args) -> None:
    roce, base, rt = _setup(args)
    rows = []
    for salt, n in enumerate(args.sizes):
        if n > rt.s.max_bytes:
            if args.rank == 0:
                print(f"{n} B: over GLM53_TF_ROCE_MAX_KB, skipped", flush=True)
            continue
        x = _pattern(args.rank, n, salt)
        ref, mine = torch.empty(2 * n, dtype=torch.uint8, device="cuda"), torch.empty(2 * n, dtype=torch.uint8,
                                                                                          device="cuda")
        base.all_gather(x, ref)
        rt.gather(x, mine)
        torch.cuda.synchronize()
        rt.check()
        exp = _expected(2, n, salt)
        same = bool(torch.equal(mine, ref)) and bool(torch.equal(ref, exp))
        xf = x.view(torch.float32) if n % 4 == 0 else x
        yf = torch.empty(2 * xf.numel(), dtype=xf.dtype, device="cuda")
        z = torch.zeros(xf.numel(), dtype=xf.dtype, device="cuda") if xf.dtype == torch.float32 else None
        res = {"bytes": n, "bits_equal": same}
        for name, fn in (("nccl", lambda: base.all_gather(xf, yf)), ("roce", lambda: rt.gather(xf, yf))):
            base.barrier()
            res[f"{name}_eager_us"] = round(_time_eager(fn, args.iters), 2)
            base.barrier()
            g = _graph(fn, args.ops)
            base.barrier()
            res[f"{name}_graph_us"] = round(_time_graph(g, args.reps, args.ops), 2)
            if z is not None:
                tail = (lambda: z.add_(yf[:xf.numel()]))
                base.barrier()
                g2 = _graph(fn, args.ops, tail)
                base.barrier()
                res[f"{name}_step_us"] = round(_time_graph(g2, args.reps, args.ops), 2)
            del g
        rt.check()
        rows.append(res)
        if args.rank == 0:
            print(json.dumps(res), flush=True)
    if args.rank == 0:
        print("\n| bytes/rank | bits | NCCL eager | RoCE eager | NCCL graph | RoCE graph | NCCL step | RoCE step |")
        print("| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in rows:
            print(f"| {r['bytes']} | {'ok' if r['bits_equal'] else 'DIFF'} | {r['nccl_eager_us']} | "
                  f"{r['roce_eager_us']} | {r['nccl_graph_us']} | {r['roce_graph_us']} | "
                  f"{r.get('nccl_step_us', '-')} | {r.get('roce_step_us', '-')} |")
        per_step = [(r["bytes"], (r["nccl_graph_us"] - r["roce_graph_us"]) * 90 / 1e3) for r in rows]
        print("saved a 90-exchange step (graph): " + ", ".join(f"{b} B: {ms:.2f} ms" for b, ms in per_step))
        print("stats:", json.dumps(rt.snapshot()), flush=True)
    if not all(r["bits_equal"] for r in rows):
        sys.exit(1)


def fault(args) -> None:
    roce, base, rt = _setup(args)
    x = _pattern(args.rank, 16 * 1024, 1)
    y = torch.empty(2 * x.numel(), dtype=torch.uint8, device="cuda")
    for _ in range(100):
        rt.gather(x, y)
    torch.cuda.synchronize()
    rt.check()
    base.barrier()
    t = time.perf_counter()
    if args.rank == 0:
        rt.gather(x, y)                  # rank 1 never joins this one
    torch.cuda.synchronize()
    waited = time.perf_counter() - t
    try:
        rt.check()
        if args.rank == 0:
            print(f"rank 0: NO ERROR after {waited:.2f} s (expected a timeout)", flush=True)
            sys.exit(1)
    except roce.RoceError as exc:
        print(f"rank {args.rank}: raised after {waited:.2f} s: {exc}", flush=True)
    if args.rank == 1:                   # rank 1's next exchange: rank 0 (poisoned) never sends
        # W9: wait out rank 0's timeout first. Launched at once (0230), this exchange IS the sequence rank 0 waits
        # for (both ranks' next sequence), so rank 0's wait was satisfied in microseconds ("NO ERROR after 0.00 s")
        time.sleep(rt.s.timeout_s + 2.0)
        t = time.perf_counter()
        rt.gather(x, y)
        torch.cuda.synchronize()
        try:
            rt.check()
            print("rank 1: its next exchange completed (rank 0 had staged and sent that sequence before it timed "
                  "out; rank 0 is poisoned and sends nothing more)", flush=True)
        except roce.RoceError as exc:
            print(f"rank 1: raised after {time.perf_counter() - t:.2f} s: {exc}", flush=True)
    print(f"rank {args.rank} snapshot: {json.dumps(rt.snapshot())}", flush=True)
    base.barrier()


def soak(args) -> None:
    roce, base, rt = _setup(args)
    sets = []
    for salt, n in enumerate((16 * 1024, 128 * 1024)):
        if n > rt.s.max_bytes:
            continue
        x = _pattern(args.rank, n, salt)
        y = torch.empty(2 * n, dtype=torch.uint8, device="cuda")
        z = torch.zeros(2 * n, dtype=torch.uint8, device="cuda")
        g = _graph(lambda x=x, y=y: rt.gather(x, y), args.ops, lambda y=y, z=z: z.copy_(y))
        # W9: x and y stay referenced with their graph (the captured kernels read / write their addresses); 0230 kept
        # only (g, z, ...), so the 16 KiB set's x / y were freed at the next iteration and reused by the 128 KiB
        # tensors: "DIFFERENT BYTES at 16384 B after 0 replays" on both ranks (results/W9/roce2/soak*)
        sets.append((g, z, _expected(2, n, salt), n, x, y))
    end = time.time() + args.minutes * 60
    last, replays = time.time(), 0
    # W9: the ranks agree when to stop (rank 0's clock, over NCCL, every 256 rounds). Each rank used its own clock:
    # one rank started a replay the other never joined and timed out after 120 s (results/W9/roce2/soak3-r1.log)
    stop = torch.zeros(1, dtype=torch.int32, device="cuda")
    stops = torch.zeros(2, dtype=torch.int32, device="cuda")
    rounds = 0
    while True:
        if rounds % 256 == 0:
            stop.fill_(int(args.rank == 0 and time.time() >= end))
            base.all_gather(stop, stops)
            if int(stops.max()):
                break
        rounds += 1
        for g, z, exp, n, _x, _y in sets:
            z.zero_()
            g.replay()
            torch.cuda.synchronize()
            rt.check()
            if not torch.equal(z, exp):
                # W9: which bytes (per rank shard), and what the output buffer itself holds
                d = (z != exp).nonzero().flatten()
                print(f"rank {args.rank}: DIFFERENT BYTES at {n} B after {replays} replays: {d.numel()} bytes, "
                      f"shard0 {int((d < n).sum())} shard1 {int((d >= n).sum())}, first {int(d[0])} last {int(d[-1])}, "
                      f"z all zero {bool((z == 0).all())}, z head {z[d[:6]].tolist()} want {exp[d[:6]].tolist()}",
                      flush=True)
                sys.exit(1)
            replays += 1
        if time.time() - last > 60:
            last = time.time()
            snap = rt.snapshot()
            print(f"rank {args.rank} {time.strftime('%H:%M:%S')} replays {replays} ops {snap['completed']} "
                  f"posted {snap.get('ops_posted')} per_hca {snap.get('per_hca')}", flush=True)
    base.barrier()
    print(f"rank {args.rank}: soak ok, {replays} replays x {args.ops} ops", flush=True)


class _Stream:
    """patches/0350: one rank's side of the stress test: an int32 payload that changes every op, the expected gathered
    words, a device mismatch counter."""

    def __init__(self, rt, rank: int, world: int, n: int, salt: int):
        from tensorfold.families.glm5_next.cuda import roce

        self.rt, self.n = rt, n
        self.x = roce.pattern(rank, n, salt).view(torch.int32).clone()
        self.ref = torch.cat([roce.pattern(r, n, salt).view(torch.int32) for r in range(world)]).clone()
        self.y = torch.empty_like(self.ref)
        self.bad = torch.zeros((), dtype=torch.int64, device=self.x.device)
        self.stream = torch.cuda.Stream()
        # W9: the mismatch check's temporaries ((y != ref), its sum) are allocated here once on ``self.stream``, so
        # the caching allocator has them before the first op: a first-time device allocation inside an op can
        # synchronize the device, and in ``--loop`` (both ranks driven by ONE host thread) that blocked the host
        # behind rank 0's spinning gather before it launched rank 1's: a 10 s "late" timeout at sequence 1
        # (results/W9/roce/stress-both.log). Two nodes (one rank a process, like the engine) cannot deadlock so.
        torch.cuda.synchronize()
        with torch.cuda.stream(self.stream):
            for _ in range(2):
                self.bad.add_((self.y != self.ref).sum())
            self.bad.zero_()
        # the tensors above are written on the default stream and the ops run on ``self.stream`` (non-blocking):
        # without this wait the first op could read a half-written payload / reference (a false mismatch)
        torch.cuda.synchronize()

    def op(self) -> None:
        self.x.add_(1)                  # every rank's payload moves on each op: a stale slot cannot match
        self.rt.gather(self.x, self.y)
        self.ref.add_(1)
        self.bad.add_((self.y != self.ref).sum())


def _stress(rts: list, ranks: list[int], world: int, args, barrier=None) -> bool:
    """``--iters`` ops a size eager, then as graphs of ``--ops`` replayed; every rank in ``rts`` (loopback: both,
    two nodes: this one). Prints a JSON line per (size, mode); False on any mismatch or failure."""

    ok = True
    for salt, n in enumerate(args.sizes):
        n = n // 4 * 4
        if n == 0 or n > rts[0].s.max_bytes:
            continue
        ss = [_Stream(rt, r, world, n, 100 + salt) for rt, r in zip(rts, ranks)]
        for mode in ("eager", "graph"):
            if barrier is not None:
                barrier()
            graphs = _graph_pair([s.op for s in ss], args.ops) if mode == "graph" else None
            done, t0, bad, failed = 0, time.perf_counter(), 0, None
            per = args.ops if mode == "graph" else 1
            while done < args.iters and failed is None:
                chunk = min(args.check, args.iters - done)
                steps = max(1, chunk // per)
                for _ in range(steps):
                    for i, s in enumerate(ss):
                        with torch.cuda.stream(s.stream):
                            if graphs is not None:
                                graphs[i].replay()
                            else:
                                s.op()
                done += steps * per
                torch.cuda.synchronize()
                bad = sum(int(s.bad.item()) for s in ss)
                for rt in rts:
                    try:
                        rt.check()
                    except Exception as exc:  # noqa: BLE001 - reported below
                        failed = str(exc)
                if bad:
                    break
            dt = time.perf_counter() - t0
            snap = rts[0].snapshot()
            row = {"bytes": n, "mode": mode, "ops": done, "mismatched_words": bad, "failed": failed,
                   "us_per_op": round(dt / max(done, 1) * 1e6, 2), "completed": snap["completed"],
                   "per_hca": snap.get("per_hca")}
            print(json.dumps(row), flush=True)
            ok = ok and not bad and failed is None
            if not ok:
                return False
    return ok


def stress(args) -> None:
    if args.loop:
        from tensorfold.families.glm5_next.cuda import roce

        torch.cuda.set_device(0)
        s = roce.settings()
        if not os.environ.get("GLM53_TF_ROCE_MAX_KB"):
            s = dataclasses.replace(s, max_bytes=max(args.sizes))
        s = dataclasses.replace(s, timeout_s=args.timeout or 10.0)
        hcas = roce.detect(s.hca_spec)[:s.hcas]
        if not hcas:
            raise SystemExit("no ACTIVE RoCE port with an IPv4-mapped RoCE v2 GID")
        print("HCAs:", [(h.name, h.gid_index, h.ipv4) for h in hcas], flush=True)
        rts = [roce.Runtime(rank=r, world=2, hcas=hcas, s=s) for r in (0, 1)]
        blobs = [rt.blob() for rt in rts]
        for rt in rts:
            rt.connect(blobs)
        ok = _stress(rts, [0, 1], 2, args)
        print("rank 0 stats:", json.dumps(rts[0].snapshot()), flush=True)
        torch.cuda.synchronize()
        for rt in rts:
            rt.close()
    else:
        roce, base, rt = _setup(args)
        ok = _stress([rt], [args.rank], 2, args, barrier=base.barrier)
        base.barrier()
        print(f"rank {args.rank} stats:", json.dumps(rt.snapshot()), flush=True)
    print("stress", "ok" if ok else "FAILED", flush=True)
    if not ok:
        sys.exit(1)


def loopback(args) -> None:
    """Two runtimes in one process on the same HCA(s): the NIC loops rank 0's writes back to rank 1's region."""

    from tensorfold.families.glm5_next.cuda import roce

    torch.cuda.set_device(0)
    s = roce.settings()
    s = dataclasses.replace(s, max_bytes=max(args.sizes), timeout_s=args.timeout or 10.0)
    hcas = roce.detect(s.hca_spec)[:s.hcas]
    if not hcas:
        raise SystemExit("no ACTIVE RoCE port with an IPv4-mapped RoCE v2 GID")
    print("HCAs:", [(h.name, h.gid_index, h.ipv4) for h in hcas], flush=True)
    rts = [roce.Runtime(rank=r, world=2, hcas=hcas, s=s) for r in (0, 1)]
    blobs = [rt.blob() for rt in rts]
    for rt in rts:
        rt.connect(blobs)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for salt, n in enumerate(args.sizes):
        xs = [_pattern(r, n, salt) for r in (0, 1)]
        ys = [torch.empty(2 * n, dtype=torch.uint8, device="cuda") for _ in (0, 1)]
        # W9: ``pattern`` runs on the default stream, the gathers on the side streams (non-blocking, no implicit
        # wait): without this the first gather could stage a half-written input. That was W7's 1 MiB
        # "bits_equal: false" (results/W9/roce-sizes*.log): the larger the input, the longer its pattern kernels
        torch.cuda.synchronize()

        def both():
            for r in (0, 1):
                with torch.cuda.stream(streams[r]):
                    rts[r].gather(xs[r], ys[r])

        both()
        torch.cuda.synchronize()
        for rt in rts:
            rt.check()
        exp = _expected(2, n, salt)
        ok = all(torch.equal(y, exp) for y in ys)
        us = _time_eager(both, args.iters)
        # patches/0350: both ranks' warm-ups together (0230 warmed rank 0 alone here and timed out at iters + 12)
        gs = _graph_pair([lambda r=r: rts[r].gather(xs[r], ys[r]) for r in (0, 1)], args.ops)
        for _ in range(3):
            for r in (0, 1):
                with torch.cuda.stream(streams[r]):
                    gs[r].replay()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(args.reps):
            for r in (0, 1):
                with torch.cuda.stream(streams[r]):
                    gs[r].replay()
        torch.cuda.synchronize()
        g_us = (time.perf_counter() - t) * 1e6 / (args.reps * args.ops)
        for rt in rts:
            rt.check()
        print(json.dumps({"bytes": n, "bits_equal": ok, "eager_pair_us": round(us, 2),
                          "graph_us": round(g_us, 2)}), flush=True)
    print("rank 0 stats:", json.dumps(rts[0].snapshot()), flush=True)
    torch.cuda.synchronize()
    for rt in rts:
        rt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="bench", choices=["bench", "fault", "soak", "loopback", "stress"])
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--master", default=os.environ.get("HEAD_IP", "127.0.0.1"), help="rank 0 address (default: $HEAD_IP)")
    ap.add_argument("--port", type=int, default=29561)
    ap.add_argument("--sizes", type=lambda v: [_size(t) for t in v.split(",")], default="16k,64k,128k,1m")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--ops", type=int, default=90)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--timeout", type=float, default=None, help="seconds (fault mode default 3)")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--loop", action="store_true", help="stress: one node, both ranks in this process (NIC loopback)")
    ap.add_argument("--check", type=int, default=10000, help="stress: synchronize and check every this many ops")
    args = ap.parse_args()
    if args.mode == "stress":
        if "--iters" not in sys.argv:
            args.iters = 100000
        if "--sizes" not in sys.argv:
            args.sizes = "16k,128k"
    if isinstance(args.sizes, str):
        args.sizes = [_size(t) for t in args.sizes.split(",")]
    if args.mode == "fault" and args.timeout is None:
        args.timeout = 3.0
    {"bench": bench, "fault": fault, "soak": soak, "loopback": loopback, "stress": stress}[args.mode](args)


if __name__ == "__main__":
    main()
