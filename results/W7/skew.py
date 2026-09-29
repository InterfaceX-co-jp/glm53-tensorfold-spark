#!/usr/bin/env python3
"""W7: rank-to-rank skew from the two ranks' NCCL all-gather kernels (analyze.py JSONs of the same capture).

  skew.py R0.json R1.json [SEGMENT]      SEGMENT: index into analyze.py's segments (each rank's own time range)

Collectives are matched in launch order after an alignment offset (the two captures start a few collectives apart; the
offset with the closest durations wins). An all-gather kernel spins until the peer's data arrive, so on each
collective the rank that arrived first shows the longer kernel: wait_r = max(0, dur_r - dur_other); the shorter
duration approximates the transfer itself."""
import json, statistics, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
la, lb = a["nccl_list"], b["nccl_list"]
if len(sys.argv) > 3:
    k = int(sys.argv[3])
    sa, sb = a["segments"][k], b["segments"][k]
    la = [x for x in la if sa["t0_ms"] * 1e6 <= x[0] < sa["t1_ms"] * 1e6]
    lb = [x for x in lb if sb["t0_ms"] * 1e6 <= x[0] < sb["t1_ms"] * 1e6]
best = None
for off in range(-5, 6):
    xa, xb = (la[off:], lb) if off >= 0 else (la, lb[-off:])
    m = min(len(xa), len(xb), 3000)
    if m < 10:
        continue
    cost = sum(abs(x[1] - y[1]) for x, y in zip(xa[:m], xb[:m])) / m
    if best is None or cost < best[0]:
        best = (cost, off)
off = best[1] if best else 0
la, lb = (la[off:], lb) if off >= 0 else (la, lb[-off:])
n = min(len(la), len(lb))
w0 = w1 = xfer = 0
d0s, d1s, mins = [], [], []
for (s0, d0), (s1, d1) in zip(la[:n], lb[:n]):
    w0 += max(0, d0 - d1); w1 += max(0, d1 - d0); m = min(d0, d1); xfer += m
    d0s.append(d0); d1s.append(d1); mins.append(m)
first0 = sum(1 for x, y in zip(d0s, d1s) if x > y)
print(json.dumps({"offset": off, "collectives": n, "r0_total_ms": round(sum(d0s) / 1e6, 2),
                  "r1_total_ms": round(sum(d1s) / 1e6, 2),
                  "r0_waits_for_r1_ms": round(w0 / 1e6, 2), "r1_waits_for_r0_ms": round(w1 / 1e6, 2),
                  "transfer_floor_ms": round(xfer / 1e6, 2), "r0_arrives_first_share": round(first0 / max(n, 1), 3),
                  "median_min_us": round(statistics.median(mins) / 1e3, 1) if mins else None,
                  "p90_min_us": round(sorted(mins)[int(0.9 * len(mins))] / 1e3, 1) if mins else None}))
