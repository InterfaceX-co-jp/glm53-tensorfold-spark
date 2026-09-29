#!/usr/bin/env python3
"""W11: in-situ per-kernel durations of the dense q4 GEMVs (_qmm by launch grid) and grouped_kernel (by rows R) in
a trace's decode rounds (w11dec.py JSON for the round windows), against kbench (isolated) and probe (plain read).

  insitu.py SQLITE DEC.json SEGMENTS(comma list)"""
import bisect
import collections
import json
import sqlite3
import statistics
import sys

db, dj, segs = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
D = json.load(open(dj))
wins = []
for i in segs:
    s = D["segments"][i]
    for j in D["masks"][i]:
        r = s["rounds_detail"][j]
        wins.append((r["t0_ms"] * 1e6, r["t0_ms"] * 1e6 + r["wall_ms"] * 1e6))
wins.sort()
ws = [w[0] for w in wins]
c = sqlite3.connect(db)
S = dict(c.execute("select id, value from StringIds"))
k = collections.defaultdict(list)
for s, e, n, x, y, z in c.execute("select start, end, shortName, gridX, gridY, gridZ from CUPTI_ACTIVITY_KIND_KERNEL"):
    i = bisect.bisect_right(ws, s) - 1
    if i < 0 or s >= wins[i][1]:
        continue
    nm = S.get(n, "")
    if nm == "_qmm":
        k[("_qmm", x, y, z)].append((e - s) / 1e3)
    elif nm == "grouped_kernel":
        k[("grouped", "gu" if y == 8 else "dn", (x - 1) // 8)].append((e - s) / 1e3)
    elif nm in ("_reduce", "gather_kernel", "rot_in_kernel", "gateup_epilogue_kernel", "down_epilogue_kernel", "_swiglu"):
        k[(nm,)].append((e - s) / 1e3)
n = len(wins)
print("rounds", n)
for key, v in sorted(k.items(), key=lambda kv: -sum(kv[1])):
    print(f"{str(key):40s} calls/round {len(v) / n:7.2f}  median {statistics.median(v):8.1f} us  mean {statistics.mean(v):8.1f}  ms/round {sum(v) / n / 1e3:7.3f}")
