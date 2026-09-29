#!/usr/bin/env python3
"""THEORY-2: duration spread of identical-work kernels (same kernel, same grid) within each rank: the R = 1 MTP expert
calls (exactly 8 experts), the head _qmm (1,1210,1), KDA projection (1,197,4), dense MLP gate/up (1,192,1).

  jitter.py cap-r0.sqlite cap-r1.sqlite > jitter.txt"""
import collections, sqlite3, statistics, sys
for db in sys.argv[1:]:
    c = sqlite3.connect(db)
    S = dict(c.execute("select id, value from StringIds"))
    d = collections.defaultdict(list)
    for s, e, n, gx, gy, gz in c.execute("select start, end, shortName, gridX, gridY, gridZ from CUPTI_ACTIVITY_KIND_KERNEL"):
        nm = S.get(n, "")
        if (nm == "grouped_kernel" and gx == 9) or (nm == "_qmm" and (gy, gz) in ((1210, 1), (197, 4), (192, 1))):
            d[(nm, gx, gy, gz)].append((e - s) / 1e3)
    for k, v in sorted(d.items()):
        q = statistics.quantiles(v, n=100)
        print(f"{db} {k}: n {len(v)} median {statistics.median(v):.1f} us  p5 {q[4]:.1f}  p25 {q[24]:.1f}  p75 {q[74]:.1f}  p95 {q[94]:.1f}  "
              f"p99 {q[98]:.1f}  (p95-p5)/median {100 * (q[94] - q[4]) / statistics.median(v):.1f}%")
