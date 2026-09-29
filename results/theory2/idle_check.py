#!/usr/bin/env python3
"""THEORY-2: device idle inside decode rounds on the device timeline (every kernel / memcpy / memset, whichever round
launched it), against W11's per-round "wall - busy of the kernels the round launched".

  idle_check.py R0.sqlite"""
import bisect, collections, sqlite3, statistics, sys
c = sqlite3.connect(sys.argv[1])
S = dict(c.execute("select id, value from StringIds"))
NV = [(s, e, t or S.get(ti, ""), g) for s, e, t, ti, g in c.execute("select start, end, text, textId, globalTid from NVTX_EVENTS")]
main = collections.Counter(g for s, e, t, g in NV if t.startswith("W7:") and e is not None).most_common(1)[0][0]
R = collections.defaultdict(list)
for s, e, t, g in NV:
    if e is not None and g == main and t.startswith("W7:"):
        R[t[3:]].append((s, e))
vs = sorted(v[0] for v in R["verify"])
rounds = sorted(r for r in R["round"] if (lambda i: i < len(vs) and vs[i] < r[1])(bisect.bisect_left(vs, r[0]))
                and not any(r[0] <= p[0] < r[1] for p in R.get("piece", [])))
iv = [(s, e) for s, e in c.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL")]
for tb in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
    iv += [(s, e) for s, e in c.execute(f"select start, end from {tb}")]
iv.sort()
# merge
m = []
for s, e in iv:
    if m and s <= m[-1][1]:
        m[-1][1] = max(m[-1][1], e)
    else:
        m.append([s, e])
ms = [x[0] for x in m]
tot_w = tot_b = 0
per = []
for r0, r1 in rounds:
    i = max(0, bisect.bisect_right(ms, r0) - 1)
    b = 0
    while i < len(m) and m[i][0] < r1:
        b += max(0, min(m[i][1], r1) - max(m[i][0], r0))
        i += 1
    per.append(((r1 - r0) - b) / 1e6)
    tot_w += r1 - r0
    tot_b += b
print(f"rounds {len(rounds)}: mean wall {tot_w / len(rounds) / 1e6:.2f} ms, device idle inside the round windows "
      f"{(tot_w - tot_b) / len(rounds) / 1e6:.3f} ms a round (median {statistics.median(per):.3f}, p90 {statistics.quantiles(per, n=10)[-1]:.3f})")
