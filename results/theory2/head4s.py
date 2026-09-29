#!/usr/bin/env python3
"""THEORY-2: the 178 MB head _qmm (grid 1,1210,1) calls a round, 1 stream (cap-r0 segment 1) vs 4 streams (cap4-r0
segment 0), with the per-round histogram.

  head4s.py cap-r0.sqlite cap4-r0.sqlite > head4s.txt"""
import bisect, collections, sqlite3, statistics, sys


def run(db, segi):
    c = sqlite3.connect(db)
    S = dict(c.execute("select id, value from StringIds"))
    NV = [(s, e, t or S.get(ti, ""), g) for s, e, t, ti, g in c.execute("select start, end, text, textId, globalTid from NVTX_EVENTS")]
    main = collections.Counter(g for s, e, t, g in NV if t.startswith("W7:") and e is not None).most_common(1)[0][0]
    R = collections.defaultdict(list)
    for s, e, t, g in NV:
        if e is not None and g == main and t.startswith("W7:"):
            R[t[3:]].append((s, e))
    for v in R.values():
        v.sort()
    vs = [v[0] for v in R["verify"]]
    rounds = [r for r in R["round"] if (lambda i: i < len(vs) and vs[i] < r[1])(bisect.bisect_left(vs, r[0]))]
    segs = []
    for r in rounds:
        if segs and r[0] - segs[-1][-1][1] < 1.5e9:
            segs[-1].append(r)
        else:
            segs.append([r])
    seg = segs[segi]
    rs = [r[0] for r in seg]
    RT = {cid: s for s, cid in c.execute("select start, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME")}
    per, us = collections.Counter(), []
    for s, e, n, cid, gy in c.execute("select start, end, shortName, correlationId, gridY from CUPTI_ACTIVITY_KIND_KERNEL"):
        if gy != 1210 or S.get(n) != "_qmm":
            continue
        t = RT.get(cid)
        i = bisect.bisect_right(rs, t) - 1 if t else -1
        if i < 0 or t >= seg[i][1]:
            continue
        per[i] += 1
        us.append((e - s) / 1e3)
    dist = collections.Counter(per[i] for i in range(len(seg)))
    print(f"{db} seg {segi}: {len(seg)} rounds, head calls a round mean {sum(per.values()) / len(seg):.2f}, "
          f"median {statistics.median(us):.0f} us each, {sum(us) / len(seg) / 1e3:.2f} ms a round; calls-per-round histogram {sorted(dist.items())}")


run(sys.argv[1], 1)
run(sys.argv[2], 0)
