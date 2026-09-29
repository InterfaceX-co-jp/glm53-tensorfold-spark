#!/usr/bin/env python3
"""THEORY-2: per-kernel-name count and time a round, 1 stream vs 4 streams (W11 captures), to split the per-slot fixed
cost by kernel. Rounds as in crit.py (kernels by launch-call host time); segment = the longest one of each trace.

  perslot.py CAP1.sqlite CAP4.sqlite"""
import bisect, collections, sqlite3, statistics, sys


def per_name(db):
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
    rounds = [r for r in R["round"] if (lambda i: i < len(vs) and vs[i] < r[1])(bisect.bisect_left(vs, r[0]))
              and not any(r[0] <= p[0] < r[1] for p in R.get("piece", []))]
    segs = []
    for r in rounds:
        if segs and r[0] - segs[-1][-1][1] < 1.5e9:
            segs[-1].append(r)
        else:
            segs.append([r])
    seg = max(segs, key=len) if "cap4" not in db else segs[0]
    rs = [r[0] for r in seg]
    RT = {cid: s for s, cid in c.execute("select start, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME")}
    ph = sorted((s, e, n) for n in ("vfwd", "propose", "sample_multi") for s, e in R.get(n, []))
    phs = [p[0] for p in ph]
    cnt, tim = collections.Counter(), collections.Counter()
    for s, e, n, cid, gx, gy, gz, bx in c.execute(
            "select start, end, shortName, correlationId, gridX, gridY, gridZ, blockX from CUPTI_ACTIVITY_KIND_KERNEL"):
        t = RT.get(cid)
        if t is None:
            continue
        i = bisect.bisect_right(rs, t) - 1
        if i < 0 or t >= seg[i][1]:
            continue
        j = bisect.bisect_right(phs, t) - 1
        p = ph[j][2] if j >= 0 and ph[j][0] <= t < ph[j][1] else "other"
        key = (S.get(n, ""), p)
        cnt[key] += 1
        tim[key] += (e - s) / 1e6
    return {k: (cnt[k] / len(seg), tim[k] / len(seg)) for k in cnt}, len(seg)


a, n1 = per_name(sys.argv[1])
b, n4 = per_name(sys.argv[2])
keys = set(a) | set(b)
rows = sorted(keys, key=lambda k: -(b.get(k, (0, 0))[1] - a.get(k, (0, 0))[1]))
print(f"rounds: 1 stream {n1}, 4 streams {n4}. Per round: count and ms at 1 / 4 streams; delta ms; per extra slot = delta / 3")
print(f"{'kernel':42} {'phase':12} {'n 1s':>7} {'n 4s':>7} {'ms 1s':>7} {'ms 4s':>7} {'d ms':>7} {'/slot':>6}")
tot = collections.Counter()
for k in rows[:45]:
    c1, t1 = a.get(k, (0, 0))
    c4, t4 = b.get(k, (0, 0))
    print(f"{k[0][:42]:42} {k[1]:12} {c1:7.1f} {c4:7.1f} {t1:7.2f} {t4:7.2f} {t4 - t1:+7.2f} {(t4 - t1) / 3:+6.2f}")
for k in keys:
    tot[k[1]] += b.get(k, (0, 0))[1] - a.get(k, (0, 0))[1]
print("delta by phase (ms a round):", {k: round(v, 2) for k, v in tot.items()})
print("kernels a round: 1s", round(sum(v[0] for v in a.values())), " 4s", round(sum(v[0] for v in b.values())))
