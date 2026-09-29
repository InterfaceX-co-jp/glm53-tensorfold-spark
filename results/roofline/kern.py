#!/usr/bin/env python3
"""Roofline input: per-kernel time inside W7 windows (nsys sqlite export of one rank).

  kern.py SQLITE MODE OUT.json      MODE = prefill | dec1 | dec4

prefill: every kernel between the first pf_begin and the last pf_end mark (the whole traced prompt: pf24 trace).
dec1 / dec4: the W7:round ranges that hold a W7:verify and no piece, in the mix trace's segment 1 (1 stream) or 2
(4 streams) (same windows as results/W7/dec.py). Kernels are attributed to the probe component named by the next
W7:<component> mark on the serving thread (results/W7/analyze.py's rule); graph-replayed kernels have no mark.
Output: per (component, kernel, grid, block): count, total ms, median us, and the median of the calls that did not
overlap an NCCL kernel ("alone")."""
import bisect, collections, json, sqlite3, statistics, sys

db, mode, out = sys.argv[1:4]
c = sqlite3.connect(db)
S = dict(c.execute("select id, value from StringIds"))
K = c.execute("select start, end, streamId, correlationId, shortName, gridX*gridY*gridZ, blockX*blockY*blockZ, "
              "registersPerThread, staticSharedMemory + dynamicSharedMemory from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
K = [(s, e, st, cid, S.get(nm, str(nm)), grid, blk, reg, smem) for s, e, st, cid, nm, grid, blk, reg, smem in K]
RT = {cid: (s, tid) for s, tid, cid in c.execute("select start, globalTid, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME")}
NV = c.execute("select start, end, coalesce(text, ''), textId, globalTid from NVTX_EVENTS").fetchall()
NV = [(s, e, t or S.get(ti, ""), g) for s, e, t, ti, g in NV]
tidc = collections.Counter(g for s, e, t, g in NV if t.startswith("W7:"))
main = tidc.most_common(1)[0][0]
marks = sorted((s, t[3:]) for s, e, t, g in NV if e is None and g == main and t.startswith("W7:"))
mt = [m[0] for m in marks]
ranges = collections.defaultdict(list)
for s, e, t, g in NV:
    if e is not None and g == main and t.startswith("W7:"):
        ranges[t[3:]].append((s, e))
for v in ranges.values():
    v.sort()


def comp(cid):
    rt = RT.get(cid)
    if rt is None or rt[1] != main:
        return "graph/other-thread"
    i = bisect.bisect_left(mt, rt[0])
    return marks[i][1] if i < len(marks) else "after-last-mark"


begins = [t for t, n in marks if n == "pf_begin"]
ends = [t for t, n in marks if n == "pf_end"]
pieces = [(b, ends[bisect.bisect_right(ends, b)]) for b in begins if bisect.bisect_right(ends, b) < len(ends)]
if mode == "prefill":
    wins = [(pieces[0][0], pieces[-1][1])]
else:
    rounds = ranges["round"]
    segs = []
    for r in rounds:
        if segs and r[0] - segs[-1][-1][1] < 300e6:
            segs[-1].append(r)
        else:
            segs.append([r])
    seg = segs[1 if mode == "dec1" else 2]
    ver = ranges["verify"]
    wins = [r for r in seg if any(r[0] <= v[0] < r[1] for v in ver) and not any(r[0] <= p[0] < r[1] for p in pieces)]
    if mode == "dec4":
        wins = wins[1:-1]          # drop the ramp rounds at the ends (in-flight 4 = the analysis' 145 of 191)
nc = sorted((s, e) for s, e, st, cid, nm, *_ in K if nm.startswith("nccl"))
ncs = [x[0] for x in nc]


def ov(s, e):
    i = max(0, bisect.bisect_right(ncs, s) - 2)
    t = 0
    while i < len(nc) and nc[i][0] < e:
        t += max(0, min(nc[i][1], e) - max(nc[i][0], s))
        i += 1
    return t


agg = collections.defaultdict(lambda: {"n": 0, "ms": 0.0, "d": [], "alone": []})
ks = [k[0] for k in K]
Ks = sorted(K)
ks = [k[0] for k in Ks]
wall = 0.0
for w0, w1 in wins:
    wall += (w1 - w0) / 1e6
    for s, e, st, cid, nm, grid, blk, reg, smem in Ks[bisect.bisect_left(ks, w0):bisect.bisect_left(ks, w1)]:
        key = (comp(cid), nm, grid, blk, reg, smem)
        a = agg[key]
        a["n"] += 1
        a["ms"] += (e - s) / 1e6
        a["d"].append(e - s)
        if not nm.startswith("nccl") and ov(s, e) < 0.05 * (e - s):
            a["alone"].append(e - s)
rows = []
for (cp, nm, grid, blk, reg, smem), a in agg.items():
    rows.append({"comp": cp, "kernel": nm, "grid": grid, "block": blk, "regs": reg, "smem": smem, "count": a["n"],
                 "ms": round(a["ms"], 3), "med_us": round(statistics.median(a["d"]) / 1e3, 1),
                 "alone_med_us": round(statistics.median(a["alone"]) / 1e3, 1) if a["alone"] else None,
                 "alone_n": len(a["alone"])})
rows.sort(key=lambda r: -r["ms"])
res = {"db": db, "mode": mode, "windows": len(wins), "wall_ms": round(wall, 3), "pieces": len(pieces), "rows": rows}
json.dump(res, open(out, "w"), indent=0)
print(mode, "windows", len(wins), "wall ms", round(wall, 1), "per window", round(wall / len(wins), 2))
for r in rows[:60]:
    print(f"{r['comp'][:18]:18s} {r['kernel'][:34]:34s} g{r['grid']:>7} b{r['block']:>4} n{r['count']:>6} {r['ms']/len(wins):9.3f}ms/w med {r['med_us']:8.1f} alone {r['alone_med_us']}")
