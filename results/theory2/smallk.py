#!/usr/bin/env python3
"""THEORY-2: inventory of the non-streaming kernels of a decode round (everything but grouped_kernel / _qmm /
gather_kernel): name, grid, block, registers, calls a round, median us, ms a round. The latency-bound set a fused /
persistent layer-boundary kernel would replace.

  smallk.py R.sqlite [segment index, default: the longest]"""
import bisect, collections, sqlite3, statistics, sys
c = sqlite3.connect(sys.argv[1])
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
seg = segs[int(sys.argv[2])] if len(sys.argv) > 2 else max(segs, key=len)
rs = [r[0] for r in seg]
RT = {cid: s for s, cid in c.execute("select start, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME")}
d = collections.defaultdict(list)
for s, e, n, cid, gx, gy, gz, bx, reg in c.execute(
        "select start, end, shortName, correlationId, gridX, gridY, gridZ, blockX, registersPerThread from CUPTI_ACTIVITY_KIND_KERNEL"):
    t = RT.get(cid)
    if t is None:
        continue
    i = bisect.bisect_right(rs, t) - 1
    if i < 0 or t >= seg[i][1]:
        continue
    nm = S.get(n, "")
    if nm in ("grouped_kernel", "_qmm", "gather_kernel", "grouped_loop_kernel"):
        continue
    d[(nm, gx * gy * gz, bx, reg)].append((e - s) / 1e3)
n = len(seg)
tot = sum(sum(v) for v in d.values()) / n / 1e3
print(f"{n} rounds; non-streaming kernels: {sum(len(v) for v in d.values()) / n:.0f} a round, {tot:.2f} ms a round")
print(f"{'kernel':34} {'CTAs':>6} {'thr':>5} {'reg':>4} {'calls':>7} {'med us':>7} {'ms/round':>9} {'cum':>6}")
cum = 0
for k, v in sorted(d.items(), key=lambda kv: -sum(kv[1]))[:40]:
    ms = sum(v) / n / 1e3
    cum += ms
    print(f"{k[0][:34]:34} {k[1]:6d} {k[2]:5d} {k[3]:4d} {len(v) / n:7.1f} {statistics.median(v):7.1f} {ms:9.3f} {cum:6.2f}")
