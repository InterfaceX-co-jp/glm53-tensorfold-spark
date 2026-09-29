#!/usr/bin/env python3
"""THEORY-2: what happens between the end of a decode round's drafting and the verify forward's first kernel (host
NVTX W7:plan, NCCL control exchanges, graph launch), per segment.

  between.py R.sqlite"""
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
RT = {cid: (s, e, S.get(n, "")) for s, e, n, cid in c.execute("select start, end, nameId, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME")}
K = sorted((s, e, S.get(n, ""), cid, g) for s, e, n, cid, g in c.execute("select start, end, shortName, correlationId, coalesce(graphId,0) from CUPTI_ACTIVITY_KIND_KERNEL"))
ks = [k[0] for k in K]
vs = [v[0] for v in R["verify"]]
rounds = [r for r in R["round"] if (lambda i: i < len(vs) and vs[i] < r[1])(bisect.bisect_left(vs, r[0]))
          and not any(r[0] <= p[0] < r[1] for p in R.get("piece", []))]
segs = []
for r in rounds:
    if segs and r[0] - segs[-1][-1][1] < 1.5e9:
        segs[-1].append(r)
    else:
        segs.append([r])
plans = R.get("plan", [])
ps = [p[0] for p in plans]
for si, seg in enumerate(segs):
    if len(seg) < 5:
        continue
    gaps, plan_ms, nccl_ms, vfirst, gl = [], [], [], [], []
    for a, b in zip(seg, seg[1:]):
        gaps.append((b[0] - a[1]) / 1e6)
        i = bisect.bisect_left(ps, a[1] - 1e5)
        pm = sum((e - s) for s, e in plans[i:] if s < b[0]) / 1e6 if i < len(plans) else 0
        plan_ms.append(pm)
        j0, j1 = bisect.bisect_left(ks, a[1] - 3e6), bisect.bisect_left(ks, b[0] + 3e6)
        nccl_ms.append(sum(e - s for s, e, n, cid, g in K[j0:j1] if n.startswith("nccl") and a[1] - 1e6 <= s < b[0] + 5e5) / 1e6)
    # the verify forward: first kernel launched inside vfwd vs the vfwd's graph launch call
    for v0, v1 in R["vfwd"]:
        if not (seg[0][0] <= v0 < seg[-1][1]):
            continue
        j = bisect.bisect_left(ks, v0)
        cand = [k for k in K[j:j + 4000] if RT.get(k[3], (0,))[0] >= v0 and RT[k[3]][0] < v1]
        if cand:
            f = min(cand)
            vfirst.append((f[0] - v0) / 1e3)
            if RT[f[3]][2].startswith("cudaGraphLaunch"):
                gl.append((RT[f[3]][1] - RT[f[3]][0]) / 1e3)
    n = len(seg)
    print(f"segment {si}: {n} rounds; host gap between rounds mean {statistics.mean(gaps):.3f} ms (median {statistics.median(gaps):.3f}); "
          f"W7:plan between rounds in {sum(1 for p in plan_ms if p > 0)} of {len(plan_ms)} gaps, mean {statistics.mean(plan_ms):.3f} ms a round; "
          f"NCCL kernel ms near the gap {statistics.mean(nccl_ms):.3f} a round; vfwd start -> first verify kernel median "
          f"{statistics.median(vfirst):.0f} us (mean {statistics.mean(vfirst):.0f}); verify graph launch call median "
          f"{statistics.median(gl) if gl else 0:.0f} us over {len(gl)} rounds")
