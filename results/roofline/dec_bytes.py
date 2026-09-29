#!/usr/bin/env python3
"""Decode rounds (W7 mix trace, dec.py's windows): expert-layer reads a round inferred from grouped_kernel time
(R = 1 bandwidth calibration, see experts_union.py), the verify window's rows (grid.x = 8R + 1), and the q4 GEMM
kernel time by grid.  dec_bytes.py SQLITE dec1|dec4"""
import bisect, collections, sqlite3, statistics, sys

db, mode = sys.argv[1:3]
c = sqlite3.connect(db)
S = dict(c.execute("select id, value from StringIds"))
NV = [(s, e, t or S.get(ti, ""), g) for s, e, t, ti, g in
      c.execute("select start, end, coalesce(text, ''), textId, globalTid from NVTX_EVENTS")]
main = collections.Counter(g for s, e, t, g in NV if t.startswith("W7:")).most_common(1)[0][0]
R_ = collections.defaultdict(list)
for s, e, t, g in NV:
    if e is not None and g == main and t.startswith("W7:"):
        R_[t[3:]].append((s, e))
pb = sorted(s for s, e, t, g in NV if e is None and g == main and t == "W7:pf_begin")
segs = []
for r in sorted(R_["round"]):
    if segs and r[0] - segs[-1][-1][1] < 300e6:
        segs[-1].append(r)
    else:
        segs.append([r])
seg = segs[1 if mode == "dec1" else 2]
wins = [r for r in seg if any(r[0] <= v[0] < r[1] for v in R_["verify"]) and not any(r[0] <= p < r[1] for p in pb)]
# keep the rounds analyze.py counted with all requests in flight (1 or 4): results/W7/analysis/mix-r0.json
import json, os
A = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "W7", "analysis", "mix-r0.json")))
want = 1 if mode == "dec1" else 4
keep = {round(w["t_ms"], 1) for w in A["segments"][1 if mode == "dec1" else 2]["decode_rounds"] if w["inflight"] == want}
wins = [w for w in wins if round(w[0] / 1e6, 1) in keep]
K = sorted(c.execute("select start, end, shortName, gridX, gridY, gridZ from CUPTI_ACTIVITY_KIND_KERNEL").fetchall())
ks = [k[0] for k in K]
T1 = {8: 167.5, 32: 83.8}           # R = 1 medians (experts_union.py): 200 GB/s
reads, rows_v, ms_ex = [], [], []
for w0, w1 in wins:
    rd = 0.0
    big = 0
    t = 0.0
    for s, e, nm, x, y, z in K[bisect.bisect_left(ks, w0):bisect.bisect_left(ks, w1)]:
        if S.get(nm) == "grouped_kernel" and y in T1:
            rd += 0.5 * 8 * (e - s) / 1e3 / T1[y]        # gate/up and down each give one estimate: average
            t += (e - s) / 1e6
            big = max(big, (x - 1) // 8)
    reads.append(rd)
    rows_v.append(big)
    ms_ex.append(t)
n = len(wins)
print(mode, "rounds", n, "wall", round(sum((b - a) for a, b in wins) / 1e6 / n, 2), "ms")
print(" grouped_kernel ms a round", round(sum(ms_ex) / n, 2), "; expert-layer reads a round", round(sum(reads) / n, 1),
      "=", round(sum(reads) / n * 6.29e-3, 2), "GB; largest window rows: mean", round(statistics.mean(rows_v), 2),
      "median", statistics.median(rows_v))
print(" rows histogram", sorted(collections.Counter(rows_v).items()))
