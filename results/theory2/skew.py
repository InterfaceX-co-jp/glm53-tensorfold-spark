#!/usr/bin/env python3
"""THEORY-2: where the two ranks' per-exchange skew comes from (W11 nsys sqlite exports, both ranks).

  skew.py R0.sqlite R1.sqlite

The ranks run the same collectives in the same order (lockstep), so the k-th gather_kernel on rank 0 and rank 1 are the
same exchange. A "block" is the device work between two consecutive exchanges. For each block: the kernel time by
family on both ranks, the device idle inside the block, and the difference. No clock alignment is needed: block
durations are measured on each rank's own clock.

Output: the per-exchange cost split into transport (the shorter of the two gather durations) and skew (the longer
minus the shorter); the distribution of the block-time difference by block kind and by kernel family (which family's
rank-to-rank variation makes the skew); and the within-rank variation of fixed-work blocks (same kernels, same bytes)
across rounds (is it jitter of identical work?).
"""
import collections
import sqlite3
import statistics
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])


def fam(n):
    if n in ("grouped_kernel", "gateup_epilogue_kernel", "down_epilogue_kernel", "rot_in_kernel"):
        return "experts"
    if n in ("_qmm", "_reduce", "_swiglu", "_fq4"):
        return "dense"
    if n == "gather_kernel":
        return "xchg"
    if n.startswith("_hc"):
        return "hc"
    if n.startswith("chain") or "kda" in n or n.startswith("_conv") or n.startswith("replay"):
        return "kda"
    if n.startswith(("_router", "_combine", "_topk", "_group", "_select", "_moe")):
        return "router"
    return "other"


def load(db):
    c = sqlite3.connect(db)
    S = dict(c.execute("select id, value from StringIds"))
    return [(s, e, S.get(n, "")) for s, e, n in
            c.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start")]


def blocks(K):
    out, cur, gat = [], collections.Counter(), []
    prev_end = None
    idle = 0
    for s, e, n in K:
        if prev_end is not None and s > prev_end:
            idle += s - prev_end
        prev_end = e if prev_end is None else max(prev_end, e)
        if n == "gather_kernel":
            out.append((cur, idle, e - s))
            cur, idle = collections.Counter(), 0
        else:
            cur[fam(n)] += e - s
            cur["_sig"] = cur.get("_sig", 0)
    return out


B0 = blocks(load(sys.argv[1]))
B1 = blocks(load(sys.argv[2]))
n = min(len(B0), len(B1))
print(f"exchanges: rank 0 {len(B0)}, rank 1 {len(B1)}")
g0 = [B0[i][2] / 1e3 for i in range(n)]
g1 = [B1[i][2] / 1e3 for i in range(n)]
tr = [min(a, b) for a, b in zip(g0, g1)]
sk = [abs(a - b) for a, b in zip(g0, g1)]
print(f"gather us: rank 0 mean {statistics.mean(g0):.1f} (median {statistics.median(g0):.1f}), rank 1 mean {statistics.mean(g1):.1f}"
      f" (median {statistics.median(g1):.1f}); transport = min of the pair: mean {statistics.mean(tr):.1f}, median "
      f"{statistics.median(tr):.1f}; skew = |difference|: mean {statistics.mean(sk):.1f}, median {statistics.median(sk):.1f}, "
      f"p90 {statistics.quantiles(sk, n=10)[-1]:.1f}")
# block time on each rank = kernels + idle between the previous exchange's end and this exchange's start
rows = collections.defaultdict(list)
famdiff = collections.defaultdict(list)
for i in range(1, n):
    c0, i0, _ = B0[i]
    c1, i1, _ = B1[i]
    t0 = sum(v for k, v in c0.items() if k != "_sig") + i0
    t1 = sum(v for k, v in c1.items() if k != "_sig") + i1
    kind = "moe" if c0.get("experts", 0) > 0 else ("kda" if c0.get("kda", 0) > 0 else "dense-only")
    rows[kind].append(((t0 - t1) / 1e3, t0 / 1e3))
    for f in set(c0) | set(c1):
        if f != "_sig":
            famdiff[f].append((c0.get(f, 0) - c1.get(f, 0)) / 1e3)
    famdiff["idle"].append((i0 - i1) / 1e3)
print("\nblock time difference rank0 - rank1 (us), by block kind (what precedes the exchange):")
for k, v in sorted(rows.items(), key=lambda kv: -len(kv[1])):
    d = [x for x, _ in v]
    t = [y for _, y in v]
    print(f"  {k:10} n {len(v):6d}  block mean {statistics.mean(t):7.1f} us  diff mean {statistics.mean(d):+6.2f}  mean |diff| "
          f"{statistics.mean(abs(x) for x in d):6.2f}  sd {statistics.pstdev(d):6.2f}  (|diff| / block {100 * statistics.mean(abs(x) for x in d) / statistics.mean(t):.1f}%)")
print("\nper family: sd of the rank0 - rank1 difference per block (us) and its mean (systematic part):")
for f, v in sorted(famdiff.items(), key=lambda kv: -statistics.pstdev(kv[1])):
    print(f"  {f:8} sd {statistics.pstdev(v):6.2f}  mean {statistics.mean(v):+6.2f}")
# fixed-work blocks: dense-only blocks (no experts, no KDA) have the same kernels and bytes every round at a given
# position; their within-rank spread is the jitter of identical work
print("\nwithin-rank spread of identical work: dense-only blocks, grouped by kernel signature (count of kernels)")
for name, B in (("rank 0", B0), ("rank 1", B1)):
    grp = collections.defaultdict(list)
    for c, idle, _ in B:
        if c.get("experts", 0) == 0 and c.get("kda", 0) == 0 and c.get("dense", 0) > 0:
            key = round(c["dense"] / 1e3 / 20) * 20  # 20 us bins of dense time as a proxy for the shape set
            grp[key].append((sum(v for k, v in c.items() if k != "_sig") + idle) / 1e3)
    for key, v in sorted(grp.items(), key=lambda kv: -len(kv[1]))[:4]:
        if len(v) > 50:
            print(f"  {name} dense~{key} us: n {len(v)}  mean {statistics.mean(v):.1f}  sd {statistics.pstdev(v):.2f} "
                  f"({100 * statistics.pstdev(v) / statistics.mean(v):.1f}%)  p5 {statistics.quantiles(v, n=20)[0]:.1f}  p95 {statistics.quantiles(v, n=20)[-1]:.1f}")
