#!/usr/bin/env python3
"""W7: decode round breakdown by kernel family and round phase (nsys sqlite + analyze.py JSON of the same rank).

  dec.py SQLITE ANALYZE.json SEGMENT INFLIGHT OUT.json

Rounds: analyze.py's decode rounds of segment SEGMENT (W7:round ranges holding a verify, no prefill piece) with
INFLIGHT requests in flight (req.py timings). Every kernel / memcpy is put in a family by its name (graph-replayed
kernels carry no probe marks) and in a phase by where it was launched on the serving thread (W7:vfwd = the verify
forward, W7:propose / mtp_chains / mtp_multi = drafting, W7:sample_* = sampling; else 'round other'). Per round:
kernel time per family and phase (sum of durations), the GPU-busy union, the NCCL part not overlapped by compute,
and GPU idle; reported as the mean over the rounds and for the median-wall round."""
import bisect
import collections
import json
import re
import sqlite3
import statistics
import sys

db, js, seg_i, want, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
c = sqlite3.connect(db)
S = dict(c.execute("select id, value from StringIds"))
A = json.load(open(js))
main = A["main_tid"]
seg = A["segments"][seg_i]
rounds = [w for w in seg["decode_rounds"] if w["inflight"] == want]

FAM = [
    ("routed experts (exl3 grouped)", r"^(grouped_kernel|grouped_loop_kernel|expert_kernel|gateup_epilogue_kernel|down_epilogue_kernel|rot_in1?_kernel|plan_kernel)$"),
    ("MoE router / grouping / combine", r"^(_router_part|_router_sum|_topk|_group|_combine|_combine_s|DeviceRadixSort.*|searchsorted.*|fill_reverse_indices_kernel|DeviceScanKernel)$"),
    ("dense q4 GEMMs (proj, shared expert, head, drafter)", r"^(_qmm|_fq4|_group_sums|_reduce|_swiglu|chain_mm.*)$"),
    ("KDA (chain, conv, replay)", r"^(chain_kernel|_kda_.*|_conv_shift|_dconv_kernel|replay_layers_kernel|_dattn_kernel)$"),
    ("DSA/MLA attention + indexer", r"^(_lchunks|_lmerge|_lsparse.*|_expand|_absorb|_lwrite8|_index_write|_pool_keys|_scores_rows|gatherTopK|compute(BlockDigitCounts|DigitCumSum|BlockwiseWithinKCounts)|DeviceScanByKeyKernel|_gather_sel)$"),
    ("hyper-connections", r"^(_hc_.*|_stream_mean)$"),
    ("NCCL", r"^nccl"),
    ("norms / elementwise / other", r".*"),
]
FAMC = [(n, re.compile(p)) for n, p in FAM]


def fam(nm):
    for n, p in FAMC:
        if p.match(nm):
            return n
    return "other"


RT = {cid: (s, tid) for s, tid, cid in c.execute("select start, globalTid, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME")}
NV = c.execute("select start, end, coalesce(text, ''), textId, globalTid from NVTX_EVENTS where end is not null").fetchall()
phase_r = []
for s, e, t, tid_, g in NV:
    t = t or S.get(tid_, "")
    if g == main and t in ("W7:vfwd", "W7:propose", "W7:mtp_chains", "W7:mtp_multi", "W7:sample_multi", "W7:sample_drafts"):
        phase_r.append((s, e, {"W7:vfwd": "verify forward", "W7:sample_multi": "sampling",
                               "W7:sample_drafts": "sampling"}.get(t, "drafting (MTP / DFlash2)")))
phase_r.sort()
ps = [p[0] for p in phase_r]


def phase(t):
    i = bisect.bisect_right(ps, t) - 1
    best = None
    while i >= 0 and i >= len(ps) - 10**9:
        s, e, n = phase_r[i]
        if s <= t < e:
            best = n if best is None or n == "verify forward" and best != "sampling" else best
            if n != "verify forward":        # innermost non-verify phase wins (sampling / drafting inside verify)
                return n
        if t - s > 5e9:
            break
        i -= 1
    return best or "round other"


def union(iv):
    iv.sort()
    tot, cur = 0, None
    for s, e in iv:
        if cur is None or s > cur[1]:
            if cur:
                tot += cur[1] - cur[0]
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    if cur:
        tot += cur[1] - cur[0]
    return tot


per = []
for w in rounds:
    t0 = w["t_ms"] * 1e6
    t1 = t0 + w["wall_ms"] * 1e6
    ks = c.execute("select start, end, shortName, correlationId from CUPTI_ACTIVITY_KIND_KERNEL where start < ? and end > ?",
                   (t1, t0)).fetchall()
    mc = c.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMCPY where start < ? and end > ?", (t1, t0)).fetchall()
    ms = c.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMSET where start < ? and end > ?", (t1, t0)).fetchall()
    famt, pht = collections.Counter(), collections.Counter()
    comp_iv, nccl_iv, all_iv = [], [], []
    for s, e, nm, cid in ks:
        s, e = max(s, t0), min(e, t1)
        n = S.get(nm, "")
        f = fam(n)
        d = (e - s) / 1e6
        famt[f] += d
        rt = RT.get(cid)
        pht[phase(rt[0]) if rt and rt[1] == main else "other thread"] += d
        (nccl_iv if f == "NCCL" else comp_iv).append((s, e))
        all_iv.append((s, e))
    for s, e in mc + ms:
        all_iv.append((max(s, t0), min(e, t1)))
        famt["memcpy / memset"] += (min(e, t1) - max(s, t0)) / 1e6
    busy = union(list(all_iv)) / 1e6
    comp_busy = union(list(comp_iv)) / 1e6
    nccl_exposed = (union(list(comp_iv) + list(nccl_iv)) - union(list(comp_iv))) / 1e6
    per.append({"wall_ms": w["wall_ms"], "gpu_busy_ms": round(busy, 3), "compute_busy_ms": round(comp_busy, 3),
                "nccl_exposed_ms": round(nccl_exposed, 3), "idle_ms": round(w["wall_ms"] - busy, 3),
                "family_ms": {k: round(v, 3) for k, v in famt.most_common()},
                "phase_ms": {k: round(v, 3) for k, v in pht.most_common()},
                "nccl_count": sum(v["count"] for v in w["nccl"].values()), "kernels": len(ks)})


def mean_of(key):
    keys = set()
    for p in per:
        keys.update(p[key])
    return {k: round(statistics.mean(p[key].get(k, 0.0) for p in per), 3) for k in sorted(keys, key=lambda k: -sum(p[key].get(k, 0) for p in per))}


res = {"segment": seg_i, "inflight": want, "rounds": len(per)}
if per:
    res["mean"] = {k: round(statistics.mean(p[k] for p in per), 3) for k in ("wall_ms", "gpu_busy_ms", "compute_busy_ms", "nccl_exposed_ms", "idle_ms", "nccl_count", "kernels")}
    res["mean"]["family_ms"] = mean_of("family_ms")
    res["mean"]["phase_ms"] = mean_of("phase_ms")
    walls = [p["wall_ms"] for p in per]
    res["median_round"] = per[sorted(range(len(per)), key=lambda i: walls[i])[len(per) // 2]]
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res.get("mean"), indent=1))
