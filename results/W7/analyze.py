#!/usr/bin/env python3
"""W7: break an nsys trace (sqlite export) of one TensorFold rank into components.

  analyze.py SQLITE OUT.json

Inputs the W7 instrumentation leaves in the trace (results/W7/profile.py, W7_NVTX=1):
  - NVTX marks "W7:<component>" on the serving thread: the kernels launched since the previous mark belong to
    <component> (the GLM53_TF_PROFILE probe sites: kda.proj, moe.routed, allgather, hc, ...; "mtp." prefix = MTP head)
  - marks W7:pf_begin / W7:pf_end around each decode.prefill (one batch piece)
  - NVTX ranges W7:round / W7:piece / W7:verify / W7:vfwd / W7:propose / W7:mtp_chains / W7:sample_* / W7:plan

Output: per window (whole traced prefill, each piece, each decode round) a partition of the wall time:
  compute by component (concurrent compute kernels split the instant evenly), comm exposed (only NCCL kernels
  running), memcpy/memset only, GPU idle (split into inside-a-piece / verify vs outside);
plus busy time per component (sum of kernel durations, overlapped or not), NCCL per kernel name (count, time,
overlapped by compute on another stream, exposed), and the NCCL kernel list for the cross-rank skew (skew.py)."""
import bisect
import collections
import json
import re
import sqlite3
import statistics
import sys

db, out = sys.argv[1], sys.argv[2]
c = sqlite3.connect(db)
tables = {n for (n,) in c.execute("select name from sqlite_master where type='table'")}
S = dict(c.execute("select id, value from StringIds"))


def rows(q):
    return c.execute(q).fetchall()


# -- raw activity ---------------------------------------------------------------------------------------------------
K = rows("select start, end, streamId, correlationId, shortName, graphId, gridX*gridY*gridZ, blockX*blockY*blockZ "
         "from CUPTI_ACTIVITY_KIND_KERNEL")
K = [(s, e, st, cid, S.get(nm, str(nm)), g, grid, blk) for s, e, st, cid, nm, g, grid, blk in K]
MC = rows("select start, end, streamId, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY") \
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables else []
MS = rows("select start, end, streamId, bytes from CUPTI_ACTIVITY_KIND_MEMSET") \
    if "CUPTI_ACTIVITY_KIND_MEMSET" in tables else []
RT = {cid: (s, tid, S.get(nm, "")) for s, tid, cid, nm in
      rows("select start, globalTid, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME")}
NV = rows("select start, end, eventType, coalesce(text, ''), textId, globalTid from NVTX_EVENTS")
NV = [(s, e, t, txt or S.get(tid_, ""), g) for s, e, t, txt, tid_, g in NV]
marks_all = [(s, txt, g) for s, e, t, txt, g in NV if e is None and txt.startswith("W7:")]
ranges_all = [(s, e, txt, g) for s, e, t, txt, g in NV if e is not None and txt.startswith("W7:")]
tid_count = collections.Counter(g for _, _, g in marks_all)
tid_count.update(g for *_, g in ranges_all)
main = tid_count.most_common(1)[0][0] if tid_count else None
marks = sorted((s, t[3:]) for s, t, g in marks_all if g == main)
mark_t = [m[0] for m in marks]
ranges = sorted((s, e, t[3:]) for s, e, t, g in ranges_all if g == main)


def is_nccl(name):
    return name.startswith("nccl")


# -- kernel -> component (the next mark after its launch on the serving thread) -------------------------------------
comp_of = []
for s, e, st, cid, nm, g, grid, blk in K:
    rt = RT.get(cid)
    if rt is None or rt[1] != main:
        comp_of.append("other-thread" if rt else "no-launch")
        continue
    i = bisect.bisect_left(mark_t, rt[0])
    comp_of.append(marks[i][1] if i < len(marks) else "after-last-mark")


def name_class(nm):
    """Fallback class by kernel name (graph replays and code without probe sites)."""
    n = nm.lower()
    if is_nccl(nm):
        return "nccl"
    for pat, cls in ((r"grouped|routed|fat|fast2|expert|exl3", "experts(exl3)"), (r"kda|chunk_gated|recur|gdn|fla", "kda"),
                     (r"sparse|index|topk|top_k|latent|attn|attention|flash|mla", "attention/indexer"),
                     (r"hc_|sinkhorn|mhc|hyper", "hc"), (r"gemm|gemv|qmm|q4|matmul|cutlass|sm90|sm100|sm120|nvjet|xmma", "gemm(dense/shared)"),
                     (r"sample|softmax|argmax|sort|cumsum|multinomial|gumbel", "sampling"),
                     (r"copy|elementwise|vectorized|reduce|fill|cat|index_put|scatter|gather", "torch-elementwise")):
        if re.search(pat, n):
            return cls
    return "other"


# -- sweep: partition a window's wall time --------------------------------------------------------------------------
# activity: (start, end, kind, label); kind: c = compute kernel, n = nccl, m = memcpy/memset
ACT = []
for (s, e, st, cid, nm, g, grid, blk), comp in zip(K, comp_of):
    ACT.append((s, e, "n" if is_nccl(nm) else "c", comp, nm, st))
for s, e, st, b, kind in MC:
    ACT.append((s, e, "m", "memcpy", f"memcpy{kind}", st))
for s, e, st, b in MS:
    ACT.append((s, e, "m", "memset", "memset", st))
ACT.sort()
starts = [a[0] for a in ACT]


def window(t0, t1, label_fn=None, gap_label="idle"):
    """Partition [t0, t1): compute split by component, 'comm exposed', 'memcpy only', gap_label; busy sums; NCCL."""
    label_fn = label_fn or (lambda a: a[3])
    lo = bisect.bisect_left(starts, t0 - 10**10)          # activities can start before t0 (a long kernel)
    acts = [a for a in ACT[lo:bisect.bisect_right(starts, t1)] if a[1] > t0 and a[0] < t1]
    ev = []
    for i, a in enumerate(acts):
        ev.append((max(a[0], t0), 1, i))
        ev.append((min(a[1], t1), 0, i))
    ev.sort()
    part = collections.Counter()
    busy = collections.Counter()
    ncc = collections.defaultdict(lambda: [0, 0.0, 0.0])       # name -> count, time, overlapped by compute
    active = set()
    prev = t0
    for t, kind, i in ev:
        if t > prev:
            d = t - prev
            comp = [j for j in active if acts[j][2] == "c"]
            if comp:
                for j in comp:
                    part[label_fn(acts[j])] += d / len(comp)
            elif any(acts[j][2] == "n" for j in active):
                part["comm exposed (NCCL only)"] += d
            elif active:
                part["memcpy/memset only"] += d
            else:
                part[gap_label] += d
            for j in active:
                if acts[j][2] == "n":
                    ncc[acts[j][4]][2] += d if comp else 0.0
            prev = t
        if kind == 1:
            active.add(i)
        else:
            active.discard(i)
    for a in acts:
        d = min(a[1], t1) - max(a[0], t0)
        busy[label_fn(a) if a[2] == "c" else ("nccl" if a[2] == "n" else a[3])] += d
        if a[2] == "n":
            ncc[a[4]][0] += 1
            ncc[a[4]][1] += d
    ms = lambda cnt: {k: round(v / 1e6, 3) for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])}
    return {"wall_ms": round((t1 - t0) / 1e6, 3), "partition_ms": ms(part), "busy_ms": ms(busy),
            "nccl": {k: {"count": v[0], "ms": round(v[1] / 1e6, 3), "overlapped_ms": round(v[2] / 1e6, 3),
                         "exposed_ms": round((v[1] - v[2]) / 1e6, 3)} for k, v in ncc.items()},
            "kernels": sum(1 for a in acts if a[2] != "m")}


res = {"db": db, "main_tid": main, "marks": len(marks), "ranges": len(ranges), "kernels": len(K)}

# -- requests (optional req.py JSONL): how many were in flight at each round --------------------------------------------
t_epoch0 = None
if "TARGET_INFO_SESSION_START_TIME" in tables:
    r0 = rows("select * from TARGET_INFO_SESSION_START_TIME")
    t_epoch0 = r0[0][0] if r0 else None
reqs = []
if len(sys.argv) > 3 and t_epoch0:
    for line in open(sys.argv[3]):
        d = json.loads(line)
        reqs.append((d["t0"] * 1e9 - t_epoch0, (d["t0"] + d["wall"]) * 1e9 - t_epoch0, d["prompt_tokens"]))


def inflight(t):
    return sum(1 for a, b, _ in reqs if a <= t < b) if reqs else None


# -- pieces (pf_begin / pf_end marks), rounds, segments (runs of rounds separated by > 300 ms) -------------------------
begins = [t for t, n in marks if n == "pf_begin"]
ends = [t for t, n in marks if n == "pf_end"]
pieces = []
for b in begins:
    j = bisect.bisect_right(ends, b)
    if j < len(ends):
        pieces.append((b, ends[j]))
round_r = sorted((s, e) for s, e, n in ranges if n == "round")
verify_r = [(s, e) for s, e, n in ranges if n == "verify"]
sub = collections.defaultdict(list)
for s, e, n in ranges:
    sub[n].append((s, e))
segs = []
for r in round_r:
    if segs and r[0] - segs[-1][-1][1] < 300e6:
        segs[-1].append(r)
    else:
        segs.append([r])
PROBE = {"kda.proj", "kda.chain", "kda.o_proj", "dsa.proj", "dsa.attn", "dsa.indexer", "dsa.sparse_attn",
         "dsa.o_proj", "mlp.dense", "moe.router", "moe.routed", "moe.shared", "moe.combine", "hc", "embed", "head"}


def dec_label(a):
    """Decode: eager kernels launched under a probe site keep its name; graph-replayed / probe-less ones: by kernel name."""
    if a[2] == "n":
        return "nccl"
    cls = name_class(a[4])
    return a[3] if a[3] in PROBE and cls != "sampling" else f"~{cls}"


res["segments"] = []
for sg in segs:
    t0, t1 = sg[0][0], sg[-1][1]
    out_s = {"t0_ms": round(t0 / 1e6, 1), "t1_ms": round(t1 / 1e6, 1), "rounds": len(sg)}
    ps = [p for p in pieces if t0 <= p[0] < t1]
    out_s["pieces"] = len(ps)
    if len(ps) >= 2:
        pr = [r for r in sg if any(r[0] <= p[0] < r[1] for p in ps)]
        a0, a1 = pr[0][0], pr[-1][1]
        whole = window(a0, a1)
        inside = sum(window(p[0], p[1])["partition_ms"].get("idle", 0.0) for p in ps)
        whole["idle_inside_pieces_ms"] = round(inside, 3)
        whole["idle_outside_pieces_ms"] = round(whole["partition_ms"].get("idle", 0.0) - inside, 3)
        whole["pieces"] = len(ps)
        whole["piece_list"] = [dict(window(p[0], p[1]), t0_ms=round((p[0] - a0) / 1e6, 3)) for p in ps]
        out_s["prefill"] = whole
    dr = [r for r in sg if any(r[0] <= v[0] < r[1] for v in verify_r) and not any(r[0] <= p[0] < r[1] for p in pieces)]
    per = []
    for r in dr:
        w = window(r[0], r[1], label_fn=dec_label)
        for n in ("plan", "vfwd", "sample_multi", "sample_drafts", "propose", "mtp_chains", "mtp_multi", "verify"):
            w[f"host_{n}_ms"] = round(sum(min(e, r[1]) - max(s, r[0]) for s, e in sub.get(n, []) if s < r[1] and e > r[0]) / 1e6, 3)
        vf = [(s, e) for s, e in sub.get("vfwd", []) if r[0] <= s < r[1]]
        w["vfwd"] = window(vf[0][0], vf[-1][1], label_fn=dec_label) if vf else None
        w["inflight"] = inflight(r[0])
        w["t_ms"] = round(r[0] / 1e6, 1)
        per.append(w)
    out_s["decode_rounds"] = per
    res["segments"].append(out_s)

# -- overlap tax: compute kernels running beside an NCCL kernel vs the same kernel (name, grid) alone ------------------
nc_iv = sorted((s, e) for s, e, st, cid, nm, g, grid, blk in K if is_nccl(nm))
nc_s = [x[0] for x in nc_iv]


def nccl_overlap(s, e):
    i = max(0, bisect.bisect_right(nc_s, s) - 2)
    tot = 0
    while i < len(nc_iv) and nc_iv[i][0] < e:
        a, b = nc_iv[i]
        tot += max(0, min(b, e) - max(a, s))
        i += 1
    return tot


groups = collections.defaultdict(lambda: ([], []))
for (s, e, st, cid, nm, g, grid, blk), comp in zip(K, comp_of):
    if is_nccl(nm) or e <= s:
        continue
    ov = nccl_overlap(s, e) / (e - s)
    groups[(nm, grid, blk, comp)][0 if ov < 0.05 else 1].append((e - s, ov))
tax = collections.Counter()
tax_n = collections.Counter()
for (nm, grid, blk, comp), (alone, beside) in groups.items():
    if len(alone) < 5 or not beside:
        continue
    base = statistics.median(d for d, _ in alone)
    for d, ov in beside:
        tax[comp] += (d - base)
        tax_n[comp] += 1
res["overlap_tax_ms"] = {k: round(v / 1e6, 3) for k, v in sorted(tax.items(), key=lambda kv: -kv[1])}
res["overlap_tax_total_ms"] = round(sum(tax.values()) / 1e6, 3)
res["overlap_tax_kernels"] = sum(tax_n.values())

# -- kernel names: time and the components they were launched in (to name graph-replayed kernels) --------------------
kn = collections.defaultdict(lambda: [0, 0.0, collections.Counter()])
for (s, e, st, cid, nm, g, grid, blk), comp in zip(K, comp_of):
    kn[nm][0] += 1
    kn[nm][1] += (e - s) / 1e6
    kn[nm][2][comp] += (e - s) / 1e6
res["kernel_names"] = {k: {"count": v[0], "ms": round(v[1], 3), "class": name_class(k),
                           "by_component": {a: round(b, 3) for a, b in v[2].most_common(6)}}
                       for k, v in sorted(kn.items(), key=lambda kv: -kv[1][1])[:120]}
res["streams"] = dict(collections.Counter(st for _, _, st, *_ in K))
# NCCL kernel list (start, duration ns) in launch order, for the cross-rank skew
res["nccl_list"] = [(s, e - s) for s, e, st, cid, nm, g, grid, blk in K if is_nccl(nm)]
json.dump(res, open(out, "w"))
print(json.dumps({k: v for k, v in res.items() if k in ("main_tid", "marks", "ranges", "kernels", "streams")}))
for sg in res["segments"]:
    print("segment", sg["t0_ms"], sg["t1_ms"], "rounds", sg["rounds"], "pieces", sg["pieces"], "decode rounds", len(sg["decode_rounds"]))
    if "prefill" in sg:
        p = sg["prefill"]
        print(" prefill wall", p["wall_ms"], "pieces", p["pieces"], "idle in/out", p["idle_inside_pieces_ms"], p["idle_outside_pieces_ms"])
        for k, v in list(p["partition_ms"].items())[:12]:
            print(f"  {k:32s} {v:10.1f} {100 * v / p['wall_ms']:6.1f}%")
        print("  nccl:", p["nccl"])
    if sg["decode_rounds"]:
        ws = [w["wall_ms"] for w in sg["decode_rounds"]]
        print(" decode rounds: median wall", statistics.median(ws), "inflight", collections.Counter(w["inflight"] for w in sg["decode_rounds"]))
print("overlap tax ms", res["overlap_tax_total_ms"], res["overlap_tax_ms"])
