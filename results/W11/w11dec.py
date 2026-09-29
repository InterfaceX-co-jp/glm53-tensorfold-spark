#!/usr/bin/env python3
"""W11: decode rounds of one rank's nsys trace (sqlite export), by kernel family, phase, exchanges, idle and bytes.

  w11dec.py SQLITE OUT.json [--req cap.jsonl] [--kb kbench.json] [--mask rank0.json]

Rounds: NVTX ranges W7:round (Batcher._execute, results/W11/profile.py) that hold a W7:verify and no W7:piece, on the
serving thread. Segments: runs of rounds separated by > 1.5 s (cap.sh sends its requests 4 s apart). With --req
(rank 0's cap.jsonl) each round gets the number of requests in flight (host epoch clock = the trace's session start);
--mask (rank 0's output) applies rank 0's per-segment round selection to rank 1 by index (the ranks run the same
rounds in lockstep).

Per round: wall; kernel time per family and per launch phase (the NVTX range the launching runtime call sat in:
verify forward, drafting, sampling, other); the GPU-busy union, idle (wall - busy), idle inside the verify forward's
host range and outside it; exchanges (RoCE gather_kernel / NCCL): count, per-exchange us, exposed ms (no compute
beside); routed experts: grouped_kernel calls by rows R (grid.x = 8R + 1), distinct experts per verify layer U
(from the kernel time: kbench's single-GPU fit t(U) if given, else U = 8 t / t(R=1)), trellis bytes and GB/s; dense q4
(_qmm + _reduce): time, bytes by launch grid (kbench's grid -> shape table), GB/s; host time between rounds.
"""
import argparse
import bisect
import collections
import json
import re
import sqlite3
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("db")
ap.add_argument("out")
ap.add_argument("--req")
ap.add_argument("--kb")
ap.add_argument("--mask")
ap.add_argument("--gap", type=float, default=1.5)
a = ap.parse_args()

c = sqlite3.connect(a.db)
S = dict(c.execute("select id, value from StringIds"))
tables = {n for (n,) in c.execute("select name from sqlite_master where type='table'")}
kcols = [r[1] for r in c.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
gcol = "graphId" if "graphId" in kcols else ("graphNodeId" if "graphNodeId" in kcols else "0")
K = c.execute(f"select start, end, shortName, gridX, gridY, gridZ, correlationId, coalesce({gcol}, 0), streamId "
              "from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
K = [(s, e, S.get(n, str(n)), x, y, z, cid, g, st) for s, e, n, x, y, z, cid, g, st in K]
ks = [k[0] for k in K]
MC = c.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMCPY").fetchall() if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables else []
MS = c.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMSET").fetchall() if "CUPTI_ACTIVITY_KIND_MEMSET" in tables else []
MC = sorted(MC + MS)
mcs = [m[0] for m in MC]
RT = {cid: (s, tid) for s, tid, cid in c.execute("select start, globalTid, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME")}
NV = c.execute("select start, end, coalesce(text, ''), textId, globalTid from NVTX_EVENTS").fetchall()
NV = [(s, e, t or S.get(ti, ""), g) for s, e, t, ti, g in NV]
main = collections.Counter(g for s, e, t, g in NV if t.startswith("W7:") and e is not None).most_common(1)[0][0]
R_ = collections.defaultdict(list)
for s, e, t, g in NV:
    if e is not None and g == main and t.startswith("W7:"):
        R_[t[3:]].append((s, e))
for v in R_.values():
    v.sort()

FAM = [
    ("routed experts", r"^(grouped_kernel|grouped_loop_kernel|grouped_epi_kernel|expert_kernel|gateup_epilogue_kernel|down_epilogue_kernel|rot_in1?_kernel|plan_kernel)$"),
    ("dense q4 GEMV", r"^(_qmm|_fq4|_reduce|_group_sums|_swiglu|_fb16|chain_mm.*)$"),
    ("router / grouping / combine", r"^(_router.*|_topk|_group|_combine.*|DeviceRadixSort.*|searchsorted.*|fill_reverse_indices_kernel|DeviceScanKernel|_select.*|_moe_.*)$"),
    ("KDA", r"^(chain_kernel|_kda.*|_conv.*|_dconv.*|replay_layers_kernel|_dattn_kernel|.*kda.*)$"),
    ("attention + indexer", r"^(_lchunks|_lmerge|_lsparse.*|_expand.*|_absorb.*|_lwrite.*|_index.*|_pool_keys|_scores.*|gatherTopK|compute(BlockDigitCounts|DigitCumSum|BlockwiseWithinKCounts)|DeviceScanByKeyKernel|_gather_sel|_sparse.*|_latent.*|_mla.*|_rope.*|_q_.*|_kv_.*)$"),
    ("hc", r"^(_hc.*|_stream_mean|.*sinkhorn.*)$"),
    ("RoCE all-gather", r"^gather_kernel$"),
    ("NCCL", r"^nccl"),
    ("sampling / argmax / softmax", r"(?i).*(argmax|softmax|topk|top_k|sort|sample|gumbel|multinomial|cumsum|max_kernel|reduce_kernel).*"),
    ("norms / elementwise / other", r".*"),
]
FAMC = [(n, re.compile(p)) for n, p in FAM]
_fam_cache = {}


def fam(nm):
    f = _fam_cache.get(nm)
    if f is None:
        f = next(n for n, p in FAMC if p.match(nm))
        _fam_cache[nm] = f
    return f


COMM = ("RoCE all-gather", "NCCL")
PH = [("vfwd", "verify forward"), ("propose", "drafting"), ("mtp_chains", "drafting"), ("mtp_multi", "drafting"),
      ("sample_multi", "sampling"), ("sample_drafts", "sampling"), ("plan", "plan")]
ph_iv = sorted((s, e, lab) for n, lab in PH for s, e in R_.get(n, []))
ph_s = [p[0] for p in ph_iv]


def phase(t):
    """innermost (latest-starting) phase range holding t"""
    i = bisect.bisect_right(ph_s, t) - 1
    n = 0
    while i >= 0 and n < 40:
        s, e, lab = ph_iv[i]
        if s <= t < e:
            return lab
        i -= 1
        n += 1
    return "round other"


def union(iv):
    iv = sorted(iv)
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


# -- rounds and segments --------------------------------------------------------------------------------------------
pieces = R_.get("piece", [])
verifies = R_.get("verify", [])
vs = [v[0] for v in verifies]
rounds = []
for r in R_["round"]:
    i = bisect.bisect_left(vs, r[0])
    if i < len(vs) and vs[i] < r[1] and not any(r[0] <= p[0] < r[1] for p in pieces):
        rounds.append(r)
segs = []
for r in rounds:
    if segs and r[0] - segs[-1][-1][1] < a.gap * 1e9:
        segs[-1].append(r)
    else:
        segs.append([r])

t_epoch0 = None
if "TARGET_INFO_SESSION_START_TIME" in tables:
    row = c.execute("select * from TARGET_INFO_SESSION_START_TIME").fetchone()
    t_epoch0 = row[0] if row else None
reqs = []
if a.req:
    for line in open(a.req):
        d = json.loads(line)
        reqs.append(((d["t0"]) * 1e9 - t_epoch0, (d["t0"] + d["wall"]) * 1e9 - t_epoch0, d.get("name"), d.get("cat")))
mask = json.load(open(a.mask))["masks"] if a.mask else None

# -- calibrations ---------------------------------------------------------------------------------------------------
MAT = 4096 * 1024 // 2
kb = json.load(open(a.kb)) if a.kb else []
fitU = {}
ex_kb = [r for r in kb if r.get("kind") == "expert"]
if ex_kb:
    # per step, linear fit t = t0 + t1 U over kbench's (U, us)
    for step in ("gu", "dn"):
        xs = [r["U"] for r in ex_kb]
        ys = [r["us"][step] for r in ex_kb]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        fitU[step] = (my - b * mx, b)
dense_map = collections.defaultdict(list)          # (gridY, gridZ) -> [(name, bytes)]
# default: every decode q4 shape of a rank (prepared manifest) with qmm.split_k's slices (SHAPE_SK / SPLIT_TARGET 192)
SHAPE_SK = {"12576x4096": 4, "4096x4096": 2, "2048x4096": 4, "8192x1536": 4, "8192x512": 4, "4096x8192": 8}


def split_k(n, k):
    if f"{n}x{k}" in SHAPE_SK:
        return SHAPE_SK[f"{n}x{k}"]
    tiles, groups, sk = -(-n // 64), k // 64, 1
    while sk < 8 and tiles * sk < 192 and groups % (sk * 2) == 0 and groups // (sk * 2) >= 8:
        sk *= 2
    return sk


# (name, n, k, matrices a verify forward + MTP layer): an ambiguous grid gets the count-weighted mean bytes
for nm, n, k, cnt in (("head", 77440, 4096, 1), ("kda.proj", 12576, 4096, 34), ("mlp.gu", 12288, 4096, 3),
                      ("dsa.o/mtp.eh", 4096, 8192, 13), ("mlp.down", 4096, 6144, 3), ("kda.o", 4096, 4096, 34),
                      ("dsa.q_b", 8192, 1536, 12), ("shared.gu/dsa.proj", 2048, 4096, 55), ("dsa.index.qb", 4096, 1536, 12),
                      ("shared.down", 4096, 1024, 43), ("dsa.kv_k/v", 8192, 512, 24), ("index.kw", 160, 4096, 12),
                      ("kda.fb/gb", 4096, 128, 68)):
    dense_map[(-(-n // 64), split_k(n, k))].append((nm, n * k * 9 // 16, cnt))
for r in kb:
    if r.get("kind") == "dense":
        key = (r["grid"][1], r["grid"][2])
        ent = (r["name"], r["n"] * r["k"] * 9 // 16, 1)
        if not any(b == ent[1] for _, b, _ in dense_map[key]):
            dense_map[key].append(ent)

# R = 1 calibration from the trace (MTP steps: exactly 8 experts)
t1 = {}
for step, gy in (("gu", 8), ("dn", 32)):
    v = [(e - s) / 1e3 for s, e, n, x, y, z, *_ in K if n == "grouped_kernel" and x == 9 and y == gy]
    t1[step] = statistics.median(v) if v else None


def u_of(step, us, R):
    if fitU:
        t0, b = fitU[step]
        return max(1.0, min(8 * R, (us - t0) / b))
    return 8 * us / t1[step] if t1[step] else float("nan")


# -- per round ------------------------------------------------------------------------------------------------------
def kernels_in(t0, t1_):
    lo = bisect.bisect_left(ks, t0 - int(5e7))
    out = []
    for k in K[lo:]:
        if k[0] >= t1_:
            break
        if k[1] > t0:
            out.append(k)
    return out


def one_round(r, nxt):
    t0, t1_ = r
    ks_ = kernels_in(t0, t1_)
    famt, pht = collections.Counter(), collections.Counter()
    comp_iv, comm_iv, all_iv = [], [], []
    comm = collections.defaultdict(list)
    gk = []
    dense_t, dense_b, dense_unk = 0.0, 0.0, 0.0
    graph_t = 0.0
    for s, e, n, x, y, z, cid, g, st in ks_:
        s2, e2 = max(s, t0), min(e, t1_)
        d = (e2 - s2) / 1e6
        f = fam(n)
        famt[f] += d
        rt = RT.get(cid)
        pht[phase(rt[0]) if rt and rt[1] == main else "other thread"] += d
        if g:
            graph_t += d
        if f in COMM:
            comm_iv.append((s2, e2))
            comm[f].append((e - s) / 1e3)
        else:
            comp_iv.append((s2, e2))
        all_iv.append((s2, e2))
        if n == "grouped_kernel" and y in (8, 32):
            R = (x - 1) // 8
            gk.append(("gu" if y == 8 else "dn", R, (e - s) / 1e3, phase(rt[0]) if rt else "?"))
        if n == "_qmm":
            cand = dense_map.get((y, z))
            dense_t += d
            if cand:
                dense_b += sum(b * w for _, b, w in cand) / sum(w for _, _, w in cand)
            else:
                dense_unk += d
        elif n == "_reduce":
            dense_t += d
    lo = bisect.bisect_left(mcs, t0 - int(5e7))
    for s, e in MC[lo:]:
        if s >= t1_:
            break
        if e > t0:
            all_iv.append((max(s, t0), min(e, t1_)))
            famt["memcpy / memset"] += (min(e, t1_) - max(s, t0)) / 1e6
    wall = (t1_ - t0) / 1e6
    busy = union(all_iv) / 1e6
    comp_busy = union(comp_iv) / 1e6
    comm_exposed = (union(comp_iv + comm_iv) - union(comp_iv)) / 1e6
    # idle inside the verify forward's host range(s) of this round vs the rest of the round
    vf = [(max(s, t0), min(e, t1_)) for s, e in R_.get("vfwd", []) if s < t1_ and e > t0]
    idle_vf = 0.0
    for s, e in vf:
        inside = [(max(x0, s), min(x1, e)) for x0, x1 in all_iv if x1 > s and x0 < e]
        idle_vf += ((e - s) - union(inside)) / 1e6
    host = {}
    for n in ("vfwd", "propose", "mtp_chains", "mtp_multi", "sample_multi", "sample_drafts", "verify"):
        host[n] = sum(min(e, t1_) - max(s, t0) for s, e in R_.get(n, []) if s < t1_ and e > t0) / 1e6
    plan_before = sum(e - s for s, e in R_.get("plan", []) if nxt and t1_ <= s < nxt[0]) / 1e6
    # experts: verify calls (R > 1 or launched in the verify forward) vs drafting calls (R = 1 outside it)
    ver = [g for g in gk if g[3] == "verify forward"]
    drf = [g for g in gk if g[3] != "verify forward"]
    reads = 0.0
    ex_bytes = 0.0
    Uv = []
    for step, R, us, _ in gk:
        U = u_of(step, us, R)
        reads += U / 2
        ex_bytes += U * MAT * (2 if step == "gu" else 1)
    for step, R, us, _ in ver:
        if step == "gu":
            Uv.append(u_of(step, us, R))
    gk_t = sum(us for *_, us, _ in [(g[0], g[1], g[2], g[3]) for g in gk]) / 1e3
    rows_v = max([g[1] for g in ver], default=0)
    return {
        "t0_ms": round(t0 / 1e6, 3), "wall_ms": round(wall, 3), "gpu_busy_ms": round(busy, 3),
        "compute_busy_ms": round(comp_busy, 3), "idle_ms": round(wall - busy, 3), "idle_in_vfwd_ms": round(idle_vf, 3),
        "idle_outside_vfwd_ms": round(wall - busy - idle_vf, 3), "comm_exposed_ms": round(comm_exposed, 3),
        "host_ms": {k: round(v, 3) for k, v in host.items()},
        "gap_to_next_ms": round((nxt[0] - t1_) / 1e6, 3) if nxt else None, "plan_between_ms": round(plan_before, 3),
        "family_ms": {k: round(v, 3) for k, v in famt.most_common()},
        "phase_ms": {k: round(v, 3) for k, v in pht.most_common()},
        "graph_kernel_ms": round(graph_t, 3), "kernels": len(ks_),
        "comm": {k: {"count": len(v), "median_us": round(statistics.median(v), 2), "mean_us": round(statistics.mean(v), 2),
                     "sum_ms": round(sum(v) / 1e3, 3)} for k, v in comm.items()},
        "experts": {"grouped_ms": round(gk_t, 3), "calls": len(gk), "verify_rows": rows_v,
                    "U_verify_layer": round(statistics.mean(Uv), 2) if Uv else None,
                    "expert_layer_reads": round(reads, 1), "bytes_gb": round(ex_bytes / 1e9, 3),
                    "grouped_gbs": round(ex_bytes / gk_t / 1e6, 1) if gk_t else None,
                    "verify_calls": len(ver), "draft_calls": len(drf)},
        "dense": {"ms": round(dense_t, 3), "bytes_gb": round(dense_b / 1e9, 3), "unmapped_ms": round(dense_unk, 3),
                  "gbs": round(dense_b / 1e6 / (dense_t - dense_unk), 1) if dense_t > dense_unk and dense_b else None},
    }


def mean_dict(ds):
    keys = []
    for d in ds:
        for k in d:
            if k not in keys:
                keys.append(k)
    return {k: round(statistics.mean(d.get(k, 0.0) for d in ds), 3) for k in keys}


def summarize(per):
    if not per:
        return None
    m = {k: round(statistics.mean(p[k] for p in per), 3) for k in
         ("wall_ms", "gpu_busy_ms", "compute_busy_ms", "idle_ms", "idle_in_vfwd_ms", "idle_outside_vfwd_ms",
          "comm_exposed_ms", "graph_kernel_ms", "kernels")}
    gaps = [p["gap_to_next_ms"] for p in per if p["gap_to_next_ms"] is not None and p["gap_to_next_ms"] < 500]
    m["gap_to_next_ms"] = round(statistics.mean(gaps), 3) if gaps else None
    m["median_wall_ms"] = round(statistics.median(p["wall_ms"] for p in per), 3)
    m["family_ms"] = dict(sorted(mean_dict([p["family_ms"] for p in per]).items(), key=lambda kv: -kv[1]))
    m["phase_ms"] = dict(sorted(mean_dict([p["phase_ms"] for p in per]).items(), key=lambda kv: -kv[1]))
    m["host_ms"] = mean_dict([p["host_ms"] for p in per])
    cm = {}
    for f in COMM:
        xs = [p["comm"][f] for p in per if f in p["comm"]]
        if xs:
            cm[f] = {"count": round(statistics.mean(x["count"] for x in xs), 1),
                     "median_us": round(statistics.median(x["median_us"] for x in xs), 2),
                     "mean_us": round(statistics.mean(x["mean_us"] for x in xs), 2),
                     "sum_ms": round(statistics.mean(x["sum_ms"] for x in xs), 3)}
    m["comm"] = cm
    ex = [p["experts"] for p in per]
    m["experts"] = {k: round(statistics.mean(e[k] for e in ex if e[k] is not None), 3)
                    for k in ex[0] if any(e[k] is not None for e in ex)}
    de = [p["dense"] for p in per]
    m["dense"] = {k: round(statistics.mean(d[k] for d in de if d[k] is not None), 3)
                  for k in de[0] if any(d[k] is not None for d in de)}
    graph_rounds = sum(1 for p in per if p["graph_kernel_ms"] > 0.5 * (p["gpu_busy_ms"] or 1))
    m["rounds_mostly_graph"] = graph_rounds
    return m


res = {"db": a.db, "main_tid": main, "t1_calibration_us": t1, "fitU": fitU, "segments": [], "masks": []}
names = []
if reqs:
    order = sorted(reqs)
    for s in segs:
        hit = [q for q in order if q[0] - 2e9 <= s[0][0] <= q[1]]
        names.append(f"{hit[0][3]}:{hit[0][2]}" if hit else "?")
for i, sg in enumerate(segs):
    per = []
    sel = []
    for j, r in enumerate(sg):
        nxt = sg[j + 1] if j + 1 < len(sg) else None
        w = one_round(r, nxt)
        if reqs:
            w["inflight"] = sum(1 for q in reqs if q[0] <= r[0] < q[1])
        per.append(w)
    if mask is not None and i < len(mask):
        keep = mask[i]
    elif reqs:
        mx = max(w["inflight"] for w in per)
        keep = [j for j, w in enumerate(per) if w["inflight"] == mx]
    else:
        keep = list(range(len(per)))
    res["masks"].append(keep)
    sel = [per[j] for j in keep if j < len(per)]
    seg = {"index": i, "name": names[i] if i < len(names) else None, "rounds": len(sg), "kept": len(sel),
           "t0_ms": round(sg[0][0] / 1e6, 1), "t1_ms": round(sg[-1][1] / 1e6, 1), "mean": summarize(sel),
           "rounds_detail": per}
    if sel:
        ws = sorted(range(len(sel)), key=lambda k: sel[k]["wall_ms"])
        seg["median_round"] = sel[ws[len(ws) // 2]]
    res["segments"].append(seg)
    m = seg["mean"] or {}
    print(f"segment {i} {seg['name']}: rounds {len(sg)} kept {len(sel)} wall {m.get('wall_ms')} (median "
          f"{m.get('median_wall_ms')}) busy {m.get('gpu_busy_ms')} idle {m.get('idle_ms')} (in vfwd "
          f"{m.get('idle_in_vfwd_ms')}) comm exposed {m.get('comm_exposed_ms')} gap {m.get('gap_to_next_ms')}")
    if m:
        print("   families", json.dumps(m["family_ms"]))
        print("   phases  ", json.dumps(m["phase_ms"]), "host", json.dumps(m["host_ms"]))
        print("   comm    ", json.dumps(m["comm"]))
        print("   experts ", json.dumps(m["experts"]), "dense", json.dumps(m["dense"]))
# unknown kernel names (the catch-all family) by time, to refine FAM
other = collections.Counter()
for s, e, n, *_ in K:
    if fam(n) == "norms / elementwise / other":
        other[n] += (e - s) / 1e6
res["other_kernels_ms"] = dict(other.most_common(40))
print("other kernels (ms, whole trace):", json.dumps(dict(other.most_common(15))))
res["t1_calibration_us"] = t1
json.dump(res, open(a.out, "w"))
