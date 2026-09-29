#!/usr/bin/env python3
"""THEORY-2: kernel-granularity critical path of W11's decode rounds (nsys sqlite exports of ~/w11-traces).

  crit.py R0.sqlite [R1.sqlite] OUT.json [--gap 1.5] [--layers 3]

Export first (local copy of the reports, never on a prod node):
  nsys export --type sqlite --output cap-r0.sqlite cap-r0.nsys-rep

Rounds: NVTX W7:round ranges holding a W7:verify and no W7:piece (w11dec.py's rule); segments split at > 1.5 s of
quiet (cap.sh sends requests 4 s apart). Kernels belong to a round by the host time of their launch call (runtime
correlation; a graph kernel by its cudaGraphLaunch), as in w11dec.py.

Per segment:
  - streams: kernels and kernel time per CUDA stream; time with >= 2 kernels in flight (true concurrency);
  - idle split by gap size (kernel end -> next kernel start on the device): < 1.5 us (launch / dependency gap),
    1.5-5, 5-20, 20-100, > 100 us; and "host-bound" idle: gaps where the next kernel's launch call returned after the
    previous kernel ended (the device waited for the host), for eager launches;
  - launches a round: kernels, graph-replayed vs eager, runtime calls (launch API) a round, host us per eager launch;
  - kernel size histogram (count and time a round);
  - the same by NVTX phase (verify forward / drafting / sampling / plan): host wall of the phase vs device busy;
  - a layer template: kernels between consecutive exchanges of the round nearest the segment's median wall
    (the kernel-granularity critical path), aggregated by exchange index;
With R1: exchange skew between the ranks: clocks aligned on the gather kernels' end times (both ranks leave an
all-gather within the transport time of each other), then per exchange which rank arrived last and by how much.
"""
import argparse
import bisect
import collections
import json
import re
import sqlite3
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("db0")
ap.add_argument("rest", nargs="+")
ap.add_argument("--gap", type=float, default=1.5)
ap.add_argument("--layers", type=int, default=4)
a = ap.parse_args()
db1 = a.rest[0] if len(a.rest) == 2 else None
out_path = a.rest[-1]

FAM = [
    ("experts", r"^(grouped_kernel|grouped_loop_kernel|grouped_epi_kernel|expert_kernel|gateup_epilogue_kernel|down_epilogue_kernel|rot_in1?_kernel|plan_kernel)$"),
    ("dense", r"^(_qmm|_fq4|_reduce|_group_sums|_swiglu|_fb16|chain_mm.*)$"),
    ("router", r"^(_router.*|_topk|_group|_combine.*|DeviceRadixSort.*|searchsorted.*|fill_reverse_indices_kernel|DeviceScanKernel|_select.*|_moe_.*)$"),
    ("kda", r"^(chain_kernel|_kda.*|_conv.*|_dconv.*|replay_layers_kernel|_dattn_kernel|.*kda.*)$"),
    ("attn", r"^(_lchunks|_lmerge|_lsparse.*|_expand.*|_absorb.*|_lwrite.*|_index.*|_pool_keys|_scores.*|gatherTopK|compute(BlockDigitCounts|DigitCumSum|BlockwiseWithinKCounts)|DeviceScanByKeyKernel|_gather_sel|_sparse.*|_latent.*|_mla.*|_rope.*|_q_.*|_kv_.*)$"),
    ("hc", r"^(_hc.*|_stream_mean|.*sinkhorn.*)$"),
    ("xchg", r"^gather_kernel$"),
    ("nccl", r"^nccl"),
    ("sample", r"(?i).*(argmax|softmax|topk|top_k|sort|sample|gumbel|multinomial|cumsum|max_kernel|reduce_kernel).*"),
    ("other", r".*"),
]
FAMC = [(n, re.compile(p)) for n, p in FAM]
_fc = {}


def fam(nm):
    f = _fc.get(nm)
    if f is None:
        f = _fc[nm] = next(n for n, p in FAMC if p.match(nm))
    return f


def load(db):
    c = sqlite3.connect(db)
    S = dict(c.execute("select id, value from StringIds"))
    RT = {}
    rtn = collections.Counter()
    for s, e, cid, nid, tid in c.execute("select start, end, correlationId, nameId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"):
        nm = S.get(nid, "")
        RT[cid] = (s, e, nm)
        rtn[nm] += 1
    K = []
    for s, e, n, cid, g, st, gx, gy, gz, bx, reg, ssm, dsm in c.execute(
            "select start, end, shortName, correlationId, coalesce(graphId, 0), streamId, gridX, gridY, gridZ, blockX, "
            "registersPerThread, staticSharedMemory, dynamicSharedMemory from CUPTI_ACTIVITY_KIND_KERNEL order by start"):
        r = RT.get(cid)
        K.append(dict(s=s, e=e, n=S.get(n, str(n)), cid=cid, g=g, st=st, grid=(gx, gy, gz), blk=bx,
                      ls=r[0] if r else None, le=r[1] if r else None, api=r[2] if r else "", reg=reg))
    NV = [(s, e, t or S.get(ti, ""), g) for s, e, t, ti, g in
          c.execute("select start, end, text, textId, globalTid from NVTX_EVENTS")]
    main = collections.Counter(g for s, e, t, g in NV if t.startswith("W7:") and e is not None).most_common(1)[0][0]
    R_ = collections.defaultdict(list)
    for s, e, t, g in NV:
        if e is not None and g == main and t.startswith("W7:"):
            R_[t[3:]].append((s, e))
    for v in R_.values():
        v.sort()
    return K, R_


def rounds_of(R_):
    pieces = R_.get("piece", [])
    vs = [v[0] for v in R_.get("verify", [])]
    rr = []
    for r in R_["round"]:
        i = bisect.bisect_left(vs, r[0])
        if i < len(vs) and vs[i] < r[1] and not any(r[0] <= p[0] < r[1] for p in pieces):
            rr.append(r)
    segs = []
    for r in rr:
        if segs and r[0] - segs[-1][-1][1] < a.gap * 1e9:
            segs[-1].append(r)
        else:
            segs.append([r])
    return segs


def assign(K, rounds):
    """kernels of each round by launch-call host time"""
    rs = [r[0] for r in rounds]
    per = [[] for _ in rounds]
    for k in K:
        t = k["ls"]
        if t is None:
            continue
        i = bisect.bisect_right(rs, t) - 1
        if i >= 0 and t < rounds[i][1]:
            per[i].append(k)
    return per


GB = [(1.5e3, "<1.5us"), (5e3, "1.5-5us"), (20e3, "5-20us"), (100e3, "20-100us"), (float("inf"), ">100us")]
KB = [(3e3, "<3us"), (10e3, "3-10us"), (50e3, "10-50us"), (200e3, "50-200us"), (float("inf"), ">=200us")]


def binlab(v, bins):
    return next(l for t, l in bins if v < t)


def phase_of(R_, t):
    for n in ("vfwd", "propose", "sample_multi", "plan"):
        iv = R_.get(n, [])
        i = bisect.bisect_right(iv, (t, float("inf"))) - 1
        if i >= 0 and iv[i][0] <= t < iv[i][1]:
            return n
    return "other"


def analyse(K, R_, label):
    segs = rounds_of(R_)
    out = []
    for si, seg in enumerate(segs):
        per = assign(K, seg)
        n = len(seg)
        if n < 5:
            continue
        acc = collections.defaultdict(float)
        stream_t = collections.Counter()
        stream_n = collections.Counter()
        walls = []
        ph_host = collections.defaultdict(float)
        ph_busy = collections.defaultdict(float)
        ph_idle_hb = collections.defaultdict(float)
        for (r0, r1), ks in zip(seg, per):
            if not ks:
                continue
            walls.append((r1 - r0) / 1e6)
            ks = sorted(ks, key=lambda k: k["s"])
            # concurrency: time with >= 2 kernels in flight
            ev = sorted([(k["s"], 1) for k in ks] + [(k["e"], -1) for k in ks])
            depth, last, conc, busy = 0, None, 0, 0
            for t, d in ev:
                if last is not None:
                    if depth >= 2:
                        conc += t - last
                    if depth >= 1:
                        busy += t - last
                depth += d
                last = t
            acc["conc_ms"] += conc / 1e6
            acc["busy_ms"] += busy / 1e6
            # device span of the round's kernels vs host range
            acc["dev_span_ms"] += (ks[-1]["e"] - ks[0]["s"]) / 1e6
            acc["kernels"] += len(ks)
            acc["graph_kernels"] += sum(1 for k in ks if k["g"])
            acc["eager_kernels"] += sum(1 for k in ks if not k["g"])
            eager = [k for k in ks if not k["g"] and k["le"] is not None]
            acc["eager_api_us"] += sum((k["le"] - k["ls"]) for k in eager) / 1e3
            acc["graph_launches"] += len({k["cid"] for k in ks if k["g"]})
            for k in ks:
                stream_t[k["st"]] += (k["e"] - k["s"]) / 1e6
                stream_n[k["st"]] += 1
                acc["k_" + binlab(k["e"] - k["s"], KB) + "_n"] += 1
                acc["k_" + binlab(k["e"] - k["s"], KB) + "_ms"] += (k["e"] - k["s"]) / 1e6
                acc["fam_" + fam(k["n"])] += (k["e"] - k["s"]) / 1e6
            # gaps on the device (single stream in practice; use the running max end)
            me = ks[0]["e"]
            for k in ks[1:]:
                gap = k["s"] - me
                if gap > 0:
                    lab = binlab(gap, GB)
                    acc["gap_" + lab + "_ms"] += gap / 1e6
                    acc["gap_" + lab + "_n"] += 1
                    ph = phase_of(R_, k["ls"]) if k["ls"] else "other"
                    # host-bound: the launch call of the next kernel returned after the device went idle
                    hb = k["le"] is not None and not k["g"] and k["le"] > me
                    if hb:
                        acc["gap_hostbound_ms"] += gap / 1e6
                        ph_idle_hb[ph] += gap / 1e6
                    acc["gapph_" + ph + "_ms"] += gap / 1e6
                me = max(me, k["e"])
            for k in ks:
                ph_busy[phase_of(R_, k["ls"])] += (k["e"] - k["s"]) / 1e6
        for nm in ("vfwd", "propose", "sample_multi", "plan"):
            for s, e in R_.get(nm, []):
                if seg[0][0] <= s < seg[-1][1]:
                    ph_host[nm] += (e - s) / 1e6
        res = dict(label=label, segment=si, rounds=n, wall_ms=statistics.mean(walls),
                   **{k: v / n for k, v in acc.items()},
                   streams={str(s): dict(kernels=stream_n[s] / n, ms=stream_t[s] / n) for s in stream_t},
                   phase_host_ms={k: v / n for k, v in ph_host.items()},
                   phase_busy_ms={k: v / n for k, v in ph_busy.items()},
                   phase_hostbound_idle_ms={k: v / n for k, v in ph_idle_hb.items()})
        res["idle_ms"] = res["wall_ms"] - res["busy_ms"]
        res["idle_by_next_launch"] = idle_attr(K, R_, seg)
        res["template"] = template(seg, per, walls)
        out.append(res)
    return out


def idle_attr(K, R_, seg):
    """device idle inside each round's host window, attributed by the next kernel's launch: its launch phase, and
    whether the device waited for the host (launch call returned after the device went idle) or not"""
    ks = [k for k in K if k["ls"] is not None and seg[0][0] - 5e6 <= k["s"] <= seg[-1][1] + 5e6]
    ks.sort(key=lambda k: k["s"])
    acc = collections.defaultdict(float)
    starts = [r[0] for r in seg]
    me = None
    for k in ks:
        if me is not None and k["s"] > me:
            # clip the idle interval to the round windows
            i = max(0, bisect.bisect_right(starts, me) - 1)
            while i < len(seg) and seg[i][0] < k["s"]:
                lo, hi = max(me, seg[i][0]), min(k["s"], seg[i][1])
                i += 1
                if hi > lo:
                    ph = phase_of(R_, k["ls"])
                    hb = k["le"] > me and not k["g"] or (k["g"] and k["ls"] > me)
                    big = "long(>20us)" if hi - lo > 20e3 else "short"
                    acc[f"{ph}|{'host' if hb else 'device'}|{big}"] += (hi - lo) / 1e6
        me = k["e"] if me is None else max(me, k["e"])
    return {k: round(v / len(seg), 3) for k, v in sorted(acc.items(), key=lambda kv: -kv[1])}


def template(seg, per, walls):
    """kernel sequence of one representative graph-replayed round (median wall), cut at exchanges"""
    med = statistics.median(walls)
    best = min(range(len(per)), key=lambda i: abs((seg[i][1] - seg[i][0]) / 1e6 - med) + (0 if per[i] and per[i][0]["g"] else 1e3))
    ks = sorted(per[best], key=lambda k: k["s"])
    blocks, cur, prev_e = [], [], ks[0]["s"]
    for k in ks:
        cur.append((k["n"], fam(k["n"]), round((k["e"] - k["s"]) / 1e3, 1), round((k["s"] - prev_e) / 1e3, 2), k["grid"]))
        prev_e = max(prev_e, k["e"])
        if k["n"] == "gather_kernel":
            blocks.append(cur)
            cur = []
    blocks.append(cur)
    summ = []
    for b in blocks:
        f = collections.Counter()
        for n, fm, d, g, gr in b:
            f[fm] += d
        summ.append(dict(kernels=len(b), us=round(sum(d for _, _, d, _, _ in b), 1),
                         gaps_us=round(sum(max(0, g) for _, _, _, g, _ in b), 1),
                         fam={k: round(v, 1) for k, v in f.most_common()}))
    return dict(round_wall_ms=(seg[best][1] - seg[best][0]) / 1e6, blocks=summ,
                first_layers=[b for b in blocks[: 2 * a.layers + 2]])


def skew(K0, K1):
    g0 = [k for k in K0 if k["n"] == "gather_kernel"]
    g1 = [k for k in K1 if k["n"] == "gather_kernel"]
    n = min(len(g0), len(g1))
    # the two lists are the same collectives in the same order if the counts agree (lockstep)
    res = dict(n0=len(g0), n1=len(g1))
    if abs(len(g0) - len(g1)) > 0:
        res["warning"] = "gather counts differ; matched by index up to the shorter list"
    offs = [g1[i]["e"] - g0[i]["e"] for i in range(n)]
    # the two hosts' clocks drift apart over the capture (a global offset spreads +-0.9 ms): a rolling median of the
    # end differences over +-100 exchanges (~2 rounds) is the local offset
    W = 100
    loc = []
    for i in range(n):
        w = offs[max(0, i - W): i + W + 1]
        loc.append(statistics.median(w))
    resid = [(offs[i] - loc[i]) / 1e3 for i in range(n)]
    res["end_offset_resid_us"] = dict(p5=statistics.quantiles(resid, n=20)[0], p95=statistics.quantiles(resid, n=20)[-1])
    d = [((g1[i]["s"] - loc[i]) - g0[i]["s"]) / 1e3 for i in range(n)]  # > 0: rank 1 arrived later
    dur0 = [(g0[i]["e"] - g0[i]["s"]) / 1e3 for i in range(n)]
    dur1 = [(g1[i]["e"] - g1[i]["s"]) / 1e3 for i in range(n)]
    res["arrival_skew_us"] = dict(median=statistics.median(d), mean=statistics.mean(d),
                                  mean_abs=statistics.mean(abs(x) for x in d),
                                  p10=statistics.quantiles(d, n=10)[0], p90=statistics.quantiles(d, n=10)[-1],
                                  r1_later_frac=sum(1 for x in d if x > 0) / n)
    res["gather_us"] = dict(r0_median=statistics.median(dur0), r0_mean=statistics.mean(dur0),
                            r1_median=statistics.median(dur1), r1_mean=statistics.mean(dur1),
                            min_of_pair_median=statistics.median(min(x, y) for x, y in zip(dur0, dur1)),
                            min_of_pair_mean=statistics.mean(min(x, y) for x, y in zip(dur0, dur1)))
    # by position inside a round-like run: which exchanges carry the skew? parity (attention vs FFN gathers)
    par = collections.defaultdict(list)
    for i in range(n):
        # previous kernel on rank 0 before this gather: what was computed just before (the producer)
        par["all"].append(d[i])
    # producer family before each gather on both ranks: skew grouped by the producer kernel's name on rank 0
    idx0 = {id(k): j for j, k in enumerate(K0)}
    by_prod = collections.defaultdict(list)
    for i in range(n):
        j = idx0[id(g0[i])]
        prod = K0[j - 1]["n"] if j else "-"
        by_prod[prod].append(d[i])
    res["skew_by_producer"] = {p: dict(n=len(v), mean_us=round(statistics.mean(v), 2),
                                       mean_abs_us=round(statistics.mean(abs(x) for x in v), 2))
                               for p, v in sorted(by_prod.items(), key=lambda kv: -len(kv[1]))[:12]}
    # waiting time a rank spends inside gathers beyond the pair minimum (= its share of the skew)
    res["wait_beyond_min_ms_total"] = dict(r0=sum(x - min(x, y) for x, y in zip(dur0, dur1)) / 1e3,
                                           r1=sum(y - min(x, y) for x, y in zip(dur0, dur1)) / 1e3)
    return res


K0, R0 = load(a.db0)
out = dict(r0=analyse(K0, R0, "r0"))
if db1:
    K1, R1 = load(db1)
    out["r1"] = analyse(K1, R1, "r1")
    out["skew"] = skew(K0, K1)
json.dump(out, open(out_path, "w"), indent=1, default=str)

for r in out["r0"] + out.get("r1", []):
    print(f"\n== {r['label']} segment {r['segment']}: {r['rounds']} rounds, wall {r['wall_ms']:.2f} ms, busy {r['busy_ms']:.2f}, "
          f"idle {r['idle_ms']:.2f}, >=2 kernels in flight {r['conc_ms']:.3f} ms")
    print(f"   kernels {r['kernels']:.0f} (graph {r['graph_kernels']:.0f}, eager {r['eager_kernels']:.0f}); graph launches "
          f"{r['graph_launches']:.1f}; host us in eager launch calls {r['eager_api_us']:.0f} "
          f"({r['eager_api_us'] / max(1, r['eager_kernels']):.2f} us each)")
    print("   streams: " + ", ".join(f"{s}: {v['kernels']:.0f} k / {v['ms']:.2f} ms" for s, v in r["streams"].items()))
    print("   gaps: " + ", ".join(f"{lab} {r.get('gap_' + lab + '_n', 0):.0f} x = {r.get('gap_' + lab + '_ms', 0):.3f} ms" for _, lab in GB)
          + f"; host-bound {r.get('gap_hostbound_ms', 0):.3f} ms")
    print("   gaps by launch phase: " + ", ".join(f"{p} {r.get('gapph_' + p + '_ms', 0):.3f}" for p in ("vfwd", "propose", "sample_multi", "plan", "other")))
    print("   kernel sizes: " + ", ".join(f"{lab} {r.get('k_' + lab + '_n', 0):.0f} x = {r.get('k_' + lab + '_ms', 0):.2f} ms" for _, lab in KB))
    print("   families ms: " + ", ".join(f"{k[4:]} {v:.2f}" for k, v in sorted(r.items(), key=lambda kv: -kv[1] if kv[0].startswith('fam_') else 0) if k.startswith("fam_")))
    print("   phase host ms: " + ", ".join(f"{k} {v:.2f}" for k, v in r["phase_host_ms"].items())
          + " | device busy by launch phase: " + ", ".join(f"{k} {v:.2f}" for k, v in r["phase_busy_ms"].items())
          + " | host-bound idle by phase: " + ", ".join(f"{k} {v:.3f}" for k, v in r["phase_hostbound_idle_ms"].items()))
    t = r["template"]
    print("   idle by the next kernel's launch (phase|who waited|size), ms a round: " + json.dumps(r["idle_by_next_launch"]))
    print(f"   template round {t['round_wall_ms']:.2f} ms, {len(t['blocks'])} blocks between exchanges; first blocks:")
    for i, b in enumerate(t["blocks"][: 2 * a.layers + 2]):
        print(f"     block {i}: {b['kernels']} kernels, {b['us']} us busy, {b['gaps_us']} us gaps: {b['fam']}")
if "skew" in out:
    print("\n== skew", json.dumps(out["skew"], indent=1))
