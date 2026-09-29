#!/usr/bin/env python3
"""W7: markdown tables from analyze.py / dec.py / skew.py outputs (run in the directory holding them).
  report.py > tables.md"""
import json
import statistics

GROUPS = [
    ("Routed experts (fat kernels, rot_in, plan)", lambda k: k == "moe.routed"),
    ("MoE router / grouping + combine", lambda k: k in ("moe.router", "moe.combine")),
    ("Shared expert + dense MLP GEMMs", lambda k: k in ("moe.shared", "mlp.dense")),
    ("KDA: projections (q4 GEMMs)", lambda k: k in ("kda.proj", "kda.o_proj")),
    ("KDA: chunked recurrence", lambda k: k == "kda.chain"),
    ("DSA/MLA: q/kv proj + absorb", lambda k: k in ("dsa.proj",)),
    ("DSA/MLA: indexer + top-k", lambda k: k == "dsa.indexer"),
    ("DSA/MLA: sparse + dense attention", lambda k: k in ("dsa.sparse_attn", "dsa.attn")),
    ("DSA/MLA: latent expand + o_proj", lambda k: k == "dsa.o_proj"),
    ("Hyper-connections (hc_post / pre / Sinkhorn)", lambda k: k in ("hc", "embed")),
    ("MTP prefill cache rows", lambda k: k.startswith("mtp.")),
    ("DFlash2 drafter taps", lambda k: k == "drafter"),
    ("NCCL exposed (nothing else running)", lambda k: k == "comm exposed (NCCL only)"),
    ("memcpy / memset only", lambda k: k == "memcpy/memset only"),
    ("GPU idle (host / Python gaps)", lambda k: k == "idle"),
]


def group(part):
    out = {g: 0.0 for g, _ in GROUPS}
    out["other (head, sample, stage, commit, piece setup)"] = 0.0
    for k, v in part.items():
        for g, f in GROUPS:
            if f(k):
                out[g] += v
                break
        else:
            out["other (head, sample, stage, commit, piece setup)"] += v
    return out


def prefill_table(title, cols):
    """cols: [(label, window dict)]"""
    print(f"\n#### {title}\n")
    print("| component | " + " | ".join(f"{l} ms | {l} %" for l, _ in cols) + " |")
    print("| --- |" + " ---: | ---: |" * len(cols))
    gs = [group(w["partition_ms"]) for _, w in cols]
    for g in gs[0]:
        print(f"| {g} | " + " | ".join(f"{x[g]:,.1f} | {100 * x[g] / w['wall_ms']:.1f}" for x, (_, w) in zip(gs, cols)) + " |")
    print("| **wall** | " + " | ".join(f"**{w['wall_ms']:,.1f}** | 100" for _, w in cols) + " |")


def nccl_line(label, w, tax=None):
    n = next(iter(w["nccl"].values()), None)
    if not n:
        return
    s = (f"| {label} | {n['count']:,} | {n['ms']:,.1f} | {n['overlapped_ms']:,.1f} | {n['exposed_ms']:,.1f} | "
         f"{100 * n['exposed_ms'] / w['wall_ms']:.1f} |")
    if tax is not None:
        s += f" {tax:,.1f} | {100 * tax / w['wall_ms']:.1f} |"
    print(s)


A = {k: json.load(open(f"{k}.json")) for k in ("pf24-r0", "pf24-r1", "mix-r0", "mix-r1")}
pf24 = {r: A[f"pf24-r{r}"]["segments"][0]["prefill"] for r in (0, 1)}
pf98 = {r: A[f"mix-r{r}"]["segments"][0]["prefill"] for r in (0, 1)}
print("### Prefill: whole prompts (nsys, both ranks)")
prefill_table("21,464-token prompt ('24.5k' cell) and 85,781-token prompt ('98k' cell); partition of the wall time",
              [("24.5k r0", pf24[0]), ("24.5k r1", pf24[1]), ("98k r0", pf98[0]), ("98k r1", pf98[1])])
# one real 2,048-row piece: the middle piece of the 24.5k prompt, and the last full piece of the 98k one
mid = len(pf24[0]["piece_list"]) // 2
last = len(pf98[0]["piece_list"]) - 2
prefill_table(f"One 2,048-row piece: piece {mid + 1}/{len(pf24[0]['piece_list'])} of the 24.5k prompt (context "
              f"{mid * 2048:,}) and piece {last + 1}/{len(pf98[0]['piece_list'])} of the 98k prompt (context {last * 2048:,})",
              [("24.5k r0", pf24[0]["piece_list"][mid]), ("24.5k r1", pf24[1]["piece_list"][mid]),
               ("98k r0", pf98[0]["piece_list"][last]), ("98k r1", pf98[1]["piece_list"][last])])
print("\n#### NCCL all-gathers in prefill (all are `ncclDevKernel_AllGather_RING_LL`, bf16 partials of 512-row sub-blocks)\n")
print("| window | count | kernel ms | overlapped by compute ms | exposed ms | exposed % of wall | overlap tax ms | tax % |")
print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for lab, key, w in (("24.5k r0", "pf24-r0", pf24[0]), ("24.5k r1", "pf24-r1", pf24[1]),
                    ("98k r0", "mix-r0", pf98[0]), ("98k r1", "mix-r1", pf98[1])):
    nccl_line(lab, w, A[key]["overlap_tax_total_ms"] if key.startswith("pf24") else A[key]["overlap_tax_total_ms"])
print("\nIdle split (ms): " + "; ".join(
    f"{lab}: inside pieces {w['idle_inside_pieces_ms']:.0f}, between pieces {w['idle_outside_pieces_ms']:.0f}"
    for lab, w in (("24.5k r0", pf24[0]), ("24.5k r1", pf24[1]), ("98k r0", pf98[0]), ("98k r1", pf98[1]))))

print("\n### Decode rounds (nsys, kernel time per round, mean over rounds)\n")
D = {k: json.load(open(f"{k}.json")) for k in ("dec1-r0", "dec1-r1", "dec4-r0", "dec4-r1")}
fams = []
for d in D.values():
    for f in d["mean"]["family_ms"]:
        if f not in fams:
            fams.append(f)
cols = [("1 stream r0", D["dec1-r0"]), ("1 stream r1", D["dec1-r1"]), ("4 streams r0", D["dec4-r0"]), ("4 streams r1", D["dec4-r1"])]
print("| kernel family | " + " | ".join(f"{l} ms | %" for l, _ in cols) + " |")
print("| --- |" + " ---: | ---: |" * len(cols))
for f in fams:
    print(f"| {f} | " + " | ".join(f"{d['mean']['family_ms'].get(f, 0):.2f} | {100 * d['mean']['family_ms'].get(f, 0) / d['mean']['wall_ms']:.1f}" for _, d in cols) + " |")
for key, lab in (("gpu_busy_ms", "GPU busy (union)"), ("nccl_exposed_ms", "NCCL exposed"), ("idle_ms", "GPU idle"),
                 ("wall_ms", "**round wall (under capture)**"), ("nccl_count", "all-gathers a round"), ("kernels", "kernels a round")):
    print(f"| {lab} | " + " | ".join(f"{d['mean'][key]:.2f} | {100 * d['mean'][key] / d['mean']['wall_ms']:.1f}" if key not in ("nccl_count", "kernels") else f"{d['mean'][key]:.0f} | " for _, d in cols) + " |")
print("\nBy phase (kernel ms a round, by where the kernel was launched):\n")
phs = []
for d in D.values():
    for f in d["mean"]["phase_ms"]:
        if f not in phs:
            phs.append(f)
print("| phase | " + " | ".join(l for l, _ in cols) + " |")
print("| --- |" + " ---: |" * len(cols))
for f in phs:
    print(f"| {f} | " + " | ".join(f"{d['mean']['phase_ms'].get(f, 0):.2f}" for _, d in cols) + " |")
