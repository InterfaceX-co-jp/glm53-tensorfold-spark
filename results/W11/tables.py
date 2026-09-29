#!/usr/bin/env python3
"""W11: the decode round tables (markdown) from w11dec.py outputs, probe.cu and kbench.py logs, and the control /
capture request stats.

  tables.py TRACEDIR > tables.md      (TRACEDIR holds cap-r0.json, cap-r1.json, cap4-r0.json, cap4-r1.json)"""
import json
import statistics
import sys
from pathlib import Path

T = Path(sys.argv[1])
H = Path(__file__).parent
CEIL_EX, CEIL_DENSE = 235.0, 230.5          # probe.cu: gathered expert reads a round / the verify's dense set


def load(n):
    return json.load(open(T / f"{n}.json"))


def seg_rounds(d, idx):
    s = d["segments"][idx]
    return [s["rounds_detail"][j] for j in d["masks"][idx]]


r0, r1 = load("cap-r0"), load("cap-r1")
q0, q1 = load("cap4-r0"), load("cap4-r1")
W = {
    "1 stream, prose (chat 256 + essay 384 + chat 256, greedy)": seg_rounds(r0, 0) + seg_rounds(r0, 1) + seg_rounds(q0, 1),
    "1 stream, code (LRU module 384, greedy)": seg_rounds(r0, 2),
    "4 streams, prose (4 x 384, greedy)": seg_rounds(q0, 0),
}
W1 = {
    "1 stream, prose (chat 256 + essay 384 + chat 256, greedy)": seg_rounds(r1, 0) + seg_rounds(r1, 1) + seg_rounds(q1, 1),
    "1 stream, code (LRU module 384, greedy)": seg_rounds(r1, 2),
    "4 streams, prose (4 x 384, greedy)": seg_rounds(q1, 0),
}


def m(per, f):
    return statistics.mean(f(p) for p in per)


def fam(p, k):
    return p["family_ms"].get(k, 0.0)


def comm(p, k, f):
    c = p["comm"].get(k)
    return c[f] if c else 0.0


def ctl_ms(fn, names):
    """uncaptured (and captured) ms a round from the engine's round_kinds verify_ms + draft_ms / rounds"""
    rows = [json.loads(line) for line in open(H / fn)]
    rows = [r for r in rows if r["name"] in names]
    v = sum(r["tf"]["round_kinds"].get("verify_ms", 0) + r["tf"]["round_kinds"].get("draft_ms", 0) for r in rows)
    n = sum(r["tf"]["rounds"] for r in rows)
    tok = sum(r["tokens"] - 1 for r in rows)
    dec = sum(r["tf"]["decode_s"] for r in rows)
    return v / n, n, tok / n, 1e3 * dec / n


out = []
p = out.append
p("## Round breakdown, rank 0 (mean over rounds; rank 1 wall in the last row)\n")
cols = list(W)
p("| | " + " | ".join(cols) + " |")
p("| --- | " + " | ".join("---:" for _ in cols) + " |")


def row(label, f, fmt="{:.2f}"):
    p(f"| {label} | " + " | ".join(fmt.format(m(W[c], f)) for c in cols) + " |")


p("| rounds (count) | " + " | ".join(str(len(W[c])) for c in cols) + " |")
row("**round wall, ms (under capture)**", lambda x: x["wall_ms"])
row("verify rows (largest window a round)", lambda x: x["experts"]["verify_rows"])
row("routed experts, ms (grouped_kernel + epilogues / rot_in)", lambda x: fam(x, "routed experts"))
row("  grouped_kernel ms", lambda x: x["experts"]["grouped_ms"])
row("  distinct experts a verify layer U", lambda x: x["experts"]["U_verify_layer"] or 0)
row("  expert-layer reads a round", lambda x: x["experts"]["expert_layer_reads"], "{:.0f}")
row("  trellis GB a round (rank)", lambda x: x["experts"]["bytes_gb"])
p("| grouped_kernel GB/s | " + " | ".join(f"{sum(q['experts']['bytes_gb'] for q in W[c]) / sum(q['experts']['grouped_ms'] for q in W[c]) * 1e3:.0f}" for c in cols) + " |")
row("dense q4 GEMV ms (_qmm + _reduce + _swiglu)", lambda x: fam(x, "dense q4 GEMV"))
row("  q4 GB a round (rank)", lambda x: x["dense"]["bytes_gb"])
p("| _qmm + _reduce GB/s | " + " | ".join(f"{sum(q['dense']['bytes_gb'] for q in W[c]) / sum(q['dense']['ms'] - q['dense']['unmapped_ms'] for q in W[c]) * 1e3:.0f}" for c in cols) + " |")
row("RoCE all-gathers a round", lambda x: comm(x, "RoCE all-gather", "count"), "{:.1f}")
p("| RoCE all-gather us, median (mean) | " + " | ".join(
    f"{statistics.median(comm(q, 'RoCE all-gather', 'median_us') for q in W[c]):.1f} ({m(W[c], lambda x: comm(x, 'RoCE all-gather', 'mean_us')):.1f})" for c in cols) + " |")
row("RoCE all-gather kernel ms", lambda x: fam(x, "RoCE all-gather"))
row("NCCL (control exchanges: 2 a round) ms", lambda x: fam(x, "NCCL"))
row("exchanges exposed (nothing else running), ms", lambda x: x["comm_exposed_ms"])
row("DSA attention + indexer ms", lambda x: fam(x, "attention + indexer"))
row("KDA ms", lambda x: fam(x, "KDA"))
row("hc ms", lambda x: fam(x, "hc"))
row("router / grouping / combine ms", lambda x: fam(x, "router / grouping / combine"))
row("norms / elementwise / memcpy / sampling ms", lambda x: fam(x, "norms / elementwise / other") + fam(x, "memcpy / memset") + fam(x, "sampling / argmax / softmax"))
row("GPU busy (union) ms", lambda x: x["gpu_busy_ms"])
row("**GPU idle inside the round, ms**", lambda x: x["idle_ms"])
row("  of it inside the verify forward's host range", lambda x: x["idle_in_vfwd_ms"])
row("host gap between rounds, ms", lambda x: x["gap_to_next_ms"] if x["gap_to_next_ms"] is not None and x["gap_to_next_ms"] < 500 else 0)
row("kernels a round", lambda x: x["kernels"], "{:.0f}")
row("by launch phase: verify forward ms", lambda x: x["phase_ms"].get("verify forward", 0))
row("by launch phase: drafting (MTP / DFlash2) ms", lambda x: x["phase_ms"].get("drafting", 0))
row("by launch phase: sampling ms", lambda x: x["phase_ms"].get("sampling", 0))
row("by launch phase: round other ms", lambda x: x["phase_ms"].get("round other", 0))
row("host: drafting ranges (propose + mtp_chains + mtp_multi) ms", lambda x: x["host_ms"]["propose"] + x["host_ms"]["mtp_chains"] + x["host_ms"]["mtp_multi"])
row("graph-replayed kernel ms", lambda x: x["graph_kernel_ms"])
p("| rounds mostly graph-replayed | " + " | ".join(f"{sum(1 for q in W[c] if q['graph_kernel_ms'] > 0.5 * q['gpu_busy_ms'])} / {len(W[c])}" for c in cols) + " |")
p("| rank 1 round wall ms | " + " | ".join(f"{m(W1[c], lambda x: x['wall_ms']):.2f}" for c in cols) + " |")
p("")

# floors
p("## Against the floor (probe ceiling: 235 GB/s expert reads, 230.5 GB/s the dense set)\n")
p("| workload | round ms | experts ms / floor | dense ms / floor | exchanges ms (exposed) / floor | small kernels ms / floor | idle ms | floor ms | % of floor |")
p("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for c in cols:
    per = W[c]
    wall = m(per, lambda x: x["wall_ms"])
    ex = m(per, lambda x: fam(x, "routed experts"))
    exf = m(per, lambda x: x["experts"]["bytes_gb"]) / CEIL_EX * 1e3
    de = m(per, lambda x: fam(x, "dense q4 GEMV"))
    def_ = m(per, lambda x: x["dense"]["bytes_gb"]) / CEIL_DENSE * 1e3
    cx = m(per, lambda x: x["comm_exposed_ms"])
    sm = m(per, lambda x: fam(x, "attention + indexer") + fam(x, "KDA") + fam(x, "hc") + fam(x, "router / grouping / combine") + fam(x, "norms / elementwise / other") + fam(x, "memcpy / memset"))
    smf = 1.7 if c.startswith("1") else 4.9
    idle = m(per, lambda x: x["idle_ms"])
    fl = exf + def_ + 0.3 + smf
    p(f"| {c} | {wall:.1f} | {ex:.1f} / {exf:.1f} | {de:.1f} / {def_:.1f} | {cx:.1f} / 0.3 | {sm:.1f} / {smf:.1f} | {idle:.1f} | {fl:.1f} | {100 * fl / wall:.0f}% |")
p("")

p("## Capture overhead: the same requests uncaptured (engine round_kinds: verify_ms + draft_ms a round)\n")
p("| requests | uncaptured ms a round | captured | tokens a round | decode_s / rounds uncaptured |")
p("| --- | ---: | ---: | ---: | ---: |")
for label, fa, fb, names in (("prose chat + essay (load 1)", "ctl.jsonl", "cap.jsonl", {"chat", "essay"}),
                             ("code-lru (load 1)", "ctl.jsonl", "cap.jsonl", {"code-lru"}),
                             ("4 x prose (load 2, 4 slots)", "ctl4.jsonl", "cap4.jsonl", {"conc4-0", "conc4-1", "conc4-2", "conc4-3"}),
                             ("chat (load 2)", "ctl4.jsonl", "cap4.jsonl", {"chat"})):
    a, n, tpr, dec = ctl_ms(fa, names)
    b, _, _, _ = ctl_ms(fb, names)
    p(f"| {label} | {a:.1f} | {b:.1f} (+{100 * (b / a - 1):.1f}%) | {tpr:.2f} | {dec:.1f} |")
p("")
print("\n".join(out))
