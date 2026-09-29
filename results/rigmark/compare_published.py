#!/usr/bin/env python3
"""W13: side-by-side markdown of our TensorFold RigMark receipt vs Alex Ellis's two published GLM-5.3 TP2 vLLM receipts
(pinned rigmark clone, results/reference/glm53-libert-nvfp4-tp2-{low,adaptive-low}.json). Not a strict `rigmark
compare`: different comparison IDs, protocol 1.0 vs 1.1, hardware units, weights and engines.

    compare_published.py OURS.json RIGMARK_DIR [--out FILE]
"""
import argparse, json, statistics
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("ours"); ap.add_argument("rigmark"); ap.add_argument("--out")
a = ap.parse_args()
ref = Path(a.rigmark) / "results/reference"
cols = [("TensorFold (ours)", json.load(open(a.ours)), True),
        ("vLLM TP2 k=7 (Alex, `glm53-libert-nvfp4-tp2-low`)", json.load(open(ref / "glm53-libert-nvfp4-tp2-low.json")), False),
        ("vLLM TP2 adaptive (Alex, `glm53-libert-nvfp4-tp2-adaptive-low`)", json.load(open(ref / "glm53-libert-nvfp4-tp2-adaptive-low.json")), False)]


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def last(run):
    return run.get("time_to_last_output_seconds") or run.get("wall_seconds")


def f(x, nd=1):
    if x is None:
        return "-"
    return f"{x:,.{nd}f}"


def rng(xs, nd=1):
    xs = [x for x in xs if x is not None]
    return f"{f(med(xs), nd)} ({f(min(xs), nd)}-{f(max(xs), nd)})" if xs else "-"


rows = []
def row(name, fn, ratio=True, lower=False):
    vals = [fn(d, ours) for _, d, ours in cols]
    cells = [v if isinstance(v, str) else f(v) for v in vals]
    num = [v if not isinstance(v, str) else None for v in vals]
    def r(i):
        if not ratio or num[0] is None or num[i] in (None, 0):
            return "-"
        return f"{num[0] / num[i]:.2f}x"
    rows.append(f"| {name}{' (lower is better)' if lower else ''} | " + " | ".join(cells) + f" | {r(1)} | {r(2)} |")


def dec(w, key):
    return lambda d, o: med([r[key] for r in d["decode"][w]["runs"]])


for w in ("code", "prose", "structured"):
    label = "structured (ceiling)" if w == "structured" else w
    row(f"{label} decode tok/s, median", dec(w, "decode_tokens_per_second"))
    rows.append(f"| {label} decode, 5-run range | " + " | ".join(
        f"{f(min(r['decode_tokens_per_second'] for r in d['decode'][w]['runs']))}-{f(max(r['decode_tokens_per_second'] for r in d['decode'][w]['runs']))}"
        for _, d, _o in cols) + " | - | - |")
    row(f"{label} time to last output s, median", lambda d, o, w=w: med([last(r) for r in d["decode"][w]["runs"]]), lower=True)
    row(f"{label} TTFT s, median", lambda d, o, w=w: med([r["ttft_seconds"] for r in d["decode"][w]["runs"]]), lower=True)
    row(f"{label} completion tokens, median", lambda d, o, w=w: med([r["completion_tokens"] for r in d["decode"][w]["runs"]]), ratio=True)
    row(f"{label} reasoning chars, median (ours halved: sent twice)",
        lambda d, o, w=w: med([r["reasoning_characters"] / (2 if o else 1) for r in d["decode"][w]["runs"]]))
    row(f"{label} basic gate", lambda d, o, w=w: f"{d['decode'][w]['completion_gate']['passed']}/{d['decode'][w]['completion_gate']['total']}", ratio=False)
for dep in ("8192", "32768", "65536"):
    row(f"{int(dep) // 1024}K prefill cold tok/s, median", lambda d, o, dep=dep: d["prefill"][dep]["cold"]["effective_prefill_tokens_per_second"]["median"])
    row(f"{int(dep) // 1024}K immediate replay tok/s, median", lambda d, o, dep=dep: d["prefill"][dep]["warm_replay"]["effective_prefill_tokens_per_second"]["median"])
for c in ("1", "2", "4"):
    row(f"C{c} aggregate end-to-end tok/s, median", lambda d, o, c=c: d["concurrency"][c]["aggregate_end_to_end_tokens_per_second"]["median"])
    row(f"C{c} per-stream decode tok/s, median", lambda d, o, c=c: d["concurrency"][c]["per_stream_decode_tokens_per_second"]["median"])
    row(f"C{c} per-stream TTFT s, median", lambda d, o, c=c: d["concurrency"][c]["per_stream_ttft_seconds"]["median"], lower=True)

meta = []
for key in ("model", "quantisation", "kv_cache_dtype", "drafter", "speculative_tokens", "speculative_policy", "context_limit",
            "serving_engine", "topology", "max_sequences", "scheduler"):
    meta.append(f"| {key} | " + " | ".join(str(d["run"]["appliance"].get(key, "-")) for _, d, _o in cols) + " |")
hdr = "| metric | " + " | ".join(n for n, _, _o in cols) + " | ours / k=7 | ours / adaptive |\n|---|" + "---:|" * (len(cols) + 2)
proto = "| protocol / rigmark rev / comparison id | " + " | ".join(
    f"{d['protocol']['version']} / {d['protocol']['repository_revision'][:12]} / {d['run']['comparison_id']}" for _, d, _o in cols) + " |"
md = "\n".join([hdr, *rows, "", "| appliance | " + " | ".join(n for n, _, _o in cols) + " |\n|---|" + "---|" * len(cols), proto, *meta]) + "\n"
if a.out:
    Path(a.out).write_text(md)
print(md)
