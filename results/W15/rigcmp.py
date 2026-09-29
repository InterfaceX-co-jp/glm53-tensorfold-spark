#!/usr/bin/env python3
"""W15: our RigMark receipt on the new production (b7) against W13's (b5), with Alex Ellis's published vLLM TP2 k=7
receipt for reference. W13 sent reasoning twice (reasoning + reasoning_content), so its reasoning characters are
halved here; W15 sends `reasoning` only. Not a strict `rigmark compare` (different comparison IDs).

    rigcmp.py W15.json W13.json RIGMARK_DIR [--out FILE]
"""
import argparse, json, statistics
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("new"); ap.add_argument("old"); ap.add_argument("rigmark"); ap.add_argument("--out")
a = ap.parse_args()
ref = Path(a.rigmark) / "results/reference/glm53-libert-nvfp4-tp2-low.json"
cols = [("W15 TensorFold b7", json.load(open(a.new)), 1), ("W13 TensorFold b5", json.load(open(a.old)), 2),
        ("vLLM TP2 k=7 (Alex)", json.load(open(ref)), 1)]


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def f(x, nd=2):
    if x is None:
        return "-"
    return f"{x:,.{nd}f}" if abs(x) < 10 else f"{x:,.1f}"


rows = []
def row(name, fn, lower=False):
    vals = [fn(d, h) for _, d, h in cols]
    r = f"{vals[0] / vals[1]:.2f}x" if vals[0] is not None and vals[1] else "-"
    rows.append(f"| {name}{' (lower is better)' if lower else ''} | " + " | ".join(f(v) for v in vals) + f" | {r} |")


def last(run):
    return run.get("time_to_last_output_seconds") or run.get("wall_seconds")


for w in ("code", "prose", "structured"):
    row(f"{w} decode tok/s", lambda d, h, w=w: med([r["decode_tokens_per_second"] for r in d["decode"][w]["runs"]]))
    row(f"{w} TTFT s", lambda d, h, w=w: med([r["ttft_seconds"] for r in d["decode"][w]["runs"]]), lower=True)
    row(f"{w} time to last output s", lambda d, h, w=w: med([last(r) for r in d["decode"][w]["runs"]]), lower=True)
    row(f"{w} reasoning chars (W13 halved)", lambda d, h, w=w: med([r["reasoning_characters"] / h for r in d["decode"][w]["runs"]]))
    row(f"{w} reasoning chars raw", lambda d, h, w=w: med([r["reasoning_characters"] for r in d["decode"][w]["runs"]]))
    rows.append(f"| {w} basic gate | " + " | ".join(f"{d['decode'][w]['completion_gate']['passed']}/{d['decode'][w]['completion_gate']['total']}"
                                                      for _, d, _h in cols) + " | - |")
for dep in ("8192", "32768", "65536"):
    row(f"{int(dep) // 1024}K cold prefill tok/s", lambda d, h, dep=dep: d["prefill"][dep]["cold"]["effective_prefill_tokens_per_second"]["median"])
    row(f"{int(dep) // 1024}K immediate replay tok/s", lambda d, h, dep=dep: d["prefill"][dep]["warm_replay"]["effective_prefill_tokens_per_second"]["median"])
for c in ("1", "2", "4"):
    row(f"C{c} aggregate tok/s", lambda d, h, c=c: d["concurrency"][c]["aggregate_end_to_end_tokens_per_second"]["median"])
    row(f"C{c} per-stream decode tok/s", lambda d, h, c=c: d["concurrency"][c]["per_stream_decode_tokens_per_second"]["median"])
    row(f"C{c} per-stream TTFT s", lambda d, h, c=c: d["concurrency"][c]["per_stream_ttft_seconds"]["median"], lower=True)
hdr = "| metric | " + " | ".join(n for n, _, _h in cols) + " | W15 / W13 |\n|---|" + "---:|" * (len(cols) + 1)
proto = "\n| receipt | " + " | ".join(n for n, _, _h in cols) + " |\n|---|" + "---|" * len(cols) + \
    "\n| protocol / rev / comparison id | " + " | ".join(
        f"{d['protocol']['version']} / {d['protocol']['repository_revision'][:12]} / {d['run']['comparison_id']}" for _, d, _h in cols) + " |"
md = "\n".join([hdr, *rows]) + "\n" + proto + "\n"
if a.out:
    Path(a.out).write_text(md)
print(md)
