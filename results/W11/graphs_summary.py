#!/usr/bin/env python3
"""W11: 4-stream graph policy table from graphs.sh outputs (conc-TAG-a.json = first 3 reps after load, -b = the next 3).

  graphs_summary.py DEF G0 CA8 DEF2"""
import json
import statistics
import sys
from pathlib import Path

H = Path(__file__).parent
print("| load | pass | aggregate tok/s, reps | mean | request-rounds graph / eager / capture / alone (%) | ms a round (engine) | batchexact |")
print("| --- | --- | --- | ---: | --- | ---: | --- |")
means = {}
for tag in sys.argv[1:]:
    bx = (H / f"batchexact-{tag}.log").read_text() if (H / f"batchexact-{tag}.log").exists() else ""
    bx = next((line.strip() for line in bx.splitlines() if "batched == alone" in line), "?")
    for ps in ("a", "b"):
        f = H / f"conc-{tag}-{ps}.json"
        if not f.exists():
            continue
        reps = json.load(open(f))["concurrent"]
        agg = [r["aggregate_tps"] for r in reps]
        kinds = {}
        vms = rounds = 0
        for r in reps:
            for s in r["stats"]:
                for k, v in (s.get("round_kinds") or {}).items():
                    if k in ("graph", "eager", "capture", "alone"):
                        kinds[k] = kinds.get(k, 0) + v
                    if k in ("verify_ms", "draft_ms"):
                        vms += v
                rounds += s.get("rounds") or 0
        tot = sum(kinds.values()) or 1
        pc = " / ".join(f"{100 * kinds.get(k, 0) / tot:.0f}" for k in ("graph", "eager", "capture", "alone"))
        means.setdefault(tag, []).extend(agg)
        print(f"| {tag} | {ps} | {', '.join(f'{x:.1f}' for x in agg)} | {statistics.mean(agg):.1f} | {pc} | "
              f"{vms / max(rounds, 1):.1f} | {bx if ps == 'a' else ''} |")
print()
base = statistics.mean(means.get("DEF", []) + means.get("DEF2", [])) if "DEF" in means else None
for tag, v in means.items():
    print(f"{tag}: mean of 6 reps {statistics.mean(v):.2f} tok/s" + (f" ({100 * (statistics.mean(v) / base - 1):+.1f}% vs DEF+DEF2)" if base else ""))
