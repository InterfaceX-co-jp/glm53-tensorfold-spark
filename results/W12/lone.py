#!/usr/bin/env python3
"""W12: lone-request probe (slots.py) per prompt against the mean of the controls: lone.py CTL1,CTL2,... TAG ..."""
import json, statistics, sys
base = {}
for t in sys.argv[1].split(","):
    for r in json.load(open(f"slots-{t}.json")):
        base.setdefault(r["i"], []).append(r["decode_tps"])
for t in sys.argv[2:]:
    try:
        rows = json.load(open(f"slots-{t}.json"))
    except Exception:
        print(t, "no slots file"); continue
    d = [(r["slot"], 100 * (r["decode_tps"] / statistics.mean(base[r["i"]]) - 1)) for r in rows]
    s0 = [x for s, x in d if s == 0]; s123 = [x for s, x in d if s != 0]
    print(f"{t:6s} all {statistics.mean(x for _, x in d):+.1f}%  slot 0 {statistics.mean(s0) if s0 else float('nan'):+.1f}%  "
          f"slots 1-3 {statistics.mean(s123):+.1f}%  (" + " ".join(f"s{s}:{x:+.1f}" for s, x in d) + ")")
