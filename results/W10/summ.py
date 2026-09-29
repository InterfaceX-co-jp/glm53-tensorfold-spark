#!/usr/bin/env python3
"""W10: one line a load from its files: summ.py TAG ... -> prefill tok/s (cold, each ab run), reply shas, exact,
batchexact, sessions, concurrent 1 / 4 stream aggregate tok/s per rep and median."""
import json, os, statistics, sys, glob, re
R = os.path.dirname(os.path.abspath(__file__))
for tag in sys.argv[1:]:
    pf, shas = {24500: [], 98000: []}, set()
    for f in sorted(glob.glob(f"{R}/ab-{tag}-[0-9].json")):
        for r in json.load(open(f)):
            c = r["cold"]; pf[r["ctx"]].append(c["prefill_tps"]); shas.add(c.get("sha"))
            if r.get("warm"): shas.add(r["warm"].get("sha"))
    ex = "?"
    try:
        ex = open(f"{R}/exact-{tag}.log").read().splitlines()[-1].count("true")
    except Exception: pass
    be = "?"
    try:
        be = [l for l in open(f"{R}/batchexact-{tag}.log") if "batched == alone" in l][0].split(":")[1].split()[0]
    except Exception: pass
    se = "-"
    try:
        se = "OK" if "SESSIONS OK" in open(f"{R}/sessions-{tag}.log").read() else "FAIL"
    except Exception: pass
    conc = {}
    try:
        for l in open(f"{R}/conc-{tag}.log"):
            m = re.search(r"(\d+) streams rep (\d+).*?aggregate ([\d.]+)", l)
            if m: conc.setdefault(int(m.group(1)), []).append(float(m.group(3)))
    except Exception: pass
    cs = " ".join(f"{k}s {'/'.join(str(v) for v in vs)} med {statistics.median(vs)}" for k, vs in sorted(conc.items()))
    print(f"{tag}: 24.5k {pf[24500]} 98k {pf[98000]} sha {sorted(s for s in shas if s)} exact {ex}/10 batchexact {be} sessions {se} | {cs}")
