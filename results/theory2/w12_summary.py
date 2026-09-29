#!/usr/bin/env python3
"""THEORY-2: W12 loads against the same-window controls (C, C2, C3), from results/W12 logs only.

  python3 results/theory2/w12_summary.py > results/theory2/w12_summary.txt
"""
import json, re, statistics, pathlib
R = pathlib.Path(__file__).resolve().parents[1] / "W12"
LOADS = ["C", "C2", "C3", "E2", "G0", "PF", "PFE", "GRS", "GRR", "GR1", "GR1L"]
CTRL = ["C", "C2", "C3"]

def glm(tag):
    p = R / f"glmbench-{tag}.log"
    if not p.exists():
        return {}
    d = json.loads(p.read_text().strip().splitlines()[-1])
    return {f"{s}:{c}:{t}": v for s, rows in d.items() for c, t, v in rows}

def conc(tag):
    out = []
    for ps in "ab":
        p = R / f"conc-{tag}-{ps}.log"
        if p.exists():
            out += [float(x) for x in re.findall(r"aggregate ([\d.]+) tok/s", p.read_text())]
    return out

def slots(tag):
    p = R / f"slots-{tag}.log"
    if not p.exists():
        return None
    v = [json.loads(l)["decode_tps"] for l in p.read_text().splitlines() if l.startswith("{\"i\"")]
    return statistics.mean(v) if v else None

G = {t: glm(t) for t in LOADS}
cells = list(G["C"])
ctrl = {c: statistics.mean(G[t][c] for t in CTRL) for c in cells}
cc = [x for t in CTRL for x in conc(t)]
sc = statistics.mean(slots(t) for t in CTRL)
print("controls: C / C2 / C3 (same image b5, prod env). Numbers: % vs the mean of the three controls.")
print(f"control spread per cell (max-min)/mean, 1 stream: " + ", ".join(
    f"{c.split(':',1)[1]} {100*(max(G[t][c] for t in CTRL)-min(G[t][c] for t in CTRL))/ctrl[c]:.1f}%" for c in cells))
print()
hdr = f"{'load':6} {'1s geo-mean':>11} {'greedy code-like':>16} {'prose':>7} {'sampled':>8} {'4s agg (n)':>14} {'lone slots':>10}"
print(hdr)
code_like = [c for c in cells if any(k in c for k in ("sequence", "code:0.0", "json", "structured", "edit"))]
prose = [c for c in cells if any(k in c for k in ("chat:0.0", "essay", "hashmap"))]
samp = [c for c in cells if c.endswith(":1.0")]
def gm(xs):
    return 100 * (statistics.geometric_mean(xs) - 1)
for t in LOADS:
    g = G[t]
    if not g:
        continue
    r = {c: g[c] / ctrl[c] for c in cells if c in g}
    cv = conc(t)
    s = slots(t)
    print(f"{t:6} {gm(list(r.values())):+10.1f}% {gm([r[c] for c in code_like]):+15.1f}% {gm([r[c] for c in prose]):+6.1f}% "
          f"{gm([r[c] for c in samp]):+7.1f}% {100*(statistics.mean(cv)/statistics.mean(cc)-1):+8.1f}% ({len(cv)}) "
          f"{100*(s/sc-1):+9.1f}%")
print()
print("per cell (tok/s): control mean, then each load's % vs it")
print(f"{'cell':28} {'ctrl':>7} " + " ".join(f"{t:>6}" for t in LOADS if t not in CTRL))
for c in cells:
    print(f"{c:28} {ctrl[c]:7.1f} " + " ".join(f"{100*(G[t][c]/ctrl[c]-1):+6.1f}" for t in LOADS if t not in CTRL and c in G[t]))
