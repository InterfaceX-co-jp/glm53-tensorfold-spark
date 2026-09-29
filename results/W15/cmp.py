#!/usr/bin/env python3
"""W15 (from W12/cmp.py): one block a load against a control: cmp.py CONTROL TAG [TAG ...]; a tag DIR/NAME reads results/DIR (e.g. W12/FIN)
1 stream: glmbench cells (median tok/s of 3) -> ratio per cell, geometric means (all / prose / code / edit), reply
hashes vs control; 4 streams: mean aggregate of 6 reps; lone-slot tok/s by slot; exact / batchexact / transcripts / sha."""
import glob, json, math, os, re, statistics, sys
R0 = os.path.dirname(os.path.abspath(__file__))
def RD(tag): return (os.path.join(R0, "..", tag.split("/")[0]), tag.split("/")[1]) if "/" in tag else (R0, tag)
PROSE = {"tf/chat/1.0", "tf/chat/0.0", "kit/hashmap/0.0", "kit/essay/0.0"}
CODE = {"tf/code/1.0", "tf/code/0.0", "tweet/code/0.0"}
EDIT = {"edit/edit-rename/0.0", "edit/edit-comments/0.0", "edit/edit-print-to-log/0.0"}
def cells(tag):
    R, tag = RD(tag)
    try:
        d = json.load(open(f"{R}/glmbench-{tag}.json"))
    except Exception:
        return {}
    out = {}
    for s, cs in d["suites"].items():
        for c in cs:
            out[f"{s}/{c['prompt']}/{c['temperature']}"] = (c["median_tps"], [r["sha256"] for r in c["runs"]])
    return out
def conc(tag):
    R, tag = RD(tag)
    v = []
    for p in "ab":
        try:
            v += [r["aggregate_tps"] for r in json.load(open(f"{R}/conc-{tag}-{p}.json"))["concurrent"]]
        except Exception:
            pass
    return v
def slots(tag):
    R, tag = RD(tag)
    try:
        rows = json.load(open(f"{R}/slots-{tag}.json"))
    except Exception:
        return {}
    by = {}
    for r in rows:
        by.setdefault(r["slot"], []).append(r["decode_tps"])
    return {k: round(statistics.mean(v), 1) for k, v in sorted(by.items(), key=lambda kv: str(kv[0]))}
def gates(tag):
    R, tag = RD(tag)
    g = {}
    try: g["exact"] = open(f"{R}/exact-{tag}.log").read().splitlines()[-1].count("true")
    except Exception: g["exact"] = "?"
    try: g["batchexact"] = [l for l in open(f"{R}/batchexact-{tag}.log") if "batched == alone" in l][0].split(":")[1].split()[0]
    except Exception: g["batchexact"] = "?"
    try:
        a = open(f"{R}/transcripts-{tag}.log").read().splitlines(); b = open(f"{R0}/../W9/transcripts-A.log").read().splitlines()
        g["transcripts"] = "same" if a[0] == b[0] and "False" not in a[-1] else "DIFF"
    except Exception: g["transcripts"] = "?"
    try:
        rs = json.load(open(f"{R}/ab-{tag}.json"))
        g["sha"] = sorted({r["cold"].get("sha") for r in rs} | {(r.get("warm") or {}).get("sha") for r in rs} - {None})
        g["prefill"] = [r["cold"]["prefill_tps"] for r in rs]
    except Exception: g["sha"] = "?"
    return g
def gm(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")
ctl = sys.argv[1]
if "," in ctl:          # several controls: per-cell mean of their medians, 4-stream reps pooled
    cs = [cells(t) for t in ctl.split(",")]
    C = {k: (statistics.mean(c[k][0] for c in cs), cs[0][k][1]) for k in cs[0] if all(k in c for c in cs)}
    CC = [x for t in ctl.split(",") for x in conc(t)]
    CS = slots(ctl.split(",")[0])
else:
    C, CC, CS = cells(ctl), conc(ctl), slots(ctl)
print(f"control {ctl}: 4 streams mean {statistics.mean(CC):.1f} ({len(CC)} reps)" if CC else f"control {ctl}: no conc")
for tag in sys.argv[2:] if "," in ctl else sys.argv[1:]:
    T, TC = cells(tag), conc(tag)
    rat = {k: T[k][0] / C[k][0] for k in T if k in C and C[k][0] and T[k][0]}
    same = sum(T[k][1] == C[k][1] for k in T if k in C); n = sum(1 for k in T if k in C)
    f = lambda ks: (gm([rat[k] for k in ks if k in rat]) - 1) * 100
    c4 = (statistics.mean(TC) / statistics.mean(CC) - 1) * 100 if TC and CC else float("nan")
    print(f"{tag:10s} 1s all {f(rat):+5.1f}% prose {f(PROSE):+5.1f}% code {f(CODE):+5.1f}% edit {f(EDIT):+5.1f}% | "
          f"4s {statistics.mean(TC) if TC else float('nan'):.1f} ({c4:+.1f}%) | hashes {same}/{n} | slots {slots(tag)} | {gates(tag)}")
    if "-v" in os.environ.get("CMP", ""):
        for k in sorted(rat):
            print(f"    {k:28s} {C[k][0]:7.2f} -> {T[k][0]:7.2f} {100 * (rat[k] - 1):+5.1f}% {'same' if T[k][1] == C[k][1] else 'DIFF'}")
