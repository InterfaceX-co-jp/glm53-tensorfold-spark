#!/usr/bin/env python3
"""W16 Part B: drafter arms vs the baseline arm inco-7d7 (docs/DRAFTER-SEARCH.md §5 item 2 / §6). Standard library.

    python3 results/W16/dscmp.py results/W16/DS > results/W16/DS/dscmp.txt

Hygiene: every (prompt, temperature, policy) reply sha equal to inco-7d7's (and to W11's acceptfix-f7 / -4 when the
image serves the same bits), MTP ("4") rows vs W11's MTP table. Decision (acceptfix, DFlash2 rows "f"): prose p_cum
1..4 >= base - 0.01; code / agent tokens a DFlash2 round >= base x 0.97; production-policy (accept) greedy decode tok/s
per class not lower (reported with the noise it has); exact 10/10 + batchexact 4/4 (modal only). Drafting cost:
round_kinds draft_ms a round on the f7 rows. glmbench tf,kit,edit x3 medians + reply hashes vs base."""
import collections
import json
import re
import statistics
import sys
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else "results/W16/DS")
REPO = Path(__file__).resolve().parents[2]
BASE = "inco-7d7"
ARMS = [a for a in ("inco-7d7", "modal", "inco-bf5", "stock") if (D / f"acceptfix-{a}.jsonl").exists()]


def rows(p):
    return [json.loads(x) for x in open(p)] if Path(p).exists() else []


def key(r):
    return (r["name"], r["temp"], r.get("policy"), bool(r.get("think")))


def shas(p):
    return {key(r): r["tf"].get("sha256") for r in rows(p)}


def glm(p):
    if not Path(p).exists():
        return {}, {}
    d = json.load(open(p))
    tps, sh = {}, {}
    for suite, cells in d.get("suites", {}).items():
        for c in cells:
            k = f"{suite}:{c['prompt']}:{c['temperature']}"
            if c.get("median_tps"):
                tps[k] = c["median_tps"]
            sh[k] = [r.get("sha256") for r in c.get("runs", [])]
    return tps, sh


def draft_cost(p):
    ms = rn = 0
    for r in rows(p):
        if r.get("policy") != "f7":
            continue
        rk = r["tf"].get("round_kinds") or {}
        ms += rk.get("draft_ms") or 0
        rn += r["tf"].get("rounds") or 0
    return ms / rn if rn else None


def tps_by_class(p):
    g = collections.defaultdict(list)
    for r in rows(p):
        if r.get("decode_tps"):
            g[(r["cat"], "greedy" if r["temp"] == 0 else "sampled")].append(r["decode_tps"])
    return {k: statistics.mean(v) for k, v in g.items()}


print(f"arms: {ARMS} (baseline {BASE})\n")
base_af = shas(D / f"acceptfix-{BASE}.jsonl")
base_ac = shas(D / f"accept-{BASE}.jsonl")
w11 = {}
for f in ("acceptfix-f7.jsonl", "acceptfix-4.jsonl"):
    w11.update(shas(REPO / "results/W11" / f))
print("== hygiene: reply sha256 per (prompt, temperature, policy)")
for a in ARMS:
    af, ac = shas(D / f"acceptfix-{a}.jsonl"), shas(D / f"accept-{a}.jsonl")
    same_af = sum(1 for k, v in af.items() if base_af.get(k) == v)
    same_ac = sum(1 for k, v in ac.items() if base_ac.get(k) == v)
    vw = sum(1 for k, v in af.items() if w11.get(k) == v)
    nw = sum(1 for k in af if k in w11)
    print(f"{a:9s} acceptfix == {BASE}: {same_af}/{len(af)}   accept == {BASE}: {same_ac}/{len(ac)}   "
          f"acceptfix == W11: {vw}/{nw}")
    diff = [k for k, v in af.items() if k in base_af and base_af[k] != v]
    if diff:
        print("   DIFFER:", diff[:8])

A = {a: json.load(open(D / f"acceptfix-{a}.json")) for a in ARMS if (D / f"acceptfix-{a}.json").exists()}
w11m = json.load(open(REPO / "results/W11/acceptfix-4.json")) if (REPO / "results/W11/acceptfix-4.json").exists() else {}
if BASE in A:
    print("\n== MTP control (policy 4, rows *|all|m): W16 inco-7d7 vs W11 acceptfix-4 (want within +-0.01)")
    for c in ("prose", "code", "agent"):
        k = f"{c}|all|m"
        me, old = A[BASE].get(k), w11m.get(k)
        if me:
            print(f"  {c:6s} a_cond W16 {me['a_cond'][:4]}  W11 {old['a_cond'][:4] if old else '-'}  "
                  f"tok/rnd {me['tokens_a_round']} vs {old['tokens_a_round'] if old else '-'}")

print("\n== decision rows (acceptfix, DFlash2 f7, all temperatures): p_cum positions 1..4, tokens a round")
gates = {}
for c in ("prose", "code", "agent"):
    k = f"{c}|all|f"
    b = A.get(BASE, {}).get(k)
    for a in ARMS:
        s = A.get(a, {}).get(k)
        if not s:
            continue
        line = f"  {c:6s} {a:9s} rounds {s['rounds']:5d} p_cum {s['p_cum'][:4]} tok/rnd {s['tokens_a_round']:.3f}"
        if b and a != BASE:
            dp = [round(x - y, 3) for x, y in zip(s["p_cum"][:4], b["p_cum"][:4])]
            dt = 100 * (s["tokens_a_round"] / b["tokens_a_round"] - 1)
            line += f"   d p_cum {dp}  d tok/rnd {dt:+.1f}%"
            ok = all(x >= -0.01 for x in dp) if c == "prose" else dt >= -3.0
            gates.setdefault(a, []).append((c, ok))
            line += "  PASS" if ok else "  FAIL"
        print(line)

print("\n== drafting cost: draft_ms a round (f7 rows)")
dc = {a: draft_cost(D / f"acceptfix-{a}.jsonl") for a in ARMS}
for a in ARMS:
    rel = f" ({100 * (dc[a] / dc[BASE] - 1):+.1f}% vs {BASE})" if dc.get(BASE) and dc[a] and a != BASE else ""
    print(f"  {a:9s} {dc[a]:.3f} ms{rel}" if dc[a] else f"  {a:9s} -")

print("\n== production policy (accept): mean decode tok/s per class")
T = {a: tps_by_class(D / f"accept-{a}.jsonl") for a in ARMS}
for k in sorted(T.get(BASE, {})):
    line = f"  {k[0]:12s} {k[1]:8s} " + " ".join(
        f"{a} {T[a].get(k, float('nan')):6.2f}" + (f" ({100 * (T[a][k] / T[BASE][k] - 1):+.1f}%)" if a != BASE and k in T[a] else "")
        for a in ARMS if T.get(a))
    print(line)

print("\n== glmbench tf,kit,edit x3 (median tok/s; reply hashes vs base)")
Gb = glm(D / f"glmbench-{BASE}.json")
for a in ARMS:
    t, h = glm(D / f"glmbench-{a}.json")
    if not t:
        continue
    common = [k for k in t if k in Gb[0]]
    rel = [t[k] / Gb[0][k] for k in common]
    same = sum(1 for k in h if Gb[1].get(k) == h[k])
    gmr = 100 * (statistics.geometric_mean(rel) - 1) if rel else float("nan")
    code = [t[k] / Gb[0][k] for k in common if re.search(r"code|fib|lru|edit|json|tool|agent", k)]
    print(f"  {a:9s} cells {len(t)} geo-mean vs base {gmr:+.1f}% (code/agent-like {100 * (statistics.geometric_mean(code) - 1) if code else float('nan'):+.1f}%) "
          f"hashes == base {same}/{len(h)}")
    if a != BASE:
        for k in common:
            print(f"      {k:40s} {Gb[0][k]:7.2f} -> {t[k]:7.2f} ({100 * (t[k] / Gb[0][k] - 1):+.1f}%)")

for a in ARMS:
    p = D / f"exact-{a}.log"
    if p.exists():
        tx = p.read_text()
        be = (D / f"batchexact-{a}.log").read_text() if (D / f"batchexact-{a}.log").exists() else ""
        m = re.findall(r"batched == alone: (\d+)/(\d+)", be)
        print(f"\n{a}: exact {tx.count('identical=True')}/{tx.count('identical=')}  batchexact {m[-1] if m else '?'}")
print("\n== acceptance gates (DRAFTER-SEARCH §5 item 2; tok/s and exactness read above):",
      {a: g for a, g in gates.items()})
