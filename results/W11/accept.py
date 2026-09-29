#!/usr/bin/env python3
"""W11: per-position draft acceptance from the engine's per-request stats (patches/0380: ``depths``; ``keeps``;
``drafters`` = the arm of each round: m = MTP, f = DFlash2, l = lookup, s = serial).

  accept.py accept-DEF.jsonl [more.jsonl ...] [--json OUT]

For each arm and content class (prose / code / agent / prose-think; greedy vs sampled): rounds, mean drafts proposed
(depth) and tokens a round (keep), and for position j = 1..: attempted_j = rounds whose depth >= j and whose drafts
1..j-1 were all accepted; accepted_j = those that also kept draft j. a_j = accepted_j / attempted_j (conditional,
the number a drafter's training moves), p_j = accepted_j / rounds with depth >= j (cumulative)."""
import collections
import json
import sys

args = [x for x in sys.argv[1:] if not x.startswith("--")]
out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
if out in args:
    args.remove(out)
reqs = [json.loads(line) for f in args for line in open(f)]


def acc(rounds):
    """rounds: [(depth, keep)] -> summary"""
    n = len(rounds)
    if not n:
        return None
    J = max(d for d, _ in rounds)
    att, ok, ge = [0] * (J + 1), [0] * (J + 1), [0] * (J + 1)
    for d, k in rounds:
        a = k - 1                              # drafts accepted
        for j in range(1, d + 1):
            ge[j] += 1
            if a >= j - 1:
                att[j] += 1
                if a >= j:
                    ok[j] += 1
    return {"rounds": n, "mean_depth": round(sum(d for d, _ in rounds) / n, 2),
            "tokens_a_round": round(sum(k for _, k in rounds) / n, 3),
            "a_cond": [round(ok[j] / att[j], 3) if att[j] else None for j in range(1, J + 1)],
            "attempted": att[1:], "p_cum": [round(ok[j] / ge[j], 3) if ge[j] else None for j in range(1, J + 1)]}


groups = collections.defaultdict(list)
for r in reqs:
    tf = r["tf"]
    arms, depths, keeps = tf.get("drafters") or "", tf.get("depths") or [], tf.get("keeps") or []
    if not (len(arms) == len(depths) == len(keeps)):
        print("skip (lengths differ)", r.get("name"), len(arms), len(depths), len(keeps), file=sys.stderr)
        continue
    temp = "sampled" if r.get("temp", 0) > 0 else "greedy"
    for a, d, k in zip(arms, depths, keeps):
        for key in ((r["cat"], temp, a), (r["cat"], "all", a), ("all", temp, a), ("all", "all", a)):
            groups[key].append((d, k))
res = {}
ARM = {"m": "MTP", "f": "DFlash2", "l": "lookup", "s": "serial"}
print(f"{'class':12s} {'temp':8s} {'arm':8s} {'rounds':>6s} {'depth':>6s} {'tok/rnd':>7s}  a_j (conditional, positions 1..)")
for key in sorted(groups, key=lambda k: (k[0] != "all", k[0], k[1], k[2])):
    s = acc(groups[key])
    res["|".join(key)] = s
    a = " ".join(f"{x:.2f}" if x is not None else "  - " for x in s["a_cond"][:8])
    print(f"{key[0]:12s} {key[1]:8s} {ARM.get(key[2], key[2]):8s} {s['rounds']:6d} {s['mean_depth']:6.2f} "
          f"{s['tokens_a_round']:7.3f}  {a}   (n {s['attempted'][:8]})")
# per request: tokens a round, arms mix, decode tok/s
print()
per = []
for r in reqs:
    tf = r["tf"]
    arms = tf.get("drafters") or ""
    c = collections.Counter(arms)
    row = dict(name=r["name"], cat=r["cat"], temp=r.get("temp"), tokens=r["tokens"], tps=r.get("decode_tps"),
               tpr=tf.get("tokens_per_round"), mix={ARM.get(k, k): v for k, v in c.items()})
    per.append(row)
    print(f"{r['name']:14s} {r['cat']:11s} T={r.get('temp')} tokens {r['tokens']:5d} {r.get('decode_tps')!s:>6} tok/s  "
          f"tok/round {tf.get('tokens_per_round')}  arms {dict(c)}")
res["requests"] = per
if out:
    json.dump(res, open(out, "w"), indent=1)
