#!/usr/bin/env python3
"""W16: every load against the control(s) and the THEORY-2 §4 gates, from results/W16 only (standard library).

  python3 results/THEORY2-SESSION/summary.py [results/W16] > results/W16/summary.txt

Controls = loads named C, C2, ... (the mean of those present). Per load: exact n/n, batchexact, ab.py reply sha,
glmbench reply hashes == control C's (every run of every cell: greedy and seeded sampled replies are deterministic),
1-stream geo-mean / greedy code-like / prose / sampled vs control, 4-stream aggregate (6 reps), lone slots; then the
gates: L / L1 (item 1 against the W12 prod control, which already has CAPTURE_AFTER=8 (+2.0% at 4s): 4s >= +0.7%
(beyond the +-0.5% control spread), lone slots >= -0.5% (G0 cost -3.1% there), 1s >= 0), K (item 6: 1s or 4s >= +0.7%; the skew half is
rocetrace.py's), VS (item 2's split: 1s >= +0.4%, replies equal).
"""
import json
import re
import statistics
import sys
from pathlib import Path

R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "W16"
SHA = "8794a3463259cc2f"


def glm(tag):
    p = R / f"glmbench-{tag}.json"
    if not p.exists():
        return {}, {}
    d = json.loads(p.read_text())
    tps, shas = {}, {}
    for suite, cells in d.get("suites", {}).items():
        for c in cells:
            key = f"{suite}:{c['prompt']}:{c['temperature']}"
            if c.get("median_tps"):
                tps[key] = c["median_tps"]
            shas[key] = [r.get("sha256") for r in c.get("runs", [])]
    return tps, shas


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
    v = [json.loads(line)["decode_tps"] for line in p.read_text().splitlines() if line.startswith('{"i"')]
    v = [x for x in v if x]
    return statistics.mean(v) if v else None


def exact(tag):
    p = R / f"exact-{tag}.log"
    if not p.exists():
        return None
    t = p.read_text()
    return f"{t.count('identical=True')}/{t.count('identical=')}"


def batchexact(tag):
    p = R / f"batchexact-{tag}.log"
    if not p.exists():
        return None
    m = re.findall(r"batched == alone: (\d+)/(\d+)", p.read_text())
    return f"{m[-1][0]}/{m[-1][1]}" if m else "?"


def absha(tag):
    p = R / f"ab-{tag}.log"
    if not p.exists():
        return None
    s = re.findall(r"sha ([0-9a-f]+)/([0-9a-f]+)", p.read_text())
    return "ok" if s and all(a == SHA and b == SHA for a, b in s) else ("DIFF " + str(s))


def loads():
    seen = []
    for line in (R / "loads.log").read_text().splitlines() if (R / "loads.log").exists() else []:
        m = re.match(r"load (\S+) \(", line)
        if m and m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def gm(xs):
    return 100 * (statistics.geometric_mean(xs) - 1) if xs else float("nan")


def main():
    L = loads()
    G = {t: glm(t) for t in L}
    ctrl_tags = [t for t in L if re.fullmatch(r"C\d*", t) and G[t][0]]
    if not ctrl_tags:
        print("no control load with glmbench results yet")
        return
    cells = list(G[ctrl_tags[0]][0])
    ctrl = {c: statistics.mean(G[t][0][c] for t in ctrl_tags if c in G[t][0]) for c in cells}
    cc = [x for t in ctrl_tags for x in conc(t)]
    sv = [s for s in (slots(t) for t in ctrl_tags) if s]
    sc = statistics.mean(sv) if sv else None
    ref_sha = G["C"][1] if "C" in G else G[ctrl_tags[0]][1]
    print(f"W16 loads {L}; controls {ctrl_tags}. Speed: % vs the controls' mean.")
    code_like = [c for c in cells if any(k in c for k in ("code:0.0", "structured", "edit"))]
    prose = [c for c in cells if any(k in c for k in ("chat:0.0", "essay", "hashmap"))]
    samp = [c for c in cells if c.endswith(":1.0")]
    print(f"{'load':6} {'exact':>6} {'bexact':>6} {'ab sha':>7} {'hashes=C':>9} {'1s geo':>7} {'code-like':>9} "
          f"{'prose':>6} {'sampled':>7} {'4s agg (n)':>12} {'slots':>7}")
    res = {}
    for t in L:
        tps, shas = G[t]
        r = {c: tps[c] / ctrl[c] for c in cells if c in tps}
        same = sum(1 for c, v in shas.items() if ref_sha.get(c) == v)
        hashes = f"{same}/{len(shas)}" if shas else "-"
        cv = conc(t)
        s = slots(t)
        row = dict(exact=exact(t), batchexact=batchexact(t), absha=absha(t), hashes=hashes,
                   g1=gm(list(r.values())), code=gm([r[c] for c in code_like if c in r]),
                   prose=gm([r[c] for c in prose if c in r]), samp=gm([r[c] for c in samp if c in r]),
                   c4=100 * (statistics.mean(cv) / statistics.mean(cc) - 1) if cv and cc else float("nan"), n4=len(cv),
                   slots=100 * (s / sc - 1) if s and sc else float("nan"))
        res[t] = row
        print(f"{t:6} {row['exact'] or '-':>6} {row['batchexact'] or '-':>6} {row['absha'] or '-':>7} {hashes:>9} "
              f"{row['g1']:+6.1f}% {row['code']:+8.1f}% {row['prose']:+5.1f}% {row['samp']:+6.1f}% "
              f"{row['c4']:+7.1f}% ({row['n4']}) {row['slots']:+6.1f}%")
    print()
    bits = lambda row: (row["exact"] in (None, "10/10") and row["batchexact"] in (None, "4/4")  # noqa: E731
                        and row["absha"] in (None, "ok"))
    for t in ("L", "L1"):
        if t not in res:
            continue
        x = res[t]
        ok = x["c4"] >= 0.7 and (x["slots"] >= -0.5 or x["slots"] != x["slots"]) and x["g1"] >= 0 and bits(x)
        print(f"GATE item1 {t} (BATCH_GRAPHS=lone{' + CAPTURE_AFTER=1' if t == 'L1' else ''} on prod's CAPTURE_AFTER=8 "
              f"+ L2PF): 4s {x['c4']:+.1f}% (>= +0.7), lone slots {x['slots']:+.1f}% (>= -0.5), 1s {x['g1']:+.1f}% "
              f"(>= 0), bits {bits(x)}, hashes {x['hashes']} -> {'PASS' if ok else 'FAIL'}")
    if "K" in res:
        x = res["K"]
        ok = (x["g1"] >= 0.7 or x["c4"] >= 0.7) and bits(x)
        print(f"GATE item6 K (lgc 2250 + CPU_PIN=http + trace): 1s {x['g1']:+.1f}% / 4s {x['c4']:+.1f}% (either >= +0.7) "
              f"-> {'PASS' if ok else 'FAIL'} (or the skew gate in rocetrace-K.txt)")
    if "VS" in res:
        x = res["VS"]
        ok = x["g1"] >= 0.4 and bits(x)
        print(f"GATE item2 VS (VERIFY_SPLIT): 1s {x['g1']:+.1f}% (>= +0.4), bits {bits(x)}, hashes {x['hashes']} "
              f"-> {'PASS' if ok else 'FAIL'} (graph probe lines: boot-*-r0.log / docker logs)")
    if "H" in res:
        x = res["H"]
        ok = x["g1"] >= 1.0 and bits(x)
        print(f"GATE item3 H (HC_CUDA=1): 1s {x['g1']:+.1f}% (>= +1.0), 4s {x['c4']:+.1f}%, bits {bits(x)}, hashes "
              f"{x['hashes']} -> {'PASS' if ok else 'FAIL'}")
    print()
    print("per cell (tok/s): control mean, then each load's % vs it")
    others = [t for t in L if t not in ctrl_tags and G[t][0]]
    print(f"{'cell':28} {'ctrl':>7} " + " ".join(f"{t:>6}" for t in others))
    for c in cells:
        print(f"{c:28} {ctrl[c]:7.1f} " + " ".join(
            f"{100 * (G[t][0][c] / ctrl[c] - 1):+6.1f}" if c in G[t][0] else f"{'-':>6}" for t in others))
    (R / "summary.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
