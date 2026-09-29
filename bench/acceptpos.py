#!/usr/bin/env python3
"""Draft acceptance by position and drafter, from the engine's per-request stats (patches/0420's GPU plan,
docs/DRAFT-VOCAB.md; any A/B of a drafter change).

Input: JSON files holding the ``tensorfold`` stats of non-streamed replies anywhere inside them (dicts with
``drafters`` = one letter a round, ``m`` MTP / ``f`` DFlash2 / ``l`` lookup, and ``keeps`` = rows kept a round;
``depths`` = drafts verified a round, patches/0380, when present): ``results/W1/ab.py`` outputs, the
``concurrent`` runs (``drafters`` beside ``stats``), a list of raw responses. For each drafter it prints rounds,
tokens a round and, for positions 1..7, the acceptance rate: the share of rounds that verified a draft at that
position (``depths``) whose drafts up to it were all kept. Without ``depths`` it prints the survival
P(kept drafts >= j) instead.

    python3 bench/acceptpos.py A.json [B.json ...]          # one table a file; with two, B - A under each row
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def rounds(obj):
    """Every (drafter, keep, depth or None) round inside ``obj``."""

    if isinstance(obj, dict):
        arms, stats = obj.get("drafters"), obj.get("stats")
        if isinstance(arms, list) and isinstance(stats, list) and len(arms) == len(stats):    # concurrent runs
            for a, st in zip(arms, stats):
                yield from rounds(dict(st, drafters=a))
            return
        arms, keeps = obj.get("drafters"), obj.get("keeps")
        if isinstance(arms, str) and isinstance(keeps, list) and len(arms) == len(keeps):
            depths = obj.get("depths")
            depths = depths if isinstance(depths, list) and len(depths) == len(keeps) else [None] * len(keeps)
            yield from zip(arms, keeps, depths)
            return
        for v in obj.values():
            yield from rounds(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from rounds(v)


def table(path: str) -> dict:
    by = collections.defaultdict(list)
    for arm, keep, depth in rounds(json.loads(Path(path).read_text())):
        by[arm].append((int(keep), None if depth is None else int(depth)))
    out = {}
    for arm, rs in sorted(by.items()):
        acc = []
        for j in range(1, 8):
            if all(d is not None for _, d in rs):
                tried = [k for k, d in rs if d >= j]
                acc.append(sum(1 for k in tried if k - 1 >= j) / len(tried) if tried else None)
            else:
                acc.append(sum(1 for k, _ in rs if k - 1 >= j) / len(rs))
        out[arm] = dict(rounds=len(rs), tpr=sum(k for k, _ in rs) / len(rs), acc=acc,
                        kind="acceptance" if all(d is not None for _, d in rs) else "survival")
    return out


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    tabs = [(p, table(p)) for p in sys.argv[1:]]
    names = {"m": "MTP", "f": "DFlash2", "l": "lookup"}
    for i, (p, t) in enumerate(tabs):
        print(p)
        for arm, r in t.items():
            cells = " ".join("   -  " if a is None else f"{a:6.3f}" for a in r["acc"])
            print(f"  {names.get(arm, arm):8s} {r['rounds']:6d} rounds  {r['tpr']:.3f} tok/round  {r['kind']} 1..7: {cells}")
            if i and arm in tabs[0][1]:
                b = tabs[0][1][arm]
                d = " ".join("   -  " if x is None or y is None else f"{x - y:+6.3f}" for x, y in zip(r["acc"], b["acc"]))
                print(f"  {'':8s} {'':6s}         {r['tpr'] - b['tpr']:+.3f} vs {Path(tabs[0][0]).name:>12s}        {d}")


if __name__ == "__main__":
    main()
