#!/usr/bin/env python3
"""THEORY-2: W12 kernel bench (results/W12/kbench.json): old vs best new vs roof, per window / shape."""
import json, pathlib
d = json.loads((pathlib.Path(__file__).resolve().parents[1] / "W12" / "kbench.json").read_text())
print("E1 routed experts (one MoE layer, gate/up + down): us / GB/s")
print(f"{'window':22} {'old':>12} {'best new (cfg)':>28} {'probe1 nodecode':>16} {'probe2 nomma':>13} {'roof':>6}")
for e in d["experts"]:
    news = {k: v for k, v in e.items() if k.startswith("new")}
    bk = min(news, key=lambda k: news[k]["us"])
    p1 = e.get("probe 1 (no decode)", {}); p2 = e.get("probe 2 (no mma)", {})
    print(f"{e['window']:22} {e['old_us']:6.0f}/{e['old_GBs']:4.0f} {news[bk]['us']:7.0f}/{news[bk]['GBs']:4.0f} {bk[4:]:>17} "
          f"{p1.get('us',0):7.0f}/{p1.get('GBs',0):4.0f}  {p2.get('us',0):6.0f}/{p2.get('GBs',0):4.0f} {e['roof_GBs']:6.0f}")
print()
print("E2 dense q4 (_qmm + _reduce vs q4_stream): us")
q = d["qmm"]
keys = list(q[0].keys()) if isinstance(q, list) else None
print(keys)
for r in q[:200]:
    news = {k: v for k, v in r.items() if isinstance(v, dict) and "us" in v and k.startswith("new")}
    if not news:
        continue
    bk = min(news, key=lambda k: news[k]["us"])
    print(f"{str(r.get('shape')):>14} M={r.get('rows', r.get('M')):>3} old {r['old_us']:7.1f} ({r.get('old_GBs',0):4.0f})  new {news[bk]['us']:7.1f} ({news[bk].get('GBs',0):4.0f}) {bk[4:]:>22}  roof {r.get('roof_GBs',0):4.0f}  x{r['old_us']/news[bk]['us']:.2f}")
