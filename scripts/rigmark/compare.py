#!/usr/bin/env python3
"""Side-by-side markdown for two RigMark receipts (vLLM kit left, TensorFold right).

    compare.py VLLM (dir or json) TENSORFOLD (dir or json) --rigmark DIR [--out FILE]

Writes: a headline table (medians, TensorFold / vLLM ratio, lower-is-better marked), the basic output gates, output
sizes, the appliance metadata side by side, then RigMark's own `compare` table (median [min-max] of every metric) and
its matched A/B card. RigMark's comparison is strict first; if the receipts differ in protocol / settings it is re-run
with --allow-mismatch and the mismatch is printed at the top (e.g. one side ran --skip-prefill).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


def receipt(path: Path) -> Path:
    if path.is_file():
        return path
    skip = {"metadata.json", "preflight.json", "models.json"}
    found = [p for p in sorted(path.glob("*.json")) if p.name not in skip]
    if len(found) != 1:
        raise SystemExit(f"{path}: expected one receipt json, found {[p.name for p in found]}")
    return found[0]


def get(d: Any, *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def med(d: Any, *keys: str) -> float | None:
    v = get(d, *keys, "median")
    return float(v) if isinstance(v, (int, float)) else None


def fmt(v: float | None, prec: int = 1) -> str:
    return "—" if v is None else f"{v:,.{prec}f}"


def rigmark(rdir: Path, *args: str) -> tuple[int, str]:
    p = subprocess.run([sys.executable, str(rdir / "rigmark"), *args], capture_output=True, text=True,
                       env={"NO_COLOR": "1", "PATH": "/usr/bin:/bin"})
    return p.returncode, (p.stdout + p.stderr).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("left", type=Path)
    ap.add_argument("right", type=Path)
    ap.add_argument("--rigmark", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    lp, rp = receipt(a.left), receipt(a.right)
    L, R = json.loads(lp.read_text()), json.loads(rp.read_text())
    ln, rn = L["run"]["label"], R["run"]["label"]
    md: list[str] = [f"# RigMark: {ln} vs {rn}", ""]

    rc, strict = rigmark(a.rigmark, "compare", str(lp), str(rp))
    if rc != 0:
        md += ["> **Not a strict match.** RigMark refused the plain comparison:", ">", f"> `{strict.splitlines()[-1]}`",
               ">", "> The tables below use `--allow-mismatch`; only the matching phases are like for like.", ""]
        rc2, table = rigmark(a.rigmark, "compare", "--allow-mismatch", str(lp), str(rp))
        card = ""
    else:
        table = strict
        _, card = rigmark(a.rigmark, "compare", "--card", str(lp), str(rp))

    md += [f"Comparison ID `{get(L, 'run', 'comparison_id')}` / `{get(R, 'run', 'comparison_id')}`; request body "
           f"`{json.dumps(get(L, 'settings', 'extra_body'))}` / `{json.dumps(get(R, 'settings', 'extra_body'))}`; "
           f"RigMark `{str(get(L, 'protocol', 'repository_revision'))[:12]}` protocol "
           f"{get(L, 'protocol', 'version')}. Started {get(L, 'run', 'started_at')} / {get(R, 'run', 'started_at')}.",
           "", "## Headline (medians)", "",
           f"| Measurement | {ln} | {rn} | right / left |", "|---|---:|---:|---:|"]
    rows: list[tuple[str, tuple[str, ...], bool]] = [
        ("Code decode estimate, tok/s", ("decode", "code", "decode_tokens_per_second"), False),
        ("Code time to last output, s ↓", ("decode", "code", "time_to_last_output_seconds"), True),
        ("Code TTFT, s ↓", ("decode", "code", "ttft_seconds"), True),
        ("Prose decode estimate, tok/s", ("decode", "prose", "decode_tokens_per_second"), False),
        ("Prose time to last output, s ↓", ("decode", "prose", "time_to_last_output_seconds"), True),
        ("Structured ceiling, tok/s", ("decode", "structured", "decode_tokens_per_second"), False),
        ("Structured time to last output, s ↓", ("decode", "structured", "time_to_last_output_seconds"), True),
    ]
    for depth in get(L, "settings", "prefill_depths") or get(R, "settings", "prefill_depths") or []:
        rows += [(f"Cold prefill {depth // 1024}K, tok/s", ("prefill", str(depth), "cold",
                                                           "effective_prefill_tokens_per_second"), False),
                 (f"Immediate replay {depth // 1024}K, tok/s", ("prefill", str(depth), "warm_replay",
                                                               "effective_prefill_tokens_per_second"), False)]
    for level in get(L, "settings", "concurrency") or get(R, "settings", "concurrency") or []:
        rows += [(f"C{level} short code-load end-to-end, tok/s",
                  ("concurrency", str(level), "aggregate_end_to_end_tokens_per_second"), False),
                 (f"C{level} per-stream TTFT, s ↓", ("concurrency", str(level), "per_stream_ttft_seconds"), True)]
    for name, path, _lower in rows:
        lv, rv = med(L, *path), med(R, *path)
        prec = 2 if "s ↓" in name else 1
        ratio = "—" if lv in (None, 0) or rv is None else f"{rv / lv:.2f}×"
        md.append(f"| {name} | {fmt(lv, prec)} | {fmt(rv, prec)} | {ratio} |")

    md += ["", "## Basic output gates and output sizes", "", f"L = {ln}, R = {rn}; token and character counts are "
           "medians over the five runs.", "",
           "| Workload | gate L | gate R | completion tokens L | completion tokens R | reasoning chars L "
           "| reasoning chars R |", "|---|---:|---:|---:|---:|---:|---:|"]
    for w in ("code", "prose", "structured"):
        cells = []
        for d in (L, R):
            g = get(d, "decode", w, "completion_gate") or {}
            cells.append(f"{g.get('passed', '?')}/{g.get('total', '?')}")
        for key in ("completion_tokens", "reasoning_characters"):
            for d in (L, R):
                runs = get(d, "decode", w, "runs") or []
                vals = [r[key] for r in runs if isinstance(r.get(key), int)]
                cells.append(fmt(statistics.median(vals), 0) if vals else "—")
        md.append(f"| {w} | " + " | ".join(cells) + " |")
    md += ["", "Reasoning characters are as RigMark counted them (it adds `delta.reasoning` and "
           "`delta.reasoning_content`; a server sending both doubles its count, see preflight.json).", ""]

    md += ["## Appliance metadata", "", f"| Field | {ln} | {rn} |", "|---|---|---|"]
    lm, rm = get(L, "run", "appliance") or {}, get(R, "run", "appliance") or {}
    for k in list(dict.fromkeys([*lm, *rm])):
        def cell(v: Any) -> str:
            v = json.dumps(v) if isinstance(v, (dict, list)) else str(v) if v is not None else "—"
            return v.replace("|", "\\|")
        md.append(f"| {k} | {cell(lm.get(k))} | {cell(rm.get(k))} |")

    md += ["", "## RigMark compare (median [min–max])", "", table, ""]
    if card:
        md += ["## Matched A/B card", "", "```text", card, "```", ""]
    md += [f"Receipts: `{lp.parent.name}/{lp.name}`, `{rp.parent.name}/{rp.name}`", ""]
    text = "\n".join(md)
    if a.out:
        a.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
