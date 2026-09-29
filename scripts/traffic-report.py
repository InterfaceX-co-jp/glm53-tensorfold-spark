#!/usr/bin/env python3
"""Summarize the per-request log of patches/0300 (GLM53_TF_REQUEST_LOG, one JSON line a request).

    scripts/traffic-report.py /sessions/requests.jsonl            # the file and its rotated .1 .2 ... siblings
    scripts/traffic-report.py log.jsonl --since 2026-09-28T12:00 --json

What it reports:

- volume: requests, errors / cancels, finish reasons, thinking / effort mix, time span;
- sizes: prompt, cached, prefilled, decode tokens, max_tokens (percentiles and a histogram);
- speed: prefill s and tok/s, decode tok/s, tokens a round, queue wait, time to first delta;
- reuse today: share of requests that resumed anything and of prompt tokens resumed, by source (slot / RAM / NVMe);
- shared-prefix headroom (patches/0310): per request, the tokens a snapshot at its fork point with ANOTHER
  conversation would have saved, ``max(0, floor64(min(lcp_other, prompt - 1)) - cached)``, and the prefill seconds
  that is (the request's own prefill rate: seconds x avoidable / prefilled). The same for the same conversation
  (``lcp_same``: a turn that should have resumed and did not, e.g. an eviction) and for either;
- shared prefixes: the most common system-prompt hashes (``sys_hash``) and 4k-token heads (``head_hash``), with
  their requests, distinct conversations, lengths and resume rate.

No dependencies beyond the standard library.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import json
import math
import os
import sys
from typing import Any, Iterable

GRID = 64                                  # fast prefill snapshots sit on multiples of 64 (patches/0085)
BUCKETS = [(0, 1024), (1024, 4096), (4096, 16384), (16384, 65536), (65536, 262144), (262144, 1 << 40)]


def files_of(path: str) -> list[str]:
    """The log and its rotated siblings, oldest first (``<file>.N`` ... ``<file>.1``, ``<file>``)."""

    rotated = []
    for p in glob.glob(glob.escape(path) + ".*"):
        tail = p[len(path) + 1:]
        if tail.isdigit():
            rotated.append((int(tail), p))
    out = [p for _, p in sorted(rotated, reverse=True)]
    if os.path.exists(path):
        out.append(path)
    return out


def load(paths: Iterable[str], since: float | None = None, until: float | None = None) -> tuple[list[dict], int]:
    recs, bad = [], 0
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1                       # a line cut by a crash mid-write
                    continue
                ts = r.get("ts") or 0
                if (since is not None and ts < since) or (until is not None and ts > until):
                    continue
                recs.append(r)
    recs.sort(key=lambda r: (r.get("start") or 0, r.get("n") or 0))
    return recs, bad


def pct(values: list[float], q: float) -> float | None:
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    i = min(len(v) - 1, max(0, math.ceil(q / 100 * len(v)) - 1))
    return v[i]


def dist(values: list) -> dict[str, Any]:
    v = [x for x in values if isinstance(x, (int, float))]
    if not v:
        return {"n": 0}
    return {"n": len(v), "mean": round(sum(v) / len(v), 3), "p50": pct(v, 50), "p90": pct(v, 90), "p99": pct(v, 99),
            "max": max(v)}


def hist(values: list) -> dict[str, int]:
    out = collections.OrderedDict()
    for lo, hi in BUCKETS:
        label = f"{_k(lo)}-{_k(hi)}" if hi < 1 << 40 else f"{_k(lo)}+"
        out[label] = sum(1 for x in values if isinstance(x, (int, float)) and lo <= x < hi)
    return out


def _k(n: int) -> str:
    return f"{n // 1024}k" if n >= 1024 else str(n)


def floor_grid(n: int) -> int:
    return n // GRID * GRID


def avoidable(r: dict, key: str) -> tuple[int, float]:
    """(tokens, seconds) of this request's prefill a snapshot at its common prefix ``key`` would have saved."""

    prompt, cached = int(r.get("prompt") or 0), int(r.get("cached") or 0)
    lcp = r.get(key)
    if not lcp or prompt <= 1:
        return 0, 0.0
    tokens = max(0, floor_grid(min(int(lcp), prompt - 1)) - cached)
    prefilled = prompt - cached
    s = r.get("prefill_s") or 0.0
    return tokens, (s * tokens / prefilled if prefilled > 0 else 0.0)


def report(recs: list[dict], *, top: int = 10, min_share: int = 1024) -> dict[str, Any]:
    ok = [r for r in recs if r.get("finish") != "error"]
    out: dict[str, Any] = {"requests": len(recs)}
    if not recs:
        return out
    t0, t1 = min(r.get("start") or r.get("ts") for r in recs), max(r.get("ts") for r in recs)
    out["span"] = {"from": _iso(t0), "to": _iso(t1), "hours": round((t1 - t0) / 3600, 2)}
    out["finish"] = dict(collections.Counter(r.get("finish") for r in recs).most_common())
    out["thinking"] = dict(collections.Counter(
        ("off" if r.get("thinking") is False else (r.get("effort") or "default")) if r.get("kind") == "chat"
        else "completion" for r in recs).most_common())
    prompts = [r.get("prompt") for r in ok]
    prefilled = [(r.get("prompt") or 0) - (r.get("cached") or 0) for r in ok]
    out["sizes"] = {
        "prompt": dist(prompts), "prompt_hist": hist(prompts),
        "cached": dist([r.get("cached") for r in ok]),
        "prefilled": dist(prefilled), "prefilled_hist": hist(prefilled),
        "decode_tokens": dist([r.get("decode_tokens") for r in ok]),
        "max_tokens_eff": dist([r.get("max_tokens_eff") for r in ok]),
    }
    out["speed"] = {
        "prefill_s": dist([r.get("prefill_s") for r in ok]),
        "prefill_tps": dist([r.get("prefill_tps") for r in ok if (r.get("prompt") or 0) - (r.get("cached") or 0)
                             >= 1024]),
        "decode_tps": dist([r.get("decode_tps") for r in ok if (r.get("decode_tokens") or 0) >= 16]),
        "tokens_per_round": dist([r.get("tokens_per_round") for r in ok]),
        "queue_s": dist([r.get("queue_s") for r in ok]),
        "first_s": dist([r.get("first_s") for r in ok]),
    }
    total_prompt = sum(r.get("prompt") or 0 for r in ok)
    total_cached = sum(r.get("cached") or 0 for r in ok)
    total_prefill_s = sum(r.get("prefill_s") or 0.0 for r in ok)
    by_src = collections.defaultdict(lambda: [0, 0])
    for r in ok:
        by_src[r.get("cache_src") or "none"][0] += 1
        by_src[r.get("cache_src") or "none"][1] += r.get("cached") or 0
    out["reuse"] = {
        "requests_resumed": _share(sum(1 for r in ok if (r.get("cached") or 0) > 0), len(ok)),
        "prompt_tokens_resumed": _share(total_cached, total_prompt),
        "by_source": {k: {"requests": v[0], "tokens": v[1]} for k, v in sorted(by_src.items())},
        "prefill_s_total": round(total_prefill_s, 1),
    }
    head = {}
    for key, label in (("lcp_other", "other_conversation"), ("lcp_same", "same_conversation"), ("lcp", "any")):
        toks = secs = 0.0
        hits = 0
        for r in ok:
            t, s = avoidable(r, key)
            if t >= min_share:
                hits += 1
                toks += t
                secs += s
        head[label] = {"requests": _share(hits, len(ok)), "tokens": int(toks),
                       "prefill_s": round(secs, 1), "prefill_share": _share(secs, total_prefill_s)}
    out["avoidable_prefill"] = head
    out["avoidable_prefill"]["note"] = (f"requests whose missed common prefix is >= {min_share} tokens; seconds at the "
                                        "request's own prefill rate")
    new_sessions = [r for r in ok if (r.get("lcp_same") or 0) == 0]
    out["new_conversations"] = {
        "requests": len(new_sessions),
        "sharing_a_prefix": _share(sum(1 for r in new_sessions if (r.get("lcp_other") or 0) >= min_share),
                                   len(new_sessions)),
        "lcp_other": dist([r.get("lcp_other") for r in new_sessions]),
        "cached": dist([r.get("cached") for r in new_sessions]),
    }
    out["shared_prefixes"] = {"sys_hash": _groups(ok, "sys_hash", "sys_len", top),
                              "head_hash": _groups(ok, "head_hash", "head_len", top)}
    return out


def _groups(recs: list[dict], key: str, length: str, top: int) -> list[dict]:
    g = collections.defaultdict(list)
    for r in recs:
        if r.get(key):
            g[r[key]].append(r)
    rows = []
    for h, rs in sorted(g.items(), key=lambda kv: -len(kv[1]))[:top]:
        rows.append({"hash": h, "requests": len(rs), "conversations": len({r.get("conv") for r in rs}),
                     "tokens": pct([r.get(length) for r in rs], 50),
                     "resumed": _share(sum(1 for r in rs if (r.get("cached") or 0) > 0), len(rs)),
                     "prefill_s": round(sum(r.get("prefill_s") or 0.0 for r in rs), 1)})
    return rows


def _share(a: float, b: float) -> float | None:
    return round(a / b, 4) if b else None


def _iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t).isoformat(timespec="seconds")


def _when(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return dt.datetime.fromisoformat(s).timestamp()


def text(rep: dict[str, Any]) -> str:
    L = []
    L.append(f"requests: {rep['requests']}")
    if not rep["requests"]:
        return "\n".join(L)
    s = rep["span"]
    L.append(f"span: {s['from']} .. {s['to']} ({s['hours']} h)")
    L.append("finish: " + ", ".join(f"{k} {v}" for k, v in rep["finish"].items()))
    L.append("thinking: " + ", ".join(f"{k} {v}" for k, v in rep["thinking"].items()))
    L.append("")
    L.append("sizes (tokens)            n     mean      p50      p90      p99      max")
    for k in ("prompt", "cached", "prefilled", "decode_tokens", "max_tokens_eff"):
        L.append(_row(k, rep["sizes"][k]))
    L.append("prompt hist:    " + "  ".join(f"{k} {v}" for k, v in rep["sizes"]["prompt_hist"].items()))
    L.append("prefilled hist: " + "  ".join(f"{k} {v}" for k, v in rep["sizes"]["prefilled_hist"].items()))
    L.append("")
    L.append("speed                     n     mean      p50      p90      p99      max")
    for k in ("prefill_s", "prefill_tps", "decode_tps", "tokens_per_round", "queue_s", "first_s"):
        L.append(_row(k, rep["speed"][k]))
    L.append("")
    r = rep["reuse"]
    L.append(f"reuse: {_p(r['requests_resumed'])} of requests resumed a prefix; {_p(r['prompt_tokens_resumed'])} of "
             f"prompt tokens were resumed; prefill total {r['prefill_s_total']} s")
    L.append("  by source: " + ", ".join(f"{k} {v['requests']} req / {v['tokens']} tok"
                                         for k, v in r["by_source"].items()))
    a = rep["avoidable_prefill"]
    L.append(f"avoidable prefill ({a['note']}):")
    for k in ("other_conversation", "same_conversation", "any"):
        x = a[k]
        L.append(f"  {k:<20} {_p(x['requests'])} of requests, {x['tokens']} tokens, {x['prefill_s']} s "
                 f"= {_p(x['prefill_share'])} of prefill time")
    n = rep["new_conversations"]
    L.append(f"new conversations: {n['requests']}, {_p(n['sharing_a_prefix'])} share a prefix with another "
             f"(lcp_other p50 {n['lcp_other'].get('p50')}, cached p50 {n['cached'].get('p50')})")
    for key in ("sys_hash", "head_hash"):
        L.append(f"top {key}: hash  requests  conversations  tokens(p50)  resumed  prefill_s")
        for g in rep["shared_prefixes"][key]:
            L.append(f"  {g['hash']}  {g['requests']:>6}  {g['conversations']:>6}  {g['tokens']}  "
                     f"{_p(g['resumed'])}  {g['prefill_s']}")
    return "\n".join(L)


def _row(k: str, d: dict) -> str:
    if not d.get("n"):
        return f"{k:<20} {0:>6}"
    f = lambda v: f"{v:>8.1f}" if isinstance(v, float) else f"{v:>8}"      # noqa: E731
    return f"{k:<20} {d['n']:>6} {f(d['mean'])} {f(d['p50'])} {f(d['p90'])} {f(d['p99'])} {f(d['max'])}"


def _p(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="GLM53_TF_REQUEST_LOG file (rotated .N siblings are read too)")
    ap.add_argument("--since", help="ISO time or unix seconds")
    ap.add_argument("--until", help="ISO time or unix seconds")
    ap.add_argument("--min-share", type=int, default=1024, help="fewest tokens of a missed prefix counted (1024)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--no-rotated", action="store_true", help="read only the file itself")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    paths = [a.log] if a.no_rotated else files_of(a.log)
    if not paths:
        print(f"no log at {a.log}", file=sys.stderr)
        return 1
    recs, bad = load(paths, _when(a.since), _when(a.until))
    rep = report(recs, top=a.top, min_share=a.min_share)
    rep["files"], rep["bad_lines"] = paths, bad
    print(json.dumps(rep, indent=1) if a.json else text(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
