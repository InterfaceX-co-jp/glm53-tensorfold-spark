#!/usr/bin/env python3
"""Post-load canary: a few short fixed greedy requests that fail the start if the engine is degenerate.

A load can finish, answer ``/v1/models`` and still be broken: a rank that loaded a stale/corrupt kernel cache, a
drafter paired with the wrong weights (every draft rejected: serial speed), NaNs that decode as one repeated token,
or a mismatched rank that emits word salad. The MiaAI-Lab vLLM kit fails its boot on such an engine (its
"degenerate-engine canary", PR #268); this is the same idea over our OpenAI endpoint, stdlib only, so
``scripts/serve.sh`` can run it on the head node without the image.

Each probe is greedy, thinking off, a few dozen tokens, and checks:

- the reply is non-empty, has no U+FFFD, and is not a loop (distinct-word ratio, longest run of one word);
- it contains an expected answer (``Paris``; the numbers 1..12 in order);
- with drafts on, the engine's ``tensorfold.tokens_per_round`` over all probes is at least ``--min-tpr``
  (a dead drafter decodes one token a round);
- optionally, decode tok/s at least ``--min-tps``.

Each probe carries a per-run nonce so a session-cache hit cannot answer it from a stored state. ``--warm-lengths``
then prefills filler prompts of about those many tokens (``max_tokens`` 4, not judged): the MiaAI-Lab kit's boot-shape
warmup (PRs #203, #254) moved first-use kernel compiles / graph captures out of the first real long request.

The probes double as the first-request warmup (Triton autotune, long-context graphs, allocator growth) that a user
would otherwise pay for.

Exit 0: pass. Exit 1: a probe failed (details on stderr). Exit 2: the server could not be reached.

    scripts/canary.py --base http://127.0.0.1:8080 [--model NAME] [--min-tpr 1.3] [--min-tps 0] [--json]
                      [--warm-lengths "4096 16384"]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

PROBES: list[dict[str, Any]] = [
    {
        "name": "capital",
        "prompt": "What is the capital of France? Answer with one word.",
        "max_tokens": 16,
        "expect": [r"\bParis\b"],
    },
    {
        "name": "count",
        "prompt": "Count from 1 to 12, separated by single spaces. Output only the numbers.",
        "max_tokens": 48,
        "expect": [r"\b1\D+2\D+3\D+4\D+5\D+6\D+7\D+8\D+9\D+10\D+11\D+12\b"],
    },
    {
        "name": "code",
        "prompt": "Write a Python function add(a, b) that returns a + b. Output only the code, no explanation.",
        "max_tokens": 48,
        "expect": [r"def\s+add\s*\(", r"return"],
    },
]


@dataclass
class ProbeResult:
    name: str
    ok: bool
    problems: list[str] = field(default_factory=list)
    content: str = ""
    completion_tokens: int = 0
    rounds: int = 0
    decode_s: float = 0.0
    seconds: float = 0.0


def degenerate(text: str, *, min_distinct: float = 0.3, max_run: int = 6) -> str | None:
    """Why ``text`` looks like a broken engine's output, or None. Short answers pass on content alone."""

    if not text.strip():
        return "empty reply"
    if "�" in text:
        return "replacement character (U+FFFD) in the reply"
    words = re.findall(r"\S+", text)
    if len(words) >= 12:
        distinct = len(set(words)) / len(words)
        if distinct < min_distinct:
            return f"repetitive reply ({distinct:.2f} distinct words)"
    run = best = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    if best > max_run:
        return f"the same word {best} times in a row"
    # one character repeated (e.g. '!!!!!!!!' or a single token looping without spaces)
    if re.search(r"(.)\1{23,}", text):
        return "one character repeated 24+ times"
    return None


def post(base: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def run_probe(base: str, model: str | None, probe: dict[str, Any], timeout: float, draft: bool,
              nonce: str = "") -> ProbeResult:
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": (f"[canary {nonce}] " if nonce else "") + probe["prompt"]}],
        "max_tokens": probe["max_tokens"],
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if model:
        body["model"] = model
    if not draft:
        body["draft"] = False
    t0 = time.perf_counter()
    payload = post(base, body, timeout)
    res = ProbeResult(probe["name"], True, seconds=time.perf_counter() - t0)
    choice = (payload.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    res.content = msg.get("content") or ""
    res.completion_tokens = int((payload.get("usage") or {}).get("completion_tokens", 0) or 0)
    stats = payload.get("tensorfold") or {}
    res.rounds = int(stats.get("rounds", 0) or 0)
    res.decode_s = float(stats.get("decode_s", 0.0) or 0.0)
    why = degenerate(res.content)
    if why:
        res.problems.append(why)
    for pattern in probe["expect"]:
        if not re.search(pattern, res.content):
            res.problems.append(f"expected /{pattern}/ in the reply")
    res.ok = not res.problems
    return res


def check(base: str, model: str | None = None, *, min_tpr: float = 1.3, min_tps: float = 0.0,
          timeout: float = 300.0, draft: bool = True, probes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Run every probe; the verdict and the per-probe details (JSON-able)."""

    nonce = f"{os.getpid()}-{int(time.time())}"
    results = [run_probe(base, model, p, timeout, draft, nonce) for p in (probes or PROBES)]
    problems = [f"{r.name}: {p}" for r in results for p in r.problems]
    # decode tokens exclude the first (it comes from the prefill)
    tokens = sum(max(r.completion_tokens - 1, 0) for r in results if r.rounds)
    rounds = sum(r.rounds for r in results)
    decode_s = sum(r.decode_s for r in results)
    tpr = tokens / rounds if rounds else None
    tps = tokens / decode_s if decode_s > 0 else None
    if draft and min_tpr > 0 and tpr is not None and tokens >= 24 and tpr < min_tpr:
        problems.append(f"drafter: {tpr:.2f} tokens a round < {min_tpr} (drafts rejected: wrong or broken drafter?)")
    if min_tps > 0 and tps is not None and tps < min_tps:
        problems.append(f"decode {tps:.1f} tok/s < {min_tps}")
    return {"ok": not problems, "problems": problems, "tokens_per_round": tpr, "decode_tps": tps,
            "probes": [r.__dict__ for r in results]}


def warm(base: str, model: str | None, lengths: list[int], timeout: float) -> list[dict[str, Any]]:
    """Prefill a filler prompt of about each length (one word ~ one token); the prefill seconds of each."""

    out = []
    for n in lengths:
        nonce = f"{os.getpid()}-{int(time.time())}-{n}"
        text = f"[warmup {nonce}] The following is filler context, ignore it: " + "warm " * max(n - 24, 1) + \
            "\nReply with OK."
        body: dict[str, Any] = {"messages": [{"role": "user", "content": text}], "max_tokens": 4, "temperature": 0,
                                "chat_template_kwargs": {"enable_thinking": False}}
        if model:
            body["model"] = model
        t0 = time.perf_counter()
        payload = post(base, body, timeout)
        out.append({"length": n, "prompt_tokens": (payload.get("usage") or {}).get("prompt_tokens"),
                    "prefill_s": (payload.get("tensorfold") or {}).get("prefill_s"),
                    "seconds": round(time.perf_counter() - t0, 2)})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default=None)
    ap.add_argument("--min-tpr", type=float, default=1.3,
                    help="least tokens a round over the probes with drafts on (0: skip the drafter check)")
    ap.add_argument("--min-tps", type=float, default=0.0, help="least decode tok/s (0: skip)")
    ap.add_argument("--no-draft", action="store_true", help="the server runs without drafts (skip the drafter check)")
    ap.add_argument("--timeout", type=float, default=300.0, help="seconds a probe (the first may compile kernels)")
    ap.add_argument("--json", action="store_true", help="print the full report as JSON on stdout")
    ap.add_argument("--warm-lengths", default="", help="space/comma separated prompt lengths to prefill after a pass")
    args = ap.parse_args(argv)
    try:
        lengths = [int(x) for x in re.split(r"[,\s]+", args.warm_lengths.strip()) if x]
    except ValueError:
        ap.error(f"--warm-lengths {args.warm_lengths!r}: integers expected")
    try:
        report = check(args.base, args.model, min_tpr=0.0 if args.no_draft else args.min_tpr, min_tps=args.min_tps,
                       timeout=args.timeout, draft=True)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"[canary] cannot query {args.base}: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=1))
    tpr = report["tokens_per_round"]
    tps = report["decode_tps"]
    summary = (f"tokens/round {tpr:.2f}" if tpr is not None else "tokens/round n/a") + \
        (f", decode {tps:.1f} tok/s" if tps is not None else "")
    if report["ok"]:
        print(f"[canary] ok: {len(report['probes'])} probes, {summary}", file=sys.stderr)
        if lengths:
            try:
                for w in warm(args.base, args.model, lengths, max(args.timeout, 3600.0)):
                    print(f"[canary] warmup {w['length']}: {w['prompt_tokens']} prompt tokens, prefill "
                          f"{w['prefill_s']} s ({w['seconds']} s)", file=sys.stderr)
            except (urllib.error.URLError, OSError, ValueError) as exc:     # warmup never fails the start
                print(f"[canary] warmup incomplete: {exc}", file=sys.stderr)
        return 0
    for p in report["problems"]:
        print(f"[canary] FAIL {p}", file=sys.stderr)
    for r in report["probes"]:
        print(f"[canary]   {r['name']}: {r['content'][:160]!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
