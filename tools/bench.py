#!/usr/bin/env python3
"""Prefill and decode benchmark against the running server.

Every prompt is fresh random prose (a new seed per run), so no kept prompt state (prefix reuse) ever matches and every
token is prefilled. Reports the server's own prefill seconds and decode tokens/s.

Usage: tools/bench.py [label]      (API_URL, default http://127.0.0.1:$PORT with PORT 8000; DECODE_ONLY=1 skips prefill)
"""
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request

API_URL = os.environ.get("API_URL", "http://127.0.0.1:" + os.environ.get("PORT", "8000")).rstrip("/")
URL = API_URL + "/v1/chat/completions"
WORDS = ("time year people way day man thing woman life child world school state family student group country "
         "problem hand part place case week company system program question work government number night point "
         "home water room mother area money story fact month lot right study book eye job word business issue "
         "side kind head house service friend father power hour game line end member law car city community name "
         "president team minute idea kid body information back parent face others level office door health person "
         "art war history party result change morning reason research girl guy moment air teacher force education "
         "river mountain signal engine garden theory market winter method bridge letter window voice paper field").split()
SIZES = [(1_000, 3), (4_000, 3), (16_000, 2), (64_000, 1)]      # (approx prompt tokens, runs)


def prose(tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    out = []
    while len(out) < tokens * 0.72:          # ~1.14 tokens a word: the prompt comes out at ~0.82 x tokens
        sentence = [rng.choice(WORDS) for _ in range(rng.randint(6, 16))]
        out += sentence[:-1] + [sentence[-1] + "."]
    return " ".join(out)


def open_url(req: urllib.request.Request, timeout: float):
    """urlopen, with a one-line message instead of a traceback when the server is not there or refuses."""

    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        sys.exit(f"the server at {API_URL} answered {exc.code}: {exc.read().decode(errors='replace')[:300]}")
    except urllib.error.URLError as exc:
        sys.exit(f"cannot reach the server at {API_URL} ({exc.reason}): is it running? (./start.sh)")


def run(prompt: str, max_tokens: int, temperature=None) -> dict:
    body = {"model": os.environ.get("MODEL", "GLM-5.3-Flash-EXL3"), "stream": True, "max_tokens": max_tokens,
            "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": prompt}]}
    if temperature is not None:
        body["temperature"] = temperature
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    start, first, stats = time.time(), None, {}
    for line in open_url(req, 3600):
        line = line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"):
            continue
        chunk = json.loads(line[5:])
        delta = (chunk.get("choices") or [{}])[0].get("delta", {})
        if first is None and (delta.get("content") or delta.get("reasoning_content")):
            first = time.time()
        if "tensorfold" in chunk:
            stats = {**chunk["tensorfold"], **chunk.get("usage", {})}
    stats["ttft"] = (first or time.time()) - start
    return stats


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    seed = int(time.time())
    print(f"== {label}")
    for tokens, runs in ([] if os.environ.get("DECODE_ONLY") else SIZES):
        rs = [run(prose(tokens, seed + i * 7919 + tokens) + "\n\nSummarize the text above in one sentence.", 16)
              for i in range(runs)]
        assert all(r.get("cached", 0) == 0 for r in rs), "prompt cache hit: prompts are not fresh"
        n = statistics.median(r["prompt_tokens"] for r in rs)
        pre = statistics.median(r["prefill_s"] for r in rs)
        ttft = statistics.median(r["ttft"] for r in rs)
        each = ", ".join("%.2f" % r["prefill_s"] for r in rs)
        print(f"prefill {n:>7,.0f} tok: {pre:7.2f} s  {n / pre:6.0f} tok/s  TTFT {ttft:6.2f} s"
              f"  ({runs} run{'s' if runs > 1 else ''}, each {each} s)")
    for name, prompt, temp in (("code greedy", "Write a Python quicksort with docstring and tests.", 0),
                               ("chat sampled", "Explain why the sky is blue in a few paragraphs.", None)):
        rs = [run(prompt, 256, temp) for _ in range(5)]
        tps = [r.get("decode_tps") or r["completion_tokens"] / r["decode_s"] for r in rs]   # parallel mode: no decode_tps
        print(f"decode {name:12s}: {statistics.median(tps):5.1f} tok/s (median of 5)")


if __name__ == "__main__":
    main()
