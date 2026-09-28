#!/usr/bin/env python3
"""One benchmark client for every GLM-5.3-Flash engine we compare (vLLM kit, TensorFold).

Standard library only, so it runs on a bare host. Every suite streams through the OpenAI API and
times decode the same way: (completion_tokens - 1) / (last content chunk - first content chunk).

Suites
  tf      TensorFold's published cells: 64-token replies, raw-completion code + chat-no-think,
          sampled (T 1, top-k 20, top-p 0.95) and greedy, seeds 1234.. (median of --reps).
  tweet   sequence / code / json prompts, thinking off, greedy, --long-tokens replies.
  kit     the vLLM kit's own decode benches: hashmap, structured (count 1-200), hard essay;
          200 tokens, greedy, thinking off (tests/bench_decode.py payloads, verbatim).
  ctx     prefill + decode behind a long filler prompt at each --ctx size.
  exact   drafted vs serial ("draft": false) token-id equality (TensorFold only).
  edit    agent-style file edits: rewrite a ~6k-character source file with small changes (copy-heavy output).

  python3 bench/glmbench.py --base http://127.0.0.1:8000 --model GLM-5.3-Flash-EXL3 \
      --suites tf,tweet,kit --label vllm-kit --out results/E1-vllm-kit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

TF_PROMPTS = [
    {"name": "code", "kind": "completion",
     "prompt": "Write a short Python function that computes the Fibonacci sequence and explain it."},
    {"name": "chat", "kind": "chat",
     "prompt": "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example."},
]
TWEET_PROMPTS = [
    {"name": "sequence", "kind": "chat",
     "prompt": "List the first 150 prime numbers in order, separated by commas. Output only the numbers."},
    {"name": "code", "kind": "chat",
     "prompt": "Write a complete Python module implementing an LRU cache class with get, put, delete and "
               "resize methods, full type hints and docstrings, followed by pytest unit tests for it."},
    {"name": "json", "kind": "chat",
     "prompt": "Output a JSON array of 25 fictional employee records. Each record has id, first_name, "
               "last_name, email, department, title, salary, start_date and a skills array. Output only JSON."},
]
KIT_PROMPTS = [
    {"name": "hashmap", "kind": "chat",
     "prompt": "Write a detailed step-by-step explanation of how a hash map works, including collision "
               "handling, resizing, and time complexity. Be thorough."},
    {"name": "structured", "kind": "chat",
     "prompt": "Count from 1 to 200. Output only the numbers, separated by spaces. No other text."},
    {"name": "essay", "kind": "chat",
     "prompt": "Write a detailed technical essay titled \"Speculative Decoding and the Hidden Cost of Failed "
               "Drafts: A Technical Analysis\". Cover draft generation, verification cost, rejection sampling, "
               "and when longer drafts stop paying. Use numbered sections. Be thorough."},
]
FILLER = ("The quick brown fox jumps over the lazy dog while the committee reviews quarterly logistics, "
          "inventory forecasts, and the maintenance schedule for the northern warehouse. ")


class Client:
    def __init__(self, base: str, model: str, key: str = "", extra: dict | None = None):
        self.base, self.model, self.key, self.extra = base.rstrip("/"), model, key, extra or {}

    def post(self, path: str, body: dict, timeout: float = 1800):
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers=headers)
        return urllib.request.urlopen(req, timeout=timeout)

    def stream(self, item: dict, tokens: int, temperature: float, seed: int | None = None,
               ignore_eos: bool = False, extra: dict | None = None) -> dict:
        body = {"model": self.model, "max_tokens": tokens, "temperature": temperature, "stream": True,
                "stream_options": {"include_usage": True}}
        if ignore_eos:
            body["ignore_eos"] = True
        if seed is not None:
            body["seed"] = seed
        if temperature > 0:
            body.update(top_k=20, top_p=0.95)
        else:
            body["top_p"] = 1
        if item["kind"] == "chat":
            path = "/v1/chat/completions"
            body["messages"] = [{"role": "user", "content": item["prompt"]}]
            body["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            path = "/v1/completions"
            body["prompt"] = item["prompt"]
        body.update(self.extra)
        body.update(extra or {})
        start = time.perf_counter()
        first = last = None
        usage, finish, text = None, None, []
        with self.post(path, body) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    d = choice.get("delta") or {}
                    piece = choice.get("text") or d.get("content") or d.get("reasoning") \
                        or d.get("reasoning_content") or ""
                    if piece:
                        now = time.perf_counter()
                        first = now if first is None else first
                        last = now
                        text.append(piece)
                    finish = choice.get("finish_reason") or finish
        n = int(usage["completion_tokens"]) if usage else None
        out = "".join(text)
        return {"ttft_s": None if first is None else first - start,
                "decode_s": None if first is None else last - first,
                "prompt_tokens": (usage or {}).get("prompt_tokens"), "tokens": n, "finish": finish,
                "decode_tps": (n - 1) / (last - first) if n and first is not None and last > first else None,
                "sha256": hashlib.sha256(out.encode()).hexdigest()[:16], "text_head": out[:160]}


def cell(c: Client, item: dict, tokens: int, temp: float, seeds: list, ignore_eos: bool, extra=None) -> dict:
    c.stream(item, min(tokens, 32), temp, seeds[0], ignore_eos, extra)  # warm-up
    runs = [c.stream(item, tokens, temp, s, ignore_eos, extra) for s in seeds]
    tps = [r["decode_tps"] for r in runs if r["decode_tps"]]
    res = {"prompt": item["name"], "temperature": temp, "tokens": tokens, "runs": runs,
           "median_tps": round(statistics.median(tps), 2) if tps else None,
           "min_tps": round(min(tps), 2) if tps else None, "max_tps": round(max(tps), 2) if tps else None}
    print(f"  {item['name']:10s} T={temp:<4} {tokens:>5} tok  median {res['median_tps']} "
          f"(min {res['min_tps']}, max {res['max_tps']})  ttft {runs[0]['ttft_s']:.2f}s", flush=True)
    return res


def suite_tf(c, a):
    out = []
    for temp in (1.0, 0.0):
        for item in TF_PROMPTS:
            out.append(cell(c, item, 64, temp, [1234 + i for i in range(a.reps)], True))
    return out


def suite_tweet(c, a):
    return [cell(c, item, a.long_tokens, 0.0, [None] * a.reps, False) for item in TWEET_PROMPTS]


def suite_kit(c, a):
    return [cell(c, item, 200, 0.0, [None] * a.reps, False) for item in KIT_PROMPTS]


def suite_ctx(c, a):
    out = []
    for n in [int(x) for x in a.ctx.split(",")]:
        reps = max(1, n // 32)  # ~32 tokens per filler sentence
        tag = f"[ctx-{n}-{time.time_ns()}] "  # unique prefix: no prefix-cache hit on the cold run
        body = tag + FILLER * reps + "\n\nIn one sentence, what is the committee reviewing? Then count from 1 to 100."
        item = {"name": f"ctx{n}", "kind": "chat", "prompt": body}
        cold = c.stream(item, 256, 0.0)
        warm = c.stream(item, 256, 0.0)
        pt = cold["prompt_tokens"] or 0
        res = {"ctx": n, "prompt_tokens": pt, "cold": cold, "warm": warm,
               "prefill_tps": round(pt / cold["ttft_s"], 1) if cold["ttft_s"] else None}
        print(f"  ctx {n:>7}  prompt {pt:>7} tok  cold ttft {cold['ttft_s']:.2f}s ({res['prefill_tps']} tok/s)  "
              f"warm ttft {warm['ttft_s']:.2f}s  decode {cold['decode_tps'] and round(cold['decode_tps'], 1)} tok/s",
              flush=True)
        out.append(res)
    return out


def suite_exact(c, a):
    out = []
    for item in TF_PROMPTS + TWEET_PROMPTS:
        for temp, seed in ((0.0, None), (1.0, 1234)):
            d = c.stream(item, 128, temp, seed, True)
            s = c.stream(item, 128, temp, seed, True, extra={"draft": False})
            ok = d["sha256"] == s["sha256"]
            print(f"  {item['name']:10s} T={temp}  drafted {d['decode_tps'] and round(d['decode_tps'], 1)} tok/s  "
                  f"serial {s['decode_tps'] and round(s['decode_tps'], 1)} tok/s  identical={ok}", flush=True)
            out.append({"prompt": item["name"], "temperature": temp, "identical": ok, "drafted": d, "serial": s})
    return out


EDIT_SOURCE = Path(__file__).resolve()          # this file: ~9k characters of real Python to edit


def suite_edit(c, a):
    """Agent-style file edits: the model rewrites a source file it was given with small changes, the workload
    where most output tokens are copies of the prompt (what lookup drafting targets)."""

    src = EDIT_SOURCE.read_text()[:6000]
    tasks = [
        ("rename", "Rename the function `cell` to `run_cell` everywhere and output the complete updated file."),
        ("comments", "Add a one-line comment above every function definition and output the complete updated file."),
        ("print-to-log", "Change every `print(` call to `log(` and output the complete updated file."),
    ]
    out = []
    for name, ask in tasks:
        item = {"name": f"edit-{name}", "kind": "chat",
                "prompt": f"Here is a Python file:\n\n```python\n{src}\n```\n\n{ask} Output only the code."}
        out.append(cell(c, item, a.edit_tokens, 0.0, [None] * max(1, a.reps // 2), False))
    return out


SUITES = {"tf": suite_tf, "tweet": suite_tweet, "kit": suite_kit, "ctx": suite_ctx, "exact": suite_exact,
          "edit": suite_edit}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--key", default="")
    p.add_argument("--suites", default="tf,tweet,kit")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--long-tokens", type=int, default=512)
    p.add_argument("--edit-tokens", type=int, default=1024)
    p.add_argument("--ctx", default="2000,8000,32000")
    p.add_argument("--extra", default="{}", help="JSON merged into every request body")
    p.add_argument("--label", default="")
    p.add_argument("--out")
    a = p.parse_args()
    c = Client(a.base, a.model, a.key, json.loads(a.extra))
    rec = {"label": a.label, "base": a.base, "model": a.model, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "args": vars(a), "suites": {}}
    for name in a.suites.split(","):
        print(f"[{a.label}] suite {name}", flush=True)
        rec["suites"][name] = SUITES[name](c, a)
        if a.out:
            with open(a.out, "w") as f:
                json.dump(rec, f, indent=1)
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rec, f, indent=1)
    print(json.dumps({s: [(r.get("prompt") or r.get("ctx"), r.get("temperature"), r.get("median_tps") or
                           r.get("prefill_tps") or r.get("identical")) for r in v]
                      for s, v in rec["suites"].items()}), file=sys.stderr)


if __name__ == "__main__":
    main()
