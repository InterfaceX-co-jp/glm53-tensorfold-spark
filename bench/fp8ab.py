#!/usr/bin/env python3
"""FP8 prefill quality A/B on a loaded TensorFold server: the same greedy requests with tf_knobs.fp8_prefill 0 and 1
(--knob fast_prefill: the same test for bf16 fast vs exact prefill, as a control).

  agree   10 short prompts (tf / tweet / kit / edit) + --long prompts built from real repo text (docs and code, 8k-28k
          tokens): identical greedy --tokens replies?  Otherwise the first divergence (characters, ~tokens).
  needle  a unique key/value line at 10% / 50% / 90% depth in --needle-ctx tokens of repo text; --trials each: accuracy.
          --values 1 runs only the knob's value 1 (e.g. a load-time A/B such as GLM53_TF_KV_DTYPE, knob fast_prefill).
  replies across two loads (a load-time switch, e.g. GLM53_TF_KV_DTYPE=bf16 vs fp8): the agree prompt set, greedy,
          with no knob; --save FILE keeps the replies, --against FILE (saved on the other load) reports identical /
          first divergence per prompt.

  python3 bench/fp8ab.py --base http://127.0.0.1:8080 --model GLM-5.3-Flash-Uncensored --corpus docs vendor/TensorFold/src
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from glmbench import KIT_PROMPTS, TF_PROMPTS, TWEET_PROMPTS, Client, suite_edit  # noqa: E402,F401
from multiturn import chat  # noqa: E402

CHARS_PER_TOKEN = 3.3        # measured roughly on this repo's markdown + Python with the GLM tokenizer


def corpus(paths: list[str]) -> str:
    parts = []
    for p in paths:
        for f in sorted(Path(p).rglob("*")):
            if f.suffix in (".md", ".py", ".cu", ".cpp") and f.is_file():
                parts.append(f"\n\n# file: {f}\n" + f.read_text(errors="ignore"))
    return "".join(parts)


def slice_tokens(text: str, tokens: int, rng: random.Random) -> str:
    n = int(tokens * CHARS_PER_TOKEN)
    if len(text) <= n:
        return (text * (n // max(len(text), 1) + 1))[:n]
    a = rng.randrange(0, len(text) - n)
    return text[a:a + n]


def edit_prompts() -> list[dict]:
    src = (Path(__file__).parent / "glmbench.py").read_text()[:6000]
    asks = ["Rename the function `cell` to `run_cell` everywhere and output the complete updated file.",
            "Add a one-line comment above every function definition and output the complete updated file.",
            "Change every `print(` call to `log(` and output the complete updated file."]
    return [{"name": f"edit{i}", "prompt": f"Here is a Python file:\n\n```python\n{src}\n```\n\n{a} Output only the code."}
            for i, a in enumerate(asks)]


def first_diff(a: str, b: str) -> int | None:
    if a == b:
        return None
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), n)


def mode_agree(c, a, text):
    rng = random.Random(1)
    items = [{"name": p["name"], "prompt": p["prompt"]} for p in [TF_PROMPTS[1]] + TWEET_PROMPTS + KIT_PROMPTS] \
        + edit_prompts()
    asks = ["Summarize the main design decisions in the text above.", "List every function defined above with a "
            "one-line description.", "What problems does the text above describe, and how are they solved?",
            "Explain the exactness guarantees described above.", "Write a short review of the code above."]
    for i in range(a.long):
        n = 8000 + (20000 * i) // max(a.long - 1, 1)
        items.append({"name": f"long{i}-{n}", "prompt": slice_tokens(text, n, rng) + "\n\n" + asks[i % len(asks)]})
    out, same = [], 0
    for it in items:
        msgs = [{"role": "user", "content": it["prompt"]}]
        r = {m: chat(c, msgs, a.tokens, extra={"tf_knobs": {a.knob: m}, "ignore_eos": True}) for m in (0, 1)}
        d = first_diff(r[0]["text"], r[1]["text"])
        same += d is None
        print(f"  {it['name']:14s} prompt {r[0]['prompt_tokens']:>6}: {'identical' if d is None else f'diverge at char {d} (~token {d / 3.5:.0f})'}"
              f"  ttft {r[0]['ttft_s']:.1f}s / {r[1]['ttft_s']:.1f}s", flush=True)
        out.append({"name": it["name"], "prompt_tokens": r[0]["prompt_tokens"], "identical": d is None,
                    "diverge_char": d, "ttft": [r[0]["ttft_s"], r[1]["ttft_s"]],
                    "heads": [r[0]["text"][:200], r[1]["text"][:200]]})
    print(f"  identical: {same}/{len(items)}")
    return out


def mode_needle(c, a, text):
    rng = random.Random(2)
    out = []
    for ctx in [int(x) for x in a.needle_ctx.split(",")]:
        for depth in (0.1, 0.5, 0.9):
            ok = {0: 0, 1: 0}
            for t in range(a.trials if ctx <= 32000 else a.trials_long):
                key = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(6))
                val = str(rng.randrange(100000, 999999))
                hay = slice_tokens(text, ctx, rng)
                cut = int(len(hay) * depth)
                cut = hay.rfind("\n", 0, cut) + 1 or cut
                needle = f"\nThe access code for vault {key} is {val}.\n"
                prompt = hay[:cut] + needle + hay[cut:] + \
                    f"\n\nWhat is the access code for vault {key}? Reply with the number only."
                for m in [int(v) for v in a.values.split(",")]:
                    r = chat(c, [{"role": "user", "content": prompt}], 16, extra={"tf_knobs": {a.knob: m}})
                    hit = val in r["text"]
                    ok[m] += hit
                    out.append({"ctx": ctx, "depth": depth, "trial": t, "fp8": m, "hit": hit, "reply": r["text"][:40],
                                "prompt_tokens": r["prompt_tokens"], "ttft_s": r["ttft_s"]})
            n = a.trials if ctx <= 32000 else a.trials_long
            print(f"  ctx {ctx} depth {depth}: " + ", ".join(f"{a.knob} {m}: {ok[m]}/{n}" for m in
                                                          [int(v) for v in a.values.split(",")]), flush=True)
    return out


def _agree_items(a, text) -> list[dict]:
    rng = random.Random(1)
    items = [{"name": p["name"], "prompt": p["prompt"]} for p in [TF_PROMPTS[1]] + TWEET_PROMPTS + KIT_PROMPTS] \
        + edit_prompts()
    asks = ["Summarize the main design decisions in the text above.", "List every function defined above with a "
            "one-line description.", "What problems does the text above describe, and how are they solved?",
            "Explain the exactness guarantees described above.", "Write a short review of the code above."]
    for i in range(a.long):
        n = 8000 + (20000 * i) // max(a.long - 1, 1)
        items.append({"name": f"long{i}-{n}", "prompt": slice_tokens(text, n, rng) + "\n\n" + asks[i % len(asks)]})
    return items


def mode_replies(c, a, text):
    """Greedy replies of the agree prompt set on this load (no knob); compared with --against (another load's)."""

    ref = {}
    if a.against:
        ref = {r["name"]: r["text"] for r in json.loads(Path(a.against).read_text())["replies"]}
    out, same, n = [], 0, 0
    for it in _agree_items(a, text):
        r = chat(c, [{"role": "user", "content": it["prompt"]}], a.tokens, extra={"ignore_eos": True})
        rec = {"name": it["name"], "prompt_tokens": r["prompt_tokens"], "text": r["text"]}
        if it["name"] in ref:
            d = first_diff(ref[it["name"]], r["text"])
            n += 1
            same += d is None
            rec["diverge_char"] = d
            print(f"  {it['name']:14s} prompt {r['prompt_tokens']:>6}: "
                  f"{'identical' if d is None else f'diverge at char {d} (~token {d / 3.5:.0f})'}", flush=True)
        out.append(rec)
    if a.save:
        Path(a.save).write_text(json.dumps({"args": vars(a), "replies": out}, indent=1))
    if n:
        print(f"  identical to {a.against}: {same}/{n}", flush=True)
    return [{k: v for k, v in r.items() if k != "text"} for r in out]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--modes", default="agree,needle")
    p.add_argument("--corpus", nargs="+", default=["docs", "vendor/TensorFold/src"])
    p.add_argument("--long", type=int, default=20)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--needle-ctx", default="28000,112000")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--trials-long", type=int, default=10)
    p.add_argument("--extra", default="{}")
    p.add_argument("--knob", default="fp8_prefill", help="the tf_knobs key switched 0 vs 1 (control: fast_prefill)")
    p.add_argument("--values", default="0,1", help="needle: the knob values to run (e.g. 1)")
    p.add_argument("--save", help="replies: write this load's replies here")
    p.add_argument("--against", help="replies: compare with replies saved on another load")
    p.add_argument("--out")
    a = p.parse_args()
    c = Client(a.base, a.model, extra=json.loads(a.extra))
    text = corpus(a.corpus)
    print(f"corpus: {len(text)} characters (~{len(text) / CHARS_PER_TOKEN:.0f} tokens)", flush=True)
    res = {"args": vars(a), "t": time.time()}
    for m in a.modes.split(","):
        print(f"[{m}]", flush=True)
        res[m] = {"agree": mode_agree, "needle": mode_needle, "replies": mode_replies}[m](c, a, text)
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
