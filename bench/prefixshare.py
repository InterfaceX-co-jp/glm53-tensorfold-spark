#!/usr/bin/env python3
"""Shared-prefix reuse across sessions (patches/0310, GLM53_TF_PREFIX_SHARE): new sessions over one agent-like system
prompt, against a running server. Standard library only.

  sequential  --sessions new sessions, one after the other, over a fresh --system-token system prompt (unique to the
              run, so nothing is stored yet), each with its own --user-token task: cached tokens, TTFT, prefill.
              Today session 2 prefills the whole system prompt (< 16,384 tokens) and 3+ resume at the fork mark;
              with the switch, 2+ resume at the system prompt's end.
  burst       --burst new sessions sent at once over another fresh system prompt: today all prefill it side by side;
              with the switch the first prefills it and the others wait for its mark and resume there.
  exact       every reply above is compared with the same request sent with "draft": false (a fresh prefill and
              serial decoding): identical text (greedy) = resumed == fresh end to end.

  python3 bench/prefixshare.py --base http://127.0.0.1:8000 --model GLM-5.3-Flash-EXL3 --system 12000 --out r.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from glmbench import Client  # noqa: E402
from multiturn import _parallel, chat, doc  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": f"tool_{i}", "description": f"Tool number {i}: does thing {i}.",
                                           "parameters": {"type": "object", "properties": {
                                               "path": {"type": "string", "description": "a file path"},
                                               "count": {"type": "integer"}}, "required": ["path"]}}}
         for i in range(12)]


def _messages(system: str, tag: str, user: int) -> list:
    return [{"role": "system", "content": system},
            {"role": "user", "content": doc(tag, user) + f"\n\nTask {tag}: in one sentence, what is this?"}]


def _row(name: str, r: dict) -> dict:
    tf = r.get("tensorfold") or {}
    row = {"name": name, "prompt_tokens": r["prompt_tokens"], "cached_tokens": r["cached_tokens"],
           "ttft_s": r["ttft_s"], "prefill_s": tf.get("prefill_s"), "restored": tf.get("restored"),
           "restored_disk": tf.get("restored_disk"), "prefix_wait": tf.get("prefix_wait"), "marks": tf.get("marks"),
           "queued_s": tf.get("queued_s"), "text": r["text"]}
    print(f"  {name:>12}: prompt {row['prompt_tokens']:>6} cached {row['cached_tokens']!s:>6} ttft "
          f"{(row['ttft_s'] or 0):6.2f}s prefill {row['prefill_s']} waits {row['prefix_wait']} marks {row['marks']}",
          flush=True)
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--system", type=int, default=12000, help="system prompt tokens (~)")
    p.add_argument("--user", type=int, default=1500, help="each session's own task tokens (~)")
    p.add_argument("--sessions", type=int, default=3)
    p.add_argument("--burst", type=int, default=4)
    p.add_argument("--tokens", type=int, default=48)
    p.add_argument("--modes", default="sequential,burst,exact")
    p.add_argument("--extra", default="{}")
    p.add_argument("--out")
    a = p.parse_args()
    c = Client(a.base, a.model, extra=json.loads(a.extra))
    modes = a.modes.split(",")
    run = time.time_ns() % 10 ** 9
    tools = {"tools": TOOLS}
    res: dict = {"args": vars(a)}
    reqs: list[tuple[str, list]] = []
    if "sequential" in modes:
        print("[sequential]", flush=True)
        system = f"You are agent {run}. Rules and reference follow.\n" + doc(f"SQ{run}", a.system)
        rows = []
        for i in range(a.sessions):
            msgs = _messages(system, f"Q{run}-{i}", a.user)
            rows.append(_row(f"session {i + 1}", chat(c, msgs, a.tokens, extra=tools)))
            reqs.append((f"seq{i}", msgs))
        res["sequential"] = rows
    if "burst" in modes:
        print("[burst]", flush=True)
        system = f"You are agent {run}b. Rules and reference follow.\n" + doc(f"BU{run}", a.system)
        all_msgs = [_messages(system, f"B{run}-{i}", a.user) for i in range(a.burst)]
        t0 = time.perf_counter()
        got = _parallel([lambda m=m: chat(c, m, a.tokens, extra=tools) for m in all_msgs])
        wall = time.perf_counter() - t0
        rows = [_row(f"burst {i + 1}", r) for i, r in enumerate(got)]
        print(f"  burst wall {wall:.1f}s, last first token {max(r['ttft_s'] or 0 for r in got):.1f}s", flush=True)
        res["burst"] = {"rows": rows, "wall_s": wall}
        reqs += [(f"burst{i}", m) for i, m in enumerate(all_msgs)]
    if "exact" in modes:
        print("[exact] each request again with draft=false (fresh prefill, serial decoding)", flush=True)
        texts = {r["name"]: r["text"] for r in res.get("sequential", [])}
        texts.update({r["name"]: r["text"] for r in (res.get("burst") or {}).get("rows", [])})
        names = [f"session {i + 1}" for i in range(len(res.get("sequential", [])))] + \
                [f"burst {i + 1}" for i in range(len((res.get("burst") or {}).get("rows", [])))]
        same = []
        for (_, msgs), name in zip(reqs, names):
            ref = chat(c, msgs, a.tokens, extra={**tools, "draft": False})
            same.append(ref["text"] == texts[name])
        print(f"  resumed == fresh: {sum(same)}/{len(same)} {same}", flush=True)
        res["exact"] = same
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
