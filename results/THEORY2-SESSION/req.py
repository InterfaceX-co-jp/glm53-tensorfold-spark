#!/usr/bin/env python3
"""W16 request driver for the nsys load (standard library only; prod API on 127.0.0.1:8000). Non-streaming, so each
reply carries the engine's per-request stats (``tensorfold``: rounds, round_kinds incl. resident / capture / graph /
eager counts, decode_s). One JSON line a request is appended to OUT.

  req.py tfcode OUT N [TOKENS]   glmbench's tf "code" cell greedy (raw completion, 64 tokens, ignore_eos), N times
                                 (THEORY-2 §2: 0450 resident lost 18% exactly here)
  req.py conc OUT K TOKENS       K concurrent prose requests (greedy, thinking off): one 4-stream rep
"""
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bench"))
from glmbench import TF_PROMPTS  # noqa: E402

B = "http://127.0.0.1:8000"
M = "GLM-5.3-Flash-EXL3"
TOPICS = ["the history of the printing press", "how a heat pump works", "the life cycle of a star",
          "why bridges use expansion joints"]


def post(path, body):
    r = urllib.request.Request(B + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(r, timeout=1800))
    tf = d.get("tensorfold", {}) or {}
    n = d["usage"]["completion_tokens"]
    return dict(t0=round(t0, 3), wall=round(time.time() - t0, 3), tokens=n, decode_s=tf.get("decode_s"),
                decode_tps=round((n - 1) / tf["decode_s"], 2) if tf.get("decode_s") else None,
                tpr=tf.get("tokens_per_round"), rounds=tf.get("rounds"), kinds=tf.get("round_kinds"), tf=tf)


def tfcode(tokens):
    item = TF_PROMPTS[0]
    assert item["name"] == "code" and item["kind"] == "completion"
    return post("/v1/completions", {"model": M, "prompt": item["prompt"], "max_tokens": tokens, "temperature": 0,
                                    "top_p": 1, "ignore_eos": True})


def prose(i, tokens):
    return post("/v1/chat/completions", {"model": M, "max_tokens": tokens, "temperature": 0, "top_p": 1,
                                         "chat_template_kwargs": {"enable_thinking": False},
                                         "messages": [{"role": "user", "content": "Write a detailed, plain-prose "
                                                       f"explanation of {TOPICS[i % len(TOPICS)]}."}]})


def main():
    mode, out = sys.argv[1], sys.argv[2]
    rows = []
    if mode == "tfcode":
        n = int(sys.argv[3])
        tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 64
        for i in range(n):
            r = tfcode(tokens)
            r.update(name="tf-code-greedy", i=i)
            rows.append(r)
    elif mode == "conc":
        k, tokens = int(sys.argv[3]), int(sys.argv[4])
        res = [None] * k

        def one(i):
            res[i] = prose(i, tokens)
            res[i].update(name=f"conc{k}-{i}")

        th = [threading.Thread(target=one, args=(i,)) for i in range(k)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        rows = res
    else:
        sys.exit(__doc__)
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
            print(time.strftime("%T"), r["name"], "tokens", r["tokens"], "decode", r["decode_tps"], "tok/s, tpr",
                  r["tpr"], "rounds", r["rounds"], "kinds", json.dumps(r["kinds"]), flush=True)


if __name__ == "__main__":
    main()
