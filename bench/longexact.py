#!/usr/bin/env python3
"""N1 (docs/SFXNZ-AUDIT.md): real-model exactness past the indexer's top-k (2,048), where a rejected draft that
corrupted a DSA indexer pool key would change which rows are selected. Standard library only.

For each prompt length (default 8k / 16k / 32k tokens of repo text, a different slice each, no unique tag so every
load sends the same bytes):
  drafted == serial  greedy (drafted first, cold; then serial, "draft": false) and sampled (T 1, top-k 20, top-p 0.95,
                     seed 1234; serial first, cold; then drafted): --tokens tokens, thinking off, ignore_eos.
                     The second request of a pair resumes the first one's prompt snapshot (resumed == fresh holds,
                     and the two orders put the cold prefill on each side).
  batched == alone   the greedy and sampled requests of every length sent together (up to 4 at once; with 3 lengths
                     the 6 go as 4 + 2) against their drafted replies alone.

  python3 bench/longexact.py --base http://127.0.0.1:8000 --model GLM-5.3-Flash-EXL3 --corpus results/W12/n1-corpus.txt \
      --out results/W12/n1-C.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
import urllib.request

ASK = ("\n\n---\nAbove is an excerpt of a technical document. Summarise it section by section in detail, naming the "
       "patches, knobs and measured numbers it gives.")


def req(a, prompt: str, sampled: bool, draft: bool) -> dict:
    body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": a.tokens,
            "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    body.update({"temperature": 1.0, "top_k": 20, "top_p": 0.95, "seed": 1234} if sampled else
                {"temperature": 0, "top_p": 1})
    if not draft:
        body["draft"] = False
    r = urllib.request.Request(a.base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(r, timeout=3600))
    tf = d.get("tensorfold", {}) or {}
    text = d["choices"][0]["message"].get("content") or ""
    n = d["usage"]["completion_tokens"]
    return {"sha": hashlib.sha256(text.encode()).hexdigest()[:16], "tokens": n,
            "prompt_tokens": d["usage"]["prompt_tokens"], "cached": tf.get("cached"), "slot": tf.get("slot"),
            "wall": round(time.time() - t0, 2),
            "decode_tps": round((n - 1) / tf["decode_s"], 1) if tf.get("decode_s") else None,
            "tpr": tf.get("tokens_per_round"), "head": text[:120]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    p.add_argument("--corpus", required=True)
    p.add_argument("--lengths", default="8000,16000,32000")
    p.add_argument("--chars-per-token", type=float, default=3.4)
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--no-batch", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    corpus = open(a.corpus).read()
    lengths = [int(x) for x in a.lengths.split(",")]
    prompts, off = {}, 0
    for L in lengths:
        n = int(L * a.chars_per_token)
        if off + n > len(corpus):
            off = 0
        prompts[L] = corpus[off:off + n] + ASK
        off += n + 1000
    out = {"args": vars(a), "alone": {}, "pairs": {}, "batched": {}}
    ok_pairs = []
    for L in lengths:
        g_d = req(a, prompts[L], False, True)
        g_s = req(a, prompts[L], False, False)
        s_s = req(a, prompts[L], True, False)
        s_d = req(a, prompts[L], True, True)
        out["alone"].update({f"{L}-g-drafted": g_d, f"{L}-g-serial": g_s, f"{L}-s-serial": s_s, f"{L}-s-drafted": s_d})
        for kind, d, s in (("greedy", g_d, g_s), ("sampled", s_d, s_s)):
            same = d["sha"] == s["sha"]
            ok_pairs.append(same)
            out["pairs"][f"{L}-{kind}"] = same
            print(f"  {L:>6} ({d['prompt_tokens']} prompt tokens) {kind:7s} drafted {d['sha']} {d['decode_tps']} tok/s "
                  f"(tpr {d['tpr']}) serial {s['sha']} {s['decode_tps']} tok/s  identical={same}", flush=True)
    print(f"N1 drafted == serial: {sum(ok_pairs)}/{len(ok_pairs)}", flush=True)
    if not a.no_batch:
        jobs = [(L, s) for L in lengths for s in (False, True)]
        same_all = []
        for i in range(0, len(jobs), 4):
            group, got = jobs[i:i + 4], {}

            def go(L, s):
                got[(L, s)] = req(a, prompts[L], s, True)
            ts = [threading.Thread(target=go, args=j) for j in group]
            [t.start() for t in ts]
            [t.join() for t in ts]
            for (L, s), r in got.items():
                k = f"{L}-{'s' if s else 'g'}"
                same = r["sha"] == out["alone"][f"{k}-drafted"]["sha"]
                same_all.append(same)
                out["batched"][k] = {"same": same, **r}
                print(f"  batched {k:10s} slot {r['slot']} {r['sha']} identical={same}", flush=True)
        print(f"N1 batched == alone: {sum(same_all)}/{len(same_all)}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
