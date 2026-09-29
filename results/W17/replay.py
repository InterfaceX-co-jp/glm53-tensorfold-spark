#!/usr/bin/env python3
"""W15: warm replay (patches/0540 §1) and grid-aligned cold prefill against a live server, RigMark's way.

    replay.py tokens OUT.json [DEPTHS] [PAIRS]   token-ID /v1/completions (max_tokens 8, greedy, ignore_eos, streamed):
        for each depth a fresh nonce prompt of exactly DEPTH ids (cold), then the same ids at once (replay), PAIRS
        times. DEPTHS default 8192,32768,65536,32700 (the last off the 64-grid), PAIRS 3.
    replay.py chat OUT.json                        a ~12k-token chat prompt, then the identical messages again,
        greedy and sampled (temperature 1, seed 7): replies identical, cached within 64 of the prompt.

Each row: client TTFT (first streamed text), the server's usage (prompt / cached tokens) and the 0300 request-log
line of the request (cached, cache_src, pieces, prefill_s, queue_s, first_s), matched by request order.
Server BASE (default http://127.0.0.1:8000); request log LOG (default rank 0's /sessions/requests.jsonl on the head node).
Stdlib only."""
import hashlib, json, os, statistics, sys, time, urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8000")
LOG = os.environ.get("LOG", "$HOME/.cache/glm53-tf/sessions/requests.jsonl")
MODEL = "GLM-5.3-Flash-EXL3"
FILLER = ("The quick brown fox jumps over the lazy dog while the committee reviews quarterly logistics, "
          "inventory forecasts, and the maintenance schedule for the northern warehouse. ")


def post_json(path, body, timeout=60):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


def log_lines():
    try:
        with open(LOG, "rb") as f:
            return f.read().splitlines()
    except OSError:
        return []


def stream(path, body):
    """POST a streamed request; returns ttft (first text), wall, text, usage and the request-log line it produced."""
    before = len(log_lines())
    body = dict(body, stream=True, stream_options={"include_usage": True})
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; text = []; usage = None
    with urllib.request.urlopen(r, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ch = json.loads(line[5:])
            if ch.get("usage"):
                usage = ch["usage"]
            for c in ch.get("choices", []):
                d = c.get("delta") or {}
                piece = c.get("text") or d.get("content") or d.get("reasoning") or d.get("reasoning_content") or ""
                if piece:
                    first = first or time.perf_counter()
                    text.append(piece)
    wall = time.perf_counter() - t0
    rec = None
    for _ in range(50):                      # the log line is written when the request ends
        lines = log_lines()
        if len(lines) > before:
            rec = json.loads(lines[before]); break
        time.sleep(0.1)
    txt = "".join(text)
    return dict(ttft=None if first is None else round(first - t0, 3), wall=round(wall, 3),
                sha=hashlib.sha256(txt.encode()).hexdigest()[:16], text=txt[:200],
                prompt_tokens=(usage or {}).get("prompt_tokens"),
                cached_tokens=((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
                log={k: (rec or {}).get(k) for k in ("n", "prompt", "cached", "cache_src", "pieces", "prefill_s", "prefill_tps",
                                                      "queue_s", "first_s", "slot", "finish")})


def ids_for(depth, tag):
    text = f"[w17-replay-{tag}-{time.time_ns()}] " + FILLER * (depth // 24 + 16)
    ids = post_json("/tokenize", {"model": MODEL, "prompt": text})["tokens"]
    assert len(ids) >= depth, (len(ids), depth)
    return ids[:depth]


def tokens(out, depths, pairs):
    rows = []
    for depth in depths:
        for p in range(pairs):
            ids = ids_for(depth, f"{depth}-{p}")
            body = {"model": MODEL, "prompt": ids, "max_tokens": 8, "temperature": 0, "top_p": 1, "ignore_eos": True}
            cold = stream("/v1/completions", body)
            warm = stream("/v1/completions", body)
            row = dict(depth=depth, pair=p, cold=cold, replay=warm, same_tokens=cold["sha"] == warm["sha"],
                       cold_tps=round(depth / cold["ttft"], 1) if cold["ttft"] else None,
                       replay_tps=round(depth / warm["ttft"], 1) if warm["ttft"] else None)
            rows.append(row)
            print(f"{time.strftime('%T')} depth {depth} pair {p}: cold ttft {cold['ttft']} s ({row['cold_tps']} tok/s, "
                  f"pieces {cold['log']['pieces']}, cached {cold['log']['cached']}) | replay ttft {warm['ttft']} s "
                  f"({row['replay_tps']} tok/s, cached {warm['log']['cached']} {warm['log']['cache_src']}, pieces "
                  f"{warm['log']['pieces']}, first_s {warm['log']['first_s']}) | same 8 tokens {row['same_tokens']}", flush=True)
    summ = {}
    for depth in depths:
        rs = [r for r in rows if r["depth"] == depth]
        summ[depth] = dict(cold_ttft_med=statistics.median(r["cold"]["ttft"] for r in rs),
                           cold_tps_med=statistics.median(r["cold_tps"] for r in rs),
                           replay_ttft_med=statistics.median(r["replay"]["ttft"] for r in rs),
                           replay_ttft_max=max(r["replay"]["ttft"] for r in rs),
                           replay_tps_med=statistics.median(r["replay_tps"] for r in rs),
                           replay_cached=sorted({r["replay"]["log"]["cached"] for r in rs}, key=str),
                           want_cached=(depth - 1) // 64 * 64,
                           replay_pieces=sorted({r["replay"]["log"]["pieces"] for r in rs}, key=str),
                           cold_pieces=sorted({r["cold"]["log"]["pieces"] for r in rs}, key=str),
                           same_tokens=all(r["same_tokens"] for r in rs))
        s = summ[depth]
        print(f"SUMMARY depth {depth}: cold {s['cold_tps_med']} tok/s (ttft {s['cold_ttft_med']} s, pieces {s['cold_pieces']}) | "
              f"replay ttft med {s['replay_ttft_med']} max {s['replay_ttft_max']} s ({s['replay_tps_med']} tok/s), cached "
              f"{s['replay_cached']} (n-64 rule: {s['want_cached']}), pieces {s['replay_pieces']} | same {s['same_tokens']}", flush=True)
    json.dump(dict(rows=rows, summary=summ), open(out, "w"), indent=1)


def chat(out):
    doc = f"[w17-regen-{time.time_ns()}] " + FILLER * 500
    msgs = [{"role": "user", "content": doc + "\n\nSummarize what the committee reviews in two sentences."}]
    rows = []
    for name, extra in (("greedy", {"temperature": 0, "top_p": 1}), ("sampled", {"temperature": 1.0, "top_p": 0.95, "seed": 7})):
        body = {"model": MODEL, "messages": msgs, "max_tokens": 128, "chat_template_kwargs": {"enable_thinking": False}, **extra}
        a = stream("/v1/chat/completions", body)
        b = stream("/v1/chat/completions", body)
        row = dict(name=name, first=a, second=b, identical=a["sha"] == b["sha"],
                   cached_within_64=b["log"]["cached"] is not None and a["prompt_tokens"] - b["log"]["cached"] <= 64)
        rows.append(row)
        print(f"{time.strftime('%T')} regenerate {name}: prompt {a['prompt_tokens']}, 2nd cached {b['log']['cached']} "
              f"({b['log']['cache_src']}), ttft {a['ttft']} -> {b['ttft']} s, identical {row['identical']}, "
              f"within 64 {row['cached_within_64']}", flush=True)
    json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    mode, outf = sys.argv[1], sys.argv[2]
    if mode == "tokens":
        d = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "8192,32768,65536,32700").split(",")]
        tokens(outf, d, int(sys.argv[4]) if len(sys.argv) > 4 else 3)
    else:
        chat(outf)
