#!/usr/bin/env python3
"""Multi-turn, multi-session and concurrent-stream measurements (standard library only).

  sessions    3 conversations A, B, C of ~--doc tokens each (a shared ~--system-token system prompt, then each its
              own document), visited A, B, A, C, B, A: TTFT, prompt tokens and usage.prompt_tokens_details
              .cached_tokens per turn (a revisit appends the previous reply and a new question).
  followup    one --doc-token prompt cold, then the same conversation + its reply + ~--append new tokens: TTFT of each.
  concurrent  1 / 2 / 4 (--streams) simultaneous streams of the tf/kit prompts: aggregate and per-stream tok/s.
  batchexact  4 greedy requests alone, then concurrently: identical replies?
  stall       --streams decoding streams (long replies); while they decode, one --doc-token prompt is sent: the
              longest gap between the decoding streams' chunks during that prefill, and its TTFT.
  stress      memory worst case (docs/GPU-PLAN-4x256k.md): 4 conversations grown together, --stress-step tokens a turn,
              to ~--stress-target prompt tokens (the 4th to target - --stress-final); then 3 of them decode long
              replies while the 4th sends a --stress-final-token turn. MemAvailable / MemFree of --mem-hosts sampled
              every 2 s (--mem-log); the minimum per host, overall and during the final phase, against --mem-min.

  python3 bench/multiturn.py --base http://127.0.0.1:8080 --model GLM-5.3-Flash-Uncensored --modes sessions,followup
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from glmbench import FILLER, KIT_PROMPTS, TF_PROMPTS, Client  # noqa: E402

WORDS = ("amber basalt cedar delta ember fjord granite harbor iris juniper kelp lagoon meadow nectar orchid "
         "pepper quartz raven saffron tundra umber violet willow xenon yarrow zephyr").split()


def doc(tag: str, tokens: int) -> str:
    """~tokens of text unique to tag (a numbered log, so no two documents share a prefix past the tag)."""

    lines, n, i = [f"Document {tag}."], 0, 0
    while n < tokens:
        w = [WORDS[(hash((tag, i, j)) & 0xffff) % len(WORDS)] for j in range(6)]
        lines.append(f"{tag}-{i:05d}: " + " ".join(w) + ".")
        n += 16
        i += 1
    return "\n".join(lines)


def chat(c: Client, messages: list, tokens: int = 64, extra: dict | None = None) -> dict:
    body = {"model": c.model, "messages": messages, "max_tokens": tokens, "temperature": 0, "top_p": 1,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    body.update(c.extra)
    body.update(extra or {})
    t0 = time.perf_counter()
    first = last = None
    usage, text, stamps, tf = None, [], [], None
    with c.post("/v1/chat/completions", body) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("tensorfold"):
                tf = chunk["tensorfold"]
            for ch in chunk.get("choices", []):
                d = ch.get("delta") or {}
                piece = d.get("content") or d.get("reasoning_content") or ""
                if piece:
                    now = time.perf_counter()
                    first = now if first is None else first
                    last = now
                    stamps.append(now)
                    text.append(piece)
    n = int((usage or {}).get("completion_tokens") or 0)
    details = (usage or {}).get("prompt_tokens_details") or {}
    return {"ttft_s": None if first is None else first - t0, "start": t0, "end": last, "stamps": stamps,
            "prompt_tokens": (usage or {}).get("prompt_tokens"), "cached_tokens": details.get("cached_tokens"),
            "tokens": n, "decode_tps": (n - 1) / (last - first) if n > 1 and last > first else None,
            "text": "".join(text), "tensorfold": tf}


def mode_sessions(c, a):
    system = {"role": "system", "content": "You are a careful analyst. Shared instructions follow.\n" + doc("SYS", a.system)}
    convs = {k: [system, {"role": "user", "content": doc(f"S{k}{time.time_ns() % 100000}", a.doc) +
                          f"\n\nSession {k}: in one sentence, what kind of document is this?"}] for k in "ABC"}
    out = []
    for turn, k in enumerate("ABACBA"):
        msgs = convs[k]
        r = chat(c, msgs, a.tokens)
        msgs.append({"role": "assistant", "content": r["text"]})
        msgs.append({"role": "user", "content": f"Turn {turn}: name one word that appears in line {turn * 7 + 3}."})
        print(f"  turn {turn} session {k}: prompt {r['prompt_tokens']:>6} cached {r['cached_tokens']}  "
              f"ttft {r['ttft_s']:.2f}s", flush=True)
        out.append({"turn": turn, "session": k, **{x: r[x] for x in ("ttft_s", "prompt_tokens", "cached_tokens",
                                                                      "decode_tps", "tensorfold")}})
    return out


def mode_followup(c, a):
    msgs = [{"role": "user", "content": doc(f"F{time.time_ns() % 100000}", a.doc) +
             "\n\nSummarize this document in two sentences."}]
    r1 = chat(c, msgs, a.tokens)
    msgs += [{"role": "assistant", "content": r1["text"]},
             {"role": "user", "content": doc("APPEND", a.append) + "\n\nNow compare the appended part with the first."}]
    r2 = chat(c, msgs, a.tokens)
    for name, r in (("cold", r1), ("follow-up", r2)):
        print(f"  {name:9s}: prompt {r['prompt_tokens']:>6} cached {r['cached_tokens']}  ttft {r['ttft_s']:.2f}s  "
              f"({(r['prompt_tokens'] or 0) / r['ttft_s']:.0f} tok/s over the whole prompt)", flush=True)
    return [{"name": n, **{x: r[x] for x in ("ttft_s", "prompt_tokens", "cached_tokens", "decode_tps", "tensorfold")}}
            for n, r in (("cold", r1), ("followup", r2))]


def _parallel(fns):
    res = [None] * len(fns)

    def run(i):
        res[i] = fns[i]()

    ts = [threading.Thread(target=run, args=(i,)) for i in range(len(fns))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return res


def _ms_round(d: dict) -> str:
    """A batched request's wall ms a round (verify + its drafting) and the seconds it waited on others' prefill
    pieces (patches/0200's ``round_kinds``)."""

    k, n = d.get("round_kinds") or {}, d.get("rounds") or 0
    if not k or not n:
        return "-"
    return f"{(k.get('verify_ms', 0) + k.get('draft_ms', 0)) / n:.0f}+{k.get('piece_ms', 0) / 1e3:.1f}s"


def mode_concurrent(c, a):
    prompts = [p for p in TF_PROMPTS + KIT_PROMPTS]
    out = []
    for n in [int(x) for x in a.streams.split(",")]:
        for rep in range(a.reps):
            fns = [lambda i=i: chat(c, [{"role": "user", "content": prompts[(i + rep) % len(prompts)]["prompt"]}],
                                    a.long_tokens, extra={"ignore_eos": True}) for i in range(n)]
            t0 = time.perf_counter()
            rs = _parallel(fns)
            first = min(r["start"] + r["ttft_s"] for r in rs)
            last = max(r["end"] for r in rs)
            toks = sum(r["tokens"] for r in rs)
            agg = toks / (last - first)
            per = [r["decode_tps"] for r in rs]
            # the engine's per-request stats (batch mode): rounds, tokens a round, round kinds (patches/0200)
            tf = [r.get("tensorfold") or {} for r in rs]
            kinds = {}
            for d in tf:
                for k, v in (d.get("round_kinds") or {}).items():
                    if not k.endswith("_ms"):
                        kinds[k] = kinds.get(k, 0) + v
            ttfts = ", ".join(f"{r['ttft_s']:.2f}" for r in rs)
            print(f"  {n} streams rep {rep}: aggregate {agg:.1f} tok/s  per stream "
                  f"{', '.join(f'{p:.1f}' for p in per)}  ttft {ttfts}"
                  f"  tok/round {', '.join(str(d.get('tokens_per_round')) for d in tf)}"
                  f"  ms/round {', '.join(_ms_round(d) for d in tf)}"
                  f"{'  rounds ' + json.dumps(kinds) if kinds else ''}  (wall {time.perf_counter() - t0:.1f}s)",
                  flush=True)
            out.append({"streams": n, "rep": rep, "aggregate_tps": agg, "per_stream_tps": per,
                        "tokens": [r["tokens"] for r in rs], "ttft_s": [r["ttft_s"] for r in rs],
                        "tokens_per_round": [d.get("tokens_per_round") for d in tf],
                        "drafters": [d.get("drafters") for d in tf], "round_kinds": kinds,
                        "stats": [{x: d.get(x) for x in ("rounds", "round_kinds", "keeps", "slot", "prefill_s",
                                                         "queued_s")} for d in tf]})
    return out


def mode_stall(c, a):
    n = int(a.streams.split(",")[-1]) - 1 or 1
    box = {}

    def decoder(i):
        return chat(c, [{"role": "user", "content": KIT_PROMPTS[2]["prompt"]}], a.long_tokens,
                    extra={"ignore_eos": True})

    def prefill():
        time.sleep(a.delay)
        box["t"] = time.perf_counter()
        return chat(c, [{"role": "user", "content": doc(f"P{time.time_ns() % 100000}", a.doc) + "\n\nOne sentence."}],
                    16)

    rs = _parallel([lambda i=i: decoder(i) for i in range(n)] + [prefill])
    dec, pf = rs[:n], rs[n]
    t0, t1 = box["t"], box["t"] + pf["ttft_s"]
    gaps, before = [], []
    for r in dec:
        st = r["stamps"]
        for x, y in zip(st, st[1:]):
            (gaps if y > t0 and x < t1 else before).append(y - x)
    res = {"decoders": n, "prefill_tokens": pf["prompt_tokens"], "prefill_ttft_s": pf["ttft_s"],
           "max_gap_during_prefill_s": max(gaps) if gaps else None,
           "sum_gaps_during_prefill_s": sum(gaps) if gaps else None,
           "median_gap_outside_s": statistics.median(before) if before else None,
           "decode_tps": [r["decode_tps"] for r in dec]}
    print(f"  {n} decoding streams + one {pf['prompt_tokens']}-token prefill (ttft {pf['ttft_s']:.1f}s): longest "
          f"decode gap during the prefill {res['max_gap_during_prefill_s']}, median gap otherwise "
          f"{res['median_gap_outside_s']}", flush=True)
    return res


def mode_batchexact(c, a):
    """The same greedy requests alone, then all at once: identical replies?"""

    prompts = [p["prompt"] for p in TF_PROMPTS + KIT_PROMPTS][:4]
    alone = [chat(c, [{"role": "user", "content": p}], 128, extra={"ignore_eos": True})["text"] for p in prompts]
    together = _parallel([lambda p=p: chat(c, [{"role": "user", "content": p}], 128, extra={"ignore_eos": True})
                          for p in prompts])
    same = [x == y["text"] for x, y in zip(alone, together)]
    print(f"  batched == alone: {sum(same)}/{len(same)} {same}", flush=True)
    return {"same": same}


class MemSampler:
    """MemAvailable / MemFree (GiB) of each host every ``every`` s: "local" reads /proc/meminfo, anything else is an
    ssh target (BatchMode). ``mark(name)`` starts a phase; ``mins()`` -> {phase: {host: min MemAvailable}}."""

    def __init__(self, hosts: list[str], log: str | None, every: float = 2.0) -> None:
        self.hosts, self.every, self.log = hosts, every, open(log, "a") if log else None
        self.phase, self.lows, self.stop = "load", {}, threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    @staticmethod
    def _read(host: str) -> dict:
        import subprocess

        if host == "local":
            text = Path("/proc/meminfo").read_text()
        else:
            text = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, "cat /proc/meminfo"],
                                  capture_output=True, text=True, timeout=10).stdout
        kv = {ln.split(":")[0]: int(ln.split()[1]) for ln in text.splitlines() if ln.split()[1:2]}
        return {k: kv.get(k, 0) / 2**20 for k in ("MemAvailable", "MemFree", "Cached")}

    def _run(self) -> None:
        while not self.stop.is_set():
            row = []
            for h in self.hosts:
                try:
                    m = self._read(h)
                except Exception as ex:                                 # noqa: BLE001 -- keep sampling
                    row.append(f"{h} ? ({ex.__class__.__name__})")
                    continue
                for ph in (self.phase, "all"):
                    d = self.lows.setdefault(ph, {})
                    d[h] = min(d.get(h, 1e9), m["MemAvailable"])
                row.append(f"{h} avail {m['MemAvailable']:.1f} free {m['MemFree']:.1f} cached {m['Cached']:.1f}")
            if self.log:
                self.log.write(f"{time.strftime('%H:%M:%S')} [{self.phase}] " + " | ".join(row) + "\n")
                self.log.flush()
            self.stop.wait(self.every)

    def mark(self, phase: str) -> None:
        self.phase = phase

    def mins(self) -> dict:
        return {ph: {h: round(v, 2) for h, v in d.items()} for ph, d in self.lows.items()}

    def close(self) -> dict:
        self.stop.set()
        self.t.join(timeout=15)
        return self.mins()


DOC_RATIO = 1.4


def mode_stress(c, a):
    """4 conversations to ~--stress-target tokens each (store saves, slot resumes), then a --stress-final-token turn on
    the 4th while the other 3 decode: the memory worst case of docs/MEMORY-4x256k.md."""

    mem = MemSampler([h for h in a.mem_hosts.split(",") if h], a.mem_log)
    tag = time.time_ns() % 100000
    # doc() sizes by ~16 tokens a line; GLM's tokenizer makes ~22 of each line (60000 -> 82.6k prompt tokens), so the
    # stress turns ask for n / DOC_RATIO to stay under the context
    dn = lambda n: int(n / DOC_RATIO)  # noqa: E731
    convs = {k: [{"role": "user", "content": doc(f"ST{k}-{tag}", dn(a.stress_step)) + f"\n\nThread {k}: one sentence."}]
             for k in range(4)}
    goal = {k: a.stress_target - (a.stress_final + 2000 if k == 3 else 0) for k in range(4)}
    size = {k: 0 for k in range(4)}
    out = {"turns": []}

    def turn(k: int, rnd: int) -> dict:
        r = chat(c, convs[k], a.tokens)
        convs[k].append({"role": "assistant", "content": r["text"]})
        size[k] = r["prompt_tokens"] or 0
        left = goal[k] - size[k] - 200
        if left > 1000:
            convs[k].append({"role": "user", "content": doc(f"ST{k}-{tag}-{rnd}", dn(min(a.stress_step, left))) +
                             f"\n\nTurn {rnd}: what changed?"})
        return r

    t0 = time.perf_counter()
    mem.mark("fill")
    rnd = 0
    while any(goal[k] - size[k] - 200 > 1000 for k in range(4)):
        live = [k for k in range(4) if goal[k] - size[k] - 200 > 1000]
        rs = _parallel([lambda k=k: turn(k, rnd) for k in live])
        for k, r in zip(live, rs):
            print(f"  round {rnd} thread {k}: prompt {r['prompt_tokens']:>7} cached {r['cached_tokens']}  "
                  f"ttft {r['ttft_s']:.1f}s  ({((r['prompt_tokens'] or 0) - (r['cached_tokens'] or 0)) / r['ttft_s']:.0f}"
                  f" new tok/s)  mem low {mem.mins().get('fill')}", flush=True)
            out["turns"].append({"round": rnd, "thread": k, **{x: r[x] for x in ("ttft_s", "prompt_tokens",
                                                                                "cached_tokens", "tensorfold")}})
        rnd += 1
    fill_s = time.perf_counter() - t0
    # final phase: threads 0-2 decode long replies; thread 3 sends --stress-final new tokens
    for k in range(3):
        convs[k].append({"role": "user", "content": "Continue: list every line tag you remember, one a line."})
    convs[3].append({"role": "user", "content": doc(f"ST3-{tag}-final", dn(a.stress_final)) + "\n\nSummarize all."})
    mem.mark("final")
    box = {}

    def final():
        time.sleep(a.delay)
        box["t"] = time.perf_counter()
        return chat(c, convs[3], 16)

    rs = _parallel([lambda k=k: chat(c, convs[k], a.long_tokens, extra={"ignore_eos": True}) for k in range(3)]
                   + [final])
    dec, pf = rs[:3], rs[3]
    t_0, t_1 = box["t"], box["t"] + (pf["ttft_s"] or 0)
    gaps = [y - x for r in dec for x, y in zip(r["stamps"], r["stamps"][1:]) if y > t_0 and x < t_1]
    time.sleep(2 * mem.every)
    lows = mem.close()
    ok = all(v >= a.mem_min for v in lows.get("all", {}).values())
    out.update(fill_s=fill_s, sizes=[r["prompt_tokens"] for r in dec] + [pf["prompt_tokens"]],
               final={"prompt_tokens": pf["prompt_tokens"], "cached_tokens": pf["cached_tokens"],
                      "ttft_s": pf["ttft_s"], "max_gap_s": max(gaps) if gaps else None,
                      "decode_tps": [r["decode_tps"] for r in dec]},
               mem_min_gib=lows, mem_ok=ok)
    print(f"  filled in {fill_s:.0f}s; final: thread 3 {pf['prompt_tokens']} tokens (cached {pf['cached_tokens']}), "
          f"ttft {pf['ttft_s']:.1f}s, longest decode gap {out['final']['max_gap_s']}; decoders at "
          f"{[r['prompt_tokens'] for r in dec]} tokens", flush=True)
    print(f"  MemAvailable minimum (GiB): {json.dumps(lows)} -> {'PASS' if ok else 'FAIL'} (>= {a.mem_min})",
          flush=True)
    return out


def mode_slots(c, a):
    """Per-slot resume under batching: sessions 1-4 get a first turn, then a second turn each (+~--append tokens); then
    a 5th session (slots full) and a third turn of session 1."""

    convs = {k: [{"role": "user", "content": doc(f"SL{k}{time.time_ns() % 100000}", a.doc) +
                  f"\n\nSession {k}: in one sentence, what is this?"}] for k in range(1, 6)}
    out = []
    for turn, k in enumerate([1, 2, 3, 4, 1, 2, 3, 4, 5, 1]):
        msgs = convs[k]
        r = chat(c, msgs, a.tokens)
        msgs.append({"role": "assistant", "content": r["text"]})
        msgs.append({"role": "user", "content": doc(f"APP{k}{turn}", a.append) + "\n\nAnd this part?"})
        print(f"  step {turn} session {k}: prompt {r['prompt_tokens']:>6} cached {r['cached_tokens']}  "
              f"ttft {r['ttft_s']:.2f}s", flush=True)
        out.append({"step": turn, "session": k, **{x: r[x] for x in ("ttft_s", "prompt_tokens", "cached_tokens")}})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--modes", default="sessions,followup")
    p.add_argument("--doc", type=int, default=30000)
    p.add_argument("--system", type=int, default=2000)
    p.add_argument("--append", type=int, default=2000)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--long-tokens", type=int, default=256)
    p.add_argument("--streams", default="1,2,4")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--delay", type=float, default=3.0)
    p.add_argument("--extra", default="{}")
    p.add_argument("--stress-target", type=int, default=250000)
    p.add_argument("--stress-step", type=int, default=60000)
    p.add_argument("--stress-final", type=int, default=32000)
    p.add_argument("--mem-hosts", default="local")
    p.add_argument("--mem-log")
    p.add_argument("--mem-min", type=float, default=8.0)
    p.add_argument("--label", default="")
    p.add_argument("--out")
    a = p.parse_args()
    c = Client(a.base, a.model, extra=json.loads(a.extra))
    res = {"label": a.label, "args": vars(a)}
    for m in a.modes.split(","):
        print(f"[{m}]", flush=True)
        res[m] = {"sessions": mode_sessions, "followup": mode_followup, "concurrent": mode_concurrent,
                  "stall": mode_stall, "batchexact": mode_batchexact, "slots": mode_slots,
                  "stress": mode_stress}[m](c, a)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
