#!/usr/bin/env python3
"""Self-distillation data for the drafters: diverse prompts through the served model with patches/0430's dump on
(standard library only). docs/DRAFTER-TRAINING.md §2 has the plan, the mix and the time budget.

The server does the recording: start it with GLM53_TF_DRAFT_DUMP=<dir> (both ranks) and GLM53_TF_BATCH >= 2, so every
prompt row (kind p) and every token the target generates (kind d: its own samples) becomes a training record. This
driver only produces the traffic and keeps a ledger of what it sent (``results.jsonl``: id, category, prompt / reply
tokens, finish, seconds), so a run can stop and resume and the budget is tracked.

    collect   build prompts.jsonl from open datasets (HF datasets-server rows API, no login), local opencode sessions
              (agent traffic: the user's own task mix, rebuilt as OpenAI tool calls) and local JSONL files:
                python3 bench/gendata.py collect --out prompts.jsonl --n 60000 \\
                    --mix chat:0.30,code:0.25,agent:0.25,reasoning:0.15,tools:0.05
    run       send them (--concurrency parallel requests; thinking on at --effort for reasoning / agent / code,
              sampled at --temperature) until --target-tokens generated tokens or --hours:
                python3 bench/gendata.py run --prompts prompts.jsonl --base http://127.0.0.1:8000 \\
                    --model GLM-5.3-Flash-EXL3 --concurrency 4 --target-tokens 20000000 --ledger results.jsonl
    tforce    teacher-force existing conversations (assistant turns included) as prefill only (max_tokens 1): rows
              at prefill speed (~1,500 tok/s); the text is not the target's, its labels (the dumped distributions)
              are:
                python3 bench/gendata.py tforce --prompts convs.jsonl --base ... --target-tokens 100000000
    plan      the time budget for a token target at given speeds:
                python3 bench/gendata.py plan --gen-tokens 20e6 --tforce-tokens 80e6

Prompt record (one JSON a line): {"id", "cat", "messages": [...], "tools": [...]?, "thinking": bool,
"effort": "high"|..., "max_tokens": int}. Categories: chat, code, agent, reasoning, tools.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROWS_API = "https://datasets-server.huggingface.co/rows"

# (category, dataset, config, split, converter, licence) - public, ungated when checked (2026-09-29)
SOURCES = {
    "ultrachat": ("chat", "HuggingFaceH4/ultrachat_200k", "default", "train_sft", "messages_first_user", "MIT"),
    "wildchat": ("chat", "allenai/WildChat-1M", "default", "train", "wildchat", "ODC-BY"),
    "evol_code": ("code", "theblackcat102/evol-codealpaca-v1", "default", "train", "instruction", "Apache-2.0"),
    "opencode_instruct": ("code", "nvidia/OpenCodeInstruct", "train", "train", "input", "CC-BY-4.0"),
    "oss_instruct": ("code", "bigcode/self-oss-instruct-sc2-exec-filter-50k", "default", "train", "oss_instruct",
                     "ODC-BY"),
    "github_code": ("code", "codeparrot/github-code-clean", "all-all", "train", "code_file", "per-file (filtered)"),
    "openr1_math": ("reasoning", "open-r1/OpenR1-Math-220k", "default", "train", "problem", "Apache-2.0"),
    "toolace": ("tools", "Team-ACE/ToolACE", "default", "train", "toolace", "Apache-2.0"),
    "glaive_fc": ("tools", "glaiveai/glaive-function-calling-v2", "default", "train", "glaive", "Apache-2.0"),
}
DEFAULT_SOURCES = {"chat": ["ultrachat", "wildchat"], "code": ["evol_code", "opencode_instruct", "oss_instruct",
                                                              "github_code"],
                   "reasoning": ["openr1_math"], "tools": ["toolace", "glaive_fc"], "agent": ["opencode"]}
PERMISSIVE = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc", "unlicense", "cc0-1.0", "mpl-2.0"}
CODE_ASKS = ["Review this file: list concrete bugs and risky spots, then propose fixes as a unified diff.",
             "Explain what this code does, then refactor it for readability without changing behaviour.",
             "Write unit tests for this file (the language's usual test framework), covering edge cases.",
             "Find the performance hot spots in this code and rewrite the worst one.",
             "Add type annotations / documentation where they are missing and explain the non-obvious parts.",
             "Port the core logic of this file to Python (or to Rust if it already is Python)."]

# OpenCode-like agent tools (names and shapes of its built-in tools; descriptions written here)
AGENT_TOOLS = [
    {"type": "function", "function": {"name": n, "description": d, "parameters": {
        "type": "object", "properties": {k: {"type": t, "description": kd} for k, t, kd in props},
        "required": [p[0] for p in props if p[0] in req]}}}
    for n, d, props, req in [
        ("bash", "Run a shell command in the project and return its output.",
         [("command", "string", "the command"), ("description", "string", "what it does, in a few words"),
          ("timeout", "number", "milliseconds")], {"command"}),
        ("read", "Read a file (optionally a line range).",
         [("filePath", "string", "absolute path"), ("offset", "number", "first line"), ("limit", "number", "lines")],
         {"filePath"}),
        ("edit", "Replace an exact string in a file.",
         [("filePath", "string", "absolute path"), ("oldString", "string", "text to replace"),
          ("newString", "string", "replacement"), ("replaceAll", "boolean", "every occurrence")],
         {"filePath", "oldString", "newString"}),
        ("write", "Write a file.", [("filePath", "string", "absolute path"), ("content", "string", "the content")],
         {"filePath", "content"}),
        ("glob", "Find files by glob pattern.", [("pattern", "string", "glob"), ("path", "string", "directory")],
         {"pattern"}),
        ("grep", "Search file contents with a regular expression.",
         [("pattern", "string", "regex"), ("path", "string", "directory"), ("include", "string", "file glob")],
         {"pattern"}),
        ("list", "List a directory.", [("path", "string", "directory")], set()),
        ("todowrite", "Replace the task list.", [("todos", "array", "the tasks")], {"todos"}),
        ("webfetch", "Fetch a URL as text.", [("url", "string", "URL"), ("format", "string", "text|markdown|html")],
         {"url"}),
    ]]
AGENT_SYSTEM = ("You are an autonomous coding agent working in the user's repository through tools. Investigate "
                "before editing, make minimal correct changes, run the tests or commands that verify them, and "
                "report briefly what you changed and why. Use absolute paths.")


# -- collect -------------------------------------------------------------------------------------------------------
def _rows(dataset: str, config: str, split: str, offset: int, length: int = 100) -> list[dict]:
    q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset,
                                "length": length})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{ROWS_API}?{q}", timeout=60) as r:
                return [x["row"] for x in json.loads(r.read()).get("rows", [])]
        except Exception as exc:  # noqa: BLE001 - network: retry, then give up on this page
            time.sleep(2 * (attempt + 1))
            err = exc
    print(f"[gendata] {dataset} offset {offset}: {err}", file=sys.stderr)
    return []


def _num_rows(dataset: str, config: str, split: str) -> int:
    q = urllib.parse.urlencode({"dataset": dataset})
    try:
        with urllib.request.urlopen(f"https://datasets-server.huggingface.co/size?{q}", timeout=60) as r:
            for s in json.loads(r.read())["size"]["splits"]:
                if s["config"] == config and s["split"] == split:
                    return int(s["num_rows"])
    except Exception:  # noqa: BLE001
        pass
    return 10_000


def _parse_functions(text: str) -> list[dict]:
    """Function specs embedded in a system prompt (JSON objects or a JSON list) as OpenAI tools."""

    out, dec, i = [], json.JSONDecoder(), 0
    while True:
        m = re.search(r"[\[{]", text[i:])
        if not m:
            break
        j = i + m.start()
        try:
            obj, end = dec.raw_decode(text[j:])
        except json.JSONDecodeError:
            i = j + 1
            continue
        for f in obj if isinstance(obj, list) else [obj]:
            if isinstance(f, dict) and "name" in f:
                params = f.get("parameters") or {"type": "object", "properties": {}}
                if params.get("type") == "dict":
                    params = dict(params, type="object")
                out.append({"type": "function", "function": {"name": f["name"],
                                                             "description": f.get("description", ""),
                                                             "parameters": params}})
        i = j + end
    return out


def convert(kind: str, row: dict, rng: random.Random) -> dict | None:
    """A dataset row -> {"messages", "tools"?} or None (unusable)."""

    if kind == "messages_first_user":
        msgs = [m for m in row.get("messages", []) if m.get("role") == "user"]
        return {"messages": [{"role": "user", "content": msgs[0]["content"]}]} if msgs else None
    if kind == "wildchat":
        if row.get("language") not in (None, "English") or row.get("toxic") or row.get("redacted"):
            return None
        conv = row.get("conversation") or []
        turns = [{"role": t["role"], "content": t["content"]} for t in conv if t.get("role") in ("user", "assistant")]
        # up to the last user turn: earlier assistant turns stay (another model's text: context, not labels)
        while turns and turns[-1]["role"] != "user":
            turns.pop()
        return {"messages": turns} if turns else None
    if kind in ("instruction", "input", "problem"):
        text = row.get(kind)
        return {"messages": [{"role": "user", "content": text}]} if text else None
    if kind == "oss_instruct":
        text = row.get("instruction") or row.get("prompt")
        return {"messages": [{"role": "user", "content": text}]} if text else None
    if kind == "code_file":
        if str(row.get("license", "")).lower() not in PERMISSIVE:
            return None
        code = row.get("code") or ""
        if not 400 <= len(code) <= 24_000:
            return None
        ask = rng.choice(CODE_ASKS)
        return {"messages": [{"role": "user", "content": f"{ask}\n\n`{row.get('path', 'file')}`:\n```\n{code}\n```"}]}
    if kind == "toolace":
        tools = _parse_functions(row.get("system", ""))
        conv = row.get("conversations") or []
        user = next((c["value"] for c in conv if c.get("from") == "user"), None)
        return {"messages": [{"role": "user", "content": user}], "tools": tools} if user and tools else None
    if kind == "glaive":
        tools = _parse_functions(row.get("system", ""))
        m = re.search(r"USER:(.*?)(?:ASSISTANT:|$)", row.get("chat", ""), re.S)
        if not m or not tools:
            return None
        return {"messages": [{"role": "user", "content": m.group(1).strip()}], "tools": tools}
    return None


def opencode_prompts(db: str, limit: int, rng: random.Random) -> list[dict]:
    """Agent steps of local opencode sessions as OpenAI requests (system, user text, assistant tool calls, tool
    results) cut before an assistant step: the model regenerates that step. Nothing leaves the machine but the
    requests this driver sends to the local server."""

    con = sqlite3.connect(f"file:{os.path.expanduser(db)}?mode=ro", uri=True)
    msgs = con.execute("select id, session_id, time_created, data from message order by session_id, time_created, id")
    by_s = collections.defaultdict(list)
    for mid, sid, t, data in msgs:
        by_s[sid].append((mid, json.loads(data)))
    parts = collections.defaultdict(list)
    for mid, data in con.execute("select message_id, data from part order by message_id, id"):
        parts[mid].append(json.loads(data))
    out = []
    for sid, ms in by_s.items():
        hist: list[dict] = [{"role": "system", "content": AGENT_SYSTEM}]
        for mid, m in ms:
            ps = parts.get(mid, [])
            if m.get("role") == "user":
                text = "".join(p.get("text", "") for p in ps if p.get("type") == "text")
                if text:
                    hist.append({"role": "user", "content": text})
                continue
            if m.get("role") != "assistant":
                continue
            if any(h["role"] == "user" for h in hist) and hist[-1]["role"] in ("user", "tool"):
                out.append({"messages": [dict(h) for h in hist], "tools": AGENT_TOOLS})
            calls, results = [], []
            text = "".join(p.get("text", "") for p in ps if p.get("type") == "text")
            for p in ps:
                if p.get("type") == "tool":
                    st = p.get("state") or {}
                    cid = p.get("callID") or f"call_{len(calls)}"
                    calls.append({"id": cid, "type": "function", "function": {
                        "name": p.get("tool", ""), "arguments": json.dumps(st.get("input") or {})}})
                    o = st.get("output")
                    results.append({"role": "tool", "tool_call_id": cid,
                                    "content": o[:30000] if isinstance(o, str) else json.dumps(o)[:30000]})
            msg = {"role": "assistant", "content": text}
            if calls:
                msg["tool_calls"] = calls
            hist.append(msg)
            hist.extend(results)
    rng.shuffle(out)
    return out[:limit] if limit else out


def collect(a) -> None:
    rng = random.Random(a.seed)
    mix = {k: float(v) for k, v in (x.split(":") for x in a.mix.split(","))}
    total = sum(mix.values())
    want = {k: int(round(a.n * v / total)) for k, v in mix.items()}
    out = Path(a.out)
    seen: set[str] = set()
    if out.exists():
        for line in out.read_text().splitlines():
            seen.add(json.loads(line)["id"])
    f = open(out, "a")
    for cat, n in want.items():
        got = 0
        srcs = DEFAULT_SOURCES[cat]
        per = max(1, n // max(1, len(srcs)))
        for src in srcs:
            if src == "opencode":
                recs = opencode_prompts(a.opencode_db, per, rng) if a.opencode_db and Path(
                    os.path.expanduser(a.opencode_db)).exists() else []
                if not recs:
                    print("[gendata] agent: no opencode database (--opencode-db); use the production dump instead",
                          file=sys.stderr)
            elif src.startswith("jsonl:"):
                recs = [json.loads(x) for x in Path(src[6:]).read_text().splitlines() if x.strip()][:per]
            else:
                _, ds, cfg, split, conv, _lic = SOURCES[src]
                size = _num_rows(ds, cfg, split)
                recs, tries = [], 0
                while len(recs) < per and tries < per // 50 + 20:
                    tries += 1
                    for row in _rows(ds, cfg, split, rng.randrange(max(1, size - 100))):
                        r = convert(conv, row, rng)
                        if r is not None:
                            recs.append(r)
                recs = recs[:per]
            for r in recs:
                if len(json.dumps(r)) > a.max_prompt_chars:
                    continue
                key = hashlib.sha1(json.dumps(r["messages"], sort_keys=True).encode()).hexdigest()[:16]
                if key in seen:
                    continue
                seen.add(key)
                thinking = cat in ("reasoning", "agent", "code") or rng.random() < 0.5
                rec = {"id": key, "cat": cat, "src": src, "messages": r["messages"], "thinking": thinking,
                       "effort": "high" if thinking else None,
                       "max_tokens": {"reasoning": 16384, "agent": 8192, "code": 8192}.get(cat, 4096)}
                if r.get("tools"):
                    rec["tools"] = r["tools"]
                f.write(json.dumps(rec) + "\n")
                got += 1
        print(f"[gendata] {cat}: {got} prompts", file=sys.stderr)
    f.close()


# -- run / tforce --------------------------------------------------------------------------------------------------
def post(base: str, path: str, body: dict, timeout: float = 3600) -> dict:
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def drive(a, tforce: bool) -> None:
    prompts = [json.loads(x) for x in Path(a.prompts).read_text().splitlines() if x.strip()]
    random.Random(a.seed).shuffle(prompts)
    ledger = Path(a.ledger)
    done, made = set(), 0
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            r = json.loads(line)
            done.add(r["id"])
            made += r.get("prompt_tokens" if tforce else "completion_tokens", 0) or 0
    todo = [p for p in prompts if p["id"] not in done]
    lock = threading.Lock()
    stop_at = time.time() + a.hours * 3600 if a.hours else None
    state = {"made": made, "i": 0, "t0": time.time(), "made0": made}
    out = open(ledger, "a", buffering=1)

    def worker() -> None:
        while True:
            with lock:
                if state["i"] >= len(todo) or state["made"] >= a.target_tokens or (stop_at and time.time() > stop_at):
                    return
                p = todo[state["i"]]
                state["i"] += 1
            body = {"model": a.model, "messages": p["messages"], "stream": False,
                    "temperature": a.temperature, "top_p": a.top_p,
                    "max_tokens": 1 if tforce else min(p.get("max_tokens", a.max_tokens), a.max_tokens),
                    "chat_template_kwargs": {"enable_thinking": bool(p.get("thinking"))}}
            if p.get("tools"):
                body["tools"] = p["tools"]
            if p.get("effort") and not tforce:
                body["reasoning_effort"] = p["effort"]
            if a.seeded:
                body["seed"] = int(p["id"][:8], 16)
            t0 = time.time()
            try:
                res = post(a.base, "/v1/chat/completions", body)
                u = res.get("usage", {})
                rec = {"id": p["id"], "cat": p.get("cat"), "prompt_tokens": u.get("prompt_tokens"),
                       "completion_tokens": u.get("completion_tokens"),
                       "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                       "finish": (res.get("choices") or [{}])[0].get("finish_reason"), "s": round(time.time() - t0, 2)}
            except Exception as exc:  # noqa: BLE001 - one failed request never stops the run
                rec = {"id": p["id"], "cat": p.get("cat"), "error": f"{type(exc).__name__}: {exc}"[:300],
                       "s": round(time.time() - t0, 2)}
            with lock:
                n = (rec.get("prompt_tokens") if tforce else rec.get("completion_tokens")) or 0
                if tforce and rec.get("cached"):
                    n -= rec["cached"]
                state["made"] += n
                out.write(json.dumps(rec) + "\n")
                dt = time.time() - state["t0"]
                rate = (state["made"] - state["made0"]) / max(dt, 1e-9)
                left = (a.target_tokens - state["made"]) / max(rate, 1e-9)
                if state["i"] % 20 == 0:
                    print(f"[gendata] {state['i']}/{len(todo)} requests, {state['made']:,} tokens, "
                          f"{rate:,.1f} tok/s, ~{left / 3600:.1f} h to {a.target_tokens:,.0f}", file=sys.stderr)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(a.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"[gendata] done: {state['made']:,} tokens", file=sys.stderr)


def plan(a) -> None:
    """Hours for a generated-token and a teacher-forced-token target at the given speeds, and the dump's size."""

    gen_h = a.gen_tokens / a.gen_tps / 3600
    pre_h = a.gen_tokens * a.prompt_ratio / a.prefill_tps / 3600
    tf_h = a.tforce_tokens / a.prefill_tps / 3600
    rows = a.gen_tokens * (1 + a.prompt_ratio) + a.tforce_tokens
    per = {"mtp only": 8388, "fp8 taps (5)": 28888, "bf16 taps (5)": 49348, "fp8 taps (9)": 45288}
    print(json.dumps({"generation_h": round(gen_h + pre_h, 1), "of_which_prompt_prefill_h": round(pre_h, 1),
                      "teacher_forcing_h": round(tf_h, 1), "total_h": round(gen_h + pre_h + tf_h, 1),
                      "rows": f"{rows:,.0f}",
                      "dump_TB": {k: round(rows * v / 1e12, 2) for k, v in per.items()}}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", required=True)
    c.add_argument("--n", type=int, default=60000)
    c.add_argument("--mix", default="chat:0.30,code:0.25,agent:0.25,reasoning:0.15,tools:0.05")
    c.add_argument("--opencode-db", default="~/.local/share/opencode/opencode.db")
    c.add_argument("--max-prompt-chars", type=int, default=600_000, help="skip longer prompts (~150k tokens)")
    c.add_argument("--seed", type=int, default=0)
    for name, tf in (("run", False), ("tforce", True)):
        r = sub.add_parser(name)
        r.add_argument("--prompts", required=True)
        r.add_argument("--base", default="http://127.0.0.1:8000")
        r.add_argument("--model", default="GLM-5.3-Flash-EXL3")
        r.add_argument("--concurrency", type=int, default=4)
        r.add_argument("--target-tokens", type=float, default=2e7)
        r.add_argument("--hours", type=float, default=0.0)
        r.add_argument("--temperature", type=float, default=0.0 if tf else 0.8)
        r.add_argument("--top-p", type=float, default=0.95)
        r.add_argument("--max-tokens", type=int, default=16384)
        r.add_argument("--seeded", type=int, default=1, help="a seed per prompt (reproducible samples)")
        r.add_argument("--ledger", default=f"{name}-results.jsonl")
        r.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("plan")
    p.add_argument("--gen-tokens", type=float, default=2e7)
    p.add_argument("--tforce-tokens", type=float, default=8e7)
    p.add_argument("--gen-tps", type=float, default=78.0, help="aggregate decode tok/s (4 streams: W10 78)")
    p.add_argument("--prefill-tps", type=float, default=1500.0, help="prefill tok/s (W10: 1,577-1,614)")
    p.add_argument("--prompt-ratio", type=float, default=1.0, help="new prompt tokens per generated token")
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(a)
    elif a.cmd == "plan":
        plan(a)
    else:
        drive(a, a.cmd == "tforce")


if __name__ == "__main__":
    main()
