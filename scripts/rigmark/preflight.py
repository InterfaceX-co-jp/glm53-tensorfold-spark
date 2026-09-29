#!/usr/bin/env python3
"""Check that an endpoint serves every request shape RigMark's standard suite sends, with a few tiny requests.

    preflight.py --base-url http://127.0.0.1:8000 --model GLM-5.3-Flash-EXL3 --extra-body '{...}' [--out FILE]

RigMark (bench.py at the pinned revision) needs:
  decode / concurrency  POST /v1/chat/completions, stream, stream_options.include_usage, temperature 0, top_p 1, seed,
                        max_tokens; a final usage chunk with integer prompt_tokens / completion_tokens, "[DONE]",
                        finish_reason "stop", one choice at index 0, output in at least two SSE events; reasoning read
                        from delta.reasoning and delta.reasoning_content (both are concatenated if both are sent)
  prefill               POST /tokenize {model, prompt, add_special_tokens: false} -> {"tokens": [int]}, then
                        POST /v1/completions with "prompt": [token ids], add_special_tokens false, ignore_eos, max_tokens
                        8, stream; usage.prompt_tokens must equal len(ids) exactly; >= 2 SSE events for the 8 tokens
Exit status: 0 everything works; 3 decode works but the prefill path does not (run with --skip-prefill); 1 decode
does not work (RigMark cannot run). Uses the GPU for a few seconds: run it inside the test window only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SYSTEM = ("Work carefully and give a complete, technically precise answer. Do not mention benchmarking or the "
          "request nonce.")   # rigmark prompts.json "system" (v1.0.0)
UNIT = ("The repository contains a service, tests, documentation, and release notes. Each observation records "
        "behaviour, evidence, constraints, and the next action. ")      # rigmark prompts.json "prefill_unit"


def post(base: str, path: str, payload: dict[str, Any] | None, timeout: float):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def sse(base: str, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Read one stream the way rigmark's Client.stream does, and also note the raw field names."""
    t0 = time.monotonic()
    out: dict[str, Any] = {"events": 0, "measured_events": 0, "done_marker": False, "usage": None,
                           "finish_reason": None, "content": "", "reasoning": "", "reasoning_content": "",
                           "delta_keys": set(), "choices_ok": True, "error_event": None}
    first = last = None
    with post(base, path, payload, timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            enc = line[5:].strip()
            if enc == "[DONE]":
                out["done_marker"] = True
                break
            ev = json.loads(enc)
            out["events"] += 1
            if "error" in ev:
                out["error_event"] = ev["error"]
            if isinstance(ev.get("usage"), dict):
                out["usage"] = ev["usage"]
            choices = ev.get("choices") or []
            if not choices:
                continue
            if len(choices) != 1 or choices[0].get("index", 0) != 0:
                out["choices_ok"] = False
            ch = choices[0]
            if ch.get("finish_reason") is not None:
                out["finish_reason"] = ch["finish_reason"]
            delta = ch.get("delta") or {}
            out["delta_keys"].update(delta.keys())
            piece = ""
            for key in ("reasoning", "reasoning_content"):
                if isinstance(delta.get(key), str):
                    out[key] += delta[key]
                    piece += delta[key]
            content = delta.get("content") if isinstance(delta.get("content"), str) else ""
            if not content and isinstance(ch.get("text"), str):
                content = ch["text"]
            out["content"] += content
            if piece or content:
                now = time.monotonic()
                first = first or now
                last = now
                out["measured_events"] += 1
    out["wall_s"] = round(time.monotonic() - t0, 3)
    out["ttft_s"] = None if first is None else round(first - t0, 3)
    out["delta_keys"] = sorted(out["delta_keys"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--extra-body", default="{}")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    base = a.base_url.rstrip("/").removesuffix("/v1")
    extra = json.loads(a.extra_body)
    report: dict[str, Any] = {"base_url_is_loopback": "127.0.0.1" in base or "localhost" in base, "checks": {}}
    gaps: list[str] = []
    decode_ok = prefill_ok = True
    c = report["checks"]

    # 1. models
    try:
        with post(base, "/v1/models", None, 30) as r:
            ids = [m.get("id") for m in (json.load(r).get("data") or [])]
        c["models"] = {"ok": a.model in ids, "ids": ids}
        if a.model not in ids:
            gaps.append(f"/v1/models does not list {a.model!r} (lists {ids})")
            decode_ok = False
    except Exception as exc:  # noqa: BLE001
        c["models"] = {"ok": False, "error": repr(exc)}
        gaps.append("/v1/models failed")
        decode_ok = False

    # 2. chat stream exactly as rigmark builds it (plus the sweep's extra body)
    payload = {"model": a.model, "messages": [{"role": "system", "content": SYSTEM},
                                              {"role": "user", "content": "Request nonce: preflight\n\n"
                                               "What is 17 * 23? Answer with the number only."}],
               "temperature": 0.0, "top_p": 1.0, "seed": 20260905, "max_tokens": 1024, "stream": True,
               "stream_options": {"include_usage": True}, **extra}
    try:
        s = sse(base, "/v1/chat/completions", payload, a.timeout)
        u = s["usage"] or {}
        both = bool(s["reasoning"]) and s["reasoning"] == s["reasoning_content"]
        chk = {
            "usage_ints": isinstance(u.get("prompt_tokens"), int) and isinstance(u.get("completion_tokens"), int),
            "done_marker": s["done_marker"], "finish_stop": s["finish_reason"] == "stop",
            "single_choice": s["choices_ok"], "multi_event": s["measured_events"] >= 2,
            "visible_answer": bool(s["content"].strip()),
            "thinking_on": bool(s["reasoning"] or s["reasoning_content"]),
        }
        c["chat"] = {"ok": all(chk.values()), **chk, "usage": u, "finish_reason": s["finish_reason"],
                     "delta_keys": s["delta_keys"], "measured_events": s["measured_events"],
                     "reasoning_chars": len(s["reasoning"]), "reasoning_content_chars": len(s["reasoning_content"]),
                     "reasoning_sent_twice": both, "answer": s["content"][:200], "ttft_s": s["ttft_s"],
                     "wall_s": s["wall_s"], "error_event": s["error_event"]}
        for k, v in chk.items():
            if not v:
                gaps.append(f"chat: {k} failed")
        if not all(v for k, v in chk.items() if k not in ("thinking_on", "finish_stop", "visible_answer")):
            decode_ok = False
        if both:
            gaps.append("chat: reasoning is sent as both delta.reasoning and delta.reasoning_content; rigmark adds "
                        "them, so this receipt's reasoning_characters are doubled (timing and tokens unaffected)")
    except Exception as exc:  # noqa: BLE001
        c["chat"] = {"ok": False, "error": repr(exc)}
        gaps.append(f"chat stream failed: {exc!r}")
        decode_ok = False

    # 3. /tokenize
    ids: list[int] = []
    try:
        with post(base, "/tokenize", {"model": a.model, "prompt": "Unique cold-prefill nonce preflight.\n",
                                      "add_special_tokens": False}, 30) as r:
            prefix = json.load(r).get("tokens") or []
        with post(base, "/tokenize", {"model": a.model, "prompt": UNIT, "add_special_tokens": False}, 30) as r:
            unit = json.load(r).get("tokens") or []
        ok = bool(prefix) and bool(unit) and all(type(t) is int for t in prefix + unit)
        c["tokenize"] = {"ok": ok, "prefix_tokens": len(prefix), "unit_tokens": len(unit)}
        ids = (prefix + unit * 8)[:256] if ok else []
        if not ok:
            gaps.append("/tokenize returned no integer token list")
            prefill_ok = False
    except urllib.error.HTTPError as exc:
        c["tokenize"] = {"ok": False, "http_status": exc.code}
        gaps.append(f"/tokenize: HTTP {exc.code} (rigmark's prefill phase needs vLLM's /tokenize)")
        prefill_ok = False
    except Exception as exc:  # noqa: BLE001
        c["tokenize"] = {"ok": False, "error": repr(exc)}
        gaps.append(f"/tokenize failed: {exc!r}")
        prefill_ok = False

    # 4. token-id completions (only with ids from the server's own tokenizer; a guessed id list could mislead)
    if ids:
        try:
            s = sse(base, "/v1/completions", {"model": a.model, "prompt": ids, "add_special_tokens": False,
                                              "max_tokens": 8, "ignore_eos": True, "temperature": 0.0,
                                              "stream": True, "stream_options": {"include_usage": True}}, a.timeout)
            u = s["usage"] or {}
            chk = {"prompt_tokens_exact": u.get("prompt_tokens") == len(ids),
                   "completion_tokens_8": u.get("completion_tokens") == 8,
                   "multi_event": s["measured_events"] >= 2, "done_marker": s["done_marker"]}
            c["completions_token_ids"] = {"ok": chk["prompt_tokens_exact"] and chk["multi_event"]
                                          and chk["done_marker"], **chk, "requested": len(ids), "usage": u,
                                          "measured_events": s["measured_events"], "ttft_s": s["ttft_s"],
                                          "error_event": s["error_event"]}
            if not c["completions_token_ids"]["ok"]:
                gaps.append(f"/v1/completions with token ids: {chk}")
                prefill_ok = False
        except urllib.error.HTTPError as exc:
            c["completions_token_ids"] = {"ok": False, "http_status": exc.code}
            gaps.append(f"/v1/completions with a token-id prompt: HTTP {exc.code}")
            prefill_ok = False
        except Exception as exc:  # noqa: BLE001
            c["completions_token_ids"] = {"ok": False, "error": repr(exc)}
            gaps.append(f"/v1/completions with a token-id prompt failed: {exc!r}")
            prefill_ok = False
    else:
        c["completions_token_ids"] = {"ok": False, "skipped": "no /tokenize ids"}
        prefill_ok = False

    report.update(decode_ok=decode_ok, prefill_ok=prefill_ok, gaps=gaps)
    text = json.dumps(report, indent=2, sort_keys=True)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)
    return 0 if decode_ok and prefill_ok else (3 if decode_ok else 1)


if __name__ == "__main__":
    sys.exit(main())
