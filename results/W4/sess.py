#!/usr/bin/env python3
"""0250 check: sessions of ~40k tokens visited in ORDER (letters), same prompt each visit (greedy, 64 tokens):
prints cached tokens, TTFT (prefill_s) and reply hash per visit. Docs are fixed by SEED so a later run can resume."""
import json, sys, time, urllib.request
FILLER = ("The quick brown fox jumps over the lazy dog while the committee reviews quarterly logistics, "
          "inventory forecasts, and the maintenance schedule for the northern warehouse. ")
seed, order = sys.argv[1], sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else None
res = []
for L in order:
    prompt = f"[session {seed}-{L}] Document {L}.\n" + FILLER * 1400 + f"\n\nSummarize document {L} in one sentence."
    body = {"model": "GLM-5.3-Flash-EXL3", "messages": [{"role": "user", "content": prompt}], "max_tokens": 64,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    t0 = time.time(); d = json.load(urllib.request.urlopen(r, timeout=1800)); tf = d.get("tensorfold", {})
    row = dict(s=L, t=round(time.time() - t0, 2), prompt=d["usage"]["prompt_tokens"], cached=tf.get("cached"),
               prefill_s=tf.get("prefill_s"), slot=tf.get("slot"), sha=tf.get("sha256"), sessions=tf.get("sessions"))
    res.append(row); print(json.dumps(row), flush=True)
if out: json.dump(res, open(out, "w"), indent=1)
