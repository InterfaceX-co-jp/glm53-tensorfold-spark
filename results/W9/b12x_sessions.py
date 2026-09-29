#!/usr/bin/env python3
"""W9 / 0360 plan step 3 on the production config: per request tf_knobs.b12x = BITS (default 4).
  sessions: 3 conversations x 3 turns over a unique ~40k-token document; turn 2+ must report cached > 0 and the same
            reply sha as the same request cold ("draft": false = a fresh prefill with the same knobs, serial decoding)
  batch:    4 unique ~8k prompts, each alone (draft false: fresh) then all 4 at once (draft false): same replies
Usage: b12x_sessions.py OUT [BITS]"""
import json, sys, threading, time, urllib.request
out = sys.argv[1]; bits = int(sys.argv[2]) if len(sys.argv) > 2 else 4
FILL = ("Section {i}: the northern warehouse logged {a} pallets of copper wire, {b} crates of valves and a note that "
        "the forklift on dock {c} needs new tyres before the audit. ")
def doc(tag, words):
    s, i = [], 0
    while sum(len(x) for x in s) < words * 6:
        s.append(FILL.format(i=i, a=(i * 37 + len(tag)) % 997, b=(i * 53) % 311, c=i % 9)); i += 1
    return f"[{tag}] " + "".join(s)
def req(messages, extra):
    body = {"model": "GLM-5.3-Flash-EXL3", "messages": messages, "max_tokens": 96, "temperature": 0, "top_p": 1,
            "chat_template_kwargs": {"enable_thinking": False}, "tf_knobs": {"b12x": bits}}
    body.update(extra)
    r = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=1800)); tf = d.get("tensorfold", {})
    return dict(text=d["choices"][0]["message"]["content"], sha=tf.get("sha256"), cached=tf.get("cached"),
                prompt=d["usage"]["prompt_tokens"], prefill_s=tf.get("prefill_s"), knobs=tf.get("tf_knobs", {}).get("b12x"))
res = {"bits": bits, "sessions": [], "batch": {}}
ok = True
qs = ["Which dock's forklift needs tyres in section 3? One line.", "How many crates of valves in section 5? One line.",
      "Summarise sections 7 and 8 in one sentence."]
for k in range(3):
    msgs = [{"role": "user", "content": doc(f"w9-b12x-{bits}-{k}-{time.time_ns()}", 40000 // 1) + "\n\n" + qs[0]}]
    for turn in range(3):
        if turn:
            msgs = msgs + [{"role": "assistant", "content": prev}, {"role": "user", "content": qs[turn]}]
        a = req(msgs, {})
        cold = req(msgs, {"draft": False}) if turn else None
        good = turn == 0 or (a["cached"] and a["cached"] > 0 and a["sha"] == cold["sha"] and a["text"] == cold["text"])
        ok = ok and bool(good)
        row = dict(conv=k, turn=turn, prompt=a["prompt"], cached=a["cached"], sha=a["sha"], prefill_s=a["prefill_s"],
                   cold_sha=cold and cold["sha"], cold_cached=cold and cold["cached"], same=bool(good), knobs=a["knobs"])
        res["sessions"].append(row); print(json.dumps(row), flush=True)
        prev = a["text"]
alone, together = [], [None] * 4
prompts = [doc(f"w9-bb-{bits}-{i}-{time.time_ns()}", 8000) + "\n\nList three numbers from section 2." for i in range(4)]
for p in prompts:
    alone.append(req([{"role": "user", "content": p}], {"draft": False}))
def go(i): together[i] = req([{"role": "user", "content": prompts[i]}], {"draft": False})
ts = [threading.Thread(target=go, args=(i,)) for i in range(4)]
[t.start() for t in ts]; [t.join() for t in ts]
same = [a["sha"] == b["sha"] and a["text"] == b["text"] for a, b in zip(alone, together)]
res["batch"] = dict(same=same, shas=[a["sha"] for a in alone], cached=[b["cached"] for b in together])
print("batch == alone:", same, flush=True)
ok = ok and all(same)
res["ok"] = ok
json.dump(res, open(out, "w"), indent=1)
print("B12X SESSIONS", "OK" if ok else "FAIL", flush=True)
