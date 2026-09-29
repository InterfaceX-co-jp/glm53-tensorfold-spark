#!/usr/bin/env python3
"""3 streams with ~98k contexts decode while a 4th 98k prompt prefills (tf_knobs from argv[1]); phase times printed."""
import json, sys, time, threading, urllib.request
FILLER = ("The quick brown fox jumps over the lazy dog while the committee reviews quarterly logistics, "
          "inventory forecasts, and the maintenance schedule for the northern warehouse. ")
knobs = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
def req(prompt, n, ignore=False):
    body = {"model": "GLM-5.3-Flash-EXL3", "messages": [{"role": "user", "content": prompt}], "max_tokens": n,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    if ignore: body["ignore_eos"] = True
    if knobs: body["tf_knobs"] = knobs
    r = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    t0 = time.time(); d = json.load(urllib.request.urlopen(r, timeout=3600)); tf = d.get("tensorfold", {})
    return dict(t=round(time.time() - t0, 1), pt=d["usage"]["prompt_tokens"], ct=d["usage"]["completion_tokens"],
                cached=tf.get("cached"), prefill_s=tf.get("prefill_s"), decode_s=tf.get("decode_s"), slot=tf.get("slot"),
                kinds=tf.get("round_kinds"), tpr=tf.get("tokens_per_round"))
def run(fns):
    out = [None] * len(fns)
    ths = [threading.Thread(target=lambda i=i, f=f: out.__setitem__(i, f())) for i, f in enumerate(fns)]
    [t.start() for t in ths]; [t.join() for t in ths]; return out
docs = [f"[mem-{i}-{time.time_ns()}] " + FILLER * 3500 for i in range(4)]
print(time.strftime("%T"), "phase A: 3 x ~98k prompts", flush=True)
print(json.dumps(run([lambda d=d: req(d + "\n\nOne sentence: what is reviewed?", 16) for d in docs[:3]])), flush=True)
print(time.strftime("%T"), "phase B: 3 decoders (resume) + a 4th 98k prefill", flush=True)
def late():
    time.sleep(3); return req(docs[3] + "\n\nOne sentence: what is reviewed?", 64)
fns = [lambda d=d: req(d + "\n\nOne sentence: what is reviewed?\n\nNow write a long essay about warehouses.", 1500, True) for d in docs[:3]] + [late]
print(json.dumps(run(fns)), flush=True)
print(time.strftime("%T"), "done", flush=True)
