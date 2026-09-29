#!/usr/bin/env python3
"""W10 (0370 plan item 4): 4 prompts greedy (256 tokens, thinking off) alone, then together with the 4th streamed and
its client disconnecting after ~50 chunks: the other 3 replies must equal alone, and the server must keep serving
(a follow-up request, /health). Usage: cancel.py OUT"""
import http.client, json, sys, threading, time, urllib.request
P = ["Explain how a refrigerator works, step by step.", "Write a Python class for a bounded LRU cache with tests.",
     "List ten facts about the Moon, one per line.", "Tell a long story about a fox who learns to sail."]
def body(p, stream=False):
    return {"model": "GLM-5.3-Flash-EXL3", "messages": [{"role": "user", "content": p}], "max_tokens": 256,
            "temperature": 0, "top_p": 1, "stream": stream, "chat_template_kwargs": {"enable_thinking": False}}
def req(p):
    r = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", data=json.dumps(body(p)).encode(),
                               headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=600)); return d["choices"][0]["message"]["content"]
alone = [req(p) for p in P[:3]]
got = [None] * 3; chunks = [0]
def go(i): got[i] = req(P[i])
def cut():
    c = http.client.HTTPConnection("127.0.0.1", 8000, timeout=600)
    c.request("POST", "/v1/chat/completions", json.dumps(body(P[3], True)), {"Content-Type": "application/json"})
    r = c.getresponse()
    while chunks[0] < 50:
        if not r.fp.readline(): break
        chunks[0] += 1
    c.sock.close(); c.close()
ts = [threading.Thread(target=go, args=(i,)) for i in range(3)] + [threading.Thread(target=cut)]
t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]
same = [a == b for a, b in zip(alone, got)]
after = req("What is 17*23? Answer with the number only.")
h = json.load(urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=10))
ok = all(same) and "391" in after and h.get("ok")
json.dump(dict(same=same, chunks_read=chunks[0], after=after, health=h, wall=round(time.time() - t0, 1), ok=ok), open(sys.argv[1], "w"), indent=1)
print("cancel: others == alone", same, "| chunks read", chunks[0], "| after", repr(after), "| health", h.get("ok"), "| CANCEL", "OK" if ok else "FAIL")
