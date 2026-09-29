#!/usr/bin/env python3
"""W12: 0490 checks on a running server: /v1/models max_model_len, an over-limit request -> 400
context_length_exceeded, and the pool edge (prompt + max_tokens within 64 of CONTEXT) admitted. api.py OUT [CONTEXT]"""
import json, sys, urllib.error, urllib.request
B = "http://127.0.0.1:8000"
out, ctx = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1048576
res = {}
def post(path, body, timeout=600):
    r = urllib.request.Request(B + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as f:
            return f.status, json.load(f)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
m = json.load(urllib.request.urlopen(B + "/v1/models", timeout=10))
e = m["data"][0]
res["models"] = e
res["max_model_len_ok"] = e.get("max_model_len") == ctx and e.get("context_length") == ctx
print("models:", {k: e.get(k) for k in ("id", "max_model_len", "context_length")}, "ok" if res["max_model_len_ok"] else "FAIL")
msgs = [{"role": "user", "content": "Reply with the single word OK."}]
st, d = post("/tokenize", {"messages": msgs, "chat_template_kwargs": {"enable_thinking": False}})
res["tokenize"] = [st, d]
p = d.get("count") if st == 200 else None
print("tokenize:", st, p, d.get("max_model_len") if isinstance(d, dict) else None)
base = {"model": "GLM-5.3-Flash-EXL3", "messages": msgs, "temperature": 0, "top_p": 1,
        "chat_template_kwargs": {"enable_thinking": False}}
st, d = post("/v1/chat/completions", dict(base, max_tokens=ctx))
err = (d.get("error") or {}) if isinstance(d, dict) else {}
res["over"] = [st, d]
res["over_ok"] = st == 400 and err.get("code") == "context_length_exceeded"
print("over limit:", st, err.get("code"), (err.get("message") or "")[:160], "ok" if res["over_ok"] else "FAIL")
if p:
    mt = ctx - p - 32          # prompt + max_tokens = CONTEXT - 32: within 64 of the limit
    st, d = post("/v1/chat/completions", dict(base, max_tokens=mt), timeout=900)
    res["edge"] = [st, d if st != 200 else {"content": d["choices"][0]["message"]["content"], "usage": d["usage"]}]
    res["edge_ok"] = st == 200
    print(f"pool edge (prompt {p} + max_tokens {mt} = {p + mt}):", st,
          repr(d["choices"][0]["message"]["content"])[:60] if st == 200 else d, "ok" if st == 200 else "FAIL")
json.dump(res, open(out, "w"), indent=1)
