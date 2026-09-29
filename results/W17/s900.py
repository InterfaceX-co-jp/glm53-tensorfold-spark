#!/usr/bin/env python3
"""W17 load S (MEMORY-SAFETY.md §5 step 5): three looping streamed conversations (~8k-token prompts, 8,192 max_tokens,
ignore_eos, thinking low) keep slots busy while a ~900k-token needle prompt (W6 needle.py's document, needle at 0.4)
prefills beside them; then the same needle prompt resent (resume at n - 64). Per stream: the largest gap between
streamed chunks while the needle ran. Run mem.sh beside it for MemAvailable.

    s900.py OUT.json [NEEDLE_N=1000000] [STREAMS=3]"""
import json, sys, threading, time, urllib.request
sys.path.insert(0, "bench")
from multiturn import doc  # noqa: E402

BASE = "http://127.0.0.1:8000"; MODEL = "GLM-5.3-Flash-EXL3"
out = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000; S = int(sys.argv[3]) if len(sys.argv) > 3 else 3
stop = threading.Event(); needle_on = threading.Event(); needle_off = threading.Event()
streams = [dict(i=i, requests=0, tokens=0, gaps_needle=[], max_gap_needle=0.0, max_gap_other=0.0, errors=[]) for i in range(S)]


def stream_loop(st):
    k = 0
    while not stop.is_set():
        text = doc(f"S{st['i']}-{k}-{time.time_ns() % 100000}", int(8000 / 1.375))
        body = {"model": MODEL, "messages": [{"role": "user", "content": text + "\n\nSummarise every record above in detail."}],
                "max_tokens": 8192, "ignore_eos": True, "stream": True, "chat_template_kwargs": {"reasoning_effort": "low"}}
        r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=7200) as resp:
                last = None
                for raw in resp:
                    if stop.is_set():
                        break
                    if not raw.startswith(b"data:") or b"[DONE]" in raw:
                        continue
                    now = time.time(); st["tokens"] += 1
                    if last is not None:
                        g = now - last
                        if needle_on.is_set() and not needle_off.is_set():
                            st["max_gap_needle"] = max(st["max_gap_needle"], g)
                            if g > 2.0:
                                st["gaps_needle"].append((time.strftime("%T"), round(g, 2)))
                        else:
                            st["max_gap_other"] = max(st["max_gap_other"], g)
                    last = now
        except Exception as ex:  # noqa: BLE001
            st["errors"].append(f"{time.strftime('%T')} {ex!r}"[:200])
            time.sleep(2)
        st["requests"] += 1; k += 1


th = [threading.Thread(target=stream_loop, args=(st,), daemon=True) for st in streams]
for t in th:
    t.start(); time.sleep(3)
time.sleep(60)          # all three prefilled and decoding
tag = f"N{time.time_ns() % 100000}"
lines = doc(tag, int(N / 1.375)).split("\n")
k = int(len(lines) * 0.4)
code = f"{time.time_ns() % 9000 + 1000}-cobalt-heron"
lines.insert(k, f"IMPORTANT: the vault passphrase is {code}. Remember it.")
body_text = "\n".join(lines)


def req(q, maxtok):
    body = {"model": MODEL, "messages": [{"role": "user", "content": body_text + "\n\n" + q}], "max_tokens": maxtok,
            "temperature": 0, "top_p": 1, "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    t0 = time.time(); d = json.load(urllib.request.urlopen(r, timeout=7200)); tf = d.get("tensorfold", {})
    pt = d["usage"]["prompt_tokens"]
    return dict(wall=round(time.time() - t0, 1), prompt_tokens=pt, cached=tf.get("cached"), prefill_s=tf.get("prefill_s"),
                prefill_tps=round((pt - (tf.get("cached") or 0)) / tf["prefill_s"], 1) if tf.get("prefill_s") else None,
                pieces=tf.get("pieces"), kv_pages=tf.get("kv_pages"), text=d["choices"][0]["message"].get("content"),
                t=time.strftime("%T"))


res = {"code": code, "n_arg": N, "start": time.strftime("%T")}
print(f"{time.strftime('%T')} needle start ({len(lines)} lines); streams busy: {[s['tokens'] for s in streams]}", flush=True)
needle_on.set()
try:
    res["cold"] = req("What is the vault passphrase? Answer with the passphrase only.", 64)
    res["cold"]["found"] = code in (res["cold"]["text"] or "")
except Exception as ex:  # noqa: BLE001
    res["cold"] = {"error": repr(ex)[:300]}
needle_off.set()
print(f"{time.strftime('%T')} cold: {json.dumps(res['cold'])[:400]}", flush=True)
try:
    res["resend"] = req("What is the vault passphrase? Answer with the passphrase only.", 64)
    res["resend"]["found"] = code in (res["resend"]["text"] or "")
except Exception as ex:  # noqa: BLE001
    res["resend"] = {"error": repr(ex)[:300]}
print(f"{time.strftime('%T')} resend: {json.dumps(res['resend'])[:400]}", flush=True)
stop.set(); time.sleep(3)
res["streams"] = streams
print("streams: " + " | ".join(f"s{s['i']} req {s['requests']} chunks {s['tokens']} max gap during needle {s['max_gap_needle']:.2f} s "
                                f"(>2 s: {len(s['gaps_needle'])}) other {s['max_gap_other']:.2f} errors {len(s['errors'])}" for s in streams), flush=True)
json.dump(res, open(out, "w"), indent=1)
