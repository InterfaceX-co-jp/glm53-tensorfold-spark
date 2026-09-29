#!/usr/bin/env python3
"""W14: image-input checks against a live TensorFold server with GLM53_TF_VISION=1 (docs/VISION.md §7 steps 3-6).

    vis.py correct OUT.json   (a)-(f): describe, read text, count, two images, transparency, URL vs data:, + 400 paths
    vis.py cache   OUT.json   same conversation twice; image A then image B with the same text; a 3-turn conversation
    vis.py replay  OUT.json CACHE.json   after a restart: the 3-turn conversation's turn 3 again (NVMe restore)
    vis.py fresh   OUT.json CACHE.json   on a fresh session tier: B alone first, the conversation first (== CACHE's)
    vis.py ttft    OUT.json   TTFT per image size (3 unique images a size, cold), stats.vision split
    vis.py probe   OUT.json   one small image request (memory check after the stress)

Images: IMG (default results/W14/img, from mkimg.py). Server: BASE (default http://127.0.0.1:8000). Greedy
(temperature 0), reasoning_effort low unless stated. Stdlib only.
"""
import base64, hashlib, http.server, json, os, re, socketserver, struct, sys, threading, time, urllib.error, urllib.request, zlib
from pathlib import Path

BASE = os.environ.get("BASE", "http://127.0.0.1:8000")
IMG = Path(os.environ.get("IMG", str(Path(__file__).resolve().parent / "img")))
MODEL = "GLM-5.3-Flash-EXL3"
IMAGE_TOKEN = 154854
truth = json.loads((IMG / "truth.json").read_text()) if (IMG / "truth.json").exists() else {}


def data_url(name):
    p = IMG / name
    mime = "image/jpeg" if p.suffix == ".jpg" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def img(url):
    return {"type": "image_url", "image_url": {"url": url}}


def txt(t):
    return {"type": "text", "text": t}


def post(path, body, timeout=1800):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(r, timeout=timeout) as f:
            return f.status, json.load(f), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), time.time() - t0


def chat(content_or_messages, effort="low", max_tokens=4096, thinking=True, **extra):
    msgs = content_or_messages if isinstance(content_or_messages, list) and content_or_messages and \
        isinstance(content_or_messages[0], dict) and "role" in content_or_messages[0] else \
        [{"role": "user", "content": content_or_messages}]
    body = {"model": MODEL, "messages": msgs, "max_tokens": max_tokens, "temperature": 0, "top_p": 1}
    kw = {"reasoning_effort": effort} if thinking else {"enable_thinking": False}
    body["chat_template_kwargs"] = kw
    body.update(extra)
    code, d, wall = post("/v1/chat/completions", body)
    if code != 200:
        return dict(code=code, error=d.get("error", d), wall=round(wall, 2))
    m = d["choices"][0]["message"]; tf = d.get("tensorfold", {})
    content = m.get("content") or ""
    return dict(code=code, content=content, reasoning_chars=len(m.get("reasoning_content") or m.get("reasoning") or ""),
                finish=d["choices"][0].get("finish_reason"), prompt_tokens=d["usage"]["prompt_tokens"],
                completion_tokens=d["usage"]["completion_tokens"], cached=tf.get("cached"), prefill_s=tf.get("prefill_s"),
                vision=tf.get("vision"), restored=tf.get("restored"), restored_disk=tf.get("restored_disk"),
                disk=tf.get("disk"), sha=hashlib.sha256(content.encode()).hexdigest()[:16],
                full_sha=hashlib.sha256(json.dumps([content, m.get("reasoning_content")]).encode()).hexdigest()[:16],
                wall=round(wall, 2), body=body)


def tokenize(messages):
    code, d, _ = post("/tokenize", {"model": MODEL, "messages": messages,
                                    "chat_template_kwargs": {"reasoning_effort": "low"}})
    return d.get("tokens", []) if code == 200 else []


def cer(ref, hyp):
    """character error rate: Levenshtein(ref, hyp) / len(ref), whitespace runs collapsed"""
    a, b = re.sub(r"\s+", " ", ref).strip(), re.sub(r"\s+", " ", hyp).strip()
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return round(prev[-1] / max(1, len(a)), 4)


def strip_fence(s):
    m = re.search(r"```[a-z]*\n(.*?)```", s, re.S)
    return m.group(1) if m else s


def first_int(s):
    m = re.search(r"\b(\d+)\b", s)
    return int(m.group(1)) if m else None


def log(res, name, r, **check):
    r = dict(r); r.pop("body", None); r.update(check); r["name"] = name
    res.append(r)
    short = (r.get("content") or str(r.get("error")))[:160].replace("\n", " | ")
    print(f"{time.strftime('%T')} {name:28s} code {r['code']} pt {r.get('prompt_tokens')} cached {r.get('cached')} "
          f"vis {r.get('vision')} {' '.join(f'{k}={v}' for k, v in check.items())} :: {short}", flush=True)


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve_dir(port=18765):
    h = lambda *a, **k: Quiet(*a, directory=str(IMG), **k)
    s = socketserver.TCPServer(("127.0.0.1", port), h)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


def big_header_png(w, h):
    def chunk(t, data):
        return struct.pack(">I", len(data)) + t + data + struct.pack(">I", zlib.crc32(t + data) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\0" * 64)) + chunk(b"IEND", b""))


def correct(out):
    res = []
    for eff in ("low",):
        r = chat([img(data_url("desk_1920x1080.png")), txt("Describe this screenshot. Which application windows are open, "
                  "and what are their exact window titles? What number does the calculator display, and what time does the clock show?")], eff)
        c = r.get("content", "")
        log(res, f"a-desk-{eff}", r, titles=all(t.lower() in c.lower() for t in truth["desk"]["titles"]),
            display="391" in c, clock="14:32" in c)
        r = chat([img(data_url("scene_1024x768.jpg")), txt("Describe this picture. List the objects you see.")], eff)
        c = r.get("content", "").lower()
        log(res, f"a-scene-{eff}", r, objects={o: o in c for o in truth["scene"]["objects"]})
        r = chat([img(data_url("receipt.png")), txt("What is the TOTAL on this receipt? List the items too.")], eff)
        c = r.get("content", "")
        log(res, f"b-receipt-{eff}", r, total="22.50" in c, items=sum(i.lower() in c.lower() for i in truth["receipt"]["items"]))
        r = chat([img(data_url("para_12px.png")), txt("Transcribe the text in this image exactly. Output only the text.")], eff)
        log(res, f"b-para12-{eff}", r, cer=cer(truth["para"], strip_fence(r.get("content", ""))))
    for eff in ("low", "max"):
        r = chat([img(data_url("term_1024x768.png")), txt("Transcribe every line of this terminal screenshot exactly, "
                  "one line per line, in a code block. Do not include the window title.")], eff, max_tokens=12000)
        log(res, f"b-term-{eff}", r, cer=cer(truth["term"], strip_fence(r.get("content", ""))))
        for n in (3, 7, 12):
            r = chat([img(data_url(f"circles_{n}.png")), txt("How many circles are in this image? Answer with the number only.")],
                     eff, max_tokens=12000)
            log(res, f"c-circles{n}-{eff}", r, want=n, got=first_int(r.get("content", "")), ok=first_int(r.get("content", "")) == n)
        r = chat([img(data_url("grid_5x4.png")), txt("How many stars are in this image? Answer with the number only.")],
                 eff, max_tokens=12000)
        log(res, f"c-grid-{eff}", r, want=20, got=first_int(r.get("content", "")), ok=first_int(r.get("content", "")) == 20)
    # (d) two images: which value is larger; image-first vs text-first; both orders of the charts
    q = "The first image and the second image are bar charts. Which one shows the larger value, the first or the second? Answer 'first' or 'second', then give both values."
    for order in (("chart_30.png", "chart_75.png"), ("chart_75.png", "chart_30.png")):
        want = "second" if order[0] == "chart_30.png" else "first"
        for layout in ("images-first", "text-first"):
            parts = [img(data_url(order[0])), img(data_url(order[1]))]
            content = parts + [txt(q)] if layout == "images-first" else [txt(q)] + parts
            r = chat(content)
            c = r.get("content", "").lower()
            got = "first" if c.find("first") != -1 and (c.find("second") == -1 or c.find("first") < c.find("second")) else "second"
            log(res, f"d-{order[0][6:8]}-{order[1][6:8]}-{layout}", r, want=want, got=got, ok=got == want,
                values="30" in c and "75" in c)
    # (e) transparency
    r = chat([img(data_url("alpha_text.png")), txt("What text is written in this image? Output only the text.")])
    log(res, "e-alpha", r, ok=truth["alpha"] in r.get("content", ""), cer=cer(truth["alpha"], strip_fence(r.get("content", ""))))
    # (f) URL (local http.server on the head; rank 0 shares the host network) vs data: -> same prompt, same reply
    srv = serve_dir()
    try:
        q = "Transcribe every line of this terminal screenshot exactly, one line per line, in a code block. Do not include the window title."
        ru = chat([img("http://127.0.0.1:18765/term_1024x768.png"), txt(q)])
        rd = chat([img(data_url("term_1024x768.png")), txt(q)])
        log(res, "f-url", ru, cer=cer(truth["term"], strip_fence(ru.get("content", ""))))
        log(res, "f-data", rd, same_reply_as_url=rd.get("full_sha") == ru.get("full_sha"),
            same_prompt_tokens=rd.get("prompt_tokens") == ru.get("prompt_tokens"))
        # a small image fetched by URL, bare-string image_url shape
        r = chat([{"type": "image_url", "image_url": "http://127.0.0.1:18765/circles_3.png"}, txt("How many circles? Number only.")])
        log(res, "f-url-bare", r, got=first_int(r.get("content", "")), ok=first_int(r.get("content", "")) == 3)
    finally:
        srv.shutdown()
    # step 6: failure paths are 400s, then a text request is served
    big = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n" + os.urandom(30 << 20)).decode()
    log(res, "x-30MB", chat([img(big), txt("hi")], max_tokens=16))
    del big
    log(res, "x-20kx20k", chat([img("data:image/png;base64," + base64.b64encode(big_header_png(20000, 20000)).decode()), txt("hi")], max_tokens=16))
    log(res, "x-dead-url", chat([img("http://127.0.0.1:1/none.png"), txt("hi")], max_tokens=16))
    log(res, "x-9-images", chat([img(data_url("circles_3.png")) for _ in range(9)] + [txt("hi")], max_tokens=16))
    log(res, "x-file-scheme", chat([img("file:///etc/passwd"), txt("hi")], max_tokens=16))
    r = chat("What is 17*23? Answer with the number only.", thinking=False, max_tokens=32)
    log(res, "x-after-text", r, ok="391" in r.get("content", ""))
    res.append({"name": "health", "health": json.load(urllib.request.urlopen(BASE + "/health", timeout=10))})
    print("health", res[-1]["health"], flush=True)
    json.dump(res, open(out, "w"), indent=1)


def cache(out):
    res = {}
    nonce = f"[w14-{time.time_ns()}] "
    conv = [{"role": "user", "content": [txt(nonce), img(data_url("cache_0.png")),
                                         txt("What is the project name written in this image? Answer in one short sentence.")]}]
    r1 = chat(conv); r2 = chat(conv)
    ids = tokenize(conv)
    before = ids.index(IMAGE_TOKEN) if IMAGE_TOKEN in ids else None
    res["same"] = dict(first={k: r1.get(k) for k in ("prompt_tokens", "cached", "vision", "full_sha", "content")},
                       second={k: r2.get(k) for k in ("prompt_tokens", "cached", "vision", "full_sha", "content")},
                       text_before_image=before, identical=r1.get("full_sha") == r2.get("full_sha"))
    print("same conversation twice:", json.dumps({k: v for k, v in res["same"].items()}, default=str)[:600], flush=True)
    # A then B: same text, different image
    nonce2 = f"[w14-pair-{time.time_ns()}]"
    tail = "Describe this image in one sentence and quote any text in it."
    ca = [{"role": "user", "content": [txt(nonce2), img(data_url("cache_0.png")), txt(tail)]}]
    cb = [{"role": "user", "content": [txt(nonce2), img(data_url("cache_1.png")), txt(tail)]}]
    ra = chat(ca); rb = chat(cb)
    ids = tokenize(cb); before = ids.index(IMAGE_TOKEN)
    res["pair"] = dict(nonce=nonce2, a={k: ra.get(k) for k in ("prompt_tokens", "cached", "vision", "full_sha", "content")},
                       b={k: rb.get(k) for k in ("prompt_tokens", "cached", "vision", "full_sha", "content")},
                       text_before_image=before, b_cached_le_text=(rb.get("cached") or 0) <= before,
                       b_body=rb["body"])
    print("A then B:", json.dumps({k: v for k, v in res["pair"].items() if k != "b_body"}, default=str)[:700], flush=True)
    # 3 turns, image in turn 1
    nonce3 = f"[w14-conv-{time.time_ns()}]"
    msgs = [{"role": "user", "content": [txt(nonce3), img(data_url("receipt.png")), txt("What shop is this receipt from?")]}]
    turns = []
    for k, follow in enumerate(["What is the most expensive item on it?", "What is the total? Number only."]):
        r = chat(msgs); turns.append({x: r.get(x) for x in ("prompt_tokens", "cached", "vision", "full_sha", "content", "restored")})
        msgs = msgs + [{"role": "assistant", "content": r.get("content", "")}, {"role": "user", "content": follow}]
    r = chat(msgs); turns.append({x: r.get(x) for x in ("prompt_tokens", "cached", "vision", "full_sha", "content", "restored")})
    res["turns"] = turns; res["turn3_body"] = r["body"]
    for i, t in enumerate(turns):
        print(f"turn {i + 1}:", json.dumps(t)[:400], flush=True)
    json.dump(res, open(out, "w"), indent=1)


def replay(out, cache_file):
    c = json.load(open(cache_file))
    code, d, wall = post("/v1/chat/completions", c["turn3_body"])
    m = d["choices"][0]["message"]; tf = d.get("tensorfold", {})
    sha = hashlib.sha256(json.dumps([m.get("content") or "", m.get("reasoning_content")]).encode()).hexdigest()[:16]
    r = dict(code=code, cached=tf.get("cached"), restored=tf.get("restored"), restored_disk=tf.get("restored_disk"),
             disk=tf.get("disk"), vision=tf.get("vision"), prompt_tokens=d["usage"]["prompt_tokens"], full_sha=sha,
             identical=sha == c["turns"][-1]["full_sha"], wall=round(wall, 2), content=m.get("content"))
    print("replay turn 3:", json.dumps(r)[:600], flush=True)
    json.dump(r, open(out, "w"), indent=1)


def fresh(out, cache_file):
    c = json.load(open(cache_file))
    code, d, wall = post("/v1/chat/completions", c["pair"]["b_body"])
    m = d["choices"][0]["message"]; tf = d.get("tensorfold", {})
    sha = hashlib.sha256(json.dumps([m.get("content") or "", m.get("reasoning_content")]).encode()).hexdigest()[:16]
    r = dict(code=code, cached=tf.get("cached"), prompt_tokens=d["usage"]["prompt_tokens"], full_sha=sha,
             b_after_a_sha=c["pair"]["b"]["full_sha"], identical=sha == c["pair"]["b"]["full_sha"], vision=tf.get("vision"))
    print("B fresh vs B after A:", json.dumps(r), flush=True)
    json.dump(r, open(out, "w"), indent=1)


def ttft(out):
    res = []
    for size in ("512x512", "1024x768", "1920x1080", "3840x2160"):
        for v in range(3):
            body = {"model": MODEL, "max_tokens": 8, "temperature": 0, "stream": True,
                    "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": [txt(f"[w14-ttft-{time.time_ns()}]"),
                                                              img(data_url(f"size_{size}_{v}.png")),
                                                              txt("What is in this picture? One word.")]}]}
            r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                       headers={"Content-Type": "application/json"})
            t0 = time.time(); first = None; usage = None; tf = None
            with urllib.request.urlopen(r, timeout=600) as f:
                for line in f:
                    line = line.decode().strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    ev = json.loads(line[6:])
                    if first is None and ev.get("choices") and (ev["choices"][0].get("delta") or {}).get("content"):
                        first = time.time() - t0
                    usage = ev.get("usage") or usage; tf = ev.get("tensorfold") or tf
            row = dict(size=size, v=v, ttft=round(first, 3) if first else None, prompt_tokens=(usage or {}).get("prompt_tokens"),
                       vision=(tf or {}).get("vision"), prefill_s=(tf or {}).get("prefill_s"), cached=(tf or {}).get("cached"))
            res.append(row)
            print(time.strftime("%T"), json.dumps(row), flush=True)
    json.dump(res, open(out, "w"), indent=1)


def probe(out):
    r = chat([img(data_url("circles_7.png")), txt("How many circles? Number only.")], max_tokens=4096)
    r.pop("body", None)
    print("probe:", json.dumps(r)[:400], flush=True)
    json.dump(r, open(out, "w"), indent=1)


PAIR2_TEXT = "[w14-pair2-v1] Background notes for the review:\n" + "".join(
    f"{i}. The quarterly inventory audit for warehouse section {i} found the shelves labelled and the counts matched.\n"
    for i in range(1, 41))


def pair2_body(k):
    return {"model": MODEL, "max_tokens": 4096, "temperature": 0, "top_p": 1,
            "chat_template_kwargs": {"reasoning_effort": "low"},
            "messages": [{"role": "user", "content": [txt(PAIR2_TEXT), img(data_url(f"cache_{k}.png")),
                                                     txt("Which project is named in the image? One sentence.")]}]}


def run_body(body):
    code, d, wall = post("/v1/chat/completions", body)
    m = d["choices"][0]["message"]; tf = d.get("tensorfold", {})
    sha = hashlib.sha256(json.dumps([m.get("content") or "", m.get("reasoning_content")]).encode()).hexdigest()[:16]
    return dict(code=code, cached=tf.get("cached"), restored=tf.get("restored"), restored_disk=tf.get("restored_disk"),
                disk=tf.get("disk"), vision=tf.get("vision"), prompt_tokens=d["usage"]["prompt_tokens"], full_sha=sha,
                wall=round(wall, 2), content=m.get("content"))


def pairfresh(out):
    ids = tokenize(pair2_body(1)["messages"])
    r = run_body(pair2_body(1)); r["text_before_image"] = ids.index(IMAGE_TOKEN)
    print("B2 fresh:", json.dumps(r), flush=True)
    json.dump(r, open(out, "w"), indent=1)


def pairab(out, fresh_file):
    f = json.load(open(fresh_file))
    ra = run_body(pair2_body(0)); rb = run_body(pair2_body(1))
    r = dict(a=ra, b=rb, text_before_image=f["text_before_image"], b_fresh_sha=f["full_sha"],
             b_identical_to_fresh=rb["full_sha"] == f["full_sha"],
             b_cached_in_text=0 < (rb["cached"] or 0) <= f["text_before_image"])
    print("A2 then B2:", json.dumps(r), flush=True)
    json.dump(r, open(out, "w"), indent=1)


def disk_body():
    return {"model": MODEL, "max_tokens": 4096, "temperature": 0, "top_p": 1,
            "chat_template_kwargs": {"reasoning_effort": "low"},
            "messages": [{"role": "user", "content": [txt("[w14-disk-v1] Here is a screenshot of my desktop."),
                                                     img(data_url("desk_1920x1080.png")),
                                                     txt("Which windows are open, and what does the note in the editor say?")]}]}


def disk1(out):
    r = run_body(disk_body())
    print("disk turn (before restart):", json.dumps(r), flush=True)
    json.dump(r, open(out, "w"), indent=1)


def disk2(out, first_file):
    f = json.load(open(first_file))
    r = run_body(disk_body()); r["identical"] = r["full_sha"] == f["full_sha"]
    print("disk turn (after restart):", json.dumps(r), flush=True)
    json.dump(r, open(out, "w"), indent=1)


def pixels(out):
    res = []
    for w, h in [(10000, 10000), (8000, 8000)]:
        body = {"model": MODEL, "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [img("data:image/png;base64," + base64.b64encode(big_header_png(w, h)).decode()), txt("hi")]}]}
        code, d, _ = post("/v1/chat/completions", body)
        res.append(dict(size=f"{w}x{h}", code=code, error=d.get("error")))
        print("pixels", res[-1], flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    mode, outf = sys.argv[1], sys.argv[2]
    {"correct": lambda: correct(outf), "cache": lambda: cache(outf), "ttft": lambda: ttft(outf), "probe": lambda: probe(outf),
     "replay": lambda: replay(outf, sys.argv[3]), "fresh": lambda: fresh(outf, sys.argv[3]),
     "pairfresh": lambda: pairfresh(outf), "pairab": lambda: pairab(outf, sys.argv[3]), "disk1": lambda: disk1(outf),
     "disk2": lambda: disk2(outf, sys.argv[3]), "pixels": lambda: pixels(outf)}[mode]()
