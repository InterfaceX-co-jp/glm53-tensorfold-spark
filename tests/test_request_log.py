"""patches/0300: the per-request log (GLM53_TF_REQUEST_LOG, ``reqlog.py``) and ``scripts/traffic-report.py``.

Host only (no torch, no GPU). Run against the patched tree:
PYTHONPATH=<tree>/src pytest -q tests/test_request_log.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

reqlog = pytest.importorskip("tensorfold.families.glm5_next.cuda.reqlog")

ROOT = Path(__file__).resolve().parents[1]
SYS, USER, ASSIST, OBS = 150001, 150002, 150003, 150004
ROLES = {"user": USER, "assistant": ASSIST, "observation": OBS}


def _report_mod():
    spec = importlib.util.spec_from_file_location("traffic_report", ROOT / "scripts" / "traffic-report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _prompt(system: list[int], user: list[int], history: list[int] = ()) -> list[int]:
    return [SYS] + system + [USER] + user + [ASSIST] + list(history)


def _result(prompt: int, *, cached=0, src=None, finish="stop", tokens=40, **extra) -> dict:
    stats = {"cached": cached, "prefill_s": 2.0, "decode_s": 1.0, "rounds": 20, "tokens_per_round": 2.0,
             "queued_s": 0.01, "slot": 1, "kv_pages": 9, "kv_free": 100, "marks": 1, "policy": "auto",
             "sessions": "3 entries", **extra}
    if src == "ram":
        stats["restored"] = 7
    if src == "disk":
        stats["restored_disk"] = 8
        stats["disk"] = {"ok": True}
    return {"finish": finish, "completion_tokens": tokens, "prompt_tokens": prompt, "stats": stats,
            "content": "SECRET-ANSWER", "reasoning": "SECRET-THOUGHT"}


def _log(tmp_path, **kw) -> "reqlog.RequestLog":
    return reqlog.RequestLog(str(tmp_path / "sub" / "requests.jsonl"), roles=ROLES, **kw)


def _lines(path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def test_off_by_default_and_settings(monkeypatch):
    for k in list(os.environ):
        if k.startswith("GLM53_TF_REQUEST_LOG"):
            monkeypatch.delenv(k)
    assert reqlog.settings() is None and reqlog.RequestLog.from_env() is None
    monkeypatch.setenv("GLM53_TF_REQUEST_LOG", "0")
    assert reqlog.settings() is None
    monkeypatch.setenv("GLM53_TF_REQUEST_LOG", "/sessions/requests.jsonl")
    s = reqlog.settings()
    assert s["path"] == "/sessions/requests.jsonl" and s["max_bytes"] == 64 << 20 and s["keep"] == 3
    assert s["window"] == 64 and s["salt"] == b""
    monkeypatch.setenv("GLM53_TF_REQUEST_LOG_MB", "0")
    with pytest.raises(ValueError):
        reqlog.settings()


def test_role_ids_from_a_tokenizer():
    tok = SimpleNamespace(token_to_id=lambda t: {"<|user|>": 5, "<|assistant|>": 6}.get(t))
    assert reqlog.role_ids(tok) == {"user": 5, "assistant": 6}


def test_common_prefix_and_first_of():
    from array import array

    a = array("i", range(10000))
    b = array("i", list(range(7777)) + [-1] * 50)
    assert reqlog.common(a, b) == 7777 and reqlog.common(a, a) == 10000 and reqlog.common(a, array("i")) == 0
    assert reqlog.first_of(array("i", [1, 2, 3, 2, 4]), [4, 2]) == 1
    assert reqlog.first_of(array("i", [1, 2]), [7]) == -1


def test_record_fields_sources_and_prefixes(tmp_path):
    log = _log(tmp_path)
    system = list(range(100, 5100))                         # a 5,000-token system prompt + tools
    other_system = list(range(5000, 5500))
    a1 = _prompt(system, [1, 2, 3])
    body = {"messages": [{"role": "user", "content": "SECRET-PROMPT"}], "tools": [{"x": 1}] * 3,
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"}, "max_tokens": 500,
            "reasoning_effort": "high"}
    t = log.begin(a1)
    log.first(t)
    r1 = log.end(t, body=body, chat=True, result=_result(len(a1)), error=None, thinking=True, max_tokens_eff=500)
    # the next turn of conversation A: it extends A's first prompt
    a2 = a1 + [11, 12, USER, 13, ASSIST]
    t = log.begin(a2)
    r2 = log.end(t, body=body, chat=True, result=_result(len(a2), cached=len(a1) - 5, src="slot"), error=None,
                 thinking=True, max_tokens_eff=500)
    # a new conversation B over the same system prompt: shares it with A (another conversation)
    b1 = _prompt(system, [4, 5])
    t = log.begin(b1)
    r3 = log.end(t, body=body, chat=True, result=_result(len(b1), src="ram", cached=2048), error=None,
                 thinking=True, max_tokens_eff=500)
    # an unrelated one, served from disk
    c1 = _prompt(other_system, [6])
    t = log.begin(c1)
    r4 = log.end(t, body={"prompt": "x"}, chat=False, result=_result(len(c1), src="disk", cached=256,
                                                                        finish="length"),
                 error=None, thinking=None, max_tokens_eff=4096)
    got = _lines(log.path)
    assert got == [r1, r2, r3, r4]
    assert [r["n"] for r in got] == [1, 2, 3, 4]
    assert r1["prompt"] == len(a1) and r1["cached"] == 0 and r1["cache_src"] == "none" and r1["lcp"] == 0
    assert r1["sys_len"] == 5001 and r1["head_len"] == 4096 and r1["tools"] == 3 and r1["messages"] == 1
    assert r1["thinking"] is True and r1["effort"] == "high" and r1["max_tokens"] == 500
    assert r1["finish"] == "stop" and r1["error"] is None and r1["first_s"] is not None
    assert r1["decode_tps"] == 40.0 and r1["prefill_tps"] == round(len(a1) / 2.0, 1)
    assert r1["kv_pages"] == 9 and r1["kv_free"] == 100 and r1["marks"] == 1 and r1["queue_s"] == 0.01
    assert r2["cache_src"] == "slot" and r2["conv"] == r1["conv"]
    assert r2["lcp"] == r2["lcp_same"] == len(a1) and r2["lcp_other"] == 0
    assert r3["cache_src"] == "ram" and r3["conv"] != r1["conv"] and r3["sys_hash"] == r1["sys_hash"]
    assert r3["head_hash"] == r1["head_hash"]                 # the first 4k tokens are the same
    assert r3["lcp_other"] == 5002 and r3["lcp_same"] == 0      # system prompt + <|user|>
    assert r4["cache_src"] == "disk" and r4["kind"] == "completion" and r4["lcp"] == 1   # <|system|>
    assert r4["finish"] == "length" and r4["thinking"] is None and r4["effort"] is None
    raw = Path(log.path).read_text()
    for secret in ("SECRET-PROMPT", "SECRET-ANSWER", "SECRET-THOUGHT", "3 entries"):
        assert secret not in raw                              # no text; not even the store's description


def test_head_hash_covers_4k_tokens_and_salt(tmp_path):
    log = _log(tmp_path)
    base = list(range(20, 5000))
    x = log.analyse(log.begin(base + [1]))
    y = log.analyse(log.begin(base[:4096] + [7] * 900 + [2]))
    z = log.analyse(log.begin([3] + base))
    assert x["head_hash"] == y["head_hash"] != z["head_hash"] and x["head_len"] == 4096
    salted = reqlog.RequestLog(str(tmp_path / "s.jsonl"), roles=ROLES, salt=b"k")
    assert salted.analyse(salted.begin(base + [1]))["head_hash"] != x["head_hash"]


def test_in_flight_requests_count_as_partners(tmp_path):
    """Two sessions arriving together: the later one sees the earlier one's prompt while it still runs."""

    log = _log(tmp_path)
    system = list(range(100, 2100))
    ta = log.begin(_prompt(system, [1]))
    tb = log.begin(_prompt(system, [2]))
    rb = log.end(tb, body={}, chat=True, result=_result(10), error=None, thinking=False, max_tokens_eff=1)
    ra = log.end(ta, body={}, chat=True, result=_result(10), error=None, thinking=False, max_tokens_eff=1)
    assert rb["lcp_other"] == 2002 and ra["lcp"] == 0          # A arrived first: nothing before it


def test_window_bounds_the_kept_prompts(tmp_path):
    log = _log(tmp_path, window=2)
    for i in range(4):
        log.end(log.begin([i] * 300), body={}, chat=False, result=_result(300), error=None, thinking=None,
                max_tokens_eff=1)
    assert len(log.prefixes.kept) == 2
    r = log.end(log.begin([0] * 300), body={}, chat=False, result=_result(300), error=None, thinking=None,
                max_tokens_eff=1)
    assert r["lcp"] == 0                                        # prompt 0 fell out of the window


def test_errors_and_cancels_are_logged_and_logging_never_raises(tmp_path, capsys):
    log = _log(tmp_path)
    r = log.end(log.begin([1, 2, 3]), body={}, chat=True, result=None, error=RuntimeError("boom"), thinking=True,
                max_tokens_eff=5)
    assert r["finish"] == "error" and r["error"] == "RuntimeError" and r["decode_tokens"] is None
    r = log.end(log.begin([1, 2, 3]), body={}, chat=True, result=_result(3, cancelled=True), error=None,
                thinking=True, max_tokens_eff=5)
    assert r["finish"] == "cancelled"
    blocked = tmp_path / "file"
    blocked.write_text("")
    bad = reqlog.RequestLog(str(blocked / "x.jsonl"), roles=ROLES)       # a file where a directory should be
    assert bad.end(bad.begin([1]), body={}, chat=True, result=_result(1), error=None, thinking=None,
                   max_tokens_eff=1) is None
    assert bad.end(bad.begin([1]), body={}, chat=True, result=_result(1), error=None, thinking=None,
                   max_tokens_eff=1) is None
    assert capsys.readouterr().err.count("request log") == 1           # reported once
    assert not bad.live                                                # nothing leaks


def test_rotation(tmp_path):
    log = _log(tmp_path, max_bytes=4000, keep=2)
    for i in range(40):
        log.end(log.begin([i] * 10), body={}, chat=False, result=_result(10), error=None, thinking=None,
                max_tokens_eff=1)
    p = Path(log.path)
    assert p.exists() and Path(f"{p}.1").exists() and Path(f"{p}.2").exists() and not Path(f"{p}.3").exists()
    for f in (p, Path(f"{p}.1"), Path(f"{p}.2")):
        assert f.stat().st_size <= 4000
        _lines(f)                                                       # whole lines only
    mod = _report_mod()
    files = mod.files_of(str(p))
    assert files == [f"{p}.2", f"{p}.1", str(p)]
    recs, bad = mod.load(files)
    assert bad == 0 and [r["n"] for r in recs] == sorted(r["n"] for r in recs) and recs[-1]["n"] == 40


def test_concurrent_writers(tmp_path):
    log = _log(tmp_path)

    def work(k):
        for i in range(50):
            log.end(log.begin([k, i] * 50), body={}, chat=False, result=_result(100), error=None, thinking=None,
                    max_tokens_eff=1)

    th = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    [t.start() for t in th]
    [t.join() for t in th]
    got = _lines(log.path)
    assert len(got) == 200 and sorted(r["n"] for r in got) == list(range(1, 201)) and not log.live


def test_glm_app_writes_one_line_per_request(tmp_path, monkeypatch):
    """``GlmApp.run`` / ``prompt_ids`` wiring: the ticket starts when the prompt is known, the line is written after
    the reply (and after a failure, which still raises)."""

    app_mod = pytest.importorskip("tensorfold.families.glm5_next.cuda.app")
    from tensorfold.cuda.server import App

    calls = []

    def fake_run(self, body, chat, emit):
        ids = self.prompt_ids("rendered")
        emit({"content": "x"})
        calls.append(ids)
        if body.get("fail"):
            raise ValueError("bad request")
        return _result(len(ids), cached=3, src="ram", finish="tool_calls")

    monkeypatch.setattr(App, "run", fake_run)
    app = object.__new__(app_mod.GlmApp)
    app.effort_field, app.default_effort, app.default_thinking, app.max_tokens = True, "high", True, 4096
    app.engine = SimpleNamespace(request=threading.local())
    app.prompt_memo = app_mod.Memo()
    app.prompt_tokens = SimpleNamespace(encode=lambda text: [SYS, 1, 2, USER, 3, ASSIST])
    app.reqlog = _log(tmp_path)
    app._rl = threading.local()
    out = app.run({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "low"}, True,
                  lambda d: True)
    assert out["finish"] == "tool_calls"
    with pytest.raises(ValueError):
        app.run({"messages": [], "fail": True}, True, lambda d: True)
    got = _lines(app.reqlog.path)
    assert len(got) == 2 and got[0]["finish"] == "tool_calls" and got[0]["cache_src"] == "ram"
    assert got[0]["thinking"] is True and got[0]["effort"] == "low" and got[0]["effort_asked"] == "low"
    assert got[0]["max_tokens_eff"] == 4096 and got[0]["prompt"] == 6 and got[0]["sys_len"] == 3
    assert got[1]["finish"] == "error" and got[1]["error"] == "ValueError" and got[1]["lcp"] == 6
    # off: no log object, nothing changes
    app.reqlog = None
    assert app.run({"messages": []}, True, lambda d: True)["finish"] == "tool_calls"


def test_traffic_report(tmp_path):
    log = _log(tmp_path)
    system = list(range(100, 12100))                        # a 12k agent system prompt, 6 sessions
    for k in range(6):
        p = _prompt(system, [k] * 200)
        cached = 0 if k < 2 else 12002 // 64 * 64               # the 3rd session on resumes at the fork mark
        res = _result(len(p), cached=cached, src="ram" if cached else None)
        res["stats"]["prefill_s"] = (len(p) - cached) / 1000.0
        log.end(log.begin(p), body={}, chat=True, result=res, error=None, thinking=True, max_tokens_eff=100)
        p2 = p + [7] * 300 + [USER, 1, ASSIST]
        res = _result(len(p2), cached=len(p) - 1, src="slot")
        res["stats"]["prefill_s"] = (len(p2) - len(p) + 1) / 1000.0
        log.end(log.begin(p2), body={}, chat=True, result=res, error=None, thinking=True, max_tokens_eff=100)
    mod = _report_mod()
    recs, bad = mod.load([log.path])
    rep = mod.report(recs)
    assert rep["requests"] == 12 and bad == 0
    # session 2 (the first sharer) could have skipped floor64(lcp_other) tokens: 12,002 shared (<|system|>, the
    # 12,000-token system prompt, <|user|>) -> 11,968 on the 64-grid
    head = rep["avoidable_prefill"]["other_conversation"]
    assert head["requests"] == round(1 / 12, 4) and head["tokens"] == 12002 // 64 * 64
    total = sum(r["prefill_s"] for r in recs)
    assert head["prefill_share"] == round((12002 // 64 * 64) / 1000.0 / total, 4)
    assert rep["reuse"]["by_source"]["ram"]["requests"] == 4 and rep["reuse"]["by_source"]["slot"]["requests"] == 6
    assert rep["new_conversations"]["requests"] == 6
    assert rep["shared_prefixes"]["sys_hash"][0]["requests"] == 12
    assert rep["shared_prefixes"]["sys_hash"][0]["conversations"] == 6
    txt = mod.text(rep)
    assert "avoidable prefill" in txt and "other_conversation" in txt
    assert mod.main([log.path]) == 0 and mod.main([log.path, "--json"]) == 0
