"""patches/0150: ``/health`` that can say no, ``/metrics``, JSON errors instead of dropped connections.

Host only (no GPU, no tokenizer): a fake app behind the real ``make_handler``. Run against the patched tree:

    PYTHONPATH=<tree>/src pytest -q tests/test_health.py
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

health_mod = pytest.importorskip("tensorfold.cuda.health")
from tensorfold.cuda.health import Health  # noqa: E402
from tensorfold.cuda.server import make_handler  # noqa: E402


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_basic_mode_stays_200_and_reports_fatal():
    h = Health(mode="basic", stall_s=0, prefill_tps=200, clock=Clock())
    rid = h.begin(10)
    h.end(rid, error=RuntimeError("NCCL error: unhandled system error"))
    code, body = h.status()
    assert code == 200 and body["ok"] is False and "NCCL" in body["fatal"]
    assert h.reject() is None                      # basic never refuses work


def test_strict_mode_503_after_fatal_and_refuses():
    h = Health(mode="strict", stall_s=0, prefill_tps=200, clock=Clock())
    assert h.status()[0] == 200
    rid = h.begin(10)
    h.end(rid, error=RuntimeError("CUDA error: an illegal memory access"))
    assert h.status()[0] == 503
    assert "restart both ranks" in h.reject()


def test_value_error_is_not_fatal():
    h = Health(mode="strict", stall_s=0, prefill_tps=200, clock=Clock())
    rid = h.begin(10)
    h.end(rid, error=ValueError("prompt too long"))
    code, body = h.status()
    assert code == 200 and body["ok"] and body["errors"] == 1 and "fatal" not in body


def test_stall_allows_prefill_time_then_counts_quiet_decode():
    clock = Clock()
    h = Health(mode="strict", stall_s=60, prefill_tps=100, clock=clock)
    rid = h.begin(30_000)                          # 60 s + 300 s allowed before the first token
    clock.t += 300
    assert h.status()[0] == 200
    clock.t += 61
    code, body = h.status()
    assert code == 503 and body["stalled"][0]["allowed_s"] == 360.0
    h.progress(rid, 1)                             # a token: the prefill allowance no longer applies
    assert h.status()[0] == 200
    clock.t += 59
    assert h.status()[0] == 200
    clock.t += 2
    assert h.status()[0] == 503
    h.end(rid, stats={"rounds": 10, "decode_s": 1.0, "prefill_s": 2.0, "cached": 5}, completion_tokens=40)
    assert h.status()[0] == 200


def test_stall_off_by_default(monkeypatch):
    monkeypatch.delenv("GLM53_TF_STALL_S", raising=False)
    monkeypatch.delenv("GLM53_TF_HEALTH", raising=False)
    clock = Clock()
    h = Health(clock=clock)
    assert h.mode == "basic" and h.stall_s == 0
    h.begin(10)
    clock.t += 1e6
    assert h.status()[1]["ok"]


def test_env_validation(monkeypatch):
    monkeypatch.setenv("GLM53_TF_HEALTH", "loud")
    with pytest.raises(ValueError):
        Health()
    monkeypatch.setenv("GLM53_TF_HEALTH", "strict")
    monkeypatch.setenv("GLM53_TF_STALL_S", "-1")
    with pytest.raises(ValueError):
        Health()


def test_metrics_counters_exact():
    h = Health(mode="basic", stall_s=0, prefill_tps=200, clock=Clock())
    for _ in range(3):
        rid = h.begin(1_234_567)
        h.end(rid, stats={"rounds": 7, "decode_s": 0.5, "prefill_s": 1.25, "cached": 100}, completion_tokens=29)
    text = h.metrics('GLM "x"')
    lines = {ln.split("{")[0]: ln.rsplit(" ", 1)[1] for ln in text.splitlines() if not ln.startswith("#")}
    assert lines["tensorfold_prompt_tokens_total"] == "3703701"      # no %g rounding of big counters
    assert lines["tensorfold_decode_rounds_total"] == "21"
    assert lines["tensorfold_completion_tokens_total"] == "87"
    assert lines["tensorfold_prefill_seconds_total"] == "3.750000"
    assert 'model="GLM \\"x\\""' in text


# -- the real handler ------------------------------------------------------------------------------------------

class FakeEngine:
    eos = (0,)

    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.request = SimpleNamespace(policy=None)

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft: bool = True):
        return self.behaviour(on_tokens)


class FakeApp:
    """``App``'s request path reduced to what the handler needs; the engine goes through ``Health.track`` as in
    ``App.__init__``."""

    served = "fake"

    def __init__(self, mode: str, behaviour):
        self.health = Health(mode=mode, stall_s=0, prefill_tps=200)
        self.engine = self.health.track(FakeEngine(behaviour))

    def check(self, body):
        return None

    def run(self, body, chat, emit):
        stats = self.engine.generate([1, 2, 3], 8, None, lambda new: not emit({"content": "hi"}))
        return {"final": {}, "calls": None, "finish": "stop", "content": "hi", "reasoning": "",
                "prompt_tokens": 3, "completion_tokens": 5, "stats": stats}


def ok_result(on_tokens):
    on_tokens([5, 6])
    on_tokens([7, 8, 9])
    return {"rounds": 2, "decode_s": 0.1, "prefill_s": 0.2, "cached": 0}


def boom(on_tokens):
    raise RuntimeError("NCCL watchdog timeout")


def refused(on_tokens):
    raise ValueError("prompt of 9 tokens: this engine serves contexts up to 8")


def test_tracked_engine_is_transparent():
    import inspect

    h = Health(mode="basic", stall_s=0, prefill_tps=200)
    raw = FakeEngine(ok_result)
    eng = h.track(raw)
    assert "draft" in inspect.signature(eng.generate).parameters     # App.check's serial-switch probe
    assert eng.eos == (0,) and eng.request is raw.request
    eng.request.policy = "auto"
    eng.limit = 99                                                    # writes reach the engine
    assert raw.limit == 99 and raw.request.policy == "auto"
    seen = []
    assert eng.generate([1] * 7, 8, None, lambda new: seen.extend(new) or False)["rounds"] == 2
    assert seen == [5, 6, 7, 8, 9]
    text = h.metrics()
    assert 'tensorfold_completion_tokens_total{model=""} 5' in text
    assert 'tensorfold_prompt_tokens_total{model=""} 7' in text


@pytest.fixture
def serve():
    servers = []

    def start(app):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    yield start
    for srv in servers:
        srv.shutdown()
        srv.server_close()


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def _post(url, body):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def test_handler_health_metrics_and_errors(serve):
    app = FakeApp("strict", ok_result)
    base = serve(app)
    code, text = _get(base + "/health")
    assert code == 200 and json.loads(text)["ok"]
    code, text = _post(base, {"messages": []})
    assert code == 200 and json.loads(text)["choices"][0]["message"]["content"] == "hi"
    code, text = _get(base + "/metrics")
    assert code == 200 and 'tensorfold_requests_total{model="fake"} 1' in text
    assert 'tensorfold_decode_rounds_total{model="fake"} 2' in text

    app.engine.behaviour = refused
    code, text = _post(base, {"messages": []})
    assert code == 400 and "contexts up to" in json.loads(text)["error"]["message"]
    assert _get(base + "/health")[0] == 200

    app.engine.behaviour = boom
    code, text = _post(base, {"messages": []})
    assert code == 500 and "NCCL" in json.loads(text)["error"]["message"]
    code, text = _get(base + "/health")
    assert code == 503 and "NCCL" in json.loads(text)["fatal"]
    app.engine.behaviour = ok_result
    code, text = _post(base, {"messages": []})       # strict: refused up front after the fatal error
    assert code == 503 and "restart both ranks" in text


def test_handler_stream_error_event(serve):
    base = serve(FakeApp("basic", boom))
    code, text = _post(base, {"messages": [], "stream": True})
    assert code == 200
    events = [ln[6:] for ln in text.splitlines() if ln.startswith("data: ")]
    assert events[-1] == "[DONE]" and "NCCL" in json.loads(events[-2])["error"]["message"]
    code, text = _get(base + "/health")
    assert code == 200 and json.loads(text)["ok"] is False      # basic: reported, still 200
