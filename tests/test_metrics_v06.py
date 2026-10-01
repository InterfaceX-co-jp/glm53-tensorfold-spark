"""patches/0660: ``/metrics`` families taken from upstream v0.6.0 (``server/metrics.py``).

Host only (no GPU, no tokenizer): ``Health`` against a fake batch engine, then the real handler. Run against the
patched tree:

    PYTHONPATH=<tree>/src pytest -q tests/test_metrics_v06.py
"""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

health_mod = pytest.importorskip("tensorfold.cuda.health")
from tensorfold.cuda.health import BUCKETS, Health  # noqa: E402

# The families upstream v0.6.0 publishes that 0150's /metrics did not have.
EXPECTED = (
    "tensorfold_requests_running",
    "tensorfold_requests_waiting",
    "tensorfold_prompt_tokens_total",
    "tensorfold_generation_tokens_total",
    "tensorfold_kv_cache_usage_ratio",
    "tensorfold_mtp_drafted_total",
    "tensorfold_mtp_accepted_total",
    "tensorfold_request_latency_seconds",
    "tensorfold_time_to_first_token_seconds",
)


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def lines_of(text: str, family: str) -> list[str]:
    """Every sample line of ``family`` (a histogram's buckets, ``_sum`` and ``_count`` included)."""

    return [ln for ln in text.splitlines() if ln and not ln.startswith("#") and ln.split("{")[0] == family]


def value_of(text: str, name: str, labels: str = 'model="m"') -> str:
    """The one value of ``name{labels}``; a missing series is a failure, not a default."""

    want = f"{name}{{{labels}}}"
    for line in text.splitlines():
        if line.startswith(want + " "):
            return line.rsplit(" ", 1)[1]
    raise AssertionError(f"{want} is not in:\n{text}")


def declared(text: str) -> tuple[list[str], list[str]]:
    """(``# HELP`` names, ``# TYPE`` names). Prometheus wants each family declared exactly once."""

    helps = [ln.split(" ", 3)[2] for ln in text.splitlines() if ln.startswith("# HELP ")]
    types = [ln.split(" ", 3)[2] for ln in text.splitlines() if ln.startswith("# TYPE ")]
    return helps, types


def test_every_family_is_declared_exactly_once():
    text = Health(mode="basic", stall_s=0, clock=Clock()).metrics("m")
    helps, types = declared(text)
    assert sorted(helps) == sorted(types), "every family needs one HELP and one TYPE"
    assert len(helps) == len(set(helps)) and len(types) == len(set(types))
    assert set(EXPECTED) <= set(types)


def test_histograms_observe_latency_and_time_to_first_token():
    clock = Clock()
    h = Health(mode="basic", stall_s=0, clock=clock)
    rid = h.begin(100)
    clock.t += 2.0
    h.progress(rid, 1)                                     # the first token, 2 s in
    clock.t += 3.0
    h.progress(rid, 4)
    clock.t += 1.0
    h.end(rid, stats={"rounds": 2, "decode_s": 0.5, "prefill_s": 1.0, "cached": 0, "drafted": 8, "accepted": 5},
          completion_tokens=5)
    text = h.metrics("m")
    assert value_of(text, "tensorfold_request_latency_seconds_count") == "1"
    assert value_of(text, "tensorfold_request_latency_seconds_sum") == "6"       # 6 s from begin to end
    assert value_of(text, "tensorfold_time_to_first_token_seconds_count") == "1"
    assert value_of(text, "tensorfold_time_to_first_token_seconds_sum") == "2"
    # buckets are cumulative: 6 s is in le=10, not le=5, and every edge is rendered with +Inf last
    assert value_of(text, "tensorfold_request_latency_seconds_bucket", 'le="5",model="m"') == "0"
    assert value_of(text, "tensorfold_request_latency_seconds_bucket", 'le="10",model="m"') == "1"
    assert value_of(text, "tensorfold_request_latency_seconds_bucket", 'le="+Inf",model="m"') == "1"
    assert value_of(text, "tensorfold_time_to_first_token_seconds_bucket", 'le="2.5",model="m"') == "1"
    assert len(lines_of(text, "tensorfold_request_latency_seconds_bucket")) == len(BUCKETS) + 1
    # the same request's stats, under upstream's names
    assert value_of(text, "tensorfold_mtp_drafted_total") == "8"
    assert value_of(text, "tensorfold_mtp_accepted_total") == "5"
    assert value_of(text, "tensorfold_generation_tokens_total") == "5"
    assert value_of(text, "tensorfold_completion_tokens_total") == "5"


def test_time_to_first_token_is_skipped_when_no_token_came():
    clock = Clock()
    h = Health(mode="basic", stall_s=0, clock=clock)
    rid = h.begin(10)
    clock.t += 4.0
    h.end(rid, error=ValueError("prompt too long"))
    text = h.metrics("m")
    assert value_of(text, "tensorfold_request_latency_seconds_count") == "1"      # it held the server
    assert value_of(text, "tensorfold_request_latency_seconds_sum") == "4"
    assert value_of(text, "tensorfold_time_to_first_token_seconds_count") == "0"
    assert value_of(text, "tensorfold_time_to_first_token_seconds_sum") == "0"


def engine_app(*, slots: list, states=None, queue: int = 0, limit=None):
    """An app whose engine is a batch engine. ``slots``: one ``seqs`` entry each, ``None`` is a free slot."""

    seqs = [None if s is None else SimpleNamespace(stepper=object(), done=s, emitted=0) for s in slots]
    batch = SimpleNamespace(seqs=seqs, n=len(seqs), queue=list(range(queue)))
    if states is not None:
        batch.states = [SimpleNamespace(pos=p) for p in states]
    engine = SimpleNamespace(batch=batch)
    if limit is not None:
        engine.limit = limit
    return SimpleNamespace(engine=engine, served="m")


def test_running_waiting_and_pools_come_from_the_engine():
    app = engine_app(slots=[100, None, 200], states=[1000, 0, 2000], queue=2, limit=4000)
    text = Health(mode="basic", stall_s=0, clock=Clock()).metrics("m", app)
    assert value_of(text, "tensorfold_requests_running") == "2"        # two occupied slots, no HTTP request
    assert value_of(text, "tensorfold_requests_waiting") == "2"
    assert value_of(text, "tensorfold_requests_inflight") == "0"       # 0150's own view is unchanged
    # states[slot].pos over the window, one series per live slot: slot 0 (pool 0) and slot 2 (pool 1)
    assert lines_of(text, "tensorfold_kv_cache_usage_ratio") == [
        'tensorfold_kv_cache_usage_ratio{pool="0"} 0.25',
        'tensorfold_kv_cache_usage_ratio{pool="1"} 0.5']


def test_lengths_fall_back_to_done_plus_emitted():
    app = engine_app(slots=[400, 100], queue=0, limit=1000)            # no states: the engine has no positions
    text = Health(mode="basic", stall_s=0, clock=Clock()).metrics("m", app)
    assert lines_of(text, "tensorfold_kv_cache_usage_ratio") == [
        'tensorfold_kv_cache_usage_ratio{pool="0"} 0.4',
        'tensorfold_kv_cache_usage_ratio{pool="1"} 0.1']


def test_a_non_batch_engine_keeps_the_0150_view():
    h = Health(mode="basic", stall_s=0, clock=Clock())
    h.begin(10)                                                       # one request in flight
    text = h.metrics("m", SimpleNamespace(engine=SimpleNamespace()))
    assert value_of(text, "tensorfold_requests_inflight") == "1"
    assert value_of(text, "tensorfold_requests_running") == "1"       # the HTTP requests, as 0150 counted them
    assert value_of(text, "tensorfold_requests_waiting") == "0"
    assert lines_of(text, "tensorfold_kv_cache_usage_ratio") == ['tensorfold_kv_cache_usage_ratio{pool="0"} 0']


class Unreadable:
    """A torn-down engine: reading it raises. A scrape must not carry that to the poller."""

    def __len__(self) -> int:
        raise RuntimeError("the batcher is gone")


def test_an_unreadable_engine_still_renders():
    batch = SimpleNamespace(seqs=[SimpleNamespace(stepper=None, done=1, emitted=0)], n=1, queue=Unreadable())
    app = SimpleNamespace(engine=SimpleNamespace(batch=batch, limit=100), served="m")
    text = Health(mode="basic", stall_s=0, clock=Clock()).metrics("m", app)
    helps, types = declared(text)                                     # the fallback is still valid exposition
    assert sorted(helps) == sorted(types)
    assert value_of(text, "tensorfold_requests_waiting") == "0"
    assert lines_of(text, "tensorfold_kv_cache_usage_ratio") == ['tensorfold_kv_cache_usage_ratio{pool="0"} 0']


@pytest.fixture
def serve():
    servers = []

    def start(app):
        from tensorfold.cuda.server import make_handler

        srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    yield start
    for srv in servers:
        srv.shutdown()
        srv.server_close()


def test_the_handler_passes_the_app_to_metrics(serve):
    batch = SimpleNamespace(seqs=[SimpleNamespace(stepper=object(), done=10, emitted=5), None], n=2, queue=[1])
    batch.states = [SimpleNamespace(pos=500), SimpleNamespace(pos=0)]
    app = SimpleNamespace(served="fake", engine=SimpleNamespace(batch=batch, limit=1000),
                          health=Health(mode="basic", stall_s=0, clock=Clock()))
    base = serve(app)
    with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
        text = r.read().decode()
    assert value_of(text, "tensorfold_requests_running", 'model="fake"') == "1"
    assert value_of(text, "tensorfold_requests_waiting", 'model="fake"') == "1"
    assert lines_of(text, "tensorfold_kv_cache_usage_ratio") == ['tensorfold_kv_cache_usage_ratio{pool="0"} 0.5']
    with urllib.request.urlopen(base + "/health", timeout=10) as r:
        body = json.loads(r.read().decode())
    assert body["streams"] == {"decoding": 1, "prefilling": 0, "max": 2}      # 0660 changed no /health field
    assert body["context_length"] == 1000
