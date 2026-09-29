"""patches/0530 (``GLM53_TF_CPU_PIN=http``, THEORY-2 idea 6): rank 0's HTTP request threads off the engine's cores,
nothing else moved (or, with ``engine=``, every other thread on the engine's cores). Host scheduling only.

- the plan on GB10's cores (a fake sysfs as on both Sparks): default http cpus = the A725s (0-4, 10-14), explicit
  ``http=`` / ``engine=fast`` / ``engine=<cpus>``, errors (empty, not allowed, overlapping);
- ``classify``: by thread name only (``tf-http`` -> http cpus; others -> engine cpus or left alone);
- on this host's real threads: ``http_thread`` moves only its own thread; ``serving`` leaves everything alone without
  ``engine=``; ``serving_request`` (no batching: the request thread decodes) runs on the engine's cpus and returns to
  the http cpus; ``sweep`` with ``engine=`` moves every other thread and leaves ``tf-http`` threads.
- 0370's modes are unchanged (``auto`` / ``fast`` plans as before).

Run: PYTHONPATH=<patched tree>/src pytest -q tests/test_http_pin.py
"""

from __future__ import annotations

import os
import sys
import threading

import pytest

from tensorfold.families.glm5_next.cuda import cpupin

linux = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux affinity calls")


def _gb10(tmp_path):
    """A fake /sys/devices/system/cpu and /proc/cpuinfo as on both Sparks (tests/cuda/test_decode_overlap_patches.py)."""

    root = tmp_path / "cpu"
    lines = []
    for n in range(20):
        fast = n in range(5, 10) or n in range(15, 20)
        d = root / f"cpu{n}"
        (d / "cpufreq").mkdir(parents=True)
        cap = (997 if n < 10 else (1024 if n == 19 else 1017)) if fast else (718 if n < 10 else 731)
        (d / "cpu_capacity").write_text(f"{cap}\n")
        (d / "cpufreq" / "cpuinfo_max_freq").write_text("3900000\n" if fast else "2808000\n")
        for i, (lvl, size) in enumerate(((1, "64K"), (2, "2048K" if fast else "512K"),
                                         (3, "8192K" if n < 10 else "16384K"))):
            idx = d / "cache" / f"index{i}"
            idx.mkdir(parents=True)
            (idx / "level").write_text(f"{lvl}\n")
            (idx / "size").write_text(size + "\n")
        lines += [f"processor\t: {n}", "CPU implementer\t: 0x41", f"CPU part\t: {'0xd85' if fast else '0xd87'}", ""]
    info = tmp_path / "cpuinfo"
    info.write_text("\n".join(lines))
    return cpupin.read_cpus(root, info)


SLOW = frozenset(range(0, 5)) | frozenset(range(10, 15))
FAST = frozenset(range(5, 10)) | frozenset(range(15, 20))


def test_http_plans(tmp_path):
    cpus = _gb10(tmp_path)
    allow = frozenset(range(20))
    p = cpupin.plan("http", cpus, allow, roce=True)
    assert p.mode == "http" and p.http == SLOW and p.engine is None and p.roce is None and p.allowed == allow
    assert "HTTP threads on 0-4,10-14" in p.describe() and "unpinned" in p.describe()
    p = cpupin.plan("http=0-3;engine=fast", cpus, allow)
    assert p.http == {0, 1, 2, 3} and p.engine == FAST
    p = cpupin.plan("HTTP=12-14;engine=5-9,15-19", cpus, allow)
    assert p.http == {12, 13, 14} and p.engine == FAST
    # a container limited to fast cores only: the lower half of what it allows
    p = cpupin.plan("http", cpus, FAST)
    assert p.http == {5, 6, 7, 8, 9}
    for bad in ("http=", "http=25", "http=0-4;engine=3-6", "http;engine=", "http;serve=3", "http=0-4;nice=3"):
        with pytest.raises(ValueError, match="GLM53_TF_CPU_PIN"):
            cpupin.plan(bad, cpus, allow)
    # 0370's modes as before
    a = cpupin.plan("auto", cpus, allow, roce=True)
    assert a.mode == "roles" and a.serve == {19} and a.roce == 18 and a.http == frozenset()


def test_classify_by_name(tmp_path):
    cpus = _gb10(tmp_path)
    allow = frozenset(range(20))
    p = cpupin.plan("http", cpus, allow)
    assert cpupin.classify(1, "tf-http", allow, p, set()) == SLOW
    for name in ("tf-serve", "NCCL Progress 0", "python", "roce-proxy"):
        assert cpupin.classify(2, name, allow, p, {2}) is None             # left where the scheduler put it
    p = cpupin.plan("http;engine=fast", cpus, allow)
    assert cpupin.classify(1, "tf-http", allow, p, set()) == SLOW
    assert cpupin.classify(2, "tf-serve", allow, p, {2}) == FAST
    assert cpupin.classify(3, "NCCL Progress 0", allow, p, set()) == FAST


@linux
def test_threads_on_this_host(monkeypatch):
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 4:
        pytest.skip("needs 4 cpus")
    before = {t: os.sched_getaffinity(t) for t in map(int, os.listdir("/proc/self/task"))}
    http = frozenset(allowed[: len(allowed) // 2])
    engine = frozenset(allowed[len(allowed) // 2:])
    p = cpupin.Plan("http", frozenset(), frozenset(), frozenset(), None, 0, (), http=http, engine=None,
                    allowed=frozenset(allowed))
    monkeypatch.setattr(cpupin, "_S", cpupin._State(plan=p))
    out: dict = {}
    stop = threading.Event()
    ready = threading.Event()

    def other():
        out["other"] = threading.get_native_id()
        ready.set()
        stop.wait(10)

    def request():
        cpupin.http_thread()
        out["http_aff"] = os.sched_getaffinity(0)
        with cpupin.serving_request():
            out["decode_aff"] = os.sched_getaffinity(0)
            out["decode_name"] = cpupin._comm(threading.get_native_id())
        out["after_aff"] = os.sched_getaffinity(0)
        out["after_name"] = cpupin._comm(threading.get_native_id())

    try:
        th = threading.Thread(target=other, daemon=True)
        th.start()
        ready.wait(10)
        other_before = os.sched_getaffinity(out["other"])
        r = threading.Thread(target=request, daemon=True)
        r.start()
        r.join(10)
        assert out["http_aff"] == http
        assert out["decode_aff"] == set(allowed) and out["decode_name"] == "tf-serve"
        assert out["after_aff"] == http and out["after_name"] == "tf-http"
        cpupin.serving()                                       # no engine=: nothing else moves
        cpupin.sweep()
        assert os.sched_getaffinity(out["other"]) == other_before
        # with engine=: the sweep moves the other threads, not tf-http ones
        p.engine = engine
        cpupin.sweep()
        assert os.sched_getaffinity(out["other"]) == engine
    finally:
        stop.set()
        for t, aff in before.items():
            try:
                os.sched_setaffinity(t, aff)
            except OSError:
                pass
        cpupin.set_thread_name("pytest")


# -- the RoCE trace dump (rocedump.py) ------------------------------------------------------------------------------
def test_roce_trace_dump(tmp_path, monkeypatch):
    import json

    pytest.importorskip("torch")
    from tensorfold.families.glm5_next.cuda import roce, rocedump

    ctrl = [0] * 16

    class RT:
        rank = 1
        s = type("S", (), {"trace": 8})()
        trace_buf = object()

        def __init__(self):
            self.ctrl = ctrl
            self.reads = 0

        def trace_rows(self):
            self.reads += 1
            last = ctrl[roce.CTRL_COMPLETED]
            return [{"seq": q, "start": 1, "bell": 2, "flag": 3, "end": 4} for q in range(max(1, last - 7), last + 1)]

    rt = RT()
    monkeypatch.setattr(rocedump, "_S", {"on": None, "path": None, "every": 0, "last": 0, "errors": 0})
    monkeypatch.setenv("GLM53_TF_ROCE_TRACE_DUMP", str(tmp_path / "w13"))
    monkeypatch.setenv("GLM53_TF_ROCE_TRACE_EVERY", "10")
    monkeypatch.setattr(roce, "COMMS", [])
    rocedump.tick()                                        # no communicator yet: looks again later
    assert rocedump._S["on"] is None
    monkeypatch.setattr(roce, "COMMS", [type("C", (), {"rt": rt})()])
    for done in (3, 9, 12, 15, 21, 40):
        ctrl[roce.CTRL_COMPLETED] = done
        rocedump.tick()
    lines = [json.loads(x) for x in (tmp_path / "w13-r1.jsonl").read_text().splitlines()]
    assert [x["completed"] for x in lines] == [12, 40] and rt.reads == 2
    assert lines[0]["rank"] == 1 and [r["seq"] for r in lines[0]["rows"]] == list(range(5, 13))
    # off: nothing set up, nothing written
    monkeypatch.setattr(rocedump, "_S", {"on": None, "path": None, "every": 0, "last": 0, "errors": 0})
    monkeypatch.delenv("GLM53_TF_ROCE_TRACE_DUMP")
    rocedump.tick()
    assert rocedump._S["on"] is False
