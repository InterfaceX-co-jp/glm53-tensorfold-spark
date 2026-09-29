"""patches/0350: ``roce_watch.h`` (the proxy thread's view of a failed RoCE wait) compiled with the host's C++
compiler and run on the CPU: idle polls record nothing; a flag already in host memory at the failure is
``not_seen`` (host_at_fail == seq); a flag that arrives 30 ms after it is ``late`` (state 2, ~30 ms); one that never
comes stays ``never``; an out-of-range record does not crash; 2,000 records written by a racing thread are always read
whole. Then ``roce.classify`` on the same records.

Run: PYTHONPATH=<patched tree>/src pytest -q tests/test_roce_watch.py   (needs g++ or clang++; torch for classify)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def _header_dir() -> Path:
    for p in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        d = Path(p) / "tensorfold" / "families" / "glm5_next" / "cuda"
        if (d / "roce_watch.h").exists():
            return d
    try:
        import tensorfold

        d = Path(tensorfold.__file__).parent / "families" / "glm5_next" / "cuda"
        if (d / "roce_watch.h").exists():
            return d
    except ImportError:
        pass
    pytest.skip("roce_watch.h not found (PYTHONPATH=<tree with patches/0350>/src)")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    cxx = shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        pytest.skip("no C++ compiler")
    exe = tmp_path_factory.mktemp("watch") / "roce_watch_test"
    subprocess.run([cxx, "-std=c++17", "-O2", "-pthread", "-Wall", "-Werror", f"-I{_header_dir()}",
                    str(HERE / "cpp" / "roce_watch_test.cpp"), "-o", str(exe)], check=True)
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True, timeout=120).stdout
    rows = {}
    for line in out.splitlines():
        name, *kv = line.split()
        rows[name] = {k: float(v) for k, v in (x.split("=") for x in kv)}
    return rows


def test_idle_records_nothing(runs):
    assert runs["idle"]["state"] == 0


def test_flag_present_at_failure_is_not_seen(runs):
    r = runs["not_seen"]
    assert (r["state"], r["seq"], r["peer"], r["hca"], r["host_at_fail"], r["gpu_seen"]) == (1, 312, 1, 1, 312, 310)


def test_flag_after_failure_is_late(runs):
    r = runs["late"]
    assert (r["state"], r["seq"], r["host_at_fail"], r["gpu_seen"]) == (2, 312, 310, 310)
    assert 25 <= r["late_ms"] < 1000


def test_flag_never_arrives(runs):
    r = runs["never"]
    assert (r["state"], r["seq"], r["host_at_fail"]) == (1, 313, 311)


def test_out_of_range_record(runs):
    assert runs["range"]["state"] == 1 and runs["range"]["host_at_fail"] == 0


def test_records_are_read_whole(runs):
    assert runs["race"]["bad"] == 0


def test_classify():
    roce = pytest.importorskip("tensorfold.families.glm5_next.cuda.roce")

    w = lambda state, at, late=0.0, seq=312: {"state": state, "seq": seq, "peer": 1, "hca": 0,  # noqa: E731
                                              "host_at_fail": at, "gpu_seen": 310, "late_ms": late}
    assert roce.classify(312, 310, 312, w(1, 312))[0] == "not_seen"
    kind, text = roce.classify(312, 310, 312, w(2, 310, 9876.5))
    assert kind == "late" and "9876.5 ms AFTER" in text and "not a visibility problem" in text
    assert roce.classify(312, 310, 310, w(1, 310))[0] == "never"
    assert roce.classify(312, 310, 312, None)[0] == "unknown"               # 0230's check-time view only
    assert roce.classify(312, 310, 312, w(0, 0))[0] == "unknown"
    assert roce.classify(312, 310, 311, None)[0] == "never"
    assert roce.classify(312, 310, 312, w(1, 312, seq=300))[0] == "unknown"   # a record of another sequence
    # the W2 log line, as 0350 classifies it: the flag rank 1 sent during its own warm-up, ~10 s later
    assert roce.classify(312, 310, 312, w(2, 310, 10012.0))[0] == "late"
