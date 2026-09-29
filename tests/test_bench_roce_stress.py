"""patches/0350: ``tests/cuda/bench_roce.py stress`` on the CPU with fake runtimes (no GPU, no RDMA): the payload moves
every op, so a peer slot read one or two sequences stale is counted, which 0230's soak (the same payload every replay)
could not see; a clean fake passes; a failed runtime stops the run. The graph path is replaced by closures replayed in
order (``_graph_pair`` itself needs CUDA).

Run: PYTHONPATH=<patched tree>/src pytest -q tests/test_bench_roce_stress.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parent / "cuda"))
roce = pytest.importorskip("tensorfold.families.glm5_next.cuda.roce")
import bench_roce  # noqa: E402


class _FakeRt:
    """One rank of a 2-rank all-gather whose peer sends base_peer + seq at sequence seq (int32 words), as the real
    peer's stress stream does; ``stale``: at those sequences the peer slot holds the payload of seq - 2 (the same
    slot's previous use)."""

    def __init__(self, rank: int, n: int, salt: int, stale=(), fail_at=None):
        self.rank, self.seq, self.stale, self.fail_at = rank, 0, set(stale), fail_at
        self.s = roce.Settings(max_bytes=256 * 1024)
        self.peer_base = roce.pattern(1 - rank, n, salt).view(torch.int32).clone()
        self.failed = False

    def gather(self, x, y):
        self.seq += 1
        k = self.seq - 2 if self.seq in self.stale else self.seq
        peer = self.peer_base + k
        y.copy_(torch.cat([x, peer] if self.rank == 0 else [peer, x]))
        if self.fail_at is not None and self.seq >= self.fail_at:
            self.failed = True

    def check(self):
        if self.failed:
            raise roce.RoceError("fake timeout")

    def snapshot(self):
        return {"completed": self.seq}


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    def pattern(rank, nbytes, salt):
        i = torch.arange(nbytes, dtype=torch.int64)
        return ((i * 131 + rank * 71 + salt * 29 + (i >> 8) * 7) & 0xFF).to(torch.uint8)

    monkeypatch.setattr(roce, "pattern", pattern)
    monkeypatch.setattr(torch.cuda, "Stream", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: _Null())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)

    def graph_pair(fns, ops, tails=None):
        return [SimpleNamespace(replay=lambda fn=fn: [fn() for _ in range(ops)]) for fn in fns]

    monkeypatch.setattr(bench_roce, "_graph_pair", graph_pair)


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _args(**kw):
    return SimpleNamespace(**dict(dict(sizes=[16 * 1024], iters=2000, ops=90, check=500), **kw))


def _run(stale=(), fail_at=None, rank=0, **kw):
    salt = 100
    rt = _FakeRt(rank, 16 * 1024, salt, stale, fail_at)
    return bench_roce._stress([rt], [rank], 2, _args(**kw))


def test_clean_run_passes(capsys):
    assert _run()
    rows = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("{")]
    assert len(rows) == 2 and all('"mismatched_words": 0' in r for r in rows)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("seq", [7, 1500, 2000 + 90 * 3])        # eager ops, then graph replays
def test_a_stale_slot_is_counted(seq, rank, capsys):
    assert not _run(stale={seq}, rank=rank)
    assert '"mismatched_words": 4096' in capsys.readouterr().out     # the whole 16 KiB peer shard


def test_a_failed_runtime_stops_the_run(capsys):
    assert not _run(fail_at=1234)
    assert "fake timeout" in capsys.readouterr().out


def test_constant_payload_would_miss_it():
    """The premise: with 0230's soak payload (the same bytes every op) a stale slot equals the fresh one."""

    base = roce.pattern(1, 64, 5).view(torch.int32)
    assert torch.equal(base + 0, base + 0)                   # seq and seq - 2 carry identical bytes in the soak
    assert not torch.equal(base + 10, base + 8)              # stress: they differ
