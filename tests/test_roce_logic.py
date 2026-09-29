"""patches/0230 (``GLM53_TF_COMM_BACKEND=roce``, ``glm5_next/cuda/roce.py``), the parts without a GPU or an RDMA
device: RoCE v2 GID detection on a fake sysfs (indices that move), pairing the ranks' ports by subnet, the knobs,
the collective ``select`` (settings compared on both ranks, marker and setup-failure fallbacks to NCCL on both
ranks, ``GLM53_TF_ROCE_FALLBACK=error``) with two fake ranks on threads, and ``RoceComm``'s size dispatch.

Run against the patched tree: PYTHONPATH=<tree>/src pytest -q tests/test_roce_logic.py   (torch: CPU is enough)
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
import torch

from tensorfold.families.glm5_next.cuda import comm as comm_mod
from tensorfold.families.glm5_next.cuda import roce


# -- a fake /sys/class/infiniband ------------------------------------------------------------------------------------
def _port(root: Path, dev: str, state: str, gids: dict[int, tuple[str, str, str]], link: str = "Ethernet") -> None:
    p = root / dev / "ports" / "1"
    (p / "gids").mkdir(parents=True)
    (p / "gid_attrs" / "types").mkdir(parents=True)
    (p / "gid_attrs" / "ndevs").mkdir(parents=True)
    (p / "state").write_text(state + "\n")
    (p / "link_layer").write_text(link + "\n")
    for i in range(8):
        gid, typ, nd = gids.get(i, ("0000:0000:0000:0000:0000:0000:0000:0000", None, None))
        (p / "gids" / str(i)).write_text(gid + "\n")
        if typ is not None:              # unpopulated entries: reading the type fails (EINVAL) on a real system
            (p / "gid_attrs" / "types" / str(i)).write_text(typ + "\n")
            (p / "gid_attrs" / "ndevs" / str(i)).write_text(nd + "\n")


def _v4(ip: str) -> str:
    a, b, c, d = (int(x) for x in ip.split("."))
    return f"0000:0000:0000:0000:0000:ffff:{a:02x}{b:02x}:{c:02x}{d:02x}"


LL = "fe80:0000:0000:0000:fe9d:05ff:fe13:5f48"


def _dgx(root: Path, host: int, shift: int = 0) -> Path:
    """a DGX Spark as seen on 2026-09-28: f0 ports down, f1 ports ACTIVE, the IPv4 RoCE v2 GID at 3 (+ ``shift``)."""

    for dev in ("rocep1s0f0", "roceP2p1s0f0"):
        _port(root, dev, "1: DOWN", {})
    for dev, net, nd in (("rocep1s0f1", 100, "enp1s0f1np1"), ("roceP2p1s0f1", 101, "enP2p1s0f1np1")):
        ip = _v4(f"198.51.{net}.{host}")
        _port(root, dev, "4: ACTIVE", {0: (LL, "IB/RoCE v1", nd), 1: (LL, "RoCE v2", nd),
                                       2 + shift: (ip, "IB/RoCE v1", nd), 3 + shift: (ip, "RoCE v2", nd)})
    return root


def test_ipv4_of_gid():
    assert roce.ipv4_of_gid("0000:0000:0000:0000:0000:ffff:c633:640a") == "198.51.100.10"
    assert roce.ipv4_of_gid(LL) is None
    assert roce.ipv4_of_gid("0000:0000:0000:0000:0000:0000:0000:0000") is None


def test_detect_finds_the_ipv4_roce_v2_gid(tmp_path):
    hcas = roce.detect(root=str(_dgx(tmp_path, 10)))
    assert [(h.name, h.gid_index, h.ipv4) for h in hcas] == [("roceP2p1s0f1", 3, "198.51.101.10"),
                                                               ("rocep1s0f1", 3, "198.51.100.10")]
    assert hcas[1].netdev == "enp1s0f1np1"


def test_detect_follows_a_moved_index(tmp_path):
    hcas = roce.detect(root=str(_dgx(tmp_path, 11, shift=2)))       # after a reboot: RoCE v2 IPv4 at index 5
    assert {h.gid_index for h in hcas} == {5}


def test_detect_spec(tmp_path):
    root = str(_dgx(tmp_path, 10))
    assert [h.name for h in roce.detect("rocep1s0f1", root)] == ["rocep1s0f1"]
    forced = roce.detect("rocep1s0f1:2", root)                      # a forced index wins (even RoCE v1)
    assert forced[0].gid_index == 2 and forced[0].ipv4 == "198.51.100.10"
    with pytest.raises(ValueError, match="not ACTIVE"):
        roce.detect("rocep1s0f0", root)
    with pytest.raises(ValueError, match="name:gid_index"):
        roce.detect("rocep1s0f1:x", root)


def test_detect_skips_infiniband_and_empty(tmp_path):
    _port(tmp_path, "mlx5_ib", "4: ACTIVE", {3: (_v4("203.0.113.1"), "RoCE v2", "ib0")}, link="InfiniBand")
    _port(tmp_path, "roce_noip", "4: ACTIVE", {1: (LL, "RoCE v2", "eth9")})
    assert roce.detect(root=str(tmp_path)) == []
    assert roce.detect(root=str(tmp_path / "missing")) == []


def test_pair_by_subnet(tmp_path):
    a = roce.detect(root=str(_dgx(tmp_path / "a", 10)))
    b = list(reversed(roce.detect(root=str(_dgx(tmp_path / "b", 11)))))     # other order on the worker
    chosen = roce.pair([a, b], 2)
    assert len(chosen[0]) == 2
    for i, j in zip(*chosen):
        assert a[i].key() == b[j].key() and a[i].name == b[j].name
    assert roce.pair([a, b], 1) == [[0], [1]]
    assert roce.pair([a, [b[0]]], 2) == [[1], [0]]                  # only 198.51.100.x in common
    other = [roce.Hca("x", 3, "203.0.113.3", 24)]
    assert roce.pair([a, other], 2) == [[], []]


def test_settings(monkeypatch):
    for k in ("GLM53_TF_ROCE_MAX_KB", "GLM53_TF_ROCE_TIMEOUT_S", "GLM53_TF_ROCE_BLOCKS", "GLM53_TF_ROCE_TC",
              "NCCL_IB_TC", "GLM53_TF_ROCE_FALLBACK", roce.BACKEND_ENV):
        monkeypatch.delenv(k, raising=False)
    s = roce.settings()
    assert (s.max_bytes, s.timeout_s, s.hcas, s.blocks, s.threads, s.traffic_class) == (262144, 120.0, 2, 8, 512, 0)
    assert roce.backend() == "nccl"
    monkeypatch.setenv("NCCL_IB_TC", "106")
    assert roce.settings().traffic_class == 106
    monkeypatch.setenv("GLM53_TF_ROCE_TC", "0")
    assert roce.settings().traffic_class == 0
    for name, bad in (("GLM53_TF_ROCE_BLOCKS", "6"), ("GLM53_TF_ROCE_MAX_KB", "0"), ("GLM53_TF_ROCE_TIMEOUT_S", "x"),
                      ("GLM53_TF_ROCE_FALLBACK", "maybe"), ("GLM53_TF_ROCE_THREADS", "100")):
        monkeypatch.setenv(name, bad)
        with pytest.raises(ValueError, match=name):
            roce.settings()
        monkeypatch.delenv(name)
    with pytest.raises(ValueError, match=roce.BACKEND_ENV):
        roce.backend("rdma")


def test_grid_blocks():
    assert roce.grid_blocks(4, 512, 8) == 1
    assert roce.grid_blocks(16 * 1024, 512, 8) == 1
    assert roce.grid_blocks(64 * 1024, 512, 8) == 4
    assert roce.grid_blocks(128 * 1024, 512, 8) == 8
    assert roce.grid_blocks(1 << 20, 512, 8) == 8


# -- two fake ranks on threads ---------------------------------------------------------------------------------------
class _Pair:
    """An all-gather between two threads (CPU tensors), in call order."""

    def __init__(self):
        self.bar = threading.Barrier(2)
        self.slots = [None, None]


class _FakeBase:
    device = "cpu"
    world = 2

    def __init__(self, pair: _Pair, rank: int):
        self.p, self.rank, self.calls = pair, rank, 0

    def all_gather(self, send, recv):
        self.calls += 1
        self.p.slots[self.rank] = send.clone()
        self.p.bar.wait()
        n = send.numel()
        for r in range(2):
            recv.view(-1)[r * n:(r + 1) * n].copy_(self.p.slots[r].view(-1))
        self.p.bar.wait()

    def barrier(self):
        self.p.bar.wait()


_LOCAL = threading.local()
_REAL = (roce.settings, roce.backend, roce.marker)


def _both(fn, envs):
    """fn(base, settings, backend, marker) on two threads, each rank with its own environment (read under a lock
    before the call, since the threads share one process environment)."""

    import os

    pair, out, lock = _Pair(), [None, None], threading.Lock()

    def run(rank):
        base = _FakeBase(pair, rank)
        with lock:
            saved = {k: os.environ.get(k) for k in envs[rank]}
            os.environ.update(envs[rank])
            try:
                s, want, mark = roce.settings(), roce.backend(), roce.marker()
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        try:
            out[rank] = fn(base, s, want, mark)
        except BaseException as exc:  # noqa: BLE001
            out[rank] = exc

    ts = [threading.Thread(target=run, args=(r,)) for r in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    return out


def _select(monkeypatch):
    """roce.select with each rank's knobs (thread-local: the module functions are patched once, for both)."""

    real = _REAL
    monkeypatch.setattr(roce, "settings", lambda: _LOCAL.knobs[0])
    monkeypatch.setattr(roce, "backend", lambda raw=None: _LOCAL.knobs[1])
    monkeypatch.setattr(roce, "marker", lambda: _LOCAL.knobs[2])

    def both(envs):
        with monkeypatch.context() as m:           # _both reads each rank's knobs with the real functions
            m.setattr(roce, "settings", real[0])
            m.setattr(roce, "backend", real[1])
            m.setattr(roce, "marker", real[2])
            knobs = []
            for env in envs:
                with monkeypatch.context() as e:
                    for k, v in env.items():
                        e.setenv(k, v)
                    knobs.append((real[0](), real[1](), real[2]()))
        pair, out = _Pair(), [None, None]

        def run(rank):
            _LOCAL.knobs = knobs[rank]
            try:
                out[rank] = roce.select(_FakeBase(pair, rank))
            except BaseException as exc:  # noqa: BLE001
                out[rank] = exc

        ts = [threading.Thread(target=run, args=(r,)) for r in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(30)
        return out

    return both


BASE_ENV = {"GLM53_TF_ROCE_MARK": "", "GLM53_TF_ROCE_FALLBACK": "nccl", "GLM53_TF_ROCE_MAX_KB": "256"}


def test_select_nccl_is_base(monkeypatch):
    env = dict(BASE_ENV, **{roce.BACKEND_ENV: "nccl"})
    got = _select(monkeypatch)([env, env])
    assert all(isinstance(g, _FakeBase) for g in got), got


def test_select_refuses_different_settings(monkeypatch):
    a = dict(BASE_ENV, **{roce.BACKEND_ENV: "roce"})
    b = dict(BASE_ENV, **{roce.BACKEND_ENV: "nccl"})
    got = _select(monkeypatch)([a, b])
    assert all(isinstance(g, RuntimeError) and "different GLM53_TF_COMM_BACKEND" in str(g) for g in got), got
    c = dict(a, GLM53_TF_ROCE_MAX_KB="128")
    got = _select(monkeypatch)([a, c])
    assert all(isinstance(g, RuntimeError) for g in got), got


def test_select_marker_falls_back_on_both(monkeypatch, tmp_path, capfd):
    mark = tmp_path / "roce-failed"
    mark.write_text("x")
    a = dict(BASE_ENV, **{roce.BACKEND_ENV: "roce", "GLM53_TF_ROCE_MARK": str(mark)})
    b = dict(BASE_ENV, **{roce.BACKEND_ENV: "roce", "GLM53_TF_ROCE_MARK": str(tmp_path / "none")})
    got = _select(monkeypatch)([a, b])
    assert all(isinstance(g, _FakeBase) for g in got), got
    assert "marker" in capfd.readouterr().err


def test_select_setup_failure_falls_back_on_both(monkeypatch, tmp_path, capfd):
    monkeypatch.setattr(roce, "SYSFS", str(tmp_path / "no-rdma"))
    env = dict(BASE_ENV, **{roce.BACKEND_ENV: "roce"})
    got = _select(monkeypatch)([env, env])
    assert all(isinstance(g, _FakeBase) for g in got), got
    assert "serving on NCCL" in capfd.readouterr().err
    strict = dict(env, GLM53_TF_ROCE_FALLBACK="error")
    got = _select(monkeypatch)([strict, strict])
    assert all(isinstance(g, RuntimeError) and "setup failed" in str(g) for g in got), got


def test_select_one_rank_without_ports(monkeypatch, tmp_path):
    """One rank has ports, the other none: both see the same verdict (no rank connects alone)."""

    good = str(_dgx(tmp_path / "good", 10))
    real, calls, lock = roce.detect, [], threading.Lock()

    def detect(spec="", root=None):
        with lock:
            calls.append(1)
            first = len(calls) == 1
        return real(spec, good) if first else []

    monkeypatch.setattr(roce, "detect", detect)
    env = dict(BASE_ENV, **{roce.BACKEND_ENV: "roce"})
    got = _select(monkeypatch)([env, env])
    assert all(isinstance(g, _FakeBase) for g in got), got
    assert len(calls) == 2


def test_exchange_roundtrip():
    objs = [{"a": 1, "blob": "00" * 300}, {"b": [1, 2, 3]}]

    def fn(base, s, want, mark):
        return roce.exchange(base, objs[base.rank])

    got = _both(fn, [{}, {}])
    assert got[0] == objs and got[1] == objs


# -- dispatch -------------------------------------------------------------------------------------------------------
class _Rt:
    def __init__(self):
        self.sizes, self.checked = [], 0

    def gather(self, send, recv):
        self.sizes.append(send.numel() * send.element_size())
        recv.view(-1)[:send.numel()].copy_(send)
        recv.view(-1)[send.numel():].copy_(send)

    def check(self):
        self.checked += 1


class _Base:
    rank, world = 0, 2
    store = "the-store"

    def __init__(self):
        self.sizes = []

    def all_gather(self, send, recv):
        self.sizes.append(send.numel() * send.element_size())
        recv.view(-1)[:send.numel()].copy_(send)
        recv.view(-1)[send.numel():].copy_(send)

    def barrier(self):
        pass


def test_roce_comm_dispatch():
    base, rt = _Base(), _Rt()
    c = roce.RoceComm(base, rt, 1024)
    x = torch.arange(256, dtype=torch.float32)          # 1 KiB: RoCE
    y = torch.empty(512)
    comm_mod.fast_gather(c, x, y)
    comm_mod.fast_gather(c, torch.arange(257, dtype=torch.float32), torch.empty(514))    # over: NCCL
    c.all_gather(x, y)                                  # control exchanges: always NCCL
    assert rt.sizes == [1024] and base.sizes == [1028, 1024]
    assert torch.equal(y[:256], x) and torch.equal(y[256:], x)
    comm_mod.check(c)
    c.barrier()
    assert rt.checked == 2
    assert c.store == "the-store" and (c.fast_ops, c.slow_ops) == (1, 1)
    comm_mod.fast_gather(base, x, y)                    # a plain communicator: its all_gather
    comm_mod.check(base)
    assert base.sizes[-1] == 1024
