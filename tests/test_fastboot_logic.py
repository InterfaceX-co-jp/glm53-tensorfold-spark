"""Fast restarts (patches/0140, ``glm5_next/cuda/fastboot.py``), the parts without torch: prepared-folder manifests and
their invalidation, where folders live, and the calibration cache (keys, storage, and rank 0 deciding for both
ranks through ``GlmEngine._calib_cache`` with two fake ranks on threads). No torch, no GPU.

Run against the patched tree: PYTHONPATH=<tree>/src pytest -q tests/test_fastboot_logic.py
"""

from __future__ import annotations

import json
import threading

import pytest

from tensorfold.families.glm5_next.cuda import fastboot
from tensorfold.families.glm5_next.cuda.engine import GlmEngine


# -- prepared folders ----------------------------------------------------------------------------------------------
KEY = {"format": fastboot.FORMAT, "what": "weights", "model": "org--m-07135ec082f8", "files": [[["a", 1]], "c"],
       "rank": 1, "world": 2, "nonexpert": "q4mse", "torch": "2.x", "device": "cuda", "code": "abc"}


def _fake_folder(tmp_path, key, size=8192, **over):
    d = fastboot.folder(tmp_path, key)
    d.mkdir(parents=True)
    (d / "data.bin").write_bytes(bytes(size))
    m = {"format": fastboot.FORMAT, "key": key, "key_hash": fastboot.digest(key), "file_size": size,
         "chunk": fastboot.CHUNK, "chunk_sha256": [], "storages": [], "tree": {"P": None}}
    m.update(over)
    (d / "manifest.json").write_text(json.dumps(m))
    return d


def test_folder_layout():
    d = fastboot.folder("/p", KEY)
    assert d.parts[-3] == "org--m-07135ec082f8" and d.name == "rank1" and len(d.parts[-2]) == 16
    dk = dict(KEY, what="drafter")
    assert fastboot.folder("/p", dk).parts[-3] == "drafter-org--m-07135ec082f8"


def test_model_rev_from_snapshot_path():
    p = "/root/.cache/huggingface/hub/models--neko-legends--GLM-5.3-Flash-Uncensored-EXL3/snapshots/" \
        "07135ec082f8f11f7a71e4244a4e5167a0f96277"
    assert fastboot.model_rev(p) == "neko-legends--GLM-5.3-Flash-Uncensored-EXL3-07135ec082f8"
    assert fastboot.model_rev("/data/glm-rank0") == "glm-rank0"


def test_manifest_valid_and_invalidated(tmp_path):
    d = _fake_folder(tmp_path, KEY)
    assert fastboot.valid(d, KEY) == (True, "ok")
    # every key field that decides the bits misses the folder, and says which
    for field, value in (("nonexpert", "bf16"), ("code", "patched"), ("rank", 0), ("torch", "2.y"),
                         ("files", [[["a", 2]], "c"]), ("model", "other")):
        other = dict(KEY, **{field: value})
        ok, why = fastboot.valid(d, other)
        assert not ok and (field in why or "no folder" in why), (field, why)
    # a folder of the same key but another format / a truncated data file
    (d / "manifest.json").write_text(json.dumps(dict(json.loads((d / "manifest.json").read_text()), format=0)))
    assert fastboot.valid(d, KEY)[0] is False
    d2 = _fake_folder(tmp_path / "b", KEY, file_size=9999)
    ok, why = fastboot.valid(d2, KEY)
    assert not ok and "truncated" in why
    assert fastboot.valid(tmp_path / "nothing", KEY) == (False, "no folder")


def test_prepared_env(monkeypatch):
    for v in ("", "0", "off"):
        monkeypatch.setenv("GLM53_TF_PREPARED", v)
        assert fastboot.prepared_root() is None
    monkeypatch.setenv("GLM53_TF_PREPARED", "/prepared")
    assert str(fastboot.prepared_root()) == "/prepared"
    monkeypatch.delenv("GLM53_TF_PREPARED_WRITE", raising=False)
    assert fastboot.write_enabled() is False
    monkeypatch.setenv("GLM53_TF_PREPARED_WRITE", "1")
    assert fastboot.write_enabled() is True
    monkeypatch.setenv("GLM53_TF_PREPARED_VERIFY", "bogus")
    with pytest.raises(ValueError):
        fastboot.verify_mode()


def test_read_plan_covers_every_byte_once():
    storages = [{"offset": 0, "nbytes": 5000}, {"offset": 8192, "nbytes": fastboot.CHUNK + 10},
                {"offset": 8192 + fastboot.CHUNK + 4096 * 3, "nbytes": 1}]
    size = storages[-1]["offset"] + 4096
    plan = fastboot._plan(storages, size)
    seen = {i: 0 for i in range(3)}
    for k, pieces in enumerate(plan):
        for i, a, s, n in pieces:
            assert 0 <= s and s + n <= fastboot.CHUNK
            assert k * fastboot.CHUNK + s == storages[i]["offset"] + a      # file position of the piece
            assert a == seen[i]                                           # in order, no gap, no overlap
            seen[i] += n
    assert seen == {0: 5000, 1: fastboot.CHUNK + 10, 2: 1}
    assert [fastboot._checked(k, 40, "sample") for k in (0, 1, 16, 39)] == [True, False, True, True]


# -- calibration cache ---------------------------------------------------------------------------------------------
def test_calib_ident_knobs(monkeypatch):
    monkeypatch.setattr(fastboot, "clock_cap", lambda: "cap")
    monkeypatch.setenv("GLM53_TF_IMAGE_ID", "sha256:img")
    monkeypatch.setenv("GLM53_TF_DECODE_KERNELS", "v2,pdl")
    for k in fastboot.CALIB_SKIP:
        monkeypatch.setenv(k, "x")
    a = fastboot.calib_ident(capacity=4096)
    assert a["knobs"].get("GLM53_TF_DECODE_KERNELS") == "v2,pdl"
    assert not set(a["knobs"]) & fastboot.CALIB_SKIP              # launch times, the calib mode, ... are not in it
    monkeypatch.setenv("GLM53_TF_CALIB", "real")                    # the mode itself never changes the key
    monkeypatch.setenv("GLM53_TF_LAUNCH_T0", "123")
    assert fastboot.ident_ints(fastboot.calib_ident(capacity=4096)) == fastboot.ident_ints(a)
    monkeypatch.setenv("GLM53_TF_DECODE_KERNELS", "v1")               # a knob that changes timings does
    assert fastboot.ident_ints(fastboot.calib_ident(capacity=4096)) != fastboot.ident_ints(a)
    monkeypatch.setenv("GLM53_TF_DECODE_KERNELS", "v2,pdl")
    assert fastboot.ident_ints(fastboot.calib_ident(capacity=8192)) != fastboot.ident_ints(a)
    monkeypatch.setattr(fastboot, "clock_cap", lambda: "other cap")    # so does a clock cap
    assert fastboot.ident_ints(fastboot.calib_ident(capacity=4096)) != fastboot.ident_ints(a)
    monkeypatch.setenv("GLM53_TF_IMAGE_ID", "sha256:other")            # and another image
    assert fastboot.ident_ints(fastboot.calib_ident(capacity=4096)) != fastboot.ident_ints(a)
    ints = fastboot.ident_ints(a)
    assert len(ints) == 8 and all(-2 ** 31 <= v < 2 ** 31 for v in ints)


def test_calib_name_uses_both_ranks():
    a, b = [1] * 8, [2] * 8
    assert fastboot.calib_name(a, b) != fastboot.calib_name(b, a) != fastboot.calib_name(a, a)
    assert fastboot.calib_name(a, b) == fastboot.calib_name(list(a), list(b))


COSTS = {"verify": [30.012345678901234, 38.0, 43.7, 49.4, 55.0, 60.7, 66.4, 72.1], "mtp": 1.97, "mtp_step": 1.52,
         "mtp_row": 0.19, "block": 3.78, "taps_row": 0.051, "timed": {"v1_0": 30.0}, "calib": "real",
         "windows": [[1, 2, 3], [4, 5]]}


def test_calib_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM53_TF_CALIB_DIR", str(tmp_path / "c"))
    assert fastboot.calib_read("calib-x.json") is None
    p = fastboot.calib_write("calib-x.json", COSTS)
    assert p is not None and p.exists()
    assert fastboot.calib_read("calib-x.json") == COSTS
    # the bytes rank 0 shares decode to the same floats on the other rank
    raw = json.dumps(COSTS).encode()
    assert json.loads(fastboot.from_ints(fastboot.to_ints(raw))) == COSTS
    p.write_text(json.dumps({"format": 999, "costs": COSTS}))
    assert fastboot.calib_read("calib-x.json") is None
    p.write_text("{not json")
    assert fastboot.calib_read("calib-x.json") is None


class _Bus:
    """Two fake ranks' all-gather: each call blocks until both ranks gave their values."""

    def __init__(self) -> None:
        self.barrier = threading.Barrier(2)
        self.slots: list = [None, None]

    def gather(self, rank: int, values: list[int]) -> list[list[int]]:
        self.slots[rank] = list(values)
        self.barrier.wait()
        got = [list(self.slots[0]), list(self.slots[1])]
        self.barrier.wait()
        return got


class _FakeRank:
    _calib_cache = GlmEngine._calib_cache

    def __init__(self, bus: _Bus, rank: int, engine: dict) -> None:
        self.bus, self.rank, self._calib_engine = bus, rank, engine

    def _gather_ints(self, values):
        return self.bus.gather(self.rank, values)

    def _share(self, values):
        got = self.bus.gather(self.rank, values if self.rank == 0 else [])
        return got[0]


def _both(how: str, engines=({"capacity": 4096}, {"capacity": 4096}), homes=None, monkeypatch=None):
    bus = _Bus()
    out: list = [None, None]

    def run(r: int) -> None:
        out[r] = _FakeRank(bus, r, dict(engines[r]))._calib_cache(how)

    ts = [threading.Thread(target=run, args=(r,)) for r in (0, 1)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    return out


def test_calib_cache_rank0_decides_and_shares(tmp_path, monkeypatch):
    monkeypatch.setattr(fastboot, "clock_cap", lambda: "cap")
    monkeypatch.setenv("GLM53_TF_CALIB_DIR", str(tmp_path))
    monkeypatch.setenv("GLM53_TF_BOOT_WARMUP", "0")
    # a miss: both ranks measure (None), both know the same name
    (n0, c0), (n1, c1) = _both("cached")
    assert c0 is None and c1 is None and n0 == n1
    # rank 0 stored a table (what _calibrate does after measuring): both ranks now get rank 0's table
    fastboot.calib_write(n0, COSTS)
    (m0, c0), (m1, c1) = _both("cached")
    assert m0 == n0 and c0 == c1 and c0["verify"] == COSTS["verify"] and c0["calib"] == "cached (real)"
    # GLM53_TF_CALIB=real: always measure (the forced re-measure), same name so the result refreshes the table
    (r0, c0), (r1, c1) = _both("real")
    assert c0 is None and c1 is None and r0 == n0
    # a different engine shape on either rank is another table
    (x0, c0), _ = _both("cached", engines=({"capacity": 4096}, {"capacity": 8192}))
    assert c0 is None and x0 != n0
