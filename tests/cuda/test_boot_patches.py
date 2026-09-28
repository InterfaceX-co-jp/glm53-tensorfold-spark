"""patches/0140 (fast restarts, ``glm5_next/cuda/fastboot.py``) on the GPU, on TensorFold's synthetic GLM checkpoint
(one GPU playing rank 0 of two). Checked here:

- the prepared weights are the load-time weights bit for bit, where it matters: built on the GPU (the q4mse clip
  search runs there), written, read back with the O_DIRECT reader into device memory: every tensor's bytes, dtype,
  shape, stride and offset, and the storage sharing; for both ranks and every GLM53_TF_NONEXPERT mode, and for the
  DFlash2 drafter. Two builds agree too (the load-time path is deterministic, so a prepared folder written once
  stands for every later build);
- an engine on prepared weights replies exactly as an engine built from the checkpoint (drafted == serial);
- the calibration cache: GLM53_TF_CALIB=cached measures once and stores the table, the next engine takes the stored
  table (no timing runs) and still replies drafted == serial; GLM53_TF_CALIB=real measures again and refreshes it;
- the [boot] timeline lines are printed;
- reader throughput on a large synthetic folder (GLM53_TF_BOOT_BENCH_GB, default 4; set 0 to skip): prints GB/s
  (target >= 4.5 GB/s a rank on the Spark's NVMe; the drive does ~10 GB/s with O_DIRECT).

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda pytest -q -s tests/cuda/test_boot_patches.py
(BOOT_TEST_DIR: where the throughput folder is written, default a pytest tmp dir; use a host bind mount on the
NVMe, e.g. -v ~/.cache/glm53-tf:/bench -e BOOT_TEST_DIR=/bench, since /tmp in a container is overlayfs.)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.glm5_next.cuda import fastboot, weights  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_fastboot_prepared import same  # noqa: E402


@pytest.fixture(autouse=True)
def _own_calib_dir(tmp_path, monkeypatch):
    """Never touch the image's /cache/calib (the server's calibration tables)."""

    monkeypatch.setenv("GLM53_TF_CALIB_DIR", str(tmp_path / "calib-default"))


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_boot")
    _checkpoint(path / "exl3", exl3=True)
    _checkpoint(path / "mlx")
    _drafter(path / "dflash2")
    return path


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM53_TF_PREPARED", str(tmp_path / "prepared"))
    monkeypatch.setenv("GLM53_TF_PREPARED_WRITE", "1")
    monkeypatch.setenv("GLM53_TF_PREPARED_VERIFY", "full")
    monkeypatch.setattr(fastboot, "CHUNK", 1 << 20)        # several chunks and threads on the small model
    return tmp_path / "prepared"


@pytest.mark.parametrize("kind,nonexpert", [("exl3", "bf16"), ("exl3", "q4"), ("exl3", "q4mse"), ("mlx", "bf16")])
@pytest.mark.parametrize("rank", [0, 1])
def test_prepared_weights_equal_load_time(ckpt, prepared, monkeypatch, kind, nonexpert, rank):
    monkeypatch.setattr(weights, "NONEXPERT", nonexpert)
    ref = weights.load_checkpoint(ckpt / kind, rank=rank)
    ref2 = weights.load_checkpoint(ckpt / kind, rank=rank)
    same(ref, ref2)                                   # the load-time path is deterministic on the GPU
    first = weights.load(ckpt / kind, rank=rank)        # built, written
    same(ref, first)
    key = fastboot.weights_key(ckpt / kind, rank)
    assert fastboot.valid(fastboot.folder(prepared, key), key)[0]
    monkeypatch.setattr(weights, "load_checkpoint", lambda *a, **k: pytest.fail("built instead of read"))
    for direct in ("1", "0"):
        monkeypatch.setenv("GLM53_TF_PREPARED_DIRECT", direct)
        got = weights.load(ckpt / kind, rank=rank)
        same(ref, got)
        assert got.head.weight.is_cuda
    del ref, ref2, first, got
    torch.cuda.empty_cache()


def test_prepared_drafter_equal_load_time(ckpt, prepared, monkeypatch):
    from tensorfold.families.glm5_next.cuda import dflash2

    monkeypatch.setattr(weights, "NONEXPERT", "q4mse")
    w = weights.load_checkpoint(ckpt / "exl3", rank=0)
    ref = dflash2.Drafter(ckpt / "dflash2", w, capacity=4096)
    dflash2.Drafter(ckpt / "dflash2", w, capacity=4096)                 # built, written
    monkeypatch.setattr(dflash2, "safe_open", lambda *a, **k: pytest.fail("built instead of read"))
    got = dflash2.Drafter(ckpt / "dflash2", w, capacity=4096)
    for name in ("fc", "hidden_norm", "norm", "hproj", "pred", "succ", "layers"):
        same(getattr(ref, name), getattr(got, name), name)
    del w, ref, got
    torch.cuda.empty_cache()


def _engine(ckpt, calib_mode: str = "real"):
    from test_glm_engine import _TwoCopies

    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GLM53_TF_CALIB", calib_mode)
        mp.delenv("GLM53_TF_BATCH", raising=False)
        return GlmEngine(ckpt / "exl3", rank=0, master="", port=0, drafter=ckpt / "dflash2", comm=_TwoCopies())


def _replies(e) -> list:
    import numpy as np

    from tensorfold.engine.exact_sampling import Sampling
    from test_glm_engine import _generate

    out = []
    prompt = [int(t) for t in np.random.default_rng(41).integers(0, 1000, size=37)]
    for sampling in (Sampling(1234, 1.0, 20, 0.95), None):
        serial, _ = _generate(e, prompt, sampling, draft=False, tokens=24)
        for policy in (None, "auto", "fc5:0.3", "c3:0.35"):
            drafted, _ = _generate(e, prompt, sampling, policy=policy, tokens=24)
            assert drafted == serial, (policy, sampling)
        out.append(serial)
    return out


def test_engine_on_prepared_weights_replies_the_same(ckpt, prepared, monkeypatch, capsys):
    monkeypatch.setattr(weights, "NONEXPERT", "q4mse")
    monkeypatch.setenv("GLM53_TF_PREPARED", "")                        # from the checkpoint
    a = _engine(ckpt)
    want = _replies(a)
    del a
    torch.cuda.empty_cache()
    monkeypatch.setenv("GLM53_TF_PREPARED", str(prepared))
    _engine(ckpt)                                                      # writes the folders
    torch.cuda.empty_cache()
    b = _engine(ckpt)                                                  # reads them
    assert "prepared folder" in capsys.readouterr().out
    assert _replies(b) == want
    del b
    torch.cuda.empty_cache()


def test_calibration_cache(ckpt, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GLM53_TF_CALIB_DIR", str(tmp_path / "calib"))
    monkeypatch.delenv("GLM53_TF_PREPARED", raising=False)
    a = _engine(ckpt, "cached")                                        # nothing stored: measures, stores
    out = capsys.readouterr().out
    assert a.costs["calib"] == "real" and "calibration table stored" in out and "[boot] r0" in out
    files = list((tmp_path / "calib").glob("calib-*.json"))
    assert len(files) == 1
    measured = dict(a.costs)
    del a
    torch.cuda.empty_cache()
    b = _engine(ckpt, "cached")                                        # the stored table, no timing runs
    assert b.costs["calib"] == "cached (real)"
    for k in ("verify", "mtp", "mtp_step", "mtp_row", "block", "taps_row", "timed", "windows"):
        assert b.costs[k] == measured[k], k
    assert "calibration table from" in capsys.readouterr().out
    _replies(b)                                                        # drafted == serial on the cached table
    del b
    torch.cuda.empty_cache()
    c = _engine(ckpt, "real")                                          # forced re-measure, refreshes the table
    assert c.costs["calib"] == "real"
    assert fastboot.calib_read(files[0].name)["timed"] == c.costs["timed"]
    del c
    torch.cuda.empty_cache()


def test_reader_throughput(tmp_path, monkeypatch):
    gb = float(os.environ.get("GLM53_TF_BOOT_BENCH_GB", "4"))
    if gb <= 0:
        pytest.skip("GLM53_TF_BOOT_BENCH_GB=0")
    root = Path(os.environ.get("BOOT_TEST_DIR", "") or tmp_path) / f"boot-bench-{os.getpid()}"
    n = int(gb * 1e9) // (256 << 20)
    # a tree like the model's: big stacked expert tensors and many small ones
    tree = {"big": [torch.randint(-2 ** 31, 2 ** 31 - 1, (64 << 20,), dtype=torch.int32, device="cuda")
                    for _ in range(n)],
            "small": [torch.randn(1000 + i, dtype=torch.bfloat16, device="cuda") for i in range(2000)]}
    key = {"format": fastboot.FORMAT, "what": "weights", "model": "bench", "rank": 0}
    try:
        fastboot.save(tree, root, key)
        del tree
        torch.cuda.empty_cache()
        for threads in (1, 4, 8, 16):
            monkeypatch.setenv("GLM53_TF_PREPARED_VERIFY", "off")
            obj, st = fastboot.read(root, "cuda", threads=threads)
            print(f"\n[reader] {st['bytes'] / 1e9:.1f} GB, {threads} threads, "
                  f"{'O_DIRECT' if st['direct'] else 'buffered'}: {st['seconds']:.2f}s = {st['gbps']:.2f} GB/s")
            del obj
            torch.cuda.empty_cache()
        monkeypatch.setenv("GLM53_TF_PREPARED_VERIFY", "sample")
        obj, st = fastboot.read(root, "cuda")
        print(f"[reader] default (8 threads, sampled checksums): {st['gbps']:.2f} GB/s, {st['verified']} chunks "
              "verified")
        assert st["gbps"] > 0.5
    finally:
        import shutil

        shutil.rmtree(root, ignore_errors=True)
