"""Prepared rank folders (patches/0140, ``glm5_next/cuda/fastboot.py``) on the CPU: what a start reads back from a
prepared folder is bit for bit what ``weights.load_checkpoint`` (and the DFlash2 drafter's ``read_weights``) built,
for both ranks and every GLM53_TF_NONEXPERT mode, on the synthetic EXL3 checkpoint of upstream's engine test
(``_checkpoint(exl3=True)``); a changed key misses the folder; a corrupted chunk is caught and the start builds
from the checkpoint instead. Needs torch (CPU is enough), no GPU. The same round trip on the GPU (where the serving
bits are made) is in tests/cuda/test_boot_patches.py.

Run against the patched tree (TF_TREE: its root, for tests/cuda/test_glm_engine.py's synthetic checkpoint):
    TF_TREE=<tree> PYTHONPATH=<tree>/src pytest -q tests/test_fastboot_prepared.py
In the image: TF_TREE=/src/TensorFold (the default when it exists).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from tensorfold.families.glm5_next.cuda import fastboot, weights  # noqa: E402

HERE = Path(__file__).resolve().parent


def _engine_test_module():
    """Upstream's tests/cuda/test_glm_engine.py (for ``_checkpoint`` / ``_drafter``), imported on a CPU-only
    machine too (it skips itself at import without CUDA; nothing it runs at import needs a GPU)."""

    roots = [os.environ.get("TF_TREE", ""), "/src/TensorFold", str(HERE.parent / "vendor" / "TensorFold")]
    for root in filter(None, roots):
        path = Path(root) / "tests" / "cuda" / "test_glm_engine.py"
        if path.exists():
            break
    else:
        pytest.skip("tests/cuda/test_glm_engine.py not found (set TF_TREE)")
    real = torch.cuda.is_available
    torch.cuda.is_available = lambda: True
    try:
        spec = importlib.util.spec_from_file_location("_tf_glm_engine_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        torch.cuda.is_available = real
    return mod


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    mod = _engine_test_module()
    d = tmp_path_factory.mktemp("ckpt")
    mod._checkpoint(d / "exl3", exl3=True)
    mod._drafter(d / "draft")
    return d / "exl3", d / "draft"


def same(a, b, path="w", shared=None) -> None:
    """Same structure, same types, same non-tensor values; every tensor with the same dtype, shape, stride, storage
    offset and storage bytes, and storages shared exactly where the original shares them."""

    shared = {} if shared is None else shared
    assert type(a) is type(b), f"{path}: {type(a)} != {type(b)}"
    if isinstance(a, torch.Tensor):
        assert (a.dtype, tuple(a.shape), a.stride(), a.storage_offset(), a.device.type) == \
               (b.dtype, tuple(b.shape), b.stride(), b.storage_offset(), b.device.type), path
        sa, sb = a.untyped_storage(), b.untyped_storage()
        ka, kb = (sa.data_ptr(), sa.nbytes()), (sb.data_ptr(), sb.nbytes())
        if ka in shared:
            assert shared[ka] == kb, f"{path}: storage sharing differs"
            return
        shared[ka] = kb
        ua = torch.empty(0, dtype=torch.uint8, device=a.device).set_(sa, 0, (sa.nbytes(),), (1,))
        ub = torch.empty(0, dtype=torch.uint8, device=b.device).set_(sb, 0, (sb.nbytes(),), (1,))
        assert torch.equal(ua.cpu(), ub.cpu()), f"{path}: bytes differ"
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), path
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{i}]", shared)
    elif isinstance(a, dict):
        assert list(a) == list(b), path
        for k in a:
            same(a[k], b[k], f"{path}.{k}", shared)
    elif hasattr(a, "__dataclass_fields__"):
        assert list(vars(a)) == list(vars(b)), path
        for k in vars(a):
            same(getattr(a, k), getattr(b, k), f"{path}.{k}", shared)
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


@pytest.fixture
def small_chunks(monkeypatch):
    """64 KiB reader chunks, so the synthetic model's storages cross chunk boundaries and several threads read."""

    monkeypatch.setattr(fastboot, "CHUNK", 64 << 10)


@pytest.mark.parametrize("nonexpert", ["bf16", "q4", "q4mse", "q8"])          # q8: patches/0470
@pytest.mark.parametrize("rank", [0, 1])
def test_weights_roundtrip(synthetic, tmp_path, monkeypatch, small_chunks, rank, nonexpert):
    ckpt, _ = synthetic
    monkeypatch.setattr(weights, "NONEXPERT", nonexpert)
    ref = weights.load_checkpoint(ckpt, rank=rank, device="cpu")
    monkeypatch.setenv("GLM53_TF_PREPARED", str(tmp_path))
    monkeypatch.setenv("GLM53_TF_PREPARED_WRITE", "1")
    monkeypatch.setenv("GLM53_TF_PREPARED_VERIFY", "full")
    first = weights.load(ckpt, rank=rank, device="cpu")          # builds, writes the folder
    same(ref, first)
    key = fastboot.weights_key(ckpt, rank, device="cpu")
    d = fastboot.folder(tmp_path, key)
    assert fastboot.valid(d, key) == (True, "ok")
    monkeypatch.setattr(weights, "load_checkpoint", lambda *a, **k: pytest.fail("built instead of read"))
    for direct in ("1", "0"):                                    # O_DIRECT (where the filesystem has it) and buffered
        monkeypatch.setenv("GLM53_TF_PREPARED_DIRECT", direct)
        again = weights.load(ckpt, rank=rank, device="cpu")
        same(ref, again)
        assert again.nbytes() == ref.nbytes()


def test_drafter_roundtrip(synthetic, tmp_path, monkeypatch, small_chunks):
    from tensorfold.families.glm5_next.cuda.dflash2 import Drafter

    ckpt, draft = synthetic
    w = weights.load_checkpoint(ckpt, rank=1, device="cpu")
    ref = Drafter(draft, w, capacity=64)
    monkeypatch.setenv("GLM53_TF_PREPARED", str(tmp_path))
    monkeypatch.setenv("GLM53_TF_PREPARED_WRITE", "1")
    monkeypatch.setenv("GLM53_TF_PREPARED_VERIFY", "full")
    Drafter(draft, w, capacity=64)                               # builds, writes
    # (not read_weights itself: its source is in the key, so replacing it is another key)
    from tensorfold.families.glm5_next.cuda import dflash2

    monkeypatch.setattr(dflash2, "safe_open", lambda *a, **k: pytest.fail("built instead of read"))
    got = Drafter(draft, w, capacity=64)
    for name in ("fc", "hidden_norm", "norm", "hproj", "pred", "succ", "layers"):
        same(getattr(ref, name), getattr(got, name), name)
    assert got.nbytes() == ref.nbytes()


def test_key_change_misses_and_corruption_falls_back(synthetic, tmp_path, monkeypatch, small_chunks, capsys):
    ckpt, _ = synthetic
    monkeypatch.setattr(weights, "NONEXPERT", "q4mse")
    monkeypatch.setenv("GLM53_TF_PREPARED", str(tmp_path))
    monkeypatch.setenv("GLM53_TF_PREPARED_WRITE", "1")
    ref = weights.load(ckpt, rank=0, device="cpu")
    key = fastboot.weights_key(ckpt, 0, device="cpu")
    d = fastboot.folder(tmp_path, key)
    # another NONEXPERT mode, or a patch that changes the code building the weights: another key, another folder
    monkeypatch.setattr(weights, "NONEXPERT", "q4")
    assert fastboot.folder(tmp_path, fastboot.weights_key(ckpt, 0, device="cpu")) != d
    monkeypatch.setattr(weights, "NONEXPERT", "q4mse")
    monkeypatch.setattr(fastboot, "WEIGHT_CODE", fastboot.WEIGHT_CODE + [f"{fastboot.PKG}.qmm:dequantize"])
    assert fastboot.weights_key(ckpt, 0, device="cpu")["code"] != key["code"]
    monkeypatch.setattr(fastboot, "WEIGHT_CODE", fastboot.WEIGHT_CODE[:-1])
    # flip a byte in the first chunk (always verified): the read fails, the start builds from the checkpoint
    monkeypatch.setenv("GLM53_TF_PREPARED_WRITE", "0")
    data = d / "data.bin"
    raw = bytearray(data.read_bytes())
    raw[100] ^= 0xFF
    data.write_bytes(bytes(raw))
    built = []
    real = weights.load_checkpoint
    monkeypatch.setattr(weights, "load_checkpoint", lambda *a, **k: built.append(1) or real(*a, **k))
    got = weights.load(ckpt, rank=0, device="cpu")
    assert built == [1] and "checksum mismatch" in capsys.readouterr().err
    same(ref, got)
    # `verify` finds it too
    assert fastboot.main(["verify", str(d)]) == 1
