"""Our decode patches on TensorFold's synthetic GLM checkpoint (one GPU playing rank 0 of two).

patches/0010  GLM53_TF_AUTO_FDRAFTS: the default policy's DFlash2 rounds verify up to 7 drafts (upstream 5), still
              cut where the drafts' probability product falls under 0.3. Drafts only propose, so every reply
              must equal serial decoding whatever the depth; windows of 7 and 8 rows (eager, past GRAPH_ROWS) must
              give the bits of serial steps, and a state left by deep rounds must resume like a fresh prefill.

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda pytest -q tests/cuda/test_decode_patches.py
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from test_glm_engine import _TwoCopies, _checkpoint, _drafter, _generate  # noqa: E402


@pytest.fixture(autouse=True)
def _no_lookup(monkeypatch):
    """patches/0020 adds lookup rounds to ``auto``; these tests pin the MTP/DFlash2 arms and window sizes."""

    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")

SAMPLINGS = [Sampling(1234, 1.0, 20, 0.95), None]
IDS = ["sampled", "greedy"]
POLICIES = (None, "auto", "auto:1:1:0", "auto:1:2:0", "f7", "fc7:0.3", "fc5:0.3", "7", "c3:0.35", "a:0.6:0.85", "2")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm_decode")
    _checkpoint(path / "model")
    _drafter(path / "dflash2")
    return path


@pytest.fixture(scope="module")
def ef(ckpt):
    """MLX-layout model with the DFlash2 drafter (``auto`` chooses between drafters), default knob."""

    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    mp = pytest.MonkeyPatch()
    mp.delenv("GLM53_TF_AUTO_FDRAFTS", raising=False)
    try:
        return GlmEngine(ckpt / "model", rank=0, master="", port=0, drafter=ckpt / "dflash2", comm=_TwoCopies())
    finally:
        mp.undo()


class _Rows:
    """Records the width of every verify window the engine runs (instance attribute over ``Engine.forward``)."""

    def __init__(self, e):
        self.e, self.orig, self.seen = e, e.forward, []

    def __enter__(self):
        def forward(tokens):
            self.seen.append(len(tokens))
            return self.orig(tokens)

        self.e.forward = forward
        return self

    def __exit__(self, *exc):
        del self.e.forward                  # back to the class method
        return False


def test_knob_default_and_bounds(monkeypatch, ef):
    from tensorfold.families.glm5_next.cuda.engine import MAX_ROWS, auto_f_most

    assert ef.f_most == 7 == MAX_ROWS - 1
    monkeypatch.delenv("GLM53_TF_AUTO_FDRAFTS", raising=False)
    assert auto_f_most() == 7
    monkeypatch.setenv("GLM53_TF_AUTO_FDRAFTS", "5")
    assert auto_f_most() == 5
    for bad in ("0", "8"):
        monkeypatch.setenv("GLM53_TF_AUTO_FDRAFTS", bad)
        with pytest.raises(ValueError, match="GLM53_TF_AUTO_FDRAFTS"):
            auto_f_most()


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
def test_deep_drafts_equal_serial(ef, sampling):
    """Every policy, the deep DFlash2 ones included, equals serial decoding; ``f7`` really verifies 8-row windows."""

    prompt = list(np.random.default_rng(31).integers(0, 1000, size=43))
    serial, stats = _generate(ef, prompt, sampling, draft=False, tokens=48)
    assert len(serial) == 48 and stats["drafts"] is False
    for policy in POLICIES:
        with _Rows(ef.e) as rows:
            drafted, stats = _generate(ef, prompt, sampling, policy=policy, tokens=48)
        assert drafted == serial, policy
        assert stats["rounds"] >= 1 and stats["min_rows"] >= 2, (policy, stats)
        assert max(rows.seen) <= 8, (policy, rows.seen)
        if policy in ("f7", "7"):                                   # fixed depth: the first round is 8 rows
            assert rows.seen[0] == 8, (policy, rows.seen)


@pytest.fixture(scope="module")
def e5(ckpt):
    """The same engine with the knob at 5 (upstream's DFlash2 depth in ``auto``)."""

    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    mp = pytest.MonkeyPatch()
    mp.setenv("GLM53_TF_AUTO_FDRAFTS", "5")
    try:
        return GlmEngine(ckpt / "model", rank=0, master="", port=0, drafter=ckpt / "dflash2", comm=_TwoCopies())
    finally:
        mp.undo()


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
def test_auto_follows_knob(e5, ef, sampling):
    """``auto`` with the knob at 5 (upstream) and at 7 gives serial decoding's reply; at 5 no window is wider than
    6 rows (MTP arm at most 4, DFlash2 arm at most 6), at 7 none wider than 8."""

    assert e5.f_most == 5
    prompt = list(np.random.default_rng(32).integers(0, 1000, size=39))
    for eng, widest in ((e5, 6), (ef, 8)):
        serial, _ = _generate(eng, prompt, sampling, draft=False, tokens=40)
        for policy in (None, "auto:1:1:0"):
            with _Rows(eng.e) as rows:
                drafted, _ = _generate(eng, prompt, sampling, policy=policy, tokens=40)
            assert drafted == serial, (eng.f_most, policy)
            assert max(rows.seen) <= widest, (eng.f_most, policy, rows.seen)
    a, _ = _generate(e5, prompt, sampling, policy="auto:1:1:0", tokens=40)
    b, _ = _generate(ef, prompt, sampling, policy="auto:1:1:0", tokens=40)
    assert a == b                                                   # the depth never changes the reply


def test_eight_row_windows_match_serial_steps(ef):
    """A window of 8 rows (and 7) keeps the bits of serial steps: sampled tokens and the committed state (checked
    through the reply that follows) match one-row steps."""

    sampling = Sampling(5, 1.0, 20, 0.95)
    prompt = list(np.random.default_rng(33).integers(0, 1000, size=29))
    serial, _ = _generate(ef, prompt, sampling, draft=False, tokens=33)
    for policy in ("7", "6", "f7"):
        drafted, _ = _generate(ef, prompt, sampling, policy=policy, tokens=33)
        assert drafted == serial, policy


def test_deep_rounds_resume(ef):
    """A reply drafted with deep DFlash2 rounds leaves a state that both drafters resume from."""

    sampling = Sampling(41, 1.0, 20, 0.95)
    rng = np.random.default_rng(42)
    first = list(rng.integers(0, 1000, size=30))
    unrelated = list(rng.integers(0, 1000, size=9))
    # auto keeps both drafters' caches, so every policy below can resume from the state after this reply
    reply, stats = _generate(ef, first, sampling, policy="auto:1:1:0", tokens=30)
    assert set(stats["drafters"]) == {"m", "f"}
    after = first + reply + [21, 22]
    for policy in ("auto", "auto:1:1:0", "f7", "2"):
        warm, stats = _generate(ef, after, sampling, policy=policy)
        assert stats["cached"] >= len(first) + len(reply) - 1, policy
        _generate(ef, unrelated, sampling)
        cold, stats = _generate(ef, after, sampling, policy=policy)
        assert stats["cached"] == 0 and warm == cold, policy
        serial, _ = _generate(ef, after, sampling, draft=False)
        assert serial == cold, policy
        _generate(ef, first, sampling, policy="auto:1:1:0", tokens=30)   # the state after the reply again
