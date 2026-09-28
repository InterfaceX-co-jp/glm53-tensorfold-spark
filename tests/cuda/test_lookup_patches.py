"""Prompt-lookup drafts (patches/0020) on TensorFold's synthetic GLM checkpoint (one GPU playing rank 0 of two).

The lookup's drafts go through the same verify window, keyed sampler and commit as MTP and DFlash2 drafts, so a
reply must equal serial decoding whatever is proposed: ``lN`` (lookup, MTP rounds without a match) and ``auto``
(lookup gated by the timed costs) on prompts that repeat a block, greedy and sampled. An oracle lookup that
proposes serial decoding's own tokens (and, some rounds, a wrong one) checks the accept path itself: every right
draft kept, a window cut at the first wrong one, long lookup streaks that the MTP head and DFlash2 then catch up
on, and a state left by them that resumes like a fresh prefill. The matcher and the decision are unit-tested
without a GPU in tests/test_glm_lookup.py.

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda pytest -q tests/cuda/test_lookup_patches.py
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.glm5_next.cuda import engine as engine_mod  # noqa: E402
from test_glm_engine import V, _TwoCopies, _checkpoint, _drafter, _generate  # noqa: E402

SAMPLINGS = [Sampling(1234, 1.0, 20, 0.95), None]
IDS = ["sampled", "greedy"]


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm_lookup")
    _checkpoint(path / "model")
    _drafter(path / "dflash2")
    return path


@pytest.fixture(scope="module")
def ef(ckpt):
    """MLX-layout model with the MTP head and the DFlash2 drafter: ``auto`` chooses between them and the lookup."""

    return engine_mod.GlmEngine(ckpt / "model", rank=0, master="", port=0, drafter=ckpt / "dflash2",
                                comm=_TwoCopies())


@pytest.fixture
def env(monkeypatch):
    monkeypatch.delenv("GLM53_TF_LOOKUP", raising=False)
    monkeypatch.delenv("GLM53_TF_LOOKUP_MIN", raising=False)
    return monkeypatch


def _repeated(seed: int, block: int = 40, times: int = 3, tail: int = 10) -> list[int]:
    """A random block repeated ``times`` times, then its first ``tail`` tokens (the end-of-sequence id 1000 kept
    out)."""

    b = [int(t) for t in np.random.default_rng(seed).integers(0, 1000, size=block)]
    return b * times + b[:tail]


class _Rows:
    """The width of every verify window the engine runs."""

    def __init__(self, e):
        self.e, self.orig, self.seen = e, e.forward, []

    def __enter__(self):
        def forward(tokens):
            self.seen.append(len(tokens))
            return self.orig(tokens)

        self.e.forward = forward
        return self

    def __exit__(self, *exc):
        del self.e.forward


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
def test_lookup_policies_equal_serial(ef, env, sampling):
    prompt = _repeated(51)
    serial, _ = _generate(ef, prompt, sampling, draft=False, tokens=64)
    seen = ""
    for policy in ("l7", "l7:1", "l3", "l1:2", "l7:8", None, "auto", "auto:1:1:0", "auto:1:2:0"):
        with _Rows(ef.e) as rows:
            drafted, stats = _generate(ef, prompt, sampling, policy=policy, tokens=64)
        assert drafted == serial, policy
        assert max(rows.seen) <= 8 and stats["min_rows"] >= 2, (policy, rows.seen, stats)
        if policy is not None and policy.startswith("l"):
            assert set(stats["drafters"]) <= {"l", "m"}, (policy, stats)      # lN never runs DFlash2
        seen += stats.get("drafters", "")
    assert "l" in seen                  # a match of 1 token (l7:1) turns up within 64 tokens


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
def test_auto_lookup_knobs(ef, env, sampling):
    """``auto`` with the lookup off, on (default) and on with any match: same reply; off never drafts "l"."""

    prompt = _repeated(52, block=24, times=4, tail=6)
    serial, _ = _generate(ef, prompt, sampling, draft=False, tokens=48)
    codes = []
    real = engine_mod.lookup_for

    def spy(code, *a, **k):
        codes.append(list(code))
        return real(code, *a, **k)

    env.setattr(engine_mod, "lookup_for", spy)
    for on, least in (("0", "4"), ("1", "4"), ("1", "1")):
        env.setenv("GLM53_TF_LOOKUP", on)
        env.setenv("GLM53_TF_LOOKUP_MIN", least)
        drafted, stats = _generate(ef, prompt, sampling, policy="auto", tokens=48)
        assert drafted == serial, (on, least)
        assert codes[-1] == engine_mod.encode_policy("auto") + [int(on), int(least)]    # in the request header
        if on == "0":
            assert "l" not in stats.get("drafters", "")


def test_same_request_same_decisions(ef, env):
    """Every decision is a function of committed tokens and load-time costs: the same request drafts alike."""

    sampling = Sampling(9, 1.0, 20, 0.95)
    prompt = _repeated(53)
    for policy in ("l7:1", "auto"):
        a, sa = _generate(ef, prompt, sampling, policy=policy, tokens=40)
        b, sb = _generate(ef, prompt, sampling, policy=policy, tokens=40)
        assert a == b and sa.get("drafters") == sb.get("drafters") and sa.get("keeps") == sb.get("keeps")


class _Oracle:
    """A lookup that proposes serial decoding's own next tokens: "all" right, "bad" wrong at the third draft,
    "none" no drafts (the round falls back to the MTP head / DFlash2), in turn."""

    def __init__(self, truth: list[int], pattern: tuple[str, ...]):
        self.truth, self.pattern = truth, pattern
        self.rounds = 0
        self.planned: list[tuple[str, list[int]]] = []
        self.log: list[tuple[str, int, int]] = []

    def plan(self, out, room):
        mode = self.pattern[self.rounds % len(self.pattern)]
        self.rounds += 1
        i = len(out)
        assert list(out) == self.truth[:i]                   # the loop's tokens are serial decoding's
        drafts = [] if mode == "none" else list(self.truth[i:i + min(7, room)])
        if mode == "bad" and len(drafts) > 2:
            drafts[2] = drafts[2] + 1 if drafts[2] + 1 < V else drafts[2] - 1
        self.planned.append((mode, drafts))
        return drafts

    def record(self, arm, rows, steps, backlog, keep):
        self.log.append((arm, rows, keep))


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
@pytest.mark.parametrize("policy,pattern", [
    ("l7", ("all",)),                                        # 300 tokens of lookup only: the MTP backlog overflows
    ("l7", ("all", "bad", "none", "all", "all", "none")),
    ("auto:1:1:0", ("all", "all", "none", "bad", "none", "all", "none")),
])
def test_oracle_lookup_keeps_right_drafts(ef, env, sampling, policy, pattern):
    prompt = _repeated(54)
    tokens = 300 if pattern == ("all",) else 96
    serial, _ = _generate(ef, prompt, sampling, draft=False, tokens=tokens)
    oracle = _Oracle(serial, pattern)
    with pytest.MonkeyPatch.context() as mp, _Rows(ef.e) as rows:
        mp.setattr(engine_mod, "lookup_for", lambda *a, **k: oracle)
        drafted, stats = _generate(ef, prompt, sampling, policy=policy, tokens=tokens)
    assert drafted == serial
    assert max(rows.seen) <= 8
    arms = stats["drafters"]
    lookups = [(mode, d) for mode, d in oracle.planned if d]
    assert arms.count("l") == len(lookups) >= 1
    kept = [keep for arm, _, keep in oracle.log if arm == "l"]
    for (mode, d), keep in zip(lookups, kept):
        assert keep == (len(d) + 1 if mode == "all" or len(d) <= 2 else 3), (mode, d, keep)
    if "none" in pattern:
        assert set(arms) - {"l"}, arms                       # the other drafters drafted between lookup rounds


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
def test_state_after_lookup_streak_resumes(ef, env, sampling):
    """A reply drafted by long lookup streaks (the MTP head's backlog past its buffer) leaves a state that resumes
    like a fresh prefill for MTP and lookup policies."""

    prompt = _repeated(55)
    rng = np.random.default_rng(56)
    unrelated = [int(t) for t in rng.integers(0, 1000, size=9)]
    serial, _ = _generate(ef, prompt, sampling, draft=False, tokens=300)

    def lookup_reply():
        oracle = _Oracle(serial, ("all",))
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(engine_mod, "lookup_for", lambda *a, **k: oracle)
            reply, _ = _generate(ef, prompt, sampling, policy="l7", tokens=300)
        assert reply == serial
        return reply

    reply = lookup_reply()
    after = prompt + reply + [21, 22]
    for policy in ("2", "c3:0.35", "l7", "l7:1"):
        warm, stats = _generate(ef, after, sampling, policy=policy, tokens=24)
        assert stats["cached"] >= len(prompt) + len(reply) - 1, policy
        _generate(ef, unrelated, sampling)
        cold, stats = _generate(ef, after, sampling, policy=policy, tokens=24)
        assert stats["cached"] == 0 and warm == cold, policy
        lookup_reply()                                       # the state after the lookup reply again


@pytest.fixture(scope="module")
def en(tmp_path_factory):
    """A checkpoint without the MTP head (DFlash2 drafts only)."""

    path = tmp_path_factory.mktemp("glm_lookup_n")
    _checkpoint(path / "model", mtp=False)
    _drafter(path / "dflash2")
    return engine_mod.GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2",
                                comm=_TwoCopies())


def test_no_mtp_head_runs_lookup_spec_as_dflash2(en, env):
    sampling = Sampling(77, 1.0, 20, 0.95)
    prompt = _repeated(57)
    serial, _ = _generate(en, prompt, sampling, draft=False, tokens=32)
    for policy in ("l7", "l3:1", "auto"):
        drafted, stats = _generate(en, prompt, sampling, policy=policy, tokens=32)
        assert drafted == serial, policy
        assert stats["min_rows"] >= 2 and not set(stats.get("drafters", "")) & {"m", "l"}, (policy, stats)
