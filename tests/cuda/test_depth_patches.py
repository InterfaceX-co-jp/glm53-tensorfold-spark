"""patches/0071 (cost-derived draft depth, ``glm5_next/cuda/depth.py``): host-only tests of the depth optimizer,
and the new policies on TensorFold's synthetic GLM checkpoint (one GPU playing rank 0 of two).

Depth only chooses which drafts a window verifies; the window, keyed sampler and commit decide every token, so
every reply must equal serial decoding. The ranks must also choose the same depths: every input of the optimizer
(the drafters' probabilities from gathered candidates, the committed tokens, the request's cost table) is the same
on both, and nothing reads a clock. Checked here:

- host only (no GPU needed): E(k), the surplus maximizer against brute force, deeper drafting as rows get cheaper
  or drafts surer, shallower as the running rate rises; the per-position correction learns an overconfident
  drafter; MTP chain decisions stop and go as they should; two optimizers fed the same rounds decide alike with
  every clock disabled; the specs (o, om[N], of[N]) and GLM53_TF_DEPTH;
- GPU: o, om, of (and N variants) and ``auto`` under GLM53_TF_DEPTH=cost equal serial decoding, sampled and
  greedy, with and without prompt-lookup matches, with and without the DFlash2 drafter; a follower engine that
  replays rank 0's headers (with its own GLM53_TF_DEPTH set differently) runs exactly rank 0's windows.

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda pytest -q tests/cuda/test_depth_patches.py
"""

from __future__ import annotations

import itertools
import time

import pytest

try:
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:          # the host-only tests still run
    CUDA = False

from tensorfold.families.glm5_next.cuda import depth  # noqa: E402

gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")

# the real model's calibration (docs/DECODE-ANALYSIS.md): verify windows of 1..8 rows, MTP 2.0 ms + 1.7 a chained
# draft, a DFlash2 block 3.9 ms
COSTS = {"verify": [30.0, 40.0, 44.7, 49.5, 54.2, 58.9, 63.7, 68.4], "mtp": 2.0, "mtp_step": 1.7, "mtp_row": 0.1,
         "block": 3.9, "taps_row": 0.05}


# -- host only -----------------------------------------------------------------------------------------------------
def test_expected_tokens():
    assert depth.expected_tokens([]) == [1.0]
    assert depth.expected_tokens([0.5, 0.5]) == [1.0, 1.5, 1.75]
    e = depth.expected_tokens([0.74, 0.45 / 0.74, 0.22 / 0.45])
    assert e[3] == pytest.approx(2.41)                        # chat greedy MTP c3: 1 + 0.74 + 0.45 + 0.22


def test_best_k_is_the_brute_force_surplus_maximum():
    v = COSTS["verify"]
    for qs, rate, row in itertools.product(([0.9] * 7, [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], [0.99, 0.1, 0.99]),
                                           (0.01, 0.023, 0.05, 0.2), (0.0, 0.1)):
        k, s = depth.best_k(qs, v, rate, row_ms=row)
        es = depth.expected_tokens(qs)
        brute = max(range(1, len(qs) + 1), key=lambda j: (es[j] - rate * (v[j] + row * es[j]), -j))
        assert k == brute and s == pytest.approx(es[k] - rate * (v[k] + row * es[k]))
    assert depth.best_k([0.0] * 7, v, 0.05)[0] == 1           # the first draft is always verified
    assert depth.best_k([0.0] * 7, v, 0.05, least=0)[0] == 0
    assert depth.best_k([0.5], v, 0.05)[0] == 1               # bounded by the drafts there are
    assert depth.best_k([0.9] * 7, v[:4], 0.001)[0] == 3      # and by the window table


def test_depth_moves_the_right_way():
    v = COSTS["verify"]
    sure, unsure = [0.95] * 7, [0.6] * 7
    rate = 1 / 22.7                                           # 44 tok/s
    assert depth.best_k(sure, v, rate)[0] > depth.best_k(unsure, v, rate)[0]
    assert depth.best_k(unsure, v, rate / 2)[0] >= depth.best_k(unsure, v, rate)[0]
    assert depth.best_k(unsure, v, rate * 2)[0] <= depth.best_k(unsure, v, rate)[0]
    cheap = [30.0 + 3.0 * r for r in range(8)]
    assert depth.best_k(unsure, cheap, rate)[0] >= depth.best_k(unsure, v, rate)[0]
    # break-even at 44 tok/s: a DFlash2 row (4.7 ms) pays for a marginal expected token above ~0.21
    assert depth.best_k([0.9, 0.9, 0.3], v, rate)[0] == 3     # 0.9 * 0.9 * 0.3 = 0.243 > 0.207
    assert depth.best_k([0.9, 0.9, 0.2], v, rate)[0] == 2     # 0.162 < 0.207


def test_calibration_learns_an_overconfident_drafter():
    cal = depth.Calibration()
    assert cal.factor("m", 0) == 1.0 and cal.q("m", 0, 0.9) == pytest.approx(0.9)
    for i in range(400):                                      # says 0.9, is right half the time
        cal.record("m", [0.9, 0.9], 2 if i % 2 else 1)
    assert cal.q("m", 0, 0.9) == pytest.approx(0.5, abs=0.05)
    assert cal.factor("f", 0) == 1.0                          # per drafter
    assert cal.tried["m"][1] > 0 and cal.kept["m"][1] == 0    # position 2 reached only after a kept first draft
    assert cal.q("m", 0, 5.0) <= depth.Q_MAX and cal.q("m", 0, -1.0) == 0.0


def test_mtp_chain_stops_and_goes():
    o = depth.DepthOptimizer(COSTS)
    o.mtp_begin()
    assert o.mtp_next(0, 0.01, 7) == (True, False)            # the first draft stays; a hopeless chain stops
    o.mtp_begin()
    assert o.mtp_next(0, 0.99, 7) == (True, True)
    take, more = o.mtp_next(1, 0.99, 7)
    assert take
    assert o.mtp_next(2, 0.01, 7) == (False, False)           # a draft not worth its row is left out
    assert o.used == [0.99, 0.99]
    o.record("m", 3, 2, 1, 3)
    assert len(o.rounds) == 1 and o.used == []
    o.mtp_begin()
    assert o.mtp_next(0, 0.99, 1) == (True, False)            # count bounds the chain


def _simulate(o: depth.DepthOptimizer, seed: int) -> list:
    """Rounds of every drafter with pseudo-random probabilities and outcomes (the committed tokens' stand-in)."""

    import random

    rng = random.Random(seed)
    trace = []
    for r in range(300):
        arm = "mfl"[r % 3]
        probs = [rng.random() for _ in range(7)]
        if arm == "m":
            o.mtp_begin()
            k = 0
            for j in range(7):
                take, more = o.mtp_next(j, probs[j], 7)
                k += take
                if not (take and more):
                    break
            steps = k
        elif arm == "f":
            k, steps = o.f_depth(probs), 0
        else:
            k, _ = o.lookup_depth(probs[0], 7, 0.1)
            steps = 0
        keep = 1 + next((j for j in range(k) if rng.random() > probs[j]), k)
        o.record(arm, k + 1, steps, 1, keep)
        trace.append((arm, k, keep, o.rate()))
    return trace


def test_two_ranks_decide_alike_without_a_clock(monkeypatch):
    def no_clock(*a):
        raise AssertionError("the depth choice read a clock")

    monkeypatch.setattr(time, "perf_counter", no_clock)
    monkeypatch.setattr(time, "time", no_clock)
    monkeypatch.setattr(time, "monotonic", no_clock)
    rank0, rank1 = _simulate(depth.DepthOptimizer(COSTS), 5), _simulate(depth.DepthOptimizer(dict(COSTS)), 5)
    assert rank0 == rank1
    assert len({k for _, k, _, _ in rank0}) > 2               # the depth really varies


def test_specs(monkeypatch):
    from tensorfold.families.glm5_next.cuda.engine import encode_policy

    assert encode_policy("o") == encode_policy("auto")        # auto's code; the header carries the depth flag
    assert encode_policy("om") == [depth.OPT_KIND, 7, 0, 0]
    assert encode_policy("of") == [depth.OPT_KIND, 7, 1, 0]
    assert encode_policy("om3") == [depth.OPT_KIND, 3, 0, 0]
    assert encode_policy("of5") == [depth.OPT_KIND, 5, 1, 0]
    for bad in ("om0", "of8", "ox", "omx", "o3"):
        with pytest.raises(ValueError):
            encode_policy(bad)
    assert depth.as_cost(encode_policy("fc5:0.3")) == [depth.OPT_KIND, 5, 1, 0]
    assert depth.as_cost(encode_policy("c3:0.35")) == [depth.OPT_KIND, 3, 0, 0]
    assert depth.as_cost(encode_policy("auto")) == encode_policy("auto")
    monkeypatch.delenv("GLM53_TF_DEPTH", raising=False)
    assert depth.env_cost() is False
    monkeypatch.setenv("GLM53_TF_DEPTH", "cost")
    assert depth.env_cost() is True
    monkeypatch.setenv("GLM53_TF_DEPTH", "bogus")
    with pytest.raises(ValueError):
        depth.env_cost()


# -- GPU: the synthetic checkpoint ---------------------------------------------------------------------------------
POLICIES = ("o", "om", "of", "om3", "of5", "auto")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_depth")
    _checkpoint(path / "model")
    _drafter(path / "dflash2")
    return path


def _engine(ckpt, drafter: bool = True):
    from test_glm_engine import _TwoCopies

    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("GLM53_TF_BATCH", raising=False)
        return GlmEngine(ckpt / "model", rank=0, master="", port=0, drafter=ckpt / "dflash2" if drafter else None,
                         comm=_TwoCopies())


@pytest.fixture(scope="module")
def ef(ckpt):
    return _engine(ckpt)


@pytest.fixture(scope="module")
def em(ckpt):
    return _engine(ckpt, drafter=False)


def _prompts():
    import numpy as np

    b = [int(t) for t in np.random.default_rng(52).integers(0, 1000, size=30)]
    return [list(np.random.default_rng(51).integers(0, 1000, size=37)), b * 3 + b[:8]]


class _Windows:
    def __init__(self, e):
        self.e, self.orig, self.seen = e.e, e.e.forward, []

        def forward(tokens):
            self.seen.append(len(tokens))
            return self.orig(tokens)

        self.e.forward = forward


@gpu
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
@pytest.mark.parametrize("which", ["drafter", "mtp-only"])
def test_cost_depth_equals_serial(ef, em, monkeypatch, greedy, which):
    from tensorfold.engine.exact_sampling import Sampling
    from test_glm_engine import _generate

    e = ef if which == "drafter" else em
    sampling = None if greedy else Sampling(1234, 1.0, 20, 0.95)
    monkeypatch.setenv("GLM53_TF_DEPTH", "cost")              # makes plain auto cost-derived too
    for prompt in _prompts():
        serial, _ = _generate(e, prompt, sampling, draft=False, tokens=40)
        for policy in POLICIES:
            w = _Windows(e)
            drafted, stats = _generate(e, prompt, sampling, policy=policy, tokens=40)
            del e.e.forward
            assert drafted == serial, (which, policy)
            assert stats["rounds"] >= 1 and stats["min_rows"] >= 2, (policy, stats)   # every round a window
            assert max(w.seen) <= 8
            assert e.e.depth is None                          # the optimizer does not outlive its request
    monkeypatch.setenv("GLM53_TF_DEPTH", "threshold")
    for policy in ("auto", "c3:0.35", "fc5:0.3"):             # upstream's thresholds still there
        drafted, _ = _generate(e, _prompts()[0], sampling, policy=policy, tokens=40)
        assert drafted == _generate(e, _prompts()[0], sampling, draft=False, tokens=40)[0], policy


class _Done(Exception):
    pass


@gpu
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
def test_follower_runs_rank0_windows(ckpt, monkeypatch, greedy):
    """Rank 0 (GLM53_TF_DEPTH=cost) serves; a second engine replays its headers through ``follow`` as rank 1 does
    while its own environment says ``threshold``. ``_TwoCopies`` makes both engines' model rank 0's, so any
    difference in the windows would be a decision the ranks made differently."""

    from tensorfold.engine.exact_sampling import Sampling
    from test_glm_engine import _generate

    r0, r1 = _engine(ckpt), _engine(ckpt)
    # On the cluster ``_calibrate`` all-gathers both ranks' timings and keeps the slower of each, so both ranks load
    # the same costs. Here each engine's ``_TwoCopies`` gathers only its own timings, so the two engines would hold
    # slightly different costs (a few microseconds), which is enough to tip a depth choice near a tie: give the
    # follower rank 0's load-time costs, as the cluster's all-gather does.
    r1.base_costs = r1.costs = r0.base_costs
    sent: list[list[int]] = []
    share = r0._share

    def record(values):
        got = share(values)
        sent.append(list(got))
        return got

    def replay(values):
        if not sent:
            raise _Done
        return sent.pop(0)

    r0._share, r1._share = record, replay
    w0, w1 = _Windows(r0), _Windows(r1)
    sampling = None if greedy else Sampling(77, 1.0, 20, 0.95)
    prompts = _prompts()
    for prompt, policy in [(prompts[0], "auto"), (prompts[1], "auto"), (prompts[1] + [3], "o"),
                           (prompts[0] + [4], "om"), (prompts[1] + [5], "of"), (prompts[0] + [6], "auto:1:1:0")]:
        monkeypatch.setenv("GLM53_TF_DEPTH", "cost")
        _generate(r0, prompt, sampling, policy=policy, tokens=40)
        monkeypatch.setenv("GLM53_TF_DEPTH", "threshold")
        with pytest.raises(_Done):
            r1.follow()
        assert r1.depth_cost == r0.depth_cost == 1, policy
        assert w1.seen == w0.seen, policy
    del r0, r1
    torch.cuda.empty_cache()
