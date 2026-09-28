"""patches/0070 (realistic draft cost calibration, ``glm5_next/cuda/calib.py``) on TensorFold's synthetic GLM
checkpoint (one GPU playing rank 0 of two), plus host-only checks of the cost arithmetic.

Costs only choose how many drafts a round verifies and which drafter proposes them, so every reply must still
equal serial decoding; what must never happen is the two ranks choosing differently. Checked here:

- host only (no GPU needed): each piece's reconciled time is a function of the all-gathered list alone (the same
  on both ranks, whichever rank computes it); random mode's windows are upstream's fit; the online table is the
  same integers, and so the same floats, on both ranks, converges to measured windows, is non-decreasing and
  shrugs off a slow outlier; GLM53_TF_CALIB is validated;
- ``real`` calibration is repeatable: two loads prefill the same prompt and time windows on the same greedy
  continuation (the timings themselves are wall clock and differ between loads; what both ranks share within a
  load is the reconciled table), and ``random`` still times upstream's seeded ids;
- in both modes, and with online refinement, every policy's reply equals serial decoding (sampled and greedy);
- online refinement keeps rank decisions consistent: rank 0 (GLM53_TF_CALIB_ONLINE=1, its table skewed by fake
  observations) serves requests while a second engine replays its headers through ``follow`` as rank 1 would,
  with its own load-time verify costs deliberately different; both use rank 0's table and run exactly the same
  verify windows.

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda pytest -q tests/cuda/test_calib_patches.py
"""

from __future__ import annotations

import math

import pytest

try:
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:          # the host-only tests still run
    CUDA = False

from tensorfold.families.glm5_next.cuda import calib  # noqa: E402

gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")

# the real model's load-time table with random ids (ms), as docs/DECODE-ANALYSIS.md reports it
RANDOM_V = [30.0, 40.0, 44.7, 49.5, 54.2, 58.9, 63.7, 68.4]
POLICIES = (None, "auto", "auto:1:1:0", "c3:0.35", "fc5:0.3", "a:0.6:0.85", "l4")


# -- host only -----------------------------------------------------------------------------------------------------
def test_slower_is_a_function_of_the_gathered_list():
    names = ["v1", "v2", "m1", "block"]
    r0, r1 = [30.1, 40.2, 2.0, 3.9], [30.4, 39.9, 2.1, 3.8]
    on0 = calib.slower(names, r0 + r1)          # the all-gather hands both ranks this same list
    on1 = calib.slower(names, r0 + r1)
    assert on0 == on1 == {"v1": 30.4, "v2": 40.2, "m1": 2.1, "block": 3.9}
    assert calib.slower(names, r1 + r0) == on0  # and the rank order does not matter


def test_random_mode_windows_are_upstreams_fit():
    import statistics

    both = {f"v{r}": v for r, v in enumerate(RANDOM_V, 1)}
    rows = list(range(2, 9))
    ys = RANDOM_V[1:]
    slope = statistics.median((ys[j] - ys[i]) / (rows[j] - rows[i]) for i in range(len(rows))
                              for j in range(i + 1, len(rows)))
    base = statistics.median(y - slope * r for r, y in zip(rows, ys))
    assert calib.window_costs(both, 8, 0) == [RANDOM_V[0]] + [base + slope * r for r in rows]


def test_real_mode_windows_average_the_spans():
    both = {f"v{r}_{s}": 30.0 + (r - 1) * (4.0 + s) for r in range(1, 9) for s in range(3)}
    v = calib.window_costs(both, 8, 3)
    assert v[0] == pytest.approx(30.0) and v[1] == pytest.approx(35.0) and v[7] == pytest.approx(65.0)


def test_online_table_is_identical_on_both_ranks():
    o = calib.OnlineCosts(RANDOM_V, alpha=0.1, least=4)
    for i in range(300):
        r = 2 + i % 6
        o.observe(r, 30.0 + 4.0 * (r - 1) + (0.3 if i % 2 else -0.3))
    ints = o.table()
    assert all(isinstance(v, int) for v in ints)
    rank0 = calib.with_table({"verify": RANDOM_V, "mtp": 2.0}, list(ints))
    rank1 = calib.with_table({"verify": [99.0] * 8, "mtp": 2.0}, list(ints))    # its own table is ignored
    assert rank0 == rank1 and rank0["verify"] == calib.decode_table(ints)
    assert calib.encode_table(rank0["verify"]) == ints                          # a lossless round trip
    assert calib.with_table(rank0, []) is rank0                                 # no table: the load-time costs


def test_online_converges_is_monotone_and_ignores_outliers():
    o = calib.OnlineCosts(RANDOM_V, alpha=0.1, least=4)
    assert o.current() == RANDOM_V                                             # nothing measured yet
    for _ in range(200):
        for r in (2, 3, 4):
            o.observe(r, 30.0 + 4.0 * (r - 1))
    v = o.current()
    assert v[1] == pytest.approx(34.0, abs=0.2) and v[3] == pytest.approx(42.0, abs=0.2)
    assert v[0] == RANDOM_V[0]                                                 # serial windows: not measured here
    assert v[7] == pytest.approx(58.0, abs=0.5)                                # unmeasured sizes: on the line
    assert all(b >= a for a, b in zip(v, v[1:]))
    before = o.current()[2]
    o.observe(3, 5000.0)                                                       # a slow moment
    assert o.current()[2] <= before * 1.051
    o.observe(3, float("nan"))
    o.observe(0, 10.0)
    o.observe(9, 10.0)
    assert all(math.isfinite(x) for x in o.current())


def test_mode_and_prompt(monkeypatch, tmp_path):
    monkeypatch.delenv("GLM53_TF_CALIB", raising=False)
    monkeypatch.delenv("GLM53_TF_CALIB_ONLINE", raising=False)
    assert calib.mode() == "real" and calib.online() is False
    monkeypatch.setenv("GLM53_TF_CALIB", "random")
    assert calib.mode() == "random"
    monkeypatch.setenv("GLM53_TF_CALIB", "bogus")
    with pytest.raises(ValueError):
        calib.mode()
    monkeypatch.setenv("GLM53_TF_CALIB_ONLINE", "1")
    assert calib.online() is True
    ids = calib.prompt_ids(tmp_path, 1024)                                     # no tokenizer.json: fixed ids
    assert ids == calib.prompt_ids(tmp_path, 1024) == calib.fallback_ids(1024)
    assert all(0 < t < 1024 for t in ids)


# -- GPU: the synthetic checkpoint ---------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_calib")
    _checkpoint(path / "model")
    _drafter(path / "dflash2")
    return path


def _engine(ckpt, calib_mode: str, online: bool = False):
    from test_glm_engine import _TwoCopies

    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GLM53_TF_CALIB", calib_mode)
        mp.setenv("GLM53_TF_CALIB_ONLINE", "1" if online else "0")
        mp.setenv("GLM53_TF_CALIB_MIN", "2")        # short test replies: trust a size after two windows
        mp.delenv("GLM53_TF_BATCH", raising=False)
        return GlmEngine(ckpt / "model", rank=0, master="", port=0, drafter=ckpt / "dflash2", comm=_TwoCopies())


def _repeated(seed: int) -> list[int]:
    import numpy as np

    b = [int(t) for t in np.random.default_rng(seed).integers(0, 1000, size=30)]
    return b * 3 + b[:8]


def _check_costs(c: dict) -> None:
    assert len(c["verify"]) == 8 and all(math.isfinite(v) and v > 0 for v in c["verify"])
    for k in ("mtp", "mtp_step", "mtp_row", "block", "taps_row"):
        assert math.isfinite(c[k]) and c[k] >= 0, k


@gpu
def test_real_calibration_is_repeatable(ckpt):
    from tensorfold.families.glm5_next.cuda.engine import MAX_ROWS

    a = _engine(ckpt, "real")
    b = _engine(ckpt, "real")
    assert a.costs["calib"] == b.costs["calib"] == "real"
    assert a.costs["windows"] == b.costs["windows"]            # same prompt, same greedy continuation
    prompt, cont = a.costs["windows"]
    assert prompt == calib.fallback_ids(a.w.cfg.vocab)         # the synthetic checkpoint has no tokenizer
    assert len(cont) == calib.SPANS * MAX_ROWS
    assert set(a.costs["timed"]) == set(b.costs["timed"]) >= {f"v{r}_{s}" for r in range(1, 9)
                                                               for s in range(calib.SPANS)}
    for e in (a, b):
        _check_costs(e.costs)
        assert e.online is None and e.base_costs is e.costs
    del a, b
    torch.cuda.empty_cache()


@gpu
def test_random_calibration_is_upstreams(ckpt):
    import numpy as np

    e = _engine(ckpt, "random")
    assert e.costs["calib"] == "random"
    assert e.costs["windows"][0] == [int(t) for t in np.random.default_rng(0).integers(0, e.w.cfg.vocab, 64)]
    assert set(e.costs["timed"]) == {f"v{r}" for r in range(1, 9)} | {"m1", "m3", "m6", "block", "taps8"}
    _check_costs(e.costs)
    del e
    torch.cuda.empty_cache()


@gpu
@pytest.mark.parametrize("calib_mode,online", [("real", False), ("random", False), ("real", True)],
                         ids=["real", "random", "real-online"])
def test_drafted_equals_serial(ckpt, calib_mode, online):
    from tensorfold.engine.exact_sampling import Sampling
    from test_glm_engine import _generate

    import numpy as np

    e = _engine(ckpt, calib_mode, online)
    prompts = [list(np.random.default_rng(41).integers(0, 1000, size=37)), _repeated(42)]
    for sampling in (Sampling(1234, 1.0, 20, 0.95), None):
        for prompt in prompts:
            serial, _ = _generate(e, prompt, sampling, draft=False, tokens=32)
            for policy in POLICIES:
                drafted, stats = _generate(e, prompt, sampling, policy=policy, tokens=32)
                assert drafted == serial, (calib_mode, online, policy, sampling)
                if online:
                    assert len(stats["verify_ms"]) == 8
    if online:
        assert sum(e.online.n) > 0                              # auto_decode fed the online costs
    del e
    torch.cuda.empty_cache()


class _Done(Exception):
    pass


class _Windows:
    """The width of every verify window an engine runs."""

    def __init__(self, e):
        self.e, self.orig, self.seen = e.e, e.e.forward, []

        def forward(tokens):
            self.seen.append(len(tokens))
            return self.orig(tokens)

        self.e.forward = forward


@gpu
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
def test_online_keeps_rank_decisions_consistent(ckpt, greedy):
    """Rank 0 refines online and sends its table; a second engine replays rank 0's headers through ``follow`` as
    rank 1 does (``_TwoCopies`` makes its all-gathers rank 0's, so its model is rank 0's and every difference would
    be a decision). Its own load-time verify costs are set far off: it must still use rank 0's table and run the
    same windows, request after request, while that table changes."""

    from tensorfold.engine.exact_sampling import Sampling
    from test_glm_engine import _generate

    import numpy as np

    r0 = _engine(ckpt, "real", online=True)
    r1 = _engine(ckpt, "real", online=False)
    assert r0.online is not None and r1.online is None
    # load time: the all-gather makes every piece the slower rank's on both; simulate that, then skew rank 1's
    # verify windows so any use of its own table would show
    r1.base_costs = dict(r0.base_costs, verify=[v * 3.0 + 7.0 for v in r0.base_costs["verify"]])
    for _ in range(6):                                          # rank 0 has run cheap deep windows before
        for r in range(2, 9):
            r0.online.observe(r, r0.base_costs["verify"][0] * (1.0 + 0.05 * r))
    sent: list[list[int]] = []
    share = r0._share

    def record(values):
        got = share(values)
        sent.append(list(got))
        return got

    r0._share = record

    def replay(values):
        assert values is None
        if not sent:
            raise _Done
        return sent.pop(0)

    r1._share = replay
    w0, w1 = _Windows(r0), _Windows(r1)
    sampling = None if greedy else Sampling(99, 1.0, 20, 0.95)
    rng = np.random.default_rng(43)
    first = _repeated(44)
    reply, _ = _generate(r0, first, sampling, policy="auto", tokens=40)
    requests = [(first + reply + [5, 6], "auto"), (list(rng.integers(0, 1000, size=33)), "auto:1:1:0"),
                (_repeated(45), "auto"), (_repeated(45) + [7], "l4")]
    tables = []
    with pytest.raises(_Done):
        r1.follow()
    assert r1.costs == r0.costs and w1.seen == w0.seen
    for prompt, policy in requests:
        n0 = len(w0.seen)
        _generate(r0, prompt, sampling, policy=policy, tokens=40)
        header = sent[0]
        with pytest.raises(_Done):
            r1.follow()
        assert r1.costs == r0.costs, policy                     # rank 0's table on both, the same floats
        assert r1.costs["verify"] != r1.base_costs["verify"]
        assert w1.seen == w0.seen, policy                       # the same windows: the same decisions
        assert len(w0.seen) > n0
        tables.append(header[13:13 + header[12]])        # the header's cost table
    assert len({tuple(t) for t in tables}) > 1                  # the table moved between requests
    del r0, r1
    torch.cuda.empty_cache()
