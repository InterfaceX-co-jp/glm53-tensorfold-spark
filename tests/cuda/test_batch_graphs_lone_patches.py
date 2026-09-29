"""patches/0510 on ONE GPU (TensorFold's synthetic checkpoint, ``_TwoCopies`` as rank 1). Nothing here may change a bit.

- ``GLM53_TF_VERIFY_SPLIT=N`` (2, 3, 8) and ``GLM53_TF_GRAPH_PROBE`` (with and without the split): the main graphs
  are ``vsplit.Split`` objects of N pieces; logits and hidden rows of windows 1-8 (graphs and eager) == the default
  engine's; replies (serial, MTP, DFlash2, the per-round choice; greedy and sampled) == the default engine's serial
  reply; the probe prints its lines;
- ``GLM53_TF_BATCH_GRAPHS=lone``: 4 slots, replies == serial; rounds of 2+ slots never capture / replay the batcher's
  graphs; a request alone in slot 1-3 does.

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q tests/cuda/test_batch_graphs_lone_patches.py
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.glm5_next.cuda import vsplit, weights  # noqa: E402
from tensorfold.families.glm5_next.cuda.forward import commit  # noqa: E402
from test_glm_engine import _TwoCopies, _checkpoint, _drafter, _generate  # noqa: E402

SAMPLINGS = [Sampling(1234, 1.0, 20, 0.95), None]
IDS = ["sampled", "greedy"]
POLICIES = (None, "auto", "2", "f3")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm_0510")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


def _engine(path, split: str | None = None, probe: str | None = None):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(weights, "NONEXPERT", "q4mse")
        mp.setenv("GLM53_TF_NONEXPERT", "q4mse")
        mp.setenv("GLM53_TF_LATENT_KV", "1")
        for k, v in (("GLM53_TF_VERIFY_SPLIT", split), ("GLM53_TF_GRAPH_PROBE", probe)):
            if v is None:
                mp.delenv(k, raising=False)
            else:
                mp.setenv(k, v)
        return GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", comm=_TwoCopies())


@pytest.fixture(scope="module")
def base(ckpt):
    return _engine(ckpt)


@pytest.fixture(scope="module", params=[("2", None), ("3", "7"), ("8", None), (None, "5")],
                ids=["split2", "split3+probe", "split8", "probe"])
def split(request, ckpt):
    n, probe = request.param
    return n, probe, _engine(ckpt, n, probe)


def test_main_graphs_are_pieces(split):
    n, probe, eng = split
    graphs = eng.e.graphs
    assert graphs.main
    for g in graphs.main.values():
        assert isinstance(g, vsplit.Split)
        assert len(g.graphs) == min(int(n or 1), len(eng.e.w.layers))
        assert (g.probe is not None) == (probe is not None)


def test_window_logits_equal_default(base, split, capfd):
    n, probe, eng = split
    rng = np.random.default_rng(51)
    prompt = [int(t) for t in rng.integers(0, 1000, size=19)]
    for R in range(1, 9):
        rows = [int(t) for t in rng.integers(0, 1000, size=R)]
        got = []
        for x in (base, eng):
            e = x.e
            e.reset()
            e.forward(prompt)
            commit(e.w, e.st, e.buf, len(prompt), len(prompt))
            for _ in range(3):                        # replays (the probe needs several)
                logits = e.forward(rows).clone()
            torch.cuda.synchronize()
            got.append((logits, e.buf.hidden[:R].clone()))
        (lb, hb), (lp, hp) = got
        assert torch.equal(lb, lp) and torch.equal(hb, hp), (n, probe, R)
    if probe is not None:
        eng.e.graphs.probe.report()
        assert "graph probe (patches/0510)" in capfd.readouterr().err
    for x in (base, eng):
        x.e.reset()
        x.cache.clear()


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
def test_replies_equal_default(base, split, sampling):
    n, probe, eng = split
    prompt = list(np.random.default_rng(52).integers(0, 1000, size=43))
    serial, _ = _generate(base, prompt, sampling, draft=False, tokens=32)
    for policy in POLICIES:
        assert _generate(eng, prompt, sampling, policy=policy, tokens=32)[0] == serial, (n, probe, policy)


def _batched(path, graphs: str):
    from test_batch_parallel_patches import _engine as engine

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GLM53_TF_LOOKUP", "0")
        mp.delenv("GLM53_TF_DEPTH", raising=False)
        return engine(path, batch=4, after=1, parity=True, mtp=True, graphs=graphs)


@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
def test_lone_batched_replies_equal_serial(ckpt, greedy):
    from test_batch2_patches import _prompt, _sampling, _serial

    lone = _batched(ckpt, "lone")
    ref = _batched(ckpt, "1")
    sampling = _sampling(greedy)
    prompts = [_prompt(90 + i, 30 + 9 * i) for i in range(4)]
    want = [_serial(ref, p, sampling, 40) for p in prompts]
    got = lone.batch.generate_batch([dict(prompt=p, max_tokens=40, sampling=sampling, policy=pol)
                                     for p, pol in zip(prompts, (None, "f3", "auto", "2"))])
    assert [t for t, _ in got] == want
    multi = [k for k in lone.batch.multi if isinstance(k, tuple) and k and isinstance(k[0], tuple) and len(k[0]) > 1]
    assert not multi, multi                                   # no graph of a 2+ slot round
    assert lone.batch.counts["eager"] >= 1


def test_lone_slot_uses_graphs(ckpt):
    """A request alone in a slot other than 0 (slot 0 is busy with a short request first) replays the batcher's
    graphs under ``lone``: some key of one slot is captured."""

    from test_batch2_patches import _prompt, _sampling

    lone = _batched(ckpt, "lone")
    reqs = [dict(prompt=_prompt(95, 20), max_tokens=2, sampling=_sampling(True), policy=None),
            dict(prompt=_prompt(96, 30), max_tokens=48, sampling=_sampling(True), policy="2")]
    lone.batch.generate_batch(reqs)
    ones = [k for k in lone.batch.multi if isinstance(k, tuple) and k and isinstance(k[0], tuple) and len(k[0]) == 1]
    assert ones, list(lone.batch.multi)
