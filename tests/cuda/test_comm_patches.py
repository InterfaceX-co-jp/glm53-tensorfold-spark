"""Our communication patch on TensorFold's synthetic GLM checkpoint (one GPU playing rank 0 of two).

patches/0040  GLM53_TF_COMM=prefetch[:MB] | ll | ll128 (comma-separated; default: upstream). ``prefetch`` reads
              the next kernels' weights into L2 on a side stream while each forward all-gather waits on the link
              (captured in the CUDA graphs with the all-gather) and runs the EXL3 shared expert before the routed
              ones. Nothing it does may change a bit: the logits of every window width (graphs and eager), and
              every reply (greedy and sampled, serial and drafted with MTP and DFlash2), must equal the default
              engine's. ``ll``/``ll128`` only set NCCL_PROTO before the communicator starts.

``_TwoCopies`` stands in for the second rank (its all-gather copies this rank's partials twice on the current
stream), so these tests check the engine's side of the patch: the side-stream fork and join inside eager and
captured forwards, that no kernel input or order that matters changed, and that the prefetch reads only weights.
They cannot check two-rank timing or NCCL itself: whether NCCL's kernel and the prefetch kernel overlap on the
GPU, the protocol variants, and the speed-up need the two Sparks (docs/COMM-ANALYSIS.md, "Validation").

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda pytest -q tests/cuda/test_comm_patches.py
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.glm5_next.cuda import overlap, weights  # noqa: E402
from tensorfold.families.glm5_next.cuda.forward import commit  # noqa: E402
from test_glm_engine import _TwoCopies, _checkpoint, _drafter, _generate  # noqa: E402

SAMPLINGS = [Sampling(1234, 1.0, 20, 0.95), None]
IDS = ["sampled", "greedy"]
POLICIES = (None, "auto", "auto:1:1:0", "2", "c3:0.35", "a:0.6:0.85", "f3", "fc5:0.3")


# -- the knob -------------------------------------------------------------------------------------------------
def test_variant_parsing(monkeypatch):
    monkeypatch.delenv(overlap.ENV, raising=False)
    assert overlap.variant() == overlap.Variant()
    assert overlap.variant("nccl") == overlap.Variant()
    assert overlap.variant("prefetch") == overlap.Variant(overlap.DEFAULT_PREFETCH_MB, None)
    assert overlap.variant("prefetch:2.5") == overlap.Variant(2.5, None)
    assert overlap.variant("prefetch:3, ll") == overlap.Variant(3.0, "LL")
    assert overlap.variant("ll128") == overlap.Variant(0.0, "LL128")
    monkeypatch.setenv(overlap.ENV, "prefetch:1")
    assert overlap.variant() == overlap.Variant(1.0, None)
    for bad in ("allreduce", "prefetch:0", "prefetch:x", "prefetch:100", "ll:1"):
        with pytest.raises(ValueError, match=overlap.ENV):
            overlap.variant(bad)


def test_nccl_env(monkeypatch):
    monkeypatch.setenv("NCCL_PROTO", "unset")             # records the original, so teardown restores it
    monkeypatch.delenv("NCCL_PROTO")
    overlap.nccl_env(overlap.variant("prefetch"))
    import os

    assert "NCCL_PROTO" not in os.environ
    overlap.nccl_env(overlap.variant("ll"))
    assert os.environ["NCCL_PROTO"] == "LL"
    overlap.nccl_env(overlap.variant("ll128"))              # an explicit setting wins
    assert os.environ["NCCL_PROTO"] == "LL"


# -- engines --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    out = {}
    for kind in ("mlx", "exl3"):
        path = tmp_path_factory.mktemp(f"glm_comm_{kind}")
        _checkpoint(path / "model", exl3=kind == "exl3")
        _drafter(path / "dflash2")
        out[kind] = path
    return out


def _engine(path, comm: str | None):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    mp = pytest.MonkeyPatch()
    mp.setattr(weights, "NONEXPERT", "q4mse")             # the deployed EXL3 configuration (patch 0001)
    mp.setenv("GLM53_TF_NONEXPERT", "q4mse")
    if comm is None:
        mp.delenv(overlap.ENV, raising=False)
    else:
        mp.setenv(overlap.ENV, comm)
    try:
        return GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", comm=_TwoCopies())
    finally:
        mp.undo()


@pytest.fixture(scope="module", params=["mlx", "exl3"])
def pair(request, ckpts):
    """(default engine, prefetch engine) on the same checkpoint."""

    path = ckpts[request.param]
    base = _engine(path, None)
    pre = _engine(path, "prefetch")
    assert base.e.w.meta.get("prefetch") is None
    assert pre.e.w.meta.get("prefetch") is not None
    return request.param, base, pre


# -- the plans ------------------------------------------------------------------------------------------------
def _weight_ranges(w):
    ranges = []
    seen = set()

    def add(t):
        if isinstance(t, torch.Tensor):
            if t.is_cuda and t.data_ptr() not in seen:
                seen.add(t.data_ptr())
                ranges.append((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()))
        elif isinstance(t, (list, tuple)):
            for v in t:
                add(v)
        elif hasattr(t, "__dict__") and not isinstance(t, type):
            for v in vars(t).values():
                add(v)

    add(w.layers)
    add([w.norm, w.head, w.mtp, w.draft_head, w.embed])
    return ranges


def test_plans_cover_every_site_and_read_only_weights(pair):
    _, _, pre = pair
    w = pre.e.w
    pf = w.meta["prefetch"]
    sites = {(layer.index, s) for layer in w.layers for s in "af"} | {("mtp", "a"), ("mtp", "f")}
    assert set(pf.plans) == sites
    ranges = _weight_ranges(w)
    for site, segs in pf.plans.items():
        assert segs, site
        assert sum(s.numel() * 4 for s in segs) <= pf.budget
        for s in segs:
            lo, hi = s.data_ptr(), s.data_ptr() + s.numel() * 4
            assert any(a <= lo and hi <= b for a, b in ranges), site          # a view into a weight
    # the site after the last layer's MoE starts with the final norm, the one after its attention with ffn_hc.fn
    last = w.layers[-1]
    assert pf.plans[(last.index, "f")][0].data_ptr() == w.norm.data_ptr()
    assert pf.plans[(last.index, "a")][0].data_ptr() == last.ffn_hc.fn.data_ptr()


def test_budget_caps_the_plans(ckpts):
    e = _engine(ckpts["mlx"], "prefetch:0.001")
    pf = e.e.w.meta["prefetch"]
    assert pf.budget == int(0.001 * 2**20)
    for segs in pf.plans.values():
        assert sum(s.numel() * 4 for s in segs) <= pf.budget


# -- bits -----------------------------------------------------------------------------------------------------
def test_window_logits_equal_default(pair):
    """Every window width from the committed state: graphs (1-6 rows) and eager (7, 8), logits and hidden rows."""

    kind, base, pre = pair
    rng = np.random.default_rng(31)
    prompt = [int(t) for t in rng.integers(0, 1000, size=19)]
    pf = pre.e.w.meta["prefetch"]
    for R in range(1, 9):
        rows = [int(t) for t in rng.integers(0, 1000, size=R)]
        got = []
        for eng in (base, pre):
            e = eng.e
            e.reset()
            e.forward(prompt)
            commit(e.w, e.st, e.buf, len(prompt), len(prompt))
            before = pf.launches
            logits = e.forward(rows).clone()
            torch.cuda.synchronize()
            got.append((logits, e.buf.hidden[:R].clone(), pf.launches - before))
        (lb, hb, _), (lp, hp, launched) = got
        assert torch.equal(lb, lp), (kind, R)
        assert torch.equal(hb, hp), (kind, R)
        if R > 6:                          # eager: one prefetch per all-gather of the forward
            assert launched == 2 * len(pre.e.w.layers), (kind, R, launched)
    for eng in (base, pre):          # the windows overwrote attention cache rows: no snapshot may resume from them
        eng.e.reset()
        eng.cache.clear()


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
def test_replies_equal_default(pair, sampling):
    """Serial and every drafting policy (MTP chains, DFlash2 blocks, the per-round choice): the prefetch
    engine's reply equals the default engine's, which equals its own serial reply."""

    kind, base, pre = pair
    prompt = list(np.random.default_rng(32).integers(0, 1000, size=43))
    serial, _ = _generate(base, prompt, sampling, draft=False, tokens=32)
    got, _ = _generate(pre, prompt, sampling, draft=False, tokens=32)
    assert got == serial, (kind, "serial")
    for policy in POLICIES:
        drafted, _ = _generate(pre, prompt, sampling, policy=policy, tokens=32)
        assert drafted == serial, (kind, policy)


def test_resume_equal_default(pair):
    kind, base, pre = pair
    sampling = Sampling(41, 1.0, 20, 0.95)
    rng = np.random.default_rng(33)
    first = list(rng.integers(0, 1000, size=70))           # more than one prefill chunk
    replies = []
    for eng in (base, pre):
        reply, _ = _generate(eng, first, sampling, policy="auto:1:1:0", tokens=20)
        warm, stats = _generate(eng, first + reply + [7, 8], sampling, tokens=20)
        assert stats["cached"] >= len(first) + len(reply) - 1
        replies.append((reply, warm))
    assert replies[0] == replies[1], kind


def test_prefetch_writes_no_weight(pair):
    _, _, pre = pair
    pf = pre.e.w.meta["prefetch"]
    segs = [s for plan in pf.plans.values() for s in plan]
    saved = [s.clone() for s in segs]
    _generate(pre, [3, 4, 5, 6, 7], None, policy="auto", tokens=16)
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(segs, saved))
