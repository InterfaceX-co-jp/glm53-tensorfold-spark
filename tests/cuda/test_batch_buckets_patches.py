"""patches/0280: batched rounds padded to a few buckets so they replay CUDA graphs (``GLM53_TF_BATCH_BUCKETS``).

W1: 70-90% of the 4-stream rounds ran eager, because a round's graph key holds one row count a slot (4 slots x 1-8 rows
x parities) and few keys repeat. With ``GLM53_TF_BATCH_BUCKETS=4,8`` every window of a multi-slot round is padded with
its last token to the smallest listed size >= the round's longest window, so the key is (slots, bucket, modes,
parities). ``GLM53_TF_BATCH_PAD_TIE`` (auto: on with buckets): a padded row's router logits are replaced by its
window's last real row's before top-k, so it routes to experts the round reads anyway (no ~6 ms of expert reads a row).

Checked:

- host only: ``bucket_mask`` / ``bucket_rows`` / ``route_sources``;
- GPU (TensorFold's synthetic EXL3 checkpoint + DFlash2 drafter, one GPU playing rank 0 of two): bucketed rounds give
  every real row its sequence's lone bits (logits, final normed rows, DFlash2 taps) on the first sighting, the capture
  and replays; two different window sets of the same bucket share ONE graph (the second replays at once); padded rows
  pick exactly their source row's experts (the last MoE layer's picks); 4 requests with buckets (and with the 0200
  knobs) == serial, sampled and greedy, for mixed policies; the ranks' settings check covers the new knobs.

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q
tests/cuda/test_batch_buckets_patches.py
"""

from __future__ import annotations

import pytest

from tensorfold.families.glm5_next.cuda import batchplan

try:
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:          # the host-only tests still run
    torch = None
    CUDA = False

gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")


# -- host only ---------------------------------------------------------------------------------------------------------
def test_bucket_mask_and_rows():
    assert batchplan.bucket_mask("", 8) == 0 and batchplan.bucket_mask("0", 8) == 0
    m = batchplan.bucket_mask("4,8", 8)
    assert m == 0b10001000
    assert [batchplan.bucket_rows([r], m) for r in range(1, 10)] == [4, 4, 4, 4, 8, 8, 8, 8, 9]
    assert batchplan.bucket_rows([1, 3, 2], m) == 4
    assert batchplan.bucket_rows([1, 5, 2, 1], m) == 8
    assert batchplan.bucket_rows([2, 2], batchplan.bucket_mask("8", 8)) == 8
    assert batchplan.bucket_rows([3, 1], 0) == 3                 # no mask: the longest window
    assert batchplan.bucket_rows([], m) == 0
    for bad in ("9", "-2", "x"):
        with pytest.raises(ValueError, match="GLM53_TF_BATCH_BUCKETS"):
            batchplan.bucket_mask(bad, 8)


def test_route_sources():
    assert batchplan.route_sources([3, 1, 2], [4, 4, 4]) == [0, 1, 2, 2, 4, 4, 4, 4, 8, 9, 9, 9]
    assert batchplan.route_sources([2, 5], [2, 5]) == list(range(7))            # nothing padded: identity
    assert batchplan.route_sources([], []) == []
    for real, ran in (([0], [4]), ([5], [4])):
        with pytest.raises(ValueError):
            batchplan.route_sources(real, ran)


# -- GPU ---------------------------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")
    monkeypatch.setenv("GLM53_TF_NONEXPERT", "q4mse")
    monkeypatch.delenv("GLM53_TF_DEPTH", raising=False)


def _engine(path, *, buckets: str = "4,8", tie: str = "auto", pad: str = "", after: int = 1, parity: bool = True,
            mtp: bool = False, **kw):
    from test_batch2_patches import _engine as engine

    with pytest.MonkeyPatch.context() as m:
        m.setenv("GLM53_TF_BATCH_BUCKETS", buckets)
        m.setenv("GLM53_TF_BATCH_PAD_TIE", tie)
        m.setenv("GLM53_TF_BATCH_MTP", "1" if mtp else "0")
        m.setenv("GLM53_TF_BATCH_PARITY_KEY", "1" if parity else "0")
        m.setenv("GLM53_TF_BATCH_PAD", pad)
        m.setenv("GLM53_TF_BATCH_CAPTURE_AFTER", str(after))
        m.setenv("GLM53_TF_BATCH_SHORT", "0")
        return engine(path, **kw)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_batch_buckets")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


@pytest.fixture(scope="module")
def ref(ckpt):
    with pytest.MonkeyPatch.context() as m:
        m.setenv("GLM53_TF_BATCH_BUCKETS", "")
        from test_batch2_patches import _engine as engine

        return engine(ckpt)


@pytest.fixture(scope="module")
def b4k(ckpt):
    """4 slots, buckets 4 / 8 with tied routing, a key captured on its 2nd sighting, parities keyed."""

    return _engine(ckpt, batch=4, after=2)


@pytest.fixture(scope="module")
def b4all(ckpt):
    """4 slots, buckets + every 0200 knob (pad 2,4,8 first, then the bucket; batched MTP)."""

    return _engine(ckpt, batch=4, buckets="8", pad="2,4,8", after=3, mtp=True)


def _serial(e, prompt, sampling, tokens):
    from test_batch2_patches import _serial as serial

    return serial(e, prompt, sampling, tokens)


@gpu
def test_settings(b4k, b4all):
    assert b4k.batch.buckets == batchplan.bucket_mask("4,8", 8) and b4k.batch.tie
    assert b4k.batch.src_dev is not None and b4k.e.buf.route_src is None
    assert b4all.batch.buckets == batchplan.bucket_mask("8", 8) and b4all.batch.tie


@gpu
def test_bucketed_rows_are_each_sequences_bits(b4k):
    """Rounds padded to the bucket: each real row has its sequence's lone bits (logits, final normed rows, DFlash2 taps)
    at the padded offsets; eager (1st sighting), capture (2nd), replay; a second window set of the same bucket replays
    the same graph at once; padded rows pick their source row's experts."""

    from test_batch2_patches import _prompt, _reset

    from tensorfold.families.glm5_next.cuda.decode import prefill
    from tensorfold.families.glm5_next.cuda.forward import forward

    bat, e, w = b4k.batch, b4k.e, b4k.w
    b = e.buf
    with torch.no_grad():
        for s in range(bat.n):
            with bat._on(s) as es, bat._borrow(s):
                prefill(es, _prompt(81 + s, 40 + 11 * s), None, mtp=True)

        def alone(active, windows):
            out = []
            for s, win in zip(active, windows):
                logits = forward(w, bat.states[s], b, win)
                taps = torch.cat([t[:len(win)] for t in b.taps], dim=1)
                out.append((logits.clone(), b.fnormed[:len(win)].clone(), taps.clone()))
            return out

        def check(active, windows, want, it):
            got = bat._forward(active, windows).clone()
            ran = list(bat.last_rows)
            Rp = batchplan.bucket_rows([len(x) for x in windows], bat.buckets)
            assert ran == [Rp] * len(windows), (ran, Rp)
            T = sum(ran)
            normed = b.fnormed[:T].clone()
            taps = torch.cat([t[:T] for t in b.taps], dim=1).clone()
            pick = b.pick[:T].clone()
            off = 0
            for (lg, nm, tp), win, n in zip(want, windows, ran):
                R = len(win)
                assert torch.equal(got[off:off + R], lg), (active, windows, it)
                assert torch.equal(normed[off:off + R], nm) and torch.equal(taps[off:off + R], tp)
                for r in range(R, n):           # padded rows route as the window's last real row
                    assert torch.equal(pick[off + r], pick[off + R - 1]), (r, it)
                off += n

        # parities: every slot at buffer 0 for the comparison (the lone forwards above do not commit)
        sets = [([0, 1, 2, 3], [[5, 6, 7], [8], [9, 10, 11, 12, 13], [17, 18]]),
                ([0, 1, 2, 3], [[20], [21, 22, 23, 24, 25, 26], [27, 28], [29, 30, 31]]),     # the same bucket (8)
                ([1, 3], [[40, 41, 42], [44]]),                                               # bucket 4
                ([1, 3], [[45], [46, 47, 48, 49]])]                                           # bucket 4 again
        before = bat.counts["capture"]
        want0 = alone(*sets[0])
        for it in range(3):                     # eager (1st sighting), capture (2nd), replay (3rd)
            check(*sets[0], want0, it)
            assert bat.last_kind == ("eager", "capture", "graph")[it]
        assert bat.counts["capture"] == before + 1
        check(*sets[1], alone(*sets[1]), 0)     # a different window set, the same key: replays at once
        assert bat.last_kind == "graph"
        want2 = alone(*sets[2])
        for it in range(2):
            check(*sets[2], want2, it)
        check(*sets[3], alone(*sets[3]), 0)
        assert bat.last_kind == "graph"
        assert bat.counts["capture"] == before + 2
    assert bat.counts["bucket_rows"] > 0
    _reset(bat)


QUADS = [(None, "f3", "o", "2"), ("auto:1:1:0", "fc5:0.3", "0", "a:0.6:0.85"), ("of", "c3:0.35", "7", "om2")]


@gpu
@pytest.mark.parametrize("which", ["buckets", "every_knob"])
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
def test_four_requests_equal_serial(ref, b4k, b4all, which, greedy):
    from test_batch2_patches import _prompt, _sampling

    eng = b4k if which == "buckets" else b4all
    sampling = _sampling(greedy)
    prompts = [_prompt(90 + i, 30 + 7 * i) for i in range(4)]
    want = [_serial(ref, p, sampling, 40) for p in prompts]
    for quad in QUADS:
        got = eng.batch.generate_batch([dict(prompt=p, max_tokens=40, sampling=sampling, policy=pol)
                                        for p, pol in zip(prompts, quad)])
        assert [t for t, _ in got] == want, quad
        kinds = [s["round_kinds"] for _, s in got]
        kind = ("alone", "graph", "eager", "capture")
        assert all(sum(d.get(k, 0) for k in kind) == s["rounds"] for d, (_, s) in zip(kinds, got))
    c = eng.batch.counts
    assert c["graph"] >= 1 and c["bucket_rows"] > 0


@gpu
def test_tie_off_is_still_exact(ckpt, ref):
    """Buckets with GLM53_TF_BATCH_PAD_TIE=0 (padded rows route on their own logits): replies still == serial."""
    from test_batch2_patches import _prompt, _sampling

    eng = _engine(ckpt, batch=4, buckets="8", tie="0", after=1)
    assert not eng.batch.tie and eng.batch.src_dev is None
    sampling = _sampling(True)
    prompts = [_prompt(110 + i, 25 + 5 * i) for i in range(4)]
    want = [_serial(ref, p, sampling, 32) for p in prompts]
    got = eng.batch.generate_batch([dict(prompt=p, max_tokens=32, sampling=sampling, policy=pol)
                                    for p, pol in zip(prompts, QUADS[0])])
    assert [t for t, _ in got] == want
    assert eng.batch.counts["bucket_rows"] > 0
