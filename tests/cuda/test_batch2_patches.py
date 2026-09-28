"""patches/0120 (GLM53_TF_BATCH=N, batching on the current engine): replaces tests/cuda/test_batch_patches.py.

2-4 requests decode together, one forward a round over every decoding request's verify window; each request keeps
its own latent KV / rings, KDA state, MTP cache, DFlash2 context, lookup index, cost-derived depths and tf_knobs;
prompts prefill in pieces through ``decode.prefill`` itself (exact, fast, lean), interleaved with the others' rounds.
Every batched reply must be byte-identical to the same request served alone (and to serial decoding). Checked:

- host only (no GPU needed): request headers and round plans round trip; piece bounds (exact and on a fast grid);
  the prefill time share; which prompt prefills next; admission order and which background request steps aside;
  the session-title heuristic; the round cost model behind batch-aware depths; tf_knobs accepted in batch mode
  (calib_online refused);
- GPU, TensorFold's synthetic EXL3 checkpoint with the DFlash2 drafter (one GPU playing rank 0 of two):
  - slots: own states (window-sized KDA rows, prefill-sized index rings), own DFlash2 contexts and graphs;
  - a batched forward's rows (logits, MTP input rows, DFlash2 taps) are the bits of each sequence's lone forward,
    2-4 sequences, lazily captured graphs (first round and replay) and eager;
  - 2 and 4 requests queued together equal serial decoding for every policy (MTP, DFlash2, auto, cost-derived o /
    om / of, thresholds, serial), sampled and greedy, graphs and eager; lookup drafts (lN, auto with lookup) on
    repeating prompts; uneven lengths with requests queued behind a full batch; per-slot prefix reuse;
  - prefill pieces: a long prompt prefills in pieces while another request decodes between them, both exact;
  - fast prefill (0080) and lean prefill (0082) admissions equal the lone fast / lean engine, only grid snapshots
    kept; per-request knobs per sequence (prefill_rows, fast_prefill, auto_fdrafts, expert_loop, depth, lookup,
    longctx_graphs) equal the lone engine with the same knobs, echoed, defaults restored;
  - latent KV (0060) and expanded, past 2,051 tokens (``_index_heads_32``): long prompts in pieces, a window crossing
    the dense limit, sparse rounds on captured graphs keyed by pool bucket;
  - a client going away mid-reply; a background request stepping aside for a foreground one and running again
    (same tokens to its caller); concurrent streaming callers; an engine trimmed to what memory holds;
  - a second engine replaying rank 0's round plans through ``follow`` (as rank 1 does) makes the same decisions.

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda pytest -q tests/cuda/test_batch2_patches.py
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from tensorfold.families.glm5_next.cuda import batchplan, knobs

try:
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:          # the host-only tests still run
    CUDA = False

gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")

COSTS = {"verify": [29.2, 38.9, 45.2, 55.2, 59.5, 63.2, 65.6, 69.2], "mtp": 1.7, "mtp_step": 1.5, "mtp_row": 0.1,
         "block": 3.0, "taps_row": 0.05}


# -- host only -----------------------------------------------------------------------------------------------------
def test_header_round_trip():
    values = {k: i + 1 for i, k in enumerate(knobs.HEADER)}
    for seed, temp, top_k, top_p, code in ((0, 0.0, 0, 1.0, [0, 0, 0, 0]),
                                           (2**64 - 1, 0.7, 20, 0.95, [4, 2, 8, 30000, 1, 4]),
                                           (1234567890123, 1.3, 50, 0.5, [6, 7, 1, 0]),
                                           (99, 1.0, 20, 0.95, [20, 7, 3, 0])):
        h = batchplan.Header(40, True, False, seed, temp, top_k, top_p, 1, dict(values), list(code))
        ints = batchplan.encode_header(h)
        assert all(-2**31 <= v < 2**31 for v in ints)
        assert batchplan.decode_header(ints) == h
    bad = batchplan.encode_header(h)
    bad[batchplan.HEADER_FIXED] += 1                 # a knob block of another patch set
    with pytest.raises(RuntimeError, match="same patched TensorFold"):
        batchplan.decode_header(bad)


def test_plan_round_trip():
    admits = [(0, 0, [5, 6, 7]), (2, 128, list(range(20)))]
    plan = batchplan.encode_plan([1, 3], admits, [2])
    assert batchplan.parse_plan(plan) == ([1, 3], admits, [2])
    assert batchplan.parse_plan(batchplan.encode_plan([], [], [])) == ([], [], [])
    with pytest.raises(RuntimeError):
        batchplan.parse_plan(plan + [9])


def test_piece_bounds():
    assert [batchplan.piece_end(d, 5000, 2048, 0) for d in (0, 2048, 4096)] == [2048, 4096, 5000]
    assert batchplan.piece_end(100, 150, 64, 0) == 150
    assert batchplan.piece_rows(2048, 1024) == 2048 and batchplan.piece_rows(1500, 1024) == 1024
    assert batchplan.piece_rows(100, 1024) == 1024 and batchplan.piece_rows(100, 0) == 100
    ends, done = [], 0
    while done < 5000:
        done = batchplan.piece_end(done, 5000, 1500, 1024)
        ends.append(done)
    assert ends == [1024, 2048, 3072, 4096, 5000]              # every piece but the last ends on the grid
    assert batchplan.piece_end(4096, 4096 + 1024, 1500, 1024) == 5120
    with pytest.raises(ValueError, match="off the chunk grid"):
        batchplan.piece_end(100, 5000, 2048, 1024)


def test_prefill_time_share():
    f = batchplan.Fairness(0.5)
    assert f.allow(False) and f.allow(True)                   # nothing owed yet
    f.after(2.0, 0.0, True)                                    # a 2 s piece, nothing decoding
    assert f.allow(False) and not f.allow(True)
    f.after(0.0, 0.5, True)                                    # 0.5 s of decode rounds repaid
    assert not f.allow(True)
    f.after(0.0, 1.6, True)
    assert f.allow(True)
    f.after(1.0, 0.1, True)                                    # a piece and a decode round in one round
    assert not f.allow(True)
    f.after(0.0, 0.0, False)                                   # nothing left to prefill: nothing owed
    assert f.allow(True)
    g = batchplan.Fairness(1.0)                                # prefill first, but one decode round between pieces
    g.after(3.0, 0.1, True)
    assert g.allow(True)
    g.after(3.0, 0.0, True)
    assert not g.allow(True) and g.allow(False)
    g.after(0.0, 0.05, True)
    assert g.allow(True)
    for bad in (0.0, -1.0, 1.5):
        with pytest.raises(ValueError, match="PREFILL_SHARE"):
            batchplan.Fairness(bad)


def test_pick_piece_and_admission():
    assert batchplan.pick_piece([]) is None
    assert batchplan.pick_piece([(0, 5000, 3), (1, 200, 5), (3, 200, 4)]) == 3
    assert batchplan.admission_order([True, False, True, False]) == [1, 3, 0, 2]
    running = [(0, False, 1), (1, True, 4), (2, True, 7)]
    assert batchplan.victim(running, True, 0) == 2             # the latest admitted background request
    assert batchplan.victim(running, False, 0) is None
    assert batchplan.victim(running, True, 1) is None          # a free slot: nobody steps aside
    assert batchplan.victim([(0, False, 1)], True, 0) is None


def test_title_request():
    title = [{"role": "system", "content": "Generate a short title for this conversation."},
             {"role": "user", "content": "fix the build"}]
    assert batchplan.is_title_request(title, None)
    assert not batchplan.is_title_request(title, [{"type": "function"}])
    assert not batchplan.is_title_request(title[1:], None)
    assert not batchplan.is_title_request([{"role": "system", "content": "title " + "x" * 5000}], None)
    assert not batchplan.is_title_request(None, None)


def test_round_costs():
    v = COSTS["verify"]
    assert batchplan.verify_ms(v, 3) == 45.2
    slope = (v[7] - v[3]) / 4
    assert batchplan.verify_ms(v, 10) == pytest.approx(v[7] + 2 * slope)
    rc = batchplan.RoundCosts(COSTS)
    assert rc.rate() is None
    rc.record([("m", 3, 1, 1)], 2)
    assert rc.rate() is None                                   # alone: the request's own rate applies
    parts = [("m", 3, 2, 2), ("f", 4, 0, 3), ("s", 1, 0, 0), ("l", 5, 0, 0)]
    ms = batchplan.verify_ms(v, 13) + (1.7 + 1.5 + 0.1) + (3.0 + 0.05 * 3)
    assert rc.round_ms(parts) == pytest.approx(ms)
    rc.record(parts, 7)
    first = batchplan.verify_ms(v, 3) + 1.7
    assert rc.rate() == pytest.approx(9 / (first + ms))
    assert rc.table(0) == v
    t = rc.table(6)
    assert t == [batchplan.verify_ms(v, 7 + k) for k in range(8)]
    assert all(b - a == pytest.approx(slope) for a, b in zip(t[1:], t[2:]))


def test_knobs_allowed_in_batch_mode():
    got = knobs.parse({"prefill_rows": 7, "fast_prefill": 1, "lookup": 0, "depth": "cost", "calib_online": 0},
                      rows_max=256, batch=True)
    assert got["prefill_rows"] == 7 and got["fast_prefill"] == 1
    with pytest.raises(ValueError, match="share their rounds"):
        knobs.parse({"calib_online": 1}, rows_max=256, batch=True)
    assert knobs.parse({"calib_online": 1}, rows_max=256) == {"calib_online": 1}


# -- GPU: engines ----------------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """Lookup off unless a test turns it on (its rounds make windows depend on repeats); 4-bit non-expert weights
    (read per request: plain ``auto`` chooses between MTP and DFlash2)."""

    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")
    monkeypatch.setenv("GLM53_TF_NONEXPERT", "q4mse")
    monkeypatch.delenv("GLM53_TF_DEPTH", raising=False)


def _engine(path, *, batch: int = 1, rows: int = 64, rows_max: int | None = None, latent_kv: bool = False,
            context: int = 0, fast: bool = False, lean: bool = False, block: int = 64, graphs: str = "1",
            piece: int | None = None, share: float = 1.0, reserve: float = 0.25, long_graphs: bool = True):
    from test_glm_engine import _TwoCopies

    from tensorfold.families.glm5_next.cuda import weights
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    with pytest.MonkeyPatch.context() as m:
        m.setattr(weights, "NONEXPERT", "q4mse")
        m.setenv("GLM53_TF_NONEXPERT", "q4mse")
        m.setenv("GLM53_TF_BATCH", str(batch))
        m.setenv("GLM53_TF_BATCH_GRAPHS", graphs)
        m.setenv("GLM53_TF_PREFILL_ROWS", str(rows))
        m.setenv("GLM53_TF_PREFILL_ROWS_MAX", str(rows_max or rows))
        m.setenv("GLM53_TF_LATENT_KV", "1" if latent_kv else "0")
        m.setenv("GLM53_TF_FAST_PREFILL", "1" if fast else "0")
        m.setenv("GLM53_TF_LEAN_PREFILL", "1" if lean else "0")
        m.setenv("GLM53_TF_LEAN_BLOCK", str(block))
        m.setenv("GLM53_TF_LONGCTX_GRAPHS", "1" if long_graphs else "0")
        m.setenv("GLM53_TF_BATCH_PREFILL_SHARE", str(share))
        m.setenv("GLM53_TF_BATCH_RESERVE_GB", str(reserve))
        m.setenv("GLM53_TF_BATCH_ADMIT_GB", "0")
        if piece is not None:
            m.setenv("GLM53_TF_BATCH_PIECE", str(piece))
        else:
            m.delenv("GLM53_TF_BATCH_PIECE", raising=False)
        for k in ("GLM53_TF_CALIB_ONLINE", "GLM53_TF_PROFILE", "GLM53_TF_DEPTH", "GLM53_TF_FAST_GATHER",
                  "GLM53_TF_BATCH_GRAPH_ROWS", "GLM53_TF_BATCH_MAX_GRAPHS"):
            m.delenv(k, raising=False)
        return GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", context=context,
                         comm=_TwoCopies())


def _free(*engines) -> None:
    """Stop the engines' batch loops (their threads would keep them alive) and give their memory back."""

    for e in engines:
        if e.batch is not None:
            e.batch.stop()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_batch2")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


@pytest.fixture(scope="module")
def ref(ckpt):
    """The same checkpoint served one request at a time."""

    return _engine(ckpt)


@pytest.fixture(scope="module")
def b2(ckpt):
    return _engine(ckpt, batch=2)


@pytest.fixture(scope="module")
def b4(ckpt):
    return _engine(ckpt, batch=4)


@pytest.fixture(scope="module")
def b2e(ckpt):
    return _engine(ckpt, batch=2, graphs="0")


def _sampling(greedy: bool):
    from tensorfold.engine.exact_sampling import Sampling

    return None if greedy else Sampling(1234, 1.0, 20, 0.95)


GREEDY = [True, False]
GIDS = ["greedy", "sampled"]


def _prompt(seed: int, n: int) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(0, 1000, size=n)]


def _repeated(seed: int, block: int = 40, times: int = 3, tail: int = 10) -> list[int]:
    b = [int(t) for t in np.random.default_rng(seed).integers(0, 1000, size=block)]
    return b * times + b[:tail]


def _run(e, prompt, sampling, *, policy=None, tokens=24, knobs_=None, draft=True, background=False,
         on_tokens=None):
    """``engine.generate`` as the server calls it (this thread's policy, knobs, priority), stop at EOS off."""

    out: list[int] = []
    e.request.policy = policy
    e.request.stop_eos = False
    e.request.knobs = knobs_
    e.request.background = background

    def got(new):
        out.extend(new)
        return on_tokens(new) if on_tokens is not None else False

    try:
        stats = e.generate(list(prompt), tokens, sampling, got, draft=draft)
    finally:
        e.request.knobs = None
        e.request.background = False
    return out, stats


def _serial(e, prompt, sampling, tokens, knobs_=None):
    out, stats = _run(e, prompt, sampling, tokens=tokens, draft=False, knobs_=knobs_)
    assert stats["drafts"] is False and len(out) == tokens
    return out


def _reset(bat) -> None:
    """After a test drove slots directly: nothing kept on them is valid any more."""

    for st in bat.states:
        st.reset()
    for d in bat.drafters:
        if d is not None:
            d.reset()
    bat.caches = [[] for _ in bat.caches]


# -- slots and the batched forward -----------------------------------------------------------------------------------
@gpu
def test_slots_have_their_own_state(b4):
    bat = b4.batch
    assert bat.n == 4 and bat.states[0] is b4.e.st and len({id(s) for s in bat.states}) == 4
    ptrs = lambda f: {f(s) for s in bat.states}            # noqa: E731
    assert len(ptrs(lambda s: s.rec.data_ptr())) == 4 and len(ptrs(lambda s: s.kc[0].data_ptr())) == 4
    assert len(ptrs(lambda s: s.mtp_kc.data_ptr())) == 4 and len(ptrs(lambda s: s.proj.data_ptr())) == 4
    for st in bat.states[1:]:                              # window rows only; a prefill borrows slot 0's
        assert st.proj.shape[1] == bat.max_rows and st.scratch_set.rows == bat.max_rows
    d = bat.drafters
    assert d[0] is b4.drafter and len({id(x) for x in d}) == 4
    assert len({x.kc[0].data_ptr() for x in d}) == 4 and len({x.pos_dev.data_ptr() for x in d}) == 4
    assert all(x.block_graph is not None and len(x.tap_graphs) == x.block for x in d)
    assert d[1].layers is d[0].layers                      # the weights are shared
    assert all(g.mtp for g in bat.graphs)


@gpu
@pytest.mark.parametrize("which", ["graphs", "eager"])
def test_batched_rows_are_each_sequences_bits(b4, b2e, which):
    """One forward over 2-4 sequences' windows gives each row the bits of that sequence's own forward: logits, the
    final-normed rows the MTP head reads and the DFlash2 taps. Graphs: the key's first round (eager through the
    capturable code) and its replay; eager: the host-position path."""

    from tensorfold.families.glm5_next.cuda.batch import compute_multi, stage_multi
    from tensorfold.families.glm5_next.cuda.decode import prefill
    from tensorfold.families.glm5_next.cuda.forward import chunks_for, forward

    eng = b4 if which == "graphs" else b2e
    with torch.no_grad():
        _rows_check(eng, which, compute_multi, stage_multi, prefill, chunks_for, forward)


def _rows_check(eng, which, compute_multi, stage_multi, prefill, chunks_for, forward):
    bat, e, w = eng.batch, eng.e, eng.w
    b = e.buf
    for s in range(bat.n):
        with bat._on(s) as es, bat._borrow(s):
            prefill(es, _prompt(31 + s, 45 + 17 * s), None, mtp=True)
    sets = [([0, 1], [[5, 6, 7], [8, 9]]), ([0, 1], [list(range(20, 28)), list(range(30, 36))]), ([1], [[11, 12]])]
    if bat.n == 4:
        sets += [([0, 1, 2, 3], [[5, 6, 7], [8], [9, 10, 11, 12, 13, 14, 15, 16], [17, 18]]),
                 ([1, 3], [[40, 41, 42, 43], [44]]), ([0, 2, 3], [[50], [51, 52], [53, 54, 55]])]
    for active, windows in sets:
        alone = []
        for s, win in zip(active, windows):
            logits = forward(w, bat.states[s], b, win)            # eager, one sequence, no commit
            taps = torch.cat([t[:len(win)] for t in b.taps], dim=1)
            alone.append((logits.clone(), b.fnormed[:len(win)].clone(), taps.clone()))
        for _ in range(2 if which == "graphs" else 1):         # graphs: the first round, then the replay
            both = bat._forward(active, windows).clone()
            normed = b.fnormed[:sum(map(len, windows))].clone()
            taps = torch.cat([t[:sum(map(len, windows))] for t in b.taps], dim=1).clone()
            off = 0
            for (lg, nm, tp), win in zip(alone, windows):
                R = len(win)
                assert torch.equal(both[off:off + R], lg), (active, windows)
                assert torch.equal(normed[off:off + R], nm) and torch.equal(taps[off:off + R], tp)
                off += R
        if len(active) > 1:
            sts = [bat.states[s] for s in active]
            stage_multi(w, sts, b, windows)
            eager = compute_multi(w, sts, b, [len(x) for x in windows],
                                  nchs=[chunks_for(st, len(x)) for st, x in zip(sts, windows)],
                                  host_pos=[st.pos for st in sts]).clone()
            assert torch.equal(eager, both)
    if which == "graphs":
        assert bat.counts["graph"] >= 1 and bat.multi
    _reset(bat)


# -- requests: every policy ------------------------------------------------------------------------------------------
PAIRS = [(None, "2"), ("auto", "c3:0.35"), ("a:0.6:0.85", "0"), ("7", "1"), ("f3", "3"), ("fc5:0.3", "auto:1:1:0"),
         ("o", "om"), ("of3", "om3")]
QUADS = [(None, "f3", "o", "2"), ("auto:1:1:0", "fc5:0.3", "0", "a:0.6:0.85"), ("of", "c3:0.35", "7", "om2")]


@gpu
@pytest.mark.parametrize("greedy", GREEDY, ids=GIDS)
@pytest.mark.parametrize("which", ["b2", "b2e"])
def test_two_requests_equal_serial(ref, b2, b2e, which, greedy):
    eng = b2 if which == "b2" else b2e
    sampling = _sampling(greedy)
    pa, pb = _prompt(5, 37), _prompt(6, 29)
    want_a, want_b = _serial(ref, pa, sampling, 32), _serial(ref, pb, sampling, 32)
    for pol_a, pol_b in PAIRS:
        (got_a, sa), (got_b, sb) = eng.batch.generate_batch([
            dict(prompt=pa, max_tokens=32, sampling=sampling, policy=pol_a),
            dict(prompt=pb, max_tokens=32, sampling=sampling, policy=pol_b)])
        assert got_a == want_a, (pol_a, pol_b)
        assert got_b == want_b, (pol_a, pol_b)
        assert sa["slot"] != sb["slot"]
        assert sa["batched_rounds"] >= 1 and sb["batched_rounds"] >= 1, (sa, sb)
    solo, stats = _run(eng, pb, sampling, policy="f3", tokens=32)            # alone through the batch engine
    assert solo == want_b and stats["batched_rounds"] == 0
    assert "f" in stats["drafters"]                                          # DFlash2 on a batch engine


@gpu
@pytest.mark.parametrize("greedy", GREEDY, ids=GIDS)
def test_four_requests_equal_serial(ref, b4, greedy):
    sampling = _sampling(greedy)
    prompts = [_prompt(20 + i, 30 + 9 * i) for i in range(4)]
    want = [_serial(ref, p, sampling, 40) for p in prompts]
    for quad in QUADS:
        got = b4.batch.generate_batch([dict(prompt=p, max_tokens=40, sampling=sampling, policy=pol)
                                       for p, pol in zip(prompts, quad)])
        assert [t for t, _ in got] == want, quad
        assert len({s["slot"] for _, s in got}) == 4
        assert all(s["batched_rounds"] >= 1 for _, s in got), [s["batched_rounds"] for _, s in got]
    arms = "".join(s.get("drafters", "") for _, s in got)
    assert "m" in arms and "f" in arms


@gpu
@pytest.mark.parametrize("greedy", GREEDY, ids=GIDS)
def test_lookup_drafts_batched(ref, b2, greedy, monkeypatch):
    """Prompt-lookup drafts (lN forced, auto gated) on prompts that repeat a block, each request with its own
    history index."""

    monkeypatch.setenv("GLM53_TF_LOOKUP", "1")
    monkeypatch.setenv("GLM53_TF_LOOKUP_MIN", "3")
    sampling = _sampling(greedy)
    pa, pb = _repeated(51), _repeated(52, block=24, times=4, tail=6)
    want_a, want_b = _serial(ref, pa, sampling, 48), _serial(ref, pb, sampling, 48)
    seen = ""
    for pol_a, pol_b in (("l7", "auto"), ("auto:1:1:0", "l3:2"), ("o", "l7:1")):
        (got_a, sa), (got_b, sb) = b2.batch.generate_batch([
            dict(prompt=pa, max_tokens=48, sampling=sampling, policy=pol_a),
            dict(prompt=pb, max_tokens=48, sampling=sampling, policy=pol_b)])
        assert got_a == want_a and got_b == want_b, (pol_a, pol_b)
        seen += sa.get("drafters", "") + sb.get("drafters", "")
    assert "l" in seen


@gpu
@pytest.mark.parametrize("greedy", GREEDY, ids=GIDS)
def test_uneven_lengths_and_queued_requests(ref, b2, greedy):
    sampling = _sampling(greedy)
    specs = [(_prompt(7, 33), 40, "2"), (_prompt(8, 21), 6, "c3:0.35"), (_prompt(9, 50), 20, None),
             (_prompt(10, 12), 12, "f3")]
    want = [_serial(ref, p, sampling, n) for p, n, _ in specs]
    got = b2.batch.generate_batch([dict(prompt=p, max_tokens=n, sampling=sampling, policy=pol)
                                   for p, n, pol in specs])
    assert [t for t, _ in got] == want
    (_, sa), (_, sb), (_, sc), (_, sd) = got
    assert sc["slot"] == sb["slot"] != sa["slot"]              # the third waited for the second's slot
    assert sd["queued_s"] >= 0 and sa["batched_rounds"] >= 1 and sc["batched_rounds"] >= 1


@gpu
@pytest.mark.parametrize("greedy", GREEDY, ids=GIDS)
def test_prefix_reuse_per_slot(ref, b2, greedy):
    """A follow-up resumes on the slot that kept its state (while the other slot serves something else) and gives
    the reply a fresh serial prefill gives."""

    sampling = _sampling(greedy)
    first, other = _prompt(10, 70), _prompt(11, 40)
    (reply, s1), (_, s2) = b2.batch.generate_batch([dict(prompt=first, max_tokens=16, sampling=sampling),
                                                     dict(prompt=other, max_tokens=16, sampling=sampling)])
    after = first + reply + [5, 6, 7]
    (warm, sw), (_, su) = b2.batch.generate_batch([dict(prompt=after, max_tokens=20, sampling=sampling),
                                                   dict(prompt=_prompt(12, 25), max_tokens=20, sampling=sampling)])
    assert sw["slot"] == s1["slot"] and sw["cached"] >= len(first) + len(reply) - 1, sw
    assert su["slot"] == s2["slot"]
    assert warm == _serial(ref, after, sampling, 20)


# -- prefill pieces -----------------------------------------------------------------------------------------------
@gpu
@pytest.mark.parametrize("greedy", GREEDY, ids=GIDS)
def test_long_prompt_prefills_in_pieces_between_rounds(ckpt, ref, greedy):
    """A 700-token prompt admitted while another request decodes prefills in 64-token pieces, and the decoding
    request verifies rounds between them (at most half the time goes to prefill); both replies are exact."""

    eng = _engine(ckpt, batch=2, piece=64, share=0.5)
    sampling = _sampling(greedy)
    pa, pb = _prompt(60, 30), _prompt(61, 700)
    want_a, want_b = _serial(ref, pa, sampling, 160), _serial(ref, pb, sampling, 24)
    started = threading.Event()
    got: dict = {}

    def run_a():
        got["a"] = _run(eng, pa, sampling, policy="2", tokens=160, on_tokens=lambda new: started.set() or False)

    t = threading.Thread(target=run_a)
    t.start()
    assert started.wait(300)
    got["b"] = _run(eng, pb, sampling, policy="auto", tokens=24)
    t.join(600)
    assert got["a"][0] == want_a and got["b"][0] == want_b
    sb = got["b"][1]
    assert sb["pieces"] == -(-700 // 64), sb
    slot_a, slot_b = got["a"][1]["slot"], sb["slot"]
    trace = list(eng.batch.trace)
    piece_rounds = [r for r, pieces, _ in trace if slot_b in pieces]
    assert len(piece_rounds) == sb["pieces"]
    span = [(pieces, active) for r, pieces, active in trace if piece_rounds[0] <= r <= piece_rounds[-1]]
    assert all(slot_a in active for _, active in span)                        # A verified every round meanwhile
    assert sum(1 for pieces, _ in span if not pieces) >= 1                    # and some rounds had no piece
    _free(eng)


@gpu
@pytest.mark.parametrize("greedy", GREEDY, ids=GIDS)
def test_fast_prefill_admissions(ckpt, greedy):
    """patches/0080 in batch mode: prompts prefill fast in pieces on the chunk grid (128), batched replies equal the
    lone fast engine's, follow-ups resume from grid snapshots, and slots keep grid snapshots only."""

    lone = _engine(ckpt, fast=True, rows=128, rows_max=256)
    eng = _engine(ckpt, batch=2, fast=True, rows=128, rows_max=256, piece=100)
    sampling = _sampling(greedy)
    pa, pb = _prompt(70, 700), _prompt(71, 300)
    want_a, _ = _run(lone, pa, sampling, policy="auto", tokens=24)
    want_b, _ = _run(lone, pb, sampling, policy="2", tokens=24)
    assert want_a == _serial(lone, pa, sampling, 24)
    (got_a, sa), (got_b, sb) = eng.batch.generate_batch([
        dict(prompt=pa, max_tokens=24, sampling=sampling, policy="auto"),
        dict(prompt=pb, max_tokens=24, sampling=sampling, policy="2")])
    assert got_a == want_a and got_b == want_b
    assert sa["fast_prefill"] == 128 and sa["pieces"] == -(-700 // 128) and sb["pieces"] == -(-300 // 128)
    for cache in eng.batch.caches:
        assert all(c.grid == 128 and len(c.ids) % 128 == 0 for c in cache)
    after = pa + got_a + [3, 4, 5]
    (warm, sw), = eng.batch.generate_batch([dict(prompt=after, max_tokens=16, sampling=sampling)])
    assert sw["cached"] == len(pa) // 128 * 128 and sw["slot"] == sa["slot"]
    cold, _ = _run(lone, after, sampling, tokens=16, draft=False)
    assert warm == cold
    # an exact request on the same engine (tf_knobs) never resumes from a fast snapshot
    (exact, se), = eng.batch.generate_batch([dict(prompt=after, max_tokens=16, sampling=sampling,
                                                  knobs={"fast_prefill": 0})])
    assert se["cached"] == 0 and "fast_prefill" not in se
    ex, _ = _run(lone, after, sampling, tokens=16, draft=False, knobs_={"fast_prefill": 0})
    assert exact == ex
    _free(lone, eng)


@gpu
@pytest.mark.parametrize("greedy", GREEDY, ids=GIDS)
def test_lean_prefill_admissions(ckpt, greedy):
    """patches/0082 in batch mode: 256-row fast chunks through the lean set on 64-row window buffers."""

    lone = _engine(ckpt, fast=True, lean=True, block=64, rows=256, rows_max=256)
    eng = _engine(ckpt, batch=2, fast=True, lean=True, block=64, rows=256, rows_max=256, piece=256)
    assert eng.e.lean is not None and eng.e.rows == 64
    sampling = _sampling(greedy)
    pa, pb = _prompt(80, 600), _prompt(81, 290)
    want = [_run(lone, p, sampling, policy=pol, tokens=24)[0] for p, pol in ((pa, "auto"), (pb, "f3"))]
    got = eng.batch.generate_batch([dict(prompt=pa, max_tokens=24, sampling=sampling, policy="auto"),
                                    dict(prompt=pb, max_tokens=24, sampling=sampling, policy="f3")])
    assert [t for t, _ in got] == want
    assert got[0][1]["pieces"] == 3
    _free(lone, eng)


# -- per-request knobs ----------------------------------------------------------------------------------------------
KNOB_SETS = [{"prefill_rows": 7, "auto_fdrafts": 2}, {"fast_prefill": 1, "prefill_rows": 128},
             {"expert_loop": 0, "lookup": 0, "depth": "cost"}, {"longctx_graphs": 0, "prefill_rows": 256},
             {"depth": "threshold", "auto_fdrafts": 7, "profile": 0}, {}]


@gpu
@pytest.mark.parametrize("greedy", GREEDY, ids=GIDS)
def test_request_knobs_per_sequence(ckpt, greedy):
    """Each request runs its own tf_knobs (sent in its header), batched with requests running others: its tokens
    are the lone engine's with the same knobs, the response echoes them, and the defaults stay."""

    lone = _engine(ckpt, rows=64, rows_max=256)
    eng = _engine(ckpt, batch=2, rows=64, rows_max=256, piece=96)
    defaults = eng._knob_state()
    sampling = _sampling(greedy)
    prompts = [_prompt(90 + i, 150 + 40 * i) for i in range(len(KNOB_SETS))]
    want = [_run(lone, p, sampling, policy="auto", tokens=24, knobs_=ks)[0] for p, ks in zip(prompts, KNOB_SETS)]
    got = eng.batch.generate_batch([dict(prompt=p, max_tokens=24, sampling=sampling, policy="auto", knobs=ks)
                                    for p, ks in zip(prompts, KNOB_SETS)])
    assert [t for t, _ in got] == want
    for (_, stats), ks in zip(got, KNOB_SETS):
        for k, v in ks.items():
            assert stats["tf_knobs"][k] == v, (k, stats["tf_knobs"])
    assert got[1][1]["fast_prefill"] == 128
    assert eng._knob_state() == defaults
    with pytest.raises(ValueError, match="share their rounds"):
        eng.batch.generate_batch([dict(prompt=prompts[0], max_tokens=4, knobs={"calib_online": 1})])
    _free(lone, eng)


# -- long contexts ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def long_ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter
    from test_patches import _index_heads_32

    path = tmp_path_factory.mktemp("glm_batch2_long")
    _checkpoint(path / "model", exl3=True)
    _index_heads_32(path / "model")
    _drafter(path / "dflash2")
    return path


@gpu
@pytest.mark.parametrize("greedy", GREEDY, ids=GIDS)
@pytest.mark.parametrize("latent_kv", [True, False], ids=["latent", "expanded"])
def test_long_contexts_batched(long_ckpt, latent_kv, greedy):
    """Past 2,051 tokens: a 3,000-token prompt (two pieces across the dense limit), one crossing the limit while it
    decodes, and a short one, batched. Replies equal the lone engine's; sparse rounds replay graphs keyed by pool
    bucket; another slot's MTP steps past the limit replay its own long-context graphs."""

    lone = _engine(long_ckpt, latent_kv=latent_kv, context=4096, rows=64)
    eng = _engine(long_ckpt, batch=2, latent_kv=latent_kv, context=4096, rows=64)
    sampling = _sampling(greedy)
    specs = [(_prompt(100, 3000), "2"), (_prompt(101, 2040), "auto"), (_prompt(102, 40), "f3"),
             (_prompt(103, 2600), "o")]
    want = [_serial(lone, p, sampling, 24) for p, _ in specs]
    got = eng.batch.generate_batch([dict(prompt=p, max_tokens=24, sampling=sampling, policy=pol)
                                    for p, pol in specs])
    assert [t for t, _ in got] == want
    assert got[0][1]["pieces"] == 2
    bat = eng.batch
    assert any(m > 0 for key in bat.multi for m in key[2]), list(bat.multi)      # a sparse round on a graph
    assert any(g.long for g in bat.graphs[1:]) or any(k[0] == "mtp" for k in bat.graphs[0].long)
    # a follow-up past the limit resumes on the slot that kept the reply's state
    (r, s0), _ = bat.generate_batch([dict(prompt=specs[0][0], max_tokens=24, sampling=sampling, policy="2"),
                                     dict(prompt=_prompt(104, 30), max_tokens=24, sampling=sampling, policy="2")])
    assert r == want[0]
    after = specs[0][0] + r + [7, 8]
    (warm, sw), = bat.generate_batch([dict(prompt=after, max_tokens=16, sampling=sampling, policy="2")])
    assert sw["cached"] >= len(specs[0][0]) and sw["slot"] == s0["slot"]
    assert warm == _serial(lone, after, sampling, 16)
    _free(lone, eng)


# -- clients -----------------------------------------------------------------------------------------------------
@gpu
def test_client_gone_mid_reply(ref, b2):
    """A request whose client stops listening ends at the next round; its batch-mate is untouched, and the slot
    serves the next request exactly."""

    pa, pb = _prompt(15, 26), _prompt(16, 31)
    want_a, want_b = _serial(ref, pa, None, 200), _serial(ref, pb, None, 30)
    seen: list[int] = []

    def gone(new: list[int]) -> bool:
        seen.extend(new)
        return len(seen) >= 3

    (got_a, sa), (got_b, sb) = b2.batch.generate_batch([
        dict(prompt=pa, max_tokens=200, on_tokens=gone), dict(prompt=pb, max_tokens=30)])
    assert sa["cancelled"] and len(got_a) < 200 and got_a == want_a[:len(got_a)], sa
    assert got_b == want_b and not sb["cancelled"]
    (again, _), = b2.batch.generate_batch([dict(prompt=pa, max_tokens=30)])
    assert again == want_a[:30]


@gpu
@pytest.mark.parametrize("greedy", GREEDY, ids=GIDS)
def test_background_request_steps_aside(ckpt, ref, greedy):
    """Two background requests fill both slots; a foreground one arrives: the later background request stops, the
    foreground one takes its slot, and the background one runs again from the start. Every caller receives
    serial decoding's tokens (the rerun's tokens it already had are not sent twice)."""

    eng = _engine(ckpt, batch=2)
    sampling = _sampling(greedy)
    pa, pb, pc = _prompt(110, 30), _prompt(111, 35), _prompt(112, 25)
    want = [_serial(ref, pa, sampling, 150), _serial(ref, pb, sampling, 150), _serial(ref, pc, sampling, 20)]
    got: dict = {}
    both = threading.Barrier(3)
    chunks: dict = {"a": [], "b": []}

    def bg(name, prompt):
        def listen(new):
            chunks[name].append(list(new))
            if len(chunks[name]) == 2:
                both.wait(120)
            return False

        got[name] = _run(eng, prompt, sampling, policy="2", tokens=150, background=True, on_tokens=listen)

    threads = [threading.Thread(target=bg, args=("a", pa)), threading.Thread(target=bg, args=("b", pb))]
    for t in threads:
        t.start()
    both.wait(300)                                            # both background requests are decoding
    got["c"] = _run(eng, pc, sampling, policy="auto", tokens=20)
    for t in threads:
        t.join(600)
    assert got["a"][0] == want[0] and got["b"][0] == want[1] and got["c"][0] == want[2]
    stepped = [n for n in "ab" if got[n][1].get("preempted")]
    assert len(stepped) == 1, (got["a"][1], got["b"][1])
    assert sum(len(c) for c in chunks[stepped[0]]) == 150        # nothing delivered twice
    _free(eng)


@gpu
def test_concurrent_streaming_callers(ref, b2):
    """Threads through ``generate`` (the server's path; each thread's own policy and knobs), tokens arriving in
    several chunks as rounds commit them."""

    from tensorfold.engine.exact_sampling import Sampling

    sampling = Sampling(99, 1.0, 20, 0.95)
    prompts = [_prompt(13, 30), _prompt(14, 44), _prompt(17, 52)]
    want = [_serial(ref, p, sampling, 32) for p in prompts]
    got: list = [None] * 3
    parts: list = [[], [], []]
    start = threading.Barrier(3)

    def run(i: int, policy, ks) -> None:
        start.wait()
        got[i] = _run(b2, prompts[i], sampling, policy=policy, tokens=32, knobs_=ks,
                      on_tokens=lambda new: parts[i].append(len(new)) and False)

    threads = [threading.Thread(target=run, args=(0, None, None)),
               threading.Thread(target=run, args=(1, "a:0.6:0.85", {"prefill_rows": 5})),
               threading.Thread(target=run, args=(2, "f3", {"auto_fdrafts": 3}))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=600)
    assert [g[0] for g in got] == want
    assert all(len(p) > 3 and sum(p) == 32 for p in parts)


@gpu
def test_memory_trims_the_slots(ckpt, ref):
    """Admission control at load: with more memory kept free than the GPU has, one slot is all that fits, and the
    engine serves requests one at a time, exactly."""

    eng = _engine(ckpt, batch=4, reserve=1e6)
    assert eng.batch.n == 1 and len(eng.batch.states) == 1
    pa, pb = _prompt(120, 30), _prompt(121, 20)
    got = eng.batch.generate_batch([dict(prompt=pa, max_tokens=16), dict(prompt=pb, max_tokens=16)])
    assert [t for t, _ in got] == [_serial(ref, pa, None, 16), _serial(ref, pb, None, 16)]
    _free(eng)


@gpu
@pytest.mark.parametrize("greedy", GREEDY, ids=GIDS)
def test_mlx_checkpoint_batched(tmp_path_factory, greedy):
    """The MLX 4-bit layout (Q4 routed experts through qmm's grouped kernels) batched."""

    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_batch2_mlx")
    _checkpoint(path / "model")
    _drafter(path / "dflash2")
    lone, eng = _engine(path), _engine(path, batch=2)
    sampling = _sampling(greedy)
    pa, pb = _prompt(130, 45), _prompt(131, 36)
    want = [_serial(lone, p, sampling, 32) for p in (pa, pb)]
    for pol_a, pol_b in (("auto:1:1:0", "2"), ("fc5:0.3", "a:0.6:0.85"), (None, "0")):
        got = eng.batch.generate_batch([dict(prompt=pa, max_tokens=32, sampling=sampling, policy=pol_a),
                                        dict(prompt=pb, max_tokens=32, sampling=sampling, policy=pol_b)])
        assert [t for t, _ in got] == want, (pol_a, pol_b)
    _free(lone, eng)


# -- tensor parallelism -------------------------------------------------------------------------------------------
class _Done(Exception):
    pass


@gpu
@pytest.mark.parametrize("greedy", GREEDY, ids=GIDS)
def test_follower_replays_rank0_rounds(ckpt, greedy):
    """A second batch engine replays rank 0's round plans and prompts through ``follow``, as rank 1 does
    (``_TwoCopies`` makes its model rank 0's, so any difference would be a decision): every request ends with the
    same tokens, keeps and drafters, and no plan is left over or missing."""

    r0 = _engine(ckpt, batch=2, piece=64)
    r1 = _engine(ckpt, batch=2, piece=64)
    # load time: both ranks price rounds with the slower rank's costs; simulate that
    r1.batch.costs = r0.batch.costs
    r1.batch.round_costs = batchplan.RoundCosts(r0.batch.costs)
    sent: list[list[int]] = []
    share = r0._share

    def record(values):
        got = share(values)
        sent.append(list(got))
        return got

    def replay(values):
        assert values is None
        if not sent:
            raise _Done
        return sent.pop(0)

    r0._share, r1._share = record, replay
    sampling = _sampling(greedy)
    reqs = [dict(prompt=_prompt(140, 200), max_tokens=40, sampling=sampling, policy="auto",
                 knobs={"prefill_rows": 9}),
            dict(prompt=_prompt(141, 60), max_tokens=24, sampling=sampling, policy="o"),
            dict(prompt=_prompt(142, 90), max_tokens=30, sampling=sampling, policy="f3"),
            dict(prompt=_repeated(143), max_tokens=20, sampling=sampling, policy="l7")]
    r0.batch.generate_batch(reqs)
    time.sleep(0.5)                                           # rank 0's loop is back waiting for work
    with pytest.raises(_Done):
        r1.batch.follow()
    key = lambda d: (d["sha256"], tuple(d["keeps"]), d["arms"], d["cancelled"])      # noqa: E731
    assert [key(d) for d in r1.batch.log] == [key(d) for d in r0.batch.log]
    assert len(r0.batch.log) == len(reqs)
    assert [t[1:] for t in r1.batch.trace] == [t[1:] for t in r0.batch.trace]
    del r0._share, r1._share
    _free(r0, r1)
