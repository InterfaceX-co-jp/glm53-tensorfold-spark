"""patches/0090 (per-request engine knobs, ``glm5_next/cuda/knobs.py``): ``"tf_knobs": {...}`` switches the
GLM53_TF_* speed knobs for one request, on both ranks, then the load-time defaults apply again.

Every knob only picks a speed path already proven exact (drafted == serial, chunk-size independence, bit-identical
expert kernels, timing-only probes), so the same request must give the same tokens under every knob set. Checked:

- host only (no GPU needed): validation (unknown keys, load-only keys with their reason, ranges, types, the
  prefill maximum, batch mode (patches/0120: only calib_online refused), long-context graphs without their
  buffers), the header block round trip, and the server's ``check`` turning a bad ``tf_knobs`` into a message
  (HTTP 400);
- GPU, TensorFold's synthetic EXL3 checkpoint with the DFlash2 drafter (one GPU playing rank 0 of two), loaded with
  GLM53_TF_PREFILL_ROWS=64 and GLM53_TF_PREFILL_ROWS_MAX=512: every knob set gives serial decoding's tokens,
  sampled and greedy, drafted and serial, fresh and resumed; the knobs are back to the defaults after each request
  (also after a failing one) and the response echoes what ran; an invalid set fails before anything is shared
  with rank 1; lookup / depth / auto_fdrafts / calib_online / profile have their effect; a second engine replaying
  rank 0's headers through ``follow`` (as rank 1 does, with different defaults of its own) runs every request with
  rank 0's knobs and exactly rank 0's windows, and returns to its own defaults.

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda pytest -q tests/cuda/test_knob_patches.py
"""

from __future__ import annotations

import pytest

try:
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:          # the host-only tests still run
    CUDA = False

from tensorfold.families.glm5_next.cuda import knobs  # noqa: E402

gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")


# -- host only -----------------------------------------------------------------------------------------------------
def test_parse_accepts_any_subset():
    assert knobs.parse(None, rows_max=512) == {}
    assert knobs.parse({}, rows_max=512) == {}
    got = knobs.parse({"lookup": 0, "lookup_min": 64, "auto_fdrafts": 1, "expert_loop": True, "prefill_rows": 512,
                       "calib_online": 1, "longctx_graphs": 0, "profile": False, "depth": "COST"}, rows_max=512)
    assert got == {"lookup": 0, "lookup_min": 64, "auto_fdrafts": 1, "expert_loop": 1, "prefill_rows": 512,
                   "calib_online": 1, "longctx_graphs": 0, "profile": 0, "depth": "cost"}
    assert knobs.parse({"prefill_rows": 1}, rows_max=8) == {"prefill_rows": 1}
    assert set(knobs.PER_REQUEST) == set(knobs.RANGES) | {"depth"}


@pytest.mark.parametrize("bad,words", [
    ({"bogus": 1}, "unknown knob"),
    ({"prefill_row": 64}, "unknown knob"),
    ({"lookup": 2}, "0 to 1"),
    ({"lookup_min": 0}, "1 to 64"),
    ({"lookup_min": 65}, "1 to 64"),
    ({"auto_fdrafts": 0}, "1 to 7"),
    ({"auto_fdrafts": 8}, "1 to 7"),
    ({"prefill_rows": 0}, "1 to 512"),
    ({"prefill_rows": 513}, "PREFILL_ROWS_MAX"),
    ({"expert_loop": "1"}, "integer"),
    ({"profile": 1.0}, "integer"),
    ({"prefill_rows": None}, "integer"),
    ({"depth": "deep"}, "cost"),
    ({"depth": 1}, "cost"),
])
def test_parse_rejects(bad, words):
    with pytest.raises(ValueError, match=words):
        knobs.parse(bad, rows_max=512)


@pytest.mark.parametrize("key", sorted(knobs.LOAD_ONLY))
def test_load_only_knobs_say_why(key):
    with pytest.raises(ValueError, match="cannot change per request") as exc:
        knobs.parse({key: 1}, rows_max=512)
    assert "GLM53_TF_" in str(exc.value)
    for k in ("nonexpert", "latent_kv", "comm", "batch"):
        assert k in knobs.LOAD_ONLY


def test_parse_engine_limits():
    with pytest.raises(ValueError, match="must be an object"):
        knobs.parse([["lookup", 0]], rows_max=64)
    # patches/0120: batched requests carry their own knobs, all but the online cost table
    assert knobs.parse({"lookup": 0}, rows_max=64, batch=True) == {"lookup": 0}
    assert knobs.parse({}, rows_max=64, batch=True) == {}
    with pytest.raises(ValueError, match="share their rounds"):
        knobs.parse({"calib_online": 1}, rows_max=64, batch=True)
    with pytest.raises(ValueError, match="GLM53_TF_LONGCTX_GRAPHS=0"):
        knobs.parse({"longctx_graphs": 1}, rows_max=64, longctx_ok=False)
    assert knobs.parse({"longctx_graphs": 0}, rows_max=64, longctx_ok=False) == {"longctx_graphs": 0}


def test_header_block_round_trip():
    values = {"expert_loop": 0, "prefill_rows": 333, "longctx_graphs": 1, "profile": 1, "auto_fdrafts": 5,
              "calib_online": 0, "fast_prefill": 1,          # fast_prefill: patches/0091
              "prefill_overlap": 1}                          # prefill_overlap: patches/0093
    if "fp8_prefill" in knobs.HEADER:                            # patches/0092
        values["fp8_prefill"] = 1
    if "fat_experts" in knobs.HEADER:                            # patches/0170
        values["fat_experts"] = 1
    for k, v in (("moe_glue", 7), ("mtp_window", 4096), ("hc_fused", 3), ("attn_bm32", 1)):   # patches/0190
        if k in knobs.HEADER:
            values[k] = v
    block = knobs.encode(values)
    assert block[0] == len(knobs.HEADER) == len(block) - 1
    code = [4, 2, 8, 30000, 1, 4]
    got, rest = knobs.decode(block + code)
    assert got == values and rest == code
    with pytest.raises(RuntimeError, match="same patched TensorFold"):
        knobs.decode([len(knobs.HEADER) - 1] + block[1:-1] + code)


def test_server_check_turns_bad_knobs_into_a_message():
    from tensorfold.cuda.server import App

    class Engine:
        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
            raise AssertionError("not called")

        def parse_knobs(self, raw):
            return knobs.parse(raw, rows_max=256)

    app = object.__new__(App)
    app.engine = Engine()
    assert app.check({"messages": [], "tf_knobs": {"prefill_rows": 256, "lookup": 0}}) is None
    assert "unknown knob" in app.check({"messages": [], "tf_knobs": {"rows": 1}})
    assert "1 to 256" in app.check({"messages": [], "tf_knobs": {"prefill_rows": 257}})
    assert "cannot change per request" in app.check({"messages": [], "tf_knobs": {"nonexpert": "q4"}})
    app.engine = type("Plain", (), {"generate": Engine.generate})()
    assert "no per-request knobs" in app.check({"messages": [], "tf_knobs": {"lookup": 0}})
    assert app.check({"messages": []}) is None


# -- GPU: the synthetic checkpoint ---------------------------------------------------------------------------------
KNOB_SETS = [
    {},
    {"prefill_rows": 512},
    {"prefill_rows": 7},
    {"prefill_rows": 100, "expert_loop": 0},
    {"prefill_rows": 333, "expert_loop": 1},
    {"lookup": 0},
    {"lookup": 1, "lookup_min": 2},
    {"auto_fdrafts": 1},
    {"auto_fdrafts": 7, "depth": "cost"},
    {"depth": "threshold"},
    {"calib_online": 1},
    {"profile": 1},
    {"longctx_graphs": 0},
    {"prefill_rows": 48, "expert_loop": 0, "lookup": 0, "auto_fdrafts": 3, "calib_online": 1, "profile": 1,
     "depth": "cost", "longctx_graphs": 1},
]


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_knobs")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


@pytest.fixture(scope="module")
def mp():
    m = pytest.MonkeyPatch()
    m.setenv("GLM53_TF_NONEXPERT", "q4mse")        # plain auto chooses between the drafters (patches/0001)
    yield m
    m.undo()


def _engine(ckpt, mp, *, rows: int = 64, rows_max: int = 512):
    from test_glm_engine import _TwoCopies

    from tensorfold.families.glm5_next.cuda import weights
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    mp.setattr(weights, "NONEXPERT", "q4mse")
    with pytest.MonkeyPatch.context() as m:
        m.setenv("GLM53_TF_PREFILL_ROWS", str(rows))
        m.setenv("GLM53_TF_PREFILL_ROWS_MAX", str(rows_max))
        for k in ("GLM53_TF_BATCH", "GLM53_TF_CALIB_ONLINE", "GLM53_TF_PROFILE", "GLM53_TF_DEPTH",
                  "GLM53_TF_LOOKUP", "GLM53_TF_LOOKUP_MIN", "GLM53_TF_AUTO_FDRAFTS"):
            m.delenv(k, raising=False)
        return GlmEngine(ckpt / "model", rank=0, master="", port=0, drafter=ckpt / "dflash2", comm=_TwoCopies())


@pytest.fixture(scope="module")
def eng(ckpt, mp):
    return _engine(ckpt, mp)


def _run(e, prompt, sampling, knob_set=None, *, draft=True, policy=None, tokens=32, on_tokens=None):
    out: list[int] = []
    e.request.policy = policy
    e.request.stop_eos = False
    e.request.knobs = knob_set
    try:
        stats = e.generate(list(prompt), tokens, sampling, on_tokens or (lambda new: out.extend(new)), draft=draft)
    finally:
        e.request.knobs = None
    return out, stats


def _prompts():
    import numpy as np

    b = [int(t) for t in np.random.default_rng(92).integers(0, 1000, size=40)]
    return [[int(t) for t in np.random.default_rng(91).integers(0, 1000, size=300)], b * 6 + b[:9]]


def _sampling(greedy: bool):
    from tensorfold.engine.exact_sampling import Sampling

    return None if greedy else Sampling(1234, 1.0, 20, 0.95)


class _Windows:
    """The width of every verify window an engine runs."""

    def __init__(self, e):
        self.e, self.orig, self.seen = e.e, e.e.forward, []

        def forward(tokens):
            self.seen.append(len(tokens))
            return self.orig(tokens)

        self.e.forward = forward

    def close(self):
        del self.e.forward


@gpu
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
def test_every_knob_set_gives_the_same_tokens(eng, greedy):
    sampling = _sampling(greedy)
    defaults = eng._knob_state()
    assert defaults["prefill_rows"] == 64 and eng.e.rows == 512 and defaults["expert_loop"] == 1
    for prompt in _prompts():
        serial, _ = _run(eng, prompt, sampling, draft=False)
        for ks in KNOB_SETS:
            drafted, stats = _run(eng, prompt, sampling, ks)
            assert drafted == serial, ks
            echo = stats["tf_knobs"]
            for k, v in ks.items():
                assert echo[k] == (0 if k == "longctx_graphs" else v), (ks, k, echo)   # dense engine: no graphs
            assert eng._knob_state() == defaults, ks          # back to the load-time values
            again, _ = _run(eng, prompt, sampling, ks, draft=False)
            assert again == serial, ks                        # serial decoding under the knobs too


@gpu
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
def test_resume_across_knob_sets(eng, greedy):
    """A reply kept under one chunk size / kernel resumes under another and still equals a fresh prefill."""

    sampling = _sampling(greedy)
    prompt = _prompts()[0]
    reply, _ = _run(eng, prompt, sampling, {"prefill_rows": 512, "expert_loop": 0})
    longer = prompt + reply + [11, 12, 13]
    fresh, _ = _run(eng, longer, sampling, draft=False)
    _run(eng, prompt, sampling, {"prefill_rows": 512, "expert_loop": 0})
    resumed, stats = _run(eng, longer, sampling, {"prefill_rows": 7, "expert_loop": 1})
    assert stats["cached"] > 0 and resumed == fresh


@gpu
def test_knobs_revert_and_echo(eng, capfd):
    from tensorfold.families.glm5_next.cuda import exl3_mm, profile

    prompt = _prompts()[1]
    defaults = eng._knob_state()
    _, stats = _run(eng, prompt, None)
    assert stats["tf_knobs"] == dict(defaults, lookup=1, lookup_min=4, depth="threshold")
    assert "verify_ms" not in stats and eng.e.calib is None
    _, stats = _run(eng, prompt, None, {"profile": 1, "calib_online": 1, "prefill_rows": 5, "expert_loop": 0})
    assert "verify_ms" in stats and eng.online is not None           # rank 0's online table, made on demand
    assert any(line.startswith("GLM53_TF_PROFILE ") for line in capfd.readouterr().err.splitlines())
    assert profile.ON is False and exl3_mm.LOOP is True and eng.e.prefill_rows == 64 and not eng.calib_on
    _, stats = _run(eng, prompt, None)
    assert "verify_ms" not in stats and stats["tf_knobs"] == dict(defaults, lookup=1, lookup_min=4,
                                                                  depth="threshold")
    assert not any(line.startswith("GLM53_TF_PROFILE ") for line in capfd.readouterr().err.splitlines())

    class Boom(Exception):
        pass

    def fail(new):
        raise Boom

    with pytest.raises(Boom):                                         # a failing request restores them too
        _run(eng, prompt, None, {"prefill_rows": 9, "auto_fdrafts": 2, "profile": 1}, on_tokens=fail)
    assert eng._knob_state() == defaults and profile.ON is False
    eng.cache = []


@gpu
def test_invalid_knobs_fail_before_rank1_hears(eng):
    sent = []
    share = eng._share
    eng._share = lambda values: sent.append(values) or share(values)
    try:
        for bad in ({"bogus": 1}, {"prefill_rows": 513}, {"nonexpert": "q4"}, {"comm": "prefetch"},
                    {"lookup_min": 0}, {"depth": "x"}, "prefill_rows=64"):
            with pytest.raises(ValueError):
                _run(eng, _prompts()[0], None, bad)
            assert sent == [], bad
    finally:
        del eng._share
    assert eng._knob_state()["prefill_rows"] == 64


def _header_parts(header):
    n = header[12]
    rest = header[13 + n:]
    values, code = knobs.decode(rest[1:])
    return rest[0], values, code


@gpu
def test_knobs_reach_the_header_and_take_effect(eng):
    sent = []
    share = eng._share

    def record(values):
        got = share(values)
        sent.append(list(got))
        return got

    eng._share = record
    try:
        prompt = _prompts()[1]
        for ks, lookup in (({"lookup": 0}, [0, 4]), ({"lookup_min": 2}, [1, 2]), ({}, [1, 4])):
            sent.clear()
            _run(eng, prompt, None, ks, policy="auto")
            cost, values, code = _header_parts(sent[0])
            assert code[0] == 4 and code[4:6] == lookup, ks          # auto's lookup settings, from the knobs
            assert values == eng._knob_state()
        for depth, flag in (("cost", 1), ("threshold", 0)):
            sent.clear()
            _run(eng, prompt, None, {"depth": depth}, policy="auto")
            assert _header_parts(sent[0])[0] == flag
        sent.clear()
        _run(eng, prompt, None, {"prefill_rows": 200, "auto_fdrafts": 2, "profile": 1})
        _, values, _ = _header_parts(sent[0])
        assert values["prefill_rows"] == 200 and values["auto_fdrafts"] == 2 and values["profile"] == 1
        w = _Windows(eng)                           # at most 2 DFlash2 drafts (3 rows); MTP c3 (4 rows); no lookup
        _run(eng, _prompts()[0], None, {"auto_fdrafts": 2, "lookup": 0, "depth": "threshold"}, tokens=48)
        w.close()
        assert w.seen and max(w.seen) <= 4
    finally:
        del eng._share


class _Done(Exception):
    pass


@gpu
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
def test_follower_runs_rank0_knobs(ckpt, mp, greedy):
    """Rank 0 serves requests with different knob sets; a second engine replays its headers through ``follow`` as
    rank 1 does (``_TwoCopies`` makes its model rank 0's, so any difference would be a decision), with its own
    defaults deliberately different. It runs each request with rank 0's knobs and windows, then its defaults."""

    r0 = _engine(ckpt, mp)
    r1 = _engine(ckpt, mp, rows=32)
    # load time: the all-gather makes every piece the slower rank's, the same on both ranks; simulate that
    r1.base_costs = r1.costs = r0.base_costs
    r1.f_most = 3                                   # rank 1's own "environment" differs from rank 0's
    own = r1._knob_state()
    assert own["prefill_rows"] == 32 and own["auto_fdrafts"] == 3
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
    during: dict[int, list] = {0: [], 1: []}
    for i, r in enumerate((r0, r1)):
        run = r._run

        def spy(*args, r=r, run=run, i=i, **kw):
            during[i].append(r._knob_state())
            return run(*args, **kw)

        r._run = spy
    w0, w1 = _Windows(r0), _Windows(r1)
    sampling = _sampling(greedy)
    prompts = _prompts()
    for prompt, ks in ((prompts[0], {}), (prompts[1], {"prefill_rows": 7, "auto_fdrafts": 7}),
                       (prompts[1] + [3], {"expert_loop": 0, "lookup": 0, "profile": 1}),
                       (prompts[0] + [4], {"calib_online": 1, "depth": "cost", "prefill_rows": 512}),
                       (prompts[0] + [5], {})):
        _run(r0, prompt, sampling, ks)
        with pytest.raises(_Done):
            r1.follow()
        assert during[1][-1] == during[0][-1], ks                    # rank 0's knobs on rank 1
        for k, v in ks.items():
            if k in knobs.HEADER:
                assert during[1][-1][k] == v, (ks, k)
        assert w1.seen == w0.seen, ks                                # the same windows: the same decisions
        assert r1._knob_state() == own, ks                           # then rank 1's own defaults again
        assert r1.costs == r0.costs, ks                              # and rank 0's cost table
    w0.close()
    w1.close()
    del r0, r1
    torch.cuda.empty_cache()


@gpu
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
def test_longctx_graphs_per_request(ckpt, mp, greedy):
    """A long-context engine (loaded with the default GLM53_TF_LONGCTX_GRAPHS=1): a request with
    longctx_graphs=0 runs upstream's eager path (no long-context graph captured, every pool scored) and gives the
    tokens of one with the graphs, crossing 2,051 tokens during decode and starting past it."""

    from test_glm_engine import _TwoCopies

    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    with pytest.MonkeyPatch.context() as m:
        m.setenv("GLM53_TF_LONGCTX_GRAPHS", "1")
        m.setenv("GLM53_TF_PREFILL_ROWS", "64")
        m.setenv("GLM53_TF_PREFILL_ROWS_MAX", "256")
        m.delenv("GLM53_TF_BATCH", raising=False)
        e = GlmEngine(ckpt / "model", rank=0, master="", port=0, drafter=ckpt / "dflash2", context=4096,
                      comm=_TwoCopies())
    assert e.longctx_buffers and e.e.longctx
    sampling = _sampling(greedy)
    import numpy as np

    for n in (2040, 2300):
        prompt = [int(t) for t in np.random.default_rng(n).integers(0, 1000, size=n)]
        before = len(e.e.graphs.long)
        off, stats = _run(e, prompt, sampling, {"longctx_graphs": 0, "prefill_rows": 256}, draft=False)
        assert stats["tf_knobs"]["longctx_graphs"] == 0 and len(e.e.graphs.long) == before
        assert e.e.longctx and e.w.meta["longctx_bound"]                  # the default again
        on, stats = _run(e, prompt, sampling, {"longctx_graphs": 1}, draft=False)
        assert on == off and e.e.graphs.long                             # replayed (captured once)
        for ks in ({"longctx_graphs": 0}, {}, {"longctx_graphs": 0, "prefill_rows": 17}):
            drafted, _ = _run(e, prompt, sampling, ks)
            assert drafted == off, ks
    del e
    torch.cuda.empty_cache()
