"""patches/0130 (GLM53_TF_DECODE_KERNELS=v2): decode-step kernels with fewer launches, on one GPU.

``decode_v2`` replaces, for windows of 1-16 rows (matmuls: up to 32), groups of kernels by fewer kernels with the same
arithmetic: split-K matmuls reduce in the kernel (the last K slice adds the slices in order), the router, its top-k
and the expert grouping are one kernel, the EXL3 experts three (the epilogues run by the last program of a Hadamard
block, the GEMV loads one k tile ahead), KDA's gate matmuls and DSA's key/value projections go in pairs, the shared
expert writes its slot directly, the hyper-connection partial dots load their K blocks at once, and (``v2,pdl``)
kernels are launched as programmatic dependents that prefetch their weights while the previous kernel drains.

Nothing may change a bit. Checked here:

- every new kernel against the kernels it replaces, bit for bit, at the model's per-rank shapes, 1-8 rows (and 16,
  24, 32 where the kernel takes them), and each row alone against the same row inside a window (row invariance);
  in CUDA graphs and eagerly; the ticket counters are left at zero;
- engines (TensorFold's synthetic checkpoint, MLX and EXL3 + q4mse; one GPU as rank 0 of two): logits and hidden rows
  of windows of 1-8 rows equal the v1 engine's (graphs 1-6, eager 7-8); a captured window equals the same window
  run eagerly; serial replies equal v1's and every drafting policy equals serial decoding, sampled and greedy; a
  state left by drafted rounds resumes like a fresh prefill;
- kernel launches per window (printed, and fewer than v1).

``-s`` prints kernel timings at the real per-rank shapes (CUDA graphs of back-to-back calls on distinct weight
copies, so weights come from DRAM) and the synthetic engine's window times, for v1 / v2 / v2,pdl.

Run inside the image:
    PYTHONPATH=/src/TensorFold/tests/cuda pytest -q -s tests/cuda/test_decode_step_patches.py
The whole existing suite can also run on v2 (the knob is read at import):
    GLM53_TF_DECODE_KERNELS=v2 PYTHONPATH=/src/TensorFold/tests/cuda pytest -q tests/cuda/
"""

from __future__ import annotations

import statistics

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.glm5_next.cuda import decode_v2 as dv2, exl3_mm, glue, qmm, weights  # noqa: E402
from tensorfold.families.glm5_next.cuda.forward import chunks_for, commit, compute, stage  # noqa: E402
from test_glm_engine import _TwoCopies, _checkpoint, _drafter, _generate  # noqa: E402

DEV = "cuda"
SPECS = ["v2"] + (["v2,pdl"] if dv2.pdl_supported() else [])
SAMPLINGS = [Sampling(1234, 1.0, 20, 0.95), None]
IDS = ["sampled", "greedy"]
POLICIES = (None, "auto", "auto:1:1:0", "auto:1:2:0", "f7", "fc7:0.3", "fc5:0.3", "7", "c3:0.35", "a:0.6:0.85", "2")

# one rank's shapes of GLM-5.3-Flash (N x K): KDA proj / o / f_b, g_b; DSA proj / q_b / kv_b halves / o; dense MLP;
# shared expert gate+up / down; head
SHAPES = [(12576, 4096), (4096, 4096), (4096, 128), (2048, 4096), (8192, 1536), (8192, 512), (4096, 8192),
          (12288, 4096), (4096, 6144), (4096, 1024), (77440, 4096)]


@pytest.fixture(autouse=True)
def _no_lookup(monkeypatch):
    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")


def _q4(n: int, k: int, seed: int) -> qmm.Q4:
    g = torch.Generator(device=DEV).manual_seed(seed)
    words = torch.randint(-2**31, 2**31 - 1, (n, k // 8), dtype=torch.int64, device=DEV, generator=g).to(torch.int32)
    scales = (torch.rand((n, k // 64), device=DEV, generator=g) * 0.01 + 0.005).to(torch.bfloat16)
    biases = (-7.5 * scales.float()).to(torch.bfloat16)
    return qmm.make_q4(words, scales, biases)


def _x(m: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn((m, k), device=DEV, generator=g).to(torch.bfloat16)


def _part() -> torch.Tensor:
    return torch.empty((8 * 32 * 16384 * 2,), dtype=torch.float32, device=DEV)


# -- the knob -------------------------------------------------------------------------------------------------
def test_knob_parsing(monkeypatch):
    assert dv2.parse(None) == dv2.parse("") == dv2.parse("v1") == frozenset()
    assert dv2.parse("v2") == frozenset(dv2.DEFAULT_V2)
    assert "pdl" not in dv2.parse("v2")
    assert dv2.parse("v2,pdl") == frozenset(dv2.DEFAULT_V2) | {"pdl"}
    assert dv2.parse("v2,-exl3,+pdl") == (frozenset(dv2.DEFAULT_V2) - {"exl3"}) | {"pdl"}
    for bad in ("v3", "v2,fast", "fixup"):
        with pytest.raises(ValueError, match=dv2.ENV):
            dv2.parse(bad)
    before = dv2.SPEC
    with dv2.using("v2,-route"):
        assert "route" not in dv2.SPEC and qmm.V2 is dv2 and exl3_mm.V2 is dv2
        assert isinstance(glue._hc_partial, dv2._Launch)
    with dv2.using(""):
        assert qmm.V2 is None and exl3_mm.V2 is None and glue._hc_partial is dv2._ORIG["_hc_partial"]
    assert dv2.SPEC == before


# -- kernels, bit for bit ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("n,k", SHAPES)
def test_matmul_equals_v1(spec, n, k):
    q = _q4(n, k, n + k)
    part = _part()
    for f32 in (False, True):
        for m in (1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32):
            x = _x(m, k, 7 * m + k)
            with dv2.using(""):
                ref = qmm.matmul(x, q, f32=f32, part=part).clone()
            with dv2.using(spec):
                got = qmm.matmul(x, q, f32=f32, part=part).clone()
                assert torch.equal(got, ref), (n, k, m, f32)
                if m <= 8:
                    for r in range(m):                          # each row alone: the bits it gets in the window
                        alone = qmm.matmul(x[r:r + 1], q, f32=f32, part=part)
                        assert torch.equal(alone[0], got[r]), (n, k, m, r, f32)
            assert not dv2.counters(part).any()                 # every launch leaves its tickets at zero
    # a strided output (the shared expert's slot of the expert outputs)
    with dv2.using(spec):
        for m in (1, 8):
            x = _x(m, k, 3 + m)
            ey = torch.zeros((m, 9, n), dtype=torch.float32, device=DEV)
            qmm.matmul(x, q, out=ey[:, 8], f32=True, part=part)
            with dv2.using(""):
                ref = qmm.matmul(x, q, f32=True, part=part)
            assert torch.equal(ey[:, 8], ref) and not ey[:, :8].any()


@pytest.mark.parametrize("spec", SPECS)
def test_matmul_in_graphs(spec):
    q = _q4(12576, 4096, 1)
    part = _part()
    for m in (1, 4, 8):
        x = _x(m, 4096, m)
        with dv2.using(""):
            ref = qmm.matmul(x, q, part=part).clone()
        with dv2.using(spec):
            out = torch.empty((m, q.n), dtype=torch.bfloat16, device=DEV)
            qmm.matmul(x, q, out=out, part=part)                         # eager first (tickets allocated)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(3):
                    qmm.matmul(x, q, out=out, part=part)
            out.zero_()
            g.replay()
            g.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, ref), m
            assert not dv2.counters(part).any()


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("n,k", [(4096, 128), (8192, 512)])
def test_pairs_equal_v1(spec, n, k):
    """KDA's f_b / g_b (one launch after one group-sum launch over [f_a | g_a]) and DSA's kv_b key / value rows."""

    q0, q1 = _q4(n, k, 11), _q4(n, k, 12)
    part = _part()
    for m in (1, 3, 8, 16, 32):
        base = _x(m, 3 * k, m)
        x0, x1 = base[:, :k], base[:, k:2 * k]                          # strided rows, adjacent columns
        with dv2.using(""):
            r0 = qmm.matmul(x0, q0, qmm.group_sums(x0), part=part).clone()
            r1 = qmm.matmul(x1, q1, qmm.group_sums(x1), part=part).clone()
        with dv2.using(spec):
            xs = qmm.group_sums(base[:, :2 * k])
            o0 = torch.empty((m, n), dtype=torch.bfloat16, device=DEV)
            o1 = torch.empty_like(o0)
            kg = k // 64
            assert dv2.pair(x0, xs[:, :kg], q0, o0, x1, xs[:, kg:], q1, o1, part)
            p0 = torch.empty_like(o0)
            p1 = torch.empty_like(o0)
            lat = base[:, :k]
            assert dv2.kv_pair(lat, qmm.group_sums(lat), q0, q1, p0, p1, part)
        assert torch.equal(o0, r0) and torch.equal(o1, r1), m
        with dv2.using(""):
            s0 = qmm.matmul(lat, q0, part=part)
            s1 = qmm.matmul(lat, q1, part=part)
        assert torch.equal(p0, s0) and torch.equal(p1, s1), m
        assert not dv2.counters(part).any()


def _route_inputs(m: int, ne: int = 288, d: int = 4096, seed: int = 0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = (torch.randn((ne, d), device=DEV, generator=g) * 0.02).to(torch.bfloat16)
    bias = torch.randn((ne,), device=DEV, generator=g) * 0.01
    x = torch.randn((m, d), device=DEV, generator=g).to(torch.bfloat16)
    return x, w, bias


def _route_run(spec, x, w, bias, top_k, ne, graph=False):
    m = x.shape[0]
    slots = top_k + 1
    maxu = min(m * top_k, ne) + 1
    mlog = torch.zeros((m, ne), dtype=torch.float32, device=DEV)
    pick = torch.zeros((m, slots), dtype=torch.int32, device=DEV)
    wts = torch.zeros((m, slots), dtype=torch.float32, device=DEV)
    grp = qmm.Group(torch.zeros((maxu,), dtype=torch.int32, device=DEV), torch.zeros((1,), dtype=torch.int32, device=DEV),
                    torch.full((maxu, m), -1, dtype=torch.int32, device=DEV))

    def once():
        if not dv2.route(x, w, bias, mlog, pick, wts, grp, top_k, ne, 2.5, True):
            glue.router(x, w, mlog)
            glue.select(mlog, bias, pick, wts, grp.ids, grp.count, grp.members, top_k, ne, 2.5, True)

    with dv2.using(spec):
        once()
        if graph:
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                once()
            for t in (mlog, pick, wts, grp.ids, grp.count, grp.members):
                t.fill_(7)
            g.replay()
        torch.cuda.synchronize()
    return mlog, pick, wts, grp.ids, grp.count, grp.members


@pytest.mark.parametrize("spec", SPECS)
def test_route_equals_v1(spec):
    for m in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16):
        x, w, bias = _route_inputs(m, seed=m)
        ref = _route_run("", x, w, bias, 8, 288)
        for graph in (False, True):
            got = _route_run(spec, x, w, bias, 8, 288, graph)
            for a, b, name in zip(ref, got, ("mlog", "pick", "wts", "ids", "count", "members")):
                assert torch.equal(a, b), (m, graph, name)
        if m <= 8:                                     # rows alone: the same logits, picks and weights
            for r in range(m):
                alone = _route_run(spec, x[r:r + 1], w, bias, 8, 288)
                for a, b in zip(alone[:3], ref[:3]):
                    assert torch.equal(a[0], b[r]), (m, r)
    # the synthetic model's router (8 experts, top 2)
    x, w, bias = _route_inputs(5, ne=8, d=512, seed=3)
    for a, b in zip(_route_run("", x, w, bias, 2, 8), _route_run(spec, x, w, bias, 2, 8)):
        assert torch.equal(a, b)


@pytest.mark.parametrize("spec", SPECS)
def test_hc_pre_equals_v1(spec):
    D, S = 4096, 4
    g = torch.Generator(device=DEV).manual_seed(5)
    fn = (torch.randn((24, S * D), device=DEV, generator=g) * 0.01).to(torch.bfloat16)
    base = torch.randn((24,), device=DEV, generator=g) * 0.1
    scale = torch.randn((3,), device=DEV, generator=g) * 0.1 + 1
    nw = (torch.randn((D,), device=DEV, generator=g) * 0.05 + 1).to(torch.bfloat16)
    for m in (1, 3, 8, 16):
        x = (torch.randn((m, S * D), device=DEV, generator=g) * 0.3).to(torch.bfloat16)

        def run(sp):
            out = torch.zeros((m, D), dtype=torch.bfloat16, device=DEV)
            xs = torch.zeros((m, D // 64), device=DEV)
            post = torch.zeros((m, S), device=DEV)
            comb = torch.zeros((m, 16), device=DEV)
            part = torch.zeros((m, glue.HC_BLOCKS, 32), device=DEV)
            with dv2.using(sp):
                glue.hc_pre(x, fn, base, scale, nw, out, xs, post, comb, part, 1e-5, 1e-6, 20)
                y = torch.empty((m, S * D), dtype=torch.bfloat16, device=DEV)
                gathered = torch.randn((2, m, D), device=DEV, generator=torch.Generator(device=DEV).manual_seed(m))
                glue.hc_post(x, y, gathered, post, comb)
            return out, xs, post, comb, part, y

        for a, b in zip(run(""), run(spec)):
            assert torch.equal(a, b), m


def _exl3(E: int, D: int, NI: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)

    def trellis(k, n):
        return torch.randint(-2**15, 2**15, (E, k // 16, n // 16, 64), dtype=torch.int16, device=DEV, generator=g)

    def hs(n, sc):
        return (torch.randn((E, n), device=DEV, generator=g) * sc).to(torch.float16)

    w = lambda t: t.view(torch.int32)                                     # noqa: E731
    return exl3_mm.Exl3Experts(w(trellis(D, NI)), w(trellis(D, NI)), w(trellis(NI, D)), hs(D, 0.02), hs(D, 0.02),
                               hs(NI, 0.5), hs(NI, 0.5), hs(NI, 0.05), hs(D, 0.2), E, NI, D)


def _exl3_route(m: int, E: int, top_k: int, seed: int, rows_max: int = 16):
    """Random distinct picks per row (the shared expert, id E, last) and the window's grouping (glue.select's)."""

    g = torch.Generator().manual_seed(seed)
    slots = top_k + 1
    pick = torch.full((rows_max, slots), E, dtype=torch.int32)
    for r in range(m):
        pick[r, :top_k] = torch.randperm(E, generator=g)[:top_k].to(torch.int32)
    used = sorted({int(e) for e in pick[:m, :top_k].flatten()})
    maxu = min(m * top_k, E) + 1
    ids = torch.zeros((maxu,), dtype=torch.int32)
    members = torch.full((maxu, m), -1, dtype=torch.int32)
    for u, e in enumerate(used):
        ids[u] = e
        j = 0
        for r in range(m):
            for s in range(top_k):
                if int(pick[r, s]) == e:
                    members[u, j] = r * 32 + s
                    j += 1
    return pick.to(DEV), qmm.Group(ids.to(DEV), torch.tensor([len(used)], dtype=torch.int32, device=DEV), members.to(DEV))


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("dims", [(4096, 1024, 64, 8), (512, 128, 8, 2)], ids=["real", "synthetic"])
def test_exl3_routed_equals_v1(spec, dims):
    """The routed experts' outputs (fp32, every routed slot) equal exl3.cu's; rows alone equal rows in the window;
    the shared slot is never written; captured and eager alike."""

    D, NI, E, top_k = dims
    ex = _exl3(E, D, NI, 3)
    slots = top_k + 1
    s = exl3_mm.Scratch(16, slots, D, NI, DEV)
    for m in (1, 2, 3, 4, 5, 6, 7, 8, 16):
        pick, grp = _exl3_route(m, E, top_k, m)
        x = _x(m, D, 100 + m)

        def run(sp, xx, pk, gg, n, graph=False):
            y = torch.full((16 * slots, D), 3.0, dtype=torch.float32, device=DEV)
            with dv2.using(sp):
                exl3_mm.routed(xx, pk, gg, ex, s, y, n, 10.0)
                if graph:
                    torch.cuda.synchronize()
                    cg = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(cg):
                        exl3_mm.routed(xx, pk, gg, ex, s, y, n, 10.0)
                    y.fill_(3.0)
                    cg.replay()
                torch.cuda.synchronize()
            return y.view(16, slots, D)[:n]

        ref = run("", x, pick, grp, m)
        assert torch.isfinite(ref[:, :top_k]).all()
        for graph in (False, True):
            got = run(spec, x, pick, grp, m, graph)
            assert torch.equal(got, ref), (m, graph)
            assert (got[:, top_k] == 3.0).all()                        # the shared slot is left alone
        cnt = getattr(s, "_v2_cnt", None)
        assert cnt is None or not cnt.any()
        if m <= 8:
            for r in range(m):
                pr, gr = _exl3_route(1, E, top_k, 0)
                pr[0] = pick[r]
                used = sorted({int(e) for e in pick[r, :top_k]})
                gr.ids[:len(used)] = torch.tensor(used, dtype=torch.int32, device=DEV)
                gr.count.fill_(len(used))
                gr.members.fill_(-1)
                for u, e in enumerate(used):
                    gr.members[u, 0] = [int(v) for v in pick[r, :top_k]].index(e)
                alone = run(spec, x[r:r + 1], pr, gr, 1)
                assert torch.equal(alone[0, :top_k], ref[r, :top_k]), (m, r)


# -- engines ----------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    out = {}
    for kind in ("mlx", "exl3"):
        path = tmp_path_factory.mktemp(f"glm_dec_{kind}")
        _checkpoint(path / "model", exl3=kind == "exl3")
        _drafter(path / "dflash2")
        out[kind] = path
    return out


def _engine(path, spec: str):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    mp = pytest.MonkeyPatch()
    mp.setattr(weights, "NONEXPERT", "q4mse")             # the deployed EXL3 configuration (patch 0001)
    mp.setenv("GLM53_TF_NONEXPERT", "q4mse")
    try:
        with dv2.using(spec):                              # the graphs capture this spec's kernels
            eng = GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", comm=_TwoCopies())
    finally:
        mp.undo()
    eng._v2 = spec
    return eng


@pytest.fixture(scope="module", params=["mlx", "exl3"])
def engines(request, ckpts):
    """{spec: engine} on one checkpoint: v1 ("") and every v2 variant."""

    path = ckpts[request.param]
    return request.param, {spec: _engine(path, spec) for spec in [""] + SPECS}


def _window(eng, prompt, rows, *, eager=False):
    e = eng.e
    with dv2.using(eng._v2):
        e.reset()
        e.forward(prompt)
        commit(e.w, e.st, e.buf, len(prompt), len(prompt))
        if eager:
            R = stage(e.w, e.st, e.buf, rows)
            logits = compute(e.w, e.st, e.buf, R, nch=chunks_for(e.st, R), host_pos=e.st.pos).clone()
        else:
            logits = e.forward(rows).clone()
        torch.cuda.synchronize()
        out = logits, e.buf.hidden[:len(rows)].clone(), e.buf.fnormed[:len(rows)].clone()
        e.reset()
        eng.cache.clear()
    return out


def test_windows_equal_v1_graphs_and_eager(engines):
    kind, engs = engines
    rng = np.random.default_rng(31)
    prompt = [int(t) for t in rng.integers(0, 1000, size=19)]
    for R in range(1, 9):
        rows = [int(t) for t in rng.integers(0, 1000, size=R)]
        ref = _window(engs[""], prompt, rows)
        for spec in SPECS:
            got = _window(engs[spec], prompt, rows)
            for a, b in zip(ref, got):
                assert torch.equal(a, b), (kind, spec, R)
            if R <= 6:                                   # the captured window against the same window run eagerly
                eag = _window(engs[spec], prompt, rows, eager=True)
                for a, b in zip(got, eag):
                    assert torch.equal(a, b), (kind, spec, R, "eager")


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
def test_replies_equal_serial(engines, sampling):
    kind, engs = engines
    prompt = list(np.random.default_rng(32).integers(0, 1000, size=43))
    with dv2.using(""):
        serial, _ = _generate(engs[""], prompt, sampling, draft=False, tokens=32)
    for spec in SPECS:
        eng = engs[spec]
        with dv2.using(spec):
            got, _ = _generate(eng, prompt, sampling, draft=False, tokens=32)
            assert got == serial, (kind, spec, "serial")
            for policy in POLICIES:
                drafted, stats = _generate(eng, prompt, sampling, policy=policy, tokens=32)
                assert drafted == serial, (kind, spec, policy)


def test_resume_equals_fresh_prefill(engines):
    kind, engs = engines
    sampling = Sampling(41, 1.0, 20, 0.95)
    rng = np.random.default_rng(33)
    first = list(rng.integers(0, 1000, size=70))           # more than one prefill chunk (v1 kernels past 32 rows)
    unrelated = list(rng.integers(0, 1000, size=9))
    results = []
    for spec in [""] + SPECS:
        eng = engs[spec]
        with dv2.using(spec):
            reply, _ = _generate(eng, first, sampling, policy="auto:1:1:0", tokens=20)
            after = first + reply + [7, 8]
            warm, stats = _generate(eng, after, sampling, tokens=20)
            assert stats["cached"] >= len(first) + len(reply) - 1
            _generate(eng, unrelated, sampling)
            cold, stats = _generate(eng, after, sampling, tokens=20)
            assert stats["cached"] == 0 and warm == cold, (kind, spec)
        results.append((reply, warm))
    assert all(r == results[0] for r in results), kind


def _launches(eng, R: int) -> int:
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    e = eng.e
    with dv2.using(eng._v2):
        e.reset()
        e.forward(list(range(1, 12)))
        commit(e.w, e.st, e.buf, 11, 11)
        R = stage(e.w, e.st, e.buf, list(range(20, 20 + R)))
        compute(e.w, e.st, e.buf, R, nch=chunks_for(e.st, R), host_pos=e.st.pos)          # warm
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            compute(e.w, e.st, e.buf, R, nch=chunks_for(e.st, R), host_pos=e.st.pos)
            torch.cuda.synchronize()
        e.reset()
        eng.cache.clear()
    return sum(1 for ev in prof.events() if ev.device_type == DeviceType.CUDA and "emcpy" not in ev.name
               and "emset" not in ev.name)


def test_fewer_launches(engines):
    kind, engs = engines
    for R in (1, 8):
        base = _launches(engs[""], R)
        for spec in SPECS:
            n = _launches(engs[spec], R)
            print(f"\n[0130] {kind} synthetic model (2 layers + head), {R}-row window: kernels v1 {base}, {spec} {n}")
            assert n < base, (kind, spec, R, n, base)


# -- timings (-s) -------------------------------------------------------------------------------------------------
def _time(fn, calls: int, reps: int = 7) -> float:
    """us per call: a CUDA graph of ``calls`` calls of ``fn(i)``, replayed ``reps`` times (median)."""

    for i in range(calls):
        fn(i)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(calls):
            fn(i)
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / calls)
    return statistics.median(ts)


def _variants():
    return [("v1", "")] + [(s, s) for s in SPECS]


def test_timing_matmuls():
    print("\n[0130] 4-bit matmuls, one rank's shapes, us per call (graph of back-to-back calls on distinct weights)")
    part = _part()
    for n, k in SHAPES:
        one = _q4(n, k, 1).nbytes()
        copies = max(2, -(-96 * 2**20 // one))
        qs = [_q4(n, k, 100 + c) for c in range(copies)]
        f32 = (n, k) == (4096, 1024)
        for m in (1, 4, 8):
            x = _x(m, k, m)
            out = torch.empty((m, 9, n) if f32 else (m, n), dtype=torch.float32 if f32 else torch.bfloat16,
                              device=DEV)
            dst = out[:, 8] if f32 else out
            row = []
            for name, spec in _variants():
                with dv2.using(spec):
                    if f32 and spec == "":
                        dst_v = torch.empty((m, n), dtype=torch.float32, device=DEV)
                        us = _time(lambda i: (qmm.matmul(x, qs[i % copies], f32=True, out=dst_v, part=part),
                                              out[:, 8].copy_(dst_v)), 2 * copies)
                    else:
                        us = _time(lambda i: qmm.matmul(x, qs[i % copies], f32=f32, out=dst, part=part), 2 * copies)
                row.append(f"{name} {us:7.1f} ({one / us / 1e3:5.0f} GB/s)")
            print(f"  {n:>6}x{k:<5} m={m}: " + "  ".join(row))


def test_timing_route_hc_exl3():
    print("\n[0130] router (288 experts, top 8), us per call")
    for m in (1, 4, 8):
        x, w, bias = _route_inputs(m, seed=m)
        row = []
        slots = 9
        maxu = min(m * 8, 288) + 1
        mlog = torch.zeros((m, 288), device=DEV)
        pick = torch.zeros((m, slots), dtype=torch.int32, device=DEV)
        wts = torch.zeros((m, slots), device=DEV)
        grp = qmm.Group(torch.zeros((maxu,), dtype=torch.int32, device=DEV),
                        torch.zeros((1,), dtype=torch.int32, device=DEV),
                        torch.full((maxu, m), -1, dtype=torch.int32, device=DEV))
        for name, spec in _variants():
            with dv2.using(spec):
                def once(i):
                    if not dv2.route(x, w, bias, mlog, pick, wts, grp, 8, 288, 2.5, True):
                        glue.router(x, w, mlog)
                        glue.select(mlog, bias, pick, wts, grp.ids, grp.count, grp.members, 8, 288, 2.5, True)
                row.append(f"{name} {_time(once, 16):7.1f}")
        print(f"  m={m}: " + "  ".join(row))

    print("[0130] hyper-connection mix (hc_pre, D 4096, 4 streams), us per call")
    D, S = 4096, 4
    fns = [(torch.randn((24, S * D), device=DEV) * 0.01).to(torch.bfloat16) for _ in range(64)]
    base = torch.zeros((24,), device=DEV)
    scale = torch.ones((3,), device=DEV)
    nw = torch.ones((D,), dtype=torch.bfloat16, device=DEV)
    for m in (1, 4, 8):
        x = (torch.randn((m, S * D), device=DEV) * 0.3).to(torch.bfloat16)
        out = torch.zeros((m, D), dtype=torch.bfloat16, device=DEV)
        xs = torch.zeros((m, D // 64), device=DEV)
        post = torch.zeros((m, S), device=DEV)
        comb = torch.zeros((m, 16), device=DEV)
        part = torch.zeros((m, glue.HC_BLOCKS, 32), device=DEV)
        row = []
        for name, spec in _variants():
            with dv2.using(spec):
                us = _time(lambda i: glue.hc_pre(x, fns[i % 64], base, scale, nw, out, xs, post, comb, part, 1e-5, 1e-6,
                                                 20), 64)
            row.append(f"{name} {us:7.1f}")
        print(f"  m={m}: " + "  ".join(row))

    print("[0130] EXL3 routed experts (288 experts, top 8, D 4096, NI 1024 a rank), us per call")
    E, D, NI, top_k = 288, 4096, 1024, 8
    ex = _exl3(E, D, NI, 9)
    s = exl3_mm.Scratch(16, top_k + 1, D, NI, DEV)
    y = torch.zeros((16 * (top_k + 1), D), device=DEV)
    for m in (1, 2, 4, 8):
        sets = [_exl3_route(m, E, top_k, 1000 * m + j) for j in range(8)]
        xs_ = [_x(m, D, j) for j in range(8)]
        per = [int(g.count.item()) * 3 * D * NI // 2 for _, g in sets]           # trellis bytes read a call
        gb = statistics.mean(per)
        row = []
        for name, spec in _variants():
            with dv2.using(spec):
                us = _time(lambda i: exl3_mm.routed(xs_[i % 8], sets[i % 8][0], sets[i % 8][1], ex, s, y, m, 10.0), 16)
            row.append(f"{name} {us:7.1f} ({gb / us / 1e3:5.0f} GB/s)")
        print(f"  m={m}: " + "  ".join(row))


def test_timing_synthetic_windows(engines):
    """Whole captured windows of the synthetic model (tiny weights: launch-bound, so this shows the launches)."""

    kind, engs = engines
    for R in (1, 4, 6):
        row = []
        for name, spec in _variants():
            eng = engs[spec]
            e = eng.e
            with dv2.using(spec):
                e.reset()
                e.forward(list(range(1, 12)))
                commit(e.w, e.st, e.buf, 11, 11)
                g = e.graphs.main[(R, e.st.parity)]
                stage(e.w, e.st, e.buf, list(range(30, 30 + R)))
                for _ in range(5):
                    g.replay()
                torch.cuda.synchronize()
                ts = []
                for _ in range(7):
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    for _ in range(20):
                        g.replay()
                    b.record()
                    b.synchronize()
                    ts.append(a.elapsed_time(b) * 1000 / 20)
                e.reset()
                eng.cache.clear()
            row.append(f"{name} {statistics.median(ts):7.1f}")
        print(f"\n[0130] {kind} synthetic {R}-row window (graph replay), us: " + "  ".join(row))
