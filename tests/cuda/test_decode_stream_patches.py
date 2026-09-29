"""patches/0440 (GLM53_TF_DEC_EXPERTS / GLM53_TF_DEC_QMM / GLM53_TF_DEC_PDL): the streaming decode kernels on one GPU.

Nothing may change a bit. Checked here:

- E1 (``exl3_stream.cu``): ``exl3_mm.routed`` with the knob on == off (exl3.cu's grouped kernels + epilogues), Xd and
  Y bit for bit, at the real per-rank shapes (4096 / 1024, 288 experts, top 8) and a synthetic one: windows of
  1-16 rows, 4-slot batched mixes (11-44 rows from four independently routed sequences), skewed windows whose hot
  experts have 17-64 members; every (column tiles, stages) of gate/up and down, PDL on / off, CTA caps; eagerly and in
  CUDA graphs; every row of a window == the row alone; the shared slot untouched; the tickets left at zero;
  repeatable run to run.
- E2 (``q4_stream.cu``): ``qmm.matmul`` on == off, bit for bit, every per-rank shape, 1-16 / 17 / 24 / 32 / 44 / 64
  rows, bf16 and fp32 outputs, strided output rows, every (groups a stage, stages), (tile, slice) and whole-tile
  items, PDL on / off, CTA caps; eagerly
  and in graphs; rows alone == rows in the window; the tickets left at zero. Skipped (with the reason) where the
  Triton is not 3.7.x (``decode_stream.qmm_reference_fused``).
- PDL races (the sfxnz kit gates PDL off on sm_12x citing races with state kernels): each PDL-launched kernel runs
  right after a "late writer" that releases its dependents at once (griddepcontrol.launch_dependents), sleeps ~0.3 ms
  and only then writes the consumer's inputs (Xg / Xu for gate/up, Xd for down, x and its group sums for the dense
  kernel), with poisoned values in them until then, while a side stream keeps DRAM busy; the outputs must equal the
  non-PDL outputs. A consumer that read an input before griddepcontrol.wait would see the poison.
- Engines (TensorFold's synthetic checkpoints, MLX and EXL3 + q4mse; one GPU as rank 0 of two): windows of 1-16
  rows equal knob-off windows (graphs and eager), serial replies and every drafting policy equal serial decoding,
  sampled and greedy; a state left by drafted rounds resumes like a fresh prefill.

``-s`` prints kernel timings; tests/cuda/bench_decode_kernels.py is the full microbench.

Run inside the image:
    PYTHONPATH=/src/TensorFold/tests/cuda pytest -q -s tests/cuda/test_decode_stream_patches.py
The whole suite on the new kernels (the knobs are read at import):
    GLM53_TF_DEC_EXPERTS=1 GLM53_TF_DEC_QMM=1 PYTHONPATH=/src/TensorFold/tests/cuda pytest -q tests/cuda/
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.glm5_next.cuda import decode_stream as ds, exl3_mm, qmm, weights  # noqa: E402
from tensorfold.families.glm5_next.cuda.forward import chunks_for, commit, compute, stage  # noqa: E402
from test_glm_engine import _TwoCopies, _checkpoint, _drafter, _generate  # noqa: E402

DEV = "cuda"
PDLS = [False] + ([True] if ds.pdl_supported() else [])
SAMPLINGS = [Sampling(1234, 1.0, 20, 0.95), None]
IDS = ["sampled", "greedy"]
POLICIES = (None, "auto", "auto:1:1:0", "auto:1:2:0", "f7", "fc7:0.3", "fc5:0.3", "7", "c3:0.35", "a:0.6:0.85", "2")
EXPERT_CFGS = [(4, 4, 4, 4), (2, 6, 8, 3), (4, 6, 2, 8), (8, 3, 4, 4), (2, 4, 2, 4)]
SHAPES = [(12576, 4096), (4096, 4096), (4096, 128), (2048, 4096), (8192, 1536), (8192, 512), (4096, 8192),
          (12288, 4096), (4096, 6144), (4096, 1024), (77440, 4096)]
QMM_ROWS = list(range(1, 17)) + [17, 24, 32, 44, 64]


@pytest.fixture(autouse=True)
def _no_lookup(monkeypatch):
    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")


# -- E1 ----------------------------------------------------------------------------------------------------------------
def _exl3(E: int, D: int, NI: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)

    def trellis(k, n):
        return torch.randint(-2**15, 2**15, (E, k // 16, n // 16, 64), dtype=torch.int16, device=DEV, generator=g)

    def hs(n, sc):
        return (torch.randn((E, n), device=DEV, generator=g) * sc).to(torch.float16)

    w = lambda t: t.view(torch.int32)                                     # noqa: E731
    ex = exl3_mm.Exl3Experts(w(trellis(D, NI)), w(trellis(D, NI)), w(trellis(NI, D)), hs(D, 0.02), hs(D, 0.02),
                             hs(NI, 0.5), hs(NI, 0.5), hs(NI, 0.05), hs(D, 0.2), E, NI, D)
    return ex


def _group(pick: torch.Tensor, E: int, top_k: int):
    """glue.select's grouping of picks [rows, slots] (shared expert E last): ids ascending with E, members first."""

    R = pick.shape[0]
    p = pick.cpu()
    used = sorted({int(e) for e in p[:, :top_k].flatten()})
    maxu = min(R * top_k, E) + 1
    ids = torch.zeros((maxu,), dtype=torch.int32)
    members = torch.full((maxu, R), -1, dtype=torch.int32)
    for u, e in enumerate(used + [E]):
        ids[u] = e
        j = 0
        for r in range(R):
            for s in range(top_k + 1):
                if int(p[r, s]) == e:
                    members[u, j] = r * 32 + s
                    j += 1
    return qmm.Group(ids.to(DEV), torch.tensor([len(used) + 1], dtype=torch.int32, device=DEV), members.to(DEV))


def _picks(rows: int, E: int, top_k: int, seed: int, weights=None) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    pick = torch.full((rows, top_k + 1), E, dtype=torch.int32)
    w = torch.ones(E) if weights is None else weights
    for r in range(rows):
        pick[r, :top_k] = torch.multinomial(w, top_k, replacement=False, generator=g).to(torch.int32)
    return pick


def _mix(sizes, E, top_k, seed):
    parts = []
    for i, n in enumerate(sizes):
        g = torch.Generator().manual_seed(seed + i)
        w = torch.full((E,), 0.2)
        w[torch.randperm(E, generator=g)[:24]] = 3.0
        parts.append(_picks(n, E, top_k, seed + 100 + i, w))
    return torch.cat(parts)


def _run_routed(ex, x, pick, grp, s, R, top_k, *, stream, cfg=None, pdl=False, ctas=0, graph=False):
    slots = top_k + 1
    y = torch.full((s.rows * slots, ex.dims), 3.0, dtype=torch.float32, device=DEV)

    def call():
        if stream:
            assert ds.run_experts(x, pick, grp, ex, s, y, R, 10.0, cfg=cfg, pdl=pdl, ctas=ctas)
        else:
            with ds.using(experts=False):                              # exl3.cu, whatever the environment says
                exl3_mm.routed(x, pick, grp, ex, s, y, R, 10.0)

    s.xd.zero_()
    call()
    if graph:
        torch.cuda.synchronize()
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg):
            call()
        y.fill_(3.0)
        s.xd.zero_()
        cg.replay()
    torch.cuda.synchronize()
    return y.view(s.rows, slots, ex.dims)[:R].clone(), s.xd.view(s.rows, slots, ex.width)[:R].clone()


def _windows(E, top_k):
    out = [(f"{m} rows", _picks(m, E, top_k, m)) for m in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16)]
    out += [("4 slots 3+3+3+2", _mix((3, 3, 3, 2), E, top_k, 7)), ("4 slots 16+12+9+7", _mix((16, 12, 9, 7), E, top_k, 8)),
            ("skewed 24", _picks(24, E, top_k, 9, 1.0 / torch.arange(1, E + 1).float() ** 1.5)),
            ("skewed 64", _picks(64, E, top_k, 10, 1.0 / torch.arange(1, E + 1).float() ** 1.5))]
    return out


@pytest.mark.parametrize("dims", [(4096, 1024, 288, 8), (512, 128, 16, 4)], ids=["real", "synthetic"])
def test_experts_equal_exl3(dims):
    D, NI, E, top_k = dims
    ex = _exl3(E, D, NI, 3)
    s = exl3_mm.Scratch(64, top_k + 1, D, NI, DEV)
    for name, pick in _windows(E, top_k):
        R = pick.shape[0]
        grp = _group(pick, E, top_k)
        assert ds.experts_ok(grp, ex, s), name
        x = (torch.randn((R, D), generator=torch.Generator().manual_seed(R)) * 0.5).to(torch.bfloat16).to(DEV)
        pk = pick.to(DEV)
        ref_y, ref_xd = _run_routed(ex, x, pk, grp, s, R, top_k, stream=False)
        assert torch.isfinite(ref_y[:, :top_k]).all()
        for cfg in EXPERT_CFGS:
            for pdl in PDLS:
                for graph in (False, True):
                    y, xd = _run_routed(ex, x, pk, grp, s, R, top_k, stream=True, cfg=cfg, pdl=pdl, graph=graph)
                    assert torch.equal(y[:, :top_k], ref_y[:, :top_k]), (name, cfg, pdl, graph)
                    assert torch.equal(xd[:, :top_k], ref_xd[:, :top_k]), (name, cfg, pdl, graph)
                    assert (y[:, top_k] == 3.0).all()                  # the shared slot is left alone
        for ctas in (1, 7, 40):
            y, _ = _run_routed(ex, x, pk, grp, s, R, top_k, stream=True, ctas=ctas)
            assert torch.equal(y[:, :top_k], ref_y[:, :top_k]), (name, ctas)
        cnt = getattr(s, "_dec_cnt", None)
        assert cnt is not None and not cnt.any()                       # the tickets are left at zero
        if R <= 8:
            for r in range(R):
                g1 = _group(pick[r:r + 1], E, top_k)
                alone, _ = _run_routed(ex, x[r:r + 1], pk[r:r + 1], g1, s, 1, top_k, stream=True)
                assert torch.equal(alone[0, :top_k], ref_y[r, :top_k]), (name, r)


def test_experts_repeatable_and_probes():
    D, NI, E, top_k = 4096, 1024, 288, 8
    ex = _exl3(E, D, NI, 4)
    s = exl3_mm.Scratch(16, top_k + 1, D, NI, DEV)
    pick = _picks(16, E, top_k, 5).to(DEV)
    grp = _group(pick.cpu(), E, top_k)
    x = torch.randn((16, D), device=DEV).to(torch.bfloat16)
    a, _ = _run_routed(ex, x, pick, grp, s, 16, top_k, stream=True)
    for _ in range(5):
        b, _ = _run_routed(ex, x, pick, grp, s, 16, top_k, stream=True)
        assert torch.equal(a, b)
    y = torch.zeros((16 * (top_k + 1), D), device=DEV)
    for probe in (1, 2):                                               # the timing probes launch (wrong results)
        ds.run_experts(x, pick, grp, ex, s, y, 16, 10.0, cfg=(4, 4, 4, 4), probe=probe)
    torch.cuda.synchronize()
    assert not s._dec_cnt.any()


# -- E2 ----------------------------------------------------------------------------------------------------------------
def _q4(n: int, k: int, seed: int) -> qmm.Q4:
    g = torch.Generator(device=DEV).manual_seed(seed)
    words = torch.randint(-2**31, 2**31 - 1, (n, k // 8), dtype=torch.int64, device=DEV, generator=g).to(torch.int32)
    scales = (torch.rand((n, k // 64), device=DEV, generator=g) * 0.01 + 0.005).to(torch.bfloat16)
    biases = (-7.5 * scales.float()).to(torch.bfloat16)
    return qmm.make_q4(words, scales, biases)


def _need_fused():
    if not ds.qmm_reference_fused():
        pytest.skip("this Triton leaves some of _qmm's epilogues unfused: the streaming dense kernel stays off")


@pytest.mark.parametrize("n,k", SHAPES)
def test_qmm_equals_v1(n, k):
    _need_fused()
    q = _q4(n, k, n + k)
    part = torch.empty((8 * 64 * max(n, 16384) * 2,), dtype=torch.float32, device=DEV)
    for m in QMM_ROWS:
        x = torch.randn((m, k), device=DEV, generator=torch.Generator(device=DEV).manual_seed(m)).to(torch.bfloat16)
        xs = qmm.group_sums(x)
        for f32 in (False, True):
            with ds.using(qmm_on=False):
                ref = qmm.matmul(x, q, xs, f32=f32, part=part).clone()
            cfgs = ds.QMM_CFGS if m <= 16 else ((1, 4),)
            for cfg in cfgs:
                for pdl in PDLS:
                    for serial in (False, True):              # (tile, slice) items + tickets / whole-tile items
                        got = ds.run_qmm(x, q, xs, f32=f32, part=part, cfg=cfg, pdl=pdl, serial=serial)
                        torch.cuda.synchronize()
                        assert torch.equal(got, ref), (n, k, m, f32, cfg, pdl, serial)
        # strided output rows (a view into a wider buffer), a CTA cap, a graph
        wide = torch.zeros((m, q.n + 64), dtype=torch.bfloat16, device=DEV)
        ds.run_qmm(x, q, xs, out=wide[:, :q.n], part=part, ctas=5)
        with ds.using(qmm_on=False):
            ref = qmm.matmul(x, q, xs, part=part)
        assert torch.equal(wide[:, :q.n], ref), (n, k, m, "strided")
        out = torch.empty((m, q.n), dtype=torch.bfloat16, device=DEV)
        cg = torch.cuda.CUDAGraph()
        ds.run_qmm(x, q, xs, out=out, part=part)
        torch.cuda.synchronize()
        with torch.cuda.graph(cg):
            ds.run_qmm(x, q, xs, out=out, part=part)
        out.zero_()
        cg.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ref), (n, k, m, "graph")
        if m in (5, 16, 44):
            for r in (0, m - 1):
                one = ds.run_qmm(x[r:r + 1], q, xs[r:r + 1], part=part)
                assert torch.equal(one[0], ref[r]), (n, k, m, r)
    assert not ds.counters(part).any()


def test_qmm_hook_dispatch():
    """With the knob on, qmm.matmul takes the stream kernel for 4-bit weights of <= 64 rows with a partial buffer (the
    forward's calls) and falls back otherwise; the results are the same either way."""

    _need_fused()
    q = _q4(12576, 4096, 1)
    part = torch.empty((4 * 64 * 12576,), dtype=torch.float32, device=DEV)
    x = torch.randn((3, 4096), device=DEV).to(torch.bfloat16)
    with ds.using(qmm_on=False):
        ref = qmm.matmul(x, q, part=part)
    with ds.using(qmm_on=True):
        assert qmm.STREAM is ds
        assert torch.equal(qmm.matmul(x, q, part=part), ref)
        assert torch.equal(qmm.matmul(x, q), ref)                    # no partial buffer: qmm's kernels
    assert qmm.STREAM is None


# -- PDL races ------------------------------------------------------------------------------------------------------------
_LATE_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void late_copy(const uint4* __restrict__ a, uint4* __restrict__ da, size_t na,
                          const uint4* __restrict__ b, uint4* __restrict__ db, size_t nb, long long wait_ns) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");     // release the dependents at once
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        __nanosleep(1000);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < (unsigned long long)wait_ns);
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < na; i += (size_t)gridDim.x * blockDim.x) da[i] = a[i];
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nb; i += (size_t)gridDim.x * blockDim.x) db[i] = b[i];
}
void late(torch::Tensor a, torch::Tensor da, torch::Tensor b, torch::Tensor db, int64_t wait_ns) {
    late_copy<<<8, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint4*>(a.data_ptr()), reinterpret_cast<uint4*>(da.data_ptr()), (size_t)a.nbytes() / 16,
        reinterpret_cast<const uint4*>(b.data_ptr()), reinterpret_cast<uint4*>(db.data_ptr()), (size_t)b.nbytes() / 16,
        (long long)wait_ns);
}
"""
_late_mod = None


def _late():
    global _late_mod
    if _late_mod is None:
        from torch.utils.cpp_extension import load_inline

        decl = ("#include <torch/extension.h>\nvoid late(torch::Tensor a, torch::Tensor da, torch::Tensor b, "
                "torch::Tensor db, int64_t wait_ns);\n")      # W12: the generated binding needs the declaration
        _late_mod = load_inline("tf_0440_late_writer", cpp_sources=decl, cuda_sources=_LATE_SRC, functions=["late"],
                                extra_cuda_cflags=["-O3", f"-arch=sm_{''.join(map(str, torch.cuda.get_device_capability()))}"],
                                verbose=False)
    return _late_mod


def _busy_side():
    side = torch.cuda.Stream()
    a = torch.empty(64 << 20, dtype=torch.uint8, device=DEV)
    b = torch.empty_like(a)

    def start(n=16):
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for i in range(n):
                (b if i % 2 else a).copy_(a if i % 2 else b)
    return start, side


def _poison(t: torch.Tensor) -> None:
    t.view(torch.int16).fill_(0x7E00 if t.dtype == torch.float16 else 0x7FC0)          # NaN in fp16 / bf16


@pytest.mark.skipif(not ds.pdl_supported(), reason="programmatic dependent launch needs sm_90+")
def test_pdl_consumers_wait_for_their_inputs():
    D, NI, E, top_k = 4096, 1024, 288, 8
    ex = _exl3(E, D, NI, 21)
    slots = top_k + 1
    s = exl3_mm.Scratch(16, slots, D, NI, DEV)
    start, side = _busy_side()
    gsk, dsk = exl3_mm.GATEUP_CFG[2], exl3_mm.DOWN_CFG[2]
    ext, base = ds._experts_ext(), exl3_mm._ext()
    P = s.rows * slots
    for R in (1, 5, 16):
        pick = _picks(R, E, top_k, 50 + R).to(DEV)
        grp = _group(pick.cpu(), E, top_k)
        x = torch.randn((R, D), device=DEV).to(torch.bfloat16)
        y_ref, xd_ref = _run_routed(ex, x, pick, grp, s, R, top_k, stream=True, pdl=False)
        cnt = s._dec_cnt
        maxu = grp.ids.shape[0]
        ng = maxu * (NI // 128)
        for cfg in EXPERT_CFGS[:3]:
            base.rot_in(x, x.stride(0), pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, slots)
            good_g, good_u = s.xg.clone(), s.xu.clone()
            for _ in range(3):
                _poison(s.xg), _poison(s.xu), _poison(s.xd)
                y = torch.full((P, D), 3.0, device=DEV)
                start()
                _late().late(good_g, s.xg, good_u, s.xu, 300_000)       # gate/up's inputs land ~0.3 ms late
                ext.gateup(s.xg, s.xu, ex.gt, ex.ut, grp.ids, grp.count, grp.members, s.z, ex.svh_g, ex.svh_u,
                           ex.suh_d, s.xd, cnt[:ng], D, NI, P, gsk, slots, cfg[0], cfg[1], 0, 10.0, True, 0)
                good_d = s.xd.clone()
                _poison(s.xd)
                _late().late(good_d, s.xd, good_d[:1], torch.empty_like(good_d[:1]), 300_000)
                ext.down(s.xd, ex.dt, grp.ids, grp.count, grp.members, s.z, ex.svh_d, y, cnt[ng:ng + maxu * (D // 128)],
                         NI, D, P, dsk, slots, cfg[2], cfg[3], 0, True, 0)
                torch.cuda.synchronize()
                side.synchronize()
                got = y.view(s.rows, slots, D)[:R]
                assert torch.equal(got[:, :top_k], y_ref[:, :top_k]), (R, cfg)
                assert torch.equal(good_d.view(s.rows, slots, NI)[:R, :top_k], xd_ref[:, :top_k]), (R, cfg)
        assert not cnt.any()
    if not ds.qmm_reference_fused():
        return
    part = torch.empty((8 * 16 * 12576,), dtype=torch.float32, device=DEV)
    for n, k in ((12576, 4096), (4096, 1024), (8192, 512), (77440, 4096)):
        q = _q4(n, k, 7)
        for m in (1, 8, 16):
            x = torch.randn((m, k), device=DEV).to(torch.bfloat16)
            xs = qmm.group_sums(x)
            with ds.using(qmm_on=False):
                ref = qmm.matmul(x, q, xs, part=part).clone()
            xp, xsp = x.clone(), xs.clone()
            for serial in (False, True):
                _poison(xp)
                xsp.fill_(float("nan"))
                start()
                _late().late(x, xp, xs, xsp, 300_000)
                got = ds.run_qmm(xp, q, xsp, part=part, pdl=True, serial=serial)
                torch.cuda.synchronize()
                side.synchronize()
                assert torch.equal(got, ref), (n, k, m, serial)
    assert not ds.counters(part).any()


# -- engines -------------------------------------------------------------------------------------------------------------
SPECS = {"off": dict(experts=False, qmm_on=False), "experts": dict(experts=True, qmm_on=False),
         "both": dict(experts=True, qmm_on=True), "both,pdl": dict(experts=True, qmm_on=True, pdl=True)}


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    out = {}
    for kind in ("mlx", "exl3"):
        path = tmp_path_factory.mktemp(f"glm_stream_{kind}")
        _checkpoint(path / "model", exl3=kind == "exl3")
        _drafter(path / "dflash2")
        out[kind] = path
    return out


def _spec_names():
    names = ["off", "experts"]
    if ds.qmm_reference_fused():
        names.append("both")
        if ds.pdl_supported():
            names.append("both,pdl")
    return names


def _engine(path, name: str):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    mp = pytest.MonkeyPatch()
    mp.setattr(weights, "NONEXPERT", "q4mse")
    mp.setenv("GLM53_TF_NONEXPERT", "q4mse")
    try:
        with ds.using(**SPECS[name]):                     # the graphs capture this setting's kernels
            eng = GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", comm=_TwoCopies())
    finally:
        mp.undo()
    eng._stream = name
    return eng


@pytest.fixture(scope="module", params=["mlx", "exl3"])
def engines(request, ckpts):
    path = ckpts[request.param]
    return request.param, {name: _engine(path, name) for name in _spec_names()}


def _window(eng, prompt, rows, *, eager=False):
    e = eng.e
    with ds.using(**SPECS[eng._stream]):
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


def test_windows_equal_off(engines):
    kind, engs = engines
    rng = np.random.default_rng(41)
    prompt = [int(t) for t in rng.integers(0, 1000, size=19)]
    for R in list(range(1, 9)) + [12, 16]:
        rows = [int(t) for t in rng.integers(0, 1000, size=R)]
        ref = _window(engs["off"], prompt, rows)
        for name, eng in engs.items():
            if name == "off":
                continue
            got = _window(eng, prompt, rows)
            for a, b in zip(ref, got):
                assert torch.equal(a, b), (kind, name, R)
            if R <= 6:
                eag = _window(eng, prompt, rows, eager=True)
                for a, b in zip(got, eag):
                    assert torch.equal(a, b), (kind, name, R, "eager")


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
def test_replies_equal_serial(engines, sampling):
    kind, engs = engines
    prompt = list(np.random.default_rng(42).integers(0, 1000, size=43))
    with ds.using(**SPECS["off"]):
        serial, _ = _generate(engs["off"], prompt, sampling, draft=False, tokens=32)
    for name, eng in engs.items():
        with ds.using(**SPECS[name]):
            got, _ = _generate(eng, prompt, sampling, draft=False, tokens=32)
            assert got == serial, (kind, name, "serial")
            for policy in POLICIES:
                drafted, _ = _generate(eng, prompt, sampling, policy=policy, tokens=32)
                assert drafted == serial, (kind, name, policy)


def test_resume_equals_fresh_prefill(engines):
    kind, engs = engines
    sampling = Sampling(41, 1.0, 20, 0.95)
    rng = np.random.default_rng(43)
    first = list(rng.integers(0, 1000, size=70))
    unrelated = list(rng.integers(0, 1000, size=9))
    results = []
    for name, eng in engs.items():
        with ds.using(**SPECS[name]):
            reply, _ = _generate(eng, first, sampling, policy="auto:1:1:0", tokens=20)
            after = first + reply + [7, 8]
            warm, stats = _generate(eng, after, sampling, tokens=20)
            assert stats["cached"] >= len(first) + len(reply) - 1
            _generate(eng, unrelated, sampling)
            cold, stats = _generate(eng, after, sampling, tokens=20)
            assert stats["cached"] == 0 and warm == cold, (kind, name)
        results.append((reply, warm))
    assert all(r == results[0] for r in results), kind


# -- timings (-s) ----------------------------------------------------------------------------------------------------------
def test_timing_summary():
    """A short version of bench_decode_kernels.py: one MoE layer at 1 / 8 rows and the KDA projection at 1 row."""

    import bench_decode_kernels as bench

    for line in bench.summary():
        print(line)
