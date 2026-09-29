"""patches/0520 (GLM53_TF_HC_CUDA=1) on the GPU: ``hc_fused.cu`` gives exactly the bits of the Triton kernels it
replaces -- ``glue.hc_post(x, x, g, post, comb)`` then ``glue.hc_pre(x, fn, base, scale, norm_w, out, xs, post,
comb, part, eps, hc_eps, iters)`` -- on every output: the new streams x (in place), the normed row, its 64-group sums,
post, comb and the partial buffer (slots 0-24; 25-31 are never written by either).

- 1-16 rows (decode / verify / MTP windows), 24 / 32 / 48 / 64 rows (batched rounds, GLM53_TF_HC_CUDA_ROWS raised),
  every row-group size (1, 2, 4, 8: placement only), seeds, input kinds: random, real-scale (streams with a few
  large channels, fn ~ N(0, 0.03), Sinkhorn-like comb), extreme (logits around the unit sequences' thresholds, huge
  streams and partials), signed zeros, subnormals, and the adversarial collapse (tests/hc_fused_emu.py);
- the gathered partials as a row slice of a larger block (rank stride > R D), world 1 and 2;
- inside a CUDA graph (replayed three times: the tickets reset themselves), and a replay on changed inputs;
- rows are independent of their window (a row alone == the same row in a 9-row launch);
- ``hc_cuda.self_check`` (what the engine runs at load) passes;
- 64 rows at one row a group (2,112 CTAs: finishers waiting while step CTAs run in waves) completes;
- GLM53_TF_HC_CUDA_PDL: the same bits, and a late writer (a kernel that releases its dependents at once, then writes
  the gathered partials after ~0.2 ms; NaN before) does not leak NaN into the outputs (the kernel waits).

A False anywhere: keep GLM53_TF_HC_CUDA unset. Print the Triton / ptxas versions with -s: the kernel was derived from
Triton 3.7.1 with its bundled ptxas-blackwell 13.1; another ptxas may compile the Triton kernels differently.

Run inside the image:
    PYTHONPATH=/src/TensorFold/src:/src/TensorFold/tests/cuda:/work/tests/cuda timeout 1200 \\
        pytest -q -s tests/cuda/test_hc_fused_patches.py          (~2-3 min, the first run compiles the extension)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

try:
    from tensorfold.families.glm5_next.cuda import glue, hc_cuda  # noqa: E402
except ImportError:
    pytest.skip("needs the patched tree on PYTHONPATH", allow_module_level=True)

for p in (Path(__file__).parent.parent, Path("/work/tests"), Path("/src/TensorFold/tests")):
    if (p / "hc_fused_emu.py").exists():
        sys.path.insert(0, str(p))
        break
import hc_fused_emu as E  # noqa: E402

DEV = "cuda"
BF, F32 = torch.bfloat16, torch.float32
D, S, WIDE = 4096, 4, 16384


def _t(a, dtype):
    if dtype == BF:
        import numpy as np

        return torch.from_numpy(np.ascontiguousarray(a).view(np.int16)).view(BF).to(DEV)
    return torch.from_numpy(a.copy()).to(dtype).to(DEV)


def _inputs(R, seed, kind, world=2, rs_pad=0):
    inp = E.make_inputs(R, seed=seed, kind=kind, world=world)
    x = _t(inp["x"], BF)
    g_full = torch.zeros((world, R + rs_pad, D), dtype=F32, device=DEV)
    g_full[:, :R] = _t(inp["g"], F32)
    g = g_full[:, :R]
    return dict(x=x, g=g, post=_t(inp["post"], F32), comb=_t(inp["comb"], F32), fn=_t(inp["fn"], BF),
                base=_t(inp["base"], F32), scale=_t(inp["scale"], F32), nw=_t(inp["nw"], BF),
                eps=float(inp["eps"]), hc_eps=float(inp["hc_eps"]), iters=int(inp["iters"]))


def _outs(R):
    return dict(out=torch.zeros((R, D), dtype=BF, device=DEV), xs=torch.zeros((R, D // 64), dtype=F32, device=DEV),
                part=torch.full((R, 16, 32), float("nan"), dtype=F32, device=DEV))


def _run(inp, fused: bool, rg: int = 4, pdl: bool = False):
    R = inp["x"].shape[0]
    x, post, comb = inp["x"].clone(), inp["post"].clone(), inp["comb"].clone()
    o = _outs(R)
    h = type("H", (), {})()
    h.fn, h.base, h.scale = inp["fn"], inp["base"], inp["scale"]
    if fused:
        why = hc_cuda.fits(x, inp["g"], post, comb, h.fn, h.base, h.scale, inp["nw"], o["out"], o["xs"], o["part"])
        assert why is None, why
        hc_cuda.launch(x, inp["g"], post, comb, h.fn, h.base, h.scale, inp["nw"], o["out"], o["xs"], o["part"],
                       inp["eps"], inp["hc_eps"], inp["iters"], rg=rg, pdl=pdl)
    else:
        glue.hc_post(x, x, inp["g"], post, comb)
        glue.hc_pre(x, h.fn, h.base, h.scale, inp["nw"], o["out"], o["xs"], post, comb, o["part"], inp["eps"],
                    inp["hc_eps"], inp["iters"])
    torch.cuda.synchronize()
    return {"x": x, "normed": o["out"], "xs": o["xs"], "post": post, "comb": comb,
            "part": o["part"][:, :, :25].contiguous()}


def _same(a, b) -> bool:
    if a.dtype == BF:
        fa, fb = a.float(), b.float()
        na, nb = torch.isnan(fa), torch.isnan(fb)
        return bool(torch.equal(na, nb) and torch.equal(a.view(torch.int16)[~na], b.view(torch.int16)[~nb]))
    na, nb = torch.isnan(a), torch.isnan(b)
    return bool(torch.equal(na, nb) and torch.equal(a.view(torch.int32)[~na], b.view(torch.int32)[~nb]))


def _diff(a: dict, b: dict) -> list[str]:
    return [k for k in a if not _same(a[k], b[k])]


@pytest.fixture(autouse=True)
def _on():
    saved = (hc_cuda.ON, hc_cuda.ROWS)
    hc_cuda.configure(True, rows=64)
    yield
    hc_cuda.ON, hc_cuda.ROWS = saved


def test_versions():
    import triton

    try:
        from triton.backends.nvidia.compiler import get_ptxas

        cap = torch.cuda.get_device_capability()
        ptxas = get_ptxas(cap[0] * 10 + cap[1])
        pv = f"{ptxas.path} {ptxas.version}"
    except Exception as exc:                        # noqa: BLE001
        pv = f"? ({exc})"
    print(f"\n[hc_fused] {torch.cuda.get_device_name(0)} sm_{''.join(map(str, torch.cuda.get_device_capability()))}, "
          f"torch {torch.__version__}, triton {triton.__version__}, Triton's ptxas: {pv}, "
          f"TRITON_PTXAS_BLACKWELL_PATH={os.environ.get('TRITON_PTXAS_BLACKWELL_PATH')}")
    if not triton.__version__.startswith("3.7."):
        print("[hc_fused] WARNING: not the image's Triton 3.7.x: the Triton kernels' arithmetic differs (expect bits "
              "to differ; the engine's self-check would keep the knob off)")


def test_self_check():
    assert hc_cuda.self_check(torch.device(DEV)) is None


@pytest.mark.parametrize("R", list(range(1, 17)))
@pytest.mark.parametrize("kind", ["random", "real"])
def test_decode_rows(R, kind):
    inp = _inputs(R, seed=R, kind=kind)
    ref = _run(inp, False)
    assert _diff(_run(inp, True), ref) == []


@pytest.mark.parametrize("kind", ["extreme", "zeros", "tiny", "collapse", "nonfinite"])
@pytest.mark.parametrize("R", [1, 3, 8, 16])
def test_edge_inputs(R, kind):
    inp = _inputs(R, seed=100 + R, kind=kind)
    ref = _run(inp, False)
    assert _diff(_run(inp, True), ref) == []


@pytest.mark.parametrize("R", [24, 32, 48, 64])
def test_batched_rows(R):
    inp = _inputs(R, seed=7 * R, kind="real")
    ref = _run(inp, False)
    for rg in (4, 8):
        assert _diff(_run(inp, True, rg=rg), ref) == [], rg


@pytest.mark.parametrize("rg", [1, 2, 4, 8])
@pytest.mark.parametrize("R", [5, 13])
def test_row_groups_place_work_only(R, rg):
    inp = _inputs(R, seed=3, kind="random")
    assert _diff(_run(inp, True, rg=rg), _run(inp, False)) == []


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_seeds(seed):
    inp = _inputs(6, seed=seed, kind="random")
    assert _diff(_run(inp, True), _run(inp, False)) == []


def test_gathered_row_slice_and_world1():
    inp = _inputs(5, seed=9, kind="real", rs_pad=11)          # rank stride (R + 11) D: a row slice of a gathered block
    assert inp["g"].stride(0) == 16 * D
    assert _diff(_run(inp, True), _run(inp, False)) == []
    inp1 = _inputs(4, seed=10, kind="random", world=1)
    assert _diff(_run(inp1, True), _run(inp1, False)) == []


def test_rows_independent_of_window():
    inp = _inputs(9, seed=21, kind="real")
    whole = _run(inp, True)
    for r in (0, 4, 8):
        one = dict(inp)
        for k in ("x", "post", "comb"):
            one[k] = inp[k][r:r + 1].clone()
        one["g"] = inp["g"][:, r:r + 1].contiguous()
        alone = _run(one, True, rg=1)
        for k in whole:
            assert _same(alone[k], whole[k][r:r + 1]), (r, k)


def test_in_a_cuda_graph():
    R = 7
    inp = _inputs(R, seed=5, kind="real")
    ref = _run(inp, False)
    x, post, comb = inp["x"].clone(), inp["post"].clone(), inp["comb"].clone()
    o = _outs(R)
    args = (x, inp["g"], post, comb, inp["fn"], inp["base"], inp["scale"], inp["nw"], o["out"], o["xs"], o["part"],
            inp["eps"], inp["hc_eps"], inp["iters"])
    hc_cuda.launch(*args)                                     # eager warm-up (compiles, allocates the scratch)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        hc_cuda.launch(*args)
    for _ in range(3):
        x.copy_(inp["x"])
        post.copy_(inp["post"])
        comb.copy_(inp["comb"])
        o["part"].fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        got = {"x": x, "normed": o["out"], "xs": o["xs"], "post": post, "comb": comb,
               "part": o["part"][:, :, :25].contiguous()}
        assert _diff(got, ref) == []
    # the next boundary's streams, partials and mixes change between replays (the weights are the captured ones)
    inp2 = dict(_inputs(R, seed=6, kind="real"), fn=inp["fn"], base=inp["base"], scale=inp["scale"], nw=inp["nw"])
    ref2 = _run(inp2, False)
    x.copy_(inp2["x"])
    inp["g"].copy_(inp2["g"])
    post.copy_(inp2["post"])
    comb.copy_(inp2["comb"])
    g.replay()
    torch.cuda.synchronize()
    got = {"x": x, "normed": o["out"], "xs": o["xs"], "post": post, "comb": comb,
           "part": o["part"][:, :, :25].contiguous()}
    assert _diff(got, ref2) == []


def test_many_ctas_complete():
    """64 rows, one row a group: 64 x 32 step CTAs + 64 finishers (more than one wave; the finishers spin on their
    group's tickets while later step CTAs are still being scheduled)."""

    inp = _inputs(64, seed=2, kind="random")
    ref = _run(inp, False)
    assert _diff(_run(inp, True, rg=1), ref) == []


def test_off_launches_nothing():
    hc_cuda.configure(False)
    x, g, post, comb = torch.zeros((2, WIDE), dtype=BF, device=DEV), torch.zeros((2, 2, D), device=DEV), \
        torch.zeros((2, 4), device=DEV), torch.zeros((2, 16), device=DEV)
    h = type("H", (), {"fn": torch.zeros((24, WIDE), dtype=BF, device=DEV), "base": torch.zeros(24, device=DEV),
                       "scale": torch.zeros(3, device=DEV)})()
    o = _outs(2)
    assert hc_cuda.post_pre(x, g, post, comb, h, torch.zeros(D, dtype=BF, device=DEV), o["out"], o["xs"], o["part"],
                            1e-5, 1e-6, 20) is False
    assert torch.isnan(o["part"]).all()


@pytest.mark.skipif(not hc_cuda.pdl_supported(), reason="PDL needs sm_90+")
def test_pdl_same_bits():
    inp = _inputs(6, seed=4, kind="real")
    assert _diff(_run(inp, True, pdl=True), _run(inp, False)) == []


def _late_writer():
    triton = pytest.importorskip("triton")
    import triton.language as tl

    try:
        from triton.language.extra.cuda import gdc_launch_dependents
    except ImportError:
        pytest.skip("this Triton has no gdc_launch_dependents")

    @triton.jit
    def late(G, SRC, n, spin, BLOCK: tl.constexpr):
        gdc_launch_dependents()
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        acc = tl.zeros((BLOCK,), dtype=tl.float32) + 1.0
        for _ in range(spin):
            acc = acc * 1.0000001 + 1e-7
        v = tl.load(SRC + i, mask=i < n, other=0.0)
        tl.store(G + i, v + (acc - acc), mask=i < n)

    return late


@pytest.mark.skipif(not hc_cuda.pdl_supported(), reason="PDL needs sm_90+")
def test_pdl_waits_for_a_late_writer():
    late = _late_writer()
    R = 4
    inp = _inputs(R, seed=8, kind="real")
    ref = _run(inp, False)
    src = inp["g"].clone()
    g = inp["g"]
    x, post, comb = inp["x"].clone(), inp["post"].clone(), inp["comb"].clone()
    o = _outs(R)
    w = _outs(R)
    hc_cuda.launch(x.clone(), g, post.clone(), comb.clone(), inp["fn"], inp["base"], inp["scale"], inp["nw"],
                   w["out"], w["xs"], w["part"], inp["eps"], inp["hc_eps"], inp["iters"], pdl=True)
    torch.cuda.synchronize()                                  # built and warm
    n = src.numel()
    for spin in (20000, 200000):
        g.fill_(float("nan"))
        x.copy_(inp["x"])
        post.copy_(inp["post"])
        comb.copy_(inp["comb"])
        late[(triton_cdiv(n, 1024),)](g, src, n, spin, BLOCK=1024)
        hc_cuda.launch(x, g, post, comb, inp["fn"], inp["base"], inp["scale"], inp["nw"], o["out"], o["xs"],
                       o["part"], inp["eps"], inp["hc_eps"], inp["iters"], pdl=True)
        torch.cuda.synchronize()
        got = {"x": x, "normed": o["out"], "xs": o["xs"], "post": post, "comb": comb,
               "part": o["part"][:, :, :25].contiguous()}
        assert _diff(got, ref) == [], spin


def triton_cdiv(a, b):
    return -(-a // b)
