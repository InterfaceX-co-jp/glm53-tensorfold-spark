"""patches/0470 (GLM53_TF_NONEXPERT=q8 / GLM53_TF_NONEXPERT_MAP) on the GPU: the 8-bit kernels on the real per-rank
shapes of GLM-5.3-Flash.

- bits: ``fast_qmm.matmul_prefill(exact=True)`` == ``qmm.matmul`` (decode / verify / MTP) bit for bit at 64-2,048
  rows; a row's bits never depend on the call's row count (rows of a 64-row call == the same rows at 1, 7, 16, 17 and
  33 rows, every row bucket); ``matmul_fast`` (one accumulator) is row-independent too; every entry of the decode
  and prefill tile tables gives the same bits (GLM53_TF_Q8_DECODE_CFG / GLM53_TF_Q8_TILE are speed knobs only); a
  CUDA graph replay gives the eager bits;
- accuracy: every kernel within fp32-summation distance of a float64 reference of the dequantized weights, and the
  8-bit matmul within ~1% of the BF16 weights' (q4mse: ~9%);
- latent MLA absorb / expand with an 8-bit kv_b: v1 == v2 bit for bit;
- the quantizer on the GPU gives the CPU's bits (the prepared folders are written on the GPU).

``-k timing`` (``-s`` to see it): the 8-bit decode kernel's GB/s at 1 / 4 / 8 / 16 rows on every shape, beside the
4-bit and BF16 kernels, and the prefill kernel at 512 / 2,048 rows (TF/s), each timed over distinct weight copies
(nothing warm in L2) with CUDA-graph replays; plus a small tile sweep for Q8_CONFIG / Q8_TILE_LOOSE. Printed only.

Run inside the image: pytest -q tests/cuda/test_q8_patches.py   (timing: -k timing -s, ~3-5 min)
"""

from __future__ import annotations

import os

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.glm5_next.cuda import fast_qmm, latent, qmm  # noqa: E402

if not hasattr(qmm, "Q8"):
    pytest.skip("needs patches/0470", allow_module_level=True)

DEV = "cuda"
SHAPES = [(12576, 4096), (4096, 128), (4096, 4096), (2048, 4096), (8192, 1536), (8192, 512), (4096, 8192),
          (12288, 4096), (4096, 6144), (4096, 1024), (160, 4096), (4096, 1536), (77440, 4096)]
IDS = [f"{n}x{k}" for n, k in SHAPES]


def _w(n, k, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(n, k, generator=g) * 0.02
    w[:, : k // 8] *= 6
    return w.to(torch.bfloat16).to(DEV)


def _x(m, k, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(m, k, generator=g).to(torch.bfloat16).to(DEV)


def _bits(t):
    return t.contiguous().view(torch.int32 if t.dtype == torch.float32 else torch.int16)


@pytest.mark.parametrize("n,k", SHAPES, ids=IDS)
def test_quantizer_gpu_equals_cpu(n, k):
    w = _w(n, k)
    a, b = qmm.quantize8(w), qmm.quantize8(w.cpu())
    assert torch.equal(a.weight.cpu(), b.weight) and torch.equal(_bits(a.scales.cpu()), _bits(b.scales))


@pytest.mark.parametrize("n,k", SHAPES, ids=IDS)
def test_bits(n, k):
    q = qmm.quantize8(_w(n, k))
    x = _x(64, k)
    dec = qmm.matmul(x, q, f32=True)
    ref = x.double() @ qmm.dequantize_q8(q).double().T
    assert ((dec.double() - ref).abs() <= 2e-5 * ref.abs().max()).all()
    for m in (1, 7, 16, 17, 33):
        assert torch.equal(_bits(qmm.matmul(x[:m], q, f32=True)), _bits(dec[:m])), m
        assert torch.equal(_bits(qmm.matmul(x[m:m + 1], q, f32=True)), _bits(dec[m:m + 1])), m
    # the bf16 output is the fp32 sum rounded once
    assert torch.equal(_bits(qmm.matmul(x, q)), _bits(dec.to(torch.bfloat16)))
    # prefill, exact: qmm's bits at any row count and tile
    big = torch.cat([x, _x(2048 - 64, k, seed=5)])
    ex = fast_qmm.matmul_prefill(big, q, f32=True, min_rows=1)
    assert torch.equal(_bits(ex[:64]), _bits(dec))
    assert torch.equal(_bits(qmm.matmul(big[1000:1016], q, f32=True)), _bits(ex[1000:1016]))
    loose = fast_qmm.matmul_prefill(big, q, f32=True, exact=False, min_rows=1)
    assert torch.equal(_bits(fast_qmm.matmul_prefill(big[5:70], q, f32=True, exact=False, min_rows=1)),
                       _bits(loose[5:70]))
    ref2 = big.double() @ qmm.dequantize_q8(q).double().T
    assert ((loose.double() - ref2).abs() <= 2e-5 * ref2.abs().max()).all()


@pytest.mark.parametrize("n,k", [(12576, 4096), (4096, 8192), (8192, 512)], ids=["kda", "dsa_o", "kvb"])
def test_tables_are_speed_only(n, k, monkeypatch):
    q = qmm.quantize8(_w(n, k))
    x = _x(300, k)
    dec = qmm.matmul(x[:16], q, f32=True)
    for cfg in ("1,4,2", "2,4,3", "1,8,3,32", "4,4,2,64", "1,4,4,128"):
        monkeypatch.setenv("GLM53_TF_Q8_DECODE_CFG", cfg)
        assert torch.equal(_bits(qmm.matmul(x[:16], q, f32=True)), _bits(dec)), cfg
    monkeypatch.delenv("GLM53_TF_Q8_DECODE_CFG")
    loose = fast_qmm.matmul_prefill(x, q, f32=True, exact=False, min_rows=1)
    exact = fast_qmm.matmul_prefill(x, q, f32=True, min_rows=1)
    for tile in ("64,64,8,2", "128,32,8,3", "64,32,4,3", "32,64,4,3", "128,64,8,2"):
        monkeypatch.setenv("GLM53_TF_Q8_TILE", tile)
        assert torch.equal(_bits(fast_qmm.matmul_prefill(x, q, f32=True, exact=False, min_rows=1)), _bits(loose))
        assert torch.equal(_bits(fast_qmm.matmul_prefill(x, q, f32=True, min_rows=1)), _bits(exact))


def test_graph_replay():
    q = qmm.quantize8(_w(4096, 4096))
    x = _x(8, 4096)
    out = torch.empty((8, 4096), dtype=torch.bfloat16, device=DEV)
    part = torch.empty((qmm.q8_split_k(4096, 4096) * 8 * 4096,), dtype=torch.float32, device=DEV)
    qmm.matmul(x, q, out=out, part=part)
    want = out.clone()
    out.zero_()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        qmm.matmul(x, q, out=out, part=part)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(_bits(out), _bits(want))


def test_accuracy_vs_bf16_and_q4():
    w = _w(4096, 4096, seed=3)
    x = _x(64, 4096, seed=4)
    ref = (x.float() @ w.float().T)
    e8 = (qmm.matmul(x, qmm.quantize8(w), f32=True) - ref).norm() / ref.norm()
    e4 = (qmm.matmul(x, qmm.quantize4(w, mse=True), f32=True) - ref).norm() / ref.norm()
    print(f"relative output error: q8 {float(e8):.4%}, q4mse {float(e4):.4%}")
    assert e8 < 0.012 and e4 / e8 > 8


@pytest.mark.parametrize("R", [1, 8, 16, 64, 513])
def test_latent_q8_v1_equals_v2(R, monkeypatch):
    H, D, L = 32, 256, 512
    kv_k, kv_v = qmm.quantize8(_w(H * D, L, seed=11)), qmm.quantize8(_w(H * D, L, seed=12))
    q = _x(R * H, D, seed=R).view(R, H, D).contiguous()
    u = torch.randn(R, H, L, device=DEV)
    got = {}
    for v2 in (False, True):
        monkeypatch.setattr(latent, "EXPAND_V2", v2)
        a = torch.empty(R, H, L, dtype=torch.bfloat16, device=DEV)
        e = torch.empty(R, H * D, dtype=torch.bfloat16, device=DEV)
        got[v2] = (latent.absorb(q, kv_k, a).clone(), latent.expand(u, kv_v, e).clone())
    assert torch.equal(_bits(got[False][0]), _bits(got[True][0]))
    assert torch.equal(_bits(got[False][1]), _bits(got[True][1]))
    wk = qmm.dequantize_q8(kv_k).double().view(H, D, L)
    ref = torch.einsum("rhi,hij->rhj", q.double(), wk)
    assert ((got[True][0].double() - ref).abs() <= 0.01 * ref.abs().max()).all()


# -- timing (printed only) ------------------------------------------------------------------------------------------
def _time(fn, reps=20):
    fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1e3            # us


@pytest.mark.skipif(os.environ.get("Q8_TIMING", "1") == "0", reason="Q8_TIMING=0")
def test_timing_decode():
    """GB/s of the weight stream (bytes of the stored matrix / kernel time) for q8, q4 (today's kernel) and bf16, over
    8 distinct copies of every shape per graph (so L2 holds none of them)."""

    copies = 8
    print("\nshape        rows   q8 us  GB/s |  q4 us  GB/s | bf16 us  GB/s   (per matrix)")
    for n, k in SHAPES:
        w = _w(n, k)
        mats = {"q8": [qmm.quantize8(w) for _ in range(copies)], "q4": [qmm.quantize4(w, mse=True) for _ in range(copies)],
                "bf16": [qmm.make_b16(w.clone()) for _ in range(copies)]}
        for m in (1, 4, 8, 16):
            x = _x(m, k)
            xs = qmm.group_sums(x)
            row = []
            for name, qs in mats.items():
                sk = max(qmm.split_k(n, k), qmm.b16_split_k(n, k))
                part = torch.empty((sk * m * n,), dtype=torch.float32, device=DEV)
                out = torch.empty((m, n), dtype=torch.bfloat16, device=DEV)

                def run(qs=qs, out=out, part=part, x=x, xs=xs):
                    for q in qs:
                        qmm.matmul(x, q, xs, out=out, part=part)
                us = _time(run) / copies
                row.append((us, qs[0].nbytes() / us / 1e3))
            print(f"{n:>6}x{k:<5} {m:>4}  " + " | ".join(f"{us:7.1f} {gbs:5.0f}" for us, gbs in row), flush=True)


@pytest.mark.skipif(os.environ.get("Q8_TIMING", "1") == "0", reason="Q8_TIMING=0")
def test_timing_prefill_and_sweep(monkeypatch):
    print("\nprefill (fast chunks, one accumulator): shape rows  q8 ms TF/s | q4 ms TF/s")
    for n, k in [(12576, 4096), (4096, 4096), (12288, 4096), (4096, 6144), (2048, 4096), (4096, 1024)]:
        w = _w(n, k)
        q8, q4 = qmm.quantize8(w), qmm.quantize4(w, mse=True)
        for m in (512, 2048):
            x = _x(m, k)
            out = torch.empty((m, n), dtype=torch.bfloat16, device=DEV)
            r = []
            for q in (q8, q4):
                us = _time(lambda q=q: fast_qmm.matmul_prefill(x, q, out=out, exact=False, min_rows=1), reps=10)
                r.append((us / 1e3, 2 * m * n * k / us / 1e6))
            print(f"  {n:>6}x{k:<5} {m:>5}  " + " | ".join(f"{ms:6.3f} {tf:5.1f}" for ms, tf in r), flush=True)
    print("decode sweep, 16-row bucket (GLM53_TF_Q8_DECODE_CFG), 12576x4096 and 4096x4096 at 1 / 8 rows, us:")
    for cfg in ("1,4,3,64", "1,4,2,64", "2,4,2,64", "1,4,4,64", "1,8,3,64", "1,4,3,32", "1,2,3,32", "2,8,3,128"):
        monkeypatch.setenv("GLM53_TF_Q8_DECODE_CFG", cfg)
        cells = []
        for n, k in ((12576, 4096), (4096, 4096)):
            qs = [qmm.quantize8(_w(n, k)) for _ in range(4)]
            for m in (1, 8):
                x = _x(m, k)
                part = torch.empty((qmm.q8_split_k(n, k) * m * n,), dtype=torch.float32, device=DEV)
                out = torch.empty((m, n), dtype=torch.bfloat16, device=DEV)
                try:
                    cells.append(_time(lambda: [qmm.matmul(x, q, out=out, part=part) for q in qs]) / 4)
                except Exception as exc:  # noqa: BLE001 - a config that does not launch is reported, not fatal
                    cells.append(float("nan"))
                    print(f"  {cfg}: {type(exc).__name__}")
        print(f"  {cfg:>10}: " + " ".join(f"{c:7.1f}" for c in cells), flush=True)
    monkeypatch.delenv("GLM53_TF_Q8_DECODE_CFG")
    print("prefill sweep (GLM53_TF_Q8_TILE), 12576x4096 at 2,048 rows, ms:")
    q8 = qmm.quantize8(_w(12576, 4096))
    x = _x(2048, 4096)
    out = torch.empty((2048, 12576), dtype=torch.bfloat16, device=DEV)
    for tile in ("128,32,8,2", "64,64,8,2", "128,64,8,2", "64,32,4,3", "128,32,4,3", "64,64,4,2", "128,128,8,2"):
        monkeypatch.setenv("GLM53_TF_Q8_TILE", tile)
        try:
            us = _time(lambda: fast_qmm.matmul_prefill(x, q8, out=out, exact=False, min_rows=1), reps=10)
            print(f"  {tile:>12}: {us / 1e3:.3f} ms", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  {tile:>12}: {type(exc).__name__}")
