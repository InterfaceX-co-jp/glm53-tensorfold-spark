"""patches/0390 (GLM53_TF_MLA_EXPAND=v2) on the GPU: the retiled latent absorb / expand (``_absorb2`` / ``_expand2``)
give exactly the bits of v1 (``_absorb`` / ``_expand``) for every row, whatever the launch's row count, on the real
per-rank shapes (32 heads, head dim 256, latent 512), 4-bit (q4 and q4mse) and BF16 kv_b.

- v2 == v1 bit for bit (int16 views of the bf16 outputs) for launches of 1, 2, 3, 7, 8, 16, 17, 31, 64, 65, 127, 513,
  2,048 and 8,192 rows, random inputs, several seeds, plus inputs with extreme spreads (tiny / huge / signed zeros);
- row invariance: rows of an 8,192-row launch == the same rows launched alone and in 7- / 64- / 513-row windows at
  odd offsets;
- every entry of a forced tile table (16-128 rows a program, 16-64 k a dot step, 4 / 8 warps, BN 32 / 64 / 128 / 256)
  gives the same bits (the table is a speed knob only);
- control: v1 against a tensor-core ``absorb_tc`` / ``expand_tc`` differs (so equal bits are not vacuous);
- the flag works where the model calls it: ``latent.absorb`` / ``expand`` under ``EXPAND_V2`` equal v1, and inside
  a CUDA graph (decode / verify steps replay graphs) too.

A False anywhere: keep GLM53_TF_MLA_EXPAND unset (v1). The engine-level gates (exact 10/10, ab.py reply hash, drafted
== serial, resumed == fresh) are in docs/MLA-EXPAND.md.

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda pytest -q tests/cuda/test_mla_expand_patches.py  (~1-2 min)
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.glm5_next.cuda import latent, qmm  # noqa: E402

if not hasattr(latent, "_expand2"):
    pytest.skip("needs patches/0390", allow_module_level=True)

DEV = "cuda"
H, D, L = 32, 256, 512
ROWS = [1, 2, 3, 7, 8, 16, 17, 31, 64, 65, 127, 513, 2048, 8192]


def _kv(kind: str, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = (torch.randn((H * D, L), generator=g) * 0.05).to(torch.bfloat16).to(DEV)
    if kind == "q4":
        return qmm.quantize4(w)
    if kind == "q4mse":
        return qmm.quantize4(w, mse=True)
    return qmm.make_b16(w)


def _inputs(R: int, seed: int, extreme: bool = False):
    g = torch.Generator(device="cpu").manual_seed(1000 + seed)
    q = torch.randn((R, H, D), generator=g)
    u = torch.randn((R, H, L), generator=g) * 0.3
    if extreme:
        q[:, :, ::5] *= 1e4
        q[:, :, 1::7] *= 1e-6
        q[:, :, 2::11] = -0.0
        u[:, :, ::5] *= 1e6
        u[:, :, 1::7] *= 1e-9
        u[:, :, 2::11] = -0.0
    return q.to(torch.bfloat16).to(DEV).contiguous(), u.float().to(DEV).contiguous()


def _bits(t):
    return t.contiguous().view(torch.int16)


def _v1(q, u, kv_k, kv_v, monkeypatch):
    monkeypatch.setattr(latent, "EXPAND_V2", False)
    R = q.shape[0]
    a = latent.absorb(q, kv_k, torch.empty((R, H, L), dtype=torch.bfloat16, device=DEV))
    e = latent.expand(u, kv_v, torch.empty((R, H * D), dtype=torch.bfloat16, device=DEV))
    return a, e


def _v2(q, u, kv_k, kv_v, monkeypatch):
    monkeypatch.setattr(latent, "EXPAND_V2", True)
    R = q.shape[0]
    a = latent.absorb(q, kv_k, torch.empty((R, H, L), dtype=torch.bfloat16, device=DEV))
    e = latent.expand(u, kv_v, torch.empty((R, H * D), dtype=torch.bfloat16, device=DEV))
    return a, e


@pytest.fixture(scope="module", params=["q4", "q4mse", "b16"])
def kv(request):
    return request.param, _kv(request.param, 1), _kv(request.param, 2)


@pytest.mark.parametrize("R", ROWS)
def test_v2_equals_v1(kv, R, monkeypatch):
    _, kv_k, kv_v = kv
    for seed in (0, 1, 2):
        q, u = _inputs(R, seed + R, extreme=(seed == 2))
        a1, e1 = _v1(q, u, kv_k, kv_v, monkeypatch)
        a2, e2 = _v2(q, u, kv_k, kv_v, monkeypatch)
        assert torch.equal(_bits(a1), _bits(a2)), f"absorb v2 != v1 (R {R}, seed {seed})"
        assert torch.equal(_bits(e1), _bits(e2)), f"expand v2 != v1 (R {R}, seed {seed})"


def test_rows_independent_of_launch(kv, monkeypatch):
    _, kv_k, kv_v = kv
    R = 8192
    q, u = _inputs(R, 77)
    a1, e1 = _v1(q, u, kv_k, kv_v, monkeypatch)
    a, e = _v2(q, u, kv_k, kv_v, monkeypatch)
    assert torch.equal(_bits(a1), _bits(a)) and torch.equal(_bits(e1), _bits(e))
    for lo, n in ((0, 1), (1, 1), (4097, 1), (8191, 1), (13, 7), (1000, 64), (3001, 513), (8190, 2)):
        for fn in (_v1, _v2):
            aw, ew = fn(q[lo:lo + n].contiguous(), u[lo:lo + n].contiguous(), kv_k, kv_v, monkeypatch)
            assert torch.equal(_bits(aw), _bits(a[lo:lo + n])), (fn.__name__, lo, n)
            assert torch.equal(_bits(ew), _bits(e[lo:lo + n])), (fn.__name__, lo, n)


def test_tile_table_does_not_change_bits(kv, monkeypatch):
    _, kv_k, kv_v = kv
    R = 700
    q, u = _inputs(R, 5)
    a1, e1 = _v1(q, u, kv_k, kv_v, monkeypatch)
    for bm in (16, 32, 64, 128):
        for bk in (16, 32, 64):
            for warps in (4, 8):
                for bn in (32, 64, 128, 256):
                    if bm * max(bk, 64) * 4 > 64 * 1024 or (bm == 128 and bk == 64):
                        continue                                    # too big to be a sensible tile; skip compile
                    monkeypatch.setattr(latent, "V2_TILES", {"expand": ((1 << 62, bm, bk, warps),),
                                                             "absorb": ((1 << 62, bm, bk, warps),)})
                    monkeypatch.setattr(latent, "V2_BN", bn)
                    a, e = _v2(q, u, kv_k, kv_v, monkeypatch)
                    assert torch.equal(_bits(a), _bits(a1)), ("absorb", bm, bk, warps)
                    assert torch.equal(_bits(e), _bits(e1)), ("expand", bm, bk, warps, bn)


def test_control_tensor_core_differs(monkeypatch):
    """The tensor-core kernels (other arithmetic) differ from v1: bit equality above is a real check."""

    kv_k, kv_v = _kv("q4", 1), _kv("q4", 2)
    q, u = _inputs(64, 3)
    a1, e1 = _v1(q, u, kv_k, kv_v, monkeypatch)
    a3 = latent.absorb_tc(q, kv_k, torch.empty_like(a1))
    e3 = latent.expand_tc(u, kv_v, torch.empty_like(e1))
    assert not torch.equal(_bits(a1), _bits(a3)) and not torch.equal(_bits(e1), _bits(e3))


def test_v2_in_a_cuda_graph(monkeypatch):
    """Decode / verify / MTP steps replay CUDA graphs: v2 captured and replayed == v1 eager."""

    kv_k, kv_v = _kv("q4", 1), _kv("q4", 2)
    for R in (1, 3, 8):
        q, u = _inputs(R, 40 + R)
        a1, e1 = _v1(q, u, kv_k, kv_v, monkeypatch)
        monkeypatch.setattr(latent, "EXPAND_V2", True)
        qa = torch.empty((R, H, L), dtype=torch.bfloat16, device=DEV)
        o = torch.empty((R, H * D), dtype=torch.bfloat16, device=DEV)
        latent.absorb(q, kv_k, qa)                                       # warm-up (compile) outside the capture
        latent.expand(u, kv_v, o)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            latent.absorb(q, kv_k, qa)
            latent.expand(u, kv_v, o)
        qa.zero_()
        o.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(_bits(qa), _bits(a1)) and torch.equal(_bits(o), _bits(e1)), R
