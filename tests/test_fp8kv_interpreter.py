"""patches/0220 (GLM53_TF_KV_DTYPE=fp8) in Triton's CPU interpreter: the FP8 latent rows on small shapes, no GPU.

- ``latent_write`` on an FP8 cache: its bytes == ``latent.quantize_rows_reference`` (torch's float8_e4m3fn
  conversion of x / 2^k) on every finite bf16 value <= 448 and on random rows over 11 decades (zeros, -0.0, tiny
  values below e4m3's normal range); rows are independent of their block; untouched slots stay zero;
- dense and sparse latent attention over FP8 rows (16- and 32-query tiles) == the same kernels over the bf16
  dequantized rows, bit for bit;
- host logic: GLM53_TF_KV_DTYPE parsing, the session store's key tag, a snapshot never restoring across formats.

The writer rounds to the e4m3 grid itself (fp32 bit arithmetic) before the conversion: the interpreter's own
fp32 -> e4m3 conversion drops the carry when rounding crosses a power of two (7.84 -> 4.0), and this way the bytes
do not depend on how any backend rounds.

Run: TRITON_INTERPRET=1 PYTHONPATH=<patched TensorFold>/src pytest -q tests/test_fp8kv_interpreter.py
(the module sets TRITON_INTERPRET itself when it is imported first). ~1 minute on a laptop CPU.
"""

from __future__ import annotations

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
latent = pytest.importorskip("tensorfold.families.glm5_next.cuda.latent")

if not hasattr(latent, "ROW8"):
    pytest.skip("patches/0220 not applied", allow_module_level=True)
if os.environ.get("TRITON_INTERPRET") != "1" or type(latent._lwrite8).__name__ != "InterpretedFunction":
    pytest.skip("needs Triton's interpreter (TRITON_INTERPRET=1 before triton is imported)", allow_module_level=True)
if torch.cuda.is_available():
    pytest.skip("CPU-interpreter checks: run where no GPU is visible (tests/cuda/test_fp8_kv_patches.py on GPUs)",
                allow_module_level=True)

L, H = 512, 4


def _write(x: torch.Tensor, cap: int | None = None, at: int = 0) -> torch.Tensor:
    lc = torch.zeros((cap or x.shape[0], latent.ROW8), dtype=torch.uint8)
    latent.latent_write(x.contiguous(), lc, torch.tensor([at], dtype=torch.int32))
    return lc


def test_writer_equals_torch_conversion_on_every_bf16_value():
    pat = torch.arange(0, 1 << 16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float()
    pat = pat[torch.isfinite(pat) & (pat.abs() <= 448)]
    pat = pat[: (pat.numel() // 511) * 511].view(-1, 511)
    x = torch.cat([pat, torch.full((pat.shape[0], 1), 448.0)], 1).to(torch.bfloat16)
    got, want = _write(x), latent.quantize_rows_reference(x)
    assert torch.equal(got, want), int((got != want).any(1).sum())
    assert torch.all(got[:, 512:516].contiguous().view(torch.float32) == 1.0)


def test_writer_random_rows_scales_and_independence():
    g = torch.Generator().manual_seed(1)
    x = torch.randn((220, L), generator=g) * torch.logspace(-8, 3, 220)[:, None]
    x[3] = 0.0
    x[4] = -0.0
    x[5, 9] = 1e-30
    x[6] = 1.75
    x[7, 0] = 448.0
    x = x.to(torch.bfloat16)
    block = _write(x, cap=260, at=20)
    assert int(block[:20].sum()) == 0 and int(block[240:].sum()) == 0
    assert torch.equal(block[20:240], latent.quantize_rows_reference(x))
    for r in (0, 3, 6, 7, 100, 219):
        assert torch.equal(_write(x[r:r + 1], cap=260, at=20 + r)[20 + r], block[20 + r]), r
    s = block[20:240, 512:516].contiguous().view(torch.float32).flatten()
    amax = x.float().abs().amax(1)
    nz = amax > 0
    assert torch.all(amax[nz] / s[nz] <= 448) and torch.all(amax[nz] / s[nz] > 224)
    assert float(s[3]) == 1.0 and float(s[6]) == 2.0 ** -8
    dq = latent.dequantize_rows(block[20:240]).float()
    err = (dq - x.float()).abs()
    assert torch.all(err <= torch.maximum(x.float().abs() * 2.0 ** -4, s[:, None] * 2.0 ** -10))


def _scratch(rows: int, nch: int):
    s = SimpleNamespace(rows=rows, heads=H, lat=L, nch=nch)
    s.po = torch.empty(nch * rows * H * L)
    s.pm = torch.empty(nch * rows * H)
    s.pl = torch.empty(nch * rows * H)
    s.u = torch.empty(rows, H, L)
    return s


@pytest.mark.parametrize("bm", [16, 32])
def test_fp8_attention_equals_bf16_on_dequantized_rows(bm):
    g = torch.Generator().manual_seed(2)
    T, R = 700, 5
    lat = (torch.randn((T, L), generator=g) * 2).to(torch.bfloat16)
    lc8 = latent.quantize_rows_reference(lat)
    lcb = latent.dequantize_rows(lc8).contiguous()
    qa = torch.randn((R, H, L), generator=g).to(torch.bfloat16)
    nch = triton.cdiv(T, latent.CHUNK)
    pos = torch.tensor([T - R], dtype=torch.int32)
    a = latent.attention_latent(qa, lc8, pos, _scratch(R, nch), scale=0.06, out=torch.empty(R, H, L), bm=bm)
    b = latent.attention_latent(qa, lcb, pos, _scratch(R, nch), scale=0.06, out=torch.empty(R, H, L), bm=bm)
    assert torch.equal(a, b)
    W = 600
    tok = torch.stack([torch.sort(torch.randperm(T, generator=g)[:W]).values for _ in range(R)]).to(torch.int32)
    cnt = torch.tensor([W, 0, 300, W, 17], dtype=torch.int32)
    ua, ub = torch.zeros(R, H, L), torch.zeros(R, H, L)
    latent.sparse_latent(qa, lc8, tok, cnt, ua, 0.06, bm=bm)
    latent.sparse_latent(qa, lcb, tok, cnt, ub, 0.06, bm=bm)
    assert torch.equal(ua, ub)


def test_kv_dtype_setting(monkeypatch):
    monkeypatch.setenv("GLM53_TF_LATENT_KV", "1")
    monkeypatch.delenv("GLM53_TF_KV_DTYPE", raising=False)
    assert latent.kv_dtype() == "bf16"
    monkeypatch.setenv("GLM53_TF_KV_DTYPE", "FP8")
    assert latent.kv_dtype() == "fp8"
    monkeypatch.setenv("GLM53_TF_KV_DTYPE", "int8")
    with pytest.raises(ValueError):
        latent.kv_dtype()
    monkeypatch.setenv("GLM53_TF_KV_DTYPE", "fp8")
    monkeypatch.setenv("GLM53_TF_LATENT_KV", "0")
    with pytest.raises(ValueError, match="LATENT_KV"):
        latent.kv_dtype()


def test_session_keys_carry_the_row_format(monkeypatch):
    from tensorfold.families.glm5_next.cuda import sessions

    monkeypatch.setattr(sessions, "KV_TAG", b"")
    k16 = sessions.ids_key(64, [1, 2, 3], True, True)
    p16 = sessions.page_keys(0, list(range(600)), sessions.chain(list(range(600))), 2, True)
    monkeypatch.setattr(sessions, "KV_TAG", b"kv:fp8")
    assert sessions.ids_key(64, [1, 2, 3], True, True) != k16
    p8 = sessions.page_keys(0, list(range(600)), sessions.chain(list(range(600))), 2, True)
    assert not set(p8) & set(p16)
    assert sessions._kv_fp8(SimpleNamespace(w=SimpleNamespace(meta={"kv_fp8": True})))
    assert not sessions._kv_fp8(SimpleNamespace())


def test_snapshot_restore_refuses_the_other_format():
    from tensorfold.families.glm5_next.cuda import decode

    snap = decode.Snapshot([1, 2], torch.zeros(1), torch.zeros(1), None, -1, -1, 0, None, 1)
    e16 = SimpleNamespace(w=SimpleNamespace(meta={"kv_fp8": False}), st=None)
    with pytest.raises(ValueError, match="fp8 latent rows"):
        decode.restore(e16, snap)
    assert decode._kv8(SimpleNamespace(w=SimpleNamespace(meta={"kv_fp8": True}))) == 1
