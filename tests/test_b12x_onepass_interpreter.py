"""patches/0360 (``tf_knobs.b12x`` bit 4, one-pass sparse latent attention, on the FP8 latent cache production uses) in
Triton's CPU interpreter: small shapes, no GPU.

The one-pass kernel runs one program per (row, head tile) whose shapes never depend on the call's row count R, so here
(unlike the chunked kernels, whose numpy matmul blocking follows R) row independence is checked bit for bit:

- every row of a launch == the same row launched alone, in any subset, order, with duplicates, and among rows of
  other counts (0, 1, a KT tile +- 1, the whole window); bf16 and FP8 caches, 16- and 32-head tiles;
- FP8 rows == the bf16 kernel on ``dequantize_rows`` (the interpreter computes element-wise, so this checks the
  dequantization path; the compiled kernel's MMA layouts are checked by ``tests/test_b12x_onepass_compile.py``);
- paged FP8 rows (patches/0290) == contiguous; deterministic; untouched rows stay untouched;
- as close to a float64 softmax as the chunked kernel (bf16 and FP8).

Run it on its own (TRITON_INTERPRET must be set before triton is imported):
    TRITON_INTERPRET=1 PYTHONPATH=<patched tree>/src pytest -q tests/test_b12x_onepass_interpreter.py   (~2 minutes)
"""

from __future__ import annotations

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

b12x_attn = pytest.importorskip("tensorfold.families.glm5_next.cuda.b12x_attn")
from tensorfold.families.glm5_next.cuda import kvpool, latent  # noqa: E402

if os.environ.get("TRITON_INTERPRET") != "1" or type(b12x_attn._lsparse_one).__name__ != "InterpretedFunction":
    pytest.skip("needs Triton's interpreter (TRITON_INTERPRET=1 before triton is imported)", allow_module_level=True)
if torch.cuda.is_available():
    pytest.skip("CPU-interpreter checks: run where no GPU is visible", allow_module_level=True)

L, T = 512, 700
FMTS, FIDS = [False, True], ["bf16", "fp8"]


def _bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def _f32_to_bf16(f32):
    t = torch.from_numpy(np.ascontiguousarray(f32, dtype=np.float32)).to(torch.bfloat16)
    return t.view(torch.int16).numpy().view(np.uint16)


@pytest.fixture(autouse=True)
def _fix_interpreter(monkeypatch):
    """As tests/test_b12x_interpreter.py: fp32 -> bf16 casts round to nearest even, bf16 dot operands widened."""

    from triton.runtime import interpreter as interp
    import triton.language as tl

    cast, dot = interp.InterpreterBuilder.cast_impl, interp.InterpreterBuilder.create_dot

    def cast_impl(self, src, dst_type):
        s, d = src.dtype.scalar, dst_type.scalar
        if s == tl.float32 and d == tl.bfloat16:
            return interp.TensorHandle(_f32_to_bf16(src.data), d)
        if s == tl.bfloat16 and d == tl.float32:
            return interp.TensorHandle(_bf16_to_f32(src.data), d)
        return cast(self, src, dst_type)

    def create_dot(self, a, b, d, input_precision, max_num_imprecise_acc):
        if a.dtype.scalar == tl.bfloat16:
            a = interp.TensorHandle(_bf16_to_f32(a.data), tl.float32)
        if b.dtype.scalar == tl.bfloat16:
            b = interp.TensorHandle(_bf16_to_f32(b.data), tl.float32)
        return dot(self, a, b, d, input_precision, max_num_imprecise_acc)

    monkeypatch.setattr(interp.InterpreterBuilder, "cast_impl", cast_impl)
    monkeypatch.setattr(interp.InterpreterBuilder, "create_fp_trunc", lambda self, s, t: cast_impl(self, s, t))
    monkeypatch.setattr(interp.InterpreterBuilder, "create_fp_ext", lambda self, s, t: cast_impl(self, s, t))
    monkeypatch.setattr(interp.InterpreterBuilder, "create_dot", create_dot)


def _cache(fp8: bool, seed: int = 1):
    """Latent rows like kv_a_layernorm's (a few large channels stress the FP8 row scale)."""

    g = torch.Generator().manual_seed(seed)
    gain = torch.exp(torch.randn(L, generator=g) * 0.5)
    gain[torch.randperm(L, generator=g)[:4]] *= 8
    lat = (torch.randn(T, L, generator=g) * gain).bfloat16()
    return latent.quantize_rows_reference(lat) if fp8 else lat


COUNTS = [300, 0, 1, 31, 32, 33, 64, 65, 137]


def _rows(H: int, counts=COUNTS, W: int = 300, seed: int = 2):
    g = torch.Generator().manual_seed(seed)
    R = len(counts)
    qa = (torch.randn(R, H, L, generator=g) * 0.3).bfloat16()
    tok = torch.full((R, W), -1, dtype=torch.int32)
    for r, n in enumerate(counts):
        tok[r, :n] = torch.sort(torch.randperm(T, generator=g)[:n]).values.int()
    return qa, tok, torch.tensor(counts, dtype=torch.int32)


def _one(qa, lc, tok, cnt, fill=7.0):
    out = torch.full(qa.shape, fill, dtype=torch.float32)
    b12x_attn.sparse_latent_one(qa.contiguous(), lc, tok.contiguous(), cnt.contiguous(), out, 256 ** -0.5)
    return out


@pytest.mark.parametrize("H", [4, 32], ids=["bm16", "bm32"])
@pytest.mark.parametrize("fp8", FMTS, ids=FIDS)
def test_rows_do_not_depend_on_the_launch(fp8, H):
    lc = _cache(fp8)
    qa, tok, cnt = _rows(H)
    full = _one(qa, lc, tok, cnt)
    m = cnt > 0
    assert bool((full[~m] == 7.0).all())                              # rows without tokens untouched
    for r in range(len(COUNTS)):                                      # every row alone
        alone = _one(qa[r:r + 1], lc, tok[r:r + 1], cnt[r:r + 1])
        assert torch.equal(alone[0], full[r]), r
    for idx in ([2, 5], [8, 0, 3], [6, 6, 6], list(range(len(COUNTS)))[::-1], [4, 1, 7, 2, 0, 8]):
        i = torch.tensor(idx)
        sub = _one(qa[i], lc, tok[i], cnt[i])
        assert torch.equal(sub, full[i]), idx                         # subsets, orders, duplicates
    assert torch.equal(_one(qa, lc, tok, cnt), full)                  # deterministic


@pytest.mark.parametrize("H", [4, 32], ids=["bm16", "bm32"])
def test_fp8_equals_the_bf16_kernel_on_dequantized_rows(H):
    lc = _cache(True)
    qa, tok, cnt = _rows(H, seed=3)
    assert torch.equal(_one(qa, lc, tok, cnt), _one(qa, latent.dequantize_rows(lc).contiguous(), tok, cnt))


def test_paged_fp8_equals_contiguous():
    lc = _cache(True, seed=4)
    page = 256
    n = kvpool.pages_for(T, page)
    book = kvpool.PoolBook(n + 2, page)
    table = torch.full((n,), book.null, dtype=torch.int32)
    sp = kvpool.SlotPages(book, T, table)
    book.slots.append(sp)
    order = [int(x) for x in torch.randperm(n + 2, generator=torch.Generator().manual_seed(5))[:n]]
    book.alloc.heap = [p for p in book.alloc.heap if p not in order]
    import heapq

    heapq.heapify(book.alloc.heap)
    sp.pages = order
    table[:] = torch.tensor(order, dtype=torch.int32)
    phys = torch.full(((book.npages + 1) * page, lc.shape[1]), 0x7F, dtype=torch.uint8)     # NaN junk elsewhere
    phys[book.null * page:(book.null + 1) * page] = 0
    pg = kvpool.Paged(phys, sp, page, T)
    for lo, hi, view in pg.segments(0, T):
        view.copy_(lc[lo:hi])
    qa, tok, cnt = _rows(4, seed=6)
    a, b = _one(qa, lc, tok, cnt), _one(qa, pg, tok, cnt)
    assert torch.equal(a, b) and not torch.isnan(b[cnt > 0]).any()


@pytest.mark.parametrize("fp8", FMTS, ids=FIDS)
def test_close_to_float64_like_the_chunked_kernel(fp8):
    lc = _cache(fp8, seed=7)
    qa, tok, cnt = _rows(4, seed=8)
    one = _one(qa, lc, tok, cnt)
    chunked = torch.full(qa.shape, 7.0)
    latent.sparse_latent(qa, lc, tok, cnt, chunked, 256 ** -0.5, None, bm=16)
    ref = b12x_attn.reference(qa, latent.dequantize_rows(lc), tok, cnt, 256 ** -0.5)
    m = cnt > 0
    e1 = (one[m].double() - ref[m]).abs().max().item()
    e0 = (chunked[m].double() - ref[m]).abs().max().item()
    assert e1 <= 1.5 * e0 + 1e-4, (e1, e0)
    assert torch.equal(one[~m], chunked[~m])                          # both leave rows without tokens alone


def test_the_interpreter_skips_the_opaque_identity():
    """The inline-asm identity is compiled-only (``OPQ``); under the interpreter the FP8 path is the plain tile."""

    assert b12x_attn._INTERPRETED
