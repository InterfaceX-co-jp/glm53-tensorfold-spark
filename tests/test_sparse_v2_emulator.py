"""patches/0410 (GLM53_TF_SPARSE_V2=1): the v2 sparse latent attention kernel == the one-pass kernel it replaces
(``b12x_attn._lsparse_one``, ``tf_knobs.b12x`` bit 4), bit for bit, offline.

The reference runs in Triton's CPU interpreter; v2 is a Gluon kernel (not interpretable), so its own source runs on the
CPU model of its Gluon ops in ``tests/sparse_v2_emu.py`` (cp.async groups land at wait_group, shared memory as bytes,
a barrier / hazard model standing in for the warps). Both sides use the same arithmetic model:

- ``tl.dot`` / ``mma_v2`` = the per-element chain over K in 16-wide blocks from the accumulator c (``sparse_v2_emu.DOT``;
  order-sensitive), with the reference's ``o * alpha + dot`` folded into the dot's c as Triton's Combine pass does on
  the GPU (checked on the compiled TTGIR by tests/test_sparse_v2_compile.py);
- float32 element-wise math, nearest-even bf16, the same numpy reductions.

Checked, bit for bit: FP8 and bf16 caches, contiguous and paged (patches/0290, pages shuffled, junk elsewhere),
STAGES 2 / 3 / 4, QKL 0 / 1, QREG 0 / 1 (every speed setting), token counts 0, 1, KT +- 1, multiples of KT, 2,051 (the production
top-k width) at contexts of 700 and 6,000 tokens; rows untouched when their count is 0; rows == the same rows alone /
in subsets. Controls: an unfolded reference differs (the model sees the chain), and the hazard model catches a
missing wait and a missing barrier.

Run on its own (TRITON_INTERPRET must be set before triton is imported):
    TRITON_INTERPRET=1 PYTHONPATH=<patched tree>/src pytest -q tests/test_sparse_v2_emulator.py   (~10 s)
"""

from __future__ import annotations

import heapq
import importlib.util
import inspect
import os
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import sparse_v2_emu as emu  # noqa: E402

b12x_attn = pytest.importorskip("tensorfold.families.glm5_next.cuda.b12x_attn")
sparse_v2 = pytest.importorskip("tensorfold.families.glm5_next.cuda.sparse_v2")
from tensorfold.families.glm5_next.cuda import kvpool, latent  # noqa: E402

if os.environ.get("TRITON_INTERPRET") != "1" or type(b12x_attn._lsparse_one).__name__ != "InterpretedFunction":
    pytest.skip("needs Triton's interpreter (TRITON_INTERPRET=1 before triton is imported)", allow_module_level=True)
if torch.cuda.is_available():
    pytest.skip("CPU checks: run where no GPU is visible", allow_module_level=True)

L, H = 512, 32
SCALE = 256 ** -0.5


def _folded_ltile(tmp_dir):
    """``latent._ltile`` with ``o * alpha + tl.dot(p, kv)`` written as ``tl.dot(p, kv, o * alpha)``: what Triton's
    Combine pass makes of it on the GPU (the interpreter runs the source as written)."""

    src = inspect.getsource(latent._ltile.fn)
    old = "o = o * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv)"
    assert old in src
    src = src.replace(old, "o = tl.dot(p.to(tl.bfloat16), kv, o * alpha[:, None])")
    path = os.path.join(tmp_dir, "folded_ltile.py")
    with open(path, "w") as f:
        f.write("import triton\nimport triton.language as tl\n\n\n" + src)
    spec = importlib.util.spec_from_file_location("folded_ltile", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._ltile


@pytest.fixture(scope="module")
def folded(tmp_path_factory):
    return _folded_ltile(str(tmp_path_factory.mktemp("fold")))


@pytest.fixture(autouse=True)
def _model(monkeypatch, folded):
    """The interpreter on the GPU's semantics where the comparison needs them: fp32 -> bf16 nearest even, bf16 dot
    operands widened exactly, dots as the chain model, the accumulator folded into the second dot."""

    from triton.runtime import interpreter as interp
    import triton.language as tl

    cast = interp.InterpreterBuilder.cast_impl

    def cast_impl(self, src, dst_type):
        s, d = src.dtype.scalar, dst_type.scalar
        if s == tl.float32 and d == tl.bfloat16:
            return interp.TensorHandle(emu.f32_to_bf16_bits(src.data), d)
        if s == tl.bfloat16 and d == tl.float32:
            return interp.TensorHandle(emu.bf16_bits_to_f32(src.data), d)
        return cast(self, src, dst_type)

    def create_dot(self, a, b, d, input_precision, max_num_imprecise_acc):
        wa = emu.bf16_bits_to_f32(a.data) if a.dtype.scalar == tl.bfloat16 else a.data
        wb = emu.bf16_bits_to_f32(b.data) if b.dtype.scalar == tl.bfloat16 else b.data
        return interp.TensorHandle(emu.DOT(wa, wb, d.data), d.dtype.scalar)

    monkeypatch.setattr(interp.InterpreterBuilder, "cast_impl", cast_impl)
    monkeypatch.setattr(interp.InterpreterBuilder, "create_fp_trunc", lambda self, s, t: cast_impl(self, s, t))
    monkeypatch.setattr(interp.InterpreterBuilder, "create_fp_ext", lambda self, s, t: cast_impl(self, s, t))
    monkeypatch.setattr(interp.InterpreterBuilder, "create_dot", create_dot)
    monkeypatch.setattr(b12x_attn, "_ltile", folded)


def _cache(fp8: bool, T: int, seed: int = 1):
    """Latent rows like kv_a_layernorm's (a few large channels stress the FP8 row scale)."""

    g = torch.Generator().manual_seed(seed)
    gain = torch.exp(torch.randn(L, generator=g) * 0.5)
    gain[torch.randperm(L, generator=g)[:4]] *= 8
    lat = (torch.randn(T, L, generator=g) * gain).bfloat16()
    return latent.quantize_rows_reference(lat) if fp8 else lat


def _rows(counts, T: int, W: int, seed: int = 2):
    g = torch.Generator().manual_seed(seed)
    R = len(counts)
    qa = (torch.randn(R, H, L, generator=g) * 0.3).bfloat16()
    tok = torch.full((R, W), -1, dtype=torch.int32)
    for r, n in enumerate(counts):
        tok[r, :n] = torch.sort(torch.randperm(T, generator=g)[:n]).values.int()
    return qa, tok, torch.tensor(counts, dtype=torch.int32)


def _ref(qa, lc, tok, cnt, fill=7.0):
    out = torch.full(qa.shape, fill, dtype=torch.float32)
    b12x_attn.sparse_latent_one(qa.contiguous(), lc, tok.contiguous(), cnt.contiguous(), out, SCALE)
    return out


def _v2(qa, lc, tok, cnt, stages=3, qkl=1, qreg=1, fill=7.0):
    out = torch.full(qa.shape, fill, dtype=torch.float32)
    emu.run_v2(sparse_v2, qa.contiguous(), lc, tok.contiguous(), cnt.contiguous(), out, SCALE, stages, qkl, qreg)
    return out


def _same(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))           # bitwise, NaN-proof, -0 != +0


COUNTS = [300, 0, 1, 31, 32, 33, 64, 65, 137]
FMTS = [pytest.param(True, id="fp8"), pytest.param(False, id="bf16")]


@pytest.mark.parametrize("fp8", FMTS)
@pytest.mark.parametrize("stages,qkl,qreg", [(3, 1, 1), (2, 1, 1), (2, 1, 0), (2, 0, 0), (4, 1, 1)])
def test_v2_equals_the_one_pass_kernel(fp8, stages, qkl, qreg):
    T = 700
    lc = _cache(fp8, T)
    qa, tok, cnt = _rows(COUNTS, T, 300)
    ref = _ref(qa, lc, tok, cnt)
    got = _v2(qa, lc, tok, cnt, stages, qkl, qreg)
    assert _same(got, ref)
    assert bool((got[cnt == 0] == 7.0).all())                              # rows without tokens untouched
    assert not torch.isnan(got[cnt > 0]).any()


@pytest.mark.parametrize("fp8", FMTS)
def test_production_width_two_contexts(fp8):
    """2,051 selected tokens (sparse.TOPK_POOLS x POOL + POOL - 1, the fast-prefill list width) and shorter lists, at
    contexts of 2,100 and 6,000 tokens."""

    for T, counts, seed in ((2100, [2051, 2050, 1900], 11), (6000, [2051, 2051, 1024, 5], 12)):
        lc = _cache(fp8, T, seed=seed)
        qa, tok, cnt = _rows(counts, T, 2051, seed=seed + 1)
        assert _same(_v2(qa, lc, tok, cnt), _ref(qa, lc, tok, cnt)), T


@pytest.mark.parametrize("fp8", FMTS)
def test_rows_do_not_depend_on_the_launch(fp8):
    T = 700
    lc = _cache(fp8, T, seed=3)
    qa, tok, cnt = _rows(COUNTS, T, 300, seed=4)
    full = _v2(qa, lc, tok, cnt)
    for idx in ([2], [5, 3], [8, 0, 3], [6, 6, 6], list(range(len(COUNTS)))[::-1]):
        i = torch.tensor(idx)
        assert _same(_v2(qa[i], lc, tok[i], cnt[i]), full[i]), idx


def _paged(lc, T, seed):
    page = 256
    n = kvpool.pages_for(T, page)
    book = kvpool.PoolBook(n + 2, page)
    table = torch.full((n,), book.null, dtype=torch.int32)
    sp = kvpool.SlotPages(book, T, table)
    book.slots.append(sp)
    order = [int(x) for x in torch.randperm(n + 2, generator=torch.Generator().manual_seed(seed))[:n]]
    book.alloc.heap = [p for p in book.alloc.heap if p not in order]
    heapq.heapify(book.alloc.heap)
    sp.pages = order
    table[:] = torch.tensor(order, dtype=torch.int32)
    if latent.is_fp8(lc):
        phys = torch.full(((book.npages + 1) * page, lc.shape[1]), 0x7F, dtype=torch.uint8)     # NaN junk elsewhere
    else:
        phys = torch.full(((book.npages + 1) * page, lc.shape[1]), float("nan"), dtype=torch.bfloat16)
    phys[book.null * page:(book.null + 1) * page] = 0
    pg = kvpool.Paged(phys, sp, page, T)
    for lo, hi, view in pg.segments(0, T):
        view.copy_(lc[lo:hi])
    return pg


@pytest.mark.parametrize("fp8", FMTS)
def test_paged_equals_the_reference_and_contiguous(fp8):
    T = 900
    lc = _cache(fp8, T, seed=5)
    pg = _paged(lc, T, seed=6)
    qa, tok, cnt = _rows([300, 0, 33, 257, 1], T, 300, seed=7)
    got = _v2(qa, pg, tok, cnt)
    assert _same(got, _ref(qa, pg, tok, cnt))
    assert _same(got, _v2(qa, lc, tok, cnt))
    assert not torch.isnan(got[cnt > 0]).any()


def test_fp8_equals_bf16_on_dequantized_rows():
    T = 700
    lc = _cache(True, T, seed=8)
    qa, tok, cnt = _rows(COUNTS, T, 300, seed=9)
    assert _same(_v2(qa, lc, tok, cnt), _v2(qa, latent.dequantize_rows(lc).contiguous(), tok, cnt))


# -- controls -------------------------------------------------------------------------------------------------------
def test_control_unfolded_reference_differs(monkeypatch):
    """Without the Combine fold (c = 0, then + o * alpha) the bits differ: the chain model is not blind to it."""

    T = 700
    lc = _cache(True, T, seed=10)
    qa, tok, cnt = _rows([300, 137], T, 300, seed=11)
    got = _v2(qa, lc, tok, cnt)
    monkeypatch.setattr(b12x_attn, "_ltile", latent._ltile)
    assert not _same(got, _ref(qa, lc, tok, cnt))


def test_control_missing_wait_is_caught(monkeypatch):
    T = 700
    lc = _cache(True, T)
    qa, tok, cnt = _rows([100], T, 300)
    monkeypatch.setattr(emu.CP, "wait_group", lambda n=0: emu.Program.arena.wait(int(n) + 1))
    with pytest.raises(emu.Hazard):
        _v2(qa, lc, tok, cnt)


@pytest.mark.parametrize("drop", [0, 2, 3, 4])
def test_control_missing_barrier_is_caught(monkeypatch, drop):
    """Drop one of the key loop's barriers (FP8, QKL 1: 0 after the wait, 1 after the new copies -- there for Triton's
    barrier analysis and ptxas' schedule, not for a hazard of its own -- 2 after the bf16 tile, 3 after the scores,
    4 after p): each of 0, 2, 3, 4 guards a hazard the model sees."""

    T = 700
    lc = _cache(True, T)
    qa, tok, cnt = _rows([100], T, 300)
    seen = {"n": 0}

    def barrier(**k):
        i = seen["n"]
        seen["n"] += 1
        if i % 5 != drop:
            emu.Program.arena.barrier()

    monkeypatch.setattr(emu.GL, "barrier", barrier)
    with pytest.raises(emu.Hazard):
        _v2(qa, lc, tok, cnt)


def test_dot_model_is_order_sensitive():
    g = np.random.default_rng(0)
    a = emu.round_bf16(g.standard_normal((4, 64)).astype(np.float32))
    b = emu.round_bf16(g.standard_normal((64, 8)).astype(np.float32))
    c = g.standard_normal((4, 8)).astype(np.float32) * 1e3
    one = emu.DOT(a, b, c)
    split = emu.DOT(a[:, 32:], b[32:], emu.DOT(a[:, :32], b[:32], c))
    swapped = emu.DOT(a[:, :32], b[:32], emu.DOT(a[:, 32:], b[32:], c))
    assert np.array_equal(one, split) and not np.array_equal(one, swapped)


def test_control_not_vacuous():
    """v2 is close to a float64 softmax, and NOT the chunked kernel's bits (the 512-token split + merge differs)."""

    T = 1500
    lc = _cache(True, T, seed=13)
    qa, tok, cnt = _rows([1100, 700], T, 1100, seed=14)
    got = _v2(qa, lc, tok, cnt)
    ref = b12x_attn.reference(qa, latent.dequantize_rows(lc), tok, cnt, SCALE)
    rel = ((got.double() - ref).norm() / ref.norm()).item()
    assert 0 < rel < 1e-2, rel
    chunked = torch.full(qa.shape, 7.0)
    latent.sparse_latent(qa, lc, tok, cnt, chunked, SCALE, None, bm=32)
    assert not _same(got, chunked)
