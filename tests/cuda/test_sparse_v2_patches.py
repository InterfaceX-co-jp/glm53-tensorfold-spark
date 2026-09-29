"""patches/0410 (GLM53_TF_SPARSE_V2=1): the v2 sparse latent attention kernel (``sparse_v2``, Gluon) == the one-pass
kernel of b12x bit 4 (``b12x_attn._lsparse_one``) bit for bit, on the GPU, on the real shapes (32 local heads, a
512-wide latent, 2,051 selected tokens):

- FP8 (production) and bf16 caches, contiguous and paged (patches/0290: shuffled pages, NaN junk elsewhere);
- every speed setting (stages / QKL / QREG);
- counts 0 (row untouched), 1, KT +- 1, 700, 2,050, 2,051; random and neighbour-overlapping token lists (the real
  top-k pattern: adjacent rows share most tokens);
- launch sizes 1 .. 2,048 rows: every row == the reference's row of a full launch (row-invariant, like the reference);
  subsets, permutations, duplicates; deterministic;
- extreme values: large logits (exp underflow in most of a row), signed zeros, zero latent rows;
- the dispatch: ``b12x_attn.sparse_latent_one`` with ``V2`` set gives the same tensor as ``v2=False``.

A single False means the arithmetic differs somewhere (a Triton whose lowering orders the chain or the reduction
differently for the two kernels): keep GLM53_TF_SPARSE_V2 off and save the case.
    PYTHONPATH=<tree>/src pytest -q -s tests/cuda/test_sparse_v2_patches.py            (~1-2 minutes)
"""

from __future__ import annotations

import heapq

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.glm5_next.cuda import b12x_attn, kvpool, latent, sparse_v2  # noqa: E402

H, L, W = 32, 512, 2051
SCALE = 256 ** -0.5
CFGS = [(3, 1, 1), (2, 1, 1), (2, 1, 0), (2, 0, 0)]      # (stages, qkl, qreg); the first is the default


def _lat(P, seed, zeros=0):
    g = torch.Generator().manual_seed(seed)
    gain = torch.exp(torch.randn(L, generator=g) * 0.5)
    gain[torch.randperm(L, generator=g)[:4]] *= 8
    lat = torch.randn(P, L, generator=g) * gain
    if zeros:
        lat[torch.randperm(P, generator=g)[:zeros]] = 0.0
    return lat.bfloat16().cuda()


def _inputs(seed, rows=96, P=6000, overlap=False, qscale=0.3, counts=None):
    g = torch.Generator().manual_seed(seed)
    qa = (torch.randn(rows, H, L, generator=g) * qscale).bfloat16()
    qa[0, 0, :7] = torch.tensor([0.0, -0.0, 0.0, -0.0, 1e-30, -1e-30, 0.0]).bfloat16()
    tok = torch.full((rows, W), -1, dtype=torch.int32)
    cnt = torch.zeros(rows, dtype=torch.int32)
    cur = torch.randperm(P, generator=g)[:W]
    pattern = counts or [W, 0, 700, 33, W - 1, 1, 32, 65, 31, W]
    for r in range(rows):
        n = pattern[r % len(pattern)]
        cnt[r] = n
        if overlap:                      # neighbours share most of their list: ~3% replaced a row
            k = max(1, W // 32)
            cand = torch.unique(torch.randint(0, P, (4 * k,), generator=g))
            new = cand[~torch.isin(cand, cur)][:k]
            cur[torch.randperm(W, generator=g)[:new.numel()]] = new
            sel = cur[:n]
        else:
            sel = torch.randperm(P, generator=g)[:n]
        tok[r, :n] = sel.sort().values.int()
    return qa.cuda(), tok.cuda(), cnt.cuda()


def _one(qa, lc, tok, cnt):
    out = torch.full(qa.shape, 7.0, device="cuda")
    b12x_attn.sparse_latent_one(qa.contiguous(), lc, tok.contiguous(), cnt.contiguous(), out, SCALE, v2=False)
    return out


def _v2(qa, lc, tok, cnt, cfg=None):
    if cfg is None:          # W10: the format's default (a bf16 cache cannot take FP8's 3 stages: 133 KB > 99 KB)
        cfg = sparse_v2.config(latent.is_fp8(lc.phys if getattr(lc, "is_paged", False) else lc))
    out = torch.full(qa.shape, 7.0, device="cuda")
    sparse_v2.sparse_latent_v2(qa.contiguous(), lc, tok.contiguous(), cnt.contiguous(), out, SCALE, stages=cfg[0],
                               qkl=cfg[1], qreg=cfg[2])
    return out


def _same(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


def _fmt(lat, cache):
    return lat if cache == "bf16" else latent.quantize_rows_reference(lat)


@pytest.mark.parametrize("cache", ["fp8", "bf16"])
@pytest.mark.parametrize("cfg", CFGS, ids=lambda c: f"s{c[0]}q{c[1]}r{c[2]}")
@pytest.mark.parametrize("seed", [4101, 4102, 4103])
def test_v2_equals_the_one_pass_kernel(cache, cfg, seed):
    if sparse_v2.smem_need(cache == "fp8", *cfg) > sparse_v2.SMEM_MAX:
        pytest.skip(f"{cfg} does not fit a GB10 block with a {cache} cache (sparse_v2.config refuses it)")
    lc = _fmt(_lat(6000, seed, zeros=20), cache)
    for overlap in (False, True):
        qa, tok, cnt = _inputs(seed + 7, overlap=overlap)
        ref, got = _one(qa, lc, tok, cnt), _v2(qa, lc, tok, cnt, cfg)
        m = cnt > 0
        assert _same(got, ref), (overlap, (got[m] - ref[m]).abs().max().item())
        assert bool((got[~m] == 7.0).all()) and not torch.isnan(got[m]).any()


@pytest.mark.parametrize("cache", ["fp8", "bf16"])
def test_v2_rows_are_launch_invariant(cache):
    lc = _fmt(_lat(30000, 4110), cache)
    qa, tok, cnt = _inputs(4111, rows=2048, P=30000, overlap=True)
    full = _one(qa, lc, tok, cnt)
    assert _same(_v2(qa, lc, tok, cnt), full)
    for R in (1, 2, 7, 31, 32, 33, 64, 65, 513):
        for o in (0, 700):
            s = slice(o, o + R)
            assert _same(_v2(qa[s], lc, tok[s], cnt[s]), full[s]), (R, o)
    g = torch.Generator().manual_seed(4112)
    for idx in (torch.randperm(2048, generator=g)[:300], torch.tensor([5, 5, 5, 2]), torch.arange(99, -1, -1)):
        i = idx.cuda()
        assert _same(_v2(qa[i], lc, tok[i], cnt[i]), full[i])
    assert _same(_v2(qa, lc, tok, cnt), _v2(qa, lc, tok, cnt))                  # deterministic


def _paged(lc, T, seed):
    page = 256
    n = kvpool.pages_for(T, page)
    book = kvpool.PoolBook(n + 2, page)
    table = torch.full((n,), book.null, dtype=torch.int32, device="cuda")
    sp = kvpool.SlotPages(book, T, table)
    book.slots.append(sp)
    order = [int(x) for x in torch.randperm(n + 2, generator=torch.Generator().manual_seed(seed))[:n]]
    book.alloc.heap = [p for p in book.alloc.heap if p not in order]
    heapq.heapify(book.alloc.heap)
    sp.pages = order
    table[:] = torch.tensor(order, dtype=torch.int32, device="cuda")
    shape = ((book.npages + 1) * page, lc.shape[1])
    if latent.is_fp8(lc):
        phys = torch.full(shape, 0x7F, dtype=torch.uint8, device="cuda")              # NaN junk elsewhere
    else:
        phys = torch.full(shape, float("nan"), dtype=torch.bfloat16, device="cuda")
    phys[book.null * page:(book.null + 1) * page] = 0
    pg = kvpool.Paged(phys, sp, page, T)
    for lo, hi, view in pg.segments(0, T):
        view.copy_(lc[lo:hi])
    return pg


@pytest.mark.parametrize("cache", ["fp8", "bf16"])
def test_v2_paged(cache):
    T = 6000
    lc = _fmt(_lat(T, 4120), cache)
    pg = _paged(lc, T, 4121)
    qa, tok, cnt = _inputs(4122, P=T)
    got = _v2(qa, pg, tok, cnt)
    assert _same(got, _one(qa, pg, tok, cnt))
    assert _same(got, _v2(qa, lc, tok, cnt))
    assert not torch.isnan(got[cnt > 0]).any()


@pytest.mark.parametrize("qscale", [3.0, 30.0])
def test_v2_extreme_logits(qscale):
    lc = _fmt(_lat(6000, 4130, zeros=200), "fp8")
    qa, tok, cnt = _inputs(4131, qscale=qscale)
    assert _same(_v2(qa, lc, tok, cnt), _one(qa, lc, tok, cnt))


def test_v2_fp8_equals_bf16_on_dequantized_rows():
    lc = _fmt(_lat(6000, 4140), "fp8")
    qa, tok, cnt = _inputs(4141)
    m = cnt > 0
    assert _same(_v2(qa, lc, tok, cnt)[m], _v2(qa, latent.dequantize_rows(lc).contiguous(), tok, cnt)[m])


def test_the_knob_dispatches_to_v2(monkeypatch):
    lc = _fmt(_lat(6000, 4150), "fp8")
    qa, tok, cnt = _inputs(4151)
    ref = _one(qa, lc, tok, cnt)
    monkeypatch.setenv(sparse_v2.ENV, "1")
    assert b12x_attn.enable_v2() and b12x_attn.V2 is sparse_v2
    out = torch.full(qa.shape, 7.0, device="cuda")
    b12x_attn.sparse_latent_one(qa, lc, tok, cnt, out, SCALE)
    monkeypatch.setattr(b12x_attn, "V2", None)
    assert _same(out, ref)


def test_timing_print():
    """Not an assertion: 1,024 rows x 2,051 tokens, FP8 (bench_sparse_v2.py has the full sweep)."""

    lc = _fmt(_lat(30000, 4160), "fp8")
    qa, tok, cnt = _inputs(4161, rows=1024, P=30000, overlap=True, counts=[W])
    fns = {"one": lambda: _one(qa, lc, tok, cnt), "v2": lambda: _v2(qa, lc, tok, cnt)}
    ms = {}
    for name, fn in fns.items():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(10):
            fn()
        e.record()
        torch.cuda.synchronize()
        ms[name] = s.elapsed_time(e) / 10
    print(f"\n[0410 fp8] one pass {ms['one']:.2f} ms -> v2 {ms['v2']:.2f} ms ({ms['one'] / ms['v2']:.2f}x)")
