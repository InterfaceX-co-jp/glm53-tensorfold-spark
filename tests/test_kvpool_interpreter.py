"""patches/0290 (GLM53_TF_KV_POOL_TOKENS) in Triton's CPU interpreter: every kernel that reads or writes a paged cache
gives the bits it gives on the contiguous cache, no GPU.

Each paged cache here is a scrambled page table (logical page i -> a physical page in shuffled / reversed order) over
a physical tensor whose unmapped pages hold junk that poisons any stray read (NaN bf16 rows, 0x7F bytes = FP8 NaN
with a huge scale), and a zero null page. Checked bit for bit (``torch.equal``) against the same call on the
contiguous cache:

- ``latent_write`` (bf16 and FP8 rows) across a page boundary: the logical rows equal the contiguous cache's, no other
  physical row changes, the null page stays zero;
- dense latent attention (16- and 32-query tiles) over 1,100 keys on 5 pages with the window straddling a page bound;
  sparse latent attention over random token lists (every page); patches/0240's one-pass sparse kernel;
- the indexer's cache update (keys, gates and pool keys all paged, as the MTP head's; or ring keys / gates with paged
  pool keys, as the model layers') over windows of 1-300 rows crossing pages;
- token selection on paged pool keys: the sorted path (decode windows), patches/0050's device path
  (``select_tokens_dev``) and patches/0065's row-block path, past the dense limit.

Plus the host bookkeeping (no interpreter needed): the allocator, a slot's table (ensure / truncate / quota), the
``Paged`` view rules, ``batchplan.pool_spills`` and the settings.

Run: TRITON_INTERPRET=1 PYTHONPATH=<patched TensorFold>/src pytest -q tests/test_kvpool_interpreter.py (~1-2 min).
"""

from __future__ import annotations

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
kvpool = pytest.importorskip("tensorfold.families.glm5_next.cuda.kvpool")
latent = pytest.importorskip("tensorfold.families.glm5_next.cuda.latent")
sparse = pytest.importorskip("tensorfold.families.glm5_next.cuda.sparse")
batchplan = pytest.importorskip("tensorfold.families.glm5_next.cuda.batchplan")

INTERP = os.environ.get("TRITON_INTERPRET") == "1" and type(latent._lwrite8).__name__ == "InterpretedFunction"
interp = pytest.mark.skipif(not INTERP or torch.cuda.is_available(),
                            reason="Triton's CPU interpreter, no GPU (tests/cuda/test_kv_pool_patches.py on GPUs)")

L, H, PAGE = 512, 4, 256


# -- paged copies of contiguous caches ------------------------------------------------------------------------------
def _slot(n_logical: int, capacity: int, *, spare: int = 3, seed: int = 0):
    """A pool with ``n_logical`` + ``spare`` pages and one slot whose first ``n_logical`` logical pages map to a
    scrambled set of physical pages."""

    book = kvpool.PoolBook(n_logical + spare, PAGE)
    g = torch.Generator().manual_seed(seed)
    order = [int(x) for x in torch.randperm(n_logical + spare, generator=g)[:n_logical]]
    table = torch.full((kvpool.pages_for(capacity, PAGE),), book.null, dtype=torch.int32)
    sp = kvpool.SlotPages(book, capacity, table)
    book.slots.append(sp)
    book.alloc.heap = [p for p in book.alloc.heap if p not in order]
    import heapq

    heapq.heapify(book.alloc.heap)
    sp.pages = order
    table[:n_logical] = torch.tensor(order, dtype=torch.int32)
    return book, sp


def _junk(shape, dtype):
    if dtype == torch.uint8:
        return torch.full(shape, 0x7F, dtype=torch.uint8)       # e4m3 NaN values, scale 3.4e38
    return torch.full(shape, float("nan"), dtype=dtype)


def _paged(t: torch.Tensor, sp, per: int) -> "kvpool.Paged":
    """A paged copy of contiguous ``t`` (rows = its logical rows): mapped pages hold ``t``'s rows, other physical pages
    junk, the null page zeros."""

    book = sp.pool
    phys = _junk(((book.npages + 1) * per, *t.shape[1:]), t.dtype)
    phys[book.null * per:(book.null + 1) * per] = 0
    p = kvpool.Paged(phys, sp, per, t.shape[0])
    end = min(t.shape[0], len(sp.pages) * per)
    for lo, hi, view in p.segments(0, end):
        view.copy_(t[lo:hi])
    return p


def _unmapped_rows(p) -> torch.Tensor:
    """The physical rows no logical row maps to (junk + the null page), for "nothing else was written" checks."""

    used = set()
    for i, pg in enumerate(p.slot.pages):
        used.update(range(pg * p.per, (pg + 1) * p.per))
    idx = [r for r in range(p.phys.shape[0]) if r not in used]
    return p.phys[idx].clone()


def _bits(t: torch.Tensor) -> torch.Tensor:
    """Raw bytes (junk is NaN, and NaN != NaN)."""

    return t.contiguous().view(torch.uint8)


def _lat(T: int, fp8: bool, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn((T, L), generator=g) * 2).to(torch.bfloat16)
    return latent.quantize_rows_reference(x) if fp8 else x


def _scratch(rows: int, nch: int):
    s = SimpleNamespace(rows=rows, heads=H, lat=L, nch=nch)
    s.po = torch.empty(nch * rows * H * L)
    s.pm = torch.empty(nch * rows * H)
    s.pl = torch.empty(nch * rows * H)
    s.u = torch.empty(rows, H, L)
    return s


FMTS = [False, True]
FIDS = ["bf16", "fp8"]


# -- latent rows ------------------------------------------------------------------------------------------------------
@interp
@pytest.mark.parametrize("fp8", FMTS, ids=FIDS)
def test_latent_write_across_a_page_boundary(fp8):
    T = 3 * PAGE
    _, sp = _slot(3, T, seed=1)
    base = _lat(T, fp8, 2)
    contig = base.clone()
    pg = _paged(base, sp, PAGE)
    before = _unmapped_rows(pg)
    g = torch.Generator().manual_seed(3)
    for at, R in ((PAGE - 7, 20), (0, 1), (2 * PAGE - 1, 2), (2 * PAGE + 5, PAGE - 5)):
        x = (torch.randn((R, L), generator=g) * 3).to(torch.bfloat16)
        pos = torch.tensor([at], dtype=torch.int32)
        latent.latent_write(x, contig, pos)
        latent.latent_write(x, pg, pos)
        assert torch.equal(pg.gather(0, T), contig), (at, R)
    assert torch.equal(_bits(_unmapped_rows(pg)), _bits(before))    # junk and the null page untouched
    assert int(pg.phys[sp.pool.null * PAGE:(sp.pool.null + 1) * PAGE].float().abs().sum()) == 0


@interp
@pytest.mark.parametrize("bm", [16, 32])
@pytest.mark.parametrize("fp8", FMTS, ids=FIDS)
def test_dense_attention_paged_equals_contiguous(fp8, bm):
    T, R = 1100, 5
    npg = kvpool.pages_for(T, PAGE)
    _, sp = _slot(npg, T + 8, seed=4)
    lc = _lat(T + 8, fp8, 5)
    pg = _paged(lc, sp, PAGE)
    g = torch.Generator().manual_seed(6)
    qa = torch.randn((R, H, L), generator=g).to(torch.bfloat16)
    nch = triton.cdiv(T, latent.CHUNK)
    for P in (T - R, 2 * PAGE - 3, 4 * PAGE - 2):              # the window inside a page / across page bounds
        pos = torch.tensor([P], dtype=torch.int32)
        a = latent.attention_latent(qa, lc, pos, _scratch(R, nch), scale=0.06, out=torch.empty(R, H, L), bm=bm)
        b = latent.attention_latent(qa, pg, pos, _scratch(R, nch), scale=0.06, out=torch.empty(R, H, L), bm=bm)
        assert torch.equal(a, b), P
        assert not torch.isnan(b).any()


@interp
@pytest.mark.parametrize("fp8", FMTS, ids=FIDS)
def test_sparse_attention_paged_equals_contiguous(fp8):
    from tensorfold.families.glm5_next.cuda import b12x_attn

    T, R, W = 1100, 5, 600
    _, sp = _slot(kvpool.pages_for(T, PAGE), T, seed=7)
    lc = _lat(T, fp8, 8)
    pg = _paged(lc, sp, PAGE)
    g = torch.Generator().manual_seed(9)
    qa = torch.randn((R, H, L), generator=g).to(torch.bfloat16)
    tok = torch.stack([torch.sort(torch.randperm(T, generator=g)[:W]).values for _ in range(R)]).to(torch.int32)
    tok[1, :4] = torch.tensor([PAGE - 1, PAGE, 2 * PAGE - 1, 2 * PAGE], dtype=torch.int32)   # page bounds
    tok[1] = torch.sort(tok[1]).values
    cnt = torch.tensor([W, W, 0, 300, 17], dtype=torch.int32)
    for bm in (16, 32):
        ua, ub = torch.zeros(R, H, L), torch.zeros(R, H, L)
        latent.sparse_latent(qa, lc, tok, cnt, ua, 0.06, bm=bm)
        latent.sparse_latent(qa, pg, tok, cnt, ub, 0.06, bm=bm)
        assert torch.equal(ua, ub), bm
    ua, ub = torch.zeros(R, H, L), torch.zeros(R, H, L)
    b12x_attn.sparse_latent_one(qa, lc, tok, cnt, ua, 0.06, bm=16)
    b12x_attn.sparse_latent_one(qa, pg, tok, cnt, ub, 0.06, bm=16)
    assert torch.equal(ua, ub)


# -- the indexer ------------------------------------------------------------------------------------------------------
def _index_inputs(R: int, g):
    kr = torch.randn((R, 128), generator=g).to(torch.bfloat16)
    gate = torch.randn((R, 128), generator=g)
    return kr, gate


@interp
@pytest.mark.parametrize("ring", [False, True], ids=["full", "ring"])
def test_index_update_paged_equals_contiguous(ring):
    cap = 1600
    npg = kvpool.pages_for(cap, PAGE)
    _, sp = _slot(npg, cap, seed=10)
    g = torch.Generator().manual_seed(11)
    ln_w = torch.randn(128, generator=g).to(torch.bfloat16)
    ln_b = torch.randn(128, generator=g).to(torch.bfloat16)
    ape = torch.randn((4, 128), generator=g).to(torch.bfloat16)
    pk_c = torch.zeros((cap // 4 + 2, 128), dtype=torch.bfloat16)
    pk_p = _paged(pk_c, sp, PAGE // 4)
    if ring:                      # the model layers': rings (contiguous, per slot) with paged pool keys
        ik_c = torch.zeros((512, 128), dtype=torch.bfloat16)
        ig_c = torch.zeros((512, 128), dtype=torch.bfloat16)
        ik_p, ig_p = ik_c.clone(), ig_c.clone()
    else:                         # the MTP head's: full keys and gates, paged
        ik_c = torch.zeros((cap, 128), dtype=torch.bfloat16)
        ig_c = torch.zeros((cap, 128), dtype=torch.bfloat16)
        ik_p, ig_p = _paged(ik_c, sp, PAGE), _paged(ig_c, sp, PAGE)
    pos = 0
    for R in (1, 3, 250, 7, 300, 2, 64, 200, 5, 1):
        kr, gate = _index_inputs(R, g)
        p = torch.tensor([pos], dtype=torch.int32)
        sparse.index_update(kr, gate, ln_w, ln_b, ape, ik_c, ig_c, pk_c, p)
        sparse.index_update(kr, gate, ln_w, ln_b, ape, ik_p, ig_p, pk_p, p)
        pos += R
        n_pools = pos // 4
        assert torch.equal(pk_p.gather(0, n_pools), pk_c[:n_pools]), (pos, R)
        if ring:
            assert torch.equal(ik_p, ik_c) and torch.equal(ig_p, ig_c)
        else:
            assert torch.equal(ik_p.gather(0, pos), ik_c[:pos]) and torch.equal(ig_p.gather(0, pos), ig_c[:pos])
    assert pos > 3 * PAGE


def _selection_inputs(R: int, nh: int, cap: int, g):
    qi = torch.randn((R, nh * 128), generator=g).to(torch.bfloat16)
    wts = torch.randn((R, nh), generator=g).to(torch.bfloat16)
    pk = torch.randn((cap // 4 + 2, 128), generator=g).to(torch.bfloat16)
    return qi, wts, pk


@interp
def test_selection_on_paged_pool_keys():
    cap, nh = 4096, 4
    _, sp = _slot(kvpool.pages_for(cap, PAGE), cap, seed=12)
    g = torch.Generator().manual_seed(13)
    # decode windows (the sorted path) and the device path (patches/0050)
    qi, wts, pk = _selection_inputs(4, nh, cap, g)
    pkp = _paged(pk, sp, PAGE // 4)
    for P in (2100, 3000, 4000):
        pos = torch.tensor([P], dtype=torch.int32)
        np_max = sparse.pool_bucket(P + 4, pk.shape[0] - 2)
        a = sparse.select_tokens(qi, wts, pk, P, 4, np_max, pos)
        b = sparse.select_tokens(qi, wts, pkp, P, 4, np_max, pos)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), P
        sc = sparse.LongScratch(8, cap, H, L, "cpu")
        a = [x.clone() for x in sparse.select_tokens_dev(qi, wts, pk, pos, 4, np_max, sc)]
        b = [x.clone() for x in sparse.select_tokens_dev(qi, wts, pkp, pos, 4, np_max, sc)]
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), P
    # prefill rows in blocks (patches/0065)
    R = 40
    qi, wts, pk = _selection_inputs(R, nh, cap, g)
    pkp = _paged(pk, sp, PAGE // 4)
    P = 2600
    pos = torch.tensor([P], dtype=torch.int32)
    a = sparse.select_pools_blocked(qi, wts, pk, P, R, pk.shape[0] - 2, pos, rows=16)
    b = sparse.select_pools_blocked(qi, wts, pkp, P, R, pk.shape[0] - 2, pos, rows=16)
    assert torch.equal(a, b)


# -- host bookkeeping -------------------------------------------------------------------------------------------------
def test_settings(monkeypatch):
    for k in (kvpool.POOL_ENV, kvpool.PAGE_ENV, kvpool.SLACK_ENV):
        monkeypatch.delenv(k, raising=False)
    assert kvpool.settings() == (0, 256, 64) and not kvpool.enabled()
    monkeypatch.setenv(kvpool.POOL_ENV, "1000")
    assert kvpool.settings() == (1024, 256, 64)                       # whole pages
    monkeypatch.setenv(kvpool.PAGE_ENV, "512")
    assert kvpool.settings()[:2] == (1024, 512)
    for bad in ("64", "384", "x"):
        monkeypatch.setenv(kvpool.PAGE_ENV, bad)
        with pytest.raises(ValueError):
            kvpool.settings()
    monkeypatch.setenv(kvpool.PAGE_ENV, "256")
    monkeypatch.setenv(kvpool.POOL_ENV, "-1")
    with pytest.raises(ValueError):
        kvpool.settings()
    assert kvpool.need_tokens(1000, 500, 1 << 20) == 1564 and kvpool.need_tokens(1000, 500, 1200) == 1200
    assert kvpool.pages_for(0, 256) == 0 and kvpool.pages_for(257, 256) == 2


def test_allocator_and_slot_tables():
    book = kvpool.PoolBook(8, 256)
    a = kvpool.SlotPages(book, 4 * 256)
    b = kvpool.SlotPages(book, 16 * 256)
    book.slots += [a, b]
    a.ensure(1)
    b.ensure(300)
    a.ensure(600)
    assert a.pages == [0, 3, 4] and b.pages == [1, 2]               # lowest free first, deterministic
    a.ensure(600)
    assert a.held == 3                                               # mapped already: nothing
    with pytest.raises(ValueError, match="capacity"):
        a.ensure(5 * 256)
    assert a.truncate(257) == 1 and a.pages == [0, 3] and book.alloc.n_free == 4
    # reservations: a promise to one slot is never given to another
    a.reserve(4)
    assert a.outstanding == 2 and book.available() == 2
    b.reserve(2)
    with pytest.raises(kvpool.PoolExhausted):
        b.ensure(5 * 256)                                            # 3 past its quota, only 2 unreserved
    b.ensure(3 * 256)                                                # 1 past its quota, unreserved: allowed, counted
    assert b.over == 1 and book.available() == 1
    a.ensure(4 * 256)
    assert a.held == 4 and book.alloc.n_free == 1
    a.truncate(0)
    b.truncate(0)
    assert book.alloc.n_free == 8 and sorted(book.alloc.heap) == list(range(8))


def test_slot_table_on_the_device_side():
    book = kvpool.PoolBook(6, 256)
    sp = kvpool.SlotPages(book, 4 * 256, torch.full((4,), book.null, dtype=torch.int32))
    book.slots.append(sp)
    book.alloc.take(2)                                               # pages 0, 1 used elsewhere
    sp.ensure(700)
    assert sp.table.tolist() == [2, 3, 4, 6]
    sp.truncate(256)
    assert sp.table.tolist() == [2, 6, 6, 6]


def test_paged_view_rules():
    book = kvpool.PoolBook(4, 256)
    sp = kvpool.SlotPages(book, 1024)
    book.slots.append(sp)
    phys = torch.arange(5 * 256 * 2, dtype=torch.int64).view(5 * 256, 2)
    p = kvpool.Paged(phys, sp, 256, 1024)
    assert p.shape == (1024, 2) and p.dtype == torch.int64 and p.is_contiguous()
    assert torch.equal(p[3], phys[book.null * 256 + 3])            # unmapped: reads the null page (size queries)
    with pytest.raises(IndexError, match="not mapped"):
        p[0:5]
    sp.ensure(512)
    sp.pages.reverse()                                               # logical 0 -> physical 1, 1 -> 0
    assert torch.equal(p[10:20], phys[256 + 10:256 + 20])
    assert p[10:20].data_ptr() == phys[266].data_ptr()               # a view, as a contiguous slice is
    with pytest.raises(IndexError, match="span pages"):
        p[250:260]
    segs = p.segments(250, 260)
    assert [(a, b) for a, b, _ in segs] == [(250, 256), (256, 260)]
    assert torch.equal(p.gather(250, 260), torch.cat([phys[256 + 250:512], phys[0:4]]))
    p[300] = torch.tensor([7, 8])
    assert phys[44].tolist() == [7, 8]
    p.write(254, torch.full((4, 2), -1, dtype=torch.int64))
    assert phys[256 + 254:512].eq(-1).all() and phys[0:2].eq(-1).all()
    with pytest.raises(IndexError):
        p[600] = torch.tensor([1, 1])                                 # an unmapped row is never written
    with pytest.raises(TypeError):
        p.clone()
    pk = kvpool.Paged(torch.zeros((5 * 64, 2)), sp, 64, 1024 // 4 + 2)
    assert pk.shape[0] == 258 and pk.shift == 6


def test_pool_spills():
    idle = [(0, 4, 10), (1, 0, 1), (2, 3, 5), (3, 6, 2)]            # (slot, pages held, admitted round)
    assert batchplan.pool_spills(5, 5, idle) == []
    assert batchplan.pool_spills(8, 5, idle) == [3]                  # least recently admitted first
    assert batchplan.pool_spills(8, 5, idle, prefer=3) == [2]        # the slot it would resume in last
    assert batchplan.pool_spills(15, 5, idle, prefer=3) == [2, 0, 3]
    assert batchplan.pool_spills(19, 5, idle) is None                # 5 + 13 < 19: it waits
