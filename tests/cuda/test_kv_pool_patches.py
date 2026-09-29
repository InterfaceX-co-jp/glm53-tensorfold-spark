"""patches/0290 (GLM53_TF_KV_POOL_TOKENS): one paged pool of latent KV rows behind every batch slot.

Host / CPU (torch on the CPU, no GPU; PYTHONPATH=<tree>/src:<tree>/tests/cuda:tests/cuda pytest -q <this file>):

- the real ``Batcher._plan`` / ``_execute`` / ``_admit`` / ``_piece`` / ``_finish`` / ``follow`` with the real session
  store on 0180's hostile fake model, every slot's caches (KV entries, prefix digests, MTP rows) paged views of one
  shared pool with 256-token pages: 4 sessions over a shared system prompt on 3 slots, exact and fast prefill; a roomy
  pool, and a tight one (admissions wait, idle slots are spilled, pages end up scattered): every reply and every
  slot's whole state at the end of a request equals a fresh prefill + serial decode on contiguous caches (resumed ==
  fresh across spills, slot moves, page boundaries and fragmented tables); the null page is never written
  (GLM53_TF_KV_POOL_CHECK in every round); nothing leaks (every page free or held, no reservation left);
- a request larger than the pool is refused and does not block the queue;
- a follower replaying rank 0's messages (plans with the spill lists) makes the same decisions: the same replies, the
  same stores, the same page tables;
- the session store across paged slots (save from one scrambled table, restore into another) and 0250's NVMe tier
  (pages written from and read into paged slots) on the same fake.

GPU (TensorFold's synthetic checkpoint, one GPU playing rank 0 of two), to run in a GPU window:

- the compiled kernels, paged (scrambled tables, poisoned unmapped pages) == contiguous bit for bit on the real shapes
  (32 local heads, latent 512, bf16 and FP8 rows, BM 16 / 32): latent writes, dense and sparse attention, b12x's
  one-pass kernel, the indexer update, the three selection paths; a captured CUDA graph replayed after the table and
  the rows moved to other pages;
- the engine with the pool vs without, same checkpoint: a 3,000-token prompt past the dense limit leaves the same
  committed rows (latent, MTP latent, pool keys, MTP index keys) and the same replies (serial, drafted, greedy,
  sampled), also with a fragmented table (the allocator scrambled first) and pages of 512; resumed == fresh;
- batching 4 on a pool smaller than 4 slots x capacity: replies == alone, waits / spills happen, null page clean;
  disk-tier sessions restored into paged slots; ``kv_bytes_per_token`` and the pool's bytes.

In the image: PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q tests/cuda/test_kv_pool_patches.py
"""

from __future__ import annotations

import collections
import contextlib
import heapq
import threading
from types import SimpleNamespace

import pytest

try:
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:          # the host-only tests still run
    torch = None
    CUDA = False

from tensorfold.families.glm5_next.cuda import batchplan, kvpool, sessions

gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")
needs_torch = pytest.mark.skipif(torch is None, reason="torch")

PAGE = 256


# -- host: paged fake slots -------------------------------------------------------------------------------------------
class _Book(kvpool.PoolBook):
    """A pool for the fake model: three int64 row tensors (KV entries, prefix digests, MTP rows), unmapped pages full
    of junk, the null page zeros."""

    def __init__(self, npages: int, slack: int = 16, page: int = PAGE) -> None:
        super().__init__(npages, page)
        self.slack = slack
        rows = (npages + 1) * page
        self.phys = {k: torch.full((rows,), -7, dtype=torch.int64) for k in ("kv", "dg", "mkv")}
        for t in self.phys.values():
            t[self.null * page:] = 0

    def new_slot(self, cap: int):
        sp = kvpool.SlotPages(self, cap, torch.full((kvpool.pages_for(cap, self.page),), self.null, dtype=torch.int32))
        self.slots.append(sp)
        return sp

    def null_clean(self) -> bool:
        return all(int(t[self.null * self.page:].abs().sum()) == 0 for t in self.phys.values())

    def nbytes(self) -> int:
        return sum(t.numel() * 8 for t in self.phys.values())


class _Dig:
    """``_St.dig`` over a paged tensor: dig[p] = digest of the entries before p (dig[0] = 0), stored at position p - 1
    (so the digest of position p's entry is paged with position p)."""

    def __init__(self, dg) -> None:
        self.dg = dg

    def __getitem__(self, p: int):
        return 0 if p == 0 else self.dg[p - 1]

    def __setitem__(self, p: int, v) -> None:
        assert p >= 1
        self.dg[p - 1] = v


class _PSt:
    """``test_batch_sessions_patches._St`` on the pool: the same fields, caches as ``kvpool.Paged`` views."""

    def __init__(self, book: _Book, cap: int = 4096) -> None:
        self.capacity = cap
        self.pages = book.new_slot(cap)
        self.kv = kvpool.Paged(book.phys["kv"], self.pages, book.page, cap)
        self.dg = kvpool.Paged(book.phys["dg"], self.pages, book.page, cap)
        self.mkv = kvpool.Paged(book.phys["mkv"], self.pages, book.page, cap)
        self.dig = _Dig(self.dg)
        self.kc = [self.kv, self.dg]
        self.vc = self.kc
        self.index, self.dsa_index = None, {}
        self.mtp_kc = self.mtp_vc = self.mkv
        self.rec = torch.zeros((2, 1), dtype=torch.int64)
        self.conv = torch.zeros((1, 3), dtype=torch.int64)
        self.proj = torch.zeros((1, 8))
        self.cur = [0]
        self.pos = self.mtp_len = self.mtp_drafted = 0
        self.win: list[int] = []
        self.hid = torch.zeros((1, 1), dtype=torch.int64)

    def reset(self) -> None:
        self.rec.zero_()
        self.conv.zero_()
        self.cur = [0]
        self.pos = self.mtp_len = self.mtp_drafted = 0

    def set_pos(self, n: int) -> None:
        self.pos = n

    def set_mtp_len(self, n: int) -> None:
        self.mtp_len = n

    def state(self) -> tuple:
        return (self.pos, int(self.rec[self.cur[0], 0]), self.conv.tolist(), self.kv.gather(0, self.pos).tolist(),
                [0] + self.dg.gather(0, self.pos).tolist(), self.mtp_len, self.mkv.gather(0, self.mtp_len).tolist())


def _stage(w, st, b, tokens):
    """``forward.stage`` for the fake: ``check_room``'s page mapping, then the model's staging."""

    from test_batch_sessions_patches import MODEL

    kvpool.ensure(st, st.pos + len(tokens))
    return MODEL.stage(w, st, b, tokens)


def _pool_batcher(monkeypatch, *, n: int = 3, npages: int = 64, rows: int = 64, fast: bool = False,
                  piece: int = 256, budget_pages: float = 4000.0, rank: int = 0, fork: int = 64, page: int = PAGE):
    """``test_batch_sessions_patches._fake_batcher`` with every slot on one ``_Book`` (the batcher's ``kvp``)."""

    import types

    import test_batch_sessions_patches as tb
    from tensorfold.families.glm5_next.cuda import batch, decode
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    monkeypatch.setattr(decode, "stage", _stage)
    for name in ("compute", "commit"):
        monkeypatch.setattr(decode, name, getattr(tb.MODEL, name))
    monkeypatch.setattr(decode, "_sync", lambda w: None)
    monkeypatch.setattr(batch, "Stepper", tb._Stepper)
    monkeypatch.setattr(batch.Batcher, "_free", staticmethod(lambda: 1 << 50))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)

    def sample_multi(w, logits, specs):
        return [[int(logits[off + i, 0]) % 5 for i in range(R)] for off, R, _, _ in specs]

    monkeypatch.setattr(batch, "sample_multi", sample_multi)
    book = _Book(npages, page=page)
    states = [_PSt(book) for _ in range(n)]
    for st in states:
        st.pages.reserve(0)                   # as ``Batcher.__init__``: slots start empty and unreserved

    class _PE(tb._E):
        def mtp(self, next_tokens, hidden):
            kvpool.ensure(self.st, self.st.mtp_len + len(next_tokens))       # ``mtp_stage``'s check_room
            return super().mtp(next_tokens, hidden)

    e = _PE(states[0], rows, fast)
    g = SimpleNamespace(e=e, w=e.w, drafter=None, rank=rank, f_most=7, costs=tb.COSTS, store=None, serial_only=False)
    for name in ("_drafters", "_grid", "_store_save"):
        setattr(g, name, types.MethodType(getattr(GlmEngine, name), g))
    g._share = lambda values: list(values)
    g._knobs = lambda values: contextlib.nullcontext()
    bat = object.__new__(batch.Batcher)
    bat.__dict__.update(
        g=g, n=n, states=states, graphs=[None] * n, drafters=[None] * n, max_rows=8, piece=piece,
        fair=batchplan.Fairness(1.0), reserve=0.0, admit_min=0.0, last_verify=False, store=None, store_budget=0,
        counts=collections.Counter(), caches=[[] for _ in range(n)], used=[0] * n, seqs=[None] * n, round=0,
        eos=(), costs=tb.COSTS, round_costs=batchplan.RoundCosts(tb.COSTS), defaults=tb._values(fast, rows),
        log=collections.deque(maxlen=1000), trace=collections.deque(maxlen=10000), last_piece_s=0.0,
        cv=threading.Condition(), stopping=False, queue=collections.deque(), error=None, thread=None,
        use_graphs=False, multi={}, pool=None, following=False, kvp=book, pool_check=True, _spills=[])

    def forward(active, windows):
        out = []
        for s, win in zip(active, windows):
            with bat._on(s) as ee:
                _stage(None, ee.st, None, win)
                out.append(tb.MODEL.compute(None, ee.st, None, len(win)))
        return torch.cat(out)

    bat._forward = forward
    store = sessions.SessionStore(e, None, 0, every_tokens=0, fork_tokens=fork)
    store.index.budget = int(budget_pages * store.index.page_bytes)
    g.store = store
    bat.attach_store(store)
    bat.ended = {}
    finish = bat._finish

    def spy(slot, *, cancelled=False):
        q = bat.seqs[slot]
        bat.ended[id(q.job)] = (bat.states[slot].state(), q.job.stats.get("restored"),
                                q.job.stats.get("restored") in store.index.entries)
        return finish(slot, cancelled=cancelled)

    bat._finish = spy
    # fragmentation seen during the run: a slot whose pages are not one ascending run
    bat.scattered = 0
    execute = bat._execute

    def watch(*args):
        execute(*args)
        for st in bat.states:
            p = st.pages.pages
            if p != list(range(p[0], p[0] + len(p))) if p else False:
                bat.scattered += 1

    bat._execute = watch
    return bat


def _pool_accounting(bat) -> None:
    book = bat.kvp
    held = sum(st.pages.held for st in bat.states)
    assert held + book.alloc.n_free == book.npages
    assert all(st.pages.quota == 0 for st in bat.states) and book.outstanding() == 0
    assert sorted(book.alloc.heap + [p for st in bat.states for p in st.pages.pages]) == list(range(book.npages))
    assert book.null_clean()


@needs_torch
@pytest.mark.parametrize("page", [256, 512])
@pytest.mark.parametrize("fast", [False, True], ids=["exact", "fast"])
@pytest.mark.parametrize("tight", [False, True], ids=["roomy", "tight"])
def test_pooled_slots_on_a_hostile_fake_model(monkeypatch, fast, tight, page):
    """Every reply and slot state == fresh prefill + serial decode on contiguous caches, with a roomy pool (3 slots of
    16 pages fit) and a tight one (2,048 tokens; these requests need 512-1,280: admissions wait, idle slots are
    spilled); pool pages of 256 tokens (store runs gathered / scattered at once) and 512 (a copy a store page)."""

    import numpy as np
    from test_batch_sessions_patches import _conversations

    npages = (2048 if tight else 16384) // page
    bat = _pool_batcher(monkeypatch, npages=npages, fast=fast, budget_pages=60.0 if tight else 4000.0, page=page)
    stats = _conversations(bat, rng=np.random.default_rng(17 + int(fast)), fast=fast, rows=64)
    assert len(stats) == 56
    assert all(s["kv_pages"] <= npages for s in stats)
    restored = [s for s in stats if s.get("restored")]
    assert restored and len({s["slot"] for s in restored}) >= 2
    if tight:
        assert bat.counts["pool_wait"] > 0 and bat.counts["pool_spills"] > 0, bat.counts
        assert bat.scattered > 0                          # tables with pages out of order occurred
    else:
        assert bat.counts["pool_wait"] == 0 and bat.counts["pool_spills"] == 0
    _pool_accounting(bat)


@needs_torch
def test_a_request_larger_than_the_pool_is_refused(monkeypatch):
    from test_batch_sessions_patches import _drain, _job, _reference

    bat = _pool_batcher(monkeypatch, npages=6)
    big = _job(list(range(3, 1800)), 8, False, 64)         # 1797 + 8 + 16 tokens: 8 pages > 6
    small = _job(list(range(5, 300)), 8, False, 64)
    bat.queue.extend([big, small])
    while bat.queue or any(s is not None for s in bat.seqs):
        bat._execute(*bat._plan())
    with pytest.raises(ValueError, match="KV pages"):
        _drain(big)
    reply, done = _drain(small)
    assert done and reply == _reference(list(range(5, 300)), 8, 64, False)[0]
    assert bat.counts["pool_refused"] == 1
    _pool_accounting(bat)


@needs_torch
def test_pooled_follower_replays_rank0(monkeypatch):
    """Rank 0's messages (plans, the spill lists, prompts, session plans, save decisions) replayed through ``follow``
    on a second pooled batcher: the same requests end the same way, the same stores, the same page tables."""

    import numpy as np
    from test_batch_sessions_patches import _conversations

    r0 = _pool_batcher(monkeypatch, npages=8, budget_pages=60.0)
    r1 = _pool_batcher(monkeypatch, npages=8, budget_pages=60.0, rank=1)
    sent: list[list[int]] = []

    def record(values):
        sent.append([int(v) for v in values])
        return list(values)

    class Done(Exception):
        pass

    def replay(values):
        assert values is None
        if not sent:
            raise Done
        return sent.pop(0)

    r0.g._share, r1.g._share = record, replay
    _conversations(r0, rng=np.random.default_rng(3), fast=False, rows=64, turns=8, check=False)
    assert r0.counts["pool_spills"] > 0
    with pytest.raises(Done):
        r1.follow()
    key = lambda d: (d["slot"], d["sha256"], tuple(d["keeps"]), d["cancelled"])      # noqa: E731
    assert [key(d) for d in r1.log] == [key(d) for d in r0.log] and len(r0.log) == 32
    assert r0.store.index.digest() == r1.store.index.digest() and r0.store.stats == r1.store.stats
    assert [st.state() for st in r0.states] == [st.state() for st in r1.states]
    assert [st.pages.pages for st in r0.states] == [st.pages.pages for st in r1.states]
    assert r0.counts["pool_spills"] == r1.counts["pool_spills"]


@needs_torch
def test_store_copies_between_scrambled_paged_slots():
    """A session saved from a slot whose pages are scrambled and restored into slots with other pages: the logical
    rows are the entry's; pages a slot holds are skipped; runs of store pages are copied page by page."""

    import test_batch_sessions_patches as tb

    book = _Book(40)
    slots = [_PSt(book, 2048) for _ in range(3)]
    for sp in (slots[1].pages, slots[2].pages):             # interleave the allocations: scattered tables
        sp.ensure(256)
    slots[0].pages.ensure(1024)
    slots[2].pages.ensure(900)
    slots[1].pages.ensure(900)
    assert slots[0].pages.pages != sorted(slots[0].pages.pages) or len(set(slots[0].pages.pages)) == 4
    e = tb._E(slots[0], 64, False)
    store = sessions.SessionStore(e, None, 0, every_tokens=0, fork_tokens=64)
    store.index.budget = 1000 * store.index.page_bytes
    store.bind_slots(slots)
    ids = list(range(3, 900))
    st0 = slots[0]
    for t in range(len(ids)):
        st0.kv[t] = t * 7 + 1
        st0.dg[t] = t * 11 + 2
        st0.mkv[t] = t * 13 + 3
    snap = SimpleNamespace(ids=ids, rec=torch.tensor([5]), conv=torch.zeros(3), pending=torch.zeros((1, 1)),
                           mtp_len=len(ids) - 1, drafter_end=-1, grid=0, window=None)
    with store.on(0):
        store.begin(None, 0)
        assert store.save(snap).status == sessions.SAVED
    entry = store.entry_of(snap)
    for s in (2, 1):
        with store.on(s):
            assert store.restore(entry) is snap
        n = len(ids)
        assert slots[s].kv.gather(0, n).tolist() == st0.kv.gather(0, n).tolist()
        assert slots[s].dg.gather(0, n).tolist() == st0.dg.gather(0, n).tolist()
        assert slots[s].mkv.gather(0, n - 1).tolist() == st0.mkv.gather(0, n - 1).tolist()
    assert store.stats["copied"] == 6
    with store.on(2):
        store.restore(entry)
    assert store.stats["skipped"] == 3 and book.null_clean()


@needs_torch
@pytest.mark.parametrize("fast", [False, True], ids=["exact", "fast"])
def test_pooled_slots_with_the_disk_tier(monkeypatch, tmp_path, fast):
    """0250's NVMe tier behind pooled slots (20-page RAM store, 8-page KV pool): pages written from paged slots and
    read back into other paged slots (the read starts at admission, after the round's spills); every reply and slot
    state == fresh prefill + serial decode."""

    import numpy as np
    from test_batch_sessions_patches import _conversations

    from tensorfold.families.glm5_next.cuda import sessdisk

    bat = _pool_batcher(monkeypatch, npages=8, fast=fast, budget_pages=20.0)
    tier = sessdisk.DiskTier(bat.store, tmp_path, 0, {"pool": 1, "fast": fast}, budget=1 << 40, min_tok=64, gain=0,
                             n_threads=3, quiet=True)
    tier.reconcile()
    bat.store.attach_disk(tier)
    stats = _conversations(bat, rng=np.random.default_rng(23 + int(fast)), fast=fast, rows=64, turns=10)
    from_disk = [s for s in stats if s.get("restored_disk")]
    assert from_disk and all(s["disk"]["ok"] for s in from_disk)
    assert bat.counts["pool_spills"] > 0
    tier.writer.drain()
    _pool_accounting(bat)


# -- GPU ------------------------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")
    for k in (kvpool.POOL_ENV, kvpool.PAGE_ENV, kvpool.SLACK_ENV, kvpool.CHECK_ENV):
        monkeypatch.delenv(k, raising=False)


def _g_slot(n_logical: int, capacity: int, *, spare: int = 3, seed: int = 0, dev: str = "cuda"):
    book = kvpool.PoolBook(n_logical + spare, PAGE)
    gen = torch.Generator().manual_seed(seed)
    order = [int(x) for x in torch.randperm(n_logical + spare, generator=gen)[:n_logical]]
    sp = kvpool.SlotPages(book, capacity, torch.full((kvpool.pages_for(capacity, PAGE),), book.null,
                                                     dtype=torch.int32, device=dev))
    book.slots.append(sp)
    book.alloc.heap = [p for p in book.alloc.heap if p not in order]
    heapq.heapify(book.alloc.heap)
    sp.pages = order
    sp.table[:n_logical] = torch.tensor(order, dtype=torch.int32, device=dev)
    return book, sp


def _g_paged(t, sp, per: int):
    book = sp.pool
    if t.dtype == torch.uint8:
        phys = torch.full(((book.npages + 1) * per, *t.shape[1:]), 0x7F, dtype=t.dtype, device=t.device)
    else:
        phys = torch.full(((book.npages + 1) * per, *t.shape[1:]), float("nan"), dtype=t.dtype, device=t.device)
    phys[book.null * per:(book.null + 1) * per] = 0
    p = kvpool.Paged(phys, sp, per, t.shape[0])
    for lo, hi, view in p.segments(0, min(t.shape[0], len(sp.pages) * per)):
        view.copy_(t[lo:hi])
    return p


@gpu
@torch.no_grad()
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
def test_gpu_kernels_paged_equal_contiguous(fp8):
    """The compiled kernels on the real shapes (32 local heads, latent 512): paged == contiguous bit for bit."""

    from tensorfold.families.glm5_next.cuda import b12x_attn, latent

    H, L, dev = 32, 512, "cuda"
    T, R = 3000, 8
    _, sp = _g_slot(kvpool.pages_for(T + 64, PAGE), T + 64, seed=1)
    g = torch.Generator().manual_seed(2)
    x = (torch.randn((T + 64, L), generator=g) * 2).to(torch.bfloat16).to(dev)
    lc = latent.quantize_rows_reference(x) if fp8 else x.clone()
    pg = _g_paged(lc, sp, PAGE)
    # writes across a page bound
    y = (torch.randn((40, L), generator=g) * 3).to(torch.bfloat16).to(dev)
    pos = torch.tensor([2 * PAGE - 17], dtype=torch.int32, device=dev)
    latent.latent_write(y, lc, pos)
    latent.latent_write(y, pg, pos)
    assert torch.equal(pg.gather(0, T), lc[:T])
    # dense (a window across a page bound) and sparse attention, both tiles
    qa = torch.randn((R, H, L), generator=g).to(torch.bfloat16).to(dev)
    s = SimpleNamespace(rows=R, heads=H, lat=L, nch=4)
    for P in (2 * PAGE - 3, 2040):
        pos = torch.tensor([P], dtype=torch.int32, device=dev)
        for bm in (16, 32):
            outs = []
            for cache in (lc, pg):
                s.po = torch.empty(4 * R * H * L, device=dev)
                s.pm = torch.empty(4 * R * H, device=dev)
                s.pl = torch.empty(4 * R * H, device=dev)
                outs.append(latent.attention_latent(qa, cache, pos, s, scale=0.06, nch=-(-(P + R) // 512),
                                                    out=torch.empty(R, H, L, device=dev), bm=bm).clone())
            assert torch.equal(outs[0], outs[1]), (P, bm)
    W = 2051
    tok = torch.stack([torch.sort(torch.randperm(T, generator=g)[:W]).values for _ in range(R)]).to(torch.int32).to(dev)
    cnt = torch.tensor([W, W, 0, 1000, 17, W, 5, W], dtype=torch.int32, device=dev)
    for bm in (16, 32):
        ua, ub = torch.zeros(R, H, L, device=dev), torch.zeros(R, H, L, device=dev)
        latent.sparse_latent(qa, lc, tok, cnt, ua, 0.06, bm=bm)
        latent.sparse_latent(qa, pg, tok, cnt, ub, 0.06, bm=bm)
        assert torch.equal(ua, ub), bm
    ua, ub = torch.zeros(R, H, L, device=dev), torch.zeros(R, H, L, device=dev)
    b12x_attn.sparse_latent_one(qa, lc, tok, cnt, ua, 0.06)
    b12x_attn.sparse_latent_one(qa, pg, tok, cnt, ub, 0.06)
    assert torch.equal(ua, ub)


@gpu
@torch.no_grad()
def test_gpu_indexer_and_selection_paged_equal_contiguous():
    from tensorfold.families.glm5_next.cuda import sparse

    dev, cap = "cuda", 8192
    _, sp = _g_slot(kvpool.pages_for(cap, PAGE), cap, seed=3)
    g = torch.Generator().manual_seed(4)
    ln_w = torch.randn(128, generator=g).to(torch.bfloat16).to(dev)
    ln_b = torch.randn(128, generator=g).to(torch.bfloat16).to(dev)
    ape = torch.randn((4, 128), generator=g).to(torch.bfloat16).to(dev)
    ik = torch.zeros((cap, 128), dtype=torch.bfloat16, device=dev)
    ig, pk = ik.clone(), torch.zeros((cap // 4 + 2, 128), dtype=torch.bfloat16, device=dev)
    ikp, igp, pkp = _g_paged(ik, sp, PAGE), _g_paged(ig, sp, PAGE), _g_paged(pk, sp, PAGE // 4)
    pos = 0
    for R in (1, 3, 2048, 7, 1000, 2, 511, 1):
        kr = torch.randn((R, 128), generator=g).to(torch.bfloat16).to(dev)
        gate = torch.randn((R, 128), generator=g).to(dev)
        p = torch.tensor([pos], dtype=torch.int32, device=dev)
        sparse.index_update(kr, gate, ln_w, ln_b, ape, ik, ig, pk, p)
        sparse.index_update(kr, gate, ln_w, ln_b, ape, ikp, igp, pkp, p)
        pos += R
    assert torch.equal(pkp.gather(0, pos // 4), pk[:pos // 4]) and torch.equal(ikp.gather(0, pos), ik[:pos])
    qi = torch.randn((64, 32 * 128), generator=g).to(torch.bfloat16).to(dev)
    wts = torch.randn((64, 32), generator=g).to(torch.bfloat16).to(dev)
    pk.copy_(torch.randn(pk.shape, generator=g).to(torch.bfloat16))
    pkp = _g_paged(pk, sp, PAGE // 4)
    for P, R in ((2100, 4), (3500, 8), (3000, 64)):
        pos = torch.tensor([P], dtype=torch.int32, device=dev)
        npm = sparse.pool_bucket(P + R, pk.shape[0] - 2)
        a = sparse.select_tokens(qi[:R], wts[:R], pk, P, R, npm, pos)
        b = sparse.select_tokens(qi[:R], wts[:R], pkp, P, R, npm, pos)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), (P, R)
        if R <= 8:
            sc = sparse.LongScratch(8, cap, 32, 512, dev)
            a = [t.clone() for t in sparse.select_tokens_dev(qi[:R], wts[:R], pk, pos, R, npm, sc)]
            b = [t.clone() for t in sparse.select_tokens_dev(qi[:R], wts[:R], pkp, pos, R, npm, sc)]
            assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), (P, R)


@gpu
@torch.no_grad()
def test_gpu_graph_replay_after_pages_move():
    """A captured sparse step reads the table at replay: rows moved to other pages (table and rows updated between
    replays, as ``ensure`` / a restore do) give the contiguous result again."""

    from tensorfold.families.glm5_next.cuda import latent

    H, L, dev, T, R = 32, 512, "cuda", 2600, 4
    book, sp = _g_slot(kvpool.pages_for(T, PAGE), T, spare=12, seed=5)
    g = torch.Generator().manual_seed(6)
    lc = (torch.randn((T, L), generator=g) * 2).to(torch.bfloat16).to(dev)
    pg = _g_paged(lc, sp, PAGE)
    qa = torch.randn((R, H, L), generator=g).to(torch.bfloat16).to(dev)
    tok = torch.stack([torch.sort(torch.randperm(T, generator=g)[:2051]).values for _ in range(R)])
    tok = tok.to(torch.int32).to(dev)
    cnt = torch.full((R,), 2051, dtype=torch.int32, device=dev)
    want = torch.zeros(R, H, L, device=dev)
    latent.sparse_latent(qa, lc, tok, cnt, want, 0.06)
    out = torch.zeros(R, H, L, device=dev)
    part = tuple(torch.empty(n, device=dev) for n in (5 * R * H * L, 5 * R * H, 5 * R * H))
    latent.sparse_latent(qa, pg, tok, cnt, out, 0.06, part)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        latent.sparse_latent(qa, pg, tok, cnt, out, 0.06, part)
    graph.replay()
    assert torch.equal(out, want)
    old = list(sp.pages)                                   # move every logical page to a spare physical page
    new = sorted(book.alloc.heap)[:len(old)] if len(book.alloc.heap) >= len(old) else None
    if new is None:
        new = list(reversed(old))
    for i, (a, b) in enumerate(zip(old, new)):
        pg.phys[b * PAGE:(b + 1) * PAGE].copy_(pg.phys[a * PAGE:(a + 1) * PAGE])
    for a in old:
        if a not in new:
            pg.phys[a * PAGE:(a + 1) * PAGE].fill_(float("nan"))
    sp.pages = new
    sp.table[:len(new)] = torch.tensor(new, dtype=torch.int32, device=dev)
    out.zero_()
    graph.replay()
    assert torch.equal(out, want)


def _engine(path, *, pool: int = 0, page: int = 256, context: int = 4096, rows: int = 64, batch: int = 1,
            gib: float = 0.0, sessions_on: bool = False, disk=None):
    from test_glm_engine import _TwoCopies

    from tensorfold.families.glm5_next.cuda import weights
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    with pytest.MonkeyPatch.context() as m:
        m.setattr(weights, "NONEXPERT", "q4mse")
        m.setenv("GLM53_TF_NONEXPERT", "q4mse")
        m.setenv("GLM53_TF_LATENT_KV", "1")
        m.setenv("GLM53_TF_PREFILL_ROWS", str(rows))
        m.setenv("GLM53_TF_BATCH", str(batch))
        m.setenv("GLM53_TF_BATCH_SESSIONS", "1" if sessions_on else "0")
        m.setenv("GLM53_TF_SESSION_GIB", str(gib))
        m.setenv("GLM53_TF_SESSION_RESERVE_GIB", "0")
        m.setenv("GLM53_TF_BATCH_RESERVE_GB", "0.25")
        m.setenv("GLM53_TF_BATCH_ADMIT_GB", "0")
        m.setenv("GLM53_TF_BATCH_PREFILL_SHARE", "1.0")
        if pool:
            m.setenv(kvpool.POOL_ENV, str(pool))
            m.setenv(kvpool.PAGE_ENV, str(page))
            m.setenv(kvpool.CHECK_ENV, "1")
        if disk is not None:
            m.setenv("GLM53_TF_SESSION_DISK", str(disk))
            m.setenv("GLM53_TF_SESSION_DISK_GIB", "4")
            m.setenv("GLM53_TF_SESSION_DISK_MIN", "64")
            m.setenv("GLM53_TF_SESSION_DISK_GAIN", "0")
        for k in ("GLM53_TF_CALIB_ONLINE", "GLM53_TF_PROFILE", "GLM53_TF_DEPTH", "GLM53_TF_LEAN_PREFILL",
                  "GLM53_TF_FAST_PREFILL", "GLM53_TF_FP8_PREFILL", "GLM53_TF_KV_DTYPE"):
            m.delenv(k, raising=False)
        return GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", context=context,
                         comm=_TwoCopies())


@pytest.fixture(scope="module")
def long_ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter
    from test_patches import _index_heads_32

    path = tmp_path_factory.mktemp("glm_kv_pool")
    _checkpoint(path / "model", exl3=True)
    _index_heads_32(path / "model")
    _drafter(path / "dflash2")
    return path


def _committed(eng, n: int) -> list:
    """The committed rows of the lone engine's slot after a prefill of n tokens, contiguous (paged or not)."""

    st = eng.e.st

    def rows(t, a, b):
        return t.gather(a, b) if kvpool.is_paged(t) else t[a:b].clone()

    out = [rows(k, 0, n) for k in st.kc] + [rows(st.mtp_kc, 0, st.mtp_len)]
    for i, (ik, ig, pk) in enumerate(st.index):
        out.append(rows(pk, 0, n // 4))
        if ik.shape[0] >= st.capacity:
            out += [rows(ik, 0, n if i < len(st.index) - 1 else st.mtp_len),
                    rows(ig, 0, n if i < len(st.index) - 1 else st.mtp_len)]
    return out + [st.rec[st.cur[0]].clone(), st.conv.clone()]


def _scramble(eng, tokens: int) -> None:
    """Map the lone slot's pages for ``tokens`` positions in a shuffled order (the allocator alone hands them out
    ascending): a fragmented table for the prefill that follows."""

    import random

    sp = eng.e.st.pages
    sp.truncate(0)
    need = kvpool.pages_for(tokens, sp.page)
    got = sp.pool.alloc.take(need)
    random.Random(9).shuffle(got)
    sp.pages = got
    sp.table[:need] = torch.tensor(got, dtype=torch.int32, device=sp.table.device)


@gpu
@torch.no_grad()
@pytest.mark.parametrize("page", [256, 512])
def test_gpu_pool_prefill_state_equals_unpaged(long_ckpt, page):
    """A 3,000-token prompt past the dense limit: the same committed rows and states with and without the pool, also
    with a fragmented table; replies equal (serial, drafted; greedy, sampled); resumed == fresh on the pool."""

    import numpy as np
    from test_glm_engine import _generate

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.glm5_next.cuda.decode import prefill

    prompt = [int(t) for t in np.random.default_rng(31).integers(0, 1000, size=3000)]
    off = _engine(long_ckpt)
    first_off = prefill(off.e, prompt, None, mtp=True)
    want = _committed(off, len(prompt))
    replies = {}
    for sampling, sid in ((None, "greedy"), (Sampling(1234, 1.0, 20, 0.95), "sampled")):
        replies[sid] = _generate(off, prompt, sampling, draft=False, tokens=24)[0]
    del off
    torch.cuda.empty_cache()
    on = _engine(long_ckpt, pool=6 * 4096, page=page)
    assert kvpool.is_paged(on.e.st.kc[0]) and on.e.st.pages is not None
    _scramble(on, 4096)
    first_on = prefill(on.e, prompt, None, mtp=True)
    got = _committed(on, len(prompt))
    assert first_on == first_off and len(got) == len(want)
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    p = on.e.st.pages.pages
    assert p != sorted(p)                                      # the table really is fragmented
    for sampling, sid in ((None, "greedy"), (Sampling(1234, 1.0, 20, 0.95), "sampled")):
        serial, _ = _generate(on, prompt, sampling, draft=False, tokens=24)
        assert serial == replies[sid], sid
        for policy in ("2", "f3", None):
            drafted, _ = _generate(on, prompt, sampling, policy=policy, tokens=24)
            assert drafted == serial, (sid, policy)
        after = prompt + serial + [3, 4]
        warm, stats = _generate(on, after, sampling, tokens=16)
        assert stats["cached"] >= len(prompt)
        _generate(on, [1, 2, 3], sampling, tokens=4)
        cold, stats = _generate(on, after, sampling, tokens=16)
        assert stats["cached"] == 0 and warm == cold, sid
    assert on.w.meta["kv_pool"].null_clean()


@gpu
def test_gpu_pool_bytes_and_accounting(long_ckpt):
    from tensorfold.families.glm5_next.cuda import latent

    off = _engine(long_ckpt)
    per = latent.kv_bytes_per_token(off.e.st)
    del off
    torch.cuda.empty_cache()
    on = _engine(long_ckpt, pool=8192)
    book = on.w.meta["kv_pool"]
    assert latent.kv_bytes_per_token(on.e.st) == per
    assert book.nbytes() == book.rows * per                     # the pool holds rows of exactly the slot's bytes
    assert book.npages == 32 and book.null == 32


@gpu
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
def test_gpu_batch_on_a_small_pool_equals_alone(long_ckpt, greedy):
    """4 slots of 4,096 tokens on a pool of 6,144 (1.5 slots): 4 concurrent 2,500-token sessions with a store; waves
    queue, wait and spill; every reply equals the request served alone; the null page is never written."""

    import numpy as np

    from tensorfold.engine.exact_sampling import Sampling

    sampling = None if greedy else Sampling(1234, 1.0, 20, 0.95)
    ref = _engine(long_ckpt)
    rng = np.random.default_rng(41)
    sys_prompt = [int(t) for t in rng.integers(0, 1000, size=2300)]
    turns = {k: sys_prompt + [int(t) for t in rng.integers(0, 1000, size=n)]
             for k, n in zip("ABCD", (150, 170, 130, 110))}
    want = {}
    from test_glm_engine import _generate

    eng = _engine(long_ckpt, pool=6144, batch=4, gib=1.0, sessions_on=True)
    assert eng.batch.n == 4 and eng.batch.kvp is not None
    for w in range(2):
        order = list("ABCD") if w == 0 else list("DBCA")
        for k in order:
            want[k] = _generate(ref, turns[k], sampling, draft=False, tokens=16)[0]
        got = eng.batch.generate_batch([dict(prompt=turns[k], max_tokens=16, sampling=sampling) for k in order])
        for k, (reply, stats) in zip(order, got):
            assert reply == want[k], (w, k, stats)
            assert stats["kv_pages"] <= 24
            turns[k] = turns[k] + reply + [7 + w]
    c = eng.batch.counts
    assert c["pool_wait"] > 0 and c["pool_spills"] > 0, c
    assert eng.batch.kvp.null_clean() and eng.batch.kvp.outstanding() == 0
    eng.batch.stop()


@gpu
def test_gpu_disk_sessions_into_pooled_slots(long_ckpt, tmp_path):
    """0250: sessions evicted from a tiny RAM store come back from disk into pooled slots, same replies."""

    import numpy as np

    from test_glm_engine import _generate

    ref = _engine(long_ckpt)
    eng = _engine(long_ckpt, pool=8192, batch=3, gib=0.05, sessions_on=True, disk=tmp_path)
    rng = np.random.default_rng(51)
    prompts = [[int(t) for t in rng.integers(0, 1000, size=n)] for n in (2200, 2300, 2400, 2500)]
    for rnd in range(2):
        if rnd:     # a later turn: a stored entry resumes only a strict prefix (an identical prompt never does)
            prompts = [p + [1] * 40 for p in prompts]
        got = eng.batch.generate_batch([dict(prompt=p, max_tokens=12, sampling=None) for p in prompts])
        for p, (reply, stats) in zip(prompts, got):
            assert reply == _generate(ref, p, None, draft=False, tokens=12)[0], (rnd, stats)
            if rnd:
                assert stats["cached"] > 0, stats
    eng.batch.stop()
