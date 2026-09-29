"""patches/0110 (GLM53_TF_SESSION_GIB > 0, ``glm5_next/cuda/sessions.py``): the multi-session state cache.

Host only (no torch; runnable anywhere: PYTHONPATH=<tree>/src pytest -q tests/cuda/test_session_patches.py):

- page rules: which pages an entry shares (exact: the next token for MTP rows; fast: the whole chunk), tail bytes;
- lookup: the longest stored strict prefix of the request's own grid whose draft caches fit; the common prefix with
  any entry; the marks (fork point, every N tokens, on the grid, not stored twice);
- pages shared between entries with reference counts, LRU eviction within the budget, empty slabs released, an
  entry larger than the budget skipped without evicting anything;
- rank 1 replaying rank 0's decisions ends with the same store (digest), and a divergence fails loudly.

On the GPU (TensorFold's synthetic GLM checkpoint, one GPU playing rank 0 of two):

- sessions A, B, A, C, B, A interleaved (a shared system prompt, forks, follow-up turns): every reply equals a fresh
  prefill + serial decoding of the same prompt on an engine without the store, and the resumes come from the store;
- the shared system prompt's pages are stored once and not copied again when the live caches hold them;
- a budget so small that sessions are evicted: still exact; an entry larger than the budget is not stored;
- past 2,051 tokens on the latent cache, exact and fast prefill (patches/0080 grid, marks on the grid);
- rank 1 following rank 0's messages (header, prompt, plan, save decisions) makes the same decisions and the same
  replies, and a store that diverged is detected.

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda pytest -q tests/cuda/test_session_patches.py
"""

from __future__ import annotations

import random

import pytest

try:
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:          # the host-only tests still run
    CUDA = False

from tensorfold.families.glm5_next.cuda import sessions
from tensorfold.families.glm5_next.cuda.sessions import (
    DUPLICATE,
    PAGE,
    SAVED,
    SKIPPED,
    SessionIndex,
    key_end,
    page_count,
)

gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")
ROWS = (1024, 64, 512, 128)          # bytes a position: main rows, main pool rows, MTP rows, MTP pool rows


def _index(budget_pages: float = 1000.0, extent: int = 4) -> SessionIndex:
    ix = SessionIndex(0, ROWS, extent=extent)
    ix.budget = int(budget_pages * ix.page_bytes)
    return ix


def _save(ix, ids, *, tag=0, mtp=True, drafter=True, snap=10_000, forced=None):
    return ix.save(tag=tag, ids=ids, mtp_len=len(ids) - 1 if mtp else -1, drafter=drafter, snap_bytes=snap,
                   forced=forced)


def _toks(rng, n):
    return [rng.randrange(1000) for _ in range(n)]


# -- host only: page rules ------------------------------------------------------------------------------------------
def test_page_rules_exact():
    # a page is shared when the entry holds the token after it (its last MTP row reads it) and its MTP rows
    assert key_end(0, 0) == 257 and key_end(0, 2) == 769
    assert page_count(0, 1000, 999) == 3
    assert page_count(0, 768, 767) == 2               # page 2's last MTP row needs token 768
    assert page_count(0, 769, 768) == 3
    assert page_count(0, 768, -1) == 2                 # no MTP rows: the key still covers the next token
    assert page_count(0, 1000, 600) == 2               # a reply snapshot with an MTP backlog: pages below mtp_len
    assert page_count(0, 255, 254) == 0


def test_page_rules_fast():
    # fast rows depend on their whole chunk: the key reaches the end of the chunk holding the page's last row
    assert key_end(384, 0) == 384 and key_end(384, 1) == 768 and key_end(384, 2) == 769
    assert page_count(384, 768, 767) == 2
    assert page_count(385, 768, 767) == 2              # FP8 tag: the same grid
    assert key_end(4096, 0) == 4096
    assert page_count(4096, 8192, 8191) == 31
    assert page_count(4096, 4096, 4095) == 15
    assert page_count(64, 512, 511) == 1               # grid 64: page 1's key needs token 512


def test_tail_bytes():
    ix = _index()
    mr, mp, tr, tp = ROWS
    # 1000 tokens, 3 pages: rows 768..999, pools 192..249; MTP rows 768..998, MTP pools 192..248
    assert ix.tail_bytes(0, 1000, 999, 3) == 232 * mr + 58 * mp + 231 * tr + 57 * tp
    assert ix.tail_bytes(0, 1000, -1, 3) == 232 * mr + 58 * mp
    assert ix.page_bytes == PAGE * (mr + tr) + PAGE // 4 * (mp + tp)


# -- host only: lookup ----------------------------------------------------------------------------------------------
def test_find_longest_fitting_prefix():
    rng = random.Random(1)
    ix = _index()
    base = _toks(rng, 900)
    short = _save(ix, base[:300]).entry
    long_ = _save(ix, base[:700]).entry
    _save(ix, base[:800], mtp=False)                   # longer, but without MTP rows
    fast = _save(ix, base[:768], tag=256).entry
    prompt = base + [1, 2, 3]
    assert ix.find(prompt, 0, True, True) is long_
    assert ix.find(prompt, 0, False, True).ids == base[:800]
    assert ix.find(prompt, 256, True, True) is fast
    assert ix.find(prompt, 128, False, False) is None                 # another grid never resumes
    assert ix.find(base[:700], 0, True, True) is short                 # strict prefixes only
    other = base[:500] + [7] + base[501:]
    assert ix.find(other, 0, True, True) is short
    assert ix.find([5] + base, 0, False, False) is None
    nodraft = _index()
    _save(nodraft, base[:700], drafter=False)
    assert nodraft.find(prompt, 0, True, True) is None and nodraft.find(prompt, 0, True, False) is not None


def test_lcp_and_marks():
    rng = random.Random(2)
    ix = _index()
    system = _toks(rng, 1100)
    a = system + _toks(rng, 300)
    _save(ix, a)
    b = system + _toks(rng, 250)
    assert ix.lcp(b) == 1100
    assert ix.lcp(system[:1099] + [a[1099] + 1]) == 1099
    # the fork point on a page boundary (exact), at least fork_min past the resume point
    assert ix.marks(b, 0, 0, has_mtp=True, drafter=True, every=0, fork_min=256) == [1024]
    assert ix.marks(b, 0, 900, has_mtp=True, drafter=True, every=0, fork_min=256) == []
    # fast: on the grid, and before the prompt's own grid snapshot (1350 // 384 * 384 = 1152)
    assert ix.marks(b, 384, 0, has_mtp=True, drafter=True, every=0, fork_min=256) == [768]
    # every N tokens, rounded to the page / grid
    long_prompt = _toks(rng, 5000)
    assert ix.marks(long_prompt, 0, 0, has_mtp=True, drafter=True, every=1000, fork_min=256) == \
        [768, 1536, 2304, 3072, 3840, 4608]
    assert ix.marks(long_prompt, 0, 1600, has_mtp=True, drafter=True, every=1024, fork_min=256) == \
        [2048, 3072, 4096]
    assert ix.marks(long_prompt, 1024, 0, has_mtp=True, drafter=True, every=2000, fork_min=256) == \
        [1024, 2048, 3072]                              # < 4096 = the prompt's own grid point
    # a mark already stored is not asked again
    _save(ix, b[:1024])
    assert ix.marks(b, 0, 0, has_mtp=True, drafter=True, every=0, fork_min=256) == []
    assert ix.marks(b, 0, 0, has_mtp=False, drafter=True, every=0, fork_min=256) == [1024]


# -- host only: pages, LRU, budget -----------------------------------------------------------------------------------
def test_shared_pages_and_refcounts():
    rng = random.Random(3)
    ix = _index(extent=4)
    system = _toks(rng, 1200)
    a = _save(ix, system + _toks(rng, 300)).entry          # 1500 tokens: 5 pages
    b_res = _save(ix, system + _toks(rng, 400))            # 1600 tokens: 6 pages
    b = b_res.entry
    assert len(a.pages) == 5 and len(b.pages) == 6
    # pages 0-3 are the system prompt's (their keys end at token 1025 <= 1200); page 4's key reaches token 1281
    assert a.pages[:4] == b.pages[:4] and a.pages[4] != b.pages[4]
    assert [p for p, _ in b_res.new] == [4, 5]
    assert len(ix.pages) == 7
    assert [ix.pages[k].refs for k in a.pages] == [2, 2, 2, 2, 1]
    ix.evict(a.id)
    assert len(ix.pages) == 6 and all(ix.pages[k].refs == 1 for k in b.pages)
    ix.evict(b.id)
    assert not ix.pages and not ix.extents and not ix.free and ix.used == 0


def test_page_keys_depend_on_tag_and_mtp():
    rng = random.Random(4)
    ids = _toks(rng, 1100)
    ix = _index()
    e0 = _save(ix, ids).entry
    e1 = _save(ix, ids, mtp=False).entry
    e2 = _save(ix, ids[:1024], tag=512).entry
    assert not set(e0.pages) & set(e1.pages) and not set(e0.pages) & set(e2.pages)
    assert _save(ix, ids).status == DUPLICATE and ix.counters["duplicate"] == 1


def test_lru_eviction_and_budget():
    rng = random.Random(5)
    ix = _index(budget_pages=13, extent=4)                # room for ~2 sessions of 1000 tokens (3 pages each + tail)
    snap = ix.page_bytes                                   # a snapshot the size of a page
    s = [_toks(rng, 1000) for _ in range(4)]
    e = [_save(ix, s[i], snap=snap).entry for i in range(2)]
    assert ix.used <= ix.budget
    ix.touch(e[0])                                         # session 0 used again: session 1 is the LRU one
    r = _save(ix, s[2], snap=snap)
    assert r.status == SAVED and r.evicted == [e[1].id] and ix.used <= ix.budget
    r = _save(ix, s[3], snap=snap)
    assert r.evicted == [e[0].id]
    assert ix.counters["evicted"] == 2 and len(ix.entries) == 2
    # larger than the whole budget: skipped, nothing evicted
    big = _save(ix, _toks(rng, 20 * PAGE), snap=snap)
    assert big.status == SKIPPED and big.evicted == [] and len(ix.entries) == 2
    assert _save(ix, s[0], snap=100 * ix.budget).status == SKIPPED


def test_slabs_reused_and_released():
    rng = random.Random(6)
    ix = _index(budget_pages=64, extent=4)
    es = [_save(ix, _toks(rng, 6 * PAGE + 10)).entry for _ in range(3)]    # 6 pages each: 18 pages, 5 slabs
    assert len(ix.extents) == 5
    ix.evict(es[1].id)                                     # its pages: the middle slabs
    assert sum(ix.extents.values()) == 12
    e = _save(ix, _toks(rng, 6 * PAGE + 10))
    assert len(ix.extents) <= 5                            # the freed pages are used first
    phys = sorted(ix.pages[k].phys for k in e.entry.pages)
    assert phys == sorted(phys) and len(set(phys)) == 6


def test_forced_replay_gives_the_same_store():
    """Rank 1 applies rank 0's decisions: the same entries, pages and digest after every step, over random mixes of
    sessions, forks, duplicates, touches and evictions."""

    rng = random.Random(7)
    a, b = _index(budget_pages=150, extent=4), _index(budget_pages=150, extent=4)
    systems = [_toks(rng, rng.randrange(200, 1500)) for _ in range(3)]
    live: list[list[int]] = []
    for step in range(300):
        r = rng.random()
        if live and r < 0.35:
            ids = rng.choice(live) + _toks(rng, rng.randrange(1, 600))
        elif live and r < 0.45:
            ids = list(rng.choice(live))                   # a duplicate
        else:
            ids = rng.choice(systems) + _toks(rng, rng.randrange(1, 900))
        tag = rng.choice([0, 0, 0, 256])
        if tag:
            ids = ids[:len(ids) // 256 * 256] or ids[:256] + _toks(rng, 256 - len(ids[:256]))
        mtp, drafter = rng.random() < 0.9, rng.random() < 0.8
        snap = rng.randrange(1, 3) * a.page_bytes
        ra = _save(a, ids, tag=tag, mtp=mtp, drafter=drafter, snap=snap)
        rb = _save(b, ids, tag=tag, mtp=mtp, drafter=drafter, snap=snap, forced=(ra.status, ra.evicted))
        assert rb.status == ra.status and rb.evicted == ra.evicted
        assert a.digest() == b.digest() and a.used <= a.budget, step
        live.append(ids)
        if rng.random() < 0.2 and a.entries:
            i = rng.choice(sorted(a.entries))
            a.touch(a.entries[i])
            b.touch(b.entries[i])
    assert a.counters == b.counters and a.counters["evicted"] > 10 and a.counters["duplicate"] > 3
    # a divergence is detected, not followed
    c = _index(budget_pages=40, extent=4)
    ids = _toks(rng, 700)
    with pytest.raises(RuntimeError, match="diverged"):
        _save(c, ids, forced=(SAVED, [12345]))
    with pytest.raises(RuntimeError, match="diverged"):
        _save(c, ids, forced=(DUPLICATE, []))


def test_plan_message_and_digest_check():
    rng = random.Random(8)
    store = object.__new__(sessions.SessionStore)
    store.index = _index()
    e = _save(store.index, _toks(rng, 900)).entry
    msg = sessions.Plan(e, [256, 512]).encode(store.index.digest())
    plan = store.follow(msg)
    assert plan.entry is e and plan.marks == [256, 512]
    assert store.follow(sessions.Plan(None, []).encode(store.index.digest())).entry is None
    with pytest.raises(RuntimeError, match="diverged"):
        store.follow(sessions.Plan(e, []).encode(store.index.digest() ^ 1))
    assert sessions.decode_saves([1, 2, 5, 6, 2, 0, 0, 1, 9], 3) == [(1, [5, 6]), (2, []), (0, [9])]
    with pytest.raises(RuntimeError):
        sessions.decode_saves([1, 0, 1], 1)


def test_settings(monkeypatch):
    monkeypatch.delenv("GLM53_TF_SESSION_GIB", raising=False)
    assert sessions.budget_bytes() == 0                     # off unless asked for
    monkeypatch.setenv("GLM53_TF_SESSION_GIB", "12")
    monkeypatch.setenv("GLM53_TF_SESSION_EVERY", "0")
    assert sessions.settings() == [12 * 1024, 0, 512]
    monkeypatch.setenv("GLM53_TF_SESSION_GIB", "-1")
    with pytest.raises(ValueError):
        sessions.budget_bytes()


# -- torch on any device: the tensor store against a model whose rows are functions of their prefix ----------------
def _row(ids, n: int) -> float:
    """A cache row's value: a function of the tokens it depends on (``n`` of them)."""

    return float(hash(tuple(ids[:n])) % 100_003)


class _FakeState:
    """A ``forward.State``'s caches, small: 2 DSA layers (latent: keys = values) with index rings, the MTP layer's
    full index caches, pool keys, the MTP head's latent cache."""

    def __init__(self, torch, cap: int = 2048) -> None:
        z = lambda n, w=3: torch.zeros((n, w))
        self.capacity = cap
        self.dsa_index = {10: 0, 20: 1}
        self.kc = [z(cap), z(cap)]
        self.vc = self.kc
        self.index = [(z(256, 2), z(256, 2), z(cap // 4 + 2, 2)), (z(256, 2), z(256, 2), z(cap // 4 + 2, 2)),
                      (z(cap, 2), z(cap, 2), z(cap // 4 + 2, 2))]
        self.mtp_kc = self.mtp_vc = z(cap)

    def write(self, ids, lo: int, hi: int, mtp_lo: int, mtp_hi: int, torch, junk: int = 0) -> None:
        """What a request does: rows [lo, hi) and MTP rows [mtp_lo, mtp_hi) of ``ids``, complete pools, then junk
        rows past them (rejected drafts, stale windows)."""

        for t in range(lo, hi + junk):
            v = _row(ids, t + 1) if t < hi else -1.0 - t
            for x in self.kc:
                x[t] = v
            for _, _, pk in self.index[:2]:
                if t % 4 == 3:
                    pk[t // 4] = v if t < hi else -7.0
        for t in range(mtp_lo, mtp_hi + junk):
            v = _row(ids, t + 2) if t < mtp_hi else -3.0 - t
            self.mtp_kc[t] = v
            ik, ig, pk = self.index[2]
            ik[t] = ig[t] = v
            if t % 4 == 3:
                pk[t // 4] = v if t < mtp_hi else -5.0

    def check(self, ids, n: int, mtp_len: int) -> None:
        for t in range(n):
            assert all(float(x[t, 0]) == _row(ids, t + 1) for x in self.kc), t
            if t % 4 == 3:
                assert all(float(pk[t // 4, 0]) == _row(ids, t + 1) for _, _, pk in self.index[:2]), t
        for t in range(max(mtp_len, 0)):
            assert float(self.mtp_kc[t, 0]) == _row(ids, t + 2) and float(self.index[2][0][t, 0]) == _row(ids, t + 2)
            if t % 4 == 3:
                assert float(self.index[2][2][t // 4, 0]) == _row(ids, t + 2), t


class _FakeDrafter:
    """The DFlash2 drafter's context ring: kc/vc [KV, cap, hd], a window of 100 positions, blocks of 8."""

    def __init__(self, torch, cap: int = 128) -> None:
        self.kc = [torch.zeros((2, cap, 4))]
        self.vc = [torch.zeros((2, cap, 4))]
        self.cap, self.window, self.block = cap, 100, 8
        self._end = self._hi = self._lo = 0
        self.lo_dev = torch.zeros((1,), dtype=torch.int64)
        self.pos_dev = torch.zeros((1,), dtype=torch.int64)

    @property
    def context_end(self) -> int:
        return self._end

    def extend(self, ids, lo: int, hi: int) -> None:
        for t in range(lo, hi):
            self.kc[0][:, t % self.cap] = _row(ids, t + 1)
            self.vc[0][:, t % self.cap] = -_row(ids, t + 1)
        self._end = hi
        self._hi = max(self._hi, hi + self.block)

    def check(self, ids) -> None:
        end = self._end
        for t in range(max(0, end - self.window), end):
            assert float(self.kc[0][0, t % self.cap, 0]) == _row(ids, t + 1), t
            assert float(self.vc[0][1, t % self.cap, 0]) == -_row(ids, t + 1), t
        assert int(self.pos_dev) == end and int(self.lo_dev) == max(0, end - self.window)


def test_store_round_trips_rows_on_any_device():
    """Sessions saved from and restored into fake live caches (random forks, follow-ups, fresh prompts, junk past
    every request's rows, evictions): after every restore the live caches hold exactly the entry's rows, whatever
    the live caches held before (pages skipped only where they already hold them)."""

    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    rng = random.Random(9)
    st, dr = _FakeState(torch), _FakeDrafter(torch)
    e = SimpleNamespace(st=st, w=SimpleNamespace(mtp=object()))
    store = sessions.SessionStore(e, dr, 0, every_tokens=0, fork_tokens=64)
    ix = store.index
    ix.budget = 40 * ix.page_bytes
    systems = [_toks(rng, rng.randrange(300, 900)) for _ in range(3)]
    for step in range(120):
        r = rng.random()
        if ix.entries and r < 0.5:
            base = rng.choice(sorted(ix.entries))
            ids = ix.entries[base].ids + _toks(rng, rng.randrange(1, 400))
        elif r < 0.8:
            ids = rng.choice(systems) + _toks(rng, rng.randrange(1, 500))
        else:
            ids = _toks(rng, rng.randrange(10, 1200))
        ids = ids[:1900]
        best = ix.find(ids, 0, True, True)
        if best is not None:
            before = dict(store.stats)
            snap = store.restore(best, dr)
            n = len(snap.ids)
            st.check(snap.ids, n, snap.mtp_len)
            dr.check(snap.ids)
            assert store.stats["copied"] - before["copied"] <= len(best.pages)
            cut, k = n, 1
        else:
            snap, cut, k = None, 0, 0
        store.begin(snap, cut)
        # the request: prompt rows from the resume point, MTP rows from its pending row, junk past both
        mtp_len = len(ids) - 1
        st.write(ids, cut, len(ids), max(cut - k, 0), mtp_len, torch, junk=rng.randrange(0, 9))
        if best is None:                                  # a fresh prefill resets the drafter
            dr._end = dr._hi = dr._lo = 0
            dr.lo_dev.zero_()
        dr.extend(ids, dr.context_end, len(ids))
        snap = SimpleNamespace(ids=list(ids), rec=torch.full((5,), float(len(ids))), conv=torch.zeros(3),
                               pending=torch.zeros((1, 2)), mtp_len=mtp_len, drafter_end=len(ids), grid=0,
                               window=sessions.capture_window(dr))
        store.save(snap)
        assert ix.used <= ix.budget
    assert ix.counters["evicted"] > 0 and store.stats["restores"] > 30
    assert store.stats["skipped"] > 0 and store.stats["copied"] > 0


# -- torch on any device: the engine's real request path with the store, on patches/0080's hostile fake model -------
def _fake_session_engine(monkeypatch, rows: int, fast: bool, budget_pages: float | None):
    """``test_fastpf_patches``' fake engine (chunk-dependent fast rows, row p reading a digest of every KV position
    before it) with the real ``GlmEngine._run`` / ``_store_save`` and, when ``budget_pages``, a session store over
    its KV cache, its prefix digests and its MTP cache."""

    import types

    import torch
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine
    from test_fastpf_patches import _Fake, _fake_engine

    g = _fake_engine(monkeypatch, rows, fast)
    f: _Fake = g.fake
    f.dig = torch.zeros(f.kv.shape[0] + 1, dtype=torch.int64)       # a tensor, so the store can copy it
    st = g.e.st
    st.capacity = f.kv.shape[0]
    st.kc = [f.kv, f.dig[1:]]                  # position t: kv[t] and the digest of kv[0:t + 1]
    st.vc = st.kc
    st.index, st.dsa_index = None, {}
    st.mtp_kc = st.mtp_vc = f.mkv
    g.rank, g.store = 0, None
    g._share = lambda values: list(values)
    g._store_save = types.MethodType(GlmEngine._store_save, g)
    if budget_pages is not None:
        g.store = sessions.SessionStore(g.e, None, 0, every_tokens=512, fork_tokens=64)
        g.store.index.budget = int(budget_pages * g.store.index.page_bytes)
    return g


def _fake_turn(g, prompt, policy: str, tokens: int = 30):
    from tensorfold.families.glm5_next.cuda.engine import encode_policy

    code = encode_policy(policy)
    hit = g._resume(list(prompt), code)
    sess = None
    if g.store is not None:                    # as ``GlmEngine.generate`` does on rank 0
        _, need_mtp, _ = g._drafters(code)
        sess = g.store.plan(list(prompt), g._grid(), need_mtp, False, len(hit.ids) if hit else 0, True)
        if sess.entry is not None:
            hit = sess.entry.payload[0]
    out: list[int] = []
    stats = g._run(list(prompt), tokens, None, False, out.extend, code, hit, True, sess)
    st, f = g.e.st, g.fake
    state = (st.pos, int(st.rec[st.cur[0], 0]), st.conv.tolist(), f.kv[:st.pos].tolist(), st.mtp_len,
             f.mkv[:st.mtp_len].tolist())
    return out, stats["cached"], state


# tight: 36 pages (40 until patches/0540, whose exact prompt snapshots end up to 64 tokens earlier and so are a little
# smaller: one seed stopped evicting at 40); every case evicts with GLM53_TF_SNAPSHOT_BEFORE_END=1 and =0
@pytest.mark.parametrize("budget", [400.0, 36.0], ids=["roomy", "tight"])
@pytest.mark.parametrize("policy", ["0", "2"])
@pytest.mark.parametrize("rows", [64, 200])
def test_sessions_on_a_hostile_fake_model(monkeypatch, rows, policy, budget):
    """Four conversations over three system prompts, interleaved at random (follow-ups after the reply or the
    prompt, forks of a system prompt, cuts), exact and fast: every request gives the reply and the whole state of a
    fresh prefill of its prompt, and most resume from the store."""

    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    from tensorfold.families.glm5_next.cuda import decode, fastpf
    from test_fastpf_patches import _fake_engine

    def use(eng):
        for name in ("stage", "compute", "commit"):
            monkeypatch.setattr(decode, name, getattr(eng.fake, name))

    rng = np.random.default_rng(rows + len(policy))
    more = lambda n: [int(t) for t in rng.integers(0, 1000, size=n)]
    C = fastpf.grid(rows)
    for fast in (False, True):
        g = _fake_session_engine(monkeypatch, rows, fast, budget)
        ref = _fake_engine(monkeypatch, rows, fast)
        systems = [more(int(n)) for n in (300, 700, 1100)]
        convs: list[tuple[list[int], list[int]]] = []
        from_store = 0
        for turn in range(40):
            k = int(rng.integers(0, 6))
            if convs and k < 3:
                i = int(rng.integers(0, len(convs)))
                last, reply = convs[i]
                prompt = (last + reply if k < 2 else last) + more(int(rng.choice([1, 5, 40, C + 3])))
                convs[i] = (prompt, [])
            elif convs and k == 3:
                last, _ = convs[int(rng.integers(0, len(convs)))]
                prompt = last[:int(rng.integers(1, len(last) + 1))] + more(7)
            else:
                prompt = systems[int(rng.integers(0, 3))] + more(int(rng.integers(1, 300)))
                convs.append((prompt, []))
                i = len(convs) - 1
            prompt = prompt[:2600]
            use(g)
            restores = g.store.stats["restores"]
            got, cached, state = _fake_turn(g, prompt, policy)
            from_store += g.store.stats["restores"] > restores
            if fast:
                assert cached % C == 0
            use(ref)
            ref.cache = []
            want, c0, want_state = _fake_request_ref(ref, prompt, policy)
            assert c0 == 0 and got == want and state == want_state, (fast, turn, len(prompt), cached)
            if k < 3 or k > 3:
                convs[i] = (prompt, got)
        ix = g.store.index
        assert from_store >= 8, from_store
        assert ix.used <= ix.budget
        if budget < 100:
            assert ix.counters["evicted"] > 0


def _fake_request_ref(ref, prompt, policy):
    from test_fastpf_patches import _fake_request

    return _fake_request(ref, prompt, policy=policy, tokens=30)


# -- GPU: engines -----------------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_lookup(monkeypatch):
    """patches/0020's lookup rounds make the windows depend on repeats; pin the MTP/DFlash2 arms."""

    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")


def _engine(path, *, gib: float = 1.0, fast: bool = False, rows: int = 64, context: int = 0,
            latent_kv: bool = False, fork: int = 256, every: int = 0, nonexpert: str | None = None):
    from tensorfold.families.glm5_next.cuda import weights
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine
    from test_glm_engine import _TwoCopies

    with pytest.MonkeyPatch.context() as m:
        if nonexpert:
            m.setattr(weights, "NONEXPERT", nonexpert)
            m.setenv("GLM53_TF_NONEXPERT", nonexpert)
        m.setenv("GLM53_TF_SESSION_GIB", str(gib))
        m.setenv("GLM53_TF_SESSION_EVERY", str(every))
        m.setenv("GLM53_TF_SESSION_FORK_MIN", str(fork))
        m.setenv("GLM53_TF_SESSION_RESERVE_GIB", "0")
        m.setenv("GLM53_TF_FAST_PREFILL", "1" if fast else "0")
        m.setenv("GLM53_TF_PREFILL_ROWS", str(rows))
        m.setenv("GLM53_TF_PREFILL_ROWS_MAX", str(max(rows, 256)))
        # patches/0085: snapshots on the chunk grid (patches/0080's rule, which these tests check)
        m.setenv("GLM53_TF_SNAPSHOT_GRID", str(max(64, rows // 64 * 64)))
        m.setenv("GLM53_TF_LATENT_KV", "1" if latent_kv else "0")
        m.setenv("GLM53_TF_LOOKUP", "0")
        for k in ("GLM53_TF_BATCH", "GLM53_TF_CALIB_ONLINE", "GLM53_TF_PROFILE", "GLM53_TF_DEPTH",
                  "GLM53_TF_FAST_GATHER", "GLM53_TF_FP8_PREFILL", "GLM53_TF_LEAN_PREFILL"):
            m.delenv(k, raising=False)
        return GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", context=context,
                         comm=_TwoCopies())


def _gen(eng, prompt, sampling, *, draft=True, policy=None, tokens=24):
    out: list[int] = []
    eng.request.policy = policy
    eng.request.stop_eos = False
    eng.request.knobs = None
    stats = eng.generate(list(prompt), tokens, sampling, lambda new: out.extend(new), draft=draft)
    return out, stats


def _fresh(ref, prompt, sampling, tokens=24):
    """The reference: a fresh prefill and serial decoding on an engine without the store."""

    ref.cache = []
    out, stats = _gen(ref, prompt, sampling, draft=False, tokens=tokens)
    assert stats["cached"] == 0
    return out


def _sampling(kind: str):
    from tensorfold.engine.exact_sampling import Sampling

    return None if kind == "greedy" else Sampling(1234, 1.0, 20, 0.95)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_sessions")
    _checkpoint(path / "model")
    _drafter(path / "dflash2")
    return path


@pytest.fixture(scope="module")
def es(ckpt):
    return _engine(ckpt)


@pytest.fixture(scope="module")
def ref(ckpt):
    return _engine(ckpt, gib=0)


def _sessions(seed: int, system: int = 600, own=(150, 170, 130)):
    import numpy as np

    rng = np.random.default_rng(seed)
    sys_prompt = [int(t) for t in rng.integers(0, 1000, size=system)]
    return [sys_prompt + [int(t) for t in rng.integers(0, 1000, size=n)] for n in own]


def _interleave(eng, ref, sampling, prompts, *, policy="auto:1:1:0", tokens=24, check=None):
    """A1 B1 A2 C1 B2 A3: each reply against the fresh reference; returns the stats in order."""

    a, b, c = prompts
    turns: dict[str, list[int]] = {"A": list(a), "B": list(b), "C": list(c)}
    order = ["A", "B", "A", "C", "B", "A"]
    seen: set[str] = set()
    out = []
    for i, name in enumerate(order):
        prompt = turns[name]
        reply, stats = _gen(eng, prompt, sampling, policy=policy, tokens=tokens)
        assert reply == _fresh(ref, prompt, sampling, tokens), (i, name, stats)
        out.append((name, name in seen, len(prompt), stats))
        if check is not None:
            check(i, name, stats)
        seen.add(name)
        turns[name] = prompt + reply + [3 + i, 17 + i]
    return out


@gpu
@pytest.mark.parametrize("sampling", ["sampled", "greedy"])
def test_interleaved_sessions_equal_fresh(es, ref, sampling):
    s = _sampling(sampling)
    es.cache = []
    before = dict(es.store.stats)
    runs = _interleave(es, ref, s, _sessions(11 if sampling == "sampled" else 12))
    (_, _, la, a1), (_, _, _, b1), (_, _, _, a2), (_, _, _, c1), (_, _, _, b2), (_, _, _, a3) = runs
    assert a1["cached"] == 0 and b1["cached"] == 0         # B1's prefill takes a mark at 512 (the fork point)
    assert a2["cached"] >= la + 24 - 1                     # A's reply snapshot, from the store (B ran in between)
    assert c1["cached"] == 512                             # the fork mark B1's prefill took at the system prompt
    assert b2["cached"] >= runs[1][2] + 24 - 1
    assert a3["cached"] >= runs[2][2] + 24 - 1
    assert es.store.stats["restores"] - before["restores"] >= 4
    # the system prompt's pages are stored once, shared by every session's entries
    ix = es.store.index
    assert sum(len(e.pages) for e in ix.entries.values()) > len(ix.pages)
    first = {e.pages[0] for e in ix.entries.values() if e.pages}
    assert any(ix.pages[k].refs >= 3 for k in first)


@gpu
def test_restore_skips_pages_the_live_caches_hold(es, ref):
    """C resumes from the fork mark while the live caches hold A: the shared system prompt page is not copied."""

    s = _sampling("greedy")
    a, b, c = _sessions(21)
    _gen(es, a, s)
    _gen(es, b, s)                              # B's prefill takes the mark at 512
    _gen(es, a + [9], s)                        # the live caches hold A again (from the live snapshot)
    copied, skipped = es.store.stats["copied"], es.store.stats["skipped"]
    reply, stats = _gen(es, c, s)
    assert stats["cached"] == 512 and reply == _fresh(ref, c, s)
    assert es.store.stats["copied"] == copied and es.store.stats["skipped"] == skipped + 1


@gpu
def test_eviction_under_a_tiny_budget_stays_exact(ckpt, ref):
    s = _sampling("sampled")
    e = _engine(ckpt)
    prompts = _sessions(31)
    _gen(e, prompts[0], s)
    ix = e.store.index
    biggest = max(x.private for x in ix.entries.values())
    # one slab of pages and about one entry's snapshots: with 0085's smaller entries, two of each no longer forced an
    # eviction during one interleave (measured: 46 evictions, every reply == fresh, at this budget)
    ix.budget = biggest + ix.extent_bytes
    _interleave(e, ref, s, _sessions(32))
    assert ix.counters["saved"] > ix.counters["evicted"] > 0 and ix.used <= ix.budget
    # smaller than one snapshot: nothing is stored, the live snapshots still resume
    ix.budget = biggest // 4
    for k in list(ix.entries):
        ix.evict(k)
    e.store._sync_slabs()
    p = _sessions(33)[0]
    reply, _ = _gen(e, p, s)
    assert not ix.entries and ix.counters["skipped"] > 0
    after = p + reply + [5]
    warm, stats = _gen(e, after, s)
    assert stats["cached"] >= len(p) + len(reply) - 1 and warm == _fresh(ref, after, s)


@gpu
def test_policies_without_a_drafter_window_or_mtp_rows(es, ref):
    """A session saved by one drafter's policy resumes a request of another only when the caches fit; replies
    stay exact either way."""

    s = _sampling("greedy")
    a, b, _ = _sessions(41)
    for first, then in (("f3", "2"), ("2", "f3"), ("auto:1:1:0", "f3"), ("0", "2")):
        es.cache = []
        reply, _ = _gen(es, a, s, policy=first)
        _gen(es, b, s, policy=first)
        after = a + reply + [4]
        warm, stats = _gen(es, after, s, policy=then)
        assert warm == _fresh(ref, after, s), (first, then, stats)


# -- GPU: past 2,051 tokens, latent cache, exact and fast prefill ----------------------------------------------------
@pytest.fixture(scope="module")
def long_ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter
    from test_patches import _index_heads_32

    path = tmp_path_factory.mktemp("glm_sessions_long")
    _checkpoint(path / "model", exl3=True)
    _index_heads_32(path / "model")
    _drafter(path / "dflash2")
    return path


@gpu
@pytest.mark.parametrize("fast", [False, True], ids=["exact", "fast"])
@pytest.mark.parametrize("sampling", ["sampled", "greedy"])
def test_long_context_latent_sessions(long_ckpt, fast, sampling):
    """A 2,300-token system prompt (past the dense limit) and three sessions on the latent cache with the index
    ring; fast prefill on a 256 grid (snapshots and marks on the grid). Every reply equals the fresh reference."""

    s = _sampling(sampling)
    kw = {"fast": fast, "rows": 256, "context": 4096, "latent_kv": True, "nonexpert": "q4mse"}
    e = _engine(long_ckpt, **kw)
    r = _engine(long_ckpt, gib=0, **kw)
    prompts = _sessions(51, system=2300, own=(200, 220, 180))
    policy = "2" if fast else "auto:1:1:0"

    def check(i, name, stats):
        if i == 3:                              # C1: from B1's fork mark (2,300 shared tokens: page / grid point 2,048)
            assert stats["cached"] == 2048
        if i == 2 and fast:                     # A2: from A1's grid snapshot, 2,304 (no reply snapshot when fast)
            assert stats["cached"] == 2304

    _interleave(e, r, s, prompts, policy=policy, tokens=16, check=check)
    if fast:
        assert all(x.tag == 256 for x in e.store.index.entries.values())


# -- GPU: rank 1 follows --------------------------------------------------------------------------------------------
class _Done(Exception):
    pass


@gpu
def test_rank1_follows_rank0(ckpt):
    """Rank 0's messages replayed to a second engine acting as rank 1: the same restores, marks, saves and
    evictions (the digest check passes every request), the same replies; a diverged store is refused."""

    s = _sampling("sampled")
    lead = _engine(ckpt)
    follower = _engine(ckpt)
    sent: list[list[int]] = []
    share = lead._share

    def record(values):
        got = share(values)
        sent.append(list(got))
        return got

    def replay(values):
        assert values is None
        if not sent:
            raise _Done
        return sent.pop(0)

    got: list = []
    run = follower._run

    def capture(*args, **kw):
        stats = run(*args, **kw)
        got.append(stats.get("sha256"))
        return stats

    lead._share = record
    follower.rank = 1
    follower._share = replay
    follower._run = capture
    shas = []
    prompts = _sessions(61)
    turns = {n: list(p) for n, p in zip("ABC", prompts)}
    reply, stats = _gen(lead, turns["A"], s, policy="auto:1:1:0")
    shas.append(stats.get("sha256"))
    turns["A"] += reply + [99]
    with pytest.raises(_Done):
        follower.follow()
    ix, fx = lead.store.index, follower.store.index
    assert fx.digest() == ix.digest()
    # evictions (W15: one extent of shared pages, was two: with patches/0540 a resend's snapshot is the prompt's own
    # (n - 64), so 6 of 19 saves are duplicates, the entries share one extent and 4 entries + 2 extents never evicted)
    ix.budget = fx.budget = 4 * max(x.private for x in ix.entries.values()) + 1 * ix.extent_bytes
    for i, name in enumerate("BACBAACB"):
        reply, stats = _gen(lead, turns[name], s, policy="auto:1:1:0")
        shas.append(stats.get("sha256"))
        turns[name] = turns[name] + reply + [i]
    with pytest.raises(_Done):
        follower.follow()
    print("rank1_follows_rank0:", dict(ix.counters), dict(lead.store.stats))
    assert got == shas
    assert ix.counters["evicted"] > 0 and lead.store.stats["restores"] > 0, (dict(ix.counters), dict(lead.store.stats))
    assert fx.digest() == ix.digest() and sorted(fx.entries) == sorted(ix.entries)
    assert fx.counters == ix.counters and follower.store.stats == lead.store.stats

    # a store that diverged: the next request's plan is refused
    _gen(lead, turns["A"], s, policy="auto:1:1:0")
    fx.evict(next(iter(fx.entries)))
    with pytest.raises(RuntimeError, match="diverged"):
        follower.follow()


@gpu
def test_spans_cut_chunks_at_marks():
    from tensorfold.families.glm5_next.cuda.decode import _spans

    assert list(_spans(0, 300, 128, [])) == [(0, 128), (128, 256), (256, 300)]
    assert list(_spans(0, 300, 128, [100, 256])) == [(0, 100), (100, 228), (228, 256), (256, 300)]
    assert list(_spans(64, 200, 64, [128])) == [(64, 128), (128, 192), (192, 200)]
