"""patches/0560 (GLM53_TF_MULTI_PREFILL, ``glm5_next/cuda/mpf.py``): several slots' prefill pieces in ONE forward (rows
stacked, per-slot KDA / DSA / carries / commits / snapshots / heads, the routed experts once over every row). Each
member's result must be byte-identical to its piece prefilled alone. docs/MULTI-PREFILL.md.

Host only (no torch): the knob and settings, ``more_pieces`` (fewest left first, within the row budget, only when the
fair share allows a piece), ``pack`` (same knobs, row budget, round order, ineligible alone), the sub-blocks (never two
members, cut from each member's start), the solo-rule guard.

With torch on the CPU, test_lean_patches' HASH MODEL (every kernel exact integer arithmetic with the real row /
position / KDA-state / cache / routing dependencies; the real 0082 / 0084 / 0320 orchestration):

- a group forward == each member's lone lean chunk, bit for bit, per member: last-row logits, final-normed rows,
  DFlash2 taps, residual streams, KDA states and conv windows (``lean.commit`` with the member's carry), attention
  caches; members of 1-600 rows (partial and several sub-blocks each), resumed at 0 / 64 / 128 / 192, 2-5 members;
  0082's loop and every 0084 variant (direct, gather, slab, gather + slab); EXL3 and MLX MoE;
- the routed experts run once per group forward (not once a member), the per-slot kernels once a member sub-block;
- controls: one shared conv carry for all members, members' positions taken from the group's rows, a sub-block
  straddling two members: each is caught;
- TWO REAL PROCESSES over gloo (test_prefill_pp_patches' communicator and rank-specific weights): with 0320's row split
  on (a group of more than one sub-block's rows), each rank's group == the members' lone chunks (which split only when
  alone they exceed a sub-block), and the ranks agree on the replicated rows.

With torch on the CPU, test_batch_sessions_patches' HOSTILE FAKE MODEL through the real ``Batcher``
(``_plan`` / ``_execute`` / ``_admit`` / ``_piece`` / ``_finish`` / ``follow``) with the real session store, the group's
forward and commit replaced by the fake's per-slot rows (whose bits depend on the chunk's start and length):

- RigMark C4: 4 short prompts queued together are admitted in one round and prefilled in ONE group, fewest tokens
  first; each first token out as its piece ends; replies and every slot's final state == the request alone; members
  whose prompt needs several chunks (64-row chunks: 2-3 lockstep forwards);
- snapshots: the prompt snapshots the group leaves (0540's strictly-before-the-end rule, marks) equal the lone
  ones; a resend resumes from them (cached n - 64) and == fresh; 4 sessions of interleaved turns with multi prefill
  (every reply and final state == fresh + serial; resumes across slots; tiny store budget with evictions);
- rank 1: a follower replaying rank 0's messages forms the same groups (same trace, replies, stores, states);
- admission: different knobs never share a group; image prompts run alone; a member's own error (a snapshot that
  cannot resume) fails that request only; decoders in the same round verify after the group; ``more_pieces`` gives
  prompts past BATCH_SHORT shared forwards; the idle wait (``coalesce``) and its bound.

Run inside the image:
    PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q tests/cuda/test_multi_prefill_patches.py
Host part without the image: PYTHONPATH=<patched tree>/src:<patched tree>/tests/cuda:tests/cuda pytest -q ...
"""

from __future__ import annotations

import os
import tempfile
import traceback

import pytest

from tensorfold.families.glm5_next.cuda import mpf

try:
    import torch
except ImportError:          # the host-only tests still run
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="torch")


# -- host only ------------------------------------------------------------------------------------------------------
def test_parse_and_settings(monkeypatch):
    assert mpf.parse("0") is False and mpf.parse("") is False and mpf.parse(" off ") is False
    assert mpf.parse("1") is True and mpf.parse("on") is True
    with pytest.raises(ValueError, match=mpf.ENV):
        mpf.parse("2")
    monkeypatch.setenv(mpf.ENV, "1")
    assert mpf.parse() is True
    monkeypatch.delenv(mpf.ROWS_ENV, raising=False)
    assert mpf.limit(4096) == 4096
    monkeypatch.setenv(mpf.ROWS_ENV, "1024")
    assert mpf.limit(4096) == 1024 and mpf.limit(512) == 512
    for bad in ("-1", "32"):
        monkeypatch.setenv(mpf.ROWS_ENV, bad)
        with pytest.raises(ValueError, match=mpf.ROWS_ENV):
            mpf.limit(4096)
    monkeypatch.delenv(mpf.ROWS_ENV)
    monkeypatch.setattr(mpf, "ON", False)
    assert mpf.settings(4096) == [0, 0]
    monkeypatch.setattr(mpf, "ON", True)
    assert mpf.settings(4096) == [1, 4096]
    monkeypatch.delenv(mpf.WAIT_ENV, raising=False)
    assert mpf.wait_ms() == 10.0
    monkeypatch.setenv(mpf.WAIT_ENV, "0")
    assert mpf.wait_ms() == 0.0


def test_more_pieces():
    pre = [(0, 3000, 1), (1, 130, 2), (2, 900, 3), (3, 5000, 4)]
    # not allowed: nothing more than pick_pieces' own
    assert mpf.more_pieces([1], pre, False, 2048, 4096) == [1]
    # allowed: fewest left first, pieces of at most 2048 rows, within 4096
    assert mpf.more_pieces([1], pre, True, 2048, 4096) == [1, 2, 0]           # 130 + 900 + 2048 = 3078
    assert mpf.more_pieces([], pre, True, 2048, 4096) == [1, 2, 0]
    assert mpf.more_pieces([], pre, True, 2048, 1000) == [1]                  # 900 more would not fit
    assert mpf.more_pieces([1, 2], pre, True, 2048, 0) == [1, 2]              # off


def test_pack():
    A, B = ("a",), ("b",)
    assert mpf.pack([(0, A, 130), (1, A, 128), (2, A, 134), (3, A, 133)], 4096) == [[0, 1, 2, 3]]
    assert mpf.pack([(0, A, 130), (1, B, 128), (2, A, 134), (3, None, 5)], 4096) == [[0, 2], [1], [3]]
    assert mpf.pack([(0, A, 2048), (1, A, 2048), (2, A, 100)], 4096) == [[0, 1], [2]]
    assert mpf.pack([(0, A, 5000), (1, A, 64)], 4096) == [[0], [1]]          # a piece past the budget alone
    assert mpf.pack([(0, A, 64)], 4096) == [[0]]


def test_sub_blocks_never_straddle_members():
    from types import SimpleNamespace as NS

    segs = [NS(off=0, R=130), NS(off=130, R=64), NS(off=194, R=1), NS(off=195, R=300)]
    bl = mpf.blocks(segs, 128)
    assert bl == [(0, 128, 0), (128, 2, 0), (130, 64, 1), (194, 1, 2), (195, 128, 3), (323, 128, 3), (451, 44, 3)]
    for a, r, i in bl:
        s = segs[i]
        assert s.off <= a and a + r <= s.off + s.R and (a - s.off) % 128 == 0      # the member's lone sub-blocks


def test_piece_bounds_solo_guard():
    from types import SimpleNamespace as NS

    e = NS(prefill_rows=64, snap_grid=64, fast_prefill=True)
    seqs = [NS(job=NS(prompt=list(range(n))), done=0) for n in (130, 3000, 5000)] + [None]
    bat = NS(g=NS(e=e), seqs=seqs, solo=4096, piece=2048)
    assert mpf.piece_bounds(bat, 0) == (130, False)                  # ends the prompt either way
    assert mpf.piece_bounds(bat, 1) == (2048, True)                  # alone it would be one solo piece of 3000
    bat.solo = 1024                                                   # a solo piece smaller than the batch piece
    assert mpf.piece_bounds(bat, 0) == (130, False) and mpf.piece_bounds(bat, 1) == (2048, True)
    bat.solo = 0
    assert mpf.piece_bounds(bat, 1) == (2048, False)


# -- the hash model: group forward == lone chunks --------------------------------------------------------------------
def _hash():
    import test_lean_patches as tl

    return tl


def _fresh(w, block, cap, pos0, seed):
    from tensorfold.families.glm5_next.cuda import forward

    tl = _hash()
    g = torch.Generator().manual_seed(seed)
    st = forward.State(w, cap, block)
    st.rec.copy_(torch.randint(0, tl.M, st.rec.shape, generator=g).float())
    st.conv.copy_(torch.randint(0, tl.M, st.conv.shape, generator=g).to(st.conv.dtype))
    for kc in st.kc:
        kc[:pos0, 0, 0] = torch.randint(0, tl.M, (pos0,), generator=g).to(kc.dtype)
    st.set_pos(pos0)
    return st


def _ids(R, seed):
    return torch.randint(0, 96, (R,), generator=torch.Generator().manual_seed(1000 + seed)).int()


def _outs(st, pos0, R, last, fn, taps, x):
    return [last, fn, taps, x, st.conv.clone(), st.rec[st.cur[0]].clone()] + [kc[:pos0 + R].clone() for kc in st.kc]


def _lone(w, sizes, pos, block, rows=1024, cap=2048):
    """Each member alone: its own lean chunk (``lean.compute`` + ``lean.commit``)."""

    from tensorfold.families.glm5_next.cuda import forward, lean

    out = []
    for i, (R, p0) in enumerate(zip(sizes, pos)):
        b = forward.Buffers(w, block, cap)
        b.set_taps((1, 3), w.cfg.hidden)
        lb = lean.LeanBuffers(w, rows, block, taps=2)
        st = _fresh(w, block, cap, p0, i)
        lb.ids[:R].copy_(_ids(R, i))
        last = lean.compute(w, st, b, lb, R, head=True).clone()
        fn, taps, x = lb.fnormed[:R].clone(), lb.tapcat[:R].clone(), lb.x[:R].clone()
        lean.commit(w, st, lb, R)
        out.append(_outs(st, p0, R, last, fn, taps, x))
    return out


def _group(w, sizes, pos, block, rows=1024, cap=2048, order=None):
    """The members in one group forward (``mpf.compute``), each committed with its own carry."""

    from tensorfold.families.glm5_next.cuda import forward, lean

    b = forward.Buffers(w, block, cap)
    b.set_taps((1, 3), w.cfg.hidden)
    lb = lean.LeanBuffers(w, rows, block, taps=2)
    order = list(range(len(sizes))) if order is None else order
    sts = {i: _fresh(w, block, cap, pos[i], i) for i in order}
    tails = mpf._tails(lb, len(order))
    segs, off = [], 0
    for j, i in enumerate(order):
        segs.append(mpf.Seg(sts[i], off, sizes[i], pos[i], tails[j], lb))
        lb.ids[off:off + sizes[i]].copy_(_ids(sizes[i], i))
        off += sizes[i]
    logits = mpf.compute(w, b, lb, segs).clone()
    res = {}
    for j, (i, s) in enumerate(zip(order, segs)):
        fn, taps, x = (lb.fnormed[s.off:s.off + s.R].clone(), lb.tapcat[s.off:s.off + s.R].clone(),
                       lb.x[s.off:s.off + s.R].clone())
        lean.commit(w, s.st, type("C", (), {"tail": s.tail})(), s.R)
        res[i] = _outs(s.st, pos[i], s.R, logits[j:j + 1].clone(), fn, taps, x)
    return [res[i] for i in range(len(sizes))]


def _same(ref, got):
    return all(len(a) == len(b) and all(x.shape == y.shape and torch.equal(x, y) for x, y in zip(a, b))
               for a, b in zip(ref, got)) and len(ref) == len(got)


def _detail(ref, got):
    return [[torch.equal(x, y) for x, y in zip(a, b)] for a, b in zip(ref, got)]


CASES = [
    ((130, 128, 134, 133), (0, 0, 0, 0)),         # RigMark C4
    ((1, 63, 65), (0, 64, 128)),
    ((200, 64, 5), (128, 0, 192)),
    ((600, 30), (0, 64)),                          # a member of several sub-blocks
    ((64, 64, 64, 64, 64), (0, 0, 64, 128, 192)),
    ((129, 257), (64, 0)),
]


@pytest.fixture
def hashed(monkeypatch):
    from tensorfold.families.glm5_next.cuda import exl3_mm, fastpf, forward, glue, lean, qmm

    k = _hash()._Hash(5)
    for name in ("embed", "hc_pre", "hc_post", "swiglu", "router", "select", "combine", "stream_mean", "rmsnorm"):
        monkeypatch.setattr(glue, name, getattr(k, name))
    monkeypatch.setattr(qmm, "matmul", k.matmul)
    monkeypatch.setattr(qmm, "group_sums", k.group_sums)
    monkeypatch.setattr(fastpf, "kda_chain", k.kda_chain)
    monkeypatch.setattr(forward, "dsa_block", k.dsa_block)
    monkeypatch.setattr(lean, "dsa_block", k.dsa_block)
    monkeypatch.setattr(exl3_mm, "routed", k.routed)
    return k


@needs_torch
@pytest.mark.parametrize("mode", ["off", "direct", "gather", "slab", "gather,slab"])
@pytest.mark.parametrize("block", [64, 128])
@pytest.mark.parametrize("case", range(len(CASES)))
def test_group_equals_lone_on_the_hash_model(hashed, monkeypatch, mode, block, case):
    from tensorfold.families.glm5_next.cuda import pfoverlap as po, pfpp

    monkeypatch.setenv(po.SLAB_ENV, "64")
    monkeypatch.setattr(po, "MODE", po.parse(mode) if mode != "off" else po.OFF)
    monkeypatch.setattr(pfpp, "ON", False)
    sizes, pos = CASES[case]
    w = _hash()._hash_weights()
    ref = _lone(w, sizes, pos, block)
    got = _group(w, sizes, pos, block)
    assert _same(ref, got), (mode, block, sizes, _detail(ref, got))
    # a member order other than fewest-first changes nothing either
    got2 = _group(w, sizes, pos, block, order=list(reversed(range(len(sizes)))))
    assert _same(ref, got2), _detail(ref, got2)


@needs_torch
def test_group_equals_lone_non_exl3(hashed, monkeypatch):
    from tensorfold.families.glm5_next.cuda import pfoverlap as po, pfpp, qmm

    import test_prefill_pp_patches as tp

    gateup, down = tp._mlx_moe()
    monkeypatch.setattr(qmm, "moe_gateup", gateup)
    monkeypatch.setattr(qmm, "moe_down", down)
    monkeypatch.setattr(pfpp, "ON", False)
    for mode in ("off", "gather,slab"):
        monkeypatch.setattr(po, "MODE", po.parse(mode) if mode != "off" else po.OFF)
        w = _hash()._hash_weights("mlx")
        sizes, pos = CASES[0]
        ref = _lone(w, sizes, pos, 64)
        got = _group(w, sizes, pos, 64)
        assert _same(ref, got), (mode, _detail(ref, got))


@needs_torch
def test_experts_once_per_group_forward(hashed, monkeypatch):
    from tensorfold.families.glm5_next.cuda import pfoverlap as po, pfpp

    monkeypatch.setattr(pfpp, "ON", False)
    monkeypatch.setattr(po, "MODE", po.ALL)
    w = _hash()._hash_weights()
    moe = sum(1 for L in w.layers if L.moe is not None)
    kda = sum(1 for L in w.layers if L.kind == "kda")
    dsa = len(w.layers) - kda
    sizes, pos = (130, 128, 600), (0, 0, 64)
    hashed.calls.clear()
    _group(w, sizes, pos, 128)
    subs = sum(-(-R // 128) for R in sizes)
    assert hashed.calls["routed"] == moe                  # once per MoE layer for the whole group
    assert hashed.calls["kda"] == kda * subs and hashed.calls["dsa"] == dsa * subs
    hashed.calls.clear()
    _lone(w, sizes, pos, 128)
    assert hashed.calls["routed"] == moe * len(sizes)


@needs_torch
@pytest.mark.parametrize("bug", ["shared-carry", "group-positions", "straddle"])
def test_hash_model_catches_group_bugs(hashed, monkeypatch, bug):
    from tensorfold.families.glm5_next.cuda import pfoverlap as po, pfpp

    monkeypatch.setattr(pfpp, "ON", False)
    monkeypatch.setattr(po, "MODE", po.ALL)
    w = _hash()._hash_weights()
    sizes, pos = (200, 64, 150), (64, 0, 128)
    ref = _lone(w, sizes, pos, 128)
    if bug == "shared-carry":                  # every member's conv carry in one tail
        orig = mpf.Seg.__init__

        def init(self, st, off, R, p, tail, lb, orig=orig):
            orig(self, st, off, R, p, mpf._tails(lb, 1)[0], lb)
        monkeypatch.setattr(mpf.Seg, "__init__", init)
    elif bug == "group-positions":             # positions counted over the group's rows
        orig = mpf.Seg.__init__

        def init(self, st, off, R, p, tail, lb, orig=orig):
            orig(self, st, off, R, p, tail, lb)
            self.pos = p + off // 64 * 64
        monkeypatch.setattr(mpf.Seg, "__init__", init)
    else:                                      # sub-blocks of the group's rows, not of each member's
        def blocks(segs, block):
            out, a, T = [], 0, sum(s.R for s in segs)
            while a < T:
                r = min(block, T - a)
                i = max(j for j, s in enumerate(segs) if s.off <= a)
                out.append((a, r, i))
                a += r
            return out
        monkeypatch.setattr(mpf, "blocks", blocks)
    try:
        got = _group(w, sizes, pos, 128)
    except (AssertionError, ValueError, RuntimeError, IndexError):
        return                                 # also a detection (e.g. the hash KDA's 64-alignment check)
    assert not _same(ref, got)


# -- two processes over gloo: 0320's row split in a group -----------------------------------------------------------
def _pp_worker(rank: int, init: str, q) -> None:
    import torch.distributed as dist

    try:
        dist.init_process_group("gloo", init_method=f"file://{init}", rank=rank, world_size=2)
        import test_prefill_pp_patches as tp

        from tensorfold.families.glm5_next.cuda import pfoverlap as po, pfpp, qmm

        os.environ[po.SLAB_ENV] = "64"
        tp._install_hash()
        qmm.moe_gateup, qmm.moe_down = tp._mlx_moe(rank)
        res = {}
        for quant in ("exl3", "mlx"):
            for sizes, pos in CASES:
                for mode in ("direct", "gather,slab"):
                    comm = tp._Gloo(rank)
                    w = tp._rank_weights(rank, quant, comm)
                    po.MODE = po.parse(mode)
                    pfpp.ON = True
                    try:
                        ref = _lone(w, sizes, pos, 64)
                        mpf.compute.last = None
                        got = _group(w, sizes, pos, 64)
                        split = isinstance(mpf.compute.last, pfpp._SplitChunk)
                    finally:
                        po.MODE, pfpp.ON = po.OFF, False
                    # [last, fnormed, taps, x, conv, rec] + kcs; x: each split keeps only its own rows' streams
                    ok = [torch.equal(a, b) for m_ref, m_got in zip(ref, got)
                          for i, (a, b) in enumerate(zip(m_ref, m_got)) if i != 3]
                    fn = torch.cat([m[1] for m in got]).contiguous()
                    both = torch.empty((2 * fn.numel(),), dtype=fn.dtype)
                    comm.all_gather(fn.view(-1), both)
                    agree = torch.equal(both[:fn.numel()], both[fn.numel():])
                    res[(quant, sizes, mode)] = (all(ok) and agree, split, sum(sizes))
        q.put((rank, res, None))
    except Exception:                    # noqa: BLE001
        q.put((rank, None, traceback.format_exc()))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:                # noqa: BLE001
            pass


@needs_torch
def test_two_ranks_row_split_group_equals_lone():
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    with tempfile.TemporaryDirectory() as d:
        init = os.path.join(d, "init")
        ps = [ctx.Process(target=_pp_worker, args=(r, init, q)) for r in range(2)]
        for p in ps:
            p.start()
        out = {}
        for _ in range(2):
            rank, res, err = q.get(timeout=900)
            assert err is None, f"rank {rank}:\n{err}"
            out[rank] = res
        for p in ps:
            p.join(60)
    for rank in (0, 1):
        bad = {k: v for k, v in out[rank].items() if not v[0]}
        assert not bad, (rank, bad)
        # every group of more than one sub-block's rows took the split (the lone rule for its row count)
        assert all(v[1] == (v[2] > 64) for v in out[rank].values()), out[rank]


# -- the real Batcher on the hostile fake model -------------------------------------------------------------------------
def _tb(monkeypatch):
    import test_replay_ttft_patches as tr

    return tr._patch_model(monkeypatch)


def _mp_batcher(monkeypatch, tb, *, n=4, rows=64, fast=True, piece=256, budget=4000.0, rank=0, rows_max=4096):
    """test_batch_sessions' fake batcher with GLM53_TF_MULTI_PREFILL on: the group forward and commit are the fake
    model's rows per member (their bits depend on each member's chunk start and length, as the lone path's)."""

    bat = tb._fake_batcher(monkeypatch, n=n, rows=rows, fast=fast, piece=piece, budget_pages=budget, rank=rank)
    bat.mpf_rows = rows_max
    bat.mpf_wait = 0.0
    bat.forwards = []
    bat.g.e.bat = bat                               # the forward hook is module-wide: find this batcher's log

    def forward_group(e, live, chunks):
        outs = []
        for m, ch in zip(live, chunks):
            tb.MODEL.stage(None, m.state, None, ch)
            outs.append(tb.MODEL.compute(None, m.state, None, len(ch), fast=True)[-1:])
        e.bat.forwards.append(tuple((m.slot, len(ch)) for m, ch in zip(live, chunks)))
        return torch.cat(outs), [None] * len(live)

    def commit_member(e, m):
        a, b = m.span()
        tb.MODEL.commit(None, m.state, None, b - a, b - a)

    monkeypatch.setattr(mpf, "forward_group", forward_group)
    monkeypatch.setattr(mpf, "commit_member", commit_member)
    monkeypatch.setattr(mpf, "room", lambda *a: None)
    monkeypatch.setattr(mpf, "engine_ok", lambda e: bool(e.fast_prefill))
    return bat


def _prompts(seed, sizes):
    import numpy as np

    rng = np.random.default_rng(seed)
    return [[int(t) for t in rng.integers(0, 1000, size=s)] for s in sizes]


@needs_torch
@pytest.mark.parametrize("rows", [64, 256], ids=["3-chunks", "1-chunk"])
def test_c4_one_group_first_tokens_and_exactness(monkeypatch, rows):
    from tensorfold.families.glm5_next.cuda import decode_overlap as dover

    tb = _tb(monkeypatch)
    bat = _mp_batcher(monkeypatch, tb, n=4, rows=rows)
    bat.overlap = dover.parse("emit")               # production: GLM53_TF_DECODE_OVERLAP=1
    bat.short = 1024                                # production: GLM53_TF_BATCH_SHORT=1024
    bat.emit_first = True
    sizes = (134, 128, 133, 130)
    prompts = _prompts(4, sizes)
    jobs = [tb._job(p, 12, True, rows) for p in prompts]
    order, ready = [], []
    piece = bat._piece

    def spy(slot):
        ready.append(sum(not j.out.empty() for j in jobs))
        order.append(jobs.index(bat.seqs[slot].job))
        return piece(slot)

    bat._piece = spy
    bat.queue.extend(jobs)
    bat._execute(*bat._plan())                      # one round: all four admitted, one group
    assert sorted(order) == [0, 1, 2, 3] and [sizes[i] for i in order] == sorted(sizes)
    assert ready == [0, 1, 2, 3] and all(not j.out.empty() for j in jobs)
    chunks = -(-min(sizes) // rows)
    assert bat.counts["multi_groups"] == 1 and len(bat.forwards) == -(-max(sizes) // rows)
    assert len(bat.forwards[0]) == 4 and len(bat.forwards[chunks - 1]) == 4
    assert all(j.stats["multi_prefill"] == 4 for j in jobs)
    tb._run_idle(bat)
    for p, job in zip(prompts, jobs):
        reply, done = tb._drain(job)
        want, state = tb._reference(p, 12, rows, True)
        assert done and reply == want                                   # grouped == alone
        assert bat.ended[id(job)][0] == state                           # the slot's whole state too


@needs_torch
def test_group_snapshots_resend_resumes_and_equals_fresh(monkeypatch):
    """Prompts sent together are grouped (their pieces too); the snapshots the group leaves are the lone ones: a
    resend of each resumes all but its last grid step, and every reply == fresh."""

    tb = _tb(monkeypatch)
    rows = 64
    bat = _mp_batcher(monkeypatch, tb, n=3, rows=rows, piece=256, budget=400.0)
    prompts = _prompts(3, (1024, 1000, 700))
    wants = [tb._reference(p, 8, rows, True)[0] for p in prompts]
    cached = []
    for _ in range(3):                                  # sent, then regenerated twice, all three together
        jobs = [tb._job(p, 8, True, rows) for p in prompts]
        bat.queue.extend(jobs)
        tb._run_idle(bat)
        for job, want in zip(jobs, wants):
            reply, done = tb._drain(job)
            assert done and reply == want, job.stats
        cached.append([j.stats["cached"] for j in jobs])
        assert all(len(c) <= 2 for c in bat.caches)
    at = [(len(p) - 1) // 64 * 64 for p in prompts]
    assert cached == [[0, 0, 0], at, at], cached
    assert bat.counts["multi_groups"] >= 4 and max(len(f) for f in bat.forwards) == 3


@needs_torch
@pytest.mark.parametrize("budget", [4000.0, 20.0], ids=["roomy", "tiny"])
def test_sessions_with_multi_prefill_equal_fresh(monkeypatch, budget):
    """test_batch_sessions' 4 interleaved sessions (shared system prompt, forks, resumes across slots, store evictions)
    with every round's pieces grouped: each reply and each slot's final state == fresh prefill + serial decode."""

    import numpy as np

    tb = _tb(monkeypatch)
    bat = _mp_batcher(monkeypatch, tb, n=3, rows=64, piece=256, budget=budget, rows_max=1024)
    stats = tb._conversations(bat, rng=np.random.default_rng(8), fast=True, rows=64)
    assert len(stats) == 56 and sum(1 for s in stats if s.get("restored")) >= 4
    assert bat.counts["multi_groups"] >= 3 and any(len(f) >= 2 for f in bat.forwards)
    assert sum(1 for s in stats if s.get("multi_prefill")) >= 3
    if budget < 100:
        assert bat.store.index.counters["evicted"] > 0


@needs_torch
def test_follower_forms_the_same_groups(monkeypatch):
    import numpy as np

    tb = _tb(monkeypatch)
    rows = 64
    r0 = _mp_batcher(monkeypatch, tb, n=3, rows=rows, piece=256, budget=20.0, rows_max=1024)
    r1 = _mp_batcher(monkeypatch, tb, n=3, rows=rows, piece=256, budget=20.0, rank=1, rows_max=1024)
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
    tb._conversations(r0, rng=np.random.default_rng(5), fast=True, rows=rows, turns=8, check=False)
    with pytest.raises(Done):
        r1.follow()
    key = lambda d: (d["slot"], d["sha256"], tuple(d["keeps"]), d["cancelled"])      # noqa: E731
    assert [key(d) for d in r1.log] == [key(d) for d in r0.log] and len(r0.log) == 32
    assert [t[1:] for t in r1.trace] == [t[1:] for t in r0.trace]
    assert r1.forwards == r0.forwards and r0.counts["multi_groups"] > 0
    a, b = r0.store.index, r1.store.index
    assert a.digest() == b.digest() and a.counters == b.counters
    assert [st.state() for st in r0.states] == [st.state() for st in r1.states]


@needs_torch
def test_admission_rules(monkeypatch):
    """Other knobs: another group; image rows: alone; a member's own error fails only that request; a request
    decoding meanwhile verifies after the group in the same round."""

    tb = _tb(monkeypatch)
    rows = 64
    bat = _mp_batcher(monkeypatch, tb, n=4, rows=rows)
    bat.short = 1024
    prompts = _prompts(11, (130, 140, 150, 160))
    prompts[3][5] = (1 << 24) + 7                     # a virtual image id (patches/0500): runs alone
    jobs = [tb._job(p, 10, True, rows) for p in prompts]
    jobs[1].values = dict(jobs[1].values, prefill_rows=128)     # other knobs: never with the others
    groups = []
    orig = mpf.groups

    def spy(b, pieces):
        out = orig(b, pieces)
        groups.append(out)
        return out

    monkeypatch.setattr(mpf, "groups", spy)
    from tensorfold.families.glm5_next.cuda import batch

    monkeypatch.setattr(batch.vision_mod, "exchange", lambda g, prompt, table: table)   # no tower in the fake
    bat.queue.extend(jobs)
    bat._execute(*bat._plan())
    slot = {id(bat.seqs[s].job): s for s in range(4) if bat.seqs[s] is not None}
    got = sorted(sorted(g) for g in groups[0])
    assert got == sorted([sorted([slot[id(jobs[0])], slot[id(jobs[2])]]), [slot[id(jobs[1])]],
                          [slot[id(jobs[3])]]]), groups
    tb._run_idle(bat)
    for p, job in zip(prompts, jobs):
        reply, done = tb._drain(job)
        # (the fake's ``_knobs`` applies nothing: every request ran 64-row chunks)
        assert done and reply == tb._reference(p, 10, rows, True)[0]

    # a member's own error (ValueError in its setup) fails that request only
    bat2 = _mp_batcher(monkeypatch, tb, n=3, rows=rows)
    bat2.short = 1024
    ps = _prompts(12, (100, 110, 120))
    js = [tb._job(p, 6, True, rows) for p in ps]
    setup = mpf._Member.setup

    def bad(self, e):
        if self.job is js[1]:
            raise ValueError("this snapshot's draft caches do not fit the request")
        return setup(self, e)

    monkeypatch.setattr(mpf._Member, "setup", bad)
    bat2.queue.extend(js)
    tb._run_idle(bat2)
    for i, (p, job) in enumerate(zip(ps, js)):
        if i == 1:
            item = job.out.get()
            assert isinstance(item, ValueError)
            continue
        reply, done = tb._drain(job)
        assert done and reply == tb._reference(p, 6, rows, True)[0]
    monkeypatch.setattr(mpf._Member, "setup", setup)

    # decoders in the same round: the group's pieces, then the verify forward of the one decoding
    bat3 = _mp_batcher(monkeypatch, tb, n=4, rows=rows)
    bat3.short = 1024
    first = tb._job(_prompts(13, (90,))[0], 40, True, rows)
    bat3.queue.append(first)
    bat3._execute(*bat3._plan())
    bat3._execute(*bat3._plan())
    rest = [tb._job(p, 8, True, rows) for p in _prompts(14, (100, 120, 110))]
    bat3.queue.extend(rest)
    bat3._execute(*bat3._plan())
    rnd, pieces, active = bat3.trace[-1]
    assert len(pieces) == 3 and len(active) == 4 and bat3.counts["multi_groups"] == 1
    tb._run_idle(bat3)
    for job in [first] + rest:
        reply, done = tb._drain(job)
        assert done and reply == tb._reference(job.prompt, job.max_tokens, rows, True)[0]


@needs_torch
def test_last_member_leaves_when_a_solo_piece_could_move_its_bound(monkeypatch):
    """GLM53_TF_SOLO_PIECE: members done at their first token free their slots; the last member would then be alone
    and ``_piece`` would cut a solo piece: it prefills alone after the group (the others keep theirs grouped)."""

    tb = _tb(monkeypatch)
    rows = 64
    bat = _mp_batcher(monkeypatch, tb, n=3, rows=rows, piece=256)
    bat.short = 1024
    bat.solo = 1024
    prompts = _prompts(41, (100, 120, 1500))
    jobs = [tb._job(p, 1, True, rows) for p in prompts[:2]] + [tb._job(prompts[2], 6, True, rows)]
    bat.queue.extend(jobs)
    tb._run_idle(bat)
    for p, job in zip(prompts, jobs):
        reply, done = tb._drain(job)
        assert done and reply == tb._reference(p, job.max_tokens, rows, True)[0]
    assert bat.counts["multi_groups"] == 1 and bat.forwards[0] and len(bat.forwards[0]) == 2
    assert jobs[2].stats.get("solo_pieces") and not jobs[2].stats.get("multi_prefill")


@needs_torch
def test_more_pieces_share_forwards_past_batch_short(monkeypatch):
    """Four 1,500-token prompts (past BATCH_SHORT = 1024): pieces of 256 rows, several a round in one forward."""

    tb = _tb(monkeypatch)
    rows = 256
    bat = _mp_batcher(monkeypatch, tb, n=4, rows=rows, piece=256, rows_max=1024)
    bat.short = 1024
    prompts = _prompts(21, (1500, 1540, 1600, 1500))
    jobs = [tb._job(p, 6, True, rows) for p in prompts]
    bat.queue.extend(jobs)
    tb._run_idle(bat)
    for p, job in zip(prompts, jobs):
        reply, done = tb._drain(job)
        assert done and reply == tb._reference(p, 6, rows, True)[0]
    assert max(len(f) for f in bat.forwards) == 4
    assert sum(r for f in bat.forwards for _, r in f) == sum(map(len, prompts))
    assert all(sum(r for _, r in f) <= 1024 for f in bat.forwards)


@needs_torch
def test_coalesce_waits_for_arrivals_when_idle(monkeypatch):
    import threading
    import time

    tb = _tb(monkeypatch)
    bat = _mp_batcher(monkeypatch, tb, n=4, rows=64)
    bat.mpf_wait = 0.2
    jobs = [tb._job(p, 4, True, 64) for p in _prompts(31, (100, 110, 120, 130))]
    for j in jobs:
        j.submitted = time.perf_counter()
    bat.queue.append(jobs[0])

    def later():
        time.sleep(0.03)
        with bat.cv:
            bat.queue.extend(jobs[1:])
            bat.cv.notify()

    threading.Thread(target=later).start()
    t0 = time.perf_counter()
    cancels, admits, pieces = bat._plan()
    assert len(admits) == 4 and time.perf_counter() - t0 < 0.19           # the slots filled: no full wait
    # alone: waits the bound, then goes
    bat2 = _mp_batcher(monkeypatch, tb, n=4, rows=64)
    bat2.mpf_wait = 0.05
    j = tb._job(_prompts(32, (100,))[0], 4, True, 64)
    j.submitted = time.perf_counter()
    bat2.queue.append(j)
    t0 = time.perf_counter()
    assert len(bat2._plan()[1]) == 1 and 0.04 <= time.perf_counter() - t0 < 0.5


@needs_torch
@pytest.mark.parametrize("mode", ["off", "gather,slab"])
def test_forward_group_and_commit_hooks_on_the_hash_model(hashed, monkeypatch, mode):
    """The driver's real hooks (``forward_group``: staging, segments, carries, views; ``commit_member``) on the hash
    model == each member's lone ``lean.stage`` / ``lean.compute`` / ``lean.commit``, the views reading each member's
    rows."""

    from types import SimpleNamespace as NS

    from tensorfold.families.glm5_next.cuda import forward, lean, pfoverlap as po, pfpp

    monkeypatch.setenv(po.SLAB_ENV, "64")
    monkeypatch.setattr(po, "MODE", po.parse(mode) if mode != "off" else po.OFF)
    monkeypatch.setattr(pfpp, "ON", False)
    rooms = []
    monkeypatch.setattr(forward, "check_room", lambda w, st, R, p=None: rooms.append((R, p)))   # tiny dense limit
    w = _hash()._hash_weights()
    sizes, pos = (130, 64, 200), (0, 128, 64)
    ref = _lone(w, sizes, pos, 64)
    b = forward.Buffers(w, 64, 2048)
    b.set_taps((1, 3), w.cfg.hidden)
    lb = lean.LeanBuffers(w, 1024, 64, taps=2)
    e = NS(w=w, buf=b, lean=lb)
    live = []
    for i, (R, p0) in enumerate(zip(sizes, pos)):
        m = NS(state=_fresh(w, 64, 2048, p0, i), slot=i, span=lambda R=R, p0=p0: (p0, p0 + R))
        live.append(m)
    chunks = [_ids(R, i).tolist() for i, R in enumerate(sizes)]
    logits, views = mpf.forward_group(e, live, chunks)
    assert rooms == list(zip(sizes, pos))                  # each member's room at its own position
    assert lb.ids[:sum(sizes)].tolist() == sum(chunks, [])
    got = []
    for i, m in enumerate(live):
        fn, taps = views[i].fnormed.clone(), views[i].tapcat.clone()
        x = lb.x[m.seg.off:m.seg.off + m.seg.R].clone()
        mpf.commit_member(e, m)
        assert m.state.pos == pos[i] + sizes[i]
        got.append(_outs(m.state, pos[i], sizes[i], logits[i:i + 1].clone(), fn, taps, x))
    assert _same(ref, got), _detail(ref, got)


# -- GPU: the engine on TensorFold's synthetic EXL3 checkpoint (one GPU playing rank 0 of two) -------------------------
CUDA = torch is not None and torch.cuda.is_available()
gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")


def _gpu_engine(path, *, batch: int, multi: bool, overlap: str = "1", block: int = 64, rows_max: int = 512,
                wait_ms: int = 0):
    from test_glm_engine import _TwoCopies

    from tensorfold.families.glm5_next.cuda import weights
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    with pytest.MonkeyPatch.context() as m:
        m.setattr(weights, "NONEXPERT", "q4mse")
        m.setattr(mpf, "ON", multi)
        env = {"GLM53_TF_NONEXPERT": "q4mse", "GLM53_TF_LOOKUP": "0", "GLM53_TF_BATCH": str(batch),
               "GLM53_TF_BATCH_SESSIONS": "1" if batch > 1 else "0", "GLM53_TF_SESSION_GIB": "1" if batch > 1 else "0",
               "GLM53_TF_SESSION_RESERVE_GIB": "0", "GLM53_TF_PREFILL_ROWS": "auto",
               "GLM53_TF_PREFILL_ROWS_MAX": str(rows_max), "GLM53_TF_FAST_PREFILL": "1", "GLM53_TF_LEAN_PREFILL": "1",
               "GLM53_TF_LEAN_BLOCK": str(block), "GLM53_TF_PREFILL_OVERLAP": overlap, "GLM53_TF_BATCH_SHORT": "1024",
               "GLM53_TF_BATCH_PREFILL_SHARE": "1.0", "GLM53_TF_BATCH_RESERVE_GB": "0.25",
               "GLM53_TF_BATCH_ADMIT_GB": "0", "GLM53_TF_BATCH_PIECE": "256", mpf.ENV: "1" if multi else "0",
               mpf.WAIT_ENV: str(wait_ms)}
        for k, v in env.items():
            m.setenv(k, v)
        for k in ("GLM53_TF_CALIB_ONLINE", "GLM53_TF_PROFILE", "GLM53_TF_DEPTH", "GLM53_TF_FAST_GATHER",
                  "GLM53_TF_LATENT_KV", "GLM53_TF_FP8_PREFILL", "GLM53_TF_SNAPSHOT_GRID", "GLM53_TF_PREFILL_PP",
                  "GLM53_TF_SOLO_PIECE", mpf.ROWS_ENV):
            m.delenv(k, raising=False)
        from tensorfold.families.glm5_next.cuda import pfoverlap

        m.setattr(pfoverlap, "DEFAULT", pfoverlap.parse(overlap))
        m.setattr(pfoverlap, "MODE", pfoverlap.parse(overlap))
        return GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", comm=_TwoCopies())


@pytest.fixture(scope="module")
def gckpt(tmp_path_factory):
    if not CUDA:
        pytest.skip("CUDA only")
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_multi_prefill")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


def _alone(lone, prompt, sampling, tokens):
    out: list[int] = []
    lone.request.policy = None
    lone.request.stop_eos = False
    lone.cache = []
    stats = lone.generate(list(prompt), tokens, sampling, lambda new: out.extend(new), draft=False)
    assert stats["cached"] == 0 and len(out) == tokens
    return out


@gpu
@pytest.mark.parametrize("overlap", ["0", "1"], ids=["lean", "pipelined"])
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
def test_gpu_group_equals_alone(gckpt, overlap, greedy):
    """4 prompts (3 short, one of several sub-blocks) submitted together: one group; every reply == the same prompt
    alone on a lone fast lean engine (fresh prefill, serial decoding); the next turns (prompt + reply) resume from the
    group's snapshots and stay exact."""

    import numpy as np

    from tensorfold.engine.exact_sampling import Sampling

    sampling = None if greedy else Sampling(99, 1.0, 20, 0.95)
    lone = _gpu_engine(gckpt, batch=1, multi=False, overlap=overlap)
    eng = _gpu_engine(gckpt, batch=4, multi=True, overlap=overlap)
    rng = np.random.default_rng(17)
    prompts = [[int(t) for t in rng.integers(0, 1000, size=n)] for n in (130, 128, 300, 70)]
    policies = ["2", "auto", "f3", "2"]
    for wave in range(2):
        got = eng.batch.generate_batch([dict(prompt=p, max_tokens=16, sampling=sampling, policy=pol)
                                        for p, pol in zip(prompts, policies)])
        for i, (p, (reply, stats)) in enumerate(zip(prompts, got)):
            assert reply == _alone(lone, p, sampling, 16), (wave, i, stats)
            if wave:
                assert stats["cached"] > 0 and stats["cached"] % 64 == 0, stats
        assert eng.batch.counts["multi_groups"] >= wave + 1 and any(s.get("multi_prefill") for _, s in got)
        prompts = [p + r + [5, 6, 7] for p, (r, _) in zip(prompts, got)]
    eng.batch.stop()
    torch.cuda.empty_cache()
