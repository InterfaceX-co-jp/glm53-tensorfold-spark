"""patches/0540: warm replay (the prompt's snapshot strictly before its end) and multi-arrival TTFT (a piece's first
token handed out at once). docs/REPLAY-TTFT.md.

Host only (no torch; runnable anywhere: PYTHONPATH=<tree>/src:<tree>/tests/cuda pytest -q <this file>):

- ``pfgrid.plan(before=True)``: an on-grid prompt's snapshot moves from n to n - G (a cut there, marks deduplicated);
  off the grid nothing changes; for every length, resume point and chunk size the snapshot is a strict prefix within
  tail + G of the end; exact prefills snapshot at the last 64-multiple before n; the knobs parse.

With torch on any device (CPU is enough), on test_batch_sessions_patches's hostile fake model:

- resume at end - grid == fresh: a prefill with the rule keeps the fresh prefill's bits (first token, whole state),
  its snapshot is a strict prefix within 64 tokens of the end, and the same prompt resumed from it gives the fresh
  prefill's first token, state and reply (exact with cut chunks, fast on the grid; on-grid and off-grid lengths);
- the real batcher + session store: an identical prompt sent again resumes all but its last 64 tokens (0 without
  the rule), replies == fresh; the rule keeps no more snapshots than before (per request, in the slots and in the
  store, under budget);
- multi-arrival: 4 short prompts queued together are admitted and prefilled in one round, fewest tokens first, and
  with GLM53_TF_EMIT_FIRST each one's first token is out before the next piece starts (held until the round's
  verify forward without it); every reply equals the same request alone.

Run inside the image:
    PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q tests/cuda/test_replay_ttft_patches.py
"""

from __future__ import annotations

import pytest

from tensorfold.families.glm5_next.cuda import pfgrid

try:
    import torch
except ImportError:          # the host-only tests still run
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="torch")


# -- host only ------------------------------------------------------------------------------------------------------
def test_on_grid_prompt_snapshots_one_grid_step_before_its_end():
    old = pfgrid.plan(0, 8192, 4096, fast=True, grid=64, tail=256)
    assert old.snap == 8192 and old.spans == [(0, 4096), (4096, 8192)]
    new = pfgrid.plan(0, 8192, 4096, fast=True, grid=64, tail=256, before=True)
    assert new.snap == 8128 and new.cut and new.spans == [(0, 4096), (4096, 8128), (8128, 8192)]
    # RigMark's pieces: the last piece of a 32K / 64K prompt (resumed from the previous piece's snapshot)
    for n in (32768, 65536):
        p = pfgrid.plan(n - 4096, n, 4096, fast=True, grid=64, tail=256, before=True)
        assert p.snap == n - 64 and p.spans == [(n - 4096, n - 64), (n - 64, n)]
    # the resend itself: resumed at n - 64, nothing new to keep (the resume snapshot stays the prompt's)
    again = pfgrid.plan(8128, 8192, 4096, fast=True, grid=64, tail=256, before=True)
    assert again.snap == 8128 and again.spans == [(8128, 8192)]
    # a session mark at the new snapshot point is the snapshot itself, taken once
    m = pfgrid.plan(0, 8192, 4096, fast=True, marks=[4096, 8128], grid=64, tail=256, before=True)
    assert m.snap == 8128 and m.marks == [4096]
    # a bigger grid: one grid step
    g = pfgrid.plan(0, 8192, 4096, fast=True, grid=256, tail=0, before=True)
    assert g.snap == 7936


def test_off_grid_prompts_unchanged_and_every_snapshot_is_a_strict_prefix():
    tail, g = 256, 64
    for step in (64, 256, 4096):
        for begin in (0, 64, 1024):
            for n in range(begin + 1, begin + 1400, 7):
                p = pfgrid.plan(begin, n, step, fast=True, grid=g, tail=tail, before=True)
                q = pfgrid.plan(begin, n, step, fast=True, grid=g, tail=tail)
                if n % g:
                    assert (p.spans, p.snap, p.marks) == (q.spans, q.snap, q.marks)
                assert begin <= p.snap < n
                assert n - p.snap <= tail + g                   # a resend re-prefills at most tail + G rows
                assert [a for a, _ in p.spans][0] == begin and p.spans[-1][1] == n
                assert all(a % 64 == 0 for a, _ in p.spans)       # fast calls start on the 64-grid


def test_exact_prefill_snapshot_before_the_end():
    p = pfgrid.plan(0, 1000, 64, fast=False, before=True)
    assert p.snap == 960 and not p.cut and p.spans == pfgrid.plan(0, 1000, 64, fast=False).spans
    q = pfgrid.plan(0, 8192, 8192, fast=False, before=True)
    assert q.snap == 8128 and q.cut and q.spans == [(0, 8128), (8128, 8192)]
    assert pfgrid.plan(0, 50, 64, fast=False, before=True).snap == -1        # nothing worth keeping
    assert pfgrid.plan(960, 1000, 64, fast=False, before=True).snap == 960   # the resume snapshot stays
    assert pfgrid.plan(0, 1000, 64, fast=False, marks=[512, 960], before=True).marks == [512]
    assert pfgrid.plan(0, 1000, 64, fast=False).snap == -1                   # without the rule: after the prompt


def test_knobs(monkeypatch):
    monkeypatch.delenv(pfgrid.BEFORE_ENV, raising=False)
    assert pfgrid.before_end()
    monkeypatch.setenv(pfgrid.BEFORE_ENV, "0")
    assert not pfgrid.before_end()
    monkeypatch.setenv(pfgrid.BEFORE_ENV, "yes")
    with pytest.raises(ValueError, match="0 or 1"):
        pfgrid.before_end()


# -- torch on any device: the fake model ----------------------------------------------------------------------------
def _patch_model(monkeypatch):
    import test_batch_sessions_patches as tb

    from tensorfold.families.glm5_next.cuda import decode

    for name in ("stage", "compute", "commit"):
        monkeypatch.setattr(decode, name, getattr(tb.MODEL, name))
    monkeypatch.setattr(decode, "_sync", lambda w: None)
    return tb


def _decode(tb, st, first: int, tokens: int) -> list[int]:
    out = [first]
    while len(out) < tokens:
        tb.MODEL.stage(None, st, None, [out[-1]])
        h = tb.MODEL.compute(None, st, None, 1)
        tb.MODEL.commit(None, st, None, 1, 1)
        out.append(int(h[0, 0]) % 5)
    return out


@needs_torch
@pytest.mark.parametrize("fast,rows", [(False, 256), (False, 64), (True, 64)], ids=["exact-cut", "exact", "fast"])
@pytest.mark.parametrize("n", [1024, 640, 700, 129, 64, 30])
def test_resume_at_end_minus_grid_equals_fresh(monkeypatch, fast, rows, n):
    import numpy as np

    from tensorfold.families.glm5_next.cuda import decode

    tb = _patch_model(monkeypatch)
    prompt = [int(t) for t in np.random.default_rng(n).integers(0, 1000, size=n)]
    want, want_state = tb._reference(prompt, 16, rows, fast)          # fresh prefill (no rule) + serial decode

    st = tb._St()
    st.kv.fill_(-99)
    e = tb._E(st, rows, fast)
    e.snap_before = n                                           # the caller's whole prompt (``engine._run``)
    first = decode.prefill(e, prompt, None, mtp=True, drafter=None)
    assert e.snap_before == 0                                   # one prefill's
    snap = e.fast_snap
    assert (first, _decode(tb, st, first, 16)[1:]) == (want[0], want[1:]) and st.state() == want_state
    if n <= 64:
        assert snap is None                                     # one short chunk: nothing worth keeping
        return
    assert snap is not None and len(snap.ids) < n and n - len(snap.ids) <= 64 and len(snap.ids) % 64 == 0

    st2 = tb._St()                                              # the resend, on the live caches the prefill left
    st2.kv.fill_(-99)
    e2 = tb._E(st2, rows, fast)
    e2.snap_before = n
    decode.prefill(e2, prompt, None, mtp=True, drafter=None)
    snap2 = e2.fast_snap
    e2.snap_before = n
    again = decode.prefill(e2, prompt, None, mtp=True, drafter=None, resume=snap2)
    assert e2.fast_snap is snap2                                # nothing new to keep
    assert again == want[0] and _decode(tb, st2, again, 16) == want and st2.state() == want_state


@needs_torch
@pytest.mark.parametrize("fast", [False, True], ids=["exact", "fast"])
@pytest.mark.parametrize("on", [True, False], ids=["rule", "no-rule"])
def test_batched_resend_resumes_all_but_the_last_grid_step(monkeypatch, fast, on):
    import numpy as np

    tb = _patch_model(monkeypatch)
    if not on:
        monkeypatch.setenv(pfgrid.BEFORE_ENV, "0")
    rows = 64
    bat = tb._fake_batcher(monkeypatch, n=2, rows=rows, fast=fast, piece=256, budget_pages=40.0)
    saves: list[int] = []
    save = bat._save

    def spy(slot, snaps):
        saves.append(len([s for s in snaps if s is not None]))
        return save(slot, snaps)

    bat._save = spy
    rng = np.random.default_rng(3)
    for n in (1024, 1000):
        prompt = [int(t) for t in rng.integers(0, 1000, size=n)]
        want = tb._reference(prompt, 8, rows, fast)[0]
        cached = []
        for _ in range(3):                                      # sent, then regenerated twice
            job = tb._job(prompt, 8, fast, rows)
            bat.queue.append(job)
            tb._run_idle(bat)
            reply, done = tb._drain(job)
            assert done and reply == want, job.stats
            cached.append(job.stats["cached"])
            assert all(len(c) <= 2 for c in bat.caches)
            assert bat.store.index.used <= bat.store.index.budget
        at = (n - 1) // 64 * 64                                 # the last grid point strictly before n
        if on:
            assert cached == [0, at, at], cached                # all but <= 64 tokens, every time
        elif n % 64 == 0:
            assert cached == [0, 0, 0], cached                  # W13: the whole prompt again (no mark below it)
    assert max(saves) == 1                                      # one snapshot a save (prompt, or exact reply)


@needs_torch
@pytest.mark.parametrize("fast", [False, True], ids=["exact", "fast"])
def test_rule_keeps_no_more_snapshots(monkeypatch, fast):
    """The same conversations with and without the rule: every reply exact (``_conversations`` checks each against a
    fresh prefill), no more snapshots saved with it, the store under budget."""

    import numpy as np

    tb = _patch_model(monkeypatch)
    counts = {}
    for on in (False, True):
        monkeypatch.setenv(pfgrid.BEFORE_ENV, "1" if on else "0")
        bat = tb._fake_batcher(monkeypatch, n=3, rows=64, fast=fast, piece=256, budget_pages=30.0)
        n_snaps = [0]
        save = bat._save

        def spy(slot, snaps, save=save, n_snaps=n_snaps):
            n_snaps[0] += len([s for s in snaps if s is not None])
            return save(slot, snaps)

        bat._save = spy
        tb._conversations(bat, rng=np.random.default_rng(21), fast=fast, rows=64, turns=6)
        counts[on] = n_snaps[0]
        assert bat.store.index.used <= bat.store.index.budget
    assert counts[True] <= counts[False], counts


@needs_torch
@pytest.mark.parametrize("fast", [False, True], ids=["exact", "fast"])
@pytest.mark.parametrize("first_now", [True, False], ids=["emit-first", "held"])
def test_multi_arrival_first_tokens_and_exactness(monkeypatch, fast, first_now):
    import numpy as np

    from tensorfold.families.glm5_next.cuda import decode_overlap as dover

    tb = _patch_model(monkeypatch)
    rows = 64
    bat = tb._fake_batcher(monkeypatch, n=4, rows=rows, fast=fast, piece=256, budget_pages=40.0)
    bat.overlap = dover.parse("emit")               # production: GLM53_TF_DECODE_OVERLAP=1 holds a round's tokens
    bat.short = 1024                                # production: GLM53_TF_BATCH_SHORT=1024
    bat.emit_first = first_now
    rng = np.random.default_rng(4)
    sizes = (134, 128, 133, 130)                    # RigMark C4: four code prompts of ~130 tokens
    prompts = [[int(t) for t in rng.integers(0, 1000, size=s)] for s in sizes]
    jobs = [tb._job(p, 12, fast, rows) for p in prompts]
    order, ready = [], []
    piece = bat._piece

    def spy(slot):
        job = bat.seqs[slot].job
        ready.append(sum(not j.out.empty() for j in jobs))
        order.append(jobs.index(job))
        return piece(slot)

    bat._piece = spy
    bat.queue.extend(jobs)
    bat._execute(*bat._plan())                      # one round: all four admitted, all four prefilled
    assert sorted(order) == [0, 1, 2, 3] and bat.round == 1
    assert [sizes[i] for i in order] == sorted(sizes)                # fewest tokens first
    firsts = [not j.out.empty() for j in jobs]
    if first_now:
        assert ready == [0, 1, 2, 3] and all(firsts)                 # each first token out before the next piece
    else:
        assert ready == [0, 0, 0, 0]                                 # held behind the round's pieces and verify
    tb._run_idle(bat)
    for p, job in zip(prompts, jobs):
        reply, done = tb._drain(job)
        assert done and reply == tb._reference(p, 12, rows, fast)[0]  # batched == alone
