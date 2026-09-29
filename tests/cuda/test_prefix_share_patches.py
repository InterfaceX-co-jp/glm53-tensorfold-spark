"""patches/0310 (GLM53_TF_PREFIX_SHARE=1): shared-prefix reuse across sessions on the session store (0110 / 0180 /
0250) and the KV pool (0290).

Host only (no torch):

- ``PrefixShare``: where a prompt's system part ends (the first user / assistant / observation role token, within
  [lo, hi], up to ``points`` of them), system-only ``prefix`` entries, the settings and the role-token lookup
  (env, tokenizer_config.json, tokenizer.json); off by default;
- ``SessionIndex.marks(extra=)``: the end of the system prompt on the snapshot grid (page for exact, 64 for fast),
  past the resume point by ``fork_min``, not stored already;
- the cap on ``prefix`` entries (rank 0 evicts the least recently used one) replays on rank 1 (same digest).

With torch on the CPU (0180's hostile fake model through the real ``Batcher._plan`` / ``_execute`` / ``_admit`` /
``_piece`` / ``_finish`` / ``follow`` and the real ``SessionStore``):

- today's behaviour, measured (the control): the second session over a 600-token system prompt prefills it all;
- with the switch: it resumes at the system prompt's end (512 exact / 576 fast), from the first session's mark;
- sessions submitted together: the later ones wait for the first one's mark and resume there (without the wait they
  prefill side by side); a partner that goes away never leaves a request waiting;
- random multi-session conversations with role tokens (turns, forks, new sessions over 2 system prompts), exact and
  fast, roomy and tiny store budgets, with and without the KV pool: every reply and every slot's whole state at the
  end of a request == a fresh prefill + serial decode;
- a follower replaying rank 0's messages ends with the same store (with ``prefix`` evictions) and slot states.

GPU (TensorFold's synthetic EXL3 checkpoint, one GPU playing rank 0; ``GLM53_TF_PREFIX_SHARE_TOKENS`` names a role
token inside its 1,024-token vocabulary): a new session resumes at the system prompt's end and replies == alone;
sessions submitted together wait and resume; exact and fast prefill; a follower replays the same decisions.

Run: PYTHONPATH=<tree>/src:<tree>/tests/cuda:tests/cuda pytest -q tests/cuda/test_prefix_share_patches.py
"""

from __future__ import annotations

import json
import time

import pytest

from tensorfold.families.glm5_next.cuda import sessions

if not hasattr(sessions, "PrefixShare"):
    pytest.skip("patches/0310 not applied", allow_module_level=True)

try:
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    CUDA = False

gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")
needs_torch = pytest.mark.skipif(torch is None, reason="torch")

USER, ASSIST, OBS = 5000, 5001, 5002
ROLES = [USER, ASSIST, OBS]


def _share(**kw) -> "sessions.PrefixShare":
    kw.setdefault("lo", 256)
    return sessions.PrefixShare(ROLES, **kw)


# -- host only ------------------------------------------------------------------------------------------------------
def test_off_by_default_and_settings(monkeypatch, tmp_path):
    for k in ("GLM53_TF_PREFIX_SHARE", "GLM53_TF_PREFIX_SHARE_TOKENS"):
        monkeypatch.delenv(k, raising=False)
    assert sessions.prefix_share() is False and sessions.PrefixShare.from_env(tmp_path) is None
    monkeypatch.setenv("GLM53_TF_PREFIX_SHARE", "2")
    with pytest.raises(ValueError):
        sessions.prefix_share()
    monkeypatch.setenv("GLM53_TF_PREFIX_SHARE", "1")
    with pytest.raises(ValueError, match="role tokens"):
        sessions.PrefixShare.from_env(tmp_path)                    # no tokenizer, no override
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"added_tokens_decoder": {
        "151336": {"content": "<|user|>"}, "151337": {"content": "<|assistant|>"}, "151338": {"content": "x"}}}))
    (tmp_path / "tokenizer.json").write_text(json.dumps({"added_tokens": [
        {"id": 151336, "content": "<|user|>"}, {"id": 151339, "content": "<|observation|>"}]}))
    assert sessions.role_token_ids(tmp_path) == [151336, 151337, 151339]
    sh = sessions.PrefixShare.from_env(tmp_path)
    assert (sh.roles, sh.lo, sh.hi, sh.points, sh.keep, sh.wait) == ([151336, 151337, 151339], 2048, 131072, 1, 4,
                                                                      True)
    monkeypatch.setenv("GLM53_TF_PREFIX_SHARE_TOKENS", "7, 9")
    monkeypatch.setenv("GLM53_TF_PREFIX_SHARE_WAIT", "0")
    monkeypatch.setenv("GLM53_TF_PREFIX_SHARE_KEEP", "0")
    sh = sessions.PrefixShare.from_env(tmp_path)
    assert sh.roles == [7, 9] and not sh.wait and sh.keep == 0
    monkeypatch.setenv("GLM53_TF_PREFIX_SHARE_MIN", "0")
    with pytest.raises(ValueError):
        sessions.PrefixShare.from_env(tmp_path)


def test_points_of_and_prefix_kind():
    sh = _share(hi=5000)
    p = [9] * 600 + [USER] + [1] * 50 + [ASSIST] + [2] * 20 + [OBS] + [3]
    assert sh.points_of(p) == [600]
    assert _share(points=3).points_of(p) == [600, 651, 672]
    assert _share(lo=700).points_of(p) == []                        # too short to be worth a snapshot
    assert _share(lo=700, points=3).points_of(p) == []
    assert _share(hi=599).points_of(p) == []                        # past the scan window
    assert sh.points_of([9] * 900) == [] and sh.points_of([]) == []
    assert sh.is_prefix(p[:600]) and sh.is_prefix(p[:601][:-1])
    assert not sh.is_prefix(p[:601]) and not sh.is_prefix(p) and not sh.is_prefix([USER])
    assert sh.is_prefix([USER] + [9] * 10)                          # a role token first is the template's own


def _index(pages: int = 4000) -> "sessions.SessionIndex":
    return sessions.SessionIndex(pages * 1000, (1, 1, 1, 1), extent=4)


@pytest.mark.parametrize("tag,want", [(0, 512), (64, 576)], ids=["exact", "fast"])
def test_marks_at_the_system_end(tag, want):
    ix = _index()
    p = [9] * 600 + [USER] + [1] * 400
    kw = dict(has_mtp=True, drafter=False, every=0, fork_min=64)
    assert ix.marks(p, tag, 0, **kw) == []                          # today: nothing to fork from, no mark
    assert ix.marks(p, tag, 0, extra=[600], **kw) == [want]
    assert ix.marks(p, tag, want - 63, extra=[600], **kw) == []     # not fork_min past the resume point
    assert ix.marks(p[:want + 1], tag, 0, extra=[600], **kw) == ([want] if tag == 0 else [])   # fast: the prompt's
    ix.save(tag=tag, ids=p[:want], mtp_len=want - 1, drafter=False, snap_bytes=10)     # last grid point is its own
    assert ix.marks(p, tag, 0, extra=[600], **kw) == []             # stored already


def test_prefix_entries_capped_and_replayed():
    r0, r1 = _index(), _index()
    r0.prefix_keep = 2
    sh = _share()
    decisions = []
    for k in range(5):
        system = [100 + k] * 300
        for ids, kind in ((system, "prefix"), (system + [USER, k, k], "turn")):
            assert sh.is_prefix(ids) == (kind == "prefix")
            got = r0.save(tag=0, ids=ids, mtp_len=len(ids) - 1, drafter=False, snap_bytes=10, kind=kind)
            decisions.append((ids, kind, got.status, list(got.evicted)))
    kinds = sorted(e.kind for e in r0.entries.values())
    assert kinds.count("prefix") == 2 and kinds.count("turn") == 5
    assert {tuple(e.ids[:1]) for e in r0.entries.values() if e.kind == "prefix"} == {(103,), (104,)}
    for ids, kind, status, evicted in decisions:                    # rank 1: no cap of its own, rank 0's decisions
        r1.save(tag=0, ids=ids, mtp_len=len(ids) - 1, drafter=False, snap_bytes=10, kind=kind,
                forced=(status, evicted))
    assert r1.digest() == r0.digest() and r1.counters == r0.counters
    # a used prefix entry is kept over an older unused one
    r0.touch(next(e for e in r0.entries.values() if e.kind == "prefix" and e.ids[0] == 103))
    r0.save(tag=0, ids=[200] * 300, mtp_len=299, drafter=False, snap_bytes=10, kind="prefix")
    assert {e.ids[0] for e in r0.entries.values() if e.kind == "prefix"} == {103, 200}


# -- torch on the CPU: the real batcher and store on the hostile fake model ------------------------------------------
def _fake(monkeypatch, *, n=3, fast=False, budget=4000.0, share=None, rank=0, fork=64):
    from test_batch_sessions_patches import _fake_batcher

    bat = _fake_batcher(monkeypatch, n=n, rows=64, fast=fast, piece=256, budget_pages=budget, rank=rank, fork=fork)
    if share is not None:
        bat.store.attach_share(share)
    return bat


def _run(bat, prompts, tokens=8, fast=False):
    """Queue ``prompts`` together, run rank 0's rounds until idle; every reply == fresh prefill + serial decode and
    every slot state at the end == the reference's. -> stats per prompt."""

    from test_batch_sessions_patches import _drain, _job, _reference, _run_idle

    jobs = [_job(p, tokens, fast, 64) for p in prompts]
    bat.queue.extend(jobs)
    _run_idle(bat)
    out = []
    for p, job in zip(prompts, jobs):
        reply, done = _drain(job)
        want, state = _reference(p, tokens, 64, fast)
        assert done and reply == want, job.stats
        assert bat.ended[id(job)][0] == state, job.stats
        out.append(job.stats)
    return out


def _system(rng, n=600):
    return [int(t) for t in rng.integers(0, 1000, size=n)]


@needs_torch
@pytest.mark.parametrize("fast", [False, True], ids=["exact", "fast"])
@pytest.mark.parametrize("on", [False, True], ids=["today", "share"])
def test_second_session_resumes_at_the_system_end(monkeypatch, fast, on):
    import numpy as np

    rng = np.random.default_rng(21)
    system = _system(rng)
    bat = _fake(monkeypatch, fast=fast, share=_share() if on else None)
    a = system + [USER] + [int(t) for t in rng.integers(0, 1000, size=150)]
    b = system + [USER] + [int(t) for t in rng.integers(0, 1000, size=170)]
    (sa,) = _run(bat, [a], fast=fast)
    assert sa["cached"] == 0
    (sb,) = _run(bat, [b], fast=fast)
    want = 512 if not fast else 576                                # the page / 64-point at or before 600
    if not on:
        assert sb["cached"] == 0 and sa["marks"] == 0              # today: the first sharer prefills it all
        return
    assert sa["marks"] == 1 and sb["cached"] == want and sb.get("restored"), sb
    ix = bat.store.index
    e = ix.entries[sb["restored"]]
    assert e.kind == "prefix" and e.ids == system[:want]
    # the third session: the same mark again, and no second one is stored
    (sc,) = _run(bat, [system + [USER, 3, 4, 5] * 40], fast=fast)
    assert sc["cached"] == want and sum(1 for x in ix.entries.values() if x.kind == "prefix") == 1


@needs_torch
@pytest.mark.parametrize("fast", [False, True], ids=["exact", "fast"])
@pytest.mark.parametrize("wait", [False, True], ids=["nowait", "wait"])
def test_sessions_submitted_together_wait_for_the_partner(monkeypatch, fast, wait):
    import numpy as np

    rng = np.random.default_rng(22)
    system = _system(rng)
    bat = _fake(monkeypatch, n=4, fast=fast, share=_share(wait=wait))
    prompts = [system + [USER] + [int(t) for t in rng.integers(0, 1000, size=k)] for k in (150, 170, 130, 110)]
    got = _run(bat, prompts, fast=fast)
    want = 512 if not fast else 576
    if wait:
        assert got[0]["cached"] == 0 and all(s["cached"] == want and s.get("restored") for s in got[1:]), got
        assert bat.counts["prefix_wait"] > 0 and all(s.get("prefix_wait", 0) > 0 for s in got[1:])
    else:
        assert all(s["cached"] == 0 for s in got) and not bat.counts["prefix_wait"]
        ix = bat.store.index
        assert ix.counters["duplicate"] >= 3                        # each took the same mark
    later = system + [USER, 1, 2, 3] * 30
    (s,) = _run(bat, [later], fast=fast)
    assert s["cached"] == want


@needs_torch
def test_a_partner_that_goes_away_never_leaves_a_request_waiting(monkeypatch):
    """The partner is cancelled before its mark: the waiting request is admitted in the next round and prefills."""

    import numpy as np

    from test_batch_sessions_patches import _drain, _job, _reference

    rng = np.random.default_rng(23)
    system = _system(rng, 1500)
    bat = _fake(monkeypatch, n=4, share=_share())
    a = _job(system + [USER, 1, 2], 6, False, 64)
    b = _job(system + [USER, 3, 4], 6, False, 64)
    bat.queue.extend([a, b])
    bat._execute(*bat._plan())                                     # a admitted (piece 1 of 6), b waits
    assert bat.counts["prefix_wait"] == 1 and list(bat.queue) == [b]
    a.cancel = True
    rounds = 0
    while bat.queue or any(s is not None for s in bat.seqs):
        bat._execute(*bat._plan())
        rounds += 1
        assert rounds < 100
    reply, done = _drain(b)
    assert done and reply == _reference(b.prompt, 6, 64, False)[0] and b.stats["cached"] == 0
    assert bat.counts["prefix_wait"] == 1


def _agent_traffic(bat, *, rng, fast: bool, turns: int = 10, tokens: int = 8):
    """Agent-like sessions over 2 system prompts (1,100 and 700 tokens + role tokens): new sessions arrive (alone and
    in bursts), sessions continue (prompt + reply + an observation / user turn), fork, and are dropped; every reply
    and end state == fresh + serial (``_run``). -> every request's stats."""

    systems = [_system(rng, 1100), _system(rng, 700)]
    live: list[list[int]] = []
    stats = []
    for step in range(turns):
        batch = []
        for _ in range(int(rng.integers(1, 4))):
            kind = int(rng.integers(0, 4))
            if kind == 0 or not live:                               # a new session
                s = systems[int(rng.integers(0, 2))]
                batch.append(s + [USER] + [int(t) for t in rng.integers(0, 1000, size=int(rng.integers(5, 200)))]
                             + [ASSIST])
            elif kind == 1:                                         # a fork of a live one
                p = live[int(rng.integers(0, len(live)))]
                batch.append(p[:int(rng.integers(1, len(p)))] + [USER, int(rng.integers(0, 1000)), ASSIST])
            else:                                                   # the next turn
                p = live.pop(int(rng.integers(0, len(live))))
                batch.append(p + [int(t) for t in rng.integers(0, 5, size=tokens)] + [OBS]
                             + [int(t) for t in rng.integers(0, 1000, size=int(rng.integers(1, 90)))] + [ASSIST])
        got = _run(bat, [p[:3000] for p in batch], tokens=tokens, fast=fast)
        stats += got
        live += [p[:3000] for p in batch]
        assert bat.store.index.used <= bat.store.index.budget
    return stats


@needs_torch
@pytest.mark.parametrize("budget", [4000.0, 12.0], ids=["roomy", "tiny"])
@pytest.mark.parametrize("fast", [False, True], ids=["exact", "fast"])
def test_agent_traffic_exact_with_share(monkeypatch, fast, budget):
    """Every request exact with the switch on (keep 4 and keep 1: two system prompts evict each other's mark); on a
    roomy store, keep 4 resumes at least as many tokens as today on the same traffic."""

    import numpy as np

    got = {}
    for name, share in (("today", None), ("keep4", _share()), ("keep1", _share(keep=1))):
        bat = _fake(monkeypatch, n=3, fast=fast, budget=budget, share=share)
        stats = _agent_traffic(bat, rng=np.random.default_rng(31 + int(fast)), fast=fast)
        got[name] = sum(s["cached"] for s in stats)
        kinds = [e.kind for e in bat.store.index.entries.values()]
        if share is not None:
            assert kinds.count("prefix") <= share.keep
            if budget > 100:
                assert kinds.count("prefix") >= 1
        else:
            assert "prefix" not in kinds
    if budget > 100:
        assert got["keep4"] >= got["today"], got


@needs_torch
@pytest.mark.parametrize("fast", [False, True], ids=["exact", "fast"])
@pytest.mark.parametrize("pages", [96, 18], ids=["roomy", "tight"])
def test_agent_traffic_exact_on_the_kv_pool(monkeypatch, fast, pages):
    """The same traffic with every slot on 0290's pool (tight: 18 pages of 256 for 3 slots of up to 12 pages each,
    so admissions wait and idle slots spill): still exact; no page leaked, the null page clean."""

    import numpy as np

    kp = pytest.importorskip("test_kv_pool_patches")
    bat = kp._pool_batcher(monkeypatch, n=3, npages=pages, fast=fast, budget_pages=4000.0)
    bat.store.attach_share(_share())
    stats = _agent_traffic(bat, rng=np.random.default_rng(41 + int(fast)), fast=fast)
    assert any(s["cached"] and s.get("restored") for s in stats)
    if pages < 40:
        assert bat.counts["pool_wait"] + bat.counts["pool_spills"] > 0, bat.counts
    kp._pool_accounting(bat)


@needs_torch
@pytest.mark.parametrize("fast", [False, True], ids=["exact", "fast"])
def test_follower_replays_rank0_with_share(monkeypatch, fast):
    import numpy as np

    r0 = _fake(monkeypatch, n=3, fast=fast, budget=30.0, share=_share(keep=1))
    r1 = _fake(monkeypatch, n=3, fast=fast, budget=30.0, share=_share(keep=1), rank=1)
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
    _agent_traffic(r0, rng=np.random.default_rng(51), fast=fast, turns=8)
    with pytest.raises(Done):
        r1.follow()
    a, b = r0.store.index, r1.store.index
    assert a.digest() == b.digest() and sorted(a.entries) == sorted(b.entries) and a.counters == b.counters
    assert a.counters["evicted"] > 0 and r0.store.stats == r1.store.stats
    assert [st.state() for st in r0.states] == [st.state() for st in r1.states]
    assert r0.counts["pieces"] == r1.counts["pieces"]


# -- GPU (synthetic checkpoint) -------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")
    monkeypatch.setenv("GLM53_TF_NONEXPERT", "q4mse")
    monkeypatch.delenv("GLM53_TF_DEPTH", raising=False)


ROLE = 1010                                                         # inside the synthetic 1,024-token vocabulary


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    if not CUDA:
        pytest.skip("CUDA only")
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_prefix_share")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


def _engine(ckpt, monkeypatch, *, on=True, wait=True, **kw):
    from test_batch_sessions_patches import _engine as build

    monkeypatch.setenv("GLM53_TF_PREFIX_SHARE", "1" if on else "0")
    monkeypatch.setenv("GLM53_TF_PREFIX_SHARE_TOKENS", str(ROLE))
    monkeypatch.setenv("GLM53_TF_PREFIX_SHARE_MIN", "256")
    monkeypatch.setenv("GLM53_TF_PREFIX_SHARE_WAIT", "1" if wait else "0")
    return build(ckpt, **kw)


def _gpu_prompts(seed: int, system: int = 700, n: int = 4):
    import numpy as np

    rng = np.random.default_rng(seed)
    sys_prompt = [int(t) for t in rng.integers(0, 1000, size=system)]
    return [sys_prompt + [ROLE] + [int(t) for t in rng.integers(0, 1000, size=int(k))]
            for k in rng.integers(60, 200, size=n)]


@gpu
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
@pytest.mark.parametrize("fast", [False, True], ids=["exact", "fast"])
def test_gpu_new_session_resumes_at_the_system_end(ckpt, monkeypatch, greedy, fast):
    from test_batch_sessions_patches import _free, _sampling, _serial

    from test_batch_sessions_patches import _engine as build

    kw = dict(fast=True, rows=128, rows_max=256, piece=256, fork=64) if fast else dict(fork=64)
    ref = build(ckpt, batch=1, gib=0, on=False, **{k: v for k, v in kw.items() if k in ("fast", "rows", "rows_max")})
    eng = _engine(ckpt, monkeypatch, batch=3, **kw)
    assert eng.store.share is not None and eng.store.share.roles == [ROLE]
    sampling = _sampling(greedy)
    a, b, c, _ = _gpu_prompts(61 + int(fast))
    want = 512 if not fast else 640                                 # 700 on the page / 64-grid
    for i, p in enumerate((a, b, c)):
        (reply, stats), = eng.batch.generate_batch([dict(prompt=p, max_tokens=16, sampling=sampling, policy="auto")])
        assert reply == _serial(ref, p, sampling, 16), (i, stats)
        assert stats["cached"] == (0 if i == 0 else want), (i, stats)
    assert sum(1 for e in eng.store.index.entries.values() if e.kind == "prefix") == 1
    _free(ref, eng)


@gpu
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
def test_gpu_sessions_submitted_together_wait_and_resume(ckpt, monkeypatch, greedy):
    from test_batch_sessions_patches import _engine as build
    from test_batch_sessions_patches import _free, _sampling, _serial

    ref = build(ckpt, batch=1, gib=0, on=False)
    eng = _engine(ckpt, monkeypatch, batch=4, piece=128, fork=64)
    sampling = _sampling(greedy)
    prompts = _gpu_prompts(71)
    got = eng.batch.generate_batch([dict(prompt=p, max_tokens=16, sampling=sampling, policy=pol)
                                    for p, pol in zip(prompts, ("auto", "f3", "2", "o"))])
    for p, (reply, stats) in zip(prompts, got):
        assert reply == _serial(ref, p, sampling, 16), stats
    assert got[0][1]["cached"] == 0
    assert all(s["cached"] == 512 and s.get("prefix_wait", 0) > 0 for _, s in got[1:]), [s for _, s in got]
    _free(ref, eng)


@gpu
def test_gpu_follower_replays_with_share(ckpt, monkeypatch):
    from test_batch_sessions_patches import _free, _sampling

    r0 = _engine(ckpt, monkeypatch, batch=3, piece=128, fork=64)
    r1 = _engine(ckpt, monkeypatch, batch=3, piece=128, fork=64)
    r1.batch.costs = r0.batch.costs
    from tensorfold.families.glm5_next.cuda import batchplan as bp

    r1.batch.round_costs = bp.RoundCosts(r0.batch.costs)
    sent: list[list[int]] = []
    share = r0._share

    def record(values):
        got = share(values)
        sent.append(list(got))
        return got

    class Done(Exception):
        pass

    def replay(values):
        assert values is None
        if not sent:
            raise Done
        return sent.pop(0)

    r0._share, r1._share = record, replay
    prompts = _gpu_prompts(81)
    r0.batch.generate_batch([dict(prompt=p, max_tokens=12, sampling=_sampling(True)) for p in prompts])
    time.sleep(0.5)
    with pytest.raises(Done):
        r1.batch.follow()
    key = lambda d: (d["sha256"], tuple(d["keeps"]), d["arms"], d["cancelled"])      # noqa: E731
    assert [key(d) for d in r1.batch.log] == [key(d) for d in r0.batch.log]
    assert r1.store.index.digest() == r0.store.index.digest()
    del r0._share, r1._share
    _free(r0, r1)
