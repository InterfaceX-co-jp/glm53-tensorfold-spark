"""patches/0420 (``GLM53_TF_DRAFT_VOCAB``, ``_ARMS``, ``_FALLBACK``; ``glm5_next/cuda/draftvocab.py``): the drafters'
heads over a ranked token list (DECODE-PLAN E7). docs/DRAFT-VOCAB.md.

Checked:

- host only: the knobs (off / N / a file / file:N, arms, fallback; refusals), the list file format, each rank's rows
  (its half of the ranking, then the half's lowest unlisted ids, a multiple of 64 rows, the same count on both ranks
  for N), the shipped ranking, the settings both ranks compare (a different list changes them), the fallback rule's
  window (reset when the head starts over, the same on both ranks);
- torch on the CPU, two ranks in two threads over a fake all-gather:
  - ``take_rows`` gives the listed rows' words, scales and biases (dequantized equal), ``attach`` builds each rank's
    head from its own 4-bit rows;
  - the TARGET's sampling is untouched: ``decode.sample_rows`` (and ``batch.sample_multi``) with the list attached
    == without it, greedy and sampled, both ranks; a whole-vocabulary draft pass (the fallback) maps as upstream;
  - draft sampling over the listed rows: both ranks draw the same token; keyed Gumbel coupling: with draft logits
    equal to the target's, the draft is the target's token whenever that token is listed (greedy; sampled without a
    nucleus cut), and a listed token otherwise; ``batch.sample_drafts`` (one gather) == per-row ``Engine.sample``;
  - drafted == serial: the REAL ``decode.mtp_decode`` / ``draft`` / ``absorb`` / ``Engine.mtp`` / ``Engine.sample``
    on a fake model whose MTP head is the target plus noise (``mtp_stage`` / ``mtp_compute`` / ``commit`` stand-ins),
    greedy and sampled, trimmed with and without the fallback rule (a stretch of unlisted tokens switches a request
    to the whole vocabulary and back), and untrimmed: every reply == ``decode.serial_decode``'s, rank 0 == rank 1,
    no draft of a trimmed pass is unlisted;
- compile (no GPU, not under the interpreter): ``qmm._qmm`` for the trimmed head's shapes (12,288 / 16,384 / 24,576 rows
  x 4,096) for sm_121; no Triton source changed by the patch;
- GPU (TensorFold's synthetic checkpoints, one GPU playing rank 0 of two; run in the image): the trimmed head's logits
  == the full head's listed columns bit for bit (eager and through the MTP graphs), drafted replies with the list ==
  serial (lone and 4 batched requests, greedy and sampled, MTP and DFlash2 arms, the fallback forced), the replies
  equal to the untrimmed engine's.

Run: PYTHONPATH=<tree>/src:<tree>/tests/cuda:tests/cuda pytest -q tests/cuda/test_draft_vocab_patches.py
"""

from __future__ import annotations

import collections
import hashlib
import os
import threading
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tensorfold.families.glm5_next.cuda import draftvocab as dvm

try:
    import torch
except ImportError:          # the host-only tests still run
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="torch")
CUDA = torch is not None and torch.cuda.is_available()
gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")

V = 154880
HALF = V // 2


# -- host only -------------------------------------------------------------------------------------------------------
def test_knobs(tmp_path):
    assert dvm.settings({}) is None
    for off in ("", "0", "off", "OFF", "no"):
        assert dvm.settings({dvm.ENV: off}) is None
    s = dvm.settings({dvm.ENV: "49152"})
    assert s.half == 24576 and s.mtp and not s.dflash and (s.fb_rate, s.fb_window) == dvm.FB_DEFAULT
    assert s.source == str(dvm.RANKING) and len(s.order) > 30000
    s = dvm.settings({dvm.ENV: "32769", dvm.ARMS_ENV: "all", dvm.FB_ENV: "0.05,128"})
    assert s.half == 16385 and s.mtp and s.dflash and (s.fb_rate, s.fb_window) == (0.05, 128)
    s = dvm.settings({dvm.ENV: "1000", dvm.ARMS_ENV: "dflash", dvm.FB_ENV: "off"})
    assert not s.mtp and s.dflash and s.fb_rate > 1
    f = tmp_path / "ids.txt"
    f.write_text("# a list\n5 3 3 77441  # repeats keep their first place\n\n100\n")
    s = dvm.settings({dvm.ENV: str(f)})
    assert s.order == (5, 3, 77441, 100) and s.half == 0 and s.source == str(f)
    s = dvm.settings({dvm.ENV: f"{f}:256"})
    assert s.order == (5, 3, 77441, 100) and s.half == 128
    for bad in ("12", "154880", "999999", "x", str(tmp_path / "missing.txt"), "-5"):
        with pytest.raises(ValueError, match=dvm.ENV):
            dvm.settings({dvm.ENV: bad})
    with pytest.raises(ValueError, match=dvm.ARMS_ENV):
        dvm.settings({dvm.ENV: "49152", dvm.ARMS_ENV: "both"})
    for bad in ("1.5", "0.03,4", "x", "0.03,y"):
        with pytest.raises(ValueError, match=dvm.FB_ENV):
            dvm.settings({dvm.ENV: "49152", dvm.FB_ENV: bad})


def test_list_file(tmp_path):
    f = tmp_path / "l.txt"
    f.write_text("1 2 x\n")
    with pytest.raises(ValueError, match="not a token id"):
        dvm.read_ids(f)
    f.write_text("1 154880\n")
    with pytest.raises(ValueError, match="outside the vocabulary"):
        dvm.read_ids(f, V)
    f.write_text("# nothing\n")
    with pytest.raises(ValueError, match="no token ids"):
        dvm.read_ids(f)


def test_shipped_ranking():
    """The ranking ships with the patch: unique ids, the chat markup and end tokens early, most ids below 77,440."""

    order = dvm.read_ids(dvm.RANKING, V)
    assert len(order) == len(set(order)) > 30000
    first = set(order[:2000])
    for special in (154827, 154829, 154842, 154843, 154844, 154847, 154848, 154849, 154850):    # (<think> is prompt)
        assert special in first, special
    assert sum(1 for i in order[:32768] if i < HALF) > 0.85 * 32768


def test_rank_rows():
    s = dvm.settings({dvm.ENV: "49152"})
    got = [dvm.rank_ids(s, r, 2, V) for r in (0, 1)]
    assert [len(g) for g in got] == [24576, 24576]
    assert got[0].min() >= 0 and got[0].max() < HALF and got[1].min() >= HALF and got[1].max() < V
    assert len(set(got[0])) == 24576 and len(set(got[1])) == 24576
    # the half's ranking first, in order, then the half's lowest unlisted ids
    mine = [i for i in s.order if i >= HALF]
    k = min(len(mine), 24576)
    assert list(got[1][:k]) == mine[:k]
    rest = [i for i in range(HALF, V) if i not in set(mine[:k])][:24576 - k]
    assert list(got[1][k:]) == rest
    # odd N: rounded up to 64 rows; a file: the larger half decides, both ranks the same count
    s = dvm.settings({dvm.ENV: "1001"})
    assert [len(dvm.rank_ids(s, r, 2, V)) for r in (0, 1)] == [512, 512]
    s = dvm.Settings(tuple(range(100)) + (HALF + 1,), 0, True, False, 0.03, 256, "t")
    r0, r1 = (dvm.rank_ids(s, r, 2, V) for r in (0, 1))
    assert len(r0) == len(r1) == 128 and list(r0[:100]) == list(range(100)) and r1[0] == HALF + 1
    assert list(r1[1:]) == [HALF] + list(range(HALF + 2, HALF + 128))
    s = dvm.Settings(tuple(range(HALF)), 0, True, False, 0.03, 256, "t")
    with pytest.raises(ValueError, match="nothing to trim"):
        dvm.rank_ids(s, 0, 2, V)


def test_settings_both_ranks_compare(tmp_path):
    a = dvm.settings({dvm.ENV: "49152"})
    assert a.ints() == dvm.settings({dvm.ENV: "49152"}).ints()
    assert all(-2**31 <= v < 2**31 for v in a.ints())
    assert a.ints() != dvm.settings({dvm.ENV: "32768"}).ints()
    assert a.ints() != dvm.settings({dvm.ENV: "49152", dvm.ARMS_ENV: "all"}).ints()
    assert a.ints() != dvm.settings({dvm.ENV: "49152", dvm.FB_ENV: "0.04"}).ints()
    f = tmp_path / "r.txt"
    order = list(a.order)
    order[5], order[6] = order[6], order[5]
    f.write_text(" ".join(map(str, order)))
    assert a.ints() != dvm.settings({dvm.ENV: f"{f}:49152"}).ints()


def _w(s, rank: int = 0, *, vocab: int = V, comm=None, head_n: int | None = None):
    """A stand-in ``Weights`` with ``draft_vocab`` for ``s`` (no head: ``head.n`` only)."""

    w = SimpleNamespace(rank=rank, world=2, vocab_offset=rank * (vocab // 2), comm=comm, device="cpu",
                        cfg=SimpleNamespace(vocab=vocab, eos=(), dense_limit=10**9),
                        draft_head=None, head=SimpleNamespace(n=head_n or vocab // 2), mtp=object(), draft_vocab=None)
    if s is not None:
        ids = dvm.rank_ids(s, rank, 2, vocab)
        gids = torch.from_numpy(ids).to(torch.int32) if torch is not None else ids
        w.draft_vocab = dvm.DraftVocab(s, SimpleNamespace(n=len(ids)), gids, len(ids))
    return w


def test_fallback_rule():
    s = dvm.Settings(tuple(range(64)), 0, True, False, 0.25, 8, "t")
    w0, w1 = _w(s, 0, vocab=1024), _w(s, 1, vocab=1024)
    listed = dvm._listed(w0.draft_vocab, w0)
    assert listed == set(range(64)) | set(range(512, 576))
    st = SimpleNamespace(mtp_len=0)
    assert not dvm.full(w0, st, "mtp")
    dvm.track(w0, st, [1, 2, 3, 600, 700, 800])        # 3 of 6 unlisted: 0.5 > 0.25
    assert dvm.full(w0, st, "mtp") and not dvm.full(w0, st, "dflash")   # the DFlash2 arm is not trimmed
    st.mtp_len = 6
    dvm.track(w0, st, [1, 2, 3, 4, 5, 6])              # the window (8) keeps 700, 800 of the misses: 0.25, not above
    assert not dvm.full(w0, st, "mtp")
    dvm.track(w0, st, [900])                            # 800 leaves, 900 enters: still 2 of 8
    assert not dvm.full(w0, st, "mtp")
    dvm.track(w0, st, [901, 902])                       # 900, 901, 902: 3 of 8
    assert dvm.full(w0, st, "mtp")
    st.mtp_len = 0                                      # the head starts over: so does the window
    dvm.track(w0, st, [1])
    assert not dvm.full(w0, st, "mtp")
    # both ranks: the same answer from the same tokens (the rule reads both halves' lists)
    a, b = SimpleNamespace(mtp_len=0), SimpleNamespace(mtp_len=0)
    rng = np.random.default_rng(0)
    for _ in range(50):
        toks = [int(t) for t in rng.integers(0, 1024, size=int(rng.integers(1, 5)))]
        dvm.track(w0, a, toks)
        dvm.track(w1, b, toks)
        a.mtp_len = b.mtp_len = a.mtp_len + len(toks)
        assert dvm.full(w0, a, "mtp") == dvm.full(w1, b, "mtp")
    off = dvm.Settings(tuple(range(64)), 0, True, False, 2.0, 1, "t")
    w = _w(off, 0, vocab=1024)
    st = SimpleNamespace(mtp_len=0)
    dvm.track(w, st, [700] * 20)
    assert not dvm.full(w, st, "mtp") and not hasattr(st, "_dv_window")
    assert not dvm.full(SimpleNamespace(draft_vocab=None), st, "mtp")


# -- torch on the CPU ---------------------------------------------------------------------------------------------------
class _Pair:
    """Two ranks in two threads: an all-gather through shared slots."""

    def __init__(self) -> None:
        self.bar = threading.Barrier(2, timeout=60)
        self.slots = [None, None]


class _Comm:
    world = 2

    def __init__(self, pair: _Pair, rank: int) -> None:
        self.pair, self.rank = pair, rank

    def all_gather(self, send, recv) -> None:
        p = self.pair
        p.slots[self.rank] = send.detach().reshape(-1).clone()
        p.bar.wait()
        recv.view(-1).copy_(torch.cat([p.slots[0], p.slots[1]]))
        p.bar.wait()


def _both(fn):
    """fn(rank, comm) on two threads -> [rank 0's result, rank 1's]."""

    pair = _Pair()
    out, err = [None, None], []

    def run(r):
        try:
            out[r] = fn(r, _Comm(pair, r))
        except BaseException as exc:  # noqa: BLE001
            err.append(exc)
            pair.bar.abort()

    ts = [threading.Thread(target=run, args=(r,)) for r in (0, 1)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if err:
        raise err[0]
    return out


@needs_torch
def test_take_rows_and_attach(monkeypatch):
    pytest.importorskip("triton")
    from tensorfold.families.glm5_next.cuda import qmm

    g = torch.Generator().manual_seed(0)
    full = (torch.randn((640, 256), generator=g) * 0.05).to(torch.bfloat16)
    q = qmm.quantize4(full, mse=True)
    rows = torch.tensor([5, 0, 639, 64, 65, 127, 300, 301])
    t = dvm.take_rows(q, rows)
    assert (t.n, t.k) == (len(rows), 256)
    wf, sf, bf = qmm.to_mlx(q)
    wt, st_, bt = qmm.to_mlx(t)
    assert torch.equal(wt, wf[rows]) and torch.equal(st_, sf[rows]) and torch.equal(bt, bf[rows])
    assert torch.equal(qmm.dequantize_q4(t), qmm.dequantize_q4(q)[rows])
    with pytest.raises(ValueError):
        dvm.take_rows(q, torch.tensor([640]))
    # attach: each rank's rows from its own half (vocabulary 1,280: 640 a rank)
    s = dvm.Settings(tuple([3, 700, 5, 1279, 64] + list(range(100, 160))), 0, True, False, 0.03, 256, "t")
    for rank in (0, 1):
        w = _w(s, rank, vocab=1280)
        w.head = q
        dv = dvm.attach(w, s)
        ids = dvm.rank_ids(s, rank, 2, 1280)
        assert dv is w.draft_vocab and dv.n == len(ids) and torch.equal(dv.gids, torch.from_numpy(ids).int())
        assert torch.equal(qmm.dequantize_q4(dv.head), qmm.dequantize_q4(q)[torch.from_numpy(ids - rank * 640)])
        assert dvm.head_for(w, "mtp") is dv.head and dvm.head_for(w, "mtp", True) is q
        assert dvm.head_for(w, "dflash") is q                    # the arm is not trimmed
    monkeypatch.delenv(dvm.ENV, raising=False)
    w = _w(None, 0, vocab=1280)
    w.head = q
    assert dvm.attach(w) is None and w.draft_vocab is None and dvm.head_for(w, "mtp") is q


def _logits(seed: int, rows: int, vocab: int, hot=None) -> "torch.Tensor":
    """Rows of float logits over the whole vocabulary; ``hot`` ids get most of the mass."""

    g = torch.Generator().manual_seed(seed)
    x = torch.randn((rows, vocab), generator=g) - 4.0
    if hot is not None:
        x[:, hot] += 4.0 + torch.randn((rows, len(hot)), generator=g) * 2.0
    return x.to(torch.bfloat16).float()


def _sampling(kind: str):
    from tensorfold.engine.exact_sampling import Sampling

    return {"greedy": None, "sampled": Sampling(1234, 1.0, 20, 0.95), "nocut": Sampling(77, 0.8, 20, 1.0),
            "k0": Sampling(5, 1.0, 0, 1.0)}[kind]


VS = 1280          # the CPU tests' vocabulary: 640 ids a rank
LIST = tuple(list(range(0, 200, 2)) + list(range(640, 760, 3)) + [1279, 1001])


@needs_torch
@pytest.mark.parametrize("kind", ["greedy", "sampled", "nocut", "k0"])
def test_target_sampling_untouched(kind):
    """The target's rows sample the same with the list attached as without, on both ranks (sample_rows and the
    batched sample_multi); a whole-vocabulary DRAFT pass maps its columns as upstream."""

    from tensorfold.families.glm5_next.cuda import batch, decode

    s = dvm.Settings(LIST, 0, True, True, 0.03, 256, "t")
    sampling = _sampling(kind)
    full = _logits(1, 12, VS, hot=list(LIST[:40]) + [641, 1111, 7])
    pos = list(range(300, 312))

    def run(rank, comm, with_list):
        w = _w(s if with_list else None, rank, vocab=VS, comm=comm)
        mine = full[:, rank * 640:(rank + 1) * 640]
        a = decode.sample_rows(w, mine, pos, sampling)
        b = batch.sample_multi(w, mine, [(0, 5, pos[:5], sampling), (5, 7, pos[5:], sampling)])
        c = decode.sample_rows(w, mine, pos, sampling, draft=True)     # a fallback draft pass: whole head
        return a, b, c

    on = _both(lambda r, c: run(r, c, True))
    off = _both(lambda r, c: run(r, c, False))
    assert on[0] == on[1] == off[0] == off[1]
    a, b, c = on[0]
    assert a == b[0] + b[1] == c


@needs_torch
@pytest.mark.parametrize("kind", ["greedy", "sampled", "nocut", "k0"])
def test_trimmed_draft_sampling(kind):
    """Draft logits over the listed rows: both ranks draw the same listed token. Keyed coupling, with the draft's
    logits equal to the target's: greedy, the draft IS the target's token whenever that token is listed. Sampled,
    the draft draws from top-k / top-p over the LISTED ids, which can take in listed tail tokens outside the target's
    top-k / nucleus (and renormalizes the nucleus), so equality is the usual case, not a rule: an acceptance effect,
    never a reply's (the target's token is drawn from its own full rows)."""

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.glm5_next.cuda import batch, decode

    s = dvm.Settings(LIST, 0, True, True, 0.03, 256, "t")
    listed = set(int(i) for r in (0, 1) for i in dvm.rank_ids(s, r, 2, VS))
    sampling = _sampling(kind)
    R = 200
    full = _logits(2, R, VS, hot=list(LIST[:30]) + [641, 1111, 7, 9, 999])
    pos = list(range(1000, 1000 + R))

    def run(rank, comm):
        w = _w(s, rank, vocab=VS, comm=comm)
        mine = full[:, rank * 640:(rank + 1) * 640]
        cols = (w.draft_vocab.gids.long() - rank * 640)
        target = decode.sample_rows(w, mine, pos, sampling)
        probs: list[float] = []
        drafts = decode.sample_rows(w, mine[:, cols], pos, sampling, probs=probs, draft=True)
        want = [(r, pos[r], sampling, True) for r in range(R)]
        one = batch.sample_drafts(w, mine[:, cols], want)
        return target, drafts, probs, one

    (t0, d0, p0, o0), (t1, d1, p1, o1) = _both(run)
    assert (t0, d0, p0, o0) == (t1, d1, p1, o1)
    assert [d for d, _ in o0] == d0 and [p for _, p in o0] == pytest.approx(p0)
    assert all(d in listed for d in d0)
    hit = sum(t in listed for t in t0)
    assert 0 < hit < R                          # both cases occur
    same = [d == t for t, d in zip(t0, d0) if t in listed]
    if sampling is None:
        assert all(same)
    else:
        assert isinstance(sampling, Sampling) and sum(same) >= 0.8 * len(same), (sum(same), len(same))


# -- drafted == serial on a fake model ----------------------------------------------------------------------------------
def _h(*xs) -> int:
    return zlib.crc32(",".join(map(str, xs)).encode())


class _Model:
    """Target: logits of the next token from the last 3 tokens and the length (a hot set, listed or not by stretch);
    MTP head: the target's logits plus noise (so drafts miss sometimes)."""

    def __init__(self, hot_listed, hot_unlisted) -> None:
        self.a, self.b = list(hot_listed), list(hot_unlisted)
        self.cache: dict = {}

    def target(self, seq) -> "torch.Tensor":
        key = (tuple(seq[-3:]), len(seq))
        got = self.cache.get(key)
        if got is None:
            g = torch.Generator().manual_seed(_h(*key))
            x = torch.randn(VS, generator=g) - 6.0
            rare = (len(seq) // 24) % 4 == 2                # a stretch of unlisted tokens (a CJK passage)
            hot = self.b if rare else self.a
            x[hot] += 6.0 + torch.randn(len(hot), generator=g) * 1.5
            if not rare and len(seq) % 7 == 0:             # now and then one unlisted favourite
                x[self.b[len(seq) % len(self.b)]] += 9.0
            got = self.cache[key] = x.to(torch.bfloat16).float()
        return got

    def draft(self, seq) -> "torch.Tensor":
        g = torch.Generator().manual_seed(_h("d", *seq[-3:], len(seq)))
        return (self.target(seq) + torch.randn(VS, generator=g) * 0.7).to(torch.bfloat16).float()


class _St:
    def __init__(self, prompt) -> None:
        self.hist = list(prompt)
        self.pos = len(prompt)
        self.window: list[int] = []
        self.mtp_len = 0
        self.mtp_drafted = 0
        self.mtp_seq: list[int] = []
        self.prompt0 = list(prompt[:1])

    def set_mtp_len(self, n: int) -> None:
        self.mtp_len = n


def _engine(model, w, prompt, log):
    from tensorfold.families.glm5_next.cuda import decode

    e = object.__new__(decode.Engine)
    e.w, e.st = w, _St(prompt)
    e.buf, e.mbuf, e.graphs = None, SimpleNamespace(rows=64, zero_first=False), None
    e.longctx, e.draft_n = False, w.head.n
    e.last_hidden = torch.zeros((1, 1))
    e.main_hidden = lambda rows: torch.zeros((rows.stop - rows.start, 1))
    e.draft_hidden = lambda row: torch.zeros((1, 1))
    r = w.rank

    def forward(tokens):
        e.st.window = list(tokens)
        return torch.stack([model.target(e.st.hist + list(tokens[:i + 1]))[r * 640:(r + 1) * 640]
                            for i in range(len(tokens))])

    e.forward = forward
    e._fake = (model, log)
    return e


def _fake_stage(w, st, b, next_tokens, hidden):
    st.mtp_seq = st.mtp_seq[:st.mtp_len] + [int(t) for t in next_tokens]
    b.zero_first = False
    return len(next_tokens)


def _fake_compute(engines):
    def compute(w, st, b, n, *, full=False, **_):
        e = engines[w.rank]
        model, log = e._fake
        x = model.draft(st.prompt0 + st.mtp_seq)[w.rank * 640:(w.rank + 1) * 640]
        head = dvm.head_for(w, "mtp", full)
        dv = dvm.of(w)
        if dv is not None and head is dv.head:
            x = x[dv.gids.long() - w.vocab_offset]
        log.append(int(x.shape[0]))
        return x[None]
    return compute


def _fake_commit(w, st, b, R, keep):
    st.hist.extend(st.window[:keep])
    st.pos += keep


def _decode_pair(monkeypatch, s, sampling, *, prompt, tokens, mtp: bool, fb: bool = True):
    """Both ranks: prefill stand-in (the MTP head absorbs the prompt), the pending token, then serial or MTP-drafted
    decoding through the real loops. -> per rank (tokens, widths of the head passes, drafts, result)."""

    from tensorfold.families.glm5_next.cuda import decode

    hot_a = [i for i in LIST if i < VS][:60]
    hot_b = [7, 9, 11, 13, 641, 643, 1111, 1113, 999, 1201]
    model = _Model(hot_a, hot_b)
    engines = [None, None]
    monkeypatch.setattr(decode, "mtp_stage", _fake_stage)
    monkeypatch.setattr(decode, "mtp_compute", _fake_compute(engines))
    monkeypatch.setattr(decode, "commit", _fake_commit)
    monkeypatch.setattr(decode, "_sync", lambda w: None)
    monkeypatch.setattr(decode, "_forward_sync", lambda: None)
    drafts_seen: list = [[], []]
    real_draft = decode.draft

    def spy(e, *a, **k):
        d = real_draft(e, *a, **k)
        drafts_seen[e.w.rank].append((list(d), e._fake[1][-1] if e._fake[1] else None))
        return d

    monkeypatch.setattr(decode, "draft", spy)

    def run(rank, comm):
        w = _w(s, rank, vocab=VS, comm=comm)
        if s is not None and not fb:
            w.draft_vocab.s = dvm.Settings(s.order, s.half, s.mtp, s.dflash, 2.0, 1, s.source)
        log: list[int] = []
        e = _engine(model, w, prompt, log)
        engines[rank] = e
        decode.absorb(e, torch.zeros((len(prompt) - 1, 1)), prompt[1:])     # prefill: the head takes the prompt
        pending = e.sample(model.target(list(prompt))[None, rank * 640:(rank + 1) * 640], [len(prompt)], sampling)[0]
        if mtp:
            res = decode.mtp_decode(e, pending, tokens, sampling,
                                    policy=decode.DepthPolicy(4, fixed=True, confidence=0.0))
        else:
            res = decode.serial_decode(e, pending, tokens, sampling)
        return res.tokens, log, list(drafts_seen[rank]), res

    return _both(run)


@needs_torch
@pytest.mark.parametrize("kind", ["greedy", "sampled"])
@pytest.mark.parametrize("mode", ["trimmed", "trimmed-no-fallback", "untrimmed"])
def test_mtp_drafted_equals_serial(monkeypatch, kind, mode):
    s = None if mode == "untrimmed" else dvm.Settings(LIST, 0, True, False, 0.2, 16, "t")
    sampling = _sampling(kind)
    listed = set(int(i) for r in (0, 1) for i in dvm.rank_ids(s, r, 2, VS)) if s is not None else set()
    for seed in (1, 2, 3):
        prompt = [int(x) for x in np.random.default_rng(seed).choice(np.array(LIST[:60]), size=30)]
        serial = _decode_pair(monkeypatch, s, sampling, prompt=prompt, tokens=120, mtp=False)
        drafted = _decode_pair(monkeypatch, s, sampling, prompt=prompt, tokens=120, mtp=True,
                               fb=mode != "trimmed-no-fallback")
        assert serial[0][0] == serial[1][0] and drafted[0][0] == drafted[1][0]      # rank 0 == rank 1
        assert drafted[0][0] == serial[0][0], (mode, kind, seed)                    # drafted == serial
        assert drafted[0][1] == drafted[1][1] and drafted[0][2] == drafted[1][2]    # the same passes and drafts
        res = drafted[0][3]
        assert res.accepted > 0 and res.rounds < 119
        widths = drafted[0][1]
        if s is None:
            assert set(widths) == {640}
            continue
        n = len(dvm.rank_ids(s, 0, 2, VS))
        assert n in widths
        # a trimmed pass never drafts an unlisted token
        for d, width in drafted[0][2]:
            if width == n:
                assert all(t in listed for t in d)
        unlisted = [t for t in serial[0][0] if t not in listed]
        assert unlisted                         # the reply has unlisted tokens (the rare stretches)
        if mode == "trimmed":
            assert 640 in widths                # the rule switched to the whole vocabulary ...
            assert n in widths[widths.index(640):]      # ... and back after the stretch
        else:
            assert 640 not in widths


@needs_torch
def test_mtp_chains_pass_is_full_when_any_slot_falls_back(monkeypatch):
    """``MtpChains`` (GLM53_TF_BATCH_MTP): the slots' backlogs feed their rule windows, and one head pass serves
    every slot, so the pass (and its chained steps) reads the whole vocabulary when any slot's rule says so."""

    from test_batch_parallel_patches import _Head, _HeadEngine, _HeadSt

    from tensorfold.families.glm5_next.cuda import batch

    s = dvm.Settings(tuple(range(64)), 0, True, False, 0.25, 8, "t")
    w = _w(s, 0, vocab=1024)
    head = _Head()
    calls = []

    def fake_multi(w_, sts, b, tokens, hidden, full=False):
        calls.append(full)
        out = []
        for i, (st, t, h) in enumerate(zip(sts, tokens, hidden)):
            lg, o = head.rows(st, t, h)
            b.fnormed[i, 0] = o
            out.append(lg)
        return torch.tensor(out, dtype=torch.int64).view(-1, 1)

    monkeypatch.setattr(batch, "mtp_multi", fake_multi)
    e = _HeadEngine(head)
    e.w = w
    bat = SimpleNamespace(g=SimpleNamespace(e=e), _on=None)

    def round_(backlogs):
        chains = batch.MtpChains()
        sts = []
        for s_, toks in enumerate(backlogs):
            st = states[s_]
            q = SimpleNamespace(slot=s_, job=SimpleNamespace(sampling=None), m_policy=SimpleNamespace(confidence=0.0),
                                opt=None, m_rows=torch.zeros((32, 1), dtype=torch.int64), m_next=list(toks), arm="s",
                                drafts=[], steps=0, backlog=0)
            chains.add(q, st, len(toks), 3)
            sts.append(st)
        calls.clear()
        assert chains.run(bat)
        return list(calls)

    states = [_HeadSt(40), _HeadSt(40)]
    assert round_([[1, 2, 3], [4, 5]]) == [False] * 3           # a pass and two chained steps, all trimmed
    assert round_([[1, 2], [700, 701, 702, 703]]) == [True] * 3  # slot 1's window: 4 of 6 unlisted
    assert dvm.full(w, states[1], "mtp") and not dvm.full(w, states[0], "mtp")
    assert round_([[1], [5, 6, 7, 8, 9, 10, 11, 12]]) == [False] * 3        # slot 1's window refilled


# -- compile (no GPU) ---------------------------------------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") == "1", reason="compiles kernels")
@pytest.mark.parametrize("n", [12288, 16384, 24576, 77440])
def test_qmm_compiles_for_trimmed_heads(n):
    """The trimmed head runs the upstream 4-bit matmul (no Triton source changed); its new N specializations
    compile for sm_121, with the K split rule's slices (1 from 12,288 rows up: the listed columns' bits)."""

    triton = pytest.importorskip("triton")
    pytest.importorskip("torch")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from tensorfold.families.glm5_next.cuda import qmm

    k = 4096
    assert qmm.split_k(n, k) == qmm.split_k(HALF, k) == 1
    for bm in (16, 32):
        gpi, warps, stages = qmm.CONFIG[bm]
        sig = {"X": "*bf16", "XS": "*fp32", "W": "*i32", "S": "*bf16", "B": "*bf16", "OUT": "*bf16", "PART": "*bf16",
               "M": "i32", "x_stride": "i32"}
        cst = {"N": n, "K": k, "SK": 1, "BM": bm, "BLOCK_N": qmm.BN, "GPI": qmm.gpi_for(k // 64, gpi), "F32": False}
        for key in cst:
            sig[key] = "constexpr"
        c = triton.compile(ASTSource(fn=qmm._qmm, signature=sig, constexprs=cst), target=GPUTarget("cuda", 121, 32),
                           options={"num_warps": warps, "num_stages": stages})
        assert ".entry" in c.asm["ptx"]


# -- GPU: TensorFold's synthetic checkpoints (one GPU playing rank 0 of two) ---------------------------------------------
@pytest.fixture(scope="module")
def dckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_dvocab")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    ranking = path / "ranking.txt"
    # the synthetic vocabulary (1,024: 512 a rank): a list of 128 a half, frequent-first by a fixed shuffle
    order = [int(i) for i in np.random.default_rng(7).permutation(1024)]
    ranking.write_text(" ".join(map(str, order)))
    return path


def _gpu_engine(path, *, vocab: str | None, arms: str = "mtp", fallback: str = "off", batch: int = 1):
    from test_batch2_patches import _engine as engine

    with pytest.MonkeyPatch.context() as m:
        m.setenv("GLM53_TF_NONEXPERT", "q4mse")
        m.setenv("GLM53_TF_DEPTH", "cost")
        m.setenv("GLM53_TF_BATCH_MTP", "1")
        m.setenv("GLM53_TF_BATCH_ROW_MS", "6.5")
        if vocab is None:
            m.delenv(dvm.ENV, raising=False)
        else:
            m.setenv(dvm.ENV, vocab)
            m.setenv(dvm.ARMS_ENV, arms)
            m.setenv(dvm.FB_ENV, fallback)
        return engine(path, batch=batch)


@gpu
def test_gpu_trimmed_head_bits(dckpt):
    """The MTP head's logits over the list == the full head's listed columns, bit for bit, eager and graphed."""

    from tensorfold.families.glm5_next.cuda.decode import prefill

    eng = _gpu_engine(dckpt, vocab=f"{dckpt / 'ranking.txt'}:256", fallback="0.5,64")
    e = eng.e
    dv = dvm.of(e.w)
    assert dv is not None and dv.n == 128 and e.graphs.mtp and e.graphs.mtp_full
    prompt = [int(t) for t in np.random.default_rng(3).integers(0, 1000, size=200)]
    prefill(e, prompt, None, mtp=True)
    hid = (torch.randn((4, e.w.cfg.hidden), generator=torch.Generator().manual_seed(1)) * 0.5).to(torch.bfloat16).cuda()
    cols = dv.gids.long() - e.w.vocab_offset
    for n in (1, 2, 3, 4):
        st = e.st
        base = st.mtp_len
        st._dv_window = None
        trimmed = e.mtp(prompt[:n], hid[:n]).clone()           # graphed, trimmed
        st.set_mtp_len(base)
        st._dv_window = [collections.deque([True] * 64, maxlen=64), 64, None]
        whole = e.mtp(prompt[:n], hid[:n]).clone()             # graphed, the fallback's whole head
        st.set_mtp_len(base)
        st._dv_window = None
        assert trimmed.shape[1] == 128 and whole.shape[1] == e.w.head.n
        assert torch.equal(trimmed, whole[:, cols]), n


@gpu
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
@pytest.mark.parametrize("arms,fallback", [("mtp", "off"), ("all", "off"), ("all", "0.01,32")])
def test_gpu_drafted_replies_equal_serial(dckpt, greedy, arms, fallback):
    """Lone and 4 batched requests (o, of7, om7, l7:3) with the list == serial == the untrimmed engine's replies."""

    from test_batch2_patches import _free, _repeated, _run, _sampling, _serial

    sampling = _sampling(greedy)
    plain = _gpu_engine(dckpt, vocab=None)
    prompts = [_repeated(300 + i, block=40, times=3, tail=8 + i) for i in range(4)]
    want = [_serial(plain, p, sampling, 64) for p in prompts]
    base = [[_run(plain, p, sampling, policy=pol, tokens=64)[0] for pol in ("o", "of7", "om7")] for p in prompts]
    assert all(b == [w_] * 3 for b, w_ in zip(base, want))
    del plain
    lone = _gpu_engine(dckpt, vocab=f"{dckpt / 'ranking.txt'}:256", arms=arms, fallback=fallback)
    for p, w_ in zip(prompts, want):
        assert _serial(lone, p, sampling, 64) == w_
        for pol in ("o", "of7", "om7", "l7:3"):
            got, _ = _run(lone, p, sampling, policy=pol, tokens=64)
            assert got == w_, pol
    del lone
    b4 = _gpu_engine(dckpt, vocab=f"{dckpt / 'ranking.txt'}:256", arms=arms, fallback=fallback, batch=4)
    try:
        got = b4.batch.generate_batch([dict(prompt=p, max_tokens=64, sampling=sampling, policy=pol)
                                       for p, pol in zip(prompts, ("o", "om7", "of7", "l7:3"))])
        assert [t for t, _ in got] == want
    finally:
        _free(b4)


def test_digest_is_stable():
    """The ranks compare a digest of the list: a function of the ids only (not of the path or the file layout)."""

    a = dvm.Settings((1, 2, 3), 0, True, False, 0.03, 256, "a")
    b = dvm.Settings((1, 2, 3), 0, True, False, 0.03, 256, "b")
    assert a.ints() == b.ints()
    d = hashlib.sha256(b"1,2,3").digest()
    assert a.ints()[-1] == int.from_bytes(d[:4], "little") & 0x7FFFFFFF
    assert Path(dvm.RANKING).name == "draft_vocab.txt"
