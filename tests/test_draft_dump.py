"""patches/0430 (GLM53_TF_DRAFT_DUMP) without a GPU: the dump's records through ``train/dumpdata.py``.

A fake two-rank engine feeds ``dump.Dumper``: prefill chunks of a prompt (the second one resumed, as a batch piece
or a session), then decode rounds of two slots, one of them a second request resuming the first's prompt. Checks:

- the merged top-k equals the top-k of the full-vocabulary log-softmax (each rank holds half of the vocabulary);
- documents assemble by prefix hashes: the resumed request links to the first request's prompt segments, each row is
  owned once, tokens / hidden rows / taps / lp come back exactly (fp8 taps within e4m3 rounding);
- nothing is written before ``arm`` and nothing on rank 1.

    TF_SRC=<patched tree>/src python -m pytest -q tests/test_draft_dump.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train"))
TF_SRC = Path(os.environ.get("TF_SRC", "/src/TensorFold/src"))
DUMP_PY = TF_SRC / "tensorfold/families/glm5_next/cuda/dump.py"

D, V, TAPS, K = 64, 200, 2, 8


def _load(tmp, taps="bf16"):
    if not DUMP_PY.exists():
        pytest.skip("TF_SRC: a patched TensorFold source tree (patches/0430)")
    os.environ.update(GLM53_TF_DRAFT_DUMP=str(tmp), GLM53_TF_DRAFT_DUMP_TOPK=str(K), GLM53_TF_DRAFT_DUMP_TAPS=taps)
    spec = importlib.util.spec_from_file_location(f"dump_{taps}", DUMP_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.FLUSH_ROWS = 5
    return mod


class TwoRanks:
    """Rank 0's all-gather: its rows, then what rank 1 would send for the same rows (``other``)."""

    def __init__(self) -> None:
        self.other = None

    def all_gather(self, send, recv):
        n = send.numel()
        recv[:n] = send
        recv[n:] = self.other.reshape(-1)


def _setup(mod, head):
    comm = TwoRanks()
    cfg = SimpleNamespace(hidden=D, vocab=V, layers=3, eos=(1,))
    w0 = SimpleNamespace(cfg=cfg, rank=0, world=2, comm=comm, vocab_offset=0, meta={})
    w1 = SimpleNamespace(cfg=cfg, rank=1, world=2, comm=None, vocab_offset=V // 2, meta={})
    d0 = mod.Dumper(w0, os.environ["GLM53_TF_DRAFT_DUMP"], (1, 2))
    d1 = mod.Dumper(w1, os.environ["GLM53_TF_DRAFT_DUMP"], (1, 2))

    def head_rows(hidden):          # rank 0's half; the fake comm carries rank 1's
        logits = (hidden.float() @ head.t()).to(torch.bfloat16)
        comm.other = d1._local_topk(logits[:, V // 2:])
        return d0._local_topk(logits[:, :V // 2])

    d0._head_rows = head_rows
    return d0, comm, d1


def _engine(hidden, taps):
    st = object()
    return SimpleNamespace(st=st, main_hidden=lambda sl: hidden[sl], tap_rows=lambda n: taps[:n])


@pytest.mark.parametrize("taps", ["bf16", "fp8"])
def test_dump_round_trip(tmp_path, taps):
    mod = _load(tmp_path, taps)
    g = torch.Generator().manual_seed(0)
    head = torch.randn(V, D, generator=g) * 0.3
    d0, comm, d1 = _setup(mod, head)
    prompt = torch.randint(2, V, (23,), generator=g).tolist()
    Hs = torch.randn(40, D, generator=g).to(torch.bfloat16)
    Ts = torch.randn(40, TAPS * D, generator=g).to(torch.bfloat16)

    # before arm: nothing
    e = _engine(Hs[:10], Ts[:10])
    d0.prefill_chunk(e, prompt, 0, 10, 0)
    assert d0.stats["prefill_rows"] == 0
    d0.armed = True
    # request A: prompt in two pieces (the second resumed at 10), then 3 decode rounds on slot A
    d0.prefill_chunk(e, prompt, 0, 10, 0)
    e2 = SimpleNamespace(st=e.st, main_hidden=lambda sl: Hs[10:23][sl], tap_rows=lambda n: Ts[10:23][:n])
    d0.prefill_chunk(e2, prompt, 10, 23, 10)
    reply = torch.randint(2, V, (7,), generator=g).tolist()
    buf_h = Hs[23:30]
    buf_t = [Ts[23:30, :D], Ts[23:30, D:]]
    eb = SimpleNamespace(buf=SimpleNamespace(fnormed=buf_h, taps=buf_t))
    logits = (buf_h.float() @ head.t()).to(torch.bfloat16)
    pos = 23
    for off, keep in ((0, 3), (3, 2), (5, 2)):
        rows = torch.arange(off, off + keep)
        comm.other = d1._local_topk(logits[rows][:, V // 2:])
        d0.decode_round(eb, logits[:, :V // 2], [(e.st, off, keep, reply[off:off + keep], pos)])
        pos += keep
    # request B on another slot: resumes A's prompt at 23 (a session), its own 4 tokens by prefill
    eB = SimpleNamespace(st=object(), main_hidden=lambda sl: Hs[30:34][sl], tap_rows=lambda n: Ts[30:34][:n])
    promptB = prompt + [5, 6, 7, 8]
    d0.prefill_chunk(eB, promptB, 23, 27, 23)
    d0.close()

    import dumpdata

    dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(dirs) == 1                                     # rank 1 wrote nothing
    docs = dumpdata.build_docs(dirs)
    assert sorted(doc.n for doc in docs) == [27, 30]
    a = next(doc for doc in docs if doc.n == 30)
    b = next(doc for doc in docs if doc.n == 27)
    assert a.tokens().tolist() == prompt + reply
    assert b.tokens().tolist() == promptB
    assert a.owned_mask().sum() + b.owned_mask().sum() == 34          # every dumped row counted once
    assert a.kinds().tolist() == [0] * 23 + [1] * 7
    assert torch.equal(a.get("hidden", 0, 30), Hs[:30])
    tp = a.taps(0, 30)
    if taps == "bf16":
        assert torch.equal(tp, Ts[:30])
    else:
        assert torch.allclose(tp.float(), Ts[:30].float(), rtol=0.07, atol=1e-6)
    sub = a.taps(0, 30, [2])                                   # one tap of the two (a 9-tap dump for a 5-tap drafter)
    assert torch.equal(sub, tp[:, D:])
    # lp / ids: the top-k of the full log-softmax, both vocabulary halves merged
    full = torch.log_softmax(torch.cat([(Hs[:30].float() @ head.t()).to(torch.bfloat16).float()]), dim=-1)
    want_v, want_i = torch.topk(full, K, dim=-1)
    got_v, got_i = a.get("lp", 0, 30).float(), a.get("ids", 0, 30).long()
    # (bf16 logits tie often at this size: compare the values the ids point at, not the order among equals)
    assert torch.allclose(full.gather(1, got_i), want_v, atol=1e-5)
    assert torch.allclose(got_v, want_v, atol=2e-3)
    assert (got_i == want_i).float().mean() > 0.9
    assert torch.equal(b.get("hidden", 23, 27), Hs[30:34])
