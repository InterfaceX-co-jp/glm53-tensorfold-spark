"""patches/0450 (GLM53_TF_GPU_ROUND ``resident`` / ``kda``): the real ``Batcher`` running GPU-resident rounds on a fake
model, on the CPU (Triton's interpreter runs every device kernel of the round: the stage, the sampler, the accept rule,
the backlog copies, the depth decisions and the DFlash2 chain).

The fake model is hostile where it matters: a token's logits read a hash of every token before it (a KDA-like state
with two parity buffers, per-row saves and replays, conv windows), the MTP head and the DFlash2 drafter read the
backlog rows and taps they are given and check them against what the committed positions really were (a wrong row, a
missed row or a row too many fails the test at once), the MTP head's cache and the drafter's context are checked
position by position when a request ends. Both ranks hold half of the vocabulary: every draw goes through the real
exchange (a fake communicator between two threads) and the device sampler.

Checked, with ``resident`` alone and with ``kda``, MTP chains batched or not, one rank and two ranks:

- every reply (greedy and sampled; auto with cost depths, MTP-only, DFlash2-only, serial; stop at EOS, max_tokens)
  equals serial decoding of the same prompt, and every slot's committed state at its end equals the serial one;
- resident rounds ran, and drained for arrivals (rank 0's flag), finished requests and context-mode boundaries, and
  the batch went resident again afterwards;
- the two ranks end the same requests the same way (sha, keeps, drafters) through the same rounds;
- while a round is launched, the driver never reads the device (no ``.item()`` / ``.cpu()`` / ``.tolist()`` /
  ``numpy()`` from ``resident.py`` / ``gpuround.py`` / ``batch.py`` between the process step and the next).

Run: TRITON_INTERPRET=1 PYTHONPATH=<patched TensorFold>/src:tests:tests/cuda \
    pytest -q tests/cuda/test_gpu_round_resident.py
"""

from __future__ import annotations

import collections
import contextlib
import os
import queue
import random
import sys
import threading
import time
from types import SimpleNamespace

os.environ.setdefault("TRITON_INTERPRET", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows  # noqa: E402
from tensorfold.families.glm5_next.cuda import batchplan  # noqa: E402
from tensorfold.families.glm5_next.cuda import gpuround as gr  # noqa: E402

CUDA = torch.cuda.is_available()
INTERP = os.environ.get("TRITON_INTERPRET") == "1" and type(gr._accept_kernel).__name__ == "InterpretedFunction"
interp = pytest.mark.skipif(not INTERP or CUDA, reason="Triton's CPU interpreter, no GPU")

V = 48                    # vocabulary
D = 8                     # hidden width (8 bytes: a hash, exactly, in bf16)
NT = 2                    # DFlash2 tap layers
EOS = (5,)
LIMIT = 10 ** 6           # the fake's "dense limit" (context modes; a test lowers it)
BACKLOG_ROWS = 32
COSTS = {"verify": [29.2, 38.9, 45.2, 55.2, 59.5, 63.2, 65.6, 69.2, 72.0, 75.1, 78.0, 80.5, 83.0, 85.2, 87.0, 89.1],
         "mtp": 1.7, "mtp_step": 1.5, "mtp_row": 0.1, "block": 3.0, "taps_row": 0.05}
M61 = (1 << 61) - 1


@pytest.fixture(autouse=True, scope="module")
def _interp():
    import gpuround_interp

    gpuround_interp.install()


# -- the fake model ----------------------------------------------------------------------------------------------------
def mix(*v: int) -> int:
    h = 1469598103934665603
    for x in v:
        h = (h * 1099511628211 + int(x) + 12345) % M61
    return h


H0 = mix(7)


def logits_full(h: int) -> np.ndarray:
    """A peaked, bf16-representable distribution over V from a state (ties happen)."""

    x = np.array([((mix(h, j) % 64) - 32) / 8.0 for j in range(V)], dtype=np.float32)
    x[h % V] += 3.5
    x[(h // V) % V] += 2.0
    return x


def encode(h: int) -> torch.Tensor:
    return torch.tensor([(h >> (8 * i)) & 0xFF for i in range(D)], dtype=torch.bfloat16)


def decode_row(row: torch.Tensor) -> int:
    v = [int(x) for x in row.float().tolist()]
    return sum(b << (8 * i) for i, b in enumerate(v))


def draw(x: np.ndarray, position: int, s, world: int) -> int:
    """The engine's draw on a full logits row: every rank's top k + MARGIN gathered (``torch.topk`` on its part: a tie
    at a rank's edge resolves as topk does, on the host path too), then the keyed rule."""

    greedy = s is None or s.temperature <= 0
    part = V // world
    k = 1 if greedy else min(part, s.top_k + MARGIN)
    vals, ids = [], []
    for r in range(world):
        half = torch.from_numpy(x[r * part:(r + 1) * part]).to(torch.bfloat16)
        v, i = torch.topk(half.float(), k)
        vals += v.tolist()
        ids += (i + r * part).tolist()
    vals = np.array(vals, dtype=np.float32)[None]
    ids = np.array(ids, dtype=np.int64)[None]
    if greedy:
        o = np.lexsort((ids, -vals), axis=-1)
        return int(ids[0, o[0, 0]])
    return choose_rows(vals, ids, [position], s)[0]


def serial(prompt: list[int], n: int, s, stop_eos: bool, world: int) -> tuple[list[int], int]:
    """Serial decoding: (the reply's tokens, the committed state after them)."""

    h = H0
    for p, t in enumerate(prompt):
        h = mix(h, t, p)
    out = [draw(logits_full(h), len(prompt), s, world)]
    pos = len(prompt)
    while len(out) < n and not (stop_eos and out[-1] in EOS):
        h = mix(h, out[-1], pos)
        pos += 1
        out.append(draw(logits_full(h), pos, s, world))
    return out, h


class FakeState:
    """A slot's State: a KDA-like hash with parity buffers and per-row saves (two sets: the State's and the ``kda``
    second one), conv windows, positions on the host and the device, the MTP head's cache, and the logs the checks
    read (the committed hash and token of every position)."""

    def __init__(self, cap: int = 4096) -> None:
        self.capacity = cap
        self.rec = torch.zeros((2,), dtype=torch.int64)
        self.cur = [0]
        self.conv = torch.zeros((3,), dtype=torch.int64)
        self.proj = torch.zeros((1, 64), dtype=torch.int64)       # [layers, rows]: ``_borrow`` swaps it
        self.scratch = []
        self.saves = {0: torch.zeros((64,), dtype=torch.int64), 1: torch.zeros((64,), dtype=torch.int64)}
        self.pos = 0
        self.pos_dev = torch.zeros((1,), dtype=torch.int32)
        self.mtp_len = self.mtp_drafted = 0
        self.mtp_pos_dev = torch.zeros((1,), dtype=torch.int32)
        self.mhist = torch.full((cap,), -1, dtype=torch.int64)
        self.mhid = torch.full((cap,), -1, dtype=torch.int64)       # the hidden the MTP head absorbed at each position
        self.hlog = torch.zeros((cap,), dtype=torch.int64)
        self.tlog = torch.full((cap,), -1, dtype=torch.int64)
        self.index = None
        self.index_ring = 0
        self.pages = None
        self.win: list[int] = []
        self.rec[0] = H0
        self.scratch_set = SimpleNamespace(layers=1, heads=1, rows=64)

    def reset(self) -> None:
        self.rec.zero_()
        self.rec[0] = H0
        self.cur = [0]
        self.conv.zero_()
        self.set_pos(0)
        self.set_mtp_len(0)
        self.mtp_drafted = 0

    def set_pos(self, n: int, *, committed: bool = False) -> None:
        self.pos = int(n)
        self.pos_dev.fill_(int(n))

    def set_mtp_len(self, n: int) -> None:
        self.mtp_len = int(n)
        self.mtp_pos_dev.fill_(int(n))

    def save_index_tail(self) -> None:
        pass

    def committed(self) -> int:
        return int(self.rec[self.cur[0]])


class Model:
    """The main model on host windows (the normal path, prefill) and on staged device rows (resident rounds)."""

    def __init__(self, rank: int, world: int) -> None:
        self.rank, self.world = rank, world
        self.half = V // world

    def part(self, h: int) -> torch.Tensor:
        x = logits_full(h)
        return torch.from_numpy(x[self.rank * self.half:(self.rank + 1) * self.half].copy()).to(torch.bfloat16)

    def run_rows(self, st: FakeState, b, h: int, pos: int, tokens: list[int], off: int, saves: torch.Tensor,
                 real: int | None = None) -> tuple[torch.Tensor, int]:
        """Rows from state ``h`` at ``pos``: saves, hidden rows and taps at ``off``, logits. -> (logits, last h)."""

        out = []
        for r, t in enumerate(tokens):
            if real is not None and r >= real:              # a padded row ``kda`` does not run: garbage
                out.append(torch.zeros((self.half,), dtype=torch.bfloat16))
                continue
            h = mix(h, t, pos + r)
            saves[r] = h
            st.proj[0, r] = t
            b.fnormed[off + r] = encode(h)
            b.taps[0][off + r] = encode(h)
            b.taps[1][off + r] = encode(h ^ 0x5A5A)
            out.append(self.part(h))
        return torch.stack(out), h

    # decode.prefill's stage / compute / commit
    def stage(self, w, st, b, tokens):
        st.win = [int(t) for t in tokens]
        return len(st.win)

    def compute(self, w, st, b, R, **kw):
        lg, h = self.run_rows(st, b, st.committed(), st.pos, st.win[:R], 0, st.saves[0])
        st.rec[1 - st.cur[0]] = h
        return lg

    def commit(self, w, st, b, R, keep):
        """``forward.commit`` (host keep): the replay into the other buffer, the flip, the conv window, positions."""

        if not 1 <= keep <= R:
            raise ValueError("keep must be in 1..R")
        cur = st.cur[0]
        if keep < R:
            st.rec[1 - cur] = st.saves[0][keep - 1]
        for r in range(keep):
            st.hlog[st.pos + r] = st.saves[0][r]
            st.tlog[st.pos + r] = st.proj[0, r]
        st.cur = [1 - cur]
        st.conv[:] = torch.cat([st.conv, st.proj[0, :keep]])[-3:]
        st.set_pos(st.pos + keep, committed=True)


class Comm:
    """Two ranks' all-gather in threads (a rendezvous): rank r's part at [r * n, (r + 1) * n)."""

    def __init__(self, box, rank: int) -> None:
        self.box, self.rank = box, rank

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        box = self.box
        n = send.numel()
        with box.cv:
            gen = box.gen
            box.parts[self.rank] = send.detach().clone().view(-1)
            box.cv.notify_all()
            ok = box.cv.wait_for(lambda: box.gen != gen or all(p is not None for p in box.parts), timeout=120)
            if not ok:
                raise TimeoutError("the other rank never came to this exchange")
            if box.gen == gen:
                box.result = [p.clone() for p in box.parts]
                box.parts = [None, None]
                box.gen += 1
                box.cv.notify_all()
            parts = box.result
        for r, p in enumerate(parts):
            if p.numel() != n:
                raise RuntimeError(f"exchange sizes differ: rank {r} sent {p.numel()}, rank {self.rank} {n}")
            recv.view(-1)[r * n:(r + 1) * n].copy_(p.view(recv.dtype) if p.dtype != recv.dtype else p)


def comm_pair():
    box = SimpleNamespace(cv=threading.Condition(), parts=[None, None], gen=0, result=None)
    return Comm(box, 0), Comm(box, 1)


class Buf:
    def __init__(self, rows: int, half: int) -> None:
        self.rows = rows
        self.ids = torch.zeros((rows,), dtype=torch.int32)
        self.hin = torch.zeros((rows, D), dtype=torch.bfloat16)
        self.fnormed = torch.zeros((rows, D), dtype=torch.bfloat16)
        self.taps = [torch.zeros((rows, D), dtype=torch.bfloat16) for _ in range(NT)]
        self.logits = torch.zeros((rows, half), dtype=torch.bfloat16)
        self.route_src = None
        self.kda_slots = None


def mtp_row(model: Model, h_in: int, tok: int, q: int) -> tuple[int, torch.Tensor]:
    """The fake MTP head on one row at head position q (the hidden of main position q, the token at q + 1): its
    output state (an exact continuation 70% of the time) and this rank's draft logits."""

    h = mix(h_in, tok, q + 1)
    good = mix(h, q, 3) % 10 < 7
    h2 = h if good else mix(h, 99)
    return h2, model.part(h2)


class Engine:
    """``decode.Engine``'s part the batcher, the Stepper and ``decode.prefill`` use."""

    def __init__(self, st: FakeState, w, model: Model, rows: int) -> None:
        from tensorfold.families.glm5_next.cuda import fastpf

        self.st, self.w, self.model, self.graphs = st, w, model, None
        self.buf = Buf(rows, model.half)
        self.mbuf = Buf(rows, model.half)
        self.rows = self.prefill_rows = self.prefill_max = rows
        self.fast_prefill = self.fp8_prefill = False
        self.snap_grid = fastpf.grid(rows)
        self.fast_snap, self.last_hidden, self.rows_from = None, None, None
        self.mark_snaps, self.checkpoints = [], ()
        self.draft_n = model.half

    def reset(self) -> None:
        self.st.reset()

    def main_hidden(self, rows):
        return self.buf.fnormed[rows]

    def tap_rows(self, n: int):
        return torch.cat([t[:n] for t in self.buf.taps], dim=1)

    def mtp(self, next_tokens, hidden):
        st = self.st
        lg = None
        for i, t in enumerate(next_tokens):
            q = st.mtp_len + i
            h_in = decode_row(hidden[i])
            if st.mtp_drafted == 0 and i < hidden.shape[0] and q < st.pos:
                assert h_in == int(st.hlog[q]), f"MTP row {q}: not the committed hidden"
            st.mhist[q] = int(t)
            st.mhid[q] = h_in
            h2, lg = mtp_row(self.model, h_in, int(t), q)
            self.mbuf.fnormed[0] = encode(h2)
        return lg[None]

    def draft_hidden(self, row: int):
        return self.mbuf.fnormed[0:1]

    def sample(self, logits, positions, sampling, *, draft=False, probs=None):
        from tensorfold.families.glm5_next.cuda.decode import sample_rows

        return sample_rows(self.w, logits, positions, sampling, None, probs, draft=draft)


class Drafter:
    """A DFlash2 drafter for one slot: its context is the taps it was given (checked against the committed
    positions); a block proposes the true continuation where a hash says so, else a near miss, with values that tell
    them apart; the device path writes ``packed`` / ``proj`` for the real chain kernel."""

    def __init__(self, model: Model, st_of, block: int = 8, top_k: int = 4) -> None:
        self.model, self.st_of = model, st_of
        self.dev = "cpu"
        self.block, self.top_k = block, top_k
        self.pos_dev = torch.zeros((1,), dtype=torch.int64)
        self.lo_dev = torch.zeros((1,), dtype=torch.int64)
        self._end = self._hi = 0
        self.ids = torch.zeros((block,), dtype=torch.int32)
        self.packed = torch.zeros((2, block - 1, 2 * top_k), dtype=torch.float32)
        self.proj = torch.zeros((block - 1, 4), dtype=torch.float32)
        self.block_graph = self.block_graph_full = None
        self.dv_full, self.dv_st = False, None
        self.tap_in = torch.zeros((64, NT * D), dtype=torch.bfloat16)
        self.ctx = {}                                    # position -> the hash its taps said

    @property
    def context_end(self) -> int:
        return self._end

    @context_end.setter
    def context_end(self, end: int) -> None:
        self._end = int(end)

    def reset(self) -> None:
        self._end = 0
        self.pos_dev.zero_()
        self.ctx = {}

    def _take(self, rows: torch.Tensor, start: int) -> None:
        """Taps of positions start..: recorded (a prefill's rows arrive before their commit), checked against the
        committed hashes when the request ends."""

        for i in range(rows.shape[0]):
            h = decode_row(rows[i, :D])
            assert decode_row(rows[i, D:2 * D]) == h ^ 0x5A5A
            self.ctx[start + i] = h

    def add_taps(self, taps) -> None:
        self._take(taps, self._end)
        self._end += taps.shape[0]
        self.pos_dev.fill_(self._end)

    def add_taps_dev(self, taps, count, n: int, end_hi: int) -> None:
        c = int(count.reshape(-1)[0])                    # (the fake is the GPU)
        start = int(self.pos_dev[0])
        self._take(taps[:c], start)
        self.pos_dev += c

    def resident_end(self, end: int) -> None:
        assert end == int(self.pos_dev[0]), (end, int(self.pos_dev[0]))
        self._end = int(end)

    def _continuation(self, pending: int, end: int, depth: int) -> list[tuple[int, bool]]:
        h = self.ctx[end - 1] if end > 0 else H0
        h = mix(h, pending, end)
        out = []
        for d in range(depth):
            x = logits_full(h)
            t = int(np.lexsort((np.arange(V), -x))[0])
            good = mix(h, d, 11) % 10 < 7
            out.append((t if good else (t + 1) % V, good))
            h = mix(h, t, end + 1 + d)
        return out

    def propose(self, pending, depth, sampling, confidence=0.0, probs=None):
        depth = min(depth, self.block - 1)
        if depth < 1 or self._end == 0:
            return []
        out = []
        chain = 1.0
        for d, (t, good) in enumerate(self._continuation(int(pending), self._end, depth)):
            p = 0.9 if good else 0.25
            if probs is not None:
                probs.append(p)
            if confidence > 0:
                chain *= p
                if d > 0 and chain < confidence:
                    break
            out.append(t)
        return out

    def _block_compute(self) -> None:
        """The device path: every position's candidates (the drafted token with a high value) and the selector rows."""

        end = int(self.pos_dev[0])
        cont = self._continuation(int(self.ids[0]), end, self.block - 1)
        self.packed.zero_()
        K = self.top_k
        for d, (t, good) in enumerate(cont):
            cands = [t] + [(t + 1 + j) % V for j in range(2 * K - 1)]
            vals = [6.0 if good else 1.0] + [0.5 - 0.1 * j for j in range(2 * K - 1)]
            for r in range(2):
                mine = [(v, c) for v, c in zip(vals, cands) if (c // (V // 2)) == r][:K]
                while len(mine) < K:
                    mine.append((-50.0 - len(mine), r * (V // 2) + (len(mine) * 7) % (V // 2)))
                self.packed[r, d, :K] = torch.tensor([v for v, _ in mine])
                self.packed[r, d, K:] = torch.tensor([c for _, c in mine], dtype=torch.int32).view(torch.float32)
        self.proj.zero_()


class Ops:
    """The model side of resident rounds on the fake (``resident.RealOps`` with the model parts replaced)."""

    def __new__(cls, res):
        from tensorfold.families.glm5_next.cuda.resident import RealOps

        class _Ops(RealOps):
            def device(self):
                return "cpu"

            def upload(self, values, dtype):
                return torch.tensor(values, dtype=dtype)

            def event(self):
                return None

            @staticmethod
            def wait(ev):
                pass

            def mode(self, s, lo, hi, R):
                a = 0 if lo + R <= LIMIT else 1 if lo >= LIMIT else None
                b = 0 if hi + R <= LIMIT else 1 if hi >= LIMIT else None
                return a if a is not None and a == b else None

            def mode_mtp(self, s, lo, hi, n):
                return 0

            def room(self, s, end):
                pass

            def enter(self, active):
                g = torch.Generator().manual_seed(4)
                self.pred = torch.rand((V, 4), generator=g)
                self.succ = torch.rand((V, 4), generator=g)

            def forward(self, active, Rs, modes):
                bat = self.bat
                b = self.e.buf
                model = self.e.model
                S = self.res.st
                out, o = [], 0
                for s, R in zip(active, Rs):
                    st = bat.states[s]
                    cur = st.cur[0]
                    toks = [int(t) for t in b.ids[o:o + R]]
                    pos = int(st.pos_dev[0])
                    h = int(st.rec[cur])
                    if self.res.kda:                     # the previous window's kept rows, then the window's real rows
                        prev = st.saves[1 - cur]
                        for r in range(int(S.I[s, gr.PREKEEP])):
                            h = int(prev[r])
                        st.rec[1 - cur] = h
                        real = min(R, 1 + int(S.I[s, gr.NREAL]))
                        lg, _ = model.run_rows(st, b, h, pos, toks, o, st.saves[cur], real)
                    else:
                        lg, h_last = model.run_rows(st, b, h, pos, toks, o, st.saves[0])
                        st.rec[1 - cur] = h_last
                    out.append(lg)
                    o += R
                self.bat.last_rows = list(Rs)
                return torch.cat(out)

            def commit(self, active, Rs):
                bat, S = self.bat, self.res.st
                for s, R in zip(active, Rs):
                    st = bat.states[s]
                    keep = int(S.I[s, gr.KEEP])
                    cur = st.cur[0]
                    saves = st.saves[cur] if self.res.kda else st.saves[0]
                    pos = int(st.pos_dev[0])
                    if not self.res.kda and keep != R:           # ``replay_slots``: skipped when every row is kept
                        st.rec[1 - cur] = saves[keep - 1] if keep else st.rec[cur]
                    for r in range(keep):
                        st.hlog[pos + r] = saves[r]
                        st.tlog[pos + r] = st.proj[0, r]
                    st.conv[:] = torch.cat([st.conv, st.proj[0, :keep]])[-3:]
                    st.pos_dev.copy_(S.I[s, gr.POS:gr.POS + 1])

            def exit(self, active):
                if not self.res.kda:
                    return
                bat, S = self.bat, self.res.st
                for s in active:
                    st = bat.states[s]
                    cur = st.cur[0]
                    keep = int(S.I[s, gr.PREKEEP])
                    h = int(st.rec[cur])
                    for r in range(keep):
                        h = int(st.saves[1 - cur][r])
                    st.rec[1 - cur] = h
                    st.cur = [1 - cur]

            def _head_pass(self, sts, ns, offs, T, sel, full, modes):
                b = self.e.mbuf
                model = self.e.model
                out = []
                for i, (st, n, o) in enumerate(zip(sts, ns, offs)):
                    last = int(sel[i]) - o                   # the item's last real row
                    q0 = int(st.mtp_pos_dev[0])
                    h2 = None
                    for r in range(last + 1):
                        q = q0 + r
                        tok = int(b.ids[o + r])
                        h_in = decode_row(b.hin[o + r])
                        if n > 1 or q < st.pos:                  # an absorbed row: the committed hidden and token
                            if q < st.pos:
                                assert h_in == int(st.hlog[q]), f"MTP row {q}: not the committed hidden"
                                assert tok == int(st.tlog[q + 1]), f"MTP token {q + 1}: not the committed token"
                        st.mhist[q] = tok
                        st.mhid[q] = h_in
                        h2, lg = mtp_row(model, h_in, tok, q)
                    b.fnormed[i] = encode(h2)
                    out.append(lg)
                return torch.stack(out)

        return _Ops(res)


# -- a batcher on the fake ---------------------------------------------------------------------------------------------
def _values(rows: int) -> dict:
    from tensorfold.families.glm5_next.cuda import knobs

    v = {k: 0 for k in knobs.HEADER}
    v.update(prefill_rows=rows, fast_prefill=0, auto_fdrafts=7)
    return v


def _fake_compute(w, st, b, R, **kw):
    return MODELS[w.rank].compute(w, st, b, R)


def _fake_stage(w, st, b, tokens):
    return MODELS[w.rank].stage(w, st, b, tokens)


def _fake_commit(w, st, b, R, keep):
    return MODELS[w.rank].commit(w, st, b, R, keep)


MODELS: dict[int, Model] = {}


def _fake_mtp_multi(w, sts, b, tokens, hidden, full=False):
    """``batch.mtp_multi`` on the fake: each sequence's rows at its head positions, its last row's logits."""

    model = MODELS[w.rank]
    out = []
    for i, (st, toks, hid) in enumerate(zip(sts, tokens, hidden)):
        h2 = lg = None
        for r, t in enumerate(toks):
            q = st.mtp_len + r
            h_in = decode_row(hid[r])
            if len(toks) > 1 and q < st.pos:
                assert h_in == int(st.hlog[q]), f"MTP row {q}: not the committed hidden"
            st.mhist[q] = int(t)
            st.mhid[q] = h_in
            h2, lg = mtp_row(model, h_in, int(t), q)
        b.fnormed[i] = encode(h2)
        out.append(lg)
    return torch.stack(out)


def make_batcher(monkeypatch, *, rank: int, world: int, comm, n: int, batch_mtp: bool, rows: int = 64):
    import types

    from tensorfold.families.glm5_next.cuda import batch, decode, decode_overlap as dover
    from tensorfold.families.glm5_next.cuda.batch import BACKLOG
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine
    from tensorfold.families.glm5_next.cuda.resident import Resident

    model = Model(rank, world)
    MODELS[rank] = model
    for name, fn in (("stage", _fake_stage), ("compute", _fake_compute), ("commit", _fake_commit)):
        monkeypatch.setattr(decode, name, fn)
    monkeypatch.setattr(batch, "commit", _fake_commit)
    monkeypatch.setattr(batch, "mtp_multi", _fake_mtp_multi)
    monkeypatch.setattr(decode, "_sync", lambda w: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(batch.Batcher, "_free", staticmethod(lambda: 1 << 50))
    half = V // world
    w = SimpleNamespace(comm=comm, world=world, rank=rank, vocab_offset=rank * half, device="cpu",
                        meta={"gpu_sample": True}, mtp=object(), draft_head=None, head=SimpleNamespace(n=half),
                        cfg=SimpleNamespace(eos=EOS, dense_limit=10 ** 9, hidden=D))
    states = [FakeState() for _ in range(n)]
    e = Engine(states[0], w, model, rows)
    drafters = [Drafter(model, (lambda s=s: states[s])) for s in range(n)]
    g = SimpleNamespace(e=e, w=w, drafter=drafters[0], rank=rank, f_most=7, costs=COSTS, store=None,
                        serial_only=False, comm=comm)
    for name in ("_drafters", "_grid", "_store_save"):
        setattr(g, name, types.MethodType(getattr(GlmEngine, name), g))
    g._knobs = lambda values: contextlib.nullcontext()
    if world == 1:
        g._share = lambda values: list(values)
    else:
        def share(values):
            cnt = torch.tensor([len(values) if rank == 0 else 0], dtype=torch.int32)
            got = torch.zeros((2,), dtype=torch.int32)
            comm.all_gather(cnt, got)
            count = int(got[0])
            buf = (torch.tensor(list(values), dtype=torch.int32) if rank == 0
                   else torch.zeros((count,), dtype=torch.int32))
            allv = torch.zeros((2 * count,), dtype=torch.int32)
            comm.all_gather(buf, allv)
            return [int(v) for v in allv[:count].tolist()]

        g._share = share
    bat = object.__new__(batch.Batcher)
    m_all = torch.zeros((n, BACKLOG + 1, D), dtype=torch.bfloat16)
    f_all = torch.zeros((n, BACKLOG + 1, NT * D), dtype=torch.bfloat16)
    ov = dover.parse("sync,emit,plan")
    bat.__dict__.update(
        g=g, n=n, states=states, graphs=[None] * n, drafters=drafters, max_rows=16, piece=256,
        fair=batchplan.Fairness(1.0), reserve=0.0, admit_min=0.0, last_verify=False, store=None, store_budget=0,
        counts=collections.Counter(), caches=[[] for _ in range(n)], used=[0] * n, seqs=[None] * n, round=0,
        eos=EOS, costs=COSTS, round_costs=batchplan.RoundCosts(COSTS, 6.5), defaults=_values(rows),
        log=collections.deque(maxlen=1000), trace=collections.deque(maxlen=10000), last_piece_s=0.0,
        cv=threading.Condition(), stopping=False, queue=collections.deque(), error=None, thread=None,
        use_graphs=False, multi={}, pool=None, following=rank == 1, graph_rows=16, max_graphs=0,
        sightings=batchplan.Sightings(1), batch_mtp=batch_mtp, row_ms=6.5, overlap=ov,
        rider_n=dover.rider_size(n), rider_host=torch.zeros((dover.rider_size(n),), dtype=torch.int32),
        rider_dev=torch.zeros((dover.rider_size(n),), dtype=torch.int32), m_all=m_all, f_all=f_all,
        m_rows=[m_all[i, :BACKLOG] for i in range(n)], f_taps=[f_all[i, :BACKLOG] for i in range(n)])

    def forward(active, windows):
        out, o = [], 0
        for s, win in zip(active, windows):
            st = bat.states[s]
            lg, h = model.run_rows(st, e.buf, st.committed(), st.pos, [int(t) for t in win], o, st.saves[0])
            st.rec[1 - st.cur[0]] = h
            out.append(lg)
            o += len(win)
        bat.last_rows = [len(x) for x in windows]
        bat.last_kind = "eager"
        return torch.cat(out)

    bat._forward = forward
    bat.resident = Resident(bat, Ops)
    bat.arrive_hook = None
    process = bat.resident._process

    def processed(h):                                 # arrivals by round, also while rounds are resident
        out = process(h)
        if bat.arrive_hook is not None:
            bat.arrive_hook()
        return out

    bat.resident._process = processed
    bat.ended = {}
    finish = bat._finish

    def spy(slot, *, cancelled=False):
        q = bat.seqs[slot]
        st = bat.states[slot]
        stp = q.stepper
        m = max(0, st.mtp_len - st.mtp_drafted)
        d = bat.drafters[slot]
        bat.ended[id(q.job)] = dict(pos=st.pos, h=st.committed(), mtp=m, mhist=st.mhist[:m].tolist(),
                                    mhid=st.mhid[:m].tolist(), hlog=st.hlog[:st.pos].tolist(),
                                    tlog=st.tlog[:st.pos + 1].tolist(), out=list(stp.out) if stp else [],
                                    ctx=[d.ctx.get(p) for p in range(d.context_end)], ctx_end=d.context_end,
                                    uses_f=bool(stp and stp.drafter is not None), cancelled=cancelled)
        return finish(slot, cancelled=cancelled)

    bat._finish = spy
    return bat


def mkjob(prompt, n: int, sampling, spec: str, *, cost: bool = False, stop_eos: bool = False, lookup: int = 0,
          rows: int = 64):
    from tensorfold.families.glm5_next.cuda import depth as depth_mod
    from tensorfold.families.glm5_next.cuda.batch import Job
    from tensorfold.families.glm5_next.cuda.engine import DFLASH_POLICY, encode_policy
    from tensorfold.families.glm5_next.cuda.lookup import with_lookup

    code = encode_policy(spec)
    if cost:
        code = depth_mod.as_cost(code)
    code = with_lookup(code, True, encode_policy(DFLASH_POLICY))
    if code[0] in (4, 5) and len(code) >= 6:
        code = code[:4] + [lookup, 4] + code[6:]
    return Job(list(prompt), n, sampling, stop_eos, True, list(code), spec, int(cost), _values(rows),
               out=queue.SimpleQueue(), submitted=time.perf_counter())


def drain(job) -> tuple[list[int], bool]:
    got, done = [], False
    while not job.out.empty():
        item = job.out.get()
        if item is None:
            done = True
        elif isinstance(item, BaseException):
            raise item
        else:
            got.extend(item)
    return got, done


def requests(seed: int, n: int = 6):
    """Prompts and settings: auto (greedy / sampled, cost depths), MTP-only, DFlash2-only, serial, stop at EOS."""

    rng = random.Random(seed)
    specs = [("auto", True, None), ("auto", True, Sampling(11, 1.0, 20, 0.95)), ("om", True, None),
             ("fc5:0.3", False, Sampling(12, 0.7, 5, 0.9)), ("0", False, None), ("c3:0.35", False, None),
             ("auto", False, Sampling(13, 1.2, 40, 1.0)), ("of", True, None)]
    out = []
    for i in range(n):
        spec, cost, s = specs[(seed + i) % len(specs)]
        prompt = [rng.randrange(V) for _ in range(rng.randint(20, 90))]
        stop = i % 3 == 1
        out.append(dict(prompt=prompt, n=rng.randint(24, 60), s=s, spec=spec, cost=cost, stop=stop,
                        lookup=int(i % 4 == 3)))
    return out


def serve(bat, reqs, arrive, *, rounds_max: int = 4000):
    """Rank 0's loop (``Batcher._serve``'s body) with the requests queued at their rounds (``bat.round``: resident
    rounds count, and arrivals come in while they run). -> the jobs."""

    jobs = [None] * len(reqs)

    def arrivals():
        for i, (r, at) in enumerate(zip(reqs, arrive)):
            if jobs[i] is None and at <= bat.round:
                jobs[i] = mkjob(r["prompt"], r["n"], r["s"], r["spec"], cost=r["cost"], stop_eos=r["stop"],
                                lookup=r["lookup"])
                with bat.cv:
                    bat.queue.append(jobs[i])

    bat.arrive_hook = arrivals
    loops = 0
    while True:
        arrivals()
        if bat.queue or any(q is not None for q in bat.seqs):
            plan = bat._take_ahead() or bat._plan()
            bat._execute(*plan)
            bat._go_resident()
        elif all(j is not None for j in jobs):
            break
        else:
            bat.round += 1                          # idle: time passes
        loops += 1
        assert loops < rounds_max
    bat.arrive_hook = None
    return jobs


def check(bat, reqs, jobs) -> None:
    world = bat.g.w.world
    for r, job in zip(reqs, jobs):
        reply, done = drain(job)
        assert done
        want, h = serial(r["prompt"], r["n"], r["s"], r["stop"], world)
        assert reply == want[:len(reply)] and len(reply) == len(want), (r["spec"], reply, want)
        end = bat.ended[id(job)]
        committed = list(r["prompt"]) + end["out"][:end["pos"] - len(r["prompt"])]
        hh = H0
        for p, t in enumerate(committed):
            hh = mix(hh, t, p)
        assert end["h"] == hh and end["pos"] == len(committed), r["spec"]      # the committed state is serial's
        assert end["tlog"][:end["pos"]] == committed
        m = end["mtp"]
        toks = list(r["prompt"]) + end["out"]                                  # the pending token included
        assert end["mhist"][:m] == toks[1:m + 1], r["spec"]      # the MTP head's cache, position by position
        assert end["mhid"][:m] == end["hlog"][:m], r["spec"]
        if end["uses_f"]:                                        # the DFlash2 context, position by position
            assert end["ctx"] == end["hlog"][:end["ctx_end"]], r["spec"]


def _knob(monkeypatch, parts: str, peek: str = "1") -> None:
    monkeypatch.setenv("GLM53_TF_GPU_ROUND", parts)
    monkeypatch.setenv("GLM53_TF_GPU_ROUND_PEEK", peek)
    gr.env.cache_clear()
    gr.peek.cache_clear()


@pytest.fixture(autouse=True)
def _clear():
    yield
    gr.env.cache_clear()
    gr.peek.cache_clear()


@interp
@pytest.mark.parametrize("peek", ["1", "0"], ids=["peek", "lag"])
@pytest.mark.parametrize("parts,batch_mtp", [("resident", False), ("resident", True), ("resident,kda", True)])
def test_resident_replies_equal_serial(monkeypatch, parts, batch_mtp, peek):
    _knob(monkeypatch, parts, peek)
    try:
        bat = make_batcher(monkeypatch, rank=0, world=1, comm=None, n=3, batch_mtp=batch_mtp)
        assert bat.resident.kda == ("kda" in parts)
        reqs = requests(1 if batch_mtp else 2)
        jobs = serve(bat, reqs, [0, 0, 1, 5, 30, 31])
        check(bat, reqs, jobs)
        res = bat.resident
        assert res.rounds > 20 and res.entries > 2, (res.rounds, res.entries, res.drains, dict(bat.counts))
        assert res.drains.get("done", 0) >= 2 and res.drains.get("rank 0", 0) >= 1, res.drains
        arms = "".join(d["arms"] for d in bat.log)
        assert set(arms) >= set("mfs"), arms
        if peek == "1":                         # windows as the device drafted them: never padded
            assert bat.counts["pad_rows"] == 0 and res.peeks >= res.rounds
        else:                                   # stale bounds: some windows ran padded rows (tie-routed, never kept)
            assert bat.counts["pad_rows"] > 0 and res.peeks == 0
    finally:
        gr.env.cache_clear()


@interp
def test_context_mode_boundary_drains_and_stays_exact(monkeypatch):
    global LIMIT
    _knob(monkeypatch, "resident,kda", "0")
    saved = LIMIT
    LIMIT = 70                                        # windows cross it while decoding
    try:
        bat = make_batcher(monkeypatch, rank=0, world=1, comm=None, n=2, batch_mtp=True)
        reqs = requests(3, 3)
        jobs = serve(bat, reqs, [0, 0, 2])
        check(bat, reqs, jobs)
        assert bat.resident.drains.get("context mode", 0) >= 1, bat.resident.drains
    finally:
        LIMIT = saved
        gr.env.cache_clear()


@interp
@pytest.mark.parametrize("parts,peek", [("resident", "1"), ("resident,kda", "1"), ("resident,kda", "0")])
def test_two_ranks_in_threads(monkeypatch, parts, peek):
    """Rank 0 serves (its loop), rank 1 follows (``Batcher.follow``) in another thread; every exchange (plans, the
    sampler's, the drafts') goes through the fake communicator. Replies == serial; both ranks end every request the
    same way through the same rounds."""

    _knob(monkeypatch, parts, peek)
    c0, c1 = comm_pair()
    try:
        b0 = make_batcher(monkeypatch, rank=0, world=2, comm=c0, n=3, batch_mtp=True)
        b1 = make_batcher(monkeypatch, rank=1, world=2, comm=c1, n=3, batch_mtp=True)
        reqs = requests(4, 5)
        err: list = []

        def follower():
            try:
                with torch.no_grad():
                    b1.follow()
            except TimeoutError:
                pass
            except BaseException as exc:  # noqa: BLE001
                err.append(exc)

        t = threading.Thread(target=follower, daemon=True)
        t.start()
        jobs = serve(b0, reqs, [0, 0, 1, 12, 13])
        t.join(timeout=300)
        assert not err, err
        check(b0, reqs, jobs)
        key = lambda d: (d["slot"], d["sha256"], tuple(d["keeps"]), d["arms"], d["cancelled"])      # noqa: E731
        assert [key(d) for d in b1.log] == [key(d) for d in b0.log]
        assert b0.resident.rounds == b1.resident.rounds > 10
        assert b0.resident.drains == b1.resident.drains
    finally:
        gr.env.cache_clear()


@interp
def test_no_device_reads_while_launching(monkeypatch):
    """Between processing one round and the next, the driver (resident.py, gpuround.py, batch.py) launches only: any
    ``.item()`` / ``.cpu()`` / ``.tolist()`` / ``.numpy()`` / ``int(tensor)`` called from those modules while a round
    is being decided or launched fails the test, unless the tensor is host staging (the pinned buffers the host writes
    for the round: ``ResidentState.h_*``, the rider's). The fake model and the interpreter may read: they are the
    GPU."""

    from tensorfold.families.glm5_next.cuda import resident as res_mod

    _knob(monkeypatch, "resident,kda", "0")         # the whole round ahead: nothing may be read while launching
    watched = {"tensorfold.families.glm5_next.cuda.resident", "tensorfold.families.glm5_next.cuda.gpuround",
               "tensorfold.families.glm5_next.cuda.batch", "tensorfold.families.glm5_next.cuda.gpusample",
               "tensorfold.families.glm5_next.cuda.dflash2"}
    state = {"on": False, "hits": [], "host": set()}

    def guard(name, fn):
        def wrapped(self, *a, **k):
            if state["on"]:
                caller = sys._getframe(1).f_globals.get("__name__", "")
                if caller in watched and self.untyped_storage().data_ptr() not in state["host"]:
                    state["hits"].append((name, caller, sys._getframe(1).f_code.co_name))
            return fn(self, *a, **k)
        return wrapped

    for name in ("item", "cpu", "tolist", "numpy", "__int__", "__index__", "__bool__", "__float__"):
        monkeypatch.setattr(torch.Tensor, name, guard(name, getattr(torch.Tensor, name)))
    launch, decide = res_mod.Resident._launch, res_mod.Resident._decide

    def on(fn):
        def wrapped(self, *a, **k):
            state["on"] = True
            try:
                return fn(self, *a, **k)
            finally:
                state["on"] = False
        return wrapped

    monkeypatch.setattr(res_mod.Resident, "_launch", on(launch))
    monkeypatch.setattr(res_mod.Resident, "_decide", on(decide))
    try:
        bat = make_batcher(monkeypatch, rank=0, world=1, comm=None, n=3, batch_mtp=True)
        S = bat.resident.st
        for name in ("h_rec", "h_recf", "h_H", "h_P", "h_rows", "h_rider"):
            state["host"] |= {t.untyped_storage().data_ptr() for t in getattr(S, name)}
        state["host"] |= {S.h_I.untyped_storage().data_ptr(), bat.rider_host.untyped_storage().data_ptr()}
        reqs = requests(5, 4)
        jobs = serve(bat, reqs, [0, 0, 0, 3])
        check(bat, reqs, jobs)
        assert bat.resident.rounds > 10
        assert not state["hits"], state["hits"][:10]
    finally:
        gr.env.cache_clear()


@interp
def test_peek_reads_only_its_records(monkeypatch):
    """With GLM53_TF_GPU_ROUND_PEEK=1 the only device reads while a round is decided or launched are ``peek_state``'s
    (one pinned copy of the counters and its event; the next window's rows, an MTP chain's continuation), never a
    sampler's candidates, a logits row or a state."""

    from tensorfold.families.glm5_next.cuda import resident as res_mod

    _knob(monkeypatch, "resident,kda", "1")
    watched = {"tensorfold.families.glm5_next.cuda." + m for m in ("resident", "gpuround", "batch", "gpusample",
                                                                   "dflash2")}
    state = {"on": False, "hits": [], "host": set()}

    def guard(name, fn):
        def wrapped(self, *a, **k):
            if state["on"]:
                f = sys._getframe(1)
                ptr = self.untyped_storage().data_ptr()
                if f.f_globals.get("__name__", "") in watched and ptr not in state["host"]:
                    state["hits"].append((name, f.f_code.co_name))
            return fn(self, *a, **k)
        return wrapped

    for name in ("item", "cpu", "tolist", "numpy", "__int__", "__index__", "__bool__", "__float__"):
        monkeypatch.setattr(torch.Tensor, name, guard(name, getattr(torch.Tensor, name)))

    def on(fn):
        def wrapped(self, *a, **k):
            state["on"] = True
            try:
                return fn(self, *a, **k)
            finally:
                state["on"] = False
        return wrapped

    monkeypatch.setattr(res_mod.Resident, "_launch", on(res_mod.Resident._launch))
    monkeypatch.setattr(res_mod.Resident, "_decide", on(res_mod.Resident._decide))
    bat = make_batcher(monkeypatch, rank=0, world=1, comm=None, n=3, batch_mtp=True)
    S = bat.resident.st
    for name in ("h_rec", "h_recf", "h_H", "h_P", "h_rows", "h_rider"):
        state["host"] |= {t.untyped_storage().data_ptr() for t in getattr(S, name)}
    state["host"] |= {S.h_I.untyped_storage().data_ptr(), bat.rider_host.untyped_storage().data_ptr()}
    reqs = requests(6, 4)
    jobs = serve(bat, reqs, [0, 0, 0, 3])
    check(bat, reqs, jobs)
    res = bat.resident
    assert res.rounds > 10 and res.peeks >= res.rounds
    assert {h for h in state["hits"] if h[1] != "peek_state"} == set(), state["hits"][:10]
