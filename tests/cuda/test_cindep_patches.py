"""patches/0085 (``glm5_next/cuda/pfgrid.py``): a fast prefill's state does not depend on its chunk size C.

Every kernel of a fast chunk is row-independent and the chunked KDA scan couples rows only inside 64-row blocks at
absolute multiples of 64, so the committed state after a fast prefill is a function of the tokens alone. Snapshots
then sit on a 64-token grid (GLM53_TF_SNAPSHOT_GRID), are tagged by mode only (fast G, FP8 fast G + 1), and a request
of any C resumes from them; ``prefill_rows`` may be "auto". Checked:

- host only: ``pfgrid`` (tags, grid, tail rule, chunk plans, auto rows, parsing); with GLM53_TF_SNAPSHOT_GRID = C the
  plan is patches/0080's grid rule exactly; the engine's real ``_run`` / ``_prefill`` / snapshot / resume code on a
  fake model whose fast chunks are as hostile as the new rule allows (a 64-row block's rows depend on every row of
  the block present in the call, on the call's offset in the block, and on the fast / FP8 mode; nothing on the
  chunk's size or start): random conversations with a random C per request (auto included), FP8 and bf16 requests
  mixed, session-style marks, serial and MTP-drafted -- every reply and the whole state equal a fresh prefill's with
  yet another C, snapshots sit on 64-token multiples and a follow-up re-prefills at most tail + 63 tokens of the old
  prompt; the control (a fake that depends on the chunk size) is caught; ``fastpf.chunk`` keeps only the head on the
  < 64-row qmm fallback and restores it;
- GPU kernels (row independence, bit for bit): ``fast_qmm.matmul_fast`` (4-bit and BF16, bf16 and FP8 prefill) on
  slices of 1-63 rows, 64-row-aligned and unaligned slices and permutations of the rows; the head's 1-row call still
  qmm's; ``hc_partial``; latent ``absorb_tc`` / ``expand_tc``; the fused EXL3 experts (a routed pair's output does not
  depend on the other members of its expert: row subsets and permutations, skewed routing past one pass of members);
  ``fast_kda`` calls cut anywhere on multiples of 64 (the prompt's partial last block included) == one call, and a
  misaligned fast call is refused;
- GPU engine (synthetic EXL3 checkpoint): the committed state (first token, KDA states, conv windows, attention /
  latent / indexer caches, MTP cache and pending row) is identical for C = 64, 1024, 4096, 8192, non-lean and lean
  (sub-blocks of 256) with and without the pipelined overlap, bf16 and FP8, for 3-1,000-token prompts and, on the
  latent cache with 32 index heads, 3,000 and 9,000-token prompts past the dense limit; a follow-up resumed from the
  64-grid snapshot with another C (and "auto") equals a fresh prefill with a third C (replies and state, MTP and
  DFlash2 drafts, FP8 too); drafted == serial with auto chunks; the knob and GLM53_TF_PREFILL_ROWS=auto.

Run inside the image:
    PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q tests/cuda/test_cindep_patches.py
"""

from __future__ import annotations

import types
from types import SimpleNamespace as NS

import numpy as np
import pytest

try:
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:          # pragma: no cover
    torch = None
    CUDA = False

from tensorfold.families.glm5_next.cuda import pfgrid  # noqa: E402

gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")
needs_torch = pytest.mark.skipif(torch is None, reason="torch")
SAMPLINGS = ["sampled", "greedy"]
CS = (64, 1024, 4096, 8192)


def _sampling(kind: str):
    from tensorfold.engine.exact_sampling import Sampling

    return None if kind == "greedy" else Sampling(1234, 1.0, 20, 0.95)


# -- host only: pfgrid ----------------------------------------------------------------------------------------------
def test_tags_and_resumable(monkeypatch):
    monkeypatch.delenv(pfgrid.GRID_ENV, raising=False)
    assert pfgrid.snapshot_grid() == 64 and pfgrid.snapshot_tail() == 256
    assert pfgrid.tag(False) == pfgrid.tag(False, True) == 0
    assert pfgrid.tag(True) == 64 and pfgrid.tag(True, True) == 65
    assert pfgrid.grid_of(64) == pfgrid.grid_of(65) == 64 and pfgrid.grid_of(0) == 0
    assert pfgrid.is_fp8(65) and not pfgrid.is_fp8(64) and not pfgrid.is_fp8(0)
    assert pfgrid.resumable(64, 64, 960) and not pfgrid.resumable(64, 64, 961)
    assert not pfgrid.resumable(65, 64, 960) and not pfgrid.resumable(0, 64, 960) and not pfgrid.resumable(64, 0, 64)
    assert pfgrid.resumable(0, 0, 961)                                  # exact snapshots: anywhere
    monkeypatch.setenv(pfgrid.GRID_ENV, "256")
    assert pfgrid.tag(True, True) == 257 and pfgrid.settings() == [256, 256]
    for bad in ("100", "32", "0"):
        monkeypatch.setenv(pfgrid.GRID_ENV, bad)
        with pytest.raises(ValueError, match="multiple of 64"):
            pfgrid.snapshot_grid()
    monkeypatch.setenv(pfgrid.TAIL_ENV, "-1")
    with pytest.raises(ValueError):
        pfgrid.snapshot_tail()


def test_rows_parse_and_auto():
    assert pfgrid.parse_rows("auto") == pfgrid.parse_rows(" AUTO ") == pfgrid.AUTO == 0
    assert pfgrid.parse_rows("8192") == 8192 and pfgrid.parse_rows(7) == 7
    for bad in ("0", "-3", 0, True, "x"):
        with pytest.raises(ValueError):
            pfgrid.parse_rows(bad)
    assert pfgrid.show_rows(0) == "auto" and pfgrid.show_rows(512) == 512
    # fast: multiples of 64 up to the buffers; auto = the rows to prefill rounded up, at most the buffers
    assert pfgrid.chunk_rows(pfgrid.AUTO, 1, 8192, True) == 64
    assert pfgrid.chunk_rows(pfgrid.AUTO, 1000, 8192, True) == 1024
    assert pfgrid.chunk_rows(pfgrid.AUTO, 28000, 8192, True) == 8192
    assert pfgrid.chunk_rows(pfgrid.AUTO, 28000, 8100, True) == 8064
    assert pfgrid.chunk_rows(100, 5000, 8192, True) == 64 and pfgrid.chunk_rows(1000, 5000, 8192, True) == 960
    with pytest.raises(ValueError):
        pfgrid.chunk_rows(8192, 9000, 4096, True)
    with pytest.raises(ValueError, match="multiples of 64"):
        pfgrid.chunk_rows(pfgrid.AUTO, 10, 32, True)
    # exact: any size up to the buffers (auto: all of them)
    assert pfgrid.chunk_rows(pfgrid.AUTO, 1000, 512, False) == 512 and pfgrid.chunk_rows(7, 1000, 512, False) == 7
    assert pfgrid.chunk_rows(pfgrid.AUTO, 3, 512, False) == 3


def _check_plan(pl, begin, n, fast=True):
    assert pl.spans[0][0] == begin and pl.spans[-1][1] == n
    assert all(a[1] == b[0] for a, b in zip(pl.spans, pl.spans[1:])) and all(s < t for s, t in pl.spans)
    if fast:
        assert all(s % 64 == 0 for s, _ in pl.spans)
        assert pl.snap == n or pl.snap in [s for s, _ in pl.spans]
    for m in pl.marks:
        assert m in [s for s, _ in pl.spans] and m != pl.snap


def test_plan_snapshot_point_and_tail():
    # the prompt's last multiple of 64, cut when the schedule does not start a chunk there
    pl = pfgrid.plan(0, 28000, 8192, fast=True, grid=64, tail=256)
    _check_plan(pl, 0, 28000)
    assert pl.snap == 27968 and pl.cut and pl.spans[-2:] == [(24576, 27968), (27968, 28000)]
    # on the grid: the end state, no cut
    pl = pfgrid.plan(0, 8192 + 640, 8192, fast=True, grid=64, tail=256)
    assert pl.snap == 8832 and not pl.cut and pl.spans == [(0, 8192), (8192, 8832)]
    # the last chunk already starts within the tail: snapshot there, no extra chunk
    pl = pfgrid.plan(0, 8300, 8192, fast=True, grid=64, tail=256)
    assert pl.snap == 8192 and not pl.cut and pl.spans == [(0, 8192), (8192, 8300)]
    pl = pfgrid.plan(0, 8300, 8192, fast=True, grid=64, tail=0)
    assert pl.snap == 8256 and pl.cut and pl.spans == [(0, 8192), (8192, 8256), (8256, 8300)]
    # resumed: from the resume point, which stays the snapshot when nothing new can be kept
    pl = pfgrid.plan(1024, 1124, 1024, fast=True, grid=64, tail=256)
    assert pl.snap == 1024 and pl.spans == [(1024, 1124)]
    pl = pfgrid.plan(1024, 1500, 1024, fast=True, grid=64, tail=256)
    assert pl.snap == 1472 and pl.spans == [(1024, 1472), (1472, 1500)]
    # short prompts: nothing (snap 0)
    assert pfgrid.plan(0, 50, 64, fast=True, grid=64, tail=0).snap == 0
    assert pfgrid.plan(0, 64, 64, fast=True, grid=64, tail=0).snap == 64
    # marks cut chunks and are snapshots of their own (never the snapshot point itself)
    pl = pfgrid.plan(0, 3000, 1024, fast=True, marks=[512, 1000, 2944, 5000], grid=64, tail=256)
    _check_plan(pl, 0, 3000)
    assert pl.marks == [512] and pl.snap == 2944
    # exact: any positions, no snapshot point
    pl = pfgrid.plan(3, 1000, 100, fast=False, marks=[7, 257, 999, 1000])
    _check_plan(pl, 3, 1000, fast=False)
    assert pl.snap == -1 and pl.marks == [7, 257, 999] and pl.spans[0] == (3, 7)
    with pytest.raises(ValueError, match="multiples of 64"):
        pfgrid.plan(32, 1000, 128, fast=True)
    with pytest.raises(ValueError, match="multiples of 64"):
        pfgrid.plan(0, 1000, 100, fast=True)


def test_grid_at_c_is_the_0080_rule():
    """GLM53_TF_SNAPSHOT_GRID = C with a fixed C: chunks at absolute multiples of C and the snapshot at
    floor(n / C) * C (end state on the grid), whatever the tail -- patches/0080's rule."""

    rng = np.random.default_rng(5)
    for C in (64, 128, 192, 1024):
        for _ in range(200):
            n = int(rng.integers(1, 5 * C))
            begin = int(rng.integers(0, (n - 1) // C + 1)) * C
            for tail in (0, 256, 10**6):
                pl = pfgrid.plan(begin, n, C, fast=True, grid=C, tail=tail)
                assert pl.spans == [(s, min(s + C, n)) for s in range(begin, n, C)]
                assert pl.snap == n // C * C and not pl.cut


def test_random_plans():
    rng = np.random.default_rng(6)
    for _ in range(2000):
        n = int(rng.integers(1, 20000))
        begin = int(rng.integers(0, (n - 1) // 64 + 1)) * 64
        step = int(rng.choice([64, 128, 960, 1024, 4096, 8192]))
        tail = int(rng.choice([0, 64, 256, 1000]))
        marks = [int(m) * 64 for m in rng.integers(0, n // 64 + 2, size=int(rng.integers(0, 4)))]
        pl = pfgrid.plan(begin, n, step, fast=True, marks=marks, grid=64, tail=tail)
        _check_plan(pl, begin, n)
        s = n // 64 * 64
        assert pl.snap <= n and (pl.snap == s or (not pl.cut and s - pl.snap <= tail) or pl.snap >= s)
        assert all(t - u <= step for u, t in pl.spans)
        if pl.snap > begin and pl.snap < n:
            assert pl.snap % 64 == 0
        # at most one chunk more than the schedule without the snapshot cut
        base = pfgrid.spans(begin, n, step, [m for m in marks if m % 64 == 0])
        assert len(pl.spans) <= len(base) + 1


# -- host only: the rule through the engine's real code, on a fake model ---------------------------------------------
class _BlockFake:
    """A CPU stand-in whose fast chunks are as hostile as patches/0085's rule allows. A fast row's value depends on
    its token and position, a digest of the KV rows before it (its attention), the state entering its 64-row block,
    EVERY row of its block present in the call (not only the earlier ones: stricter than the real scan), the call's
    offset inside the block (a call starting mid-block would change it) and the mode (bf16 / FP8 fast; exact rows
    are per row). It does NOT depend on the chunk's size or start beyond that: the real kernels' guarantee.
    ``chunk_dependent`` (the control) mixes the chunk's size in, as patches/0080's kernels were allowed to."""

    M = (1 << 61) - 1

    def __init__(self, cap: int = 4096, chunk_dependent: bool = False):
        self.kv = torch.zeros(cap, dtype=torch.int64)
        self.dig = [0] * (cap + 1)
        self.mkv = torch.zeros(cap, dtype=torch.int64)
        self.hidden = torch.zeros((cap, 1), dtype=torch.int64)
        self.ids: list[int] = []
        self.chunk_dependent = chunk_dependent

    def mix(self, *v: int) -> int:
        h = 1469598103934665603
        for x in v:
            h = (h * 1099511628211 + int(x) + 12345) % self.M
        return h

    def stage(self, w, st, b, tokens):
        self.ids = list(tokens)
        return len(tokens)

    def compute(self, w, st, b, R, *, logits=True, nch=None, host_pos=None, npb=None, fast=False, head=True):
        from tensorfold.families.glm5_next.cuda import fp8pf

        s = int(st.rec[st.cur[0], 0])
        mode = (2 if fp8pf.ON else 1) if fast else 0
        pos = st.pos
        i = 0
        while i < R:
            p0 = pos + i
            end = min(R, i + (64 - p0 % 64)) if fast else i + 1        # the rest of this 64-block in the call
            blk = self.mix(*self.ids[i:end], p0 % 64, end - i) if fast else 0
            extra = R if self.chunk_dependent and fast else -1
            for j in range(i, end):
                p = pos + j
                s = self.mix(s, self.ids[j], p, self.dig[p], blk, mode, extra)
                self.kv[p] = self.mix(s, 7)
                self.dig[p + 1] = self.mix(self.dig[p], int(self.kv[p]))
                self.hidden[j, 0] = s
            i = end
        st.rec[1 - st.cur[0], 0] = s
        return self.hidden[:R].clone()

    def commit(self, w, st, b, R, keep):
        st.rec[1 - st.cur[0], 0] = self.hidden[keep - 1, 0]
        st.cur = [1 - st.cur[0]]
        st.conv[0] = torch.tensor(([0, 0, 0] + list(self.ids[:keep]))[-3:], dtype=torch.int64)
        st.set_pos(st.pos + keep)


def _block_engine(monkeypatch, cap: int, tail: int, chunk_dependent: bool = False):
    from test_fastpf_patches import _FakeEngine

    from tensorfold.families.glm5_next.cuda import decode
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    class Eng(_FakeEngine):
        snap_grid = 64                      # patches/0085's default grid (the base fake runs 0080's grid at C)

    fake = _BlockFake(chunk_dependent=chunk_dependent)
    monkeypatch.setattr(decode, "_sync", lambda w: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    e = Eng(fake, cap, True)
    e.snap_tail = tail
    e.fp8_prefill = False
    g = NS(e=e, drafter=None, cache=[], eos=(), online=None, calib_on=False, costs={}, f_most=7, depth_cost=0,
           fake=fake)
    for name in ("_run", "_resume", "_remember", "_grid", "_drafters"):
        setattr(g, name, types.MethodType(getattr(GlmEngine, name), g))
    return g


def _conversation(monkeypatch, *, policy: str, tail: int, seed: int, chunk_dependent: bool = False, turns: int = 60):
    """Random turns (extend the last prompt, or its reply, or a prefix; or a new prompt), each with a random chunk
    size (auto included) and mode; the reference is a fresh prefill with another random chunk size. Returns
    (mismatches, resumed turns, the cached counts of continuation turns and their old prompt lengths)."""

    from test_fastpf_patches import _fake_request

    from tensorfold.families.glm5_next.cuda import decode

    rng = np.random.default_rng(seed)
    cap = 1024
    sizes = [pfgrid.AUTO, 64, 128, 192, 320, 1024]

    def use(g):
        for name in ("stage", "compute", "commit"):
            monkeypatch.setattr(decode, name, getattr(g.fake, name))

    g = _block_engine(monkeypatch, cap, tail, chunk_dependent)
    ref = _block_engine(monkeypatch, cap, tail, chunk_dependent)
    more = lambda n: [int(t) for t in rng.integers(0, 1000, size=n)]         # noqa: E731
    last, reply, last_fp8 = [], [], False
    bad, resumed, follow = [], 0, []
    for turn in range(turns):
        fp8 = bool(rng.integers(0, 4) == 0) if turn >= 2 else False
        kind = int(rng.integers(0, 5)) if turn else 4
        grow = more(int(rng.choice([1, 3, 30, 63, 64, 65, 200, 700]))) if turn else more(700)
        prompt = (last + reply + grow if kind <= 1 else last + grow if kind == 2 else
                  last[:int(rng.integers(0, len(last) + 1))] + grow if kind == 3 else grow)[:3000]
        use(g)
        g.e.prefill_rows = int(rng.choice(sizes))
        g.e.fp8_prefill = fp8
        g.e.checkpoints = sorted({int(m) * 64 for m in rng.integers(1, len(prompt) // 64 + 2, size=2)})
        got, cached, state = _fake_request(g, prompt, policy=policy)
        g.e.checkpoints = ()
        assert all(s.grid == pfgrid.tag(True, bool(s.grid % 64)) and len(s.ids) % 64 == 0 for s in g.cache)
        assert cached % 64 == 0
        if kind <= 1 and fp8 == last_fp8 and prompt[:len(last)] == last:
            follow.append((len(last), cached))
        resumed += cached > 0
        use(ref)
        ref.cache = []
        ref.e.prefill_rows = int(rng.choice(sizes))
        ref.e.fp8_prefill = fp8
        want, c0, want_state = _fake_request(ref, prompt, policy=policy)
        assert c0 == 0
        if (got, state) != (want, want_state):
            bad.append((turn, len(prompt), cached, fp8))
        last, reply, last_fp8 = prompt, got, fp8
    return bad, resumed, follow


@needs_torch
@pytest.mark.parametrize("tail", [0, 256])
@pytest.mark.parametrize("policy", ["0", "2"])
def test_rule_any_chunk_size_on_a_hostile_fake(monkeypatch, policy, tail):
    bad, resumed, follow = _conversation(monkeypatch, policy=policy, tail=tail, seed=850 + tail + len(policy))
    assert not bad, bad
    assert resumed >= 10, resumed
    # a follow-up turn resumes from the old prompt's last 64-token point (tail rule: at most tail rows before it)
    assert follow and all(c >= n - 63 - tail for n, c in follow if n >= 64 + tail), follow


@needs_torch
def test_rule_control_chunk_dependent_fake_is_caught(monkeypatch):
    """The same conversations on a fake whose fast chunks depend on their size: resumes with another C diverge, so
    the test above passes because the state does not depend on C, not by luck."""

    bad, _, _ = _conversation(monkeypatch, policy="0", tail=256, seed=851, chunk_dependent=True, turns=30)
    assert bad


@needs_torch
def test_misaligned_resume_and_fast_call_refused(monkeypatch):
    from tensorfold.families.glm5_next.cuda import decode, fastpf

    g = _block_engine(monkeypatch, 1024, 256)
    for name in ("stage", "compute", "commit"):
        monkeypatch.setattr(decode, name, getattr(g.fake, name))
    prompt = list(range(300))
    decode.prefill(g.e, prompt, None, mtp=True)
    off = decode.take_snapshot(g.e, prompt, g.e.last_hidden, mtp=True, grid=64, mtp_len=299)
    with pytest.raises(ValueError, match="grid"):
        decode.prefill(g.e, prompt + [1, 2], None, mtp=True, resume=off)
    fp8 = decode.take_snapshot(g.e, prompt[:256], g.e.last_hidden, mtp=True, grid=65, mtp_len=255)
    with pytest.raises(ValueError, match="grid"):
        decode.prefill(g.e, prompt + [1, 2], None, mtp=True, resume=fp8)
    with pytest.raises(ValueError, match="multiples of 64"):
        fastpf.kda_chain(*([None] * 15), pos=100)


def test_fast_chunk_keeps_only_the_head_on_qmm(monkeypatch):
    """``fastpf.chunk`` installs the head as the one matrix with the < 64-row qmm fallback, restores it after, and
    ``matmul_fast`` asks the fast kernels for every other row count (bf16 and FP8 prefill)."""

    pytest.importorskip("triton")
    from tensorfold.families.glm5_next.cuda import fast_qmm, fastpf, fp8pf, qmm

    calls = []
    monkeypatch.setattr(fast_qmm, "matmul_prefill", lambda x, q, xs=None, **kw: calls.append(("bf16", kw)))
    monkeypatch.setattr(fast_qmm, "matmul_fp8", lambda x, q, xs=None, **kw: calls.append(("fp8", kw)))
    monkeypatch.setattr(fastpf, "fast_qmm", fast_qmm)
    head, other = object(), object()
    x = NS(shape=(1, 64))
    b = NS()
    assert fast_qmm.FAST_HEAD is None
    for on in (False, True):
        monkeypatch.setattr(fp8pf, "ON", on)
        with fastpf.chunk(b, head):
            assert b.fast and fast_qmm.FAST_HEAD is head and qmm.FAST_MM is fast_qmm.matmul_fast
            fast_qmm.matmul_fast(x, other)
            fast_qmm.matmul_fast(x, head)
        assert fast_qmm.FAST_HEAD is None and not b.fast and qmm.FAST_MM is None
    kind = ["fp8" if on else "bf16" for on in (False, True) for _ in range(2)]
    assert [c[0] for c in calls] == kind
    assert [c[1]["min_rows"] for c in calls] == [1, fast_qmm.MIN_ROWS] * 2


# -- GPU: kernels, row independence -----------------------------------------------------------------------------------
def _rows_same(fn, x, full, cuts):
    """fn on row slices / permutations of x gives the rows of fn(x) bit for bit."""

    for a, b in cuts:
        got = fn(x[a:b].contiguous())
        assert torch.equal(got, full[a:b]), (a, b)
    perm = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(x.shape[0])).to(x.device)
    assert torch.equal(fn(x[perm].contiguous()), full[perm])


CUTS = [(0, 1), (5, 22), (0, 63), (63, 127), (64, 128), (100, 300), (256, 300), (299, 300), (0, 300)]


@gpu
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize("n,k", [(12576, 4096), (4096, 4096), (4096, 8192), (8192, 512), (4096, 128), (160, 4096)])
def test_matmul_fast_row_independent_any_rows(monkeypatch, n, k, fp8):
    """A row's bits are the same in a call of 1, 17, 63, 64 or 300 rows (patches/0085: no qmm fallback below 64
    rows in a fast chunk), sliced anywhere or permuted; the head keeps qmm's bits below 64 rows."""

    from tensorfold.families.glm5_next.cuda import fast_qmm, fp8pf, qmm

    if fp8 and not fp8pf.available():
        pytest.skip("no FP8 tensor cores")
    monkeypatch.setattr(fp8pf, "ON", fp8)
    g = torch.Generator(device="cpu").manual_seed(n + k)
    q = qmm.quantize4((torch.randn(n, k, generator=g) * 0.02).cuda())
    x = (torch.randn(300, k, generator=g) * (1 + 10 * (torch.rand(300, 1, generator=g) > 0.9))).cuda().bfloat16()
    fn = lambda t: fast_qmm.matmul_fast(t, q, f32=True)              # noqa: E731
    full = fn(x)
    assert torch.equal(full, fn(x))
    _rows_same(fn, x, full, CUTS)
    monkeypatch.setattr(fast_qmm, "FAST_HEAD", q)
    assert torch.equal(fast_qmm.matmul_fast(x[:1], q, f32=True), qmm.matmul(x[:1], q, f32=True))
    if k % 64 == 0:
        b = qmm.make_b16((torch.randn(n, k, generator=g) * 0.02).cuda())
        monkeypatch.setattr(fast_qmm, "FAST_HEAD", None)
        fnb = lambda t: fast_qmm.matmul_fast(t, b, f32=True)         # noqa: E731
        _rows_same(fnb, x, fnb(x), CUTS)


@gpu
def test_hc_partial_row_independent():
    from tensorfold.families.glm5_next.cuda import fast_qmm, glue

    wide, nb = 4 * 4096, glue.HC_BLOCKS
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(300, wide, generator=g, device="cuda").bfloat16()
    fn_w = (torch.randn(24, wide, generator=g, device="cuda") * 0.02).bfloat16()

    def fn(t):
        out = torch.zeros(t.shape[0], nb, 32, device="cuda")
        fast_qmm.hc_partial(t, fn_w, out, nb)
        return out[..., :25]

    _rows_same(fn, x, fn(x), CUTS)


@gpu
def test_latent_tc_row_independent():
    from tensorfold.families.glm5_next.cuda import latent, qmm

    H, DQ, L, DV = 32, 256, 512, 256
    g = torch.Generator(device="cpu").manual_seed(9)
    kk = qmm.quantize4((torch.randn(H * DQ, L, generator=g) * 0.05).cuda())
    kv = qmm.quantize4((torch.randn(H * DV, L, generator=g) * 0.05).cuda())
    q = torch.randn(300, H, DQ, generator=g).cuda().bfloat16()
    u = torch.randn(300, H, L, generator=g).cuda()
    fa = lambda t: latent.absorb_tc(t, kk, torch.empty((t.shape[0], H, L), dtype=torch.bfloat16, device="cuda"))  # noqa
    fe = lambda t: latent.expand_tc(t, kv, torch.empty((t.shape[0], H * DV), dtype=torch.bfloat16, device="cuda"))  # noqa
    _rows_same(fa, q, fa(q), CUTS)
    _rows_same(fe, u, fe(u), CUTS)


@gpu
@pytest.mark.parametrize("R", [300, 4200])
@pytest.mark.parametrize("routing", ["uniform", "skewed"])
def test_fused_experts_pair_independent_of_its_expert_mates(routing, R):
    """A routed pair's gate/up/down result does not depend on which other pairs its expert serves (member count,
    passes of members, the warp and tile the pair lands in): row subsets and permutations of the window. At 4,200
    rows the down kernel takes fast2's large-window tiles (member buffers of 4096+), its subsets the default ones."""

    from test_fastpf_patches import _fast_group
    from test_patches import _exl3_layer

    from tensorfold.families.glm5_next.cuda import exl3_mm

    D, NI, E, TOP, LIMIT = 512, 256, 16, 8, 10.0
    ex = _exl3_layer(D, NI, E, seed=11)
    g = torch.Generator().manual_seed(12)
    x = (torch.randn((R, D), generator=g) * 0.5).to(torch.bfloat16).cuda()
    w = torch.ones(E) if routing == "uniform" else 1.0 / torch.arange(1, E + 1).float() ** 2
    picks = torch.full((R, TOP + 1), E, dtype=torch.int32)
    picks[:, :TOP] = torch.multinomial(w.expand(R, E), TOP, replacement=False, generator=g).int()

    def run(xs, ps):
        n = xs.shape[0]
        grp = _fast_group(ps, E)
        scratch = exl3_mm.Scratch(n, TOP + 1, D, NI, "cuda")
        y = torch.zeros((n * (TOP + 1), D), dtype=torch.float32, device="cuda")
        exl3_mm.routed(xs, ps.cuda(), grp, ex, scratch, y, n, LIMIT, fast=True)
        return y.view(n, TOP + 1, D)[:, :TOP].clone()

    full = run(x, picks)
    assert int((_fast_group(picks, E).members >= 0).sum(1).max()) > 128 or routing == "uniform"
    cuts = CUTS if R == 300 else [(0, 1), (0, 300), (4000, 4200), (1000, 4100), (0, R)]
    for a, b in cuts:
        assert torch.equal(run(x[a:b].contiguous(), picks[a:b].contiguous()), full[a:b]), (a, b)
    perm = torch.randperm(R, generator=torch.Generator().manual_seed(13))
    assert torch.equal(run(x[perm.cuda()].contiguous(), picks[perm].contiguous()), full[perm.cuda()])


@gpu
def test_fast_kda_any_aligned_split_is_one_call():
    """A prompt's KDA through calls cut anywhere on multiples of 64 (64-row calls, 960 + the partial last block,
    uneven cuts) gives the one call's outputs and state bit for bit; fastpf refuses a call off the 64 grid."""

    import test_fastk_patches as fk

    from tensorfold.families.glm5_next.cuda import fastpf

    rows, pos0 = 1000, 2048
    d = fk._kda_inputs(rows, seed=21)
    o1, s1 = fk._fast(d, rows, pos0)

    def split(cuts):
        outs, state, conv = [], d["state_in"], d["conv_state"]
        edges = [0, *cuts, rows]
        for a, b in zip(edges, edges[1:]):
            part = dict(d, p=d["p"][a:], a=d["a"][a:], g=d["g"][a:], conv_state=conv, state_in=state)
            o, s = fk._fast(part, b - a, pos0 + a)
            outs.append(o.clone())
            state = s
            conv = d["p"][b - 3:b, :fk.C].contiguous()
        return torch.cat(outs), state

    for cuts in (list(range(64, rows, 64)), [960], [64], [192, 448, 960], [512]):
        o, s = split(cuts)
        assert torch.equal(o, o1) and torch.equal(s, s1), cuts
    with pytest.raises(ValueError, match="multiples of 64"):
        fastpf.kda_chain(*fk._args(d, 10), None, torch.empty_like(d["state_in"]), pos=pos0 + 3)


# -- GPU: the engine on the synthetic checkpoint ---------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_lookup(monkeypatch):
    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")
    monkeypatch.setenv("GLM53_TF_NONEXPERT", "q4mse")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_cindep")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


@pytest.fixture(scope="module")
def long_ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter
    from test_patches import _index_heads_32

    path = tmp_path_factory.mktemp("glm_cindep_long")
    _checkpoint(path / "model", exl3=True)
    _index_heads_32(path / "model")
    _drafter(path / "dflash2")
    return path


def _engine(path, *, lean_block: int = 0, rows_max: int = 1024, rows="auto", context: int = 0,
            latent_kv: bool = False):
    from test_glm_engine import _TwoCopies

    from tensorfold.families.glm5_next.cuda import weights
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    with pytest.MonkeyPatch.context() as m:
        m.setattr(weights, "NONEXPERT", "q4mse")
        m.setenv("GLM53_TF_NONEXPERT", "q4mse")
        m.setenv("GLM53_TF_FAST_PREFILL", "1")
        m.setenv("GLM53_TF_PREFILL_ROWS", str(rows))
        m.setenv("GLM53_TF_PREFILL_ROWS_MAX", str(rows_max))
        m.setenv("GLM53_TF_LEAN_PREFILL", "1" if lean_block else "0")
        if lean_block:
            m.setenv("GLM53_TF_LEAN_BLOCK", str(lean_block))
        m.setenv("GLM53_TF_LATENT_KV", "1" if latent_kv else "0")
        m.setenv("GLM53_TF_LOOKUP", "0")
        for k in ("GLM53_TF_BATCH", "GLM53_TF_CALIB_ONLINE", "GLM53_TF_PROFILE", "GLM53_TF_DEPTH",
                  "GLM53_TF_FAST_GATHER", "GLM53_TF_FP8_PREFILL", "GLM53_TF_SNAPSHOT_GRID", "GLM53_TF_SNAPSHOT_TAIL",
                  "GLM53_TF_PREFILL_OVERLAP", "GLM53_TF_SESSION_GIB", "GLM53_TF_FP8_TILE"):
            m.delenv(k, raising=False)
        return GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", context=context,
                         comm=_TwoCopies())


@pytest.fixture(scope="module")
def ea(ckpt):
    return _engine(ckpt)                                            # non-lean, 1024-row buffers


@pytest.fixture(scope="module")
def eb(ckpt):
    return _engine(ckpt, lean_block=256, rows_max=8192)            # lean: sub-blocks of 256, chunks up to 8192


@pytest.fixture(scope="module")
def la(long_ckpt):
    return _engine(long_ckpt, context=9300, latent_kv=True)


@pytest.fixture(scope="module")
def lb(long_ckpt):
    return _engine(long_ckpt, lean_block=256, rows_max=8192, context=9300, latent_kv=True)


class _Variant:
    """Sets a fast prefill variant on an engine: chunk rows, FP8, 0084's pipeline (lean chunks only)."""

    def __init__(self, eng, rows: int, fp8: bool = False, overlap: str = "0"):
        self.eng, self.rows, self.fp8, self.overlap = eng, rows, fp8, overlap

    def __enter__(self):
        from tensorfold.families.glm5_next.cuda import fast_qmm, pfoverlap

        self.saved = (self.eng.e.prefill_rows, self.eng.e.fp8_prefill, pfoverlap.MODE, fast_qmm.FP8_MIN_NK)
        self.eng.e.prefill_rows = self.rows
        self.eng.e.fp8_prefill = self.fp8
        if self.fp8:
            fast_qmm.FP8_MIN_NK = 0              # the synthetic matrices are small: every one of them in fp8
        pfoverlap.MODE = pfoverlap.parse(self.overlap)
        return self

    def __exit__(self, *exc):
        from tensorfold.families.glm5_next.cuda import fast_qmm, fp8pf, pfoverlap

        self.eng.e.prefill_rows, self.eng.e.fp8_prefill, pfoverlap.MODE, fast_qmm.FP8_MIN_NK = self.saved
        fp8pf.ON = False


def _state(eng, prompt) -> list:
    """A fresh fast prefill of ``prompt`` (MTP head absorbing) and every piece of committed state: the first token,
    KDA states and conv windows (0065's index-ring tail included), the MTP input row, the attention / latent caches,
    the indexer's full caches and pool keys, the MTP head's caches."""

    from tensorfold.families.glm5_next.cuda.decode import prefill

    e = eng.e
    first = prefill(e, list(prompt), None, mtp=True)
    st = e.st
    n = len(prompt)
    out = [torch.tensor([first]), st.rec[st.cur[0]].clone(), st.conv.clone(), e.last_hidden.clone()]
    kc = list(st.kc) + ([] if st.vc is st.kc else list(st.vc))
    out += [k[:n].clone() for k in kc]
    for ik, ig, pk in (st.index or ()):
        out += [t[:n].clone() for t in (ik, ig) if t.shape[0] >= n] + [pk[:n // 4].clone()]
    out += [st.mtp_kc[:st.mtp_len].clone(), st.mtp_vc[:st.mtp_len].clone(), torch.tensor([st.mtp_len])]
    eng.cache = []
    return out


def _same(a, b) -> bool:
    return len(a) == len(b) and all(x.shape == y.shape and torch.equal(x, y) for x, y in zip(a, b))


def _fp8_ok() -> bool:
    from tensorfold.families.glm5_next.cuda import fp8pf

    return fp8pf.available()


def _states_over_c(ref_eng, lean_eng, prompt, fp8: bool):
    """{variant: state}: the non-lean engine at 64 and 1024 rows, the lean one at every C, pipelined or not."""

    got = {}
    for C in (64, 1024):
        with _Variant(ref_eng, C, fp8):
            got[("plain", C, "0")] = _state(ref_eng, prompt)
    for C in CS:
        for ov in ("0", "1"):
            with _Variant(lean_eng, C, fp8, ov):
                got[("lean", C, ov)] = _state(lean_eng, prompt)
    return got


@gpu
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize("n", [3, 63, 64, 65, 300, 1000])
def test_state_same_for_every_chunk_size(ea, eb, n, fp8):
    if fp8 and not _fp8_ok():
        pytest.skip("no FP8 tensor cores")
    prompt = list(np.random.default_rng(860 + n).integers(0, 1000, size=n))
    got = _states_over_c(ea, eb, prompt, fp8)
    ref = got[("plain", 64, "0")]
    diff = {k: [i for i, (x, y) in enumerate(zip(ref, v)) if not torch.equal(x, y)] for k, v in got.items()
            if not _same(ref, v)}
    assert not diff, diff


@gpu
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize("n", [3000, 9000])
def test_state_same_for_every_chunk_size_long_latent(la, lb, n, fp8):
    """Past the dense limit (sparse top-k attention, 32 index heads, index ring) on the latent cache; at 9,000 tokens
    an 8192-row chunk is followed by a partial one."""

    if fp8 and not _fp8_ok():
        pytest.skip("no FP8 tensor cores")
    prompt = list(np.random.default_rng(870 + n).integers(0, 1000, size=n))
    got = _states_over_c(la, lb, prompt, fp8)
    ref = got[("plain", 64, "0")]
    diff = {k: [i for i, (x, y) in enumerate(zip(ref, v)) if not torch.equal(x, y)] for k, v in got.items()
            if not _same(ref, v)}
    assert not diff, diff


def _gen(eng, prompt, sampling, *, draft=True, policy=None, tokens=24, knobs=None):
    out: list[int] = []
    eng.request.policy = policy
    eng.request.stop_eos = False
    eng.request.knobs = knobs
    try:
        stats = eng.generate(list(prompt), tokens, sampling, lambda new: out.extend(new), draft=draft)
    finally:
        eng.request.knobs = None
    return out, stats


def _cold(eng, prompt, sampling, **kw):
    eng.cache = []
    out, stats = _gen(eng, prompt, sampling, draft=False, **kw)
    assert stats["cached"] == 0
    return out


@gpu
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_resume_with_another_chunk_size_equals_fresh(eb, sampling, fp8):
    """p1 prefilled with 8192-row chunks keeps its state at its last multiple of 64; follow-ups (the reply and new
    tokens) resumed from there with 64 / 1024 / auto rows equal a fresh prefill with yet another C, for MTP, DFlash2
    and auto drafts; a chain of resumes with changing C too."""

    if fp8 and not _fp8_ok():
        pytest.skip("no FP8 tensor cores")
    s = _sampling(sampling)
    rng = np.random.default_rng(880)
    more = lambda n: [int(t) for t in rng.integers(0, 1000, size=n)]       # noqa: E731
    with _Variant(eb, 8192, fp8):
        p1 = more(1500)
        eb.cache = []
        r1, stats = _gen(eb, p1, s, policy="auto:1:1:0", knobs={"prefill_rows": 8192})
        assert stats["fast_prefill"] == 8192 and stats.get("fp8_prefill", 0) == int(fp8)
        assert [len(c.ids) for c in eb.cache] == [1472] and eb.cache[0].grid == 64 + int(fp8)
        for rows, ref_rows, policy in ((64, 1024, "auto:1:1:0"), (1024, "auto", "2"), ("auto", 64, "f3")):
            _gen(eb, p1, s, policy=policy, knobs={"prefill_rows": 8192})       # back to p1's snapshot at 1472
            p2 = p1 + r1 + more(100)
            warm, stats = _gen(eb, p2, s, policy=policy, knobs={"prefill_rows": rows})
            assert stats["cached"] == 1472, (rows, stats["cached"])
            assert warm == _cold(eb, p2, s, knobs={"prefill_rows": ref_rows}), (rows, ref_rows, policy)
        # a chain: each turn resumes from the last one's snapshot (at most tail + 63 tokens before its prompt's
        # end), with a different C each time; then every turn against a fresh prefill with another C
        eb.cache = []
        prompt = more(700)
        reply, _ = _gen(eb, prompt, s, knobs={"prefill_rows": 64})
        chain = []
        for i, rows in enumerate((4096, "auto", 1024, 8192)):
            prev = len(prompt)
            prompt = prompt + reply + more(37 + 90 * i)
            reply, stats = _gen(eb, prompt, s, knobs={"prefill_rows": rows})
            assert stats["cached"] >= prev - 63 - 256, (i, stats["cached"], prev)
            chain.append((prompt, reply))
        for i, (prompt, reply) in enumerate(chain):
            assert reply == _cold(eb, prompt, s, knobs={"prefill_rows": CS[i]}), i


@gpu
@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_drafted_equals_serial_with_auto_rows(eb, sampling):
    s = _sampling(sampling)
    prompt = list(np.random.default_rng(890).integers(0, 1000, size=2000))        # below the synthetic dense limit
    serial = _cold(eb, prompt, s, tokens=32, knobs={"prefill_rows": 64})
    for policy in (None, "auto:1:1:0", "f3", "fc5:0.3", "2", "c3:0.35", "a:0.6:0.85"):
        eb.cache = []
        drafted, stats = _gen(eb, prompt, s, policy=policy, tokens=32, knobs={"prefill_rows": "auto"})
        assert drafted == serial, policy
        assert stats["fast_prefill"] == 2048 and stats["tf_knobs"]["prefill_rows"] == "auto"
        assert [len(c.ids) for c in eb.cache] == [1984]


@gpu
def test_auto_rows_default_and_knob(ckpt, eb):
    """GLM53_TF_PREFILL_ROWS=auto at load (the knob's default, 0 in the header, "auto" in the echo), its choices by
    the rows to prefill, the knob per request, and an integer 0 refused."""

    e = _engine(ckpt, lean_block=256, rows_max=8192, rows="auto")
    assert e.e.prefill_rows == 0 and e._knob_state()["prefill_rows"] == 0
    rng = np.random.default_rng(891)
    for n, C in ((50, 64), (700, 704), (2000, 2048)):
        e.cache = []
        _, stats = _gen(e, list(rng.integers(0, 1000, size=n)), None, tokens=4)
        assert stats["fast_prefill"] == C and stats["tf_knobs"]["prefill_rows"] == "auto", (n, stats)
    _, stats = _gen(e, list(rng.integers(0, 1000, size=700)), None, tokens=4, knobs={"prefill_rows": 128})
    assert stats["fast_prefill"] == 128 and stats["tf_knobs"]["prefill_rows"] == 128
    assert e.e.prefill_rows == 0
    assert eb.parse_knobs({"prefill_rows": "auto"}) == {"prefill_rows": 0}
    with pytest.raises(ValueError, match="1 to 8192"):
        eb.parse_knobs({"prefill_rows": 0})
