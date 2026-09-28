"""patches/0080 (GLM53_TF_FAST_PREFILL=1, ``glm5_next/cuda/fastpf.py``) and 0091 (``tf_knobs.fast_prefill``) on
TensorFold's synthetic GLM checkpoint (one GPU playing rank 0 of two).

Fast prefill runs a prompt's chunks through kernels that are NOT row-invariant (patches/0080's fused EXL3 expert
GEMMs and bf16 rank partials; patches/0081's chunked KDA scan and large-M matmuls when installed), at absolute
multiples of a chunk grid C, and keeps only the state at the prompt's last grid point. Checked:

- the fused expert kernels (``exl3_fast.cu``) against a float64 reference with the kernels' roundings and against
  the row-invariant path (tolerance), deterministic, and a pair's outputs independent of its window, on the synthetic
  shapes and the real model's (4096 x 1024 a rank), with experts past one pass of members;
- the engine with fast prefill: a prefill's state is deterministic; drafted == serial for every policy; a prompt
  resumed from a fast snapshot equals a fresh fast prefill (extension within the last chunk, across grid points,
  after a reply, a chain of resumes, a prompt ending on the grid, a prompt shorter than the grid); only grid
  snapshots are kept (none after a reply); fast and exact snapshots never mix; a resume off the grid is refused;
- the rule, not luck: with ADVERSARIAL stand-ins for patches/0081 (a matmul and a KDA whose bits depend on the
  chunk's row count), two chunk grids give different states, yet resumed == fresh and drafted == serial still hold;
- quality: the fast prefill's hidden rows, logits and KDA states stay within bf16-level tolerance of the
  row-invariant prefill's, and most rows keep their argmax;
- past 2,051 tokens with the latent cache (patches/0060, GLM53_TF_LATENT_KV=1): sparse top-k attention in fast
  chunks, drafted == serial and resumed == fresh.

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda pytest -q tests/cuda/test_fastpf_patches.py
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

try:
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:          # the host-only tests still run
    CUDA = False

from tensorfold.families.glm5_next.cuda import fastpf  # noqa: E402

gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")
SAMPLINGS = ["sampled", "greedy"]
GRID = 128                   # the fast engines' chunk grid (GLM53_TF_PREFILL_ROWS=128)


def _sampling(kind: str):
    from tensorfold.engine.exact_sampling import Sampling

    return None if kind == "greedy" else Sampling(1234, 1.0, 20, 0.95)


# -- host only ------------------------------------------------------------------------------------------------------
def test_grid_is_a_multiple_of_64():
    assert [fastpf.grid(r) for r in (1, 63, 64, 100, 128, 1000, 1024, 2047)] == [64, 64, 64, 64, 128, 960, 1024, 1984]


def test_settings_and_switch(monkeypatch):
    monkeypatch.delenv("GLM53_TF_FAST_PREFILL", raising=False)
    monkeypatch.delenv("GLM53_TF_FAST_GATHER", raising=False)
    assert not fastpf.enabled() and fastpf.gather16()               # off by default; bf16 gathers when on
    monkeypatch.setenv("GLM53_TF_FAST_PREFILL", "1")
    monkeypatch.setenv("GLM53_TF_FAST_GATHER", "fp32")
    assert fastpf.enabled() and not fastpf.gather16()
    assert fastpf.settings()[:2] == [1, 0]
    monkeypatch.setenv("GLM53_TF_FAST_GATHER", "fp8")
    with pytest.raises(ValueError, match="bf16 or fp32"):
        fastpf.gather16()


def test_knob_parses():
    from tensorfold.families.glm5_next.cuda import knobs

    assert knobs.parse({"fast_prefill": 1}, rows_max=512) == {"fast_prefill": 1}
    assert "fast_prefill" in knobs.HEADER
    with pytest.raises(ValueError, match="multiples of 64"):
        knobs.parse({"fast_prefill": 1}, rows_max=32)


# -- host only: the rule on a fake model, through the engine's real prefill / snapshot / resume code ---------------------
class _Fake:
    """A stand-in model on the CPU whose prefill chunks are as hostile as the rule allows: a chunk's rows, its KV
    entries and the state after it depend on the chunk's start and length (a fast chunk) or only on each row
    (an exact chunk, and every decode step), and row p reads a digest of KV positions 0..p-1 (its attention), so a
    stale cache position, a wrong resume point, a wrong MTP length or a mis-chunked resume changes the result.
    Integer hashes, so equal means equal."""

    M = (1 << 61) - 1

    def __init__(self, cap: int = 4096):
        self.kv = torch.zeros(cap, dtype=torch.int64)
        self.dig = [0] * (cap + 1)            # dig[p]: the digest of kv[0:p]
        self.mkv = torch.zeros(cap, dtype=torch.int64)
        self.hidden = torch.zeros((cap, 1), dtype=torch.int64)
        self.ids: list[int] = []

    def mix(self, *v: int) -> int:
        h = 1469598103934665603
        for x in v:
            h = (h * 1099511628211 + int(x) + 12345) % self.M
        return h

    # decode.stage / compute / commit
    def stage(self, w, st, b, tokens):
        self.ids = list(tokens)
        return len(tokens)

    def compute(self, w, st, b, R, *, logits=True, nch=None, host_pos=None, npb=None, fast=False, head=True):
        s = int(st.rec[st.cur[0], 0])
        for i, t in enumerate(self.ids[:R]):
            p = st.pos + i
            s = self.mix(s, t, p, self.dig[p], st.pos if fast else -1, R if fast else -1)
            self.kv[p] = self.mix(s, 7)
            self.dig[p + 1] = self.mix(self.dig[p], int(self.kv[p]))
            self.hidden[i, 0] = s
        st.rec[1 - st.cur[0], 0] = s
        return self.hidden[:R].clone()

    def commit(self, w, st, b, R, keep):
        st.rec[1 - st.cur[0], 0] = self.hidden[keep - 1, 0]          # the state after the kept rows
        st.cur = [1 - st.cur[0]]
        st.conv[0] = torch.tensor(([0, 0, 0] + list(self.ids[:keep]))[-3:], dtype=torch.int64)
        st.set_pos(st.pos + keep)


class _FakeState:
    def __init__(self):
        self.rec = torch.zeros((2, 1), dtype=torch.int64)
        self.conv = torch.zeros((1, 3), dtype=torch.int64)
        self.cur = [0]
        self.pos = self.mtp_len = self.mtp_drafted = 0

    def reset(self):
        self.rec.zero_()
        self.conv.zero_()
        self.cur = [0]
        self.pos = self.mtp_len = self.mtp_drafted = 0

    def set_pos(self, n):
        self.pos = n

    def set_mtp_len(self, n):
        self.mtp_len = n


class _FakeEngine:
    def __init__(self, fake: _Fake, rows: int, fast: bool):
        self.fake, self.st = fake, _FakeState()
        self.w = SimpleNamespace(mtp=object())
        self.buf = None
        self.mbuf = SimpleNamespace(rows=rows)
        self.rows = self.prefill_rows = rows
        self.fast_prefill, self.fast_snap, self.last_hidden = fast, None, None

    @property
    def snap_grid(self) -> int:
        """patches/0085: this fake's fast chunks depend on their start and length (what patches/0080's grid rule
        allowed), so it runs ``pfgrid``'s rule with the snapshot grid at C, which is patches/0080's rule exactly.
        tests/cuda/test_cindep_patches.py checks the 64-token grid on fakes whose chunks do not matter."""

        return fastpf.grid(self.prefill_rows)

    def reset(self):
        self.st.reset()

    def main_hidden(self, rows):
        return self.fake.hidden[rows]

    def forward(self, tokens):
        R = self.fake.stage(None, self.st, None, tokens)
        return self.fake.compute(None, self.st, None, R)

    def mtp(self, next_tokens, hidden):
        for i, t in enumerate(next_tokens):
            self.fake.mkv[self.st.mtp_len + i] = self.fake.mix(int(hidden[i, 0]), t)
        return hidden[-1:]

    def draft_hidden(self, row):
        return self.fake.hidden[:1] * 3 + 1

    def sample(self, logits, positions, sampling, **kw):
        return [int(logits[i, 0]) % 5 for i in range(logits.shape[0])]      # few values: drafts often right


def _fake_engine(monkeypatch, rows: int, fast: bool):
    """A GlmEngine-shaped object over the fake model: the real ``_run`` / ``_resume`` / ``_remember`` / ``_grid``
    and decode's real ``_prefill`` / ``take_snapshot`` / ``restore`` / ``absorb`` / ``serial_decode``."""

    import types

    from tensorfold.families.glm5_next.cuda import decode
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    fake = _Fake()
    for name in ("stage", "compute", "commit"):
        monkeypatch.setattr(decode, name, getattr(fake, name))
    monkeypatch.setattr(decode, "_sync", lambda w: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    g = SimpleNamespace(e=_FakeEngine(fake, rows, fast), drafter=None, cache=[], eos=(), online=None,
                        calib_on=False, costs={}, f_most=7, depth_cost=0, fake=fake)
    for name in ("_run", "_resume", "_remember", "_grid", "_drafters"):
        setattr(g, name, types.MethodType(getattr(GlmEngine, name), g))
    return g


def _fake_request(g, prompt, tokens: int = 40, policy: str = "0"):
    """One request (``policy``: "0" serial, "2" two MTP drafts a round; snapshots kept as for a drafted request): its
    reply and the whole state after it."""

    from tensorfold.families.glm5_next.cuda.engine import encode_policy

    code = encode_policy(policy)
    hit = g._resume(list(prompt), code)
    out: list[int] = []
    stats = g._run(list(prompt), tokens, None, False, out.extend, code, hit, True)
    st, f = g.e.st, g.fake
    state = (st.pos, int(st.rec[st.cur[0], 0]), st.conv.tolist(), f.kv[:st.pos].tolist(), st.mtp_len,
             f.mkv[:st.mtp_len].tolist())
    return out, stats["cached"], state


@pytest.mark.parametrize("policy", ["0", "2"])
@pytest.mark.parametrize("rows", [64, 128, 200])
def test_rule_on_a_hostile_fake_model(monkeypatch, rows, policy):
    """Conversations (a scripted start, then random turns: a new prompt extends the last prompt, or its reply, or a
    prefix of it, or nothing; lengths across and within grid points, on the grid, below it), serial or MTP-drafted:
    every resumed request gives the reply and the whole state (KDA-like state, conv window, KV and MTP caches) of a
    fresh prefill of its prompt, under the fast rule; the exact mode resumes after replies as before."""

    from tensorfold.families.glm5_next.cuda import decode

    rng = np.random.default_rng(rows)
    C = fastpf.grid(rows)

    def use(eng):
        for name in ("stage", "compute", "commit"):
            monkeypatch.setattr(decode, name, getattr(eng.fake, name))

    for fast in (True, False):
        g = _fake_engine(monkeypatch, rows, fast)
        ref = _fake_engine(monkeypatch, rows, fast)          # fresh prefills only, its own fake model
        more = lambda n: [int(t) for t in rng.integers(0, 1000, size=n)]        # noqa: E731
        long = more(2 * C + 17)
        script = [long, more(5), long + more(3), long[:C + 1] + more(C), long + more(2 * C)]
        last, reply = [], []
        resumed = 0
        for turn in range(44):
            if turn < len(script):
                prompt = script[turn]
            else:
                kind = rng.integers(0, 4)
                grow = more(int(rng.choice([1, 3, 30, C - 1, C, C + 5, 2 * C])))
                prompt = (last + reply + grow if kind == 0 else last + grow if kind == 1 else
                          last[:int(rng.integers(0, len(last) + 1))] + grow if kind == 2 else grow)[:3000]
            use(g)
            got, cached, state = _fake_request(g, prompt, policy=policy)
            if fast:
                assert cached % C == 0 and all(s.grid == C and len(s.ids) % C == 0 for s in g.cache)
            else:
                assert all(s.grid == 0 for s in g.cache)
            resumed += cached > 0
            use(ref)
            ref.cache = []
            want, c0, want_state = _fake_request(ref, prompt, policy=policy)
            assert c0 == 0 and got == want and state == want_state, (fast, turn, len(prompt), cached)
            last, reply = prompt, got
        assert resumed >= 10, resumed


# -- the fused expert kernels -----------------------------------------------------------------------------------------
def _reference_routed(x, picks, ex, limit):
    """Float64 routed outputs [R, slots - 1, D] with the kernels' roundings (fp16 rotated inputs, bf16 SwiGLU steps,
    fp16 down input): only the fp32 summation order is left to differ."""

    from tensorfold.families.glm5_next.cuda import exl3

    def unpack(words):          # int32 [K/16, N/16, 32] -> W_q [K, N] float64
        return exl3.unpack(words.cpu().contiguous().view(torch.int16)).double()

    def bf(t):
        return t.to(torch.bfloat16).double()

    R, slots = picks.shape
    out = torch.zeros((R, slots - 1, ex.dims), dtype=torch.float64)
    cache = {}
    xs = x.double().cpu()
    for r in range(R):
        for s in range(slots - 1):
            e = int(picks[r, s])
            if e not in cache:
                cache[e] = [unpack(t[e]) for t in (ex.gt, ex.ut, ex.dt)]
            wg, wu, wd = cache[e]
            xg = exl3.rotate(xs[r] * ex.suh_g[e].cpu().double(), -1).half().double()
            xu = exl3.rotate(xs[r] * ex.suh_u[e].cpu().double(), -1).half().double()
            g = bf(exl3.rotate(xg @ wg, -1) * ex.svh_g[e].cpu().double()).clamp(max=limit)
            u = bf(exl3.rotate(xu @ wu, -1) * ex.svh_u[e].cpu().double()).clamp(-limit, limit)
            act = bf(bf(g * torch.sigmoid(g)) * u)
            xd = exl3.rotate(act * ex.suh_d[e].cpu().double(), -1).half().double()
            out[r, s] = exl3.rotate(xd @ wd, -1) * ex.svh_d[e].cpu().double()
    return out


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().cpu(), b.double().cpu()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


@gpu
@pytest.mark.parametrize("shape", ["synthetic", "real"])
def test_fast_expert_kernels(shape):
    """300 rows over 4 routed experts: experts 0-2 get ~190 members (3 gate/up passes of 64, 2 down passes of 128),
    expert 3 exactly 32. The fused kernels match the float64 reference as closely as the row-invariant path does,
    repeat their bits, and give a pair the same bits alone, in any sub-window and in any order."""

    from test_patches import _exl3_group, _exl3_layer

    from tensorfold.families.glm5_next.cuda import exl3_mm

    D, NI = (512, 128) if shape == "synthetic" else (4096, 1024)
    E, SLOTS, R, LIMIT = 4, 3, 300, 10.0
    ex = _exl3_layer(D, NI, E, seed=17)
    rng = np.random.default_rng(18)
    picks = torch.full((R, SLOTS), E, dtype=torch.int32)
    for r in range(R):
        pair = [3, int(rng.integers(0, 3))] if r < 32 else list(rng.choice(3, size=2, replace=False))
        picks[r, 0], picks[r, 1] = pair[0], pair[1]
    x = (torch.randn((R, D), generator=torch.Generator().manual_seed(19)) * 0.5).to(torch.bfloat16).cuda()

    def run(rows, fast):
        n = len(rows)
        p = picks[rows]
        scratch = exl3_mm.Scratch(n, SLOTS, D, NI, "cuda")
        y = torch.zeros((n * SLOTS, D), dtype=torch.float32, device="cuda")
        exl3_mm.routed(x[rows].contiguous(), p.cuda(), _exl3_group(p), ex, scratch, y, n, LIMIT, fast=fast)
        torch.cuda.synchronize()
        return y.view(n, SLOTS, D)[:, :SLOTS - 1].cpu()

    every = list(range(R))
    fast = run(every, True)
    exact = run(every, False)
    assert torch.isfinite(fast).all() and fast.abs().max() > 0
    check = every if shape == "synthetic" else every[::10]         # the float64 reference is slow on real shapes
    ref = _reference_routed(x[check].cpu(), picks[check], ex, LIMIT)
    e_fast, e_exact = _rel(fast[check], ref), _rel(exact[check], ref)
    assert e_fast < 1e-2 and e_fast < max(3 * e_exact, 1e-3), (e_fast, e_exact)
    assert _rel(fast, exact) < 1e-2
    assert torch.equal(run(every, True), fast)                      # deterministic
    for r in (0, 31, 32, 77, 299):                                  # alone
        assert torch.equal(run([r], True)[0], fast[r]), r
    assert torch.equal(run(list(range(100, 140)), True), fast[100:140])
    rev = every[::-1]                                               # other tile-mates, tile positions and passes
    assert torch.equal(run(rev, True), fast[rev])


# -- engines --------------------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_lookup(monkeypatch):
    """patches/0020's lookup rounds make the windows depend on repeats; pin the MTP/DFlash2 arms. The 4-bit
    non-expert setting is read per request too (plain ``auto`` then chooses between MTP and DFlash2)."""

    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")
    monkeypatch.setenv("GLM53_TF_NONEXPERT", "q4mse")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_fastpf")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


def _engine(path, *, fast: bool = True, rows: int = GRID, rows_max: int = 256, context: int = 0,
            latent_kv: bool = False):
    from test_glm_engine import _TwoCopies

    from tensorfold.families.glm5_next.cuda import weights
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    with pytest.MonkeyPatch.context() as m:
        m.setattr(weights, "NONEXPERT", "q4mse")
        m.setenv("GLM53_TF_NONEXPERT", "q4mse")
        m.setenv("GLM53_TF_FAST_PREFILL", "1" if fast else "0")
        m.setenv("GLM53_TF_PREFILL_ROWS", str(rows))
        m.setenv("GLM53_TF_PREFILL_ROWS_MAX", str(rows_max))
        # patches/0085: snapshots on the chunk grid (patches/0080's rule, which these tests check; the 64-token
        # grid is tests/cuda/test_cindep_patches.py's)
        m.setenv("GLM53_TF_SNAPSHOT_GRID", str(max(64, rows // 64 * 64)))
        m.setenv("GLM53_TF_LATENT_KV", "1" if latent_kv else "0")
        m.setenv("GLM53_TF_LOOKUP", "0")
        for k in ("GLM53_TF_BATCH", "GLM53_TF_CALIB_ONLINE", "GLM53_TF_PROFILE", "GLM53_TF_DEPTH",
                  "GLM53_TF_FAST_GATHER"):
            m.delenv(k, raising=False)
        return GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", context=context,
                         comm=_TwoCopies())


@pytest.fixture(scope="module")
def ef(ckpt):
    return _engine(ckpt)


@pytest.fixture(scope="module")
def ex(ckpt):
    return _engine(ckpt, fast=False)


def _adversarial():
    """Stand-ins for patches/0081 whose bits depend on a chunk's row count (as large-M tiles and chunked scans may):
    deterministic, close to the row-invariant kernels, and useless to the rule unless chunks sit on the grid."""

    from tensorfold.families.glm5_next.cuda import kda, qmm

    def matmul_prefill(x, q, xs=None, *, out=None, f32=False, part=None):
        y = qmm.matmul(x, q, xs, out=out, f32=f32, part=part)         # the hook is cleared during this call
        return y.mul_(1.0 + 2.0 ** -7 * ((x.shape[0] % 5) - 2))

    def kda_prefill_chunked(*args, pos):
        assert pos % fastpf.GRID == 0
        out = kda.chain(*args)
        return out.mul_(1.0 + 2.0 ** -8 * ((args[12] % 3) - 1))       # args[12]: the chunk's rows

    return SimpleNamespace(matmul_prefill=matmul_prefill), SimpleNamespace(kda_prefill_chunked=kda_prefill_chunked)


@pytest.fixture(params=["installed", "adversarial"])
def kernels(request, monkeypatch):
    """The fast kernels of this patch set (0081's when present), or the adversarial stand-ins for 0081."""

    if request.param == "adversarial":
        mm, kd = _adversarial()
        monkeypatch.setattr(fastpf, "fast_qmm", mm)
        monkeypatch.setattr(fastpf, "fast_kda", kd)
    return request.param


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
    """A fresh prefill (every snapshot dropped) and serial decoding: the reference."""

    eng.cache = []
    out, stats = _gen(eng, prompt, sampling, draft=False, **kw)
    assert stats["cached"] == 0
    return out


def _grid_only(eng, n: int):
    """Every kept snapshot is a fast one on the grid, at most at the last grid point of an n-token prompt."""

    assert eng.cache and all(s.grid == GRID and len(s.ids) % GRID == 0 and 0 < len(s.ids) <= n for s in eng.cache)
    assert len(eng.cache[-1].ids) == (n // GRID) * GRID


@gpu
def test_fast_prefill_on_and_deterministic(ef, kernels):
    from test_patches import _prefill_state, _same

    assert ef.e.fast_prefill and ef._grid() == GRID
    prompt = list(np.random.default_rng(71).integers(0, 1000, size=700))
    a = _prefill_state(ef, prompt)
    _gen(ef, list(np.random.default_rng(72).integers(0, 1000, size=90)), None, tokens=4)   # anything in between
    b = _prefill_state(ef, prompt)
    assert _same(a, b)


@gpu
@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_fast_drafted_equals_serial(ef, kernels, sampling):
    s = _sampling(sampling)
    prompt = list(np.random.default_rng(73).integers(0, 1000, size=300))
    serial = _cold(ef, prompt, s, tokens=32)
    for policy in (None, "auto:1:1:0", "f3", "fc5:0.3", "2", "c3:0.35", "a:0.6:0.85"):
        ef.cache = []
        drafted, stats = _gen(ef, prompt, s, policy=policy, tokens=32)
        assert drafted == serial, policy
        assert stats["fast_prefill"] == GRID
        _grid_only(ef, len(prompt))


@gpu
@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_fast_resumed_equals_fresh(ef, kernels, sampling):
    """Resume == fresh under the grid rule (C = 128): within the last chunk, across grid points, after a reply,
    along a chain of resumes, on the grid, and shorter than the grid; with MTP and DFlash2 drafts."""

    s = _sampling(sampling)
    rng = np.random.default_rng(74)
    more = lambda n: [int(t) for t in rng.integers(0, 1000, size=n)]       # noqa: E731
    p1 = more(300)                                   # grid points 128, 256; the snapshot at 256
    ef.cache = []
    r1, stats = _gen(ef, p1, s, policy="auto:1:1:0")
    _grid_only(ef, len(p1))                          # the reply's rows are not kept
    cases = [
        ("within", p1 + more(3), 256),               # the new prompt's last grid point is still 256
        ("reply", p1 + r1 + more(5), 256),           # 329 tokens: re-prefills 256.. (tail, reply, new)
        ("across", p1 + r1 + more(120), 256),        # 444 tokens: the new snapshot at 384
    ]
    for name, prompt, cached in cases:
        for policy in ("auto:1:1:0", "2", "f3"):
            _gen(ef, p1, s, policy="auto:1:1:0")     # back to the state after p1 (its snapshot at 256)
            warm, stats = _gen(ef, prompt, s, policy=policy)
            assert stats["cached"] == cached, (name, policy, stats["cached"])
            _grid_only(ef, len(prompt))
            assert warm == _cold(ef, prompt, s), (name, policy)
    # a chain: p2 resumes from p1's snapshot, p3 from p2's (made by a resumed prefill)
    _gen(ef, p1, s)
    p2 = p1 + r1 + more(120)
    _gen(ef, p2, s)
    p3 = p2 + more(200)
    warm, stats = _gen(ef, p3, s)
    assert stats["cached"] == 384 and warm == _cold(ef, p3, s)
    # a prompt ending on the grid keeps its end state, pending MTP row included
    p4 = more(256)
    ef.cache = []
    _gen(ef, p4, s)
    assert len(ef.cache[-1].ids) == 256 and ef.cache[-1].mtp_len == 255
    tail = more(2)
    warm, stats = _gen(ef, p4 + tail, s, policy="2")
    assert stats["cached"] == 256 and warm == _cold(ef, p4 + tail, s)
    # shorter than the grid: nothing to keep, nothing to resume
    ef.cache = []
    _gen(ef, more(50), s)
    assert ef.cache == []


@gpu
def test_fast_and_exact_snapshots_never_mix(ef):
    """patches/0091: a request with fast_prefill=0 on the fast engine ignores fast snapshots (and the other way
    round), and its reply equals the exact engine's serial reference."""

    s = None
    prompt = list(np.random.default_rng(75).integers(0, 1000, size=300))
    longer = prompt + [7, 8, 9]
    ef.cache = []
    _gen(ef, prompt, s)                                              # a fast snapshot at 256
    exact, stats = _gen(ef, longer, s, knobs={"fast_prefill": 0})
    assert stats["cached"] == 0 and "fast_prefill" not in stats and stats["tf_knobs"]["fast_prefill"] == 0
    assert ef.cache and all(c.grid == 0 for c in ef.cache)          # exact snapshots (prompt and reply)
    assert exact == _cold(ef, longer, s, knobs={"fast_prefill": 0})
    _gen(ef, prompt, s, knobs={"fast_prefill": 0})                  # an exact snapshot of the prompt
    fast, stats = _gen(ef, longer, s)
    assert stats["cached"] == 0 and stats["fast_prefill"] == GRID
    assert ef._knob_state()["fast_prefill"] == 1                     # the load-time default is back


@gpu
def test_resume_off_the_grid_is_refused(ef):
    from tensorfold.families.glm5_next.cuda.decode import prefill, take_snapshot

    e = ef.e
    prompt = list(np.random.default_rng(76).integers(0, 1000, size=300))
    ef.cache = []
    prefill(e, prompt, None, mtp=True)
    off = take_snapshot(e, prompt, e.last_hidden, mtp=True, grid=GRID)        # 300: not a grid point
    with pytest.raises(ValueError, match="grid"):
        prefill(e, prompt + [1, 2], None, mtp=True, resume=off)
    exact = take_snapshot(e, prompt[:256], e.last_hidden, mtp=True, grid=0)
    with pytest.raises(ValueError, match="grid"):
        prefill(e, prompt + [1, 2], None, mtp=True, resume=exact)
    ef.cache = []


@gpu
def test_adversarial_kernels_do_depend_on_the_chunks(ckpt, monkeypatch):
    """The control for the adversarial runs above: with the stand-ins, grids 128 and 192 give different states, so
    their resumed == fresh holds because of the grid rule, not because the chunking happens not to matter."""

    from test_patches import _prefill_state, _same

    mm, kd = _adversarial()
    monkeypatch.setattr(fastpf, "fast_qmm", mm)
    monkeypatch.setattr(fastpf, "fast_kda", kd)
    e = _engine(ckpt, rows=GRID)
    prompt = list(np.random.default_rng(77).integers(0, 1000, size=500))
    a = _prefill_state(e, prompt)
    e.e.prefill_rows = 192
    b = _prefill_state(e, prompt)
    assert not _same(a, b)


@gpu
def test_fast_state_close_to_exact(ef, ex):
    """Quality: the same 300-token prompt through the fast prefill and the row-invariant one. The last chunk's
    final-normed rows, their logits and every KDA state agree to bf16-level tolerance; most rows keep their argmax."""

    from tensorfold.families.glm5_next.cuda import qmm
    from tensorfold.families.glm5_next.cuda.decode import prefill

    prompt = list(np.random.default_rng(78).integers(0, 1000, size=300))
    got = {}
    for name, eng in (("fast", ef), ("exact", ex)):
        eng.cache = []
        prefill(eng.e, prompt, None, mtp=True)
        rows = eng.e.buf.fnormed[:len(prompt) - 256].clone()        # the last chunk, rows 256..299, in both
        got[name] = (rows, qmm.matmul(rows, eng.w.head).float(), eng.e.st.rec[eng.e.st.cur[0]].clone())
        eng.cache = []
    (hf, lf, sf), (he, le, se) = got["fast"], got["exact"]
    assert _rel(hf, he) < 3e-2 and _rel(lf, le) < 5e-2 and _rel(sf, se) < 5e-2, (_rel(hf, he), _rel(lf, le),
                                                                                 _rel(sf, se))
    same = (lf.argmax(-1) == le.argmax(-1)).float().mean().item()
    assert same >= 0.75, same


# -- past 2,051 tokens, latent cache ----------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def long_ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter
    from test_patches import _index_heads_32

    path = tmp_path_factory.mktemp("glm_fastpf_long")
    _checkpoint(path / "model", exl3=True)
    _index_heads_32(path / "model")
    _drafter(path / "dflash2")
    return path


@gpu
@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_fast_long_context_latent(long_ckpt, sampling):
    """A 3,000-token prompt crosses the dense limit in fast chunks of 256 on the latent cache: drafted == serial, and
    a follow-up resumed from the snapshot at 2,816 equals a fresh prefill."""

    s = _sampling(sampling)
    e = _engine(long_ckpt, rows=256, rows_max=256, context=4096, latent_kv=True)
    prompt = list(np.random.default_rng(79).integers(0, 1000, size=3000))
    serial = _cold(e, prompt, s)
    for policy in (None, "2", "f3"):
        e.cache = []
        drafted, _ = _gen(e, prompt, s, policy=policy)
        assert drafted == serial, policy
    assert len(e.cache[-1].ids) == 2816 and e.cache[-1].grid == 256
    after = prompt + serial + [3, 4]
    cold = _cold(e, after, s, tokens=16)
    # A snapshot keeps only the draft caches of the request that made it (f3: DFlash2 taps, no MTP rows), so the
    # follow-up names a policy it fits. The default ``auto`` may or may not need MTP rows: ``_effective`` narrows it
    # from load-time timings, which made this check flaky when it resumed with the default policy.
    for policy in ("f3", "2"):
        e.cache = []
        _gen(e, prompt, s, policy=policy)
        warm, stats = _gen(e, after, s, tokens=16, policy=policy)
        assert stats["cached"] == 2816, policy
        assert warm == cold, policy


def _fast_group(picks: torch.Tensor, E: int):
    """``_exl3_group`` vectorized (a 1024-row window of 288 experts): distinct experts ascending, each one's
    (row * 32 + slot) members in row order, one spare entry."""

    from types import SimpleNamespace

    n, slots = picks.shape
    flat = picks.flatten().long()
    code = (torch.arange(n).repeat_interleave(slots) * 32 + torch.arange(slots).repeat(n)).int()
    used = torch.unique(flat)
    members = torch.full((used.numel() + 1, n), -1, dtype=torch.int32)
    for u, e in enumerate(used.tolist()):
        m = code[flat == e]
        members[u, :m.numel()] = m
    ids = torch.zeros((used.numel() + 1,), dtype=torch.int32)
    ids[:used.numel()] = used.int()
    return SimpleNamespace(ids=ids.cuda(), count=torch.tensor([used.numel()], dtype=torch.int32).cuda(),
                           members=members.cuda())


@gpu
@pytest.mark.parametrize("rows", [1024, 2048])
def test_fast_expert_timing(rows):
    """Prints the routed experts of one MoE layer on GLM-5.3-Flash's per-rank shapes (288 experts, top 8, hidden
    4096, 1024 of each expert's 2048 width), row-invariant path (grouped loop + epilogues) vs the fused kernels, with
    uniform and skewed (Zipf-like) routing. No assertion on speed."""

    from test_patches import _exl3_layer

    from tensorfold.families.glm5_next.cuda import exl3_mm

    D, NI, E, TOP, LIMIT = 4096, 1024, 288, 8, 10.0
    ex = _exl3_layer(D, NI, E, seed=5)
    x = (torch.randn((rows, D), generator=torch.Generator().manual_seed(6)) * 0.5).to(torch.bfloat16).cuda()
    g = torch.Generator().manual_seed(7)
    for kind in ("uniform", "skewed"):
        w = torch.ones(E) if kind == "uniform" else 1.0 / torch.arange(1, E + 1).float() ** 0.8
        picks = torch.full((rows, TOP + 1), E, dtype=torch.int32)
        picks[:, :TOP] = torch.multinomial(w.expand(rows, E), TOP, replacement=False, generator=g).int()
        grp = _fast_group(picks, E)
        dev_picks = picks.cuda()
        scratch = exl3_mm.Scratch(rows, TOP + 1, D, NI, "cuda")
        y = torch.zeros((rows * (TOP + 1), D), dtype=torch.float32, device="cuda")
        prev = exl3_mm.LOOP
        exl3_mm.LOOP = True
        try:
            t0 = _time(lambda: exl3_mm.routed(x, dev_picks, grp, ex, scratch, y, rows, LIMIT, fast=False))
            t1 = _time(lambda: exl3_mm.routed(x, dev_picks, grp, ex, scratch, y, rows, LIMIT, fast=True))
        finally:
            exl3_mm.LOOP = prev
        busiest = int((grp.members >= 0).sum(1).max())
        print(f"\nexperts {rows} rows {kind} ({int(grp.count[0])} experts, busiest {busiest} members): "
              f"row-invariant {t0:.2f} ms, fused {t1:.2f} ms, x{t0 / t1:.2f}")


def _time(fn, reps=10):
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps
