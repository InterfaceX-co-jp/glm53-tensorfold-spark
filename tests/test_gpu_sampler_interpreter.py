"""patches/0450 (GLM53_TF_GPU_ROUND ``sample``): the device sampler (``gpusample``) in Triton's CPU interpreter
against the host sampler, bit for bit.

- ``_log`` / ``_exp`` (glibc 2.39's aarch64 ``__log`` / ``__exp``, ported instruction for instruction) against numpy's
  ``np.log`` / ``np.exp`` on this machine: the sampler's domains (uniforms 2^-54 .. 1 - 2^-54 and their -log for the
  Gumbel noise; exp over (-inf, 0] with gaps up to 1e6: tiny, normal, subnormal, underflow) and random doubles of every
  exponent. numpy calls libm for float64 on aarch64 (the image); on x86-64 without AVX-512 it does too (checked:
  glibc's FMA build gives the same bits). tests/libm_a64.py runs the port against Ubuntu's aarch64 libm itself.
- The draw (``_choose_kernel``) against ``exact_sampling.choose_rows`` (sampled), the greedy lexsort
  (``batch.sample_multi``) and ``decode._probability`` (the drafts' probabilities): one and two ranks, top_k 0 / 1 / 5 /
  20 / 40 / 64 / 120, top_p 0 / 0.5 / 0.9 / 0.95 / 1, temperatures 1e-7 .. 1.5, tied values, huge positions and seeds.
- numpy's own float64 ``sum`` along a row is the pairwise sum the kernel reproduces (a numpy that changed it would fail
  here first).
- ``batch.sample_multi`` / ``sample_drafts`` / ``decode.sample_rows`` with the knob give the host path's tokens (one
  and two ranks, riders included), and ``gpusample.self_check`` passes.

Run: TRITON_INTERPRET=1 PYTHONPATH=<patched TensorFold>/src:tests pytest -q tests/test_gpu_sampler_interpreter.py
(~3 min; GLM53_TF_SAMPLER_CASES=N raises the random draw cases from 300).
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

os.environ.setdefault("TRITON_INTERPRET", "1")
sys.path.insert(0, os.path.dirname(__file__))

import pytest  # noqa: E402

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
gs = pytest.importorskip("tensorfold.families.glm5_next.cuda.gpusample")

import gpuround_interp  # noqa: E402
from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows  # noqa: E402

INTERP = os.environ.get("TRITON_INTERPRET") == "1" and type(gs._choose_kernel).__name__ == "InterpretedFunction"
interp = pytest.mark.skipif(not INTERP or torch.cuda.is_available(),
                            reason="Triton's CPU interpreter, no GPU (tests/cuda/test_gpu_round_patches.py on GPUs)")
CASES = int(os.environ.get("GLM53_TF_SAMPLER_CASES", "300"))


@pytest.fixture(autouse=True, scope="module")
def _interp():
    gpuround_interp.install()


def _bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64).view(np.int64)


# -- libm --------------------------------------------------------------------------------------------------------------
@interp
def test_log_on_the_uniforms_and_their_gumbel():
    rng = np.random.default_rng(1)
    q = rng.integers(0, 1 << 53, size=60000, dtype=np.int64)
    q[:6] = [0, 1, 2, (1 << 53) - 1, (1 << 52), (1 << 52) - 1]
    u = q.astype(np.float64) * 2.0 ** -53 + 2.0 ** -54
    got = gs.libm(torch.from_numpy(u), "log").numpy()
    assert np.array_equal(_bits(got), _bits(np.log(u)))
    got = gs.libm(torch.from_numpy(u), "gumbel").numpy()
    assert np.array_equal(_bits(got), _bits(np.log(-np.log(u))))


@interp
def test_log_near_one_and_everywhere():
    rng = np.random.default_rng(2)
    near = 1.0 + rng.uniform(-0.07, 0.07, size=30000)
    near[:4] = [1.0, np.nextafter(1.0, 0), np.nextafter(1.0, 2), 1.0 - 2.0 ** -4]
    wide = np.exp(rng.uniform(-700, 700, size=30000))
    for x in (near, wide):
        got = gs.libm(torch.from_numpy(x), "log").numpy()
        assert np.array_equal(_bits(got), _bits(np.log(x)))


@interp
def test_exp_every_range():
    rng = np.random.default_rng(3)
    xs = [-np.abs(rng.standard_normal(20000) * s) for s in (1e-17, 1e-3, 1.0, 30.0, 700.0, 745.0, 2000.0)]
    edge = np.array([0.0, -0.0, -2.0 ** -54, -2.0 ** -55, -708.3964185322641, -708.4, -745.1332191019411, -745.14,
                     -1023.9, -1024.0, -1e6, -np.inf, -512.0, -511.99, 1e-20, 5.0, 709.7])
    for x in xs + [edge, rng.uniform(-1100, 709, size=20000)]:
        with np.errstate(all="ignore"):
            want = np.exp(x)
        got = gs.libm(torch.from_numpy(x), "exp").numpy()
        assert np.array_equal(_bits(got), _bits(want)), x[_bits(got) != _bits(want)][:4]


def test_numpy_sums_rows_pairwise():
    """The kernel's ``_pw_sum`` is numpy's pairwise summation from the identity (8 accumulators up to 128)."""

    def pw(a):
        n = len(a)
        if n < 8:
            r = -0.0
            for x in a:
                r += x
            return r
        r = list(a[:8])
        i = 8
        while i < n - n % 8:
            for j in range(8):
                r[j] += a[i + j]
            i += 8
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        while i < n:
            res += a[i]
            i += 1
        return res

    rng = np.random.default_rng(4)
    for _ in range(2000):
        k = int(rng.integers(1, 129))
        x = np.exp(rng.standard_normal((2, k)) * 3)
        s = x.sum(axis=-1, keepdims=True)
        for r in range(2):
            assert float(s[r, 0]) == pw([float(v) for v in x[r]])
        c = np.cumsum(x[0])
        run = 0.0
        for i in range(k):
            run += float(x[0, i])
            assert float(c[i]) == run


# -- the draw ----------------------------------------------------------------------------------------------------------
def _case(rng, world: int, k: int, rows: int, s, want: bool):
    vals = (rng.standard_normal((rows, world * k)) * rng.choice([0.5, 3, 20])).astype(np.float32)
    if rng.random() < 0.5:                      # ties
        vals = np.round(vals * 2) / 2
    if rng.random() < 0.2:
        vals[:, ::3] = -0.0
    ids = np.stack([rng.permutation(154880)[:world * k] for _ in range(rows)]).astype(np.int64)
    P = rows * 2 * k + 5
    got = np.zeros((world, P), dtype=np.float32)
    for r in range(world):
        seg = np.concatenate([vals[:, r * k:(r + 1) * k], ids[:, r * k:(r + 1) * k].astype(np.int32).view(np.float32)],
                             axis=1)
        got[r, :rows * 2 * k] = seg.reshape(-1)
    pos = [int(p) for p in rng.integers(0, 1 << 40, size=rows)]
    tok, prob = gs.choose(torch.from_numpy(got.reshape(-1)), P, world,
                          [(i * 2 * k, k, pos[i], s, want) for i in range(rows)], want=want)
    return vals, ids, pos, tok.tolist(), prob


@interp
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_draws_equal_the_host(seed):
    from tensorfold.families.glm5_next.cuda.decode import _probability

    rng = np.random.default_rng(100 + seed)
    done = 0
    while done < CASES // 3:
        world = int(rng.choice([1, 2]))
        if rng.random() < 0.25:
            s = None if rng.random() < 0.5 else Sampling(1, 0.0, 20, 0.95)
            k = 1 if rng.random() < 0.5 else 20 + MARGIN
        else:
            s = Sampling(seed=int(rng.integers(0, 1 << 63)), temperature=float(rng.choice([1e-7, 0.3, 0.7, 1.0, 1.5])),
                         top_k=int(rng.choice([0, 1, 5, 20, 40, 64, 120])),
                         top_p=float(rng.choice([0.0, 0.5, 0.9, 0.95, 1.0])))
            k = s.top_k + MARGIN
        want = bool(rng.random() < 0.5)
        if want and (s is None or s.temperature <= 0):
            k = 20 + MARGIN
        if not gs.fits(k, world, s, want):
            continue
        vals, ids, pos, tok, prob = _case(rng, world, k, int(rng.integers(1, 9)), s, want)
        if s is None or s.temperature <= 0:
            order = np.lexsort((ids, -vals), axis=-1)
            host = [int(ids[i, order[i, 0]]) for i in range(len(pos))]
        else:
            host = choose_rows(vals, ids, pos, s)
        assert tok == host, (s, world, k)
        if want:
            assert [float(p) for p in prob.tolist()] == _probability(vals, ids, host, s), (s, world, k)
        done += 1


def test_fits():
    assert gs.fits(28, 2, Sampling(1, 1.0, 20, 0.95))
    assert gs.fits(1, 2, None)
    assert not gs.fits(129 + MARGIN, 2, Sampling(1, 1.0, 129, 0.95))           # 129 kept: the host draws it
    assert not gs.fits(200, 2, Sampling(1, 1.0, 0, 0.95))                      # 400 gathered
    assert gs.fits(8, 2, Sampling(1, 1.0, 0, 0.95))                            # top_k 0: every gathered candidate


# -- the samplers with the knob ----------------------------------------------------------------------------------------
class _Comm:
    """Rank 0 of two: the other rank's part comes from ``other(n)``."""

    def __init__(self, other):
        self.other = other

    def all_gather(self, send, recv):
        n = send.numel()
        recv[:n].copy_(send)
        recv[n:].copy_(self.other(n))


def _w(world: int, gpu: bool, other=None):
    return SimpleNamespace(comm=None if world == 1 else _Comm(other), vocab_offset=0, world=world,
                           meta={"gpu_sample": gpu})


@interp
@pytest.mark.parametrize("world", [1, 2])
def test_samplers_with_the_knob_equal_the_host(world):
    from tensorfold.families.glm5_next.cuda import batch, decode

    rng = np.random.default_rng(7 + world)
    V = 300
    for trial in range(8):
        logits = torch.from_numpy((rng.standard_normal((9, V)) * 3).astype(np.float32)).to(torch.bfloat16)
        other = torch.from_numpy((rng.standard_normal((9, V)) * 3).astype(np.float32)).to(torch.bfloat16)
        s = [None, Sampling(seed=123 + trial, temperature=0.8, top_k=20, top_p=0.9)][trial % 2]
        specs = [(0, 3, [10, 11, 12], s), (3, 6, [50 + i for i in range(6)], Sampling(7, 1.0, 5, 1.0))]

        def other_part(n):
            parts = []
            for off, R, _, sp in specs:
                k = 1 if sp is None or sp.temperature <= 0 else min(V, sp.top_k + MARGIN)
                vals, ids = torch.topk(other[off:off + R].float(), k, dim=-1)
                parts.append(torch.cat([vals, (ids + V).to(torch.int32).view(torch.float32)], 1).reshape(-1))
            flat = torch.cat(parts)
            return torch.cat([flat, torch.zeros(n - flat.numel())])       # a zero rider

        rider = torch.tensor([1, 2, 7, 9, 0, 0], dtype=torch.int32)
        a = batch.sample_multi(_w(world, False, other_part), logits, specs, rider=rider)
        b = batch.sample_multi(_w(world, True, other_part), logits, specs, rider=rider)
        assert a == b
        if world == 1:
            pa, pb = [], []
            ra = decode.sample_rows(_w(1, False), logits[:4], [5, 6, 7, 8], s, probs=pa, draft=True)
            rb = decode.sample_rows(_w(1, True), logits[:4], [5, 6, 7, 8], s, probs=pb, draft=True)
            assert ra == rb and pa == pb
            ds = [(r, 40 + r, s, bool(r % 2)) for r in range(4)]
            assert batch.sample_drafts(_w(1, False), logits, ds) == batch.sample_drafts(_w(1, True), logits, ds)


@interp
def test_self_check_passes_here():
    assert gs.self_check("cpu", rows=64) is None
