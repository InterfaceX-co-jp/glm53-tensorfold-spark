"""patches/0360: ``tf_knobs.b12x`` bit 4 (one-pass sparse latent attention, patches/0240) made fit for the production
configuration (FP8 latent KV, patches/0220; sessions, 0110 / 0180 / 0250). Scope: bit 4 only (bits 1 / 2 stay as 0240
left them, not adopted).

What 0360 changes and what is checked here:

1. Session reuse: the lone engine looked snapshots up with the rank's DEFAULT b12x bits (``generate`` called ``_grid``
   without ``b12x``) while the prefill tagged them with the request's bits, so a request with b12x bits never resumed
   (W3: ``cached`` 0 in every drafted == serial / resumed == fresh test and in the same-bits control). Now
   ``GlmEngine._request_grid(values)`` is the lookup tag (the batcher already passed the bits). Host: the real
   ``_run`` / ``_resume`` / ``decode._prefill`` on the hostile fake model, bits 4 / 7: follow-ups resume (cached > 0) and
   end in the fresh state; other bits never resume them; the 0240 lookup (control) never resumes; the batcher's tag ==
   the engine's. The NVMe tier's compat hash no longer includes GLM53_TF_B12X (the default; entries carry the bits
   in their tag) but keeps GLM53_TF_B12X_KDA_*.
2. FP8 == bf16 on the dequantized rows, by construction (``b12x_attn._opaque``: the dots' operand layouts of the bf16
   kernel; tests/test_b12x_onepass_compile.py checks them offline). GPU: on the real shapes (32 heads, 2,051 tokens),
   the W3 check that failed; row subsets / single rows / permutations / duplicates bitwise, bf16 and FP8, paged.
3. Engine with bit 4 on the latent cache past the dense limit (the toy checkpoint's latent is 128 wide, so bf16 KV;
   FP8 engine checks run on the real model, docs/PATCHES.md 0360's GPU plan): committed state C- and pipeline-
   independent, drafted == serial, resumed == fresh WITH cached > 0 through ``generate``, snapshots never cross
   bits 4 / 0, batch-mode sessions.

Host-only tests run anywhere with torch (CPU); kernel and engine tests need a GPU.
    PYTHONPATH=<tree>/src:/src/TensorFold/tests/cuda:tests/cuda pytest -q -s tests/cuda/test_b12x_attn_patches.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

try:
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    CUDA = False

gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")
needs_torch = pytest.mark.skipif(torch is None, reason="torch")

sys.path.insert(0, str(Path(__file__).parent))

ROWS = 200


# -- host: the lookup tag and session reuse -------------------------------------------------------------------------
def _values(bits: int, fast: int = 1, fp8: int = 0, rows: int = ROWS) -> dict:
    return {"fast_prefill": fast, "prefill_rows": rows, "fp8_prefill": fp8, "b12x": bits}


def _g(monkeypatch, fast: bool = True):
    from test_fastpf_patches import _fake_engine

    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    g = _fake_engine(monkeypatch, ROWS, fast)
    g._request_grid = types.MethodType(GlmEngine._request_grid, g)
    return g


class _Bits:
    """What ``GlmEngine._knobs`` does with the request's b12x value around ``_run`` (both ranks)."""

    def __init__(self, bits: int):
        self.bits = bits

    def __enter__(self):
        from tensorfold.families.glm5_next.cuda import b12xpf

        self.saved = b12xpf.BITS
        b12xpf.set_bits(self.bits)

    def __exit__(self, *exc):
        from tensorfold.families.glm5_next.cuda import b12xpf

        b12xpf.set_bits(self.saved)


def _turn(g, prompt, bits: int, policy: str = "2", lookup: str = "0360", tokens: int = 30):
    """``GlmEngine.generate`` on rank 0 without a store: the lookup tag from the request's knobs BEFORE they are
    applied (0360: ``_request_grid``; control "0240": ``_grid(fast, rows, fp8)``), then ``_run`` under the knobs."""

    from tensorfold.families.glm5_next.cuda.engine import encode_policy

    code = encode_policy(policy)
    v = _values(bits)
    grid = g._request_grid(v) if lookup == "0360" else g._grid(bool(v["fast_prefill"]), v["prefill_rows"],
                                                                   bool(v["fp8_prefill"]))
    hit = g._resume(list(prompt), code, grid)
    out: list[int] = []
    with _Bits(bits):
        stats = g._run(list(prompt), tokens, None, False, out.extend, code, hit, True)
    st, f = g.e.st, g.fake
    state = (st.pos, int(st.rec[st.cur[0], 0]), st.conv.tolist(), f.kv[:st.pos].tolist(), st.mtp_len,
             f.mkv[:st.mtp_len].tolist())
    return out, stats, state


@needs_torch
def test_request_grid_carries_the_bits(monkeypatch):
    from tensorfold.families.glm5_next.cuda import b12xpf, pfgrid

    g = _g(monkeypatch)
    assert b12xpf.BITS == 0                                  # between requests: the default
    G = g._grid()
    for bits in range(8):
        assert g._request_grid(_values(bits)) == G + pfgrid.B12X * bits
        with _Bits(bits):
            assert g._grid() == g._request_grid(_values(bits))          # == what the prefill tags
    assert g._request_grid(_values(4, fast=0)) == 0                      # exact prefill: no bits
    assert g._request_grid(_values(4, fp8=1)) == G + 1 + 16


@needs_torch
def test_batcher_tag_equals_the_engine_tag(monkeypatch):
    from tensorfold.families.glm5_next.cuda import batch

    g = _g(monkeypatch)
    fake = types.SimpleNamespace(g=g)
    for bits in (0, 4, 7):
        for fp8 in (0, 1):
            assert batch.Batcher._grid(fake, _values(bits, fp8=fp8)) == g._request_grid(_values(bits, fp8=fp8))


@needs_torch
@pytest.mark.parametrize("policy", ["0", "2"])
@pytest.mark.parametrize("bits", [4, 7, 0])
def test_b12x_follow_ups_resume_and_equal_fresh(monkeypatch, bits, policy):
    """Conversations with the bits on: every follow-up resumes a snapshot of the same bits (cached > 0, on the grid)
    and ends in the state and reply of a fresh prefill (a fresh fake engine)."""

    from test_fastpf_patches import _fake_engine, _fake_request
    from tensorfold.families.glm5_next.cuda import decode

    rng = np.random.default_rng(3600 + bits + len(policy))
    g = _g(monkeypatch)
    ref = _fake_engine(monkeypatch, ROWS, True)

    def use(eng):
        for name in ("stage", "compute", "commit"):
            monkeypatch.setattr(decode, name, getattr(eng.fake, name))

    prompt = [int(t) for t in rng.integers(0, 1000, size=700)]
    use(g)
    reply, stats, _ = _turn(g, prompt, bits, policy)
    assert stats["cached"] == 0
    for turn in range(6):
        prompt = prompt + reply + [int(t) for t in rng.integers(0, 1000, size=int(rng.integers(5, 300)))]
        use(g)
        reply, stats, state = _turn(g, prompt, bits, policy)
        assert stats["cached"] > 0 and stats["cached"] % g.e.snap_grid == 0, (turn, stats)
        if bits:
            assert stats.get("b12x") == bits
        use(ref)
        ref.cache = []
        with _Bits(bits):
            want, c0, want_state = _fake_request(ref, prompt, policy=policy, tokens=30)
        assert c0 == 0 and reply == want and state == want_state, turn


@needs_torch
def test_control_the_0240_lookup_never_resumes(monkeypatch):
    rng = np.random.default_rng(3610)
    g = _g(monkeypatch)
    prompt = [int(t) for t in rng.integers(0, 1000, size=700)]
    reply, _, _ = _turn(g, prompt, 4, lookup="0240")
    for _ in range(3):
        prompt = prompt + reply + [1, 2, 3]
        reply, stats, _ = _turn(g, prompt, 4, lookup="0240")
        assert stats["cached"] == 0                               # W3's "cached always 0"


@needs_torch
def test_snapshots_never_cross_bits(monkeypatch):
    rng = np.random.default_rng(3620)
    p1 = [int(t) for t in rng.integers(0, 1000, size=700)]
    for first, second in ((4, 0), (0, 4), (4, 7), (7, 4)):
        g = _g(monkeypatch)
        r1, _, _ = _turn(g, p1, first)
        _, stats, _ = _turn(g, p1 + r1 + [5, 6, 7], second)
        assert stats["cached"] == 0, (first, second)
    g = _g(monkeypatch)
    r1, _, _ = _turn(g, p1, 4)
    _, stats, _ = _turn(g, p1 + r1 + [5, 6, 7], 4)
    assert stats["cached"] > 0                                          # control: same bits resume


def test_disk_compat_hash_leaves_out_the_default_bits_only():
    from tensorfold.families.glm5_next.cuda import sessdisk

    env = {"GLM53_TF_B12X": "4", "GLM53_TF_B12X_KDA_PREC": "bf16", "GLM53_TF_B12X_BV": "32",
           "GLM53_TF_B12X_WARPS": "8", "GLM53_TF_KV_DTYPE": "fp8"}
    got = sessdisk.knobs(env)
    assert "GLM53_TF_B12X" not in got
    assert {"GLM53_TF_B12X_KDA_PREC", "GLM53_TF_B12X_BV", "GLM53_TF_B12X_WARPS", "GLM53_TF_KV_DTYPE"} <= set(got)
    assert sessdisk.knobs(dict(env, GLM53_TF_B12X="0")) == got


# -- GPU: the kernel on the real shapes ----------------------------------------------------------------------------
def _attn_inputs(seed: int, rows: int = 96, H: int = 32, L: int = 512, P: int = 6000, W: int = 2051):
    g = torch.Generator().manual_seed(seed)
    gain = torch.exp(torch.randn(L, generator=g) * 0.5)
    gain[torch.randperm(L, generator=g)[:4]] *= 8
    lat = (torch.randn(P, L, generator=g) * gain).bfloat16().cuda()
    qa = (torch.randn(rows, H, L, generator=g) * 0.3).bfloat16().cuda()
    tok = torch.full((rows, W), -1, dtype=torch.int32)
    cnt = torch.zeros(rows, dtype=torch.int32)
    for r in range(rows):
        n = [W, 0, 700, 33, W - 1, 1, 32, 65][r % 8]
        cnt[r] = n
        tok[r, :n] = torch.randperm(P, generator=g)[:n].sort().values.int()
    return qa, lat, tok.cuda(), cnt.cuda()


def _one(qa, lc, tok, cnt):
    from tensorfold.families.glm5_next.cuda import b12x_attn

    out = torch.full(qa.shape, 7.0, device="cuda")
    b12x_attn.sparse_latent_one(qa.contiguous(), lc, tok.contiguous(), cnt.contiguous(), out, 256 ** -0.5)
    return out


@gpu
@pytest.mark.parametrize("cache", ["bf16", "fp8"])
def test_one_pass_rows_independent(cache):
    from tensorfold.families.glm5_next.cuda import latent

    qa, lat, tok, cnt = _attn_inputs(3601)
    lc = lat if cache == "bf16" else latent.quantize_rows_reference(lat)
    full = _one(qa, lc, tok, cnt)
    m = cnt > 0
    assert bool((full[~m] == 7.0).all())
    assert torch.equal(_one(qa, lc, tok, cnt), full)                          # deterministic
    for r in range(0, 16):
        assert torch.equal(_one(qa[r:r + 1], lc, tok[r:r + 1], cnt[r:r + 1])[0], full[r]), r
    g = torch.Generator().manual_seed(7)
    for idx in (torch.randperm(qa.shape[0], generator=g), torch.tensor([5, 5, 5, 2]),
                torch.arange(qa.shape[0] - 1, -1, -1), torch.randperm(qa.shape[0], generator=g)[:37]):
        i = idx.cuda()
        assert torch.equal(_one(qa[i], lc, tok[i], cnt[i]), full[i])


@gpu
def test_fp8_equals_bf16_on_dequantized_rows():
    """The W3 failure (0240's FP8 one-pass kernel fed its dots in the 8-bit upcast layout)."""

    from tensorfold.families.glm5_next.cuda import b12x_attn, latent

    for seed in (3602, 3603, 3604):
        qa, lat, tok, cnt = _attn_inputs(seed)
        lc = latent.quantize_rows_reference(lat)
        a = _one(qa, lc, tok, cnt)
        b = _one(qa, latent.dequantize_rows(lc).contiguous(), tok, cnt)
        m = cnt > 0
        assert torch.equal(a[m], b[m]), (seed, (a[m] - b[m]).abs().max().item())
        ref = b12x_attn.reference(qa[:8], latent.dequantize_rows(lc), tok[:8], cnt[:8], 256 ** -0.5)
        mm = cnt[:8] > 0
        chunked = torch.full(qa[:8].shape, 7.0, device="cuda")
        latent.sparse_latent(qa[:8].contiguous(), lc, tok[:8].contiguous(), cnt[:8].contiguous(), chunked,
                             256 ** -0.5, bm=latent.FAST_BM)
        e1 = (a[:8][mm].double() - ref[mm]).abs().max().item()
        e0 = (chunked[mm].double() - ref[mm]).abs().max().item()
        print(f"\n[0360 fp8] one pass vs f64 {e1:.3e}, chunked vs f64 {e0:.3e}")
        assert e1 <= 1.5 * e0 + 1e-4


@gpu
def test_timing_print():
    """Not an assertion: one pass (bf16 / FP8) vs the chunked kernel at 1,024 rows (bench_b12x.py has the full one)."""

    from tensorfold.families.glm5_next.cuda import latent

    qa, lat, tok, cnt = _attn_inputs(3605, rows=1024, P=30000)
    cnt.fill_(2051)
    tok = torch.stack([torch.randperm(30000)[:2051].sort().values.int() for _ in range(1024)]).cuda()
    for cache in ("bf16", "fp8"):
        lc = lat if cache == "bf16" else latent.quantize_rows_reference(lat)
        out = torch.empty(qa.shape, device="cuda")
        fns = {"one": lambda: _one(qa, lc, tok, cnt),
               "chunked": lambda: latent.sparse_latent(qa, lc, tok, cnt, out, 256 ** -0.5, bm=latent.FAST_BM)}
        ms = {}
        for name, fn in fns.items():
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(10):
                fn()
            e.record()
            torch.cuda.synchronize()
            ms[name] = s.elapsed_time(e) / 10
        print(f"\n[0360 {cache}] chunked {ms['chunked']:.2f} ms -> one pass {ms['one']:.2f} ms "
              f"({ms['chunked'] / ms['one']:.2f}x)")


# -- GPU: the engine (bit 4 on the latent cache past the dense limit) -------------------------------------------------
@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")
    monkeypatch.setenv("GLM53_TF_NONEXPERT", "q4mse")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    if not CUDA:
        pytest.skip("CUDA only")
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_b12x360")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


@pytest.fixture(scope="module")
def el(ckpt):
    from test_cindep_patches import _engine

    return _engine(ckpt, lean_block=256, rows_max=8192, context=8192, latent_kv=True)


@gpu
def test_engine_bit4_state_independent_of_chunks(el):
    from test_cindep_patches import _Variant, _same, _state

    with _Bits(4):
        for n in (2100, 3000):
            prompt = list(np.random.default_rng(3630 + n).integers(0, 1000, size=n))
            with _Variant(el, 1024):
                a = _state(el, prompt)
            for C, ov in ((8192, "1"), (256, "0"), (4096, "1")):
                with _Variant(el, C, overlap=ov):
                    assert _same(a, _state(el, prompt)), (n, C, ov)


@gpu
@pytest.mark.parametrize("sampling", ["greedy", "sampled"])
@pytest.mark.parametrize("policy", [None, "2", "f3"])
def test_engine_bit4_drafted_serial_resumed_fresh(el, sampling, policy):
    """Through ``generate`` (the lookup 0360 fixed): past the dense limit, a follow-up resumes (cached > 0) and equals
    a fresh prefill with another chunk size; drafted == serial."""

    from test_cindep_patches import _cold, _gen, _sampling

    s = _sampling(sampling)
    rng = np.random.default_rng(3640 + [None, "2", "f3"].index(policy))
    p1 = [int(t) for t in rng.integers(0, 1000, size=2600)]
    kn = {"b12x": 4}
    el.cache = []
    r1, st1 = _gen(el, p1, s, policy=policy, knobs=dict(kn, prefill_rows=8192))
    assert st1["tf_knobs"]["b12x"] == 4 and st1.get("b12x") == 4
    assert r1 == _cold(el, p1, s, knobs=dict(kn, prefill_rows=1024))[:len(r1)]
    _gen(el, p1, s, policy=policy, knobs=dict(kn, prefill_rows=8192))
    p2 = p1 + r1 + [int(t) for t in rng.integers(0, 1000, size=90)]
    warm, st2 = _gen(el, p2, s, policy=policy, knobs=dict(kn, prefill_rows=256))
    assert st2["cached"] > 0 and st2["cached"] % 64 == 0
    assert warm == _cold(el, p2, s, knobs=dict(kn, prefill_rows=4096))


@gpu
def test_engine_bit4_snapshots_never_cross(el):
    from test_cindep_patches import _gen

    rng = np.random.default_rng(3650)
    p1 = [int(t) for t in rng.integers(0, 1000, size=2300)]
    for first, second, resumes in ((4, 0, False), (0, 4, False), (4, 4, True)):
        el.cache = []
        r1, _ = _gen(el, p1, None, knobs={"b12x": first, "prefill_rows": 1024})
        _, st = _gen(el, p1 + r1 + [5, 6, 7], None, knobs={"b12x": second, "prefill_rows": 1024})
        assert (st["cached"] > 0) == resumes, (first, second)
