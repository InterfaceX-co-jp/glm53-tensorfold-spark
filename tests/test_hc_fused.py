"""patches/0520 (GLM53_TF_HC_CUDA): the fused hyper-connection boundary kernel, CPU tests (no GPU, no nvcc).

- knobs: GLM53_TF_HC_CUDA (0 | 1), _ROWS (1 .. 64), _RG (1, 2, 4, 8); ``settings`` for the ranks' agreement.
- the CPU bit model: ``tests/hc_fused_emu.py``'s lane-level port of ``hc_fused.cu`` (its CTAs, thread mapping,
  shuffles, smem exchanges, tickets; step CTAs in random orders) equals the specification of glue._hc_post ->
  _hc_partial -> _hc_finish as Triton 3.7.1 + its ptxas execute them, bit for bit, on every output (x, the partials,
  normed, xs, post, comb), for 1-16 rows, every row-group size and seven input kinds (random, signed zeros, the
  unit sequences' scaling paths, subnormals, inf / NaN, real-scale, and an adversarial collapse). Negative controls:
  each of seven one-detail variants of the specification differs from the port on some kind (so the comparison
  would see such a mistake). The fp helpers are checked against exact rational arithmetic.
- dispatch: ``hc_cuda.fits`` rules; ``forward.layer_forward`` and ``batch.compute_multi`` with the knob off call
  exactly the Triton kernels they called before (same functions, same arguments, same order), with it on they call
  the fused kernel at both boundaries of a layer and skip the next layer's first hc_pre only when the previous
  boundary's kernel computed it; a stale or mismatched mark only costs a recomputation.

The PTX-level checks (the Triton kernels' and the CUDA kernel's own PTX in an interpreter, SASS counts, sm_121
compile) are in tests/test_hc_fused_compile.py.

    PYTHONPATH=<patched tree>/src pytest -q tests/test_hc_fused.py
"""

from __future__ import annotations

import fractions
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import hc_fused_emu as E  # noqa: E402


def _hc():
    try:
        from tensorfold.families.glm5_next.cuda import hc_cuda
    except ImportError:
        pytest.skip("needs the patched tree on PYTHONPATH")
    return hc_cuda


# -- knobs -----------------------------------------------------------------------------------------------------------
def test_knob_parse():
    hc = _hc()
    for v in (None, "", "0", "off", "false", "no", " 0 "):
        assert hc.parse(v) is False
    for v in ("1", "on", "true", "yes", "ON"):
        assert hc.parse(v) is True
    for v in ("2", "v2", "auto", "-1"):
        with pytest.raises(ValueError):
            hc.parse(v)
    assert hc.parse_rows(None) == 16 and hc.parse_rows("64") == 64 and hc.parse_rows("1") == 1
    for v in ("0", "65", "x"):
        with pytest.raises(ValueError):
            hc.parse_rows(v)
    assert hc.parse_rg("") == 4 and [hc.parse_rg(str(v)) for v in (1, 2, 4, 8)] == [1, 2, 4, 8]
    for v in ("3", "16", "0"):
        with pytest.raises(ValueError):
            hc.parse_rg(v)


def test_settings_and_default_off(monkeypatch):
    hc = _hc()
    if os.environ.get("GLM53_TF_HC_CUDA") is None:
        assert hc.REQUESTED is False
    assert hc.ON is False                       # never on at import: only after the engine's self-check
    monkeypatch.setattr(hc, "REQUESTED", True)
    assert hc.settings() == [1]
    monkeypatch.setattr(hc, "REQUESTED", False)
    assert hc.settings() == [0]
    saved = (hc.ON, hc.ROWS, hc.RG)
    try:
        hc.configure(True, rows=8, rg=2)
        assert (hc.ON, hc.ROWS, hc.RG) == (True, 8, 2)
        with pytest.raises(ValueError):
            hc.configure(True, rows=100)
    finally:
        hc.ON, hc.ROWS, hc.RG = saved


# -- fp helpers ------------------------------------------------------------------------------------------------------
def _frac(v) -> fractions.Fraction:
    return fractions.Fraction(float(v))


def _round32(q: fractions.Fraction) -> np.float32:
    """Exact rational -> nearest fp32, ties to even (normal range)."""

    f = np.float32(float(q))                       # a neighbour; fix up by comparing distances
    cands = [np.nextafter(f, np.float32(-np.inf)), f, np.nextafter(f, np.float32(np.inf))]
    best = min(cands, key=lambda c: (abs(_frac(c) - q), int(np.float32(c).view(np.uint32)) & 1))
    return np.float32(best)


def test_fma32_exact():
    rng = np.random.default_rng(0)
    a = (rng.standard_normal(3000) * np.exp2(rng.integers(-20, 20, 3000))).astype(np.float32)
    b = (rng.standard_normal(3000) * np.exp2(rng.integers(-20, 20, 3000))).astype(np.float32)
    c = (rng.standard_normal(3000) * np.exp2(rng.integers(-40, 40, 3000))).astype(np.float32)
    got = E.fma32(a, b, c)
    for i in range(0, 3000, 7):
        assert got[i] == _round32(_frac(a[i]) * _frac(b[i]) + _frac(c[i])), i


def test_bf16_rounding():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(5000).astype(np.float32)
    u = E.f32_to_bf16(x)
    back = E.bf16_to_f32(u)
    lo = E.bf16_to_f32(u - 1)
    hi = E.bf16_to_f32(u + 1)
    assert np.all(np.abs(back - x) <= np.minimum(np.abs(lo - x), np.abs(hi - x)))
    a16 = E.f32_to_bf16(rng.standard_normal(2000))
    b16 = E.f32_to_bf16(rng.standard_normal(2000))
    prod = E.bf16_mul_bits(a16, b16)
    ref = E.f32_to_bf16(E.bf16_to_f32(a16) * E.bf16_to_f32(b16))        # exact in fp32, then one rounding
    assert np.array_equal(prod, ref)


def test_unit_sequences():
    # div.full's scaling paths and ex2's halving path give finite, sensible values, and the stand-ins are not exact
    b = np.array([2.0 ** 127, 3.0e38, 1e-39, 2.0 ** -127, 3.0, 1.0], np.float32)
    q = E.div_full(np.float32(1.0), b)
    assert q[5] == 1.0 and q[3] == 2.0 ** 127 and np.isinf(q[2])      # 1 / 1e-39 overflows fp32, 2^-127 does not
    assert np.all(np.abs(q[:2].astype(np.float64) * b[:2] - 1) < 1e-5) and q[0] > 0     # not flushed to zero
    assert E.ex2_approx(np.float32(-130.0)) > 0                       # subnormal result through the halving
    x = np.linspace(0.1, 10, 4001).astype(np.float32)
    exact = (1.0 / x.astype(np.float64)).astype(np.float32)
    assert 0 < np.count_nonzero(E.mufu_rcp(x) != exact) < len(x)      # deliberately approximate


# -- the CPU bit model: port == specification ------------------------------------------------------------------------
CASES = [(1, 4), (2, 4), (3, 2), (5, 4), (6, 1), (9, 8), (16, 4)]


@pytest.mark.parametrize("R,rg", CASES)
@pytest.mark.parametrize("kind", E.KINDS)
def test_port_equals_spec(R, rg, kind):
    inp = E.make_inputs(R, seed=R * 7 + rg, kind=kind)
    rng = np.random.default_rng(R + 100 * rg)
    groups = (R + min(R, rg) - 1) // min(R, rg)
    got = E.port_of(inp, rg=rg, step_order=rng.permutation(32 * groups))
    ref = E.ref_of(inp)
    assert E.diff(got, ref) == []


def test_rows_independent_of_window():
    """A row's outputs do not depend on the other rows of its window or on the grouping (placement only)."""

    inp = E.make_inputs(7, seed=3, kind="real")
    whole = E.port_of(inp, rg=4)
    for r in (0, 3, 6):
        one = {k: (v[r:r + 1] if k not in ("fn", "base", "scale", "nw", "eps", "hc_eps", "iters") else v)
               for k, v in inp.items()}
        one["g"] = inp["g"][:, r:r + 1].copy()
        alone = E.port_of(one, rg=1)
        for k in E.OUTPUTS:
            a = alone[k]
            b = whole[k][r:r + 1]
            if k == "part":
                a, b = a[:, :, :25], b[:, :, :25]
            assert E.same(a, b), (r, k)


def test_negative_controls():
    """Every one-detail variant of the specification differs from the port on some input kind."""

    ports = {kind: (E.make_inputs(4, 11, kind), None) for kind in ("random", "real", "collapse")}
    ports = {k: (inp, E.port_of(inp, rg=2)) for k, (inp, _) in ports.items()}
    for kind, (inp, got) in ports.items():
        assert E.diff(got, E.ref_of(inp)) == [], kind
    for v in E.VARIANTS:
        caught = [kind for kind, (inp, got) in ports.items() if E.diff(got, E.ref_of(inp, variant=v))]
        assert caught, f"variant {v} not seen"


# -- dispatch ---------------------------------------------------------------------------------------------------------
torch = None


def _torch():
    global torch
    torch = pytest.importorskip("torch")
    return torch


def _tensors(R=3, world=2):
    t = _torch()
    bf, f32 = t.bfloat16, t.float32
    x = t.zeros((R, 16384), dtype=bf)
    g = t.zeros((world, R, 4096), dtype=f32)
    post = t.zeros((R, 4), dtype=f32)
    comb = t.zeros((R, 16), dtype=f32)
    fn = t.zeros((24, 16384), dtype=bf)
    base = t.zeros((24,), dtype=f32)
    scale = t.zeros((3,), dtype=f32)
    nw = t.zeros((4096,), dtype=bf)
    out = t.zeros((R, 4096), dtype=bf)
    xs = t.zeros((R, 64), dtype=f32)
    part = t.zeros((R, 16, 32), dtype=f32)
    return [x, g, post, comb, fn, base, scale, nw, out, xs, part]


def test_fits_rules(monkeypatch):
    hc = _hc()
    from tensorfold.families.glm5_next.cuda import glue

    args = _tensors()
    monkeypatch.setattr(hc, "ON", False)
    assert hc.fits(*args, cuda=False) == "off"
    monkeypatch.setattr(hc, "ON", True)
    monkeypatch.setattr(hc, "ROWS", 16)
    assert hc.fits(*args, cuda=False) is None
    assert hc.fits(*args) == "not CUDA"
    monkeypatch.setattr(glue, "FAST_HC", lambda *a: None)
    assert "fast-prefill" in hc.fits(*args, cuda=False)
    monkeypatch.setattr(glue, "FAST_HC", None)
    monkeypatch.setattr(glue, "MHC", True)
    assert "fast-prefill" in hc.fits(*args, cuda=False)
    monkeypatch.setattr(glue, "MHC", False)
    big = _tensors(R=17)
    assert hc.fits(*big, cuda=False) == "17 rows"
    monkeypatch.setattr(hc, "ROWS", 64)
    assert hc.fits(*big, cuda=False) is None
    assert hc.fits(*_tensors(R=65), cuda=False) == "65 rows"
    bad = list(args)
    bad[1] = args[1].to(torch.bfloat16)                  # 0084's bf16 partials: not decode
    assert hc.fits(*bad, cuda=False) == "layout"
    bad = list(args)
    bad[0] = args[0][:, :8192]                           # not 4 streams of 4096
    assert hc.fits(*bad, cuda=False) is not None
    bad = list(args)
    bad[4] = args[4].float()
    assert hc.fits(*bad, cuda=False) == "weights layout"
    bad = list(args)
    bad[10] = torch.zeros((3, 16, 25))
    assert hc.fits(*bad, cuda=False) == "output layout"
    gw = torch.zeros((2, 5, 4096))[:, :3]                # a row slice of a larger gathered block: rows contiguous
    assert hc.fits(args[0], gw, *args[2:], cuda=False) is None
    monkeypatch.setattr(hc, "BROKEN", "earlier failure")
    assert hc.fits(*args, cuda=False) == "earlier failure"


def test_post_pre_declines_without_cuda(monkeypatch):
    hc = _hc()
    monkeypatch.setattr(hc, "ON", True)
    x, g, post, comb, fn, base, scale, nw, out, xs, part = _tensors()
    h = SimpleNamespace(fn=fn, base=base, scale=scale)
    launched = []
    monkeypatch.setattr(hc, "launch", lambda *a, **k: launched.append(1))
    assert hc.post_pre(x, g, post, comb, h, nw, out, xs, part, 1e-5, 1e-6, 20) is False
    assert hc.post_pre(x, g, post, comb, None, nw, out, xs, part, 1e-5, 1e-6, 20) is False
    monkeypatch.setattr(hc, "ON", False)
    assert hc.post_pre(x, g, post, comb, h, nw, out, xs, part, 1e-5, 1e-6, 20) is False
    assert not launched


class _Rec:
    """Records the hc calls of a forward: ("pre", site), ("post", g), ("fused", g, site)."""

    def __init__(self, names):
        self.calls = []
        self.names = names                                # id(fn tensor) -> "attn0" / "ffn0" / ..

    def pre(self, x, fn, base, scale, nw, out, xs, post, comb, part, eps, hc_eps, iters):
        self.calls.append(("pre", self.names[id(fn)], x.shape[0], out.shape[0], part.shape[0]))

    def post(self, x, xout, g, post, comb):
        assert x is xout or x.data_ptr() == xout.data_ptr()
        self.calls.append(("post", g, x.shape[0], post.shape[0]))


def _model(n_layers=3, R=3):
    t = _torch()
    bf = t.bfloat16
    names = {}
    layers = []
    for i in range(n_layers):
        hs = []
        for site in ("attn", "ffn"):
            h = SimpleNamespace(fn=t.zeros((24, 16384), dtype=bf), base=t.zeros(24), scale=t.zeros(3))
            names[id(h.fn)] = f"{site}{i}"
            hs.append(h)
        layers.append(SimpleNamespace(index=i, kind="kda", attn_hc=hs[0], ffn_hc=hs[1],
                                      in_norm=t.zeros(4096, dtype=bf), post_norm=t.zeros(4096, dtype=bf),
                                      mlp=object(), moe=None))
    w = SimpleNamespace(cfg=SimpleNamespace(eps=1e-5, hc_eps=1e-6, hc_iters=20, hidden=4096, streams=4),
                        layers=layers, meta={}, embed=None)
    rows = 8
    b = SimpleNamespace(x=t.zeros((rows, 16384), dtype=bf), post=t.zeros((rows, 4)), comb=t.zeros((rows, 16)),
                        normed=t.zeros((rows, 4096), dtype=bf), xs=t.zeros((rows, 64)), hcpart=t.zeros((rows, 16, 32)),
                        ids=t.zeros(rows, dtype=t.int32), hidden=t.zeros((rows, 4096), dtype=bf), tap_at={}, taps={},
                        site=None)
    return w, b, names, layers


def _patch_forward(monkeypatch, fwd, rec, fused_ok, fused_calls):
    hc = fwd.hc_cuda
    monkeypatch.setattr(fwd.glue, "hc_pre", rec.pre)
    monkeypatch.setattr(fwd.glue, "hc_post", rec.post)
    monkeypatch.setattr(fwd, "kda_block", lambda layer, w, st, b, R: f"g_attn{layer.index}")
    monkeypatch.setattr(fwd, "mlp_block", lambda layer, w, b, R: f"g_ffn{layer.index}")

    real = hc.post_pre

    def post_pre(x, g, post, comb, h, norm_w, out, xs, part, eps, hc_eps, iters):
        site = rec.names[id(h.fn)]
        fused_calls.append((g, site))
        if not hc.ON:                                     # the real entry point (it declines when off)
            return real(x, g, post, comb, h, norm_w, out, xs, part, eps, hc_eps, iters)
        assert x.shape[0] == post.shape[0] == comb.shape[0] == out.shape[0] == xs.shape[0] == part.shape[0]
        if fused_ok:
            rec.calls.append(("fused", g, site))
        return fused_ok

    monkeypatch.setattr(hc, "post_pre", post_pre)


OFF_SEQ = [("pre", "attn0"), ("post", "g_attn0"), ("pre", "ffn0"), ("post", "g_ffn0"),
           ("pre", "attn1"), ("post", "g_attn1"), ("pre", "ffn1"), ("post", "g_ffn1"),
           ("pre", "attn2"), ("post", "g_attn2"), ("pre", "ffn2"), ("post", "g_ffn2")]
ON_SEQ = [("pre", "attn0"), ("fused", "g_attn0", "ffn0"), ("fused", "g_ffn0", "attn1"),
          ("fused", "g_attn1", "ffn1"), ("fused", "g_ffn1", "attn2"),
          ("fused", "g_attn2", "ffn2"), ("post", "g_ffn2")]


def _short(calls):
    return [c[:3] if c[0] == "fused" else c[:2] for c in calls]


def test_layer_forward_off_is_untouched(monkeypatch):
    _hc()
    from tensorfold.families.glm5_next.cuda import forward as fwd

    w, b, names, layers = _model()
    rec = _Rec(names)
    fused = []
    _patch_forward(monkeypatch, fwd, rec, True, fused)
    monkeypatch.setattr(fwd.hc_cuda, "ON", False)
    for layer in layers:
        fwd.layer_forward(layer, w, SimpleNamespace(), b, 3)
    assert _short(rec.calls) == OFF_SEQ
    assert all(c[2] == 3 for c in rec.calls)             # every call on the window's rows ([:R] views)
    assert fused == []                                   # off: the fused entry point is never even called
    assert getattr(b, "hc_cuda_ready", None) is None


def test_layer_forward_on(monkeypatch):
    _hc()
    from tensorfold.families.glm5_next.cuda import forward as fwd

    w, b, names, layers = _model()
    rec = _Rec(names)
    fused = []
    _patch_forward(monkeypatch, fwd, rec, True, fused)
    monkeypatch.setattr(fwd.hc_cuda, "ON", True)
    for layer in layers:
        fwd.layer_forward(layer, w, SimpleNamespace(), b, 3)
    assert _short(rec.calls) == ON_SEQ
    assert b.hc_cuda_ready is None                       # the last mark was consumed
    # a second forward on the same buffers starts clean
    rec.calls.clear()
    for layer in layers:
        fwd.layer_forward(layer, w, SimpleNamespace(), b, 3)
    assert _short(rec.calls) == ON_SEQ


def test_layer_forward_on_but_declined(monkeypatch):
    _hc()
    from tensorfold.families.glm5_next.cuda import forward as fwd

    w, b, names, layers = _model()
    rec = _Rec(names)
    fused = []
    _patch_forward(monkeypatch, fwd, rec, False, fused)
    monkeypatch.setattr(fwd.hc_cuda, "ON", True)
    for layer in layers:
        fwd.layer_forward(layer, w, SimpleNamespace(), b, 3)
    assert _short(rec.calls) == OFF_SEQ                  # the Triton kernels, exactly as before
    assert [f[1] for f in fused] == ["ffn0", "attn1", "ffn1", "attn2", "ffn2"]


def test_stale_marks_cost_a_recomputation_only(monkeypatch):
    _hc()
    from tensorfold.families.glm5_next.cuda import forward as fwd

    w, b, names, layers = _model()
    rec = _Rec(names)
    _patch_forward(monkeypatch, fwd, rec, True, [])
    monkeypatch.setattr(fwd.hc_cuda, "ON", True)
    # an interrupted forward left a mark for layer 2: a new forward's layer 0 recomputes and clears it
    fwd.hc_cuda.mark_ready(b, layers[2], 3)
    fwd.layer_forward(layers[0], w, SimpleNamespace(), b, 3)
    assert _short(rec.calls)[0] == ("pre", "attn0")
    # a mark for other rows is not used
    rec.calls.clear()
    fwd.hc_cuda.mark_ready(b, layers[1], 2)
    fwd.layer_forward(layers[1], w, SimpleNamespace(), b, 3)
    assert _short(rec.calls)[0] == ("pre", "attn1")
    assert b.hc_cuda_ready is None or b.hc_cuda_ready == (id(layers[2]), 3)


def test_next_layer(monkeypatch):
    hc = _hc()
    w, b, names, layers = _model(4)
    assert hc.next_layer(w, layers[0]) is layers[1]
    assert hc.next_layer(w, layers[3]) is None
    layers[2].attn_hc = None                             # a layer without hyper-connections: no fusion into it
    assert hc.next_layer(w, layers[1]) is None
    w.layers = layers[:2]                                # a new layer list: the map follows it
    assert hc.next_layer(w, layers[1]) is None


def _patch_batch(monkeypatch, bmod, rec, fused_ok):
    from tensorfold.families.glm5_next.cuda import hc_cuda as hc

    monkeypatch.setattr(bmod.glue, "hc_pre", rec.pre)
    monkeypatch.setattr(bmod.glue, "hc_post", rec.post)
    monkeypatch.setattr(bmod.glue, "embed", lambda *a, **k: None)
    monkeypatch.setattr(bmod.glue, "stream_mean", lambda x, out: rec.calls.append(("mean", out.shape[0])))
    monkeypatch.setattr(bmod, "_kda", lambda layer, w, sts, b, Rs, offs, T: f"g_attn{layer.index}")
    monkeypatch.setattr(bmod, "mlp_block", lambda layer, w, b, T: f"g_ffn{layer.index}")

    real = hc.post_pre

    def post_pre(x, g, post, comb, h, norm_w, out, xs, part, eps, hc_eps, iters):
        if not hc.ON:
            return real(x, g, post, comb, h, norm_w, out, xs, part, eps, hc_eps, iters)
        if fused_ok:
            rec.calls.append(("fused", g, rec.names[id(h.fn)]))
        return fused_ok

    monkeypatch.setattr(hc, "post_pre", post_pre)
    return hc


@pytest.mark.parametrize("on,ok", [(False, True), (True, True), (True, False)])
def test_compute_multi_dispatch(monkeypatch, on, ok):
    _hc()
    from tensorfold.families.glm5_next.cuda import batch as bmod

    w, b, names, layers = _model()
    rec = _Rec(names)
    hc = _patch_batch(monkeypatch, bmod, rec, ok)
    monkeypatch.setattr(hc, "ON", on)
    b.tap_at = {1: (0,)}
    b.taps = {0: b.hidden}
    assert bmod.compute_multi(w, [None, None], b, [2, 3], logits=False) is None
    seq = [c for c in _short(rec.calls)]
    means = [c for c in seq if c[0] == "mean"]
    seq = [c for c in seq if c[0] != "mean"]
    assert seq == (ON_SEQ if on and ok else OFF_SEQ)
    assert len(means) == 2                               # the tap after layer 1 (reads x: the kernel wrote it) + final
    assert rec.calls.index(("mean", 5)) > [i for i, c in enumerate(rec.calls) if c[1] == "g_ffn1"][0]


def test_bench_gate():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cuda"))
    import bench_hc_fused as bh

    rows = {1: {"triton_us": 16.0, "fused_us": 7.0, "same_bits": True},
            4: {"triton_us": 17.0, "fused_us": 8.0, "same_bits": True},
            32: {"triton_us": 30.0, "fused_us": 29.0, "same_bits": True}}
    g = bh.gate(rows)
    assert g["pass"] and g["worst_rows"] == 4 and abs(g["worst"] - 8 / 17) < 1e-12
    rows[4]["fused_us"] = 9.0
    assert not bh.gate(rows)["pass"]
    rows[4]["fused_us"] = 8.0
    rows[1]["same_bits"] = False
    assert not bh.gate(rows)["pass"]
    assert not bh.gate({32: rows[32]})["pass"]


SYNTH = """
.version 9.1
.target sm_121a
.address_size 64
.visible .entry k(
    .param .u64 .ptr .global .align 1 k_param_0
)
{
    .reg .b32 %r<24>;
    .reg .b64 %rd<3>;
    ld.param.b64 %rd1, [k_param_0];
    ld.global.b32 %r1, [%rd1];
    ld.global.b32 %r2, [%rd1+4];
    ld.global.b32 %r3, [%rd1+8];
    mul.f32 %r4, %r1, %r2;
    add.f32 %r5, %r4, %r3;
    mul.f32 %r6, %r1, %r3;
    sub.f32 %r7, %r6, %r2;
    mul.f32 %r8, %r2, %r3;
    sub.f32 %r9, %r1, %r8;
    mul.f32 %r10, %r1, %r1;
    add.f32 %r11, %r10, %r10;
    mul.rn.f32 %r12, %r2, %r2;
    add.f32 %r13, %r12, %r1;
    mul.f32 %r14, %r3, %r3;
    add.f32 %r15, %r14, %r1;
    add.f32 %r16, %r14, %r2;
    mov.b32 %r17, 0f46800000;
    div.full.f32 %r18, %r1, %r17;
    add.f32 %r19, %r18, %r3;
    st.global.b32 [%rd1+12], %r5;
    st.global.b32 [%rd1+16], %r7;
    st.global.b32 [%rd1+20], %r9;
    st.global.b32 [%rd1+24], %r11;
    st.global.b32 [%rd1+28], %r13;
    st.global.b32 [%rd1+32], %r15;
    st.global.b32 [%rd1+36], %r16;
    st.global.b32 [%rd1+40], %r19;
    ret;
}
"""


def test_ptxas_view_rules():
    """What the view fuses: a single-use non-rounding mul into its non-rounding add / sub (either side), and a
    div.full by a constant 2^k (an exact multiply) into the next add; what it leaves alone: mul.rn, a mul with two
    uses, x + x of one mul."""

    k = E.parse(SYNTH)
    view, stats = E.ptxas_view(k)
    assert stats["contracted"] == 4 and stats["div_pow2"] == 1
    rng = np.random.default_rng(3)
    for _ in range(200):
        a, b, c = (np.float32(v) for v in rng.standard_normal(3) * 10 ** rng.uniform(-3, 3, 3))
        mem = E.Memory()
        buf = mem.alloc(np.array([a, b, c] + [0] * 8, dtype=np.float32))
        E.launch(view, mem, [buf], (1,), 32)
        got = mem.get(buf, np.float32, (11,))[3:]
        want = [E.fma32(a, b, c), E.fma32(a, c, -b), E.fma32(-b, c, a), E.add32(E.mul32(a, a), E.mul32(a, a)),
                E.add32(E.mul32(b, b), a), E.add32(E.mul32(c, c), a), E.add32(E.mul32(c, c), b),
                E.fma32(a, np.float32(2.0 ** -14), c)]
        assert [float(v) for v in got] == [float(v) for v in want]
