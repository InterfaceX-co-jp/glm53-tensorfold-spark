"""patches/0400 (``kda_v2``, GLM53_TF_KDA_V2) in Triton's CPU interpreter: bit-identical to ``fast_kda``, no GPU.

Every comparison is BITWISE (``torch.equal``): the rows' outputs and the state after the last row, for
``kda_v2.kda_prefill_v2`` in mode 1 (split: ``fast_kda``'s prep + ``_state2``, value blocks of 16 / 32 / 64 / 128,
with and without the K split) and mode 2 (fused persistent kernel: 1 / 2 / 4 value blocks a head, lag / ring
settings, 1 or more programs) against ``fast_kda.kda_prefill_chunked`` (today's kernels), at 1 / 63 / 64 / 65 / 200 /
2,048 / 8,192 rows, aligned and unaligned starts, with an initial state; a prompt cut at multiples of 64 into calls of
any sizes (the engine's 64-row grid: resume == fresh prefill, and every snapshot state between calls) equals one call;
``state_out`` aliasing ``state_in``; the fused kernel's ticket order (every wait points to an earlier ticket).

The interpreter is made to compute like the GPU where it matters for bits:
- fp32 -> bf16 casts round to nearest even (Triton 3.8's interpreter truncates);
- ``tl.dot`` is an mma-style chain: the accumulator starts at the given value and takes one rounded fp32 step per
  8 of K (the in-step sum exact: tf32 operands for "tf32", one fma per element for "ieee"), so a dot's result does not
  depend on the tile's M / N and a K split whose second dot starts from the first one's result is the same chain;
- Triton's combine rewrite ``addf(x, dot(a, b, 0)) -> dot(a, b, x)`` (and the mirrored one), which the compiled
  ``fast_kda._kda_state`` has (``tests/kda_v2_compile.py`` checks its TTGIR), is applied to additions of a fresh dot.
The emulated chain is not the tensor cores' exact internal rounding; the GPU tests decide that. What these tests pin
down is the data flow: which values meet in which order, every rounding, every chunk and call boundary.

Run: TRITON_INTERPRET=1 PYTHONPATH=<patched TensorFold>/src pytest -q tests/test_kda_v2_interpreter.py (~10 minutes;
-m "not slow" skips the 8,192-row cases).
"""

from __future__ import annotations

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

kda_v2 = pytest.importorskip("tensorfold.families.glm5_next.cuda.kda_v2")
from tensorfold.families.glm5_next.cuda import fast_kda  # noqa: E402

if os.environ.get("TRITON_INTERPRET") != "1" or type(kda_v2._fused).__name__ != "InterpretedFunction":
    pytest.skip("needs Triton's interpreter (TRITON_INTERPRET=1 before triton is imported)", allow_module_level=True)
if torch.cuda.is_available():
    pytest.skip("CPU-interpreter checks: run where no GPU is visible", allow_module_level=True)

H = 2
C = 3 * H * 128


def _bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def _f32_to_bf16(f32):
    t = torch.from_numpy(np.ascontiguousarray(f32, dtype=np.float32)).to(torch.bfloat16)
    return t.view(torch.int16).numpy().view(np.uint16)


def _rnd_tf32(x):
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0xFFF + ((u >> 13) & 1)) & 0xFFFFE000
    return u.astype(np.uint32).view(np.float32)


def _chain(a, b, acc, prec):
    """The mma-style chain: acc, then one fp32 rounding per k-step of 8 (tf32) or per k (ieee: fma)."""

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    acc = np.array(acc, dtype=np.float32, copy=True)
    if "TF32" in str(prec).upper() and "X3" not in str(prec).upper():
        a, b, step = _rnd_tf32(a), _rnd_tf32(b), 8
    else:
        step = 1
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    for k0 in range(0, a.shape[-1], step):
        part = a64[..., :, k0:k0 + step] @ b64[..., k0:k0 + step, :]
        acc = (acc.astype(np.float64) + part).astype(np.float32)
    return acc


_PLAIN_FADD: dict = {}


@pytest.fixture(autouse=True)
def _gpu_like_interpreter(monkeypatch):
    from triton.runtime import interpreter as interp
    import triton.language as tl

    cast = interp.InterpreterBuilder.cast_impl

    def cast_impl(self, src, dst_type):
        s, d = src.dtype.scalar, dst_type.scalar
        if s == tl.float32 and d == tl.bfloat16:
            return interp.TensorHandle(_f32_to_bf16(src.data), d)
        if s == tl.bfloat16 and d == tl.float32:
            return interp.TensorHandle(_bf16_to_f32(src.data), d)
        return cast(self, src, dst_type)

    def create_dot(self, a, b, d, input_precision, max_num_imprecise_acc):
        ad = _bf16_to_f32(a.data) if a.dtype.scalar == tl.bfloat16 else a.data
        bd = _bf16_to_f32(b.data) if b.dtype.scalar == tl.bfloat16 else b.data
        out = interp.TensorHandle(_chain(ad, bd, d.data, input_precision), tl.float32)
        if not np.any(d.data):
            out.attr["dot0"] = (ad, bd, input_precision)          # a fresh dot: Triton's combine may absorb an add
        return out

    fadd = _PLAIN_FADD.setdefault("fadd", interp.InterpreterBuilder.create_fadd)

    def create_fadd(self, lhs, rhs):
        for dot, other in ((rhs, lhs), (lhs, rhs)):             # addf(x, dot(a, b, 0)) first, as the TTGIR shows
            spec = getattr(dot, "attr", {}).get("dot0")
            if spec is not None and np.shape(other.data) == np.shape(dot.data):
                return interp.TensorHandle(_chain(spec[0], spec[1], other.data, spec[2]), tl.float32)
        return fadd(self, lhs, rhs)

    monkeypatch.setattr(interp.InterpreterBuilder, "cast_impl", cast_impl)
    monkeypatch.setattr(interp.InterpreterBuilder, "create_fp_trunc", lambda self, s, t: cast_impl(self, s, t))
    monkeypatch.setattr(interp.InterpreterBuilder, "create_fp_ext", lambda self, s, t: cast_impl(self, s, t))
    monkeypatch.setattr(interp.InterpreterBuilder, "create_dot", create_dot)
    monkeypatch.setattr(interp.InterpreterBuilder, "create_fadd", create_fadd)


def _inputs(rows: int, seed: int = 0, state_scale: float = 3.0, v_scale: float = 0.05):
    """Random KDA inputs. The defaults make the state's products as large as the values they are subtracted from
    (a large state kept by slow channels, small v), so a last-bit change anywhere in the scan survives to the outputs
    and the state (with randn * 0.1 states, O(1) values and each row's decay drawn per element, the state was gone
    after a few rows and a swapped dot order was absorbed: checked)."""

    g = torch.Generator().manual_seed(seed)
    b_off = C + 256
    p = torch.randn(rows, b_off + H + 3, generator=g)
    p[:, 2 * H * 128:3 * H * 128] *= v_scale
    a = torch.randn(rows, H * 128, generator=g)
    slow = torch.rand(1, H * 128, generator=g) < 0.75         # per channel: 3/4 keep their state (~e^-6e-4 a row)
    a = torch.where(slow, a * 0.5 - 9.0, a * 0.5 + 12.0)
    return dict(p=p.bfloat16(), b_off=b_off, a=a.bfloat16(), g=torch.randn(rows, H * 128, generator=g).bfloat16(),
                conv_state=torch.randn(3, C, generator=g).bfloat16(),
                conv_w=(torch.randn(C, 4, generator=g) * 0.5).bfloat16(),
                state_in=torch.randn(H, 128, 128, generator=g) * state_scale,
                a_log=torch.randn(H, generator=g) * 0.2, dt_bias=torch.randn(H * 128, generator=g) * 0.2,
                norm_w=(1 + 0.1 * torch.randn(128, generator=g)).bfloat16(), eps=1e-6, lower=-5.0)


def _args(d, rows):
    return (d["p"], d["b_off"], d["a"], d["g"], d["conv_state"], d["conv_w"], d["state_in"], d["a_log"],
            d["dt_bias"], d["norm_w"], d["eps"], d["lower"], rows)


def _old(d, rows, pos):
    so = torch.empty_like(d["state_in"])
    return fast_kda.kda_prefill_chunked(*_args(d, rows), None, so, pos=pos), so


def _cfg(**kw):
    cfg = dict(bv=32, warps=4, maxnreg=168, ksplit=1, ctas=0, lag=1, ring=3, fbv=64, fmaxnreg=0)
    cfg.update(kw)
    return cfg


def _new(d, rows, pos, mode, **kw):
    so = torch.empty_like(d["state_in"])
    return kda_v2.kda_prefill_v2(*_args(d, rows), None, so, pos=pos, v2=mode, cfg=_cfg(**kw)), so


def _same(x, y):
    (o1, s1), (o2, s2) = x, y
    assert torch.equal(o1.view(torch.int16), o2.view(torch.int16)), "outputs differ"
    assert torch.equal(s1.view(torch.int32), s2.view(torch.int32)), "state differs"


VARIANTS = [
    (1, dict(bv=32, ksplit=1)), (1, dict(bv=32, ksplit=0)), (1, dict(bv=16, ksplit=1)), (1, dict(bv=64, ksplit=1)),
    (1, dict(bv=64, ksplit=0)),
    (2, dict(fbv=64, ksplit=0)), (2, dict(fbv=64, ksplit=1)), (2, dict(fbv=32, ksplit=1)),
    (2, dict(fbv=64, ksplit=1, lag=2, ring=3)), (2, dict(fbv=64, ksplit=0, lag=1, ring=2, ctas=1)),
    (2, dict(fbv=32, ksplit=0, lag=3, ring=4, ctas=3)),
]


@pytest.mark.parametrize("rows,pos", [(1, 0), (63, 0), (64, 0), (65, 128), (200, 64), (200, 10), (130, 7)])
@pytest.mark.parametrize("mode,kw", VARIANTS, ids=[f"m{m}-" + "-".join(f"{k}{v}" for k, v in kw.items())
                                                   for m, kw in VARIANTS])
def test_v2_equals_fast_kda(rows, pos, mode, kw):
    d = _inputs(rows, seed=rows + pos)
    _same(_new(d, rows, pos, mode, **kw), _old(d, rows, pos))


@pytest.mark.parametrize("mode,kw", [(1, dict(bv=32, ksplit=1)), (2, dict(fbv=64, ksplit=0)),
                                     (2, dict(fbv=32, ksplit=1, ring=2))])
def test_v2_equals_fast_kda_2048(mode, kw):
    rows = 2048
    d = _inputs(rows, seed=11)
    _same(_new(d, rows, 0, mode, **kw), _old(d, rows, 0))


@pytest.mark.slow
@pytest.mark.parametrize("mode,kw", [(1, dict(bv=32, ksplit=1)), (2, dict(fbv=64, ksplit=0, ring=3))])
def test_v2_equals_fast_kda_8192(mode, kw):
    rows = 8192
    d = _inputs(rows, seed=12)
    _same(_new(d, rows, 0, mode, **kw), _old(d, rows, 0))


def _second(d, cut, state):
    e = dict(d)
    e.update(p=d["p"][cut:], a=d["a"][cut:], g=d["g"][cut:], conv_state=d["p"][cut - 3:cut, :C].contiguous(),
             state_in=state)
    return e


@pytest.mark.parametrize("mode,kw", [(1, dict(bv=32, ksplit=1)), (2, dict(fbv=64, ksplit=0)),
                                     (2, dict(fbv=32, ksplit=1, ring=2, ctas=2))])
def test_resume_equals_fresh(mode, kw):
    """A prompt cut on the 64-row grid into calls of any sizes == one call (outputs and final state), and the state
    at each cut (a snapshot) == the old kernels' state at that cut."""

    rows, pos0 = 448, 128
    d = _inputs(rows, seed=21)
    whole = _new(d, rows, pos0, mode, **kw)
    _same(whole, _old(d, rows, pos0))
    for cuts in ([64], [128, 192], [64, 128, 320], [384]):
        outs, st, prev = [], d["state_in"], 0
        for cut in cuts + [rows]:
            e = _second(d, prev, st) if prev else d
            o, st = _new(e, cut - prev, pos0 + prev, mode, **kw)
            eo, est = _old(e, cut - prev, pos0 + prev)
            _same((o, st), (eo, est))                           # snapshot at the cut == today's
            outs.append(o)
            prev = cut
        assert torch.equal(torch.cat(outs).view(torch.int16), whole[0].view(torch.int16))
        assert torch.equal(st.view(torch.int32), whole[1].view(torch.int32))


@pytest.mark.parametrize("mode,kw", [(1, dict(bv=32, ksplit=1)), (2, dict(fbv=64, ksplit=1))])
def test_state_out_may_alias_state_in(mode, kw):
    rows = 200
    d = _inputs(rows, seed=31)
    ref = _old(d, rows, 0)
    s = d["state_in"].clone()
    o = kda_v2.kda_prefill_v2(*_args(dict(d, state_in=s), rows), None, s, pos=0, v2=mode, cfg=_cfg(**kw))
    _same((o, s), ref)


def test_deterministic_and_off_is_fast_kda():
    rows = 130
    d = _inputs(rows, seed=41)
    a = _new(d, rows, 0, 2, fbv=32, ksplit=1)
    b = _new(d, rows, 0, 2, fbv=32, ksplit=1)
    _same(a, b)
    _same(_new(d, rows, 0, 0), _old(d, rows, 0))


@pytest.mark.parametrize("nc,nvb,lag,ring", [(1, 2, 1, 2), (2, 2, 1, 2), (8, 2, 1, 2), (8, 4, 2, 3), (5, 1, 3, 4),
                                             (128, 2, 1, 3)])
def test_fused_schedule_waits_point_backwards(nc, nvb, lag, ring):
    """Every item appears once; a state step's waits (its chunk's prep, its block's previous step) and a prep's wait
    (the state steps of the chunk that last used its ring slot) are all on EARLIER tickets: whichever programs are
    resident, the one a program waits for already holds its ticket, so the kernel cannot deadlock."""

    heads = 3
    order = kda_v2.schedule(nc, heads, nvb, lag)
    where = {it: n for n, it in enumerate(order)}
    assert len(where) == len(order) == nc * heads * (1 + nvb)
    for it, n in where.items():
        if it[0] == "p":
            _, c, h = it
            if c >= ring:
                assert all(where[("s", c - ring, h, vb)] < n for vb in range(nvb))
        else:
            _, c, h, vb = it
            assert where[("p", c, h)] < n
            if c:
                assert where[("s", c - 1, h, vb)] < n


def test_control_the_checks_see_chain_order(monkeypatch):
    """Controls (the tests above would catch a wrong order): the scan redone in numpy from the workspace with the
    same emulated chains equals the kernel's state; with the two key halves' dots swapped it does not; and without
    Triton's combine rewrite (``dot + dot`` as a sum of two chains: what ``_kda_state``'s source says, not what it
    compiles to) the K-split scan's bits differ from ``fast_kda``'s."""

    from triton.runtime import interpreter as interp

    rows = 200
    d = _inputs(rows, seed=51)
    ref = _old(d, rows, 64)
    _same(_new(d, rows, 64, 1, bv=32, ksplit=1), ref)
    assert torch.equal(_scan_numpy(d, rows, swap=False), ref[1])
    assert not torch.equal(_scan_numpy(d, rows, swap=True), ref[1])
    with monkeypatch.context() as m:
        m.setattr(interp.InterpreterBuilder, "create_fadd", _PLAIN_FADD["fadd"])
        plain_old = _old(d, rows, 64)
        plain_new = _new(d, rows, 64, 1, bv=32, ksplit=1)
    assert not torch.equal(plain_new[1], plain_old[1])


def _scan_numpy(d, rows, swap):
    """The state after ``rows`` rows from the prep's workspace (left by the last call), in numpy with the emulated
    chains: E = U - (W S^T over the key halves, in order or swapped), S <- S e^{G_C} + E^T K^."""

    fk = kda_v2.fk
    so = torch.empty_like(d["state_in"])
    kda_v2.kda_prefill_v2(*_args(d, rows), None, so, pos=64, v2=1, cfg=_cfg(bv=32, ksplit=1))
    work = fk._WORK[(H, "cpu")].numpy()
    s = d["state_in"].clone().numpy()
    z = np.zeros((64, 128), np.float32)
    for c in range(-(-rows // 64)):
        for h in range(H):
            wb = work[(c * H + h) * fk.WORK:(c * H + h + 1) * fk.WORK]
            w = wb[fk._G:fk._G + 8192].reshape(64, 128)
            u = wb[fk._V:fk._V + 8192].reshape(64, 128)
            kh = wb[fk._K:fk._K + 8192].reshape(64, 128)
            dec = wb[fk._DEC:fk._DEC + 128]
            sh = s[h]
            halves = [(w[:, :64], sh[:, :64].T), (w[:, 64:], sh[:, 64:].T)]
            if swap:
                halves.reverse()
            ws = _chain(*halves[1], _chain(*halves[0], z, "tf32"), "tf32")
            eb = (u - ws).astype(np.float32)
            s[h] = _chain(eb.T, kh, (sh * dec[None, :]).astype(np.float32), "tf32")
    return torch.from_numpy(s)
