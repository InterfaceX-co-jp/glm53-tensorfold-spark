"""patches/0081: the fast prefill kernels (``fast_qmm.matmul_prefill``, ``fast_kda.kda_prefill_chunked``) on a GPU.

- fast_qmm against ``qmm.matmul`` on GLM-5.3-Flash's per-rank shapes (4-bit and BF16), M in {256, 1024, 1500}:
  close (fp32 rounding), deterministic across runs, and -- the design goal, which makes the kernel usable on the
  exact paths too -- equal bit for bit (``test_fast_qmm_bitwise``); plus a timing print (``-s``) with the
  achieved TFLOP/s;
- fast_kda against the serial CUDA chain (``kda.chain``) on the real shape (32 heads of 128): outputs and state
  within tf32 tolerance for aligned and unaligned ``pos``; deterministic; a prompt cut at multiples of 64 gives the
  bits of one call; the fp32-dot variant against the fp32 torch reference; ``save_replay`` feeds ``kda.replay``.

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda pytest -q -s tests/cuda/test_fastk_patches.py
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.glm5_next.cuda import fast_kda, fast_qmm, kda, qmm  # noqa: E402

DEV = "cuda"
H = 32                                   # KDA heads per rank (64 / TP 2)
C = 3 * H * 128
B_OFF = C + 256                          # [q | k | v | f_a | g_a | b]
WIDTH = B_OFF + H

# (N, K) per rank at TP = 2
SHAPES = [(12576, 4096), (4096, 4096), (2048, 4096), (8192, 1536), (8192, 512), (4096, 8192), (12288, 4096),
          (4096, 6144), (4096, 1024), (4096, 128), (160, 4096), (4096, 1536)]
ROWS = [256, 1024, 1500]


def _q4(n, k, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return qmm.quantize4(torch.randn(n, k, generator=g, device=DEV) * 0.02)


def _x(m, k, seed):
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    return torch.randn(m, k, generator=g, device=DEV).bfloat16()


@pytest.mark.parametrize("m", ROWS)
@pytest.mark.parametrize("n,k", SHAPES)
def test_fast_qmm_close_and_deterministic(n, k, m):
    q = _q4(n, k, n + k)
    x = _x(m, k, n)
    xs = qmm.group_sums(x)
    for f32 in (True, False):
        want = qmm.matmul(x, q, xs, f32=f32)
        got = fast_qmm.matmul_prefill(x, q, xs, f32=f32)
        tol = 1e-5 if f32 else 1e-2
        assert torch.allclose(got.float(), want.float(), rtol=tol, atol=tol * want.float().abs().max().item())
        for _ in range(2):
            assert torch.equal(fast_qmm.matmul_prefill(x, q, xs, f32=f32), got)


@pytest.mark.parametrize("m", ROWS)
@pytest.mark.parametrize("n,k", SHAPES)
def test_fast_qmm_bitwise(n, k, m):
    """Same per-group arithmetic, qmm's K slices summed in slice order: qmm's bits. If this fails on some shape,
    fast_qmm is still a valid (deterministic, row-invariant) fast-prefill kernel, but not a drop-in on exact paths."""

    q = _q4(n, k, n + k)
    x = _x(m, k, n)
    xs = qmm.group_sums(x)
    assert torch.equal(fast_qmm.matmul_prefill(x, q, xs, f32=True), qmm.matmul(x, q, xs, f32=True))
    assert torch.equal(fast_qmm.matmul_prefill(x, q, xs), qmm.matmul(x, q, xs))


@pytest.mark.parametrize("m", [256, 1024])
@pytest.mark.parametrize("n,k", [(12576, 4096), (4096, 4096), (4096, 8192)])
def test_fast_qmm_b16(n, k, m):
    g = torch.Generator(device=DEV).manual_seed(n)
    b = qmm.make_b16(torch.randn(n, k, generator=g, device=DEV) * 0.02)
    x = _x(m, k, k)
    want = qmm.matmul(x, b, f32=True)
    got = fast_qmm.matmul_prefill(x, b, f32=True)
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-5 * want.abs().max().item())
    assert torch.equal(fast_qmm.matmul_prefill(x, b, f32=True), got)


@pytest.mark.parametrize("m", [100, 400, 1024, 2048])
@pytest.mark.parametrize("n,k", SHAPES)
def test_fast_qmm_loose(n, k, m):
    """``matmul_fast`` (fast-prefill chunks: one accumulator): close to qmm, deterministic, and a row's bits do not
    depend on the other rows of the call."""

    q = _q4(n, k, n + k)
    x = _x(m, k, n)
    xs = qmm.group_sums(x)
    for f32 in (True, False):
        want = qmm.matmul(x, q, xs, f32=f32)
        got = fast_qmm.matmul_fast(x, q, xs, f32=f32)
        tol = 1e-5 if f32 else 1e-2
        assert torch.allclose(got.float(), want.float(), rtol=tol, atol=tol * want.float().abs().max().item())
        assert torch.equal(fast_qmm.matmul_fast(x, q, xs, f32=f32), got)
    got = fast_qmm.matmul_fast(x, q, xs, f32=True)
    lo = m // 3
    part = x[lo:lo + 70]
    assert torch.equal(fast_qmm.matmul_fast(part, q, qmm.group_sums(part), f32=True), got[lo:lo + 70])


def test_fast_qmm_strided_rows():
    """The KDA f_b/g_b inputs are column slices of the projection rows."""

    p = _x(1024, 12576, 3)
    fa = p[:, 12288:12288 + 128]
    q = _q4(4096, 128, 11)
    xs = qmm.group_sums(fa)
    assert torch.equal(fast_qmm.matmul_prefill(fa, q, xs), qmm.matmul(fa, q, xs))


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


def test_fast_qmm_timing():
    """Prints qmm vs fast_qmm ms and TFLOP/s at 1024 rows (no assertion on speed)."""

    part = torch.empty((8 * 2048 * 16384,), dtype=torch.float32, device=DEV)
    for m in (1024, 2048):
        for n, k in SHAPES:
            q = _q4(n, k, 1)
            x = _x(m, k, 2)
            xs = qmm.group_sums(x)
            out = torch.empty((m, n), dtype=torch.bfloat16, device=DEV)
            t0 = _time(lambda: qmm.matmul(x, q, xs, out=out, part=part))
            t1 = _time(lambda: fast_qmm.matmul_prefill(x, q, xs, out=out, part=part))
            t2 = _time(lambda: fast_qmm.matmul_fast(x, q, xs, out=out, part=part))
            fl = fast_qmm.flops(m, n, k)
            cfg = fast_qmm.config(n, k, m, qmm.split_k(n, k))
            print(f"\n{n}x{k} M={m}: qmm {t0:.3f} ms ({fl / t0 / 1e9:.1f} TF/s), exact tiles {t1:.3f} ms "
                  f"({fl / t1 / 1e9:.1f} TF/s) [{cfg}], matmul_fast {t2:.3f} ms ({fl / t2 / 1e9:.1f} TF/s)")


# -- fast_kda ----------------------------------------------------------------------------------------------------
def _kda_inputs(rows, seed=0, mode="real"):
    g = torch.Generator(device=DEV).manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, device=DEV)                          # noqa: E731
    p = r(rows, WIDTH)
    a = r(rows, H * 128)
    if mode == "slow":
        a = a * 0.5 - 9.0
        p[:, B_OFF:] += 4.0
    return dict(p=p.bfloat16(), a=a.bfloat16(), g=r(rows, H * 128).bfloat16(), conv_state=r(3, C).bfloat16(),
                conv_w=(r(C, 4) * 0.5).bfloat16(), state_in=r(H, 128, 128) * 0.1, a_log=r(H) * 0.3,
                dt_bias=r(H * 128) * 0.3, norm_w=(1 + 0.1 * r(128)).bfloat16(), eps=1e-6, lower=-5.0)


def _args(d, rows):
    return (d["p"], B_OFF, d["a"], d["g"], d["conv_state"], d["conv_w"], d["state_in"], d["a_log"], d["dt_bias"],
            d["norm_w"], d["eps"], d["lower"], rows)


def _serial(d, rows):
    scratch = kda.KDAScratch(rows, H, DEV)
    so = torch.empty_like(d["state_in"])
    o = kda.chain(*_args(d, rows), scratch, so)
    return o.clone(), so


def _fast(d, rows, pos, **kw):
    so = torch.empty_like(d["state_in"])
    o = fast_kda.kda_prefill_chunked(*_args(d, rows), None, so, pos=pos, **kw)
    return o, so


@pytest.mark.parametrize("mode", ["real", "slow"])
@pytest.mark.parametrize("rows,pos", [(1024, 0), (1024, 64), (1000, 37), (256, 4095), (1, 3)])
def test_fast_kda_vs_serial_chain(rows, pos, mode):
    d = _kda_inputs(rows, seed=rows + pos, mode=mode)
    ref_o, ref_s = _serial(d, rows)
    o, s = _fast(d, rows, pos)
    assert torch.isfinite(o.float()).all() and torch.isfinite(s).all()
    rel = ((s - ref_s).norm() / ref_s.norm()).item()
    assert rel < 3e-3, rel
    diff = (o.float() - ref_o.float()).abs()
    assert diff.mean() < 3e-3, diff.mean()
    assert (diff <= 0.05 + 0.02 * ref_o.float().abs()).all(), diff.max()


@pytest.mark.parametrize("pos", [0, 21])
def test_fast_kda_ieee_vs_reference(pos):
    rows = 200
    d = _kda_inputs(rows, seed=4, mode="slow")
    ref_o, ref_s = fast_kda.serial_reference(*_args(d, rows))
    o, s = _fast(d, rows, pos, precision="ieee")
    assert (s - ref_s).abs().max() <= 1e-4 * max(1.0, ref_s.abs().max().item())
    diff = (o.float() - ref_o.float()).abs()
    assert (diff <= 1e-2 + 8e-3 * ref_o.float().abs()).all(), diff.max()


def test_fast_kda_deterministic_and_split_invariant():
    rows = 1024
    d = _kda_inputs(rows, seed=8)
    o1, s1 = _fast(d, rows, 2048)
    for _ in range(2):
        o2, s2 = _fast(d, rows, 2048)
        assert torch.equal(o1, o2) and torch.equal(s1, s2)
    cut = 320                                                                         # a multiple of 64
    o_a, s_a = _fast(d, cut, 2048)
    rest = dict(d, p=d["p"][cut:], a=d["a"][cut:], g=d["g"][cut:],
                conv_state=d["p"][cut - 3:cut, :C].contiguous(), state_in=s_a)
    o_b, s_b = _fast(rest, rows - cut, 2048 + cut)
    assert torch.equal(torch.cat([o_a, o_b]), o1)
    assert torch.equal(s_b, s1)


def test_fast_kda_save_replay():
    rows = 300
    d = _kda_inputs(rows, seed=12)
    scratch = kda.KDAScratch(rows, H, DEV)
    so = torch.empty_like(d["state_in"])
    fast_kda.kda_prefill_chunked(*_args(d, rows), scratch, so, pos=10, save_replay=True)
    ref_scratch = kda.KDAScratch(rows, H, DEV)
    kda.chain(*_args(d, rows), ref_scratch, torch.empty_like(so))
    # The conv's fp32 sum is ordered differently from the chain kernel's, so a few activations round to the
    # neighbouring bf16 value (measured: 4 of 300 rows' k, 3 v elements); such a row's k differs by up to one bf16
    # step in one channel and ~1e-5 in the others (its norm). Everything else matches to fp32 rounding.
    for name in ("k", "v", "g", "b"):
        got, want = getattr(scratch, name)[:rows].float(), getattr(ref_scratch, name)[:rows].float()
        diff = (got - want).abs()
        assert (diff <= 1e-6 + 1e-5 * want.abs()).float().mean() > 0.999, name
        assert (diff <= 1e-6 + 2 ** -6 * want.abs()).all(), (name, diff.max())
    replayed = torch.empty_like(so)
    kda.replay(d["state_in"], scratch, rows, replayed)
    assert ((replayed - so).norm() / so.norm()).item() < 3e-3


def test_fast_kda_timing():
    """Prints kda.chain vs the chunked kernel at 1024 rows, 32 heads (no assertion on speed)."""

    rows = 1024
    d = _kda_inputs(rows, seed=1)
    scratch = kda.KDAScratch(rows, H, DEV)
    so = torch.empty_like(d["state_in"])
    t0 = _time(lambda: kda.chain(*_args(d, rows), scratch, so))
    t1 = _time(lambda: fast_kda.kda_prefill_chunked(*_args(d, rows), None, so, pos=0))
    t2 = _time(lambda: fast_kda.kda_prefill_chunked(*_args(d, rows), None, so, pos=0, precision="tf32x3"))
    print(f"\nKDA 1024 rows x 32 heads: chain {t0:.3f} ms, chunked tf32 {t1:.3f} ms (x{t0 / t1:.1f}), "
          f"tf32x3 {t2:.3f} ms")


# -- hc_partial (hyper-connection mixing dots of fast chunks) -------------------------------------------------------
@pytest.mark.parametrize("rows", [1, 100, 1024, 2048])
def test_fast_hc_partial(rows):
    """Row-tiled tensor-core version of ``glue._hc_partial``: close to it, deterministic, and a row's values do not
    depend on the other rows of the call. Prints the timing at 1024+ rows."""

    from tensorfold.families.glm5_next.cuda import glue

    wide, nb = 4 * 4096, glue.HC_BLOCKS
    g = torch.Generator(device=DEV).manual_seed(rows)
    x = torch.randn(rows, wide, generator=g, device=DEV).bfloat16()
    fn = (torch.randn(24, wide, generator=g, device=DEV) * 0.02).bfloat16()
    want = torch.zeros(rows, nb, 32, device=DEV)
    got = torch.zeros(rows, nb, 32, device=DEV)
    glue._hc_partial[(rows, nb)](x, fn, want, WIDE=wide, NB=nb, SUB=128, num_warps=4)
    fast_qmm.hc_partial(x, fn, got, nb)
    assert torch.allclose(got[..., :25], want[..., :25], rtol=1e-4, atol=1e-4 * want[..., :25].abs().max().item())
    again = torch.zeros_like(got)
    fast_qmm.hc_partial(x, fn, again, nb)
    assert torch.equal(again, got)
    for r in {0, rows // 2, rows - 1}:
        one = torch.zeros(1, nb, 32, device=DEV)
        fast_qmm.hc_partial(x[r:r + 1].contiguous(), fn, one, nb)
        assert torch.equal(one[0, :, :25], got[r, :, :25]), r
    if rows >= 1024:
        t0 = _time(lambda: glue._hc_partial[(rows, nb)](x, fn, want, WIDE=wide, NB=nb, SUB=128, num_warps=4))
        t1 = _time(lambda: fast_qmm.hc_partial(x, fn, got, nb))
        print(f"\nhc_pre dots {rows} rows: per-row kernel {t0:.3f} ms, row tiles {t1:.3f} ms, x{t0 / t1:.1f}")
