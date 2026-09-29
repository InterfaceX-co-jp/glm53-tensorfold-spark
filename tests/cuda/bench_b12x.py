"""patches/0240 micro-benchmark: the b12x-derived fast-prefill kernels against today's fast path, on the real per-rank
shapes of GLM-5.3-Flash (KDA 32 heads of 128; hidden 4096 x 4 streams; 32 latent heads of 512 over a 32k context), one
GPU, about a minute. Checks the numbers while it is at it.

    PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda python tests/cuda/bench_b12x.py [rows ...] [--sweep]

Per row count (default 1024 2048 8192):
- [kda]  ``fast_kda.kda_prefill_chunked`` (today, ``kda.chain`` in the profile) vs ``b12x_kda.kda_prefill16`` (bit 2);
         the error of both against the serial CUDA chain (``kda.chain``, the decode kernel), the max difference between
         them, determinism and "split at 64 == one call" (both must say True). ``--sweep``: value block / warps /
         stages of the new kernel (with bitwise against its default: a False means that setting is other arithmetic);
- [hc]   one hc site (hc_post, then the next hc_pre) in 0084's 384-row slabs: today's kernels (``_hc_post`` +
         ``fast_qmm.hc_partial`` + ``_hc_finish``) vs ``b12x_mhc`` fused (bit 1) and unfused; fused == unfused must be
         True (else bit 1 must stay off: the pipelined and plain lean chunks would disagree); max difference vs today;
- [attn] sparse latent attention, 2,048 selected tokens a row (1,024 rows; bf16 and FP8 caches): today's 512-token
         chunks + merge at 32 queries a tile vs ``b12x_attn`` one pass (bit 4), max difference, row subsets bitwise;
- the expected saving over a 28k / 112k prompt (34 KDA layers, 90 hc sites, 11 DSA layers).
"""

from __future__ import annotations

import sys

import torch
import triton

H, DK, D, S = 32, 128, 4096, 4
C = 3 * H * DK
B_OFF = C + 2 * H * DK            # after [q | k | v | f_a | g_a]: where the beta logits sit (layout only)
ITERS = 20


def _ms(fn, reps=10, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


# -- KDA ---------------------------------------------------------------------------------------------------------------
def _kda_inputs(rows, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    p = torch.randn(rows, B_OFF + H + 8, device=dev, generator=g)
    a = torch.randn(rows, H * DK, device=dev, generator=g)
    # decays: half the channels slow (~1 - 2e-4 a step), half fast (~e^-5): both ends of the tile's exponent range
    a = torch.where(torch.rand(rows, H * DK, device=dev, generator=g) < 0.5, a * 0.5 - 9.0, a * 0.5 + 12.0)
    return dict(p=p.bfloat16(), b_off=B_OFF, a=a.bfloat16(),
                g=torch.randn(rows, H * DK, device=dev, generator=g).bfloat16(),
                conv_state=torch.randn(3, C, device=dev, generator=g).bfloat16(),
                conv_w=(torch.randn(C, 4, device=dev, generator=g) * 0.5).bfloat16(),
                state_in=torch.randn(H, DK, DK, device=dev, generator=g) * 0.1,
                a_log=torch.randn(H, device=dev, generator=g) * 0.2,
                dt_bias=torch.randn(H * DK, device=dev, generator=g) * 0.2,
                norm_w=(1 + 0.1 * torch.randn(DK, device=dev, generator=g)).bfloat16(), eps=1e-6, lower=-5.0)


def _args(d, rows):
    return (d["p"], d["b_off"], d["a"], d["g"], d["conv_state"], d["conv_w"], d["state_in"], d["a_log"], d["dt_bias"],
            d["norm_w"], d["eps"], d["lower"], rows)


def bench_kda(rows, sweep=False):
    from tensorfold.families.glm5_next.cuda import b12x_kda, fast_kda, kda

    d = _kda_inputs(rows, seed=rows)
    pos = 4096                                          # a 64-aligned fast call past the start
    so_old, so_new, so_ref = (torch.empty_like(d["state_in"]) for _ in range(3))
    out_old = torch.empty(rows, H * DK, dtype=torch.bfloat16, device="cuda")
    out_new = torch.empty_like(out_old)

    def old():
        fast_kda.kda_prefill_chunked(*_args(d, rows), None, so_old, pos=pos, out=out_old)

    def new(**kw):
        b12x_kda.kda_prefill16(*_args(d, rows), None, so_new, pos=pos, out=out_new, **kw)

    old()
    new()
    torch.cuda.synchronize()
    ref_scr = kda.KDAScratch(rows, H, "cuda")
    ref_o = kda.chain(*_args(d, rows), ref_scr, so_ref).clone()
    o_old, s_old, o_new, s_new = out_old.clone(), so_old.clone(), out_new.clone(), so_new.clone()

    def rel(s):
        return ((s - so_ref).norm() / so_ref.norm()).item()

    e_old = (o_old.float() - ref_o.float()).abs()
    e_new = (o_new.float() - ref_o.float()).abs()
    new()
    det = torch.equal(out_new, o_new) and torch.equal(so_new, s_new)
    # split at a multiple of 64 == one call
    cut = (rows // 2) // 64 * 64
    d2 = dict(d)
    sa = torch.empty_like(d["state_in"])
    oa = b12x_kda.kda_prefill16(*_args(d, cut), None, sa, pos=pos)
    d2.update(p=d["p"][cut:], a=d["a"][cut:], g=d["g"][cut:], conv_state=d["p"][cut - 3:cut, :C].contiguous(),
              state_in=sa)
    sb = torch.empty_like(d["state_in"])
    ob = b12x_kda.kda_prefill16(*_args(d2, rows - cut), None, sb, pos=pos + cut)
    split = torch.equal(torch.cat([oa, ob]), o_new) and torch.equal(sb, s_new)
    t_old = _ms(old)
    t_new = _ms(new)
    t_chain = _ms(lambda: kda.chain(*_args(d, rows), ref_scr, so_ref), reps=3, warm=1)
    print(f"[kda] rows {rows:5d}: kda.chain {t_chain:7.3f} ms | fast_kda {t_old:7.3f} ms -> b12x_kda {t_new:7.3f} ms "
          f"({t_old / t_new:4.2f}x, {b12x_kda.PREC}, BV {b12x_kda.BV})")
    print(f"          vs chain: state rel {rel(s_old):.2e} -> {rel(s_new):.2e}, out mean {e_old.mean():.2e} -> "
          f"{e_new.mean():.2e}, max {e_old.max():.3e} -> {e_new.max():.3e}; new vs old max diff "
          f"{(o_new.float() - o_old.float()).abs().max():.3e} (bitwise {torch.equal(o_new, o_old)}); deterministic "
          f"{det}; split at 64 == one call {split}")
    if sweep:
        for prec in ("tf32", "bf16"):
            for bv in (16, 32, 64, 128):
                for warps in (4, 8):
                    for stages in (1, 2, 3):
                        try:
                            new(bv=bv, prec=prec, num_warps=warps, num_stages=stages)
                            same = torch.equal(out_new, o_new) and torch.equal(so_new, s_new)
                            t = _ms(lambda: new(bv=bv, prec=prec, num_warps=warps, num_stages=stages), reps=5)
                            print(f"          {prec} BV {bv:3d} warps {warps} stages {stages}: {t:7.3f} ms "
                                  f"(bitwise vs default {same}, state rel {rel(so_new):.2e})")
                        except Exception as exc:                            # noqa: BLE001
                            print(f"          {prec} BV {bv:3d} warps {warps} stages {stages}: FAILED "
                                  f"{type(exc).__name__}: {str(exc)[:100]}")
    return (t_old - t_new) / rows


# -- hyper-connections -------------------------------------------------------------------------------------------------
def bench_hc(rows, slab=384):
    from tensorfold.families.glm5_next.cuda import b12x_mhc, fast_qmm, glue

    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(7)
    x0 = torch.randn(rows, S * D, device=dev, generator=g).to(torch.bfloat16)
    part16 = (torch.randn(2, rows, D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    post0 = torch.rand(rows, S, device=dev, generator=g) * 2
    comb0 = torch.rand(rows, S * S, device=dev, generator=g) / 4
    fn = (torch.randn(24, S * D, device=dev, generator=g) * 0.02).to(torch.bfloat16)
    base = torch.randn(24, device=dev, generator=g) * 0.1
    scale = torch.tensor([0.4, 0.6, 0.8], device=dev)
    nw = (torch.rand(D, device=dev, generator=g) + 0.5).to(torch.bfloat16)
    hcpart = torch.empty(slab, glue.HC_BLOCKS, 32, device=dev)
    normed = torch.empty(rows, D, dtype=torch.bfloat16, device=dev)
    xs = torch.empty(rows, D // 64, device=dev)

    def site(x, post, comb, mhc, fused):
        saved = (glue.HC_FUSED, glue.FAST_HC, glue.MHC)
        glue.HC_FUSED, glue.FAST_HC, glue.MHC = 0, fast_qmm.hc_partial, mhc
        try:
            for s0 in range(0, rows, slab):
                n = min(slab, rows - s0)
                xx, pp, cc = x[s0:s0 + n], post[s0:s0 + n], comb[s0:s0 + n]
                if fused:
                    glue.hc_post_pre(xx, xx, part16[:, s0:s0 + n], pp, cc, fn, base, scale, nw, normed[s0:s0 + n],
                                     xs[s0:s0 + n], pp, cc, hcpart[:n], 1e-6, 1e-6, ITERS)
                else:
                    glue.hc_post(xx, xx, part16[:, s0:s0 + n], pp, cc)
                    glue.hc_pre(xx, fn, base, scale, nw, normed[s0:s0 + n], xs[s0:s0 + n], pp, cc, hcpart[:n], 1e-6,
                                1e-6, ITERS)
        finally:
            glue.HC_FUSED, glue.FAST_HC, glue.MHC = saved

    def once(mhc, fused):
        x, post, comb = x0.clone(), post0.clone(), comb0.clone()
        site(x, post, comb, mhc, fused)
        torch.cuda.synchronize()
        return x, post, comb, normed.clone(), xs.clone()

    ref = once(False, False)
    fu = once(True, True)
    un = once(True, False)
    same = all(torch.equal(a, b) for a, b in zip(fu, un))
    xb, pb, cb = x0.clone(), post0.clone(), comb0.clone()
    t0 = _ms(lambda: site(xb, pb, cb, False, False))
    t1 = _ms(lambda: site(xb, pb, cb, True, True))
    t2 = _ms(lambda: site(xb, pb, cb, True, False))
    dn = (fu[3].float() - ref[3].float()).abs().max().item()
    dp = max((fu[1] - ref[1]).abs().max().item(), (fu[2] - ref[2]).abs().max().item())
    broken = f" (fused kernel unavailable: {b12x_mhc._BROKEN[:100]})" if b12x_mhc._BROKEN else ""
    print(f"[hc] rows {rows:5d}, slabs of {slab}: today {t0:7.3f} ms ({t0 * 1e3 / rows:5.3f} us/row/site) -> b12x fused "
          f"{t1:7.3f} ({t0 / t1:4.2f}x), unfused {t2:7.3f}; fused == unfused bitwise {same}{broken}; vs today: new "
          f"streams bitwise {torch.equal(fu[0], ref[0])}, post/comb max diff {dp:.2e}, normed rows max diff {dn:.3e}")
    return (t0 - t1) / rows if same else 0.0


# -- sparse attention --------------------------------------------------------------------------------------------------
def bench_attn(rows=1024):
    from tensorfold.families.glm5_next.cuda import b12x_attn, latent

    Hh, L, P = 32, 512, 32768
    g = torch.Generator(device="cpu").manual_seed(1)
    lat = torch.randn(P + rows + 64, L, generator=g).cuda().bfloat16()
    qa = (torch.randn(rows, Hh, L, generator=g) * 0.3).cuda().bfloat16()
    W = 2051
    tok = torch.stack([torch.randperm(P, generator=g)[:W].sort().values for _ in range(rows)]).cuda().int()
    cnt = torch.full((rows,), W, dtype=torch.int32, device="cuda")
    scale = 256 ** -0.5
    total = 0.0
    for name, lc in (("bf16", lat), ("fp8", latent.quantize_rows_reference(lat))):
        o_old = torch.zeros((rows, Hh, L), device="cuda")
        o_new = torch.zeros((rows, Hh, L), device="cuda")
        latent.sparse_latent(qa, lc, tok, cnt, o_old, scale, bm=latent.FAST_BM)
        b12x_attn.sparse_latent_one(qa, lc, tok, cnt, o_new, scale)
        torch.cuda.synchronize()
        t_old = _ms(lambda: latent.sparse_latent(qa, lc, tok, cnt, o_old, scale, bm=latent.FAST_BM), reps=5)
        t_new = _ms(lambda: b12x_attn.sparse_latent_one(qa, lc, tok, cnt, o_new, scale), reps=5)
        sub = torch.zeros((rows // 3, Hh, L), device="cuda")
        b12x_attn.sparse_latent_one(qa[5:5 + rows // 3].contiguous(), lc, tok[5:5 + rows // 3].contiguous(),
                                    cnt[5:5 + rows // 3].contiguous(), sub, scale)
        rows_ok = torch.equal(sub, o_new[5:5 + rows // 3])
        again = torch.zeros_like(o_new)
        b12x_attn.sparse_latent_one(qa, lc, tok, cnt, again, scale)
        print(f"[attn] {rows} rows, {W} tokens a row, {name} cache: chunked BM 32 {t_old:6.2f} ms -> one pass {t_new:6.2f} "
              f"ms ({t_old / t_new:4.2f}x); max diff {(o_new - o_old).abs().max():.3e} (rel "
              f"{((o_new - o_old).norm() / o_old.norm()).item():.2e}); row subset bitwise {rows_ok}; deterministic "
              f"{torch.equal(again, o_new)}")
        if name == "fp8":
            total = (t_old - t_new) / rows
    return total


def main():
    if not torch.cuda.is_available():
        print("needs a GPU")
        return
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sweep = "--sweep" in sys.argv
    rows_list = [int(a) for a in args] or [1024, 2048, 8192]
    print(f"[bench_b12x] {torch.cuda.get_device_name()}, triton {triton.__version__}")

    def guard(name, fn, *a):
        try:
            return fn(*a)
        except Exception as exc:                                        # noqa: BLE001
            print(f"[{name}] FAILED: {type(exc).__name__}: {str(exc)[:300]}")
            torch.cuda.synchronize()
            return None

    for rows in rows_list:
        dd = guard("kda", bench_kda, rows, sweep and rows == rows_list[0])
        if dd:
            print(f"          b12x kda saves {dd * 1e3:.3f} us/row/layer: 28k prompt {dd * 34 * 28045:.0f} ms, "
                  f"112k {dd * 34 * 112000:.0f} ms (34 KDA layers)")
    for rows in rows_list:
        dh = guard("hc", bench_hc, rows)
        if dh:
            print(f"          b12x hc saves {dh * 1e3:.3f} us/row/site: 28k prompt {dh * 90 * 28045:.0f} ms, "
                  f"112k {dh * 90 * 112000:.0f} ms (90 sites a token)")
    da = guard("attn", bench_attn)
    if da:
        print(f"          b12x attn saves {da * 1e3:.2f} us/row/layer: 28k prompt ~{da * 11 * (28045 - 2051):.0f} ms, "
              f"112k ~{da * 11 * (112000 - 2051):.0f} ms (11 DSA layers; the MTP head keeps the chunked kernel)")


if __name__ == "__main__":
    main()
