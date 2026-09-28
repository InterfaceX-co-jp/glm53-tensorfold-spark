"""patches/0190 micro-benchmark: old vs new prefill glue kernels on the real per-rank shapes (GLM-5.3-Flash: hidden 4096,
4 streams, 288 experts top 8), one GPU, ~1 minute. Checks the outputs are equal while it is at it.

    PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda python tests/cuda/bench_glue.py [rows ...]

Prints, per row count (default 1024 4096 8192):
- MoE routing (``moe_glue``): router in 8 K-slice partials + sum vs one kernel (bit 2), top-k, ``_group`` (serial) vs
  ``group_sorted`` (bit 1), the combine after a copy of the shared expert's rows vs reading them in place (bit 4);
- hyper-connections: hc_post + hc_pre as today (``_hc_post``, ``fast_qmm.hc_partial``, ``_hc_finish``) vs
  ``hc_fused`` modes 1 / 2 / 3 (and fused row tiles of 16 / 32), in 384-row slabs like 0084's pipeline;
- latent attention, 16- vs 32-query tiles (``attn_bm32``), sparse and dense, 1,024 rows;
- the expected saving over a 28k / 112k prompt (42 MoE layers, 90 hc sites, 12 DSA layers a token).
Every line says whether the new kernel gave the same bits; a knob whose line says False must stay off.
"""

from __future__ import annotations

import sys

import torch
import triton

from tensorfold.families.glm5_next.cuda import glue

D, S, E, TOP = 4096, 4, 288, 8
SLOTS = TOP + 1
ITERS = 20


def _ms(fn, reps=20, warm=3):
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


def _picks(rows):
    g = torch.Generator(device="cuda").manual_seed(rows)
    # locality like a real prompt: blocks of 64 rows share a preference
    base = torch.randn(rows // 64 + 1, E, device="cuda", generator=g).repeat_interleave(64, 0)[:rows]
    logits = torch.randn(rows, E, device="cuda", generator=g) * 0.3 + base
    return logits


def bench_routing(rows):
    dev = "cuda"
    x = torch.randn(rows, D, device=dev).to(torch.bfloat16)
    w = (torch.randn(E, D, device=dev) * 0.02).to(torch.bfloat16)
    mlog = torch.empty(rows, E, device=dev)
    bias = torch.zeros(E, device=dev)
    maxu = min(rows * TOP, E) + 1
    pick = torch.empty(rows, SLOTS, dtype=torch.int32, device=dev)
    wts = torch.empty(rows, SLOTS, dtype=torch.float32, device=dev)
    ids = torch.zeros(maxu, dtype=torch.int32, device=dev)
    cnt = torch.zeros(1, dtype=torch.int32, device=dev)
    mem = torch.full((maxu, rows), -1, dtype=torch.int32, device=dev)
    t_router = _ms(lambda: glue.router(x, w, mlog))
    ref_log = mlog.clone()
    glue.ROUTER_FUSED = True
    try:
        glue.router(x, w, mlog)
        same_r = torch.equal(ref_log, mlog)
        t_router2 = _ms(lambda: glue.router(x, w, mlog))
    finally:
        glue.ROUTER_FUSED = False
    logits = _picks(rows)
    block = triton.next_power_of_2(E + 1)
    t_topk = _ms(lambda: glue._topk[(rows,)](logits, bias, pick, wts, 2.5, NE=E, TOPK=TOP, SLOTS=SLOTS, BLOCK=block,
                                             SLOTP=triton.next_power_of_2(SLOTS), NORM=True, num_warps=4))
    old = lambda: glue._group[(1,)](pick, ids, cnt, mem, rows, SLOTS=SLOTS, MAXU=maxu, MAXM=rows,  # noqa: E731
                                    BLOCK=block, num_warps=8)
    t_old = _ms(old, reps=5, warm=1)
    ref = (ids.clone(), cnt.clone(), mem.clone())
    new = lambda: glue.group_sorted(pick, rows, ids, cnt, mem, E)          # noqa: E731
    t_new = _ms(new)
    same = torch.equal(ref[0], ids) and torch.equal(ref[1], cnt) and torch.equal(ref[2], mem)
    # combine: copy the shared expert's rows into ey's last slot, then _combine; vs _combine_s reading them in place
    ey = torch.randn(rows, SLOTS, D, device=dev)
    sy = torch.randn(rows, D, device=dev)
    out1 = torch.empty(rows, D, dtype=torch.bfloat16, device=dev)
    out2 = torch.empty(rows, D, dtype=torch.bfloat16, device=dev)

    def copy_combine():
        ey[:, TOP].copy_(sy)
        glue.combine(ey, wts, out1)

    copy_combine()
    glue.combine(ey, wts, out2, shared=sy)
    same_c = torch.equal(out1, out2)
    t_c1 = _ms(copy_combine)
    t_c2 = _ms(lambda: glue.combine(ey, wts, out2, shared=sy))
    print(f"[routing] rows {rows:5d}: router {t_router:7.3f} -> one kernel {t_router2:7.3f} ms (bitwise {same_r}), "
          f"top-k {t_topk:6.3f}, group serial {t_old:7.3f} -> sorted {t_new:6.3f} ms ({t_old / max(t_new, 1e-6):5.1f}x, "
          f"same ints: {same}), copy + combine {t_c1:6.3f} -> in place {t_c2:6.3f} ms (bitwise {same_c})")
    return (t_old - t_new) + (t_router - t_router2) + (t_c1 - t_c2)


def bench_hc(rows, slab=384):
    """One hc site (hc_post, then the next hc_pre) over ``rows`` rows in 0084's slabs: separate kernels vs each
    ``hc_fused`` mode (1: post + partial fused, 2: unrolled finish, 3: both) and fused tiles of 16 / 32 rows."""

    from tensorfold.families.glm5_next.cuda import fast_qmm

    if not hasattr(glue, "hc_post_pre"):
        print(f"[hc] rows {rows}: fused kernels not in this build")
        return 0.0
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

    def site(x, post, comb, mode):
        saved = (glue.HC_FUSED, glue.FAST_HC)
        glue.HC_FUSED, glue.FAST_HC = mode, fast_qmm.hc_partial
        try:
            for s0 in range(0, rows, slab):
                n = min(slab, rows - s0)
                glue.hc_post_pre(x[s0:s0 + n], x[s0:s0 + n], part16[:, s0:s0 + n], post[s0:s0 + n], comb[s0:s0 + n],
                                 fn, base, scale, nw, normed[s0:s0 + n], xs[s0:s0 + n], post[s0:s0 + n],
                                 comb[s0:s0 + n], hcpart[:n], 1e-6, 1e-6, ITERS)
        finally:
            glue.HC_FUSED, glue.FAST_HC = saved

    def once(mode):
        x, post, comb = x0.clone(), post0.clone(), comb0.clone()
        site(x, post, comb, mode)
        torch.cuda.synchronize()
        return x, post, comb, normed.clone(), xs.clone()

    ref = once(0)
    xb, pb, cb = x0.clone(), post0.clone(), comb0.clone()
    t = {0: _ms(lambda: site(xb, pb, cb, 0))}
    line = [f"[hc] rows {rows:5d}, slabs of {slab}: separate {t[0]:7.3f} ms ({t[0] * 1e3 / rows:5.3f} us/row/site)"]
    best = t[0]
    for mode in (1, 2, 3):
        try:
            got = once(mode)
            same = all(torch.equal(a, b) for a, b in zip(ref, got))
            if mode & 1 and glue._HC_FUSED_BROKEN:
                line.append(f"mode {mode}: fused kernel unavailable ({glue._HC_FUSED_BROKEN[:120]})")
                continue
            t[mode] = _ms(lambda: site(xb, pb, cb, mode))
        except Exception as exc:                                        # noqa: BLE001
            line.append(f"mode {mode} FAILED {type(exc).__name__}: {str(exc)[:160]}")
            continue
        line.append(f"mode {mode} {t[mode]:7.3f} ({t[0] / max(t[mode], 1e-6):4.2f}x, bitwise {same})")
        if same:
            best = min(best, t[mode])
    saved_bm = glue.HC_FUSED_BM
    for bm in (16, 32):
        glue.HC_FUSED_BM = bm
        try:
            got = once(3)
            same = all(torch.equal(a, b) for a, b in zip(ref, got))
            tt = _ms(lambda: site(xb, pb, cb, 3))
            line.append(f"mode 3 BM {bm} {tt:7.3f} (bitwise {same})")
        except Exception as exc:                                        # noqa: BLE001
            line.append(f"mode 3 BM {bm} FAILED {type(exc).__name__}")
        finally:
            glue.HC_FUSED_BM = saved_bm
    print("; ".join(line))
    return (t[0] - best) / rows


def bench_attn(rows=1024):
    """Latent attention, 16- vs 32-query tiles (``attn_bm32``), dense (1,024 rows at 1,024) and sparse (2,048
    selected tokens a row of a 32k context), bit-identity checked."""

    from types import SimpleNamespace

    from tensorfold.families.glm5_next.cuda import latent

    H, L, P = 32, 512, 32768
    g = torch.Generator(device="cpu").manual_seed(1)
    lc = torch.randn(P + rows + 64, L, generator=g).cuda().bfloat16()
    qa = torch.randn(rows, H, L, generator=g).cuda().bfloat16()
    W = 2048
    tok = torch.stack([torch.randperm(P, generator=g)[:W].sort().values for _ in range(rows)]).cuda().int()
    cnt = torch.full((rows,), W, dtype=torch.int32, device="cuda")
    outs, ts = [], []
    for bm in (latent.BM, latent.FAST_BM):
        o = torch.zeros((rows, H, L), device="cuda")
        latent.sparse_latent(qa, lc, tok, cnt, o, 0.05, bm=bm)
        outs.append(o)
        ts.append(_ms(lambda: latent.sparse_latent(qa, lc, tok, cnt, o, 0.05, bm=bm), reps=5))
    same_s = torch.equal(outs[0], outs[1])
    Pd = 1024
    cfg = SimpleNamespace(heads=H * 2, kv_lora=L, v_dim=128, dense_limit=2051)
    sc = latent.Scratch(SimpleNamespace(cfg=cfg, world=2), rows, Pd + rows + 64, "cuda")
    pos = torch.tensor([Pd], dtype=torch.int32, device="cuda")
    nch = -(-(Pd + rows) // latent.CHUNK)
    douts, dts = [], []
    for bm in (latent.BM, latent.FAST_BM):
        o = torch.empty((rows, H, L), device="cuda")
        latent.attention_latent(qa, lc, pos, sc, scale=0.05, nch=nch, out=o, bm=bm)
        douts.append(o.clone())
        dts.append(_ms(lambda: latent.attention_latent(qa, lc, pos, sc, scale=0.05, nch=nch, out=o, bm=bm), reps=5))
    same_d = torch.equal(douts[0], douts[1])
    print(f"[attn] {rows} rows, one layer: sparse BM 16 {ts[0]:6.2f} ms -> BM 32 {ts[1]:6.2f} ms (bitwise {same_s}); "
          f"dense {dts[0]:6.2f} -> {dts[1]:6.2f} ms (bitwise {same_d})")
    d = (ts[0] - ts[1]) / rows
    print(f"          sparse saves {d * 1e3:.2f} us/row/layer: 28k prompt ~{d * 12 * (28045 - 2051):.0f} ms, "
          f"112k ~{d * 12 * (112000 - 2051):.0f} ms (11 DSA layers + the MTP head's)")


def main():
    if not torch.cuda.is_available():
        print("needs a GPU")
        return
    rows_list = [int(a) for a in sys.argv[1:]] or [1024, 4096, 8192]
    print(f"[bench_glue] {torch.cuda.get_device_name()}, triton {triton.__version__}")
    def guard(name, fn, *a):
        """One section; an error (e.g. a kernel that cannot run here) is printed and the next section runs."""
        try:
            return fn(*a)
        except Exception as exc:                                        # noqa: BLE001
            print(f"[{name}] FAILED: {type(exc).__name__}: {str(exc)[:300]}")
            torch.cuda.synchronize()
            return None

    for rows in rows_list:
        saved = guard("routing", bench_routing, rows)
        if saved is None:
            continue
        per_tok = saved / rows
        print(f"          moe_glue=7 saves {saved:.3f} ms a MoE layer at {rows} rows = {per_tok * 1e3:.2f} us/token/layer; "
              f"28k prompt: {per_tok * 42 * 28045:.0f} ms, 112k: {per_tok * 42 * 112000:.0f} ms")
    for rows in rows_list:
        d = guard("hc", bench_hc, rows)
        if d:
            print(f"          hc saves {d * 1e3:.3f} us/row/site: 28k prompt {d * 90 * 28045:.0f} ms, "
                  f"112k {d * 90 * 112000:.0f} ms (90 sites a token)")
    guard("attn", bench_attn)


if __name__ == "__main__":
    main()
