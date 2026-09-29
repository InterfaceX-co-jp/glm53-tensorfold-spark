"""patches/0400 micro-benchmark: the KDA prefill recurrence, ``fast_kda`` (today: ``_kda_prep`` + ``_kda_state`` +
``_kda_norm``) against ``kda_v2`` (GLM53_TF_KDA_V2 split / fused) on the real per-rank shape (32 heads of 128), one
GPU, a few minutes. Every timed setting is also checked BITWISE against ``fast_kda`` (outputs + state); a setting that
is not bit-identical is reported as such and never recommended.

    PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda python tests/cuda/bench_kda_v2.py [rows ...] [--sweep]
        [--reps N] [--csv FILE]

Default rows: 512 (production: the lean sub-block of GLM53_TF_LEAN_BLOCK=512), 1024, 2048, 8192. Per row count:
- ``fast_kda`` total and its three kernels alone (prep / state / norm: the profile's 568 / 334 / 41 us at 512 rows);
- each v2 setting: ms a call, speedup, bitwise equal (outputs, state), and the projection to a prefill token:
  us a token = ms / rows * 34 KDA layers (the roofline's "KDA chunked recurrence" line: 63.5 us a token today);
- ``--sweep``: every split (bv x warps x ksplit x register cap) and fused (fbv x ksplit x lag / ring x programs x
  register cap) setting;
  default: a short list. The last line names the fastest bit-identical setting per row count as env lines.
"""

from __future__ import annotations

import csv
import itertools
import sys

import torch

from tensorfold.families.glm5_next.cuda import fast_kda, kda_v2

H = 32
C = 3 * H * 128
B_OFF = C + 256
WIDTH = B_OFF + H
LAYERS = 34


def _inputs(rows, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, device="cuda")                       # noqa: E731
    p = r(rows, WIDTH)
    a = r(rows, H * 128)
    slow = torch.rand(1, H * 128, generator=g, device="cuda") < 0.75
    a = torch.where(slow, a * 0.5 - 9.0, a * 0.5 + 12.0)
    return dict(p=p.bfloat16(), a=a.bfloat16(), g=r(rows, H * 128).bfloat16(), conv_state=r(3, C).bfloat16(),
                conv_w=(r(C, 4) * 0.5).bfloat16(), state_in=r(H, 128, 128), a_log=r(H) * 0.3,
                dt_bias=r(H * 128) * 0.3, norm_w=(1 + 0.1 * r(128)).bfloat16(), eps=1e-6, lower=-5.0)


def _args(d, rows):
    return (d["p"], B_OFF, d["a"], d["g"], d["conv_state"], d["conv_w"], d["state_in"], d["a_log"], d["dt_bias"],
            d["norm_w"], d["eps"], d["lower"], rows)


def _ms(fn, reps, warm=3):
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


def _parts(d, rows, reps):
    """fast_kda's three kernels timed alone (the same launches as kda_prefill_chunked)."""

    import triton

    off, nc = 0, -(-rows // 64)
    work = fast_kda.workspace(H, nc, "cuda")
    o = torch.empty((rows, H * 128), dtype=torch.bfloat16, device="cuda")
    so = torch.empty_like(d["state_in"])
    p, a, g = d["p"], d["a"], d["g"]

    def prep():
        fast_kda._kda_prep[(nc, H)](p, p.stride(0), B_OFF, a, a.stride(0), d["conv_state"], d["conv_w"], d["a_log"],
                                    d["dt_bias"], work, work, work, work, work, work, work, work, work, d["lower"],
                                    rows, off, H=H, SAVE=False, BCC=fast_kda.BC, PREC="tf32", num_warps=8,
                                    num_stages=1)

    def state():
        fast_kda._kda_state[(H, 2)](work, work, work, work, work, d["state_in"], so, o, rows, off, nc, H=H, BV=64,
                                    PREC="tf32", num_warps=8, num_stages=1)

    def norm():
        fast_kda._kda_norm[(triton.cdiv(rows, 32), H)](o, g, g.stride(0), d["norm_w"], d["eps"], rows, H=H, BR=32,
                                                       num_warps=4)

    prep()
    return _ms(prep, reps), _ms(state, reps), _ms(norm, reps)


def _settings(sweep: bool):
    if not sweep:
        split = [dict(bv=32, warps=4, ksplit=1, maxnreg=168), dict(bv=32, warps=4, ksplit=1, maxnreg=0),
                 dict(bv=16, warps=4, ksplit=1, maxnreg=0), dict(bv=16, warps=4, ksplit=1, maxnreg=128),
                 dict(bv=32, warps=8, ksplit=1, maxnreg=0), dict(bv=64, warps=8, ksplit=0, maxnreg=0)]
        fused = [dict(fbv=64, ksplit=0), dict(fbv=64, ksplit=1), dict(fbv=32, ksplit=1),
                 dict(fbv=64, ksplit=0, ring=2), dict(fbv=64, ksplit=0, lag=2, ring=4),
                 dict(fbv=32, ksplit=1, fmaxnreg=128)]
    else:
        split = [dict(bv=bv, warps=w, ksplit=ks, maxnreg=mr) for bv, w, ks, mr in
                 itertools.product((16, 32, 64), (2, 4, 8), (0, 1), (0, 128, 168))]
        fused = [dict(fbv=fbv, ksplit=ks, lag=lag, ring=ring, ctas=ctas, fmaxnreg=mr)
                 for fbv, ks, (lag, ring), ctas, mr in
                 itertools.product((32, 64), (0, 1), ((1, 2), (1, 3), (2, 3), (2, 4)), (0, 96), (0, 128))]
    return [(1, kw) for kw in split] + [(2, kw) for kw in fused]


def main(argv):
    sweep = "--sweep" in argv
    reps = 20
    out_csv = None
    rows_list = []
    it = iter(argv)
    for x in it:
        if x == "--reps":
            reps = int(next(it))
        elif x == "--csv":
            out_csv = next(it)
        elif not x.startswith("--"):
            rows_list.append(int(x))
    rows_list = rows_list or [512, 1024, 2048, 8192]
    print(f"GPU {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).multi_processor_count} SMs, "
          f"triton {__import__('triton').__version__}")
    table = []
    best = {}
    for rows in rows_list:
        d = _inputs(rows, seed=rows)
        so_ref = torch.empty_like(d["state_in"])
        ref_o = fast_kda.kda_prefill_chunked(*_args(d, rows), None, so_ref, pos=0).clone()
        t_old = _ms(lambda: fast_kda.kda_prefill_chunked(*_args(d, rows), None, torch.empty_like(so_ref), pos=0), reps)
        tp, ts, tn = _parts(d, rows, reps)
        print(f"\n[{rows} rows] fast_kda {t_old:.3f} ms (prep {tp:.3f} + state {ts:.3f} + norm {tn:.3f}) = "
              f"{t_old / rows * LAYERS * 1e3:.1f} us a token over {LAYERS} layers")
        table.append(dict(rows=rows, mode=0, cfg="fast_kda", ms=t_old, speedup=1.0, same=True))
        for mode, kw in _settings(sweep):
            cfg = dict(bv=32, warps=4, maxnreg=168, ksplit=1, ctas=0, lag=1, ring=3, fbv=64, fmaxnreg=0)
            cfg.update(kw)
            so = torch.empty_like(so_ref)
            try:
                o = kda_v2.kda_prefill_v2(*_args(d, rows), None, so, pos=0, v2=mode, cfg=cfg).clone()
                same = torch.equal(o.view(torch.int16), ref_o.view(torch.int16)) and \
                    torch.equal(so.view(torch.int32), so_ref.view(torch.int32))
                t = _ms(lambda: kda_v2.kda_prefill_v2(*_args(d, rows), None, torch.empty_like(so_ref), pos=0,
                                                      v2=mode, cfg=cfg), reps)
            except Exception as exc:  # noqa: BLE001 - a setting that cannot launch (shared memory) is reported
                print(f"  mode {mode} {kw}: FAILED {type(exc).__name__}: {str(exc)[:120]}")
                continue
            tag = ("split " if mode == 1 else "fused ") + " ".join(f"{k}={v}" for k, v in kw.items())
            print(f"  {tag:48s} {t:7.3f} ms  x{t_old / t:4.2f}  bitwise {'EQUAL' if same else 'DIFFERENT'}  "
                  f"{t / rows * LAYERS * 1e3:6.1f} us/token (saves {(t_old - t) / rows * LAYERS * 1e3:5.1f})")
            table.append(dict(rows=rows, mode=mode, cfg=tag, ms=t, speedup=t_old / t, same=same))
            if same and (rows not in best or t < best[rows][0]):
                best[rows] = (t, mode, cfg)
    print()
    for rows, (t, mode, cfg) in best.items():
        env = [f"GLM53_TF_KDA_V2={mode}"]
        keys = ("bv", "warps", "maxnreg", "ksplit") if mode == 1 else ("fbv", "ksplit", "lag", "ring", "ctas",
                                                                        "fmaxnreg")
        names = {"fbv": "FUSED_BV", "fmaxnreg": "FUSED_MAXNREG"}
        env += [f"GLM53_TF_KDA_V2_{names.get(k, k.upper())}={cfg[k]}" for k in keys]
        print(f"best bit-identical at {rows} rows: {t:.3f} ms -> {' '.join(env)}")
    if out_csv:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)


if __name__ == "__main__":
    main(sys.argv[1:])
