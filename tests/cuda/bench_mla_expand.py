"""patches/0390 micro-benchmark: latent absorb / expand, v1 (``_absorb`` / ``_expand``) vs v2 (``_absorb2`` /
``_expand2``, GLM53_TF_MLA_EXPAND=v2) on the real per-rank shapes (32 heads, head dim 256, latent 512; kv_b in 4 bits
as production's q4mse), one GPU, ~2-4 minutes. Checks the bits while it is at it.

    PYTHONPATH=/src/TensorFold/tests/cuda python tests/cuda/bench_mla_expand.py [--sweep] [--b16] [rows ...]

Prints, per row count (default 1 8 512 2048 8192):
- us a call for v1 and v2 (the default tile table), achieved TFLOP/s (2 R N L each; 4.3 GFLOP a 512-row call), the
  speed-up, and whether v2's bits equal v1's (a False: do not enable the knob);
- with --sweep: every candidate tile (rows a program 16 / 32 / 64 / 128, k a step 16 / 32, warps 4 / 8, expand's BN
  32 / 64 / 128), the best per kernel and row count, and the ``V2_TILES`` table that would pick them. The table only
  changes speed: paste the printed one into ``latent.V2_TILES`` (or keep the offline default).
- the projected saving a token of prefill (11 DSA layers: 2 calls a layer a 512-row sub-block), against W7's
  2,415 us (``_expand``) + 676 us (``_absorb``) a sub-block.

Roofline (GB10): 30.7 TFLOP/s fp32 FMA -> 140 us a 512-row call at the roof; DRAM: u is 32 MB fp32 a 512-row call,
read once = 140 us at 230 GB/s (L2-resident when produced just before, as in the model).
"""

from __future__ import annotations

import argparse

import torch
import triton

from tensorfold.families.glm5_next.cuda import latent, qmm

H, D, L = 32, 256, 512
N = H * D


def _us(fn, reps=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1000


def _kv(b16: bool, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = (torch.randn((N, L), generator=g) * 0.05).to(torch.bfloat16).cuda()
    return qmm.make_b16(w) if b16 else qmm.quantize4(w, mse=True)


def _launch(kind, R, x, kv, out, bm, bk, warps, bn=64):
    W, S, B, is_q4 = latent._parts(kv)
    if kind == "expand":
        latent._expand2[(N // bn, triton.cdiv(R, bm))](x, W, S, B, out, R, H=H, DV=D, L=L, N=N, IS_Q4=is_q4, BMR=bm,
                                                       BN=bn, BK=bk, num_warps=warps, num_stages=1)
    else:
        latent._absorb2[(H * (L // 64), triton.cdiv(R, bm))](x, W, S, B, out, R, H=H, DQ=D, L=L, N=N, IS_Q4=is_q4,
                                                             BMR=bm, BK=bk, num_warps=warps, num_stages=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="*", type=int, default=[1, 8, 512, 2048, 8192])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--b16", action="store_true", help="BF16 kv_b (an EXL3 checkpoint without q4) instead of q4mse")
    a = ap.parse_args()
    print(f"[bench_mla_expand] {torch.cuda.get_device_name()}, triton {triton.__version__}, kv_b "
          f"{'bf16' if a.b16 else 'q4mse'}", flush=True)
    kv_k, kv_v = _kv(a.b16, 1), _kv(a.b16, 2)
    best: dict[tuple[str, int], tuple[float, tuple]] = {}
    per512 = {}
    for R in a.rows:
        g = torch.Generator(device="cpu").manual_seed(R)
        q = torch.randn((R, H, D), generator=g).to(torch.bfloat16).cuda()
        u = (torch.randn((R, H, L), generator=g) * 0.3).cuda()
        outs = {}
        for kind, x, kv, shape in (("absorb", q, kv_k, (R, H, L)), ("expand", u, kv_v, (R, N))):
            fn = getattr(latent, kind)
            o1 = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
            o2 = torch.empty_like(o1)
            latent.EXPAND_V2 = False
            t1 = _us(lambda: fn(x, kv, o1))
            latent.EXPAND_V2 = True
            t2 = _us(lambda: fn(x, kv, o2))
            latent.EXPAND_V2 = False
            same = torch.equal(o1.view(torch.int16), o2.view(torch.int16))
            fl = 2 * R * N * L
            print(f"  R {R:5d} {kind:6s}: v1 {t1:9.1f} us ({fl / t1 / 1e6:5.2f} TFLOP/s)   v2 {t2:9.1f} us "
                  f"({fl / t2 / 1e6:5.2f} TFLOP/s)   x{t1 / t2:5.2f}   same bits {same}   tile {latent.v2_tiles(R, kind)}",
                  flush=True)
            outs[kind] = (t1, t2)
            if a.sweep:
                for bm in (16, 32, 64, 128):
                    if bm > 16 and bm // 2 >= R:
                        continue
                    for bk in (16, 32):
                        for warps in (4, 8):
                            for bn in ((32, 64, 128) if kind == "expand" else (64,)):
                                o3 = torch.empty_like(o1)
                                try:
                                    t = _us(lambda: _launch(kind, R, x, kv, o3, bm, bk, warps, bn), reps=10)
                                except Exception as exc:  # noqa: BLE001 - out of resources etc.
                                    print(f"      {kind} bm {bm} bk {bk} w {warps} bn {bn}: {type(exc).__name__}")
                                    continue
                                ok = torch.equal(o1.view(torch.int16), o3.view(torch.int16))
                                print(f"      {kind} bm {bm:3d} bk {bk} w {warps} bn {bn:3d}: {t:9.1f} us  same {ok}",
                                      flush=True)
                                if ok and ((kind, R) not in best or t < best[(kind, R)][0]):
                                    best[(kind, R)] = (t, (bm, bk, warps, bn))
        if R == 512 or (512 not in a.rows and R >= 512 and not per512):
            per512 = {k: (v[0] * 512 / R, v[1] * 512 / R) for k, v in outs.items()}
    if per512:
        v1 = sum(v[0] for v in per512.values())
        v2 = sum(v[1] for v in per512.values())
        print(f"per 512-row sub-block and layer: v1 {v1:.0f} us, v2 {v2:.0f} us (W7 trace: 2,415 + 676 = 3,091 us); "
              f"11 DSA layers: {11 * (v1 - v2) / 512:.1f} us a token saved (prefill ~680 us a token: "
              f"{100 * 11 * (v1 - v2) / 512 / 680:.1f}%)")
    if best:
        print("best tiles (us, (rows a program, k a step, warps, expand BN)):")
        for (kind, R), (t, cfg) in sorted(best.items()):
            print(f"  {kind:6s} R {R:5d}: {t:9.1f} us {cfg}")


if __name__ == "__main__":
    main()
