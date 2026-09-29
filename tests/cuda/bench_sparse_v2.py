"""patches/0410 micro-benchmark: sparse latent attention v2 (``sparse_v2``, GLM53_TF_SPARSE_V2=1) against the one-pass
kernel it replaces (``b12x_attn._lsparse_one``, b12x bit 4, production since W9) and the chunked kernel (bit 4 off),
on the real per-rank shapes (32 local heads, 512-wide latent, 2,051 selected tokens a row), one GPU, a few minutes.

    PYTHONPATH=<tree>/src timeout 1800 python tests/cuda/bench_sparse_v2.py [rows ...] [--ctx 10700,85800] [--bf16]
                                                                         [--random] [--reps N]

Per context (latent rows in the cache: 10,700 = the mean context of the 24.5k prompt, 85,800 = the 98k prompt, whose
FP8 cache (45 MB a layer) no longer fits L2) and row count (default 512 2048 8192):

- token lists like the prefill's: every row 2,051 tokens, neighbouring rows sharing ~97% of theirs (``--random``:
  independent lists, the worst case for L2);
- ms a call, us a row, TFLOP/s (32 x 2,051 x 512 x 4 FLOP a row), and the speedup of v2 over the one-pass kernel;
- ``same bits``: v2 == the one-pass kernel for EVERY row (torch.equal on the fp32 bits). Must say True on every line;
  a False means keep GLM53_TF_SPARSE_V2 off and run tests/cuda/test_sparse_v2_patches.py;
- the projected saving a prefill token: 11 DSA layers x (one pass - v2) a row, and what it is of today's ~638 us a
  token (1,566 tok/s at 24.5k with b12x bit 4, W9).
"""

from __future__ import annotations

import subprocess
import sys

import torch
import triton

H, L, W = 32, 512, 2051
SCALE = 256 ** -0.5
US_TOKEN = 1e6 / 1566.0          # W9 production prefill at 24.5k (b12x bit 4)
LAYERS = 11


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


def _clock():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,power.draw", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception:                                                     # noqa: BLE001
        return "?"


def _tokens(rows, P, random, seed):
    """[rows, W] sorted token lists: independent (``random``) or like the prefill's top-k, each row replacing ~3% of
    its neighbour's tokens."""

    g = torch.Generator().manual_seed(seed)
    if random:
        return torch.stack([torch.randperm(P, generator=g)[:W].sort().values for _ in range(rows)]).int()
    cur = torch.randperm(P, generator=g)[:W]
    out = torch.empty(rows, W, dtype=torch.int32)
    k = max(1, W // 32)
    for r in range(rows):
        cand = torch.unique(torch.randint(0, P, (4 * k,), generator=g))
        new = cand[~torch.isin(cand, cur)][:k]
        cur[torch.randperm(W, generator=g)[:new.numel()]] = new
        out[r] = cur.sort().values.int()
    return out


def bench(P, rows_list, bf16, random, reps):
    from tensorfold.families.glm5_next.cuda import b12x_attn, latent, sparse_v2

    g = torch.Generator().manual_seed(P)
    gain = torch.exp(torch.randn(L, generator=g) * 0.5)
    lat = (torch.randn(P, L, generator=g) * gain).bfloat16().cuda()
    lc = lat if bf16 else latent.quantize_rows_reference(lat)
    fmt = "bf16" if bf16 else "fp8"
    cfgs = [(2, 0, 0)] if bf16 else [(3, 1, 1), (2, 1, 1), (2, 1, 0), (2, 0, 0)]
    for rows in rows_list:
        qa = (torch.randn(rows, H, L, generator=g) * 0.3).bfloat16().cuda()
        tok = _tokens(rows, P, random, rows + P).cuda()
        cnt = torch.full((rows,), W, dtype=torch.int32, device="cuda")
        flop = rows * H * W * L * 4
        o_one = torch.empty(rows, H, L, device="cuda")
        o_chk = torch.empty(rows, H, L, device="cuda")
        o_v2 = torch.empty(rows, H, L, device="cuda")
        t_chk = _ms(lambda: latent.sparse_latent(qa, lc, tok, cnt, o_chk, SCALE, bm=latent.FAST_BM), reps)
        t_one = _ms(lambda: b12x_attn.sparse_latent_one(qa, lc, tok, cnt, o_one, SCALE, v2=False), reps)
        line = (f"[{fmt} ctx {P:6d} rows {rows:5d}{' random' if random else ''}] chunked {t_chk:7.2f} ms | one pass "
                f"{t_one:7.2f} ms ({flop / t_one / 1e9:5.1f} TF/s)")
        best = None
        for st, qkl, qreg in cfgs:
            o_v2.fill_(7.0)
            t = _ms(lambda: sparse_v2.sparse_latent_v2(qa, lc, tok, cnt, o_v2, SCALE, stages=st, qkl=qkl, qreg=qreg),
                    reps)
            same = torch.equal(o_v2.view(torch.int32), o_one.view(torch.int32))
            line += (f" | v2 s{st}q{qkl}r{qreg} {t:7.2f} ms ({flop / t / 1e9:5.1f} TF/s, {t_one / t:4.2f}x, same bits "
                     f"{same})")
            if best is None or t < best[0]:
                best = (t, f"{st},{qkl},{qreg}")
        print(line, flush=True)
        d = (t_one - best[0]) / rows * 1e3 * LAYERS                   # us a prefill token (11 DSA layers)
        print(f"          best v2 (GLM53_TF_SPARSE_V2_CFG={best[1]}): {best[0] * 1e3 / rows:6.2f} us/row/layer "
              f"(one pass {t_one * 1e3 / rows:6.2f}); saves {d:5.1f} us a token = {d / US_TOKEN * 100:4.1f}% of "
              f"{US_TOKEN:.0f} us -> ~{1e6 / (US_TOKEN - d):6.0f} tok/s at 24.5k (from 1,566)", flush=True)


def main():
    if not torch.cuda.is_available():
        print("needs a GPU")
        return
    argv = sys.argv[1:]
    rows_list = [int(a) for a in argv if a.isdigit()] or [512, 2048, 8192]
    ctx = [10700, 85800]
    reps = 10
    for i, a in enumerate(argv):
        if a == "--ctx":
            ctx = [int(x) for x in argv[i + 1].split(",")]
        if a == "--reps":
            reps = int(argv[i + 1])
    print(f"[bench_sparse_v2] {torch.cuda.get_device_name()}, triton {triton.__version__}, clocks {_clock()}")
    for P in ctx:
        bench(P, rows_list, "--bf16" in argv, "--random" in argv, reps)
    print(f"[bench_sparse_v2] clocks after {_clock()}")


if __name__ == "__main__":
    main()
