"""W12: 0450's sampler at scale (inside the image, one GPU). gpusample.self_check's two halves, split so each runs at
the size it can: (1) libm on the device vs the node's numpy, bit for bit: log of uniforms, log of their -log (the
Gumbel chain) and exp over its range, LIBM_ROWS x 8 values each a seed; (2) keyed draws: gpusample.choose vs the
host's exact_sampling.choose_rows on DRAW_ROWS random rows a seed (2 ranks x 28 candidates, ties, T 0.7, top-k 20,
top-p 0.9; the host side is a per-row Python loop). python sampler.py SEEDS LIBM_ROWS DRAW_ROWS"""
import sys, time
import numpy as np
import torch
from tensorfold.families.glm5_next.cuda import gpusample as gs
from tensorfold.engine.exact_sampling import Sampling, choose_rows

seeds, lrows, drows = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
tot_l = tot_d = bad = 0
t0 = time.time()
for seed in range(seeds):
    rng = np.random.default_rng(seed)
    u = (rng.integers(0, 1 << 53, size=lrows * 8, dtype=np.int64).astype(np.float64) * 2.0 ** -53 + 2.0 ** -54)
    e = -np.abs(rng.standard_normal(lrows * 8) * rng.choice([0.01, 1.0, 30.0, 800.0], size=lrows * 8))
    for which, x, want in (("log", u, np.log(u)), ("log", -np.log(u), np.log(-np.log(u))), ("exp", e, np.exp(e))):
        got = gs.libm(torch.from_numpy(x).to("cuda"), which).cpu().numpy()
        nb = int(np.count_nonzero(got.view(np.int64) != want.view(np.int64)))
        tot_l += len(x); bad += nb
        if nb:
            print(f"seed {seed} {which}: {nb} of {len(x)} differ", flush=True)
    world, k = 2, 28
    vals = (rng.standard_normal((drows, world * k)) * 3).astype(np.float32)
    vals[:, 1::7] = vals[:, ::7][:, :vals[:, 1::7].shape[1]]
    ids = np.stack([rng.permutation(1000)[:world * k] for _ in range(drows)]).astype(np.int64)
    got = np.zeros((world, drows * 2 * k), dtype=np.float32)
    for r in range(world):
        seg = np.concatenate([vals[:, r * k:(r + 1) * k], ids[:, r * k:(r + 1) * k].astype(np.int32).view(np.float32)], axis=1)
        got[r] = seg.reshape(-1)
    gt = torch.from_numpy(got.reshape(-1)).to("cuda")
    s = Sampling(seed=int(rng.integers(1 << 62)), temperature=0.7, top_k=20, top_p=0.9)
    pos = [int(p) for p in rng.integers(0, 1 << 20, size=drows)]
    td = time.time()
    tok, _ = gs.choose(gt, drows * 2 * k, world, [(r * 2 * k, k, pos[r], s, False) for r in range(drows)])
    dev = tok.cpu().tolist()
    tdev = time.time() - td
    host = choose_rows(vals.astype(np.float32), ids, pos, s)
    nd = sum(a != b for a, b in zip(dev, host))
    tot_d += drows; bad += nd
    print(f"seed {seed}: libm {lrows * 24} values, draws {drows} ({nd} differ; device {tdev:.2f}s, total {time.time() - td:.1f}s)"
          f"  elapsed {time.time() - t0:.0f}s", flush=True)
print(f"SAMPLER libm values {tot_l}, keyed draws {tot_d}, differences {bad}: {'OK' if bad == 0 else 'FAIL'}")
