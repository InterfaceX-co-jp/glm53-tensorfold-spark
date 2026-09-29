#!/usr/bin/env python3
"""W11: the production decode kernels on one GPU with random weights of the real per-rank shapes (measurement only).

    PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda python kbench.py [--only expert|dense] [--ncu]

- Routed experts (``exl3_mm.routed``'s decode path: rot_in, grouped gate/up, gateup_epilogue, grouped down,
  down_epilogue) for R rows with exactly U distinct experts (the engine's group layout: ids / members padded to
  min(8R, 288) + 1 entries, so grid.x = 8R + 1 as in production), alternating two random layers (3.6 GB, nothing
  L2-resident). Per step: median us and GB/s of the trellis bytes (U x 4.19 MB gate+up, U x 2.10 MB down).
- Dense q4 (``qmm.matmul``: ``_qmm`` [+ ``_reduce``]) for every decode shape at M = 1 / 4 / 8 / 16 rows: us, GB/s
  of n x k x 0.5625 bytes, and the launch grid (to name the trace's kernels by grid).
- ``--ncu``: a few calls a configuration only (for ncu --launch-skip / --launch-count), no timing table.
"""
import argparse
import json
import statistics
import sys
from types import SimpleNamespace

import numpy as np
import torch

D, NI, E, TOP = 4096, 1024, 288, 8
SLOTS = TOP + 1
MAT = D * NI // 2                          # one 4-bit matrix, bytes


def ms_of(fn, reps=30, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        t.append(a.elapsed_time(b))
    return statistics.median(t)


def group_for(R, U, rng):
    """picks [R, 9] with exactly U distinct routed experts (each row 8 distinct), slot 8 = shared id E; the engine's
    group arrays (forward.Weights.group: maxu = min(8R, 288) + 1 rows, members [maxu, R])."""
    ex = np.sort(rng.choice(E, size=U, replace=False))
    picks = np.full((R, SLOTS), E, dtype=np.int32)
    for r in range(R):
        for j in range(TOP):
            picks[r, j] = ex[(r * TOP + j) % U]
    maxu = min(R * TOP, E) + 1
    used = sorted(set(picks.flatten().tolist()))          # includes E (the shared id) last
    ids = np.zeros(maxu, dtype=np.int32)
    members = np.full((maxu, R), -1, dtype=np.int32)
    for u, e in enumerate(used):
        m = [r * 32 + s for r in range(R) for s in range(SLOTS) if picks[r, s] == e]
        ids[u] = e
        members[u, :len(m)] = m[:R]
    g = SimpleNamespace(ids=torch.from_numpy(ids).cuda(), count=torch.tensor([len(used)], dtype=torch.int32).cuda(),
                        members=torch.from_numpy(members).cuda())
    return torch.from_numpy(picks).cuda(), g


def _exl3_layer(D, NI, E, seed):
    """tests/cuda/test_patches.py's random EXL3 layer (copied: that module imports pytest)."""
    from tensorfold.families.glm5_next.cuda import exl3_mm

    rng = np.random.default_rng(seed)

    def trellis(k, n):
        return torch.from_numpy(rng.integers(-2**15, 2**15, size=(k // 16, n // 16, 64)).astype(np.int16))

    def scales(n, sc):
        return torch.from_numpy((rng.standard_normal(n) * sc).astype(np.float16))

    ex = [(trellis(D, NI), trellis(D, NI), trellis(NI, D), scales(D, 0.02), scales(D, 0.02), scales(NI, 0.5),
           scales(NI, 0.5), scales(NI, 0.05), scales(D, 0.2)) for _ in range(E)]
    cols = list(zip(*ex))
    wds = lambda ts: torch.stack([exl3_mm.words(t) for t in ts]).cuda()        # noqa: E731
    hs = lambda ts: torch.stack(list(ts)).cuda()                               # noqa: E731
    return exl3_mm.Exl3Experts(wds(cols[0]), wds(cols[1]), wds(cols[2]), hs(cols[3]), hs(cols[4]), hs(cols[5]),
                               hs(cols[6]), hs(cols[7]), hs(cols[8]), E, NI, D)


def experts(args, out):
    from tensorfold.families.glm5_next.cuda import exl3_mm

    ext = exl3_mm._ext()
    layers = [_exl3_layer(D, NI, E, seed=5 + i) for i in range(2)]
    for ex in layers:
        ex.suh_u = ex.suh_g
    rng = np.random.default_rng(3)
    cases = [(1, 8), (2, 12), (3, 17), (4, 21), (5, 24), (8, 35), (8, 40), (12, 54), (16, 65), (16, 104)]
    if args.ncu:
        cases = [(1, 8), (3, 17), (8, 40), (16, 65)]
    print("R   U   rot_in  gate/up us  GB/s   gu_epi  down us  GB/s   dn_epi  total us  GB/s(all trellis)", flush=True)
    for R, U in cases:
        picks, g = group_for(R, U, rng)
        x = (torch.randn((R, D), device="cuda") * 0.5).to(torch.bfloat16)
        s = exl3_mm.Scratch(R, SLOTS, D, NI, "cuda")
        y = torch.zeros((R * SLOTS, D), dtype=torch.float32, device="cuda")
        P = s.rows * s.slots
        st = {"i": 0}

        def ex_():
            st["i"] ^= 1
            return layers[st["i"]]

        def rot():
            e = ex_()
            ext.rot_in(x, x.stride(0), picks, e.suh_g, e.suh_u, s.xg, s.xu, R, D, SLOTS)

        def gu():
            e = ex_()
            nt, w, sk = exl3_mm.GATEUP_CFG
            ext.grouped(s.xg, s.xu, e.gt, e.ut, g.ids, g.count, g.members, s.z, 2, D, NI, P, sk, SLOTS, nt, w)

        def gepi():
            e = ex_()
            ext.gateup_epilogue(s.z, picks, e.svh_g, e.svh_u, e.suh_d, s.xd, R, P, NI, exl3_mm.GATEUP_CFG[2], SLOTS, 10.0)

        def dn():
            e = ex_()
            nt, w, sk = exl3_mm.DOWN_CFG
            ext.grouped(s.xd, s.xd, e.dt, e.dt, g.ids, g.count, g.members, s.z, 1, NI, D, P, sk, SLOTS, nt, w)

        def depi():
            e = ex_()
            ext.down_epilogue(s.z, picks, e.svh_d, y, R, P, D, exl3_mm.DOWN_CFG[2], SLOTS)

        def full():
            e = ex_()
            exl3_mm.routed(x, picks, g, e, s, y, R, 10.0, fast=False)

        if args.ncu:
            for _ in range(3):
                full()
            torch.cuda.synchronize()
            continue
        t = {k: ms_of(f) * 1e3 for k, f in (("rot", rot), ("gu", gu), ("gepi", gepi), ("dn", dn), ("depi", depi),
                                             ("full", full))}
        rec = dict(kind="expert", R=R, U=U, grid_x=8 * R + 1 if 8 * R < E else E + 1, us=t,
                   gu_gbs=2 * U * MAT / t["gu"] / 1e3, dn_gbs=U * MAT / t["dn"] / 1e3,
                   full_gbs=3 * U * MAT / t["full"] / 1e3)
        out.append(rec)
        print(f"{R:<3} {U:<3} {t['rot']:7.1f} {t['gu']:10.1f} {rec['gu_gbs']:6.1f} {t['gepi']:7.1f} {t['dn']:8.1f} "
              f"{rec['dn_gbs']:6.1f} {t['depi']:7.1f} {t['full']:9.1f} {rec['full_gbs']:6.1f}", flush=True)
        print("W11KB " + json.dumps(rec), flush=True)


SHAPES = [("head", 77440, 4096), ("kda.proj", 12576, 4096), ("mlp.gu", 12288, 4096), ("dsa.o/mtp.eh", 4096, 8192),
          ("mlp.down", 4096, 6144), ("kda.o", 4096, 4096), ("dsa.q_b", 8192, 1536), ("shared.gu/dsa.proj", 2048, 4096),
          ("dsa.index.qb", 4096, 1536), ("shared.down", 4096, 1024), ("dsa.kv_k/v", 8192, 512),
          ("index.kw", 160, 4096), ("kda.fb/gb", 4096, 128)]


def dense(args, out):
    from tensorfold.families.glm5_next.cuda import qmm

    rows = [1, 4, 8, 16] if not args.ncu else [1, 8]
    print("shape                     n      k      M   grid          us     GB/s", flush=True)
    for name, n, k in SHAPES:
        qs = []
        for i in range(3):                  # rotate 3 copies (the head is 178 MB; small shapes stay L2-cold-ish)
            w = torch.randint(-2**31, 2**31 - 1, (n // 64, k // 64, 64, 8), dtype=torch.int32, device="cuda")
            sc = (torch.rand((k // 64, n), device="cuda") * 0.01).to(torch.bfloat16)
            bi = (torch.randn((k // 64, n), device="cuda") * 0.01).to(torch.bfloat16)
            qs.append(qmm.Q4(w, sc, bi, n, k))
        by = n * k * 9 // 16
        for M in rows:
            x = (torch.randn((M, k), device="cuda")).to(torch.bfloat16)
            xs = qmm.group_sums(x)
            o = torch.empty((M, n), dtype=torch.bfloat16, device="cuda")
            part = torch.empty((8 * M * n,), dtype=torch.float32, device="cuda")
            st = {"i": 0}

            def f():
                st["i"] = (st["i"] + 1) % 3
                qmm.matmul(x, qs[st["i"]], xs, out=o, part=part)

            if args.ncu:
                for _ in range(3):
                    f()
                torch.cuda.synchronize()
                continue
            us = ms_of(f, reps=50) * 1e3
            sk = qmm.split_k(n, k)
            grid = (-(-M // qmm.bucket(M)), -(-n // qmm.BN), sk)
            rec = dict(kind="dense", name=name, n=n, k=k, M=M, grid=grid, us=us, gbs=by / us / 1e3)
            out.append(rec)
            print(f"{name:22s} {n:6d} {k:6d} {M:4d}   {str(grid):12s} {us:8.1f} {rec['gbs']:7.1f}", flush=True)
            print("W11KB " + json.dumps(rec), flush=True)
        del qs
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="expert,dense")
    p.add_argument("--ncu", action="store_true")
    p.add_argument("--out")
    args = p.parse_args()
    torch.cuda.init()
    out = []
    if "expert" in args.only:
        experts(args, out)
    if "dense" in args.only:
        dense(args, out)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


main()
