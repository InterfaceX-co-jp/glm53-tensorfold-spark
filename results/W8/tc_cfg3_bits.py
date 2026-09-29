"""W8: patches/0330 cfg 1/2 (544-thread CTAs) cannot launch on GB10 (occupancy 0); check cfg 3 alone, bit for bit vs
fast2 / fat, with the test file's helpers (tests/cuda/test_expert_tc_patches.py)."""
import sys, torch
sys.path.insert(0, "/work/tests/cuda")
import test_expert_tc_patches as T
ex = T._layer()
for rows, kind in ((64, "uniform"), (300, "skewed"), (2048, "uniform"), (2048, "skewed"), (4160, "skewed"), (8192, "uniform")):
    x, picks = T._picks(rows, kind)
    y2, xd2 = T._run(ex, x, picks, "fast2")
    for cfg, ticket, ctas in (((3, 3), True, 0), ((3, 3), False, 0), ((3, 3), True, 7)):
        yt, xdt = T._run(ex, x, picks, "tc", cfg=cfg, ticket=ticket, ctas=ctas)
        print(rows, kind, cfg, ticket, ctas, "Xd same", torch.equal(xdt, xd2), "Y same", torch.equal(yt, y2), flush=True)
for cfg in ((1, 1), (2, 2)):
    try:
        x, picks = T._picks(2048, "uniform"); T._run(ex, x, picks, "tc", cfg=cfg)
        print(cfg, "launched")
    except RuntimeError as e:
        print(cfg, "RuntimeError:", e)
