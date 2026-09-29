"""patches/0520 microbench (THEORY-2 item 3 / session step 1d): one hyper-connection boundary -- glue.hc_post then
glue.hc_pre, i.e. the three Triton kernels _hc_post, _hc_partial, _hc_finish -- against hc_fused.cu's one launch.

In the server's state (THEORY-2 section 7 rule 1): each boundary is captured in a CUDA graph right after a ~50 MB
streaming kernel (plain loads, so it evicts the 24 MB L2 and leaves a realistic predecessor tail): graph A = N x
(pred, boundary), graph P = N x pred, replayed interleaved; us a boundary = median over reps of (A - P) / N. The
boundary's weights (fn: 768 KB bf16, the norm weight) and streams come from DRAM, as at a real layer boundary. A
``hot`` row (N x boundary, no predecessor) is printed next to it. Every row first checks the fused kernel's bits
against the Triton kernels' (every output), and the timing is only reported for configurations with the same bits.

Rows: windows of 1-16 rows (decode / verify / MTP; the gated range) and 32 / 64 (batched rounds). Row-group sizes
GLM53_TF_HC_CUDA_RG = 1, 2, 4, 8 (placement only) are all timed, each also as a programmatic dependent
(GLM53_TF_HC_CUDA_PDL=1; its fn / norm-weight loads overlap the predecessor's tail); the best is reported and the
knob values to set are printed.

    GATE item3: fused <= 50% of the three Triton kernels (cold, every row count 1-16): PASS / FAIL

plus the estimated saving a 1-stream round (89 boundaries a forward; the W11 trace's ~90 hc boundaries a round at
~15.6 us) and whether raising GLM53_TF_HC_CUDA_ROWS past 16 pays (the 32 / 64-row rows).

    PYTHONPATH=/src/TensorFold/src:/src/TensorFold/tests/cuda:/work/tests/cuda \\
        python tests/cuda/bench_hc_fused.py --json out.json          # ~2-4 min (the first run compiles the extension)
    ... --quick                                                       # rows 1, 4, 16; rg 4 only; ~1 min

Lock the SM clock at the production cap first (nvidia-smi -lgc 2250,2250); one GPU, nothing else on it.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

try:
    import triton
    import triton.language as tl
except ImportError:                      # pragma: no cover
    triton = tl = None

ROWS = (1, 2, 3, 4, 6, 8, 12, 16, 32, 64)
GATE_ROWS = tuple(range(1, 17))
GATE_RATIO = 0.50
RGS = (1, 2, 4, 8)
BOUNDARIES = 90                          # hc boundaries a 1-stream round in the W11 trace (THEORY-2 section 1.5)
D, WIDE = 4096, 16384


def stats(xs) -> dict:
    v = sorted(float(x) for x in xs)
    if not v:
        return {}
    return {"median": statistics.median(v), "min": v[0], "max": v[-1], "n": len(v)}


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def gate(rows: dict, gate_rows=GATE_ROWS, ratio=GATE_RATIO) -> dict:
    """rows: {R: {"triton_us", "fused_us", "same_bits"}} (cold). PASS when every gated row count that was timed has
    same bits and fused / triton <= ratio."""

    got = {R: r for R, r in rows.items() if R in gate_rows and r.get("triton_us") and r.get("fused_us")}
    if not got:
        return {"pass": False, "why": "no gated rows timed"}
    rat = {R: r["fused_us"] / r["triton_us"] for R, r in got.items()}
    bits = all(r.get("same_bits") for r in got.values())
    worst = max(rat, key=rat.get)
    return {"pass": bits and all(v <= ratio for v in rat.values()), "same_bits": bits, "ratios": rat,
            "worst_rows": worst, "worst": rat[worst], "geomean": geomean(list(rat.values()))}


_PRED = None


def pred_kernel():
    global _PRED
    if _PRED is None:
        @triton.jit
        def stream_k(P, OUT, n, BLOCK: tl.constexpr, ITERS: tl.constexpr):
            pid = tl.program_id(0)
            acc = tl.zeros((BLOCK,), dtype=tl.int32)
            for i in range(ITERS):
                off = (pid * ITERS + i) * BLOCK + tl.arange(0, BLOCK)
                acc += tl.load(P + off, mask=off < n, other=0)
            tl.store(OUT + pid, tl.sum(acc, axis=0))

        _PRED = stream_k
    return _PRED


class Bench:
    def __init__(self, pred_mb: int):
        import torch

        from tensorfold.families.glm5_next.cuda import glue, hc_cuda

        self.torch, self.glue, self.hc = torch, glue, hc_cuda
        self.dev = "cuda"
        n = pred_mb * 1024 * 1024 // 4
        self.pbuf = torch.randint(0, 1 << 20, (n,), dtype=torch.int32, device=self.dev)
        self.pn, self.PB, self.PI = n, 2048, 8
        self.pgrid = -(-n // (self.PB * self.PI))
        self.pout = torch.zeros(self.pgrid, dtype=torch.int32, device=self.dev)

    def pred(self):
        pred_kernel()[(self.pgrid,)](self.pbuf, self.pout, self.pn, BLOCK=self.PB, ITERS=self.PI, num_warps=8)

    def inputs(self, R: int, seed: int = 0) -> dict:
        torch = self.torch
        g = torch.Generator(device="cpu").manual_seed(seed)
        bf, f32 = torch.bfloat16, torch.float32
        x = torch.randn((R, WIDE), generator=g)
        x[:, torch.randperm(WIDE, generator=g)[:40]] *= 40
        m = torch.rand((R, 4, 4), generator=g) + 0.05
        for _ in range(5):
            m = m / m.sum(dim=2, keepdim=True)
            m = m / m.sum(dim=1, keepdim=True)
        d = {"x": x.to(bf), "g": torch.randn((2, R, D), generator=g) * 0.3,
             "post": 2 * torch.sigmoid(torch.randn((R, 4), generator=g)), "comb": m.reshape(R, 16),
             "fn": (torch.randn((24, WIDE), generator=g) * 0.03).to(bf), "base": torch.randn(24, generator=g),
             "scale": 1 + 0.1 * torch.randn(3, generator=g), "nw": (1 + 0.2 * torch.randn(D, generator=g)).to(bf)}
        d = {k: v.to(self.dev).contiguous() for k, v in d.items()}
        d.update(out=torch.zeros((R, D), dtype=bf, device=self.dev), xs=torch.zeros((R, 64), dtype=f32, device=self.dev),
                 part=torch.zeros((R, 16, 32), dtype=f32, device=self.dev))
        return d

    def triton_fn(self, d):
        glue = self.glue

        def f():
            glue.hc_post(d["x"], d["x"], d["g"], d["post"], d["comb"])
            glue.hc_pre(d["x"], d["fn"], d["base"], d["scale"], d["nw"], d["out"], d["xs"], d["post"], d["comb"],
                        d["part"], 1e-5, 1e-6, 20)
        return f

    def fused_fn(self, d, rg: int, pdl: bool = False):
        hc = self.hc

        def f():
            hc.launch(d["x"], d["g"], d["post"], d["comb"], d["fn"], d["base"], d["scale"], d["nw"], d["out"],
                      d["xs"], d["part"], 1e-5, 1e-6, 20, rg=rg, pdl=pdl)
        return f

    def same_bits(self, R: int, rg: int, pdl: bool = False) -> bool:
        torch = self.torch
        a, b = self.inputs(R, 1), self.inputs(R, 1)
        self.triton_fn(a)()
        self.fused_fn(b, rg, pdl)()
        torch.cuda.synchronize()
        for k in ("x", "out", "xs", "post", "comb"):
            u, v = a[k], b[k]
            vu = u.view(torch.int16) if u.dtype == torch.bfloat16 else u.view(torch.int32)
            vv = v.view(torch.int16) if v.dtype == torch.bfloat16 else v.view(torch.int32)
            if not torch.equal(vu, vv):
                return False
        return torch.equal(a["part"][:, :, :25].view(torch.int32), b["part"][:, :, :25].view(torch.int32))

    def graph(self, fns):
        torch = self.torch
        for f in fns:
            f()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for f in fns:
                f()
        g.replay()
        torch.cuda.synchronize()
        return g

    def time_graph(self, g) -> float:
        torch = self.torch
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) * 1e3

    def cold(self, fns: dict, calls: int, reps: int) -> dict:
        """{name: us a boundary}: N x (pred, fn) vs N x pred, interleaved replays."""

        gp = self.graph([self.pred] * calls)
        graphs = {}
        for name, fn in fns.items():
            seq = []
            for _ in range(calls):
                seq += [self.pred, fn]
            graphs[name] = self.graph(seq)
        t = {name: [] for name in fns}
        for _ in range(reps):
            for name, g in graphs.items():
                base = self.time_graph(gp)                     # the predecessors alone, right before
                t[name].append((self.time_graph(g) - base) / calls)
        return {name: statistics.median(v) for name, v in t.items()}

    def hot(self, fns: dict, calls: int, reps: int) -> dict:
        out = {}
        for name, fn in fns.items():
            g = self.graph([fn] * calls)
            out[name] = statistics.median(self.time_graph(g) for _ in range(reps)) / calls
        return out


def clocks() -> str:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,clocks.applications.graphics",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception as exc:                                # noqa: BLE001
        return f"? ({exc})"


def main() -> int:
    ap = argparse.ArgumentParser(description="hc boundary: 3 Triton kernels vs hc_fused.cu (THEORY-2 item 3)")
    ap.add_argument("--rows", type=int, nargs="+", default=None, help=f"row counts (default {ROWS})")
    ap.add_argument("--rg", type=int, nargs="+", default=None, help=f"row-group sizes (default {RGS})")
    ap.add_argument("--calls", type=int, default=12, help="boundaries a graph")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--pred-mb", type=int, default=50)
    ap.add_argument("--no-hot", action="store_true")
    ap.add_argument("--no-pdl", action="store_true", help="skip the GLM53_TF_HC_CUDA_PDL=1 rows")
    ap.add_argument("--quick", action="store_true", help="rows 1, 4, 16; rg 4")
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = tuple(a.rows) if a.rows else ((1, 4, 16) if a.quick else ROWS)
    rgs = tuple(a.rg) if a.rg else ((4,) if a.quick else RGS)
    pdls = (False,)

    def log(s):
        print(s, flush=True)

    res = {"argv": sys.argv, "errors": [], "rows": {}}
    try:
        import torch

        B = Bench(a.pred_mb)
        B.hc.configure(True, rows=64)
        res.update(device=torch.cuda.get_device_name(0), torch=torch.__version__,
                   triton=getattr(triton, "__version__", None), clocks=clocks())
        why = B.hc.self_check(torch.device("cuda"))
        res["self_check"] = why or "ok"
        if B.hc.pdl_supported() and not a.no_pdl:
            pdls = (False, True)
        res["pdl"] = pdls
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        res["errors"].append(f"setup: {type(e).__name__}: {e}")
        if a.json:
            Path(a.json).write_text(json.dumps(res, indent=1, default=str))
        print("GATE item3 hc_fused: ERROR (setup)")
        return 1
    log(f"[hc] {res['device']} triton={res['triton']} clocks(sm,max,app)={res['clocks']} self_check={res['self_check']}"
        f" rows={rows} rg={rgs} pred={a.pred_mb} MB")
    t0 = time.perf_counter()
    for R in rows:
        row = {"rows": R}
        try:
            d = B.inputs(R)
            fns = {"triton": B.triton_fn(d)}
            bits = {}
            for rg in rgs:
                if rg > 1 and rg > R and R <= 8:
                    continue                                   # the same launch as rg = R
                for pdl in pdls:
                    name = f"fused rg{rg}" + (" pdl" if pdl else "")
                    bits[name] = B.same_bits(R, rg, pdl)
                    if bits[name]:
                        fns[name] = B.fused_fn(d, rg, pdl)
            row["same_bits_by_rg"] = bits
            row["same_bits"] = bool(bits) and all(bits.values())
            cold = B.cold(fns, a.calls, a.reps)
            row["cold_us"] = cold
            if not a.no_hot:
                row["hot_us"] = B.hot(fns, a.calls, a.reps)
            fused = {k: v for k, v in cold.items() if k.startswith("fused")}
            row["triton_us"] = cold["triton"]
            if fused:
                best = min(fused, key=fused.get)
                row["best"] = best
                row["fused_us"] = fused[best]
                row["ratio"] = fused[best] / cold["triton"]
            hot = row.get("hot_us", {})
            log(f"[hc] R={R:3d}: triton {cold['triton']:7.2f} us cold"
                + (f" ({hot.get('triton', 0):6.2f} hot)" if hot else "")
                + "".join(f" | {k} {v:7.2f}" + (f" ({hot.get(k, 0):6.2f})" if hot else "") for k, v in fused.items())
                + (f" | best {row['best']}: {row['ratio']:.2f}x" if fused else " | NO FUSED ROW WITH SAME BITS")
                + f" | same bits {row['same_bits']} {bits}")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            res["errors"].append(f"R={R}: {type(e).__name__}: {e}")
            log(f"[hc] R={R}: ERROR {type(e).__name__}: {e}")
        res["rows"][R] = row
    g = gate(res["rows"])
    res["gate"] = g
    res["seconds"] = time.perf_counter() - t0
    typ = [R for R in (1, 4, 8) if R in res["rows"] and res["rows"][R].get("fused_us")]
    save = {R: BOUNDARIES * (res["rows"][R]["triton_us"] - res["rows"][R]["fused_us"]) / 1e3 for R in typ}
    res["saving_ms_a_round"] = save
    big = {R: res["rows"][R].get("ratio") for R in (32, 64) if R in res["rows"]}
    bests = [res["rows"][R].get("best") for R in GATE_ROWS if R in res["rows"] and res["rows"][R].get("best")]
    if bests:
        pick = max(set(bests), key=bests.count)
        rg_best = int(pick.split("rg")[1].split()[0])
        res["recommend"] = {"GLM53_TF_HC_CUDA_RG": rg_best, "GLM53_TF_HC_CUDA_PDL": int(pick.endswith("pdl"))}
        log(f"[hc] most often best at 1-16 rows: {pick} -> GLM53_TF_HC_CUDA_RG={rg_best} "
            f"GLM53_TF_HC_CUDA_PDL={int(pick.endswith('pdl'))} (placement / timing only)")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str))
    if "ratios" in g:
        log(f"GATE item3 hc_fused: fused <= {GATE_RATIO:.0%} of the 3 Triton kernels (cold, rows 1-16): "
            f"{'PASS' if g['pass'] else 'FAIL'} (worst R={g['worst_rows']} {g['worst']:.2f}x, geomean "
            f"{g['geomean']:.2f}x, same bits {g['same_bits']}); est. saving a 1-stream round "
            + ", ".join(f"{v:.2f} ms at R={R}" for R, v in save.items())
            + (f"; 32/64 rows ratio {big} (raise GLM53_TF_HC_CUDA_ROWS if < 1)" if big else ""))
    else:
        log(f"GATE item3 hc_fused: FAIL ({g.get('why')})")
    return 0 if g.get("pass") else 2


if __name__ == "__main__":
    sys.exit(main())
