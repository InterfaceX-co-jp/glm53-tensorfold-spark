#!/usr/bin/env python3
"""THEORY-2 item 2 / session step 1a: how long does the GPU sit idle when a big CUDA graph is launched?

Measures, without per-node tracing, for graphs of 100 / 400 / 1,650 nodes of production-like kernels:

- host time inside ``CUDAGraph.replay()`` (cudaGraphLaunch), time.perf_counter_ns around the call;
- first-node delay: how long after ``replay()`` is called the graph's first kernel starts on the GPU.

Graph composition (``plan_nodes``):

- ``mix``: repeating block of 9 nodes, 1 mid-size streaming Triton kernel (64 MB read, ~0.27 ms at 235 GB/s) and 8
  tiny ones (Triton glue-like kernels: an RMS-norm of 1-4 rows x 4096, a 16-CTA elementwise, plus torch add / mul_ /
  copy_ on small tensors). 1,650 nodes run ~50 ms, like a real verify; 100 / 400 run ~3 / ~12 ms.
- ``tiny``: the same block without the streaming kernel (only launch-bound kernels).
- ``calib``: a 2-node graph (the two stamps only): the fixed launch latency floor of the method.

Every graph's node 0 is a stamp kernel that writes %globaltimer (Triton inline asm ``mov.u64 $0, %globaltimer;``)
into a device buffer, and its last node is another stamp (the graph's device span). Per rep:

    stamps.zero_(); synchronize()                          (stream idle)
    stamp_eager()   -> eager stamp kernel on the same stream
    t_b = perf_counter_ns()                                (the eager launch has returned)
    t_call = perf_counter_ns(); graph.replay(); t_ret = perf_counter_ns()
    synchronize(); read stamps

    first-node delay = (stamp(node 0) - stamp(eager)) - (t_call - t_b)          [first_node_delay_us]

The eager stamp starts L after its launch returns (L = launch latency of one kernel, a few us, unknown sign), so the
delay is biased by -L; the ``calib`` 2-node graph shows the floor of the method (report both raw and minus calib).
%globaltimer's resolution is 32 ns - 1 us depending on the GPU: irrelevant at the 400 us gate.

Chained split (``--splits``, default 2 4 8, on the 1,650-node graphs): the same body cut into k contiguous pieces,
each captured as its own graph with its own start / end stamps, replayed back to back. Reported per piece: host time
of its replay() and its first-kernel delay from its own replay() call, device gap before it (its start stamp minus
the previous piece's end stamp: > 0 means the launch of that piece was not hidden), and end-to-end = last end stamp
minus the eager stamp (corrected like the delay), vs the single graph's. Expected saving of the split =
delay(1 graph) - delay(first piece); the end-to-end difference is the saving actually realised.

cudaGraphUpload: for the single graphs, the first replay after instantiation is also timed three ways (fresh captures,
``--first-reps`` each): torch default (keep_graph=False, instantiated at capture end), keep_graph + instantiate()
without upload, and keep_graph + instantiate() + cuGraphUpload (driver API via ctypes) before the first replay. Node
counts come from cuGraphGetNodes(raw_cuda_graph()) when torch has keep_graph (else the planned count is reported).

Reading it: the GATE line reads the steady-state median first-node delay of the ``mix`` 1,650-node graph of a PLAIN
(no nsys) run:

    GATE item2: uncaptured first-node delay at ~1650 nodes = X us (>= 400 us -> build the split): PASS/FAIL

PASS -> build item 2's split (2-4 chained graphs). The nsys runs (``--tag nsys-graph`` / ``--tag nsys-node``) show
how much of the W11 trace's 1,105 us is the tracer's own overhead: compare their host / delay columns with the plain
run. A run under nsys prints its gate as informational only.

    PYTHONPATH=... python results/THEORY2-SESSION/probes/glprobe.py --tag plain --json glprobe-plain.json
    nsys profile --cuda-graph-trace=graph -o glp-graph python .../glprobe.py --tag nsys-graph --json glprobe-graph.json
    nsys profile --cuda-graph-trace=node  -o glp-node  python .../glprobe.py --tag nsys-node  --json glprobe-node.json

Runtime: ~1-2 min plain (Triton compiles included), a few minutes under node tracing. Needs one GPU; no other
process should use it.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

try:                                     # module globals for the jit kernels; importing on a box without Triton works
    import triton
    import triton.language as tl
except ImportError:                      # pragma: no cover
    triton = tl = None

GATE_US = 400.0
SIZES = (100, 400, 1650)
SPLITS = (2, 4, 8)
STREAM_MB = 64
# 9-node blocks: 1 streaming + 8 tiny (mix), or the 8 tiny ones (tiny)
MIX_BLOCK = ("stream", "glue", "t_add", "glue_wide", "glue", "t_mul", "glue", "t_copy", "glue_wide")
TINY_BLOCK = tuple(k for k in MIX_BLOCK if k != "stream")


# -- pure helpers (CPU-testable) ---------------------------------------------------------------------------------------
def stats(xs) -> dict:
    """median / p10 / p90 / mean / min / max / n of a list of numbers (None and NaN dropped)."""

    v = sorted(float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x)))
    if not v:
        return {"n": 0, "median": None, "p10": None, "p90": None, "mean": None, "min": None, "max": None}

    def pct(p):
        if len(v) == 1:
            return v[0]
        pos = p * (len(v) - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(v) - 1)
        return v[lo] + (v[hi] - v[lo]) * (pos - lo)

    return {"n": len(v), "median": statistics.median(v), "p10": pct(0.10), "p90": pct(0.90),
            "mean": statistics.fmean(v), "min": v[0], "max": v[-1]}


def first_node_delay_us(stamp_eager_ns: int, stamp_node0_ns: int, t_eager_ret_ns: int, t_call_ns: int) -> float:
    """GPU time from the eager stamp to node 0's stamp, minus the host time from the eager launch's return to the
    replay() call: how long after replay() was called the first node started (biased by -L, see the docstring)."""

    return (stamp_node0_ns - stamp_eager_ns) / 1e3 - (t_call_ns - t_eager_ret_ns) / 1e3


def chain_analysis(stamp_eager: int, starts: list, ends: list, t_eager_ret: int, t_calls: list, t_rets: list) -> dict:
    """One rep of k chained graphs (k = 1 is a single graph). ``starts`` / ``ends``: each piece's first / last stamp
    (ns, GPU clock); ``t_calls`` / ``t_rets``: host ns around each replay(). All us:

    first_delay: first piece's start after the first replay() call; piece_delay[i]: piece i's start after its own
    replay() call; host[i]: host time of replay() i; gap[i]: piece i's start minus piece i-1's end (0 for i = 0);
    span: last end - first start; end_to_end: last end after the first replay() call; host_total: first call ->
    last return."""

    k = len(starts)
    if not (len(ends) == len(t_calls) == len(t_rets) == k) or k == 0:
        raise ValueError("chain_analysis: starts / ends / t_calls / t_rets must have the same length >= 1")
    piece_delay = [first_node_delay_us(stamp_eager, starts[i], t_eager_ret, t_calls[i]) for i in range(k)]
    host = [(t_rets[i] - t_calls[i]) / 1e3 for i in range(k)]
    gaps = [0.0] + [(starts[i] - ends[i - 1]) / 1e3 for i in range(1, k)]
    return {"first_delay": piece_delay[0], "piece_delay": piece_delay, "host": host, "gap": gaps,
            "span": (ends[-1] - starts[0]) / 1e3,
            "end_to_end": first_node_delay_us(stamp_eager, ends[-1], t_eager_ret, t_calls[0]),
            "host_total": (t_rets[-1] - t_calls[0]) / 1e3}


def plan_nodes(n_total: int, variant: str) -> list:
    """Kinds of the ``n_total`` nodes of a graph: a start stamp, n_total - 2 body nodes cycling the variant's block,
    an end stamp. ``calib``: the two stamps only."""

    if variant == "calib":
        return ["stamp", "stamp"]
    block = {"mix": MIX_BLOCK, "tiny": TINY_BLOCK}[variant]
    if n_total < 2:
        raise ValueError("plan_nodes: a graph needs >= 2 nodes (the stamps)")
    body = [block[i % len(block)] for i in range(n_total - 2)]
    return ["stamp"] + body + ["stamp"]


def split_sizes(n: int, k: int) -> list:
    """n items cut into k contiguous pieces as equal as possible (the first n % k pieces one larger)."""

    if k < 1 or n < k:
        raise ValueError(f"split_sizes: cannot cut {n} into {k}")
    q, r = divmod(n, k)
    return [q + (1 if i < r else 0) for i in range(k)]


def split_plan(n_total: int, variant: str, k: int) -> list:
    """The single graph's body cut into k contiguous pieces, each wrapped in its own start / end stamps (so the
    pieces hold the same work; total nodes = n_total - 2 + 2k)."""

    body = plan_nodes(n_total, variant)[1:-1]
    out, i = [], 0
    for s in split_sizes(len(body), k):
        out.append(["stamp"] + body[i:i + s] + ["stamp"])
        i += s
    return out


def gate_item2(delay_us, threshold: float = GATE_US, informational: bool = False) -> tuple:
    """(pass, GATE line) for the uncaptured first-node delay at ~1,650 nodes."""

    if delay_us is None:
        return False, "GATE item2: uncaptured first-node delay at ~1650 nodes = n/a (>= 400 us -> build the split): FAIL (no measurement)"
    ok = delay_us >= threshold
    line = (f"GATE item2: uncaptured first-node delay at ~1650 nodes = {delay_us:.0f} us (>= {threshold:.0f} us -> "
            f"build the split): {'PASS' if ok else 'FAIL'}")
    if informational:
        line += "  [informational: this run is under a profiler / not tagged plain; the gate reads the plain run]"
    return ok, line


def split_saving(single: dict | None, pieces: dict | None) -> dict | None:
    """Expected saving of a split: delay(1 graph) - delay(first piece), and the realised end-to-end difference (us,
    medians)."""

    if not single or not pieces:
        return None
    try:
        return {"expected_us": single["first_delay"]["median"] - pieces["first_delay"]["median"],
                "end_to_end_us": single["end_to_end"]["median"] - pieces["end_to_end"]["median"]}
    except (KeyError, TypeError):
        return None


def under_profiler() -> bool:
    """Best-effort: nsys injects itself via LD_PRELOAD / CUDA_INJECTION64_PATH or sets NSYS_* variables."""

    env = os.environ
    if any(k.startswith("NSYS_") for k in env):
        return True
    inj = env.get("CUDA_INJECTION64_PATH", "") + env.get("LD_PRELOAD", "")
    return "nsys" in inj.lower() or "toolsinjection" in inj.lower()


def fmt_stat(s: dict | None, w: int = 7) -> str:
    if not s or s.get("median") is None:
        return f"{'n/a':>{w}} ({'':>5}-{'':>5})"
    return f"{s['median']:{w}.0f} ({s['p10']:5.0f}-{s['p90']:5.0f})"


# -- driver API (ctypes, guarded) ----------------------------------------------------------------------------------------
_CU = None


def _libcuda():
    global _CU
    if _CU is None:
        try:
            _CU = ctypes.CDLL("libcuda.so.1")
            _CU.cuGraphGetNodes.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
            _CU.cuGraphGetNodes.restype = ctypes.c_int
            _CU.cuGraphUpload.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            _CU.cuGraphUpload.restype = ctypes.c_int
        except (OSError, AttributeError):
            _CU = False
    return _CU or None


def _as_ptr(h) -> int | None:
    if h is None:
        return None
    if isinstance(h, int):
        return h
    try:                                              # a PyCapsule
        f = ctypes.pythonapi.PyCapsule_GetPointer
        f.restype, f.argtypes = ctypes.c_void_p, [ctypes.py_object, ctypes.c_char_p]
        name = ctypes.pythonapi.PyCapsule_GetName
        name.restype, name.argtypes = ctypes.c_char_p, [ctypes.py_object]
        return f(h, name(h))
    except Exception:
        try:
            return int(h)
        except Exception:
            return None


def count_nodes(g) -> int | None:
    cu = _libcuda()
    if cu is None:
        return None
    try:
        p = _as_ptr(g.raw_cuda_graph())
    except Exception:
        return None
    if not p:
        return None
    n = ctypes.c_size_t(0)
    rc = cu.cuGraphGetNodes(ctypes.c_void_p(p), None, ctypes.byref(n))
    return int(n.value) if rc == 0 else None


def upload(g, stream) -> bool:
    cu = _libcuda()
    if cu is None:
        return False
    try:
        p = _as_ptr(g.raw_cuda_graph_exec())
    except Exception:
        return False
    if not p:
        return False
    return cu.cuGraphUpload(ctypes.c_void_p(p), ctypes.c_void_p(stream.cuda_stream)) == 0


# -- GPU part (lazy) --------------------------------------------------------------------------------------------------------
_TK = None


def triton_kernels() -> dict:
    """The probe's Triton kernels (jit functions; built lazily, cached; importing this module needs no Triton)."""

    global _TK
    if _TK is not None:
        return _TK
    if triton is None:
        raise RuntimeError("triton is not installed")

    @triton.jit
    def stamp_k(OUT, slot):
        # the same asm as triton.language.extra.cuda.globaltimer()
        t = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False,
                                      pack=1)
        tl.store(OUT + slot, t)

    @triton.jit
    def stream_k(P, OUT, n, BLOCK: tl.constexpr, ITERS: tl.constexpr):
        pid = tl.program_id(0)
        acc = tl.zeros((BLOCK,), dtype=tl.int32)
        for i in range(ITERS):
            off = (pid * ITERS + i) * BLOCK + tl.arange(0, BLOCK)
            acc += tl.load(P + off, mask=off < n, other=0)
        tl.store(OUT + pid, tl.sum(acc, axis=0))

    @triton.jit
    def glue_k(X, W, Y, N: tl.constexpr, EPS: tl.constexpr):
        r = tl.program_id(0)
        o = tl.arange(0, N)
        x = tl.load(X + r * N + o).to(tl.float32)
        w = tl.load(W + o).to(tl.float32)
        inv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + EPS)
        tl.store(Y + r * N + o, (x * inv * w).to(tl.bfloat16))

    @triton.jit
    def wide_k(X, Y, n, BLOCK: tl.constexpr):
        o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = o < n
        x = tl.load(X + o, mask=m, other=0.0).to(tl.float32)
        tl.store(Y + o, (x * tl.sigmoid(x)).to(tl.bfloat16), mask=m)

    _TK = {"stamp": stamp_k, "stream": stream_k, "glue": glue_k, "wide": wide_k}
    return _TK


class Kernels:
    """The kernels + the tensors every node kind uses. Built on first use (needs a GPU)."""

    def __init__(self):
        import torch

        self.torch = torch
        tk = triton_kernels()
        stamp_k, stream_k, glue_k, wide_k = tk["stamp"], tk["stream"], tk["glue"], tk["wide"]
        self.stamp_k, self.stream_k, self.glue_k, self.wide_k = stamp_k, stream_k, glue_k, wide_k
        dev = "cuda"
        self.stamps = torch.zeros(256, dtype=torch.int64, device=dev)
        self.sbytes = STREAM_MB * 1024 * 1024
        self.sbufs = [torch.randint(0, 1 << 20, (self.sbytes // 4,), dtype=torch.int32, device=dev) for _ in range(2)]
        self.n_stream = self.sbytes // 4
        self.SB, self.SI = 2048, 8
        self.sgrid = triton.cdiv(self.n_stream, self.SB * self.SI)
        self.sout = torch.zeros(self.sgrid, dtype=torch.int32, device=dev)
        self.D = 4096
        self.x = [torch.randn((r, self.D), device=dev).to(torch.bfloat16) for r in (1, 2, 4)]
        self.w = torch.randn((self.D,), device=dev).to(torch.bfloat16)
        self.y = [torch.empty_like(t) for t in self.x]
        self.a = torch.randn((4, 4096), device=dev)
        self.b = torch.randn((4, 4096), device=dev)
        self.c = torch.empty_like(self.a)
        self.i = 0

    def stamp(self, slot: int):
        self.stamp_k[(1,)](self.stamps, slot, num_warps=1)

    def launcher(self, kind: str, idx: int):
        """A zero-arg callable launching one node of ``kind`` (idx varies the tensors a little)."""

        torch = self.torch
        if kind == "stream":
            buf = self.sbufs[idx % 2]
            return lambda: self.stream_k[(self.sgrid,)](buf, self.sout, self.n_stream, BLOCK=self.SB, ITERS=self.SI,
                                                        num_warps=8)
        if kind == "glue":
            j = idx % 3
            x, y = self.x[j], self.y[j]
            return lambda: self.glue_k[(x.shape[0],)](x, self.w, y, N=self.D, EPS=1e-6, num_warps=4)
        if kind == "glue_wide":
            x, y = self.x[2], self.y[2]
            n = x.numel()
            return lambda: self.wide_k[(n // 1024,)](x, y, n, BLOCK=1024, num_warps=4)
        if kind == "t_add":
            return lambda: torch.add(self.a, self.b, out=self.c)
        if kind == "t_mul":
            return lambda: self.c.mul_(0.5)
        if kind == "t_copy":
            return lambda: self.a.copy_(self.c)
        raise ValueError(kind)

    def build(self, kinds: list, start_slot: int):
        """Callables for one graph: node 0 stamps into start_slot, the last node into start_slot + 1."""

        fns = []
        for i, k in enumerate(kinds):
            if k == "stamp":
                s = start_slot if i == 0 else start_slot + 1
                fns.append(lambda s=s: self.stamp(s))
            else:
                fns.append(self.launcher(k, i))
        return fns


def _capture(K, fns, keep: bool):
    torch = K.torch
    g = None
    if keep:
        try:
            g = torch.cuda.CUDAGraph(keep_graph=True)
        except TypeError:
            g = None
    kept = g is not None
    if g is None:
        g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    return g, kept


def _one_rep(K, graphs, starts_slots):
    """One timed rep of back-to-back replays of ``graphs``; returns chain_analysis()."""

    torch = K.torch
    pc = time.perf_counter_ns
    K.stamps.zero_()
    torch.cuda.synchronize()
    time.sleep(0.0002)
    K.stamp(0)
    t_b = pc()
    t_calls, t_rets = [], []
    for g in graphs:
        t_calls.append(pc())
        g.replay()
        t_rets.append(pc())
    torch.cuda.synchronize()
    st = K.stamps.cpu().tolist()
    starts = [st[s] for s in starts_slots]
    ends = [st[s + 1] for s in starts_slots]
    if st[0] == 0 or 0 in starts or 0 in ends:
        raise RuntimeError("a stamp was not written")
    return chain_analysis(st[0], starts, ends, t_b, t_calls, t_rets)


def _summ(reps: list) -> dict:
    out = {k: stats([r[k] for r in reps]) for k in ("first_delay", "end_to_end", "span", "host_total")}
    k = len(reps[0]["host"]) if reps else 0
    out["pieces"] = [{"host": stats([r["host"][i] for r in reps]), "delay": stats([r["piece_delay"][i] for r in reps]),
                      "gap": stats([r["gap"][i] for r in reps])} for i in range(k)]
    return out


def run_case(K, name: str, plans: list, reps: int, warm: int, first_reps: int) -> dict:
    """plans: list of per-piece kind lists (one entry = a single graph)."""

    torch = K.torch
    res = {"case": name, "pieces": len(plans), "planned_nodes": [len(p) for p in plans]}
    slots = [1 + 2 * i for i in range(len(plans))]
    fns = [K.build(p, s) for p, s in zip(plans, slots)]
    for fl in fns:                                   # compile / warm every kernel eagerly
        for f in fl:
            f()
    torch.cuda.synchronize()
    graphs, kept = [], False
    t0 = time.perf_counter()
    for fl in fns:
        g, kept = _capture(K, fl, keep=True)
        if kept:
            try:
                g.instantiate()
            except Exception:
                pass
        graphs.append(g)
    res["capture_s"] = time.perf_counter() - t0
    res["keep_graph"] = kept
    res["nodes"] = [count_nodes(g) if kept else None for g in graphs]
    for _ in range(warm):
        for g in graphs:
            g.replay()
    torch.cuda.synchronize()
    rr = [_one_rep(K, graphs, slots) for _ in range(reps)]
    res["steady"] = _summ(rr)
    del graphs
    if len(plans) == 1 and first_reps > 0:           # first replay after instantiation, three ways
        fr = {}
        for mode in ("torch_default", "instantiate_no_upload", "instantiate_upload"):
            try:
                reps_m, extra = [], []
                for _ in range(first_reps):
                    keep = mode != "torch_default"
                    g, kept = _capture(K, fns[0], keep=keep)
                    if keep and not kept:
                        raise RuntimeError("torch has no CUDAGraph(keep_graph=True)")
                    if keep:
                        t = time.perf_counter_ns()
                        g.instantiate()
                        extra.append(("instantiate_us", (time.perf_counter_ns() - t) / 1e3))
                    if mode == "instantiate_upload":
                        torch.cuda.synchronize()
                        t = time.perf_counter_ns()
                        if not upload(g, torch.cuda.current_stream()):
                            raise RuntimeError("cuGraphUpload unavailable or failed")
                        torch.cuda.synchronize()
                        extra.append(("upload_us", (time.perf_counter_ns() - t) / 1e3))
                    reps_m.append(_one_rep(K, [g], slots[:1]))
                    del g
                fr[mode] = _summ(reps_m)
                for key in ("instantiate_us", "upload_us"):
                    vals = [v for k, v in extra if k == key]
                    if vals:
                        fr[mode][key] = stats(vals)
            except Exception as e:  # noqa: BLE001
                fr[mode] = {"error": f"{type(e).__name__}: {e}"}
        res["first_replay"] = fr
    torch.cuda.empty_cache()
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="plain", help="label for this run (plain / nsys-graph / nsys-node)")
    ap.add_argument("--json", help="write every number here")
    ap.add_argument("--reps", type=int, default=60, help="timed reps per case (>= 50)")
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--first-reps", type=int, default=10, help="fresh captures per first-replay mode (0: skip)")
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    ap.add_argument("--splits", type=int, nargs="*", default=list(SPLITS))
    ap.add_argument("--split-sizes", type=int, nargs="*", default=[1650], help="graph sizes that also get splits")
    ap.add_argument("--variants", nargs="+", default=["mix", "tiny"], choices=["mix", "tiny"])
    a = ap.parse_args()

    import torch

    prof = under_profiler()
    out = {"tag": a.tag, "under_profiler": prof, "argv": sys.argv, "cases": [], "errors": []}
    try:
        out["device"] = torch.cuda.get_device_name(0)
        out["torch"] = torch.__version__
        import triton
        out["triton"] = triton.__version__
        out["sm_clock_mhz"] = None
        try:
            out["sm_clock_mhz"] = torch.cuda.clock_rate()
        except Exception:
            pass
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"setup: {e}")
    print(f"[glprobe] tag={a.tag} under_profiler={prof} device={out.get('device')} torch={out.get('torch')} "
          f"triton={out.get('triton')} reps={a.reps}", flush=True)
    try:
        K = Kernels()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        out["errors"].append(f"kernels: {type(e).__name__}: {e}")
        if a.json:
            Path(a.json).write_text(json.dumps(out, indent=1, default=str))
        return 1

    cases = [("calib", "calib", 2, 1)]
    for v in a.variants:
        for n in a.sizes:
            cases.append((f"{v} {n}", v, n, 1))
            if n in a.split_sizes:
                for k in a.splits:
                    cases.append((f"{v} {n} / {k}", v, n, k))
    t_start = time.perf_counter()
    for name, v, n, k in cases:
        try:
            plans = [plan_nodes(n, v)] if k == 1 else split_plan(n, v, k)
            r = run_case(K, name, plans, a.reps, a.warm, a.first_reps if k == 1 and v != "calib" else 0)
            r.update({"variant": v, "size": n, "k": k})
            out["cases"].append(r)
            s = r["steady"]
            print(f"[glprobe] {name:16s} nodes {r['nodes'] if any(r['nodes']) else r['planned_nodes']} | host "
                  f"{fmt_stat(s['pieces'][0]['host'])} us | first delay {fmt_stat(s['first_delay'])} us | end-to-end "
                  f"{fmt_stat(s['end_to_end'], 8)} us | {time.perf_counter() - t_start:.0f} s", flush=True)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            out["errors"].append(f"{name}: {type(e).__name__}: {e}")
            print(f"[glprobe] {name}: ERROR {type(e).__name__}: {e}", flush=True)

    # -- report -----------------------------------------------------------------------------------------------------
    by = {c["case"]: c for c in out["cases"]}
    calib = by.get("calib", {}).get("steady", {}).get("first_delay", {}).get("median")
    print(f"\n== glprobe [{a.tag}] (us: median (p10-p90) over {a.reps} reps; delay = first kernel start after "
          f"replay() is called; calib 2-node delay = {calib if calib is None else round(calib, 1)} us)", flush=True)
    print(f"{'case':16s} {'nodes':>12s} {'host replay':>21s} {'first delay':>21s} {'delay-calib':>11s} "
          f"{'end-to-end':>22s} {'dev span':>9s}")
    for c in out["cases"]:
        s = c["steady"]
        nl = c["nodes"] if all(c["nodes"]) else c["planned_nodes"]
        nodes = "+".join(str(x) for x in nl) if len(nl) <= 2 else f"{len(nl)}x{max(nl)}"
        host = s["host_total"] if c["k"] > 1 else s["pieces"][0]["host"]
        dc = "" if calib is None or s["first_delay"]["median"] is None else f"{s['first_delay']['median'] - calib:11.0f}"
        span = s["span"]["median"]
        print(f"{c['case']:16s} {nodes[-12:]:>12s} {fmt_stat(host)} {fmt_stat(s['first_delay'])} {dc:>11s} "
              f"{fmt_stat(s['end_to_end'], 8)} {span if span is None else round(span):>9}")
        if c["k"] > 1:
            for i, p in enumerate(s["pieces"]):
                print(f"    piece {i}: host {fmt_stat(p['host'])}  delay from own call {fmt_stat(p['delay'])}  "
                      f"device gap before {fmt_stat(p['gap'])}")
        for mode, f in (c.get("first_replay") or {}).items():
            if "error" in f:
                print(f"    first replay {mode:22s} ERROR {f['error']}")
            else:
                ex = "".join(f"  {k[:-3]} {f[k]['median']:.0f} us" for k in ("instantiate_us", "upload_us") if k in f)
                print(f"    first replay {mode:22s} host {fmt_stat(f['pieces'][0]['host'])}  delay "
                      f"{fmt_stat(f['first_delay'])}  end-to-end {fmt_stat(f['end_to_end'], 8)}{ex}")
    out["savings"] = {}
    for v in a.variants:
        for n in a.split_sizes:
            single = by.get(f"{v} {n}", {}).get("steady")
            for k in a.splits:
                sv = split_saving(single, by.get(f"{v} {n} / {k}", {}).get("steady"))
                if sv:
                    out["savings"][f"{v} {n} / {k}"] = sv
                    print(f"split saving {v} {n} / {k}: expected (delay 1 graph - delay first piece) "
                          f"{sv['expected_us']:.0f} us, realised end-to-end {sv['end_to_end_us']:.0f} us")
    g1650 = by.get("mix 1650", {}).get("steady", {}).get("first_delay", {}).get("median")
    ok, line = gate_item2(g1650, informational=prof or a.tag != "plain")
    out["gate"] = {"item2_delay_us": g1650, "pass": ok, "line": line}
    print(line, flush=True)
    if out["errors"]:
        print(f"[glprobe] {len(out['errors'])} error(s): " + " | ".join(out["errors"]), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str))
        print(f"[glprobe] wrote {a.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
