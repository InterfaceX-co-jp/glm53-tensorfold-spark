"""W7 MEASUREMENT COPY (results/W7, bind-mounted over the image's profile.py for the W7 window only; not a patch):
both ranks print their report (tagged "rank"), and with W7_NVTX=1 every probe site also drops an NVTX mark (name =
the work since the previous mark) and batch rounds / pieces / verify / drafting get NVTX ranges, so an nsys trace
can attribute each kernel to a component on both ranks. Timing-only: no tensor is read or written.

Prefill timing probe for GLM-5.3-Flash (GLM53_TF_PROFILE=1): where a prompt's GPU time goes, by component.

Off (the default), every probe site is one ``if profile.ON`` test and nothing else runs. On, ``prefill`` opens a
session and each site records one CUDA event on the current stream ("the work enqueued since the previous mark was
<name>"); the whole forward runs on one stream, NCCL's all-gathers included, so the intervals add up to the GPU
time of the prefill. Nothing waits on an event until the session ends (one synchronize after the first token is
sampled, which synchronizes anyway), no tensor is read or written, so no result changes. Rank 0 prints a table
and one ``GLM53_TF_PROFILE {json}`` line to stderr per prefill.

Names: ``kda.proj`` (projections, f_b/g_b), ``kda.chain`` (the recurrence kernel), ``kda.o_proj``; ``dsa.proj``
(q/kv projections, norms, cache write), ``dsa.attn`` (dense attention), ``dsa.indexer`` (index keys, pools, query,
scores, top-k), ``dsa.sparse_attn``, ``dsa.o_proj``; ``mlp.dense``; ``moe.router`` (router, top-k, grouping),
``moe.routed`` (routed experts), ``moe.shared`` (shared expert), ``moe.combine``; ``allgather``; ``hc``
(hyper-connections, taps); ``embed``; ``head`` (final norm, lm_head over the chunk); ``mtp.*`` (the MTP head's
absorb, same names); ``drafter`` (DFlash2 taps); ``commit``; ``stage``/``resume``/``sample``. An all-gather's time
includes waiting for the other rank.
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

ON = os.environ.get("GLM53_TF_PROFILE", "").strip().lower() not in ("", "0", "false", "no", "off")

GROUPS = {
    "kda.proj": "KDA block", "kda.chain": "KDA block", "kda.o_proj": "KDA block",
    "dsa.proj": "DSA attention", "dsa.attn": "DSA attention", "dsa.sparse_attn": "DSA attention",
    "dsa.o_proj": "DSA attention", "dsa.indexer": "DSA indexer/top-k",
    "moe.routed": "MoE routed experts", "moe.router": "MoE router/grouping",
    "moe.shared": "shared expert + dense MLP", "moe.combine": "shared expert + dense MLP",
    "mlp.dense": "shared expert + dense MLP", "allgather": "all-gathers",
    "hc": "hyper-connections", "embed": "hyper-connections", "head": "head (all chunk rows)",
}


class Probe:
    """One prefill's marks: (event, name of the work since the previous mark)."""

    def __init__(self) -> None:
        self.active = False
        self.prefix = ""
        self.pool: list[torch.cuda.Event] = []
        self.marks: list[tuple[torch.cuda.Event, str | None]] = []
        self.chunks = 0
        self.rows = 0
        self.t0 = 0.0

    def _event(self) -> torch.cuda.Event:
        i = len(self.marks)
        if i == len(self.pool):
            self.pool.append(torch.cuda.Event(enable_timing=True))
        ev = self.pool[i]
        ev.record()
        return ev

    def begin(self) -> None:
        self.active = True
        self.prefix = ""
        self.marks = []
        self.chunks = 0
        self.rows = 0
        self.t0 = time.perf_counter()
        if NVTX:
            torch.cuda.nvtx.mark("W7:pf_begin")
        self.marks.append((self._event(), None))

    def lap(self, name: str) -> None:
        self.marks.append((self._event(), self.prefix + name))

    def end(self, tokens: int, cached: int, rank: int) -> dict | None:
        """Close the session: per-name GPU milliseconds (rank 0 prints them)."""

        if not self.active:
            return None
        self.active = False
        self.prefix = ""
        torch.cuda.synchronize()
        wall = time.perf_counter() - self.t0
        totals: dict[str, float] = {}
        for (a, _), (b, name) in zip(self.marks, self.marks[1:]):
            totals[name] = totals.get(name, 0.0) + a.elapsed_time(b)
        self.marks = []
        report = {"tokens": tokens, "cached": cached, "rows": self.rows, "chunks": self.chunks,
                  "wall_s": round(wall, 4), "gpu_ms": round(sum(totals.values()), 3),
                  "ms": {k: round(v, 3) for k, v in sorted(totals.items(), key=lambda kv: -kv[1])}}
        report["rank"] = rank
        if NVTX:
            torch.cuda.nvtx.mark("W7:pf_end")
        _print(report)
        return report


def _group_of(name: str) -> str:
    if name.startswith("mtp."):
        return "MTP head absorb"
    return GROUPS.get(name, f"other ({name})")


def _print(report: dict) -> None:
    ms = report["ms"]
    total = max(report["gpu_ms"], 1e-9)
    n = max(report["chunks"], 1)
    rows = report["rows"]
    groups: dict[str, float] = {}
    for name, v in ms.items():
        g = _group_of(name)
        groups[g] = groups.get(g, 0.0) + v
    out = [f"[GLM53_TF_PROFILE] r{report.get('rank', 0)} prefill {report['tokens']} tokens ({report['cached']} resumed, {rows} computed in "
           f"{report['chunks']} chunks): GPU {total:.1f} ms, wall {report['wall_s'] * 1e3:.1f} ms, "
           f"{rows / max(report['wall_s'], 1e-9):.1f} tok/s"]
    out.append(f"  {'component':<34}{'ms':>11}{'ms/chunk':>11}{'%':>7}")
    for g, v in sorted(groups.items(), key=lambda kv: -kv[1]):
        out.append(f"  {g:<34}{v:>11.1f}{v / n:>11.2f}{100 * v / total:>7.1f}")
    out.append("  detail: " + ", ".join(f"{k} {v:.1f}" for k, v in ms.items()))
    print("\n".join(out), file=sys.stderr, flush=True)
    print("GLM53_TF_PROFILE " + json.dumps(report), file=sys.stderr, flush=True)


NVTX = os.environ.get("W7_NVTX", "0") == "1"

P = Probe()


def lap(name: str) -> None:
    """A probe site: the work enqueued since the previous mark was ``name`` (call only under ``if profile.ON``)."""

    if P.active:
        P.lap(name)
    if NVTX:
        torch.cuda.nvtx.mark("W7:" + P.prefix + name)


# -- W7: NVTX ranges around batch rounds (installed once tensorfold's batch module is imported) -------------------
def _w7_wrap(obj, attr, label):
    fn = getattr(obj, attr, None)
    if fn is None or getattr(fn, "_w7", False):
        return
    rp, rpop = torch.cuda.nvtx.range_push, torch.cuda.nvtx.range_pop

    def wrapped(*a, **k):
        rp(label)
        try:
            return fn(*a, **k)
        finally:
            rpop()

    wrapped._w7 = True
    setattr(obj, attr, wrapped)


def _w7_install() -> None:
    import threading

    def run() -> None:
        for _ in range(3600):
            m = sys.modules.get("tensorfold.families.glm5_next.cuda.batch")
            if m is not None and hasattr(m, "Batcher") and hasattr(m, "Stepper"):
                B = m.Batcher
                for attr, label in (("_execute", "W7:round"), ("_piece", "W7:piece"), ("_verify", "W7:verify"),
                                    ("_forward", "W7:vfwd"), ("_plan", "W7:plan")):
                    _w7_wrap(B, attr, label)
                _w7_wrap(m.Stepper, "propose", "W7:propose")
                if hasattr(m, "MtpChains"):
                    _w7_wrap(m.MtpChains, "run", "W7:mtp_chains")
                for fn, label in (("sample_multi", "W7:sample_multi"), ("sample_drafts", "W7:sample_drafts"),
                                  ("mtp_multi", "W7:mtp_multi")):
                    _w7_wrap(m, fn, label)
                print("[W7] NVTX batch ranges installed", file=sys.stderr, flush=True)
                return
            time.sleep(0.2)

    threading.Thread(target=run, name="w7-nvtx", daemon=True).start()


if NVTX:
    _w7_install()
