#!/usr/bin/env python3
"""patches/0550 GPU plan, step P (docs/MEMORY-SAFETY.md): can a CUDA allocation on GB10 take reclaimable page cache?

Admission with GLM53_TF_ADMIT_MEM=available counts clean page cache as usable; that is safe only if cudaMalloc gets
those pages when MemFree runs out (the kernel reclaims clean file pages for the driver's allocation) instead of
failing. This probe measures it, with the model server STOPPED on this node:

1. fill the page cache by reading FILE (buffered, e.g. a prepared weight shard or a scratch file of tens of GB) until
   Cached grew by FILL_GIB or the file ends;
2. allocate and touch device memory in STEP_GIB tensors until the total passes the MemFree seen after step 1 by
   BEYOND_GIB, but never while MemAvailable would fall under FLOOR_GIB (the probe stops first);
3. per step: seconds, cudaMemGetInfo free, MemFree / MemAvailable / Cached / Dirty; then free everything.

    python3 bench/pagecache_probe.py OUT.json FILE [FILL_GIB=20] [BEYOND_GIB=6] [STEP_GIB=1] [FLOOR_GIB=8]

Verdict ``cache_reclaimed``: device memory beyond the starting MemFree was allocated while Cached fell (the kernel
handed page cache to CUDA); ``failed_at`` is set if an allocation raised before the floor. Nothing else is touched:
no drop_caches, no swap. Needs torch with CUDA (run it in the serving image).
"""

from __future__ import annotations

import json
import os
import sys
import time

GiB = 1 << 30


def meminfo() -> dict[str, float]:
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            if k in ("MemFree", "MemAvailable", "Cached", "Dirty", "Writeback", "Mapped"):
                out[k] = int(v.split()[0]) / 2 ** 20
    return out


def fill(path: str, gib: float) -> dict:
    start = meminfo()
    t0 = time.perf_counter()
    read = 0
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(64 << 20)
            if not b:
                break
            read += len(b)
            if read % GiB < (64 << 20) and meminfo()["Cached"] - start["Cached"] >= gib:
                break
    return {"read_gib": read / GiB, "s": round(time.perf_counter() - t0, 1), "before": start, "after": meminfo()}


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    out, path = sys.argv[1], sys.argv[2]
    fill_gib, beyond, step, floor = (float(x) for x in (sys.argv[3:] + ["20", "6", "1", "8"][len(sys.argv) - 3:])[:4])
    import torch

    torch.cuda.init()
    rep = {"file": path, "fill": fill(path, fill_gib), "steps": [], "failed_at": None}
    base = meminfo()
    target = base["MemFree"] + beyond
    held = []
    t_all = time.perf_counter()
    while len(held) * step < target:
        m = meminfo()
        if m["MemAvailable"] - step < floor:
            rep["stopped"] = f"MemAvailable {m['MemAvailable']:.2f} GiB would fall under the floor {floor:g}"
            break
        t0 = time.perf_counter()
        try:
            x = torch.empty((int(step * GiB),), dtype=torch.uint8, device="cuda")
            x.fill_(1)
            torch.cuda.synchronize()
        except RuntimeError as exc:                                  # torch.OutOfMemoryError is one
            rep["failed_at"] = {"allocated_gib": len(held) * step, "error": str(exc)[:300], "meminfo": meminfo()}
            break
        held.append(x)
        free, _ = torch.cuda.mem_get_info()
        rep["steps"].append({"gib": len(held) * step, "s": round(time.perf_counter() - t0, 3),
                             "cuda_free": round(free / GiB, 2), **{k: round(v, 2) for k, v in meminfo().items()}})
    rep["alloc_s"] = round(time.perf_counter() - t_all, 1)
    total = len(held) * step
    last = rep["steps"][-1] if rep["steps"] else base
    rep["summary"] = {"memfree_start": round(base["MemFree"], 2), "cached_start": round(base["Cached"], 2),
                      "allocated_gib": total, "cached_end": round(last["Cached"], 2),
                      "cache_reclaimed": bool(total > base["MemFree"] and last["Cached"] < base["Cached"] - 0.5)}
    del held
    torch.cuda.empty_cache()
    rep["after_free"] = meminfo()
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep["summary"]))
    return 0 if rep["failed_at"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
