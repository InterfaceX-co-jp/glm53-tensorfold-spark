#!/usr/bin/env python3
"""Decode: distinct experts read per verify layer, inferred from the row-invariant grouped EXL3 kernel (W7 mix trace).

  experts_union.py SQLITE

grouped_kernel's grid.x = 8 R + 1 (a slot per routed pick + the shared id), so grid.x gives the window's rows R.
The kernel reads each DISTINCT expert's trellis once (programs past ucount exit), so its time is ~ distinct experts x
bytes / bandwidth. Calibration: the R = 1 calls (grid.x 9: MTP draft steps, exactly 8 experts) give the kernel's
bandwidth; U(R) = 8 x t(R) / t(1). Output: per R, calls, median gate/up and down us, inferred U, the uniform-
independent expectation 288 (1 - (280/288)^R), and the MB a layer and rank (U x 6.29 MB)."""
import collections, sqlite3, statistics, sys

c = sqlite3.connect(sys.argv[1])
S = dict(c.execute("select id, value from StringIds"))
rows = c.execute("select start, end, gridX, gridY, gridZ, shortName from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
d = collections.defaultdict(list)
for s, e, x, y, z, nm in rows:
    if S.get(nm) == "grouped_kernel":
        d[("gu" if y == 8 else "dn", (x - 1) // 8)].append((e - s) / 1e3)
EXPERT_MB = 3 * 4096 * 1024 * 0.5 / 1e6          # gate + up + down, one rank's half, 4.0 bpw trellis
t1 = {k: statistics.median(d[(k, 1)]) for k in ("gu", "dn")}
print(f"R=1 calibration: gate/up {t1['gu']:.1f} us, down {t1['dn']:.1f} us -> "
      f"{8 * 2 * 4096 * 1024 * 0.5 / t1['gu'] / 1e3:.0f} / {8 * 4096 * 1024 * 0.5 / t1['dn'] / 1e3:.0f} GB/s")
print("R  calls  gu_us   dn_us   U(gu)  U(dn)  U_indep  MB/layer")
for R in sorted({r for k, r in d}):
    if not d[("gu", R)] or not d[("dn", R)]:
        continue
    gu, dn = statistics.median(d[("gu", R)]), statistics.median(d[("dn", R)])
    ug, ud = 8 * gu / t1["gu"], 8 * dn / t1["dn"]
    ind = 288 * (1 - (280 / 288) ** R)
    print(f"{R:2d} {len(d[('gu', R)]):6d} {gu:7.1f} {dn:7.1f} {ug:6.1f} {ud:6.1f} {ind:7.1f} {EXPERT_MB * (ug + ud) / 2:9.1f}")
