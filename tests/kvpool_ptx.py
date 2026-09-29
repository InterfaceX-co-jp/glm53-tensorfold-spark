#!/usr/bin/env python3
"""patches/0290: compile the Triton kernels the KV pool touches for sm_121 (GB10) WITHOUT a GPU, from one TensorFold
tree, and print a hash of each kernel's PTX (debug lines stripped) with the pool off (PT None, PSH 0), plus a compile
of every paged variant. Run it on the tree with 0290 and on the tree without (0001-0280): equal hashes mean the pool
off compiles to the same PTX as before the patch (the "code path identical when off" check).

    PYTHONPATH=<tree without 0290>/src python3 tests/kvpool_ptx.py > /tmp/ptx-base.json
    PYTHONPATH=<tree with 0290>/src    python3 tests/kvpool_ptx.py --against /tmp/ptx-base.json

Offline result (2026-09-28, triton 3.8.0 CPU wheel): 16 / 16 unpaged kernels identical, every paged variant compiled.
Not a pytest file (it needs two trees); ~1 minute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

os.environ.pop("TRITON_INTERPRET", None)          # compile, not interpret

import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

from tensorfold.families.glm5_next.cuda import b12x_attn, latent, sparse  # noqa: E402

paged_tree = hasattr(latent, "_prow")
TGT = GPUTarget("cuda", 121, 32)

def strip(ptx):
    out = []
    skip = False
    for ln in ptx.splitlines():
        s = ln.strip()
        if s.startswith(".section") and "debug" in s:
            skip = True
        if skip:
            if s == "}":
                skip = False
            continue
        if s.startswith((".loc", ".file", "//")) or not s:
            continue
        out.append(ln)
    return "\n".join(out)

def comp(fn, sig, cst, warps=4, stages=1, paged=None):
    sig = dict(sig); cst = dict(cst)
    if paged_tree:
        names = [p.name for p in fn.params]
        for pt, psh in (("PT", "PSH"), ("PTI", "PSHI"), ("PTP", "PSHP")):
            if pt in names:
                if paged and (paged is True or pt in paged):
                    sig[pt] = "*i32"; sig[psh] = "constexpr"; cst[psh] = paged_shift.get(pt, 8)
                else:
                    sig[pt] = "constexpr"; sig[psh] = "constexpr"; cst[pt] = None; cst[psh] = 0
    if "OPQ" in [p.name for p in fn.params] and "OPQ" not in cst:
        cst["OPQ"] = False          # patches/0360's opaque FP8 tile: off here, so pool-off PTX compares with 0290's
    for k in cst:
        sig[k] = "constexpr"
    src = ASTSource(fn=fn, signature=sig, constexprs=cst)
    k = triton.compile(src, target=TGT, options={"num_warps": warps, "num_stages": stages})
    return strip(k.asm["ptx"])

paged_shift = {"PT": 8, "PTI": 8, "PTP": 6}
cases = {}
for f8 in (False, True):
    lc = "*u8" if f8 else "*bf16"; lcs = "*fp32" if f8 else "*bf16"
    rb, sb = (528, 132) if f8 else (512, 0)
    if f8:
        cases["lwrite8"] = (latent._lwrite8, {"LAT": "*bf16", "LC": "*u8", "LCS": "*fp32", "POS": "*i32"},
                            {"L": 512, "RB": 528, "SB": 132, "SA": 128}, 4, 1)
    else:
        cases["lwrite"] = (latent._lwrite, {"LAT": "*bf16", "LC": "*bf16", "POS": "*i32"}, {"L": 512}, 4, 1)
    for bm in (16, 32):
        cases[f"lchunks_f8{int(f8)}_bm{bm}"] = (latent._lchunks, {"QA": "*bf16", "LC": lc, "LCS": lcs, "POS": "*i32",
            "PO": "*fp32", "PM": "*fp32", "PL": "*fp32", "R": "i32"},
            {"H": 32, "L": 512, "CH": 512, "KTS": 32, "SCALE": 0.0625, "BMQ": bm, "FP8": f8, "RB": rb, "SB": sb,
             "SA": 128}, 8, 1)
        cases[f"lsparse_f8{int(f8)}_bm{bm}"] = (latent._lsparse_chunks, {"QA": "*bf16", "LC": lc, "LCS": lcs,
            "TOK": "*i32", "CNT": "*i32", "PO": "*fp32", "PM": "*fp32", "PL": "*fp32", "RS": "i32"},
            {"W": 2051, "H": 32, "L": 512, "CH": 512, "KTS": 32, "SCALE": 0.0625, "BMQ": bm, "FP8": f8, "RB": rb,
             "SB": sb, "SA": 128}, 8, 1)
    cases[f"lsparse_one_f8{int(f8)}"] = (b12x_attn._lsparse_one, {"QA": "*bf16", "LC": lc, "LCS": lcs, "TOK": "*i32",
        "CNT": "*i32", "OUT": "*fp32"}, {"W": 2051, "H": 32, "L": 512, "KTS": 32, "SCALE": 0.0625, "BMQ": 32,
        "FP8": f8, "RB": rb, "SB": sb, "SA": 128}, 8, 1)
cases["index_write"] = (sparse._index_write, {"KR": "*bf16", "k_stride": "i32", "GR": "*fp32", "LNW": "*bf16",
    "LNB": "*bf16", "IK": "*bf16", "IG": "*bf16", "POS": "*i32", "NR": "i32", "eps": "fp32"}, {"D": 128}, 1, 1)
cases["pool_keys"] = (sparse._pool_keys, {"IK": "*bf16", "IG": "*bf16", "APE": "*bf16", "PK": "*bf16", "POS": "*i32",
    "R": "i32", "NR": "i32"}, {"D": 128}, 1, 1)
cases["scores"] = (sparse._scores, {"QI": "*bf16", "W": "*bf16", "w_stride": "i32", "PK": "*bf16", "OUT": "*fp32",
    "POS": "*i32", "R": "i32", "NP": "i32", "scale": "fp32"}, {"H": 32, "D": 128, "BP": 64, "NH": 32,
    "WS": 1.0 / 5.656854249492381}, 4, 1)
cases["scores_rows"] = (sparse._scores_rows, {"QI": "*bf16", "W": "*bf16", "w_stride": "i32", "PK": "*bf16",
    "OUT": "*i32", "POS": "*i32", "R": "i32", "NP": "i32", "row0": "i32", "scale": "fp32"}, {"H": 32, "D": 128,
    "BP": 64, "BRB": 16, "NH": 32, "WS": 1.0 / 5.656854249492381, "KEYS": True}, 4, 1)

res = {}
for name, (fn, sig, cst, w, s) in cases.items():
    res[name] = hashlib.sha256(comp(fn, sig, cst, w, s).encode()).hexdigest()[:16]
    if paged_tree:
        ptx = comp(fn, sig, cst, w, s, paged=True)
        res[name + "+paged"] = f"ok {len(ptx.splitlines())} lines"
        if name == "pool_keys":
            comp(fn, sig, cst, w, s, paged={"PTP"})           # ring keys, paged pool keys
            res[name + "+paged_pk_only"] = "ok"
ap = argparse.ArgumentParser()
ap.add_argument("--against", help="the JSON this script printed on the tree without patches/0290")
args = ap.parse_args()
print(json.dumps(res, indent=0))
if args.against:
    base = json.load(open(args.against))
    diff = [k for k in base if not k.endswith(("+paged", "+paged_pk_only")) and base[k] != res.get(k)]
    print(f"unpaged kernels identical to the base tree: {len(base) - len(diff)} of {len(base)}; differ: {diff}",
          file=sys.stderr)
    sys.exit(1 if diff else 0)
