"""patches/0460, compiled for sm_121 (GB10) without a GPU (nvcc; set NVCC=... when it is not on PATH):

- ``l2pf.cu``: every kernel builds (bulk and lines forms of the segment and expert kernels), ptxas accepts
  ``cp.async.bulk.prefetch.L2.global`` for sm_121, no spills, no stack; the PTX holds loads only from the tables /
  router output and the prefetch instructions -- **no store to global memory** (the kernels cannot change a bit);
- ``roce.cu``: still 0350's 64 registers or fewer, no spills; the lean path has the release / acquire forms
  (``st.release.sys`` doorbell, ``atom.acq_rel.sys`` arrival) and the classic path keeps its ``fence.sc.sys`` pairs;
  in the lean doorbell the byte-count stores come before the release store of the sequence (the order the proxy's
  catch-up relies on);
- the C++ bindings (``l2pf.cpp``, ``roce.cpp``) pass ``g++ -fsyntax-only`` when ``TF_CXX_INC`` names the extra include
  directory (c10/cuda + infiniband headers; the CPU torch wheel has no c10/cuda) -- skipped otherwise.

    NVCC=<nvcc> PYTHONPATH=<patched tree>/src pytest -q tests/test_prefetch_comm_compile.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sysconfig
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import tensorfold.families.glm5_next.cuda as cudapkg  # noqa: E402

SRC = Path(cudapkg.__file__).parent
FLAGS = ["-arch=sm_121", "-O3", "-include", "cstdint"]


def _nvcc() -> str:
    venv = Path.home() / ".cache/roce-port-venv/lib/python3.14/site-packages/nvidia/cu13/bin/nvcc"
    for c in (os.environ.get("NVCC"), shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc", str(venv)):
        if c and Path(c).exists():
            return c
    pytest.skip("no nvcc (set NVCC=...)")


def _build(name: str) -> tuple[str, str]:
    """(ptxas -v output, PTX) of SRC/name for sm_121."""

    nvcc = _nvcc()
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([nvcc, *FLAGS, "-cubin", "-Xptxas", "-v", "-o", f"{td}/k.cubin", str(SRC / name)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        p = subprocess.run([nvcc, *FLAGS, "-ptx", "-o", f"{td}/k.ptx", str(SRC / name)], capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-3000:]
        return r.stderr, Path(f"{td}/k.ptx").read_text()


def _entries(ptxas: str) -> dict:
    out, cur = {}, None
    for line in ptxas.splitlines():
        m = re.search(r"Compiling entry function '(\S+)'", line)
        if m:
            cur = m.group(1)
            out[cur] = {}
        m = re.search(r"Used (\d+) registers", line)
        if m and cur:
            out[cur]["regs"] = int(m.group(1))
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m and cur:
            out[cur]["stack"], out[cur]["spill"] = int(m.group(1)), int(m.group(2)) + int(m.group(3))
    return out


def _functions(ptx: str) -> dict[str, str]:
    """PTX body of every .entry."""

    out = {}
    for m in re.finditer(r"\.entry\s+(\S+)\s*\((.*?)\n\}", ptx, re.S):
        out[m.group(1)] = m.group(2)
    return out


def test_l2pf_kernels():
    ptxas, ptx = _build("l2pf.cu")
    ents = _entries(ptxas)
    assert len(ents) == 4, ents
    for name, e in ents.items():
        assert e["spill"] == 0 and e["stack"] == 0, (name, e)
        assert e["regs"] <= 40, (name, e)
    fns = _functions(ptx)
    assert len(fns) == 4
    for name, body in fns.items():
        bulk = "ILb1E" in name
        assert ("cp.async.bulk.prefetch.L2.global" in body) == bulk, name
        assert ("prefetch.global.L2 " in body) == (not bulk), name
        # read-only: the only memory operations are loads (the table / router output) and the prefetches
        ops = set(re.findall(r"\b(st|atom|red|ld|cp)\.[A-Za-z0-9.:]+", body))
        assert ops <= {"ld", "cp"}, (name, ops)
        assert set(re.findall(r"\bcp\.[A-Za-z0-9.:]+", body)) <= {"cp.async.bulk.prefetch.L2.global"}, name


def test_roce_kernel_lean_and_classic_paths():
    ptxas, ptx = _build("roce.cu")
    ents = _entries(ptxas)
    (name, e), = ents.items()
    assert e["regs"] <= 64 and e["spill"] == 0, e
    body = _functions(ptx)[name]
    assert "st.release.sys.global.u32" in body and "atom.acq_rel.sys.global.add.u32" in body
    assert "atom.acq_rel.gpu.global.add.u32" in body
    assert body.count("fence.sc.sys") >= 3                     # 0350's per-block + doorbell + failure record fences
    # the lean doorbell: both byte-count stores precede the release store of the sequence
    rel = body.index("st.release.sys.global.u32")
    before = body[:rel]
    assert before.count("st.relaxed.sys.global.u32") >= 2
    # the one-HCA rule is in the kernel (a remainder by the HCA count selects the waiter)
    assert "rem.u32" in body or "rem.s32" in body


def _cxx_inc() -> str:
    inc = os.environ.get("TF_CXX_INC", "")
    if not inc or not Path(inc).is_dir():
        pytest.skip("set TF_CXX_INC to a directory with c10/cuda and infiniband headers")
    return inc


@pytest.mark.parametrize("name", ["l2pf.cpp", "roce.cpp"])
def test_bindings_syntax(name):
    inc = _cxx_inc()
    cxx = shutil.which("g++")
    if cxx is None:
        pytest.skip("no g++")
    tinc = Path(torch.__file__).parent / "include"
    cuda_inc = Path(_nvcc()).parent.parent / "include"
    r = subprocess.run([cxx, "-std=c++20", "-fsyntax-only", "-DTORCH_EXTENSION_NAME=x",
                        "-DTORCH_API_INCLUDE_EXTENSION_H", f"-I{inc}", f"-I{tinc}",
                        f"-I{tinc / 'torch/csrc/api/include'}", f"-I{cuda_inc}",
                        f"-I{sysconfig.get_paths()['include']}", str(SRC / name)], capture_output=True, text=True)
    errors = [ln for ln in r.stderr.splitlines() if " error" in ln or ln.startswith("error")]
    assert r.returncode == 0 and not errors, "\n".join(errors[:20])
