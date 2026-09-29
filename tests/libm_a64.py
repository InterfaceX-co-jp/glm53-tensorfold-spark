"""patches/0450: the sampler's libm port against the aarch64 libm itself, executed in unicorn (no aarch64 host needed).

``gpusample._log`` / ``_exp`` are glibc 2.39's ``__log`` / ``__exp`` as Ubuntu 24.04 builds them for arm64 (the image's
libm: GCC contracted glibc's C into ``fmadd`` / ``fmsub`` at places no source shows, so the port follows the binary).
This file runs that binary's functions (the GLIBC_2.29 symbols numpy calls) on the CPU emulator and compares them with
the port (in Triton's interpreter) on the sampler's inputs and on random doubles.

Needs ``pip install unicorn pyelftools`` and the library:
    curl -O http://ports.ubuntu.com/ubuntu-ports/pool/main/g/glibc/libc6_2.39-0ubuntu8.9_arm64.deb
    ar x libc6_*.deb && tar xf data.tar.* ./usr/lib/aarch64-linux-gnu/libm.so.6
    GLM53_TF_LIBM_A64=$PWD/usr/lib/aarch64-linux-gnu/libm.so.6 TRITON_INTERPRET=1 \\
        PYTHONPATH=<patched TensorFold>/src:tests pytest -q tests/libm_a64.py
(2026-09-29: 10.5M comparisons of the scalar port and 5.4M of the Triton port (3 seeds x 1.8M), 0 differences.)
"""

from __future__ import annotations

import os
import struct
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")
sys.path.insert(0, os.path.dirname(__file__))

import pytest  # noqa: E402

LIBM = os.environ.get("GLM53_TF_LIBM_A64", "")
pytestmark = pytest.mark.skipif(not LIBM or not os.path.exists(LIBM), reason="GLM53_TF_LIBM_A64: an aarch64 libm.so.6")
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
unicorn = pytest.importorskip("unicorn")
elf = pytest.importorskip("elftools.elf.elffile")

STOP = 0x7000000


class A64Libm:
    """The library's ``log`` / ``exp`` (the GLIBC_2.29 versions) called in unicorn."""

    def __init__(self, path: str) -> None:
        from unicorn import UC_ARCH_ARM64, UC_MODE_ARM, Uc
        from unicorn.arm64_const import UC_ARM64_REG_CPACR_EL1, UC_ARM64_REG_TPIDR_EL0

        data = open(path, "rb").read()
        with open(path, "rb") as f:
            e = elf.ELFFile(f)
            self.fn = {}
            for sym in e.get_section_by_name(".dynsym").iter_symbols():
                if sym.name in ("log", "exp") and sym["st_size"]:
                    # the larger body is the GLIBC_2.29 one (the older symbol is an errno wrapper around it)
                    if sym["st_size"] > self.fn.get(sym.name, (0, 0))[1]:
                        self.fn[sym.name] = (sym["st_value"], sym["st_size"])
            segs = [(s["p_vaddr"], s["p_offset"], s["p_filesz"], s["p_memsz"]) for s in e.iter_segments()
                    if s["p_type"] == "PT_LOAD"]
        self.uc = uc = Uc(UC_ARCH_ARM64, UC_MODE_ARM)
        uc.reg_write(UC_ARM64_REG_CPACR_EL1, 3 << 20)
        for va, off, fsz, msz in segs:
            lo, hi = va & ~0xFFF, (va + msz + 0xFFF) & ~0xFFF
            uc.mem_map(lo, hi - lo)
            uc.mem_write(va, data[off:off + fsz])
        uc.mem_map(0x6000000, 0x100000)
        uc.mem_map(STOP, 0x1000)
        uc.mem_map(0x5000000, 0x10000)
        uc.reg_write(UC_ARM64_REG_TPIDR_EL0, 0x5008000)

    def __call__(self, name: str, x: float) -> float:
        from unicorn.arm64_const import UC_ARM64_REG_D0, UC_ARM64_REG_SP, UC_ARM64_REG_X30

        uc = self.uc
        uc.reg_write(UC_ARM64_REG_SP, 0x60F0000)
        uc.reg_write(UC_ARM64_REG_X30, STOP)
        uc.reg_write(UC_ARM64_REG_D0, struct.unpack("<Q", struct.pack("<d", x))[0])
        uc.emu_start(self.fn[name][0], STOP)
        return struct.unpack("<d", struct.pack("<Q", uc.reg_read(UC_ARM64_REG_D0)))[0]


@pytest.fixture(scope="module")
def a64():
    return A64Libm(LIBM)


def _inputs(rng, n: int):
    q = rng.integers(0, 1 << 53, size=n, dtype=np.int64)
    u = q.astype(np.float64) * 2.0 ** -53 + 2.0 ** -54
    return {
        "log": np.concatenate([u, np.exp(rng.uniform(-700, 700, size=n)), 1.0 + rng.uniform(-0.07, 0.07, size=n)]),
        "exp": np.concatenate([-np.abs(rng.standard_normal(n) * s) for s in (1e-3, 1.0, 30.0, 700.0, 745.0)]
                              + [rng.uniform(-1100, 709, size=n)]),
    }


def test_port_equals_the_aarch64_libm(a64):
    import gpuround_interp
    from tensorfold.families.glm5_next.cuda import gpusample as gs

    gpuround_interp.install()
    rng = np.random.default_rng(int(os.environ.get("GLM53_TF_LIBM_SEED", "5")))
    n = int(os.environ.get("GLM53_TF_LIBM_N", "20000"))
    for which, x in _inputs(rng, n).items():
        want = np.array([a64(which, float(v)) for v in x])
        got = gs.libm(torch.from_numpy(x), which).numpy()
        bad = np.nonzero(got.view(np.int64) != want.view(np.int64))[0]
        assert not len(bad), (which, x[bad[:4]], got[bad[:4]], want[bad[:4]])
