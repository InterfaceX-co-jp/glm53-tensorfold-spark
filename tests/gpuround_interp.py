"""patches/0450 test helper: Triton's CPU interpreter made fit for the GPU round's kernels.

- ``tl.fma``: the interpreter computes ``x * y + z`` with two roundings; the GPU's ``fma.rn.f64`` rounds once. The
  sampler's libm port (``gpusample._log`` / ``_exp``) is glibc's instruction sequence with its fused multiply-adds, so
  the interpreter must round once too: float64 operands go through ``math.fma`` (Python 3.13+; IEEE non-trapping: an
  overflow is an infinity, an invalid operation a NaN, as on the GPU), else through an exact integer-ratio sum rounded
  by Python's correctly rounded int division.
- Kernel launches from several threads (the two-rank tests): the interpreter keeps its grid state in module globals,
  so launches are serialized.

``install()`` is idempotent; call it before any interpreted kernel runs (TRITON_INTERPRET=1 set before triton import).
"""

from __future__ import annotations

import math
import threading

import numpy as np

_LOCK = threading.RLock()
_DONE = False


def _fma_exact(a: float, b: float, c: float) -> float:
    """a * b + c rounded once (to nearest, ties to even) without ``math.fma`` (Python < 3.13, the image's 3.12): the
    exact sum of the integer ratios, divided by Python's correctly rounded int true division."""

    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
        with np.errstate(all="ignore"):
            return float(np.float64(a) * np.float64(b) + np.float64(c))     # inf / nan: no rounding involved
    na, da = a.as_integer_ratio()
    nb, db = b.as_integer_ratio()
    nc, dc = c.as_integer_ratio()
    num = na * nb * dc + nc * da * db
    if num == 0:                                   # an exact zero: its sign as IEEE gives it (a * b is exact then)
        with np.errstate(all="ignore"):
            return float(np.float64(a) * np.float64(b) + np.float64(c))
    try:
        return num / (da * db * dc)
    except OverflowError:
        return math.inf if num > 0 else -math.inf


def _fma1(a: float, b: float, c: float) -> float:
    if not hasattr(math, "fma"):
        return _fma_exact(float(a), float(b), float(c))
    try:
        return math.fma(a, b, c)
    except OverflowError:
        with np.errstate(all="ignore"):
            return float(np.float64(a) * np.float64(b) + np.float64(c))
    except ValueError:
        return math.nan


_VFMA = np.frompyfunc(_fma1, 3, 1)


def install() -> None:
    global _DONE
    if _DONE:
        return
    import triton.runtime.interpreter as interp

    def create_fma(self, x, y, z):
        a, b, c = x.data, y.data, z.data
        if np.asarray(c).dtype == np.float64:
            with np.errstate(all="ignore"):
                out = np.asarray(_VFMA(a, b, c), dtype=np.float64)
            return interp.TensorHandle(out, z.dtype.scalar)
        with np.errstate(all="ignore"):
            out = (np.asarray(a, dtype=np.float64) * b + c).astype(np.asarray(c).dtype)
        return interp.TensorHandle(out, z.dtype.scalar)

    interp.InterpreterBuilder.create_fma = create_fma
    call = interp.GridExecutor.__call__

    def locked(self, *args, **kwargs):
        with _LOCK, np.errstate(all="ignore"):
            return call(self, *args, **kwargs)

    interp.GridExecutor.__call__ = locked
    _DONE = True
