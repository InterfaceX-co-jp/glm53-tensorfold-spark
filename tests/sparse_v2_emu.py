"""A CPU model of the Gluon ops ``sparse_v2._lsparse_v2`` uses (patches/0410), for offline bitwise tests.

Gluon kernels do not run under Triton's interpreter, so this module executes the kernel's own Python source (the
``GluonJITFunction``'s ``fn``) with ``gl`` / ``cp`` / ``mma_v2`` replaced by numpy implementations, one program at a
time. Layouts are ignored (they are checked on the compiled IR by tests/test_sparse_v2_compile.py); what is modelled:

- element-wise float32 arithmetic, bf16 rounding (nearest even) and the e4m3 decode, with the same numpy calls
  Triton's interpreter makes for the reference (np.sum / np.nanmax / np.exp / np.maximum on float32 arrays);
- ``mma_v2`` as ``DOT``: the per-element chain over K in 16-wide blocks from the accumulator c (the model the reference
  runs under in the interpreter too), so the chain's order and its starting value are compared, not just the sum;
- shared memory as bytes: ``_reinterpret`` views alias the ring slots exactly as on the GPU (logical byte ranges);
- cp.async: a copy is gathered at issue and lands at ``wait_group`` (all but the newest N committed groups);
- a hazard model standing in for the 8 warps: reading bytes that are still in flight, or that were written (store or
  landed copy) since the last ``gl.barrier()``, or writing / issuing into bytes read since the last barrier, raises
  ``Hazard``. A wrong slot, a missing wait or a missing barrier fails the test instead of passing by luck.
"""

from __future__ import annotations

import types

import numpy as np
import torch


class Hazard(AssertionError):
    pass


def f32_to_bf16_bits(x):
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(torch.bfloat16)
    return t.view(torch.int16).numpy().view(np.uint16)


def bf16_bits_to_f32(u):
    return (np.asarray(u).astype(np.uint32) << 16).view(np.float32)


def round_bf16(x):
    return bf16_bits_to_f32(f32_to_bf16_bits(x))


def e4m3_to_f32(u8):
    t = torch.from_numpy(np.ascontiguousarray(u8, dtype=np.uint8)).view(torch.float8_e4m3fn)
    return t.to(torch.float32).numpy()


def DOT(a, b, c):
    """The tensor-core chain model (shared with the reference's interpreter run): for each 16-wide K block in order,
    acc = fp32(acc + the block's exact products summed in float64). Order-sensitive across blocks, starts from c."""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    acc = np.array(c, dtype=np.float32, copy=True)
    K = a.shape[1]
    assert K % 16 == 0 and b.shape[0] == K
    for k0 in range(0, K, 16):
        acc = (acc.astype(np.float64) + a[:, k0:k0 + 16] @ b[k0:k0 + 16, :]).astype(np.float32)
    return acc


# -- dtypes ------------------------------------------------------------------------------------------------------
class DT:
    def __init__(self, name, np_store, size):
        self.name, self.np_store, self.size = name, np_store, size

    def __repr__(self):
        return self.name


INT1 = DT("i1", np.bool_, 1)
INT32 = DT("i32", np.int32, 4)
INT64 = DT("i64", np.int64, 8)
UINT8 = DT("u8", np.uint8, 1)
F8 = DT("f8e4nv", np.uint8, 1)
BF16 = DT("bf16", np.uint16, 2)
F32 = DT("f32", np.float32, 4)


def _vals_from_store(raw, dt):
    """Stored representation -> value array (bf16 as exact float32, f8 kept as bits)."""

    if dt is BF16:
        return bf16_bits_to_f32(raw)
    return np.asarray(raw)


def _store_from_vals(v, dt):
    if dt is BF16:
        return f32_to_bf16_bits(v)
    return np.asarray(v).astype(dt.np_store)


# -- tensors --------------------------------------------------------------------------------------------------------
class T:
    """A value tensor: numpy data + dtype tag (bf16 data held as exact float32 values, f8 as its bits)."""

    def __init__(self, a, dt):
        self.a = np.asarray(a)
        self.dt = dt

    @property
    def shape(self):
        return list(self.a.shape)

    def __bool__(self):
        return bool(self.a)

    def __index__(self):
        return int(self.a)

    def __int__(self):
        return int(self.a)

    def __getitem__(self, k):
        return T(self.a[k], self.dt)

    def to(self, dt, bitcast=False):
        if bitcast:
            assert dt.size == self.dt.size, (self.dt, dt)
            return T(self.a, dt)
        if dt is F32:
            if self.dt is F8:
                return T(e4m3_to_f32(self.a), F32)
            return T(self.a.astype(np.float32), F32)
        if dt is BF16:
            assert self.dt is F32
            return T(round_bf16(self.a), BF16)
        if dt in (INT32, INT64):
            return T(self.a.astype(dt.np_store), dt)
        raise NotImplementedError((self.dt, dt))

    # arithmetic
    def _bin(self, o, f, cmp=False, rev=False):
        if isinstance(o, P):
            return NotImplemented
        if isinstance(o, T):
            ob, odt = o.a, o.dt
        else:
            odt = None
            ob = o
        dt = self.dt
        if odt is not None and odt is not dt:
            order = [INT1, INT32, INT64, F32]
            dt = max(dt, odt, key=lambda d: order.index(d) if d in order else 99)
        if dt in (F32, BF16):
            dt = F32
            if not isinstance(ob, np.ndarray):
                ob = np.float32(ob)
            x, y = self.a.astype(np.float32), np.asarray(ob).astype(np.float32)
        elif dt in (INT32, INT64):
            x, y = self.a.astype(dt.np_store), np.asarray(ob).astype(dt.np_store)
        else:
            x, y = self.a, np.asarray(ob)
        if rev:
            x, y = y, x
        with np.errstate(all="ignore"):
            r = f(x, y)
        return T(r, INT1 if cmp else dt)

    __add__ = lambda s, o: s._bin(o, np.add)
    __radd__ = lambda s, o: s._bin(o, np.add, rev=True)
    __sub__ = lambda s, o: s._bin(o, np.subtract)
    __rsub__ = lambda s, o: s._bin(o, np.subtract, rev=True)
    __mul__ = lambda s, o: s._bin(o, np.multiply)
    __rmul__ = lambda s, o: s._bin(o, np.multiply, rev=True)
    __truediv__ = lambda s, o: s._bin(o, np.divide)
    __rshift__ = lambda s, o: s._bin(o, np.right_shift)
    __lshift__ = lambda s, o: s._bin(o, np.left_shift)
    __and__ = lambda s, o: s._bin(o, np.bitwise_and)
    __or__ = lambda s, o: s._bin(o, np.bitwise_or)
    __mod__ = lambda s, o: s._bin(o, np.mod)
    __lt__ = lambda s, o: s._bin(o, np.less, cmp=True)
    __le__ = lambda s, o: s._bin(o, np.less_equal, cmp=True)
    __gt__ = lambda s, o: s._bin(o, np.greater, cmp=True)
    __ge__ = lambda s, o: s._bin(o, np.greater_equal, cmp=True)
    __eq__ = lambda s, o: s._bin(o, np.equal, cmp=True)
    __ne__ = lambda s, o: s._bin(o, np.not_equal, cmp=True)
    __hash__ = None


class P:
    """A pointer tensor: a typed flat buffer and element offsets."""

    def __init__(self, buf, dt, off=None):
        self.buf, self.dt = buf, dt
        self.off = np.zeros((), np.int64) if off is None else np.asarray(off, dtype=np.int64)

    def __add__(self, o):
        return P(self.buf, self.dt, self.off + (o.a if isinstance(o, T) else np.int64(o)))

    __radd__ = __add__

    @property
    def shape(self):
        return list(self.off.shape)


def _gather(p: P, mask, other=0):
    off = p.off
    if mask is not None:
        m = np.broadcast_to(mask.a if isinstance(mask, T) else mask, off.shape)
        safe = np.where(m, off, 0)
        if m.any() and (safe[m].min() < 0 or safe[m].max() >= p.buf.size):
            raise IndexError("load out of bounds")
        v = p.buf[safe]
        v = np.where(m, v, np.asarray(other).astype(v.dtype))
    else:
        if off.size and (off.min() < 0 or off.max() >= p.buf.size):
            raise IndexError("load out of bounds")
        v = p.buf[off]
    return T(_vals_from_store(v, p.dt), p.dt)


# -- shared memory -----------------------------------------------------------------------------------------------------
class Arena:
    """Shared memory of one program: bytes, plus per-byte state for the hazard model."""

    def __init__(self, size=1 << 20):
        self.mem = np.zeros(size, np.uint8)
        self.top = 0
        self.pending = []              # groups: list of [(byte_idx array, bytes)]
        self.cur = []                  # copies issued since the last commit
        self.inflight = np.zeros(size, bool)
        self.dirty = np.zeros(size, bool)      # written since the last barrier
        self.read = np.zeros(size, bool)       # read since the last barrier

    def alloc(self, nbytes):
        base = (self.top + 127) // 128 * 128
        self.top = base + nbytes
        assert self.top <= self.mem.size
        return base

    def barrier(self):
        self.dirty[:] = False
        self.read[:] = False

    def on_read(self, idx):
        if self.inflight[idx].any():
            raise Hazard("read of shared memory with a cp.async still in flight")
        if self.dirty[idx].any():
            raise Hazard("read of shared memory written since the last barrier (other warps' writes not visible)")
        self.read[idx] = True

    def on_write(self, idx):
        if self.inflight[idx].any():
            raise Hazard("write to shared memory with a cp.async in flight")
        if self.read[idx].any():
            raise Hazard("write to shared memory read since the last barrier (other warps may still be reading)")
        self.dirty[idx] = True

    def issue(self, idx, data):
        if self.inflight[idx].any():
            raise Hazard("cp.async into bytes already in flight")
        if self.read[idx].any():
            raise Hazard("cp.async into shared memory read since the last barrier")
        self.inflight[idx] = True
        self.cur.append((idx, data))

    def commit(self):
        self.pending.append(self.cur)
        self.cur = []

    def wait(self, n):
        assert not self.cur, "wait_group with uncommitted copies"
        while len(self.pending) > n:
            for idx, data in self.pending.pop(0):
                self.mem[idx] = data
                self.inflight[idx] = False
                self.dirty[idx] = True        # landed: visible to other threads only after a barrier


class S:
    """A shared-memory descriptor: element dtype, shape, byte offset in the arena (row-major; swizzles ignored), and an
    optional view on top (``permute`` order, then ``slice`` ranges in the permuted coordinates)."""

    def __init__(self, arena, dt, shape, base, perm=None, cut=None):
        self.arena, self.dt, self.sh, self.base = arena, dt, list(shape), base
        self.perm = perm
        self.cut = cut                                          # [(start, length)] per dim of the permuted shape

    @property
    def shape(self):
        s = [self.sh[i] for i in self.perm] if self.perm else list(self.sh)
        if self.cut:
            s = [c[1] for c in self.cut]
        return s

    def nbytes(self):
        return int(np.prod(self.sh)) * self.dt.size

    def _idx(self):
        return self.base + np.arange(self.nbytes())

    def _plain(self):
        return self.perm is None and self.cut is None

    def index(self, i):
        assert self._plain()
        i = int(i)
        sub = self.sh[1:]
        assert 0 <= i < self.sh[0]
        return S(self.arena, self.dt, sub, self.base + i * int(np.prod(sub)) * self.dt.size)

    def permute(self, order):
        assert self._plain()
        return S(self.arena, self.dt, self.sh, self.base, list(order))

    def slice(self, start, length, dim=0):
        cut = list(self.cut) if self.cut else [(0, n) for n in self.shape]
        s0, n0 = cut[dim]
        assert 0 <= start and start + length <= n0
        cut[dim] = (s0 + start, length)
        return S(self.arena, self.dt, self.sh, self.base, self.perm, cut)

    def _reinterpret(self, dt, shape, layout=None):
        assert int(np.prod(shape)) * dt.size <= self.nbytes() and self._plain()
        return S(self.arena, dt, shape, self.base)

    def load(self, layout=None):
        idx = self._idx()
        self.arena.on_read(idx)                                 # the whole buffer (coarse: reads never overlap writes)
        raw = self.arena.mem[idx].view(self.dt.np_store).reshape(self.sh)
        v = _vals_from_store(raw, self.dt)
        if self.perm:
            v = np.transpose(v, self.perm)
        if self.cut:
            v = v[tuple(slice(a, a + n) for a, n in self.cut)]
        return T(np.array(v), self.dt)

    def store(self, value):
        assert self._plain()
        v = value.a if isinstance(value, T) else np.asarray(value)
        assert list(v.shape) == self.sh, (v.shape, self.sh)
        idx = self._idx()
        self.arena.on_write(idx)
        self.arena.mem[idx] = np.ascontiguousarray(_store_from_vals(v, self.dt)).view(np.uint8).reshape(-1)


# -- the fake modules ------------------------------------------------------------------------------------------------
class Program:
    """State of the program being emulated (program id, its arena)."""

    pid = 0
    arena: Arena | None = None


def _layout(*a, **k):
    return None


def _mk_gl():
    g = types.SimpleNamespace()
    g.constexpr = object
    g.int1, g.int32, g.int64, g.uint8, g.float8e4nv, g.bfloat16, g.float32 = INT1, INT32, INT64, UINT8, F8, BF16, F32
    for name in ("NVMMADistributedLayout", "DotOperandLayout", "BlockedLayout", "SliceLayout", "SwizzledSharedLayout"):
        setattr(g, name, _layout)
    g.program_id = lambda axis: Program.pid
    g.static_range = range
    g.cdiv = lambda a, b: -(-int(a) // int(b))
    g.num_warps = lambda: 8

    def load(ptr, mask=None, other=0):
        return _gather(ptr, mask, other)

    def store(ptr, value, mask=None):
        v = value.a if isinstance(value, T) else np.asarray(value)
        v = np.broadcast_to(v, ptr.off.shape)
        off = ptr.off
        if mask is not None:
            m = np.broadcast_to(mask.a, off.shape)
            off, v = off[m], v[m]
        ptr.buf[off.reshape(-1)] = _store_from_vals(v.reshape(-1), ptr.dt)

    def arange(start, end, layout=None):
        return T(np.arange(start, end, dtype=np.int32), INT32)

    def full(shape, value, dtype, layout=None):
        return T(np.full(shape, value, dtype=dtype.np_store if dtype is not BF16 else np.float32), dtype)

    def zeros(shape, dtype, layout=None):
        return full(shape, 0, dtype)

    def where(c, x, y):
        ca = c.a if isinstance(c, T) else c
        xs = [v for v in (x, y) if isinstance(v, T)]
        dt = F32 if any(v.dt in (F32, BF16) for v in xs) or any(isinstance(v, float) for v in (x, y)) else xs[0].dt
        xa = x.a if isinstance(x, T) else x
        ya = y.a if isinstance(y, T) else y
        r = np.where(ca, xa, ya)
        return T(r.astype(np.float32) if dt is F32 else r, dt)

    def reduce_max(x, axis):
        return T(np.nanmax(x.a, axis=axis), x.dt)

    def reduce_sum(x, axis):
        return T(np.sum(x.a, axis=axis), x.dt)

    def maximum(x, y):
        return T(np.maximum(x.a, y.a), x.dt)

    def exp(x):
        with np.errstate(all="ignore"):
            return T(np.exp(x.a), x.dt)

    def expand_dims(x, axis):
        return T(np.expand_dims(x.a, axis), x.dt)

    def broadcast(a, b):
        x, y = np.broadcast_arrays(a.a, b.a)
        return T(np.array(x), a.dt), T(np.array(y), b.dt)

    def allocate_shared_memory(dt, shape, layout, value=None):
        ar = Program.arena
        s = S(ar, dt, shape, 0)
        s.base = ar.alloc(s.nbytes())
        if value is not None:
            s.store(value)
        return s

    g.load, g.store, g.arange, g.full, g.zeros, g.where = load, store, arange, full, zeros, where
    g.max, g.sum, g.maximum, g.exp = reduce_max, reduce_sum, maximum, exp
    g.expand_dims, g.broadcast = expand_dims, broadcast
    g.convert_layout = lambda x, layout, assert_trivial=False: x
    g.allocate_shared_memory = allocate_shared_memory
    g.barrier = lambda **k: Program.arena.barrier()
    return g


def _mk_cp():
    c = types.SimpleNamespace()

    def async_copy_global_to_shared(smem, pointer, mask=None):
        assert smem._plain() and list(pointer.off.shape) == smem.sh
        v = _gather(pointer, mask, 0)                          # masked lanes: zero-filled (cp.async src-size 0)
        data = np.ascontiguousarray(_store_from_vals(v.a, smem.dt)).view(np.uint8).reshape(-1)
        Program.arena.issue(smem._idx(), data)

    c.async_copy_global_to_shared = async_copy_global_to_shared
    c.commit_group = lambda: Program.arena.commit()
    c.wait_group = lambda n=0: Program.arena.wait(int(n))
    return c


def _mma_v2(a, b, acc, input_precision=None):
    assert a.dt is BF16 and b.dt is BF16 and acc.dt is F32
    return T(DOT(a.a, b.a, acc.a), F32)


GL, CP = _mk_gl(), _mk_cp()


def emulated(jitfn, extra=None):
    """The Python function of a Gluon kernel with the fake modules; every Gluon function of its module (its helpers)
    runs on the fakes too, all sharing one globals dict."""

    fn = jitfn.fn
    g = dict(fn.__globals__)
    g.update(gl=GL, cp=CP, mma_v2=_mma_v2)
    for name, v in list(g.items()):
        if type(v).__name__ == "GluonJITFunction":
            f = v.fn
            g[name] = types.FunctionType(f.__code__, g, f.__name__, f.__defaults__, f.__closure__)
    if extra:
        g.update(extra)
    return types.FunctionType(fn.__code__, g, fn.__name__, fn.__defaults__, fn.__closure__)


def run_v2(sparse_v2, qa, lc, tokens, counts, out, scale, stages, qkl, qreg=0):
    """Emulate ``sparse_v2._lsparse_v2`` over every row (grid (R,)) on CPU torch tensors, with the host wrapper's
    argument rules (``latent._views`` / ``_paging``: FP8 or bf16 rows, contiguous or a patches/0290 paged cache);
    ``out`` is updated in place."""

    from tensorfold.families.glm5_next.cuda import latent

    R, H, L = qa.shape
    W = tokens.shape[1]
    lcv, lcs, fp8, rb, sb = latent._views(lc, L)
    pg = latent._paging(lc)
    qbuf = qa.contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1)
    if fp8:
        lbuf = lcv.contiguous().numpy().reshape(-1)
        lp, sp = P(lbuf, UINT8), P(lbuf.view(np.float32), F32)
    else:
        lbuf = lcv.contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1)
        lp = sp = P(lbuf, BF16)
        qkl = 0
    qreg = qreg if qkl else 0
    ptp = None if pg["PT"] is None else P(pg["PT"].contiguous().numpy().reshape(-1), INT32)
    obuf = out.numpy().reshape(-1)                              # shares memory with ``out``
    kern = emulated(sparse_v2._lsparse_v2)
    for r in range(R):
        Program.pid = r
        Program.arena = Arena()
        kern(P(qbuf, BF16), lp, sp, P(tokens.contiguous().numpy().reshape(-1), INT32),
             P(counts.contiguous().numpy().reshape(-1), INT32), P(obuf, F32), ptp, W, H=H, L=L, KTS=latent.KT,
             SCALE=scale, FP8=fp8, RB=rb, SB=sb, SA=latent.SCALE_AT // 4, PSH=pg["PSH"], STAGES=stages, QKL=qkl, QREG=qreg)
        ar = Program.arena
        assert not ar.cur and not ar.pending, "program ended with cp.async groups outstanding"
