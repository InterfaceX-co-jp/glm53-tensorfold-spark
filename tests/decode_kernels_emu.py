"""patches/0440: a CPU emulation of the streaming decode kernels (``exl3_stream.cu``, ``q4_stream.cu``) and of the
kernels whose bits they must keep (exl3.cu's ``grouped_kernel`` + epilogues; qmm.py's ``_qmm`` + ``_reduce`` as
Triton compiles them), for ``tests/test_decode_kernels_emulator.py``.

Two levels, on purpose:

- the references are written at MATRIX level from the reference sources: exl3.cu's work item (member tile x K split x
  column block; warp w's k tiles in order; warps added 0..3; Z; the epilogues summing splits from 0.f), and ``_qmm``'s
  per-group chain as its PTX shows it (four chained m16n8k16 from +0.0, then fma(xs, b, fma(p, s, acc)); slices added
  in order);
- the new kernels are ported at LANE level, line for line from the CUDA: the per-warp (E1) / per-CTA (E2) shared-memory
  rings with their slot arithmetic and swizzles, 16-byte copies that land at random times between issue and the
  ``cp.async.wait_group`` that covers them (and never later), the fragment reads by lane, the item walks (static
  stride, the E1 extra-member-tile list), the tickets under random CTA interleavings, the stale shared memory of
  earlier items (the ring starts as NaN garbage; dead member rows are never copied).

The tensor-core mma is replaced by an ORDER-SENSITIVE model (``mma_model``: the 16 exact products added to the
accumulator one k at a time, fp32 rounding each step) through the canonical m16n8k16 fragment layout. Real hardware
adds differently, but identically for both kernels when every mma gets the same operands in the same positions -- which
is the property under test: a k position moved inside a fragment, a chunk out of order, a different reduction order or a
stale row changes the model's bits. fp32 fma is emulated exactly (float64 product, round-to-odd sum, then fp32).
"""

from __future__ import annotations

import numpy as np

M32 = np.uint64(0xFFFFFFFF)
LANE = np.arange(32)
G = LANE >> 2
T = LANE & 3


# -- floating point --------------------------------------------------------------------------------------------------
def fma32(a, b, c) -> np.ndarray:
    """fp32 fma(a, b, c), one rounding (nearest-even): exact float64 product, float64 sum rounded to odd, then fp32."""

    a, b, c = (np.asarray(v, dtype=np.float32) for v in (a, b, c))
    p = a.astype(np.float64) * b.astype(np.float64)
    c = np.broadcast_to(c.astype(np.float64), p.shape)
    with np.errstate(invalid="ignore", over="ignore"):
        s = p + c
        bb = s - p
        err = (p - (s - bb)) + (c - bb)
    even = (s.view(np.int64) & 1) == 0
    fix = (err != 0) & even & np.isfinite(s)
    if fix.any():
        s = np.where(fix, np.nextafter(s, np.where(err > 0, np.inf, -np.inf)), s)
    with np.errstate(over="ignore"):
        return s.astype(np.float32)


def bf16_to_f32(u16) -> np.ndarray:
    return (np.asarray(u16, dtype=np.uint32) << 16).view(np.float32)


def f32_to_bf16(x) -> np.ndarray:
    """fp32 -> bf16 bits, round to nearest even (cvt.rn.bf16.f32; NaN kept NaN)."""

    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + np.uint64(0x7FFF) + ((u >> np.uint64(16)) & np.uint64(1))) >> np.uint64(16)).astype(np.uint16)
    nan = np.isnan(np.asarray(x, dtype=np.float32))
    return np.where(nan, np.uint16(0x7FC0), r)


def bf16r(x) -> np.ndarray:
    return bf16_to_f32(f32_to_bf16(x))


def f16_to_f32(u16) -> np.ndarray:
    return np.asarray(u16, dtype=np.uint16).view(np.float16).astype(np.float32)


def f32_to_f16(x) -> np.ndarray:
    with np.errstate(over="ignore"):
        return np.asarray(x, dtype=np.float32).astype(np.float16).view(np.uint16)


def halves(u32, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """A 32-bit register as its two 16-bit values (low half first) in fp32."""

    u32 = np.asarray(u32, dtype=np.uint32)
    lo, hi = (u32 & 0xFFFF).astype(np.uint16), (u32 >> 16).astype(np.uint16)
    if kind == "f16":
        return f16_to_f32(lo), f16_to_f32(hi)
    return bf16_to_f32(lo), bf16_to_f32(hi)


# -- the mma model -----------------------------------------------------------------------------------------------------
def mma_model(C: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """D[16, 8] = C + A[16, 16] @ B[16, 8], the products (exact in fp32 for fp16 / bf16 inputs) added one k at a time
    in k order with an fp32 rounding each step: order- and position-sensitive by construction."""

    acc = C.astype(np.float32).copy()
    with np.errstate(invalid="ignore", over="ignore"):
        for k in range(16):
            acc = (acc + (A[:, k:k + 1] * B[k:k + 1, :]).astype(np.float32)).astype(np.float32)
    return acc


def frag_a(a: np.ndarray, kind: str) -> np.ndarray:
    """A [16, 16] from the four A registers of 32 lanes (a[lane, 0..3]): a0 row g k 2t..2t+1, a1 row g + 8, a2 row g
    k 2t + 8.., a3 row g + 8 k 2t + 8.. (low half = lower k)."""

    A = np.full((16, 16), np.nan, dtype=np.float32)
    for r, (dr, dk) in enumerate(((0, 0), (8, 0), (0, 8), (8, 8))):
        lo, hi = halves(a[:, r], kind)
        A[G + dr, dk + 2 * T] = lo
        A[G + dr, dk + 2 * T + 1] = hi
    return A


def frag_b(b0: np.ndarray, b1: np.ndarray, kind: str) -> np.ndarray:
    """B [16, 8] (k, n) from the two B registers of 32 lanes: b0 k 2t..2t+1 of column g, b1 k 2t + 8.."""

    B = np.full((16, 8), np.nan, dtype=np.float32)
    for reg, dk in ((b0, 0), (b1, 8)):
        lo, hi = halves(reg, kind)
        B[dk + 2 * T, G] = lo
        B[dk + 2 * T + 1, G] = hi
    return B


def frag_d(D: np.ndarray) -> np.ndarray:
    """[16, 8] -> the C / D registers of 32 lanes: d0 (g, 2t), d1 (g, 2t + 1), d2 (g + 8, 2t), d3 (g + 8, 2t + 1)."""

    return np.stack([D[G, 2 * T], D[G, 2 * T + 1], D[G + 8, 2 * T], D[G + 8, 2 * T + 1]], axis=1).astype(np.float32)


def unfrag_d(d: np.ndarray) -> np.ndarray:
    D = np.full((16, 8), np.nan, dtype=np.float32)
    D[G, 2 * T], D[G, 2 * T + 1], D[G + 8, 2 * T], D[G + 8, 2 * T + 1] = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    return D


def mma_frag(d: np.ndarray, a: np.ndarray, b0: np.ndarray, b1: np.ndarray, kind: str) -> np.ndarray:
    """mma.m16n8k16 on lane registers (d [32, 4] fp32 in place semantics: returns the new d)."""

    return frag_d(mma_model(unfrag_d(d), frag_a(a, kind), frag_b(b0, b1, kind)))


# -- EXL3 trellis decode (exl3.cu) -------------------------------------------------------------------------------------
def _hadd2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xl, xh = halves(x, "f16")
    yl, yh = halves(y, "f16")
    lo = (xl.astype(np.float16) + yl.astype(np.float16)).view(np.uint16).astype(np.uint32)
    hi = (xh.astype(np.float16) + yh.astype(np.float16)).view(np.uint16).astype(np.uint32)
    return (lo | (hi << 16)).astype(np.uint32)


def mcg2(s0, s1) -> np.ndarray:
    s0 = np.asarray(s0, dtype=np.uint64)
    s1 = np.asarray(s1, dtype=np.uint64)
    x0 = (s0 * np.uint64(0xCBAC1FED)) & M32
    x1 = (s1 * np.uint64(0xCBAC1FED)) & M32
    x0 = (x0 & np.uint64(0x8FFF8FFF)) ^ np.uint64(0x3B603B60)
    x1 = (x1 & np.uint64(0x8FFF8FFF)) ^ np.uint64(0x3B603B60)
    lo = (x0 & np.uint64(0xFFFF)) | ((x1 & np.uint64(0xFFFF)) << np.uint64(16))          # __byte_perm 0x5410
    hi = (x0 >> np.uint64(16)) | (x1 & np.uint64(0xFFFF0000))                           # __byte_perm 0x7632
    with np.errstate(over="ignore"):
        return _hadd2(lo.astype(np.uint32), hi.astype(np.uint32))


def decode_tile(w: np.ndarray):
    """exl3.cu's decode_tile for the 32 lanes of a warp (w[lane] = the tile's word lane): b0[2], b1[2] per lane."""

    w64 = np.asarray(w, dtype=np.uint32).astype(np.uint64)
    p = w64[(LANE + 31) & 31]                                                          # __shfl_sync(w, lane - 1)
    s = ((w64 >> np.uint64(20)) | (p << np.uint64(12))) & M32                          # __funnelshift_r(w, p, 20)
    F = np.uint64(0xFFFF)
    b00 = mcg2((s >> np.uint64(8)) & F, (s >> np.uint64(4)) & F)
    b01 = mcg2(s & F, w64 >> np.uint64(16))
    b10 = mcg2((w64 >> np.uint64(12)) & F, (w64 >> np.uint64(8)) & F)
    b11 = mcg2((w64 >> np.uint64(4)) & F, w64 & F)
    return (b00, b01), (b10, b11)


def tile_matrix(words: np.ndarray) -> np.ndarray:
    """A 16x16 trellis tile (32 words) as the (k, n) matrix its decode feeds to the two n8 halves' mma."""

    (b00, b01), (b10, b11) = decode_tile(words)
    return np.concatenate([frag_b(b00, b01, "f16"), frag_b(b10, b11, "f16")], axis=1)


def fwht128(v: np.ndarray) -> np.ndarray:
    """exl3.cu's fwht128 for 32 lanes x 4 values (lane L: values 4L..4L+3), fp32."""

    v = v.astype(np.float32).copy()
    a, b, c, d = v[:, 0] + v[:, 1], v[:, 0] - v[:, 1], v[:, 2] + v[:, 3], v[:, 2] - v[:, 3]
    v[:, 0], v[:, 1], v[:, 2], v[:, 3] = a + c, b + d, a - c, b - d
    m = 1
    while m < 32:
        o = v[LANE ^ m]
        up = (LANE & m) != 0
        v = np.where(up[:, None], o - v, v + o).astype(np.float32)
        m <<= 1
    return v


HAD = np.float32(0.08838834764831845)


def gateup_epilogue(Z, e, svh_g, svh_u, suh_d, xd, p, blk, P, N, SK, limit):
    """exl3.cu's gateup_epilogue body for member row p and Hadamard block blk (Z flat fp32; xd [P, N] fp16 bits)."""

    n = blk * 128 + 4 * LANE
    gv = np.zeros((32, 4), np.float32)
    uv = np.zeros((32, 4), np.float32)
    for j in range(4):
        sg = np.zeros(32, np.float32)
        su = np.zeros(32, np.float32)
        for s in range(SK):
            sg = (sg + Z[((0 * SK + s) * P + p) * N + n + j]).astype(np.float32)
            su = (su + Z[((1 * SK + s) * P + p) * N + n + j]).astype(np.float32)
        gv[:, j], uv[:, j] = sg, su
    gv, uv = fwht128(gv), fwht128(uv)
    v = np.zeros((32, 4), np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        for j in range(4):
            gg = np.minimum(bf16r(gv[:, j] * HAD * f16_to_f32(svh_g[e, n + j])), np.float32(limit))
            uu = np.minimum(np.maximum(bf16r(uv[:, j] * HAD * f16_to_f32(svh_u[e, n + j])), np.float32(-limit)),
                            np.float32(limit))
            act = bf16r(bf16r(gg / (np.float32(1) + np.exp(-gg).astype(np.float32))) * uu)
            v[:, j] = act * f16_to_f32(suh_d[e, n + j])
    v = fwht128(v)
    for j in range(4):
        xd[p, n + j] = f32_to_f16(v[:, j] * HAD)


def down_epilogue(Z, e, svh_d, y, p, blk, P, D, SK):
    n = blk * 128 + 4 * LANE
    v = np.zeros((32, 4), np.float32)
    for j in range(4):
        s = np.zeros(32, np.float32)
        for k in range(SK):
            s = (s + Z[(k * P + p) * D + n + j]).astype(np.float32)
        v[:, j] = s
    v = fwht128(v)
    for j in range(4):
        y[p, n + j] = (v[:, j] * HAD * f16_to_f32(svh_d[e, n + j])).astype(np.float32)


def rot_in(x_bf16, pick, suh0, suh1, K, slots, P):
    """exl3.cu's rot_in_kernel: Xh[mat][row * slots + slot] (fp16 bits) for every routed slot."""

    out = [np.zeros((P, K), np.uint16), np.zeros((P, K), np.uint16)]
    rows = x_bf16.shape[0]
    for mat, suh in enumerate((suh0, suh1)):
        for row in range(rows):
            for slot in range(slots - 1):
                e = pick[row, slot]
                for blk in range(K // 128):
                    idx = blk * 128 + 4 * LANE
                    v = np.stack([bf16_to_f32(x_bf16[row, idx + j]) * f16_to_f32(suh[e, idx + j]) for j in range(4)], 1)
                    v = fwht128(v)
                    for j in range(4):
                        out[mat][row * slots + slot, idx + j] = f32_to_f16(v[:, j] * HAD)
    return out


# -- E1 reference: exl3.cu grouped_kernel (member_tiles<NT, W, 1>) at matrix level ---------------------------------------
def exl3_reference(X0, X1, T0, T1, uids, ucount, members, K, N, P, SK, slots, E, mats, W=4):
    """Z [mats, SK, P, N] fp32 as exl3.cu's grouped kernels leave it (rows that are no member: NaN)."""

    KT, NTILES = K // 16, N // 16
    per_split, maxm = KT // SK, members.shape[1]
    PW = per_split // W
    Z = np.full((mats, SK, P, N), np.nan, np.float32)
    for u in range(int(ucount)):
        e = int(uids[u])
        if e >= E:
            continue
        for mt in range((maxm + 15) // 16):
            codes = [int(members[u, mt * 16 + i]) if mt * 16 + i < maxm else -1 for i in range(16)]
            rows = [(c >> 5) * slots + (c & 31) if c >= 0 else -1 for c in codes]
            if rows[0] < 0:
                continue
            for mat in range(mats):
                X, Tw = (X1, T1) if mat else (X0, T0)
                for split in range(SK):
                    parts = []
                    for w in range(W):
                        kt0 = split * per_split + w * PW
                        acc = np.zeros((16, N), np.float32)
                        for kt in range(kt0, kt0 + PW):
                            A = np.zeros((16, 16), np.float32)
                            for i, r in enumerate(rows):
                                if r >= 0:
                                    A[i] = f16_to_f32(X[r, kt * 16:kt * 16 + 16])
                            for nt in range(NTILES):
                                Bm = tile_matrix(Tw[e, kt, nt])
                                for h in range(2):
                                    c0 = nt * 16 + h * 8
                                    acc[:, c0:c0 + 8] = mma_model(acc[:, c0:c0 + 8], A, Bm[:, h * 8:h * 8 + 8])
                        parts.append(acc)
                    s = parts[0].copy()
                    for w in range(1, W):
                        s = (s + parts[w]).astype(np.float32)
                    for i, r in enumerate(rows):
                        if r >= 0:
                            Z[mat, split, r] = s[i]
    return Z


def exl3_routed_reference(x, pick, uids, ucount, members, ex, K_D, NI, slots, E, SKg, SKd, limit):
    """exl3_mm.routed's non-fast path (rot_in, grouped, gateup_epilogue, grouped, down_epilogue): (xd, y) for every
    routed pair (fp16 bits / fp32)."""

    rows = x.shape[0]
    P = rows * slots
    xg, xu = rot_in(x, pick, ex["suh_g"], ex["suh_u"], K_D, slots, P)
    Zg = exl3_reference(xg, xu, ex["gt"], ex["ut"], uids, ucount, members, K_D, NI, P, SKg, slots, E, 2)
    xd = np.zeros((P, NI), np.uint16)
    Zf = Zg.reshape(-1)
    for row in range(rows):
        for slot in range(slots - 1):
            p = row * slots + slot
            e = pick[row, slot]
            for blk in range(NI // 128):
                gateup_epilogue(Zf, e, ex["svh_g"], ex["svh_u"], ex["suh_d"], xd, p, blk, P, NI, SKg, limit)
    Zd = exl3_reference(xd, xd, ex["dt"], ex["dt"], uids, ucount, members, NI, K_D, P, SKd, slots, E, 1)
    y = np.full((P, K_D), np.nan, np.float32)
    Zf = Zd.reshape(-1)
    for row in range(rows):
        for slot in range(slots - 1):
            p = row * slots + slot
            for blk in range(K_D // 128):
                down_epilogue(Zf, pick[row, slot], ex["svh_d"], y, p, blk, P, K_D, SKd)
    return xd, y


# -- asynchronous copies ---------------------------------------------------------------------------------------------
class AsyncCopies:
    """cp.async groups of one thread group (a warp or a CTA): copies land at random times after their issue and no
    later than the ``wait(n)`` that leaves at most n newer groups pending."""

    def __init__(self, mem: np.ndarray, rng: np.random.Generator) -> None:
        self.mem, self.rng = mem, rng
        self.open: list = []
        self.groups: list[list] = []

    def copy(self, dst: int, data: np.ndarray) -> None:
        self.open.append((dst, np.array(data, copy=True)))

    def commit(self) -> None:
        self.groups.append(self.open)
        self.open = []

    def _land(self, grp) -> None:
        for dst, data in grp:
            self.mem[dst:dst + data.size] = data

    def jitter(self) -> None:
        """Some pending copies land now (any subset, in any order)."""

        keep = []
        for grp in self.groups:
            rest = []
            for c in grp:
                if self.rng.random() < 0.3:
                    self._land([c])
                else:
                    rest.append(c)
            keep.append(rest)
        self.groups = keep

    def wait(self, n: int) -> None:
        while len(self.groups) > n:
            self._land(self.groups.pop(0))


# -- E1 port: exl3_stream.cu ---------------------------------------------------------------------------------------------
def a_word(row, half, t):
    return row * 8 + ((half ^ ((row >> 2) & 1)) << 2) + t


class StreamLaunch:
    def __init__(self, X0, X1, T0, T1, uids, ucount, members, K, N, P, SK, slots, E, mats, epi):
        self.X0, self.X1, self.T0, self.T1 = X0, X1, T0, T1
        self.uids, self.ucount, self.members = uids, ucount, members
        self.K, self.N, self.P, self.SK, self.slots, self.E, self.mats = K, N, P, SK, slots, E, mats
        self.maxm = members.shape[1]
        self.epi = epi                         # (Z flat, e, p, hb) -> None


MAX_EXTRA = 1024


def exl3_stream_port(L: StreamLaunch, NT: int, STAGES: int, grid: int, cnt: np.ndarray, rng: np.random.Generator,
                     W: int = 4, mutate: str = "") -> tuple[np.ndarray, int]:
    """Run exl3_stream.cu's stream_kernel<NT, STAGES, MODE, 0> over ``grid`` CTAs, CTAs interleaved at random item by
    item. Returns (Z flat, epilogues run). Z starts as NaN; ``L.epi`` receives every epilogue call.

    ``mutate`` (negative controls, each must break the bits): "wait" waits for one group too few; "swizzle" reads the
    A block without the swizzle; "warps" adds the warps in reverse order; "rows" copies one live member row too few."""

    KT, NTILES = L.K >> 4, L.N >> 4
    PW = KT // (L.SK * W)
    SLOT = NT * 32 + 128
    Z = np.full(L.mats * L.SK * L.P * L.N, np.nan, np.float32)
    epis = 0

    # -- the walk (every CTA computes the same one)
    nexp = int(L.ucount)
    if nexp > 0 and int(L.uids[nexp - 1]) >= L.E:
        nexp -= 1
    NB = L.N // (16 * NT)
    ipe = L.mats * L.SK * NB
    MT = (L.maxm + 15) // 16
    extra = []
    if MT > 1:
        cand = (MT - 1) * nexp
        live = [int(L.members[j - ((1 + j // nexp) - 1) * nexp, (1 + j // nexp) * 16]) >= 0 for j in range(cand)]
        n = 0
        for j0 in range(0, cand, 32):
            on = [j0 + ln < cand and live[j0 + ln] for ln in range(32)]
            for ln in range(32):
                if on[ln]:
                    pos = n + sum(on[:ln])
                    if pos < MAX_EXTRA:
                        j = j0 + ln
                        mt, u = 1 + j // nexp, j - (j // nexp) * nexp
                        extra.append((mt << 10) | u)
            n += sum(on)
        n_items = (nexp + min(n, MAX_EXTRA)) * ipe
    else:
        n_items = nexp * ipe

    def item_at(it):
        pi = it // ipe
        r = it - pi * ipe
        if pi < nexp:
            mt, u = 0, pi
        else:
            x = extra[pi - nexp]
            mt, u = x >> 10, x & 1023
        ms = r // NB
        nb = r - ms * NB
        mat = ms // L.SK
        split = ms - mat * L.SK
        e = int(L.uids[u])
        rows = np.full(32, -1)
        for lane in range(16):
            m = mt * 16 + lane
            code = int(L.members[u, m]) if m < L.maxm else -1
            rows[lane] = (code >> 5) * L.slots + (code & 31) if code >= 0 else -1
        live = int(((rows >= 0) & (LANE < 16)).sum())
        return dict(u=u, e=e, mt=mt, mat=mat, split=split, nb=nb, row=rows, live=live)

    class Warp:
        def __init__(self, cta, w):
            self.cta, self.w = cta, w
            self.ring = np.full(STAGES * SLOT, np.float32(np.nan)).view(np.uint32).copy()   # NaN garbage
            self.cp = AsyncCopies(self.ring, rng)
            self.c_it, self.c_step, self.c_I = cta, 0, None
            if self.c_it < n_items:
                self.c_I = item_at(self.c_it)

        def cursor(self):
            I = self.c_I
            kt0 = I["split"] * (KT // L.SK) + self.w * PW
            Tw = L.T1 if I["mat"] else L.T0
            X = L.X1 if I["mat"] else L.X0
            return kt0, Tw, X

        def load_stage(self, slot):
            if self.c_it < n_items:
                I = self.c_I
                kt0, Tw, X = self.cursor()
                kt = kt0 + self.c_step
                words = Tw[I["e"], kt, I["nb"] * NT:I["nb"] * NT + NT].reshape(-1)   # NT tiles x 32 words
                base = slot * SLOT
                for q in range(NT * 8):                           # lane q (q += 32): 16-byte chunk q
                    self.cp.copy(base + q * 4, words[q * 4:q * 4 + 4])
                for lane in range(32):
                    r = I["row"][lane >> 1]
                    if mutate == "rows" and (lane >> 1) == I["live"] - 1:
                        r = -1
                    if r >= 0:
                        h = lane & 1
                        src = X[r, kt * 16 + h * 8:kt * 16 + h * 8 + 8]      # 8 fp16 = 16 bytes
                        pair = src.astype(np.uint32)
                        chunk = pair[0::2] | (pair[1::2] << 16)
                        self.cp.copy(base + NT * 32 + a_word(lane >> 1, h, 0), chunk)
                self.c_step += 1
                if self.c_step == PW:
                    self.c_step = 0
                    self.c_it += grid
                    if self.c_it < n_items:
                        self.c_I = item_at(self.c_it)
            self.cp.commit()

    def cta_items(cta):
        it = cta
        while it < n_items:
            yield it
            it += grid

    # CTAs as generators that yield after each item; the scheduler interleaves them at random
    def run_cta(cta):
        nonlocal epis
        if cta >= n_items:
            return
        warps = [Warp(cta, w) for w in range(W)]
        for wp in warps:
            for s in range(STAGES - 1):
                wp.load_stage(s)
                wp.cp.jitter()
        slot = 0
        for it in cta_items(cta):
            I = item_at(it)
            accs = []
            slots_w = [slot] * W
            for wp in warps:
                sl = slot
                acc = np.zeros((NT, 2, 32, 4), np.float32)
                for step in range(PW):
                    wp.cp.wait(STAGES - 1 if mutate == "wait" else STAGES - 2)
                    fill = STAGES - 1 if sl == 0 else sl - 1
                    wp.load_stage(fill)
                    wp.cp.jitter()
                    base = sl * SLOT
                    words = wp.ring[base:base + NT * 32]
                    asl = wp.ring[base + NT * 32:base + NT * 32 + 128]
                    aw = (lambda r, h, t: r * 8 + (h << 2) + t) if mutate == "swizzle" else a_word
                    a = np.stack([asl[aw(G, 0, T)], asl[aw(G + 8, 0, T)], asl[aw(G, 1, T)],
                                  asl[aw(G + 8, 1, T)]], axis=1)
                    for i in range(NT):
                        (b00, b01), (b10, b11) = decode_tile(words[i * 32 + LANE])
                        acc[i, 0] = mma_frag(acc[i, 0], a, b00, b01, "f16")
                        acc[i, 1] = mma_frag(acc[i, 1], a, b10, b11, "f16")
                    sl = 0 if sl == STAGES - 1 else sl + 1
                accs.append(acc)
                slots_w[wp.w] = sl
            assert len(set(slots_w)) == 1
            slot = slots_w[0]
            # reduce_store: warp 0 adds warps 1..3 in order, live rows only
            live = I["live"]
            lo = G < live
            hi = G + 8 < live
            zb = ((I["mat"] * L.SK + I["split"]) * L.P) * L.N + I["nb"] * NT * 16
            r0 = I["row"][G]
            r1 = I["row"][np.minimum(G + 8, 31)]
            for i in range(NT):
                for h in range(2):
                    col = i * 16 + h * 8 + 2 * T
                    order = list(range(W))[::-1] if mutate == "warps" else list(range(W))
                    s = [accs[order[0]][i, h][:, e].copy() for e in range(4)]
                    for w in order[1:]:
                        for e in range(4):
                            s[e] = (s[e] + accs[w][i, h][:, e]).astype(np.float32)
                    for ln in range(32):
                        if lo[ln]:
                            Z[zb + r0[ln] * L.N + col[ln]] = s[0][ln]
                            Z[zb + r0[ln] * L.N + col[ln] + 1] = s[1][ln]
                        if hi[ln]:
                            Z[zb + r1[ln] * L.N + col[ln]] = s[2][ln]
                            Z[zb + r1[ln] * L.N + col[ln] + 1] = s[3][ln]
            # finish: ticket on (mt, u, hb)
            PER_HB = 128 // (NT * 16)
            hb = I["nb"] // PER_HB
            expected = L.mats * L.SK * PER_HB
            ci = (I["mt"] * nexp + I["u"]) * (L.N // 128) + hb
            last = True
            if expected > 1:
                prev = int(cnt[ci])
                cnt[ci] += 1
                last = prev == expected - 1
                if last:
                    cnt[ci] = 0
            if last:
                for m in range(live):
                    L.epi(Z, I["e"], int(I["row"][m]), hb)
                    epis += 1
            yield
        for wp in warps:
            wp.cp.wait(0)

    runs = [run_cta(c) for c in range(grid)]
    active = list(range(grid))
    while active:
        k = active[int(rng.integers(len(active)))]
        try:
            next(runs[k])
        except StopIteration:
            active.remove(k)
    return Z, epis


def exl3_items(nexp: int, n_extra: int, mats: int, SK: int, N: int, NT: int) -> int:
    return (nexp + n_extra) * mats * SK * (N // (16 * NT))


# -- E2 reference: qmm._qmm + _reduce as compiled ------------------------------------------------------------------------
def unpack_q4(words: np.ndarray, n: int, k: int) -> np.ndarray:
    """MLX (n, k / 8) int32 words -> (n, k) integer values 0..15 (k = 8 * word + nibble)."""

    w = np.asarray(words, dtype=np.int64) & 0xFFFFFFFF
    q = (w[..., None] >> (np.arange(8) * 4)) & 0xF
    return q.reshape(n, k).astype(np.float32)


def qmm_reference(x_bf16, xs, words, scales, biases, n, k, sk, f32):
    """``_qmm`` + ``_reduce``: out[m, n] (bf16 bits or fp32). x_bf16 [M, K] bits; xs [M, K/64] fp32; words MLX
    (n, k/8) int32; scales / biases MLX (n, k/64) bf16 bits."""

    M = x_bf16.shape[0]
    X = bf16_to_f32(x_bf16)
    Q = unpack_q4(words, n, k)
    S = bf16_to_f32(scales)
    Bb = bf16_to_f32(biases)
    KG = k // 64
    per = KG // sk
    parts = []
    for s in range(sk):
        acc = np.zeros((M, n), np.float32)
        for g in range(s * per, s * per + per):
            p = np.zeros((M, n), np.float32)
            for c in range(4):                                  # four chained m16n8k16 from +0.0, k chunks in order
                k0 = g * 64 + c * 16
                with np.errstate(invalid="ignore", over="ignore"):
                    for kk in range(16):
                        p = (p + (X[:, k0 + kk:k0 + kk + 1] * Q[None, :, k0 + kk]).astype(np.float32)).astype(np.float32)
            acc = fma32(xs[:, g:g + 1], Bb[None, :, g], fma32(p, S[None, :, g], acc))
        parts.append(acc)
    tot = parts[0]
    for s in range(1, sk):
        tot = (tot + parts[s]).astype(np.float32)
    return tot if f32 else f32_to_bf16(tot)


# -- E2 port: q4_stream.cu ---------------------------------------------------------------------------------------------
def tile_words_np(words: np.ndarray, n: int) -> np.ndarray:
    """qmm.tile_words: (n, k/8) -> (ceil(n/64), k/64, 64, 8), n padded with zero rows."""

    k8 = words.shape[1]
    npad = -(-n // 64) * 64
    w = np.zeros((npad, k8), np.int32)
    w[:n] = words
    return w.reshape(npad // 64, 64, k8 // 8, 8).transpose(0, 2, 1, 3).copy()


def q4_stream_port(x_bf16, xs, words, scales, biases, n, k, sk, f32, RT, GPS, STAGES, grid, rng, mutate: str = "",
                   serial: bool = False):
    """q4_stream.cu's q4_kernel<RT, GPS, STAGES> over ``grid`` CTAs, interleaved at random item by item; ``serial``: an
    item is a whole tile, its K slices in order, added in registers (Args.serial; always so when sk == 1).

    ``mutate`` (negative controls): "wait" one group too few waited; "khalf" a2 / a0 swapped (k 2t+8 read as 2t);
    "fma" the two fmas in the other order; "chunks" the four k chunks in reverse; "slices" partials added in reverse."""

    M = x_bf16.shape[0]
    BN, XROW = 64, 72
    KG = k // 64
    PER = KG // sk
    STEPS = PER // GPS
    NT = -(-n // BN)
    serial = serial or sk == 1
    SLICES = sk if serial else 1
    n_items = NT if serial else NT * sk

    def tile_of(it):
        return it if serial else it // sk

    def first_group(it):
        return 0 if serial else (it % sk) * PER
    Wt = tile_words_np(words, n).view(np.uint32).reshape(-1)       # the stored tiles, flat words
    St = np.ascontiguousarray(np.asarray(scales, np.uint16).T).reshape(-1)    # [K/64][N] bf16 bits
    Bt = np.ascontiguousarray(np.asarray(biases, np.uint16).T).reshape(-1)
    xflat = np.asarray(x_bf16, np.uint16)
    WORDS = GPS * BN * 8 * 4
    SB = GPS * BN * 2
    XB = GPS * RT * 16 * XROW * 2
    XSB = GPS * RT * 16 * 4
    BYTES = WORDS + 2 * SB + XB + XSB
    out = np.full((M, n), np.nan, np.float32) if f32 else np.full((M, n), 0x7FC1, np.uint16)
    part = np.full(sk * M * n, np.nan, np.float32)
    cnt = np.zeros(NT, np.int64)

    def load_stage(cp, base_b, tile, g0):
        """smem as bytes (uint8); copies of 16 bytes (4 for group sums), as the CUDA."""

        wsrc = ((tile * KG + g0) * (BN * 8))
        for q in range(GPS * BN * 2):
            cp.copy(base_b + q * 16, Wt[wsrc + q * 4:wsrc + q * 4 + 4].view(np.uint8))
        n0 = tile * BN
        for q in range(GPS * 16):
            gi, which, c8 = q >> 4, (q >> 3) & 1, (q & 7) * 8
            arr = Bt if which else St
            left = n - (n0 + c8)
            dst = base_b + WORDS + which * SB + (gi * BN + c8) * 2
            if left >= 8:
                cp.copy(dst, arr[(g0 + gi) * n + n0 + c8:(g0 + gi) * n + n0 + c8 + 8].view(np.uint8))
            else:
                cp.copy(dst, np.zeros(16, np.uint8))                 # src-size 0: zero fill
        xb = base_b + WORDS + 2 * SB
        for q in range(GPS * RT * 16 * 8):
            gi, r, c = q // (RT * 16 * 8), (q >> 3) % (RT * 16), q & 7
            if r < M:
                src = xflat[r, (g0 + gi) * 64 + c * 8:(g0 + gi) * 64 + c * 8 + 8]
                cp.copy(xb + ((gi * RT * 16 + r) * XROW + c * 8) * 2, src.view(np.uint8))
        xsb = xb + XB
        for q in range(GPS * RT * 16):
            gi, r = q // (RT * 16), q % (RT * 16)
            if r < M:
                cp.copy(xsb + (gi * RT * 16 + r) * 4, np.asarray([xs[r, g0 + gi]], np.float32).view(np.uint8))

    def run_cta(cta):
        if cta >= n_items:
            return
        smem = np.full(STAGES * BYTES // 4, np.float32(np.nan)).view(np.uint8).copy()
        cp = AsyncCopies(smem, rng)
        l = {"it": cta, "step": 0}

        def load_next(slot_):
            if l["it"] < n_items:
                load_stage(cp, slot_ * BYTES, tile_of(l["it"]), first_group(l["it"]) + l["step"] * GPS)
                l["step"] += 1
                if l["step"] == SLICES * STEPS:
                    l["step"] = 0
                    l["it"] += grid
            cp.commit()
            cp.jitter()

        for s in range(STAGES - 1):
            load_next(s)
        slot = 0
        it = cta
        while it < n_items:
            tile, sl = tile_of(it), (0 if serial else it % sk)
            tot = None
            for slc in range(SLICES):
                acc = np.zeros((4, RT, 2, 32, 4), np.float32)          # [warp][rt][h][lane][e]
                for step in range(STEPS):
                    cp.wait(STAGES - 1 if mutate == "wait" else STAGES - 2)
                    fill = STAGES - 1 if slot == 0 else slot - 1
                    load_next(fill)
                    st = slot * BYTES
                    for gi in range(GPS):
                        wd = smem[st:st + WORDS].view(np.uint32)[gi * BN * 8:(gi + 1) * BN * 8]
                        ss = smem[st + WORDS:st + WORDS + SB].view(np.uint16)[gi * BN:(gi + 1) * BN]
                        bs = smem[st + WORDS + SB:st + WORDS + 2 * SB].view(np.uint16)[gi * BN:(gi + 1) * BN]
                        xr = smem[st + WORDS + 2 * SB:st + WORDS + 2 * SB + XB].view(np.uint16)
                        xr = xr[gi * RT * 16 * XROW:(gi + 1) * RT * 16 * XROW]
                        xsv = smem[st + WORDS + 2 * SB + XB:st + BYTES].view(np.float32)[gi * RT * 16:(gi + 1) * RT * 16]
                        for warp in range(4):
                            bq = np.zeros((2, 4, 2, 32), np.uint32)
                            for h in range(2):
                                col = warp * 16 + h * 8 + G
                                w8 = np.stack([wd[col * 8 + j] for j in range(8)], 0)      # [8, lanes]
                                for c in range(4):
                                    for jj, wsel in ((0, 2 * c), (1, 2 * c + 1)):
                                        v = w8[wsel] >> (8 * T).astype(np.uint32)
                                        biased = (v & 0xF) | ((v & 0xF0) << 12) | 0x43004300
                                        lo = bf16_to_f32((biased & 0xFFFF).astype(np.uint16)) - np.float32(128)
                                        hi = bf16_to_f32((biased >> 16).astype(np.uint16)) - np.float32(128)
                                        bq[h, c, jj] = (f32_to_bf16(lo).astype(np.uint32) |
                                                        (f32_to_bf16(hi).astype(np.uint32) << 16))
                            sv = np.zeros((2, 2, 32), np.float32)
                            bv = np.zeros((2, 2, 32), np.float32)
                            for h in range(2):
                                for j in range(2):
                                    col = warp * 16 + h * 8 + 2 * T + j
                                    sv[h, j] = bf16_to_f32(ss[col])
                                    bv[h, j] = bf16_to_f32(bs[col])
                            xr32 = xr.view(np.uint32)
                            for r in range(RT):
                                row0 = (r * 16 + G) * XROW // 2
                                row1 = (r * 16 + G + 8) * XROW // 2
                                p = np.zeros((2, 32, 4), np.float32)
                                for c in (range(3, -1, -1) if mutate == "chunks" else range(4)):
                                    a = np.stack([xr32[row0 + c * 8 + T], xr32[row1 + c * 8 + T],
                                                  xr32[row0 + c * 8 + 4 + T], xr32[row1 + c * 8 + 4 + T]], 1)
                                    if mutate == "khalf":
                                        a = a[:, [2, 1, 0, 3]]
                                    p[0] = mma_frag(p[0], a, bq[0, c, 0], bq[0, c, 1], "bf16")
                                    p[1] = mma_frag(p[1], a, bq[1, c, 0], bq[1, c, 1], "bf16")
                                xs0, xs1 = xsv[r * 16 + G], xsv[r * 16 + G + 8]
                                for h in range(2):
                                    A_ = acc[warp, r, h]
                                    if mutate == "fma":
                                        A_[:, 0] = fma32(p[h][:, 0], sv[h, 0], fma32(xs0, bv[h, 0], A_[:, 0]))
                                    else:
                                        A_[:, 0] = fma32(xs0, bv[h, 0], fma32(p[h][:, 0], sv[h, 0], A_[:, 0]))
                                    A_[:, 1] = fma32(xs0, bv[h, 1], fma32(p[h][:, 1], sv[h, 1], A_[:, 1]))
                                    A_[:, 2] = fma32(xs1, bv[h, 0], fma32(p[h][:, 2], sv[h, 0], A_[:, 2]))
                                    A_[:, 3] = fma32(xs1, bv[h, 1], fma32(p[h][:, 3], sv[h, 1], A_[:, 3]))
                    slot = 0 if slot == STAGES - 1 else slot + 1
                if slc == 0:
                    tot = acc.copy()
                else:
                    tot = (tot + acc).astype(np.float32)          # qmm._reduce's sum, slices in order
            # outputs
            for warp in range(4):
                n_base = tile * BN + warp * 16 + 2 * T
                for r in range(RT):
                    for h in range(2):
                        for e in range(4):
                            m = r * 16 + G + (e >> 1) * 8
                            nn = n_base + h * 8 + (e & 1)
                            ok = (m < M) & (nn < n)
                            v = (tot if serial else acc)[warp, r, h][:, e]
                            for ln in np.nonzero(ok)[0]:
                                if serial:
                                    if f32:
                                        out[m[ln], nn[ln]] = v[ln]
                                    else:
                                        out[m[ln], nn[ln]] = f32_to_bf16(v[ln])
                                else:
                                    part[(sl * M + m[ln]) * n + nn[ln]] = v[ln]
            if not serial:
                prev = int(cnt[tile])
                cnt[tile] += 1
                if prev == sk - 1:
                    cnt[tile] = 0
                    cols = min(BN, n - tile * BN)
                    for idx in range(M * cols):
                        m, nn = idx // cols, tile * BN + idx % cols
                        order = list(range(sk))[::-1] if mutate == "slices" else list(range(sk))
                        tot = np.float32(part[(order[0] * M + m) * n + nn])
                        for s in order[1:]:
                            tot = np.float32(tot + part[(s * M + m) * n + nn])
                        out[m, nn] = tot if f32 else f32_to_bf16(tot)
            yield
            it += grid
        cp.wait(0)

    runs = [run_cta(c) for c in range(grid)]
    active = list(range(grid))
    while active:
        kk = active[int(rng.integers(len(active)))]
        try:
            next(runs[kk])
        except StopIteration:
            active.remove(kk)
    assert not cnt.any()
    return out
