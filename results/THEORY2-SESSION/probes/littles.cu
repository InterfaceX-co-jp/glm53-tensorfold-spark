// littles.cu -- THEORY-2 item 8 gate probe (§6 step 1b): GB10 DRAM latency and Little's-law curves for four load
// mechanisms, cold, on the gathered EXL3 expert layout. Measurement only; no engine code.
//
//   nvcc -O3 -arch=sm_121 -lineinfo -o littles littles.cu
//   ./littles [--quick] [--json out.json] [--gb 8] [--only chase|nc|cpasync|bulk] [--reps N] [--no-beside]
//             [--gate-gbs 230]
//
// Why: 0440's E1 ring (cp.async, warp-private, ~18 KB an SM in flight) capped at ~158 GB/s with no ALU work, while
// W11's plain-read probe reached 235 GB/s at ~32 KB an SM. By Little's law, throughput = bytes in flight / latency,
// so the design question is "how many bytes must be in flight per SM, with which mechanism, to hold >= 230 GB/s on
// GB10", measured instead of assumed (THEORY-2 §2, §3.3, §7 rule 2).
//
// Memory image. A pool of --gb GB (default 8, --quick 4; >> the 24 MB L2) is laid out as EXL3 experts: one expert =
// gate (2 MiB) + up (2 MiB) + down (2 MiB) = 6.29 MB of trellis words, each matrix T[kt][nt][32 words] (exl3.cu):
//   gate / up: K 4096 x N 1024 -> 256 k rows x 64 tiles x 128 B (8 KB a k row)
//   down:      K 1024 x N 4096 ->  64 k rows x 256 tiles x 128 B (32 KB a k row)
// A warp "segment" is 64 steps of 512 B (32 lanes x 16 B; one step = NT 4 trellis tiles of one k row), 32 KB.
// Patterns (all mechanisms read the same segments):
//   tile   random experts; a segment is one warp's 16-row tile stream: 4 column tiles (512 B) of 64 consecutive k
//          rows (stride 8 KB for gate / up, 32 KB for down), i.e. what exl3.cu's member_tiles<NT=4> warp reads.
//          Concurrent warps take neighbouring column blocks (exl3's item order). This is the pattern the gate reads.
//   contig random experts, each read as contiguous 32 KB segments (W11 "gathered").
//   flat   consecutive experts: one plain contiguous region (W11 "contiguous").
// Every timed launch reads a fresh random expert set (rotation over reps + 3 sets), so nothing is L2-resident.
//
// Sections:
//  1. Latency (pointer chase, one thread, ld.global.cg, clock64 + %globaltimer):
//     - idle DRAM latency: a chain over random distinct g-aligned slots, g = 64 B / 128 B / 4 KB / 2 MB (the 2 MB
//       one spans the whole pool: every step a new 2 MB page), L2 flushed first, one cold lap timed;
//     - L2-hit latency: a 16K-node chain in 8 MB, warm lap then timed lap;
//     - loaded latency: the same thread chasing (512 B random nodes over 2 GB) while N other CTAs (4 warps, 4
//       independent ld.global.nc.v4 a lane) stream the expert pool until the chase ends; reports the streamers' GB/s.
//  2. Little's-law sweep (throughput vs nominal bytes in flight per SM; one config = one row):
//     (a) nc      ld.global.nc.v4, D = 1..8 independent 16-B loads a lane issued, then consumed; the next batch's
//                 address depends on the consumed data (a masked zero), so in flight = D x 512 B a warp, exactly.
//                 4-warp CTAs, 1..12 CTAs an SM (4..48 warps).
//     (b) cpasync 0440-E1-style warp-private ring: S stages x G steps (G x 512 B a stage), 16-B cp.async.cg,
//                 one commit group a stage, wait_group S-2, __syncwarp; consumer reads a neighbour lane's 16 B from
//                 shared. In flight = (S-1) x G x 512 B a warp. (E1 default = S 4, G 1, 3 CTAs an SM.)
//     (c) bulk    cp.async.bulk (TMA 1D) global -> shared with mbarrier complete_tx: CH = 4/8/16 KB chunks, S = 2-4
//                 stages, 256-thread CTAs, 1-4 CTAs an SM; one issuing lane for contiguous chunks (contig / flat);
//                 for `tile` a chunk is CH/512 separate 512-B copies at the tile stride, issued by lanes of warp 0
//                 onto one mbarrier. All 256 threads consume, __syncthreads, warp 0 re-arms. In flight = S x CH a CTA.
//     (d) chase   warp-cooperative dependent chase over random 512-B nodes (the next pointer is in the node, lane 0
//                 broadcasts it), 1/2/4 independent chains a warp: the fully latency-bound reference curve.
//     Derived per row: effective latency = device bytes in flight / throughput (Little's law).
//  3. "Beside": every sweep row is re-run (3x the bytes) concurrently with latk, a 32-CTA x 1024-thread latency-bound
//     chain kernel (KDA-chain-like: a dependent 16-B ld.cg a thread per step, 16 dependent FMAs, two __syncthreads,
//     256 steps over a cold 128 MB window) on a second stream. %globaltimer stamps give each kernel's own span:
//     streamer GB/s beside vs alone, latk slowdown, the fraction of latk inside the streamer, and the makespan vs
//     running the two back to back.
//  4. Registers / spills / static smem of every kernel (cudaFuncGetAttributes), SM clock at start and end
//     (nvidia-smi + a clock64/globaltimer measurement), a table and --json.
//
// Timing: cudaEvent median of --reps (11; --quick 5) for alone rows, device stamps for beside rows. Runtime: ~1-3 min
// full, < 1 min --quick on GB10 (the caller wraps it in `timeout`).
//
// How to read the gate. The candidate mechanisms are nc and cpasync on the `tile` pattern (a warp's own 4-tile
// stream, as the EXL3 kernel consumes it) and bulk on any pattern (a CTA owning whole k rows is a valid EXL3 design).
// A row qualifies if (i) alone GB/s >= --gate-gbs (230), (ii) the probe kernel uses <= 112 registers and (iii) its
// warps an SM would still fit at 112 registers a thread (<= 65536 / (112 x 32) = 18 warps an SM), i.e. the real
// kernel (decode + mma, ~100-112 registers) could run at that occupancy. X = the smallest in-flight KB an SM among
// qualifying rows. For nc, the D x 4 data registers held in flight must also fit inside the real kernel's 112
// (reported as data_regs; D = 8 needs 32). The last line reads
//   GATE item8: <mechanism> reaches >= 230 GB/s at <= X KB/SM in flight with <= 112 regs/thread: PASS|FAIL
// Rows above 250 GB/s are flagged: DRAM cannot deliver that, so such a row would be L2-resident (§7 rule 1).
// Caveats: the consumers here do no decode / mma, so these are upper bounds for a kernel that also computes; the SM
// clock is whatever the node is set to (the session locks 2,250 MHz; printed at start / end).

#include <cuda_runtime.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#define CK(x)                                                                                         \
    do {                                                                                              \
        cudaError_t e_ = (x);                                                                         \
        if (e_ != cudaSuccess) {                                                                      \
            fprintf(stderr, "CUDA %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
            exit(1);                                                                                  \
        }                                                                                             \
    } while (0)

// ---- layout constants --------------------------------------------------------------------------------------------
constexpr uint64_t MAT_BYTES = 2097152ull;             // one 4-bit 4096 x 1024 matrix, this rank's half
constexpr uint64_t EXPERT_BYTES = 3 * MAT_BYTES;       // gate + up + down: 6.29 MB
constexpr int STEP_BYTES = 512;                        // a warp step: 32 lanes x 16 B = 4 trellis tiles
constexpr int SEG_STEPS = 64;                          // steps a segment
constexpr int SEG_BYTES = STEP_BYTES * SEG_STEPS;      // 32 KB
constexpr int SEGS_PER_EXPERT = (int)(EXPERT_BYTES / SEG_BYTES);   // 192
enum { PAT_TILE = 0, PAT_CONTIG = 1 };                 // device patterns (flat = contig over consecutive experts)
constexpr int NODE_BYTES = 512;                        // chase-throughput / loaded-chase node

// ---- device helpers ----------------------------------------------------------------------------------------------
struct Stamp {
    unsigned long long start, end;
};

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ void stamp_begin(Stamp *s) {
    if (s && threadIdx.x == 0) atomicMin(&s->start, gtimer());
}
__device__ __forceinline__ void stamp_end(Stamp *s) {
    __syncthreads();
    if (s && threadIdx.x == 0) atomicMax(&s->end, gtimer());
}

__device__ __forceinline__ uint4 ldg_stream(const void *p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}
__device__ __forceinline__ uint64_t ld_cg_u64(const void *p) {
    uint64_t v;
    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void xor4(uint4 &a, const uint4 &v) {
    a.x ^= v.x; a.y ^= v.y; a.z ^= v.z; a.w ^= v.w;
}
__device__ __forceinline__ void sink_if(const uint4 &a, uint4 *sink) {
    if ((a.x & a.y & a.z & a.w) == 0xdeadbeefu) sink[threadIdx.x & 255] = a;   // never true in practice; keeps loads
}

// Byte offset (from the pool) of segment `seg` of the expert set and its step stride.
__device__ __forceinline__ uint64_t seg_off(const int *__restrict__ ex, int pattern, int seg, uint32_t &stride) {
    const int u = seg / SEGS_PER_EXPERT, j = seg - u * SEGS_PER_EXPERT;
    const uint64_t base = (uint64_t)__ldg(ex + u) * EXPERT_BYTES;
    if (pattern == PAT_CONTIG) {
        stride = STEP_BYTES;
        return base + (uint64_t)j * SEG_BYTES;
    }
    const int mat = j >> 6, r = j & 63;
    if (mat < 2) {   // gate / up: 16 column blocks (512 B of an 8 KB k row) x 4 k quarters of 64 rows
        stride = 8192;
        return base + mat * MAT_BYTES + (uint64_t)(r >> 4) * (64 * 8192) + (uint64_t)(r & 15) * STEP_BYTES;
    }
    stride = 32768;  // down: 64 column blocks of a 32 KB k row, all 64 k rows
    return base + 2 * MAT_BYTES + (uint64_t)r * STEP_BYTES;
}

// cp.async (0440 E1 style)
__device__ __forceinline__ void cp_async16(void *smem, const void *gmem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

// cp.async.bulk (TMA 1D) + mbarrier (sm_90+; sm_120 / sm_121 included)
__device__ __forceinline__ uint32_t smem_u32(const void *p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t *bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint64_t *bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes)
                 : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t *bar, uint32_t parity) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra LAB_WAIT;\n"
        "}\n" ::"r"(smem_u32(bar)),
        "r"(parity)
        : "memory");
}
__device__ __forceinline__ void bulk_g2s(void *dst, const void *src, uint32_t bytes, uint64_t *bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar))
                 : "memory");
}

// ---- kernels -----------------------------------------------------------------------------------------------------
struct SArgs {
    const uint8_t *pool;
    const int *ex;        // expert set of this launch (indices into the pool, in EXPERT_BYTES units)
    int pattern;          // PAT_TILE / PAT_CONTIG
    int segs_per_warp;    // nc / cpasync: segments each warp reads (warp gw reads gw, gw + TW, ...)
    int G;                // cpasync: steps a stage
    int chunk;            // bulk: bytes a chunk (4 / 8 / 16 KB)
    int nck;              // bulk: chunks a CTA (chunk c = blockIdx + i * grid)
    uint32_t zero;        // always 0 (opaque to the compiler)
    uint4 *sink;
    Stamp *st;            // optional span stamps
};

// (a) D independent 16-B non-coherent loads a lane, then use; the next batch's address depends on the data.
template <int D>
__global__ void __launch_bounds__(128) nc_kernel(SArgs a) {
    stamp_begin(a.st);
    const int lane = threadIdx.x & 31;
    const int gw = blockIdx.x * 4 + (threadIdx.x >> 5), TW = gridDim.x * 4;
    uint4 acc = make_uint4(0, 0, 0, 0);
    uint64_t dep = 0;
    for (int i = 0; i < a.segs_per_warp; ++i) {
        uint32_t stride;
        const uint8_t *p = a.pool + seg_off(a.ex, a.pattern, gw + i * TW, stride) + lane * 16;
        for (int s = 0; s < SEG_STEPS; s += D) {
            uint4 v[D];
#pragma unroll
            for (int d = 0; d < D; ++d) v[d] = ldg_stream(p + (uint64_t)(s + d) * stride + dep);
#pragma unroll
            for (int d = 0; d < D; ++d) xor4(acc, v[d]);
            dep = acc.x & a.zero;
        }
    }
    sink_if(acc, a.sink);
    stamp_end(a.st);
}

// (b) warp-private cp.async ring, S stages of G steps; S - 1 stages in flight (0440 E1: S 4, G 1 = NT 4 tiles).
template <int S>
__global__ void __launch_bounds__(128) cpa_kernel(SArgs a) {
    extern __shared__ __align__(128) uint8_t dsm[];
    stamp_begin(a.st);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gw = blockIdx.x * 4 + warp, TW = gridDim.x * 4;
    const int G = a.G, SB = G * STEP_BYTES;
    uint8_t *ring = dsm + warp * S * SB;
    const int n = a.segs_per_warp * SEG_STEPS / G;   // stages this warp consumes
    auto issue = [&](int k) {
        uint8_t *slot = ring + (k % S) * SB;
        for (int g = 0; g < G; ++g) {
            const int t = k * G + g;
            uint32_t stride;
            const uint64_t off = seg_off(a.ex, a.pattern, gw + (t >> 6) * TW, stride);
            cp_async16(slot + g * STEP_BYTES + lane * 16, a.pool + off + (uint64_t)(t & 63) * stride + lane * 16);
        }
    };
#pragma unroll
    for (int k = 0; k < S - 1; ++k) {
        if (k < n) issue(k);
        cp_commit();
    }
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (int k = 0; k < n; ++k) {
        cp_wait<S - 2>();          // stage k landed (S - 1 + k groups committed)
        __syncwarp();
        if (k + S - 1 < n) issue(k + S - 1);   // into slot (k - 1) % S, consumed last iteration
        cp_commit();
        const uint8_t *slot = ring + (k % S) * SB;
        for (int g = 0; g < G; ++g) xor4(acc, *reinterpret_cast<const uint4 *>(slot + g * STEP_BYTES + ((lane + 1) & 31) * 16));
        __syncwarp();
    }
    cp_wait<0>();
    sink_if(acc, a.sink);
    stamp_end(a.st);
}

// (c) TMA 1D bulk copies into an S-stage CTA ring with one mbarrier a stage; warp 0 issues, all threads consume.
template <int S>
__global__ void __launch_bounds__(256) bulk_kernel(SArgs a) {
    extern __shared__ __align__(128) uint8_t dsm[];
    __shared__ __align__(8) uint64_t bar[S];
    stamp_begin(a.st);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int CH = a.chunk, P = SEG_BYTES / CH, NR = CH / STEP_BYTES;
    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < S; ++s) mbar_init(&bar[s], 1);
        mbar_fence_init();
    }
    __syncthreads();
    auto issue = [&](int i) {   // warp 0, all lanes
        const int c = blockIdx.x + i * gridDim.x;
        const int seg = c / P, part = c - seg * P;
        uint32_t stride;
        const uint64_t off = seg_off(a.ex, a.pattern, seg, stride);
        uint8_t *dst = dsm + (i % S) * CH;
        uint64_t *b = &bar[i % S];
        if (lane == 0) mbar_expect_tx(b, (uint32_t)CH);
        if (stride == STEP_BYTES) {          // contiguous chunk: one copy, one issuing lane
            if (lane == 0) bulk_g2s(dst, a.pool + off + (uint64_t)part * CH, (uint32_t)CH, b);
        } else {                              // tile: CH / 512 runs of 512 B at the tile stride
            __syncwarp();
            for (int r = lane; r < NR; r += 32)
                bulk_g2s(dst + r * STEP_BYTES, a.pool + off + (uint64_t)(part * NR + r) * stride, STEP_BYTES, b);
        }
    };
    if (warp == 0)
        for (int i = 0; i < S && i < a.nck; ++i) issue(i);
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (int i = 0; i < a.nck; ++i) {
        mbar_wait(&bar[i % S], (uint32_t)((i / S) & 1));
        const uint4 *src = reinterpret_cast<const uint4 *>(dsm + (i % S) * CH);
        for (int j = tid; j < CH / 16; j += 256) xor4(acc, src[j]);
        __syncthreads();              // everyone is done with stage i % S before it is re-armed
        if (warp == 0 && i + S < a.nck) issue(i + S);
    }
    sink_if(acc, a.sink);
    stamp_end(a.st);
}

// (d) warp-cooperative dependent chase over 512-B nodes: CH independent chains a warp.
template <int CH>
__global__ void __launch_bounds__(128) chase_tput_kernel(const uint8_t *base, const uint64_t *starts, int L,
                                                         uint4 *sink, Stamp *st) {
    stamp_begin(st);
    const int lane = threadIdx.x & 31;
    const int gw = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    uint64_t p[CH];
#pragma unroll
    for (int c = 0; c < CH; ++c) p[c] = starts[gw * CH + c];
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (int s = 0; s < L; ++s) {
        uint4 v[CH];
#pragma unroll
        for (int c = 0; c < CH; ++c) v[c] = ldg_stream(base + p[c] + lane * 16);
#pragma unroll
        for (int c = 0; c < CH; ++c) {
            xor4(acc, v[c]);
            const uint64_t nx = ((uint64_t)v[c].y << 32) | v[c].x;
            p[c] = __shfl_sync(0xffffffffu, nx, 0);
        }
    }
    sink_if(acc, sink);
    stamp_end(st);
}

// Idle chase: one thread; `warm` untimed steps, then `n` timed. out = {cycles, ns, final pointer}.
__global__ void chase1_kernel(const uint8_t *base, uint64_t start, int warm, int n, unsigned long long *out) {
    uint64_t p = start;
    for (int i = 0; i < warm; ++i) p = ld_cg_u64(base + p);
    const long long c0 = clock64();
    const unsigned long long g0 = gtimer();
    for (int i = 0; i < n; ++i) p = ld_cg_u64(base + p);
    const long long c1 = clock64();
    const unsigned long long g1 = gtimer();
    out[0] = (unsigned long long)(c1 - c0);
    out[1] = g1 - g0;
    out[2] = p;
}

// Loaded chase: block 0 thread 0 chases (as chase1_kernel), blocks 1.. stream the expert set (nc, D = 4, wrapping)
// until the chase is done (or 500 ms). out = {cycles, ns, streamed bytes, final pointer}; st = streamers' span.
__global__ void __launch_bounds__(128) loaded_kernel(SArgs a, int nsegs, const uint8_t *cbase, uint64_t cstart, int warm,
                                                     int n, int *done, unsigned long long *out) {
    if (blockIdx.x == 0) {
        if (threadIdx.x == 0) {
            uint64_t p = cstart;
            for (int i = 0; i < warm; ++i) p = ld_cg_u64(cbase + p);
            const long long c0 = clock64();
            const unsigned long long g0 = gtimer();
            for (int i = 0; i < n; ++i) p = ld_cg_u64(cbase + p);
            const long long c1 = clock64();
            const unsigned long long g1 = gtimer();
            out[0] = (unsigned long long)(c1 - c0);
            out[1] = g1 - g0;
            out[3] = p;
            __threadfence();
            atomicExch(done, 1);
        }
        return;
    }
    const int lane = threadIdx.x & 31;
    const int gw = (blockIdx.x - 1) * 4 + (threadIdx.x >> 5), TW = (gridDim.x - 1) * 4;
    const unsigned long long t0 = gtimer();
    if (threadIdx.x == 0) atomicMin(&a.st->start, t0);
    uint4 acc = make_uint4(0, 0, 0, 0);
    uint64_t dep = 0, bytes = 0;
    for (long long i = 0;; ++i) {
        if (*(volatile int *)done) break;
        if (gtimer() - t0 > 500000000ull) break;   // safety: never outlive the chase by much
        uint32_t stride;
        const int seg = (int)(((long long)gw + i * TW) % nsegs);
        const uint8_t *p = a.pool + seg_off(a.ex, a.pattern, seg, stride) + lane * 16;
        for (int s = 0; s < SEG_STEPS; s += 4) {
            uint4 v0 = ldg_stream(p + (uint64_t)s * stride + dep), v1 = ldg_stream(p + (uint64_t)(s + 1) * stride + dep),
                  v2 = ldg_stream(p + (uint64_t)(s + 2) * stride + dep), v3 = ldg_stream(p + (uint64_t)(s + 3) * stride + dep);
            xor4(acc, v0); xor4(acc, v1); xor4(acc, v2); xor4(acc, v3);
            dep = acc.x & a.zero;
        }
        bytes += SEG_BYTES;
    }
    sink_if(acc, a.sink);
    if (lane == 0) {
        atomicAdd(&out[2], (unsigned long long)bytes);
        atomicMax(&a.st->end, gtimer());
    }
}

// latk: the "latency-bound neighbour" (KDA-chain-like). 32 CTAs x 1024 threads; per step one dependent 16-B ld.cg a
// thread (16 KB a CTA, cold), 16 dependent FMAs, a shared-memory exchange between two __syncthreads.
__global__ void __launch_bounds__(1024) latk_kernel(const float4 *buf, int T, uint32_t zero, float *out, Stamp *st) {
    __shared__ float red[1024];
    stamp_begin(st);
    const int tid = threadIdx.x;
    const float4 *p = buf + (size_t)blockIdx.x * T * 1024 + tid;
    float s = tid * 1e-3f;
    uint32_t dep = 0;
    for (int t = 0; t < T; ++t) {
        const float4 v = __ldcg(p + (size_t)t * 1024 + dep);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            s = fmaf(s, v.x, v.y);
            s = fmaf(s, v.z, v.w);
        }
        red[tid] = s;
        __syncthreads();
        s += red[tid ^ 32];
        dep = __float_as_uint(red[(tid + 1) & 1023]) & zero;
        __syncthreads();
    }
    if (s == 12345.f) out[blockIdx.x] = s;
    stamp_end(st);
}

// Chain builders: node at base + offs[i] stores offs[(i + 1) % n] (a cycle).
__global__ void link_kernel(uint8_t *base, const uint64_t *offs, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        *reinterpret_cast<uint64_t *>(base + offs[i]) = offs[(i + 1) % n];
}
__global__ void cycle_kernel(uint8_t *base, const uint32_t *perm, int m) {   // 512-B nodes in cycle order
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += gridDim.x * blockDim.x)
        *reinterpret_cast<uint64_t *>(base + (uint64_t)perm[i] * NODE_BYTES) = (uint64_t)perm[(i + 1) % m] * NODE_BYTES;
}

// L2 flush: stream-read a region (>= 4x L2).
__global__ void flush_kernel(const uint4 *p, uint64_t n16, uint4 *sink) {
    uint4 a = make_uint4(0, 0, 0, 0);
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (uint64_t)gridDim.x * blockDim.x)
        xor4(a, __ldcg(p + i));
    sink_if(a, sink);
}

// SM clock: clock64 cycles over ~2 ms of %globaltimer.
__global__ void clock_kernel(unsigned long long *out) {
    const unsigned long long g0 = gtimer();
    const long long c0 = clock64();
    unsigned long long g;
    do { g = gtimer(); } while (g - g0 < 2000000ull);
    const long long c1 = clock64();
    out[0] = (unsigned long long)(c1 - c0);
    out[1] = g - g0;
}

// ---- host ----------------------------------------------------------------------------------------------------------
struct Opt {
    bool quick = false, beside = true;
    const char *json = nullptr;
    double gb = 0;
    std::string only;
    int reps = 0;
    double gate_gbs = 230.0;
};

static int g_sms, g_dev = 0;
static cudaStream_t sA, sB;
static uint4 *g_sink;
static Stamp *d_st;
static std::string J;   // JSON body pieces

static std::string fmt(const char *f, ...) __attribute__((format(printf, 1, 2)));
static std::string fmt(const char *f, ...) {
    char b[2048];
    va_list ap;
    va_start(ap, f);
    vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}
static double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static std::string nvsmi() {
    FILE *f = popen("nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu,pstate "
                    "--format=csv,noheader 2>/dev/null",
                    "r");
    if (!f) return "n/a";
    char b[512] = {0};
    std::string s;
    while (fgets(b, sizeof b, f)) s += b;
    pclose(f);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    for (auto &c : s)
        if (c == '"' || c == '\n') c = ' ';
    return s.empty() ? "n/a" : s;
}
static double measured_mhz() {
    unsigned long long *d, h[2];
    CK(cudaMalloc(&d, 16));
    clock_kernel<<<1, 1>>>(d);
    CK(cudaMemcpy(h, d, 16, cudaMemcpyDeviceToHost));
    CK(cudaFree(d));
    return (double)h[0] / (double)h[1] * 1e3;
}

struct KInfo {
    std::string name;
    int regs, local, smem;
};
static std::vector<KInfo> g_kinfo;
template <class K>
static int kinfo(const char *name, K k) {
    for (auto &x : g_kinfo)
        if (x.name == name) return x.regs;
    cudaFuncAttributes at;
    CK(cudaFuncGetAttributes(&at, k));
    g_kinfo.push_back({name, at.numRegs, (int)at.localSizeBytes, (int)at.sharedSizeBytes});
    return at.numRegs;
}

// One sweep row.
struct Row {
    std::string mech, pattern, cfg;
    int ctas_sm, warps_sm, regs, data_regs;
    double inflight_kb_sm, smem_kb_sm, bytes, us, gbs, eff_lat_us;
    bool has_b = false;
    double gbs_b = 0, str_slow = 0, latk_us = 0, latk_us_b = 0, latk_slow = 0, overlap = 0, makespan_ratio = 0;
};
static std::vector<Row> g_rows;

// latk (the latency-bound neighbour)
static const int LATK_CTAS = 32, LATK_T = 256;
static const uint64_t LATK_BYTES = (uint64_t)LATK_CTAS * LATK_T * 1024 * 16;   // 128 MB a launch
static uint8_t *g_pool;
static uint64_t g_pool_bytes;
static float *g_latk_out;
static double g_latk_alone_us = 0;
static void launch_latk(cudaStream_t s, Stamp *st, int r) {
    const uint64_t slots = g_pool_bytes / LATK_BYTES - 1;
    const uint64_t off = ((uint64_t)(r * 7 + 3) % slots) * LATK_BYTES;
    latk_kernel<<<LATK_CTAS, 1024, 0, s>>>(reinterpret_cast<const float4 *>(g_pool + off), LATK_T, 0u, g_latk_out, st);
}

static void reset_stamps(int n) {
    std::vector<Stamp> h(n, Stamp{~0ull, 0ull});
    CK(cudaMemcpyAsync(d_st, h.data(), n * sizeof(Stamp), cudaMemcpyHostToDevice, sA));
    CK(cudaStreamSynchronize(sA));
}

// launch(stream, stamp, set index, scale): scale 1 = the alone launch, 3 = the beside launch (3x the bytes).
using Launch = std::function<void(cudaStream_t, Stamp *, int, int)>;

// Alone: cudaEvent median over reps, each rep a different expert set.
static void measure_alone(Row &row, const Launch &L, const std::function<double(int)> &bytes_of, int reps) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    L(sA, nullptr, reps + 1, 1);   // warm-up (instruction cache, TLB for the code; its set is not reused next)
    CK(cudaStreamSynchronize(sA));
    std::vector<double> t;
    for (int r = 0; r < reps; ++r) {
        CK(cudaEventRecord(a, sA));
        L(sA, nullptr, r, 1);
        CK(cudaEventRecord(b, sA));
        CK(cudaEventSynchronize(b));
        float ms;
        CK(cudaEventElapsedTime(&ms, a, b));
        t.push_back(ms * 1e3);
    }
    CK(cudaGetLastError());
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    row.bytes = bytes_of(1);
    row.us = median(t);
    row.gbs = row.bytes / row.us / 1e3;
    row.eff_lat_us = row.inflight_kb_sm * 1024.0 * g_sms / (row.gbs * 1e9) * 1e6;
}

// Beside: the streamer (3x bytes) on sA, then latk on sB; each kernel's span from its own %globaltimer stamps.
static void measure_beside(Row &row, const Launch &L, double b3, int reps) {
    std::vector<double> gb, lu, ov, mk;
    for (int r = 0; r < reps; ++r) {
        reset_stamps(2);
        L(sA, d_st, r, 3);
        launch_latk(sB, d_st + 1, r + 11);
        CK(cudaStreamSynchronize(sA));
        CK(cudaStreamSynchronize(sB));
        Stamp h[2];
        CK(cudaMemcpy(h, d_st, sizeof h, cudaMemcpyDeviceToHost));
        const double s_ns = (double)(h[0].end - h[0].start), l_ns = (double)(h[1].end - h[1].start);
        const double lo = (double)std::max(h[0].start, h[1].start), hi = (double)std::min(h[0].end, h[1].end);
        const double span = (double)(std::max(h[0].end, h[1].end) - std::min(h[0].start, h[1].start));
        gb.push_back(b3 / s_ns);                                          // bytes / ns = GB/s
        lu.push_back(l_ns / 1e3);
        ov.push_back(std::max(0.0, hi - lo) / l_ns);                      // fraction of latk inside the streamer
        mk.push_back(span / (b3 / row.gbs + g_latk_alone_us * 1e3));      // vs the two back to back, alone speeds
    }
    CK(cudaGetLastError());
    row.has_b = true;
    row.gbs_b = median(gb);
    row.str_slow = row.gbs / row.gbs_b;
    row.latk_us = g_latk_alone_us;
    row.latk_us_b = median(lu);
    row.latk_slow = row.latk_us_b / g_latk_alone_us;
    row.overlap = median(ov);
    row.makespan_ratio = median(mk);
}

static void measure(Row &row, const Launch &L, const std::function<double(int)> &bytes_of, const Opt &o) {
    measure_alone(row, L, bytes_of, o.reps);
    if (o.beside) measure_beside(row, L, bytes_of(3), o.reps);
}

static void print_row_header() {
    printf("%-8s %-7s %-22s %4s %5s %8s %6s %5s %8s %8s %8s | %8s %6s %8s %6s %5s %6s\n", "mech", "pattern", "config",
           "CTA", "warps", "KB/SM", "smemKB", "regs", "us", "GB/s", "effLat", "GB/s-b", "str-x", "latk-us", "latk-x",
           "ovl", "mksp");
}
static void print_row(const Row &r) {
    printf("%-8s %-7s %-22s %4d %5d %8.1f %6.1f %5d %8.1f %8.1f %8.2f", r.mech.c_str(), r.pattern.c_str(),
           r.cfg.c_str(), r.ctas_sm, r.warps_sm, r.inflight_kb_sm, r.smem_kb_sm, r.regs, r.us, r.gbs, r.eff_lat_us);
    if (r.has_b)
        printf(" | %8.1f %6.2f %8.1f %6.2f %5.2f %6.2f", r.gbs_b, r.str_slow, r.latk_us_b, r.latk_slow, r.overlap,
               r.makespan_ratio);
    printf("%s\n", r.gbs > 250 ? "  <-- > 250 GB/s: L2-resident?" : "");
    fflush(stdout);
}

static int occupancy(const void *k, int threads, size_t dsm) {
    int n = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k, threads, dsm));
    return n;
}

// ---- random helpers ------------------------------------------------------------------------------------------------
static std::mt19937_64 rng(0x1177);

// n distinct values in [0, range)
static std::vector<uint64_t> distinct(uint64_t n, uint64_t range) {
    std::vector<uint64_t> v;
    if (n * 4 > range) {
        std::vector<uint64_t> all(range);
        for (uint64_t i = 0; i < range; ++i) all[i] = i;
        std::shuffle(all.begin(), all.end(), rng);
        all.resize(n);
        return all;
    }
    std::unordered_set<uint64_t> seen;
    while (v.size() < n) {
        uint64_t x = rng() % range;
        if (seen.insert(x).second) v.push_back(x);
    }
    return v;
}

int main(int argc, char **argv) {
    Opt o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); }
            return argv[++i];
        };
        if (a == "--quick") o.quick = true;
        else if (a == "--json") o.json = next();
        else if (a == "--gb") o.gb = atof(next());
        else if (a == "--only") o.only = next();
        else if (a == "--reps") o.reps = atoi(next());
        else if (a == "--no-beside") o.beside = false;
        else if (a == "--gate-gbs") o.gate_gbs = atof(next());
        else {
            fprintf(stderr, "usage: %s [--quick] [--json path] [--gb GB] [--only chase|nc|cpasync|bulk] [--reps N] "
                            "[--no-beside] [--gate-gbs 230]\n", argv[0]);
            return 2;
        }
    }
    if (o.gb <= 0) o.gb = o.quick ? 4.0 : 8.0;
    if (o.reps <= 0) o.reps = o.quick ? 5 : 11;
    auto run = [&](const char *m) { return o.only.empty() || o.only == m; };
    if (!o.only.empty() && !run("chase") && !run("nc") && !run("cpasync") && !run("bulk")) {
        fprintf(stderr, "--only must be chase, nc, cpasync or bulk\n");
        return 2;
    }

    CK(cudaSetDevice(g_dev));
    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, g_dev));
    g_sms = pr.multiProcessorCount;
    int l2 = 0, clk = 0, memclk = 0, busw = 0, optin = 0, maxthr = 0, regs_sm = 0, maxblk = 0;
    cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, g_dev);
    cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, g_dev);
    cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, g_dev);
    cudaDeviceGetAttribute(&busw, cudaDevAttrGlobalMemoryBusWidth, g_dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, g_dev);
    cudaDeviceGetAttribute(&maxthr, cudaDevAttrMaxThreadsPerMultiProcessor, g_dev);
    cudaDeviceGetAttribute(&regs_sm, cudaDevAttrMaxRegistersPerMultiprocessor, g_dev);
    cudaDeviceGetAttribute(&maxblk, cudaDevAttrMaxBlocksPerMultiprocessor, g_dev);
    const std::string smi0 = nvsmi();
    const double mhz0 = measured_mhz();
    printf("littles: device %s cc %d.%d SMs %d L2 %.1f MB clockattr %.0f MHz memclk %d kHz bus %d bit smem-optin %d "
           "thr/SM %d regs/SM %d blocks/SM %d\n",
           pr.name, pr.major, pr.minor, g_sms, l2 / 1048576.0, clk / 1e3, memclk, busw, optin, maxthr, regs_sm, maxblk);
    printf("clock start: nvidia-smi [%s] measured %.0f MHz\n", smi0.c_str(), mhz0);
    printf("options: quick %d gb %.2f reps %d beside %d only %s gate %.0f GB/s\n", o.quick, o.gb, o.reps, o.beside,
           o.only.empty() ? "all" : o.only.c_str(), o.gate_gbs);
    fflush(stdout);

    CK(cudaStreamCreateWithFlags(&sA, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&sB, cudaStreamNonBlocking));
    CK(cudaMalloc(&g_sink, 256 * sizeof(uint4)));
    CK(cudaMalloc(&d_st, 4 * sizeof(Stamp)));
    CK(cudaMalloc(&g_latk_out, LATK_CTAS * sizeof(float)));

    // ---- pool ----------------------------------------------------------------------------------------------------
    const int pool_experts = (int)(o.gb * 1e9 / EXPERT_BYTES);
    if (pool_experts < 240) {
        fprintf(stderr, "--gb %.2f too small (need >= 1.5 GB)\n", o.gb);
        return 2;
    }
    g_pool_bytes = (uint64_t)pool_experts * EXPERT_BYTES;
    CK(cudaMalloc(&g_pool, g_pool_bytes));
    CK(cudaMemset(g_pool, 0x5a, g_pool_bytes));
    const uint64_t FLUSH_BYTES = 128ull << 20;                  // >= 4x L2
    const uint8_t *flush_base = g_pool + g_pool_bytes - FLUSH_BYTES;
    auto flush_l2 = [&]() {
        flush_kernel<<<4 * g_sms, 256, 0, sA>>>(reinterpret_cast<const uint4 *>(flush_base), FLUSH_BYTES / 16, g_sink);
        CK(cudaStreamSynchronize(sA));
    };
    const uint64_t CHASE_REGION = std::min<uint64_t>(2ull << 30, (g_pool_bytes - FLUSH_BYTES) / 2);
    printf("pool %d experts x %.2f MB = %.2f GB, chase region %.2f GB, flush %llu MB\n", pool_experts,
           EXPERT_BYTES / 1e6, g_pool_bytes / 1e9, CHASE_REGION / 1e9, (unsigned long long)(FLUSH_BYTES >> 20));

    // expert sets: gathered (random distinct experts) and flat (consecutive experts), NSETS each
    const double TARGET = 256.0 * 1048576.0;                    // bytes an alone launch (~1.1 ms at 235 GB/s)
    const int nexp_max = std::min(pool_experts, (int)(3 * 1.5 * TARGET / EXPERT_BYTES) + 3);
    const int NSETS = o.reps + 3 + 12;                           // + latk / warm-up offsets
    std::vector<int> hg((size_t)NSETS * nexp_max), hf((size_t)NSETS * nexp_max);
    for (int s = 0; s < NSETS; ++s) {
        auto d = distinct(nexp_max, pool_experts);
        for (int u = 0; u < nexp_max; ++u) hg[(size_t)s * nexp_max + u] = (int)d[u];
        const int s0 = (int)(((uint64_t)s * (nexp_max + 13)) % (uint64_t)(pool_experts - nexp_max + 1));
        for (int u = 0; u < nexp_max; ++u) hf[(size_t)s * nexp_max + u] = s0 + u;
    }
    int *d_sets_g, *d_sets_f;
    CK(cudaMalloc(&d_sets_g, hg.size() * sizeof(int)));
    CK(cudaMalloc(&d_sets_f, hf.size() * sizeof(int)));
    CK(cudaMemcpy(d_sets_g, hg.data(), hg.size() * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_sets_f, hf.data(), hf.size() * sizeof(int), cudaMemcpyHostToDevice));
    const int max_segs = nexp_max * SEGS_PER_EXPERT;
    struct Pat {
        const char *name;
        int dev;
        int *sets;
    };
    std::vector<Pat> pats = {{"tile", PAT_TILE, d_sets_g}, {"contig", PAT_CONTIG, d_sets_g}, {"flat", PAT_CONTIG, d_sets_f}};
    auto set_ptr = [&](const Pat &p, int s) { return p.sets + (size_t)(s % NSETS) * nexp_max; };

    // ---- registers of every kernel ---------------------------------------------------------------------------------
    int r_nc[9] = {0}, r_cpa[9] = {0}, r_bulk[5] = {0}, r_ct[5] = {0};
    r_nc[1] = kinfo("nc_kernel<1>", nc_kernel<1>);
    r_nc[2] = kinfo("nc_kernel<2>", nc_kernel<2>);
    r_nc[4] = kinfo("nc_kernel<4>", nc_kernel<4>);
    r_nc[8] = kinfo("nc_kernel<8>", nc_kernel<8>);
    r_cpa[2] = kinfo("cpa_kernel<2>", cpa_kernel<2>);
    r_cpa[3] = kinfo("cpa_kernel<3>", cpa_kernel<3>);
    r_cpa[4] = kinfo("cpa_kernel<4>", cpa_kernel<4>);
    r_cpa[6] = kinfo("cpa_kernel<6>", cpa_kernel<6>);
    r_cpa[8] = kinfo("cpa_kernel<8>", cpa_kernel<8>);
    r_bulk[2] = kinfo("bulk_kernel<2>", bulk_kernel<2>);
    r_bulk[3] = kinfo("bulk_kernel<3>", bulk_kernel<3>);
    r_bulk[4] = kinfo("bulk_kernel<4>", bulk_kernel<4>);
    r_ct[1] = kinfo("chase_tput_kernel<1>", chase_tput_kernel<1>);
    r_ct[2] = kinfo("chase_tput_kernel<2>", chase_tput_kernel<2>);
    r_ct[4] = kinfo("chase_tput_kernel<4>", chase_tput_kernel<4>);
    kinfo("chase1_kernel", chase1_kernel);
    kinfo("loaded_kernel", loaded_kernel);
    kinfo("latk_kernel", latk_kernel);
    printf("\n== kernels (cudaFuncGetAttributes) ==\n%-24s %5s %8s %8s\n", "kernel", "regs", "local B", "smem B");
    for (auto &k : g_kinfo) printf("%-24s %5d %8d %8d%s\n", k.name.c_str(), k.regs, k.local, k.smem, k.local ? "  SPILL" : "");
    for (auto k : {(const void *)cpa_kernel<2>, (const void *)cpa_kernel<3>, (const void *)cpa_kernel<4>,
                   (const void *)cpa_kernel<6>, (const void *)cpa_kernel<8>, (const void *)bulk_kernel<2>,
                   (const void *)bulk_kernel<3>, (const void *)bulk_kernel<4>})
        CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));

    // ---- latk alone ------------------------------------------------------------------------------------------------
    {
        std::vector<double> t;
        launch_latk(sA, nullptr, 0);
        CK(cudaStreamSynchronize(sA));
        for (int r = 0; r < o.reps; ++r) {
            reset_stamps(1);
            launch_latk(sA, d_st, r + 11);
            CK(cudaStreamSynchronize(sA));
            Stamp h;
            CK(cudaMemcpy(&h, d_st, sizeof h, cudaMemcpyDeviceToHost));
            t.push_back((h.end - h.start) / 1e3);
        }
        g_latk_alone_us = median(t);
        printf("\nlatk alone (32 CTAs x 1024 thr, %d dependent steps): %.1f us (%.2f us a step)\n", LATK_T,
               g_latk_alone_us, g_latk_alone_us / LATK_T);
    }

    std::string jlat, jload;
    // ================================================================================================================
    // 1. latency
    // ================================================================================================================
    unsigned long long *d_out;
    CK(cudaMalloc(&d_out, 8 * sizeof(unsigned long long)));
    int *d_done;
    CK(cudaMalloc(&d_done, sizeof(int)));
    double lat_idle_ns = 0;
    if (run("chase")) {
        printf("\n== 1. pointer-chase latency (1 thread, ld.global.cg) ==\n");
        printf("%-10s %-10s %10s %8s %10s %10s %8s\n", "kind", "granule", "region MB", "steps", "ns/step", "cyc/step",
               "MHz");
        const int steps = o.quick ? 8192 : 32768;
        struct LC {
            const char *kind;
            uint64_t g, region;
            bool l2;
        };
        std::vector<LC> lcs = {{"l2-hit", 128, 8ull << 20, true},       {"dram", 64, CHASE_REGION, false},
                               {"dram", 128, CHASE_REGION, false},      {"dram", 4096, CHASE_REGION, false},
                               {"dram", 2ull << 20, g_pool_bytes - FLUSH_BYTES, false}};
        for (auto &lc : lcs) {
            const uint64_t nslots = lc.region / lc.g;
            const int m = (int)std::min<uint64_t>(lc.l2 ? 16384 : (uint64_t)steps + 1, nslots);
            auto sl = distinct(m, nslots);
            std::vector<uint64_t> offs(m);
            for (int i = 0; i < m; ++i)
                offs[i] = sl[i] * lc.g + (lc.g >= 4096 ? (rng() % (lc.g / 128)) * 128 : 0);
            uint64_t *d_offs;
            CK(cudaMalloc(&d_offs, m * sizeof(uint64_t)));
            CK(cudaMemcpy(d_offs, offs.data(), m * sizeof(uint64_t), cudaMemcpyHostToDevice));
            link_kernel<<<64, 256>>>(g_pool, d_offs, m);
            CK(cudaDeviceSynchronize());
            CK(cudaFree(d_offs));
            std::vector<double> ns, cy;
            const int nt = lc.l2 ? m : m - 1, warm = lc.l2 ? m : 0;
            for (int r = 0; r < (o.quick ? 2 : 3); ++r) {
                flush_l2();
                chase1_kernel<<<1, 1, 0, sA>>>(g_pool, offs[0], warm, nt, d_out);
                unsigned long long h[3];
                CK(cudaMemcpy(h, d_out, sizeof h, cudaMemcpyDeviceToHost));
                ns.push_back((double)h[1] / nt);
                cy.push_back((double)h[0] / nt);
            }
            const double n_ = median(ns), c_ = median(cy);
            if (!lc.l2 && lc.g == 128) lat_idle_ns = n_;
            const std::string gs = lc.g >= (1u << 20) ? fmt("%llu MB", (unsigned long long)(lc.g >> 20))
                                   : lc.g >= 1024 ? fmt("%llu KB", (unsigned long long)(lc.g >> 10))
                                                  : fmt("%llu B", (unsigned long long)lc.g);
            printf("%-10s %-10s %10.0f %8d %10.1f %10.1f %8.0f\n", lc.kind, gs.c_str(), lc.region / 1048576.0, nt, n_,
                   c_, c_ / n_ * 1e3);
            jlat += fmt("%s{\"kind\":\"%s\",\"granule_bytes\":%llu,\"region_bytes\":%llu,\"steps\":%d,\"ns\":%.2f,"
                        "\"cycles\":%.1f,\"mhz\":%.0f}",
                        jlat.empty() ? "" : ",", lc.kind, (unsigned long long)lc.g, (unsigned long long)lc.region, nt,
                        n_, c_, c_ / n_ * 1e3);
        }
    }

    // 512-B node cycle over the chase region (loaded chase + chase throughput)
    const int M = (int)(CHASE_REGION / NODE_BYTES);
    std::vector<uint32_t> perm;
    if (run("chase")) {
        perm.resize(M);
        for (int i = 0; i < M; ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        uint32_t *d_perm;
        CK(cudaMalloc(&d_perm, (size_t)M * 4));
        CK(cudaMemcpy(d_perm, perm.data(), (size_t)M * 4, cudaMemcpyHostToDevice));
        cycle_kernel<<<4 * g_sms, 256>>>(g_pool, d_perm, M);
        CK(cudaDeviceSynchronize());
        CK(cudaFree(d_perm));

        printf("\n== 1b. loaded latency: 1 chasing thread (512-B random nodes) + N streaming CTAs (nc D=4, tile) ==\n");
        printf("%8s %8s %10s %10s %10s %10s\n", "stream", "CTAs/SM", "ns/step", "cyc/step", "x idle", "stream GB/s");
        const int occ = occupancy((const void *)loaded_kernel, 128, 0);
        const int warm = 256, n = o.quick ? 2048 : 4096;
        for (int N : {0, 12, 24, 48, 96, 192, 384}) {
            if (N + 1 > occ * g_sms) continue;
            std::vector<double> ns, cy, gbs;
            for (int r = 0; r < (o.quick ? 2 : 3); ++r) {
                flush_l2();
                reset_stamps(1);
                CK(cudaMemset(d_out, 0, 8 * sizeof(unsigned long long)));
                CK(cudaMemset(d_done, 0, sizeof(int)));
                SArgs a{g_pool, set_ptr(pats[0], r), PAT_TILE, 0, 0, 0, 0, 0u, g_sink, d_st};
                const uint64_t start = (uint64_t)perm[(size_t)r * (M / 8)] * NODE_BYTES;
                loaded_kernel<<<N + 1, 128, 0, sA>>>(a, max_segs, g_pool, start, warm, n, d_done, d_out);
                CK(cudaStreamSynchronize(sA));
                unsigned long long h[4];
                Stamp hs;
                CK(cudaMemcpy(h, d_out, sizeof h, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(&hs, d_st, sizeof hs, cudaMemcpyDeviceToHost));
                ns.push_back((double)h[1] / n);
                cy.push_back((double)h[0] / n);
                gbs.push_back(N ? (double)h[2] / (double)(hs.end - hs.start) : 0.0);
            }
            const double n_ = median(ns), c_ = median(cy), g_ = median(gbs);
            if (N == 0 && lat_idle_ns == 0) lat_idle_ns = n_;
            printf("%8d %8.2f %10.1f %10.1f %10.2f %10.1f\n", N, (double)N / g_sms, n_, c_,
                   lat_idle_ns > 0 ? n_ / lat_idle_ns : 0.0, g_);
            jload += fmt("%s{\"stream_ctas\":%d,\"ns\":%.2f,\"cycles\":%.1f,\"stream_gbs\":%.1f}", jload.empty() ? "" : ",",
                         N, n_, c_, g_);
        }
    }

    // ================================================================================================================
    // 2 + 3. Little's-law sweep, alone and beside latk
    // ================================================================================================================
    printf("\n== 2. Little's-law sweep (KB/SM = nominal bytes in flight an SM; effLat = in flight / throughput, us; "
           "-b = beside latk: streamer GB/s, x slowdown, latk us / x, overlap of latk inside the streamer, makespan / "
           "back-to-back) ==\n");
    print_row_header();
    auto segs_for = [&](int TW, int scale) {
        int s = std::max(1, (int)llround(TARGET / ((double)TW * SEG_BYTES)));
        s *= scale;
        return std::min(s, max_segs / TW);
    };

    // (d) chase throughput
    if (run("chase")) {
        uint64_t *d_starts;
        const int maxQ = g_sms * 48 * 4;
        CK(cudaMalloc(&d_starts, (size_t)maxQ * sizeof(uint64_t)));
        struct CC {
            int thr, cps, ch;
        };
        std::vector<CC> ccs;
        for (int ch : {1, 2, 4}) {
            ccs.push_back({32, 1, ch});
            for (int c : {1, 2, 4, 8, 12}) ccs.push_back({128, c, ch});
        }
        for (auto &cc : ccs) {
            if (o.quick && (cc.cps == 2 || cc.cps == 12)) continue;
            const void *k = cc.ch == 1 ? (const void *)chase_tput_kernel<1> : cc.ch == 2 ? (const void *)chase_tput_kernel<2>
                                                                                            : (const void *)chase_tput_kernel<4>;
            if (occupancy(k, cc.thr, 0) < cc.cps) continue;
            const int grid = g_sms * cc.cps, warps = grid * cc.thr / 32, Q = warps * cc.ch;
            std::vector<uint64_t> st(Q);
            for (int q = 0; q < Q; ++q) st[q] = (uint64_t)perm[(size_t)q * (M / Q)] * NODE_BYTES;
            CK(cudaMemcpy(d_starts, st.data(), Q * sizeof(uint64_t), cudaMemcpyHostToDevice));
            const int L1 = std::min(256, M / Q), L3 = std::min(768, M / Q);
            Row row;
            row.mech = "chase";
            row.pattern = "random";
            row.cfg = fmt("thr%d chains%d", cc.thr, cc.ch);
            row.ctas_sm = cc.cps;
            row.warps_sm = cc.cps * cc.thr / 32;
            row.regs = r_ct[cc.ch];
            row.data_regs = 4 * cc.ch;
            row.inflight_kb_sm = row.warps_sm * cc.ch * NODE_BYTES / 1024.0;
            row.smem_kb_sm = 0;
            // successive reps restart the same chains: flush first so they stay cold
            Launch L = [&, cc, grid, L1, L3](cudaStream_t s, Stamp *stp, int, int scale) {
                flush_kernel<<<4 * g_sms, 256, 0, s>>>(reinterpret_cast<const uint4 *>(flush_base), FLUSH_BYTES / 16, g_sink);
                if (stp) CK(cudaStreamSynchronize(s));   // beside: keep the flush out of the streamer's span
                const int Ls = scale == 1 ? L1 : L3;
                if (cc.ch == 1) chase_tput_kernel<1><<<grid, cc.thr, 0, s>>>(g_pool, d_starts, Ls, g_sink, stp);
                else if (cc.ch == 2) chase_tput_kernel<2><<<grid, cc.thr, 0, s>>>(g_pool, d_starts, Ls, g_sink, stp);
                else chase_tput_kernel<4><<<grid, cc.thr, 0, s>>>(g_pool, d_starts, Ls, g_sink, stp);
            };
            // alone timing must exclude the flush: time the chase with its own events
            {
                std::vector<double> t;
                cudaEvent_t a, b;
                CK(cudaEventCreate(&a));
                CK(cudaEventCreate(&b));
                for (int r = 0; r < o.reps; ++r) {
                    flush_l2();
                    CK(cudaEventRecord(a, sA));
                    if (cc.ch == 1) chase_tput_kernel<1><<<grid, cc.thr, 0, sA>>>(g_pool, d_starts, L1, g_sink, nullptr);
                    else if (cc.ch == 2) chase_tput_kernel<2><<<grid, cc.thr, 0, sA>>>(g_pool, d_starts, L1, g_sink, nullptr);
                    else chase_tput_kernel<4><<<grid, cc.thr, 0, sA>>>(g_pool, d_starts, L1, g_sink, nullptr);
                    CK(cudaEventRecord(b, sA));
                    CK(cudaEventSynchronize(b));
                    float ms;
                    CK(cudaEventElapsedTime(&ms, a, b));
                    t.push_back(ms * 1e3);
                }
                CK(cudaEventDestroy(a));
                CK(cudaEventDestroy(b));
                row.bytes = (double)Q * L1 * NODE_BYTES;
                row.us = median(t);
                row.gbs = row.bytes / row.us / 1e3;
                row.eff_lat_us = row.inflight_kb_sm * 1024.0 * g_sms / (row.gbs * 1e9) * 1e6;
            }
            if (o.beside) measure_beside(row, L, (double)Q * L3 * NODE_BYTES, std::max(3, o.reps / 2));
            print_row(row);
            g_rows.push_back(row);
        }
        CK(cudaFree(d_starts));
    }

    // (a) nc
    if (run("nc")) {
        for (auto &p : pats) {
            if (o.quick && strcmp(p.name, "tile") && strcmp(p.name, "flat")) continue;
            for (int C : {1, 2, 4, 8, 12})
                for (int D : {1, 2, 4, 8}) {
                    if (o.quick && C == 12) continue;
                    const void *k = D == 1 ? (const void *)nc_kernel<1> : D == 2 ? (const void *)nc_kernel<2>
                                              : D == 4 ? (const void *)nc_kernel<4> : (const void *)nc_kernel<8>;
                    if (occupancy(k, 128, 0) < C) continue;
                    const int grid = g_sms * C, TW = grid * 4;
                    Row row;
                    row.mech = "nc";
                    row.pattern = p.name;
                    row.cfg = fmt("D%d", D);
                    row.ctas_sm = C;
                    row.warps_sm = 4 * C;
                    row.regs = r_nc[D];
                    row.data_regs = 4 * D;
                    row.inflight_kb_sm = row.warps_sm * D * STEP_BYTES / 1024.0;
                    row.smem_kb_sm = 0;
                    Launch L = [&, D, grid, TW, p](cudaStream_t s, Stamp *stp, int set, int scale) {
                        SArgs a{g_pool, set_ptr(p, set), p.dev, segs_for(TW, scale), 0, 0, 0, 0u, g_sink, stp};
                        if (D == 1) nc_kernel<1><<<grid, 128, 0, s>>>(a);
                        else if (D == 2) nc_kernel<2><<<grid, 128, 0, s>>>(a);
                        else if (D == 4) nc_kernel<4><<<grid, 128, 0, s>>>(a);
                        else nc_kernel<8><<<grid, 128, 0, s>>>(a);
                    };
                    measure(row, L, [&, TW](int sc) { return (double)TW * segs_for(TW, sc) * SEG_BYTES; }, o);
                    print_row(row);
                    g_rows.push_back(row);
                }
        }
    }

    // (b) cp.async ring
    if (run("cpasync")) {
        for (auto &p : pats) {
            if (o.quick && strcmp(p.name, "tile") && strcmp(p.name, "flat")) continue;
            for (int C : {2, 3, 4, 8})
                for (int S : {2, 3, 4, 6, 8})
                    for (int G : {1, 2, 4}) {
                        if (o.quick && (S == 3 || S == 6 || C == 2)) continue;
                        const void *k = S == 2 ? (const void *)cpa_kernel<2> : S == 3 ? (const void *)cpa_kernel<3>
                                        : S == 4 ? (const void *)cpa_kernel<4> : S == 6 ? (const void *)cpa_kernel<6>
                                                                                 : (const void *)cpa_kernel<8>;
                        const size_t dsm = (size_t)4 * S * G * STEP_BYTES;
                        if ((int)dsm > optin || occupancy(k, 128, dsm) < C) continue;
                        const int grid = g_sms * C, TW = grid * 4;
                        Row row;
                        row.mech = "cpasync";
                        row.pattern = p.name;
                        row.cfg = fmt("S%d G%d%s", S, G, (S == 4 && G == 1 && C == 3) ? " (E1)" : "");
                        row.ctas_sm = C;
                        row.warps_sm = 4 * C;
                        row.regs = r_cpa[S];
                        row.data_regs = 0;
                        row.inflight_kb_sm = row.warps_sm * (S - 1) * G * STEP_BYTES / 1024.0;
                        row.smem_kb_sm = C * dsm / 1024.0;
                        Launch L = [&, S, G, grid, TW, dsm, p](cudaStream_t s, Stamp *stp, int set, int scale) {
                            SArgs a{g_pool, set_ptr(p, set), p.dev, segs_for(TW, scale), G, 0, 0, 0u, g_sink, stp};
                            if (S == 2) cpa_kernel<2><<<grid, 128, dsm, s>>>(a);
                            else if (S == 3) cpa_kernel<3><<<grid, 128, dsm, s>>>(a);
                            else if (S == 4) cpa_kernel<4><<<grid, 128, dsm, s>>>(a);
                            else if (S == 6) cpa_kernel<6><<<grid, 128, dsm, s>>>(a);
                            else cpa_kernel<8><<<grid, 128, dsm, s>>>(a);
                        };
                        measure(row, L, [&, TW](int sc) { return (double)TW * segs_for(TW, sc) * SEG_BYTES; }, o);
                        print_row(row);
                        g_rows.push_back(row);
                    }
        }
    }

    // (c) TMA bulk
    if (run("bulk")) {
        for (auto &p : pats) {
            if (o.quick && strcmp(p.name, "tile") && strcmp(p.name, "flat")) continue;
            for (int CHk : {4, 8, 16})
                for (int S : {2, 3, 4})
                    for (int C : {1, 2, 4}) {
                        if (o.quick && S == 3) continue;
                        const int CH = CHk * 1024;
                        const void *k = S == 2 ? (const void *)bulk_kernel<2> : S == 3 ? (const void *)bulk_kernel<3>
                                                                                        : (const void *)bulk_kernel<4>;
                        const size_t dsm = (size_t)S * CH;
                        if ((int)dsm > optin || occupancy(k, 256, dsm) < C) continue;
                        const int grid = g_sms * C;
                        auto nck_for = [&, grid, CH](int scale) {
                            int n = std::max(1, (int)llround(TARGET / ((double)grid * CH))) * scale;
                            const int cap = (int)((double)max_segs * SEG_BYTES / ((double)grid * CH));
                            return std::min(n, cap);
                        };
                        Row row;
                        row.mech = "bulk";
                        row.pattern = p.name;
                        row.cfg = fmt("CH%dK S%d", CHk, S);
                        row.ctas_sm = C;
                        row.warps_sm = 8 * C;
                        row.regs = r_bulk[S];
                        row.data_regs = 0;
                        row.inflight_kb_sm = (double)C * S * CHk;
                        row.smem_kb_sm = C * dsm / 1024.0;
                        Launch L = [&, S, CH, grid, dsm, p, nck_for](cudaStream_t s, Stamp *stp, int set, int scale) {
                            SArgs a{g_pool, set_ptr(p, set), p.dev, 0, 0, CH, nck_for(scale), 0u, g_sink, stp};
                            if (S == 2) bulk_kernel<2><<<grid, 256, dsm, s>>>(a);
                            else if (S == 3) bulk_kernel<3><<<grid, 256, dsm, s>>>(a);
                            else bulk_kernel<4><<<grid, 256, dsm, s>>>(a);
                        };
                        measure(row, L, [&, grid, CH, nck_for](int sc) { return (double)grid * nck_for(sc) * CH; }, o);
                        print_row(row);
                        g_rows.push_back(row);
                    }
        }
    }
    CK(cudaDeviceSynchronize());

    // ================================================================================================================
    // 4. summary + gate
    // ================================================================================================================
    const int warps_at_112 = regs_sm / (112 * 32);   // 18 on a 64K-register SM
    printf("\n== minimum bytes in flight an SM for >= %.0f GB/s (alone) ==\n", o.gate_gbs);
    printf("%-8s %-7s %12s %28s %12s %28s\n", "mech", "pattern", "any KB/SM", "row", "gate KB/SM", "row (<=112 regs, <=" "18 warps)");
    std::string jmin;
    struct Best {
        std::string mech, pattern, cfg;
        double kb = 1e30, gbs = 0;
        int warps = 0, regs = 0, data_regs = 0;
    };
    std::vector<Best> gate_best;
    Best overall_best_gbs;   // for the FAIL message
    for (const char *mech : {"chase", "nc", "cpasync", "bulk"})
        for (const char *pat : {"random", "tile", "contig", "flat"}) {
            Best any, gate;
            bool seen = false;
            double best_gbs = 0;
            for (auto &r : g_rows) {
                if (r.mech != mech || r.pattern != pat) continue;
                seen = true;
                best_gbs = std::max(best_gbs, r.gbs);
                const bool fast = r.gbs >= o.gate_gbs;
                if (fast && r.inflight_kb_sm < any.kb) any = {r.mech, r.pattern, fmt("%s C%d", r.cfg.c_str(), r.ctas_sm), r.inflight_kb_sm, r.gbs, r.warps_sm, r.regs, r.data_regs};
                const bool eligible = (!strcmp(mech, "bulk") || !strcmp(pat, "tile")) && strcmp(mech, "chase");
                if (fast && eligible && r.regs <= 112 && r.warps_sm <= warps_at_112 && r.inflight_kb_sm < gate.kb)
                    gate = {r.mech, r.pattern, fmt("%s C%d", r.cfg.c_str(), r.ctas_sm), r.inflight_kb_sm, r.gbs, r.warps_sm, r.regs, r.data_regs};
                if (eligible && r.gbs > overall_best_gbs.gbs)
                    overall_best_gbs = {r.mech, r.pattern, fmt("%s C%d", r.cfg.c_str(), r.ctas_sm), r.inflight_kb_sm, r.gbs, r.warps_sm, r.regs, r.data_regs};
            }
            if (!seen) continue;
            if (gate.kb < 1e29) gate_best.push_back(gate);
            printf("%-8s %-7s %12s %28s %12s %28s   (best %.1f GB/s)\n", mech, pat,
                   any.kb < 1e29 ? fmt("%.1f", any.kb).c_str() : "none", any.kb < 1e29 ? fmt("%s %.0f", any.cfg.c_str(), any.gbs).c_str() : "-",
                   gate.kb < 1e29 ? fmt("%.1f", gate.kb).c_str() : "none",
                   gate.kb < 1e29 ? fmt("%s %.0f", gate.cfg.c_str(), gate.gbs).c_str() : "-", best_gbs);
            jmin += fmt("%s{\"mech\":\"%s\",\"pattern\":\"%s\",\"best_gbs\":%.1f,\"min_kb_sm_any\":%s,\"row_any\":\"%s\","
                        "\"min_kb_sm_gate\":%s,\"row_gate\":\"%s\"}",
                        jmin.empty() ? "" : ",", mech, pat, best_gbs, any.kb < 1e29 ? fmt("%.2f", any.kb).c_str() : "null",
                        any.kb < 1e29 ? any.cfg.c_str() : "", gate.kb < 1e29 ? fmt("%.2f", gate.kb).c_str() : "null",
                        gate.kb < 1e29 ? gate.cfg.c_str() : "");
        }
    bool any_high = false;
    for (auto &r : g_rows) any_high |= r.gbs > 250;
    if (any_high) printf("WARNING: rows above 250 GB/s (cannot be DRAM; check for L2 residency)\n");

    const std::string smi1 = nvsmi();
    const double mhz1 = measured_mhz();
    printf("\nclock end: nvidia-smi [%s] measured %.0f MHz (start %.0f)\n", smi1.c_str(), mhz1, mhz0);

    std::string gate_line, gate_mech = "none";
    bool pass = false;
    double gate_kb = 0;
    const bool measured = run("nc") || run("cpasync") || run("bulk");
    if (!gate_best.empty()) {
        auto b = *std::min_element(gate_best.begin(), gate_best.end(), [](const Best &x, const Best &y) { return x.kb < y.kb; });
        pass = true;
        gate_mech = b.mech;
        gate_kb = b.kb;
        gate_line = fmt("GATE item8: %s reaches >= %.0f GB/s at <= %.1f KB/SM in flight with <= 112 regs/thread: PASS "
                        "(%s %s, %.1f GB/s, %d warps/SM, %d regs, %d data regs)",
                        b.mech.c_str(), o.gate_gbs, b.kb, b.pattern.c_str(), b.cfg.c_str(), b.gbs, b.warps, b.regs, b.data_regs);
    } else if (measured) {
        gate_line = fmt("GATE item8: %s reaches >= %.0f GB/s at <= n/a KB/SM in flight with <= 112 regs/thread: FAIL "
                        "(best eligible: %s %s %s, %.1f GB/s at %.1f KB/SM, %d warps/SM)",
                        overall_best_gbs.mech.empty() ? "none" : overall_best_gbs.mech.c_str(), o.gate_gbs,
                        overall_best_gbs.mech.c_str(), overall_best_gbs.pattern.c_str(), overall_best_gbs.cfg.c_str(),
                        overall_best_gbs.gbs, overall_best_gbs.kb, overall_best_gbs.warps);
    } else {
        gate_line = fmt("GATE item8: none reaches >= %.0f GB/s at <= n/a KB/SM in flight with <= 112 regs/thread: FAIL "
                        "(no load mechanism run: --only %s)", o.gate_gbs, o.only.c_str());
    }

    // ---- JSON ------------------------------------------------------------------------------------------------------
    if (o.json) {
        std::string js = "{";
        js += fmt("\"device\":{\"name\":\"%s\",\"cc\":\"%d.%d\",\"sms\":%d,\"l2_mb\":%.1f,\"clock_attr_mhz\":%.0f,"
                  "\"memclk_khz\":%d,\"bus_bits\":%d,\"smem_optin\":%d,\"regs_sm\":%d},",
                  pr.name, pr.major, pr.minor, g_sms, l2 / 1048576.0, clk / 1e3, memclk, busw, optin, regs_sm);
        js += fmt("\"options\":{\"quick\":%s,\"gb\":%.2f,\"reps\":%d,\"beside\":%s,\"only\":\"%s\",\"gate_gbs\":%.1f,"
                  "\"launch_bytes_target\":%.0f},",
                  o.quick ? "true" : "false", o.gb, o.reps, o.beside ? "true" : "false", o.only.c_str(), o.gate_gbs, TARGET);
        js += fmt("\"clocks\":{\"start_nvsmi\":\"%s\",\"end_nvsmi\":\"%s\",\"start_mhz\":%.0f,\"end_mhz\":%.0f},",
                  smi0.c_str(), smi1.c_str(), mhz0, mhz1);
        js += "\"kernels\":[";
        for (size_t i = 0; i < g_kinfo.size(); ++i)
            js += fmt("%s{\"name\":\"%s\",\"regs\":%d,\"local_bytes\":%d,\"static_smem\":%d}", i ? "," : "",
                      g_kinfo[i].name.c_str(), g_kinfo[i].regs, g_kinfo[i].local, g_kinfo[i].smem);
        js += fmt("],\"latk\":{\"ctas\":%d,\"threads\":1024,\"steps\":%d,\"alone_us\":%.2f},", LATK_CTAS, LATK_T,
                  g_latk_alone_us);
        js += "\"latency\":[" + jlat + "],\"loaded_latency\":[" + jload + "],\"sweep\":[";
        for (size_t i = 0; i < g_rows.size(); ++i) {
            const Row &r = g_rows[i];
            js += fmt("%s{\"mech\":\"%s\",\"pattern\":\"%s\",\"cfg\":\"%s\",\"ctas_sm\":%d,\"warps_sm\":%d,\"regs\":%d,"
                      "\"data_regs\":%d,\"inflight_kb_sm\":%.2f,\"smem_kb_sm\":%.2f,\"bytes\":%.0f,\"us\":%.2f,\"gbs\":%.2f,"
                      "\"eff_latency_us\":%.3f",
                      i ? "," : "", r.mech.c_str(), r.pattern.c_str(), r.cfg.c_str(), r.ctas_sm, r.warps_sm, r.regs,
                      r.data_regs, r.inflight_kb_sm, r.smem_kb_sm, r.bytes, r.us, r.gbs, r.eff_lat_us);
            if (r.has_b)
                js += fmt(",\"beside\":{\"gbs\":%.2f,\"stream_slowdown\":%.3f,\"latk_us\":%.2f,\"latk_slowdown\":%.3f,"
                          "\"overlap\":%.3f,\"makespan_vs_serial\":%.3f}",
                          r.gbs_b, r.str_slow, r.latk_us_b, r.latk_slow, r.overlap, r.makespan_ratio);
            js += "}";
        }
        js += "],\"min_inflight\":[" + jmin + "],";
        js += fmt("\"gate\":{\"pass\":%s,\"mechanism\":\"%s\",\"kb_sm\":%.2f,\"gbs_threshold\":%.1f,\"max_regs\":112,"
                  "\"max_warps_sm\":%d,\"line\":\"%s\"}}",
                  pass ? "true" : "false", gate_mech.c_str(), gate_kb, o.gate_gbs, warps_at_112, gate_line.c_str());
        FILE *f = fopen(o.json, "w");
        if (!f) {
            fprintf(stderr, "cannot write %s\n", o.json);
        } else {
            fputs(js.c_str(), f);
            fputs("\n", f);
            fclose(f);
            printf("json: %s\n", o.json);
        }
    }
    printf("\n%s\n", gate_line.c_str());
    return 0;
}
