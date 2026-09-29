// W11 bandwidth / launch probe for one GB10 GPU (measurement only; no engine code).
//
//   nvcc -O3 -arch=sm_121 -o probe probe.cu && ./probe [--quick]
//
// 1. Streaming READ (128-bit loads, 4 in flight a thread, xor into a sink) and COPY (read + write) kernels over the
//    byte volumes of one decode round:
//    - routed experts, one layer: U experts' gate+up (U x 4.19 MB) and down (U x 2.10 MB) per launch, the experts
//      either CONTIGUOUS (U neighbours) or GATHERED (U distinct random experts of 288, one 2D block row per expert as
//      the grouped kernel does), over a pool of 4 layers x 288 experts (7.2 GB: nothing is L2-resident);
//    - the whole round's expert reads: 42 layers x (gate/up, down) launches back to back at U, from the pool;
//    - dense q4 shapes: each rank's weight bytes (n x k x 0.5625: words + bf16 scales / biases) of every decode
//      shape, one launch each, and the verify's whole dense set (2.33 GB) in layer order back to back;
//    - a size sweep 0.25 MB .. 1 GB (one launch, cold) for the ramp: GB/s against bytes.
//    Each: best of the grid shapes (SMs x 1..8 blocks of 256 / 512 threads), median of reps.
// 2. Launch overhead and idle between small kernels: an empty kernel x 2,000 (eager stream, CUDA graph), a chain of
//    small dependent kernels (each reads 1 MB) eager / graph / graph + PDL, and the measured gap between one kernel's
//    last block and the next kernel's first block (%globaltimer), with and without programmatic dependent launch.
// 3. PDL availability on this GPU: a PDL launch (cudaLaunchAttributeProgrammaticStreamSerialization) and whether the
//    dependent kernel's prologue starts before the primary ends (early start observed = PDL is effective).
//
// Output: human table + one "W11PROBE {json}" line per measurement.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#define CK(x)                                                                                        \
    do {                                                                                             \
        cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
            exit(1);                                                                                 \
        }                                                                                            \
    } while (0)

__device__ __forceinline__ uint64_t gtimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ uint4 ldg_stream(const uint4 *p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

// 2D: blockIdx.y = expert u (base + idx[u] * chunk16), blockIdx.x strides within the chunk
__global__ void read_gather(const uint4 *__restrict__ base, const int *__restrict__ idx, uint64_t chunk16,
                            uint4 *sink) {
    const uint4 *p = base + (uint64_t)idx[blockIdx.y] * chunk16;
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t st = (uint64_t)gridDim.x * blockDim.x;
    uint4 a = make_uint4(0, 0, 0, 0);
    for (; i + 3 * st < chunk16; i += 4 * st) {
        uint4 v0 = ldg_stream(p + i), v1 = ldg_stream(p + i + st), v2 = ldg_stream(p + i + 2 * st),
              v3 = ldg_stream(p + i + 3 * st);
        a.x ^= v0.x ^ v1.x ^ v2.x ^ v3.x;
        a.y ^= v0.y ^ v1.y ^ v2.y ^ v3.y;
        a.z ^= v0.z ^ v1.z ^ v2.z ^ v3.z;
        a.w ^= v0.w ^ v1.w ^ v2.w ^ v3.w;
    }
    for (; i < chunk16; i += st) {
        uint4 v = ldg_stream(p + i);
        a.x ^= v.x; a.y ^= v.y; a.z ^= v.z; a.w ^= v.w;
    }
    if ((a.x & a.y & a.z & a.w) == 0xdeadbeef) sink[threadIdx.x] = a;
}

// flat read of n16 uint4 (grid-stride)
__global__ void read_flat(const uint4 *__restrict__ p, uint64_t n16, uint4 *sink) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t st = (uint64_t)gridDim.x * blockDim.x;
    uint4 a = make_uint4(0, 0, 0, 0);
    for (; i + 3 * st < n16; i += 4 * st) {
        uint4 v0 = ldg_stream(p + i), v1 = ldg_stream(p + i + st), v2 = ldg_stream(p + i + 2 * st),
              v3 = ldg_stream(p + i + 3 * st);
        a.x ^= v0.x ^ v1.x ^ v2.x ^ v3.x;
        a.y ^= v0.y ^ v1.y ^ v2.y ^ v3.y;
        a.z ^= v0.z ^ v1.z ^ v2.z ^ v3.z;
        a.w ^= v0.w ^ v1.w ^ v2.w ^ v3.w;
    }
    for (; i < n16; i += st) {
        uint4 v = ldg_stream(p + i);
        a.x ^= v.x; a.y ^= v.y; a.z ^= v.z; a.w ^= v.w;
    }
    if ((a.x & a.y & a.z & a.w) == 0xdeadbeef) sink[threadIdx.x] = a;
}

__global__ void copy_flat(const uint4 *__restrict__ s, uint4 *__restrict__ d, uint64_t n16) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t st = (uint64_t)gridDim.x * blockDim.x;
    for (; i + 3 * st < n16; i += 4 * st) {
        uint4 v0 = ldg_stream(s + i), v1 = ldg_stream(s + i + st), v2 = ldg_stream(s + i + 2 * st),
              v3 = ldg_stream(s + i + 3 * st);
        d[i] = v0; d[i + st] = v1; d[i + 2 * st] = v2; d[i + 3 * st] = v3;
    }
    for (; i < n16; i += st) d[i] = ldg_stream(s + i);
}

__global__ void empty_kernel() {}

// small dependent kernel: reads n16 (1 MB), stamps start / after-wait / end; pdl: griddepcontrol.wait first
struct Stamp {
    unsigned long long start, go, end;
};
__global__ void small_kernel(const uint4 *__restrict__ p, uint64_t n16, uint4 *sink, Stamp *st, int k,
                             unsigned int *done, int pdl) {
    if (threadIdx.x == 0) atomicMin(&st[k].start, (unsigned long long)gtimer());
    if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
    if (threadIdx.x == 0) atomicMin(&st[k].go, (unsigned long long)gtimer());
    if (pdl) asm volatile("griddepcontrol.launch_dependents;");
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t s = (uint64_t)gridDim.x * blockDim.x;
    uint4 a = make_uint4(0, 0, 0, 0);
    for (; i < n16; i += s) {
        uint4 v = ldg_stream(p + i);
        a.x ^= v.x; a.y ^= v.y; a.z ^= v.z; a.w ^= v.w;
    }
    if ((a.x & a.y & a.z & a.w) == 0xdeadbeef) sink[threadIdx.x] = a;
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        atomicMax(&st[k].end, (unsigned long long)gtimer());
    }
}

// ---------------------------------------------------------------------------------------------------------------
static int g_sms = 0;
static uint4 *g_sink = nullptr;
static cudaStream_t g_s;

static float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static void emit(const char *what, const std::string &json) { printf("W11PROBE {\"what\":\"%s\",%s}\n", what, json.c_str()); }

// time fn over reps with events; returns median ms
template <class F>
static float time_ms(F fn, int reps, int warm = 2) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    for (int i = 0; i < warm; i++) fn(i);
    CK(cudaStreamSynchronize(g_s));
    std::vector<float> t;
    for (int r = 0; r < reps; r++) {
        CK(cudaEventRecord(a, g_s));
        fn(r + warm);
        CK(cudaEventRecord(b, g_s));
        CK(cudaEventSynchronize(b));
        float ms;
        CK(cudaEventElapsedTime(&ms, a, b));
        t.push_back(ms);
    }
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    return median(t);
}

struct Grid {
    int bpsm, thr;
};
static const Grid GRIDS[] = {{1, 512}, {2, 512}, {4, 256}, {4, 512}, {8, 256}};

int main(int argc, char **argv) {
    bool quick = argc > 1 && !strcmp(argv[1], "--quick");
    int dev = 0;
    CK(cudaSetDevice(dev));
    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, dev));
    g_sms = pr.multiProcessorCount;
    int l2 = 0, clk = 0, memclk = 0, busw = 0;
    cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev);
    cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&busw, cudaDevAttrGlobalMemoryBusWidth, dev);
    printf("device %s cc %d.%d SMs %d L2 %.1f MB clock %.0f MHz memclk %d kHz bus %d bit\n", pr.name, pr.major,
           pr.minor, g_sms, l2 / 1048576.0, clk / 1e3, memclk, busw);
    char buf[512];
    snprintf(buf, sizeof buf, "\"name\":\"%s\",\"cc\":\"%d.%d\",\"sms\":%d,\"l2_mb\":%.1f,\"clock_mhz\":%.0f", pr.name,
             pr.major, pr.minor, g_sms, l2 / 1048576.0, clk / 1e3);
    emit("device", buf);
    CK(cudaStreamCreateWithFlags(&g_s, cudaStreamNonBlocking));
    CK(cudaMalloc(&g_sink, 1 << 16));

    // ---- pool: 4 layers x 288 experts x 6.29 MB (gate 2.10 + up 2.10 + down 2.10 MB, each rank's half) ------------
    const uint64_t MAT = 4096ull * 1024 / 2;              // one matrix, 4 bits: 2,097,152 B
    const uint64_t EXPERT = 3 * MAT;                        // 6.29 MB
    const int E = 288, LAYERS = 4;
    const uint64_t pool_bytes = (uint64_t)LAYERS * E * EXPERT;   // 7.25 GB
    uint4 *pool;
    CK(cudaMalloc(&pool, pool_bytes));
    CK(cudaMemset(pool, 0x5a, pool_bytes));
    // gate+up of expert e in layer l: pool + (l*E + e) * EXPERT (4.19 MB contiguous), down: + 2 MAT (2.10 MB)
    std::mt19937 rng(11);
    int *d_idx;
    CK(cudaMalloc(&d_idx, 4096 * sizeof(int)));

    // one launch reads `parts` consecutive MAT chunks (gate+up: 2 from offset 0; down: 1 from offset 2 MAT) of U
    // experts, one 2D block row per chunk (the grouped kernel's layout: a block row per distinct expert and matrix)
    auto best_gather = [&](int U, int parts, int first, bool contiguous, float *ms_out, int *bpsm_out, int *thr_out) {
        float best = 1e30f;
        int bb = 0, bt = 0;
        int reps = quick ? 5 : 15;
        int nsets = reps + 4;
        std::vector<int> flat;
        for (int s = 0; s < nsets; s++) {
            int l = rng() % LAYERS;
            std::vector<int> ex(E);
            for (int i = 0; i < E; i++) ex[i] = i;
            if (contiguous) {
                int s0 = rng() % (E - U + 1);
                for (int i = 0; i < U; i++) ex[i] = s0 + i;
            } else {
                std::shuffle(ex.begin(), ex.end(), rng);
                std::sort(ex.begin(), ex.begin() + U);
            }
            for (int i = 0; i < U; i++)
                for (int p = 0; p < parts; p++) flat.push_back((l * E + ex[i]) * 3 + first + p);   // in MAT units
        }
        int *d_sets;
        CK(cudaMalloc(&d_sets, flat.size() * sizeof(int)));
        CK(cudaMemcpy(d_sets, flat.data(), flat.size() * sizeof(int), cudaMemcpyHostToDevice));
        int rows = U * parts;
        for (const Grid &g : GRIDS) {
            int bx = std::max(1, g.bpsm * g_sms / rows);
            float ms = time_ms(
                [&](int r) {
                    read_gather<<<dim3(bx, rows), g.thr, 0, g_s>>>(pool, d_sets + (r % nsets) * rows, MAT / 16, g_sink);
                },
                reps);
            if (ms < best) best = ms, bb = g.bpsm, bt = g.thr;
        }
        CK(cudaFree(d_sets));
        *ms_out = best;
        *bpsm_out = bb;
        *thr_out = bt;
    };

    printf("\n== routed experts, one layer (a launch reads U experts' gate+up or down) ==\n");
    printf("%4s %-10s %10s %9s %9s %10s %9s %9s\n", "U", "layout", "gu MB", "gu us", "gu GB/s", "down MB", "dn us",
           "dn GB/s");
    int Us[] = {8, 12, 13, 17, 19, 21, 25, 33, 40, 51, 65};
    for (int U : Us) {
        for (int c = 0; c < 2; c++) {
            float gu, dn;
            int b1, t1, b2, t2;
            best_gather(U, 2, 0, c == 1, &gu, &b1, &t1);
            best_gather(U, 1, 2, c == 1, &dn, &b2, &t2);
            double gub = U * 2.0 * MAT, dnb = U * 1.0 * MAT;
            printf("%4d %-10s %10.1f %9.1f %9.1f %10.1f %9.1f %9.1f\n", U, c ? "contiguous" : "gathered", gub / 1e6,
                   gu * 1e3, gub / gu / 1e6, dnb / 1e6, dn * 1e3, dnb / dn / 1e6);
            snprintf(buf, sizeof buf,
                     "\"U\":%d,\"layout\":\"%s\",\"gu_bytes\":%.0f,\"gu_us\":%.2f,\"gu_gbs\":%.1f,\"gu_grid\":\"%dx%d\","
                     "\"dn_bytes\":%.0f,\"dn_us\":%.2f,\"dn_gbs\":%.1f,\"dn_grid\":\"%dx%d\"",
                     U, c ? "contiguous" : "gathered", gub, gu * 1e3, gub / gu / 1e6, b1, t1, dnb, dn * 1e3,
                     dnb / dn / 1e6, b2, t2);
            emit("expert_layer", buf);
        }
    }

    // ---- a whole round's expert reads: 42 layers x (gate/up, down), back to back, gathered ------------------------
    printf("\n== a round's routed-expert reads: 42 layers x (gate/up, down) launches back to back, gathered ==\n");
    int roundU[] = {8, 13, 17, 19, 21, 25, 40, 51};
    for (int U : roundU) {
        const int L = 42;
        std::vector<int> flat;
        int nsets = 3;
        for (int s = 0; s < nsets; s++)
            for (int l = 0; l < L; l++) {
                int layer = (s * L + l) % LAYERS;
                std::vector<int> ex(E);
                for (int i = 0; i < E; i++) ex[i] = i;
                std::shuffle(ex.begin(), ex.end(), rng);
                std::sort(ex.begin(), ex.begin() + U);
                for (int i = 0; i < U; i++) flat.push_back(layer * E + ex[i]);   // expert index; chunk = EXPERT
            }
        std::vector<int> dn_idx;
        for (int v : flat) dn_idx.push_back(v * 3 + 2);
        std::vector<int> gu2;
        for (int v : flat) {
            gu2.push_back(v * 3);
            gu2.push_back(v * 3 + 1);
        }
        int *d_gu, *d_dn;
        CK(cudaMalloc(&d_gu, gu2.size() * sizeof(int)));
        CK(cudaMalloc(&d_dn, dn_idx.size() * sizeof(int)));
        CK(cudaMemcpy(d_gu, gu2.data(), gu2.size() * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_dn, dn_idx.data(), dn_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
        float best = 1e30f;
        int bb = 0, bt = 0;
        for (const Grid &g : GRIDS) {
            int bx1 = std::max(1, g.bpsm * g_sms / (2 * U)), bx2 = std::max(1, g.bpsm * g_sms / U);
            float ms = time_ms(
                [&](int r) {
                    int s = r % nsets;
                    for (int l = 0; l < L; l++) {
                        read_gather<<<dim3(bx1, 2 * U), g.thr, 0, g_s>>>(pool, d_gu + ((s * L + l) * U) * 2, MAT / 16,
                                                                         g_sink);
                        read_gather<<<dim3(bx2, U), g.thr, 0, g_s>>>(pool, d_dn + (s * L + l) * U, MAT / 16, g_sink);
                    }
                },
                quick ? 3 : 7, 1);
            if (ms < best) best = ms, bb = g.bpsm, bt = g.thr;
        }
        double bytes = 42.0 * U * EXPERT;
        printf("U %3d: %6.2f GB in %7.2f ms = %6.1f GB/s (84 launches, grid %dx%d)\n", U, bytes / 1e9, best,
               bytes / best / 1e6, bb, bt);
        snprintf(buf, sizeof buf, "\"U\":%d,\"bytes\":%.0f,\"ms\":%.3f,\"gbs\":%.1f,\"grid\":\"%dx%d\"", U, bytes, best,
                 bytes / best / 1e6, bb, bt);
        emit("expert_round", buf);
        CK(cudaFree(d_gu));
        CK(cudaFree(d_dn));
    }

    // ---- dense q4 shapes (n x k x 0.5625 B), one launch each, from rotating places in the pool -----------------------
    printf("\n== dense q4 shapes: one read launch of the weight bytes (cold) ==\n");
    struct Sh {
        const char *name;
        int n, k, count;
    } SH[] = {{"head 77440x4096", 77440, 4096, 1},       {"kda.proj 12576x4096", 12576, 4096, 34},
              {"mlp.gu 12288x4096", 12288, 4096, 3},     {"dsa.o / mtp.eh 4096x8192", 4096, 8192, 11},
              {"mlp.down 4096x6144", 4096, 6144, 3},     {"kda.o 4096x4096", 4096, 4096, 34},
              {"dsa.q_b 8192x1536", 8192, 1536, 11},     {"shared.gu / dsa.proj 2048x4096", 2048, 4096, 53},
              {"dsa.index.qb 4096x1536", 4096, 1536, 11}, {"shared.down 4096x1024", 4096, 1024, 42},
              {"dsa.kv 8192x512", 8192, 512, 22},        {"index.kw 160x4096", 160, 4096, 11},
              {"kda.fb/gb 4096x128", 4096, 128, 68}};
    printf("%-32s %10s %9s %9s %8s\n", "shape", "MB", "us", "GB/s", "grid");
    for (auto &s : SH) {
        uint64_t bytes = (uint64_t)s.n * s.k * 9 / 16;
        bytes = (bytes + 15) / 16 * 16;
        float best = 1e30f;
        int bb = 0, bt = 0;
        uint64_t slots = pool_bytes / bytes - 1;
        for (const Grid &g : GRIDS) {
            float ms = time_ms(
                [&](int r) {
                    uint64_t off = ((uint64_t)(r * 7919 + 13) % slots) * bytes;
                    read_flat<<<g.bpsm * g_sms, g.thr, 0, g_s>>>(pool + off / 16, bytes / 16, g_sink);
                },
                quick ? 7 : 25);
            if (ms < best) best = ms, bb = g.bpsm, bt = g.thr;
        }
        printf("%-32s %10.2f %9.1f %9.1f %5dx%d\n", s.name, bytes / 1e6, best * 1e3, bytes / best / 1e6, bb, bt);
        snprintf(buf, sizeof buf, "\"shape\":\"%s\",\"n\":%d,\"k\":%d,\"count\":%d,\"bytes\":%llu,\"us\":%.2f,\"gbs\":%.1f",
                 s.name, s.n, s.k, s.count, (unsigned long long)bytes, best * 1e3, bytes / best / 1e6);
        emit("dense_shape", buf);
    }
    // the verify's dense set in layer order, back to back (one launch a matrix, best grid per launch = 4x256)
    {
        std::vector<uint64_t> seq;
        for (int l = 0; l < 45; l++) {
            bool dsa = (l % 4) == 3, dense = l < 3;
            if (dsa) {
                for (uint64_t b : {2048ull * 4096, 8192ull * 1536, 8192ull * 512, 8192ull * 512, 4096ull * 1536,
                                   160ull * 4096, 4096ull * 8192})
                    seq.push_back(b * 9 / 16);
            } else {
                for (uint64_t b : {12576ull * 4096, 4096ull * 128, 4096ull * 128, 4096ull * 4096}) seq.push_back(b * 9 / 16);
            }
            if (dense) {
                seq.push_back(12288ull * 4096 * 9 / 16);
                seq.push_back(4096ull * 6144 * 9 / 16);
            } else {
                seq.push_back(2048ull * 4096 * 9 / 16);
                seq.push_back(4096ull * 1024 * 9 / 16);
            }
        }
        seq.push_back(77440ull * 4096 * 9 / 16);
        double tot = 0;
        for (auto b : seq) tot += b;
        for (const Grid &g : GRIDS) {
            float ms = time_ms(
                [&](int r) {
                    uint64_t off = (uint64_t)(r % 3) * 2400000000ull;
                    for (auto b : seq) {
                        uint64_t bb = (b + 15) / 16 * 16;
                        read_flat<<<g.bpsm * g_sms, g.thr, 0, g_s>>>(pool + off / 16, bb / 16, g_sink);
                        off += bb;
                    }
                },
                quick ? 3 : 7, 1);
            printf("verify dense set: %zu launches, %.3f GB in %.2f ms = %.1f GB/s (grid %dx%d)\n", seq.size(), tot / 1e9,
                   ms, tot / ms / 1e6, g.bpsm, g.thr);
            snprintf(buf, sizeof buf, "\"launches\":%zu,\"bytes\":%.0f,\"ms\":%.3f,\"gbs\":%.1f,\"grid\":\"%dx%d\"",
                     seq.size(), tot, ms, tot / ms / 1e6, g.bpsm, g.thr);
            emit("dense_verify_set", buf);
        }
    }

    // ---- size sweep: read and copy, one launch, cold ---------------------------------------------------------------
    printf("\n== size sweep, one launch (cold): read GB/s, copy GB/s (read + write bytes) ==\n");
    for (double mb : {0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0}) {
        uint64_t bytes = (uint64_t)(mb * 1048576.0);
        uint64_t slots = (pool_bytes / 2) / bytes - 1;
        float br = 1e30f, bc = 1e30f;
        for (const Grid &g : GRIDS) {
            float ms = time_ms(
                [&](int r) {
                    uint64_t off = ((uint64_t)(r * 7919 + 13) % slots) * bytes;
                    read_flat<<<g.bpsm * g_sms, g.thr, 0, g_s>>>(pool + off / 16, bytes / 16, g_sink);
                },
                quick ? 7 : 25);
            br = std::min(br, ms);
            float mc = time_ms(
                [&](int r) {
                    uint64_t off = ((uint64_t)(r * 7919 + 13) % slots) * bytes;
                    copy_flat<<<g.bpsm * g_sms, g.thr, 0, g_s>>>(pool + off / 16, pool + (pool_bytes / 2 + off) / 16,
                                                                 bytes / 16);
                },
                quick ? 7 : 25);
            bc = std::min(bc, mc);
        }
        printf("%8.2f MB: read %7.1f us %6.1f GB/s | copy %7.1f us %6.1f GB/s\n", mb, br * 1e3, bytes / br / 1e6,
               bc * 1e3, 2.0 * bytes / bc / 1e6);
        snprintf(buf, sizeof buf, "\"mb\":%.2f,\"read_us\":%.2f,\"read_gbs\":%.1f,\"copy_us\":%.2f,\"copy_gbs\":%.1f", mb,
                 br * 1e3, bytes / br / 1e6, bc * 1e3, 2.0 * bytes / bc / 1e6);
        emit("size_sweep", buf);
    }

    // ---- launch overhead ------------------------------------------------------------------------------------------
    printf("\n== launch overhead ==\n");
    const int NK = 2000;
    {
        float ms = time_ms([&](int) { for (int i = 0; i < NK; i++) empty_kernel<<<1, 32, 0, g_s>>>(); }, 5, 1);
        printf("empty kernel, eager stream: %.2f us a launch\n", ms * 1e3 / NK);
        snprintf(buf, sizeof buf, "\"mode\":\"eager\",\"us_per_kernel\":%.3f", ms * 1e3 / NK);
        emit("empty", buf);
        cudaGraph_t gr;
        cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(g_s, cudaStreamCaptureModeGlobal));
        for (int i = 0; i < NK; i++) empty_kernel<<<1, 32, 0, g_s>>>();
        CK(cudaStreamEndCapture(g_s, &gr));
        CK(cudaGraphInstantiate(&ge, gr, 0));
        ms = time_ms([&](int) { CK(cudaGraphLaunch(ge, g_s)); }, 5, 1);
        printf("empty kernel, CUDA graph: %.2f us a kernel\n", ms * 1e3 / NK);
        snprintf(buf, sizeof buf, "\"mode\":\"graph\",\"us_per_kernel\":%.3f", ms * 1e3 / NK);
        emit("empty", buf);
        CK(cudaGraphExecDestroy(ge));
        CK(cudaGraphDestroy(gr));
    }
    // small dependent kernels: chain of NC kernels each reading SZ bytes; eager / graph / graph+PDL / eager+PDL
    for (uint64_t SZ : {0ull, 1ull << 20, 4ull << 20}) {
        const int NC = 400;
        Stamp *st;
        unsigned int *done;
        CK(cudaMalloc(&st, NC * sizeof(Stamp)));
        CK(cudaMalloc(&done, 4));
        std::vector<Stamp> hs(NC);
        auto reset = [&]() {
            std::vector<Stamp> init(NC, Stamp{~0ull, ~0ull, 0ull});
            CK(cudaMemcpyAsync(st, init.data(), NC * sizeof(Stamp), cudaMemcpyHostToDevice, g_s));
            CK(cudaStreamSynchronize(g_s));
        };
        int blocks = SZ ? g_sms * 2 : 1;
        auto launch_chain = [&](bool pdl) {
            for (int k = 0; k < NC; k++) {
                const uint4 *src = pool + ((uint64_t)k * (SZ ? SZ : 16) % (pool_bytes / 2)) / 16;
                if (pdl) {
                    cudaLaunchConfig_t cfg = {};
                    cfg.gridDim = dim3(blocks);
                    cfg.blockDim = dim3(256);
                    cfg.stream = g_s;
                    cudaLaunchAttribute at[1];
                    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
                    at[0].val.programmaticStreamSerializationAllowed = 1;
                    cfg.attrs = at;
                    cfg.numAttrs = 1;
                    CK(cudaLaunchKernelEx(&cfg, small_kernel, src, (uint64_t)(SZ / 16), g_sink, st, k, done, 1));
                } else {
                    small_kernel<<<blocks, 256, 0, g_s>>>(src, SZ / 16, g_sink, st, k, done, 0);
                }
            }
        };
        for (int mode = 0; mode < 4; mode++) {       // 0 eager, 1 eager+PDL, 2 graph, 3 graph+PDL
            bool pdl = mode & 1, graph = mode >= 2;
            cudaGraphExec_t ge = nullptr;
            cudaGraph_t gr;
            if (graph) {
                CK(cudaStreamBeginCapture(g_s, cudaStreamCaptureModeGlobal));
                launch_chain(pdl);
                CK(cudaStreamEndCapture(g_s, &gr));
                CK(cudaGraphInstantiate(&ge, gr, 0));
            }
            float ms = 0;
            std::vector<float> ts;
            double gap = 0, early = 0, busy = 0;
            int ng = 0, nearly = 0;
            for (int rep = 0; rep < 5; rep++) {
                reset();
                cudaEvent_t a, b;
                CK(cudaEventCreate(&a));
                CK(cudaEventCreate(&b));
                CK(cudaEventRecord(a, g_s));
                if (graph) CK(cudaGraphLaunch(ge, g_s)); else launch_chain(pdl);
                CK(cudaEventRecord(b, g_s));
                CK(cudaEventSynchronize(b));
                CK(cudaEventElapsedTime(&ms, a, b));
                ts.push_back(ms);
                CK(cudaMemcpy(hs.data(), st, NC * sizeof(Stamp), cudaMemcpyDeviceToHost));
                if (rep == 4) {
                    for (int k = 1; k < NC; k++) {
                        gap += (double)hs[k].go - (double)hs[k - 1].end;        // prev last block -> this one's work
                        early += (double)hs[k - 1].end - (double)hs[k].start;   // > 0: started before prev ended
                        nearly += hs[k].start < hs[k - 1].end;
                        ng++;
                    }
                    for (int k = 0; k < NC; k++) busy += (double)hs[k].end - (double)hs[k].go;
                }
                CK(cudaEventDestroy(a));
                CK(cudaEventDestroy(b));
            }
            float med = median(ts);
            const char *mn[] = {"eager", "eager+PDL", "graph", "graph+PDL"};
            printf("chain %4d x %5.1f MB %-10s: %7.2f us a kernel, kernel busy %7.2f us, gap last-block -> next "
                   "work %6.2f us, early starts %d/%d (mean %.2f us)\n",
                   NC, SZ / 1048576.0, mn[mode], med * 1e3 / NC, busy / NC / 1e3, gap / ng / 1e3, nearly, ng,
                   early / ng / 1e3);
            snprintf(buf, sizeof buf,
                     "\"mb\":%.1f,\"mode\":\"%s\",\"us_per_kernel\":%.3f,\"busy_us\":%.3f,\"gap_us\":%.3f,"
                     "\"early_starts\":%d,\"n\":%d,\"early_us\":%.3f",
                     SZ / 1048576.0, mn[mode], med * 1e3 / NC, busy / NC / 1e3, gap / ng / 1e3, nearly, ng,
                     early / ng / 1e3);
            emit("chain", buf);
            if (graph) {
                CK(cudaGraphExecDestroy(ge));
                CK(cudaGraphDestroy(gr));
            }
        }
        CK(cudaFree(st));
        CK(cudaFree(done));
    }
    CK(cudaDeviceSynchronize());
    printf("\nDONE\n");
    return 0;
}
