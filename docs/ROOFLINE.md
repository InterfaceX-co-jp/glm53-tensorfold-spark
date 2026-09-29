# Roofline: where the remaining time is, measured against what GB10 can do (2026-09-28)

Offline analysis. No GPU was used and production was not touched. Inputs:

- W7's nsys traces, both ranks. `pf24-r{0,1}` is the 21,464-token "24.5k" prefill; `mix-r{0,1}` is the 85,781-token "98k"
  prefill plus 1-stream and 4-stream decode. They were copied from the head node and exported to sqlite locally with nsys
  2026.3.2.
- The model config and the safetensors headers of the production snapshot.
- Our docs: `PROFILE.md`, `RESULTS.md` (W7, W8), `EXPERT-TC.md`, `PATCHES.md`, `DECODE-ANALYSIS.md`, `EXPERIMENTS.md`.

Scripts and their outputs are in `results/roofline/`:

- `model.py`: per-stage FLOPs and bytes, and the prefill table.
- `kern.py`: per-kernel time by probe component.
- `experts_union.py`: distinct experts per decode window.
- `dec_bytes.py`: expert bytes per decode round.
- `*.txt`: what each script printed.

Every number below is **per rank (= per node)**. TP=2 splits every large matrix in two, so the ranks do the same work;
the W7 ranks agree within 0.5 point.

## 0. Roofs and the model

**GB10, per node**

| roof | value | source |
| --- | ---: | --- |
| DRAM (LPDDR5x, unified) | **230 GB/s** attainable (273 nominal); **W11 measured: 235 GB/s streaming read** (233-238 for one decode round's expert volumes, gathered or contiguous; 230.5 for the verify's dense q4 set in 304 launches; copy 216-220; a single launch needs ~8 MB for 219, ~16 MB for 229) | the best kernels here run at 207-233 GB/s: the head `_qmm` at 207, `grouped_kernel` at 200, `_combine_s` at 202; OPS-GPUWATCH: 224-233; `results/W11/probe.cu` |
| bf16 / f16 `mma.sync` m16n8k16, fp32 accumulate | **110 TFLOP/s** | measured (EXPERT-TC.md, PATCHES 0080); spec 48 SMs x 1,024 FLOP/clk x ~2.5 GHz = 123; cuBLAS-class health check 94.8 (OPS-GPUWATCH) |
| e4m3 / int8 `mma.sync` m16n8k32 | ~220 TFLOP/s (TOPS), 2x bf16 | spec ratio ("1 PFLOP FP4 sparse" = 500 FP4 dense = 250 FP8 = 125 BF16). Repo: "about twice the bf16 rate" (0083). Needs both operands in 8 bits, so FP8 activations, which were rejected for quality. **Not available to this quant policy.** |
| FP4 block-scaled mma (`kind::mxf4`) | ~440 dense | sm_121a only, not sm_121 (EXPERT-TC.md section 2); new bits |
| FP32 FMA (CUDA cores) | 30.7 TFLOP/s | 48 x 128 x 2 x 2.5 GHz |
| no `wgmma`, no `tcgen05`, no `setmaxnreg` | | EXPERT-TC.md: `mma.sync` is the only tensor-core path |
| CX7 RoCE between the Sparks | 25 GB/s line, 13.6 GB/s per QP; NCCL `RING_LL` 9-10.5 GB/s at 4-8 MiB; 24-29 us median for a 16-128 KiB decode all-gather (graph); 0230's GPU-initiated path 11-27 us at 16-128 KiB (0350 harness) | PROFILE.md, EXPERIMENTS.md, RESULTS W7 |

**Model: GLM-5.3-Flash, 45 layers + 1 MTP layer.**

- **Attention:** 34 KDA layers (64 heads x 128) and 11 DSA/MLA layers (64 heads, q_lora 1,536, kv_lora 512,
  qk = v = 256, indexer 32 x 128 with top-2,048 over kpool-4 keys).
- **MLP:** layers 0-2 are dense (12,288 wide). The other 42 are MoE: 288 experts x 2,048, top-8 + 1 shared.
- **Residual:** hyper-connections, 4 bf16 streams, 20 Sinkhorn steps.
- **Head:** vocab 154,880.

Bytes per rank:

| weights a rank | params | bytes |
| --- | ---: | ---: |
| Routed experts, EXL3 4.0 bpw. One expert's half is 3 x 4,096 x 1,024 = **6.29 MB**. | 42 x 288 x 12.6 M | **76.1 GB** (1.81 GB a layer) |
| q4mse non-expert matrices, 4.5 bpw. KDA 1.326, DSA 0.436, shared 0.297, dense MLP 0.127, head 0.178 (vocab split). | 4.20 G | **2.365 GB** |
| router + hc weights, bf16, replicated | 0.12 G | 0.170 GB |
| One MTP draft step: its DSA layer, shared expert, eh_proj, 8 experts, head | | 0.297 GB |
| KDA recurrent state (fp32, per slot) | | 71 MB |

**Prefill FLOPs per token per rank** (`model.py`) at the 24.5k prompt (mean context 10.7k):

| part | GFLOP a token |
| --- | ---: |
| routed experts (8 picks x 3 matrices, 42 layers) | 8.46 |
| KDA projections | 4.72 |
| shared expert + dense MLP | 1.51 |
| sparse attention (32 heads x 2,048 keys x 512 latent, 11 layers) | 1.48 |
| MLA expand + o_proj | 0.83 |
| DSA q/kv projections + absorb | 0.72 |
| indexer (at 98k context: 0.97) | 0.24 |
| KDA recurrence | 0.20 |
| router | 0.10 |
| **total** | **18.4** (19.1 at 98k) |

On bf16 `mma.sync` that is **167 us a token = 5,980 tok/s**, the absolute compute ceiling of this quant policy. The
minimum DRAM traffic of today's dataflow (including the fp32 expert outputs, the hc streams and weights read once per
chunk / sub-block) is 44 MB a token, or **193 us**.

Link: the prefill exchanges move ~720 KB a token a rank (90 exchanges x 8 KB after 0320). That is 70-80 us at NCCL's
9-10.5 GB/s against 680 us of compute, so it is **not a binding roof**; its cost is the overlap tax (section 1).

## 1. Prefill (production: 8,192-row solo chunks, 512-row lean sub-blocks, 0320 row split; 1,468-1,473 tok/s at 24.5k)

**Where the "measured" column comes from.** The only traces are W7's, which ran 2,048-row chunks without 0320. The
production column (`now`) takes W7's rank-0 partition per token and applies three corrections (`model.py`, `ADJ`):

1. **Capture distortion:** x 0.982. W7 traced 1,239 tok/s; the same configuration without nsys (W8 load A) ran
   1,261.
2. **8,192-row chunks** (W8 D vs A: 1,261 -> 1,342.5 tok/s, -48.1 us a token):
   - -3.7 us is fewer piece boundaries (~10 ms each);
   - the remaining -44.4 us goes to the routed experts, the only stage whose cost depends on chunk size.
3. **0320 row split** (W8 E vs D: 1,342.5 -> 1,470.9, -65.0 us):
   - -57 us is hyper-connections: half the `_hc_post` / `_hc_pre` work, plus most of the all-gather overlap tax, which
     sat on `_hc_post`;
   - -8 us is the shared expert, whose `_fq4` also ran beside the all-gathers.

The corrected total reproduces the measured 1,471 tok/s. The split of the two W8 gains across stages is an
attribution, not a measurement. **One nsys capture of today's configuration would replace it** (method: PROFILE.md).

"roof" = max(FLOPs / roof rate, minimum bytes / 230 GB/s) for the stage. The roof rate is 110 TFLOP/s for tensor-core
work, or 30.7 for work that is inherently FP32 FMA (KDA recurrence, router, hc). "gap" = now - roof.

### 1.1 Ranked by gap

Columns: W7 and now are us a token; ms / 8k is ms per 8,192-row chunk; TF/s and GB/s are achieved (now).

| rank | stage | W7 | now | ms / 8k | GFLOP | MB | TF/s | GB/s | roof us | bound | % of roof | gap us | gap ms / 8k |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 1 | Routed experts (fat + rot_in + plan) | 216.0 | 167.8 | 1,375 | 8.46 | 16.9 | 50.4 | 100 | 76.9 | MMA (DRAM 73.3) | 46 | **91.0** | 745 |
| 2 | DSA sparse + dense latent attention | 97.6 | 95.8 | 785 | 1.48 | 0.8 | 15.4 | 9 | 13.4 | MMA | 14 | **82.4** | 675 |
| 3 | MLA latent expand + o_proj | 69.9 | 68.6 | 562 | 0.83 | 1.5 | 12.1 | 22 | 7.5 | MMA | 11 | **61.1** | 500 |
| 4 | KDA projections (q4 GEMMs) | 93.1 | 91.4 | 749 | 4.72 | 4.6 | 51.6 | 50 | 42.9 | MMA | 47 | **48.6** | 398 |
| 5 | KDA chunked recurrence | 64.7 | 63.5 | 520 | 0.20 | 3.6 | 3.2 | 57 | 15.7 | DRAM | 25 | **47.8** | 391 |
| 6 | Shared expert + dense MLP (q4 GEMMs) | 46.4 | 37.6 | 308 | 1.51 | 3.0 | 40.1 | 81 | 13.7 | MMA | 37 | 23.9 | 196 |
| 7 | Hyper-connections | 104.6 | 45.8 | 375 | 0.04 | 5.5 | - | 121 | 24.0 | DRAM | 52 | 21.7 | 178 |
| 8 | DSA q/kv projections + absorb | 28.1 | 27.6 | 226 | 0.72 | 1.0 | 26.0 | 37 | 6.5 | MMA | 24 | 21.1 | 173 |
| 9 | NCCL exposed + memcpy + host gaps + other | 21.7 | 17.7 | 145 | - | - | - | - | 0 | - | 0 | 17.7 | 145 |
| 10 | DSA indexer + top-k (98k: 31.6 us in W7) | 13.2 | 13.0 | 106 | 0.24 | 0.3 | 18.6 | 19 | 2.2 | MMA | 17 | 10.8 | 88 |
| 11 | MoE router / grouping + combine | 40.8 | 40.1 | 329 | 0.10 | 6.9 | - | 173 | 30.1 | DRAM | 75 | 10.0 | 82 |
| 12 | MTP cache rows + DFlash2 taps | 10.9 | 10.8 | 88 | 0.10 | 0.3 | 9.4 | 28 | 1.3 | - | 12 | 9.4 | 77 |
| | **total** | **807.1** | **679.8** | **5,569** | **18.4** | **44.4** | 27.0 | 65 | **234.4** | | **34** | **445.4** | **3,649** |

**Ceiling.**

- **Sum of the per-stage roofs: 234 us a token = ~4,270 tok/s (2.9x today)** with today's dataflow (fp32 expert
  outputs, bf16 hc streams, 512-row sub-blocks).
- **Pure MMA bound: 167 us = ~5,980 tok/s** (4.1x), if every byte were hidden under the tensor cores.
- **Realistic engineering target: ~3,000 tok/s (~335 us a token)**, with every stage at ~70% of its roof.

Today's prefill runs at **34% of the sum of roofs**: 445 us of the 680 us a token is gap.

### 1.2 What the traces say per stage (W7 kernels, rank 0; rank 1 within 1%)

Times are per call on a 512-row lean sub-block unless noted.

1. **Routed experts** (`expert_kernel`: fat, grid 96 = 2 CTAs an SM, ticket scheduling).
   - **At 2,048 rows: DRAM-bound.** gate/up 4,952 us + down 3,645 us per layer and 2,048-row chunk is 8.6 ms for 1.81 GB
     of trellis, i.e. **213 GB/s on the weights alone**. `rot_in1_kernel` adds 730 us.
   - **At 8,192 rows the weights shrink to 1.13 us a token**, and the kernel meets two roofs of the same size:
     - MMA: 76.9 us a token at 110 TFLOP/s;
     - DRAM: 73.3 us a token. The weights are 40 of it. The rest is activations: Xg, Xd, and above all **Y, 8 x 4,096
       fp32 = 128 KB a token and layer, written here and read back by `_combine_s`**.
   - Today's 168 us is roughly the **sum** of the two roofs (150), not the max. MMA and data movement do not overlap.
     W2 saw the same: no-mma saves 21-27% at 8,192 rows.
2. **Sparse latent attention** (`_lsparse_chunks`, grid 2,560: **3,693 us** at 10.7k context, **3,677 us** at 43k).
   - Plus `_lsparse_merge` at 883 us.
   - That is 134 MFLOP a row and layer at **15-19 TFLOP/s (14-17% of MMA)**.
   - **The time does not move with context** (10k vs 98k prompt: same per-call time), although the latent KV no longer
     fits L2 at 98k. So the kernel is bound by its own issue and latency (unpipelined per-tile row gathers, small online
     softmax tiles, FP8 -> bf16 conversion), not by DRAM or gather volume.
3. **MLA expand / absorb run on the FMA pipe** (`_expand`, grid 4,096: **2,415 us**; `_absorb`, grid 8,192: 676 us).
   - Each is 4.3 GFLOP per sub-block.
   - `_expand` runs at **1.8 TFLOP/s: 6% of the FP32 FMA peak, 1.6% of the tensor roof**. `_absorb` runs at 6.4.
   - Cause (0060's Triton kernels): `tl.dot(..., input_precision="ieee")` on 16-row tiles that dequantize the whole q4
     kv_b tile in fp32 again for every 16 rows.
   - `_expand` alone is 11 layers x 16 sub-blocks x 2.4 ms = **425 ms of every 8,192-row chunk (7.6% of prefill)**.
   - The tensor-core versions exist (`absorb_tc` / `expand_tc`, `GLM53_TF_LATENT_TC=1`). They round u / q and kv_b to
     bf16, and **they are off (they change replies)**.
   - The o_proj half of the stage is `_fq4` grid 256 at 670 us = 51 TFLOP/s.
4. **q4 GEMM family** (`_fq4`).
   - Rates:
     - KDA q/k/v/f/g/b (N = 12,576): grid 788, 938 us = **56 TFLOP/s**;
     - KDA o_proj: grid 256, 338 us = 51;
     - shared gate/up: grid 128, 199 us = 43; shared down: grid 256, 102 us = 42;
     - dense MLP gate/up: grid 768, 896 us = 57;
     - DSA q/kv: grids 512 / 128 / 12.
   - That is 38-52% of the MMA roof. It is compute-bound with 128-row M tiles: weights are re-read per 512-row
     sub-block, which is only 2.6 MB a token, so bytes are not the issue.
5. **KDA recurrence** (a 512-row sub-block):
   - `_kda_prep`: grid 256, 568 us;
   - `_kda_state`: **grid 64** on 48 SMs, 334 us;
   - `_kda_norm`: 41 us.
   - That is 0.2 GFLOP of FP32-class math and ~3.6 MB a token (bf16 q/k/v/g, **fp32 W/U intermediates written by
     prep and read by state**). The run is 4x its DRAM roof: one and a third waves of CTAs, a sequential chunk scan and
     an fp32 round trip between the kernels.
6. **Hyper-connections**:
   - `_hc_post` (grid 1,536) took 139 us alone and 344 us median in W7, because it sat beside the 4 MiB all-gathers.
   - 0320 halved the work and removed most of that tax, so hc is now ~46 us a token.
   - It is DRAM-bound streaming: 4 bf16 streams read and written, plus `_hc_pre` re-reading them. hc fusion
     (`hc_fused` bit 1) is still off.
7. **Router + combine**: `_combine_s` (374 us per sub-block) reads the fp32 Y at **202 GB/s**. It is at its roof; only
   fewer bytes help (see gap 3). `_router_part` does 2,048-row fp32 dots at 248 us.
8. **Communication**: 90 exchanges a sub-block. W7 showed 0.8% exposed plus an 8.2-8.7% overlap tax; 0320 took most of
   the tax away. Exposed NCCL + memcpy + host gaps are ~18 us a token (2.6%).

## 2. Decode

> **Today's round is measured in W11** (`docs/RESULTS.md` W11 §2, `docs/DECODE-PLAN.md` §1): b4 + RoCE + 0370 / 0380 /
> 0390, 1 stream prose **53.3 ms uncaptured (54.6 captured)** = experts 27.8 (205 GB/s, U 20) + dense 16.1 (**175 GB/s**)
> + exchanges 2.3 exposed (100 RoCE all-gathers, 14.5 us median) + attention 2.1 + hc / router 2.6 + KDA 1.6 + other 0.3
> + idle 1.8, no gap between rounds; floor 37.2 ms (68%). 4 streams **121.3 ms** (experts 75.5, dense 21.7, exchanges
> 4.2, KDA 6.0, attention 4.8, idle 5.1; floor 89.0). Sections 2.1-2.3 below are W7's (pre-RoCE) trace.

### 2.1 Bytes a round, from first principles

`experts_union.py`:

- `grouped_kernel` (the row-invariant EXL3 decode kernel) launches with **grid.x = 8R + 1**, so every call names its
  window's rows R. It reads each **distinct** expert once: programs past `ucount` exit.
- The R = 1 calls are MTP draft steps with exactly 8 experts. They run at **200 GB/s** (gate/up 167.5 us, down 83.8 us).
  That calibrates the distinct-expert count U(R) = 8 t(R) / t(1).
- The gate/up and down estimates agree to within 1-3%, and both ranks give the same U.

| rows R | 1 | 2 | 3 | 4 | 5 | 8* | 10* | 12* | 16* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| distinct experts U (measured) | 8 | 12.6 | 17.0 | 21.5 | 24.4 | 40.5 | 48.2 | 54.0 | 64.8 |
| if routing were independent, 288 (1 - (280/288)^R) | 8 | 15.8 | 23.3 | 30.7 | 37.8 | 58.1 | 70.7 | 82.6 | 104.5 |
| expert MB a layer and rank (U x 6.29) | 50 | 79 | 107 | 135 | 153 | 255 | 303 | 340 | 408 |

\* Rows of 2-4 different sequences (4-stream rounds). A single stream's 8-row window reads ~33-36 experts (the few
1-stream 57- and 65-grid calls).

Consecutive tokens of one sequence share ~half their experts; rows of different sequences share almost nothing.

A verify forward of R rows reads, per rank:

- 2.365 GB of q4 weights;
- 0.17 GB of router + hc weights;
- **U(R) x 0.264 GB of experts** (42 layers);
- the KDA state (71 MB read + written);
- selected KV (R x 11 x 2,048 x 528 B: 12 MB a row).

| verify window | bytes | floor at 230 GB/s |
| --- | ---: | ---: |
| R = 1 (serial decode) | 4.80 GB | 20.9 ms = **48 tok/s** |
| R = 3 (U 17) | 7.20 GB | 31.3 ms |
| R = 8 (U 33-40) | 11.5-13.3 GB | 50-58 ms |

Best case, 8 rows with all 7 drafts accepted: **6.3-7.2 ms a token (140-160 tok/s)** is the hardware floor for one
stream at this quant.

### 2.2 One stream (W7 trace: 106 rounds, 59.47 ms under capture, 54-60 without, 2.4 tokens a round)

**Correction to the premise.** The 54-60 ms rounds carry **2.4** tokens: plain prose, verify windows of 3.7 rows mean,
3 median. A **5.7-token** round is the counting canary: 8-row windows at 76.3 tok/s, which is **~75 ms a round**. Both
cases are below.

`dec_bytes.py`: **809.5 expert-layer reads a round = 5.09 GB** (verify ~19 experts a layer + MTP steps).

| stage (kernel families, dec.py) | measured ms | bytes GB | floor ms | achieved GB/s | % of floor | gap ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Routed experts (`grouped_kernel` 25.4 + epilogues / rot_in / plan) | 26.43 | 5.09 | 22.1 | 193 | 84 | 4.3 |
| Dense q4 GEMMs (`_qmm`: KDA qkv grid 1x197x4 at 150 us = 193 GB/s, head 1x1210 at 859 us = 207 GB/s, shared, o_proj, drafter) | 16.29 | ~3.1 (verify 2.37 + 1.8 extra heads + MTP layer + DFlash2) | 13.5 | 190 | 83 | 2.8 |
| NCCL: 100 all-gathers of 16-128 KiB, ~48 us each incl. peer wait, **none hidden** | 4.77 | 0.003 | 0.5 (5 us GPU-initiated) | - | 10 | 4.3 |
| GPU idle inside the round (hard syncs: sampling `.cpu()`, numpy, staging) | 4.65 | - | 0 | - | 0 | 4.65 |
| DSA attention + indexer (`_lsparse*`, `_expand` 109 us a call = 43 GB/s, `_absorb`) | 2.60 | 0.04 | 0.2 | - | 8 | 2.4 |
| hc + router / grouping / combine | 2.76 | 0.17 | 0.74 | - | 27 | 2.0 |
| KDA (`chain_kernel`, conv, `replay_layers_kernel`) | 1.60 | 0.15 | 0.65 | - | 41 | 0.95 |
| norms / elementwise / memcpy | 0.38 | - | 0.1 | - | - | 0.3 |
| **round** | **59.47** | **~8.6** | **37.8** | 145 | **64** | **21.7** |

Plus ~3.9 ms of host work between rounds (plan share, HTTP emit), outside the round wall.

By phase: verify forward 49.2 ms, drafting 4.8, sampling 0.3, round other 0.4.

Per token:

| case | floor | measured |
| --- | --- | --- |
| prose, 2.4 tokens a round | 15.8 ms (63 tok/s) | 24.8 ms under capture; 23-25 ms / 40-43 tok/s without |
| canary, 5.7 tokens a round | 53-61 ms a round (R = 8, U 33-40, + DFlash2 block ~2.5-3 ms) = 9.3-10.7 ms (93-107 tok/s) | ~75 ms = 13.1 ms (76 tok/s): **72-82% of floor** |

**Where the ~57 ms goes vs the floor** (1 stream, prose): 38 ms is bytes that must move. Of the 22 ms of gap:

- **9.4 ms is synchronisation**: NCCL latency + GPU idle. Add the ~3.9 ms between rounds and it is 13.3 ms: **a
  fifth of wall time**.
- 7.1 ms is the two big bandwidth kernels running at 190-193 instead of 230 GB/s.
- 5.3 ms is small latency-bound kernels: attention, hc, router, KDA.

### 2.3 Four streams (W7 trace: 145 rounds with 4 in flight, 125.5 ms under capture, 113-122 without)

Verify windows of 10.8 rows mean; **2,196 expert-layer reads = 13.81 GB a round, ~51 experts a layer**. Aggregate 71-73
tok/s, ~9 tokens a round.

| stage | measured ms | floor ms | gap ms |
| --- | ---: | ---: | ---: |
| Routed experts | 72.31 | 60.0 (13.81 GB) | 12.3 |
| Dense q4 GEMMs (weights shared by the 4 rows' verify; drafting batched) | 22.98 | ~14.6 (2.37 + ~1.0 GB) | 8.4 |
| NCCL (120 all-gathers, exposed) | 7.67 | 0.6 | 7.1 |
| GPU idle | 7.29 | 0 | 7.3 |
| KDA (4 slots' state) | 5.53 | 2.5 | 3.0 |
| DSA attention + indexer | 5.21 | ~1.0 | 4.2 |
| hc + router / combine | 3.34 | 0.74 | 2.6 |
| other + memcpy | 1.18 | 0.1 | 1.1 |
| **round** | **125.5** | **~79.5** | **~46** |

Floor per aggregate token: 8.8 ms, i.e. **~113 tok/s against 71-73 measured (63%)**.

## 3. The top 5 gaps and how to close them

Constraints:

- Replies must stay byte-identical for drafted == serial, resumed == fresh and batched == alone.
- FP8 activations are rejected.
- Weights stay EXL3 4-bit.
- `GLM53_TF_LATENT_TC` (bf16 absorb/expand) is off (it changes replies).

**On bits.** "Same bits" below means the new kernel does every output's arithmetic in the same instruction sequence,
so it is bit-identical to today's. That is checkable with the existing GPU tests (`test_glue`, `test_fast_*`). "New
fast-prefill bits" is a different but deterministic, row- and C-independent arithmetic. It keeps all three exactness
properties, because every fast chunk uses it, and it needs a new snapshot tag (as `LATENT_TC` does) plus the quality
gate (MMLU-200 / needle / exact / batchexact).

| # | gap | today -> target | engineering idea | est. gain | effort | exactness risk |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | **Routed experts at 8,192 rows**: 91 us a token (13% of prefill) | 168 -> 110-120 us | **(a) Same bits:** overlap MMA with the loads. The kernel today costs the sum of its two roofs: 77 us MMA and 73 us DRAM. Use a producer warp with a `cp.async` / `cp.async.bulk` + mbarrier ring, as 0330, but in a shape GB10 can launch: at most 16 warps or 96 registers a CTA (cfg 1/2 need 19,200 registers on a 16,384-register sub-partition), 128-member passes so each trellis tile is decoded half as often, and keep fat's ticket (W5 showed static striding loses under the overlap stream). **(b) New bits:** store Y in bf16. It is 128 KB a token and layer written, and read again by `_combine_s`: -5.4 MB a token, -23 us of DRAM on each side. Fast prefill's partials already cross the link in bf16. | (a) -35..-50 us; (b) a further -15..-25 us (combine too). **+6-10% prefill** | (a) 1-2 weeks (0330's scaffolding, tests and bench exist); (b) 2-3 days | (a) low: the per-element mma chain is fat's; the 0330 tests already check this. (b) medium: new fast-prefill bits plus the quality gate |
| 2 | **Sparse latent attention**: 82 us a token (12%). 15-19 TF/s, context-independent | 96 -> 30-40 us | Rewrite `_lsparse_chunks` + `_lsparse_merge` as one pipelined kernel: multi-stage `cp.async` gathers of the selected 528-B FP8 rows (today not pipelined), FP8 -> bf16 in shared memory, 64-key tiles, m16n8k16 for QK and PV, the split merge fused. Start from the 32-query (one row's 32 heads) tile, which keeps row independence by construction. A second step shares the gathered keys between the rows of an absolute 64-row group. Adjacent rows' top-2,048 sets overlap heavily. The group must be absolute, not chunk-relative, to stay C-independent. Also re-A/B 0360 (b12x bit 4, 1.47x kernel, now fit for FP8 KV + sessions) at 8,192-row chunks as the quick check. | -55..-65 us: **+9-10%** (more at long context, where the indexer also grows) | quick A/B: 1 window. Own kernel: 1-2 weeks | Per-row tile with today's tile order: can be same bits. Group-shared keys: new fast-prefill bits (online-softmax order changes), still deterministic and row / C-independent. Medium |
| 3 | **dense q4 GEMM family (`_fq4`)**: KDA proj 48.6 + shared / dense 23.9 + DSA / o_proj ~10 = ~82 us (12%) | 38-57 -> 80-90 TFLOP/s | One better q4 x bf16 kernel for all `matmul_fast` calls, compute-bound at 512-row sub-blocks: 256 x 128 tiles, 3-4-stage `cp.async` pipeline, q4 weights pre-shuffled at load into the m16n8k16 fragment layout (no shuffles, 128-bit loads), a persistent grid sized to 48 SMs (the 12,576-column KDA projection is 788 tiles = 16.4 waves today). **Keep `matmul_fast`'s contract:** one fp32 accumulator per output walking K in group order, no split-K, tiles never by M. Also: run the lean sub-block's KDA and shared-expert GEMMs at 1,024 rows, since they are compute-bound and fewer tails help. | -40..-50 us: **+6-8%** | 1-2 weeks | **Low: same bits** if the per-output k16 mma chain and the group order are kept (tiling in M/N does not change bits; the existing `_fq4` bitwise tests decide) |
| 4 | **MLA expand / absorb on the FMA pipe**: ~75 us (11%): `_expand` 2,415 us + `_absorb` 676 us a sub-block at 1.8 / 6.4 TFLOP/s | 75 -> ~15 us | Keep the fp32 IEEE arithmetic kept over `LATENT_TC`, but write a proper SIMT kernel: 64-128-row tiles; the kv_b tile dequantized once per CTA into shared memory and reused by every row (today: again every 16 rows); register-blocked 8x8 fp32 FMA micro-tiles; the same per-output FMA order (64-wide groups ascending, k ascending within a group). FMA roof 140 us a call; 50% of it is 280 us vs 2,415. Fallback with the same precision class: 3-term bf16 split (hi*hi + hi*lo + lo*hi) on tensor cores, ~fp32 accuracy, 3x the MMA FLOPs, still ~20x faster than today. | -55..-60 us: **+8-9%** (1,471 -> ~1,600 tok/s) | 2-4 days | Low if the FMA order is matched (bitwise test vs `_expand` / `_absorb`; Triton's `ieee` dot is a k-ordered FMA chain, but whether it fuses `acc + dot` must be checked on the GPU). Otherwise new fast-prefill bits at fp32 precision: quality-neutral by construction, needs the gate |
| 5 | **Decode synchronisation**: NCCL latency 4.8 / 7.7 ms + GPU idle 4.7 / 7.3 ms + ~3.9 ms between rounds = **13-19 ms a round (22-25% of 1-stream wall)** | 1 stream: 57 -> ~47 ms a prose round | **(a)** 0230 GPU-initiated RoCE for the ≤256 KiB decode exchanges. The 0350 harness moves 16 / 64 / 128 KiB bit-exact in 11.3 / 17.7 / 26.9 us inside a graph, vs ~48 us a NCCL all-gather incl. peer wait. Keep `ROCE_MAX_KB=256`: 1 MiB is not bit-exact yet. **(b)** Take the host off the critical path: device-side top-k / acceptance for greedy and keyed-sampled rows (a GPU port of the splitmix64 / float64 Gumbel rule), next round's staging and plan share while the GPU verifies, the HTTP emit on another thread. | (a) -2.5..-3.5 ms a round; (b) -4..-8 ms. **1 stream +12-19%** (prose ~42 -> ~47-50 tok/s; canary 76 -> ~84-89), **4 streams +8-12%** | (a) 1 window: code and harness exist; (b) 1-2 weeks | (a) low-medium: the same bytes move, only ≤256 KiB is verified, and 0230's reliability history argues for a soak. (b) medium: the device sampler must reproduce the host's keyed draw bit for bit (test against the host rule over millions of draws); the overlap-only part is low |

**Next after these:**

- **KDA recurrence** (48 us, 7%): split `_kda_state`'s value columns into 32-wide blocks. The delta rule is
  independent per value column, so this is the same arithmetic per element, same bits, and gives 128-256 CTAs instead
  of 64. Also fuse prep -> state -> norm to drop the fp32 W/U round trip. -25..-35 us, +4-5%.
- **hc fusion** (`hc_fused` bit 1, now launchable with `num_stages=1`): ~-8 us a token.
- **Non-expert weights q4mse (4.5 bpw affine) -> EXL3 4.0 bpw** (the same codec family as the experts): -0.26 GB a
  verify, -1.1 ms a round (+2% decode), -0.5 us a token prefill. It needs the quality check; trellis + Hadamard has
  lower MSE than affine g64.
- **Decode, 4 streams**: the expert kernel runs at 200 of 230 GB/s (12 ms a round).
- **Beyond engineering**: tokens per round. Bytes a committed token fall from 3.0 GB (R = 3, 2.4 tokens) to 1.4-1.7 GB
  (R = 8, 8 tokens). Drafter acceptance is the decode lever the roofline cannot remove.

**Stacked.**

- Prefill items 1-4 are independent stages: -185..-250 us of 680, i.e. **~2,000-2,300 tok/s at 24.5k (+37-58%)**,
  still under the ~3,000 tok/s engineering ceiling. Add the KDA recurrence and it is ~2,100-2,500.
- Decode item 5 plus the 4-stream expert efficiency: +12-19% single stream, +10-15% aggregate.

**Recommended order, by gain per effort and risk:**

1. Gap 4 (days, likely same bits).
2. Gap 3 (same bits).
3. Gap 5a (one window).
4. The b12x re-A/B of gap 2.
5. Gap 1a.
6. Gap 2's own kernel and 5b.
7. Anything with new bits (1b, the group-shared attention) last, behind the quality gate.

**Before any of it:** one nsys capture of today's production config (8,192-row solo chunks + 0320) turns section 1's
`now` column from an attribution into a measurement.

## Reproduce

```sh
# traces: the head node ~/w7-traces/*.nsys-rep -> local; nsys 2026.3.2 CLI (x86_64 deb, extracted, no install)
nsys export --type sqlite --output pf24-r0.sqlite pf24-r0.nsys-rep            # also pf24-r1, mix-r0, mix-r1
python3 results/roofline/kern.py pf24-r0.sqlite prefill out.json              # per-kernel prefill table
python3 results/roofline/experts_union.py mix-r0.sqlite                       # U(R) from grouped_kernel grids
python3 results/roofline/dec_bytes.py mix-r0.sqlite dec1                      # expert bytes a round (dec1 / dec4)
python3 results/roofline/model.py                                             # FLOPs / bytes / roofs / gap table
```
