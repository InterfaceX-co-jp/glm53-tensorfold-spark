# Where the time goes: prefill and decode on the production load (W7, 2026-09-28)

Measured in the W7 window (16:53-17:13, prod down 20 min) on the production configuration (`config/prod.env`: image
`glm53-tensorfold:kvpool`, TP=2 over the two Sparks, 4 batch slots, 2,048-token pieces / 2,048-row fast chunks in
512-row lean sub-blocks, pipelined overlap on, fat experts, FP8 latent KV, shared 1M KV pool). Raw data, scripts and
the analysis JSONs: `results/W7/` (method below; results summary in `docs/RESULTS.md`, "W7: profile"). Nothing was
adopted.

## Method

- **nsys on both ranks, inside the containers.** The image's nsys 2026.3 (`/usr/local/bin/nsys`) ran the server under
  `nsys launch --trace=cuda-sw,nvtx --cuda-graph-trace=node` (`results/W7/bin/tensorfold` shim, `entry.sh`,
  `serve-w7.sh` = `scripts/serve.sh` + `W7_DOCKER` docker args); collection was started / stopped per window with
  `nsys start|stop --session=w7` on both ranks at once (`nsysctl.sh`). Needs `--cap-add SYS_ADMIN` (the driver has
  `RmProfilingAdminOnly: 1`; without it CUPTI returns INSUFFICIENT_PRIVILEGES and no kernel is recorded). The hardware
  CUDA trace keeps no kernels after the first start/stop cycle; `cuda-sw` does. **A second start/stop cycle in the
  same session lost the nsys agent and took both ranks down (load A1)**: take one capture per server lifetime.
- **Attribution: NVTX marks from the existing probe sites.** A bind-mounted copy of `profile.py`
  (`results/W7/profile.py`, measurement only) makes every `GLM53_TF_PROFILE` probe site also drop an NVTX mark, marks
  each `decode.prefill` (one batch piece) begin / end, prints the per-piece event table on BOTH ranks, and wraps the
  batcher's round / piece / verify / drafting / sampling calls in NVTX ranges. A kernel belongs to the component
  named by the first mark after its launch on the serving thread (`analyze.py`). Graph-replayed decode kernels carry
  no marks, so decode is broken down by kernel-name family (`dec.py`).
- **Partition of the wall time** (`analyze.py`): every instant of a window is given to the compute kernels running
  then (split evenly when several run on different streams), else to "NCCL exposed" when only an all-gather runs,
  else memcpy/memset, else GPU idle. So each column adds up to the wall time, and "NCCL exposed" is the part of the
  communication nothing hides. **Overlap tax**: each compute kernel that ran beside an NCCL kernel against the median
  of the same kernel (name, grid, block, component) running alone.
- **Rank skew** (`skew.py`): the two ranks' all-gathers matched in order; an all-gather kernel spins until the peer's
  data arrive, so `max(0, own - peer)` duration is that rank's wait and the shorter one approximates the transfer.
- **Distortion check.** 24.5k cold prefill: 1,264 tok/s with nsys attached but idle and probes off, 1,258 with probes
  on, 1,247 while capturing (-1.3%); 98k 1,237.5 under capture vs 1,251.5 without. Decode rounds: 54-60 ms
  (1 stream) and 113-122 ms (4 streams) without capture vs 59.5 / 125.5 ms under capture (<= 5-10%, mostly in the
  host gaps; kernel times are unaffected).

The "24.5k" / "98k" cells are `ab.py`'s (and W6's) prompt sizes: 21,464 and 85,781 prompt tokens.

## 1. Prefill, whole prompts, both ranks (ms and % of wall)

| component | 24.5k r0 ms | 24.5k r0 % | 24.5k r1 ms | 24.5k r1 % | 98k r0 ms | 98k r0 % | 98k r1 ms | 98k r1 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Routed experts (fat kernels, rot_in, plan) | 4,637.2 | 26.8 | 4,643.8 | 26.8 | 17,182.5 | 24.7 | 17,226.8 | 24.8 |
| MoE router / grouping + combine | 876.4 | 5.1 | 884.2 | 5.1 | 3,417.4 | 4.9 | 3,479.0 | 5.0 |
| Shared expert + dense MLP GEMMs | 996.9 | 5.8 | 989.3 | 5.7 | 3,983.3 | 5.7 | 3,802.3 | 5.5 |
| KDA: projections (q4 GEMMs) | 1,997.3 | 11.5 | 2,023.1 | 11.7 | 7,951.6 | 11.4 | 8,048.0 | 11.6 |
| KDA: chunked recurrence | 1,387.7 | 8.0 | 1,386.8 | 8.0 | 5,594.7 | 8.1 | 5,507.2 | 7.9 |
| DSA/MLA: q/kv proj + absorb | 603.1 | 3.5 | 606.8 | 3.5 | 2,400.5 | 3.5 | 2,419.2 | 3.5 |
| DSA/MLA: indexer + top-k | 283.7 | 1.6 | 291.0 | 1.7 | 2,709.6 | 3.9 | 2,737.2 | 3.9 |
| DSA/MLA: sparse + dense attention | 2,094.0 | 12.1 | 2,091.4 | 12.1 | 8,513.0 | 12.3 | 8,517.2 | 12.3 |
| DSA/MLA: latent expand + o_proj | 1,499.4 | 8.7 | 1,512.7 | 8.7 | 6,012.2 | 8.7 | 6,049.5 | 8.7 |
| Hyper-connections (hc_post / pre / Sinkhorn) | 2,248.5 | 13.0 | 2,187.4 | 12.6 | 8,965.1 | 12.9 | 8,938.4 | 12.9 |
| MTP prefill cache rows | 63.3 | 0.4 | 60.3 | 0.3 | 248.6 | 0.4 | 245.8 | 0.4 |
| DFlash2 drafter taps | 171.6 | 1.0 | 166.8 | 1.0 | 659.6 | 0.9 | 671.0 | 1.0 |
| NCCL exposed (nothing else running) | 144.0 | 0.8 | 148.7 | 0.9 | 538.6 | 0.8 | 565.0 | 0.8 |
| memcpy / memset only | 68.3 | 0.4 | 70.6 | 0.4 | 276.6 | 0.4 | 267.3 | 0.4 |
| GPU idle (host / Python gaps) | 213.1 | 1.2 | 220.7 | 1.3 | 915.2 | 1.3 | 896.3 | 1.3 |
| other (head, sample, stage, commit, piece setup) | 38.4 | 0.2 | 38.9 | 0.2 | 94.4 | 0.1 | 92.6 | 0.1 |
| **wall** | **17,323.0** | 100 | **17,322.6** | 100 | **69,463.0** | 100 | **69,463.0** | 100 |

## 2. One real 2,048-row piece, both ranks

One 2,048-row piece: piece 6/11 of the 24.5k prompt (context 10,240) and piece 41/42 of the 98k prompt (context 81,920).

| component | 24.5k r0 ms | 24.5k r0 % | 24.5k r1 ms | 24.5k r1 % | 98k r0 ms | 98k r0 % | 98k r1 ms | 98k r1 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Routed experts (fat kernels, rot_in, plan) | 393.0 | 24.8 | 394.8 | 24.9 | 405.5 | 23.9 | 401.9 | 23.7 |
| MoE router / grouping + combine | 83.1 | 5.2 | 84.0 | 5.3 | 82.4 | 4.9 | 83.0 | 4.9 |
| Shared expert + dense MLP GEMMs | 91.7 | 5.8 | 90.1 | 5.7 | 92.8 | 5.5 | 93.9 | 5.5 |
| KDA: projections (q4 GEMMs) | 186.7 | 11.8 | 191.1 | 12.1 | 189.5 | 11.2 | 191.4 | 11.3 |
| KDA: chunked recurrence | 132.2 | 8.3 | 133.5 | 8.4 | 134.1 | 7.9 | 133.0 | 7.9 |
| DSA/MLA: q/kv proj + absorb | 56.2 | 3.5 | 57.1 | 3.6 | 57.1 | 3.4 | 57.2 | 3.4 |
| DSA/MLA: indexer + top-k | 30.4 | 1.9 | 30.3 | 1.9 | 120.2 | 7.1 | 119.6 | 7.1 |
| DSA/MLA: sparse + dense attention | 203.4 | 12.8 | 204.8 | 12.9 | 204.2 | 12.0 | 203.3 | 12.0 |
| DSA/MLA: latent expand + o_proj | 141.3 | 8.9 | 143.7 | 9.1 | 142.9 | 8.4 | 142.5 | 8.4 |
| Hyper-connections (hc_post / pre / Sinkhorn) | 216.2 | 13.6 | 203.8 | 12.9 | 213.2 | 12.6 | 215.9 | 12.7 |
| MTP prefill cache rows | 5.7 | 0.4 | 6.0 | 0.4 | 6.4 | 0.4 | 5.6 | 0.3 |
| DFlash2 drafter taps | 16.0 | 1.0 | 15.9 | 1.0 | 15.7 | 0.9 | 15.0 | 0.9 |
| NCCL exposed (nothing else running) | 5.9 | 0.4 | 6.6 | 0.4 | 6.0 | 0.4 | 8.1 | 0.5 |
| memcpy / memset only | 6.1 | 0.4 | 5.8 | 0.4 | 6.1 | 0.4 | 5.9 | 0.3 |
| GPU idle (host / Python gaps) | 9.7 | 0.6 | 9.8 | 0.6 | 10.9 | 0.6 | 10.9 | 0.6 |
| other (head, sample, stage, commit, piece setup) | 1.4 | 0.1 | 1.6 | 0.1 | 1.5 | 0.1 | 1.5 | 0.1 |
| **wall** | **1,584.5** | 100 | **1,584.8** | 100 | **1,694.3** | 100 | **1,693.9** | 100 |

## 3. Communication in prefill

| window | count | kernel ms | overlapped by compute ms | exposed ms | exposed % of wall | overlap tax ms | tax % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24.5k r0 | 4,038 | 2,302.9 | 2,158.9 | 144.0 | 0.8 | 1,495.6 | 8.6 |
| 24.5k r1 | 4,038 | 2,249.7 | 2,101.0 | 148.7 | 0.9 | 1,428.4 | 8.2 |
| 98k r0 | 15,541 | 9,187.8 | 8,649.2 | 538.6 | 0.8 | 6,030.5 | 8.7 |
| 98k r1 | 15,541 | 9,127.7 | 8,562.8 | 565.0 | 0.8 | 5,843.5 | 8.4 |

- 90 row-parallel exchanges a token chunk: every lean 512-row sub-block all-gathers its bf16 partial (512 x 4,096 x
  2 B = 4 MiB a rank) with NCCL's `RING_LL` protocol. Median transfer (the shorter of the two ranks' kernels) 456 us
  = ~9 GB/s effective, p90 620 us. 94% of the all-gather kernel time runs beside compute (the 0084 pipeline); only 0.8% of
  the wall is exposed.
- **The hidden cost is the overlap tax: 8.2-8.7% of the wall.** The kernels that share the GPU with the all-gathers
  are the hyper-connection slabs and the shared expert: `_hc_post` (grid 1,536) takes a median 139 us alone and
  413 us beside an all-gather (3,307 of 3,780 calls in the 24.5k prompt ran beside one); shared-expert `_fq4` GEMMs
  lose ~0.25 s. So communication costs ~9-9.5% of prefill (0.8% exposed + ~8.5% tax), and "hyper-connections 13%"
  is roughly half hc work and half tax.
- Turning the overlap off (per request `prefill_overlap: 0`, same bits): 1,184 vs 1,266 tok/s at 24.5k (-6.4%),
  1,178 vs 1,252 at 98k (-5.8%): exposing the exchanges costs more than the tax.

**Rank skew.** Both ranks are balanced: at 24.5k rank 0 waits 0.39 s and rank 1 0.34 s inside all-gathers (of
17.3 s; who arrives first is 51/49), at 98k 1.60 / 1.54 s (of 69.5 s, 46/54). The per-piece GPU time from the event
probes is identical on both ranks to 0.01 s (16.96 / 16.97 s at 24.5k, 68.61 / 68.61 s at 98k). The waits are
per-collective jitter inside the overlapped kernels, not a slow rank.

**GPU idle / host gaps: 1.2-1.3% of prefill**, split evenly between gaps inside pieces (Python between kernels) and
between pieces (the round loop, plan share, snapshot, next piece's admission: ~10 ms a piece boundary).

## 4. Prefill by kernel family (24.5k, rank 0, kernel busy time; overlapping kernels both count)

| kernel family | ms | % of wall |
| --- | ---: | ---: |
| routed experts (`expert_kernel` fat + rot_in / plan / epilogues) | 4,689 | 27.1 |
| dense q4 GEMMs (`_fq4` / `_qmm`: KDA / DSA projections, o_proj, shared expert, dense MLP, drafter) | 3,896 | 22.5 |
| NCCL all-gather (97% overlapped) | 2,313 | 13.4 |
| hyper-connections (`_hc_post` 1,832 of it) | 2,237 | 12.9 |
| sparse latent attention (`_lsparse_chunks` / `_merge`, dense `_lchunks`) | 2,095 | 12.1 |
| MLA latent expand (`_expand`, 2.4 ms a 512-row sub-block) + absorb | 1,490 | 8.6 |
| KDA chunked recurrence (`_kda_prep` / `_state` / `_norm`) | 1,385 | 8.0 |
| MoE combine (`_combine_s`) | 665 | 3.8 |
| router / grouping, indexer, elementwise | 517 | 3.0 |

## 5. Decode rounds (nsys; kernel ms a round, mean over rounds)

1 stream: 106 rounds of a 256-token plain-prose reply (2.4 tokens a round). 4 streams: the 145 rounds with all four
requests in flight (384-token replies, 1.9-2.6 tokens a round each; 71-73 tok/s aggregate).

| kernel family | 1 stream r0 ms | % | 1 stream r1 ms | % | 4 streams r0 ms | % | 4 streams r1 ms | % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| routed experts (exl3 grouped) | 26.43 | 44.4 | 26.32 | 44.3 | 72.31 | 57.6 | 71.77 | 57.2 |
| dense q4 GEMMs (proj, shared expert, head, drafter) | 16.29 | 27.4 | 16.33 | 27.5 | 22.98 | 18.3 | 23.02 | 18.3 |
| NCCL | 4.77 | 8.0 | 4.89 | 8.2 | 7.67 | 6.1 | 8.34 | 6.6 |
| DSA/MLA attention + indexer | 2.60 | 4.4 | 2.64 | 4.4 | 5.21 | 4.2 | 5.24 | 4.2 |
| KDA (chain, conv, replay) | 1.60 | 2.7 | 1.61 | 2.7 | 5.53 | 4.4 | 5.60 | 4.5 |
| hyper-connections | 1.51 | 2.5 | 1.52 | 2.6 | 1.76 | 1.4 | 1.75 | 1.4 |
| MoE router / grouping / combine | 1.25 | 2.1 | 1.26 | 2.1 | 1.58 | 1.3 | 1.58 | 1.3 |
| norms / elementwise / other | 0.28 | 0.5 | 0.28 | 0.5 | 0.72 | 0.6 | 0.74 | 0.6 |
| memcpy / memset | 0.10 | 0.2 | 0.10 | 0.2 | 0.45 | 0.4 | 0.52 | 0.4 |
| GPU busy (union) | 54.82 | 92.2 | 54.94 | 92.4 | 118.22 | 94.2 | 118.55 | 94.5 |
| NCCL exposed | 4.77 | 8.0 | 4.89 | 8.2 | 7.67 | 6.1 | 8.34 | 6.6 |
| GPU idle | 4.65 | 7.8 | 4.51 | 7.6 | 7.29 | 5.8 | 6.96 | 5.5 |
| **round wall (under capture)** | 59.47 | 100.0 | 59.45 | 100.0 | 125.51 | 100.0 | 125.51 | 100.0 |
| all-gathers a round | 100 |  | 100 |  | 120 |  | 120 |  |
| kernels a round | 1846 |  | 1846 |  | 2672 |  | 2673 |  |

By phase (where the kernel was launched):

| phase | 1 stream r0 | 1 stream r1 | 4 streams r0 | 4 streams r1 |
| --- | ---: | ---: | ---: | ---: |
| verify forward | 49.20 | 49.05 | 102.42 | 102.36 |
| drafting (MTP / DFlash2) | 4.80 | 4.82 | 13.65 | 13.85 |
| round other | 0.44 | 0.44 | 1.06 | 1.06 |
| sampling | 0.30 | 0.55 | 0.64 | 0.77 |

- The verify forward is 83% (1 stream) / 82% (4 streams) of a round; drafting (MTP head, DFlash2 block) 8% / 11%.
- Routed experts are 44% of a single-stream round and 58% of a 4-stream round: 4 streams add ~46 ms of expert reads
  (rows of different sequences rarely share experts) against +7 ms of dense GEMMs.
- Communication in decode is not hidden: 100-120 all-gathers a round of 16-128 KiB, median transfer 24-29 us, 4.8-8.3 ms
  a round (6-8%); about half of it is waiting for the peer (rank skew per round ~3 ms each way, both directions:
  jitter, not a slow rank).
- GPU idle inside a round 4.5-7.3 ms (5.5-7.8%), plus ~3.9 ms between single-stream rounds (plan share, HTTP emit).

## 6. Piece / chunk size for one active request (no capture; `results/W7/ab-B.*`, `ab-C.*`)

| load (piece / PREFILL_ROWS_MAX) | chunk rows (per request) | 24.5k tok/s | 98k tok/s | vs 2,048 | lean chunk buffers a rank | MemAvailable min r0 / r1 (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| prod: 2,048 / 2,048 (A1/A2, nsys idle) | 2,048 | 1,264 | (1,237 under capture) | - | 0.78 GiB | 14.97 / 13.60 (98k + decode) |
| B: 8,192 / 8,192 | 2,048 | 1,266 | 1,252 | base | 3.10 GiB | 13.38 / 12.17 (sweep) |
| B | 4,096 | 1,310 | 1,294 | +3.5% / +3.4% | | |
| B | 8,192 | **1,339** | **1,319** | **+5.8% / +5.4%** | | |
| B | 2,048, `prefill_overlap: 0` | 1,184 | 1,178 | -6.4% / -5.8% | | |
| C: 4,096 / 4,096 | 4,096 | 1,307 | 1,290 | +3.2% / +3.0% | 1.55 GiB | 15.91 / 14.71 |

- Every cell has the same reply sha (`8794a3463259cc2f`, as W6); `exact` 10/10 on load B (8,192-row chunks by
  default). Decode after the prompt 85-89 tok/s everywhere.
- The piece itself hardly matters: 4,096-row chunks in 8,192-token pieces (B) and in 4,096-token pieces (C) are within
  0.3%. The gain is the chunk size (fewer, fuller expert passes; the per-boundary cost is ~10 ms).
- Memory: +0.8 GiB a rank at 4,096 rows, +2.3 GiB at 8,192 (buffers allocated at load, used or not). Load B went to
  12.38 / 11.18 GiB during its 4-stream decode + `exact` phase. The W6 worst case (4 x 300k overfill) was 15.1 / 14.05
  at 2,048; the 8,192 load's worst case was not measured (needs the stress run before any adoption; target >= 8).
- **A larger piece only while one request is active** is possible and exact (bits do not depend on piece or chunk:
  same sha in every cell, fast snapshots on the 64-token grid), but not with today's code: `GLM53_TF_BATCH_PIECE` is
  fixed at load and shared by both ranks, and `batchplan.piece_rows` uses it for every piece. A rule "use an 8,192
  piece / chunk when no other slot is decoding, else 2,048" would be a small scheduler change (rank 0 decides, the
  piece end already travels in the plan), but the 8,192-row buffers (+2.3 GiB a rank) must be allocated at load
  either way. With decoders present an 8,192 piece (~6.3 s) stalls them 4x longer per piece (the 0.5 fair share still
  bounds the average). Per request today: `tf_knobs.prefill_rows` picks the chunk up to `PREFILL_ROWS_MAX`.

## 7. Conclusions: which lever is largest

1. **Expert GEMM efficiency (routed experts, then the dense q4 GEMMs) is the largest lever.** Routed experts are
   25-27% of prefill wall and 44% / 58% of a 1- / 4-stream decode round; the q4 GEMMs another ~22% of prefill kernel
   time and 27% / 18% of decode. It is the only block that dominates both prefill and decode. W2/W5 showed the 2,048-row
   expert kernel is bound by data movement and scheduling, not decode or MMA: that is where to dig (fewer passes over
   member rows, better L2 reuse of weight slices).
2. **Attention (DSA/MLA) is the second block and the long-context lever:** 26-28% of prefill (sparse attention 12%,
   latent expand + o_proj 8.7%, q/kv 3.5%, indexer 1.6-3.9%). The indexer grows with context (1.9% of a piece at 10k
   context, 7.1% at 82k), and `_expand` (2.4 ms a sub-block) is a single kernel worth a look.
3. **Communication: ~9% of prefill, but almost all of it is the overlap tax, not exposed time.** Pipeline
   parallelism can at best recover that ~9% in prefill, while splitting layers across the ranks would make each
   single-stream decode step run the two halves one after the other (each Spark's bandwidth idle half the time)
   unless several sequences are micro-batched: it would roughly halve single-stream decode. **Not worth it.** The
   cheaper communication levers are the 4 MiB prefill all-gathers themselves (RING_LL moves them at ~9 GB/s on a
   25 GB/s link; protocol / channel count) and where they overlap (beside `_hc_post`, which runs 3x slower next to
   them), each with an ~8% ceiling. Decode's exposed 6-8% is latency (100-120 small exchanges a round).
4. **Piece / chunk size: +3.4% at 4,096 rows, +5.4-5.8% at 8,192, available now** (same bits), for +0.8 / +2.3 GiB a
   rank. The cheapest win measured; needs the worst-case memory run and a scheduler rule to use it only when one
   request is active.
5. **Host gaps are not a lever:** 1.2-1.3% of prefill; in decode 6-8% idle in a round plus ~4 ms between rounds.

Ranked by size: experts (GEMMs) > attention > hyper-connections (half of it comm tax) > communication (~9%, mostly
tax) > piece size (+3-6%, cheapest) > host gaps (~1%).
