# GLM-5.3-Flash TP=2: where the cross-rank communication goes, and what can be cut exactly

Offline analysis of `vendor/TensorFold` @ 2f8e514 + patches 0001-0010 (engine:
`src/tensorfold/families/glm5_next/cuda/{forward,mtp,decode,dflash2,comm,graphs,glue}.py`). Nothing was run on the
Sparks or a GPU. Timings come from the recipe ("Where the time goes") and `docs/DECODE-ANALYSIS.md`.

## 1. Every collective, per step

The model: 45 layers over a hidden size D = 4,096 (34 KDA + 11 DSA attention; 3 dense MLP + 42 MoE), 4
hyper-connection streams, vocabulary 154,880 split in halves over the ranks. Every block is row-parallel: each rank
computes its heads or its half of every MLP/expert width, writes an fp32 partial `[R, D]` (`b.part`), and
`forward.gather` all-gathers the partials (NCCL `ncclAllGather`, fp32, on the current stream, captured in the step
graphs). `glue._hc_post` (or `_residual_add` in the MTP head) adds them rank 0 first, rounds once to bf16 and applies
the hyper-connection epilogue, in one kernel.

| Site | Code | Per | Count | Send per rank | Recv |
| --- | --- | --- | ---: | ---: | ---: |
| KDA attention output (`o_proj` partial) | `kda_block -> out_proj -> gather` | main forward | 34 | 16 KiB x R | 32 KiB x R |
| DSA attention output (`o_proj` partial) | `dsa_block -> out_proj -> gather` | main forward | 11 | 16 KiB x R | 32 KiB x R |
| Dense MLP output (layers 0-2, `down` partial) | `mlp_block -> out_proj -> gather` | main forward | 3 | 16 KiB x R | 32 KiB x R |
| MoE output (routed + shared, `combine` partial) | `moe_block -> gather` | main forward | 42 | 16 KiB x R | 32 KiB x R |
| Head logits | none in the forward: each rank keeps its vocabulary half | - | 0 | - | - |
| Sampling: every row's top candidates | `decode.sample_rows` (eager, then `.cpu()`) | verify, and each MTP draft | 1 | R x 2k x 4 B: greedy 8 B/row, sampled (k = 20 + 8) 224 B/row | 2x |
| MTP head: DSA output, MoE output | `mtp_compute` (graph per row count) | MTP step | 2 | 16 KiB x n | 32 KiB x n |
| DFlash2 drafter: attention + MLP output per layer | `Drafter._row` (in the block graph) | DFlash2 block | 2L | 128 KiB (8 rows) | 256 KiB |
| DFlash2: packed candidates | `Drafter._block_compute` | DFlash2 block | 1 | 7 x 2 x 8 x 4 = 448 B | 2x |

Per one-row verify step: **90 captured all-gathers of 16 KiB + 1 eager of 8-224 B**, 1.41 MiB sent a rank. An 8-row
window: 90 x 128 KiB = 11.3 MiB. An MTP step: 2 + 1. A DFlash2 block: 2L + 1 (about 11 with the published drafter,
per `DECODE-ANALYSIS.md`).

### Latency model

`t(bytes) = alpha + bytes / beta`, with alpha about 27-28 us for an all-gather captured in a CUDA graph and about 17
us eager (recipe), and beta = 25 GB/s for 200 Gb/s RoCE (NCCL's LL protocol sends 4 B of flag per 4 B of data, so
about 12.5 GB/s on the wire).

| Window | Bytes a gather | Wire time (Simple / LL) | alpha | Share that is latency | 90 gathers | Share of step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 row | 16 KiB | 0.7 / 1.3 us | 27 us | 95-98% | 2.5 ms (measured about 2.4) | 8% of 30.0 ms |
| 4 rows | 64 KiB | 2.6 / 5.2 us | 27 us | 84-91% | 2.7-2.9 ms | 6% of 49.5 ms |
| 8 rows | 128 KiB | 5.2 / 10.5 us | 27 us | 72-84% | 2.9-3.4 ms | 5% of 68.4 ms |

The step's bandwidth floor is about 18 ms (5.0 GB a rank at the ~270 GB/s peak); measured 30.0 ms. Communication is
2.4 ms of the 12 ms gap, and all of it is latency: halving the bytes would save under 0.1 ms at one row.

## 2. Options, checked for exactness

Requirement: both ranks hold identical bits after every reduction, and the bits equal the serial engine's (drafted
replies byte-identical to serial ones, and to the default engine's).

**(a) Merging the attention all-gather into the FFN block's: not possible exactly.** Between the two exchanges
the engine computes `X_s = bf16(post_s * bf16(p0 + p1) + sum_j comb[j, s] X_j)` (hc_post), then hc_pre's mix
`rsqrt(mean(X^2)) * (X . fn)` over all four updated streams, sigmoid and 20 Sinkhorn iterations, the collapsed row
`bf16(sum_s pre_s X_s)`, and RMSNorm. Every one of those is nonlinear in the gathered sum (and the bf16 rounding of
the branch alone is), so the FFN's input cannot be computed from the partials, and neither can the router, the
shared expert or any expert. The only exact way to drop an exchange is to stop splitting a block (replicate it):
the cheapest candidates, the 3 dense MLPs, would save 3 x 27 us = 0.08 ms and read 0.13 GB more a rank (+0.5 ms).
Replicating attention's `o_proj` needs every head's output on both ranks, which is itself an exchange. **90 is the
minimum for row-parallel TP on this model.**

**(b) All-reduce / reduce-scatter instead of all-gather + local sum: exact for two ranks, but slower.** With two
ranks an fp32 sum is `p0 + p1` on one rank and `p1 + p0` on the other; IEEE addition is commutative, so both are the
same bits and equal the rank-0-first sum `_hc_post` computes (with three or more ranks the order would matter). But
NCCL's all-reduce for two ranks is a reduce-scatter then an all-gather: two dependent network steps where the
all-gather is one, about 2 alpha instead of alpha. Reduce-scatter alone leaves each rank half a row, and every
consumer (hc_post, RMSNorm) needs the whole row. No gain.

**(c) Overlapping the exchange with independent compute: there is no independent compute, but there is idle
bandwidth.** The dependency chain is strict: partial -> all-gather -> hc_post -> hc_pre (needs the whole row) ->
router / shared expert / experts / attention projections -> next partial. The task's candidates (router logits,
shared expert, the next layer's pre-mix) all read the normed row that the exchange produces. The only pre-gather
computable piece is hc_post's `sum_j comb[j, s] X_j`, about 1-2 us, and splitting it out of the fused kernel risks a
different FMA contraction (bits). What is known before the exchange ends is *which weights come next*, and for
27 us the GPU's memory system reads nothing: the step is bandwidth-bound everywhere else. **Prefetching the next
kernels' weights into L2 during the exchange is exact by construction (loads only) and is what patch 0040
implements.**

**(d) Protocol and transport.** `NCCL_PROTO=LL`/`LL128` only changes how bytes travel; an all-gather is a copy, so
it is exact. At 16 KiB NCCL's tuner almost certainly picks LL already; at 4-8 rows LL128 or Simple might win a few us
per gather (at most 0.2-0.5 ms on a wide window). The 11 us captured-vs-eager gap is most likely NCCL's graph-mode
handoff to its network proxy thread, which no protocol setting removes. `NCCL_GRAPH_MIXING_SUPPORT=0` is **not safe
here**: NCCL only guarantees correctness with it off if an outstanding graph launch is never followed by an
un-captured launch, and `decode.draft` replays the MTP graph and then calls `sample_rows`' eager all-gather without a
sync. A custom GPU-initiated exchange (NVSHMEM/IBGDA put + signal over the ConnectX-7, capturable, no proxy) could
bring alpha to maybe 5-8 us: 90 x ~20 us = about 1.8 ms a step. That is the largest exact win available, but it is
a transport project (NVSHMEM bootstrap, symmetric buffers for `b.gath`, IBGDA on RoCE on GB10) that needs the
cluster to build and test.

**(e) Fused epilogue: already done.** `_hc_post` reads the gathered `[2, R, D]` fp32 partials, sums rank 0 first,
rounds to bf16 and applies the post/comb mixing for all four streams in one kernel; the MTP head's `_residual_add`
does the same. The remaining fusion, hc_post into the next hc_pre's `_hc_partial`, would save one kernel boundary
(about 2-3 us in a graph) per exchange, about 0.2 ms a step, but needs out-of-place streams (hc_post updates X in
place while other programs of the fused kernel would still read the old X). It is compute-side work; left as a
follow-up.

**Also small:** the sample exchange runs eagerly after each graph (about 17 us + a launch); capturing top-k and the
all-gather into the verify graph would save ~10-20 us a step.

## 3. What patch 0040 does (`GLM53_TF_COMM`)

`patches/0040-glm-comm-prefetch.patch` (new `cuda/overlap.py`; small hooks in `forward.py`, `mtp.py`, `decode.py`,
`comm.py`). Default (unset, `nccl`): every hook is a Python `None` check; the kernel sequence is upstream's.

- `prefetch[:MB]` (default 4 MB, capped at half the device's L2): `layer_forward` and `mtp_compute` tag each block
  with its all-gather site. At the all-gather, `gather` forks a side stream from the current stream (after the block's
  partial is written) and launches `_touch`: half the SMs read the first MB of the weights the next kernels read,
  int32 loads with `.cg` / `evict_last`, summed into a dummy sink. Then the all-gather is queued on the main stream
  as before. Every forward and MTP step joins the side stream before it returns, so the fork is inside every
  captured graph (the graphs capture it like they capture the all-gather). Plans, built once at engine start:

  | After the all-gather of | Prefetched, in order, up to the budget | Real-model sizes a rank |
  | --- | --- | --- |
  | a layer's attention | `ffn_hc.fn`, `post_norm`, then the dense MLP's gate/up, or the router, its bias and the shared expert's gate/up | fn 768 KiB, router 2.25 MiB, shared gate/up 4.5 MiB |
  | a layer's MLP/MoE | the next layer's `attn_hc.fn`, `in_norm`, its KDA/DSA input projection (scales first, then the words' first N tiles); after the last layer the final norm and the head | fn 768 KiB, KDA projection ~24 MiB, head 151 MiB |
  | the MTP head's attention / MoE | its router and shared gate/up / `shared_head.norm` and the (draft) head | as above |

  With prefetch on, the EXL3 MoE block computes the shared expert (its own buffers `sgu`, `sact`, `sy`, `b.sk`)
  before the routed experts (their own `b.exl3` scratch), so its prefetched gate/up is still in L2; its result is
  copied into the last slot after the routed kernels exactly as before.
- `ll` / `ll128`: sets `NCCL_PROTO` (unless already set) before the communicator is created. Both ranks must use
  the same value (mismatched protocols can hang).
- Combine with commas: `GLM53_TF_COMM=prefetch:4,ll`.

### Exactness argument

- The prefetch kernel only loads from weight tensors and writes a private sink nothing reads. It changes no input of
  any model kernel, and each model kernel is the same launch with the same arguments. L2 residency changes timing,
  not values.
- The shared-first order in the EXL3 MoE block reorders two kernel chains with disjoint inputs and outputs
  (`normed`/`xs` read-only; shared writes `sgu`, `sact`, `sxs`, `sy`, and `b.sk` as split-K scratch; routed writes
  `b.exl3.*` and routed slots of `ey`); the `ey[:, top_k] <- sy` copy and `combine` run after both, as before.
  Stream order on the main stream is unchanged otherwise; the side stream only reads weights.
- The all-gather, its buffers, the rank-0-first sum in `_hc_post` and every rounding are untouched, so both ranks
  still hold identical bits after each exchange, equal to the default engine's.
- The protocol variants change only NCCL's transport of a copy.

### Expected gain (to be measured)

Per exchange the prefetch can save at most min(alpha, budget / DRAM rate): 4 MiB at about 230 GB/s is about 18 us
of later DRAM time, under the 27 us alpha. Upper bound 90 x 18 us = **1.6 ms of a 30 ms one-row step (5%)**, the
same absolute amount on wider windows and about 2 x 18 us on an MTP step. Realistic: **0.5-1.2 ms**, depending on
how much of the prefetched data survives in L2 until its kernel runs (the first kernels after the exchange,
hc_post/hc_pre, touch well under 1 MiB) and on whether the touch kernel delays the start of NCCL's kernel. If the
exchange is shorter than the prefetch the main stream does not wait (the join is at the end of the forward), so the
downside is bandwidth contention with the kernels after the exchange, i.e. a few us.

## 4. Validation that needs the two Sparks

`tests/cuda/test_comm_patches.py` (synthetic checkpoint, `_TwoCopies` stand-in: one GPU as rank 0, an all-gather that
returns its own partial twice on the current stream) checks the engine side: the side-stream fork/join in eager and
captured forwards, logits and hidden rows of windows 1-8 (graphs and eager) bit-equal to the default engine, replies
equal to the default engine's serial reply for greedy and sampled, serial and every drafting policy (MTP, DFlash2,
the per-round choice) on an MLX and an EXL3 + q4mse checkpoint, resumes, the plans reading only weights and staying
within the budget, and the knob parsing. It cannot check:

1. Two-rank bits: run the exactness suite (serial vs drafted, sampled and greedy, both prompts) with
   `GLM53_TF_COMM=prefetch` on both ranks and diff the transcripts against `GLM53_TF_COMM` unset. The patch does not
   touch the exchange or the summation order, so this is a regression check, not a new ordering question.
2. Overlap on the GPU: an `nsys` trace of one verify step to confirm `_touch` runs concurrently with
   `ncclDevKernel_AllGather_*` and does not delay its start.
3. Speed: the load-time calibration line (`drafter timings (ms)`, v1..v8, m1, block) for unset / `prefetch:2` /
   `prefetch:4` / `prefetch:6` / `prefetch:4,ll128`, then the tok/s cells. Keep the variant only if v1 drops.
4. `ll`/`ll128`: both ranks set identically; watch for hangs at start (the engine's first barrier).

## 5. Bottom line

The exchange count cannot be cut exactly: 90 all-gathers a forward is the minimum for row-parallel TP on this model,
the payloads are tiny, and the 2.4 ms (8% of a one-row step) is all per-exchange latency. What can be done exactly
is to use that latency: patch 0040 prefetches the next weights during each exchange (0.5-1.6 ms a step expected,
unmeasured). The protocol knob might give a few tenths of a ms on wide windows. The largest remaining exact
option is a GPU-initiated RDMA exchange (NVSHMEM/IBGDA) to cut alpha from ~27 to ~5-8 us, about 1.8 ms a step, as a
separate project on the cluster. Every other option that cuts the count changes the model (for example parallel
attention and FFN from the same input: 45 exchanges, different bits), so it is not worth pursuing under the
byte-identical rule.
