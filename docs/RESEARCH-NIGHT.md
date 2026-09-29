# Research night: paradigm shifts for decode (and prefill) on 2x DGX Spark, ranked for an 8-hour window

> Research note, 2026-09-28. Web research (2025-2026 papers, kits, vendor docs) plus this repo's docs (README,
> PARADIGMS, PROFILE, RESULTS W5-W8, ADAPTIVE-DRAFT, PATCHES, EXPERIMENTS, DECODE-/COMM-ANALYSIS, ROCE-FIX). No GPU
> and no Spark were used. Most gains below are **arithmetic**. Two numbers are new **offline measurements** made for
> this note: (a) the entropy of EXL3 trellis words and of 4-bit affine codes (section 3), and (b) the
> acceptance-ceiling histogram of 32k recorded drafted rounds (section 1). "Exact" means drafted == serial,
> batched == alone and resumed == fresh still hold with the same weights.
>
> **Not repeated here** (see PARADIGMS.md for them): the NVMe session tier (0250, adopted), prefix sharing (0310,
> adopted), row-split prefill (0320, adopted), 8,192-row solo pieces (0335, adopted), the b12x prefill kernels (0240,
> only bit 4 helps), expert `once` / `tc` kernels (0260 / 0330, not adopted), and 0230 RoCE, which the W9 window is
> testing now (`results/W9/roce/SUMMARY`: transport clean, stress ok). They appear below only as dependencies.

## 0. Bottom line

- **Nothing that fits an 8-hour window gives +20% on every cell.** Single-stream decode on a one-row step is ~30 ms
  against a ~24.6 ms floor. The only ways past +20% on general text are **more tokens per round** (better drafters)
  or **fewer weight bytes** (3.x-bpw experts, which is parked and needs a quality gate). Neither is an overnight job.
- **Lossless weight compression is dead for this checkpoint** (measured, section 3). EXL3 trellis words carry
  15.97 of 16 bits of entropy (zstd-19 ratio 1.0000). The 4-bit affine codes of the non-expert weights carry
  3.75-3.87 of 4 bits. The most an entropy coder could save is ≤ 1.5% of a round, and decoding it costs more.
- **What overnight work can buy:**
  - ~+4-9% on every decode cell, from the host path: pin threads to the Cortex-X925 cores, and take the per-round
    host tail and rank jitter off the critical path.
  - +5-15% on agent turns only, from a cross-request suffix tree in the lookup arm.
  - +3-5% from 0230, if W9 passes. That is its own window.
- **Largest levers for the following week(s):**
  - Drafters trained on the model's own outputs (FastMTP-style MTP self-distillation, then an on-policy distilled
    block drafter with a longer block): **prose +10-25%**, where decode is slowest.
  - Deeper verify windows (12-16 rows) for the rounds that hit today's 8-row ceiling. That is **33-40% of DFlash2
    rounds on code-like streams and 83% on repetitive ones** (measured below).
  - A weight-major persistent MoE decode kernel (MonoMoE-style) for the 4-stream case, where routed experts are 58%
    of the round.

## 1. Where a round goes, and the acceptance ceiling (new measurement)

From PROFILE.md §5 (nsys, production):

| 1 stream, 59.5 ms round (under capture) | ms | 4 streams, 125.5 ms | ms |
| --- | ---: | --- | ---: |
| routed experts | 26.4 | routed experts | 72.3 |
| dense q4 GEMMs | 16.3 | dense q4 GEMMs | 23.0 |
| NCCL exposed (100 gathers; ~half is waiting for the peer) | 4.8 | NCCL exposed (120) | 7.7-8.3 |
| GPU idle inside the round | 4.5 | GPU idle | 7.0-7.3 |
| attention + KDA + hc + router | 7.0 | attention + KDA + hc + router | 14.1 |
| **plus between rounds (plan share, HTTP emit)** | **~3.9** | | |

So ~13 ms of a ~63 ms single-stream cycle (~21%) is not bandwidth work: idle, the host gap between rounds, and
all-gather waiting. That is the biggest *exact, engineering-only* pool left for decode.

**Acceptance ceiling.** This is DBloom's preflight check ([arXiv 2608.30427](https://arxiv.org/abs/2608.30427)):
a spike in the "accepted the whole block" bin means block-limited acceptance. I computed it from the per-round
`keeps` of every recorded concurrent run (`results/W1/conc.json`, `results/W5/conc-*.json`, `results/W6/conc-A.json`, `results/W8/conc-*.json`;
32,425 rounds; `keep` = tokens committed, 8 = all 7 drafts + the bonus).

| arm | rounds | keep 1 | 2 | 3 | 4 | 5 | 6 | 7 | **8 (ceiling)** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DFlash2 (`f`) | 18,087 | 0.190 | 0.272 | 0.137 | 0.090 | 0.043 | 0.059 | 0.018 | **0.190** |
| MTP (`m`) | 14,296 | 0.220 | 0.451 | 0.190 | 0.091 | 0.030 | 0.012 | 0.005 | 0.002 |

Per stream the picture is bimodal:

| stream type | tokens a round | DFlash2 rounds at the ceiling |
| --- | ---: | ---: |
| prose / chat | 2.0-2.7 | 0-4% |
| code-like | 4.1-4.9 | 32-40% |
| repetitive ("sequence") | 7.0 | 81-83% |

Two different problems follow:

- **Prose** is limited by **drafter quality**: rounds end early, and the ceiling is irrelevant. The lever is
  training (L1).
- **Code / structured** is limited by the **8-row window**: a third of its rounds would have gone on. The lever is
  deeper windows (L2), and suffix drafts for copies (N3).

## 2. Ranked: overnight (≤ 8 h to implement; GPU window for the A/B separate)

Ranked by expected gain × probability it works × fits in 8 h. All are exact.

| # | Idea | Evidence / source | What it needs here | Exactness | Estimated gain here | Effort | First check |
| ---: | --- | --- | --- | --- | --- | ---: | --- |
| N1a | **Pin the host threads to Cortex-X925 cores** (the engine's serving thread on each rank, NCCL's proxy, 0230's busy-spin proxy on its own X925 core; HTTP / tokenizer threads on A725) | GB10 is 10 x X925 + 10 x A725 (NVIDIA DGX Spark Porting Guide §1.1.1; A725 has 1/4 the L2 and a smaller L3). Nothing in `scripts/`, `docker/` or `patches/` sets affinity today (grep: no `cpuset`, `taskset`, `sched_setaffinity`). The decode round has d + 2 hard syncs, ~3.9 ms of host work between rounds, and ~3 ms of per-round rank skew that PROFILE calls "jitter, not a slow rank". Host-side jitter on either rank shows up as the other rank waiting inside all-gathers | `docker run --cpuset-cpus` or `os.sched_setaffinity` at engine start, with X925 ids from `/proc/cpuinfo` (`CPU part` 0xd85 = X925, 0xd87 = A725; verify on the box). Same on both ranks | Exact (scheduling only) | **+0-5% decode**, 1 and 4 streams. Unknown until measured: if the scheduler already keeps the hot thread on an X925, ~0. The rank-skew part (up to ~2.4 ms of the 4.8 ms NCCL exposure) is the upside | 1 h | During a decode, `ps -eLo tid,psr,pcpu,comm` on both ranks, sampled every 100 ms: how often is the busiest Python thread on an A725 core? Then A/B `canary` + `multiturn --streams 1,4` pinned vs unpinned |
| N1b | **Take the host tail off the critical path** ("Spec V2"-style overlap). Rank 0 launches the next round's draft before the HTTP emit / request-log / stats work; the plan share rides on an exchange that already happens; drop the timing-only `torch.cuda.synchronize()` after verify (DECODE-ANALYSIS §5.3) | SGLang Spec V2 overlap scheduling: "host-device synchronization kills inference performance", +33% at concurrency 32 on B200 ([LMSYS, 2026-06-15](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)). Ours: 3.9 ms between single-stream rounds + 4.5 ms idle inside a round (PROFILE §5) | Reorder `auto_decode` / the batcher round loop so that everything not needed for the next launch runs after it (emit on a queue to the HTTP thread). The plan share must stay rank-deterministic | Exact: the same kernels and inputs; only when host work runs changes | **+3-6% single stream** (recover ~2-4 ms of ~8.4), +2-4% at 4 streams | 5-6 h | Split the 3.9 ms from W7's trace (`~/w7-traces/mix-r0`, NVTX ranges already exist) into emit / plan share / stats. Go if ≥ 2 ms can move |
| N2 | **0230 RoCE one-shot all-gather in the engine** (already in W9) | b12x RoCEnante: 10.5-12.5 us all-gathers vs NCCL ~27 us in graphs and ~40 us for a 16-32 KB all-reduce between two Sparks ([Sangiorgi](https://contact.alessandrosangiorgi.net/posts/dgx-spark-nccl-collective-latency/)). Upstream still has an open wedge bug under graph replay ([b12x #313](https://github.com/local-inference-lab/b12x/issues/313)) | W9's engine A/B; keep 0350's timeout / poison / marker path, and the 256 KiB cap (the 1 MiB `bits_equal: false` in W7 is unexplained) | Exact (a data move) | **+3-5% decode** (90-120 x ~15 us) | in progress | Already running: do not duplicate. Soak ≥ 2 h at 4 streams before adoption, because of #313 |
| N3 | **Cross-request suffix tree for the lookup arm** (0020 today matches only the current request's prompt + reply). A frequency-scored suffix tree over all recent requests' prompts and replies on both ranks, proposing the most frequent continuation up to 7 tokens, gated by 0020's existing cost rule | SuffixDecoding (NeurIPS 2025 spotlight, [arXiv 2411.04975](https://arxiv.org/abs/2411.04975)): 6.3 accepted tokens a step on AgenticSQL, 7.8 on SWE-Bench; the global tree alone beats the per-request tree on most AgenticSQL stages; 1.66x alone on Spec-Bench chat. Production: 1.96-3.12x end to end in vLLM / Arctic Inference ([Snowflake](https://www.snowflake.com/en/engineering-blog/suffixdecoding-arctic-inference-vllm/)). Ours: 0020 measured +9-12% on edit cells; only 42 `l` rounds appear in the concurrent benches (the bench prompts are not agentic) | A host-only structure in `lookup.py` fed from committed tokens. Both ranks see the same requests in the same order, so the trees stay identical (assert a hash in the plan every N rounds). A size cap (e.g. 2M tokens, LRU by request). No new graph sizes: stays ≤ 8 rows | Exact (drafts only propose) | **Agent turns: +5-15% over 0020; prose: 0.** Most cross-turn repetition is already inside an agent's own prompt (0020 sees it). The global tree adds other sessions' outputs: subagents, the same tool-call scaffolding, repeated file emits | 6-8 h | Offline replay first. 0300's request log has no text, so use local agent transcripts (opencode sessions) re-tokenized with the GLM tokenizer. At each output position, measure the accepted length of the per-request lookup vs the global tree. Build only if the global tree adds ≥ 0.3 tokens a round on ≥ 20% of agent-turn tokens |
| N4 | **Acceptance-histogram instrumentation in /metrics** (keep-bin counts per arm and per request class) | DBloom's recommendation to look at the histogram before spending training compute (above) | Counters in 0150's `/metrics` from `keeps` / `arms`, which the engine already records | No behaviour change | 0 directly; decides L1 / L2 on real agent traffic instead of bench prompts | 1 h | Section 1's table, recomputed on a day of production |

Stacked estimate for N1a + N1b + N2, all exact: **single-stream decode +6-12%, 4-stream +4-9%**. Add N3 on agent
turns. That is the honest overnight ceiling.

## 3. Checked tonight and rejected

### Lossless / entropy-coded weights (DFloat11, Huff-LLM, EntroLLM, ANS): no gain on this checkpoint (measured)

The literature:

- DFloat11 compresses **BF16** to ~11 bits, bit-exact. At batch 1 it is ~2x *slower* than BF16 on GPU
  ([arXiv 2504.11651](https://arxiv.org/abs/2504.11651)).
- EntroLLM reports 1.39 bits/weight for 4-bit codes, but those are per-layer (not group-wise) quantized, so most
  codes sit in a few bins ([arXiv 2505.02380](https://arxiv.org/abs/2505.02380)).
- "Approaching Shannon Bound" (ANS; [arXiv 2606.15789](https://arxiv.org/abs/2606.15789)) finds only 1.1-1.3x on
  group-quantized weights, and has no batch-1 GEMV results.

What matters is the entropy of *our* bits, so I measured it:

| data | source (public, range-fetched) | entropy | general-purpose compressors |
| --- | --- | --- | --- |
| EXL3 4.0-bpw trellis words (48 MB of routed-expert `.trellis` tensors) | `turboderp/Qwen3-30B-A3B-exl3@4.0bpw` (same format; our checkpoint is gated) | 7.9995 bits/byte; 15.974 bits per 16-bit word; nibbles 3.9999 of 4 | zlib-9 1.0003, zstd-19 1.0000, lzma 1.0001 |
| 4-bit affine g64 codes of a real GLM-5.3-Flash weight (layer 20 `o_proj`, 4096 x 8192, from the public BF16 base) | `zai-org/GLM-5.3-Flash` | min/max scales 3.748 bits; clipped (MSE-style) 3.87-3.92 bits | - |

- **Experts** (44-58% of a round): the trellis bitstream is incompressible. QTIP / EXL3 already spend every bit.
- **Non-experts** (q4mse, 27% of a single-stream round): at best 3.75 / 4 of the code bits, with the scales untouched,
  so ≤ ~6% of those bytes, ≤ 1.5% of a round. A Huffman / ANS decode in the GEMV's inner loop costs more than that on
  GB10's ALUs (the W2 no-decode probe already showed that data movement, not ALU work, binds these kernels).
- **Rejected.** The same measurement also closes "compress the KV cache losslessly": FP8 latent rows are near-max
  entropy as well, and attention is 4% of decode anyway.

### Other rejections

| Idea | Why not here |
| --- | --- |
| **Tree verification, now that KDA trees are solvable** (STree [2505.14969](https://arxiv.org/abs/2505.14969), TreeWY [2608.20961](https://arxiv.org/abs/2608.20961), SGLang's Bole [2608.01651](https://arxiv.org/abs/2608.01651), SpecLA [2607.16673](https://arxiv.org/abs/2607.16673)) | These remove the *KDA* obstacle (Bole: exact closed-form tree recurrence, 1.26x over tree baselines, 2.03x peak on GB10, but dense and GDN models, no MoE at batch 1). Our obstacle was never KDA: it is **~4.7 ms of new experts per extra row** (EXPERIMENTS §5: a sibling is worth ~0.12 tokens ≈ 2.7 ms against a 4.7 ms row). "The Limits of Speculation" ([2609.22156](https://arxiv.org/abs/2609.22156)) bounds MoE speculation by the expert union and finds the oracle caps draft length at ~2.8 on Qwen3-30B-A3B. Stays rejected. Revisit only together with expert-aware tree pruning (EcoSpec, below), and only if a measured sibling acceptance is > 0.35 |
| **EcoSpec / EVICT / AcceptMoE / MoE-Spec** expert-aware draft selection ([2607.12696](https://arxiv.org/abs/2607.12696): up to 1.62x on DeepSeek-V3.1 / Qwen3-235B / GPT-OSS; [2605.00342](https://arxiv.org/abs/2605.00342), [2608.02989](https://arxiv.org/abs/2608.02989), [2602.16052](https://arxiv.org/abs/2602.16052)) | The cost-aware part is what 0071 already does: marginal row prices, stop at the cost boundary. The rest assumes trees, a router predictor for draft tokens, or (MoE-Spec, AcceptMoE) a **reduced expert set in verification, which is not exact**. Chain-only expert-aware pricing (predict a draft row's new experts) is worth ≤ 2-4%: longer-term L7 at most |
| Self-speculation with fewer experts / layer skip (DraftExpert [2607.24434](https://arxiv.org/abs/2607.24434), S2-MoE [2608.15018](https://arxiv.org/abs/2608.15018), MoE-SpeQ [2511.14102](https://arxiv.org/abs/2511.14102)) | Built for expert-*offloaded* edge MoE, where the target's experts are far away. Here a top-1/2 self-draft still reads ~2-4 GB a token (~8 ms), against DFlash2's 3.9 ms block for 7 positions |
| NVFP4 (W4A4) experts / non-experts | 4.5 bpw is more bytes than EXL3 4.0 (decode -4-5%), W4A4 changes outputs, and published GLM NVFP4 KLD is 2.5-3x ours (PARADIGMS §1 #10). vLLM itself notes SM121 NVFP4 is "generally slow" and still routes around sm_100-only paths ([vLLM #40082](https://github.com/vllm-project/vllm/pull/40082), [FlashInfer #3013](https://github.com/flashinfer-ai/flashinfer/issues/3013)). Unchanged |
| FP8 dense projections (MiaAI kit: "~11 ms per step saved") | Their baseline is BF16 dense. Ours is 4.5-bpw q4mse, so FP8 (8 bits) would *add* bytes. It is also output-changing (same class as the rejected FP8 prefill) |
| GPUDirect RDMA / NVSHMEM / IBGDA / NCCL symmetric-memory kernels | Not supported on Spark: iGPU `cudaMalloc` memory is not coherent to I/O, so nvidia-peermem / dma-buf / GDRCopy do not work; NVIDIA recommends `cudaHostAlloc` + `ibv_reg_mr` (DGX Spark Porting Guide §4.6.1, [NVIDIA answer 5780](https://nvidia.custhelp.com/app/answers/detail/a_id/5780/~/is-gpudirect-rdma-supported-on-dgx-spark)). That is exactly 0230's design. NCCL 2.28's symmetric kernels and the "near speed-of-light collectives" work ([2607.16100](https://arxiv.org/abs/2607.16100)) are NVLink / scale-up only |
| Persistent megakernel (Mirage MPK, AutoMegaKernel) | MPK's 1.2-1.7x is on dense models, H100 / B200. MoE support is on the fall-2026 roadmap ([mirage #773](https://github.com/mirage-project/mirage/issues/773)); nothing covers KDA + DSA + mHC + EXL3. Stays PARADIGMS #12 (25+ days) |
| DFlash2 drafts for sampled requests | TensorFold upstream measured DFlash2's sampled chains slower than MTP `a:0.6:0.85` (engine.py docstring); `auto:E:EVERY:MARGIN` already lets a request try both |
| Jacobi / lookahead decoding | Low acceptance on non-repetitive text. On repetitive text the suffix / lookup arm does the same job for free |
| IndexCache (cross-layer index reuse, [2603.12201](https://arxiv.org/abs/2603.12201)), token-dropping prefill | Not exact |

## 4. Longer-term (days; ranked by expected gain × feasibility)

| # | Idea | Evidence | Needs here | Exactness | Estimated gain | Effort |
| ---: | --- | --- | --- | --- | --- | ---: |
| L1 | **Train the drafters on the abliterated model's own outputs**: (a) FastMTP-style self-distillation of the MTP layer (LoRA on attention / `eh_proj` / shared expert, recursive training-time test); (b) an on-policy distilled block drafter (AdaFlash reverse-KL OPD + adaptive length head), started from a licence-clean base (canada-quant DFlash2-G, Apache-2.0, or RedHat DSpark, MIT) | FastMTP ([2509.18362](https://arxiv.org/abs/2509.18362)): 2.03x vs NTP, +82% over vanilla MTP. Red Hat's recipe on Qwen3-Next lifts acceptance at positions 0/1/2 from 0.897/0.719/0.476 to 0.912/0.776/0.616 with ~8k samples and ~7 minutes on 2 x H200 ([Red Hat, 2026-09-08](https://developers.redhat.com/articles/2026/09/08/optimize-vllm-speculative-decoding-fastmtp-heads)). AdaFlash ([2607.19223](https://arxiv.org/abs/2607.19223)): up to +66% throughput over prior SOTA, most at high concurrency. GLM-5's own paper: accept length 2.76 at 4 MTP steps with shared MTP layers ([2602.15763](https://arxiv.org/abs/2602.15763)). Ours: MTP chained a = 0.74 / 0.45 / 0.22; prose streams 2.0-2.7 tokens a round | Teacher-forced taps + MTP inputs from our model (prefill speed: 10M tokens ≈ 2.5 h at ~1,150-1,450 tok/s, prod down), training on one Spark with the target unloaded, a new drafter loader for DFlash2-G / DSpark shapes | Exact (drafters only) | MTP: a2 0.45 → ~0.55, a3 0.22 → ~0.35 gives E[T] 2.41 → ~2.65, **+8-12% on sampled / chat cells**. Block drafter: prose E[T] 2.4 → 2.9-3.2 at ~+1 row, **+12-25% prose**, and the 4-stream aggregate likewise (ADAPTIVE-DRAFT: the unreachable-by-policy oracle is +15-21%, and only better prediction reaches it) | 3-4 d (MTP), 7-10 d (block drafter) |
| L2 | **Verify windows of 12-16 rows** for rounds whose drafts are confident: the lookup / suffix arm first, then a block-16 drafter (DBloom-style block expansion needs the drafter post-trained, so it comes with L1b) | Section 1: 19% of all DFlash2 rounds, 32-40% on code-like streams and 81-83% on repetitive ones, hit the 8-row ceiling. DBloom ([2608.30427](https://arxiv.org/abs/2608.30427)): B16 → B24 adds a median +0.8 committed tokens (up to +1.37); naive widening at inference does not work. SuffixDecoding uses up to 66-token speculation on structured tasks | Buffers, graphs and the 0071 cost table past `MAX_ROWS = 8` (sized everywhere); V(12) ≈ 87, V(16) ≈ 106 ms | Exact | Code-like streams: ~35% of rounds gaining ~3 tokens for ~4 extra rows gives **+8-12%**. Copy spans at q = 0.98: 108 → 130 tok/s (**+20%**) | 2-4 d |
| L3 | **Routed experts at 3.0-3.5 bpw** (parked) | EXPERIMENTS Q1, PARADIGMS #6 | Quality gate (MMLU, refusals, tool calls, top-1 vs 4.0), K = 3 / `mul1` kernels | Output-changing (new weights); drafted == serial holds | +5.5% (3.5) to **+12%** (3.0) on every cell; 9.5-19 GB a rank freed | 7.5 d |
| L4 | **Weight-major persistent MoE decode kernel** (MonoMoE-style): CTAs over expert-weight tiles, the whole window's rows on the MMA N dimension, router / top-k / gate-up / act / down / combine in one launch, row-invariant K order | MonoMoE ([2609.04244](https://arxiv.org/abs/2609.04244), FlashInfer): token-major grouped GEMMs sustain only 17-22% of DRAM peak at small batch; routed MoE up to 1.54x faster than vLLM's Triton grouped GEMM, up to -18.7% TPOT end to end. Ours: routed experts are 26.4 ms (1 stream) and **72.3 ms (4 streams)** a round; 4 streams add ~46 ms of expert reads | First measure the grouped EXL3 decode kernel's effective GB/s at 4-20 rows (distinct experts x 6.3 MB a rank / kernel ms). Then a weight-major kernel that keeps each row's accumulation order fixed (row-invariant, so exact), with the trellis decode per weight tile shared by all rows | Exact if row-invariant (the 0170 test pattern) | Single stream **+3-6%** if the kernel is at ~200 of ~230 GB/s. 4 streams **+8-15%** if the many-experts-few-rows regime is as far below peak as MonoMoE found on H200 | 6-9 d |
| L5 | Fused-round decode: device sampler, MTP chain in one graph | PARADIGMS #7, DECODE-ANALYSIS §3a | Merges with N1b | Exact (the new sampler becomes the reference) | +2-4% beyond N1b | 5-7 d |
| L6 | Context-aware draft depth (V(R) per context bucket) | PARADIGMS #5 | Calibration at 2k / 64k / 200k | Exact | +3-10% past 100k | 1-1.5 d |
| L7 | Chain-only expert-aware row pricing (EcoSpec-lite): price a draft row by its predicted *new* experts | EcoSpec [2607.12696](https://arxiv.org/abs/2607.12696) | A cheap router predictor from the drafter's hidden state | Exact | ≤ +2-4% | 3-4 d |

## 5. Fastest known GLM-5.3-Flash on 2 x Spark (for calibration)

| kit | prefill tok/s | decode prose / structured (1 stream) | 4 streams | notes |
| --- | --- | --- | --- | --- |
| **this repo (b2, 2026-09-28)** | 1,447-1,473 (24.5k-98k) | ~40-60 (chat/code) / ~86-96 | 72-78 | exact; FP8 latent KV; 1M pool |
| [MiaAI-Lab kit](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) | 1,492-1,554 (8k-256k) | 37.1 / 62.9 | prose 75.3, structured 146.5 | vLLM, DFlash2 k=7, EMA adaptive k, E3 grouped MoE, opt-in FP8 dense |
| [Reederey87 kit](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark) | 1,408-1,454 (60k-240k) | ~33 / ~74 | 63-66 warm | vLLM, DFlash2 k=7, fat-expert kernels |
| single Spark ([gitcommit90](https://github.com/gitcommit90/glm-5.3-one-spark), 2.05 bpw) | 786-846 | 29.9 / 64 | C4 181.9 | quality loss (KLD 0.12) |

- Prefill is now at parity (0320 + 0335).
- Single-stream decode leads by 1.2-1.6x.
- The one cell where a kit leads is **4-stream structured** (MiaAI 146.5 aggregate). That is the DFlash2 k=7 ceiling
  multiplied across streams, and it argues for L2 (deeper windows) and L4 (the multi-row expert kernel) over
  anything else for concurrency.

## 6. GB10 facts used above

- 20 Arm cores = 10 Cortex-X925 (2 MB L2 each, 16 MB L3) + 10 Cortex-A725 (512 KB L2, 8 MB L3) (Porting Guide
  §1.1.1). This makes host-thread placement a real variable (N1a).
- GPUDirect RDMA unsupported; `cudaHostAlloc` + `ibv_reg_mr` is the sanctioned path (§4.6.1).
- DRAM 273 GB/s nominal; ~231-234 GB/s attainable (PARADIGMS §0).
- FP4 `mma` exists on sm_121a, but software support is thin (see the NVFP4 row).
- The clock-clamp / slow-state issue is covered by PARADIGMS #3 (operational). Pinning (N1a) does not touch it.

## Sources

- SuffixDecoding: [arXiv 2411.04975](https://arxiv.org/abs/2411.04975), [project page](https://suffix-decoding.github.io/), [Snowflake / Arctic Inference](https://www.snowflake.com/en/engineering-blog/suffixdecoding-arctic-inference-vllm/)
- AgentSpec (batch agent speculation): [arXiv 2608.24004](https://arxiv.org/abs/2608.24004)
- DFlash / Spec V2: [arXiv 2602.06036](https://arxiv.org/abs/2602.06036), [LMSYS 2026-06-15](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/); DFlare [2606.02091](https://arxiv.org/abs/2606.02091); DBloom / ceiling clipping [2608.30427](https://arxiv.org/abs/2608.30427); AdaFlash [2607.19223](https://arxiv.org/abs/2607.19223)
- FastMTP: [arXiv 2509.18362](https://arxiv.org/abs/2509.18362), [Red Hat Developer 2026-09-08](https://developers.redhat.com/articles/2026/09/08/optimize-vllm-speculative-decoding-fastmtp-heads); GLM-5 MTP: [arXiv 2602.15763](https://arxiv.org/abs/2602.15763); DeepSeek-V3 MTP 85-90% second-token acceptance: [arXiv 2412.19437](https://arxiv.org/abs/2412.19437)
- MoE speculation: EcoSpec [2607.12696](https://arxiv.org/abs/2607.12696), Limits of Speculation [2609.22156](https://arxiv.org/abs/2609.22156), EVICT [2605.00342](https://arxiv.org/abs/2605.00342), AcceptMoE [2608.02989](https://arxiv.org/abs/2608.02989), MoE-Spec [2602.16052](https://arxiv.org/abs/2602.16052), MoESD [2505.19645](https://arxiv.org/abs/2505.19645), DraftExpert [2607.24434](https://arxiv.org/abs/2607.24434), S2-MoE [2608.15018](https://arxiv.org/abs/2608.15018), MoE-SpeQ [2511.14102](https://arxiv.org/abs/2511.14102)
- Linear-attention trees: STree [2505.14969](https://arxiv.org/abs/2505.14969), TreeWY [2608.20961](https://arxiv.org/abs/2608.20961), Bole [2608.01651](https://arxiv.org/abs/2608.01651), SpecLA [2607.16673](https://arxiv.org/abs/2607.16673)
- Lossless compression: DFloat11 [2504.11651](https://arxiv.org/abs/2504.11651) / [code](https://github.com/LeanModels/DFloat11), Huff-LLM [2502.00922](https://arxiv.org/abs/2502.00922), EntroLLM [2505.02380](https://arxiv.org/abs/2505.02380), Shannon-bound ANS [2606.15789](https://arxiv.org/abs/2606.15789)
- MoE decode kernels / megakernels: MonoMoE [2609.04244](https://arxiv.org/abs/2609.04244), [Mirage roadmap #773](https://github.com/mirage-project/mirage/issues/773)
- KDA / DSA kernels (prefill; no decode lever found): [MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA), Kimi Linear [2510.26692](https://arxiv.org/abs/2510.26692), [FlashMLA](https://github.com/deepseek-ai/FlashMLA), [vLLM DeepSeek-V3.2 blog](https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html)
- DGX Spark: [Porting Guide (PDF)](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/dgx-spark-porting-guide.pdf), [GPUDirect RDMA answer](https://nvidia.custhelp.com/app/answers/detail/a_id/5780/~/is-gpudirect-rdma-supported-on-dgx-spark), [NCCL small-message latency between two Sparks](https://contact.alessandrosangiorgi.net/posts/dgx-spark-nccl-collective-latency/), [b12x](https://github.com/local-inference-lab/b12x) and [issue #313](https://github.com/local-inference-lab/b12x/issues/313), [vLLM #40082](https://github.com/vllm-project/vllm/pull/40082), [FlashInfer #3013](https://github.com/flashinfer-ai/flashinfer/issues/3013), [SM121 CUTLASS results](https://forums.developer.nvidia.com/t/sm121-cutlass-kernel-optimization-results-nvfp4-356-tflops-moe-grouped-gemm-on-dgx-spark/359960)
- Kits: [MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks), [Reederey87](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark), [gitcommit90 one-Spark](https://github.com/gitcommit90/glm-5.3-one-spark)
- Measurement inputs: `turboderp/Qwen3-30B-A3B-exl3` (branch 4.0bpw, `model-00002-of-00002.safetensors`, byte range of `layers.30.mlp.experts.5+.*.trellis`), `zai-org/GLM-5.3-Flash` (`model-00018-of-00062.safetensors`, `layers.20.self_attn.o_proj.weight`); histogram from `results/W1/conc.json`, `results/W5/conc-*.json`, `results/W6/conc-A.json`, `results/W8/conc-*.json`
