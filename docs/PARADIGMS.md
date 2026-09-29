# Paradigm shifts for GLM-5.3-Flash on 2x DGX Spark: ranked

> Research note, 2026-09-28. Web research plus our own docs (RESULTS, EXPERIMENTS, PREFILL-/DECODE-/COMM-ANALYSIS,
> MEMORY-4x256k, MIA-AUDIT, PATCHES). No GPU was used, so every gain below is **arithmetic**, not a measurement. "Exact"
> means drafted == serial and resumed == fresh still hold, with the same weights. "Output-changing" means greedy replies
> move, the way FP8 prefill (rejected) and FP8 KV (accepted) did.
>
> **Already in progress, not repeated here:** 0230 (b12x RoCEnante one-shot gather) and 0240 (b12x prefill kernels:
> fused mHC norm, `kda_prefill`, `dsa_indexer`, `sparse_mla`). They show up in the table only as dependencies.

## 0. Baseline and cost model

- Production today (4 x 262k, FP8 latent KV): decode 41-75 tok/s single stream (kit structured ~96); 4-stream aggregate
  72-77 tok/s; prefill 1,154 (24.5k) / 1,127 (98k) tok/s alone; 1,266 / 1,238 at 28k / 112k on the single-stream image.
- Decode verify window V(R) = 30.0, 40.0, 44.7, 49.5, 54.2, 58.9, 63.7, 68.4 ms (R = 1..8), then about +4.7 ms a row.
  The weights are ~5.0 GB a rank a token. At the DRAM bandwidth people actually reach on GB10 (231-234 GB/s,
  STREAM-class), the floor is ~21.5 ms. With KV and the replicated reads it is ~24.6 ms (the agents' figure), so the
  realistic engineering headroom on a one-row step is ~5 ms (-17%), not 9 ms.
- Prefill at 28k: ~790 us a token (both ranks in lockstep). The shares are experts ~26% (~205 us), sparse attention 13%,
  hc 13%, KDA proj/chain + DSA proj ~30%, router/combine/shared ~13%. Per rank that is 18 GFLOP a token x ~1,200 tok/s,
  about **22 TFLOP/s**, against a measured GB10 dense peak of **99.8 TF BF16, 188-208 TF FP8 and ~356 TF NVFP4**
  (CUTLASS microbenchmarks). **Prefill is not tensor-core-bound.** The experts are limited by EXL3 trellis-decode ALU
  work (~15 TF/s in `exl3_fast.cu`, PATCHES 0083), and the rest by memory, gathers and small kernels.
- GB10 facts that decide several rows below:
  - FP4 `mma.sync` exists (`kind::mxf4nvf4.block_scale`, m16n8k64). It needs the arch-specific `sm_121a` / family
    `sm_120f` target. There is no tcgen05/TMEM.
  - vLLM on GB10 runs NVFP4 MoE through **Marlin** (weight-only FP4 dequantized to BF16 MMA), because its CUTLASS/FlashInfer
    FP4 MoE paths are gated to sm_100. That is the "target-gated rather than silicon-absent" point in the Reederey87 docs:
    the silicon has the instructions, and the software builds don't target them.
  - **There is no GPUDirect RDMA on Spark**, so NVSHMEM/IBGDA/DeepEP-style GPU-initiated comms are out.
  - GB10 has a hidden slow state and clock clamps (e.g. the 507 MHz stuck head node) that swing throughput by
    ±30-70%.

## 1. Ranked table

Rank = expected gain for our workload (agent sessions: long prompts, tool calls, edits, 1-4 streams) / effort, with
exact items preferred. Effort is in engineer-days.

| # | Idea | Evidence | Expected gain for us (arithmetic) | Effort | Risk / exactness | Cheap first experiment |
| ---: | --- | --- | --- | ---: | --- | --- |
| 1 | **Tiered session store on NVMe** (spill the 0110/0180 store to local NVMe, and persist it across restarts). *Written offline as patches/0250 (`GLM53_TF_SESSION_DISK`); GPU tests pending* | Mia [#232](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/232): 46.8k-token prompt, cold TTFT 32.4 s, **1.58 s after a restart** (4.2 GB a rank written). Our batch production has only a **2 GiB** RAM store, and `slots` measured a session evicted by a 5th coming back as a **42 s cold prefill** | A 40k session is 40k x 7.6 KB (FP8 latent) + 74 MB of KDA/conv snapshot ≈ **0.38 GB a rank**, i.e. ~0.1-0.2 s at 2-4 GB/s NVMe read, against 35 s of re-prefill at 1,150 tok/s. For agent traffic with more than 4 live sessions (subagents, several users) this removes most cold prefills, a **~100x TTFT cut on the evicted-session case**. After a watchdog heal, sessions come back in ~1-2 s instead of minutes. It also lets the RAM store shrink to 1 GiB (memory headroom on the worker node) | 2-3 | **Exact** (copies bits). Risks: file format and invalidation (key on the 0140 image id + the knobs that tag snapshots, `fp8pf.tag`, KV dtype); NVMe wear (~0.4 GB a session write; fine); rank 1 must read the same file set (rank 0's plan already names the entry). Mia's code is AGPL: re-implement from the idea only | `dd`/`fio` sequential read and write on both nodes' NVMe with O_DIRECT (expect 3-6 GB/s), then time `torch.save`/`load` of a real 0110 entry (~0.4 GB). Go if a 40k entry restores in < 0.5 s |
| 2 | **Global suffix-tree drafting** (SuffixDecoding-style: 0020's prompt lookup grows into a suffix tree over *all* earlier prompts and replies of every session, frequency-scored, with deep windows of 12-16 rows) | SuffixDecoding ([arXiv 2411.04975](https://arxiv.org/abs/2411.04975); Arctic/vLLM integration): up to **5.3x** on agentic (SWE-bench / AgenticSQL-style) workloads, ~0 on open chat. Our 0020 (current request only, windows ≤ 8) measured +9-12% on edit cells | EXPERIMENTS S1 arithmetic with our V(R): at per-token acceptance q = 0.95, 7 drafts give 98 tok/s, 11 give 105, 15 give 106; at q = 0.98: 108 / 124 / 130. Today edit cells run ~82-92. Agent output that repeats across turns (the same file re-emitted, the same tool-call scaffolding, JSON keys, paths) is exactly what a cross-request tree catches and a per-request lookup misses. Blended over agent turns with a copy fraction of 0.3-0.6: **+15-40% decode on agent turns**, ~0 on prose. Costs no draft time (tree lookup on the host, < 0.1 ms) | 4-5 (graphs + buffers for 12/16-row windows are most of it) | **Exact** (drafts only propose). Risks: 12/16-row buffers and graphs cost memory (the 8-row buffers are sized everywhere); with batching, deep windows compete with other slots' rows (cap depth by slot count) | Offline, no GPU: replay the production logs of the last days (opencode / agent traffic). At each output token, measure the suffix-tree proposal's accepted length against the real continuation, with the tree built from all earlier traffic. Report copy fraction and E[T] per round. Build if E[T] ≥ 4 on ≥ 25% of agent-turn tokens |
| 3 | **Clock and slow-state control on GB10** (detect and escape the hidden slow state; pin clocks and power mode; gate on a per-boot throughput probe) | Agent report: GB10's hidden slow state and clock clamps give **±30-70%** throughput swings. Reederey87 docs: the head GPU stuck at 507 MHz until a cold power cycle. MIA-AUDIT item 8 | Not a speedup of the good case. It removes the bad case: a node in the slow state makes the pair run at the slower node's rate (TP lockstep), so -30-70% on everything. If that happens even 5% of the time, it costs more than most kernel projects gain | 0.5-1 | None (operational). Watch for power-cap interactions | Log `nvidia-smi -q -d CLOCK,PERFORMANCE,POWER` on both nodes every minute for a few days next to `/metrics` tok/s. Then set `CANARY_MIN_TPS` from a healthy run and add a clock-reason check to `serve.sh watch` (alert only) |
| 4 | **Bit-identical EXL3 expert kernel that decodes each trellis tile once per chunk** (decode a 16x16 tile once, reuse it for all ~230 members an expert has at 8,192-row chunks, instead of MG = 2 member tiles) | PATCHES 0083: `exl3_fast.cu` runs at ~15 TF/s, "bound by trellis decode ALU and the slab barrier (43% of stalls), not the mma rate". GB10 BF16 dense is 99.8 TF measured. The Mia/Reederey E3 fat MoE reached 1,428-1,587 tok/s prefill with 3 launches a layer | Experts ≈ 205 us of 790 a token at 28k. Decoding each weight tile once per chunk and staging it in shared memory (or bf16 into L2-resident scratch) moves the kernel toward MMA-bound: 2-3x on the expert kernel, so -100 to -135 us a token, i.e. **1,266 -> ~1,520-1,600 tok/s at 28k (+20-27%)**. Decode unaffected | 5-7 | **Exact if built as 0170 was**: the same decoded fp16 B values and the same ascending m16n8k16 chain per output, with only data movement changed. Risk: shared-memory budget on sm_121 (101 KB; `hc_fused` already hit it) | Nsight Compute on the current `fat` gate/up kernel at 8,192 rows: ALU vs MMA vs memory stall split. Then a standalone microbench: decode one expert's tiles once into shared memory and run 256 member rows through them. Go if ≥ 1.8x on the kernel |
| 5 | **Long-context-aware draft depth** (depth and arm chosen by position and context, not only by acceptance; cheaper rounds past ~100k) | Agent report: other kits found **k ≈ 3 beats k = 7 on long agentic prompts**. Our decode after a 112k prompt was 56-81 tok/s depending on knobs. Verify rows get dearer with context (sparse attention and the indexer per row) | The 0071 cost rule already prices rows, but with a V(R) calibrated at short context. If V(R) at 200k has a steeper per-row slope (e.g. 6-7 ms instead of 4.7), depth 7 over-drafts. Re-pricing per context bucket: **+3-10% decode past 100k**, 0 below | 1-1.5 | **Exact** (depth only). Low risk | Calibrate V(1..8) at 2k / 64k / 200k context on real-token windows (EXPERIMENTS D1) and compare the per-row slope. Then A/B `fc3` vs `fc7` at 200k on the kit cells |
| 6 | **Routed experts at ~3.0-3.5 bpw (EXL3)**, parked, still the only large decode lever | EXPERIMENTS Q1. `satgeze/...-TR3-3.5bpw` KLD 0.0297 vs 0.0246 at 4.0; `MikeRoz/...-Uncensored-3.05bpw-h6-exl3` (our abliteration, codebook `mul1`) | 3.0 bpw: -2.3 ms on a 1-row window and -1.2 ms a later row, so **+12% decode** on a 4-row round (EXPERIMENTS), and 19 GB a rank freed (more slots, a bigger store, or bf16 KV back). 3.5 bpw: +5.5%, 9.5 GB freed | 7.5 | New weights: output-changing, quality gate needed. Drafted == serial holds. Kernel work: K = 3 trellis + `mul1` decode | Serve MikeRoz 3.05 in the vLLM kit for 1 day of MMLU-200 / refusals / tool-call harness / 50k-token top-1 agreement vs 4.05 |
| 7 | **Fused-round decode: device-side sampling + one graph per round** (includes the O1 fusions not covered by 0240's fused hc) | DECODE-ANALYSIS: host bubbles ~1.1 ms of a 55 ms MTP round, d + 2 hard syncs a round; ~1,500 kernels a window at 1-2 us of gap. b12x's fused `norm.mhc` (0240) covers part of the hc cost | -1 to -2.5 ms a round, **+3-6% decode**; with batching the host work grows with slots, so +5-10% aggregate at 4 streams | 5-7 | Exact, with the device sampler as the new reference (serial decoding uses it too) | nsys of one 4-row round at 28k: count the GPU-idle gaps between kernels and around the syncs. Go if > 2 ms |
| 8 | **Better drafters (licence-clean): DSpark confidence head, DFlash2-G** | [RedHatAI/GLM-5.3-Flash-speculator.dspark-preview](https://huggingface.co/RedHatAI/GLM-5.3-Flash-speculator.dspark-preview) (MIT, ~3 B, confidence head, τ 3.77); [canada-quant/GLM-5.3-Flash-DFlash2-G](https://huggingface.co/canada-quant/GLM-5.3-Flash-DFlash2-G) (Apache-2.0, 8 layers, 9 taps, τ 3.68) vs incoai 3.63 (CC BY-NC-ND). No EAGLE-3 exists for 5.3 | τ +0.05-0.14 (+1.4-3.9% tokens a round). The DSpark block costs ~5-6 ms against DFlash2's 3.9: at ~65 ms rounds -2-3%, so net **-1 to +2%**. The confidence head is the real value: per-position stop probabilities for 0071. For a public image, DFlash2-G removes the NC-ND blocker at ~0% speed | 3-4 | Exact. Unknown acceptance on the abliterated taps (deep-layer drift) | Teacher-forced acceptance of both on 2k tokens of our model's own replies (taps from a prefill), offline on one GPU window |
| 9 | **Fine-tune the MTP layer (or a small EAGLE-3-style head) on the abliterated model's own outputs** | EXPERIMENTS S5 / S5'. MTP chained acceptance decays a1 0.74 -> a3 0.22; training-time-test recipes (EAGLE-3 / SpecForge) lift deep positions | a3 0.22 -> 0.35 gives E[T] +0.13, about **+5% on sampled/chat cells** (the MTP-heavy ones). A full drafter: +8-15% on non-copy text | 10-16 | Exact. Data generation dominates (teacher-forced taps at prefill speed: 10M tokens ≈ 2.5 h at 1,150 tok/s) | Dump 1M tokens of taps + MTP inputs from production-like prompts. Train a LoRA on MTP attention + eh_proj for 1 epoch on one Spark. Measure chained a2/a3 teacher-forced |
| 10 | **NVFP4 / FP4 tensor-core path** (W4A4 NVFP4 experts and/or non-experts) | Silicon: yes (sm_121a `mxf4nvf4`, ~356 TF measured). Checkpoints: nvidia / dealignai / orcarouter / LibertAI NVFP4 builds of GLM-5.3-Flash, **KLD ~0.06-0.073 vs 0.0246 for our EXL3 4.0** (Reederey87 docs/01). Best native-FP4 GLM prefill: 1,279-1,384 tok/s on **4** Sparks (shankinsonhf), 1,997 cold on 4 (tonyd2wild); alexellis TP4 2,276 cold at 64k with Marlin (weight-only) MoE. vLLM on GB10 runs Marlin, not FP4 MMA | Decode: NVFP4 is 4.5 bpw, experts **+12.5% bytes** vs EXL3 4.0, so -4-5% decode. Prefill: we run at ~22 TF/s against 99.8 TF BF16, so FP4 MMA speeds up a part that does not bind. The real prefill win from dropping EXL3 is the cheaper dequant (item 4 gets it exactly). Net: **decode -5%, prefill maybe +10-20%, quality 2.5-3x worse KLD** | 15-25 | Output-changing (new weights, and W4A4 quantizes activations: stricter than the rejected FP8 prefill). Not recommended | None needed. Revisit only if a 4-bit NVFP4 build with KLD ≤ 0.03 appears |
| 11 | **Batching refinements for 4 x 256k**: prefill/decode stall and depth taper | Ours: a 35-40k prefill beside 3 decoders gives **2.1-2.9 s decode gaps and 45-50 s TTFT**. Mia #231: taper k by batch width, c8 aggregate 75.3 -> 91.9. EXPERIMENTS B1 model: 4 streams ideal ~82 | Smaller prefill pieces while decoders are live (e.g. 512-row pieces, a time-share per round) cut the gap to < 0.5 s at ~5-10% cost to the prefill. Depth taper: +0-15% aggregate at 3-4 streams | 2-3 | Exact (scheduling only) | Rerun `multiturn.py --modes stall` with `BATCH_PREFILL_SHARE` 0.25 / 0.5 and `PREFILL_ROWS_MAX` 1024 vs 2048; log per-slot depth |
| 12 | **Persistent megakernel for the decode step** (Mirage MPK / Hazy-style, layer-group persistent kernel) | Hazy/MPK report 1.2-1.7x on batch-1 decode for dense models on H100/B200; nothing ships for sm_120/121 MoE + linear attention + mHC | Would close most of the ~5 ms between our 30 ms and the ~24.6 ms floor: **+10-18% decode** at best. 0230 (gather latency) and 0240 (fused hc) take part of the same gap first, which makes the rest smaller | 25+ | Exact if row-invariant; very high effort and upstream divergence | Only after 0230 + 0240: nsys the one-row step again. If the non-bandwidth gap is still > 3 ms, prototype one layer group (KDA block + MoE) as one persistent kernel |
| 13 | **KV compression beyond FP8** (4-bit latent, FP8 indexer keys) | FP8 latent (0220) already 13.6 -> 7.7 KB a token a rank; KIVI/KVQuant-style 4-bit KV on MLA latents is lightly studied | Memory only: 4 x 262k slots 7.45 -> ~4 GiB a node (+3.4 GiB headroom, or ~1 more slot). No speed gain (the sparse read is 22 MB a window) | 4-6 | Output-changing (more than FP8). Low value while NVMe tiering (item 1) solves the store pressure | Not recommended now |
| 14 | **Two single-Spark replicas instead of TP=2** (data parallel at ~2 bpw) | [gitcommit90/glm-5.3-one-spark](https://github.com/gitcommit90/glm-5.3-one-spark): EXL3 2.05 bpw, 64 tok/s structured, ~30 prose, prefill 786-846 tok/s, C4 aggregate ~182; [forum thread](https://forums.developer.nvidia.com/t/60-tok-s-glm-5-3-flash-on-a-single-dgx-spark/382140) reports crashes past ~10k tokens | Aggregate could double for many short streams, but **KLD 0.1216 / top-1 88.9%** (5x worse than ours), prefill -30%, and single-stream prose slower | - | Output-changing, large quality loss | Rejected |
| 15 | **EP / PP / prefill-decode disaggregation / GPU-initiated comms across the two Sparks** | No GPUDirect RDMA on Spark, so no NVSHMEM/IBGDA/DeepEP. EXPERIMENTS §3: EP busier rank 5.09 of 8 experts (+27% at R = 1); PP 42 ms single-stream | Negative for one stream. Disaggregation needs each node to hold the whole model (≥ 2 bpw, as in item 14) | - | - | Rejected. The comm-latency lever is 0230's host-staged one-shot gather (all-gather 10.5-12.5 us vs NCCL ~40-60 us), already in progress |

Items 0230/0240 (in progress) would, by the agents' figures, give: all-gather latency -1.4 to -2 ms a decode step
(**+5-7% decode**, stacking with 7); KDA prefill 2.2-2.55x on `kda.chain` + fused mHC + indexer/sparse-MLA kernels
(**~+10-20% prefill**, which stacks with item 4 because it touches the non-expert 74%).

## 2. Notes behind the ranking

- **Why NVFP4 ranks low despite 356 TF.** Our prefill does ~22 TF/s a GPU. Tensor-core peak is not the limit. Trellis
  decode in the expert kernel is (item 4), and so are memory, gathers and hc for the rest (0230/0240). FP4 decode
  bytes are 4.5 bpw, worse than EXL3's 4.0. The published NVFP4 checkpoints are 2.5-3x worse in KLD than our weights.
  W4A4 also changes activations, which is a larger behaviour change than the FP8 prefill already rejected.
  The one scenario that changes this: a GB10 FP4 grouped-GEMM MoE that makes a 4-bit checkpoint with KLD ≤ 0.03 run
  prefill ≥ 2x, and that is not available anywhere today (vLLM falls back to Marlin on sm_121).
- **Fastest known GLM-5.3-Flash on 2 Sparks:**
  - MiaAI-Lab kit: prefill **1,492-1,587 tok/s**, decode prose ~32.
  - Reederey87 kit: 1,408 tok/s at 240k, prose 29-32 / structured 69-70, 4 agents 63-66 aggregate.
  - Ours: decode 41-75 (structured ~96), 4-stream 72-77, prefill 1,154-1,266.
  - We lead on decode and concurrency, and trail on prefill by ~20-25%. Items 4 + 0240 are the path to parity or better.
  - On 4 Sparks: TP4 NVFP4 gets 1,997-2,276 tok/s cold prefill, 71-75 code / ~30 prose decode.
- **Speculation.** Trees stay rejected: KDA recurrence plus the ~4.7 ms MoE row cost (DECODE-ANALYSIS §3b; EcoSpec
  and others find little for MoE + linear attention). The gain left is (a) free drafts on repeated text (item 2),
  (b) depth pricing by context (item 5), and (c) deeper acceptance from training (item 9). DSpark / EVICT / LibraSpec
  cost-aware truncation is what 0071 already does.
- **Decode floor.** With 231-234 GB/s attainable, the honest floor is ~24.6 ms (with KV and replicated reads), so
  engineering alone (0230 + 7 + 12) caps at ~+20%. Anything beyond that takes fewer bytes (item 6) or more tokens a
  round (items 2, 5, 9).

## 3. Top-5 program

1. **Week 1: NVMe session tier (item 1) + clock/slow-state watch (item 3).** 3-4 d, both exact. Success: an evicted
   40k session resumes in ≤ 1 s (from 42 s); sessions survive a restart; a slow-state node is detected and alerted
   within a minute; the RAM store can drop to 1 GiB with no rise in cold prefills on replayed agent traffic.
2. **Week 1-2 (offline first): suffix-tree drafting (item 2).** A 0.5 d log replay decides it; 4-5 d to build with
   12/16-row windows. Success: ≥ 1.3x decode on agent edit/tool turns, no `tf`/`kit` cell down more than 2%,
   `exact` 10/10.
3. **Week 2-3: single-decode EXL3 expert kernel (item 4)**, alongside the 0240 port. Success: expert kernel ≥ 1.8x at
   8,192 rows, bit-identical to `fat` (the 0170 test pattern), prefill ≥ 1,500 tok/s at 28k with 0240 → at or
   above the MiaAI kit.
4. **Week 3: context-aware draft depth (item 5) + batching stall fix (item 11).** 3-4 d, exact. Success: decode after a
   200k prompt +5% or more; decode gaps during a concurrent 40k prefill ≤ 0.5 s; 4-stream aggregate ≥ 80 tok/s.
5. **Week 4+: the 3.x-bpw expert gate (item 6)**, only if it is unparked: 1 d quality check with MikeRoz 3.05
   in the vLLM kit, then 7 d of K = 3 / `mul1` kernel work. Success: one-row verify ≤ 27 ms, chat greedy ≥ 48,
   MMLU ≥ 87%, top-1 ≥ 97% vs 4.0, tool-call harness unchanged. The fused-round decode (7) and a trained MTP (9) come
   after, if decode on prose remains the complaint.

Compounded estimate after 1-4 + 0230/0240 (all exact, same weights):

- Prefill: 1,266 -> ~1,600-1,800 tok/s at 28k.
- Decode: prose +5-7%, agent edit/tool turns +20-40%.
- Evicted and restarted sessions: 35-45 s -> ~1 s TTFT.
- 4-stream aggregate: ~80-85 tok/s.

With item 6 (new weights) add ~+12% decode on every cell.


> **Correction (0260 analysis, 2026-09-28):** the "expert kernel decodes each tile once per chunk" premise does not hold for the production kernels — fast2/fat already decode each trellis tile once per 64-member pass (≈1.0-1.15 decodes per tile at 1024-2048-row chunks). Patch 0260 (`GLM53_TF_FAST_EXPERTS=once`) reaches ~1.0 at large chunks, but the expected gain is only ~0-2.5% prefill, not 1,266→1,520-1,600 tok/s. Its bench_experts.py "no-decode" probe decides whether trellis decode is the expert bottleneck at all.
