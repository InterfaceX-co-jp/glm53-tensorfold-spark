# Architecture and paradigm experiments: GLM-5.3-Flash on TensorFold, 2x DGX Spark

A ranked, quantified backlog. Written offline on 2026-09-27 from `vendor/TensorFold` @ `2f8e514` (the
`glm5_next/cuda` engine), patches 0001-0010 and 0040, the sibling analyses (`PREFILL-ANALYSIS.md`,
`DECODE-ANALYSIS.md`, `COMM-ANALYSIS.md`), and the Hugging Face API. Nothing ran on the Sparks. A number is
**measured** only where it says so. Everything else is arithmetic from bytes, FLOPs and the measured costs below,
and each item names the cheap experiment that would confirm or kill it.

## 0. Summary

| Rank | ID | Experiment | Expected gain | Effort | Exact? |
| ---: | --- | --- | --- | ---: | --- |
| 1 | P1 | Land 0005/0006, sweep prefill chunks to 1024/2048, skip the head on non-final chunks | prefill 420 -> ~800-1,100 tok/s | 1.5 d | yes (bit-identical) |
| 2 | D1 | Decode profile: 1->2-row jump, eager steps past 2,051 tokens, calibration with real tokens | finds 3-8 ms a window (enabler) | 1.5 d | n/a |
| 3 | S1 | Suffix / prompt-lookup drafts as a third `auto` arm, windows up to 16 rows for that arm | +10-40% on agent edit output, ~0 on prose | 3.5 d | yes (drafts only propose) |
| 4 | P2 | Canonical-chunk fast prefill (tensor-core GEMMs that need not be row-invariant, overlapped gathers) | prefill -> 1,500-2,000 tok/s | 12 d | yes, under the canonical-chunk rule (section 7) |
| 5 | M1+M2 | Latent (absorbed) MLA cache, then a multi-session snapshot cache | KV 21x smaller (1M context fits); +8-12% decode past 2k; session-switch TTFT 70 s -> <3 s at 30k | 7 + 5 d | yes (new kernels, still row-invariant) |
| 6 | L1 | Long-context decode path: graphs past 2,051 tokens, indexer bounded by visible pools, not capacity | +10-25% decode past 2k (if D1 confirms) | 4.5 d | yes |
| 7 | O2 | Measure and keep patch 0040 (L2 prefetch in all-gather gaps) | -0.5 to -1.6 ms a window (+2-5%) | 1 d | yes |
| 8 | S2 | Cost-aware draft depth from real-token V(R); a 4th sampled MTP draft | +2-4% | 1.5 d | yes |
| 9 | Q1 | Routed experts at 3.5 or 3.0 bpw (mcg), requantized from `orcarouter/...-Uncensored-FP8` | +6-15% decode on every cell; frees 10-19 GB a rank | 7.5 d | new weights; drafted == serial holds |
| 10 | B1 | Continuous batching of 2-8 sequences in one window | aggregate 33 -> 52/74/93 tok/s serial at B = 2/4/8 | 17 d | yes (rows independent) |
| 11 | S3 | Frequency-ranked draft-only head | +1-3% (MTP-heavy cells) | 1.5 d | yes |
| 12 | S4 | DFlash2 `fc` ridge recalibration to the abliterated taps (private use only) | +1-4% | 2 d | yes |
| 13 | Q2 | Non-experts in EXL3 trellis (4.0 bpw, then 3.x) instead of affine q4 g64 (4.5 bpw) | +2.5% (4.0, better quality) to +7% (3.0) | 6 d | new weights |
| 14 | O1 | Kernel fusion plus concurrent graph branches | -1.5 to -2.5 ms a window (+4-7%) | 7 d | yes |
| 15 | O3 | GPU-initiated all-gather (NVSHMEM/IBGDA or RDMA write + flag) | -1.4 to -1.8 ms a window (+4-6%) | 7.5 d | yes |
| 16 | S6 | Evaluate alternative drafters (RedHat DSpark, canada-quant DFlash2-G) | -5% to +8% | 3.5 d | yes |
| 17 | S5 | Train a drafter / fine-tune MTP on the abliterated model's own outputs | +8-15% on non-copy text | 16 d | yes |
| 18 | O5 | Persistent megakernel per layer group | +15-25% (subsumes O1-O3) | 25+ d | yes if row-invariant |
| - | - | Rejected: expert or pipeline parallelism, verify trees, bf16 decode partials, token-dropping prefill, decode-side chunked KDA | section 9 | | |

Rank is gain / effort. The weights are 1.0 for decode on every cell, 0.8 for agent-only decode, 0.6 for
prefill/TTFT and 0.3 for multi-stream aggregate throughput, with dependencies respected. Section 10 has the scores
and section 11 the recommended program.

**Projected ladder (chat greedy / code greedy / agent edits / prefill).** These are estimates that compound the
midpoints above.

| After | chat greedy | code greedy | agent edit output | prefill tok/s | context |
| --- | ---: | ---: | ---: | ---: | --- |
| today (measured) | 44 | 61 | ~55-60 (est.) | 420 | ~100k tokens a rank of free KV |
| P1 + D1 + S1 + O2 | 46 | 64 | 75-90 | 800-1,100 | same |
| + P2 + M1/M2 + L1 | 48 | 66 | 80-95 | 1,500-2,000 | 1M; many sessions cached |
| + Q1 (3.0) + Q2 + O1/O3 | 55-60 | 75-82 | 95-115 | 1,700-2,200 | 1M |

## 1. Cost model used throughout

- **Verify window.** V(R) = 30.0, 40.0, 44.7, 49.5, 54.2, 58.9, 63.7, 68.4 ms for R = 1..8 (measured,
  calibrated at load). Beyond 8 rows extrapolate +4.7 ms a row: V(12) ≈ 87, V(16) ≈ 106. The 2nd row costs 10 ms
  and each later row 4.7 ms, almost all of it the new experts that row routes to (~1.1 GB a rank a row).
- **Drafts.** An MTP draft costs 2.0 ms, +1.7 per chained draft, ~0.3-0.4 of which is a host bubble. A DFlash2
  block is 3.9 ms (up to 7 positions). A round's host tail is ~0.3 ms.
- **Tokens a round.** E[T] = 1 + Σ_k a_k, where a_k = P(drafts 1..k all accepted). The 1 is the verify row's
  own token. Rate = E[T] / (V(1 + d) + draft ms + 0.3).
  - Check (chat greedy, MTP c3): a = (0.74, 0.45, 0.22), so E[T] = 2.41. Round = 49.5 + 2.0 + 3.4 + 0.3 =
    55.2 ms, giving **43.7 tok/s**, against 44 measured.
- **Break-even for draft k.** a_k × (ms per committed token) > marginal row cost. At 44 tok/s (22.7 ms a token)
  a DFlash2 row (4.7 ms) needs a_k > 0.21, and an MTP row (4.7 + 1.7 ms) needs a_k > 0.28.
- **Bytes a rank a token.** ~5.0 GB: routed experts 2.11 (EXL3 4.0 bpw), shared experts 0.30, KDA projections
  1.46, DSA 0.39, dense MLP 0.13, head 0.18 (4.5 bpw affine q4mse for all of these), router + hyper-connection
  weights 0.17 (replicated), plus KV reads that grow with context.

## 2. Roofline gap: where 30 ms goes against the 18-21 ms floor

The 18 ms floor uses the nominal 273 GB/s. GB10 kernels attain ~240 GB/s at best (recipe), so the realistic
floor is 5.0 / 0.24 = **20.8 ms**. The breakdown of a one-row step comes from the upstream MLX checkpoint, whose
per-rank bytes are the same as ours:

| Component | Bytes a rank | Measured ms | Effective GB/s | Ideal at 240 GB/s | Target | Lever |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Routed + shared experts | 2.4-2.7 GB | 13.4 | ~200 | 10.0-11.2 | 11.5 | Q1 (bytes), kernel tuning |
| Dense 4-bit matmuls (KDA, DSA, MLP, head) | 2.16 GB | 9.7 | ~223 | 9.0 | 9.2 | Q2 (bytes) |
| Hyper-connections (90 pre + 90 post, 20 Sinkhorn steps) and KDA chains, router/HC weights 0.17 GB | 0.17 GB | 2.2 | - | 0.7 | 1.0 | O1 |
| 90 all-gathers of 16 KiB | 1.4 MiB | 2.4 | latency-bound (~27 us each) | 0.5-0.9 (5-10 us link) | 0.9 | O2, O3 |
| Router, select, combine, norms, embed, launch gaps | - | ~1.9 | - | ~0.3 | 0.8 | O1 |
| **Total** | **~5.0 GB** | **29.6-30.0** | 167 average | **20.8** | **~23.5-24.5** | |

So the gap is about 9 ms:

- ~2.3 ms of bandwidth kernels running below 240 GB/s;
- 2.4 ms of all-gather latency;
- ~2.2 ms of hyper-connection and KDA work;
- ~2 ms of small kernels and launch gaps. A layer runs about 30-36 kernels (hc_pre is 2, each qmm 1-2 with its
  K-split reduce, the EXL3 MoE 5, the shared expert 4, plus the router, select, combine, gathers and hc_post), so
  ~1,400-1,600 kernels a window. At 1-2 us of graph gap each that is 1.5-3 ms.

Engineering alone (O1, O2, O3) reaches about **24 ms** (-20%). Below that it takes fewer bytes (Q1, Q2).

**All-gathers: fewer, larger.** `COMM-ANALYSIS.md` shows 90 is the minimum for row-parallel TP on this model:
hc_post, the Sinkhorn mix and the norm are nonlinear in the reduced sum, so no two exchanges can merge exactly.
There are only two ways to have fewer and larger exchanges:

- **Batching (B1).** Two sequences share the same 90 exchanges.
- **Wider verify windows.** 8 rows cost 2.9-3.4 ms against 2.5 ms for 1 row.

Otherwise the lever is per-exchange latency:

- **O2 (patch 0040, written, unmeasured).** Prefetches the next weights into L2 during the ~27 us of each
  exchange. Upper bound 90 x 18 us = 1.6 ms; realistic 0.5-1.2 ms.
- **O3.** A GPU-initiated put + flag (no NCCL proxy thread) takes alpha from ~27 us to ~5-8 us: about -1.8 ms a
  window. The risk is that IBGDA / GPUDirect on GB10's ConnectX-7 is unverified. First test: `ib_write_lat` with
  16 KiB between the Sparks, then an NVSHMEM `put_signal` ping-pong. Go only if the one-way time is under 8 us.

**Overlapping comm with compute.** There is no independent compute to overlap within one sequence: the partial
feeds the gather, which feeds hc_post, which feeds hc_pre, which feeds everything else. There is idle DRAM, which
O2 uses. Real overlap exists in two places:

- in prefill: split a chunk into two row halves and gather half A while half B computes (rows are independent,
  so this is exact; section 7);
- with batching: sequence A's gather overlaps sequence B's compute, at the cost of two graph streams.

**CUDA-graphing whole rounds (O4).** Today each MTP draft does a hard sync, a numpy draw and a relaunch (d + 2
hard syncs a round), and DFlash2 rounds do 4. The host bubbles total ~1.1 ms of a 55 ms MTP round, so a
device-resident round (draw on the GPU, accept/commit with a device-side keep, chain the MTP drafts in one graph)
saves about **1-3%**.

- **Exactness note.** A GPU sampler does not have to reproduce numpy bit for bit. The contract is drafted ==
  serial on the same engine, so serial decoding switches to the same device sampler, and the sampler itself
  becomes the reference.
- Effort 4-5 d, low priority. The exception: batching (B1) needs it, because host work per round grows with B.

**Graph coverage is a larger lever than whole-round graphs.** `Engine.forward` replays a graph only while
`st.pos + R <= dense_limit` (2,051). Past that, every window and every MTP step runs eagerly: ~1,500 Triton
launches from Python plus `sparse.select_tokens` with host-side positions. Agent sessions are almost always past
2k tokens (the system prompt alone), so this is the default agent path. Three more costs grow past 2k:

- **Indexer cost follows the configured capacity, not the context.** `select_tokens` launches `_scores` over
  `capacity / 4` pools and runs two `torch.sort` over [R, capacity / 4] in each of 12 layers. With `--context 1M`
  that is 2 x 12 sorts of 262k elements a window, whatever the actual length: an estimated 3-6 ms.
- **KV reads.** The per-head K/V cache is 32 KB a token for each DSA layer. Sparse attention reads ~2,048 tokens
  x 32 KB x 11 layers = 0.72 GB a window, about +3.4 ms (M1 removes this).
- **Eager launches.** Python/Triton launch overhead is likely 10-25 ms of host time a window, which may exceed
  the 30 ms of GPU time at R = 1.

The measured 70.8 tok/s at 28k context (greedy, a predictable reply) says the path is not catastrophic, but it is
unprofiled. D1 measures it and L1 fixes it:

- device-side positions and a fixed-size top-512 over `npool` (a radix select bounded by the visible pools),
  making the whole path capturable;
- graphs for R = 1..8 past 2,051 tokens.

**The 1->2-row jump.** Row 2 costs 10.0 ms, and every later row costs 4.7 ms.

- **Cause.** Load-time calibration (`GlmEngine._calibrate`) times windows on uniformly random token ids. Random
  tokens share few experts, while real consecutive tokens share ~50% (the "routing is local" note in
  `PREFILL-ANALYSIS.md`).
- **Effect.** The engine prices real windows with random-token routing. If real V(2) is ~35 ms, every
  depth/drafter decision is biased toward too little drafting. The MTP break-even for the first draft drops from
  a_1 > 0.44 to ~0.22.
- **Test (D1, 0.5 d).** Time V(1..8) on windows cut from the benchmark replies (teacher-forced), next to the
  random-token numbers.

## 3. Parallelism layout

Decode, one sequence, one-row window, per rank. EP counts assume independent uniform routing: with 8 picks split
over 2 owners, E[max(k, 8 - k)] = 1304 / 256 = **5.09 experts** on the busier rank against 4.0 under TP.

| Layout | Critical-path bytes a rank | Collectives a step | Link bytes a step a direction | Estimated step | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| **TP = 2 (now)** | 5.0 GB, balanced | 90 all-gathers + 1 sample exchange | 1.4 MiB | **30 ms** | keep |
| EP experts (144 per Spark), TP attention | 2.6 + 2.69 = 5.3 GB (busier rank) | 90. hc_pre's output is already on both ranks, so there is no dispatch; the combine is one exchange | 1.4 MiB | 31.5-33 ms (+5-10%) | worse: the expert imbalance is +27% at R = 1 and ~+12% at R = 8 (E[max] of ~40 distinct experts) |
| EP + hot-expert replication (~30 GB a rank) | ~5.0-5.1 GB | 90 | 1.4 MiB | ~30 ms | no gain, and it spends the memory KV needs |
| EP + replicated attention | 2.6 + 1.85 (full KDA + DSA) + 2.69 = 7.1 GB | 42 | 0.7 MiB | ~38 ms | worse |
| PP, 23 / 22 layers, two requests in flight | 5.0 GB a stage, run in sequence | 2 point-to-point (32 KiB of streams, a token id) | 32 KiB | 2 x ~21 = **42 ms** of latency; 2 tokens / 42 ms = 48 tok/s aggregate | single stream -30%; loses to TP + batching (below) |
| TP + batching, B = 2 (compare) | 5.0 + ~1.9 GB | 90, shared by both | 2.8 MiB | ~38.5 ms, 52 tok/s aggregate | beats PP at the same concurrency, and keeps 30 ms single-stream |

- **Link latency.** At 5-10 us a collective, TP's minimum is 0.45-0.9 ms a step (90 x alpha), against 2.4 ms
  measured. The gap is NCCL's graph-mode proxy (O3), not the wire.
- **Link bandwidth.** It matters only for prefill: a 2,048-row chunk gathers 33.5 MB a rank per exchange, which
  is 2.5 ms at 109 Gb/s and 222 ms per chunk (section 7).

**Verdict: stay TP = 2.** Spend the effort on per-exchange latency (O2, O3) and on amortizing exchanges over
sequences (B1). Neither EP nor PP reduces bytes on the critical path for one stream, and both lose on balance or
latency.

## 4. Lower bits

### Bytes and time (a rank, R = 1, at ~230 GB/s)

| Component | Now | Option | After | Δ GB | Δ ms a window | Δ ms a later verify row |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Routed experts | 2.11 GB, EXL3 4.0 bpw | EXL3 3.5 (per-tensor K3/K4 mix) | 1.85 | -0.26 | -1.1 | -0.6 |
| | | EXL3 3.0 | 1.58 | -0.53 | -2.3 | -1.2 |
| Shared experts | 0.30 GB, affine 4.5 bpw | EXL3 4.0 / 3.0 | 0.27 / 0.20 | -0.03 / -0.10 | -0.1 / -0.4 | - |
| KDA projections | 1.46 GB, affine 4.5 bpw | EXL3 4.0 / 3.0 | 1.30 / 0.97 | -0.16 / -0.49 | -0.7 / -2.1 | - |
| DSA + dense MLP | 0.52 GB, affine 4.5 bpw | EXL3 4.0 / 3.0 | 0.46 / 0.35 | -0.06 / -0.17 | -0.3 / -0.7 | - |
| Head | 0.18 GB, affine 4.5 bpw | keep | 0.18 | 0 | 0 | - |

- **Q1 (experts at 3.0).** Chat round (4 rows): -2.3 - 3 x 1.2 = -5.9 ms of 55.2, so **+12%**. At 3.5: -2.9 ms,
  **+5.5%**. It frees 76 GB x 25% = **19 GB a rank** at 3.0 (9.5 GB at 3.5), which goes to KV and sessions.
- **Q2 (non-experts).** Affine q4mse at 4.5 bpw becomes EXL3 trellis at 4.0 bpw: -1.1 ms (+2%). Trellis plus
  Hadamard has lower MSE than affine g64, so quality goes up, not down. At 3.0 bpw: -0.76 GB, -3.3 ms (+6.5%), with real
  quality risk on attention.
- **Both at 3.0.** 55.2 -> ~46.0 ms a chat round: **+20%** (chat ~52, code ~73 tok/s).

### Sources (Hugging Face API, `models?search=GLM-5.3-Flash`, 350 results, 2026-09-27)

| Repository | What it is | Use |
| --- | --- | --- |
| `orcarouter/GLM-5.3-Flash-Uncensored-FP8` (gated) | The abliterated block-FP8 source of **our** weights (`neko-legends/...-Uncensored-EXL3` lists it as `base_model`). F8_E4M3 314.4 B params + BF16 6.9 B; attention stays BF16 | **The right source for requantizing experts at 3.0/3.5 bpw with the same abliteration.** exllamav3 convert with `mcg`, experts only, then graft the result into our layout |
| `MikeRoz/GLM-5.3-Flash-Uncensored-{2.51,3.05,4.05}bpw-h6-exl3` | Same orcarouter source; the whole model in EXL3 (attention included), 6-bit head, **codebook `mul1`**; 3.05 bpw is 117.5 GiB (~59 GiB a rank) | Ready-made 3-bit build of our abliteration, but the engine reads only 4-bit mcg experts. Needs a K = 3 trellis and `mul1` decode in `exl3.cu`, plus EXL3 for non-experts (Q2's kernel work) |
| `satgeze/GLM-5.3-Flash-EXL3-TR3-3.5bpw` (stock base) | Per-tensor K3/K4 mix of mcg experts | Quality evidence: **KLD 0.0297 nats against 0.0246 for 4.0 bpw** (five-run gate) |
| `0xSero/GLM-5.3-Flash-EXL3-TR3-3.0bpw`, `...-EXL3-3.0bpw` (stock) | 3.0 bpw builds | KLD/quality comparisons at 3.0 |
| `dealignai/GLM-5.3-Flash-UNCENSORED-FP8`, `genevera/...-Uncensored-EXL3-2.5bpw`, `huihui-ai/GLM-5.3-Flash-abliterated-GGUF`, `Blackfrost-AI/...-DERISKED-BF16` | Other abliteration methods | Not our model; a different refusal edit. Use only if we change base |

- **Quality risk.** 3.5 bpw experts are low risk (+0.005 nats KLD over 4.0 on the stock model). At 3.0, expect
  roughly 2x the 4.0 KLD, which needs a gate.
- **Gate.** Teacher-forced KLD and top-1 agreement against our 4.0 build on 100k tokens (target ≤ 0.02 nats
  mean, ≥ 97% top-1), MMLU-200 ≥ 87%, refusals 0/10, and the tool-call harness unchanged.
- **Cheap first experiment (1 d).** Serve `MikeRoz/...-3.05bpw` in the vLLM prod kit, whose EXL3 kernels take
  any K and codebook. Run MMLU-200, refusals, the tool-call harness and a 50k-token top-1 agreement against the
  4.05 build. If 3.05 whole-model passes, experts-only 3.0 will too.

### KV and indexer compression (M1, Q3)

- **KV per token per rank today.**
  - The engine caches decompressed per-head K and V: 32 heads x (256 + 256) x 2 B, times 12 layers (11 DSA plus
    the MTP layer) = **393 KB**.
  - The indexer adds ~6 KB: keys 128 x 2 B, gates 128 x 2 B, and pools.
  - About 0.4 MB/token, so the ~35-40 GB free per rank holds ~100k tokens. **1M context would need 400 GB a
    rank.**
- **Latent (absorbed MLA).** GLM-5.3 has `qk_rope_head_dim = 0`, so the latent is just `kv_lora = 512`: 1 KB a
  layer a token.
  - Cache size: 12 KB + 6 KB indexer = **18.6 KB/token (21x smaller)**, or ~9.5 KB with an fp8 latent and fp8
    indexer keys (41x). 1M tokens = 18.6 GB (bf16) or 9.5 GB (fp8) a rank.
  - Decode reads: 2,048 selected tokens x 1 KB x 11 = 22 MB a window instead of 0.72 GB, so **-3.4 ms a window
    past ~2k**. Below 2,051 the dense path reads pos x 32 KB x 12, which is 0.77 GB (3.3 ms) at 2,000 tokens.
  - Extra compute: absorbing `W_uk` / `W_uv` costs ~1.7 GFLOP a row a rank, ~0.05 ms on tensor cores.
  - Exactness: new arithmetic, still row-invariant, so drafted == serial holds. The replies differ from today's
    engine at the last bits (like any kernel change).

## 5. Speculation

### S1. Suffix / prompt-lookup drafts (the free drafter)

- **Mechanism.** A suffix index over prompt + reply + earlier turns on both ranks (the same token history, so the
  same drafts with no exchange). `tensorfold/drafters/draft_ngram.py` (`SessionNGram`) can be reused.
  - When the longest match is ≥ 3 tokens, propose its continuation, with depth set by match length and frequency.
  - Add it as arm "p" to `DrafterChoice`, whose cost is V(R) only: no 2-3.9 ms draft.
  - Let "p" rounds use R up to 16 (buffers and graphs for 12 and 16 rows).
- **Arithmetic** (per-token acceptance q on copied spans; the round includes a 0.35 ms host tail):

  | q | 7 drafts (V(8)) | 11 drafts (V(12)) | 15 drafts (V(16)) |
  | ---: | ---: | ---: | ---: |
  | 0.95 | E[T] 6.73 -> 98 tok/s | 9.19 -> 105 | 11.2 -> 106 |
  | 0.98 | 7.46 -> 108 | 10.8 -> 124 | 13.8 -> 130 |

  - DFlash2 at 7 drafts and r = 0.95 gives 92.7 tok/s (`DECODE-ANALYSIS.md`).
  - On verbatim spans (edit tools' `old_string`, re-emitted files, JSON arguments quoting paths) PLD adds
    **+15-40% on those spans**, more against MTP rounds, which cap at 3 drafts.
  - Blended over agent output with copy fraction f = 0.3-0.6: **+10-40%**. Prose and chat: ~0.
- **Effort** 3.5 d. **Risk** low. **Exactness** none; the verify rule decides.
- **First cheap experiment (0.5 d, no GPU).** Replay recorded opencode transcripts from the vLLM kit logs (greedy
  turns). At each output position, compute the suffix proposal and its accepted length against the real
  continuation. Output: the copy fraction f, the E[T] histogram, and projected tok/s from V(R). Build S1 only if
  f ≥ 0.2 on agent turns.

### Better drafters

| Option | Mechanism | Arithmetic | Effort | Risk / license |
| --- | --- | --- | ---: | --- |
| S4. DFlash2 `fc` ridge recalibration | DFlash2 reads target taps at layers [5, 14, 24, 33, 42]; abliteration shifts the deep ones (a sibling abliteration measured up to 2.0% drift at L42, mean cosine 0.926). Solve min ‖W H_ablit − fc_old H_stock‖² + λ‖W − fc_old‖² on ~50k tokens of taps from our model and from the stock MLX checkpoint on the same text | Restores part of the lost acceptance: +1-4% on DFlash2 rounds (about half of greedy rounds) | 2 d | incoai's DFlash2 is CC BY-NC-ND: a private modification only, never published (`gorbatjovy/...-ablit-fc` did this for a different abliteration) |
| S6. RedHat DSpark (`RedHatAI/GLM-5.3-Flash-speculator.dspark-preview`, MIT, 2.5 B, 5 layers, up to 8 drafts, **confidence head**) | A DFlash-style block drafter plus per-position acceptance prediction, which is exactly the signal S2 needs | Block ~0.7 GB a rank at 4 bits, ~5-6 ms against 3.9. Pays only if E[T] rises ≥ 0.15 a round | 3.5 d | Unknown acceptance on our target |
| S6'. `canada-quant/GLM-5.3-Flash-DFlash2-G` (Apache-2.0, 8 layers, ~3.1 B) | Drop-in DFlash2, license-clean for a public image | Mean acceptance 3.676 against incoai's 3.632 at K = 7 (+1%), but a ~2.6x larger block (~6 ms): slower here | 1 d | Use only for licensing |
| S5. Own drafter on abliterated outputs | EAGLE-3 / DFlash-style, 0.5-1 B, trained on our target's taps and outputs | Data costs more than training. Teacher-forced taps come at prefill speed: 10M tokens in 7 h today, ~1.5 h after P2. Regenerated replies at ~50 tok/s take 55 h per 10M tokens (~15 h with B1). Training on one Spark: hours. Expected +5-10 points of a_1..a_3, **+8-15%** on non-copy text | 16 d | Training code for the GLM5 taps; storage (10M x 40 KB taps = 400 GB, so train streaming or in fp8) |
| S5'. MTP fine-tune | LoRA on the MTP layer's attention, `eh_proj` and shared expert (7.4 B params in bf16; the experts stay frozen), trained on chained inputs (its own previous draft as input, as in EAGLE's training-time test) | Mainly lifts a_2 and a_3 (now 10-36% at depth 3): a_3 0.22 -> 0.35 gives E[T] +0.13, about +5% chat | 10-14 d | MIT, publishable |

### Multi-token MTP chains, trees, dynamic depth

- **Chains.** GLM has one MTP layer, so chains reuse it, and a_3 decays fast. Depth 4 for sampled requests pays
  only when running acceptance is ≥ 0.92 (`DECODE-ANALYSIS.md`). S5' is the real fix.
- **Trees against the KDA recurrence: rejected.**
  - State copies are not the problem. A branch needs the state after its parent: 64 KB a head (16 fp32
    registers x 1,024 threads), which fits in a register or shared-memory copy inside `chain_kernel`. A full
    state per rank is 34 x 32 x 128 x 128 x 4 B = 71 MB, about 0.6 ms to copy through DRAM, and is never
    needed. The commit replays along the accepted path (`qwen3_5/cuda/gdn_tree.cu` already does this for GDN).
  - The cost is experts: each node is a row (4.7-10 ms). A second first-position candidate is accepted with
    P ≈ (1 − a_1) x P(2nd | miss) ≈ 0.26 x 0.4 ≈ 0.10, worth ~0.12 tokens ≈ 2.7 ms at 44 tok/s against a
    4.7 ms row. **Net -2 ms a round.** Trees would pay only if rows cost < ~2.5 ms.
- **S2. Dynamic depth.** `cN:P` and `fcN:P` already stop on the chain probability product. What is missing:
  - calibration on real tokens (section 2);
  - a threshold derived from the measured marginal row cost and the running ms per token, instead of the fixed
    0.35 / 0.3;
  - per-position confidence from a DSpark-style head.

  Expected **+2-4%**.

## 6. Throughput paradigm for agents

### B1. Continuous batching (2-8 sequences in one window)

**Experts are not "nearly free" for extra sequences here.** Consecutive tokens of one sequence share ~50% of
their experts (4.7 ms a row). Independent sequences share few (6-10 ms a row, taken as ~8 ms). The shared part is
all non-expert bytes (~2.6 GB), the exchanges and the overheads: about 16-20 ms of the 30.

| B | Rows | Step (serial, est.) | Aggregate | Per sequence | With 1 MTP draft each (E[T] 1.74) | Aggregate | Per sequence |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 30.0 | 33 | 33 | 42 ms | 41 | 41 |
| 2 | 2 | 38.5 | 52 | 26 | ~56 ms | 62 | 31 |
| 4 | 4 | 54 | 74 | 18.5 | ~85 ms | 82 | 20 |
| 8 | 8 | 86 | 93 | 11.6 | ~142 ms | 98 | 12 |

- **Compute never binds.** 16 rows x 18 GFLOP a rank = 0.29 TFLOP, ~10 ms at 30 TFLOP/s, against a
  ~140 ms step. MoE decode stays bound by the union of experts until ~all 288 are read (76 GB a rank, 330 ms),
  i.e. hundreds of rows.
- **Draft depth under batching.** It should drop to 0-2 per sequence as B grows, because the break-even a_k
  rises when rows are shared.
- **What it needs.** Per-sequence `State` (KDA state, conv, KV pages); the KDA chain over (head, sequence); DSA
  attention with a per-row sequence id and position; commit per sequence; graphs per (B, R); a scheduler; and
  device-side sampling (O4), because host work grows with B.
- **Exactness.** Unchanged: kernels are row-invariant, so a sequence's rows never depend on their batch-mates.
- **Effort** 15-20 d. **Value** is multi-agent and multi-user aggregate throughput (vLLM's kit runs 4 streams).
  A single user's single stream does not benefit.

### M2. Multi-session prefix / state cache

- **Snapshot size per rank.**
  - KDA state: 34 layers x 32 heads x 128 x 128 x 4 B = **71.3 MB (68 MiB)**.
  - Conv windows: 34 x 3 x 12,288 x 2 B = 2.5 MB.
  - So **~74 MB a snapshot**; 100 snapshots are 7.4 GB.
- **Attention KV.** The KV cannot be shared between sessions without per-session pages. Today one sequence owns
  the cache, and only the last prompt and reply are resumable (`GlmEngine.cache` holds 2 entries).
  - At 0.39 MB/token a 30k session is 11.7 GB, so only ~3 such sessions fit.
  - With M1 it is 0.56 GB, so **~60 sessions of 30k** fit in 35 GB.
- **Mechanism.**
  - A paged latent KV store per session.
  - KDA/conv snapshots at canonical boundaries (every 4k tokens and at each turn end).
  - LRU keyed by a token-prefix hash; both ranks key alike (the header already carries the resume length).
- **Gain.** A switch back to a cached 30k-token session re-prefills only the new tokens: ~2k at 420-1,500 tok/s
  is 1.3-5 s, against **70 s** today (30k at 420). That is the biggest TTFT lever for agents that interleave
  sessions or subagents. vLLM's kit gets 97% prefix-cache hits.
- **Effort** 5 d after M1. **Exactness** unchanged: resumed prefills already equal fresh ones, and snapshots copy
  bits.

### Speculative prefill

- **Token-dropping "SpecPrefill"** (prefill only the tokens a small model deems important): not exact, rejected.
- **What remains.** The engine already keeps the state after the last reply. With M2 the only work left per agent
  turn is the tool result (typically 0.5-5k tokens). "Speculative" pre-work would be re-canonicalizing the reply
  rows while the tool runs (section 7), which is idle-time work.

## 7. Prefill paradigm

**Measured.** 404-420 tok/s at 512-row chunks, 1.25 s a chunk. `PREFILL-ANALYSIS.md` attributes 50-75% of it to
the EXL3 grouped kernel re-reading each expert per 16-member tile. Patch 0006 (written, unmeasured) fixes that
bit-identically, estimating 510-640 tok/s at 512 rows and **800-1,100 at 2048**.

### P1. Land 0005/0006 and sweep chunk size (1.5 d)

- **Chunk size.** Bigger chunks amortize the 76 GB of experts a rank that every chunk of ≥ 256 rows reads (at
  512 rows, 1 − (1 − 8/288)^512 ≈ 1). At 230 GB/s that is 331 ms a chunk: 0.65 ms/token at 512 rows, 0.16 at
  2,048.
- **Head skip.** Prefill needs only the last row's logits (the MTP absorb reads `fnormed`, not logits). Skipping
  the head on non-final chunks saves ~10 ms a 512-row chunk (~1%) and is bit-identical.
- **Memory.** Buffers cost ~5 MB/row, so 2,048 rows is ~10 GB a rank. That fits today, but it competes with KV,
  so check `torch.cuda.mem_get_info` at the served `--context`.
- **Success.** ≥ 800 tok/s at 8k and 28k prompts; drafted == serial 10/10; resumed == fresh.

### P2. Canonical-chunk fast prefill (12 d)

**The contract question.** Prefill bits never have to equal decode-step bits:

- drafted and serial decoding of a request both start from the *same* prefill, so drafted == serial holds for any
  deterministic prefill;
- the `draft: false` reference also takes that prefill path.

What breaks is **resume == fresh prefill**. A snapshot taken after a reply holds reply rows built by decode
windows, while a fresh prefill builds them with the fast kernels. Two rules keep it exact:

1. **Canonical chunks.**
   - Chunk boundaries sit at absolute multiples of C (e.g. 256), and each chunk's shape depends only on its
     (start, length).
   - A resumable snapshot sits only at a boundary. A resume re-prefills the partial tail from the last boundary
     (< C tokens).
   - Then a resumed prompt equals a fresh prefill bit for bit, even with non-row-invariant GEMMs (cuBLAS/CUTLASS
     are deterministic for a fixed shape once the algorithm is pinned) or a chunked KDA scan.
2. **Re-canonicalize replies.**
   - After a reply, re-prefill its rows from the prompt-end canonical snapshot in idle time (500 tokens at
     1,500 tok/s = 0.33 s, usually while the tool runs), then store that snapshot.
   - If the next turn arrives first, resume from the prompt-end snapshot and prefill reply + new tokens together:
     TTFT rises by reply_len / prefill_rate.

This is the answer to "can prefill use a chunked KDA scan if decode replays from the same chunk boundaries":
**yes**. Decode does not have to replay anything. Decode continues serially from the prefill's final state, and
only snapshots must sit on canonical boundaries.

**What P2 changes, and the arithmetic per 2,048-row chunk.** Per rank ~9 B active params, so 18 GFLOP/token and
37 TFLOP a chunk. GB10 BF16 `mma.sync` peaks at ~60 TFLOP/s (estimate in `PREFILL-ANALYSIS.md`).

| Piece | Today (row-invariant, ≤ 128-row tiles) | P2 | Arithmetic |
| --- | --- | --- | --- |
| Experts | GEMV-style mma, 16-row tiles, decode per tile | Weight-stationary grouped GEMM (57 rows/expert on average at 2,048 rows), one trellis decode per tile per chunk, large N tiles | 76 GB read = 331 ms, overlapped with ~24 TFLOP at 30-40 TFLOP/s (600-800 ms) |
| Non-expert 4-bit matmuls | 128-row tiles: 2.46 GB re-read once per 128 rows (16x = 39 GB a chunk if L2 misses) | Tiles of 256 rows or more; dequantize once to bf16 in shared memory; cuBLAS-class efficiency | 4.4 G params x 2 x 2,048 = 18 TFLOP, 0.4-0.6 s at 30-45 TFLOP/s |
| All-gathers | 90 x 33.5 MB fp32 = 2.5 ms each at 109 Gb/s: **222 ms a chunk** | Two QPs (`NCCL_IB_QPS_PER_CONNECTION=2`), bf16 partials (prefill bits are free), and half-chunk pipelining (gather half A while half B computes; rows are independent) | Down to ~60 ms (bf16 + 2 QPs), then hidden |
| KDA recurrence | 0.6-1.0 us a row, serial: 10-17 ms a 512-row chunk (~1% today) | Leave serial until the rest is fast. At 2,000 tok/s it is ~5% (P3: chunked WY scan, allowed by the canonical rule) | - |
| Sparse attention past 2k | per row, 1/16 mma use | 16-row tiles over rows that share pools | 60-180 ms a 512-row chunk, down 3-5x |
| Optional: FP8 activations (`mma` kind f8f6f4 on sm_121) | - | 2x tensor rate for prefill only | Quality check against the bf16 prefill (top-1 ≥ 99.5% on teacher-forced text) |

- **Target: ≥ 1,500 tok/s** at 8k-28k (vLLM's kit: 960-1,448), ~2,000 with FP8 activations. At 1,500 tok/s a
  30k cold prompt takes 20 s instead of 71 s.
- **Risk.** Medium: new GEMMs, and pinning cuBLAS algorithms for determinism.
- **Exactness.** It holds under the two rules above. Add a test: resumed == fresh with the fast path, and
  drafted == serial.

**P3. Chunked KDA scan.** Only after P2: +3-5% prefill for 5 d.

## 8. Other levers found in the code

- **Calibration on random tokens** (`engine._calibrate`, `tokens(r)` = uniform ids) misprices windows. See D1
  and S2.
- **Eager past 2,051 tokens.** Graphs apply only while `pos + R ≤ dense_limit` (`decode.Engine.forward`,
  `.mtp`). See L1.
- **The indexer is sized by capacity.** `select_tokens` scores and sorts `capacity / 4` pools, whatever the
  length. See L1: bound it to `npool` with a device-side top-512.
- **Replicated reads** (router 42 x ~2.4 MB, HC `fn` 90 x 1.5 MB fp32, MLA `q_a`/`kv_a`, indexer): 0.17-0.23 GB,
  0.7-1 ms a window. Splitting them would add exchanges (27 us each), so leave them. Storing HC `fn` in bf16 would
  change the model's arithmetic: not worth it.
- **`GRAPH_ROWS = 1..6`.** 7- and 8-row windows (now common with patch 0010) run eagerly. The recipe's direct
  timings sit on the fitted line, so this is likely ≤ 1 ms. D1 confirms; add 7 and 8 if it is not.
- **The timing-only `torch.cuda.synchronize()`** after each verify forward in `auto_decode`: ~0.05-0.1 ms a
  round, trivial to drop.
- **NVFP4 weights.** 4 bits + an fp8 scale per 16 = 4.5 bpw, the same bytes as today's affine q4, so no decode
  gain. It is interesting only for prefill compute (FP4 `mma`) with W4A4 quality risk. Not recommended.

## 9. Rejected, with numbers

| Idea | Why |
| --- | --- |
| Expert parallelism (any variant) | The busier rank carries 5.09 of 8 experts (+27% expert time at R = 1); same exchange count; replicating attention costs +1.85 GB a rank |
| Pipeline parallelism | 42 ms single-stream latency (2 x ~21); 2 in flight give 48 tok/s aggregate, below TP + batching (52) at the same concurrency |
| Verify trees | Sibling acceptance ~0.10 < break-even ~0.21-0.28; net -2 ms a round |
| Fewer, merged all-gathers in TP | Nonlinear hc_post / Sinkhorn / norm between exchanges; 90 is the exact minimum (`COMM-ANALYSIS.md`) |
| bf16 partials or all-reduce in decode | Latency-bound (≤ 0.1 ms of wire time at 1 row); all-reduce is 2 alpha |
| NCCL protocol / channel tuning | Tried upstream: defaults best; LL made wide windows 56% slower on the 27B |
| SpecPrefill (token dropping) | Not exact |
| Chunked KDA for decode windows | Windows are ≤ 16 rows; the chain is ~1% of a step |
| 2-2.5 bpw experts | KLD far past the gate; tool-call reliability risk |

## 10. Full scoring

Score = midpoint gain x weight / days (weights in section 0). "Cap" marks items whose main value is a capability
(1M context, sessions), which a speed score understates.

| ID | Metric | Gain (mid) | Weight | Days | Score | Depends on |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| P1 | prefill | +100% | 0.6 | 1.5 | 40 | - |
| M2 | TTFT on a session switch | 70 s -> 3 s (take +100%) | 0.6 x 0.5 | 5 | 6 | M1 |
| S1 | agent decode | +25% | 0.8 | 3.5 | 5.7 | - |
| P2 | prefill | +100% over P1 | 0.6 | 12 | 5.0 | P1 |
| D1 | enabler | ~+8% found | 0.9 | 1.5 | 4.8 | - |
| O2 | decode | +3% | 1.0 | 1 | 3.0 | - |
| L1 | long-context decode | +15% | 0.8 | 4.5 | 2.7 | D1 |
| S2 | decode | +3% | 1.0 | 1.5 | 2.0 | D1 |
| B1 | aggregate | +100% | 0.3 | 17 | 1.8 | O4 |
| Q1 | decode | +11% (3.0) | 1.0 | 7.5 | 1.5 | - |
| S3 | decode (MTP cells) | +2% | 1.0 | 1.5 | 1.3 | - |
| S4 | decode (DFlash2 rounds) | +2.5% | 1.0 | 2 | 1.2 | - |
| M1 | long-context decode + 21x KV (cap) | +10% | 0.8 | 7 | 1.1 + cap | - |
| S6 | decode | +3% (uncertain) | 1.0 | 3.5 | 0.9 | - |
| Q2 | decode | +5% | 1.0 | 6 | 0.8 | shares kernel work with Q1 |
| O1 | decode | +5.5% | 1.0 | 7 | 0.8 | - |
| O5 | decode | +20% | 1.0 | 25 | 0.8 | - |
| O3 | decode | +5% | 1.0 | 7.5 | 0.7 | - |
| S5 | decode (non-copy) | +11% | 1.0 | 16 | 0.7 | P2 speeds up data generation |
| P3 | prefill | +4% | 0.6 | 5 | 0.5 | P2 |
| O4 | decode | +2% | 1.0 | 4.5 | 0.4 | - (needed by B1) |

## 11. Recommended program: the top 5, in order

1. **D1 + P1: instrument, then take the free prefill win (week 1, 3 d).**
   - Work:
     - nsys plus the 0005 probes extended to decode, for V(1..8) on real-token windows at 300 / 2,000 / 2,100 /
       28k context;
     - `--context 32k` against `256k` at the same position (does the indexer cost follow capacity?);
     - a 7- and 8-row eager-versus-graph A/B;
     - land 0005/0006/0040 and sweep `GLM53_TF_PREFILL_ROWS` 512/1024/2048.
   - **Success:**
     - a per-kernel breakdown that sums to within 5% of the measured window at every size and context;
     - prefill **≥ 800 tok/s** at 8k and 28k;
     - 0040 keeps V(1) ≤ 29.2 ms or is dropped;
     - `exact` suite 10/10.
2. **S1: suffix / prompt-lookup arm (week 2, 3.5 d, after a 0.5 d transcript replay showing f ≥ 0.2).**
   - **Success:**
     - on an agent-edit suite (re-emit a 200-line file with 3 edits; tool calls quoting context paths),
       **≥ 1.4x** the current `auto` tok/s;
     - no existing `tf` / `kit` / `tweet` cell regresses by more than 2%;
     - drafted == serial 10/10.
3. **P2: canonical-chunk fast prefill (weeks 3-5, 12 d).**
   - **Success:**
     - **≥ 1,500 tok/s** at 8k and 28k prompts;
     - resumed == fresh prefill by token-id SHA-256 on 5 multi-turn transcripts;
     - drafted == serial 10/10;
     - teacher-forced top-1 agreement ≥ 99.5% against the row-invariant prefill on 150-token continuations.
4. **M1 + M2 (+ L1 if D1 shows the eager and capacity costs): long context and sessions (weeks 6-8, 12-16 d).**
   - **Success:**
     - `--context 1M` starts with ≤ 20 GB of KV a rank;
     - one-row decode at 28k within **10%** of 300-token context;
     - switching between 4 cached 30k sessions gives **TTFT ≤ 3 s** for a 2k-token tool result;
     - resumed == fresh and drafted == serial hold.
5. **Q1 (+ Q2 at 4.0 bpw): fewer bytes (weeks 9-10, 10-12 d, gated by the 1-day MikeRoz 3.05 quality check).**
   - **Success:**
     - one-row verify **≤ 25 ms** (from 30.0);
     - chat greedy **≥ 52** and code greedy **≥ 70** tok/s;
     - KLD against the current 4.0 build ≤ 0.02 nats (mean over 100k tokens), top-1 ≥ 97%;
     - MMLU-200 ≥ 87%, refusals 0/10, tool-call harness unchanged.

**Next after the five:**

- O1 + O3 (overheads, target V(1) ≤ 25 ms before the bit cuts, stacking with Q1 toward ~21 ms);
- B1 if multi-agent aggregate throughput becomes the goal;
- S5' / S5 for acceptance on non-copy text.

## Appendix: constants

- GLM-5.3-Flash config (HF): hidden 4,096; 64 heads; `qk_nope_head_dim` 256, `qk_rope_head_dim` 0, `v_head_dim`
  256; `kv_lora_rank` 512, `q_lora_rank` 1,536; KDA 64 heads x 128; 288 experts, top 8, width 2,048, 1 shared;
  dense layers 0-2 (12,288); index top-k 2,048 in pools of 4; vocab 154,880; 1 MTP layer.
- Per rank:
  - one expert = 3 x 4,096 x 1,024 x 0.5 B = 6.29 MB;
  - all experts = 76.1 GB;
  - a token's routed experts = 8 x 6.29 x 42 = 2.11 GB;
  - all-gather payload = R x 4,096 x 4 B = 16 KiB x R.
- Link: 109 Gb/s measured per QP = 13.6 GB/s; captured all-gather ~27-28 us, eager ~17 us.

## Backlog: host-language rewrite (user question, 2026-09-28)

Question: rewrite the Python parts in Rust or Go for speed?

- The math already runs in CUDA/Triton kernels. Decode replays CUDA graphs; measured host overhead is about 2.5% of
  a decode round (~0.3 ms of ~30 ms, docs/DECODE-ANALYSIS.md). Prefill is GPU-bound. A full rewrite would recover a
  few percent at most, cost weeks, and fork us permanently from upstream TensorFold.
- Where host code does hurt: the batching scheduler (per-slot Python loops and per-slot kernel launches each round),
  host round trips in draft selection (`.cpu()`, numpy in the DFlash2 candidate path), and the stdlib HTTP server
  under concurrent load.
- Plan: (1) move per-round work onto the GPU (one launch across slots, graphs keyed by total rows, draft selection
  on device; partly in patch 0200); (2) a Rust front end (HTTP server, queue, scheduler) talking to the Python engine
  processes, as SGLang/vLLM routers do; the tokenizer is already Rust (HF `tokenizers`). Keep the engine in Python on
  upstream TensorFold.
