# What the sfxnz NVFP4 vLLM kit does that we could use

This audit covers [sfxnz/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark](https://github.com/sfxnz/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark),
main @ `072c3f6` (2026-09-28). It reads the whole repo:

- the README, `AGENTS.md` and `recipe.yaml`;
- `run.sh`, the v8-v13 Dockerfiles and every `docker/patch_*.py` with its tests;
- `docker/README-v13.md`;
- `bench_decode.py`, `kit/compare.py` and `quality/`;
- `tools/`;
- the evidence tree, `evidence/e0`-`e6`, `iter-*` and the 2026-09-25 review (`evidence/review-20260925/`).

The kit serves `nvidia/GLM-5.3-Flash-NVFP4` (ModelOpt NVFP4 on the routed experts and the layer 0-2 MLP) on vLLM at
TP=2 across two Sparks. It drafts with DFlash2 k=7 (the same `incoai` revision `7d74cdd` we load) and runs CUDA graphs.

No GPU was used and the Sparks were not touched. "Their receipt" is a number quoted from their evidence files; we did not
re-measure any of it. "Gain for us" is our estimate against this repo's round model (DECODE-PLAN §1, ROOFLINE §2, PROFILE
§5). The one exception is the kernel table in §3.2, which also uses our own W11 kernel bench (`results/W11/kbench.log`,
uncommitted at the time of writing).

**Credit and licence.**

- The recipe, scripts, patches, bench and quality tools are MIT: `LICENSE`, "Copyright (c) 2026", with no holder named.
  The README says "Recipe scripts are MIT".
- Code can be reused here (this repo is MIT) as long as their copyright and permission notice travels with the copied
  files. Add them to `NOTICE` if anything is copied.
- Several review findings cite files in the author's private lab (`/home/<user>/projects/ai-lab/...`, a DeepSeek-V4.1
  sibling recipe). Those files are not in the repo and cannot be checked.
- Their Tier-0 corpus (`quality/data/corpus.jsonl`, 96 documents) carries per-row licences, listed in
  `quality/data/SOURCES.md`: Project Gutenberg public domain, PSF / BSD / MIT / Apache code, and MIT GSM8K / MATH. It
  can be reused with that file kept alongside.
- The DFlash2 drafter is CC BY-NC-ND 4.0 for them as for us.
- vLLM's Marlin kernels, relevant to §4 item 1, are Apache-2.0: compatible, with its NOTICE kept.

## 0. Bottom line

- **They are not close to our speed on like-for-like cells** (table §2):
  - single-stream prose: we are about 1.5x (44.9 vs 30.0 tok/s);
  - count-to-200: 1.18x;
  - four concurrent clients: ~1.9x (they admit 2 sequences);
  - prefill: 1.35x.

  The "similar speed" impression most likely comes from mismatched cells. Their long-context cell E@32k decodes at 44.6
  tok/s, the same digits as our chat-greedy 44.6. But E@32k is a 128-token summary after a 32k filler prompt, with
  acceptance swinging 2.2-4.3 between panels. Their count-to-200 at 85 tok/s is the closest real pair (ours 100.6).
- **Bytes per token.** At the same 4 bits, they read ~1.7x our bytes per token and take ~1.5x our time per token.
  - Experts: NVFP4 is 4.5 bpw (e2m1 + an e4m3 scale per 16) against EXL3's 4.0, so 7.08 against 6.29 MB per expert
    half a rank.
  - Non-expert weights: they keep them at INT8 (8.125 bpw) or BF16 for quality; ours are 4.5 bpw.
  - So per verify step they stream ~13.5 GB a rank against our ~8.6 GB a round.
  - Their step converts bytes to time **~15-20% more efficiently than our round**: ~185 GB/s whole-step effective
    against ~158.
- **Where that efficiency comes from: kernels, nothing else.**
  - **Dense weights:** vLLM's Marlin runs at 210-229 GB/s on the dense projections at M = 1-16. On **the exact per-rank
    shapes we run**, at the **same 4.5 bpw**, Marlin NVFP4 is 1.07-1.75x faster than our q4 `_qmm` (§3.2).
  - **Routed experts:** Marlin MoE runs at ~213 GB/s in the step, against our `grouped_kernel` family at ~193 in the
    round. Our kernel alone already reaches 205-222 GB/s (W11), so the gap is ramp, tail and the epilogue / rot_in
    launches.
  - **Everything else favours us:**
    - CUDA graphs: both stacks capture.
    - Speculation: their 2.21 tokens a step against our 2.4.
    - Communication: their NCCL all-reduce costs 3.8 ms a step, our RoCE all-gather ~1.7 ms.
    - Parallelism: TP=2 on both, no EP.
    - FP4 paths: FlashInfer / CUTLASS FP4 is refused on sm_121 (JIT OOM); Marlin dequantizes to BF16 `mma.sync`, the
      same class as ours.
    - Clocks: none set.
    - Idle: theirs 5.5 ms a step, ours ~3.8.
- **What to adopt, in order (§5):**
  1. **E2 with Marlin as the target.** It is now a measured kernel on this GPU, not an estimate: -2.6 to -3.0 ms a
     round, about +5-6% single stream and +4-6% at 4 streams.
  2. **Two cheap correctness and quality gates:**
     - a real-model drafted == serial check past 2,048 tokens of context (the bug class they found in vLLM's indexer
       ring);
     - a teacher-forced KL / top-1 measurement of q4mse against BF16 non-expert weights. Their data says 4.5-bpw
       non-expert weights cost measurable KL.
  3. **E1 retargeted at the in-round overhead**, with 213 GB/s as the demonstrated bar.
  4. **Re-base T2's prose expectation downward and T1's upward.** DFlash2 on the *un-abliterated* NVFP4 target accepts
     only .64 / .35 / .17 on prose, which is below our MTP head's .74 / .45 / .22.
  5. **Measurement hygiene:**
     - their 8-essay x 512-token prose cell, so the two kits can be compared directly;
     - boot-as-unit statistics;
     - swap-growth invalidation;
     - a PM QoS A/B.

## 1. Their numbers and how they are measured

Their ruler is `bench_decode.py`, "ruler v2". It is stdlib-only and uses streamed chat completions with thinking off
(`chat_template_kwargs.enable_thinking: false`) except cell T.

- **Timing.** decode tok/s = (completion_tokens - 1) / (last - first streamed token), the same formula as our
  `glmbench.py`.
- **Forced lengths.** `min_tokens = max_tokens`, so a reply cannot stop early.
- **Factorization.** Every wave is factorized from vLLM `/metrics` deltas:
  - acceptance_len = 1 + accepted / drafts (tokens a verify step);
  - step_ms = decode_s / drafts;
  - tok_s ≈ acceptance_len x 1000 / step_ms (checked to 3%).
- **Invalid cells.** A cell is INVALID if swap grows more than 64 MiB while it runs.

Published values are the mean of three plain `./run.sh` boots on 2026-09-28 (F1, F2b, G2), each boot's two panels
averaged first (`evidence/e6-prefill/compare-f12g2-vs-e0.txt`, `recipe.yaml` `measured`). Cells marked * come from
fewer boots.

| Cell | Workload | Context | Concurrency | Tokens | Their tok/s (per stream) | Step ms | Tokens a step |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| A (published) | 8 distinct essays / stories / letters ("about 1000 words"), greedy | short | 1 | 512 forced | **30.04** | 73.2 | 2.21 |
| B | 8 distinct code prompts (LRU cache, Rust parser, Go server, ...), greedy | short | 1 | 512 forced | 46.46 | 88 | 4.1 |
| J@c1 | "Count from 1 to 200 ... separated by commas", greedy | short | 1 | max 200 | 85.18 | 93.5 | 8.00 (the k=7 cap) |
| J@c2 | the same prompt on both streams (they share experts) | short | 2 | max 200 | 70.33 (~140 aggregate) | 111-117 | 7.9-8.0 |
| H (published) | A's prompts, two streams (prefix-cache hits) | short | 2 | 512 forced | 20.69 (**39.43 aggregate**) | 105.3 | 2.18 |
| I* | A's prompts, 4 clients; only 2 run (`--max-num-seqs 2`), TTFT 12 s queued | short | 4 | 512 forced | 20.66 (~41 aggregate) | 107 | 2.2 |
| G* | A's prompts sampled, T 1.0 / top_p 0.95 | short | 1 | 512 forced | 28.45 | 73.7 | 2.10 |
| T* | A's prompts, thinking on (Max effort), reasoning + content | short | 1 | 512 forced | 34.5-35.4 | 75 | 2.6 |
| K* | legacy ~98-token prose paragraph | short | 1 / 2 | max 200 | 27.98 / 21.36 | | |
| F* | generated PNG, "describe in 400 words" | image | 1 | 256 forced | 36.08 | 74.7 | 2.7 |
| E@32k* | unique salt + seeded word filler, "in about 200 words describe which words appear most often" | 32k | 1 | 128 forced | 44.61 (panels 31.8-51.8) | 69-79 | 2.2-4.3 |
| E@128k* | same | 128k | 1 | 128 forced | 33.34 | 72 | 2.5 |
| Prefill | cold, unique salt, server-side time | 32k / 128k | 1 | | **1,187-1,192 / 1,183-1,190 tok/s** (TTFT 128k ~110 s) | | |

- Chunks are 1,152 prefill tokens (a mamba-aligned split, not the 2,048 they configure).
- Prefill was **-10.5% against their own E0** (1,331 tok/s), because INT8 Marlin runs 3.6x slower than BF16 on `kda_in`
  at M = 1,152.

**Their noise and determinism.**

- Their serve is not run-to-run deterministic. In E0's same-boot A/A, greedy output diverged on 14 of 20 prompts and
  top-1 disagreed on 1.55% of teacher-forced positions.
- Same-code boots moved tok/s by up to 3.7% (E1a vs E0), so they judge on step_ms and on multi-boot means with a ±4%
  band.
- Our engine gives byte-identical replies across loads, so only timing noise applies to us.

## 2. Like-for-like comparison

Ours is the production config (image b4 + the W10 knobs). The cells come from `bench/glmbench.py`: median of 5 reps of
**one** prompt, thinking off, greedy unless noted. The 4-stream figure is W10 FIN's concurrency bench.

| Workload | Theirs | Ours | Ours / theirs | Methodology differences (read before quoting the ratio) |
| --- | --- | --- | ---: | --- |
| Prose, 1 stream, greedy | A **30.04** (8 prompts x 512 forced) | kit essay **44.9** (200 tok); kit hashmap 53.2 (200); tf chat 44.6 (64 tok) | **1.49** (essay) to 1.77 | Same timing formula, prompts differ. Ours are one prompt each and 64-200 tokens, theirs 512 forced tokens over 8 prompts. Later positions of a long essay accept a little less, so the fair ratio is probably ~1.4-1.5x. Run their cell A on our server to close this (§5, N3) |
| Code, 1 stream, greedy | B 46.46 (chat, 512 forced) | tf code 77.6 is a **raw completion, 64 tokens**: not comparable. Our matching cell is tweet code 512 (LRU cache, chat): 58.4 in the README table, measured before W1-W10 and not republished | ≥ 1.26 (against the stale 58.4) | Apples to oranges if 77.6 is quoted. Re-run tweet code 512 |
| Count 1-200, 1 stream | J@c1 **85.18** (commas, acceptance capped at 8 a step) | kit structured **100.6** (spaces, 200 tok) | **1.18** | Nearly the same prompt. Our windows go to 16 rows (0380) and theirs stop at 8 |
| Sampled prose, 1 stream | G 28.45 (T 1, top_p .95, 512 forced) | tf chat sampled 41.7 (T 1, top-k 20, top-p .95, 64 tok; README table, older engine) | ~1.47 | Older number and different sampler settings |
| Concurrent clients | 4 clients: ~41 aggregate (2 run, 2 queue). 2 clients: 39.43 aggregate (**identical prompts**, so the streams share experts) | 4 streams: **~78 aggregate** (W10 FIN median 78.1; 5 reps over a mixed prompt set, one stream structured-like at ~7 tokens a round) | **~1.9** | Our mix includes a high-acceptance stream. Our prose streams at 4-way run 19-22 tok/s each, i.e. their c=2 per-stream rate at twice the streams |
| Thinking on | T ~35 | not measured on this ruler | - | |
| Decode behind a long prompt | E@32k 44.61, E@128k 33.34 (128-token word-frequency summary) | ctx 28k: 70.8 (256-token reply, "one sentence + count to 100", README, older) | not comparable | Different reply kind, and the count part is high-acceptance |
| Cold prefill | 1,187-1,192 @ 32k, 1,183-1,190 @ 128k | 1,602-1,614 @ 24.5k, 1,577-1,606 @ 98k (W10 FIN) | **~1.35** | Different prompt lengths (both are flat across length), different filler. Both are cold with a unique prefix |
| Max context a request / slots | 327,680 / 2 sequences (372,877-token pool; 1M refused on their fp8 pin) | 1,048,576 / 4 slots sharing a 1M pool | | |
| Exactness | lossless in distribution; greedy is not reproducible run to run | drafted == serial byte-identical, same reply sha across loads | | |

## 3. Bytes per token and where their time goes

### 3.1 One prose verify step against one prose round (per rank)

Theirs comes from:

- the F3 profile boot (`evidence/e5-final/notes.txt`, "profile" and "census");
- the E1/E4 microbenches;
- `docker/README-v13.md`.

Ours is DECODE-PLAN §1 (W7 trace with the W9/W10 corrections).

| | Theirs (DFlash2-7, adaptive verify tau 0.3, 8 rows at full shape) | Ours (MTP + DFlash2, cost depth, mean window 3.7 rows) |
| --- | --- | --- |
| Routed experts, bytes | 24.1 distinct a layer (census) x 42 x 7.08 MB = **7.17 GB** | 809.5 expert-layer reads a round (19.3 a layer, incl. MTP steps) x 6.29 MB = **5.09 GB** |
| Routed experts, time | 33.6 ms profiled (**213 GB/s**) | 25.9 ms (**~197 GB/s**; 193 in W7) |
| Non-expert quantized weights | INT8 g128 on shared / MLA / KDA o / KDA in / lm_head (read twice: target + drafter) 4.06 GB, plus NVFP4 layer 0-2 MLP 0.13 GB = **4.19 GB in 18.6 ms (~225 GB/s)** | q4mse 4.5 bpw on everything incl. heads, MTP layer and DFlash2: **~3.1 GB in 16.0 ms (~190 GB/s)** |
| Remaining BF16 weights | fused_qkv_a, indexer, kv_b, KDA f_b / g_b, router: 6.3 ms (~1.3 GB, our estimate) | router + hc 0.17 GB (the small projections are q4) |
| Drafter | NVFP4 W4A16 0.34 GB, 5.2 ms bucket | inside the q4 line above |
| KDA state | an fp32 state per verify row: ~0.63 GB a step (review RF-6) | 71 MB read + written (commit replay, parity keys) |
| Communication | NCCL, ~91 all-reduces: **3.8 ms** | RoCE one-shot all-gather, 100 a round: **~1.7 ms** |
| KDA + mHC + indexer + logits + other | 8.8 ms | attention + indexer 2.1, hc + router 2.7, KDA 1.6, other 0.7 = 7.1 ms |
| GPU idle | 5.5 ms | ~3.8 ms |
| **Total bytes** | **~13.5 GB** (±1) | **~8.6 GB** |
| **Step / round** | **73.2 ms** unprofiled (79.4 profiled) | **~54.4 ms** |
| Tokens a step / round | 2.21 | 2.4 |
| **GB a token** | **~6.1** | **~3.6** (1.7x less) |
| **ms a token** | **33.3** | **22.4** (1.49x less) |
| Whole-step effective GB/s (bytes / wall) | **~185** | **~158** |

- **Expert bytes: they read 1.41x ours.**
  - 1.125x is the format (NVFP4 4.5 bpw vs EXL3 4.0).
  - 1.25x is more distinct experts a step. Their verify runs all 8 rows, and masked rows reuse the anchor's experts.
    Their mean live rows are 4.33, ours a 3.7-row window.
- **Routing cross-check.** Their single-sequence distinct-expert prefix curve is 8 / 13.5 / 17.7 / 20.8 / 22.8 at 1-5
  rows. Our measured U(R) is 8 / 12.6 / 17.0 / 21.5 / 24.4 (ROOFLINE §2.1). Two quants and two engines give the same
  routing locality, which supports the expert-byte model in ADAPTIVE-DRAFT and DECODE-PLAN.
- **Non-expert bytes: they read ~1.9x ours**, because they hold these weights at 8 bits or 16 (§4, item 5).
- **Everything that is not a weight stream costs them more:** 18 ms against 12.6.
- **The efficiency difference is in the weight-streaming kernels.** Their big streams run at 213-225 GB/s and ours at
  190-197. Nothing in their host, graph, speculation or communication setup is faster than ours.

### 3.2 Same shapes, same bytes: Marlin NVFP4 against our q4 `_qmm` (µs a GEMM, one rank, M = 8)

Sources:

- theirs: `evidence/e1-microbench/table.txt` (checkpoint weights, CUDA-graph replay, 50 iterations);
- ours: `results/W11/kbench.log`.

Both formats are 4.5 bits a weight: e2m1 + an e4m3 scale per 16, against 4-bit + bf16 scale and bias per 64. The byte
counts are therefore equal, and the time ratio is the efficiency ratio.

| Per-rank shape (N x K) | Where | Ours q4 | Marlin NVFP4 | Ours / Marlin | Calls a verify forward | ms a forward, ours → at Marlin's rate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 12576 x 4096 | KDA in_proj | 157.8 | 137.7 | 1.15 | 34 | 5.37 → 4.68 |
| 4096 x 4096 | KDA o_proj | 60.8 | 45.2 | 1.35 | 34 | 2.07 → 1.54 |
| 4096 x 8192 | DSA o_proj | 110.8 | 86.0 | 1.29 | 11 | 1.22 → 0.95 |
| 8192 x 1536 | DSA q_b | 41.2 | 35.0 | 1.18 | 11 | 0.45 → 0.39 |
| 2048 x 4096 | shared gate/up | 34.2 | 25.4 | 1.35 | 42 | 1.44 → 1.07 |
| 4096 x 1024 | shared down | 24.8 | 14.2 | **1.75** | 42 | 1.04 → 0.60 |
| 12288 x 4096 | dense MLP gate/up (their drafter gate/up) | 151.0 | 125.8 | 1.20 | 3 | 0.45 → 0.38 |
| 4096 x 6144 | dense MLP down (their drafter down) | 90.0 | 70.9 | 1.27 | 3 | 0.27 → 0.21 |
| 77440 x 4096 | LM head | 833.4 | 778.3 | 1.07 | 1 | 0.83 → 0.78 |
| **sum** | | | | | | **13.1 → 10.6 ms (-2.6 ms)** |

- The gap is largest on the narrow and short shapes: shared down, KDA o and shared gate/up. Those are the ones where
  our grid is 32-128 CTAs on 48 SMs. Marlin stripes K across SMs and reduces in a fixed order, so every SM streams
  even for N = 1,024-4,096.
- The MTP step and the DFlash2 block call the same kernel on smaller sets. That adds another ~0.3-0.5 ms.

## 4. What they do that we don't, item by item

"Applies" means under our rules: drafted == serial, batched == alone, resumed == fresh; no FP8 or FP4 activations.

| # | Technique (where) | What it does | Their receipt | Applies to us? | Gain for us (est.) | Effort |
| ---: | --- | --- | --- | --- | --- | ---: |
| 1 | **Marlin weight-only GEMM for every decode / verify linear** (`patch_v13_fp8.py`; vLLM `MarlinLinearKernel`, `apply_gptq_marlin_linear`) | Offline-permuted 4/8-bit weights for 128-bit loads, lop3 dequant in registers, multi-stage cp.async pipeline, K striped over all SMs with a serialized fixed-order reduction, BF16 `mma.sync` | NVFP4 / FP8 / INT8 at 210-231 GB/s on every large shape at M = 1-32 (`e1-microbench`, `e4-microbench`). The step model predicted -17.2 ms for their INT8 swap and E4a measured -17 ms | **Yes, as DECODE-PLAN E2.** There are two routes.<br>(a) Port the techniques into our own `_qmm`, keeping q4mse's bits.<br>(b) Call vLLM's `gptq_marlin_gemm` directly.<br>Route (b) has a catch. q4mse is 4-bit + a float scale and min per 64. Marlin's float-zero-point (HQQ) path is, as far as we know, fp16-activation only (check before building). With BF16 activations it needs integer zero points (the AWQ-Marlin layout), which means re-quantizing the BF16 non-expert weights at load, as q4mse already does. Row-invariant for M ≤ 16 if the config does not change inside 1-16 rows and `use_atomic_add` stays off (it is non-deterministic). Bits change against today (Marlin applies group scales in BF16 before the MMA), so this is a new-arithmetic change like 0060: re-gate MMLU / Tier 0, new reply sha, and drafted == serial must hold within it | **-2.6 ms a verify forward** (§3.2) + ~0.4 ms of draft steps: about **+5-6% single stream**, +4-6% at 4 streams (dense is 23 ms there at 146 GB/s) | 4-8 d |
| 2 | **Marlin MoE efficiency in the step** (`--moe-backend marlin`) | 5 launches a layer; `block_size_m` 8; fp32 reduce, no atomics | 213 GB/s in the profiled step (census bytes / bucket time) | Not the kernel (NVFP4, not EXL3 trellis). **It is the bar for E1.** Our `grouped_kernel` alone runs 205-222 GB/s (W11 kbench, R 3-16), but ~193-197 in the round. rot_in + gate/up epilogue + down epilogue are ~29 µs a layer = **1.2 ms a forward** of small launches (W11) | E1 mid (-2.7 ms) confirmed as reachable; aim E1 at fusing rot_in and the two epilogues into the grouped launch first | (E1) |
| 3 | **Fixed-shape adaptive verify** (`patch_v13_verify.py`, `GLM53_ADAPTIVE_VERIFY_TAU=0.3`) | Verify only the leading drafts whose running DFlash2 top-probability product ≥ tau. Masked rows keep the graph shape and take their anchor's top-8 experts at weight 0 | tau 0.2 over off: A +14.9%, H +20.0%, B +5.3%, T +10.1%. tau 0.3 over 0.2: A +5.2%, H +8.0%, B +4.3% | **Already ours, and stronger.** 0071 prices rows by drafter probability x per-position calibration and runs only the chosen rows (per-row graphs), so masked rows cost nothing, not even attention / KDA / shared / head. 0280 pads batch windows with the last real row's router logits, the same trick as their anchor reuse | 0 | - |
| 4 | **kpool tail ring sized for the verify window** (`patch_v13_kpool_tail.py`) | Fixes vLLM's DSA-indexer ring (4 slots addressed `pos % 4`), where rejected drafts and the other request overwrote committed slots, and pool keys built past 2,048 tokens compressed rejected tokens | CPU simulation: 18-53 of 66-154 committed pools differ on v11, 0 with the fix. Needles 3/3 at 8k-128k | Our indexer pools are our own code (0065 rings; `test_1m_patches` ring == full; `test_longctx_patches` drafted == serial at 2.5-7k on the **synthetic** checkpoint). **The lesson applies:** this bug is invisible below index_topk = 2,048, where every pool is selected. Our real-model `exact` suite uses short prompts | Correctness only. See N1 | 0.5 d |
| 5 | **Non-expert weights at 8 bits, gated by teacher-forced KL** (`TARGET_WEIGHT_GROUPS_INT8`, `quality/tier0.py`) | Keep KDA / MLA / shared / head at INT8 g128 (relative error 0.007) because 4.5-bpw NVFP4 on the same groups (error 0.085) failed their gate | E2b NVFP4 on all non-expert groups: KL top-20 **3.7e-2 against a 5.9e-3 A/A**, top-1 **95.6%** against 98.3%, dNLL +0.019: FAIL. FP8 per channel (error 0.026) also failed. E2c: the output-side groups (KDA o, MLA, shared) carried ~3.8e-3 of FP8's 5.2e-3 excess KL | **A quality question for us, not a speed lever.** q4mse is the same bit budget as their failed NVFP4 (4.5 bpw, g64 affine with MSE clip, so probably somewhat lower error). Our only gate is MMLU-200, where 13 of 200 answers flipped (87.0 → 88.0%). Measure it (N2) before deciding anything. If it fails, 8-bit on the output-side groups only costs ~+0.9 GB a round, **about -6% decode** | Measurement 1-1.5 d; the trade only if the measurement says so | 1.5 d |
| 6 | **Tier 0 / Tier 1 quality gates + boot-as-unit statistics** (`quality/`, `kit/compare.py`) | Tier 0 teacher-forces 96 frozen documents (186,526 tokens), taking KL over the reference's top-20 + rest, top-1 agreement, dNLL by domain, greedy "hazard" over 20 x 200 tokens, and probes. Gates are margins around the reference's own A/A. Tier 1 is 1,010 paired items (IFEval, GSM8K, MMLU-Pro, BFCL, ChartQA, OCRBench, MMMU): pooled paired difference ≥ -2 pt, plus McNemar p < 0.01 and ≥ 5 pt per group. `compare.py` treats the boot as the sample: KEEP / REVERT / INCONCLUSIVE with a ±4% band on single-boot arms | Caught every failing quant (E2a-E2d) that byte counts and smoke probes passed | **Yes.** The MIT code can be reused. Our API has no `prompt_logprobs`, but 0430's dump already writes the target's top-32 log-probabilities for every committed row, including prefill rows. So Tier 0's KL / top-1 / dNLL can be computed from two dumps (`GLM53_TF_FAST_PREFILL=0`, so the rows use decode arithmetic). 0430 is CPU-tested only so far, so its first GPU run is part of this item. We are deterministic, so their A/A margins collapse to zero noise, and the absolute floors become usable | Stronger quality gate than MMLU-200 for every future weight or kernel change | 1-2 d |
| 7 | **Ruler v2: forced 512-token prose over 8 prompts, per-step factorization, INVALID on swap growth** (`bench_decode.py`) | Forces the length (`min_tokens`), reports tokens a step, ms a step and per-position acceptance for every wave, and invalidates a cell if swap grows > 64 MiB | Their cross-boot band is ~±4%. Within-boot structured step_ms CV is 0.2-0.3% | **Yes.** Our cells are 64-200 tokens of one prompt each. The engine already returns `speculative` (rounds, drafted, accepted) per request, and `bench/acceptpos.py` exists | Makes every decode claim comparable with this kit and catches swap-noisy runs | 0.5-1 d |
| 8 | **PM QoS: hold `/dev/cpu_dma_latency` at 20 µs** (review XP-11, from their DSv4.1 sibling on the same hosts) | Keeps host cores out of the LPI-2/3 idle states (exit 231 / 433 µs on GB10) | Graph host-node wake 481-586 µs → 3.4-14.7 µs, about -1.1 ms a step (modelled). **Not serve-validated anywhere** | **Plausibly.** Each round has host syncs (drafting 1.5 ms of syncs; 0370 moved some off the path) and ~3.8 ms of GPU idle. Our RoCE proxy spins on its own core, but the serve thread and rank 1's plan thread block. Host config only, same bits | 0-1 ms a round: **+0-2%**. Risk: more idle power and heat on nodes already at 79-83 °C under the `-lgc` cap | 0.5 d + 20 min GPU |
| 9 | **PDL gated off on SM12x** (`Dockerfile.sm121-v8`: "unvalidated on SM12x and races KDA state kernels") | vLLM's `is_arch_support_pdl` limited to majors 9 and 10 | No repro or trace in the repo (review NMK-10: "keep PDL off; re-enabling needs a KDA state race analysis") | **A caution for E4.** Our W11 probe shows PDL works on this GPU: a graph chain gap of 0.79 → 0.44 µs a kernel, and 399/399 early starts on 1-4 MB kernels. Their warning names the risky spot: any kernel that reads the previous kernel's output (KDA state, chain, replay) must execute `griddepcontrol.wait` before its first dependent load. E4's bitwise tests must run under PDL with a busy stream | Keeps E4's estimate; adds a test requirement | (E4) |
| 10 | **Large-M dequant → cuBLAS for prefill** (`GLM53_WQ_DEQUANT_MIN_M`, E6) | Dequantize `kda_in` to a BF16 workspace for M ≥ 512 | Microbench: -2.48 ms a GEMM at M = 1,152. End to end only +6.4% / +3.5% at 32k / 128k (+9.5% at 7k), so reverted; the length dependence is unexplained | Our analogue is MIA-AUDIT's later item 2. Our fast-prefill `_fq4` already dequantizes in-kernel at 51-56 TF/s on the KDA projections, so the vLLM gap it closed (Marlin re-streaming weights per 64-row block) is not ours. **Lesson:** microbench prefill gains shrank 2-3x at long context. Profile a 32k prefill before building | Low | - |
| 11 | **Keep GEMV kernels out of prefill** (E5 diagnosis) | INT8 Marlin re-streams a weight that misses L2 once per 64-row M block: `kda_in` 3.6x BF16 at M = 1,152, and the head 3.2x | -10.5% prefill for their INT8 swap | **Yes, if item 1 lands:** dispatch by M. Marlin for ≤ 16-row decode / verify / draft calls; `_fq4` for fast-prefill chunks. Exact prefill (the `exact` path, M = 64) must use the same kernel as decode if exact snapshots are to keep today's rule | Avoids a prefill regression | (in item 1) |
| 12 | **Swap / reclaim hygiene** (review SYS-2, PROTO-6) | swappiness 60 → low with swap kept on; gate benches on active paging (vmstat si/so), not swap level | spark1 had ~1.15 TiB of lifetime swap-out; multi-second TTFT stalls on a warm serve. Swap level alone: no measured cost (120.2 vs 120.3 ms with 7.3 GiB swapped) | Partly ours: MEM_GATE, stress floors, gpuwatch. Missing: si/so in the bench record and an INVALID rule | Variance, not mean | 0.5 d |
| 13 | **RoCE one-shot collectives elsewhere: known failure modes** (review SYS-6, EXT-12) | - | sparkring: "peer-wait timeout poisons runtime after 1.5 days / 18.5M collectives"; b12x #313: "graph-replayed collective can wedge one rank" | We run 0230/0350 in production with a fallback marker. A soak past 2 days and ~20M exchanges has not been reported | Reliability | (E8) |
| 14 | **Expert census with an oracle bound** (`patch_v13_census.py`, `tools/census_report.py`) | Records every verify row's top-8 (rejected rows included) and prices the SD-1 oracle ("keep only sampled rows") | Prose: 36.7% of routed bytes are the oracle's floor | Ours already: `experts_union.py`, `dec_bytes.py` (W7 traces), and the ADAPTIVE-DRAFT simulator's oracle gap | 0 | - |
| 15 | **Build once, ship with `docker save \| docker load`; per-image JIT cache; MHC warmup** | Identical image IDs on both ranks; no compiler at boot | Ready 21.2 → 16.3 min; first-c=2 TTFT 8.7 → 0.6 s | Ours already (`serve.sh build`, the `glm53-tf-cache` volume, `WARMUP_LENGTHS`); our loads are 23-40 s | 0 | - |

## 5. What they do worse, or where we are already ahead (brief)

- **Speculation.** Their DFlash2-only prose step commits 2.21 tokens; our MTP + DFlash2 with cost depth commits 2.4.
  They cannot run MTP on the nvidia pack: its layer-45 MTP weights are 13.84 GiB of BF16 that do not load. The one
  same-session MTP-4 vs DFlash2 comparison on the LibertAI pack gave 2.175 vs 2.292.
- **Verify shape.** Masked rows still run attention, KDA, the shared expert and the head at full shape (8 rows). We run
  only the rows we verify.
- **Communication.** Their NCCL all-reduce costs 3.8 ms a step; our RoCE all-gather ~1.7 ms. Custom all-reduce and
  symm-mem are unavailable on 12.1, and their sibling measured dual-rail and NCCL knob sweeps as null or negative.
- **KDA state.** They write one fp32 state per verify row (~0.63 GB a step, review RF-6). We replay commits with parity
  keys.
- **Draft head.** Their drafter re-reads the full target head in INT8. Ours reads a 4-bit copy, and 0420 trims the
  vocabulary.
- **Concurrency and context.**
  - Their KV is pinned at 4.14 GiB: 2 sequences and 327,680 tokens, with 1M refused and a 4-sequence boot that failed
    its memory gate.
  - Ours is 4 slots sharing a 1M pool, 1M a request, sessions on NVMe and shared-prefix marks.
- **Prefill.** 1.35x ours is slower, and their INT8 swap cost them 10.5% of it.
- **Determinism.** Their greedy output is not reproducible run to run (14/20 prompts diverge within one boot). Every one
  of our gates is a byte-identical hash.
- **Load time.** Their ready time is 15-17 min with a warm JIT cache (the head reads weights at ~730 s); ours is 23-40 s
  (0140).

## 6. Ranked adoptions, mapped onto DECODE-PLAN

| Rank | Item | Maps to | What changes in the plan | Gain (est.) | Effort | Gate to adopt |
| ---: | --- | --- | --- | --- | ---: | --- |
| 1 | **Marlin-class dense GEMV**: port Marlin's layout, pipeline and K-striping into `_qmm` (keeps q4mse), or link vLLM's `gptq_marlin_gemm` for M ≤ 16 (needs an integer-zero-point re-quantization with BF16 activations). Keep `_fq4` for prefill | **E2** | E2 moves from an estimate to a demonstrated bar on this GPU and these exact shapes: -2.6 ms a forward at Marlin's speed (§3.2), against E2's mid of -1.9. Start with shared down, KDA o and shared gate/up (1.35-1.75x) | 1 stream **+5-6%**, 4 streams +4-6% | 4-8 d | Bitwise row-invariance 1-16 rows (0170 / 0260 pattern), drafted == serial, batchexact, MMLU / N2 on the new bits, 1 stream ≥ +3% |
| 2 | **N1: real-model drafted == serial past 2,048 tokens.** Add `exact` cases to `glmbench` with 8k and 32k prompts, 512-token greedy and sampled replies, and a 4-slot batch variant | new (correctness) | The class of bug their kpool fix found: rejected rows corrupting indexer pool keys that only rank above 2,048 tokens | Correctness | 0.5 d + 20 min GPU | 100% hash equality |
| 3 | **N2: Tier 0 for q4mse.** 0430 dumps of their 96-document corpus (MIT, `quality/data/corpus.jsonl`) on `NONEXPERT=bf16` and `q4mse` loads, exact prefill. Compute KL top-20, top-1 and dNLL with their `tier0.py` formulas | new (quality) | Their data says 4.5-bpw non-expert weights cost ~6x the A/A KL on this model. We have never measured ours beyond MMLU-200 | Decides whether q4mse stays or the output-side groups go to 8 bits (-6% decode) | 1-1.5 d + 2 loads | Report; decision is the user's |
| 4 | **E1 retargeted**: fuse rot_in and the gate/up and down epilogues into the grouped launch (1.2 ms a forward of ~10 µs launches, W11), then attack ramp and tail | **E1** | Bar: Marlin MoE's 213 GB/s in their step, against our 193-197. Our kernel alone already reaches 205-222 | 1 stream +4%, 4 streams +5-7% | (E1: 7-12 d) | As E1 |
| 5 | **Re-base T1 / T2 with their per-position curves** (docs only now; 20 min GPU to confirm) | **T1, T2, T4** | DFlash2 on the **stock** NVFP4 target, prose, thinking off, k=7: .640 .345 .169 .065 .029 .012 .003 (E0 A, 2.29 tokens). Code: .817 .657 .515 .406 .322 .238 .182. Thinking on (tau 0.3): .682 .412 .240 .133 .079 .050 .030. That base-model prose curve is below our MTP head's .74 / .45 / .22, and matches our 2.0-2.7-token prose DFlash2 rounds. **So abliteration is not what keeps DFlash2 weak on prose; the regime is.** T2's prose share (+10-20%) should come down, T1 (the prose drafter) goes first, and T4 (block 16) keeps its code case (.18 still accepted at position 7) | Changes priorities, not ms | 20 min GPU: run `bench/acceptpos.py` on their 8 prose prompts with DFlash2 forced | - |
| 6 | **N3: their ruler in our bench.** Cell A's 8 prompts x 512 forced tokens (and B, J), per-round ms / tokens a round / per-position acceptance per cell, INVALID on swap growth, and `kit/compare.py`-style boot-as-unit verdicts with ABAB for any < 3% decision | new (method) | Makes "us vs them" a single-table fact, and fixes the §2 caveats (code 77.6 is a raw completion; the prose cells are short) | Method | 1 d | - |
| 7 | **N4: PM QoS A/B** (`/dev/cpu_dma_latency` held at 20 µs on both hosts) | **E6** neighbour (host latency) | Config-only probe of whether host wake latency is part of our 3.8 ms idle | +0-2% | 0.5 d + 20 min GPU | ≥ +1% ABAB and temperature not higher than today's 79-83 °C |
| 8 | **E4 test requirement from their PDL gate** | **E4** | E4 keeps its estimate (our probe: -0.35 µs a kernel in graphs), but its bitwise tests run with PDL on, a busy stream, and every KDA state / chain / replay consumer checked for `griddepcontrol.wait` before its first dependent load | - | - | - |
| 9 | **E8 soak** | **E8** | A ≥ 48 h, ≥ 20M-exchange RoCE soak with graph replay before tightening the proxy loop (their ecosystem's two known failure modes) | Reliability | GPU time | 0 timeouts |

Not adopted, and why:

- **INT8 non-expert weights for speed.** They are the other direction for us (+1.7 GB a round).
- **Their adaptive verify.** Our cost depth already skips rows entirely.
- **NVFP4 drafter weights.** Ours are already 4-bit.
- **Prefill dequant.** Our `_fq4` does not have the problem it fixed.
- **Everything vLLM-specific:**
  - the NoPE FA2 backend patch;
  - the FlashInfer 0.6.18 pin;
  - the DFlash2 backport and aux hidden states;
  - the draft KV group;
  - the indexer workspace factor;
  - KDA `.contiguous()` trims;
  - `SKIP_MTP_WEIGHTS`;
  - the DFlash prefix-cache fix;
  - the chat-template `enable_thinking` alias;
  - the KV pin;
  - breakable graphs;
  - the `_compact` allocator fix.

## 7. What needs a GPU run before relying on any of this

1. Their cell A prompts (512 forced, greedy, thinking off) on our production server. This gives the real prose ratio
   for §2; expect ~1.4-1.5x.
2. `bench/acceptpos.py` over the same 8 prompts with the drafter forced to DFlash2 and to MTP, which gives our
   per-position curves beside theirs (§6 rank 5).
3. The §3.2 table re-timed in one window: our `_qmm` and vLLM's `gptq_marlin_gemm` (4-bit, g64, bf16 activations) on the same
   GPU, same shapes, M = 1 / 4 / 8 / 16. Plus a row-invariance check of Marlin across M = 1..16 with atomic-add off.
4. N1 and N2 as above.

## Sources (their repo, `072c3f6`)

- `README.md`, `AGENTS.md`, `recipe.yaml`, `run.sh`, `LICENSE`
- `docker/Dockerfile.sm121-v8` (PDL gate, FlashInfer / NCCL pins), `docker/README-v13.md` (adaptive verify, kpool tail
  ring, weight-only Marlin, large-M dequant), `docker/patch_v13_*.py`
- `bench_decode.py`, `kit/compare.py`, `quality/README.md`, `quality/tier0.py`, `quality/tier1.py`, `tools/README.md`
- `evidence/e0-nvidia-v11/SUMMARY.md`, `evidence/e1-microbench/table.txt`, `evidence/e4-microbench/table.txt`,
  `evidence/e3b-av-tau0.2/notes.txt`, `evidence/e4a-int8/notes.txt`, `evidence/e5-final/notes.txt` (F3 step buckets and
  census), `evidence/e6-prefill/notes.txt`, `evidence/e6-prefill/prefill-table.txt`,
  `evidence/e6-prefill/compare-f12g2-vs-e0.txt`, `evidence/decision.tsv`, `evidence/architecture-model.md`
- `evidence/review-20260925/findings/*.md`: byte-roofline RF-1/4/5/6/11/12, moe-kernels, attn-kernels NMK-10,
  system-uma-comm SYS-2/6/16, sibling-recipes XP-10/11, spec-decode SD-1/3/6/10, quality, regression-bench

This repo: DECODE-PLAN §1 / §3, ROOFLINE §2, PROFILE §5, RESULTS (README table, W10), PATCHES table, MIA-AUDIT,
`results/W10/conc-FIN.log`, `results/W11/kbench.log` and `results/W11/probe-head.log` (uncommitted W11 files).
