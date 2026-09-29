# Decode plan: how a +40% decode target decomposes on 2x DGX Spark (2026-09-29)

> Research and arithmetic only. No GPU was used and the Sparks were not touched. Inputs:
>
> - This repo's docs: ROOFLINE §2 (decode rounds and floors), PROFILE §5, RESULTS W7-W10, RESEARCH-NIGHT,
>   ADAPTIVE-DRAFT, DEEP-VERIFY, DECODE-OVERLAP, DECODE-ANALYSIS, COMM-ANALYSIS, EXPERIMENTS, PATCHES and
>   `config/prod.env`.
> - `results/W10/conc-*.json` (round kinds).
> - Web research on 2025-2026 work (Sources at the end).
>
> "Exact" means drafted == serial, batched == alone and resumed == fresh all still hold, with the same EXL3 4-bit
> weights. Every gain below is **derived from the measured round decomposition**, not measured itself. Section 5
> lists the measurements that would turn the estimates into numbers.

## 0. Bottom line

- **+40% is reachable on paper, but only by stacking two independent families, each at its middle estimate:**
  - Engineering takes ms off each round: about +23% single-stream and +28% at 4 streams.
  - Better drafters commit more tokens a round: about +13% single-stream and +14% at 4 streams on their own.
  - Neither family reaches +40% alone. Engineering alone would need the round at ~39 ms against a 37.8 ms floor.
    A drafter alone would need prose rounds to commit ~3.5 tokens instead of 2.4, a jump no published recipe gives
    on a model that is already MTP- and DFlash2-drafted.
- **Middle-case stack:**

| workload | today (est.) | engineering only | drafters only | both | honest range, both (low / high) |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 stream, prose (2.4 tokens a round) | 54.4 ms a round, **44 tok/s** | 44.4 ms, 54 tok/s (+23%) | 50 tok/s (+13%) | **61 tok/s (+38%)** | +19% / +64% |
| 1 stream, code / structured (5.7 tokens a round) | ~70 ms, ~82 tok/s | +15-22% | +8-15% (block-16 drafter) | +25-40% | |
| 4 streams, aggregate | 119 ms, 9.1 tokens a round, **76.5 tok/s** | 93 ms (+28%) | +14% | **110 tok/s (+44%)** | +19% / +76% |

  The high ends are bounds: they put the round at 97-100% of the bandwidth floor, which nothing on GB10 reaches.
  The realistic reading: **+25-40% single-stream prose and +30-45% at 4 streams.** +40% single-stream needs every
  engineering item at its middle estimate *and* a self-distilled drafter.
- **The three largest single items, in order:**
  1. **Drafters trained on the abliterated target's own outputs.** Two parts:
     - T1: MTP self-distillation, FastMTP recipe, days.
     - T2: a block-drafter re-fit, one to two weeks.

     Worth +7-19% single-stream alone and the multiplier on everything else. Our stock MTP acceptance
     (0.74 / 0.45 / 0.22) is the same as vLLM's on GLM-5.3-Flash (0.70 / 0.45 / 0.26), which is exactly the
     weak-deep-position profile FastMTP and Red Hat lift. DFlash2 reads target taps that the abliteration shifted
     (a sibling abliteration measured cosine 0.926 at L42).
  2. **Routed-expert and dense decode kernels from 190-193 to ~215 GB/s** (E1 + E2): -4.6 ms of a 54 ms round,
     -11 ms of a 119 ms 4-stream round. The same GB10 reaches 224-233 GB/s on a plain GEMV and 231 GB/s end to end
     on a dense NVFP4 model in SGLang, so ~215 on EXL3 grouped reads is plausible, not proven.
  3. **The per-slot fixed cost at 4 streams (F3):** 6.6 ms a slot, measured in W9. Fused multi-slot
     KDA / attention / indexer / commit launches are worth -9 ms a 4-stream round.
- **First steps for the next session**, each small and each deciding a larger item (§5):
  - one nsys + ncu capture of today's production (b4 + FIN knobs);
  - the draft-head vocabulary trim (E7), a 2-3 day exact patch;
  - the MTP self-distillation data and reference path (T1), days 1-2;
  - a roof-probe microbench for the expert and q4 GEMV kernels (E1 / E2 go / no-go);
  - `GLM53_TF_BATCH_GRAPHS=0` or a higher `CAPTURE_AFTER` at 4 streams (config only).

## 1. Today's decode round, reconstructed after W10

W7's trace (ROOFLINE §2.2) predates RoCE (W9), 0370 overlap and 0390 (W10). The table below applies each change's
measured or bounded delta to W7's kernel families, then the capture correction (x 0.98). **It is a reconstruction;
§5 step 1 replaces it with a measurement.**

| 1 stream, prose (2.4 tokens, mean window 3.7 rows) | W7 ms | change since | now (est.) ms | floor ms (230 GB/s) |
| --- | ---: | --- | ---: | ---: |
| routed experts (`grouped_kernel` + epilogues; 5.09 GB, 193 GB/s) | 26.43 | - | 25.9 | 22.1 |
| dense q4 GEMMs (`_qmm`; ~3.1 GB, 190 GB/s) | 16.29 | - | 16.0 | 13.5 |
| exchanges (100 a round) | 4.77 | RoCE: W9 rounds 3-5 ms shorter | ~1.7 | 0.2-0.5 |
| GPU idle inside the round | 4.65 | 0370: -0.6..-1.1 ms | ~3.8 | 0 |
| DSA attention + indexer | 2.60 | 0390: expand 114 -> 68 us a call | ~2.1 | 0.2 |
| hc + router / combine | 2.76 | - | 2.7 | 0.74 |
| KDA chain / conv / replay | 1.60 | - | 1.6 | 0.65 |
| other + gap between rounds | 0.68 | 0370 `plan`: gap 0.9 -> ~0.3 | 0.7 | 0.1 |
| **round** | **59.5** (capture) | | **54.4 = 44 tok/s** | **37.8** |

Cross-checks:

- W10's 1-stream medians are 52-53 tok/s over a prompt mix: prose reps 43-52, code / sequence reps 71-97.
- The task's "prose rounds ~2.4 tokens, ~55 ms" matches.

**What +40% means in ms.** At constant tokens a round, +40% needs the round at 54.4 / 1.4 = **38.9 ms, i.e. at 97%
of the 37.8 ms floor.** That is not available: our best rounds (canary) run at 72-82% of their floor, and the best
published decode kernels anywhere run at ~78-85% of DRAM peak (Hazy's megakernel 78% on H100; SGLang 85% on GB10,
dense). So +40% on prose has to be split between ms a round and tokens a round.

4 streams (ROOFLINE §2.3; W7 125.5 ms under capture, minus ~3 ms RoCE and ~1 ms overlap): **~119 ms, ~9.1 tokens a
round, 76.5 tok/s** (W10 FIN median 78.1). The floor is 79.5 ms: routed experts 13.81 GB = 60 ms, dense 14.6 ms,
the rest ~5 ms.

## 2. What others reach (calibration)

| claim | number | relevance here |
| --- | --- | --- |
| GLM-5.3-Flash on 2 Sparks, other kits | MiaAI EXL3 + DFlash2: ~37 tok/s prose, 62.9 structured. vLLM NVFP4 + MTP-3 over RoCE: 20-23 tok/s, MTP per-position acceptance **0.70 / 0.45 / 0.26** | We lead single-stream by 1.2-2x. Our MTP (0.74 / 0.45 / 0.22) is the stock head's ceiling everywhere: nobody has a better one for this model yet |
| Attainable GB10 bandwidth | GEMV microbench **224-233 GB/s** (82-85% of 273) in the fast state, 66-80 in the hidden slow state. Qwen3.8-27B NVFP4 end to end in SGLang: 18.77 GB a token at 12.32 tok/s = **231 GB/s** | Our big decode kernels run at 190-207. A whole-model 231 GB/s on GB10 shows ~215-225 is not a fantasy for large GEMVs |
| MoE decode kernels | MonoMoE (weight-major persistent MoE, FlashInfer): vLLM's Triton fused_moe reaches only 10.6 / 19.4 / 30.8 / 45.1% of H200 peak at batch 1 / 2 / 4 / 8; MonoMoE 21-58%; kernel 1.02-1.54x | Our EXL3 grouped kernel is already at 83-84% of attainable. The weight-major idea helps at 4 streams (U ≈ 51 experts a layer, few rows each); expect +5-15% on the kernel, not 1.5x |
| Megakernels | Hazy "No Bubbles" (Llama-1B): 78% of H100 bandwidth vs ~50% for vLLM / SGLang; 1.3 us per graph kernel launch still paid. Hazy TP8 70B: +22% throughput. MPK (Mirage, OSDI'26): up to 1.7x lower latency, no MoE / linear-attention support | We already run 92% GPU-busy at ~84% bandwidth on the big kernels. A megakernel would buy the 7-9 ms of small-kernel, idle and exchange time (E3-E6, E8), not 1.7x. PARADIGMS #12: +10-18%, 25+ days |
| 2-Spark small-message latency | RDMA write floor ~2 us; NCCL all-reduce 16-32 KB 40-43 us | Our RoCE all-gather is 11.7 / 16.9 / 20.3 us at 16 / 64 / 128 KiB. ~10-15 us a hop above the wire floor is proxy / flag overhead: the E8 pool |
| EP vs TP at batch 1 | EP loses: two all-to-alls a MoE layer, poor load balance at batch 1; Qwen3.8 TP4-EP decodes slower than TP2 | Confirms PARADIGMS #15: TP=2 stays |
| Clock state | `-lgc` works, `-pl` does not; the hidden slow state costs 30-50% decode; 83 °C steps the clock down 150 MHz every 30 s | Prod runs `-lgc 300,2250` at 79-83 °C under load. Not a speed lever (decode is DRAM-bound); a hygiene item (§3.4) |

## 3. Candidates

Columns:

- **ms**: saved from the round in the 1-stream (1s) and 4-stream (4s) models of §1, as low / mid / high.
- **%**: the middle estimate alone.
- The effort scale is engineer-days, excluding GPU windows.

### 3.1 Engineering: fewer ms a round (all exact unless marked)

| # | lever | mechanism | evidence | 1s ms (low / mid / high) | 4s ms | exactness | effort | risk |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| E1 / F1 | **Routed-expert decode kernel 193 -> 205 / 215 / 225 GB/s** | One persistent launch per MoE layer: gate/up -> act -> down chained per expert with device flags, so the down pass has no launch ramp or tail. Rot_in / plan / epilogues fused in; PDL on the launch edges; tiles striped evenly over the 48 SMs by distinct-expert count U(R). At 4 streams, weight-major order (MonoMoE-style): each trellis tile is decoded once for all its member rows, rows on the MMA N dimension | R = 1 calls run at 200 GB/s (gate/up 167.5 us, down 83.8 us a layer). 84 launches a round at ~80-170 us each, so ramp and tail are 3-6% of each. GB10 GEMV 224-233; SGLang dense 231 end to end; MonoMoE | 1.5 / **2.7** / 3.8 (5.09 GB) | 4.9 / **8.1** / 10.9 (13.81 GB) | Same bits if each output keeps its k-ordered chain (the 0170 / 0260 bitwise test pattern; the 16+-member `grouped_loop` path already proves it for large windows) | 7-12 | Medium: EXL3 trellis decode ALU at small R; 0330 showed warp-specialized designs can lose on GB10 (register file per sub-partition) |
| E2 / F2 | **Dense q4 GEMV family (`_qmm`) 190 -> 215** | Pre-shuffled q4 weights (128-bit loads, no shuffles), a persistent grid sized to 48 SMs, and the small projections of one layer (DSA q_a / kv_a / indexer: grids of 12-128 today) grouped into one launch. At 4 streams, dense is 23 ms against a 14.6 floor (146 GB/s): the batched MTP passes and per-slot calls are the gap | Head `_qmm` 207 GB/s, KDA qkv 193; small-grid calls leave SMs idle | 1.2 / **1.9** / 2.5 | 1.5 / **3.0** / 6.0 | Same bits (keep `matmul_fast`'s per-output k order; no split-K) | 5-7 | Low |
| E4 | **Small-kernel fusion + Programmatic Dependent Launch** | Attention + hc + router / combine + KDA + other are 6.9 ms against a ~1.7 ms floor, and a round is 1,846 kernels. Fuse `hc_post` -> `hc_pre` -> RMSNorm -> router into one kernel a layer boundary (decode's `hc_fused`). Add PDL (`griddepcontrol`, captured in graphs) so each kernel's prologue overlaps the previous tail | Hazy: the per-kernel bubble is the gap between 50% and 78% of peak. SGLang ships PDL in its DeepSeek fused kernels. PARADIGMS #7 | 0.8 / **1.5** / 2.3 | (in F3) | Fusion: same bits if each element's op order is kept (bitwise tests as 0390 / 0400). PDL: scheduling only | 7-10 | Low-medium (sm_121 PDL support to confirm) |
| E5 | **Graph coverage for lone decode** (no eager or capture rounds in steady state) | In W7, 8 of 106 single-stream rounds (eager / capture) held 55% of the round's idle, 32-35 ms each. W10 FIN still has 2-17% eager + capture rounds per 1-stream rep (`conc-FIN.json`: 6 / 212, 58 / 350, 24 / 144). 16-row windows and power-of-2 context buckets add keys. Pre-capture rows 1..16 x both parities for the current and the next bucket while the slot is idle, instead of on the first sighting | `results/W10/conc-*.json` `round_kinds`; DECODE-OVERLAP §2 | 0.5 / **1.0** / 2.0 | ~0 (W5: 4-slot rounds are GPU-bound, eager costs nothing) | Same code path, graph or not (0200) | 2-3 | Low (graph memory; `LONGCTX_MAX_GRAPHS` 256) |
| E6 / F4 | **Device-side sampling and draft chains** | Drafting syncs are 1.5 ms median a round (1s) and 3.0-3.2 ms (4s): MTP d + 2 hard syncs, DFlash2 `candidates` readbacks. A GPU port of `choose_rows` (greedy argmax; keyed splitmix64 / float64 Gumbel for sampled rows), the MTP chain as one graph, DFlash2 chain selection on the device. At 4 streams, add batched DFlash2 blocks (ADAPTIVE-DRAFT §7: +2%) | DECODE-OVERLAP §2 (idle split); DECODE-ANALYSIS §3a; SGLang Spec V2 overlap scheduling | 0.6 / **1.0** / 1.5 | 1.5 / **3.0** / 5.0 | The device sampler must reproduce the host's keyed draw bit for bit (tested over millions of draws), or become the reference for serial too | 7-10 | Medium (sampler equivalence) |
| E7 | **Draft-head vocabulary trim** (drafts only) | An MTP step reads ~270 MB a rank, ~165 MB of it the 154,880-row head. A DFlash2 block runs the head for 7 positions (~0.8 ms). Draft over the top 32k tokens by frequency in our model's replies (FR-Spec); verification keeps the full head | FastMTP's 32k-vocab compression costs 0.03-0.07 τ. FR-Spec: -75% LM-head compute, ~1.12x over EAGLE-2 (not re-verified this session). The tonyd2wild Qwen3.8 Spark kit ships a reduced-vocab draft. DECODE-ANALYSIS §3c | 0.5 / **0.8** / 1.2 | 0.6 / **1.2** / 2.0 (batched MTP reads the head once a step) | Exact: drafts only; a token outside the subset only ends a chain | 2-3 (+ offline frequency pass) | Low. Coverage must be ≥ 97-98% of reply tokens |
| E3 / F7 | **L2 prefetch during latency windows** (0040 revived) | During each exchange and each small-kernel stretch (~9 ms a round with DRAM mostly idle), stream the next weights (shared-expert gate/up, KDA projection head tiles, router) into L2. 0040 exists, was never measured, and sets **no prefetch sites in `compute_multi`**. With `GLM53_TF_BATCH=4`, production's rounds go through the Batcher, so 0040 must be ported first | COMM-ANALYSIS §3: upper bound 90 x 18 us = 1.6 ms (NCCL-era alpha); RoCE shortens the exchange windows, but the hc / router / attention windows remain | 0.3 / **0.7** / 1.5 | 0.3 / **0.8** / 1.5 | Exact (loads into a sink) | 3-4 | Low (L2 thrash; measure) |
| E8 / F5 | **RoCE residual latency** | 11.7-20.3 us an exchange against a ~2 us RDMA floor. Tighten 0230's proxy loop and flag path (the GPU writes the flag in the producing kernel's epilogue; the proxy spins on one cache line on its own X925), and cut the peer-wait skew (~half of W7's exchange time) | W9 bench (RoCE 11.7 / 16.9 / 20.3 us vs NCCL 45 / 76 / 66); Sangiorgi's 2 us `ib_write_lat` | 0.2 / **0.4** / 0.8 | 0.3 / **0.8** / 1.5 | Exact (a copy) | 3-5 | Medium (0230's reliability history: soak again) |
| F3 | **Per-slot fixed cost at 4 streams** | W9 measured ~6.6 ms a slot a round on top of rows: 26 ms of a ~115 ms 4-stream round. That is per-slot KDA chains, attention / indexer launches over 45 layers with small grids, and commit replay. One launch per layer across slots: `chain_kernel` grid (heads, slots) over a slot table; attention / indexer with a per-row sequence id; a batched KDA replay | RESULTS W9 §5 (fit 7 + 6 confirmed: 33 / 45.5 / 69.3 ms); ADAPTIVE-DRAFT simulator: halving it = +6% (finite bench), more in steady serving; 0200 design notes | - | 5 / **9** / 13 | Same bits per row (row-local kernels; the 0200 / 0290 row-independence tests) | 10-14 | Medium (`kda.cu` work) |
| F0 | **4-stream graph policy** (config only) | W5: at 4 slots, capture rounds cost a full extra forward (8-12% of rounds in W10), and eager rounds cost nothing. `GLM53_TF_BATCH_GRAPHS=0` measured +3.5% (inside the spread, not A/B'd for exactness); a higher `CAPTURE_AFTER` is the gentler form | RESULTS W5; `conc-FIN.json`: 70-158 captures per 4-stream rep | - | 0 / ~2 / 4 | Same code path | 0.5 | Low |

**Ranges.** Engineering stacked (the ms add; they are distinct kernels and syncs):

- **1 stream:** low -5.6 ms (+12%), **mid -10.0 ms (+23%)**, high -15.6 ms (+40%, a bound: 97% of floor).
- **4 streams:** low -14 ms (+13%), **mid -26 ms (+28%)**, high -40 ms (+50%, at the floor).

### 3.2 Drafters: more tokens a round (all exact: drafts only propose)

| # | lever | mechanism | evidence | gain here | effort | risk |
| --- | --- | --- | --- | --- | ---: | --- |
| T1 | **MTP self-distillation** (FastMTP recipe on the abliterated target) | Fine-tune the MTP layer's attention, `eh_proj`, norms and shared expert (routed experts frozen, dequantized to bf16 for the backward pass, ~14.5 GB) on the target's own continuations. Use recursive 3-step training-time test (its own drafts as inputs) and a KL loss to the target's distribution at positions t+2..t+4 | FastMTP: 389k self-distilled samples, **< 1 day on one H20 server**; acceptance 70 -> 81 / 11 -> 56 / 2 -> 36% at positions 1-3; 2.03x vs NTP. Red Hat on Qwen3-Next-80B-A3B: 0.897 / 0.719 / 0.476 -> 0.912 / 0.776 / 0.616 with **~5k samples in 443 s on 2x H200**. MTP-D: +7.5% acceptance. GLM-5 trained shared-parameter multi-step MTP (τ 2.76 at 4 steps) | a = 0.74 / 0.45 / 0.22 -> ~0.80 / 0.58 / 0.38: MTP-round E[T] 2.41 -> ~2.76 for ~+2.5 ms (a chained step + a row): **+9-10% on MTP rounds**. That is sampled cells (MTP only) +8-12% and greedy prose (about half MTP rounds) +4-6% | 5-8 (reference MTP forward in PyTorch: DSA at ≤ 2,048 context is dense MLA, so exact; data hook; trainer) | Low-medium (the first attempt may undershoot on prose; FastMTP's biggest lift is at positions 2-3, which is where ours is weak) |
| T2 | **Block drafter re-fit to the abliterated target** (on-policy distillation) | Fine-tune a DFlash-style block drafter on teacher-forced taps (layers 5 / 14 / 24 / 33 / 42) and target logits of the target's own replies, with KL and a confidence / stop head (DSpark-style). Start from `canada-quant/GLM-5.3-Flash-DFlash2-G` (Apache-2.0) or RedHat DSpark (MIT). incoai's DFlash2 is CC BY-NC-ND: private use only, no derivative may be shared | EAGLE-3: training on target-regenerated data plus training-time test gives +30-50% τ over EAGLE-2. DFlash: τ 4.35-7.84 with 800k target-generated samples. DFlash τ scales steeply with data (1.06 -> 2.47 -> 6.12 at 694 -> 20k -> 1.3M samples, **unverified secondary**). Ours: taps drift under abliteration (EXPERIMENTS S4: cosine 0.926 at L42). Prose rounds keep 2.0-2.7 against DFlash2's published MT-Bench τ 4.19 (base GLM-5.3) | Prose DFlash2 rounds +0.4-0.7 tokens: **+10-20% prose alone**. Code +5-10% | 7-12 | Medium: training compute on GB10 (below); acceptance on prose is the least certain number here |
| T3 | **Calibrated stop / confidence** | 0071 prices each row from the drafter's own probability times a per-position correction. A learned stop head (SpecDec++-style) closes part of the oracle gap: +18.8% prose at 8 rows, +15-21% at 4 streams, all of it unreachable by policy alone (ADAPTIVE-DRAFT) | SpecDec++: +7-11% over fixed K; DSpark's confidence-scheduled verification | +3-6% on top of T2, trained with it | +2-3 on T2 | Low |
| T4 | **Block-16 drafter for code / structured** (DBloom curriculum B8 -> B16) | 0380's 16-row windows are in production, but DFlash2 proposes at most 7. 33-40% of code-like and 81-83% of repetitive DFlash2 rounds keep all 8 | DBloom: +0.8 τ median (up to +1.37) from a short post-training curriculum; naive widening at inference fails. DEEP-VERIFY: block 16 at tail 1.0 = +26% repetitive, +2.7% code | Code / structured **+8-15%**, prose ~0 | +3-5 on T2 | Medium |
| T5 | Expert-union-aware draft pricing (EcoSpec-lite, chain-only) | Price a draft row by its predicted *new* experts; at 4 streams, choose which slot goes deeper by expected overlap | EcoSpec: DeepSeek-V3.1 + MTP only 1.10 -> 1.15x (union near saturation); Qwen3-235B 1.22 -> 1.36x with trees. ADAPTIVE-DRAFT §4.2: ≤ 1% for the pricing part | ≤ +2-4% (4 streams), ~0 (1 stream) | 3-4 | Low value; last |

**Middle-case drafter effect** in the round model:

- **1 stream:** prose E[T] 2.4 -> 2.9 for +0.6 row (~3.2 ms): +13% alone.
- **4 streams:** +2 tokens a round for +1.6 rows (~8.6 ms): +14% alone.

**Training cost on this hardware.**

- **Data** is the long pole:
  - Teacher-forced taps and logits run at prefill speed: ~1,500 tok/s, i.e. 10M tokens in ~2 h.
  - Regenerated replies run at decode speed: ~80 tok/s aggregate at 4 streams, ~7M tokens a day.
  - Recipe:
    - Near-policy text: our server's replies, plus GLM-5.3-Flash replies (the same base model) to UltraChat / ShareGPT prompts.
    - Teacher-force it through the abliterated target, and train on the target's soft labels (KL), so the labels are
      on-policy even where the text is not.
- **Data needed:**
  - T1: 5-50k samples (Red Hat 5k, FastMTP 389k).
  - T2: 20-50M tokens.
- **Storage** for T2: taps at 5 layers x 4,096 x bf16 = 40 KB a token, so 20M tokens = 800 GB (400 GB in FP8). That
  fits the Spark's NVMe, or is streamed.
- **Compute:**
  - T1: < 0.5 B trainable parameters, short contexts. Hours on one Spark.
  - T2: fine-tuning a ~3 B block drafter on 50M tokens. ~6 x 3.1e9 x 5e7 x ~2.7 (block anchors) ≈ 2.5e18 FLOP:
    ~14 h an epoch at ~50 TFLOP/s achieved bf16 on GB10, or ~2-3 h on one rented H100.
- **The real cost is production downtime.** The target needs both Sparks to generate taps, and training needs one
  Spark's memory. So:
  - data generation is a planned 3-6 h prod-down window;
  - T2 training runs on a rented GPU (taps uploaded) or in a second window.
- **Licences:**
  - GLM-5.3-Flash is MIT (verified).
  - Check the abliterated checkpoint's card before sharing anything.
  - T1 output (fine-tuned MTP weights) is ours to keep.
  - T2 must not start from incoai's NC-ND weights if it is ever shared.

### 3.3 Checked and not in the plan

| idea | why not |
| --- | --- |
| Trees / sibling candidates | Each node is a row of new experts (~5.5 ms); a sibling is worth ~0.12 tokens (EXPERIMENTS §5). "Limits of Speculation" caps MoE speculation by the expert union. Stays rejected |
| Cross-request suffix tree (N3) | Measured +0.0% on 826 real agent steps (DEEP-VERIFY §4) |
| Lossless weight compression | Trellis words at 15.97 / 16 bits of entropy (RESEARCH-NIGHT §3) |
| Reduced expert set at verify (MoE-Spec, AcceptMoE) | Not exact |
| EP / PP across the two Sparks | EP loses at batch 1 (two all-to-alls a layer, imbalance); PP halves single-stream |
| Ladder residual / parallel attention + FFN | Model change (retraining); not exact |
| Full persistent megakernel (MPK / Hazy-style) | Would capture E3-E6 and E8 at once (+10-18%), but nothing supports MoE + KDA + DSA + mHC + EXL3, and it is 25+ days. E1's persistent MoE layer and E4's fusion + PDL are its incremental path |
| Per-slot drafter choice (0340), deeper rows for prose | ADAPTIVE-DRAFT / DEEP-VERIFY: ≈ 0 |

### 3.4 Weight bytes and clocks (outside the exact plan)

- **Non-expert q4mse 4.5 bpw -> EXL3 4.0** (the same codec family as the experts): -0.26 GB a verify, -1.1 ms a
  round, +2%. It changes outputs, so it needs the quality gate (MMLU-200, needle, exact / batchexact on the new
  snapshot).
- **3.x-bpw experts** (parked): +5.5% at 3.5 bpw, +12% at 3.0 bpw on every cell. Not in the plan.
- **Clocks.**
  - Decode is DRAM-bound, and the memory clock is not settable on GB10.
  - The ~7-9 ms of latency-bound kernels scale with the SM clock, but raising the `-lgc 300,2250` cap risks the
    prefill power-off it exists for, at 79-83 °C already. **Do not touch.**
  - The value is guarding against the slow state (-30-50%) and lockstep drag, which `gpuwatch` already does.
  - Add the per-round histogram to `/metrics` (OPS-GPUWATCH's granularity note) so a slow state inside a long reply
    is visible.

## 4. The combined plan

### 4.1 Single stream (prose; 44 -> ~61 tok/s at mid)

| # | contribution | ms a round (mid) | cumulative round | tok/s (2.4 tokens) | cumulative gain |
| --- | --- | ---: | ---: | ---: | ---: |
| | today (reconstructed) | | 54.4 | 44.1 | |
| E7 | draft-head vocab trim | -0.8 | 53.6 | 44.8 | +1.5% |
| E5 | graph coverage | -1.0 | 52.6 | 45.6 | +3.4% |
| E3 | L2 prefetch in the Batcher path | -0.7 | 51.9 | 46.2 | +4.8% |
| E2 | dense q4 GEMV -> 215 GB/s | -1.9 | 50.0 | 48.0 | +8.8% |
| E1 | routed experts -> 215 GB/s | -2.7 | 47.3 | 50.7 | +15% |
| E4 | fusion + PDL | -1.5 | 45.8 | 52.4 | +19% |
| E6 | device sampler + draft chains | -1.0 | 44.8 | 53.6 | +21% |
| E8 | RoCE residual | -0.4 | **44.4** | **54.0** | **+23%** |
| T1 + T2 (+T3) | drafters: E[T] 2.4 -> 2.9, +0.6 row at 215 GB/s (+3.2 ms) | +3.2 | 47.7 (2.9 tokens) | **60.8** | **+38%** |

Sensitivity (1 stream, prose):

| engineering \ drafters | none | low (2.65 tokens, +0.3 row) | mid (2.9, +0.6) | high (3.1, +0.8) |
| --- | ---: | ---: | ---: | ---: |
| none | 0 | +7% | +13% | +19% |
| low (-5.6 ms) | +12% | +19% | +26% | +32% |
| **mid (-10.0 ms)** | +23% | +31% | **+38%** | +44% |
| high (-15.6 ms, a bound) | +40% | +49% | +57% | +64% |

**Code / structured** (~70 ms, 5.7 tokens a round): the same engineering saves ~9-16 ms (+15-30%; the rounds are
longer, so the fixed savings weigh less), plus T4's +8-15%.

### 4.2 Four streams (aggregate; 76.5 -> ~110 tok/s at mid)

| # | contribution | ms a round (mid) | cumulative | gain |
| --- | --- | ---: | ---: | ---: |
| | today (reconstructed) | | 119.0 | |
| F0 | graph policy (config) | -2 (not added to the stack below: inside the spread until A/B'd) | | (+0-3.5%) |
| F1 | routed experts -> 215 GB/s (weight-major at U ≈ 51) | -8.1 | 110.9 | +7% |
| F3 | per-slot fixed cost halved-plus | -9.0 | 101.9 | +17% |
| F2 | dense q4 in batched rounds | -3.0 | 98.9 | +20% |
| F4 | device sampler + batched DFlash2 blocks | -3.0 | 95.9 | +24% |
| F6 / F7 / F5 | vocab trim, prefetch, RoCE | -2.8 | **93.1** | **+28%** |
| T1 + T2 | +2 tokens a round (9.1 -> 11.1), +1.6 rows (+7.7 ms) | +7.7 | 100.8 (11.1 tokens) | **+44%** |

4 streams: engineering low / mid / high = +13 / +28 / +50%. With middle drafters: +29 / **+44** / +68%.

### 4.3 Sequence (by gain per effort and risk; each step has its own A/B gate)

| order | item | days | GPU window | expected (mid) | gate to adopt |
| ---: | --- | ---: | --- | --- | --- |
| 1 | **Measure**: nsys + ncu on production b4 + FIN (§5.1) | 0.5 | 30-40 min | replaces §1's reconstruction | - |
| 2 | F0 config A/B (`BATCH_GRAPHS=0` / `CAPTURE_AFTER=6`) | 0 | 20 min | 4s +0-3.5% | batchexact 4/4, ≥ +2% at 4 streams |
| 3 | E7 vocab trim | 2-3 | 30 min | 1s +1.5%, 4s +1% | exact 10/10, sampled cells not lower |
| 4 | E5 graph pre-capture + E3 prefetch port | 5-7 | 40 min | 1s +3.3% | 1s ≥ +2%, memory ≥ 8 GiB in the stress |
| 5 | **T1 MTP self-distillation** (data window 3-6 h, training offline) | 5-8 | data 3-6 h + A/B 40 min | sampled +8-12%, prose +4-6% | teacher-forced a2 / a3 up ≥ 0.08 first; then exact, cells |
| 6 | E1 + E2 decode kernels (roof probe first, §5.4) | 12-19 | 2 x 40 min | 1s +9%, 4s +10% | bitwise tests; 1s ≥ +5% |
| 7 | F3 multi-slot launches + E6 device sampler | 17-24 | 2 x 60 min | 4s +12%, 1s +2% | batchexact; sampler equivalence over 10^7 draws |
| 8 | **T2 block-drafter re-fit** (+T3 stop head, +T4 block 16) | 10-15 + compute | data window + A/B | prose +10-20%, code +8-15% | teacher-forced τ on held-out replies ≥ +0.4 before any serving A/B |
| 9 | E4 fusion + PDL, E8 RoCE path | 10-15 | 2 x 40 min | 1s +3.5% | bitwise; RoCE soak ≥ 2 h |

Items 1-5 fit roughly one week and should show **+8-12% single-stream** (more on sampled requests). The drafter
number from item 5 decides how hard items 6-9 must push.

## 5. First steps for the next session (buildable and testable)

1. **One capture of today's production** (W7 harness, `GLM53_TF_PROFILE` probes, one nsys per server lifetime):
   - a 256-token prose reply and a 4-stream 384-token set;
   - `dec.py` / `dec_bytes.py` / `experts_union.py` on the new trace;
   - ncu on `grouped_kernel` and the four largest `_qmm` shapes at R = 1 / 4 / 8 / 16: dram__throughput, launch
     ramp and tail, achieved occupancy;
   - `nvidia-smi` clocks during the run.

   This yields the real §1 table (RoCE exposed ms, post-0370 idle, the 1-stream eager / capture share) and the per
   kernel GB/s that E1 / E2 start from.
2. **E7 offline now, no GPU.**
   - Tokenize the local opencode GLM-5.3-Flash transcripts (826 agent steps) plus glmbench / multiturn replies
     with the GLM tokenizer.
   - Rank token frequency; report reply-token coverage of the top 16k / 32k / 48k.
   - Build the patch if 32k covers ≥ 97.5%: a `draft_head_ids` index plus a sliced head copy per rank (~40 MB);
     MTP and DFlash2 draft over it; the verify head untouched.
   - CPU tests on 0180's hostile fake model (drafted == serial with a subset that misses tokens).
3. **T1 day 1-2.**
   - (a) A dump hook on the prefill path: final hidden rows, next-token ids and the target's top-k logits at
     t+1..t+4. `MTP_PREFILL_CACHE` already computes the MTP inputs.
   - (b) A PyTorch reference of the MTP layer: embed / norm / `eh_proj` / MLA dense at ≤ 2,048 context, which
     equals DSA top-2,048 there; MoE with dequantized experts; the shared head.
   - (c) A teacher-forced acceptance script that must reproduce today's a = 0.74 / 0.45 / 0.22 on dumped data
     before any training.
   - Everything but the dump is offline.
4. **Roof probe for E1 / E2.** A microbench that streams exactly the bytes of a decode round's expert set (U = 8,
   17, 24, 51 experts x 6.29 MB) and of each `_qmm` shape with a plain 128-bit-load kernel, next to the real kernels.
   The gap between the probe's GB/s and the kernel's is E1 / E2's real headroom. If the probe tops out at ~205,
   E1 / E2 shrink to the low column.
5. **F0 config A/B** in the same window as step 1 (no build).

## Sources

**Speculative decoding and drafters:**

- EAGLE-3: [arXiv 2503.01840](https://arxiv.org/abs/2503.01840)
- HASS: [arXiv 2408.15766](https://arxiv.org/abs/2408.15766)
- GRIFFIN: [arXiv 2502.11018](https://arxiv.org/abs/2502.11018)
- FastMTP: [arXiv 2509.18362](https://arxiv.org/abs/2509.18362)
- Red Hat FastMTP heads: [developers.redhat.com, 2026-09-08](https://developers.redhat.com/articles/2026/09/08/optimize-vllm-speculative-decoding-fastmtp-heads)
- MTP-D: [arXiv 2603.23911](https://arxiv.org/abs/2603.23911)
- GLM-5: [arXiv 2602.15763](https://arxiv.org/abs/2602.15763)
- DeepSeek-V3: [arXiv 2412.19437](https://arxiv.org/abs/2412.19437)
- MiMo-V2-Flash: [arXiv 2601.02780](https://arxiv.org/abs/2601.02780)
- DFlash: [arXiv 2602.06036](https://arxiv.org/abs/2602.06036), [z-lab/dflash](https://github.com/z-lab/dflash)
- DBloom: [arXiv 2608.30427](https://arxiv.org/abs/2608.30427)
- DSpark: [arXiv 2607.05147](https://arxiv.org/abs/2607.05147)
- PARD: [arXiv 2504.18583](https://arxiv.org/abs/2504.18583)
- SpecDec++ (COLM 2025)
- SpecForge: [LMSYS](https://www.lmsys.org/blog/2025-07-25-spec-forge/)
- SuffixDecoding: [arXiv 2411.04975](https://arxiv.org/abs/2411.04975)
- Arctic hybrid: [Snowflake](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/)
- Not re-verified this session: FR-Spec [arXiv 2502.14856](https://arxiv.org/abs/2502.14856) and DistillSpec
  [arXiv 2310.08461](https://arxiv.org/abs/2310.08461).
- DSpark data-scaling figures: [GLM-5 planning issue](https://github.com/PhillipChaffee/GLM-5/issues/1)
  (secondary, unverified).

**MoE speculation:**

- EcoSpec: [arXiv 2607.12696](https://arxiv.org/abs/2607.12696)
- EVICT: [arXiv 2605.00342](https://arxiv.org/abs/2605.00342)
- MoE-Spec: [arXiv 2602.16052](https://arxiv.org/abs/2602.16052)
- Cascade: [arXiv 2506.20675](https://arxiv.org/abs/2506.20675)
- Limits of Speculation: [arXiv 2609.22156](https://arxiv.org/abs/2609.22156)
- MoESD: [arXiv 2505.19645](https://arxiv.org/abs/2505.19645)

**Kernels and megakernels:**

- MonoMoE: [arXiv 2609.04244](https://arxiv.org/html/2609.04244v1)
- Hazy "No Bubbles": [blog](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- Hazy TP Llama-70B: [blog](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)
- MPK: [arXiv 2512.22219](https://arxiv.org/abs/2512.22219)
- TRT-LLM min-latency DeepSeek-R1: [blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog1_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.html)
- Ladder Residual: [arXiv 2501.06589](https://arxiv.org/abs/2501.06589)
- TokenWeave: [arXiv 2505.11329](https://arxiv.org/abs/2505.11329)

**DGX Spark / GB10:**

- GEMV fast / slow state: [tonyd2wild issue #1](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/issues/1)
- 231 GB/s end to end: [ai-muninn](https://ai-muninn.com/en/blog/dgx-spark-bandwidth-ceiling-85-percent)
- llama.cpp gpt-oss-120b: [discussion 16578](https://github.com/ggml-org/llama.cpp/discussions/16578)
- NVIDIA Spark performance: [blog](https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks)
- vLLM on Spark: [blog](https://vllm.ai/blog/2026-06-01-vllm-dgx-spark)
- GLM-5.3-Flash vLLM TP2 MTP acceptance: [HF discussion 52](https://huggingface.co/zai-org/GLM-5.3-Flash/discussions/52)
- MiaAI kit: [GitHub](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
- Reduced-vocab draft on Spark: [tonyd2wild Qwen3.8 kit](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark)
- NCCL latency between two Sparks: [Sangiorgi](https://contact.alessandrosangiorgi.net/posts/dgx-spark-nccl-collective-latency/)
- 721 MHz clamp: [NVIDIA forum](https://forums.developer.nvidia.com/t/dgx-spark-gb10-gpu-clock-pinned-at-721-mhz-under-full-load-no-throttling-not-liftable-via-nvidia-smi/376039)
- Hard power-off fix: [tonyd2wild](https://github.com/tonyd2wild/dgx-spark-hard-poweroff-fix)
- Two-node clocks: [WeZZard bench](https://github.com/WeZZard/dgx-spark-bench)

**This repo:**

- ROOFLINE.md §2
- PROFILE.md §5
- RESULTS.md W5, W7, W9 §3 / §5, W10
- ADAPTIVE-DRAFT.md §7
- DEEP-VERIFY.md §3-4
- DECODE-OVERLAP.md §2
- DECODE-ANALYSIS.md §2-3
- COMM-ANALYSIS.md §2-3
- EXPERIMENTS.md §1, S4 / S5
- RESEARCH-NIGHT.md §1, §4
- OPS-GPUWATCH.md
- `results/W10/conc-*.json`
