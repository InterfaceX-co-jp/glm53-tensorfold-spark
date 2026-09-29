# Quality plan: higher-precision non-expert weights (patches/0470), and how to measure what they buy

Offline work (2026-09-29): no GPU was used and the Sparks were not touched. Every speed and memory number below is an
**estimate** from the repo's measured round model (ROOFLINE §2, DECODE-PLAN §1, W10 memory gates). The weight-error
numbers were measured on the CPU with real GLM-5.3-Flash tensors (§1).

## 0. Bottom line

- **What 0470 adds.**
  - `GLM53_TF_NONEXPERT=q8`: int8, symmetric, one bf16 scale per 128 inputs, 8.125 bits a weight (the sfxnz kit's
    INT8 g128).
  - Decode GEMV and prefill GEMM kernels for it, for sm_121. They keep the engine's exactness rules: row-invariant,
    and exact prefill gives decode's bits.
  - `GLM53_TF_NONEXPERT_MAP`: the format per module class, e.g. `o_proj=q8,down=q8,lm_head=bf16`.
  - Prepared folders keyed by the resolved precision.
- **Error on the model's own weights.**
  - q8 relative weight error: 0.65-0.73% (eh_proj 1.9%).
  - q4mse: 8.6-9.4%, about 13x more.
  - For scale, the checkpoint's own bf16 rounding is ~0.2%.
  - The bf16 mode is the ceiling: it holds exactly the checkpoint's values, and for the FP8-origin classes those are
    upcast block-FP8 (§1.1).
- **Cost** (1 stream / 4 streams, today's kernel efficiency):

  | config | decode cost | memory a node | fits the 1M pool at >= 8 GiB? |
  | --- | --- | --- | --- |
  | output side at q8 (`out=q8`) | -6.5% / -2.9% | +0.82 GiB | yes (8.54 GiB) |
  | the example map from the request | -8.8% / -4.0% | +1.08 GiB | yes (8.28 GiB) |
  | q8 everywhere | -16% / -8% | +1.99 GiB | no: pool 917,504 tokens |
  | bf16 everywhere | -38% / -21% | +5.94 GiB | no: pool 393,216 tokens |

  - "Fits" is against the W10 4 x 250k stress minimum of 9.36 GiB on the worker node.
  - The 1M pool keeps >= 8 GiB for the output-side and example maps only.
  - The kernel work in flight (0440 + 0460, mid estimates -6.1 ms a 1-stream round) pays for the output-side map
    (+3.8 ms) or the example map (+5.2 ms). It does not pay for q8 everywhere (+10.5 ms).
- **MMLU > 99% is not reachable, with any weights** (§6).
  - MMLU's own labels carry an estimated ~6.5% error rate (MMLU-Redux).
  - Frontier models sit around 90%.
  - Our generative, thinking-off MMLU-200 is 87-88%, with a +-4.5 pt interval at n = 200.
  - Non-expert precision can move MMLU by about a point at most. The routed experts (EXL3 4.0 bpw, unchanged by
    0470) and the abliteration set the model.
  - Measure precision with KL / top-1 against the bf16 configuration (`bench/divergence.py`), and MMLU in full
    (`bench/mmlu_full.py`, +-0.55 pt) as the paired sanity check.

## 1. What the non-expert weights are, and how 8 bits compares

### 1.1 Lineage: the "bf16" non-experts are partly FP8 values

`neko-legends/GLM-5.3-Flash-Uncensored-EXL3` declares `base_model: orcarouter/GLM-5.3-Flash-Uncensored-FP8`. That
card says the abliteration was written back into the official **block-FP8 shards**:

- e4m3 values with 128 x 128 block scales;
- FP8 matrices dequantized, projected in fp32, requantized with their own scales;
- BF16 matrices (the 34 KDA `o_proj`, the MTP `eh_proj`) projected and kept BF16.

The public `zai-org/GLM-5.3-Flash` headers (read here with HTTP range requests) give the classes by storage:

| engine class (0470) | tensors | stored as in the release |
| --- | --- | --- |
| `dsa_proj`, `dsa_qb`, `dsa_out` | q_a / kv_a, q_b, o_proj of the 11 MLA layers + MTP | **FP8** e4m3, 128 x 128 blocks |
| `dense_gu`, `dense_down` | layers 0-2 MLP | **FP8** |
| `shared_gu`, `shared_down` | shared expert, 42 layers + MTP | **FP8** |
| `kda_proj`, `kda_fb`, `kda_gb`, `kda_out` | KDA q/k/v/f/g/b, f_b/g_b, o_proj | BF16 |
| `dsa_kvb`, `idx_proj`, `idx_qb` | MLA kv_b, indexer | BF16 |
| `mtp_eh`, `lm_head` | MTP eh_proj, head | BF16 |

So the EXL3 checkpoint's "BF16" non-experts are exact upcasts of FP8 for 7 of the 16 classes. Consequences:

- `GLM53_TF_NONEXPERT=bf16` is the whole-model reference. No higher precision exists anywhere.
- A future `fp8` mode (e4m3 + 128 x 128 block scales) would be **lossless** for the FP8-origin classes, at 8.0 bpw.
  It needs an e4m3 -> bf16 GEMV path; not built here.

### 1.2 Format study (real tensors, CPU)

Setup:

- 12 matrices fetched from `zai-org/GLM-5.3-Flash`, covering every class. Row subsets of the large ones; the FP8
  ones were dequantized with their block scales.
- Relative Frobenius error of the stored weights. Output error with Gaussian inputs gives the same numbers to
  +-0.02 pt.

| tensor (origin) | q4 | q4mse | int8 per channel | **int8 g128 sym (0470)** | int8 g128 sym + MSE clip | int8 g64 sym | int8 g128 asym |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KDA q_proj (bf16) | 9.03% | 8.63% | 0.90% | **0.66%** | 0.66% | 0.61% | 0.60% |
| KDA o_proj (bf16) | 9.25% | 8.82% | 1.09% | **0.69%** | 0.69% | 0.63% | 0.62% |
| KDA f_b (bf16) | 8.97% | 8.57% | 0.65% | **0.65%** | 0.65% | 0.60% | 0.59% |
| MLA kv_b (bf16) | 8.98% | 8.59% | 0.75% | **0.65%** | 0.65% | 0.60% | 0.59% |
| indexer wq_b (bf16) | 9.58% | 9.16% | 0.92% | **0.72%** | 0.72% | 0.65% | 0.64% |
| MTP eh_proj (bf16) | 14.93% | 9.47% | **14.69%** | **1.90%** | 1.90% | 1.37% | 1.04% |
| lm_head (bf16) | 9.00% | 8.61% | 0.93% | **0.66%** | 0.66% | 0.60% | 0.59% |
| MLA o_proj (fp8) | 9.04% | 8.61% | 1.25% | **0.66%** | 0.66% | 0.61% | 0.60% |
| MLA q_b (fp8) | 9.78% | 9.35% | 0.98% | **0.73%** | 0.73% | 0.66% | 0.67% |
| shared down (fp8) | 9.24% | 8.82% | 0.91% | **0.68%** | 0.68% | 0.62% | 0.62% |
| shared gate (fp8) | 9.00% | 8.59% | 0.88% | **0.66%** | 0.66% | 0.60% | 0.60% |
| dense down, layer 0 (fp8) | 9.11% | 8.67% | 1.00% | **0.68%** | 0.68% | 0.62% | 0.60% |

Choice: **symmetric, groups of 128, a bf16 scale rounded up** (nothing clips).

- **Per channel is unsafe.** eh_proj's input is [embedding | hidden], two halves with different scales: 14.7%, as
  bad as q4.
- **A zero-point or g64 buys little.**
  - Error falls only 5-10% (eh_proj 1.9 -> 1.0-1.4%).
  - It costs 1.5% more bytes (8.25 bpw).
  - A zero-point also needs the kernels to read 64/128-input group sums: Q4's `xs` term, one more FMA and one more
    load stream a group.
- **An MSE clip search changes nothing at 8 bits.** The unclipped scale always won.

## 2. patches/0470

Knobs (load time, both ranks equal):

- `GLM53_TF_NONEXPERT=bf16|q4|q4mse|q8`.
- `GLM53_TF_NONEXPERT_MAP=key=mode,...`.
  - Keys: the 16 classes of §1.1, the groups below, or `default`, each optionally prefixed `mtp.` (the MTP layer
    only).
  - The most specific key wins: `mtp.<class>` > `<class>` > `mtp.<group>` > `<group>` > `mtp.default` > `default`
    > `GLM53_TF_NONEXPERT`.
  - Two groups that disagree on a class at one level are an error until that class is named alone.

| group | classes |
| --- | --- |
| `o_proj` | kda_out, dsa_out |
| `down` | dense_down, shared_down |
| `out` | kda_out, dsa_out, dense_down, shared_down, lm_head (every block's output side + the head) |
| `kda` / `dsa` / `indexer` / `attn` | the attention classes |
| `dense` / `shared` / `mlp` | the MLP classes |
| `kv_b` | dsa_kvb |
| `head` | lm_head |

Examples:

- `GLM53_TF_NONEXPERT=q4mse GLM53_TF_NONEXPERT_MAP=out=q8`: the output side at 8 bits.
- `GLM53_TF_NONEXPERT_MAP=o_proj=q8,shared_down=q8,kda_out=q8,lm_head=bf16,default=q4mse`: the requested example.
  kda_out is inside `o_proj` too, which is allowed.

The sfxnz evidence behind the output-side idea is SFXNZ-AUDIT §4 item 5 (their E2c: KDA o, MLA and the shared
expert carried ~3.8e-3 of FP8's 5.2e-3 excess KL).

What changes where (`families/glm5_next/cuda/`):

- **`qmm.py`.**
  - `Q8` (int8 [N, K] row-major, bf16 scales [K/128, N]), `quantize8`, `dequantize_q8` / `dequantize_any`.
  - `_q8mm`: the decode / verify / MTP kernel, with qmm's split-K slices `q8_split_k`. They never need a larger
    partial buffer than Q4's.
  - The weights are read as int32 words and unpacked in registers (`_q8_unpack`). int8 tensors compiled to one
    `ld.global.b8` a byte with no pipelining; words stream with 4-byte `cp.async`, as the 4-bit kernels do.
  - A group's dot runs as one mma chain in two 64-input halves, then one FMA with the group scale.
  - Tiles by row bucket: `Q8_CONFIG` (16 rows: 1 group a step, 4 warps, 3 stages, 64 columns). Every entry compiles
    for sm_121 without spills on Triton 3.7.1 and 3.8.0. Env override `GLM53_TF_Q8_DECODE_CFG`.
- **`fast_qmm.py`.**
  - `_fq8`: fast prefill chunks run it with one accumulator; the `exact` variant keeps qmm's slice order in one loop,
    so it gives qmm's bits.
  - Tiles: `Q8_TILE_LOOSE` 128 x 32 / 8 warps, `Q8_TILE_EXACT` 64 x 32. Env override `GLM53_TF_Q8_TILE`.
  - FP8 prefill (0083, off in prod) runs 8-bit weights on the bf16 fast kernel.
- **`latent.py`.** kv_b may be 8-bit (`IS_Q4 = 2`) in every absorb / expand variant (v1, 0390's v2, tc). The
  dequantization `s * q` is exact in fp32, so 0390's "exactly dequantized kv_b" property holds.
- **`weights.py`.**
  - Every non-expert matrix is built with its class.
  - An 8-bit (or BF16) head gets the 4-bit draft copy the drafters read (+0.18 GB), as BF16 did.
- **`fastboot.py`.**
  - The prepared-folder key's `nonexpert` field is `weights.precision_key()`: the plain mode when every class
    resolves to one mode, else the full resolved table. A MAP that changes nothing keys the same folder.
  - The Q8 code is part of the key's source digest. Like any weights.py change, the first start on the new image
    rebuilds the prepared folders.
- **Small edits.** `overlap.py` / `l2pf.py` (0460) prefetch Q8 scales + words; `fastpf.py` (KDA bf16 copies) and
  `dump.py` (meta) are format-generic; `engine.py`'s EXL3 drafter policy keys on "every class bf16".

**Exactness.** Same guarantees as Q4 within a configuration, from the same row-invariant structure:

- drafted == serial;
- batched == alone;
- resumed == fresh;
- exact prefill == decode bits.

A new configuration is a new model: reply hashes differ from q4mse, as bf16's always did.

**Tests.**

| test | what it checks | result |
| --- | --- | --- |
| `tests/test_q8_interpreter.py` | the format; decode and prefill against a float64 reference; row invariance across every bucket; exact prefill == qmm bits under a tile table; latent v1 == v2 with 8-bit kv_b | 21 passed, Triton interpreter |
| `tests/test_q8_compile.py` | sm_121 PTX: mma.sync bf16, cp.async weight stream, no byte loads, FMA-only one-slice kernels, no spills, shared memory <= 99 KB | 90 passed on 3.7.1 and 3.8.0 |
| `tests/test_nonexpert_map.py` | parsing, resolution order, `precision_key`; what `load_checkpoint` stores per class on the synthetic EXL3 checkpoint; prepared folders keyed and round-tripped bit for bit | CPU |
| `tests/test_fastboot_prepared.py` | now also `q8` | CPU |
| `tests/cuda/test_q8_patches.py` | bitwise on the GPU, plus `-k timing -s` for GB/s and tile sweeps | for the GPU window |

## 3. Configurations: bytes, decode cost, memory

**Model.** Per rank, TP=2.

- Non-expert matrices: 4,198 M elements, plus 116 M in the MTP layer.
- By class: kda_proj 1,751 M, kda_out 570, dsa_out 369, shared_gu 352, lm_head 317, shared_down 176, dense_gu 151,
  dsa_qb 138, dsa_proj 92, dsa_kvb 92, dense_down 76, idx_qb 69, kda_fb / gb 18 each, idx_proj 7.
- Bytes a weight: q4 / q4mse 0.5625, q8 1.0156, bf16 2.
- Decode round: 54.4 ms, 2.4 tokens (1 stream prose; DECODE-PLAN §1), 125.5 ms (4 streams, W7).
- Non-expert stream: 3.1 GB at ~190 GB/s = 16.3 ms. That is the verify's 2.36 GB plus ~1.8 MTP steps. Each MTP step
  reads the MTP layer and the 4-bit draft head, whatever the head's format.
- Δ round = Δverify + 1.8 x ΔMTP layer, at 190 GB/s: today's q4 kernel efficiency. The q8 kernel is untimed; sfxnz's
  INT8 g128 Marlin reached ~225 GB/s on GB10, which would make every q8 Δ ~15% smaller.

**Memory.** Rank 1 (worker), which binds.

- Stress minimum: 9.36 GiB MemAvailable on W10's 4 x 250k stress.
- A node's resident Δ is the weights plus the 4-bit draft head copy.
- "Pool for >= 8 GiB": the 1M KV pool (7,616 B a token a rank) cut in 65,536-token steps until the stress minimum is
  back at >= 8 GiB.
- "Needle min" (W10: 7.39 GiB) is the needle-after-stress+MMLU scenario, already under 8 at today's config, shifted
  by Δ. It needs the same pool cut (or dropping the 2 GiB RAM session store) to clear 8.

| # | config (env) | verify GB | resident GB | Δ GiB a node | Δ GB a round | Δ ms | 1 stream ms / tok/s / Δ | 4 streams ms / Δ | stress min GiB | pool for >= 8 GiB | needle min GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| A | `NONEXPERT=q4mse` (prod) | 2.361 | 2.427 | 0 | 0 | 0 | 54.4 / 44.1 / - | 125.5 / - | 9.36 | 1,048,576 | 7.39 |
| B | `NONEXPERT=q4` | 2.361 | 2.427 | 0 | 0 | 0 | same | same | 9.36 | 1,048,576 | 7.39 |
| D | `q4mse` + `MAP=out=q8` | 3.045 | 3.306 | +0.82 | +0.71 | +3.8 | 58.2 / 41.3 / **-6.5%** | 129.3 / **-2.9%** | **8.54** | **1,048,576** | 6.57 |
| D2 | `MAP=o_proj=q8,shared_down=q8,kda_out=q8,lm_head=bf16,default=q4mse` | 3.323 | 3.584 | +1.08 | +0.99 | +5.2 | 59.6 / 40.3 / -8.8% | 130.7 / -4.0% | 8.28 | 1,048,576 | 6.31 |
| E | `NONEXPERT=q8` + `MAP=kda_proj=q4mse` | 3.470 | 3.766 | +1.25 | +1.20 | +6.3 | 60.7 / 39.5 / -10.4% | 131.8 / -4.8% | 8.11 | 1,048,576 (thin) | 6.14 |
| C | `NONEXPERT=q8` | 4.264 | 4.560 | +1.99 | +2.00 | +10.5 | 64.9 / 37.0 / **-16%** | 136.0 / -7.7% | 7.37 | **917,504** (-0.93 GiB) | 5.40 |
| F | `NONEXPERT=bf16` | 8.396 | 8.807 | +5.94 | +6.34 | +33.3 | 87.7 / 27.4 / **-38%** | 158.8 / -21% | 3.42 | **393,216** (-4.76 GiB) | 1.45 |

- **bf16 does not fit production.**
  - At the 1M pool the stress minimum would be ~3.4 GiB.
  - It fits only with the pool at ~393k tokens: 4 slots of ~98k, or one 393k context.
  - Alternatively, drop the 2 GiB RAM session store and cut less: ~655k with the store off.
  - It is a measurement reference, run with a small pool (§5), not a serving candidate.
- **q8 everywhere** needs the pool at 917,504 tokens (CONTEXT 917,504), or the session store at 1 GiB instead of 2.
- **Prefill.**
  - The non-expert GEMMs in fast chunks are compute-bound at 512-row sub-blocks. The 8-bit kernel's work a weight
    is no larger than q4's: a shift pair and a convert, against a shift, a mask, a convert and the bias FMA.
  - Its weight re-reads are ~1.8x the bytes: +2-4 MB a token, mostly hidden.
  - But its tile is untuned (128 x 32 against q4's measured 128 x 64). Expect **-2 to -8% prefill for C until the
    sweep in `test_q8_patches.py -k timing` picks tiles**, less for D / D2.
  - bf16 runs 0081's `_fb16`, not measured on this engine since W1: measure it.
- **The kernel budget.**
  - In flight (mid estimates, 1 stream / 4 streams): 0440 -4.8 / -13.5 ms, 0460 -1.3 / -1.5 ms. That is ~-6 ms a
    1-stream round.
  - D (+3.8 ms) and D2 (+5.2 ms) fit inside it: decode would still end up ~+2-5% above today with 8-bit output
    sides.
  - C (+10.5 ms) does not: ~-8% net.
  - Caveat: 0440's E2 speeds up the 4-bit dense GEMVs only, and in D / D2 about a third of the q4 bytes become q8.
    E2's saving shrinks by that share unless the q8 GEMV gets the same persistent-kernel treatment (a Marlin-style
    int8 path; sfxnz's bar 210-229 GB/s).

## 4. Tools

**`bench/mmlu_full.py`: full MMLU.**

- Scope: 57 subjects, 14,042 test questions.
- Data: HF `cais/mmlu`, MIT, fetched once into `bench/data/mmlu/`, not committed.
- Prompt: `quality.py`'s MMLU-200 prompt exactly. MMLU-200 is a checked subset: all 200 items and answers found in
  the full test split.
- Sampling: thinking off, greedy, first A-D letter.
- `--shots 5`: the subject's dev questions as earlier chat turns, so they form a shared prefix. `--shots-format
  inline` gives the classic layout.
- Concurrency 4 by default; resumable (`answers.jsonl`).
- Report: overall with a Wilson 95% CI, macro average, the 4 categories, per subject.
- `compare A B` gives the paired difference, McNemar's exact p and the subjects that moved.
- Time: MMLU-200 + refusals took 130-185 s at concurrency 1 (~0.55-0.75 s a question).
  - 0-shot at concurrency 4: **~50-70 min**.
  - 5-shot (~900-token prompts): **~2-4 h**, nearer 2 when sessions resume the per-subject few-shot prefix (fork
    marks, 0110).
  - The run prints its rate and ETA every 200 questions.

**`bench/divergence.py`: KL / top-1 against a reference configuration.**

- `corpus`: ~200k tokens, cached in `bench/data/divergence/`:
  - 40% Wikipedia (CC BY-SA, cached for evaluation only);
  - 25% HumanEvalPack code in 6 languages (MIT);
  - 25% UltraChat multi-turn (MIT, sent as chat);
  - 10% GSM8K worked solutions (MIT).
- `run`: every document as a raw completion or a chat, `max_tokens` 1, to a server with the 0430 dump on.
- `compare`:
  - Documents are matched by the hash of their token ids, so each position is teacher-forced on identical context.
  - Per position: KL over the reference's top-32 + a rest bucket (a lower bound of the full-vocabulary KL, exact
    when the lists cover the mass; the report says how often they do), top-1 agreement, and the true next token's
    NLL under both (a perplexity ratio).
  - Output: mean with a document-bootstrap 95% CI, median / p90 / p95 / p99 / p99.9 / max, a log-bin histogram, and
    per source (wiki / code / chat / math).
- Tests: `tests/test_divergence.py`, on dumps written by the engine's own writer.
- Scale: the dump without taps is ~8.4 KB a token, ~1.7 GB a configuration. A run is one prefill of ~200k tokens:
  ~3-6 min.

## 5. GPU test plan

**Image.** b5 = the stack through 0470. The patch applies after 0440 and 0460, checked in name order on the pinned
submodule.

**Prepared folders.** Each configuration is its own prepared folder, ~82-88 GB a node, so check NVMe space.
`scripts/prepare.sh` with the configuration's env writes it ahead; otherwise the first start builds from the
checkpoint. q8 quantization is a few elementwise passes and cheap next to q4mse's clip search.

**Step 1: kernels (~20 min).**

```
pytest -q tests/cuda/test_q8_patches.py
pytest -q tests/cuda/test_q8_patches.py -k timing -s
```

- The first run must pass bitwise.
- The timing run prints q8 / q4 / bf16 GB/s at 1 / 4 / 8 / 16 rows on every shape, and prefill TF/s at 512 / 2,048
  rows.
- From the sweep, set `Q8_CONFIG[16]` and `Q8_TILE_LOOSE` (speed-only tables, no bit changes).
- Bar: >= 200 GB/s at 1-16 rows on the >= 7 MB shapes. Marlin INT8 g128 did 210-229 on these shapes.

**Step 2: divergence (~1 h for 7 loads).**

- Quality server settings, the same for every configuration except the weight knobs:

  ```
  CONTEXT=131072 GLM53_TF_KV_POOL_TOKENS=131072
  GLM53_TF_SESSION_GIB=0 GLM53_TF_BATCH_SESSIONS=0 GLM53_TF_PREFIX_SHARE=0
  GLM53_TF_FAST_PREFILL=0
  GLM53_TF_DRAFT_DUMP=/sessions/div-<cfg> GLM53_TF_DRAFT_DUMP_TAPS=0 GLM53_TF_DRAFT_DUMP_WHAT=prefill
  ```

- `FAST_PREFILL=0` makes the rows use decode arithmetic, as SFXNZ-AUDIT N2 advises: it is what generated tokens
  see. Optionally repeat A and D with fast prefill.
- The small pool is what lets bf16 load: F is +5.94 GiB, the pool 6.5 GiB smaller.
- Loads:
  1. F: the bf16 reference.
  2. A, then A again: the A/A check, which must give KL 0 and top-1 1.000 exactly.
  3. B, C, D, D2, E.
- For each: `divergence.py run`, then `compare --ref div-F --cand div-X --run <F's run.json>`.
- Expected, from sfxnz's numbers on this model (their 4.5 bpw NVFP4 non-experts: KL top-20 3.7e-2, top-1 95.6%):
  - A / B: KL ~2-4e-2, top-1 ~95-96%;
  - C: an order of magnitude lower, KL < 3e-3, top-1 > 99%;
  - D / D2 in between.
- The dump's first GPU run is part of this step (0430 is CPU-tested only).

**Step 3: MMLU full (~1 h a configuration, 0-shot).**

- Configurations: F, A, C, D (and D2 if it differs from D in KL), same quality server settings minus the dump.
- Run `mmlu_full.py run --concurrency 4`, then `compare` each against A (paired).
- 5-shot for A and the best candidate only, if the window allows.
- Refusals (`quality.py`): 0/10 must hold on every configuration.

**Step 4: speed and gates on the production config** (per candidate that passes steps 2-3; ~45 min each):

- `glmbench` tf / kit suites;
- exact 10/10 and batchexact 4/4 (drafted == serial within the configuration);
- decode 1 / 4 streams x 5 reps, and prefill 24.5k / 98k;
- the 4 x 250k stress with MemAvailable sampling. C needs `GLM53_TF_KV_POOL_TOKENS=917504 CONTEXT=917504`, or a
  1 GiB session store.

**Step 5: decision.** Suggested rule:

- Adopt the cheapest configuration whose KL against F is <= 1/4 of A's, with top-1 >= 99%.
- MMLU full may not be significantly worse than A (McNemar p > 0.05 or a positive Δ).
- Refusals 0/10.
- Decode within the 0440 / 0460 budget, memory >= 8 GiB.
- Expected pick: D (output side q8). Pick C only if D's KL stays close to A's, and then with the smaller pool.

## 6. MMLU ceilings (the "> 99%?" question)

- **No official MMLU for GLM-5.3-Flash.**
  - Z.ai's card reports agentic benchmarks only: Terminal Bench 2.1 84.3, DeepSWE 63.4, Agents' Last Exam 26.3,
    AutomationBench 48.8, HLE w/ tools 55.3, GDPval-AA 1773.
  - The abliterated FP8 build's card measured MMLU on 300 questions, scored by option-letter logits with
    `reasoning_effort=low`: **base 83.3%, abliterated 82.7%** (-0.7 pt, which they call noise). Their MMLU-Pro
    was 43.8 -> 45.3.
  - So the abliteration costs about nothing measurable, and a logit-scored MMLU on this model sits in the low 80s.
- **Ours:**
  - MMLU-200 (generative, thinking off): bf16 87.0%, q4mse 88.0% (13 of 200 answers flipped).
  - At n = 200 the 95% interval is about +-4.5 pt, so the two are not distinguishable and "88%" means ~84-92%.
  - Full MMLU narrows this to +-0.55 pt, and paired comparisons are tighter still.
- **Label noise caps every model.**
  - MMLU-Redux (Gema et al., "Are We Done with MMLU?") re-annotated 3,000 questions over 30 subjects. It estimated
    that ~6.5% of MMLU questions have an error: a wrong key, no correct option, several correct options, or an
    unclear question. Some subjects are far worse (virology about half).
  - Even the optimistic 2-3% figure puts a perfect model at ~97-98% against the keys.
  - With ~6.5%, a perfect model lands around 93-95%.
  - The best frontier models report ~90-93%. **99% is not reachable on MMLU by anyone, and not a weight-precision
    question.**
- **Where our remaining headroom is:**
  - The routed experts hold ~97% of the weight bytes, at EXL3 4.0 bpw, in every configuration here. That quantization is the
    largest remaining deviation from the release.
  - Thinking off costs MMLU points on hard subjects: effort `high` or `max` would likely add more than any weight
    change, at ~10-50x the tokens.
  - Non-expert precision can recover only the non-expert share of the quantization error. On MMLU that is expected
    to be <= ~1 pt, which is why KL / top-1 is the primary metric.
  - For a benchmark with headroom, use MMLU-Pro or MMLU-Redux; the same tool pattern applies.
