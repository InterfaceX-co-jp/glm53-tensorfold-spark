# Drafter training: MTP self-distillation (T1) and a block-drafter re-fit (T2)

> Offline work, 2026-09-29. No GPU was used and the Sparks were not touched. Everything below the "measured" line in
> §7 is an estimate from this repo's measured speeds (W10) and published recipes; the GPU phase (§8) turns it into
> numbers. The pipeline is built and tested on the CPU with tiny synthetic models (17 tests); the GPU tests are
> written and not yet run.
>
> Inputs: docs/DECODE-PLAN.md (T1 / T2 / T3 / T4), docs/RESEARCH-NIGHT.md (L1), docs/DRAFT-VOCAB.md (0420),
> `config/prod.env`, the engine's MTP / DFlash2 / prefill / batch code (patches 0020, 0071, 0120, 0190, 0200, 0420),
> the HF model cards and licences named in §10.

## 0. Bottom line

- **Everything the GPU phase needs is in place:**
  - `patches/0430` records training data while the engine serves or generates;
  - `bench/gendata.py` produces the traffic;
  - `train/` holds the PyTorch reference, both trainers and the offline acceptance evaluator;
  - `GLM53_TF_MTP_WEIGHTS` serves a trained MTP head without touching the checkpoint.
- **Data is the long pole, and generation is its slow part.** Measured speeds: ~78 tok/s aggregate decode at
  4 streams, ~1,500-1,600 tok/s prefill. So:
  - one day of generation gives **~5.5-6.5M target-generated tokens**;
  - one day of teacher forcing (prefill over existing text) gives **~120M rows**.
- **The plan mixes both sources** (§2.4):
  - ~10-12M self-generated tokens (2 days) plus the prompts they answer;
  - 30-60M teacher-forced rows (6-12 h);
  - optionally, a day of production traffic with the dump on.

  That is ~50-80M training rows, well above what published MTP recipes needed (Red Hat 5-8k samples, FastMTP 389k
  samples). For T2 it sits at the low end of what block drafters used (DFlash2-G 737k samples, DSpark 1.7M).
- **Training fits one Spark each, in parallel on the two Sparks once generation ends:**
  - T1 (185M trainable parameters): ~40 GB, **~3-8 h** for 20-40M anchor rows.
  - T2 (a 1.1-1.8B-parameter drafter): ~35-50 GB, **~6-20 h** for 50M rows.
- **Expected** (DECODE-PLAN §3.2, not measured here):
  - T1: MTP acceptance 0.74 / 0.45 / 0.22 → ~0.80 / 0.58 / 0.38. That is +9-10% on MTP rounds: sampled cells
    +8-12%, greedy prose +4-6%.
  - T2: prose DFlash2 rounds +0.4-0.7 tokens, i.e. prose +10-20%.
- **Two gates before any engine A/B:**
  1. The reference must reproduce the engine's stock MTP acceptance on dumped data (GPU test, ±0.08).
  2. A trained drafter must beat the stock one offline on held-out documents by the margins in §6.
- **Licences allow private use of everything.** Sharing a trained drafter needs:
  - ShapleyMcg-style attribution for anything derived from the EXL3 checkpoint;
  - for T2, a start from DFlash2-G (Apache-2.0) or DSpark (MIT), never incoai's CC BY-NC-ND drafter.

  The engine cannot load DFlash2-G or DSpark yet: 2-3 days of loader work, §5.3.

## 1. What was built

| Path | What it is |
| --- | --- |
| `patches/0430-glm-draft-dump.patch` | `cuda/dump.py`: per-token records, hooked into `decode._prefill` (every prefill mode) and `batch.Batcher._verify` (committed decode rows). Also the `setup` / `arm` calls in `engine.py`, and `cuda/mtpw.py`: `GLM53_TF_MTP_WEIGHTS`, a distilled head loaded in place of the checkpoint's |
| `bench/gendata.py` | `collect` (prompts from open datasets and local opencode sessions), `run` (generation), `tforce` (teacher forcing), `plan` (time and disk budget). Standard library only |
| `train/glmref.py` | The PyTorch reference, differentiable. The MTP layer as the engine computes it: q4mse non-experts, EXL3 experts through the engine's decoder, latent MLA with FP8 rows, DSA selection past 2,051 keys. Also the DFlash2 drafter (incoai and DFlash2-G layouts), the loaders and the exporters |
| `train/dumpdata.py` | Reads dumps and assembles documents by prefix hash. `stats` CLI |
| `train/common.py` | The KL-to-top-k loss, window sampling, LR schedule and log |
| `train/mtp_distill.py` | T1 trainer (FastMTP recipe, §4) |
| `train/dflash_distill.py` | T2 trainer (§5) |
| `train/eval_accept.py` | Offline acceptance by draft position: greedy, data, and coupled-sampled at T = 0.6 / 1.0 |
| `tests/test_drafter_ref.py` | CPU, 14 tests. `quantize4` / `_clip_search` == the engine's source (run without triton). `exl3_weight` == the engine's float64 decoder. FP8 rows == 0220's reference. Chain == step-by-step drafting. Context rows, sparse selection, gradients, export round trip, shard halves, DFlash2 block independence and sliding window, the evaluator |
| `tests/test_draft_dump.py` | CPU. `Dumper` on a fake two-rank engine: records, the merged top-k, prefix-hash linking of a resumed request, one owner per row, fp8 taps, a tap subset |
| `tests/test_drafter_train.py` | CPU end to end: dump → `mtp_distill` → export → reload through the override → `eval_accept` → `dflash_distill` → export → reload |
| `tests/cuda/test_drafter_ref_patches.py` | GPU, written, not run. On the synthetic checkpoint: reference MTP and DFlash2 == engine (rank 0's shard); `GLM53_TF_MTP_WEIGHTS` loads an export; the dump leaves replies byte-identical and each greedy decode row's dumped top-1 is the emitted token. On the real model (env-gated): the reference reproduces the engine's MTP acceptance |

Run the CPU tests against a patched source tree:

```
TF_SRC=<tree>/src python -m pytest -q tests/test_drafter_ref.py tests/test_draft_dump.py tests/test_drafter_train.py
```

## 2. Data

### 2.1 What a record is (patches/0430)

Per committed position the dump writes five things:

- the input token;
- `hidden`: the main model's final-normed row. This is what the MTP head reads (`Engine.main_hidden`, vLLM's
  convention).
- `taps`: the stream mean after each tap layer, i.e. the DFlash2 context rows (`Engine.tap_rows`);
- `lp` / `ids`: the target's next-token distribution as the top-k log-probabilities over the whole vocabulary.

How the target's distribution is formed:

- Each rank takes the top-k and the log-sum-exp of its vocabulary half.
- One all-gather on the NCCL control path brings both halves to both ranks.
- Both ranks merge. Rank 0 writes; rank 1 computes and writes nothing.

Where the rows come from:

- **Prefill chunks** (kind `p`). The engine computes the head for the chunk's last row only, so the hook runs the
  head (the same `qmm.matmul`) on the other rows, 256 at a time.
- **Kept decode rows** of every batch slot (kind `d`). Their logits are already there from the verify forward.

Linking:

- A segment carries the hash of the ids before it and through it.
- A request that resumed from a session or a shared prefix links to whichever request wrote that prefix.
- `dumpdata.build_docs` rebuilds documents from position 0, and each row is owned by exactly one document.

Bytes per token for GLM-5.3-Flash (D 4,096, k 32):

| taps | bytes / token | 10M rows |
| --- | ---: | ---: |
| none (`GLM53_TF_DRAFT_DUMP_TAPS=0`, MTP only) | 8,388 | 84 GB |
| 5 incoai taps, fp8 | 28,888 | 289 GB |
| 5 incoai taps, bf16 | 49,348 | 493 GB |
| 9 DFlash2-G taps, fp8 (a superset of incoai's 5) | 45,288 | 453 GB |

Throughput impact (estimated; §8 step 2 measures it):

- **Prefill +4-6%.**
  - The extra head is 0.63 GFLOP a row a rank, ~25 µs a row at ~25 TFLOP/s, against ~625 µs a row of prefill at
    1,600 tok/s.
  - Top-k / log-sum-exp over 77,440 logits a row: ~1 µs.
  - One 1 MB gather a 4,096-row chunk.
  - A synchronous device-to-host copy: ~10 ms a chunk.
  - Disk: ~70 MB/s from a background thread.
- **Decode < 1%.** A round adds a top-k over ≤ 64 rows, one small gather and a ~0.4 MB copy: ~0.2-0.4 ms against
  54-119 ms rounds.
- **Memory:**
  - GPU: ~120 MB of head scratch on each rank;
  - host: up to 4 segments queued, ≤ 0.8 GB on rank 0.

  Run the dump with the usual `MEM_GATE` and not inside the 4 x 250k stress configuration.

Exactness: the hook only reads buffers before the next forward overwrites them, and runs its own head into its own
scratch. Replies are byte-identical with the dump on or off; the GPU test checks this.

Knobs (load time, both ranks the same; checked at start):

- `GLM53_TF_DRAFT_DUMP=<dir>`
- `_TOPK` (32)
- `_TAPS=bf16|fp8|0`
- `_WHAT=prefill,decode`
- `_GIB` (512: raise it for a long run)
- `_TAP_LAYERS`: a tap set other than the served drafter's (for DFlash2-G's 9 or DSpark's). The DFlash2 drafter is
  then not used while its taps differ, so decode is MTP / lookup only.

### 2.2 Three ways to make rows

| Source | Speed | On-policy? | Use |
| --- | --- | --- | --- |
| **Generation** (`gendata.py run`): prompts through the server, the target samples its own replies | ~78 tok/s aggregate at 4 streams (W10 FIN 76.5-78.1): **~6.5M tokens a day** | yes: the tokens, the hidden rows and the labels are all the target's | the core of T1 (FastMTP uses self-generated data) and of T2's reply positions |
| **Teacher forcing** (`gendata.py tforce`): existing conversations, assistant turns included, as prefill with `max_tokens` 1 | ~1,500-1,600 tok/s: **~130M rows a day**, ~20x generation | the labels are (the target's full distribution at every position); the input text is not (another model's words) | volume, especially for T2; KL to soft labels keeps the target's distribution even on foreign text |
| **Production traffic** (`GLM53_TF_DRAFT_DUMP` on the serving config for a day) | whatever the user sends: agent sessions with 20-100k-token prompts, mostly prefix-shared | yes, and it is exactly the traffic to be sped up | the best agent data there is. It stores the user's own sessions on the Spark's NVMe (tokens and hidden states): keep it local, delete it after training |

Prefill-only teacher forcing on target-generated text is the same thing as the decode dump, at 20x the speed. The
dump already records every generated token at decode time, so there is nothing to re-prefill. The only reason to
teacher-force is text the target did not write.

FastMTP's and EAGLE-3's results argue for self-generated data. So:

- T1 trains on generated rows first (`--kinds d`), then mixes in teacher-forced rows as a second-stage experiment.
- T2 needs volume and uses both.

### 2.3 Prompt mix (`gendata.py collect`)

The default mix is chat 0.30, code 0.25, agent 0.25, reasoning 0.15, tools 0.05. Thinking is on at high effort for
reasoning, agent and code (production's default effort is high).

All sources below were public and ungated when checked on 2026-09-29 through the HF datasets-server rows API:

| Category | Sources | Licence |
| --- | --- | --- |
| chat | HuggingFaceH4/ultrachat_200k (first user turn); allenai/WildChat-1M (English, not toxic, conversation up to the last user turn) | MIT; ODC-BY |
| code | theblackcat102/evol-codealpaca-v1, nvidia/OpenCodeInstruct, bigcode/self-oss-instruct-sc2-exec-filter-50k, and codeparrot/github-code-clean files under permissive licences with a review / test / refactor / port ask. Stands in for "The Stack" samples: the-stack-smol is gated | Apache-2.0; CC-BY-4.0; ODC-BY; per file |
| agent | **local opencode sessions**: every assistant step becomes a request of the session so far (system, user turns, assistant tool calls, tool results) with OpenCode-shaped tools (bash, read, edit, write, glob, grep, list, todowrite, webfetch). The model regenerates the step. This is the user's own task mix; the production dump is better still | the user's own |
| reasoning | open-r1/OpenR1-Math-220k problems (up to 16k-token replies) | Apache-2.0 |
| tools | Team-ACE/ToolACE, glaiveai/glaive-function-calling-v2 (function specs parsed into OpenAI `tools`) | Apache-2.0 |

Not accessible without login: Salesforce/xlam-function-calling-60k, lmsys/lmsys-chat-1m, Nemotron post-training v2,
ShareGPT (not served by the rows API). A test `collect --n 40` returned 38 prompts from every source.

Generation settings:

- `--temperature 0.8 --top-p 0.95` by default, a seed per prompt. Sampled data covers more of the distribution, and
  the MTP head also drafts sampled requests.
- A greedy run can be mixed in with `--temperature 0`.
- The server stays in its production shape (4 slots, sessions, prefix sharing) with `WARMUP_LENGTHS=""` and
  `CANARY=off`, so warm-up and canary requests are not recorded. Unarmed passes (calibration, capture) are never
  written anyway.

### 2.4 Budget (`gendata.py plan`)

| Phase | Tokens | Wall clock | Dump size (9 fp8 taps / no taps) |
| --- | ---: | ---: | --- |
| generation: ~12k prompts, avg ~900-token replies | 11M generated + ~11M prompt rows | ~48 h (78 tok/s; less with long reasoning prompts in the mix) | ~1.0 TB / 0.18 TB |
| teacher forcing with taps (T2 volume) | 20M | ~4 h | 0.9 TB |
| teacher forcing without taps (T1 volume), a second server start with `_TAPS=0` | 40M | ~8 h | 0.34 TB |
| optional: production traffic, one day | ~2-5M rows, mostly prefill | 24 h (serving as usual) | ~0.1-0.2 TB |
| **total** | **~80M rows** | **~2.5-3 days** | **~2.3 TB** |

- Check the NVMe's free space first (DGX Spark: 4 TB).
- Set `GLM53_TF_DRAFT_DUMP_GIB` to the space you can give. The dump stops cleanly at the cap.
- 50-200M tokens is reachable only with teacher forcing: 200M generated tokens would take ~30 days at 4 streams.
- Generation at 8 slots (`GLM53_TF_BATCH=8`, shorter contexts) could reach ~110-130 tok/s. That is an untested
  extrapolation of the 1 → 4 stream scaling; try it in step 2 of §8, and gate on memory and batchexact.

## 3. The reference (`train/glmref.py`)

The MTP layer follows `mtp.mtp_compute` op by op:

```
x = eh_proj([enorm(e) | hnorm(h)])     # e zeroed at the head's first position
x = x + attn(input_layernorm(x))       # plain residual
x = x + moe(post_attention_layernorm(x))
out = shared_head.norm(x)
logits = head(out)
```

- `out` is the next chained step's `h`.
- Each op uses the engine's roundings: RMSNorm, gate / up bf16, the swiglu limit, fp32 combine, bf16 residual.
- It uses the engine's weights:
  - non-experts quantized bit for bit as `qmm.quantize4(mse=True)` does (tested against the engine's own source);
  - routed experts from the EXL3 trellis through the engine's reference decoder (tested equal);
  - the target's head as the engine stores it (q4mse).
- Attention is absorbed MLA over the head's own cache with FP8 latent rows (tested equal to 0220's reference).
- DSA's top-2,048 selection applies past 2,051 keys (`Indexer`, following `sparse.py`'s definition).

It is **near-exact, not bitwise**: the engine's kernels sum in a fixed tiled order, the reference in torch's. The CPU
tests hold the reference to itself (the chain vs step-by-step drafting, shards vs the whole). The GPU test holds it
to the engine:

- logits within 3% of their range;
- argmax equal on ≥ 97% of rows;
- the stock head's acceptance within ±0.08 on real dumps.

`MTP.chain` is FastMTP's training-time recurrence:

- Step 1 absorbs every position. Entry i reads (h_i, t_{i+1}) and predicts t_{i+2}.
- For anchor a, step k sits at entry a + k - 1. It reads step k-1's output and t_{a+k}.
- It attends to step-1 entries 0..a plus the anchor's own chain entries, i.e. what a drafted chain sees at inference
  while its drafts are right.
- A CPU test checks that one parallel pass equals drafting step by step through `MTP.step`.

`DFlash` follows `dflash2.py`:

- context = hidden_norm(fc(taps)), per-layer k / v with k_norm and RoPE;
- blocks [pending, mask x 7] with grouped dynamic convolutions around attention and the MLP;
- bidirectional block attention plus the context within the sliding window;
- the target's head, and the engine's chain rule (top-16 plus 0.6 x the selector edge).

It also takes DFlash2-G's layout: no window, a learnt mask embedding, 9 taps, 8 layers.

## 4. T1: MTP self-distillation (`train/mtp_distill.py`)

The recipe (FastMTP, arXiv 2509.18362; Red Hat's vLLM heads):

- **3 chained steps** with shared weights (training-time test).
- **Loss per step:** KL(target || head) to the dumped top-32 plus a rest bucket, plus 0.1 x cross-entropy on the
  document token.
- **Step weights 0.51 / 0.31 / 0.18:** FastMTP's β = 0.6 decay, normalized.
- **Trained:** attention, `eh_proj`, the norms, the router and the shared expert; 185M parameters, fp32 master
  weights.
- **Frozen:** the 288 routed experts (EXL3 dequantized to bf16, 14.5 GB), the embedding and the target's head.
- **Quantization-aware:** the forward uses the q4mse copy the engine will make of the exported BF16 weights
  (straight-through). The head that trains is the head that serves.
- **Optimizer:** AdamW (0.9, 0.95), LR 3e-5, 200 warmup steps, cosine to 10%, clip 1.0.
- **Windows:** 1,024 anchors after 1,024 context rows, 2 windows a step, 6,000 steps (~12M anchor rows ≈ 1 epoch
  of ~12M loss rows).

Dense attention over ≤ 2,051 rows is the engine's exact rule. A window of a longer document sees its last 2,048
rows instead of DSA's selection; that is the one approximation. `eval_accept --sparse-docs N` measures its effect
with the exact selection on whole documents up to N tokens.

**Cost on one Spark (engine stopped):**

- Memory:
  - frozen weights ~18 GB;
  - optimizer and master weights ~3 GB;
  - activations ~15-25 GB (logits dominate: 1.5 MB an anchor-step with gradients).

  Total **~40 GB of 121 GiB**.
- Compute: ~7 GFLOP an anchor-step (the head's 634M parameters, 201M of active experts, attention over 2,048 keys;
  backward ~2x), x 3 steps ≈ 21 GFLOP an anchor. At 15-25 TFLOP/s achieved that is ~700-1,200 anchors/s, so
  **12M anchors ≈ 3-5 h**, 40M ≈ 9-16 h. The python-loop MoE is the likely bottleneck; the logged `anchors_per_s`
  decides.

**Export and serving:**

- `<out>/export/mtp.safetensors` holds BF16 tensors under the checkpoint's own names
  (`model.language_model.layers.45.*`).
- Serve with `GLM53_TF_MTP_WEIGHTS=<out>/export` on both ranks (`cuda/mtpw.py`, patches/0430):
  - each rank slices its part by the checkpoint's own split rules;
  - the tensors are quantized exactly as the checkpoint's would be (q4mse);
  - the prepared folders key on the export's files.
- It lives outside `weights.py`, so production's prepared folders stay valid with the knob unset.
- The routed experts are not re-encoded. Training them would need exllamav3's EXL3 encoder and ShapleyMcg-licensed
  scales; that is not planned.

## 5. T2: block-drafter re-fit (`train/dflash_distill.py`)

### 5.1 Recipe

- DFlash / DSpark-style on-policy distillation: 128 anchors a window of 2,048 rows, and 2,048 context rows before
  the first anchor (the sliding window is 2,047).
- Loss: KL to the dumped top-32 plus 0.1 x cross-entropy. Block row j is weighted exp(-(j-1)/7).
- Every weight trains except the candidate selector.
- Quantization-aware to the engine's 4-bit drafter copy (non-mse `quantize4`).
- AdamW, LR 2e-5, 500 warmup steps, 20k steps.

### 5.2 Cost

- incoai layout (5 layers, 1.1B parameters): ~18 GB of weights and optimizer state.
- DFlash2-G (8 layers, 1.84B parameters): ~30 GB.
- Plus the frozen head and embedding (2.5 GB) and activations. **~35-50 GB.**
- Compute: ~11 GFLOP a block row trained (x 8 rows a block) plus ~1.5 GFLOP a context row. At 1 anchor per 16 rows:
  **50M rows ≈ 5-10 h**; at 1 per 4 rows, 15-25 h.
- It runs on the second Spark while T1 runs on the first.
- Copy the taps over the 200 Gb link first: ~1 TB takes ~5-10 min at 2-4 GB/s.

### 5.3 Starting weights and what the engine can load

| Drafter | Licence | Architecture | Taps | Engine today |
| --- | --- | --- | --- | --- |
| incoai/GLM-5.3-Flash-DFlash2 (served now) | **CC BY-NC-ND 4.0** | 5 Qwen3 layers, dynamic conv, selector (rank 256, top 16), sliding window 2,048, block 8 | 5, 14, 24, 33, 42 | loads (`dflash2.py`). A re-fit loads the same way (`DRAFTER=<out>/export`) but may **never be shared** |
| canada-quant/GLM-5.3-Flash-DFlash2-G | **Apache-2.0** | the same family: 8 layers, full attention (`sliding_window: null`), learnt `mask_embedding.pt`, ships its own `embed_tokens` and `lm_head`. 1.84B parameters, 6.2 GB. Trained on 737k self-generated samples against the non-abliterated W4A16 target; 3.676 vs incoai's 3.632 mean acceptance at K = 7 (B300) | 5, 9, 14, 19, 24, 28, 33, 38, 42 | **no.** Needs `dflash2.py` support for a null window (a full-context cache instead of the 4,096-slot ring), the mask embedding in place of `embed(mask_id)`, and its own embed / head (or a check that they equal the target's) |
| RedHatAI/GLM-5.3-Flash-speculator.dspark-preview | **MIT** | DSparkDraftModel: 5 Qwen3 layers, 64 heads x 64, a Markov head (rank 256) and a confidence head, sliding window 2,048, block 8 (up to 8 drafts). Preview: the epoch-2 checkpoint of a 3-epoch run on 1.7M GLM-5.3-Flash-regenerated samples | 20, 28, 32, 36, 40, 44 + the final layer (45) | **no.** A new drafter module. Its final-layer input is likely the pre-norm hidden, not the final-normed row the dump stores. The reference does not implement it either |

- Hidden size 4,096 and vocabulary 154,880 match the target everywhere. All three are compatible at the tensor level.
- **For anything shareable: dump 9 taps, start from DFlash2-G, and spend the 2-3 engine days.**
  - `GLM53_TF_DRAFT_DUMP_TAP_LAYERS=5,9,14,19,24,28,33,38,42`. That is a superset of incoai's 5, so the same dump
    also re-fits incoai. `Doc.taps(layers=...)` slices it.
  - The cost of those taps: generation runs without DFlash2 drafts, so code-like streams decode ~15-30% slower; prose
    is about the same.
- **For private use only:** re-fit incoai. It is served as is today.

## 6. Evaluation and gates (`train/eval_accept.py`)

On held-out documents (2% of documents, split by conversation), per draft position k, cumulative acceptance
a_k = P(drafts 1..k all kept), three ways:

- **greedy**: the draft equals the target's argmax.
- **data**: the draft equals the token the document has. On decode rows that is the target's own sample, the
  engine's measure for the same traffic.
- **T = 0.6 / 1.0**: the expected acceptance of sampled serving. The engine draws drafts with the same keyed Gumbel
  noise as the target's sample, so a draft is kept when both noisy argmaxes agree. Estimated by Monte Carlo over the
  union of both top-ks.

Each is also reported as E[tokens a round], split by row kind and by position below / past the dense limit. For
DFlash2 it uses the engine's chain rule.

Gates, in order:

1. **Parity:** the stock head through the reference reproduces the engine's MTP acceptance within ±0.08 (the GPU
   test's real-model part). If it does not, stop: the reference is wrong.
2. **T1:** held-out a2 and a3 each up by ≥ 0.08 (DECODE-PLAN's gate), a1 not lower, on decode rows and on the
   sampled estimate. Then the engine A/B:
   - `GLM53_TF_MTP_WEIGHTS` on the b-image, exact 10/10 and batchexact 4/4 (drafts cannot change replies; this
     checks that);
   - `bench/acceptpos.py` acceptance by position;
   - the tf / kit / edit suites and 1 / 4-stream multiturn.
3. **T2:** held-out τ up by ≥ 0.4 on prose-like documents (DECODE-PLAN) before any serving A/B.

## 7. Expected gains (from DECODE-PLAN; not measured)

- **T1:** 0.74 / 0.45 / 0.22 → ~0.80 / 0.58 / 0.38.
  - MTP-round E[T] goes 2.41 → ~2.76 for ~+2.5 ms a round.
  - +9-10% on MTP rounds: **sampled cells +8-12%, greedy prose +4-6%**.
  - Calibration: FastMTP's biggest lift is at positions 2-3 (11 → 56%, 2 → 36% on its model). Red Hat on
    Qwen3-Next: 0.897 / 0.719 / 0.476 → 0.912 / 0.776 / 0.616.
  - Our head is strong at position 1 and weak at 2-3, the profile those recipes fix.
- **T2:** prose DFlash2 rounds +0.4-0.7 tokens: **prose +10-20%**, code +5-10%. The least certain number.
  DFlash2-G itself only edges incoai on the base model (+0.044 τ); the gain here comes from fitting the abliterated
  target, whose taps drifted (EXPERIMENTS S4: cosine 0.926 at layer 42).
- **With 0420** (trimmed draft vocabulary): the trimmed head only drafts, and the evaluator scores the full
  vocabulary. With `GLM53_TF_DRAFT_VOCAB` on, expect served acceptance ~0.01-0.02 lower than offline at most (coverage
  98%).

## 8. GPU-phase runbook (order and budget)

| # | Step | Time | Output / gate |
| ---: | --- | ---: | --- |
| 0 | Build the image with every patch through 0430 on both nodes. Copy `train/` next to `tests/` | 40 min | - |
| 1 | GPU tests: `tests/cuda/test_drafter_ref_patches.py` (synthetic parts), 0420's and upstream's GLM tests | 15 min | parity asserts pass |
| 2 | Smoke:<br>- serve production config + `GLM53_TF_DRAFT_DUMP=/sessions/draftdata`, `_TAPS=fp8`, `_TAP_LAYERS=5,9,...,42`, `_GIB=2500`, `WARMUP_LENGTHS=""`;<br>- `gendata.py run` on 300 prompts;<br>- `dumpdata.py stats`;<br>- canary + multiturn 1 / 4 streams with the dump on vs off (the §2.1 overhead);<br>- optional: `GLM53_TF_BATCH=8` generation rate | 1.5 h | bytes / token; prefill ≤ +6%; decode ≤ +1%; replies identical |
| 3 | Parity on real data: stop the server; `DRAFT_TEST_MODEL` / `DRAFT_TEST_DUMPS` run the GPU test's real-model part (expert dequant ~5-10 min, cached with `DRAFT_TEST_EXPERTS`) | 40 min | reference ≈ engine acceptance (gate 1) |
| 4 | Generation: `gendata.py run --target-tokens 1.1e7`, prompts from `collect --n 15000` (run `collect` on a machine with internet and the opencode DB) | ~48 h | ~11M generated tokens, ~1 TB |
| 5 | Teacher forcing: `gendata.py tforce` on full conversations (WildChat / UltraChat with their replies, OpenR1 solutions, code instruct outputs):<br>- 20M rows with taps;<br>- restart with `_TAPS=0`, then 40M rows | ~12 h | +1.2 TB |
| 6 | Train, both Sparks in parallel:<br>- T1 `mtp_distill.py` on the head node (6k steps, ~3-8 h; a second run adding `--kinds d,p` teacher-forced rows);<br>- T2 `dflash_distill.py` on the worker node after copying the dump (incoai re-fit for private use: 20k steps, ~6-20 h) | ~1 day | `eval-*.json` a checkpoint |
| 7 | Offline gates (§6), then the engine A/B:<br>- `GLM53_TF_MTP_WEIGHTS` on production + the usual gates;<br>- `DRAFTER=<export>` for the incoai re-fit | 2 h | adopt / not |

Total downtime is ~4 days, most of it step 4. Shrinking step 4 to 24 h (5-6M tokens) is a reasonable first pass for
T1 alone. Steps 2-3 decide early whether the whole program is worth running.

## 9. Risks

- **Reference drift.** The reference is near-exact, not bitwise. Gate 1 catches a real mismatch; ±0.02 of noise
  between the reference's and the engine's acceptance is expected.
- **Prefill rows are fast-kernel rows** (0080 / 0240 prefill arithmetic); decode rows are exact-path rows. Both are
  the target's; the difference is last-bit noise.
- **The long-context approximation** in training windows. Agent traffic has 20-100k prompts, and at inference the
  head attends to DSA's top-2,048 of the whole history. Measure with `--sparse-docs`. If position ≥ 2,051 acceptance
  lags, train some windows whole from position 0 (the chain supports it; slower).
- **Overfitting to the prompt mix.** Hold out by conversation (done). Keep the stock head's acceptance on the
  held-out set as the floor for every position.
- **Disk and memory.** The dump stops at `_GIB`. Unified memory: the host queue is bounded at 4 segments. Do not run
  it inside the 4 x 250k stress configuration.
- **Privacy.** The production dump holds the user's sessions as token ids and hidden states (text can be decoded
  from the ids). Keep it on the Spark and delete it after training.
- **T2 serving gap.** A shareable DFlash2-G-based drafter needs the loader work in §5.3 before it can be A/B'd.
- **The one-sided tap knob.** `_TAP_LAYERS` switches off DFlash2 drafts while it differs from the drafter's taps.
  Generation is then slower on code, and a data server with that knob must not be left as production.

## 10. Licences (checked 2026-09-29; not legal advice)

- **zai-org/GLM-5.3-Flash: MIT** (HF metadata `license:mit`).
- **neko-legends/GLM-5.3-Flash-Uncensored-EXL3** (our `MODEL_PATH`; gated): card licence
  `other: shapleymcg-license-1.0`. The card calls it a mixed-license artifact:
  - The weights chain (zai-org MIT → orcarouter's uncensored FP8, MIT) is "redistributable under MIT terms". This
    covers every BF16 non-expert tensor, i.e. everything T1 trains and exports (the MTP layer's attention, `eh_proj`,
    norms, router, shared expert).
  - The per-expert `suh` / `svh` / `mcg` scales and the trellis codes are ShapleyMcg-licensed.
  - The quantizer's own contributions are under either licence.
- **The ShapleyMcg licence, two versions:**
  - v1.0, the GitHub text of 2026-08-23 / 24: attribution-required and source-available, free to everyone except one
    named excluded party. Its "Derivative" is broad: any model or dataset "produced in whole or in part by running the
    Work or by applying the recipe, schema, or format it implements".
  - The current GitHub LICENSE (2026-09-10): the "Local Inference Lab Attribution License 1.0", MIT-derived. It
    explicitly extends the grant to fine-tunes and distillations, on condition of attribution (author, a link to the
    upstream source, a link to the project home, in the model card's first screen).
  - The copy inside the gated HF repo (`ShapleyMcg-LICENSE`) could not be read without a login. Check it before
    publishing anything.
- **What that means for us:**
  - Private use of dumps and trained heads: allowed.
  - Sharing: the conservative reading treats our dumps and trained drafters as derivatives (they were produced by
    running the EXL3 model and trained through its experts). Publish with the attribution notice and the upstream
    links, and without the EXL3 tensors themselves.
  - T1's export contains only MIT-chain tensors.
  - We are not the excluded party.
- **incoai/GLM-5.3-Flash-DFlash2: CC BY-NC-ND 4.0.** A private re-fit is fine; sharing it is not.
- **canada-quant/GLM-5.3-Flash-DFlash2-G: Apache-2.0.** Self-generated training data, no third-party drafter weights
  (its PROVENANCE.txt).
- **RedHatAI/GLM-5.3-Flash-speculator.dspark-preview: MIT.**
- **Prompt datasets:** §2.3. They supply prompts only; every reply and label is the target's.

## Sources

- **Recipes:**
  - FastMTP [arXiv 2509.18362](https://arxiv.org/abs/2509.18362)
  - Red Hat FastMTP heads [2026-09-08](https://developers.redhat.com/articles/2026/09/08/optimize-vllm-speculative-decoding-fastmtp-heads)
  - DFlash [arXiv 2602.06036](https://arxiv.org/abs/2602.06036)
  - DSpark [arXiv 2607.05147](https://arxiv.org/abs/2607.05147)
  - EAGLE-3 [arXiv 2503.01840](https://arxiv.org/abs/2503.01840)
- **Models:**
  - [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
  - [neko-legends/GLM-5.3-Flash-Uncensored-EXL3](https://huggingface.co/neko-legends/GLM-5.3-Flash-Uncensored-EXL3)
  - [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
  - [canada-quant/GLM-5.3-Flash-DFlash2-G](https://huggingface.co/canada-quant/GLM-5.3-Flash-DFlash2-G), including
    its PROVENANCE.txt and config.json
  - [RedHatAI/GLM-5.3-Flash-speculator.dspark-preview](https://huggingface.co/RedHatAI/GLM-5.3-Flash-speculator.dspark-preview)
- **Licences:**
  - [ShapleyMcg LICENSE](https://github.com/brandonmmusic-max/shapleymcg/blob/main/LICENSE): current text and commit
    25794f78
- **This repo:**
  - DECODE-PLAN §3.2 / §4.3
  - RESEARCH-NIGHT §1 / §4 (L1)
  - DRAFT-VOCAB (0420)
  - RESULTS W10 (speeds)
  - EXPERIMENTS S4 (tap drift)
