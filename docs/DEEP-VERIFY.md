# Deeper verify windows (patches/0380) and cross-request lookup (N3): estimates and GPU plan

Offline work, 2026-09-28. No GPU and no Spark were used. The patch, `patches/0380-glm-deep-verify.patch`, applies on
the production stack (through 0360; also with 0390). New files: `bench/lookupsim.py`,
`tests/cuda/test_deep_verify_patches.py`. `bench/draftsim.py` was extended. Simulator outputs: `results/sim0380/`.
Background: RESEARCH-NIGHT.md §1 (the 8-row ceiling) and L2 / N3; ADAPTIVE-DRAFT.md (the simulator).

## 0. Bottom line

- **The patch.** `GLM53_TF_MAX_DRAFT_ROWS` = 8..16 (load-time, both ranks equal, **default 8 = today**). A round may
  verify up to N - 1 drafts. Nothing new decides depth: 0071's per-draft cost depths and 0020's lookup gate have
  their caps raised, so a row past the 8th is drafted only when its expected token beats the request's rate times
  its marginal ms. `GLM53_TF_DFLASH_BLOCK` is an opt-in experiment that widens the DFlash2 block pass.
- **Exact.** Every decode / verify / MTP kernel is already row-independent from 1 to 512 rows. Exact prefill
  chunks run the same code, and `qmm.bucket` puts 1-16 rows in one tile size. The accept rule and `commit` are
  unchanged. No Triton kernel changed. The kernel PTX of 0290's check is identical at 8 and 16 (33 / 33).
  CPU tests of the real batch `Stepper` and the lone `auto_decode` at 16 rows give replies equal to serial.
- **Cost.** About +115 MB a rank at a 1M capacity (below). The load-time calibration times windows up to N rows.
- **Gains are narrow.** The published DFlash2 drafter has `block_size` 8, so it proposes 7 positions and never
  more. DFlash2 rounds are the ones that hit the ceiling on code-like and repetitive streams. The extra rows
  therefore help only content that other drafters can extend:

| content (single stream unless noted) | rows 16, block 8 (the patch as shipped) | + DFlash2 block 16 (experiment) | source |
| --- | ---: | ---: | --- |
| **copy / edit cells** (file re-emitted with an edit) | **+16% to +30%** (98 -> 114-128 tok/s) | same | lookupsim, token-exact |
| JSON records | +1% to 0 | not simulated | lookupsim |
| agent turns (826 real GLM-5.3-Flash agent steps, 386k tokens) | **+0.2% to +0.6%** | not simulated | lookupsim |
| prose / chat (63 recorded streams) | -0.1% | -0.5% to -0.8% | draftsim |
| code generation (20 recorded code-like streams) | -0.2% | -1.9% .. +2.7% (tail 0 .. 1) | draftsim |
| repetitive (count, lists; 23 streams) | +1.5% | -1% .. +14-26% (tail 0 .. 1) | draftsim |
| 4 streams, aggregate (23 runs) | -0.1% | -1.1% .. +1.7% | draftsim |

  "tail" is how sure the drafter would be at positions 8-15, relative to position 7 (1 = as sure, 0 = never
  right). DBloom (arXiv 2608.30427) finds that widening a block at inference, without post-training, does not work.
  The realistic reading of the block-16 column is therefore its lower half (tail ≤ 0.75: -2% to +3%).
- **Recommendation.**
  - Keep the default 8.
  - Give `GLM53_TF_MAX_DRAFT_ROWS=16` one GPU A/B. It is worth adopting if the edit / copy cells gain ≥ +10% and
    nothing else moves outside noise; that is the expected result.
  - Run `GLM53_TF_DFLASH_BLOCK=16` in the same window only to measure DFlash2's acceptance at positions 8-15. It
    decides whether a block-16 drafter (RESEARCH-NIGHT L1b) is worth training.
- **N3 (cross-request suffix index): not built.** On the recorded agent traffic, a global index of every earlier
  reply adds 0.0% (+0.0 on top of per-request lookup at 8 or 16 rows). With the gate off it raises copyable coverage
  from 61.4% to 63.4% of tool-call tokens. Almost everything an agent copies is already in its own prompt, which
  0020 indexes. It fails the research note's bar (≥ 0.3 tokens a round on ≥ 20% of agent tokens).

## 1. What fills rows past the 8th

| drafter | proposes | past 7 drafts? |
| --- | --- | --- |
| DFlash2 (`incoai/GLM-5.3-Flash-DFlash2`) | one block pass over `[pending, mask x (block_size - 1)]` | **no**: `dflash_config.block_size` = 8 (the 7-position block the engine's docstrings and EXPERIMENTS §5 describe; `Drafter.propose` clips at `block - 1`). `GLM53_TF_DFLASH_BLOCK=16` runs 16-row passes (masks 8-15 out of training distribution; the bidirectional block can also change picks 1-7) |
| MTP head | a chain, one head step (1.68 ms) a draft | yes, up to N - 1; each step must pay (`DepthOptimizer.mtp_next`). The recorded MTP acceptance is too low past 3-4 for this to matter, except on repetitive streams (+1.5%) |
| lookup (0020) | a copy of the tokens that followed the longest earlier match of the history's suffix | yes, up to N - 1 (`lookup.MAX_DRAFTS`), priced per match-length band |

So the ceiling rounds of RESEARCH-NIGHT §1 are DFlash2 rounds that 0380 alone cannot extend: 34% of code-like and
82% of repetitive DFlash2 rounds keep all 8 tokens. Lookup rows help where the reply copies its context.

Considered and not built: extending a full DFlash2 chain with a lookup copy (DFlash2 for rows 1-7, the copy after).
On agent traffic the gate-free potential is large (below: 40% of tool-call tokens sit in copy runs of 9+ tokens),
but where lookup has a copy, DFlash2 usually has it too. A lookup round already takes those regions whenever its
band pays. The upper bound, if every such region ran at 16 tokens a round instead of 8, is about +3% on agent
turns. It is only worth building after the GPU window measures how often DFlash2 hits 8 inside copy regions (the
new `depths` stat below).

## 2. The patch (`deep.py` + small edits)

| where | before | with `GLM53_TF_MAX_DRAFT_ROWS=N` |
| --- | --- | --- |
| `engine.MAX_ROWS` | 8 | N: `Engine(max_rows=N)` (buffers already hold `max(max_rows, prefill rows)` = 512 in production), capacity context + N, policy validation (`cN:P`, `fN`, `lN`, `GLM53_TF_AUTO_FDRAFTS` up to N - 1), the rank check (with `GLM53_TF_DFLASH_BLOCK`) |
| `depth.MAX_DRAFTS` | 7 | N - 1: `Calibration` keeps each of N - 1 positions apart; `om[N]` / `of[N]`, `as_cost` |
| `lookup.MAX_DRAFTS` | 7 | N - 1 (still bounded by the request's cost table) |
| `knobs` `auto_fdrafts` | 1..7 | 1..N - 1 |
| `latent.SPARSE_ROWS` | 8 | max(8, N): preallocated sparse partials (captured long-context steps allocate nothing) |
| `sparse.LongScratch` (0050) | 8 rows | N rows (via `max_rows`); long-context graph rows 1..N, captured lazily as before |
| batch slots' `State` rows (0120) | min(8, e.rows) | min(N, e.rows): KDA projection and replay rows of slots 1-3 |
| `Batcher` graph rows, pad / bucket masks, batched KDA rows | 8 | N (`GLM53_TF_BATCH_GRAPH_ROWS` defaults to N; `n x N` <= buffer rows: 64 <= 512) |
| `calib.window_costs` | windows 2..8 on one line | the 1..8 table exactly as an 8-row calibration fits it (the spans still start every 8 tokens, so the first 8 windows time the same tokens); 9..N on a second line through the 8-row cost, along the slope measured over 8..N |
| stats | `keeps`, `drafters` | + `depths` (drafts verified each round; lone engine and batch) |

Unchanged (checked): `qmm` / `glue` / KDA (`kda.cu` loops rows at run time) / EXL3 expert kernels (windows with
16+ members per expert already take the same-bits `grouped_loop`), dense graphs (`GRAPH_ROWS` 1..6; windows of 7+
rows run eagerly as 7 and 8 did), the index rings (`index_ring_rows` = 256 for any N ≤ 192), the KV pool's slack
(64 tokens past prompt + max_tokens; a window never passes `max_tokens`), snapshots and sessions (no row-sized
state), the DFlash2 context update (graphs for 1..block rows; longer catch-ups run eagerly, as backlogs of up to 32
rows already did), the MTP backlogs (32 rows), 0230's RoCE path (exchanges over 256 KiB fall back to NCCL on both
ranks by size, as before), `decode_v2` (off in production; its limits are 16 rows a window and 32 a call, with
checked fallbacks).

**Memory**, per rank at a 1M-token capacity, 16 rows against 8:
- `LongScratch` of the main and MTP buffer sets: +25 MB each. Scores and int64 keys: rows x 262k pools x 12 B.
- latent sparse partials: +2.6 MB each.
- slots 1-3: +20 MB each. 34 KDA layers x 8 rows x (12,576-wide bf16 projection + 32 heads x 1.5 KB replay).
- Total: **about +115 MB a rank**, plus whatever the extra lazily captured graphs take (measured in the GPU plan).
  the worker node's 4 x 250k stress minimum was 8.14 GiB (W8), so there is room.

**Exactness.** A token at position p is the keyed sample of p's own logits row. Rows do not depend on their window
mates or the window's length. The same compute path runs exact prefill chunks of any size, and 0003 / 0050 / 0120 /
0200 test that bit for bit. `commit(R, keep)` replays the kept prefix. A deeper window changes how many positions a
round confirms, never which token a position gets.

- **CPU tests** (`tests/cuda/test_deep_verify_patches.py`; the file re-runs itself with the knob at 16 in a child
  process, because the knob is read at import):
  - knobs, the caps and the calibration table;
  - cost depths past 7 only for confident drafts (DFlash2 picks, lookup bands, MTP chains);
  - lookup copies of 15 tokens;
  - on 0180's hostile fake model: the real `Stepper` / `Batcher` with `auto` + gated lookup, forced `l15:3`,
    `of15`, `om15` and a mix, requests alone and in shared rounds (windows of 9-16 rows verified, kept and cut);
  - the same for the lone `decode.auto_decode`, and a follower batcher deciding the same windows.

  Every reply equals serial decoding. All pass at 8 and at 16, as do the existing depth / lookup / calib / knob /
  adapt / batch / session / pool suites at 8 (164 passed). At 16, five existing tests fail by design: they assert
  the 7-draft cap (`test_depth_patches::test_specs`, two `test_knob_patches::test_parse_rejects` cases, two in
  `tests/test_glm_lookup.py`). Run them at the default.
- **GPU tests in the same file** (not run yet), all on the synthetic checkpoints:
  - dense windows 1-16 == serial rows;
  - long-context windows and MTP steps of 1-16 rows through the graphs == upstream eager;
  - drafted replies, lone and batched, greedy and sampled, with a 16-row DFlash2 block, == serial.
- **PTX**: `tests/kvpool_ptx.py --against` the tree without 0380 gives 33 / 33 kernels identical, with the knob at
  8 and at 16. No Triton source changed.

## 3. Cost model and how the estimates were made

- **Row cost past 8.** Each extra row reads the experts no earlier row of the window read: 4.7-6.5 ms a row
  (RESEARCH-NIGHT: V(12) ≈ 87, V(16) ≈ 106 ms from the table's own curve, i.e. ~4 ms; W5: 6-7 ms a real row across
  sequences). The engine's table past 8 is what the load-time calibration measures on consecutive tokens. Both sims
  run 4.7 and 6.5 ms.
- **draftsim** (`bench/draftsim.py`; ADAPTIVE-DRAFT.md §3) replays the recorded concurrent runs (W1, W5, W6, W8:
  23 four-stream runs, 106 streams) through the engine's own `depth` / `batchplan` / `DrafterChoice` code on
  de-censored per-stream acceptance. New in 0380:
  - `rows=`, `block=`, `tail=` variants;
  - `--alone` (each stream on its own, gains per class);
  - `--deep-row`, `--block-ms` (a 16-row block pass: 4.4 ms against 3.9, estimated).

  The base variant reproduces the 0340 numbers exactly (81.8 / 98.9 tok/s over the 11 W1 / W5 runs).
- **lookupsim** (`bench/lookupsim.py`, new) replays real token streams, tokenized with the GLM-5.3 tokenizer,
  through the engine's `lookup.Lookup` gate and `DepthOptimizer`:
  - Lookup rounds keep exactly the drafts that equal the next reply tokens.
  - Other rounds draw DFlash2 keeps from the recorded per-class distributions (prose for reasoning and text,
    code-like for tool calls and code), at the same draw per position in every variant.
  - A round also costs 3.9 ms of host work (PROFILE §5).
  - Sources: glmbench's writable cells (count, primes, 25 JSON records, three edit cells re-emitting a file of this
    repo with an edit), and 826 agent steps served by GLM-5.3-Flash. Those are 203 by this cluster (`dgxspark`) and
    623 by the Z.ai API, from the local opencode database. They are read locally; only aggregates are printed or
    stored.
  - Caveat: drafter rounds are drawn independently of the text, so lookup gains are upper-leaning.

### 3.1 Single stream, by recorded stream class (draftsim `--alone`, 8 seeds; `results/sim0380/alone-row*.txt`)

| variant | prose (63) | code-like (20) | repetitive (23) |
| --- | ---: | ---: | ---: |
| base (8 rows) | 48.1 tok/s | 71.3 | 92.6 |
| rows 16 | -0.1% | -0.2% | +1.5% |
| rows 16, block 16, tail 1.0 | -0.5 / -0.7% | +2.7 / -0.1% | +26.4 / +13.7% |
| tail 0.9 | -0.5 / -0.6% | +1.6 / +0.2% | +5.6 / +2.1% |
| tail 0.75 | -0.7 / -0.8% | -0.5 / -1.1% | +2.6 / +0.6% |
| tail 0 (positions 8-15 useless) | -0.5% | -1.9% | -1.0% |
| oracle at 8 rows (knows L) | +18.8% | +17.3% | +4.6% |
| oracle, 16 rows, block 16, tail 1 | +22 / +22% | +36 / +30% | +36 / +24% |

(pairs: deep row 4.7 / 6.5 ms)

- The code-like class shows that a longer block is not enough by itself. Even at tail 1, the cost depths buy
  ≤ +3% where an oracle buys +30-36%. The drafter's per-draft confidence is too noisy for a 5-6 ms row to be priced
  well at positions 8-15.
- The repetitive class is the only one where a better block pays in practice, and only if the drafter stays sure
  past position 7.

### 3.2 Four streams (draftsim, 23 runs, 8 seeds; `results/sim0380/batch4-row*.txt`)

base 82.4 tok/s:
- rows 16: -0.1%;
- block 16 with tail 1.0 / 0.9 / 0.75 / 0: +1.4..1.7% / -0.6..-0.8% / -0.8..-0.9% / -0.9..-1.1%;
- oracle: +21.0% at 8 rows, +32.5% at 16.

In shared rounds a row prices at the aggregate rate and the `ROW_MS` floor, so deep rows rarely pay. Per slot in
the batch the patch is free (it sizes every slot), but it is not a lever.

### 3.3 Token-exact lookup replays (lookupsim, 3 seeds; `results/sim0380/lookup-*.txt`)

| cell | 8 rows | 16 rows (4.7 / 6.5 ms a row) | 16 + global index |
| --- | ---: | ---: | ---: |
| edit (3 files, 1.6-3.6k tokens) | 97-98 tok/s | 125-128 / 113-114 (**+30% / +16%**) | same |
| count 1..200, primes | 92-93 | same | same |
| JSON records | 74.3 | 74.9 / 74.2 | same |
| agent steps (826) | 57.6 | 58.0 / 57.8 (+0.6% / +0.2%) | +0.6% / +0.2% |

- Agent steps: 8% of the requests gain ≥ 5%; the median request is unchanged.
- Gate-free potential on the agent steps: every position with a 4-token match runs a copy.
  - 61.4% of tool-call tokens and 14.9% of prose tokens could be committed by lookup rounds.
  - 40.6% and 4.3% of them in rounds of 9+ tokens.
  - A global index raises this to 63.4% / 15.5%.
- The gate takes little of it. Those copies mostly start from 4-15-token matches, a band whose learned acceptance
  (0.6-0.65) does not beat a DFlash2 round's rate. That is correct pricing when DFlash2 would do as well there,
  which the replay cannot see (§1).

## 4. N3: cross-request suffix drafts

- **Data.** The 0300 request log holds no text (hashes only), and it lives on the Sparks. The local opencode
  database has real agent transcripts from this cluster's GLM-5.3-Flash (203 steps) and the same model via Z.ai
  (623 steps). They were replayed in time order: the global index holds every earlier reply, and a round uses the
  global match when it is longer than the request's own.
- **Result.** +0.0% over per-request lookup at 8 or 16 rows. There were 3.1k global matches, but they are rarely
  longer than the prompt's own and rarely pass the gate. Gate-free coverage rises by only +2.0 points on tool-call
  tokens. Agents copy from their own context (files read, earlier edits), and 0020 indexes all of it.
- **Decision: not implemented.** A global tree would matter for many sub-agents or users repeating the same outputs
  with different prompts. That is not this traffic. Revisit if N4's acceptance histogram on production shows
  lookup rounds with long matches the per-request index misses.

## 5. GPU test plan (one window, ~75 min, prod down)

Image: through 0380 (other patches as the window carries). Loads from `config/prod.env` plus overrides (W5's
`load.sh` pattern). The calibration table re-measures on the first start of each config: its key holds the knobs.

1. **Tests** (the worker node, ~15 min):
   - `tests/cuda/test_deep_verify_patches.py` in the image. It spawns the 16-row child itself.
   - Gates: GPU dense 1-16 == serial rows; long 1-16 + MTP 1-16 == eager; lone + batched drafted == serial,
     greedy and sampled, with a 16-row DFlash2 block.
   - Also, at the default: `test_longctx_patches.py`, `test_depth_patches.py`, `test_lookup_patches.py`,
     `test_batch_parallel_patches.py`.
2. **A: production** (as W8) vs **B: `GLM53_TF_MAX_DRAFT_ROWS=16`** vs **C: B + `GLM53_TF_DFLASH_BLOCK=16` +
   `GLM53_TF_AUTO_FDRAFTS=15`**. Each load (~15 min):
   - boot log: `drafter costs (ms): verify ...` must list N windows, with the 9..16 slope (expected 4-6.5 ms);
   - `glmbench --suites tf,tweet,kit,edit --reps 3 --long-tokens 512` (code, chat, sequence, code, json, hashmap,
     structured, essay, edit cells);
   - `glmbench --suites exact` (gate **10/10**);
   - `multiturn --modes batchexact` (gate **4/4**);
   - `multiturn --modes concurrent --streams 1,4 --reps 3 --long-tokens 512`.

   Keep every stats `keeps`, `depths` and `drafters`.
3. **Per-position acceptance (C).**
   - From `depths` and `keeps` of DFlash2 rounds: P(draft j kept | drafts < j kept, j verified) for j = 1..15.
   - Compare j = 1..7 with A's. If C's first 7 are lower, the 16-row block hurts the positions that matter.
   - Re-fit draftsim with C's streams (`--variants base:rows=16:block=16`) to turn the measured tail into a number.
4. **Memory.**
   - `nvidia-smi` / MemAvailable after load, A vs B: expected +~0.1 GiB a rank.
   - `multiturn --modes stress` (4 x 250k) on B. Gate: MemAvailable ≥ 8 GiB on both nodes (W8's minimum 8.14 on
     the worker node).
   - After the concurrent runs: the long-context graph count (`GLM53_TF_LONGCTX_MAX_GRAPHS` is 256; 16 rows x 2
     parities x ~10 buckets can reach it; later keys then run eagerly).
5. **Adoption.**
   - Adopt B if the edit cells gain ≥ +10%, every other cell and the 4-stream aggregate are within noise (±3%),
     exact 10/10, batchexact 4/4, and memory holds. Expected: edit +16-30%, the rest ±1%.
   - Adopt C only if its measured tail makes draftsim's code / repetitive gain ≥ +5% with prose unchanged. Expected:
     no (DBloom); C's value is the measurement for a block-16 drafter (L1b).

## 6. Files

- `patches/0380-glm-deep-verify.patch`:
  - `deep.py` (new): the knobs, 8..16 / the block ≤ N;
  - `engine.py`: `MAX_ROWS`, the rank check, the drafter block, the calibration spans, `depths`;
  - `depth.py`, `lookup.py`, `knobs.py`: the caps;
  - `latent.py`: `SPARSE_ROWS`;
  - `calib.py`: the two-line table;
  - `batch.py`: `depths`, docstring.
- `tests/cuda/test_deep_verify_patches.py`: host, CPU fake-model and GPU tests.
- `bench/draftsim.py`: deep variants, `--alone`. `bench/lookupsim.py`: token-exact lookup replay and N3.
- `results/sim0380/`: the simulator outputs quoted above. Aggregates only; no transcript text.
