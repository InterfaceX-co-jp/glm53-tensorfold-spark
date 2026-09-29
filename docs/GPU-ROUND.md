# GPU-resident decode round (patch 0450, `GLM53_TF_GPU_ROUND`, off by default)

> Offline work: no GPU was used and the Sparks were not touched. Round-time numbers are W11's measurements
> (`docs/RESULTS.md` W11, `docs/DECODE-PLAN.md` after W11). Code citations are `glm5_next/cuda/` in the tree through
> 0430 (before this patch; later patches move the lines). Savings are estimates for the GPU window to confirm (§6).

## 0. Bottom line

- **What 0450 is.** Three parts behind one knob, each off unless named:
  - `sample`: the exact keyed sampler, the greedy pick and the drafts' probabilities run on the GPU with the host's bits.
  - `resident`: steady batched decode runs as GPU-resident rounds. Staging, sampling, the accept rule, commits,
    backlogs, MTP chains and DFlash2 chains/depths all run on the device. The host does its bookkeeping for round
    k - 1 while round k's forward runs.
  - `kda`: KDA commits are folded into the next window's chain, with every slot's chain in one launch a layer.
- **Exactness is proven offline for what can be proven offline.**
  - The device draw reproduces `choose_rows` / the greedy lexsort / `_probability` bit for bit. This covers 1,900+
    random row sets in Triton's interpreter, on both Triton 3.7.1 (the image's) and 3.8.
  - The libm port equals the real Ubuntu 24.04 aarch64 libm. The scalar spec was checked on 10.5M inputs and the
    Triton port on 5.4M inputs, both run in unicorn (§4).
  - Resident rounds on a hostile fake model give replies == serial, committed states == serial, and the MTP-head
    cache / DFlash2 context correct position by position. This holds with one rank and with two ranks in threads,
    for both host modes (§4).
- **Expected gain, sized on W11: small, not +40%.**
  - 1 stream: -0.6 to -1.2 ms a round (+1-2%).
  - 4 streams: -3.5 to -7 ms (+3-6%, mid +4.5%).
  - Why small: after 0370 the single-stream GPU idle is 1.8 ms, only 0.7 ms of it outside the verify forward. At 4
    streams, the per-slot items this patch batches (KDA, idle) are about 8 of the 26 ms the W9 fit attributes to
    slots. The rest is per-slot attention (0.9 ms a slot) and per-slot DFlash2 passes (2.4 ms a slot), which are the
    next steps (§5).
- **Deliberate deviations from the brief, with W11 as the reason.**
  1. **The whole round is not one captured CUDA graph.**
     - At 4 streams, W11 found graph keys (slots x rows x modes x parities) rarely repeat: 60-80% of rounds run eager
       and captures never pay back (`BATCH_GRAPHS=0` is +2.6%).
     - Resident rounds therefore keep the batcher's graph policy per key for the verify forward and for the MTP head
       passes (`CAPTURE_AFTER`). The glue is 5 fused kernels a round, not a graph.
  2. **MTP chains are paced by the device's decisions (peek mode, the default), not run blind a round ahead.**
     - The chain's length depends on each draft's probability. An unneeded MTP head step costs ~1.3 ms (W11: head
       834 us + experts 240 us + the rest). That is more than the 0.7 ms of single-stream idle the fully lagged mode
       removes.
     - Peek mode reads a small pinned record (the slots' counters, a few hundred bytes) per chain step (a copy and its event: no stream synchronization, no
       candidate readback, no numpy), so chains run exactly the steps they need and windows carry no padding.
     - The fully lagged mode (`GLM53_TF_GPU_ROUND_PEEK=0`: no read at all while launching, stale bounds, padded rows)
       is kept for the 4-stream A/B.
  3. **Lookup rounds are not resident.** A prompt-lookup match drains to the normal path, which drafts it. A device
     n-gram index is future work.

## 1. Where a decode round's time goes today

### 1.1 The measured round (W11, rank 0, ms a round)

| | 1 stream, prose | 4 streams |
| --- | ---: | ---: |
| round (uncaptured) | **53.3** (2.42 tokens) | **121.3** (9.1 tokens) |
| routed experts + dense q4 (bandwidth kernels) | 43.9 | 97.2 |
| exchanges exposed (RoCE, 100-115 a round) | 2.3 | 4.2 |
| DSA attention + indexer | 2.1 | 4.8 |
| KDA (chain, conv, replay) | 1.6 | 6.0 |
| hc + router / combine | 2.65 | 3.4 |
| **GPU idle inside the round** | **1.8** (1.1 inside the verify forward) | **5.1** (2.5 inside the verify forward) |
| drafting kernels (launch phase) / host time in the drafting ranges | 4.3 / 5.0 | 11.4 / **16.7** |
| sampling kernels | 0.08 | 0.6 |
| kernels a round | 1,853 | 2,621 |
| empty kernel launch, eager / in a graph (probe) | 1.62 / 0.47 us | |

The host-side critical path lies in the gaps between GPU work. Every one of them is a host readback followed by
host arithmetic and a launch:

| step | code (tree through 0430) | host work on the critical path |
| --- | --- | --- |
| verify sampling | `batch.sample_multi` (`batch.py:306-347`): per-slot `topk` + `cat`, one all-gather, then **`buf.view(world, -1).cpu()` (:333)**, `choose_rows` in numpy | device idle from the gather's end until the next launch: the readback, the numpy draw (~35 us / 16 rows), Python |
| accept + commit | `Stepper.accept` (:755-816) → `forward.commit` (`forward.py:705-724`): per slot `replay_layers` (one launch, 71 MB read + written when a draft was rejected), `_conv_shift`, `set_pos` (a fill), `save_index_tail` (ring copies) | per-slot launches from Python after the readback; the backlog copies (`m_rows` / `f_taps` `copy_`) likewise |
| MTP chain | `decode.draft` (`decode.py:279-326`): each draft `e.sample` → `sample_rows` **`.cpu()` (:54)**, numpy draw + `_probability`, `opt.mtp_next`, then `e.mtp([d], prev)` (a graph) | one hard sync + numpy + Python a chained draft; batched chains (`MtpChains.run`, `batch.py:420-560`) one `sample_drafts` **`.cpu()` (:379)** a step |
| DFlash2 | `Drafter.candidates` **`self.packed[:, :depth].cpu()` (`dflash2.py:526`)**, then `Drafter.chain` in numpy (:538-567: selector edges `succ @ (pred * proj)`, keyed noise, `f_depth`) | one hard sync + the numpy chain a slot |
| between rounds | 0370's plan rider | ~0 (W11: 0.00-0.01 ms) |

In W11 the single-stream idle outside the forward is 0.7 ms: 0370 already took the gap between rounds away. At 4 streams it
is 2.6 ms, plus 2.5 ms inside the forward. Those rounds are 60-80% eager, and the host's launch stream falls behind the
GPU's small kernels. Host time in the drafting ranges (16.7 ms) exceeds the drafting kernels' GPU time (11.4 ms).

### 1.2 The per-slot fixed cost (~6.6 ms a slot, W9), from the code

W9's fit: a round is ~21 ms, plus 12.1 ms per slot-row, of which ~5.5 is the row and ~6.6 is fixed per slot. W11's
4-stream vs 1-stream kernel families decompose that fixed part (ms a round, (4-stream - 1-stream) / 3 = per extra
slot):

| per-slot item | 4 s - 1 s | per slot | what it is in the code |
| --- | ---: | ---: | --- |
| drafting passes | 11.4 - 4.3 | **2.4** | per-slot DFlash2 block passes (`Drafter.propose` per slot from `Stepper.propose`, `batch.py:743-747`): 5 drafter layers + the target head over 7 rows (178 MB, 0.83 ms) + taps; MTP chains are batched (0200 `BATCH_MTP`) |
| KDA | 6.0 - 1.6 | **1.5** | `batch._kda` (`:155-176`) per slot per KDA layer (34): `sp.copy_`, `kda.chain` (32 blocks of 1,024 threads on 48 SMs: a third of the GPU idle), `bkout.copy_`; `commit` per slot: `replay_layers` (71 MB read + written a slot a rejected round) |
| GPU idle | 5.1 - 1.8 | **1.1** | per-slot sampling / accept / commit / drafting from Python after hard syncs; eager rounds whose launch stream the host cannot keep ahead of |
| attention + indexer | 4.8 - 2.1 | **0.9** | `batch._dsa` (`:180-252`) per slot per DSA layer (11 + the MTP layer): `latent._attend` (`latent.py:893-945`): the latent cache write, `index_update`, dense attention or `select_tokens` + sparse attention, all small grids |
| exchanges | 4.2 - 2.3 | 0.6 | more and larger all-gathers (per-slot DFlash2 passes' o / down partials, candidates) |
| **sum** | | **~6.5** | vs W9's 6.6 |

The patch addresses the KDA (§2.3), idle (§2.2) and host-side per-slot work (§2.2). It does not address per-slot
attention or per-slot DFlash2 passes (§5).

## 2. Design

### 2.1 `sample`: the exact keyed draw on the device (`gpusample.py`)

The host's rule (`exact_sampling.choose_rows`, `batch.sample_multi`, `decode._probability`) is float64 numpy: an
order by (value desc, id asc), `scaled = value / max(T, 1e-6)`, a top-p cut from `exp`, `sum` and `cumsum`, and
`argmax(scaled - log(-log(u)))`, with `u` from splitmix64. `_choose_kernel` computes the same operations in the same
order, one program a row:

- **The order.** A candidate's rank is the number of candidates before it (float32 compares, ids distinct across
  ranks). The top-k keep is `rank < k`.
- **Uniforms and noise.** The splitmix64 uniforms use uint64 arithmetic. `(x >> 11) * 2^-53 + 2^-54` is an exact
  product and one rounded add.
- **log / exp: glibc 2.39's aarch64 build, instruction for instruction.** numpy calls libm for float64 on aarch64.
  The image is Ubuntu 24.04. GCC contracted glibc's C into `fmadd` / `fmsub` at places the source does not show, so
  the port follows the disassembly of Ubuntu's `libm.so.6` (2.39-0ubuntu8.9, `__log` / `__exp`, the GLIBC_2.29
  symbols). It covers both `log` paths (near 1, table), every `exp` path (tiny, normal, `specialcase` overflow and
  subnormal, huge), glibc's own tables (read from the binary, equal to `e_log_data.c` / `e_exp_data.c`), and
  `log(+-0) = -inf`. `u` rounds to 1.0 when `x >> 11 = 2^53 - 1`, and the host then takes `log(-0.0)`.
- **Sums.** `probs.sum()` is numpy's pairwise sum (8 accumulators, the fixed combine). This is checked against numpy
  1.26 / 2.0 / 2.2 / 2.5: the reduction starts from the identity, not the first element. `cumsum` is in order.
- **No contraction.** The kernels compile with `enable_fp_fusion=False` (every float64 op an explicit `.rn`; the only
  fmas are the written ones), and every float64 constant comes from a table (a float literal in Triton is float32).
- **Limits (`fits`).** At most 128 kept and 256 gathered candidates a row. Other rows are drawn on the host.
- **A load check.** Both ranks compare the kernels with the host's own numpy (libm on uniforms and exp ranges, 4,096
  random draws) before serving. A difference on either rank turns the device draw off on both, with the reason in
  the log.

The host reads the tokens (and probabilities) only, and runs no numpy. Used in `sample_multi` (rider words included),
`sample_drafts` and `decode.sample_rows` (the lone engine, prefill's first token, `decode.draft`).

### 2.2 `resident`: GPU-resident rounds (`resident.py`, `gpuround.py`)

**When.** At the end of a normal batch round, both ranks enter resident rounds when:

- rank 0's 0370 plan rider rode empty (no queue, no cancel, no prompt prefilling, not stopping), and
- every active slot is eligible. Eligible means:
  - decoding, not finished;
  - drafting with s / m / f (not lookup);
  - its sampling within `fits`;
  - its MTP head started;
  - room in its backlogs for the first window;
  - the window's context mode certain.

This needs 0370's `plan` part (prod has `DECODE_OVERLAP=1`). Everything is decided from shared values, so both ranks
enter at the same round.

**A round** (`Resident._launch`; every step a device kernel or an existing capturable path):

| step | device work | replaces |
| --- | --- | --- |
| stage | `_stage_kernel`: every row's token from the device state (pending token, drafts; a padded row repeats the last real token), its route row (0280's tie: padded rows read no experts of their own), its sampler position | `stage_multi` / `stage` host staging |
| forward | `compute_multi` with `npbs` (capturable; device positions), graphs per key under the batcher's policy | the same |
| sample | per-slot top-k + one all-gather (rank 0's drain flag in the rider) + `_choose_kernel` for all rows | `sample_multi`'s readback + numpy |
| accept | `_accept_kernel`, one program a slot: `Stepper.accept`'s keep rule (drafts until the first mismatch or an end token with stop-at-EOS), a finished slot keeps nothing; counters, `Stepper.done` (max_tokens, EOS), the MTP token backlog, the round's record, the next window's room | the host loop in `_verify` + `Stepper.accept` |
| commit | `kda.replay_slots`: every slot's replay in one launch (keeps on the device; skipped where the whole window was kept; keep 0 copies); `_conv_shift_dev`; positions copied from the device counters. With `kda`: nothing (§2.3) | per-slot `forward.commit` |
| backlogs | `_backlog_kernel`: the kept rows' final-normed rows and taps into `m_all` / `f_all` (one tensor each, the slots' `m_rows` / `f_taps` are views) | per-slot `copy_` |
| draft | MTP: every m-slot's backlog in one head pass (`batch.mtp_multi` on device-staged rows, each slot's last real row selected on the device), then one pass a chained step: `_draw` (the exact sampler with probabilities) → `_mtp_next_kernel` (`DepthOptimizer.mtp_next` / the confidence chain, calibration one round stale) → the draft fed to the next step on the device. DFlash2: `add_taps_dev` (a device count), the block graph, `dflash2._chain_kernel` (selector edges, keyed noise, picks and probabilities), `_f_depth_kernel` (`f_depth` / the confidence cut) | `Stepper.propose`, `MtpChains`, `Drafter.propose` |
| record | `_finish_kernel` (the next drafts become the window's) + one pinned copy of the record + an event | the host's bookkeeping inputs |

**The host.**

- **Peek mode (default).**
  - After enqueuing round k's front (stage through backlogs), the host processes round k - 1's record while round
    k's forward runs:
    - `Stepper` records: choice, adapt, lookup, opt calibration;
    - emits;
    - `round_costs`;
    - the next window's drafter choice.
  - It then enqueues the draft stage. Only there does it wait, on events of small pinned copies of the counters (a few hundred bytes):
    after each MTP chain step (stop when every chain stopped) and once after the stage (the next window's exact
    rows).
  - No stream synchronization, no pageable copy, no candidate readback and no numpy runs on the critical path. The
    GPU idles for the event latency plus a launch (estimate 10-30 us) at each wait.
- **Lag mode (`GLM53_TF_GPU_ROUND_PEEK=0`).**
  - The host launches round k + 1 before it reads round k at all (no read while launching; tested).
  - Its drafter bounds are one round stale (the chain the device ran lately + 1; the block's recent depth + 2).
  - MTP steps past a chain's end run for nothing. Windows are padded to the bound; padded rows are tie-routed and
    never kept.

**Exits (drains).** Nothing is ever cut half-way: every check is made before a round is launched. Then the round in
flight is processed and the host state materialized:

- positions and the index-ring tail (`save_index_tail`: resident commits skip it);
- the KDA state (`kda`);
- the MTP head's length / drafted entries;
- DFlash2's context end;
- every `Stepper`'s next drafts, arm, steps, backlogs and `opt.used`.

The normal path continues from exactly that state. The reasons:

- a finished request (it is then finished by `_finish`, snapshot included);
- rank 0's flag (a queued request, a cancel, stopping);
- a lookup match;
- a context mode that is not certain for a window (crossing 2,051, or a pool bucket);
- the cache's end or a KV-pool shortage.

The batch goes resident again at the next steady round.

**Lockstep.**

- Both ranks run the same stages and the same collectives in the same order: the candidates' all-gather, the drafts'
  gathers, the drafter's own exchanges.
- They make the same host decisions from identical data: gathered candidates, device counters and records that are
  equal on both ranks, and the stale calibration.
- Rank 0's drain flag travels in the verify gather. Both ranks read it with the record, so both drain after the same
  round.
- Tested with two ranks in threads over a fake communicator (§4).

**Early exit.**

- EOS (with stop-at-EOS) and `max_tokens` are detected on the device (`_accept_kernel`). A finished slot's later
  rounds keep nothing (keep 0: its state is copied, not advanced) until the host drains.
- Stop strings stay host-side (0160). The HTTP callback cancels the request, rank 0's flag drains, and the request
  ends `cancelled`. As today, a cancelled request stores no reply snapshot, so tokens computed past a stop string never
  persist.
- A RoCE failure raises at the next record's `comm.check`, before any of its tokens is emitted.

### 2.3 `kda`: commits folded into the next chain, every slot in one launch (`kda.cu`)

- **`chain_slots_kernel`**, one launch per KDA layer for every slot (block (head, slot), a table of pointers per slot):
  - The prologue replays the previous window's kept rows (`PREKEEP`, read on the device) from that window's saves.
    This is `replay_kernel`'s loop and `update` routine. It writes the committed state.
  - Then it runs the window's real rows (`1 + NREAL`: padded rows are not run) with `chain_kernel`'s statements.
  - It saves what the next prologue needs into the other scratch set (a second `KDAScratchSet` per slot, 16 rows:
    ~27 MB a slot a rank).
  - It stores each row's conv channels for the conv shift.
- **What it removes.**
  - The separate replay pass: 71 MB read + 71 MB written per slot per rejected window.
  - Two copies per slot per layer.
  - Three launches per slot per layer, which become one launch across slots: 128 blocks for 4 slots instead of four
    32-block launches on 48 SMs.
- **Same bits.** The same `update` routine in the same row order. A replayed state read back from memory is the
  register state. PTX float-op histogram: `chain_slots` = `chain` + `replay_layers` exactly, per opcode; `replay_slots`
  = `replay_layers`; both at 64 registers (sm_121, nvcc 13.4). GPU tests compare them bitwise with the per-slot path
  (§6).
- **At a drain.** `replay_slots` replays the last window's kept rows from its saves (`RealOps.exit`). The State then
  holds the committed state again.

### 2.4 Knobs, state, stats

| knob | values | effect |
| --- | --- | --- |
| `GLM53_TF_GPU_ROUND` | `0` (default) / `1` / `sample,resident,kda` | parts as above; `resident` implies `sample`; `kda` needs `resident`; both ranks must agree (checked at load) |
| `GLM53_TF_GPU_ROUND_PEEK` | `1` (default) / `0` | peek vs lag host mode (both ranks agree) |

- **Memory.**
  - Resident state: kilobytes.
  - Backlogs: one tensor each, with one extra row.
  - `kda`: a 16-row second scratch set per slot, ~27 MB a slot a rank (~110 MB a rank at 4 slots).
  - Graphs: the batcher's pool and cap (`BATCH_MAX_GRAPHS`).
- **Stats.**
  - Per request: `round_kinds.resident` (resident windows) and `pad_rows` (lag mode).
  - Batcher counts: `resident_runs`, and `resident_no_<why>` for rounds that could not go resident.
  - `Resident.drains` by reason, `.rounds`, `.entries`, `.peeks`.

## 3. Expected savings (W11 round, ms a round; to be measured)

| item | mechanism | 1 stream | 4 streams |
| --- | --- | ---: | ---: |
| idle outside the verify forward | peeks cost ~10-30 us each; no numpy, no per-slot Python between device steps | -0.4 .. -0.6 (of 0.7) | -1.8 .. -2.3 (of 2.6) |
| idle inside the verify forward | peek: unchanged (the host launches the forward right after the draft decisions, as today). Lag: host a round ahead | 0 | 0 (peek); -1 .. -2 (lag, minus its waste) |
| KDA replay folded (`kda`) | replay pass gone on rejected windows (P ≈ 0.75 on prose); prologue compute added | -0.3 .. -0.5 | -1.5 .. -2.2 |
| KDA fused across slots (`kda`) | 1 launch a layer for all slots (3 waves instead of 4 on 48 SMs), 2 copies a slot a layer gone | -0.1 | -0.8 .. -1.3 |
| replay in one launch (`resident` without `kda`) | the four per-slot replays together | 0 | -0.3 .. -0.5 |
| sampler kernel | float64 draw on the device (+10-40 us), top-k / gather unchanged | +0.02 | +0.05 |
| MTP head passes | graphed per key after sightings (as today's graphed steps); peek: exactly today's steps | 0 | 0 |
| **total** (`1`, peek) | | **-0.6 .. -1.2 (+1-2%)** | **-3.5 .. -7 (+3-6%)** |

Cross-check against DECODE-PLAN (post-W11):

- E6 (device sampling + draft chains) is -0.6 mid at 1 stream and -3.0 at 4 streams; F3's KDA share is ~-2 to -3
  of its -6 mid.
- The patch covers E6 and the KDA part of F3.
- F3's attention part and F4's batched DFlash2 blocks remain (§5).

## 4. Offline tests (all pass)

| file | what | result |
| --- | --- | --- |
| `tests/test_gpu_sampler_interpreter.py` | libm port vs numpy (60k uniforms, their Gumbel, log near 1 and on random doubles, exp on every range incl. edges); the draw vs `choose_rows` / greedy lexsort / `_probability` on 300 random cases (1-2 ranks, top_k 0-120, top_p 0-1, T 1e-7-1.5, ties, -0.0); numpy's pairwise sum; `sample_multi` / `sample_drafts` / `sample_rows` with the knob == host (riders, 1-2 ranks); `self_check` | 11 passed (3.8); 1,600 more random cases in scratch runs, 0 differences |
| `tests/libm_a64.py` (optional: `GLM53_TF_LIBM_A64`) | the Triton port vs Ubuntu's aarch64 `libm.so.6` executed in unicorn | 5.4M inputs (3 seeds), 0 differences; the scalar spec 10.5M, 0 |
| `tests/cuda/test_gpu_round_patches.py` | knob parsing; `_accept_kernel` vs `Stepper.accept` + `done` + room (40 random batches); `_stage_kernel`; `_mtp_next_kernel` vs `DepthOptimizer.mtp_next` and the confidence chain (120 random chains, calibrated histories); `_f_depth_kernel` vs `f_depth` and `Drafter.chain`'s cut; `_conv_shift_dev` vs `_conv_shift`; `_backlog_kernel`. GPU (skipped here): `replay_slots` == `replay_layers`, `chain_slots` == replay + `chain` bitwise, the load check on GB10, the synthetic checkpoint's replies with each part == serial, a lone request + session resume | 9 passed, 16 GPU skipped |
| `tests/cuda/test_gpu_round_resident.py` | the real `Batcher` / `Stepper` / `Resident` on a hostile fake model, every device kernel in the interpreter: replies == serial, committed states == serial, MTP cache and DFlash2 context position by position (peek and lag x MTP batched or not x `kda`); context-mode drains; **two ranks in threads** (identical logs, rounds and drains); no device read while launching (lag) / only peek records (peek) | 12 passed |
| `tests/test_gpu_round_compile.py` | sm_121: the sampler's float64 is exactly as written (no contraction, fma counts = the port's); glue and the DFlash2 chain compile; `kda.cu` via nvcc: `chain_slots` = `chain` + `replay_layers` per opcode, `replay_slots` = `replay_layers`, <= 64 registers | 14 passed on Triton 3.7.1 and 3.8 |

The harness caught three real bugs before any GPU time:

- the first resident round could overflow an MTP backlog the normal path had nearly filled (entry now checks room);
- serial requests' backlogs were tracked, then flushed as garbage (tracking now follows `Stepper.accept`: drafting
  requests only);
- a device read (`float(dc[2])`) in the DFlash2 chain setup.

Regression: every existing CPU suite gives the same results on the full patch stack with and without 0450
(`tests/cuda/*_patches.py` + `tests/test_*.py`, knob unset). Patch order: 0450 applies after 0440, and 0460 / 0470 /
0490 apply after it.

## 5. Not done here (next steps, in order of value at 4 streams)

1. **Batched DFlash2 blocks** (F4, ~2.4 ms a slot at 4 streams). One drafter pass for every f-slot: row-local
   layers over all slots' 8-row blocks (`_dconv` needs a block-boundary mask: row 0 of a block must not mix with the
   previous block's row 7), attention per slot on its own context (as `batch._dsa` does), one head read for 7 x n
   rows, one candidates exchange.
   - Resident rounds already run the chain on the device, so batching only changes `RealOps.dflash`.
   - Estimate -2 to -5 ms at 4 streams.
2. **Per-slot attention in one launch per layer** (F3's other half, 0.9 ms a slot): the latent cache write,
   `index_update`, selection and sparse attention with a per-row sequence table.
3. **Conditional graph nodes for MTP chains.** CUDA 12.4+ IF / WHILE nodes, driven by `_mtp_next_kernel`'s live flag,
   would give lag mode's zero reads without its wasted steps. This needs cuda-python graph construction around
   torch's captured steps (`CUDAGraph.raw_cuda_graph`) and a GPU to validate.
4. **A device lookup index** (prompt-lookup drafts without a drain), and PDL on the glue kernels.

## 6. GPU test plan (one window, ~75 min; image = the stack with 0450, knob off == today)

**0. Unit, worker (prod may stay up if memory allows), ~10 min.**

- `tests/cuda/test_gpu_round_patches.py` GPU tests:
  - `replay_slots` / `chain_slots` bitwise (rows 1-16, keeps 0..R, prologues 0-16);
  - the load check at 65,536 rows;
  - the synthetic checkpoint: 4 requests x (sample / resident / 1) x (greedy / sampled) == serial;
  - a lone request + resume == fresh.
- `test_batch_parallel_patches.py`, `test_batch_sessions_patches.py` and `test_decode_overlap_patches.py` with
  `GLM53_TF_GPU_ROUND=1` in the environment.
- The sampler at scale: `python -c "from tensorfold.families.glm5_next.cuda import gpusample; [print(s,
  gpusample.self_check('cuda', rows=1 << 20, seed=s)) for s in range(100)]"`. That is 10^8 draws against the node's
  numpy. Every line must print `None`.

**1. Loads** (`config/prod.env` + ...; W11's harness):

| load | adds |
| --- | --- |
| A | nothing (control) |
| B | `GLM53_TF_GPU_ROUND=sample` |
| C | `GLM53_TF_GPU_ROUND=resident` |
| D | `GLM53_TF_GPU_ROUND=1` |
| E | D + `GLM53_TF_GPU_ROUND_PEEK=0` |
| F (if D wins at 4 streams) | D + `GLM53_TF_BATCH_GRAPHS=0` (W11's F0) |

**2. Per load.**

1. Exactness:
   - `glmbench.py --suites exact` 10/10;
   - `multiturn.py --modes batchexact` 4/4;
   - the W9/W10 transcripts byte-identical to A (6 prompts, greedy and sampled seed 1234, alone and 4 together);
   - `--suites tf,kit,edit` reply hashes identical to A.
2. Speed: `multiturn.py --modes concurrent --streams 1,4 --reps 5 --long-tokens 512` (median of 5) and a lone request
   in slot 2 (W11's note).
3. From the stats:
   - `round_kinds.resident` share;
   - `pad_rows` (0 with peek);
   - `Resident.drains` by reason (expect `done` and `rank 0` to dominate; `lookup` on edit / agent content);
   - `resident_no_*` counts.
4. Session resume == cold on a 2-turn conversation after resident rounds.
5. Cancel: a streamed client disconnecting after ~50 tokens beside 3 decoders ends `cancelled`, and the others' replies
   == batchexact.
6. Arrival: a 5th request while 4 decode. Its `queued_s` may grow by at most two rounds (one more than 0370).

**3. nsys, one capture per server lifetime** (loads A and D; W11's `w11dec.py`):

| where | expect |
| --- | --- |
| idle outside the verify forward, 1 stream | 0.7 -> <= 0.2 ms |
| idle outside the verify forward, 4 streams | 2.6 -> <= 0.6 ms |
| the KDA family at 4 streams | 6.0 -> ~3.5 ms; no `replay_layers_kernel` in D |
| host time in drafting ranges | well under the drafting kernels' time |
| peeks a round (NVTX) | 1 + the MTP chain's continuations |

**4. Adopt / reject.**

- Adopt the best of C / D / E only if every exactness gate passes, 4 streams gain ≥ +3% (median of 5) and 1 stream is
  not lower.
- If only `sample` is clean and neutral, adopt nothing (it is a building block).
- Revert: unset the knobs.

## 7. Files

`patches/0450-glm-gpu-round.patch`:

- **New.**
  - `gpusample.py`: the sampler, the libm port and its tables, the load check.
  - `gpuround.py`: knobs, the device state layout, the glue kernels.
  - `resident.py`: the driver and `RealOps`.
- **Changed.**
  - `batch.py`: `sample_multi` / `sample_drafts` device paths, the entry decision in `_verify`, `_go_resident` in
    `_serve` / `follow`, `_mode_at`, backlogs as views of one tensor, `_kda_slots`.
  - `decode.py`: `sample_rows`' device path.
  - `dflash2.py`: `_taps_compute` with a device count, `add_taps_dev`, `resident_end`, `_chain_kernel` / `chain_dev`.
  - `kda.cu` / `kda.cpp` / `kda.py`: `replay_slots`, `chain_slots`, `SlotTables`.
  - `engine.py`: the knob check and the load check.
- **Tests.** `tests/test_gpu_sampler_interpreter.py`, `tests/test_gpu_round_compile.py`, `tests/libm_a64.py`,
  `tests/gpuround_interp.py`, `tests/cuda/test_gpu_round_patches.py`, `tests/cuda/test_gpu_round_resident.py`.
