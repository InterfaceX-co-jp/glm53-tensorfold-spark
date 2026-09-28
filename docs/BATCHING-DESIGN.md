# Continuous batching for the GLM-5.3-Flash CUDA engine

Status: phase-1 prototype in `patches/0030-glm-batch2.patch` (`GLM53_TF_BATCH=2`, 23 synthetic GPU tests green),
rebuilt on the current engine by `patches/0120-glm-batch-v2.patch` (written offline, not yet run on a GPU): see
[v2 on the current engine](#v2-on-the-current-engine-patches0120). Sections 1-7 are the original design.
`patches/0200-glm-batch-parallel.patch` (opt-in knobs, offline, not yet run on a GPU) adds graph-key capture on the
N-th sighting, parity-keyed graphs, MTP drafts of all slots in one head pass, a row-cost floor for batch-aware depths,
short prompts prefilled in their admission round and optional window padding; its analysis of mixed prefill/decode
rounds (not worth it on these kernels) and of one-launch per-slot kernels is in `docs/PATCHES.md` (0200).

## v2 on the current engine (patches/0120)

0030 was written against 0001-0020; later features were off or bypassed in batch mode (DFlash2 ran as MTP, lookup,
cost depths and `tf_knobs` refused or ignored, fast / lean prefill dropped, windows over 4 rows and contexts past
2,051 tokens eager). 0120 keeps 0030's data layout (a `State` per slot, shared `Buffers`, one forward over every
slot's window) and brings every per-request feature into it. Code: `batch.py` (rounds), `batchplan.py` (host-only
scheduling, unit-tested anywhere), small hooks in `engine.py` (build the batcher after the knob defaults and before
0110's session store; no knob refusal), `knobs.py` (only `calib_online` refused in batch mode) and `app.py`
(background priority).

**Per sequence.** Latent KV (0060) and the index rings (0065) come with each slot's `State` (rings sized for the
engine's prefill rows). KDA states, conv windows, MTP cache and MTP CUDA graphs (0050's long-context ones too) per
slot. **DFlash2 per slot**: `_drafter_view` shares the drafter's weights and per-pass buffers and gives each slot its
own context caches (a 4,096-slot ring with 0065: ~42 MB a rank), positions and CUDA graphs. **The decode loop per
slot** is `Stepper`: `decode.auto_decode`'s round cut at the forward (propose drafts / accept rows), with MTP,
DFlash2, `auto`'s per-round choice, lookup (0020, each request its own index), threshold depths and cost-derived
depths (0071), the same backlogs and records as alone, so the same drafts. Slots other than 0 keep window-sized KDA
projection / replay rows (8) and borrow slot 0's for a prefill (those rows only live between a forward and its
commit): ~2.6 GB a slot less at 1,024-row buffers.

**Per-request knobs (0090-0093).** Each request's `tf_knobs` are resolved on rank 0 and travel in its admission
header (`batchplan.encode_header`: sampling, stop, max tokens, cost flag, the knob block, the policy code). Its
prefill runs inside `GlmEngine._knobs(values)` (prefill rows, fast / FP8 / overlap, expert loop, long-context graphs,
profile); its `Stepper` uses its `auto_fdrafts`, lookup settings and depth mode. Batched rounds run at the load-time
defaults (the round-level knobs pick between paths with the same bits). `calib_online` stays refused: its table times
one request's windows.

**Prefill.** An admitted prompt prefills in pieces of `GLM53_TF_BATCH_PIECE` tokens (default 2,048; a fast prefill's
pieces are multiples of its chunk grid C, at least C), each through `decode.prefill` itself (exact, fast, lean, FP8,
pipelined: whatever the request's knobs say) resumed from the snapshot the previous piece left (a fast piece ends on
the grid, where `_prefill` takes its grid snapshot). Resumed == fresh holds in every prefill mode, so a prompt
prefilled in pieces has the bits of one prefill, and snapshot semantics are the lone engine's (fast: grid snapshots
only). One piece a round at most, the prompt with the fewest tokens left first; while other requests decode, pieces
take at most `GLM53_TF_BATCH_PREFILL_SHARE` (default 0.5) of the time (`batchplan.Fairness`: a piece of t seconds is
followed by at least t (1 - share) / share seconds and one round of decode). So a 60 s prompt no longer stalls the
others: they lose at most one piece (~2-3 s at 2,048 tokens) at a time. With nothing decoding, pieces run back to
back (new arrivals are admitted between them).

*Deferred: vLLM-style mixing* (a prefill chunk's rows in the same forward as decode windows). Exact chunks could
mix (every kernel is row-invariant: add the chunk as one more window, commit all its rows, head on its last row only),
fast chunks cannot (their kernels are not the decode kernels). Most prompts run fast, so pieces cover both first;
mixing exact chunks is ~2-3 days (the chunk's MTP absorb and DFlash2 taps from its rows, graph keys by chunk size).

**Rounds.** One forward over all decoding windows: row-local work once over T rows, KDA chain / attention / indexer
per slot (patches/0050's device-side selection, `npb`, per slot past 2,051 tokens; latent path through `latent`'s
pieces). Keyed sampling for every request's rows with ONE all-gather (`sample_multi`: the same per-request top-k as
`sample_rows`). CUDA graphs: slot 0 alone replays the engine's graphs; other rounds replay graphs captured lazily per
(slots, rows per slot, per-slot mode: dense or pool bucket), windows up to 8 rows (`GLM53_TF_BATCH_GRAPH_ROWS`), at
most `GLM53_TF_BATCH_MAX_GRAPHS` (256); a key's first round runs eagerly through the capturable code and is captured
after (as 0050 does). KDA parities are normalized to buffer 0 before a graphed round (a 71 MB copy a slot when a
commit left the state in buffer 1, ~0.3-0.6 ms), which removes 2^N from the key count. Windows crossing 2,051 run
eager.

**Batch-aware depths.** Cost-derived depths (`o`, `om`, `of`, `auto` with `depth: cost`) price a slot's draft rows at
the verify curve's slope past the other slots' rows (`RoundCosts.table`) against the rounds' aggregate rate while
more than one request shares them (`RoundCosts.rate`, from the load-time costs and committed tokens: the same on both
ranks): a row that slows every request must buy more than one that slows only its own. Threshold policies keep their
thresholds (exact either way).

**Server.** Requests queue on the batcher (the app lock is lifted); tokens stream per round to the caller's thread.
A client whose stream write fails is cancelled at the next round. `"priority": "background"` (or a session-title
request, TensorFold's heuristic) is admitted after every foreground request and, when a foreground one waits and no
slot is free, steps aside (the latest admitted background request); it runs again from the start later and its caller
only receives the tokens it did not have (same prompt and keyed sampling: same tokens).

**Memory and admission.** Every slot has the engine's full context. At load the batcher adds slots while at least
`GLM53_TF_BATCH_RESERVE_GB` (4) stays free, the same count on both ranks, and serves that many (printing why). At
admission a request waits while fewer than `GLM53_TF_BATCH_ADMIT_GB` (1) are free and others run. Per extra
sequence and rank (real model, latent KV, rings on, arithmetic):

| Item | Size |
| --- | --- |
| latent KV + indexer (12 layers' latents, MTP index keys, pool keys; rings) | 13.3 KB a token: 0.43 GB at 32k, 1.7 GB at 128k |
| KDA states (two buffers) + conv windows + ring tail | 146 MB |
| window rows (8-row projections and replay scratch), index rings at 1,024-row buffers | ~32 MB |
| DFlash2 context ring (4,096 slots) + its graphs | ~60 MB |
| MTP graphs, drafter backlogs | ~30 MB |
| kept snapshots (up to 2) | ~150 MB |
| **total** | **~0.85 GB at 32k, ~2.1 GB at 128k** (expanded KV: ~12.9 GB at 32k) |

So 4 sequences at 128k cost ~6.4 GB a rank beyond one; the lazily captured round graphs share one pool.

**Expected throughput** (the section 6 model with one sampling all-gather a round, ~0.3-0.6 ms a slot of parity
copies, graphs hit): 2 x 3 rows ~73 ms a round, 4.4 tokens: **~60 tok/s** (1.3-1.35x one stream's 45); 4 x 2-3 rows
85-105 ms, 7.2-8.8 tokens: **~80-90 tok/s** (1.25-1.4x vLLM's 63-66). DFlash2 / lookup-heavy requests (single-stream
70-101 tok/s) gain less: their windows are already wide. Eager rounds (a new graph key, a window crossing 2,051)
cost ~10-25 ms more.

**Deferred** (effort): mixing exact prefill chunks into rounds (2-3 d); device-side per-slot offsets / pointers in the
KDA chain and attention so graphs are keyed by T only (3-4 d); MTP drafts batched across slots (2 d); detecting a
non-streamed client that went away (the HTTP layer only sees failed stream writes). 0110's session store behind the
slots is `patches/0180` (opt-in `GLM53_TF_BATCH_SESSIONS=1`: restores copied into a free slot, saves from every slot;
`docs/PATCHES.md`).

## Goal

TensorFold serves one request at a time (`cuda/server.py` holds one lock around `generate`). Coding agents
(opencode with subagents) keep 2-4 requests in flight. The production vLLM kit serves 4 sequences at about
63-66 tok/s in total. Decode is bound by memory bandwidth: a one-row step reads about 5.0 GB a rank (about 30 ms),
and each extra verify row costs 6-10 ms, almost all of it for the routed experts the row adds (8 of 288 per layer).
A row from a second sequence should cost about what an extra draft row of the same sequence costs. So running 2-4
sequences' verify windows as ONE forward should raise total throughput while each request stays exact: every
kernel already computes each row without reference to the other rows in the window.

## 1. What the engine assumes today

`forward.compute(w, st, b, R)` runs R consecutive rows of ONE sequence (`State st`) through static buffers
(`Buffers b`). Grouped by whether a kernel reads any per-sequence state:

**Row-local, so they batch as they are** (a row's bits do not depend on the other rows, and none of them reads a
`State`):

| Work | Where | Note |
| --- | --- | --- |
| embedding, hyper-connections (`hc_pre`/`hc_post`, Sinkhorn), RMSNorms, `stream_mean`, DFlash2 taps | `glue.py` | one program per row |
| every dense matmul (KDA `proj`/`f_b`/`g_b`/`o`, DSA `proj`/`q_b`/`kv_b`/`o`, indexer `wk`/`wq_b`, dense MLPs, shared expert, head) | `qmm.matmul` / `_matmul_b16` | split-K fixed by the weight's shape; BM tile chosen by row count, and a row's result does not depend on the tile size (the same property that makes drafted replies equal serial ones) |
| MoE router, top-k `select`, grouped routed experts (Q4 `moe_gateup`/`moe_down`, EXL3 `exl3_mm.routed`), `combine` | `glue.py`, `qmm.py`, `exl3_mm.py` | experts grouped by id over all rows (`Buffers.group(R)`); the members of each expert are independent |
| all-gathers of fp32 partials | `forward.gather` | `[world, R, D]`, one per block |
| sampling | `decode.sample_rows` | keyed by (seed, absolute position) per request; rank-local top-k then all-gather |

**Tied to one sequence:**

| Work | Where | What ties it to one sequence |
| --- | --- | --- |
| KDA chain | `kda.cu` `chain_kernel`, grid = heads | one block per head walks rows 0..R-1 **sequentially over ONE recurrent state** (`state_in` -> `state_out`) and ONE conv window (`cs`); it writes the replay inputs (k, v, g, beta) to one `KDAScratch` |
| KDA commit | `kda.replay_layers`, `forward._conv_shift` | replays the kept prefix from `State.scratch_set` into the State's other buffer; the conv shift reads `State.proj` (the window's q\|k\|v rows) |
| attention cache write | `attention.kv_write` | writes rows at `POS + r`: ONE `pos` scalar on the device, ONE cache |
| dense attention | `attention._chunks`/`_merge` | ONE KV cache and ONE `pos` scalar a launch; row r attends to keys `0..pos + r` |
| DSA indexer (contexts past 2,051 tokens) | `sparse.index_update`, `select_tokens`, `sparse_attention` | index-key/pool caches per sequence, `pos_dev` scalar, and a host `pos` (eager only) |
| MTP head | `mtp.mtp_compute` | the head's own cache (`State.mtp_kc/vc`, `mtp_pos_dev`) |
| CUDA graphs | `graphs.py` | keyed by (rows, KDA parity) with the pointers of `e.st` baked in; the MTP graphs likewise |
| commit / rollback, snapshots | `forward.commit`, `decode.Snapshot` | per `State`; a snapshot copies the KDA state and conv window and **leaves the attention caches in place**, so it is only valid on the State it came from |
| the request loop | `decode.*_decode`, `GlmEngine._run` | one `Engine.st`, one monolithic loop per request; the server lock serializes requests |
| TP lockstep | `GlmEngine.generate/follow` | rank 0 shares one request's header and prompt; rank 1 mirrors the whole request |
| DFlash2 drafter | `dflash2.Drafter` | one context (KV caches, `context_end`) next to the weights |

`State` already holds everything per sequence (KDA states x2 and conv windows, window projections and replay
scratch, the DSA and MTP KV caches, indexer caches, positions), and `Buffers` holds only per-window scratch. So a
second sequence is a second `State` with the same shared `Buffers`.

## 2. Minimal data-structure change

**Slots.** `N` `State` objects ("slots"); slot 0 is the engine's own `e.st`. `Buffers` stays shared. A round's
rows are the active slots' windows back to back in slot order: slot i owns rows `[off_i, off_i + R_i)`.
`sum R_i <= Buffers.rows` (64 by default, so 8 windows of 8 fit).

**Batched forward** (`batch.compute_multi`). The same layer sequence as `forward.compute`, over `T = sum R_i`
rows:

- Row-local work runs once over all T rows (so the weights are read once a round).
- KDA layer: the `proj`, `f_b`, `g_b` matmuls over T rows into a shared `Buffers.bproj`. Then, per slot, its rows
  are copied into `State.proj[layer, :R_i]` (the conv shift at commit reads them there), `kda.chain` runs over
  the slot's state, conv window and scratch, and its output is copied into `Buffers.bkout[off_i:]`. One `o_proj`
  over T rows.
- DSA layer: projections over T rows (and the indexer's `wk`/gate when long contexts are on). Per slot:
  `kv_write` at the slot's `pos_dev`, `index_update`, and `attention(..., out=b.attn.out[off_i:off_i+R_i])` (the
  one kernel change: `attention` takes an `out`), or sparse selection/attention past 2,051 tokens. One `o_proj`
  over T rows.
- MLP/MoE, head: unchanged, over T rows.

**Commit, sampling, drafting per slot.** `forward.commit(w, st_i, b, R_i, keep_i)` is already per State.
`sample_rows(logits[off_i:off_i+R_i], positions_i, sampling_i)`. MTP drafting runs per slot with the engine
pointed at the slot's State (`Batcher._on(slot)` swaps `e.st` and `e.graphs`), reading its kept rows
`b.fnormed[off_i:off_i+keep_i]` before the next forward overwrites them.

**Graphs.** Slot 0 alone replays the engine's own graphs. Other compositions are captured lazily with key
(active slots, rows per slot, KDA parity per slot), for windows of up to 4 rows while the context is dense; the rest
runs eager. For N=2 that is at most 8 (slot 1 alone) + 64 (both) keys, captured once each (one warm-up forward
plus capture). The MTP head's graphs are captured per slot at start.

**Phase 2 kernel changes (to scale past 2 and cut graph count).** Pass device-side arrays instead of per-slot
launches:

- KDA: `chain_kernel` grid (heads, sequences); per sequence a row offset, a row count and pointers to its state
  in/out, conv window and scratch (or all slots' states in one `[slots, 2, layers, H, 128, 128]` tensor and a
  parity array). One launch a layer whatever N.
- Attention: a per-row sequence id (or per 16-row tile, with each sequence's rows padded to tiles so no tile
  mixes caches) giving the cache base and `pos`; `kv_write` likewise. One launch a layer.
- Then row counts per slot and parities live on the device, and a graph only depends on T: graphs keyed by
  T = 1..32 instead of per composition.

## 3. Scheduler

A loop thread on rank 0 (`Batcher._serve`) runs rounds; HTTP threads only queue a `Job` and relay its tokens
(the server lock is lifted for the GLM app in batch mode).

Each round:

1. **Plan** (rank 0): cancels (clients that went away), then admissions: queued jobs into free slots, FIFO. A
   job goes to the free slot whose kept snapshots resume the longest prefix of its prompt (agents re-send a
   growing conversation, so affinity keeps prefix reuse working per slot), else to the least recently admitted
   slot. The plan is shared with rank 1 (below).
2. **Admit**: prefill each admitted prompt on its slot (phase 1: whole prompt, the other slots wait), emit the
   first token, draft its first window.
3. **Verify**: one forward over every active slot's window (pending token + drafts); then per slot, in slot order:
   sample its rows with its own keyed sampling, keep up to the first mismatch, commit its prefix, emit, and either
   finish (max tokens, EOS: snapshot of the reply kept on the slot) or draft its next window.

Round-robin is implicit: every active sequence contributes one window every round, so no sequence starves; a
long prefill is the one head-of-line block (phase 2 fixes it, below).

**Exactness.** A request's tokens are the keyed samples of its own logits at its own positions. Its logit rows
are the bits of its lone forward (row-local kernels; per-slot kernels see only the slot's state and rows), its
state advances by `commit` exactly as alone, and the accept rule only decides how many of its rows to keep. Drafts
only propose. So each batched reply is byte-identical to the same request served alone, and to serial decoding,
whatever else shares its rounds and whatever depth policy it gets.

**Draft depth under batching.** Alone, a draft row costs 6-10 ms against a ~30 ms base, so 2-3 drafts pay. In a
batch the fixed ~20 ms (dense matmuls, all-gathers, head) is shared, so each row's marginal cost is a larger share
of the round; the best depth per sequence drops as N grows. Phase 1 keeps each request's policy (exact either
way); phase 2 should pick depths per round to maximize expected committed tokens / round cost from the calibrated
verify curve (extended to 32 rows).

**Drafters.** Phase 1 drafts with the MTP head only (its cache is in `State`). DFlash2 holds a single context;
batched requests run DFlash2 specs as the MTP policy with the same depth rule, and `auto` as its MTP arm
(`c3:0.35` greedy, `a:0.6:0.85` sampled). Phase 2: split `Drafter` into weights + a per-slot context.

## 4. Tensor-parallel lockstep

Rank 0 alone decides what depends on the outside world: which clients cancelled, which queued jobs are admitted,
into which slot, resuming from how many cached tokens. It shares that as one int list before every round
(`GlmEngine._share`: `[n_cancel, slots..., n_admit, (slot, cached, 15-int header)...]`, then one `_share` per
admitted prompt). Everything else in a round (prefill, drafts, window sizes, graph keys and lazy captures,
sampling, keeps, end of a request) is a function of the shared inputs, computed alike on both ranks. Rank 1's
`follow` runs the same `_execute(cancels, admits)`. The ranks refuse to start with different `GLM53_TF_BATCH`,
`GLM53_TF_BATCH_GRAPHS` or `GLM53_TF_BATCH_GRAPH_ROWS`. The per-round plan costs two small all-gathers
(tens of microseconds on the link) against a 40-100 ms round.

## 5. Memory per extra sequence (per rank)

Model: 45 layers (34 KDA with 64 heads of 128, 11 DSA with 64 heads of 256), plus the MTP layer (DSA). Per rank
HL = 32 attention heads and 32 KDA heads.

| Item | Size per extra slot per rank |
| --- | --- |
| DSA + MTP KV caches: 12 layers x 32 heads x (256 + 256) x 2 B | **0.39 MB a token of capacity**: 1.0 GB at the default 2,560 slots, 3.2 GB at 8k, 12.6 GB at 32k |
| indexer caches (long contexts only) | ~7 KB a token (0.2 GB at 32k) |
| KDA states, two buffers: 2 x 34 x 32 x 128 x 128 x 4 B | 143 MB |
| KDA conv windows | 2.5 MB |
| window projections `State.proj` (34 x rows x 12,576 x 2 B) | 55 MB at 64 rows (440 MB with `GLM53_TF_PREFILL_ROWS=512`) |
| KDA replay scratch (34 x rows x 49 KB) | 107 MB at 64 rows (860 MB at 512) |
| kept snapshots (up to 2 a slot: KDA state + conv) | ~150 MB |
| **total** | **~1.45 GB at 2,560 context**, ~3.6 GB at 8k, ~13 GB at 32k (64-row prefill) |

A rank holds roughly 85-91 GB of weights of its 128 GB, so N=4 fits up to roughly 8-16k context per slot; long
contexts for all slots need paged KV (one pool shared by the slots; phase 2+). Every slot has the capacity the
server was started with (`--context`). The prototype prints what its extra slots took at start.

## 6. Expected throughput

Measured verify cost per window on two Sparks (whole step, graphs and all-gathers; `docs/recipes/glm-5.3-flash.md`):
29.2, 38.9, 45.2, 55.2, 59.5, 63.2, 65.6, 69.2 ms at 1-8 rows; past 8 rows extrapolated at +4.5 ms a row (the
curve's slope at 6-8 rows is 2.4-3.7; rows of different sequences share fewer experts than draft rows of one,
so this is a conservative figure). Per sequence and round: sampling ~0.8 ms, per-slot KDA/attention state traffic
~0.6 ms, MTP drafting 1.7 ms + 1.5 ms per chained draft. Committed tokens per round: ~2.2 at 3 rows (MTP c3,
matching today's ~45 tok/s single stream), ~1.8 at 2 rows, 1 serial.

| Sequences x rows | Round | Tokens/round | Total tok/s | Per request |
| --- | ---: | ---: | ---: | ---: |
| 1 x 3 (today) | 49 ms | 2.2 | **45** | 45 |
| 2 x 1 | 42 ms | 2 | 48 | 24 |
| 2 x 3 | 72 ms | 4.4 | **61** | 30 |
| 3 x 3 | 88 ms | 6.6 | 75 | 25 |
| 4 x 2 | 82 ms | 7.2 | **88** | 22 |
| 4 x 3 | 106 ms | 8.8 | 83 | 21 |
| 4 x 3, phase 2 (batched MTP drafts, one sampling gather) | 93 ms | 8.8 | ~95 | 24 |

So roughly 1.35x total at N=2 and ~2x at N=4 over today's single stream, about 1.3-1.45x vLLM's 63-66 tok/s at 4
sequences. That is less than 2-3x because extra rows are not free: each costs 6-10 ms (20-30% of a one-row
step), mostly for the experts it brings in. The gain is the amortized fixed part (dense matmuls ~9.7 ms, 90
all-gathers ~2.4 ms, head, launch/host time).

## 7. Phases and effort

**Phase 1: 2 sequences, any lengths (prototype written: `patches/0030`).** Slots, `compute_multi` with
per-slot KDA/attention launches, the round loop with admission/cancel/placement, per-slot snapshots, lazy
graphs, MTP drafting, TP plan sharing, server lock lifted. Remaining: GPU bring-up and the tests below (1-2
days), a two-rank soak with concurrent clients and a concurrent benchmark (`bench/glmbench.py` with N
parallel streams) (1 day), capturing the N=2 graphs at start instead of lazily, and a batch-aware depth rule
(1 day). **~3-4 days.**

**Phase 2: N sequences, efficient.** (days)

| Item | Days |
| --- | ---: |
| KDA chain grid (heads, sequences) and attention/`kv_write` with per-row sequence id -> cache base, pos; device-side offsets; graphs keyed by T only | 3-4 |
| MTP drafts batched across sequences (one head forward per chained step for all slots) | 2 |
| one all-gather + host sort for all slots' sampling | 0.5 |
| chunked prefill inside rounds (a prefill chunk is one more slot window, rows per round capped) so a long prompt no longer stalls the others | 2-3 |
| DFlash2 per-slot contexts (`Drafter` = weights + per-slot caches) and `auto` choice per slot | 2-3 |
| batch-aware depth policy from the calibrated cost curve | 1 |
| optional: paged KV shared by slots (long contexts x N) | 3-5 |

**~10-15 days** for phase 2 without paged KV.

## Prototype status (`patches/0030`)

`GLM53_TF_BATCH=N` (default 1, which is upstream behaviour plus one start-up all-gather comparing N across ranks).

| File | Change |
| --- | --- |
| `families/glm5_next/cuda/batch.py` (new) | `compute_multi`, `stage_multi`, `Batcher` (slots, loop, plan sharing, admission, verify rounds, snapshots, lazy graphs), `generate`, `generate_batch` (queue several requests at once; tests), `follow` |
| `attention.py` | `attention(..., out=None)`: write a sequence's rows into its part of the window |
| `engine.py` | `self.batch = Batcher(self, n)` when `GLM53_TF_BATCH > 1`; `generate`/`follow` delegate to it |
| `app.py` | the GLM app lifts `App.run`'s request lock when the engine batches |

Knobs: `GLM53_TF_BATCH_GRAPHS=0` (multi-sequence rounds eager), `GLM53_TF_BATCH_GRAPH_ROWS` (largest per-slot
window captured, default 4). `N x 8 <= GLM53_TF_PREFILL_ROWS` is required (raise it for N > 8).

Tests: `tests/cuda/test_batch_patches.py` (one GPU playing rank 0 of two, TensorFold's synthetic checkpoint):
batched forward rows == each sequence's own forward rows (logits and MTP inputs; graphs, eager, 8+6-row eager
round, slot 1 alone); two queued requests == their lone and serial replies for 6 policy pairs, sampled and greedy,
with graphs and eager; uneven lengths with a third request queued behind a full batch; per-slot prefix reuse ==
fresh serial; two concurrent `generate` callers; a client going away mid-reply; the EXL3 checkpoint. Run inside
the image: `PYTHONPATH=/src/TensorFold/tests/cuda pytest -q tests/cuda/test_batch_patches.py`.

**Untested:** nothing in the patch has run. It was checked only by `py_compile`, pyflakes, and applying on top of
0001-0006, 0010 and 0040. In particular unverified: the bit-equality of rows across batched windows up to 16 rows
(it rests on the same row-independence the drafted==serial tests rely on, which covers windows up to 8 and
prefill chunks of 64/512), lazy CUDA-graph capture mid-serving (including NCCL inside the capture on two ranks),
rank-1 `follow` in batch mode (the tests only play rank 0), and any speed.

**Known limits of phase 1:**

- An admitted prompt prefills whole while the other slot waits.
- DFlash2 is not used in batch mode (MTP drafts only). Without an MTP head, batched requests decode one row a round.
- MTP drafting and sampling run once per slot on the host path.
- Windows over 4 rows, and contexts past 2,051 tokens, run eager.
- `compute_multi` repeats `forward.layer_forward`'s layer loop, so `GLM53_TF_COMM=prefetch` (0040) sets no
  prefetch sites in multi-sequence rounds. That is correct (no prefetch is launched) but slower. Folding both loops
  into one that takes a per-sequence "attention mixer" would remove the duplication.
- A queued request whose client has gone is still admitted, then cancelled at the next round.
