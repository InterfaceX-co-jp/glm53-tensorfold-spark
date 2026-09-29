# Multi-slot prefill (patch 0560, `GLM53_TF_MULTI_PREFILL`)

Several requests waiting to prefill (RigMark C4's four ~130-token prompts, an agent's burst of tool results) go
through **one** forward: their rows stacked, every per-sequence part per slot, the routed experts read once for all
of them. Each request's bits are those of its piece prefilled alone. Offline work (no GPU run yet): knob **off** by
default, CPU tests pass, the GPU plan is §9. `docs/REPLAY-TTFT.md` §2 has the analysis this implements.

> **Update 2026-09-30 (W17, docs/RESULTS.md):** measured on the GPU and **adopted** (`GLM53_TF_MULTI_PREFILL=1` in
> `config/prod.env.example`, image b9): 92/92 grouped replies equal to the same request alone, C4 per-stream first
> tokens 1.64 -> 0.82 s with thinking on (RigMark 3-run mean 0.89 s), prefill and decode unchanged.

## 1. Why

A short prompt's prefill is weight-read bound. A fast chunk of R rows reads every routed expert that one of its rows
picks: 130 rows x 8 picks over 288 experts touch ~97% of them, ~74 GB a rank of ~76 GB (`docs/ROOFLINE.md`: U(R) x
0.264 GB), ~0.27 s at the read rate, before any arithmetic. W13's request log: a 54-row piece 0.285 s, 130 rows 0.41 s,
241 rows 0.45 s. With `GLM53_TF_BATCH_SHORT=1024` a round prefills its short prompts one after another, so C4 read the
experts four times (first tokens 0.47 / 0.9 / 1.3 / 1.7 s with 0540's `EMIT_FIRST`; with thinking on, the text starts
after the round's verify, 1.9 s for all but the first; vLLM 0.8 s).

## 2. A group forward (`mpf.compute`)

The lean fast chunk (0082), with its sub-blocks taken from the members:

- member i's rows sit at `[off_i, off_i + R_i)` of the lean set (no gaps; members in the round's piece order, fewest
  tokens first) and are cut into `GLM53_TF_LEAN_BLOCK` sub-blocks **from its own start** (`mpf.blocks`): a sub-block
  never holds two prompts' rows, and a member's sub-blocks are exactly those of its lone chunk of R_i rows;
- per sub-block (member i, local row la, position pos_i + la), the attention half as alone: hc_pre, the KDA block from
  the member's committed state (la = 0) or its own carry (`Seg.tail`: the conv window; the state goes through the
  member's output buffer as in 0082), the DSA block on the member's latent / index caches and page tables (0290) at
  its positions (`lb.pos_dev`); the FFN half's hc_pre, shared expert, combine, exchange and hc_post per sub-block;
- **once over all T = sum R_i rows**: embed (0082), the router, top-k, grouping and the routed experts;
- the head on each member's last row (one row: qmm, as alone) into its own `b.logits` row; the final norm's group sums
  kept per row (`pfpp._fxs_rows`) for it;
- the loops: 0082's (`_plain`), 0084's pipeline (`_MultiChunk`: the same pre(k) / post(k - 1) order, the same
  drains, slabs, direct partials) and 0320's row split (`_MultiSplit`, used when the lone rule would split a chunk of
  T rows: more than one sub-block). The piece classes are 0084's / 0320's with one method replaced (`attn_pre`: the
  sub-block's member, local row and position) and, without the split, the final norm's sums per row;
- `lean.commit` per member with its carry. The KDA projection / replay rows are slot 0's for every member
  (`Batcher._borrow` for each): they hold one sub-block's rows between its projection and its chain.

The driver (`mpf.run`, both ranks) is `decode._prefill` cut at its forwards: every member's setup (reset or restore
from its resume snapshot, the pending MTP rows re-absorbed, `pfgrid.plan` with its marks and 0540's rule, its room
and pages), then the members' chunks in lockstep (a member whose piece is several chunks, e.g. 0540's cut at n - 64 or
64-row chunks, takes part in several forwards; one forward a chunk index), and per member after each forward: the
last row, the MTP absorb, the DFlash2 taps, the commit, and the snapshot at the end when the piece ends on the grid.
Then `Batcher._piece` runs for each member as before, the group's result standing in for `decode.prefill`
(`mpf.take`: the snapshots, marks, last hidden row, and the first token sampled there), so saves, `_remember`, stats,
the first token's emit (0540), the Stepper and its first drafts are the lone code.

## 3. Exactness

Each member's first token, committed state (KDA states and conv windows, latent / index / MTP / DFlash2 caches),
snapshots and reply are the bits of its piece prefilled alone:

| part | in a group | why the bits are the lone ones |
| --- | --- | --- |
| KDA chain, carries, DSA write / attention / indexer, commit | the member's lone calls (same rows, shapes, state, positions) | its sub-blocks are its lone sub-blocks |
| head, sample, MTP absorb, DFlash2 taps, snapshots, marks, saves | per member, in the lone order | same calls on the same rows |
| matmuls (`fast_qmm.matmul_fast`, FP8, the KDA bf16 copy) | per sub-block (attention, shared, dense) | same calls as alone |
| hc_pre / hc_post / 0190 fused / norms / stream means / embed | slabs of other row ranges, embed over all rows | per-row arithmetic (0084 / 0320 already call them on any 64-aligned or partial slab, and on half a sub-block) |
| router, top-k, grouping, routed experts (`exl3_fast.cu` fat / fast2 / tc), bf16 exchanges | once over all rows | 0085's property: no fast kernel's result for a row depends on the call's other rows or their count (a pair's mma chain is one warp's walk over K; the grouping decides only which warp holds it); a group is one more chunking of the same rows |

Kernel **choices** by the call's total rows: `fast_qmm`'s < 64-row qmm fallback is off in fast chunks (0085) except
for the head, which is per member; the routed experts pick by rows only under `GLM53_TF_FAST_EXPERTS=auto` (0270)
and `once` with `GLM53_TF_ONCE_MIN_ROWS` (0260), both documented as the same bits, but groups are simply not formed
in those modes (`mpf.kernels_ok`). Production runs `fat` (one kernel family for any row count). pfpp's split and
pfoverlap's drain depend on the sub-block count: the same bits by 0320 / 0084. No new kernel, so no new arithmetic:
nothing to check in the interpreter or for PTX (every kernel a group runs is one the lone path runs, with the lone
path's tests).

What the CPU cannot see: that the real routed-expert / matmul kernels are row-independent (0085's GPU tests already
check chunkings; §9 step 3 checks groups end to end).

## 4. Which pieces go together

- **Groups** (`mpf.groups`, both ranks, in `_execute`): the round's pieces (`batchplan.pick_pieces`: short prompts
  fewest-left first, then the fair share's piece) packed in that order into groups of equal request knobs
  (`knobs.HEADER`: the chunk kernels are knob-dependent), at most `GLM53_TF_MULTI_PREFILL_ROWS` piece rows (default:
  the lean rows, `GLM53_TF_PREFILL_ROWS_MAX`, 4,096 in prod). A piece joins when its request prefills through a fast
  lean chunk, has no image rows (0500's `vision.active` holds one table) and profile / 0430 dump are off. `_piece`
  recomputes each member's bound when its turn comes; the members after it still hold their slots, so only the LAST
  member could find itself alone (the others done at their first token) and get 0335's solo bound: it leaves the group
  (prefills alone after it) when `GLM53_TF_SOLO_PIECE` would move its bound and no other request holds a slot.
  Others run alone as before.
- **More pieces a round** (`mpf.more_pieces`, rank 0's plan): when the fair share allows a piece, more waiting
  prompts' pieces join the round (fewest tokens left first) while their rows fit the budget, so prompts past
  `BATCH_SHORT` (agent bursts of 1-4k-token tool results) share forwards too.
- **Arrivals** (`mpf.coalesce`, rank 0, `GLM53_TF_MULTI_PREFILL_WAIT_MS`, default 10): when no slot holds a request
  and fewer requests wait than there are slots, the plan waits until 10 ms after the oldest arrival (or until the
  slots are covered). Without it C4's first request is admitted alone (the others arrive a few ms later, W13
  `queue_s` 0.004 s) and only three share. A lone request pays <= 10 ms; nothing waits while anything runs.
- **Decoders in the same round** (0200): the group's forward(s) and the members' first tokens / first drafts, then
  the round's verify forward, as with serial pieces; the fairness debt counts the group's time.
- **Rank lockstep**: the groups are a function of what both ranks hold (the plan's pieces, the slots' prompts, knobs,
  progress, marks, the settings); `GLM53_TF_MULTI_PREFILL` and the row budget are checked equal at load. The
  collectives: the group forward's (identical on both ranks), then per member its MTP absorb, sample, saves, drafts
  in slot order, on both ranks.

## 5. Knobs

| knob | default | |
| --- | --- | --- |
| `GLM53_TF_MULTI_PREFILL` | `0` | `1`: group pieces (load-time, both ranks, checked) |
| `GLM53_TF_MULTI_PREFILL_ROWS` | `0` (= the lean rows) | piece rows a group may hold (>= 64; checked) |
| `GLM53_TF_MULTI_PREFILL_WAIT_MS` | `10` | rank 0's idle wait for more arrivals; `0`: none |

Boot line (rank 0): `multi prefill (patches/0560): a round's pieces in one forward, up to 4096 rows (10 ms wait ...)`.
Request log (0300): `multi` = members of the request's group (null: alone). Batcher counts `multi_groups`,
`multi_members`.

Memory: nothing new beyond the member carries (34 KDA layers x 3 x 12,288 bf16 = 2.5 MB a member, 4 kept) and the
per-row final sums 0320 already keeps (1 MB at 4,096 rows): groups fit the lean set that exists.

## 6. Estimates

Model: a lone piece of R rows takes t(R) ~ 0.30 + 0.5 ms x R (s) (W13: 54 / 130 / 241 rows -> 0.285 / 0.41 / 0.45 s;
W15 4,096-row chunks at ~1,600 tok/s), incl. the member's MTP absorb and sample. A group of k members and T rows:
t(T) + (k - 1) x d, d ~ 12-20 ms for the member's own attention / shared / dense weight reads (2.2 GB a rank, ~9 ms)
plus its launches and exchanges (0084 hides most of the latter), + 1 ms a head. The expert read (~74-76 GB) is paid
once.

**RigMark C4** (4 x ~130 tokens, T ~ 525): group ~0.55-0.6 s (vs 4 x 0.41 = 1.64 s of pieces).

| C4 per-stream first token | W13 (b5) | 0540 (b7, measured W15) | 0560, all four in one group | 0560, first one alone |
| --- | --- | --- | --- | --- |
| thinking off (first token = text) | 1.91 s | median 0.93 s | **~0.6-0.7 s** (all four) | 0.47 s, then ~1.0 s x3 |
| thinking on (text after the first verify round) | 1.91 s | ~1.9 s (unchanged) | **~0.7-0.8 s** | ~0.5 s, then ~1.1 s x3 |
| vLLM TP2 k=7 (Alex) | 0.81 s | | | |

"All four" needs the arrivals within `WAIT_MS` of each other (W13: a few ms). **C2** (2 streams): group of 2 x ~130
rows ~0.45 s -> first tokens ~0.5 s (thinking off) / ~0.6 s (on), was ~0.9 s median.

**Agent bursts** (aggregate prefill tok/s on the 4 slots; prompts fit a piece; 0540's cut at n - 64 makes a prompt
of more than ~320 tokens two chunks, the second a ~40-row one that re-reads the experts; a group pays it once):

| burst | serial pieces | grouped | gain |
| --- | --- | --- | --- |
| 4 x 130 tokens | 1.64 s, 317 tok/s | ~0.58 s, ~900 tok/s | ~2.8x |
| 8 x 250 tokens (4 slots, 2 rounds) | ~3.6 s, ~560 tok/s | ~1.75 s, ~1,150 tok/s | ~2x |
| 4 x 1,000 tokens (2 chunks each) | ~4.6 s, ~870 tok/s | ~2.7 s, ~1,500 tok/s | ~1.7x |
| 4 x 4,000 tokens (pieces of 2,048; 2 a group) | ~11 s, ~1,450 tok/s | ~9.5 s, ~1,700 tok/s | ~1.15x |
| one long prompt | unchanged (not grouped) | | |

Decoders sharing the rounds wait for a group as for a piece (fair share); a group is longer than a piece but serves
several, so the prefill time per decoded token does not rise.

## 7. Limits and follow-ups

- Attention projections, o_proj, shared expert and dense MLP are read once a **member** sub-block (not once a group):
  the per-slot sub-blocks keep every per-slot call identical to alone (the bits argument is 0085's for experts only).
  Sharing them (projections once over the group, per-member chains inside a sub-block) saves the d above (~15 ms a
  member, ~3% of a C4 group) and needs segment-aware DSA / KDA blocks: a follow-up once groups are proven on the GPU.
- The members' MTP absorbs and first samples run one after another (a few ms each); `batch.mtp_multi` /
  `sample_multi` could batch them.
- Exact (non-fast) prefills, image prompts, profile / dump runs: alone.
- 0550 (memory safety), if present: its selection scratch is reserved per member before the group forward (as
  `_prefill` does); its allocator trim runs in each member's `_piece` (after the group).

## 8. Tests (CPU, offline)

`tests/cuda/test_multi_prefill_patches.py` (83 on the CPU + 4 GPU; `PYTHONPATH=<patched tree>/src:<tree>/tests/cuda:tests/cuda`):

- host: knob / settings / budget, `more_pieces`, `pack`, per-member sub-blocks, the solo guard;
- test_lean_patches' **hash model** (exact integer kernels with the real row / position / KDA / cache / routing
  dependencies, the real 0082 / 0084 orchestration): a group forward == each member's lone lean chunk bit for bit
  (last logits, final rows, taps, streams, KDA states and conv windows, caches) for 6 member sets (1-600 rows, 2-5
  members, resume points 0-192), sub-blocks of 64 / 128, 0082's loop and the four 0084 variants, EXL3 and MLX MoE, any
  member order; the routed experts run once a group, KDA / DSA once a member sub-block; three planted bugs are caught
  (one carry for all members, positions over the group's rows, sub-blocks of the group's rows); the driver's real
  hooks (`forward_group`, `commit_member`);
- **two real processes over gloo** (test_prefill_pp_patches' communicator, rank-specific weights): with 0320 on, each
  rank's group (split when over one sub-block) == the lone chunks; the ranks agree on the replicated rows;
- the real **Batcher** on test_batch_sessions' hostile fake model: C4 in one round and one group (fewest first, first
  tokens out in turn, replies and final slot states == alone; 1 or 3 lockstep chunks); resends resume at n - 64 from
  the group's snapshots, == fresh; 4 interleaved sessions with grouped pieces (roomy and tiny store, evictions) ==
  fresh prefill + serial decode; a follower (rank 1) replaying rank 0's messages forms the same groups (forwards,
  trace, logs, stores, states equal); other knobs / image rows / a member's error / a decoder in the round; more
  pieces past BATCH_SHORT (4 x 1.5k tokens, 4 members a forward, <= the budget); the last member leaving for a solo
  piece; the idle wait and its bound.
- No new kernel (every call is one the lone path makes), so no interpreter or PTX test is added; the kernels' own
  tests stand. Regression: the batch / session / replay / lean / overlap / pp / kv-pool / prefix-share / decode-overlap
  suites give the same results with and without 0560 (knob off; `test_solo_piece_patches.py`'s 3 failures are the
  stack's, 0500's fake-job gap). 0560 applies on the committed stack, on b7's patch list and on 0550's current tree;
  the tests pass on all three.

## 9. GPU test plan

Image `glm53-tensorfold:b8` = b7's patch list (0001-0490, 0500, 0540) + 0560 (0560 applies on b7's list and on the
whole stack through 0550; with 0550 too if it is adopted first). Prod down ~1.5 h.

1. **Tests in the image** (one GPU): `run_tests_in_image.sh` for `test_multi_prefill_patches.py` (host + hash-model
   + gloo parts, and `test_gpu_group_equals_alone`: synthetic EXL3 checkpoint, lean and pipelined, greedy and sampled,
   4 prompts incl. one of several sub-blocks grouped == each alone, next turns resumed from the group's snapshots),
   plus `test_lean_patches.py`, `test_overlap_patches.py`, `test_prefill_pp_patches.py`, `test_batch_sessions_patches.py`,
   `test_replay_ttft_patches.py`, `test_fastpf_patches.py`, `test_cindep_patches.py`: all pass.
2. **Candidate**: `config/prod.env` + `IMAGE=glm53-tensorfold:b8` + `GLM53_TF_MULTI_PREFILL=1`. Boot line present, no
   rank mismatch. Gates (`gates.sh`): exact 10/10, **batchexact 4/4** (4 concurrent requests: now grouped; replies
   == alone), W9 transcripts identical, reply sha `8794a3463259cc2f`, N1 12/12 (batched == alone, drafted == serial),
   prefill 24.5k / 98k equal to b7 (lone prompts are never grouped: same numbers), decode not lower, 4 x 250k stress
   MemAvailable >= 8 GiB both nodes, MMLU-200 >= 87%. With `GLM53_TF_MULTI_PREFILL=0` on b8: b7's numbers.
3. **Group exactness on the real model**: 4 short prompts (130 / 128 / 134 / 133 tokens, greedy and seeded sampling,
   thinking on and off, `max_tokens` 64) sent concurrently, 10 rounds: request log `multi` = 4 (or 3 + the first
   alone if the arrival spread exceeds 10 ms: note `queue_s`), each reply == the same request sent alone (sha), also
   with the knob off; a burst of 8 prompts of 200-3,000 tokens (mixed, some past `BATCH_SHORT`) twice: replies == alone,
   `multi` >= 2 on most, the second send `cached` = n - 64 (0540's snapshots from the groups); a session conversation
   (4 sessions over a 20k system prompt, 0310 marks) with the knob on: replies == knob off.
4. **TTFT**: RigMark C4 / C2 concurrency phases (`scripts/rigmark/run.sh tensorfold`, new `COMPARISON_ID`) and
   `multiturn.py --modes concurrent --streams 4`: per-stream TTFT median target <= 0.8 s (thinking on and off), C2
   <= 0.65 s; request log `prefill_s` of a 4-group ~0.55-0.65 s. If arrivals spread: `GLM53_TF_MULTI_PREFILL_WAIT_MS=20`.
5. **Throughput**: 8 / 16 concurrent 250-1,000-token completions (`bench/`'s burst or a loop of `req.py`): aggregate
   prefill tok/s vs knob off (expected 1.7-2.8x, §6); `GLM53_TF_PROFILE` off. Decode rate of a running stream during
   a burst: not lower than with serial pieces per prefilled token.
6. **0320 with groups** (two Sparks only): a 4 x 130 group splits (525 rows > 512): the pp gates of step 2 cover it;
   also `GLM53_TF_PREFILL_PP=0` once (the same replies).
7. **Adopt** when 2-5 pass: `GLM53_TF_MULTI_PREFILL=1` in `config/prod.env` with `IMAGE=glm53-tensorfold:b8`. Revert:
   drop the knob (b8 without it == b7).
