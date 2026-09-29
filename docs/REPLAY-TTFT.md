# Warm replay and multi-arrival TTFT (patch 0540), from RigMark W13

W13 (`docs/RESULTS.md`, `results/rigmark/tensorfold-20260929/`) found four things to fix. This page is the
analysis, what `patches/0540-glm-replay-ttft.patch` changes, the estimates, and the GPU plan. Everything here was done
offline (no GPU, the Sparks untouched); nothing is adopted until the GPU gates below pass.

**W15 (docs/RESULTS.md):** GPU-validated and adopted in image b7 (with 0500). Replay of identical 8K / 32K / 64K prompts:
cached n - 64, TTFT 0.23 / 0.26 / 0.29 s (b5: 5.1 / 10.2 / 10.9 s); grid-aligned 8K cold -1.8%. `GLM53_TF_EMIT_FIRST`
works with thinking off (C4 median TTFT 0.93 s vs 1.46 s) but not with thinking on: the first generated token streams no
text, so RigMark's C4 (reasoning low) is unchanged. Nine GPU test failures were old-rule expectations; the tests (and a
`test_glm_engine.py` hunk in this patch) now follow the rule, knob on and off.

| # | Finding (W13) | Change | Default | Replies |
| --- | --- | --- | --- | --- |
| 1 | Immediate replay 1,597 / 3,319 / 6,304 tok/s at 8K / 32K / 64K (vLLM 1,812 / 11,046 / 11,364): the replays resumed only 0 / 16,384 / 49,152 tokens | the prompt's snapshot sits at the last grid point **strictly before** its end (`GLM53_TF_SNAPSHOT_BEFORE_END`) | on | unchanged (a cut and a snapshot on the 64-grid) |
| 2 | C4 per-stream TTFT 1.91 s (vLLM 0.81) | a prompt's first token goes out when its piece ends, not after the round's other pieces and verify forward (`GLM53_TF_EMIT_FIRST`); multi-slot prefill analysed, not implemented | on | unchanged (scheduling only) |
| 3 | reasoning sent in both `reasoning` and `reasoning_content`, RigMark counts it twice | recommendation only: `GLM53_TF_REASONING_FIELDS=reasoning` (what the vLLM kit sends) | - | - |
| 4 | `tests/cuda/test_vision_patches.py`'s bf16-vs-fp32 bound (2%) is below what bf16 gives (W14: 6.0% in transformers' own tower) | bound 8%, comment citing W14 | - | - |

## 1. Warm replay

### Why the replay re-prefilled 8-16K tokens

The request log of the run (`requests.jsonl`, n = 50-61) says it directly:

| depth | cold prefill | replay `cached` | replay `pieces` | replay prefill |
| --- | --- | --- | --- | --- |
| 8,192 | 5.07 s | 0 | 2 | 5.04 s (the whole prompt again) |
| 32,768 | 19.8 s | 16,384 (`cache_src` ram) | 4 | 9.8 s |
| 65,536 | 40.2 s | 49,152 (ram) | 4 | 10.3 s |

A resume needs a snapshot of a **strict** prefix of the prompt (`decode._prefill`: the first token is sampled from the
prompt's last row, and a snapshot does not hold it). Fast prefill (patches/0085) keeps one snapshot per prompt, at
S = the prompt's last multiple of the 64-token grid. RigMark's depths are multiples of 64, so S = n: the only snapshot
near the end is the whole prompt, which an identical prompt cannot resume from. The next one down is a session mark
(`GLM53_TF_SESSION_EVERY` = 16,384; for 8,192 there is none). Chat prompts rarely end on the grid (63 lengths in 64
already snapshot at n // 64 * 64 < n), so this is mostly a benchmark edge, but it also hits a regenerate / retry of
a prompt that happens to end on the grid, and every exact-mode (`GLM53_TF_FAST_PREFILL=0`) prompt, whose snapshot was
always at n.

### The change (`pfgrid.plan(before=True)`, `decode._prefill`, `engine._run`, `batch.Batcher._piece`)

For the prefill that ends at the prompt's end (the lone engine's `_run`, the batcher's last piece; the callers set
`e.snap_before = len(prompt)` and `_prefill` applies it only to a prefill of exactly that length, once), the snapshot
point is S = (n - 1) // G * G, the last grid point strictly before n, for every n:

- off the grid this is the old S (n // G * G): **nothing changes** for 63 of every 64 lengths (checked exhaustively
  in the tests: same chunks, same snapshot, same marks);
- on the grid, S = n - G: the last chunk is cut at n - 64 (as every off-grid prompt's already is), the snapshot is
  taken there, and it **replaces** the one at n. rule 2's tail exception is unchanged (when the last chunk already
  starts within `GLM53_TF_SNAPSHOT_TAIL` = 256 rows of S, the snapshot is its start, no cut);
- a resend resumes at S and prefills the last <= 64 rows (<= tail + 64 with the tail exception); the resumed
  prefill's plan gives snap == begin, so the resume snapshot stays the prompt's snapshot (nothing new stored);
- exact prefills (grid 0) take theirs at the same S, as a chunk cut (exact chunks never depend on their cuts; with the
  default 64-row exact chunks it costs nothing), instead of `take_snapshot` after the prompt; a fresh exact prompt of
  <= 64 tokens keeps none (the reply snapshot still covers its next turn);
- intermediate batch pieces keep their snapshot at their end (the next piece resumes from it), as before.

`GLM53_TF_SNAPSHOT_BEFORE_END=1` (default) / `0` (patches/0085's rule). It moves a cut, so both ranks must agree
(checked at load, like `fastpf.settings`).

**Why it is exact.** A cut and a snapshot at a multiple of 64 are exactly what every off-grid prompt, every session
mark and every piece bound already get (patches/0085: fast chunks' bits do not depend on the chunking; a state at a
multiple of 64 is the fresh prefill's state there; exact rows are row-invariant). The resend is a resume from such a
snapshot, and resumed == fresh holds for it as for any. No new arithmetic.

**Can an identical prompt be served?** Yes, now: resume at n - 64 and re-prefill 64 rows; the first token, the state
and the reply are the fresh prefill's (test: `test_resume_at_end_minus_grid_equals_fresh`). The rule "a resume needs a
strict prefix" stays; the snapshot just is one. (A zero-row variant, storing the last row's logits shard, ~300 KB a
rank, in the snapshot at n and sampling the first token from it, would skip even the 64-row forward and the cold cut;
it touches the resume path, the snapshot format and the NVMe tier's entry format, so it is left as a follow-up that
needs GPU validation.)

**Memory.** One prompt snapshot per request, as before (it moved, it was not added): the slots keep at most 2 snapshots
each (`_remember`), the store stays under `GLM53_TF_SESSION_GIB`, and each save carries one snapshot. The moved
snapshot holds up to 64 fewer rows (exact prompts' entries get slightly smaller; one seed of
`test_session_patches.py`'s tight-budget case stopped evicting, so its budget went from 40 to 36 pages). No extra KDA
state (~94 MB a rank a snapshot, 0310's figure) is kept. Tests: `test_rule_keeps_no_more_snapshots`,
`test_batched_resend_resumes_all_but_the_last_grid_step`.

**Cost.** One more chunk (one more read of the weights, ~0.2-0.25 s: W13's 54-row prefill took 0.285 s, W10's
314,304-token resume with 22 rows 0.25 s) on a **cold** prompt that ends on the grid and whose last chunk starts more
than 256 rows before n - 64. Off-grid prompts pay nothing new (they always had this cut).

### Estimate for RigMark's prefill phase

Replay TTFT ~ restore from the RAM store (the latent rows and one KDA snapshot: a few ms to ~10 ms) + one 64-row fast
chunk + sample: ~0.25-0.35 s (W10 needle: 0.25 s for a 22-row resume at 314k).

| | cold 8K / 32K / 64K tok/s | replay 8K / 32K / 64K tok/s |
| --- | --- | --- |
| W13 (b5) | 1,598 / 1,641 / 1,621 | 1,597 / 3,319 / 6,304 |
| 0540, estimate | ~1,530 / ~1,625 / ~1,613 (-4 / -1 / -0.5%: the extra 64-row chunk) | **~23,000-33,000 / ~94,000-131,000 / ~187,000-262,000** |
| vLLM TP2 k=7 (Alex) | 1,813 / 1,908 / 1,922 | 1,812 / 11,046 / 11,364 |

The cold cost falls only on on-grid prompts; with `GLM53_TF_SNAPSHOT_BEFORE_END=0` the cold numbers are W13's and the
replay ones too.

## 2. C4 time to first token

### What the request log shows

C4 rounds (n = 71-82; `first_s` = time to first token, `queue_s` = before admission, `prefill_s` = from admission to
the end of the request's piece):

| round | request | `queue_s` | `prefill_s` | `first_s` |
| --- | --- | --- | --- | --- |
| 1 | 71 | 0.004 | 0.41 | 0.47 |
| 1 | 72 / 73 / 74 | 0.47 | 0.41 / 0.82 / 1.24 | 1.91 / 1.91 / 1.91 |
| 2 | 76 / 75 | 0.004 | 0.41 / 0.82 | 0.90 / 0.90 |
| 2 | 78 / 77 | 0.90 | 0.42 / 0.83 | 1.89 / 1.89 |
| 3 | 79 | 0.006 | 0.42 | 0.49 |
| 3 | 80 / 81 / 82 | 0.49 | 0.42 / 0.83 / 1.24 | 1.93 / 1.93 / 1.93 |

Two things add up:

1. **Pieces are serial.** The first arrival is admitted alone (the others arrive a few ms later, during its piece);
   the next round admits the other three and, with `GLM53_TF_BATCH_SHORT=1024`, prefills all three in that round, one
   ~130-row forward (~0.41 s) after another.
2. **First tokens are held to the end of the round.** With `GLM53_TF_DECODE_OVERLAP=1` (patches/0370's `emit`),
   rank 0 holds a round's tokens until the round's verify forward has been launched. A piece's first token is
   held too, so all three first tokens of a round go out together, after the last piece and the verify launch: 1.91 s
   although the first of them was ready at 0.88 s.

### The change: `GLM53_TF_EMIT_FIRST` (default 1)

`Batcher._piece` flushes the held tokens right after it emits a prompt's first token (they are only this round's
earlier first tokens: the previous round's were flushed at this round's plan), so each first token goes out when its
piece ends. Rank 0 only, host only: the same kernels, the same rounds and collectives on both ranks, the same tokens
in the same order a request (test: `test_multi_arrival_first_tokens_and_exactness`: 4 short prompts queued together
are admitted and prefilled in one round, fewest tokens first; each first token is out before the next piece starts;
every reply == the same request alone; `test_emit_after_the_next_forward_launch[first-now]`). The cost is a few
queue puts on the round loop's thread between two pieces.

Estimate (the W13 log re-timed: first token = queue + own prefill + 0.06 s): per-stream C4 TTFT
0.53-1.85 s, **median ~1.15-1.2 s** (from 1.91; vLLM 0.81). C1 unchanged; C2 median ~0.92 -> ~0.7 s (in the round
where both streams were admitted together, n = 67 / 68, the second first token moves from 0.92 to ~0.47 s).

### Multi-slot prefill (several prompts in one forward): analysed, not implemented

**Exact?** Yes, in principle. Every kernel of a fast chunk other than the per-sequence ones is row-independent
(patches/0085: `fast_qmm`, the routed experts' mma chains, hyper-connections, norms, the router, the head), so rows of
several prompts can share one chunk as decode rows of several slots already share one verify forward
(`batch.compute_multi`). What must stay per slot, as `compute_multi` does for decode: the KDA chains (each slot's
state and conv window, `fast_kda`'s 64-row blocks at the slot's own absolute positions: every slot's segment must
start on its own 64-grid), the DSA attention and indexer (the slot's latent / index caches and positions), the KV and
index writes, the MTP absorb, the DFlash2 taps, the last-row head and the first token, the commit and the snapshots.

**Why not now.** Production's prefill is the lean chunk (patches/0082) with 0084's pipelined exchanges and 0320's
row-split hyper-connections (`GLM53_TF_LEAN_PREFILL=1`, `PREFILL_OVERLAP=1`, `PREFILL_PP=1`), each with its own
sub-block loop over one `State`. A multi-slot chunk needs segment-aware versions of all three loops (sub-blocks
never straddling two slots, per-slot `_kda_block` / `_attn_half` state, carries and projection rows: slots other than
0 hold only window-sized projection rows and borrow slot 0's, one slot at a time today), one head row and one keyed
sample per slot, and per-slot snapshots / marks / session saves, on both ranks in the same order. That is a new path
through the most tuned code, and its bits can only be checked on the GPU (the fake model cannot see a sub-block that
straddles two slots). It is the right next step if C4 TTFT matters: estimate one ~520-row forward ~0.45-0.55 s
(W13: 54 rows 0.285 s, 130 rows 0.41 s, 241 rows 0.45 s), so **all four first tokens at ~0.55-0.65 s** (vLLM
0.78-0.81).

**Also considered.** Coalescing arrivals (wait a few ms when idle so the four are admitted in one round): with
serial pieces it gains ~0.05 s of median TTFT and costs every lone request that wait; not done. A smaller first
piece does nothing for ~130-token prompts (one piece already); `GLM53_TF_BATCH_SHORT` already orders short prompts
fewest-left first.

## 3. Reasoning fields: `GLM53_TF_REASONING_FIELDS=reasoning`

What the vLLM kit sends (read-only on the head node, `~/glm53-exl3-2x-kit`): `--reasoning-parser glm45` on vLLM
`0.1.dev20051` (`.glm53-exl3-head.inner.sh`); every acceptance run of the kit prints `reasoning field: reasoning`
(`local/task6-pr125-gates-20260906.txt`, `local/task28/*`, `local/task30/*`, `local/task42-*`, `local/task24-*`); the
kit's own probes read `delta.reasoning` first. Alex's published vLLM receipts count 60 prose reasoning characters where
our raw receipt counts 120 for the same 60 (RigMark concatenates `reasoning` + `reasoning_content`): vLLM sends one
field. TensorFold's default `both` sends two.

**Recommendation for `config/prod.env`** (not edited here; it takes a restart):

```bash
GLM53_TF_REASONING_FIELDS=reasoning
```

Then TensorFold's stream matches the kit's (`delta.reasoning` only), RigMark's reasoning characters and hashes stop
doubling, and clients that worked against the vLLM kit see the same field. A client that reads only
`reasoning_content` (older OpenAI-compatible UIs) would stop showing thinking; the kit never sent it either, so any
client that worked with the kit is fine. Until prod.env has it, RigMark runs can set it for the run
(`GLM53_TF_REASONING_FIELDS=reasoning scripts/rigmark/window.sh tf-up`, docs/RIGMARK.md gap 2).

## 4. Vision test bound

`tests/cuda/test_vision_patches.py`: bf16 vs fp32 tower `rel < 8e-2` (was `2e-2`), with a comment citing W14
(transformers' own `Glm5NextVisionModel` bf16 vs fp32: 0.060 on the random image, ours 0.059; our fp32 vs theirs
0.004).

## Tests (CPU)

`tests/cuda/test_replay_ttft_patches.py` (32; host-only parts need no torch; the rest on
`test_batch_sessions_patches`' hostile fake model, whose rows depend on everything they may depend on):

- plans: on-grid snapshot at n - 64 with the cut (RigMark's 8K / 32K / 64K last pieces), the resend's plan
  (snap == begin), marks deduplicated, other grids; off-grid plans identical to 0085's for every length / resume
  point / chunk size tried, every snapshot a strict prefix within tail + G; exact plans; the knob;
- resume at end - grid == fresh: first token, whole state and a 16-token reply, exact with cut chunks, exact, fast,
  on- and off-grid lengths, and the rule itself leaves the fresh prefill's bits;
- the real batcher and session store: a prompt sent three times resumes at (n - 1) // 64 * 64 on the 2nd and 3rd
  (0 without the rule for an on-grid prompt, W13's case), replies == fresh, <= 2 snapshots a slot, store under
  budget, one snapshot a save; the same conversations with and without the rule save no more snapshots with it;
- multi-arrival: see §2.

Also changed: `test_decode_overlap_patches.py::test_emit_after_the_next_forward_launch` runs with and without
`EMIT_FIRST` (with it, no token waits for a launch there); `test_session_patches.py`'s tight budget 40 -> 36 pages.
The whole CPU suite: no new failures against the stack without 0540 (the failures there are the environment's:
interpreter / emulator tests that need their own process or a GPU toolchain, 0500's fake-job gap in
`test_solo_piece_patches.py`).

## GPU plan

Image `glm53-tensorfold:b7` = b5's patch list (0001-0490) + 0540 (0540 applies on it and on the whole stack through
0530; `PATCHES` = b5's `build-patches.txt` + `0540`). One window, prod down ~1.5-2 h. The usual W10 gate set
(`results/W10/gates.sh`), then the replay / TTFT checks, then RigMark.

1. **Image and tests in it.** `run_tests_in_image.sh` for the CPU files above plus `test_batch_sessions_patches.py`,
   `test_fastpf_patches.py`, `test_cindep_patches.py`, `test_session_patches.py`; the GPU parts of
   `test_batch_sessions_patches.py` (fast prefill sessions batched, follower replay) and `test_fastpf_patches.py` on
   the synthetic checkpoint: all pass.
2. **Control and candidate.** Start b7 with `config/prod.env` + `IMAGE=glm53-tensorfold:b7`
   (+ `GLM53_TF_REASONING_FIELDS=reasoning` for the RigMark part). Boot log: no rank mismatch on
   `GLM53_TF_SNAPSHOT_BEFORE_END`.
3. **Gates** (`gates.sh B7`): exact **10/10**, batchexact **4/4**, `ab.py` 24.5k / 98k x2 **reply sha
   `8794a3463259cc2f`** in every cell (those prompts are off the grid: prefill tok/s must equal the controls),
   decode 1 / 4 streams not lower, **4 x 250k stress MemAvailable >= 8 GiB on both nodes** (compare with FIN's
   9.74 / 8.32), 0 OOM, MMLU-200 >= 87%, exact / batchexact again.
4. **Replay** (token-ID `/v1/completions`, `max_tokens` 8, `ignore_eos`, as RigMark): for 8,192 / 32,768 / 65,536 and an
   off-grid 32,700, a fresh nonce prompt then the same ids at once, 3 pairs each:
   - request log: replay `cached` = n - 64 (8,128 / 32,704 / 65,472), 32,640 for 32,700 (as before), replay
     `pieces` 1; cold `pieces` as W13 (the cut is one more chunk inside the last piece, not a piece);
   - the 8 tokens of cold and replay equal (greedy);
   - replay TTFT <= 0.5 s at every depth; cold within 5% of W13 (8K ~5.3 s);
   - the same with a chat prompt regenerated (`messages` identical, `temperature` 0 and a sampled one with a seed):
     identical replies, `cached` within 64 of the prompt.
5. **C4 TTFT.** `multiturn.py --modes concurrent --streams 4` or RigMark's concurrency phase: request log
   `first_s` of the 2nd-4th request of a round ~0.9 / 1.3 / 1.7 s (was 1.9 for all), `queue_s` unchanged; batchexact
   4/4 already covers the replies.
6. **RigMark rerun** (`scripts/rigmark/run.sh tensorfold`, a new `COMPARISON_ID`, e.g.
   `2026-10-glm53-exl3-2xspark-tensorfold-0540-v1`, because the NVMe session tier survives restarts): 15/15 gates,
   prefill rows `prompt_tokens` == depth, replay 8K / 32K / 64K > 11,364 tok/s (vLLM's), C4 per-stream TTFT median
   <= 1.3 s, `reasoning_sent_twice` false in `preflight.json`. Record in RESULTS.md as W15 with the compare table.
7. **Adopt** when 3-6 pass: `IMAGE=glm53-tensorfold:b7` and `GLM53_TF_REASONING_FIELDS=reasoning` in
   `config/prod.env`. If only the cold 8K cost is objectionable, `GLM53_TF_SNAPSHOT_BEFORE_END=0` keeps the rest.
