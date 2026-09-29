# Trimmed draft vocabulary (patches/0420, DECODE-PLAN E7): coverage study, patch, estimate, GPU plan

Offline work, 2026-09-29. No GPU and no Spark were used. The only Spark access was a read-only `scp` of the
tokenizer and config files from the head node's HF snapshot (`MODEL_PATH` in `config/prod.env`).

> **Public release note (2026-09-30): the shipped ranking is built from public text.** The study below (sections
> 1-3, `results/draftvocab/`) ranks tokens by their frequency in the maintainer's own coding-agent sessions, which are
> not published; only aggregate counts are. The `draft_vocab.txt` in `patches/0420-glm-draft-vocab.patch` is instead
> built by `bench/draftvocab_public.py` from pinned, permissively licensed public text: OpenAssistant oasst1
> (Apache-2.0), SWE-Gym/OpenHands-SFT-Trajectories (MIT; its tool calls rewritten into GLM's `<tool_call>` markup),
> CPython 3.14.7 `Lib/` and `Doc/` (PSF-2.0), denoland/std (MIT) and ripgrep 14.1.1 (MIT / Unlicense): 13.7M
> reply-like tokens ranked first, 13.0M other tokens as the prior, then id; the chat / tool-call format tokens lead
> the list. 118,040 ids (66,797 in rank 0's half, 51,243 in rank 1's). Coverage of the public list on the private
> sessions' GLM replies (aggregates only, held out as in §2): **78.8% at N = 16k, 87.1% at 32k, 91.3% at 48k**,
> against 90.4% / 95.6% / 98.1% for a list built from those sessions themselves (the gap is mostly project-specific
> identifiers). The estimates in §0 and §3 assume the private list; with the public one expect less, likely nothing
> at 1 stream. For your own traffic, write a ranking with `bench/draftvocab.py --write-list <file>` (it reads an
> opencode database) and point `GLM53_TF_DRAFT_VOCAB=<file>[:N]` at it. 0420 is off by default and not adopted.

Deliverables:

- the patch, `patches/0420-glm-draft-vocab.patch`, which applies on the production stack through 0410;
- `bench/draftvocab.py`, the study, which also writes a ranking file from your own sessions;
- `bench/draftvocab_public.py`, which builds the shipped ranking from public text (see the note above);
- `bench/acceptpos.py`, acceptance by position from the engine's stats;
- the tests `tests/cuda/test_draft_vocab_patches.py` and `tests/test_draft_vocab_interpreter.py`;
- the study output in `results/draftvocab/` (aggregates only).

## 0. Bottom line

- **The trim pays less here than E7 assumed.** E7's +1.5% (1 stream) needed ≥ 97-98% coverage and both drafters
  on the list. What the data supports:
  - **MTP arm only, 24,576 listed rows a rank** (`GLM53_TF_DRAFT_VOCAB=49152`, the default fallback rule): about
    **+1.2-1.7% on MTP rounds** (held-out). That is roughly **+0.5-0.8% for 1-stream decode overall**, because
    DFlash2 and lookup rounds are unchanged.
  - The upper bound, if unlisted tokens were never drafted right anyway: +3% on MTP rounds, about +1.4% overall.
  - At 4 streams it is likely nil: one batched head pass saves ~1.8 ms of a ~115 ms round, and one slot's fallback
    makes the whole pass full.
- **DFlash2 should stay on the full head.** One head pass a round saves ~0.6 ms of a 58-82 ms round. DFlash2's long
  chains lose more to unlisted tokens than that: net -0.5% prose and -2% to -4% code / tool calls, even with the
  fallback. The patch supports it (`_ARMS=dflash|all`) only so the GPU window can confirm this.
- **Coverage** (held out, per-rank list, 24,576 rows a rank): 98.1-98.7% of reply tokens (prose 98.2-98.5%, code
  97.8-98.8%, tool calls 98.3-99.2%). CJK text gets only 63-64%.
  - The prompt-driven fallback rule is therefore necessary: without it a CJK conversation loses 28-38% of its MTP
    tokens a round. With it the loss is 0.
  - At 16k rows a rank, 95.6-96.9%: net negative.
- **Exact by construction, and checked.**
  - Every committed token is still the target's keyed sample of its own full-vocabulary rows. The target's sampling
    code path is not touched: `sample_rows` without `draft`, `sample_multi`.
  - Draft columns map to token ids before the exchange, so the keyed Gumbel noise is the target's for the same
    (seed, position, token).
  - Tests: drafted == serial through the real `mtp_decode` / `draft` / `absorb` / `Engine.mtp` / `Engine.sample`,
    greedy and sampled, with the list, with the fallback, and without. Rank 0 == rank 1.
  - No Triton source changed: PTX 33 / 33 identical. The trimmed head's logits are the full head's listed columns bit
    for bit (interpreter).
- **Recommendation.**
  - Keep it off: one A/B in a GPU window with spare time (section 7).
  - Adopt `GLM53_TF_DRAFT_VOCAB=49152` (MTP arm, fallback `0.03,256`) only if 1-stream decode gains ≥ +0.5% with MTP
    position-1 acceptance down ≤ 0.02, and the gates hold.
  - The larger lever, not built, is a per-request extension. The request's own history tokens would join the list.
    The study's `+ctx` rows show 99.3-99.5% coverage at 24,576 rows. That needs per-request head rows, a design of
    its own.

## 1. Where the time goes

The drafters' last matmul is the vocabulary head. Each rank holds 77,440 rows × 4,096 as 4-bit words plus bf16
scales and biases (production's `q4mse` head; the drafters read `w.draft_head` if the head is bf16, else `w.head`):

| per rank | rows | MB | ms at 200 GB/s | ms at 215 GB/s |
| --- | ---: | ---: | ---: | ---: |
| full head | 77,440 | 158.6 + 19.8 = **178.4** | 0.89 | 0.83 |
| list, 32,768 a rank | 32,768 | 75.5 | 0.38 | 0.35 |
| list, 24,576 a rank | 24,576 | 56.6 | 0.28 | 0.26 |
| list, 16,384 a rank | 16,384 | 37.7 | 0.19 | 0.18 |

- **Saved per head evaluation, 24,576 rows: 0.61 ms at 200 GB/s** (0.57 at 215). At 16,384 it is 0.70, at 32,768
  it is 0.51.
- Per round:
  - **MTP rounds**: the absorb pass plus one step per chained draft each read the head once. That is about 3
    evaluations at depth 3, so **~1.8 ms of a ~60 ms round** (W10: MTP step 1.68 ms).
  - **DFlash2 rounds**: one block pass reads the head once for its 7 positions. **~0.6 ms of 58-82 ms.**
  - **Batched MTP (4 streams)**: one pass for all slots, so the same ~1.8 ms of a ~115 ms round.
- **The cost is per rank.** Both ranks exchange candidates every draft step, so a step lasts as long as the larger
  half.
  - A global top-N list costs its larger half. In this corpus that is almost all of it: rank 1's half (ids ≥
    77,440) holds rare ids and the chat specials. At N = 32,768: 31,999 / 769.
  - So the list is built per rank, the M most frequent ids of each half (N = 2M). At equal cost that beats the global
    list (section 2).
  - Both ranks get the same row count, a multiple of 64.

## 2. Coverage

**Source**: the maintainer's local opencode database (not published; aggregates only), as `bench/lookupsim.py` reads it. It covers 826 assistant steps served by
GLM-5.3-Flash: 203 by this cluster's abliterated model, 623 by the Z.ai API. That is 385,061 reply tokens in 15
sessions, in the chat template's shape: reasoning, `</think>`, text, tool calls in GLM's `<tool_call>` markup, and
the end token.

Classes:

- `prose`: reasoning, and text outside code fences (181,559 tokens);
- `code`: fenced code, and the `content` / `oldString` / `newString` of write / edit calls (136,395);
- `tool`: the rest of a tool call, meaning markup, names, commands and paths (67,107).

The prior (ranking beyond GLM replies) is every other text of the database: user turns, tool outputs, other models'
replies. That is 3.3M tokens in 60 sessions.

**CJK**: there is no CJK in GLM's replies here (0 CJK characters in 747k characters of reasoning and text). The
`cjk` column is a stand-in: 1.9M tokens of local CJK text (zh / ja / ko gettext catalogs, CJK READMEs) through the
same list.

**Held out**: the list is built on one part and measured on the other.

- `session` split: sessions by hash, 676 / 150 steps.
- `time` split: the first 75% of steps / the last 25%.

In-sample is 100% from 12,288 rows a rank up: the corpus only has 12,443 distinct reply ids, so in-sample numbers
are not informative.

Coverage (share of reply tokens that are listed), per-rank list, ranking = GLM frequency, then prior, then id:

| rows a rank (N) | split | prose | code | tool calls | CJK | all |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 8,192 (16k) | session / time | 89.6 / 90.6 | 91.2 / 89.0 | 92.0 / 94.7 | 46.4 / 43.6 | 90.4 / 90.8 |
| 12,288 (24k) | session / time | 93.6 / 94.7 | 93.6 / 95.4 | 94.6 / 97.9 | 51.0 / 49.7 | 93.7 / 95.4 |
| 16,384 (32k) | session / time | 95.5 / 96.3 | 95.5 / 97.2 | 96.2 / 98.6 | 56.5 / 55.1 | 95.6 / 96.9 |
| **24,576 (48k)** | session / time | **98.2 / 98.5** | **97.8 / 98.8** | **98.3 / 99.2** | 64.3 / 62.9 | **98.1 / 98.7** |
| 32,768 (64k) | session / time | 98.6 / 99.1 | 98.2 / 99.3 | 98.9 / 99.5 | 67.8 / 66.8 | 98.5 / 99.2 |

Three comparisons, all in percent (`results/draftvocab/study.txt`):

- **Without the prior** (GLM frequency, then id), 16,384 a rank gives 93.2 / 93.6 all: the prior adds 2.4-3.3
  points.
- **A global top-2M list** at 16,384 gives 98.1 / 98.8, but costs 29,504 rows a rank, not 16,384. At equal cost the
  per-rank list wins: at 24,576 it gives 98.1 / 98.7, while the global list gives 98.3 / 99.0 for 31,104 rows.
- **`+ctx`** (listed, or already in the request's own history) at 24,576: 99.3 / 99.5. At 16,384: 98.6 / 98.8. That
  is the per-request extension of section 0.

**Caveats.**

- The corpus is small (15 GLM sessions, mostly this project's own agent work), so the held-out misses are mostly ids
  never seen. A broader corpus would move coverage more than any choice of M.
- The ranking ships in the patch. It lists token ids only, but its tail reflects which rare subwords appeared in
  these private transcripts. Regenerate it from other text before publishing the image
  (`bench/draftvocab.py --write-list`).

## 3. Acceptance loss and net estimate

A round's accepted drafts must all be listed. The round's last token is the target's own and need not be. Each
simulated round draws its keep from the recorded keeps:

- DFlash2 by stream class (`bench/lookupsim.py`'s `KEEPS`);
- MTP from a = 0.74 / 0.45 / 0.22 (DECODE-PLAN T1).

The keep is then cut at the first unlisted token among the accepted drafts. The session split is held out; the CJK
stand-in runs as prose.

- **Net** is tokens per ms against the full head, with the saved head ms (200 GB/s) taken off each trimmed round.
  Round model: MTP 3 drafts, a 4-row window, absorb + 2 chained steps; DFlash2 one block with a 4-row (prose) or
  8-row (code) window; 3.9 ms host.
- **fb** is the fallback rule: a request drafts over the whole vocabulary while more than 3% of the last 256 tokens
  its MTP head absorbed (prompt first) are unlisted.

**The loss is upper-leaning.** Draws ignore the text, so a rare token counts as drafted as often as a common one.
The truth lies between this and the zero-loss bound.

| arm, rows a rank | tokens a round prose (full -> list) | net prose | net code | net tool calls | net CJK | with fb: prose / code / tool / CJK |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| MTP 8,192 | 2.42 -> 2.18 | -6.0% | -5.0% | -3.9% | -38% | -0.1 / +0.1 / +0.6 / 0.0% |
| MTP 16,384 | 2.42 -> 2.31 | -0.9% | -0.7% | -0.1% | -33% | +0.1 / +0.5 / +1.2 / 0.0% |
| **MTP 24,576** | 2.42 -> 2.37 | **+1.3%** | **+0.9%** | **+1.7%** | -28% | **+1.4 / +1.2 / +1.7 / 0.0%** (20% of rounds full) |
| MTP 32,768 | 2.42 -> 2.38 | +1.3% | +0.5% | +2.0% | -26% | +1.3 / +0.9 / +2.1 / 0.0% |
| DFlash2 16,384 | 2.61 -> 2.47 | -4.2% | -10.9% | -9.7% | -38% | -0.8 / -2.9 / -2.4 / 0.0% |
| DFlash2 24,576 | 2.61 -> 2.55 | -1.1% | -5.4% | -4.6% | -34% | -0.5 / -3.7 / -2.0 / 0.0% |
| DFlash2 32,768 | 2.61 -> 2.56 | -0.9% | -4.4% | -2.9% | -31% | -0.4 / -3.1 / -1.0 / -0.1% |

- **Zero-loss bounds**: MTP 24,576 at 3 evaluations is 1.83 / 60.3 ms = +3.1%; DFlash2 is 0.61 / 58-82 ms =
  +0.7-1.0%.
- **Production mix.** Recorded rounds are about 45% MTP (W5: 907 of 1,987 drafter rounds), and lookup rounds never
  read the head. So MTP-arm-only at 24,576 comes to about **+0.5-0.8% overall at 1 stream** (the bound is ~+1.4%).
- **4 streams**: ≤ +1.6% before loss, and one slot in fallback makes the pass full. Expect 0 to +0.5%.

## 4. The patch (`draftvocab.py` + small edits)

The knobs are read at load. Both ranks compare them: the settings plus a digest of the ranking.

- **`GLM53_TF_DRAFT_VOCAB`** (list size or file; default off):
  - `""` / `0` / `off`: upstream code and graphs, bit for bit.
  - `N`: each rank takes the first N / 2 ids of its half in the shipped ranking (`glm5_next/cuda/draft_vocab.txt`:
    in the public release 118,040 ids from public text, most frequent first; the study's private list had 37,074), then the half's lowest unlisted ids, rounded up to 64.
  - `PATH`: that file's ids (whitespace separated, `#` comments, most frequent first). Both ranks get the larger
    half's count. `PATH:N` takes the first N / 2 of each half.
- **`GLM53_TF_DRAFT_VOCAB_ARMS`** (which drafters use the list): `mtp` (default) / `dflash` / `all`.
- **`GLM53_TF_DRAFT_VOCAB_FALLBACK`** (when a request goes back to the whole vocabulary): `RATE[,WINDOW]`, default
  `0.03,256`; `off` disables it.

| where | change |
| --- | --- |
| `draftvocab.py` (new) | settings / list file / per-rank ids. `take_rows` builds the trimmed 4-bit head: the listed rows' words, scales and biases, gathered from the tiled head without materializing the whole head. `attach` builds it at load from the head the drafters read. `head_for(w, arm, full)`: the head a draft pass reads. `ids_of(w, local, width)`: column -> token id, through the list when the pass was trimmed (a gather, capturable), else + the rank's offset. `track` / `full` implement the fallback rule. |
| `mtp.py` | `mtp_compute(..., full=)`: the head matmul over `head_for(w, "mtp", full)`, output `[k, head.n]` |
| `decode.py` | `sample_rows(..., draft=)`: draft logits map their columns with `ids_of` **before** the exchange and the keyed draw; the target's calls pass no `draft`, so the target path is unchanged. `Engine.sample(draft=True)` passes it (the keyword existed, unused). `Engine.mtp` picks trimmed or full per the request's rule: graphs `mtp` / `mtp_full`, long-context key + `"full"`, eager. `absorb` feeds the rule. |
| `graphs.py` | the MTP graphs over the list, plus whole-vocabulary MTP graphs (`mtp_full`) when the MTP arm is trimmed and the fallback is on |
| `batch.py` | `sample_drafts` maps with `ids_of`. `mtp_multi(..., full=)`. `MtpChains.run` feeds each slot's backlog to its rule, and one pass serves all slots, so the pass is full if any slot's rule says so (passed only when true, so stand-ins keep their signature). `Stepper` gives the slot's DFlash2 view its State. |
| `dflash2.py` | `_block_compute` reads `head_for(w, "dflash", dv_full)` and maps with `ids_of`. `capture` also captures the whole-vocabulary block when the DFlash2 arm is trimmed with a fallback. `candidates` picks the graph from the rule (`dv_st`). |
| `pfglue.py` | prefill's MTP absorbs (`cache_absorb`, the `mtp_window` skip) feed the rule |
| `overlap.py` | 0040's prefetch plan (off in production) prefetches the head the MTP pass reads |
| `engine.py` | rank check, `attach` before the drafter / engine / graphs, the load message, the lone drafter's `dv_st`, and the list's settings in the calibration key |
| `pyproject.toml` | `*.txt` in `glm5_next.cuda`'s package data (the ranking) |

**Memory**: the trimmed head is 56.6 MB a rank at 24,576 rows (37.7 MB at 16,384). The extra MTP graphs for the
fallback use the same static buffers, so they add nothing sized per head.

**The fallback rule**: a deque of the last WINDOW tokens the request's MTP head absorbed, whether each was listed.

- It is fed with the prompt at prefill and the committed tokens at each MTP absorb. Chained draft steps are not
  fed.
- It starts over when the head starts over (`mtp_len` 0).
- It reads both ranks' lists (computed from the settings), so both ranks decide alike.
- DFlash2 requests read the same window. Their MTP backlog is absorbed at MTP rounds or every 32 tokens, so the
  window lags by up to 32 tokens.
- After a session restore, the window holds what that State last saw until WINDOW new tokens pass. That affects
  drafts only.

**Bits**:

- `qmm.split_k` gives 1 K slice for the full head and for 12,288+ listed rows, so the trimmed logits are the full
  head's listed columns bit for bit (checked in the interpreter).
- Smaller lists may split K (a different sum order). That is fine, since the logits only draft.
- The load-time calibration times the trimmed MTP step. Fallback requests' steps cost the full step, a pricing error
  on those requests only.

## 5. Exactness, and the keyed-Gumbel check

**Replies.** Every committed token is `choose_rows` over the target's candidates from its own full-vocabulary rows
(`decode.sample_rows(w, logits, ...)` without `draft`, `batch.sample_multi`). The patch changes neither function's
target path. A draft is kept exactly when it equals that token. Trimming only changes which drafts are proposed, so
drafted == serial holds by construction.

**Keyed noise.**

- The target's token at position p is argmax over its candidates of logit / T + g(seed, p, token id).
- The patch maps draft columns to token ids before the exchange (`ids_of`), so a draft of token t uses the same
  g(seed, p, t) as the target. DFlash2's chain uses the target's noise on the token ids of its candidates, which
  are now mapped the same way.
- The noise is a stateless hash, with no RNG stream to advance, so draft draws can never shift the target's.
- **Greedy**: with draft logits equal to the target's, the trimmed draft is the target's token whenever that token
  is listed. Checked on 200 rows a case, both ranks.
- **Sampled**: the draft draws top-k / top-p over the listed ids. That can take in listed tail tokens outside the
  target's top-k and renormalizes the nucleus, so equality is the usual case (≥ 80% in the test, where the target's
  token is listed), not a rule.
- That is an acceptance effect, included in no estimate above (small: it needs a tail token's noise to beat the
  target's pick). It never affects a reply.

## 6. Tests (offline, this session)

- **`tests/cuda/test_draft_vocab_patches.py`: 27 passed** (CPU torch 2.14 + Triton 3.8 CPU wheel), 7 GPU tests
  skipped.
  - Host only:
    - knobs and refusals;
    - the list file format;
    - per-rank rows (order, fill, a multiple of 64, equal counts);
    - the shipped ranking (unique; end tokens and tool markup in the first 2,000);
    - the settings the ranks compare (the list's digest moves them);
    - the fallback window (thresholds, reset, both ranks alike, off).
  - Torch on the CPU, two ranks in two threads over a fake all-gather:
    - `take_rows` / `attach` (words, scales, biases, dequantized equal);
    - the target's sampling identical with and without the list: greedy, top-k / top-p, no nucleus cut, top-k 0;
      `sample_rows` and `sample_multi`; a full-width draft pass;
    - trimmed draft sampling equal on both ranks, always listed, with the greedy coupling;
    - `sample_drafts` == per-row sampling;
    - `MtpChains`: the pass is full iff any slot's rule says so.
    - **Drafted == serial** through the real `mtp_decode` / `draft` / `absorb` / `Engine.mtp` / `Engine.sample` on a
      fake model with rare-token stretches: greedy and sampled; the list with the fallback (it switches to the
      whole vocabulary in a stretch and back after), without it, and untrimmed. 3 prompts each. Rank 0 == rank 1 in
      replies, passes and drafts. No trimmed pass drafts an unlisted token.
  - Compile: `qmm._qmm` at 12,288 / 16,384 / 24,576 / 77,440 rows × 4,096 for sm_121, 16- and 32-row buckets.
- **`tests/test_draft_vocab_interpreter.py`: 3 passed.** In Triton's interpreter, the trimmed head through the real
  `qmm.matmul` == the full head's listed columns, bit for bit, for 1-16 rows.
- **Regression**: 19 existing suites plus this one (decode, batch parallel, deep verify, batch sessions, adapt,
  depth, lookup, decode overlap, overlap, knob, MIA prefill, calib, boot, batch2, decode step, prefix share, batch
  buckets, `test_glm_lookup`, `test_fastboot_logic`): **253 passed**.
  - The unpatched stack gives 226 on the same 19 suites, so the difference is this file.
  - One existing test's `mtp_multi` stand-in had 5 arguments. The patch passes `full` only when true, so it passes
    unchanged.
- **PTX**: `tests/kvpool_ptx.py --against` the stack without 0420 gives **33 / 33 kernels identical**. No Triton
  source changed; `dflash2.py`'s kernels are untouched.
- The patch applies with `git apply` on 0001-0410, and the result equals the working tree.

## 7. GPU test plan (one window; no model change)

1. **Unit tests in the image**:
   `PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q tests/cuda/test_draft_vocab_patches.py`, and
   `tests/test_draft_vocab_interpreter.py` in its own process (it sets `TRITON_INTERPRET`). Gate: all pass.
   - The GPU tests run on the synthetic checkpoint (1,024-token vocabulary, a 128-a-half list).
   - The trimmed MTP logits must equal the full head's listed columns bit for bit, graphed, including the
     `mtp_full` graphs.
   - Drafted replies must equal serial and the untrimmed engine's replies, lone and 4 batched, greedy and sampled,
     with arms `mtp` / `all` and the fallback off / forced.
2. **Control load (production config)**: record replies with non-streamed requests that return the `tensorfold`
   stats (`results/W1/ab.py` style), plus `glmbench --suites tf,kit,edit --reps 3 --long-tokens 512` and the
   concurrent 1 / 4-stream runs.
3. **Load L1** (production + `GLM53_TF_DRAFT_VOCAB=49152`): check the load message, `24576 listed token ids a
   rank ... 56.6 MB head`, and the calibration's MTP step. It should drop by ~0.5-0.6 ms from W10's ~1.68.
   - `glmbench --suites exact`: gate **10/10**.
   - `multiturn --modes batchexact`: gate **4/4**.
   - Reply sha == production (8794a3463259cc2f).
   - `glmbench --suites tf,kit,edit` (same reps).
   - Concurrent `--streams 1,4 --reps 5`.
   - The same stats requests as the control.
   - **Acceptance by position**: `python3 bench/acceptpos.py control.json L1.json` gives, per drafter, tokens a
     round and acceptance at positions 1..7 (from `depths`), with L1 - control below each row.
     - Expected: MTP position 1 at most -0.01-0.02; DFlash2 and lookup unchanged (not trimmed).
     - A CJK prompt (e.g. a Chinese chat of 1k tokens) should show MTP acceptance equal to the control: the
       fallback.
4. **Load L2, optional, to confirm section 3**: `GLM53_TF_DRAFT_VOCAB_ARMS=all`. Expected: DFlash2 tokens a round
   -2% to -6% on code / tool cells, decode lower. Only `acceptpos` and the tf cells are needed.
5. **Adopt L1** if all of these hold:
   - exact / batchexact / sha are clean;
   - 1-stream decode is ≥ +0.5% (median over the mix);
   - no cell is below -1%;
   - MTP position-1 acceptance is within -0.02;
   - 4 streams are not lower.
   Otherwise leave it off. Revert: unset the knob.

## 8. What would change the answer

- **A corpus of this model's own replies across real traffic.** The request log (0300) holds hashes, not tokens. A
  one-off token-id dump of a few days of replies (ids only) would replace the 15-session sample.
- **The per-request extension** (`+ctx`: 99.3-99.5% at 24,576). The request's history ids that are not listed would
  get rows of their own. That means a small per-request head gathered at prefill; in batch mode, one per slot, and
  the batched pass computes each slot's extra rows.
- **A cheaper head read** from other work (e.g. E2's GEMV at 215 GB/s) shrinks the saving further. MTP
  self-distillation (T1) raises acceptance at positions 2-3, where a trimmed chain loses the most, which makes each
  unlisted token cost more.
