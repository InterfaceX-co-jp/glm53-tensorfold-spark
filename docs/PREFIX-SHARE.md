# Shared-prefix reuse across sessions (patches/0310, `GLM53_TF_PREFIX_SHARE`) and the request log (patches/0300)

Status (2026-09-28): written and tested offline (host and CPU torch tests, no GPU). Both patches are off by default,
so production is unchanged. The GPU test plan is at the end.

## The question

Take a NEW session whose prompt starts with a long prefix that another session already prefilled. The typical case
is the same 10-20k-token agent system prompt plus tools. Can it skip prefilling that prefix?

## What hits today (0110 / 0180 / 0250 / 0290, production settings)

The session store resumes a request from **the longest stored entry whose tokens are a strict prefix of its prompt**
(`SessionIndex.find`, then the NVMe tier's copy of it). Other conditions: the same prefill mode (snapshot tag: fast 64
vs exact 0, FP8 / TC / b12x bits), and draft caches that fit the request (MTP rows, the DFlash2 window). An entry is a
snapshot plus its rows. Attention rows alone are not enough: the 36 KDA layers need their **recurrent state at that
exact position**, which only a snapshot holds. So a request can resume only where some earlier prefill took a
snapshot. Snapshots are taken at:

1. the prompt snapshot: exact prefills at the prompt end; fast prefills (production) at the prompt's last multiple of
   64 (or up to 256 earlier, `pfgrid`'s tail rule);
2. the reply snapshot (exact requests only);
3. `GLM53_TF_SESSION_EVERY` marks: every 16,384 tokens of a prompt, past the resume point by at least 512;
4. **fork marks**: at the page (exact) or 64-point (fast) at or before the longest common prefix with a stored entry
   (RAM or NVMe) or with a prompt in flight (0180 fix). A fork mark must be at least `GLM53_TF_SESSION_FORK_MIN` = 512
   past the resume point.

So, for sessions S1, S2, S3, ... whose prompts are `system (L tokens) + <|user|> + own task`:

| case | S1 | S2 | S3 and later |
| --- | --- | --- | --- |
| L < 16,384 (e.g. a 12k system prompt), one after another | cold | **cold: prefills all L** (nothing stored ends inside the shared part; S2 takes the fork mark) | resume at floor64(lcp) ~ L |
| 16,384 <= L < 32,768 (e.g. 20k) | cold | resumes at 16,384 (S1's EVERY mark), prefills L - 16,384 again | resume ~ L |
| N sessions arriving together (subagent burst) | cold | **all N prefill all L side by side** (each takes the same fork mark; the 2nd..Nth saves are duplicates) | the next session after the burst resumes ~ L |
| system prompt differs per session after token X (a date or cwd at its end) | cold | cold (or the EVERY mark below X) | resume at floor64(X) (the fork mark is at the lcp, role tokens play no part) |
| a different prefill mode (a `tf_knobs.fast_prefill=0` request) or drafter set | never shares with the other mode's entries | | |
| the entry evicted from the 2 GiB RAM store | still found on NVMe (write-through, 64 GiB) at ~0.15-0.3 s | | |

The gap is the **first sharer of every distinct system prompt**, and every session of a burst. The 12k case never
hits for S2 because of the marks' granularity. The fork mark exists, but only the session that *discovers* the fork
takes it, and that session has already paid for the prefill.

## What 0310 adds

Two mechanisms, both under `GLM53_TF_PREFIX_SHARE=1`.

**1. System-prompt marks** (`sessions.PrefixShare`, `SessionStore.plan`). Rank 0 adds one more mark to every
request's plan: the position of the prompt's first `<|user|>` / `<|assistant|>` / `<|observation|>` token. Everything
before that token is `[gMASK]<sop><|system|>` + system prompt + tools. The mark is rounded down to the snapshot grid
like every fork mark: the page for exact prefills, 64 for fast ones, so a fast request loses at most 63 tokens. It is
added only when:

- it lies in [`GLM53_TF_PREFIX_SHARE_MIN` (2,048), `GLM53_TF_PREFIX_SHARE_MAX` (131,072)];
- it is at least `fork_min` past the request's resume point;
- it is before the prompt's own snapshot;
- no entry for it exists already, in RAM or on disk.

So S1 leaves the snapshot that S2 resumes at. Each distinct system prompt costs one snapshot, taken once.

**2. Waiting for a partner's mark** (`Batcher._share_wait`, `GLM53_TF_PREFIX_SHARE_WAIT=1`). Rank 0 admits requests
in `_plan`. A waiting request is held back for one round if a request in flight (still prefilling, or admitted
earlier in this round) meets all of these:

- it has the same prefill mode;
- its drafters cover this request's;
- it has a planned mark q it has not reached yet;
- q lies inside the prefix the two prompts share;
- q is at least `fork_min` beyond what this request could resume now (any slot's snapshot, the RAM store, the disk).

After the partner's piece passes q, the mark is saved, and the next round's plan restores it. A burst of N subagents
then prefills the shared prefix once, not N times.

The wait never outlives its reason:

- it ends when the partner passes the mark (stored or skipped), finishes, is cancelled or is pre-empted;
- the check re-runs every round;
- other waiting requests are admitted meanwhile, so it does not block the queue.

Only rank 0 decides it. Rank 1 just runs the rounds, so the protocol is unchanged.

**Bounds.**

- System-prompt marks are tagged `kind == "prefix"` (rank 0's bookkeeping, from its own plan).
- At most `GLM53_TF_PREFIX_SHARE_KEEP` (4) stay in the RAM store. Rank 0 evicts the least recently used one first; rank
  1 replays that eviction like any other. With write-through they stay on NVMe.
- Every other entry is under the store's LRU as before.
- A request adds at most `GLM53_TF_PREFIX_SHARE_POINTS` (1) mark.

**Exactness.** Nothing new is computed:

- A system-prompt mark is a mark. `decode._prefill` cuts an exact chunk there (no bit changes, as for every fork mark)
  or takes a fast snapshot on the 64-grid (0085: the state at a multiple of 64 is what every fast prefill of a prompt
  extending it holds there).
- The restore is 0110's copy-in (0290: into the slot's pool pages).
- The wait is scheduling only.

So resumed == fresh and batched == alone, as for every other session resume. The CPU tests check it on 0180's hostile
fake model: every reply and every slot's whole end state against a fresh prefill + serial decode.

## Alternatives considered

- **Finer early marks** (every 1-2k tokens of the first 64k). Each mark is a full KDA snapshot. At 2k spacing that is
  32 snapshots x 94 MB = 3 GB a rank for one new 64k prompt, more than the whole 2 GiB RAM store. Each mark also cuts
  a prefill chunk. A capped version (a few marks) does not know where the fork will be. The system-prompt boundary is
  where agent sessions fork, and costs one snapshot. If the request log shows system prompts that differ near their
  end, `GLM53_TF_SESSION_EVERY` (today 16,384) can be lowered, knowing the cost: 94 MB a mark per long new prompt.
- **Copy-free refcounted pool pages** (KV-POOL stage 2). This saves the restore's copy of the prefix rows: 12k tokens
  x 7.4 KB = 89 MB a rank, ~1 ms at device bandwidth. It also saves the pool memory of a prefix duplicated across
  slots, at most 4 x 12k x 7.4 KB = 0.36 GiB of the 7.44 GiB pool. It does not save prefill: the KDA state still
  needs the snapshot, which is what 0310 adds. It is a large change touching every writer (copy-on-write of the shared
  last page, refcounts across spills and the NVMe loader) for ~1 ms a resume, so it stays deferred.
- **Retroactive snapshots.** Recompute the KDA state at a fork point from the attention rows: impossible (the KDA
  recurrence reads its own previous state, not the cache).

## Memory

Numbers are per rank (both ranks hold the same entries). Snapshot, from 0110's measurements:

| part | size |
| --- | ---: |
| KDA recurrent state | 71.3 MB |
| conv windows (with 0065's index-ring tail) | 2.6 MB |
| DFlash2 window | ~20 MB |
| pending MTP rows | < 1 MB |
| **total** | **~94 MB** (0.09 GiB) |

A system-prompt entry adds that snapshot plus at most a page (256 tokens x 7.4 KB = 1.9 MB) of private tail. Its full
pages are the same page keys as the prompt entries' (reference counted), so they cost nothing more.

| | RAM store (2 GiB a rank in prod) | NVMe tier (64 GiB) |
| --- | --- | --- |
| one distinct system prompt | +94 MB, once | +94 MB snapshot file, once |
| cap (`KEEP` = 4) | <= 0.37 GiB (4 distinct system prompts at once; the 5th evicts the oldest) | LRU with everything else |
| a burst of N sessions without the wait | N-1 duplicate snapshot copies are made and dropped (transient) | |

No new device memory outside the store's budget, no new pool pages (a restore maps the slot's own pages, which its
reservation covers), and no change to the snapshots the slots keep.

## Expected savings

Prefill runs at ~1,250-1,280 tok/s at 24-98k (RESULTS W5 / W6) and a little faster on short prompts. Time is taken
as linear in tokens here.

| traffic | today | with 0310 |
| --- | --- | --- |
| a 2nd session over a 12k system prompt | prefills 12k + task: ~9.5 s + task | resumes at ~12k (restore ~0.02 s RAM / ~0.1 s NVMe): **-9.5 s TTFT** |
| a 2nd session over a 20k system prompt | resumes at 16,384, prefills ~3.6k + task: ~2.9 s + task | resumes at ~20k: **-2.9 s** |
| 4 subagents at once, 12k shared + 2k own each | 4 x 14k = 56k tokens side by side: ~44 s of prefill, every first token ~40-44 s | 12k once, then 4 x 2k: 20k tokens, ~16 s; first tokens at ~11-16 s: **-28 s of GPU prefill, TTFT ~-28 s** |
| 3rd+ session, next turns of a session | resumes | unchanged |
| requests without role tokens before `MIN` (raw completions, short system prompts) | | unchanged (no mark) |

How much this is worth in production depends on how often new sessions start relative to follow-up turns. The
request log (0300) measures exactly that. `scripts/traffic-report.py`'s *avoidable prefill / other_conversation*
line is the share of prefill time a snapshot at every cross-session fork point would have saved. That is the upper
bound of what 0310 can recover. The sessions whose fork is the system-prompt boundary are 0310's share of it.

## The request log (0300)

`GLM53_TF_REQUEST_LOG=/sessions/requests.jsonl` (on the `/sessions` mount; rank 0 only) writes one JSON line per
request. The fields (`reqlog.py` docstring):

- sizes: prompt, cached, and `cache_src` (slot / ram / disk / none);
- timings: prefill s and tok/s, queue wait, first delta, decode tokens / s / tok/s, tokens a round;
- KV pool: pages reserved and free;
- the marks the prefill took, effort / thinking, max_tokens, and the finish reason;
- prefix fields: `head_hash` (the first 4,096 token ids), `sys_len` / `sys_hash` (up to the first role token), `conv`
  (a conversation key: the prompt through its first `<|assistant|>`);
- `lcp` / `lcp_same` / `lcp_other`: the longest common prefix with the last 64 prompts, with the same conversation and
  with others.

No text is logged; `GLM53_TF_REQUEST_LOG_SALT` keys the hashes. It is rotated at 64 MiB with 3 old files kept. All
the work is on the request's HTTP thread after the reply: ~1-20 ms of host time at 100k tokens (the 64-prompt
comparison), no GPU sync, nothing on the engine loop.

`python3 scripts/traffic-report.py /sessions/requests.jsonl` prints:

- volume and finish reasons;
- size percentiles / histograms;
- prefill / decode speed;
- the reuse rate today by source;
- avoidable prefill by kind (other conversation / same conversation / either);
- the share of new conversations that share a prefix;
- the top system-prompt and 4k-head hashes.

`--json` gives the same data as JSON.

## Knobs

| knob | default | meaning |
| --- | --- | --- |
| `GLM53_TF_PREFIX_SHARE` | `0` | `1`: system-prompt marks and the wait |
| `GLM53_TF_PREFIX_SHARE_MIN` / `_MAX` | `2048` / `131072` | the boundary must lie in this range (tokens) |
| `GLM53_TF_PREFIX_SHARE_POINTS` | `1` | role boundaries marked a prompt (2: also the end of the first user message) |
| `GLM53_TF_PREFIX_SHARE_KEEP` | `4` | system-prompt entries kept in RAM (0: no cap) |
| `GLM53_TF_PREFIX_SHARE_WAIT` | `1` | batch admissions wait for an in-flight partner's mark |
| `GLM53_TF_PREFIX_SHARE_TOKENS` | from the tokenizer | role token ids (comma-separated), overriding `tokenizer_config.json` / `tokenizer.json` |
| `GLM53_TF_REQUEST_LOG` | unset (off) | the request log's file |
| `GLM53_TF_REQUEST_LOG_MB` / `_KEEP` / `_PROMPTS` / `_SALT` | `64` / `3` / `64` / none | rotation size, old files, prompts compared, hash key |

Both knob families are left out of 0250's compat hash (they never change a stored bit), so turning them on keeps
the NVMe sessions. The ranks need not agree on `PREFIX_SHARE*`: marks and waits are rank 0's decisions, sent in the
plans as before.

## Risks

1. **Template assumptions.** The boundary is the first role token. GLM-5.3's template puts tools and system text
   before the first `<|user|>`, so the boundary is where agents fork.
   - If a client puts per-session text in the system prompt (a date, a cwd), the boundary mark is per session. It is
     then useless but cheap: one 94 MB snapshot per new session, capped at 4 in RAM. The log's `sys_hash` shows it
     (as many hashes as conversations).
   - A template that renders the effort line into the system block gives each effort level its own system prefix.
2. **Snapshot churn on NVMe.** Each new distinct system prompt writes one more ~94 MB entry file (write-through):
   negligible next to the per-turn entries.
3. **The wait adds latency when the partner is slow to reach its mark.** For example, the partner is a background
   request that keeps stepping aside, or it prefills a much longer prompt while this one would have finished its
   own prefix first.
   - It never waits on a request that is not prefilling, and it re-checks every round.
   - A background partner that is pre-empted leaves the queue's head, which ends the wait.
   - Worst case: this request's first token comes when the partner's mark does. Its own prefill is then a few pieces,
     not the whole prefix.
4. **Duplicate prefill when the wait is off** (`WAIT=0`): the burst case stays as today, apart from later sessions.
5. **GPU-only unknowns.** None expected: no kernel, no arithmetic, no new buffer. The paths are 0110 / 0180 marks and
   restores, which the GPU has run since W1-W6. The FP8 / pool restores are 0290's (W6).

## GPU test plan

Before the window (no GPU):

1. Build the image with every patch through 0310 (0300 and 0310 each also apply alone on 0290, and under 0320):
   `IMAGE=glm53-tensorfold:prefix CONFIG=config/prod.env scripts/serve.sh build`. The CPU tests in the image:
   `tests/test_request_log.py`, `tests/cuda/test_prefix_share_patches.py` (host parts), plus 0180 / 0250 / 0290's
   host parts.
2. Optional, on production as it runs (no restart): nothing. 0300 needs a restart to turn on.

In the window (`R=results/W7p`, `B=http://127.0.0.1:8000`, `M=GLM-5.3-Flash-EXL3`):

| t (min) | step | pass gate |
| ---: | --- | --- |
| 0 | Stop production; lease; `nvidia-smi` clean on both nodes. | |
| 2 | GPU tests, the head node: `tests/cuda/test_prefix_share_patches.py` (GPU part: a new session resumes at the system end, exact and fast, greedy and sampled; a burst of 4 waits and resumes; a follower replays the same decisions). the worker node in parallel: `test_batch_sessions_patches.py`, `test_session_disk_patches.py`, `test_kv_pool_patches.py` (regressions). Commands as in KV-POOL.md step 2 with these files. ~10 min. | all prefix-share tests pass; the others' counts as in W6 |
| 12 | **Load A** (control): production + `GLM53_TF_REQUEST_LOG=/sessions/requests.jsonl` on image `prefix` (0310 present, off). Boot line `request log (patches/0300)`. | boot as production |
| 15 | **Same bits** (A): `bench/glmbench.py --suites exact`, `bench/multiturn.py --modes batchexact`. | 10/10, 4/4 |
| 19 | **Today's reuse** (A): `python3 bench/prefixshare.py --base $B --model $M --system 12000 --user 1500 --out $R/prefixA-12k.json`, then `--system 20000 --out $R/prefixA-20k.json`. | sequential: session 2 `cached` 0 (12k) / 16,384 (20k), session 3 ~ system prompt; burst: all `cached` 0; exact all `true` |
| 27 | **Load B**: A + `GLM53_TF_PREFIX_SHARE=1`. Boot line `shared prefixes (patches/0310): ... role tokens [<3 ids>]`. | the ids are GLM-5.3's `<|user|>`, `<|assistant|>`, `<|observation|>` (tokenizer_config.json) |
| 30 | **Same bits** (B): as step 15. | 10/10, 4/4 |
| 34 | **Prefix reuse** (B): `prefixshare.py` as step 19 into `$R/prefixB-*.json`. | sequential: sessions 2 and 3 `cached` ~ system prompt (within 64 + template tokens), session 2's TTFT -8 to -10 s (12k) / -2 to -3 s (20k) against A's; burst: bursts 2-4 `prefix_wait` > 0 and `cached` ~ system prompt, wall well below A's; exact all `true` |
| 40 | **Log**: `python3 scripts/traffic-report.py <head sessions dir>/requests.jsonl --since <load B time>` and the same for A's span. | a line a request; B: `cache_src` ram for sessions 2+, avoidable *other_conversation* ~0; A: > 0 |
| 45 | **Restore production** (with or without the switches, per the results), https check, watchdog re-armed, lease deleted. | |

Keep the log on in production afterwards (it only writes lines). A day of real traffic through
`scripts/traffic-report.py` says whether `PREFIX_SHARE` pays: the *other_conversation* avoidable prefill share and the
*new conversations sharing a prefix* rate.
