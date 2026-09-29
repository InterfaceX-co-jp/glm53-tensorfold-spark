# RigMark: TensorFold on 2x DGX Spark

Three sets of receipts, newest first:

1. **{{AVG_RUN_LABEL}}: the 3-round averaged run on the release config** ({{AVG_RUN_DATE}}, image {{AVG_IMAGE}}):
   `{{AVG_RUN_DIRS}}`. See [below](#release-run-3-rounds-averaged).
2. **W15 validation (2026-09-29, image b7 = patches 0001-0490 + 0500 + 0540, `config/prod.env.example` of this
   release):** two full standard-suite runs, [`tensorfold-20260929-w15/`](tensorfold-20260929-w15/) and
   [`tensorfold-20260929-w15-run2/`](tensorfold-20260929-w15-run2/). See [below](#w15-validation-runs-2026-09-29).
3. **The W13 baseline (2026-09-29, image b5 = patches through 0490):** [`tensorfold-20260929/`](tensorfold-20260929/),
   the section after that.

## Release run (3 rounds, averaged)

{{AVG_RUN_SECTION: method (3 back-to-back standard-suite runs, fresh comparison IDs, prod serving, no restart), mean
and min-max of each row below, the receipts' sha256, anything that differed from W15}}

| RigMark (mean of 3 runs' medians) | TensorFold release | vLLM TP2 k=7 (Alex) |
|---|---:|---:|
| Code / prose / structured decode tok/s | {{AVG_CODE_TPS}} / {{AVG_PROSE_TPS}} / {{AVG_STRUCT_TPS}} | 44.0 / 18.9 / 64.9 |
| Cold prefill 8K / 32K / 64K tok/s | {{AVG_COLD_8K}} / {{AVG_COLD_32K}} / {{AVG_COLD_64K}} | 1,813 / 1,908 / 1,922 |
| Immediate replay 8K / 32K / 64K tok/s | {{AVG_REPLAY_8K}} / {{AVG_REPLAY_32K}} / {{AVG_REPLAY_64K}} | 1,812 / 11,046 / 11,364 |
| Replay TTFT 8K / 32K / 64K s | {{AVG_REPLAY_TTFT_8K}} / {{AVG_REPLAY_TTFT_32K}} / {{AVG_REPLAY_TTFT_64K}} | 4.52 / 2.97 / 5.77 |
| C1 / C2 / C4 aggregate tok/s | {{AVG_C1}} / {{AVG_C2}} / {{AVG_C4}} | 31.6 / 42.0 / 66.1 |
| C4 per-stream TTFT s | {{AVG_C4_TTFT}} | 0.81 |

## W15 validation runs (2026-09-29)

Production after W15 (image b7: image input on, 0540's replay snapshot and early first token, reasoning in the
`reasoning` field only), RigMark pinned at `c5a0db01b054` (clean), standard suite, `reasoning_effort` low, two new
comparison IDs (`2026-09-glm53-exl3-2xspark-tensorfold-w15-b7-v1` / `-v2`) so no earlier sweep's prompt could sit in
the NVMe session tier. Both runs valid: 15/15 basic output gates, no preflight gaps (reasoning is no longer sent twice),
every prefill row's `prompt_tokens` == its depth. Each directory: the receipt JSON + `.sha256` (run 1
`94533a0f...`, run 2 `aaeb3751...`), the card, `command.txt` (the directories were renamed after the runs, so it names
`tensorfold-20260929-141934` / `-143739`), `metadata.json`, `preflight.json`, `models.json` and `requests.jsonl`
(the server's request log of the run: token counts, cache source, timings; no text). RigMark's `run.log` is not
included. Side by side with W13 and Alex Ellis's k=7 receipt: [`../W15/rigmark-w13-w15.md`](../W15/rigmark-w13-w15.md).

| RigMark (median) | W13 b5 | W15 b7 run 1 | W15 b7 run 2 | vLLM TP2 k=7 (Alex) |
|---|---:|---:|---:|---:|
| Code / prose / structured decode tok/s | 68.6 / 43.2 / 88.2 | 67.5 / 44.0 / 89.0 | 67.2 / 42.7 / 88.7 | 44.0 / 18.9 / 64.9 |
| Cold prefill 8K / 32K / 64K tok/s | 1,598 / 1,641 / 1,621 | 1,559 / 1,635 / 1,619 | 1,564 / 1,638 / 1,619 | 1,813 / 1,908 / 1,922 |
| Immediate replay 8K / 32K / 64K tok/s | 1,597 / 3,319 / 6,304 | **38,881 / 146,067 / 257,263** | **37,117 / 142,316 / 250,472** | 1,812 / 11,046 / 11,364 |
| Replay TTFT 8K / 32K / 64K s | 5.13 / 9.87 / 10.4 | 0.21 / 0.22 / 0.25 | 0.22 / 0.23 / 0.26 | 4.52 / 2.97 / 5.77 |
| C1 / C2 / C4 aggregate tok/s | 54.3 / 65.5 / 82.2 | 54.8 / 67.0 / 81.8 | 54.3 / 67.3 / 82.8 | 31.6 / 42.0 / 66.1 |
| C4 per-stream TTFT s | 1.91 | 1.63 | 1.91 | 0.81 |

**What "immediate replay" is here, and how it was checked.** RigMark's replay rate is `prompt_tokens / TTFT` for an
identical prompt sent again right after its cold run. TensorFold keeps a snapshot of the whole model state (latent KV,
KDA recurrent state, MTP / DFlash2 context) at the last 64-token grid point strictly before a prompt's end (patch 0540),
so an identical resend resumes at n - 64 and computes only the last 64 tokens through every layer (~0.17-0.20 s), then
samples. That is prefix / session caching of an identical prompt, not a faster prefill: the cold rows above are the
prefill speed. Checks (docs/RESULTS.md W15 §6, `results/W15/verify.py`, `verify-prod.json`):

- **The request log** (`requests.jsonl`, 9 replays a run): every replay `cached` = n - 64 (8,128 / 32,704 / 65,472),
  `cache_src` `slot`, one 64-row piece; the cold runs `cached` 0.
- **Same output as computing it:** each replay's 8-token output sha equals its cold run's in RigMark's own receipts
  (9/9 in both runs). Separately, prompts of exactly 8,192 / 32,768 / 65,536 token ids the server had never seen
  (`cached` 0), 64 greedy tokens with `ignore_eos`: the replay and a second replay returned **byte-identical 64
  tokens** at all three depths.
- **Negative controls:** variants with one token changed at the start, the middle, n - 10 and n - 100, and a
  different prompt of the same length. `cached` never exceeded the prefix the variant shares with anything stored
  (start / different prompt: 0 and a full cold prefill; middle: the last 16,384 session mark before the change;
  n - 100: below the change). Every variant's output differs from the base's except one (64K, middle: a word
  changed 31k tokens before the end did not change the next 64 greedy tokens; its log shows 32,768 tokens recomputed).
- **Why vLLM replays at ~11k tok/s on this model:** GLM-5.3-Flash is a hybrid (KDA linear attention + MLA). vLLM's
  prefix cache can only restore the recurrent KDA state at page-aligned checkpoints (`mamba_cache_mode = "align"`),
  caps a hit at n - 1 rounded down to a whole block, never prefix-caches the DSA indexer's scratch, and drops the last
  matched page per cache group with a drafter. A replay therefore restarts several thousand tokens before the end
  (Alex's receipts imply ~8,200 / ~5,700 / ~11,100 recomputed tokens at 8K / 32K / 64K) and at 8K reuses nothing.
  This is vLLM's stock hybrid caching, not a TensorFold shortcut.

Against W13: replay 24x / 44x / 41x faster, reproduced within 5% by run 2; cold 8K -2% (0540 adds one 64-row chunk to a
grid-aligned cold prompt; 32K / 64K equal); decode and aggregates within run-to-run noise. C4 per-stream TTFT is
unchanged within noise: 0540 hands a piece's first token over when the piece ends, but with thinking on (RigMark's
low effort) the first token carries no visible text, so the first visible delta still waits for the round
(docs/RESULTS.md W15 §2; patch 0560, multi-slot prefill, is the fix in progress). Reasoning characters are now
single-counted (60, as vLLM's 60).

## W13 baseline (2026-09-29)

[RigMark](https://github.com/alexellis/rigmark) by Alex Ellis, pinned revision `c5a0db0` (protocol 1.1.0), **standard
suite, unmodified settings**, with the same `{"chat_template_kwargs":{"reasoning_effort":"low"}}` request body Alex
uses for his published GLM-5.3 runs. Run from the head node against `127.0.0.1:8000` (loopback, no proxy, no other
traffic during the run).

**This is a baseline.** We think we can do better with more optimizations over the coming days, and will rerun RigMark
and publish the new receipt here when we do.

- Receipt (the content-hashed result JSON): [`tensorfold-20260929/glm53-flash-exl3-tensorfold-tp2-low.json`](tensorfold-20260929/glm53-flash-exl3-tensorfold-tp2-low.json)
  (`sha256sum -c` against the `.sha256` file; `./rigmark report --save <json>` regenerates the card byte-identically)
- Card: [`tensorfold-20260929/glm53-flash-exl3-tensorfold-tp2-low.card.txt`](tensorfold-20260929/glm53-flash-exl3-tensorfold-tp2-low.card.txt)
- Command: [`tensorfold-20260929/command.txt`](tensorfold-20260929/command.txt)
- Side-by-side with Alex Ellis's two published vLLM GLM-5.3 TP2 receipts: [`compare-published-20260929.md`](compare-published-20260929.md)

```
15/15 BASIC OUTPUT GATES PASSED | GLM-5.3-Flash-EXL3 | 2x DGX Spark | reasoning=low | protocol 1.1.0 | git:c5a0db01b054 clean
CODE 68.6 tok/s 26.3s last (66.5–69.3) 5/5 | PROSE 43.2 tok/s 24.1s last (43.0–43.5) 5/5 | STRUCTURED* 88.2 tok/s 8.2s last (87.3–89.1) 5/5
64K PREFILL cold 1,621 • replay 6,304 tok/s | C1 54.3 • C2 65.5 • C4 82.2 tok/s | C4 normal stop 0/12, visible 12/12 | sha256:4af01c23364e22b6…
```

| RigMark (median) | TensorFold (this repo) | vLLM TP2 k=7 (Alex) | vLLM TP2 adaptive (Alex) |
|---|---:|---:|---:|
| Code decode tok/s | **68.6** | 44.0 | 42.6 |
| Prose decode tok/s | **43.2** | 18.9 | 22.2 |
| Structured decode tok/s | **88.2** | 64.9 | 54.6 |
| Code / prose time to last output (s) | **26.3 / 24.1** | 46.6 / 55.7 | 50.2 / 45.4 |
| Cold prefill 8K / 32K / 64K tok/s | 1,598 / 1,641 / 1,621 | **1,813 / 1,908 / 1,922** | 1,835 / 1,898 / 1,905 |
| Immediate replay 32K / 64K tok/s | 3,319 / 6,304 | **11,046 / 11,364** | 11,339 / 11,464 |
| C1 / C2 / C4 aggregate tok/s | **54.3 / 65.5 / 82.2** | 31.6 / 42.0 / 66.1 | 31.2 / 42.8 / 61.1 |
| C4 per-stream TTFT (s) | 1.9 | **0.8** | 0.8 |

### Read this before comparing

- Not a strict RigMark comparison (RigMark refuses one across different receipts/protocols; the table uses
  `--allow-mismatch`): different weights (abliterated EXL3 4-bit here vs LibertAIDAI NVFP4), drafter policy, context
  limit (1M vs 262k), protocol (1.1.0 vs 1.0.0), day and machines.
- Structured output: our model emits pretty-printed JSON (687 tokens) vs 441 compact tokens in Alex's runs, so that
  row compares different outputs (all 5/5 valid in every run).
- Our reasoning text is emitted in both `reasoning` and `reasoning_content`, which RigMark sums; reasoning character
  counts in our receipt are doubled (timings and token counts are unaffected). Fixed in a later config.
- Where we lose: cold prefill (12-16% behind), immediate replay (our snapshot grid made identical-prompt replays resume
  from the last 16k mark, not the end) and C4 time to first token (concurrent prompts admitted one prefill piece at a
  time). Fixes for replay and C4 TTFT are in progress.
- The receipt's `competing_traffic` string says "test window held": that text came from our run script; the endpoint
  was idle and nothing else ran, but no maintenance window was held.

### Recipe

The server was this repo's production config (`config/prod.env.example`) on image `b5`: TensorFold v0.3.4 (`2f8e514`)
plus this repo's patches through 0490. Patches 0420-0490 (L2 weight prefetch, OpenAI-style context errors,
`/tokenize` for RigMark's prefill phase, and others) are in the repo since the 2026-09-30 update. To reproduce:
`docs/RIGMARK.md` and `scripts/rigmark/` (`install.sh`, then `run.sh tensorfold` on the head node). The directory
also holds the run's `metadata.json`, `preflight.json`, `models.json` and `requests.jsonl` (request log, no text).
