# RigMark baseline: TensorFold on 2x DGX Spark (2026-09-29)

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

## Read this before comparing

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

## Recipe

The server was this repo's production config (`config/prod.env.example`) on image `b5`: TensorFold v0.3.4 (`2f8e514`)
plus this repo's patches through 0490. Patches 0420-0490 (L2 weight prefetch, OpenAI-style context errors,
`/tokenize` for RigMark's prefill phase, and others) land in the next update PR together with image input; until then
`main` has patches through 0410, so RigMark's prefill phase needs that PR's 0490 to run against a server built from
`main`. To reproduce: `docs/RIGMARK.md` / `scripts/rigmark/` in that update.
