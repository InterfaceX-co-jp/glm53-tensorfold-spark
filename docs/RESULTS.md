# Results

> **Work in progress.** Measured on one pair of DGX Sparks, 2026-09-27 to 2026-09-29, with the **abliterated** checkpoint
> below. The sections are in the order the work happened; the newest ones are the current state: **W10** (the end of
> this file) for the production config (`config/prod.env.example`: 4 requests sharing a 1M-token KV pool), "Stacked run
> and production config" for the older single-stream config. The public repo carries the benchmark JSON and the
> window scripts of the runs in `results/` (E*, F*, L*, M*, P*, Q*, A1, B1, B2, S1, X1, Y1, Z1, Z2, W1-W10, roofline,
> sim0340, sim0380); logs (`*.log`, `*.out`, `*.err`), nsys traces, test output of the early runs and some runs (B3,
> K0, K1, P1, T*) are summarized here only. Hosts in the JSON were normalized to `127.0.0.1` / `<worker-ssh>`, node
> names to head / worker, and paths in the window scripts to `$HOME/glm53-tensorfold-spark`. Config names refer to
> the `config/*.env.example` files; `results/W9/prod.env.before-W9` and `results/W10/prod.env.before-W10` are the
> production config before each window, with the same placeholders.

GLM-5.3-Flash abliterated EXL3 (`neko-legends/GLM-5.3-Flash-Uncensored-EXL3` @ `07135ec0`) on two DGX Sparks
(GB10, TP=2 over the 200 Gb/s CX7 link). Three stacks on the same pair, the same weights and the same client:

| Stack | What it is |
| --- | --- |
| vLLM prod kit | [Reederey87/glm53-flash-exl3-2x-dgx-spark](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark) @ `8e443d6` (2026-09-20; a fork of [MiaAI-Lab's kit](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)) as we ran it in production on this pair: vLLM, 1M context, FP8 KV, DFlash2 drafter k=7 with adaptive k (EMA), fused EXL3 MoE kernels, the same abliterated weights |
| TensorFold upstream | `vendor/TensorFold` @ `2f8e514` (0.3.4) with `GLM53_TF_NONEXPERT=bf16`, i.e. upstream behaviour; drafter policy `auto` (upstream `EXL3_AUTO`) |
| TensorFold + patches | the same engine with `patches/0001`-`0004`, `GLM53_TF_NONEXPERT=q4mse`, drafter policy `auto` |

Both TensorFold stacks load the same DFlash2 drafter revision as the vLLM kit (`incoai/GLM-5.3-Flash-DFlash2` @
`7d74cdd`) next to the checkpoint's own MTP layer.

## Decode (tok/s, median of 5, single stream, thinking off)

| Cell | vLLM prod kit | TF upstream (bf16) | TF + patches (q4mse) | vs vLLM | vs upstream |
| --- | ---: | ---: | ---: | ---: | ---: |
| tf code, sampled, 64 tok | 35.2 | 32.3 | **44.9** | 1.28x | 1.39x |
| tf chat, sampled, 64 tok | 27.1 | 30.4 | **41.7** | 1.54x | 1.37x |
| tf code, greedy, 64 tok | 41.9 | 46.4 | **61.0** | 1.46x | 1.31x |
| tf chat, greedy, 64 tok | 22.8 | 28.7 | **44.3** | 1.94x | 1.54x |
| sequence, 512 tok | 67.5 | 61.8 | **80.6** | 1.19x | 1.30x |
| code, 512 tok | 42.4 | 44.3 | **58.4** | 1.38x | 1.32x |
| json, 512 tok | 50.4 | 55.9 | **68.5** | 1.36x | 1.23x |
| kit hashmap, 200 tok | 30.0 | 36.4 | **49.1** | 1.64x | 1.35x |
| kit structured, 200 tok | 72.7 | 65.1 | **85.9** | 1.18x | 1.32x |
| kit essay, 200 tok | 26.1 | 32.7 | **43.9** | 1.68x | 1.34x |

Serial decode (no drafts, `"draft": false`): 17 tok/s with BF16 non-expert weights, 33 tok/s with q4mse. The
one-row verify step went from 57 ms to 30 ms; that is where most of the drafted gain comes from.

## Prefill and long context

| Stack | Prefill tok/s |
| --- | --- |
| vLLM prod kit | 960 @ 1.8k, 1340 @ 7k, 1448 @ 28k prompt tokens |
| TF upstream (bf16, 64-row chunks) | 256-266; prompts past 2,051 tokens get HTTP 400 unless `--context` is set |
| TF + patches (q4mse) | 404-420 over the measured prompt sizes |

Prefill tok/s = prompt tokens / cold time to first token (unique prompt prefix, so no cache hit). Prefill is
where vLLM stays well ahead (2.3-3.6x); see `docs/PREFILL-ANALYSIS.md` for the chunk-size work.

Decode behind a ~28k-token prompt: vLLM 65.4 tok/s, TF + patches 70.8 tok/s.

## Exactness and quality

| Check | TF upstream (bf16) | TF + patches (q4mse) | vLLM prod kit |
| --- | --- | --- | --- |
| drafted == serial, byte-identical (10 cases) | 10/10 | 10/10 | n/a |
| MMLU-200, greedy, thinking off | 87.0% | 88.0% (13/200 answers differ from bf16) | pending |
| refusals (10 prompts) | 0/10 | 0/10 | 0/10 |

"drafted == serial" means that for every request the drafted reply is the same bytes as the one-token-a-round
reply of the same engine and weights. q4mse is a different set of weights from bf16 (the non-expert matrices are
re-quantized), so bf16 and q4mse replies differ from each other; the MMLU and refusal rows are the check that
the re-quantization does not cost quality.

## Methodology

- Client: `bench/glmbench.py` (standard library only), the same script against every stack, from the head node.
  Every request streams through the OpenAI API; decode tok/s = `(completion_tokens - 1) / (last content chunk -
  first content chunk)`, so prefill and time to first token are excluded.
- Each cell: one short warm-up request, then 5 measured requests; the table reports the median.
- `tf` suite (TensorFold's published cells): 64-token replies with `ignore_eos`; `code` is a raw completion,
  `chat` is chat with thinking off. Sampled = temperature 1, top-k 20, top-p 0.95, seeds 1234-1238; greedy =
  temperature 0.
- `tweet` suite (sequence / code / json): chat, thinking off, greedy, 512-token replies.
- `kit` suite (hashmap / structured / essay): the vLLM kit's own decode prompts verbatim, chat, thinking off,
  greedy, 200 tokens.
- `ctx` suite: a unique tag plus filler text sized for ~2k / 8k / 32k tokens (1.8k / 7k / 28k actual prompt
  tokens), then a short question and a 256-token greedy reply; cold and warm runs.
- `exact` suite: 5 prompts x (greedy, sampled seed 1234), 128 tokens each, drafted vs `"draft": false`,
  compared by SHA-256 of the reply text.
- Quality: `bench/quality.py`, MMLU 200 questions stratified over the 57 subjects with a fixed seed
  (`bench/data/mmlu200.jsonl`), greedy, thinking off, first A-D letter of the reply; plus 10 prompts a
  safety-tuned model tends to decline, counting replies that open with a refusal.
- One request at a time, nothing else on the GPUs. Only one stack runs at a time (`scripts/serve.sh start`
  refuses to start while another CUDA process is up on either node).

Commands:

```bash
python3 bench/glmbench.py --base http://127.0.0.1:8080 --model GLM-5.3-Flash-Uncensored \
    --suites tf,tweet,kit,ctx,exact --label tf-q4mse --out results/tf-q4mse.json
python3 bench/quality.py --base http://127.0.0.1:8080 --model GLM-5.3-Flash-Uncensored \
    --label tf-q4mse --out results/Q-tf-q4mse.json
```

## Drafter policy sweep

_Placeholder._ Per-policy decode (MTP-only, DFlash2-only, `auto`, draft lengths) with q4mse non-expert weights,
same cells as above. To be filled in.

## Stacked run and production config, 2026-09-27/28

One image with every committed patch (0001-0180; 0080 fast2 experts, 0083 FP8 tile 128,64,4,2, 0085 chunk-size-independent
prefill, 0110 sessions, 0120 batching, 0130 decode kernels, 0140 fast boot, 0150 health/metrics, 0160 OpenAI compat,
0170 fat experts / BF16 KDA copy, 0180 batch sessions). Load A (`results/A1/`): `CONTEXT=262144`, q4mse, latent KV,
fast + lean prefill (block 1024, rows `auto`, max 8192), overlap, bf16 gathers, expert loop, real calibration, cost
depths, lookup, `SESSION_GIB=12`. Production (`config/prod-single.env.example`, `results/S1/`): the same with `CONTEXT=524288`,
`GLM53_TF_FAST_EXPERTS=fat`, cached calibration, decode kernels off, FP8 off, on :8000 as `GLM-5.3-Flash-EXL3`.

| | earlier best (F9/F10) | Load A, FP8 off | Load A, FP8 on | **production** | vLLM prod kit |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefill 7k (tok/s, cold prompt, warm kernels) | 1,024 (FP8 on) | 1,084-1,122 | - | **1,162** | 1,340 |
| prefill 28k | 1,034 (FP8 on) | 1,062 | 1,177 | **1,209** | 1,448 |
| prefill 112k | 1,013 (FP8 on) | 1,105 | 1,186 | **1,162** | - |
| routed experts in a 28k / 112k prompt (s, rank 0) | 10.0 / 39.3 | 5.9 / 23.1 | 6.1 / 25.8 | | |
| follow-up: 34.8k conversation + reply + 2.2k new tokens (TTFT) | 3.6-6.8 s | 2.9 s | | 3.3 s | |
| session revisit, ~37-39k tokens (A,B,A,C,B,A) | ~36 s (re-prefill) | 0.45-0.54 s | | 0.38-1.01 s | |
| decode tf code greedy / chat greedy / kit structured / tweet sequence | 65.7 / 41.6 / 95.8 / - | 64.3 / 47.6 / 96.7 / 90.1 | | 65.8 / 44.7 / 98.1 / 93.3 | 41.9 / 22.8 / 72.7 / 67.5 |
| drafted == serial (`exact`) | 10/10 | 10/10 | | 10/10 | |
| MMLU-200 / refusals | 89.0% / 0 | 89.5% / 0 | 88.5% (Q5-1) | 89.5% / 0 | |
| tool calls, opencode toolset, T 0 (clean / corrupt) | | 200/210 / 0 (thinking off) | 200/210 / 0 | 95/105 / 0 (thinking on) | |
| load time | ~6-8 min | 490 s | | **34-37 s** (prepared folders) | |

Prefill tok/s = prompt tokens / cold TTFT on a unique prompt, after the kernels compiled (production: the canary's
4k / 16k warm-up at start does that). At 28k vLLM is still ~20% ahead on prefill; decode is 1.3-2x vLLM.

**Final production (2026-09-28 01:55, image `glm53-tensorfold:z` = patches through 0210, `config/prod-single.env.example`)**: the
column above plus 0190's `GLM53_TF_MOE_GLUE=5` (parallel MoE grouping + in-place combine; same bits) and 0210's
prompt-token cache (on by default; `verify` mode showed no difference). Measured on that load (`results/Z1/`):

| | production (final) | vLLM prod kit |
| --- | ---: | ---: |
| prefill 28k / 112k (tok/s) | **1,266 / 1,238** (`moe_glue` 0: 1,197 / 1,177) | 1,448 / - |
| follow-up: 34.8k + reply + 2.2k new tokens | **2.35 s** | |
| session revisit ~37k tokens | 0.41-0.51 s | |
| exact (drafted == serial) | 10/10 | |
| load time | 36 s | |
| verified through the HTTPS reverse proxy in front of the API | models, thinking (`reasoning` == `reasoning_content`), `stop` streamed / not, stream == non-stream, tool calls 38/42 clean, 0 corrupt | |

0190 knobs A/B'd per request on one load (`results/X1/`, `results/Z1/`, `results/bench_glue_*.txt`):

| knob | 28k / 112k tok/s | GPU tests | production |
| --- | ---: | --- | --- |
| none | 1,195-1,197 / 1,176-1,177 | | |
| `moe_glue` 1 (grouping) | 1,266 / 1,230 | pass | on (in 5) |
| `moe_glue` 5 (+ in-place combine) | 1,266 / 1,238 | pass | **on** |
| `moe_glue` 7 (+ one-kernel router: 2.2 vs 1.3 ms at 8192 rows) | 1,218 / 1,229 | pass | off |
| `attn_bm32` | 1,247 / 1,211 | engine test fails (tile needs 128 KB of shared memory on the test shapes) | off |
| `mtp_window` 8192 | 1,194 / 1,216 | main-model state / resume tests fail | off |
| grouping + bm32 + mtp_window | 1,276 / 1,289 | | off |
| `hc_fused` | - | out of shared memory (131 KB > 101 KB) | off |

Batching with 0200's on-set (`results/Y1/`), BATCH=4 at 262k after dropping page caches (4 slots fit):
- batched == alone: 4/4;
- aggregate 76-79 tok/s at 4 streams (single 47-69);
- per-slot resume: 24.5k cached, ~2 s; a 5th session evicts a slot;
- a 35k prefill beside 3 decoders: 62 s TTFT, 2.4 s decode gaps;
- single-request prefill 5% below single-stream;
- MemAvailable fell to 2-4 GiB under load.

Not used in production (memory).

**Last test window (01:35-02:00, image `z2` = 0190 fixes, `results/Z2/`, `results/T8/`).**

- `test_glue_patches`: 79/80 pass; the one failure is `latent_tc`, which is off.
- `hc_fused` is bitwise but 7-8x slower (num_stages=1): off.
- Per request at 28k / 112k (tok/s):

  | knobs | 28k | 112k |
  | --- | ---: | ---: |
  | production knobs | 1,261 | 1,235 |
  | + `attn_bm32` | 1,286 | 1,284 |
  | + `mtp_window` 4096 | 1,338 | 1,306 |
  | both | 1,389 | 1,357 |

  `exact` 10/10 with both. Decode right after the 112k prompt with both: 56 tok/s (81 without).
- Rank 0 then aborted during a repeat `attn_bm32` run (`terminate called without an active exception`, exit 133). The
  kernel log shows `NVRM ... Out of memory` at the same minute: unified memory ran out at 524k context + 12 GiB store.
- Production went back to image `z` with the committed single-stream config and was re-verified through the API (02:00).
  `attn_bm32` / `mtp_window` stay off until they are run at a smaller CONTEXT or SESSION_GIB and checked for memory.

What was tried and left out of production, and why:

- **Decode kernels (0130)**. On the real model they cost time instead of saving it. 1-row verify: off 31.8 ms,
  `v2` 32.7 ms, `v2,pdl` 32.9 ms (the L1 load: 31.3 ms). Decode, off / v2 / v2,pdl (tok/s):

  | cell | off | v2 | v2,pdl |
  | --- | ---: | ---: | ---: |
  | tf code greedy | 64.3 | 62.4 | 56.4 |
  | kit structured | 96.7 | 96.4 | 93.6 |
  | tweet sequence | 90.1 | 88.5 | 87.7 |

  This was the decode regression of Load A.
- **FP8 prefill (0083)**: +11% at 28k, +7% at 112k with the new tile. Greedy replies diverge from FP8-off within the
  first ~20 tokens on 25 of 30 prompts. Needle retrieval at 28k: 10/10 at 10 / 50 / 90% depth both ways. Tool calls
  equal. Decision: off (a real behaviour change for a single-digit gain).
- **Batching (0120)**, `GLM53_TF_BATCH=4`. At 262k context only 1 sequence fits (3.57 GB of cache slots each, 4 GB
  kept free); at 131k, 2 fit. With 2 slots:
  - batched == alone 4/4;
  - aggregate 56-74 tok/s against 46-72 single-stream (per stream ~30);
  - a 35k prefill stalls a decoding stream for up to 3.2 s.

  0180 (sessions in batch slots) failed 4 of its GPU tests (no cache reuse, follower replay). Production is single
  request + session store.
- **BF16 KDA projection copy (0170)**: 1.45x on that matmul (~7% of a 28k prefill), for +3.26 GiB a rank. At 524k
  context with the session store that would take MemAvailable below 8 GiB during a 128k request (measured minimum
  without it: 11 / 10 GiB). Off.
- **Fat experts (0170)**: bitwise equal to fast2 (tests) and +3-5% end to end (fast2 1,129 / 1,154 / 1,122 tok/s at
  7k / 28k / 112k). On.
- **Cold first requests**: right after a load the first fast prefill compiles Triton kernels (7k: 727 tok/s).
  Production's canary warm-up (`WARMUP_LENGTHS="4096 16384"`) pays that at start.
- **Sessions (0110)**: a revisit resumes (36.8k of 36.9k tokens cached). A shared ~2.4k-token system prompt was
  reused by the third session (1,984-2,048 tokens) but not the second (its first visit came before a fork mark
  existed). Eviction is exact: 46 evictions under a one-entry budget, every reply == fresh (sampled and greedy).
- **Known in production**: with one request at a time, a `stop` match ends the reply but the engine keeps decoding
  silently to EOS / `max_tokens` before the next queued request (0160).

## 4 x 256k batch production, 2026-09-28 (09:15-11:45)

**Incident first.** The 524k single-stream production (`config/prod-single.env.example` with `SESSION_GIB=12`) died at
~07:00-07:04: both kernel logs show `NVRM ... Out of memory [NV_ERR_NO_MEMORY]` (07:00:38-07:01:28 on the head node), rank 1
exited 137 and rank 0 exited; nothing restarted it until this window (~2 h down). Most likely cause: the 12 GiB
session store filling under the morning automations on top of 524k of caches. No watchdog was installed.

**What runs now** (`config/prod.env.example`, an image with every patch through 0220, built on
both nodes): 4 concurrent requests x 262,144 tokens each (`GLM53_TF_BATCH=4`), FP8 latent KV (0220), sessions inside
the batch slots (0180, `BATCH_SESSIONS=1`) with a **2 GiB** store, `PREFILL_ROWS_MAX=2048`, `LEAN_BLOCK=512`, the 0200
on-set. It is `config/prod-batch.env.example` with `SESSION_GIB` 4 -> 2 (see the stress row). The watchdog user timer is
installed on the head node (`glm53-tf-watchdog.timer`, `CONFIG=config/prod.env`, `WATCH_HEAL=1`).

### Why only 1 slot fit before, and what the memory goes to

- Every slot is allocated at full capacity at load (3.59 GiB a slot at 262k bf16, 2.1 GiB FP8). The load-time rule adds a
  slot while `cudaMemGetInfo free - slot - store budget >= BATCH_RESERVE_GB` on both ranks. On GB10 that free figure is
  MemFree: page cache counts as used. The "1 slot" runs had `BATCH_SESSIONS=1` with the 12 GiB store budget counted
  (3.6 + 12 + 4 GiB needed before the first extra slot). With the store off and 0140's O_DIRECT loads (no page cache
  from the weights), 4 bf16 slots fit without dropping caches (B1).
- The rest, per rank: weights 78 GiB, window buffers 8.5 GiB at `LEAN_BLOCK` 1024 (4.2 at 512), the lean set 3.1 / 1.55 /
  0.78 GiB at `PREFILL_ROWS_MAX` 8192 / 4096 / 2048 (batching prefills in 2048-token pieces, so more is never used),
  latent KV 3.31 GiB a slot bf16 / 1.86 FP8. Breakdown: `docs/MEMORY-4x256k.md`.
- The worker node (rank 1) is the binding node: 2 GiB less MemTotal, ~1.5 GiB lower in every measurement below.

### Phase 1: bf16 KV, image `z`, BATCH=4, CONTEXT=262144, store off (`results/B1-B3/`)

| | B1: rows 8192 | B2: rows 4096 | B3: rows 2048 |
| --- | ---: | ---: | ---: |
| slots | 4 | 4 | 4 |
| MemAvailable after load, r0 | 8.0 | 9.6 | 10.4 (r1 8.2) |
| minimum under the quick bench, r0 / r1 (GiB) | 5.7 / 4.4 | 6.7 / 5.0 | 7.5 / 5.9 |
| prefill 28.7k alone (tok/s) | 1,197 | 1,202 | 1,198 |
| 1 / 2 / 4 streams aggregate (tok/s) | - / - / 74-82 | 44-64 / 58-62 / 74-79 | 44-64 / 57-60 / 75-81 |
| batched == alone | 4/4 | 4/4 | 4/4 |
| stall: longest decode gap during a ~35k prefill | 2.0 s (2 decoders) | 2.1 s (3) | 2.1 s (3), TTFT 38.9 s |

B1 decode (tok/s): tf code greedy 61.5, chat greedy 47.1, kit structured 97.1, hashmap 49.4, essay 44.2; decode
after the 28.7k prompt 88. B3 stress, 4 x 112k prompts at once: minimum MemAvailable 6.3 / 4.8 GiB. bf16 at 4 x 262k
cannot reach 8 GiB of headroom with these knobs.

### Phase 2: FP8 latent KV, `config/prod-batch.env` (`results/K1/`, `results/K1-tests*`)

GPU tests on image `fp8kv`:

| file | result | note |
| --- | --- | --- |
| test_fp8_kv_patches | 4 pass, 6 fail, 16 errors | **the engine-level FP8 tests do not run**: the toy model has `kv_lora` 128 and 0220 only lays out 512 (`ValueError`). Kernel tests: FP8 attention != bf16 attention on the dequantized rows bit for bit (2 fails); FP8 rel. error vs the fp32 expanded reference 5.7e-2 against a 4e-2 bound (bf16: 3.4e-3) (2 fails). To fix in 0220 / its tests. |
| test_latent / test_1m | 20/20, 45/45 | |
| test_glue | 79/80 | `latent_tc` (off), as before |
| test_batch_sessions (0180 fixed) | 28/28 | was 4 failures |
| test_batch_parallel / test_session | 26/26, 32/32 | |
| test_batch2 | 40/44 | the same 4 as in T-runs before (fast-prefill admissions, per-sequence knobs) |

So FP8 was checked on the real model instead:

| | K1: FP8, store 4 GiB | production before (524k, single) | vLLM prod kit |
| --- | ---: | ---: | ---: |
| load | 150 s first (calibration re-measured), 35 s after | 36 s | |
| MemAvailable after load, r0 / r1 | 19.4 / 17.8 GiB | | |
| exact (drafted == serial) / batched == alone | 10/10 / 4/4 | 10/10 / - | |
| prefill alone 24.5k / 98k (tok/s) | 1,154 / 1,127 | 1,266 / 1,238 (28k / 112k) | 1,448 (28k) |
| decode tf code greedy / chat greedy / kit structured / hashmap / essay | 75.3 / 41.2 / 95.7 / 49.5 / 42.0 | 65.8 / 44.7 / 98.1 / - / - | 41.9 / 22.8 / 72.7 / 30.0 / 26.1 |
| concurrent streams 1 / 2 / 4, aggregate tok/s | 43-70 / 56-62 / 72-77 | one at a time | |
| stall: 3 decoders + a 39.8k prefill | gap 2.1 s, TTFT 45 s | queued behind | |
| slot resume (~40k sessions, 4 slots + a 5th) | revisits 2.5 s (39.8k cached); session 1 resumed after the 5th | | |
| MMLU-200 / refusals | 88.0% / 0/10 | 89.5% / 0 | |
| greedy replies vs bf16 KV (fp8ab `replies`, 20 prompts) | 5/20 identical; the rest diverge in the first 0-78 tokens, stay on topic (see below) | | |
| needle, fast prefill: 28k 3 depths x 3, 112k 3 depths x 1 | 9/9, 3/3 | 10/10 at 28k | |

**Memory stress** (`multiturn.py --modes stress`: 4 conversations grown together by ~60k-token turns, resumed in their
slots, store filling; then 3 decode 512 tokens while the 4th adds a 32k turn):

| run | sizes | fill | final 32k turn | MemAvailable min r0 / r1 | gate (>= 8) |
| --- | --- | ---: | --- | ---: | --- |
| K1 stress2, store 4 GiB | 251.6k, 251.9k, 251.9k, 250.2k | 1,049 s | TTFT 50.7 s, decode gap 2.9 s | 8.85 / **7.32** | fail |
| P1 production, store 2 GiB | 162.3k, 162.2k, 162.3k, 158.5k | 638 s | TTFT 49.2 s, decode gap 2.8 s | **16.06 / 14.44** | pass |

No NVRM OOM, no request errors, `/health` ok after both. Resumed turns show `cached` = the previous prompt (e.g. 187,264
of 251,562). K1's floor came after ~50 minutes of every other
benchmark on the same load (MemAvailable r1 went 17.8 -> 11.3 GiB during exact / concurrency / slots / 112k prefill /
decode, before the stress began), then sank ~0.3 GiB a stress round while the 4 GiB store filled. P1 ran on a fresh
production load (heal restart), so it does not show that drift: with the drift and a full 2 GiB store, the expected
long-uptime floor on the worker node is ~9.3 GiB (K1's 7.3 + the 2 GiB of store). Worth watching `MemAvailable` on the worker node over
the first days; the knobs if it goes under 8: `SESSION_GIB=1`, `BATCH_MAX_GRAPHS` below 256. After P1, a 98k prefill alone (1,099 tok/s,
decode after it 80 tok/s) and the `slots` run kept the floor at 15.9 / 14.3 GiB. Cost of the smaller store: in `slots`,
session 1 coming back after a 5th session was a cold 42 s prefill with 2 GiB (it resumed from the store with 4 GiB in K1);
revisits while it still holds its slot resume in 2.5 s either way.

Watchdog heal test: `docker kill glm53-tf-r0` at 10:53:40. The first heal (10:57) did nothing: the unit is a oneshot and
systemd killed the detached `serve.sh restart` with the tick's cgroup (empty `heal.log`). Fixed with `KillMode=process` in
`scripts/systemd/glm53-tf-watchdog.service`; the next heal (11:04:50) restarted both ranks from `config/prod.env`, ready
after 35 s, canary ok.

Not run in this window (time went to the stress runs): the 0190 per-request knobs `attn_bm32` and `mtp_window` on the
FP8 load, the `LEAN_BLOCK` 512 vs 1024 A/B (the ~4% lower prefill than B3's 1,198 at 28k is block 512 plus FP8 rows;
unseparated), and `followup,sessions` (the `slots` mode covered resume).

**What still needs real-use testing**: FP8 KV changes greedy replies from the first tokens on (expected). Reviewing
the bf16 replies against the FP8 replies (`bench/fp8ab.py --modes replies`; not in `results/`), the long-prompt FP8 replies read as
correct and specific as the bf16 ones, but that is a spot check. Please use it on real long agent sessions (past 100k:
recall of early details, tool-call formatting). Fallback without a rebuild: `GLM53_TF_KV_DTYPE=bf16` with
`SESSION_GIB=0`/`BATCH_SESSIONS=0` (worst case ~5-6 GiB on the worker node at 4 x 262k: under target), or 3 slots, or the
single-stream config (`config/prod-single.env.example`, `SESSION_GIB` <= 6).

Verified through the HTTPS reverse proxy after the switch (11:18): `/v1/models` lists
`GLM-5.3-Flash-EXL3`; a thinking reply returns `reasoning_content` and the answer (391, finish `stop`); a tool call returns
`get_weather({"city":"Paris"})` with finish `tool_calls`; streaming delivers the reply in chunks.

## W1: 0190 prefill knobs on the 4 x 256k production, 2026-09-28 (12:18-12:55)

Load: `config/prod.env` (image `fp8kv`, then `w1` = the same with 0190's new MTP prefill cache rows), `GLM53_TF_PROFILE=1`,
per-request `tf_knobs`, unique filler prompts, non-streaming (engine `prefill_s`, `decode_s`, `tokens_per_round`),
256-token greedy replies. JSON / logs: `results/W1/` on the head node (`ab*.json`, `memtest.log`, `conc.json`, `tests.log`).

| prefill tok/s | 24.5k | 98k |
| --- | ---: | ---: |
| production (`fp8kv`, knobs off) | 1,146-1,154 (4 runs) | 1,133 |
| + `attn_bm32` | 1,202-1,207 (3 runs; the first run, 1,141, compiled the tile) | 1,190 |
| + `mtp_window` 4096 | 1,156 | 1,130 |
| + both | 1,209 | 1,185 |
| `w1`, `GLM53_TF_MTP_PREFILL_CACHE=1` | 1,214 / 1,214 | 1,193 |
| `w1` cache rows + `attn_bm32` (**production now**) | **1,283 / 1,280** | **1,258** |

- Decode after the prompt: 83-91 tok/s cold and warm in every cell, tokens a round 7.29-7.5, the same drafter mix
  (5 MTP / 29-30 DFlash2 rounds) and the same reply hash everywhere: no knob changes decode.
- `mtp_window` does nothing here: batch mode prefills 2,048-token pieces, each a prefill of `prompt[:end]`, so its
  start `end - W` is below the piece for any W >= 2,048. Why it cost decode in Z2 (single stream): the zeroed head
  rows score exactly 0 in the head's indexer, above real pools with negative (signed-weight) scores, so the head
  attends to zero rows and its drafts get worse; larger windows only push that back. Not measured in single-stream
  mode this window (production is batch). Replaced by the cache-rows switch (docs/PATCHES.md, 0190 update): prefill
  never reads the head's outputs, so only its cache writes run; same bits, drafts unchanged. GPU tests 21/21
  (`-k "mtp_prefill_cache or mtp_window or mtp_absorb"`).
- `attn_bm32` + cache rows together: +11% at 24.5k and 98k (vs 1,448 for the vLLM kit at 28k).
- Memory (`memtest.py`, `attn_bm32` on, cache rows on): 3 x 98k prompts, then those 3 decoding 1,500 tokens each
  (resumed from their slots, 97,984 cached) while a 4th 98k prompt prefilled (TTFT 151 s): minimum MemAvailable
  **14.8 / 13.4 GiB** (r0 / r1); a single 98k prefill 16.6 / 15.0. Neither knob allocates memory.

**Batched decode, 4 streams** (`multiturn.py --modes concurrent --streams 1,4 --long-tokens 512`): aggregate 79 / 71
tok/s (1 stream 64 / 44). Per round: verify (forward, sampling, accept, commit) 95-139 ms, batched MTP drafting
4-5 ms (`mtp_batched` in 20-45% of rounds), so the verify forward is ~95% of a round. 70-90% of the rounds are
`eager` (keys met fewer than `CAPTURE_AFTER=3` times in a 25 s run; captures 9-22 a stream): the next cost to attack
is the eager rounds (+10-25 ms each per 0200's model), then the rows themselves (~6-7 ms a verify row: 4 x 2-7
rows). During a 98k prefill beside 3 decoders (memtest phase B), a decoder's 145 s went ~70 s to waiting through
prefill pieces, ~65 s to verify, ~3 s to drafting (2.0 tokens a round, 10 tok/s each).

## W2-W4: 0260 expert bench, 0230 RoCE, 0240 b12x, 0250 NVMe sessions, 2026-09-28 (13:00-13:44)

Images built from committed trees only: `w2` (through 0260 at 57d4631), `w3` (+ 0240), `w4` = `sessdisk` (+ 0250 at
74191b1). Logs: `results/W2`, `W3`, `W4`.

**0260 (`bench_experts.py 1024 2048 4096 8192`, one GPU, prod stopped; all kernels "same" bits, 48/48 tests).**
Sum of gate/up + down, ms (uniform routing; skewed in brackets):

| rows | fast2 | fat | once | once, no decode | once, no mma |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 9.4 [10.0] | 11.6 [11.9] | 11.6 [12.1] | 11.5 [12.4] | 11.5 [11.7] |
| 2048 (prod piece) | 10.9 [12.5] | 13.0 [14.2] | 13.0 [14.6] | 12.9 [14.7] | 12.9 [13.3] |
| 4096 | 16.9 [19.0] | 16.0 [19.0] | 16.7 [20.9] | 16.4 [19.7] | 15.8 [16.9] |
| 8192 | 31.1 [32.9] | 28.6 [30.2] | 32.1 [33.4] | 30.1 [31.2] | 22.5 [24.4] |

- `once` never beats `fat` (0.89-1.01x): do not use.
- At <= 2,048 rows (what batch production runs), removing the trellis decode or the MMA changes nothing (within 1%):
  the kernel is bound by data movement / scheduling (weight reads + member-row gathers), not by either. Also there
  **fast2 is 1.13-1.24x faster than fat** (the reverse of 8,192 rows): with `PREFILL_ROWS_MAX=2048` production
  should A/B `GLM53_TF_FAST_EXPERTS=fast2` end to end (bitwise equal, so a per-load swap is safe).
- At 8,192 rows: no-mma saves 21-27% (MMA-bound share), no-decode 3-5%: decode-sharing cannot pay; better MMA
  (tiling / tensor-core use) is where the kernel time is.

**0230 RoCE: stopped at step 2.** Unit tests 38/38 (one GPU, no RDMA). NIC loopback on the head node (both CX7 functions,
16k first size) timed out at sequence 312 on roceP2p1s0f1: "the flag HAS reached this host's memory (the GPU did not
observe it: sparkring #278's signature)". That is the failure mode the design must not have (GPU-side visibility of
RDMA-written host memory); the 2-node bench / fault / soak and the engine A/B were not run. To investigate before
the next attempt: the kernel's acquire load path on pinned host memory (volatile / `ld.acquire.sys` vs caching),
single-HCA loopback (`GLM53_TF_ROCE_HCAS=1`). A `/cache/roce-failed` marker already existed (06:17, earlier run).

**0240 b12x: not adopted.** `bench_b12x.py --sweep`: KDA 0.94x of fast_kda (slower; no BV / warps / stages setting
beats the default tf32 BV 64 w4 s1 at 1.37-2.05 ms, stages 2-3 exceed shared memory), fused mhc 0.69-0.74x (slower;
fused == unfused bitwise True), one-pass sparse attention 1.47x (bf16 and FP8). Tests 39 pass / 14 fail: resumes never
hit with b12x bits on (`cached` 0: drafted == serial / resumed == fresh, snapshot control), state differs across
C / overlap for bits 3, FP8 one-pass attention row subsets not bitwise, mhc vs float64 3.8e-4 (bound 1e-5), mhc vs
today 3e-2, a KDA dispatch test error. End to end on the production load (`w3`, per request): 24.5k: b0 1,275 / 1,278,
b1 1,163, b2 1,112, b4 1,290 / 1,322, b7 1,145; 98k: b0 1,248, b4 1,303 (+4.4%). Decode unchanged. Only bit 4 helps
(+1-4%), below the 5% bar, and its FP8 row-independence test fails (production is FP8 KV).

**0250 NVMe session tier: adopted.** GPU tests 30/32 (the 2 failures: the FP8 engine tests cannot run on the toy
model's 128-wide latent, as in 0220). `sessdisk bench` on the head node's NVMe: 40k write 0.30 s / read 0.14 s, 100k 0.60 /
0.28 s, exact. Load with the tier on (prod config): 6 sessions of 39,226 tokens A..F (31 s cold each) on 4 slots +
2 GiB RAM, then A, B: 38,912 cached, prefill 0.42 s, same reply hash; server restarted, then C, F: 20 entries indexed
in 0.2 s, 38,912 cached, 0.43-0.45 s, same hashes. `exact` 10/10. MemAvailable after it 18 / 16 GiB. Production
now runs `glm53-tensorfold:sessdisk` with `GLM53_TF_SESSION_DISK=/sessions`, 64 GiB; verified through https
(13:43: models, thinking reply 391, tool call `get_weather({"city":"Paris"})`, streaming), watchdog re-armed.

## W5: fast2 vs fat end to end (0270), round buckets for batched graphs (0280), 2026-09-28 (13:57-14:24)

Image `glm53-tensorfold:w5` (every patch through 0280, built on both nodes), loads from `config/prod.env` with
overrides (`results/W5/load.sh`). Prod down 27 min. Logs and JSON: `results/W5/`. **Nothing adopted: production
stays `glm53-tensorfold:sessdisk` / `config/prod.env` unchanged**, restored 14:23, checked through https (models,
`17*23` -> `391`, canary), watchdog timer re-armed, lease deleted.

**Why fat was picked (history).** 0170 made `fat` the production kernel on the 8,192-row lean chunks of that time:
+3-5% end to end (RESULTS "Fat experts (0170)"). At 8,192 rows fat is still the faster kernel (W2: 28.6 vs 31.1 ms).
`PREFILL_ROWS_MAX` went to 2,048 with the 4 x 256k batch production, where W2 found fast2 1.13-1.24x faster in
isolation. Decode never runs either kernel: fast2 / fat / once serve fast prefill chunks only (`exl3_mm.routed(...,
fast=True)`); decode and verify windows use the row-invariant grouped kernels. So there is no decode-side reason for
either; the rows that matter are fast-chunk sizes (multiples of 64 up to 2,048: batch pieces, prompt tails, short
prompts).

**Kernel sweep** (`bench_experts.py`, one GPU, gate/up + down ms, uniform [skewed]; 1-40 rows listed for the decode
question only; 1024-8192 from W2; fast2 here reads the shared rotated input, as 0270's auto does):

| rows | 1 | 8 | 40 | 64 | 128 | 256 | 512 | 1024 (W2) | 2048 (W2) | 4096 (W2) | 8192 (W2) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fast2 | 0.29 | 1.66 [1.50] | 5.62 [4.40] | 6.61 [5.28] | 8.03 [6.92] | 8.37 [8.17] | 8.69 [8.96] | 9.4 [10.0] | 10.9 [12.5] | 16.9 [19.0] | 31.1 [32.9] |
| fat | 0.28 | 2.01 [1.90] | 7.23 [5.63] | 8.28 [7.02] | 10.10 [8.72] | 11.15 [10.19] | 10.91 [11.04] | 11.6 [11.9] | 13.0 [14.2] | 16.0 [19.0] | 28.6 [30.2] |

fast2 wins from 8 to 2,048 rows (1.19-1.33x), ties at 1 row, loses from ~4,096. All bits "same". So 0270's
`GLM53_TF_FAST_EXPERTS=auto` (fast2 below `GLM53_TF_FAST2_ROWS` hi = 4096, fat at and above) was the candidate.

**End to end, the kernel win does not transfer** (one load, `FAST_EXPERTS=auto`, per request `tf_knobs.fat_experts`
1 = fat, 2 = auto, 0 = plain fast2; `results/W5/ab1.log`; unique prompts, cold prefill):

| prefill tok/s | 24.5k (2 runs) | 98k |
| --- | ---: | ---: |
| fat (production) | 1,284 / 1,278 | 1,259 |
| auto (fast2 on rot_in1 below 4,096 rows) | 1,251 / 1,253 (-2.2%) | 1,232 (-2.1%) |
| fast2 (0170's plain path, two rotations) | 1,224 / 1,222 (-4.6%) | 1,199 (-4.7%) |

Same reply hash everywhere; decode after the prompt 80-90 tok/s in every cell. With production's pipelined lean
chunks (`PREFILL_OVERLAP=1`: all-gathers and hc slabs on another stream while the experts run) fat is faster in the
engine even though fast2 wins the isolated kernel by ~2 ms a layer. Likely cause (not profiled): fast2's gate/up is
register-bound at one CTA an SM (224 registers) and competes worse with the overlapped kernels than fat's 2 CTAs an SM
at 128 registers. auto on the same load: `exact` 10/10, tf suite code/chat sampled 40.3 / 38.7, greedy 72.6 / 40.5
(unchanged kernels for decode). **0270 not adopted** (-2%, gate was +3%); it stays in the tree, off, as a tool
(bitwise-tested, per-request `fat_experts=2`). MMLU was not run (nothing to adopt; fat == auto bit for bit anyway).

**Batched decode graphs (0280).** 4 streams, `multiturn.py --modes batchexact,concurrent --streams 4 --reps 3
--long-tokens 512`, all on image w5 with fat:

| load | aggregate tok/s (3 reps) | mean | rounds graph / eager / capture | verify ms a round | batchexact / exact |
| --- | --- | ---: | --- | ---: | --- |
| production knobs (buckets off) | 78.6 / 70.3 / 73.7 | 74.2 | 8-33% / 49-82% / 8-10% | 105-115 | 4/4 / 10/10 |
| `GLM53_TF_BATCH_BUCKETS=4,8` (tied routing) | 73.6 / 67.5 / 70.8 | 70.6 (-5%) | 80-91% / 6-12% / 3-5% | 114-121 | 4/4 / 10/10 |
| `GLM53_TF_BATCH_GRAPHS=0` (every round eager) | 80.7 / 74.3 / 75.5 | 76.8 (+3.5%) | 0 / 100% / 0 | 103-110 | - |

- Buckets did what they were built for: graph replays went from 8-33% to 80-91% of rounds, byte-identical output
  (batchexact 4/4, exact 10/10, GPU tests 9/9). But each round got ~6-9 ms slower: ~1,750-1,980 padded rows a run
  (~2.7 a slot and round). Even routed to their window's last row (no new expert reads) a padded row costs ~0.8-1 ms
  (KDA chain steps, attention, dense rows). Not adopted (-5%, gate was +10%).
- The decisive number is the third row: with no batched graphs at all the aggregate is the same or slightly higher.
  So W1's premise ("eager rounds cost +10-25 ms each") does not hold at these shapes: a 4-slot verify of ~10-20 rows
  is GPU-bound at ~100-120 ms, the host enqueues the eager kernels faster than the GPU runs them, and a replay saves
  ~nothing, while captures cost a full extra forward each (8-10% of rounds). No bucketing / padding / capture policy
  can reach +10%: the ceiling of "every round a graph for free" is ~0 here. `GRAPHS=0`'s +3.5% is within the
  rep-to-rep spread (70-81) and was not A/B'd for exactness or memory; not adopted. A cheaper knob to try in a later
  window: `GLM53_TF_BATCH_CAPTURE_AFTER` higher (fewer captures) or `GLM53_TF_BATCH_GRAPHS=0`, both with batchexact.
- What would move 4-stream throughput instead: the verify rows themselves (~6-7 ms a real row, mostly routed-expert
  reads that rows of different sequences rarely share) and fewer rejected draft rows (per-stream tokens a round
  2.0-2.5 for the MTP-heavy streams). 0200's "one launch per layer for per-slot KDA / attention" would only remove
  launches, which this window shows are not the bottleneck.
- Memory: not measured for 0280 (not adopted). Graphs are few with buckets (fewer than without), so it would not
  have been the constraint.

Tests (the worker node, image w5): `test_fast_experts_auto_patches` 29/29, `test_batch_buckets_patches` 9/9,
`test_batch_parallel_patches` 26/26; host-only parts of `test_mia_prefill`, `test_expert_once`, `test_knob`,
`test_batch2`, `test_batch_sessions` 112 passed (GPU parts skipped there).

## Prefill work

Fast prefill (patches 0080-0084, knobs 0091-0093) on the latent-KV load (`CONTEXT=262144`, q4mse, expert loop,
real calibration, cost depths, lookup, bf16 gathers). Prefill tok/s = prompt tokens / cold TTFT (unique prompt, no
cache hit), 1.8k / 7k / 28k / 112k-token prompts. Every fast run follows a warm-up pass (the first fast request of a
row bucket compiles Triton kernels: F4's 2k cell took 49 s). JSON in `results/F*-ctx*.json`, per-request profiles
in `results/profiles/`.

| Config | 1.8k | 7k | 28k | 112k | warm TTFT 28k |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline: exact, 1024-row chunks (L1) | 498 | 625 | 660 | 582 | 42 s |
| exact + 0065 (blocked index selection), 1024 (F5) | 649 | 662 | 659 | 638 | 43 s |
| fast, 1024 (F5: chunked KDA, exact-order qmm tiles, fused experts) | 754 | 782 | 767 | 772 | 0.9 s |
| fast, 1024, + one-accumulator matmuls, hc_pre row tiles (F9) | | 837 | 863 | 852 | 0.7 s |
| fast, lean 2048 (F9) | | 866 | 892 | 871 | 1.8 s |
| fast, lean 4096 (F9) | | 892 | 908 | 890 | 4.0 s |
| fast, lean 8192 (F9) | | 913 | 920 | 898 | 4.0 s |
| lean 8192 + `fp8_prefill` (F9) | | 962 | 966 | 944 | 3.8 s |
| lean 8192 + `prefill_overlap` (F9) | | 974 | 982 | 961 | 3.8 s |
| **lean 8192 + fp8 + overlap (F9; the defaults left running, F10: 1,032 at 28k)** | | **1,024** | **1,034** | **1,013** | 3.6 s |
| 1M context load, fast 1024 (F6; `CONTEXT=1000000`, `PREFILL_ROWS_MAX=1024`) | | | 863 | 848 | 0.7 s |
| 1M context load, exact 1024 (F6) | | | 673 | 656 | 42 s |
| vLLM prod kit | 960 | 1,340 | 1,448 | | |

Warm TTFT: a follow-up that extends the prompt resumes from the last grid point, so it re-prefills up to one chunk
(C - 1 tokens): ~1 s at C = 1024, 4-7 s at C = 8192 (7k prompt: 6.8 s). `tf_knobs.prefill_rows` picks C per request.

Profile of a 28k prompt (s, rank 0; F9 and L1):

| Component | exact 1024 (L1/F5) | fast 1024 (F9) | fast lean 8192 | 8192 + fp8 + overlap |
| --- | ---: | ---: | ---: | ---: |
| routed experts | 14.6-15.0 | 12.3 | 9.9 | 10.0 |
| all-gathers | 4.6-5.7 | 3.0 | 3.0 | 0.0 (overlapped) |
| DSA sparse attention | 3.1 | 3.0 | 3.1 | 2.3 |
| KDA chain | 3.5 | 1.8 | 1.8 | 1.8 |
| KDA projections | 3.1 | 1.9 | 1.9 | 2.2 |
| hyper-connections | 2.2 | 1.9 | 2.0 | 2.9 (hc slabs, incl. gather waits) |
| DSA o-proj | 2.2 | 1.9 | 1.9 | 0.7 |
| DSA indexer | 1.5 (0.5 with 0065) | 0.4 | 0.4 | 0.4 |
| MTP head absorb | 1.3-1.4 | 1.3 | 1.3 | 1.3 |
| total GPU | 42.5 | 32.4 | 30.4 | 27.0 |

At 112k (8192 + fp8 + overlap, 110 s): routed experts 39.3, hc 11.5, sparse attention 9.7, KDA proj 8.8, KDA chain
7.2, MTP 5.4, router 5.0, shared expert 4.9, indexer 4.8 (26 s before 0065).

Exactness and quality with fast prefill: `--suites exact` 10/10 identical with fast on (F5, lean 2048) and with
lean 8192 + fp8 + overlap (F9). MMLU-200: exact latent (Q3) 88.5%, fast 2048 (Q4) 88.5% (5 answers differ from
Q3), lean 8192 fp8 off (Q5-0) 89.0%, fp8 on (Q5-1) 88.5% (5 answers differ from fp8 off); refusals 0/10 everywhere.
Decode is unchanged (F5 fast vs L1: tf/kit/edit cells within run-to-run spread).

## W6: patch 0290 shared KV pool (2026-09-28 15:27-16:35) — adopted

Image `glm53-tensorfold:kvpool` (0290 on the prod set), FP8 KV, BATCH=4. Load A: pool 1,048,576, CONTEXT 262,144.
Load B/C: pool 1,048,576, CONTEXT 1,048,576, `GLM53_TF_KV_POOL_CHECK=1`. Files: `results/W6/`.

| Check | Production (sessdisk, 4 x 262k) | Pool, CONTEXT 262k | Pool, CONTEXT 1M |
| --- | ---: | ---: | ---: |
| Prefill 24.5k (tok/s) | 1,278-1,284 | 1,258 / 1,259 | 1,252 |
| Prefill 98k (tok/s) | 1,259 | 1,252 / 1,253 | 1,250 |
| Reply sha (ab.py) | 8794a3463259cc2f | same | same |
| exact / batchexact | 10/10 / 4/4 | 10/10 / 4/4 | 10/10 / - |
| 4-stream aggregate (tok/s, 3 reps) | 70-81 (W5 spread) | 76.9 / 70.5 / 73.8 | - |
| MMLU-200 / refusals | 88.0% / 0 | - | 88.0% / 0 |
| Max context a request | 262,144 | 262,144 | 1,048,576 |

- Needle at 357,820 prompt tokens in one slot: found; prefill 1,034.6 tok/s (345.8 s), decode 54-59 tok/s; the
  follow-up resumed 357,760 cached tokens (prefill 0.40 s).
- Stress 4 x ~300k (1.22M tokens against a 1.05M pool, so spills/waits happen): completed in 1,162 s, longest decode
  gap 6.8 s, MemAvailable minimum 15.11 / 14.05 GiB (head / worker) -> PASS (>= 8). health: 0 errors.
- GPU tests: test_kv_pool_patches 22/23 first run; the failure (`test_gpu_disk_sessions_into_pooled_slots`) was a
  test bug (an identical prompt never resumes by design); fixed to extend the prompt on round 2, rerun 23/23.
  test_latent_patches 20/20, test_1m_patches 45/45.
- Cost: prefill -1.5 to -2% at 24.5k, -0.5 to -0.7% at 98k (over the 1% bar at 24.5k); accepted for 4x the per-request
  context at the same memory.
- Adopted: prod.env IMAGE=kvpool, CONTEXT=1048576, GLM53_TF_KV_POOL_TOKENS=1048576 (pool check off). Restarted 16:3x,
  ready in 40 s, canary ok (decode 76.3 tok/s), https chat ok, MemAvailable 18 / 17 GiB idle. The benchmark driver died of an
  API error during load B; the window was finished by hand.

## W7: profile (2026-09-28 16:53-17:13, prod down 20 min) — measurement only, nothing adopted

Per-stage breakdown of prefill (one 2,048-row piece, whole 21,464- and 85,781-token prompts = the "24.5k" / "98k"
cells) and decode rounds (1 stream, 4 streams) on the production config, both ranks, with nsys (CUDA + NVTX marks from
the `GLM53_TF_PROFILE` probe sites); piece / chunk size sweep; the 0230 RoCE loopback rerun. Full tables, method and
conclusions: **`docs/PROFILE.md`**. Files: `results/W7/` (scripts, logs, `tables.md`, `analysis/*.json`); the four
nsys reports (`pf24-r{0,1}`, `mix-r{0,1}` = 98k + decodes) are on the head node in `~/w7-traces/` (not in git, 15-106 MB).

**Prefill, % of wall (both ranks within 0.5 point of each other; 24.5k / 98k):** routed experts 26.8 / 24.8, DSA/MLA
attention 25.9 / 28.4 (sparse attention 12.1-12.3, latent expand + o_proj 8.7, q/kv 3.5, indexer 1.6 / 3.9),
KDA 19.5 (projections 11.5, recurrence 8.0), hyper-connections 13.0 / 12.9, shared expert + dense MLP 5.5-5.8,
router + combine 5.0, DFlash2 taps 1.0, MTP cache rows 0.4, **NCCL exposed 0.8, GPU idle 1.2-1.3**, memcpy 0.4.
The all-gathers (4,038 / 15,541 of 4 MiB, RING_LL, ~456 us each) run 94% beside compute, but the kernels beside them
pay an **overlap tax of 8.2-8.7% of the wall** (`_hc_post` 139 us alone vs 413 us beside an all-gather): communication
costs ~9% of prefill in total. `prefill_overlap: 0` costs -6.4% / -5.8%. Ranks are balanced (each waits 2-2.3% of the
wall inside all-gathers, jitter; per-piece GPU time equal to 0.01 s).

**Decode round, kernel ms (1 stream 59.5 ms / 4 streams 125.5 ms under capture; 54-60 / 113-122 ms without):** routed
experts 26.4 / 72.3 (44% / 58%), dense q4 GEMMs 16.3 / 23.0, NCCL 4.8 / 7.7-8.3 (fully exposed; 100 / 120 all-gathers,
half of it waiting for the peer), attention 2.6 / 5.2, KDA 1.6 / 5.5, hc 1.5 / 1.8, idle 4.5-7.3 ms; verify forward
82-83% of a round, drafting 8% / 11%.

**Piece / chunk sweep, one active request (cold prefill tok/s, 24.5k / 98k):**

| load | chunk rows | 24.5k | 98k | vs 2,048 |
| --- | ---: | ---: | ---: | ---: |
| B: `BATCH_PIECE=8192`, `PREFILL_ROWS_MAX=8192` | 2,048 | 1,266 | 1,252 | base |
| B | 4,096 | 1,310 | 1,294 | +3.5% / +3.4% |
| B | **8,192** | **1,339** | **1,319** | **+5.8% / +5.4%** |
| B | 2,048, `prefill_overlap: 0` | 1,184 | 1,178 | -6.4% / -5.8% |
| C: `BATCH_PIECE=4096`, `PREFILL_ROWS_MAX=4096` | 4,096 | 1,307 | 1,290 | +3.2% / +3.0% |
| prod (A1, nsys attached, idle) | 2,048 | 1,264 | - | |

Same reply sha in every cell (`8794a3463259cc2f`), `exact` 10/10 on load B (8,192-row chunks). The gain is the chunk
size, not the piece (B at 4,096 rows == C within 0.3%). Cost: lean chunk buffers 0.78 -> 1.55 -> 3.10 GiB a rank;
MemAvailable minimum 14.97 / 13.60 (prod config, 98k + decodes), 15.91 / 14.71 (C, one 98k), 13.38 / 12.17 (B sweep),
12.38 / 11.18 (B incl. 4-stream decode + exact); worst case (4 full slots) at 8,192 not measured. A larger piece only
while one request is active is exact but needs a scheduler change (`GLM53_TF_BATCH_PIECE` is fixed at load) and the
8,192-row buffers loaded regardless (docs/PROFILE.md section 6).

**0230 RoCE loopback rerun (image kvpool, W2's harness): same failure, same place** — timeout at sequence 312 on
roceP2p1s0f1 with the same "flag HAS reached this host's memory" diagnosis (`results/W7/roce-loopback.log`). So it is
deterministic, not a transient. 312 = 1 + 10 + 300 + 1 is the first collective of the harness's single-rank graph
warm-up, which matches 0350's explanation (f45b7ef: the harness, not GPU visibility). With 0350's harness
(`tests/cuda/bench_roce.py` at f45b7ef, md5 86ab4277, run from a copy against the image's 0230 runtime; RoCE not
enabled in the server) the same loopback completes 21,044 exchanges with no timeout, bit-exact at 16k / 64k / 128k
(graph 11.3 / 17.7 / 26.9 us), but **`bits_equal: false` at 1 MiB** (97 us), above the engine's 256 KiB default
`GLM53_TF_ROCE_MAX_KB`: a separate data problem to look at before any size above 256 KiB goes over RoCE
(`results/W7/roce-loopback-0350harness.log`).

**Conclusions (docs/PROFILE.md section 7):** the largest lever is expert GEMM efficiency (routed experts 25-27% of
prefill and 44-58% of decode rounds, dense q4 GEMMs next); attention is second and grows with context; communication
is ~9% of prefill but nearly all overlap tax, so pipeline parallelism (which would roughly halve single-stream decode)
is not worth it; the cheapest win is the chunk size (+3.4% at 4,096, +5.4-5.8% at 8,192, same bits, +0.8 / +2.3 GiB a
rank); host gaps are ~1%.

Ops notes: nsys needs `--cap-add SYS_ADMIN` here (`RmProfilingAdminOnly: 1`) and `--trace=cuda-sw` for more than one
capture; a second `nsys start` in the same session lost the agent and took both ranks down (load A1, 16:55; the
captured trace was intact). Prod restored 17:13 from `config/prod.env` (image kvpool, standard entrypoint), ready in
25 s, canary ok (decode 74.1 tok/s), https `/v1/models` ok, `17*23` -> `391`, watchdog timer re-armed, lease
refresher and memory sampler stopped, lease deleted.

## W8: batch 2 (0300, 0310, 0320, 0330, 0335 + chunk size), 2026-09-28 (17:30-18:25, 18:26-19:15) — adopted

Image `glm53-tensorfold:b2` = every patch through 0360, built on the head node and loaded on the worker node (after the 0330 loader fix
below). Off == production: `tests/kvpool_ptx.py` against `kvpool` finds 33 of 33 Triton kernels' PTX identical, and load
A (b2 with only the request log on) gives kvpool's reply sha, exact 10/10 and batchexact 4/4. Every load is
`config/prod.env` plus overrides (`results/W8/load.sh`); per load `results/W8/run.sh` = exact, batchexact, `ab.py`
24.5k / 98k twice (cold prefill tok/s, reply sha, decode after the prompt), `multiturn.py --modes concurrent --streams
1,4 --reps 3`. Two windows, prod restored in between (18:25, ~1 min before the second window). Files: `results/W8/`.

**GPU unit tests** (`results/W8/tests-head*`, `tests-worker*`):

| test | result |
| --- | --- |
| 0320 `test_prefill_pp_patches` (incl. the one-GPU row-split kernel identity test on real shapes, fused hc 0 / 3) | 15 passed; **the kernels are bitwise on half sub-blocks, so the row split is usable** |
| 0320 two engines on one GPU (`PP_TWO_PROC=1`) | failed first: the test's host-staged gloo communicator synchronizes inside the engine's decode-graph capture (`cudaErrorStreamCaptureUnsupported`), a test bug; fixed (no decode graphs in that test), then passed |
| 0310 `test_prefix_share_patches` | 31 passed on the rerun; the first full run hit an illegal memory access while building a reference engine (prefix share off, in the Graphs warm-up) after 28 passes; not reproduced in two more runs (one with `CUDA_LAUNCH_BLOCKING=1`). Unexplained, recorded |
| 0335 `test_solo_piece_patches` | 12 of 14. `test_second_request_turns_pieces_normal` needed a 4,096-token test context (fixed); its greedy case still ends with every piece solo (the newcomer was admitted after the toy prompt finished: timing, the replies are equal). `test_lazy_xu_engine_same_reply` fails (Xu allocated anyway under the test's default fast2): `LEAN_LAZY_XU` not used |
| 0300 `test_request_log` | 12 passed |
| regressions: overlap 149, kv_pool 23 passed; batch2 `fast_prefill_admissions` 2 failed | the batch2 failure (grid stat 64 != 128) is the same on image kvpool: a stale test expectation, not a regression |
| 0330 tc kernels (under `timeout`) | **the extension did not build**: the base image's `TORCH_CUDA_ARCH_LIST` includes 8.0, and ptxas refuses mbarrier / cp.async.bulk below sm_90. Fixed in patches/0330 (`_tc_ext` builds for the device's arch only). Then: **cfg 1 / 2 (544-thread CTAs) cannot launch on GB10** ("does not fit an SM": 17 warps x 120 registers need 5 warps x 3,840 = 19,200 registers on one SM sub-partition, which has 16,384). cfg 3 (fat's tile, 288 threads): bit-identical to fast2 / fat at 64-8,192 rows, ticket on / off, CTA cap; no hang |

**Expert bench** (`bench_experts.py 2048 4096 8192 --tc --contend`, gate/up + down ms, isolated / contended, uniform
routing; only cfg 3 runs):

| rows | fast2 | fat (prod) | tc cfg 3 | tc vs fat isolated / contended |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 11.03 / 12.99 | 13.18 / 15.24 | 11.11 / 13.62 | 1.19x / 1.12x (skewed 1.07x / 1.06x) |
| 4,096 | 40.41 / 19.76 | 15.96 / 18.10 | 18.68 / 21.30 | 0.85x / 0.85x |
| 8,192 | 30.92 / 35.05 | 27.70 / 28.79 | 35.68 / 38.85 | 0.78x / 0.74x |

All bits "same". Like fast2 in W5, tc cfg 3's 2,048-row kernel win did not survive end to end (load C below).

**Per-feature loads** (prefill tok/s, cold, two runs; every cell reply sha `8794a3463259cc2f`, exact 10/10 and
batchexact 4/4; decode = `multiturn.py` aggregate tok/s, 3 reps):

| load | 24.5k | 98k | vs A | 1 stream | 4 streams |
| --- | --- | --- | ---: | --- | --- |
| prod (kvpool, live, before the window) | 1,265.5 / 1,269.4 | 1,254.1 / 1,254.1 | +0.5% / +0.5% | 60.9 / 43.4 / 47.7 | 78.5 / 71.4 / 53.2 (a stray request) |
| A: b2 + request log (control) | 1,259.7 / 1,262.7 | 1,247.9 / 1,248.0 | base | 62.8 / 45.1 / 47.7 | 79.0 / 71.9 / 74.1 |
| B: A + `PREFILL_PP=1` + `PREFIX_SHARE=1` | 1,348.4 / 1,358.9 | 1,348.3 / 1,349.6 | **+7.3% / +8.1%** | 64.0 / 44.7 / 47.6 | 78.1 / 71.6 / 74.0 |
| C: A + `FAST_EXPERTS=tc`, `TC_CFG=3,3` | 1,210.3 / 1,210.8 | 1,198.5 / 1,192.2 | -4.0% / -4.2% | 63.9 / 45.0 / 47.4 | 79.0 / 71.3 / 73.2 |
| D: A + `PREFILL_ROWS_MAX=8192`, `SOLO_PIECE=8192`; per request 8,192 rows | 1,343.1 / 1,341.9 | 1,322.4 / 1,319.4 | **+6.4% / +5.8%** | 63.5 / 45.2 / 47.2 | 77.0 / 70.7 / 72.0 |
| D, per request `prefill_rows: 4096` | 1,310.4 / 1,309.0 | 1,295.4 / 1,275.1 | +3.8% / +3.0% | | |
| D, per request `prefill_rows: 2048` | 1,265.7 / 1,257.9 | 1,253.5 / 1,254.2 | 0 / +0.5% | | |
| **E: A + PP + PREFIX_SHARE + ROWS_MAX / SOLO_PIECE 8192 (combined)** | **1,468.4 / 1,473.4** | **1,446.9 / 1,451.1** | **+16.6% / +16.1%** | 63.1 / 45.0 / 47.0 | 77.7 / 70.7 / 72.7 |

- 0300 request log: A vs live prod -0.5% at both lengths, inside the boot-to-boot spread (the log's work runs on the
  HTTP thread after the reply and cannot touch the engine's `prefill_s`); one line a request (177 lines by load C's
  start), no text. Adopted.
- 0320 row split: +7.3-8.1%, twice the estimate (+3-5%); at 8,192-row chunks (E) it compounds with the chunk gain
  (+6.1% x +7.7% would be +14%; measured +16%). Adopted.
- 0330 tc: -4%, not adopted (and cfg 1 / 2 cannot run on GB10; a 544-thread design needs <= 96 registers a thread or
  <= 16 warps a CTA). The patch stays, off.
- Chunk size: +6.4% / +5.8% at 8,192 (W7: +5.8 / +5.4), +3.8 / +3.0% at 4,096. The per-request 2,048 cell inside load D
  equals A, so the gain is the chunk. With `SOLO_PIECE` the 8,192 pieces run only for a lone request; 4-stream decode
  keeps 2,048 pieces (aggregate within the 70-81 spread). Adopted at 8,192 via 0335 (0335's solo rule is what keeps
  pieces at 2,048 beside decoders; the piece size itself does not matter, W7).
- Decode after the prompt (`ab.py`, "count to 100") 80-89 tok/s in every load; single-stream and 4-stream aggregates
  within rep-to-rep spread everywhere.

**0310 shared prefixes** (`bench/prefixshare.py`; "12k" / "20k" arguments give ~19.4-20.3k and ~31.7k-token prompts:
the system prompt is ~18k / ~29.5k tokens; A = today, B = PP + share, E = combined):

| case | A (off) | B (on) | E (on, combined) |
| --- | --- | --- | --- |
| 12k: session 2 cached / TTFT | 16,384 / 3.25 s | **17,920 / 2.24 s** | 17,920 / 2.13 s |
| 12k: session 3 cached / TTFT | 17,152 / 2.27 s | 17,920 / 2.22 s | 17,920 / 2.11 s |
| 12k: burst of 4, wall / cached | 72.4 s / 0-896 | **28.2 s** / 3 of 4 at 17,920 (waits 8-9 rounds) | 24.8 s / 3 of 4 at 17,920 |
| 20k: session 2 cached / TTFT | 16,384 / 13.88 s | **29,376 / 2.21 s** | - |
| 20k: burst of 4, wall | 114.1 s | **35.7 s** (3 of 4 at 29,376) | - |
| resumed == fresh (draft off, fresh prefill) | 7/7, 7/7 | 7/7, 7/7 | 7/7 |

The expected cached tokens (system prompt rounded down to 64 and the template tokens) appear, and replies are identical.
Adopted.

**Combined config E: worst case and quality** (`results/W8/runE2.sh`, after E's gates and prefix bench, so the session
store and allocator caches were already warm):

- 4 x ~250k stress (`multiturn.py --modes stress --stress-target 250000`; decoders at 252.8k, the 4th prefilling 32k):
  filled in 845 s, final TTFT 49.0 s, longest decode gap 4.2 s; **MemAvailable minimum 9.23 / 8.14 GiB** (head /
  worker; final phase 9.45 / 8.42) -> PASS (>= 8), no OOM in dmesg, health 0 errors. The margin on the worker node is thin:
  W6's 2,048-row stress had 14.05. Of the ~6 GiB, 2.3 is the 8,192-row lean buffers; the rest is the drift of a load
  that has served (session store full, allocator caches after 98k-300k prompts: every load in this window drifts from
  ~18 / 17 after boot to ~13 / 12 at 2,048 rows and ~11 / 10 at 8,192). If memory ever runs short, the first step back
  is `PREFILL_ROWS_MAX` / `SOLO_PIECE` 4,096 (+0.8 GiB instead of +2.3, keeps +3-4%).
- Needle, one prompt of 314,262 tokens (the "350k" argument of `needle.py`) alone after the stress (pool holding the
  stress sessions, solo 8,192-row chunks): **found**, prefill 1,287 tok/s (W6 at 358k: 1,035), decode 58.9 tok/s; the
  follow-up resumed 314,240 cached (0.25 s). MemAvailable minimum during it 9.28 / 8.35.
- MMLU-200 **88.0%** (176/200, same as W6), refusals 0/10; exact 10/10 and batchexact 4/4 again after the stress.

**Adopted:** `config/prod.env` -> `IMAGE=glm53-tensorfold:b2`, `GLM53_TF_PREFILL_PP=1`, `GLM53_TF_PREFILL_ROWS_MAX=8192`,
`GLM53_TF_SOLO_PIECE=8192`, `GLM53_TF_PREFIX_SHARE=1`, `GLM53_TF_REQUEST_LOG=/sessions/requests.jsonl` (fat experts
stay). Prod restarted 19:14 on it: ready in ~1 min, canary ok (decode 76.3 tok/s), warm-up 16,384 prefill 10.98 s
(kvpool: 12.57 s), https `/v1/models` ok, `17*23` -> `391`, watchdog timer re-armed, lease refresher and memory
sampler stopped, lease deleted. MemAvailable idle after start 15 / 13 GiB. The NVMe session store starts cold (the
image id is in 0250's compat hash).

Ops notes: `pkill -f results/W8/lease.sh` inside an `ssh '...'` command also matches (and killed) that ssh's own shell;
kill by pid instead. The canary's tokens-a-round can differ between loads (5.00 vs 5.71 on load E: the drafter cost
calibration at load differs by ~0.2 ms); its replies are exact.

## W9: batch 3 (0310 IMA, memory margin, 0350 RoCE, 0360 b12x bit 4, per-slot verify cost), 2026-09-28 (19:23-20:26, 20:36-21:20, 21:30-22:25, 22:35-23:26) — RoCE, b12x bit 4 and 4,096-row chunks adopted

Image `glm53-tensorfold:b2` throughout (no rebuild: only test harnesses changed). Loads from `config/prod.env` plus
overrides (`results/W9/load.sh`). Four windows, prod restored and verified in between (https, `17*23` -> `391`,
canary, watchdog timer active, lease deleted). Files: `results/W9/` (scripts, logs, JSONs; `SUMMARY` files per step).

### 1. The 0310 illegal memory access (W8): not a production bug; PREFIX_SHARE stays on

`tests/cuda/test_prefix_share_patches.py`, whole file, 22 runs in W9:

| condition | runs | IMA |
| --- | ---: | ---: |
| warm caches (the worker node) | 6 | 0 |
| cold Triton cache (`TRITON_CACHE_DIR` in the container, 43 s runs) | 3 | 0 |
| fresh empty `/cache` volume (extensions, Triton, calibration built in the run), alone, `CUDA_LAUNCH_BLOCKING=1` | 1 | 0 |
| fresh volume after W8's crashed two-engine test (0320 `test_two_engines_one_gpu` at a08d7d6, two processes on one GPU), CLB | 1 | 0 |
| fresh volume after that test, no CLB | 4 | 0 |
| copy of the warm prod cache minus the kda / exl3 extension builds, after that test | 4 | 0 |
| **W8's order on the head node's own cache: the crashed two-engine test (which rebuilt the kda / exl3 extensions), then the file** | 3 | **1 (the first)** |
| memcheck (`compute-sanitizer`, GPU tests, warm) | stopped after 7 min to free the GPU | - |

The one reproduction (`results/W9/tests-head-seq/seq1-prefix.log`) is W8's failure exactly: the same three tests,
the same place (test 29, the first engine built after the four `test_gpu_new_session_resumes_at_the_system_end`
cases: a reference engine with prefix share **off**, IMA surfacing at the synchronize after `Graphs`' eager warm-up
in `GlmEngine.__init__`), 28 passes before it. Both failing runs (W8, W9 seq1) were the first run of the file after a
crashed two-process test and both spent ~50 s in the container before pytest started (63 s wall vs 10.6 s in pytest;
W8: 124 vs 71); every passing run spent 3-4 s. A hypothesis that a warm-up kernel reads an uninitialized device
buffer as an index (recycled memory from earlier engines in the process) was tested directly: device memory poisoned
with 0xFF / 0x7F (60 GiB, freed back) before building the test's engines, 3 rounds each, under CLB -> no fault
(`results/W9/poison_engine.py`, `ima/p1.log`, `ima-worker/p2.log`).

Verdict: an artifact of the test process after a crashed two-process GPU run, not of the production path. The fault
is in an engine with 0310 disabled, in engine construction, in a process that had already built and freed ~20
engines (production builds one engine per process and restarts clean: every start in W6-W9 passed its canary), and
it appeared only right after a crashed two-process run: 2 of the 13 runs that followed one (W8 + W9), 0 of the 12
without one (W9's 10, W8's 2 reruns). The faulting kernel could not be named (the two
reproductions were without CLB; the 11 other post-crash runs, 1 with CLB, did not), so a latent engine-construction bug
cannot be excluded completely, but nothing points at the live 0310 path. `GLM53_TF_PREFIX_SHARE=1` stays.

### 2. Memory margin: under 8 GiB today at 8,192-row chunks -> 4,096 (W8's documented step back)

| when | head MemAvailable | worker MemAvailable |
| --- | ---: | ---: |
| prod idle, 9 min after W8's 19:14 start (before W9) | 16.05 GiB | 14.62 GiB |
| prod idle, 10 min after the W9 window-1 restore / at the window-4 start | 15.52 / 15.62 GiB | 14.08 / 14.32 GiB |
| minimum in window 2 (loads A, C, B incl. 98k prompts, 60k sessions) | 10.32 GiB | 9.05 GiB |
| 4 x 250k stress, F = prod + RoCE + b12x (8,192 rows) | 8.89 | **7.82** |
| 4 x 250k stress, G = prod + RoCE (8,192 rows) | 8.91 | **7.83** |
| 4 x 250k stress, **H = prod unchanged** (W8 config, 8,192 rows) | 8.44 | **7.21** |
| 4 x 250k stress, **I = prod + RoCE + b12x + 4,096 rows (adopted)** | **9.72** | **8.67** |

Each stress ran after the same ~10 min of gates (exact, batchexact, 24.5k / 98k twice, 10 concurrent reps), as W8's E
(9.23 / 8.14). The production config itself now bottoms out under 8 GiB on the worker node (H: 7.21), so W8's 8.14 was a
thin pass, not a stable margin; RoCE (a ~1.5 MB pinned region) and b12x do not account for it (G == F; both above H).
W8's documented first step, `PREFILL_ROWS_MAX` / `SOLO_PIECE` 4,096 (lean chunk buffers 1.55 instead of 3.10 GiB a
rank), gives +1.45 GiB on the worker node over H and passes. `scripts/gpuwatch.py` has no memory threshold (clocks, power,
the slow state); the memory guards are the engine's (`BATCH_RESERVE_GB=11`, `SESSION_RESERVE_GIB=6`, `ADMIT_GB=2`)
and `MEM_GATE_GIB=108` (MemFree at start); unchanged, they still fit (idle after start ~15.5 / ~14.3 GiB).

### 3. 0350 RoCE: every stage clean after four harness fixes; +4-11% decode, adopted

Details in docs/ROCE-FIX.md "W9". Summary:

- **W7's 1 MiB mismatch was a harness race**, not a chunk / slot boundary: `loopback` built its inputs on the default
  stream and gathered them on non-blocking side streams without a wait. With the race, 512 KiB-4 MiB first gathers
  differ in 19-20 of 20 trials, **including the rank's own shard** (copied from its input, never on the wire); 16k-256k
  in 0 of 20. With a synchronize: no difference in 20 trials at 16k / 128k / 4 MiB and 320 trials each at 256 KiB-2
  MiB. The engine gathers on the producing stream, so it cannot hit this. Fixed loopback: `bits_equal: true` at 1
  MiB. One unexplained difference: a second gather of fully written 2 MiB inputs, once, not again in 360 more (above
  the engine's 256 KiB).
- Three more harness bugs found and fixed in `tests/cuda/bench_roce.py` (`stress --loop` host deadlock on a
  first-time allocation; `fault` launching rank 1's "next" op at once; `soak` graphs whose inputs were freed, and ranks
  stopping on their own clocks).
- Preflight: `PCI_WR_ORDERING = per_mkey(0)` on all four functions. GPU unit tests 25/25.
- Stage 1-2 (one node): loopback both functions, `HCAS=1`, each function alone: bits equal 16k-1m. `stress --loop`
  100k ops per size and mode (16k, 128k; eager, graph), striped and per function, plus 20k at 256k / 1m: 0 mismatches.
- Stage 3 (two nodes): `bench` bits equal at 16k / 64k / 128k / 1m, graph latency RoCE 11.7 / 16.9 / 20.3 / 77.7 us vs
  NCCL 45.3 / 76.1 / 66.1 / 204.3 us (a 90-exchange step saves 3.0 ms at 16k); `fault`: rank 0 raises after 3.00 s,
  `never`, rank 1's next exchange completes; `stress` 100k a size and mode: 0 mismatches; `soak` 20 min (~69.5M ops a
  rank, every replay compared) + 3 min after the end-condition fix (113,664 replays): clean.
- **Engine A/B** (load A = prod, NCCL; load C = A + `GLM53_TF_COMM_BACKEND=roce`, 256 KiB max over both functions):

| check | A (NCCL) | C (RoCE) |
| --- | --- | --- |
| exact / batchexact | 10/10, 4/4 | 10/10, 4/4 |
| transcripts (6 prompts x greedy / sampled seed 1234, alone; 4 together) | reference | **12/12 and 4/4 byte-identical** |
| decode 1 stream, 5 reps (tok/s; each rep its own prompt) | 63.8 / 45.2 / 47.2 / 91.8 / 39.3, median 47.2 | 66.4 / 48.1 / 52.3 / 96.7 / 41.5, median **52.3 (+10.8%)**; per rep +4.0 to +10.7% |
| decode 4 streams, 5 reps (aggregate) | 78.2 / 71.5 / 73.2 / 72.9 / 69.1, median 72.9 | 81.5 / 74.8 / 76.1 / 76.0 / 72.3, median **76.0 (+4.3%)**; per rep +3.9 to +4.6% |
| round time at equal tokens / round | 1 stream 55-77 ms, 4 streams 98-148 ms | 3-5 ms shorter a round (1 stream), ~3 ms (4 streams) |
| load-time calibration: verify 1 row / DFlash2 block | 31.8 / 3.74 ms | 30.4 / 3.11 ms |
| RoCE failure / fallback lines | - | none |

Adopted (rule: every stage clean, identical transcripts, decode >= +3%): `GLM53_TF_COMM_BACKEND=roce`,
`GLM53_TF_ROCE_MARK=/cache/roce-failed` (a run-time failure pins the next start to NCCL). The stale W2 marker
(`/cache/roce-failed` on the head node, 09:59) and the markers W9's own harness failures wrote were saved
(`results/W9/roce/roce-failed-marker-head.txt`, `roce-marks-before-*.txt`) and deleted.

### 4. 0360 b12x bit 4 under FP8: +6.3% / +6.7% prefill, adopted

- GPU tests (`test_b12x_attn_patches.py`): 23 passed, including FP8 == dequantized at 2,051 tokens (W3's failure),
  rows / subsets / permutations, engine resumed == fresh with cached > 0, bits never cross. Kernel: FP8 one pass 5.01-
  5.40 ms vs chunked 9.08-9.24 ms (1.71-1.81x), bf16 1.40-1.50x. The 0240 subset: 5 of 6; the failure is
  `test_engine_snapshots_never_cross_bits`' control for **bits 3** (0240's KDA / hc kernels, not adopted; the
  bit-4 path has its own passing test), recorded, not pursued.
- Load B = prod + `GLM53_TF_B12X=4`: exact 10/10, batchexact 4/4; sessions (3 conversations x 3 turns, ~60k-token
  prompts, `tf_knobs.b12x: 4`): every follow-up cached 60,864 and the same reply sha as the same request cold
  (`draft: false`), 6/6; 4 concurrent == alone 4/4.
- Prefill A/B in the same load, per request `b12x` 4 vs 0, 3 runs (cold tok/s):

| prompt | b12x 0 | b12x 4 | gain | reply sha (all cells, cold and warm) |
| --- | --- | --- | ---: | --- |
| 24.5k (21,464 tokens) | 1,475.1 / 1,469.4 / 1,471.3 | 1,566.4 / 1,564.1 / 1,561.6 | **+6.3%** | 8794a3463259cc2f |
| 98k (85,782 tokens) | 1,452.1 / 1,450.3 / 1,373.6 | 1,547.2 / 1,546.1 / 1,548.3 | **+6.7%** | 8794a3463259cc2f |

Decode unchanged (after the prompt 72.7-89.2 tok/s both ways; concurrent 1 / 4 streams 63.5 / 45.1 / 47.5 and 78.4 /
71.3 / 73.1 vs load A's 63.8 / 45.2 / 47.2 and 78.2 / 71.5 / 73.2). Adopted: `GLM53_TF_B12X=4`.

### 5. Per-slot fixed verify cost (ADAPTIVE-DRAFT.md §6 step 2): measured; the lever exists

Load A, `bench/multiturn.py --modes concurrent --streams 1,2,4 --reps 3 --long-tokens 512 --extra '{"draft": false}'`
(serial: 1 row a slot a round), and the same with drafting (5 reps):

| slots x rows a round | ms / round (3 reps) | aggregate tok/s |
| --- | --- | --- |
| 1 x 1 (serial) | 33.0 / 33.1 / 33.2 | 29.2-29.6 |
| 2 x 1 (serial) | 45.5 / 45.5 / 45.5 | 42.5-42.6 |
| 4 x 1 (serial) | 69.3 / 69.0 / 69.5 | 55.5-55.9 |
| 1 x ~8 (drafting, mostly DFlash2 blocks: rep 3, 7.2 tokens a round) | 71.6 verify + 5.8 drafting | 91.8 |
| 1 x mixed (drafting, 2.2-2.9 tokens a round) | 50.4-54.1 verify + 5.1-5.9 drafting | 39.3-47.2 |

So a round is ~21 ms plus **12.1 ms per slot-row**; going from 1 to ~8 rows in one slot adds ~38 ms (~5.5 ms a row,
mostly expert reads), which leaves **~6.6 ms fixed per slot** (the fit's 7 + 6 holds: predicted ~31 / ~44 / ~70 ms,
measured 33 / 45.5 / 69.3; the "cheap slots" case, ~31 / ~38 / ~50, is ruled out). In a 4-stream round (98-148 ms,
~115 typical) the fixed per-slot part is ~26 ms (~23%); drafting is 4-5.5 ms a round. The dominant term is still
per-row work (routed experts), but the per-slot fixed cost is real and large enough that halving it (fused per-slot
KDA / attention / commit across slots) is worth the simulator's **~+6%** at 4 streams. Measurement only.

### Combined gates (windows 3 and 4)

Each load: `results/W9/gates.sh` (`w3.sh` / `w3g.sh` for F / G) = exact, batchexact, `ab.py` 24.5k / 98k twice,
concurrent 1 / 4 streams x 5 reps, 4 x ~250k stress (`multiturn.py --modes stress`), MMLU-200 + refusals, exact /
batchexact again, dmesg OOM count. Decode vs load A (window 2, same prompts per rep; H is the same-window control):

| load | exact / batchexact (before, after stress) | reply sha (8 cells) | prefill 24.5k / 98k tok/s | decode 1 / 4 streams, median of 5 (vs A) | stress MemAvailable min head / worker | MMLU-200, refusals | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F: RoCE + b12x 4 (8,192 rows) | 10/10, 4/4; 10/10, 4/4 | 8794a3463259cc2f | 1,575 / 1,555 | 50.7 (+7.4%) / 74.5 (+2.2%) | 8.89 / **7.82** | 88.0%, 0/10 | **fail (memory)** |
| G: RoCE (8,192 rows) | 10/10, 4/4; 10/10, 4/4 | 8794a3463259cc2f | 1,465 / 1,445 | 50.1 (+6.1%) / 75.0 (+2.9%) | 8.91 / **7.83** | 88.0%, 0/10 | fail (memory) |
| H: prod unchanged | 10/10, 4/4 | 8794a3463259cc2f | 1,473 / 1,450 | 46.1 (-2.3%) / 73.4 (+0.7%) | 8.44 / **7.21** | - | fail (memory) |
| **I: RoCE + b12x 4 + 4,096 rows** | **10/10, 4/4; 10/10, 4/4** | **8794a3463259cc2f** | **1,499 / 1,496** | **50.5 (+7.0%) / 76.1 (+4.4%)** | **9.72 / 8.67** | **88.0%, 0/10** | **pass: adopted** |

I against H (same window, same prompts): prefill +1.7% / +3.2%, decode per rep +4.3 to +9.5% (mean +6.6%) at 1
stream and +3.4 to +5.1% (mean +4.0%) at 4 streams. OOM kills 0 / 0 in every load; the stress filled in 806-847 s,
longest decode gap 3.8-4.0 s. (b12x's +6.5% at 8,192 rows mostly pays for the chunk step back: 8,192 -> 4,096 costs
~2.6% in W8.)

**Adopted:** `config/prod.env` -> `GLM53_TF_COMM_BACKEND=roce`, `GLM53_TF_ROCE_MARK=/cache/roce-failed`,
`GLM53_TF_B12X=4`, `GLM53_TF_PREFILL_ROWS_MAX=4096`, `GLM53_TF_SOLO_PIECE=4096`; image `glm53-tensorfold:b2` (tagged on
both nodes, unchanged). Prod restarted on it 23:26: ready in 22 s, canary ok (decode 77.7 tok/s; warm-up 16,384 prefill
10.54 s, was 11.0), RoCE connected on both ranks (256 KiB, both functions), lean set 4,096 rows (1.55 GiB), https
`/v1/models` ok, `17*23` -> `391`, watchdog timer active, lease refresher stopped, lease deleted. First hour on RoCE
watched every 5 min (`results/W9/prod-watch.log`): 23:26-00:26, 13 samples: 0 RoCE failure / fallback lines on either rank, no marker, health errors 0 over 101 requests; light
traffic every 5 min (`results/W9/prod-traffic.log`: 1 stream 64.2-69.1 tok/s, 4 streams 78.4-79.4 tok/s, 256 tokens) and
`exact` 10/10 on the live server at the start and at the end; MemAvailable 17.39 / 16.07 GiB after start, 14.87 / 13.59
after the hour (allocator and session-store warm-up, as in W8).

Ops notes: `pkill -f <script>` inside an `ssh '...'` command killed that ssh's own shell again (W8's note); kill by
pid. A local tmp quota error lost one launching command's output (window 3 had started; the windows.log line was added
afterwards). The b12x bits-3 test and the 2 MiB single mismatch are the open ends.

## W10: batch 4 (0390 MLA expand v2, 0400 KDA v2, 0410 sparse v2, 0370 decode overlap / CPU pin, 0380 deep verify, 8,192-row chunks), 2026-09-29 (00:36-01:30, 01:41-02:27, 02:38-03:22, 03:33-04:25) — 0390, 0370 overlap and 0380 16 rows adopted

Image `glm53-tensorfold:b4` = every patch through 0410, built on the head node and loaded on the worker node (tagged on both). Two
patch fixes made during window 1 went into the final b4 (built three times; the first two never served):

- **0410 had a syntax error** in the `engine.py` hunk: the load-time message split an f-string expression across two
  string literals (`... else 'in "` / `f"shared memory'}`). `engine.py` did not import, so every engine-level test
  of the first b4 failed (`SyntaxError: unterminated string literal`, line 303) and **no load could have started**.
  The kernel-level tests import `sparse_v2` / `latent` / `kda_v2` directly and passed. Fixed in
  `patches/0410-glm-sparse-v2.patch` (same hunk sizes); `compileall` over the installed package is clean.
- **0390's tile table** (`latent.V2_TILES`) retuned from the GPU sweep (`bench_mla_expand.py --sweep`): 4 warps
  instead of 8 everywhere, absorb 64 rows x k 32 (was 128 x 16), expand 64 x 16 with BN 64 unchanged. Speed only;
  the bitwise test covers every forced tile and was re-run on the final image (50 passed).

**Off == production** on the final b4: `tests/kvpool_ptx.py` against b2 **33 of 33** Triton kernels' PTX identical;
the patches' own compile checks in the image (0390 21, 0410 36, 0400 17 passed: v1 / one-pass / fast_kda PTX
unchanged); load P0 (b4, no new knob) gives b2's reply sha `8794a3463259cc2f`, exact 10/10, batchexact 4/4, session
follow-up resumed == cold, prefill 1,500 / 1,503 tok/s (W9 load I: 1,499 / 1,496). Files: `results/W10/`
(`tests-head*/SUMMARY`, `tests-worker*/SUMMARY`, `summ.py` prints a line a load).

### 1. Kernel bitwise tests and microbenches (GPU, both nodes, everything under `timeout`)

| test | result |
| --- | --- |
| 0390 `test_mla_expand_patches.py` | **50 passed** (first b4 and final tiles): v2 == v1 bit for bit, q4 / q4mse / bf16 kv_b, 1-8,192 rows, windows, every forced tile, in a CUDA graph; control differs |
| 0390 regressions with `GLM53_TF_MLA_EXPAND=v2` | `test_latent_patches.py` 20 passed; `test_kv_pool_patches.py` / `test_fp8_kv_patches.py -k latent`: 2 failed, **the same 2 with v1** (`test_fp8_latent_close_to_expanded_reference`: 0.0566 > 0.04 tolerance, identical numbers both ways: a stale tolerance, not 0390) |
| 0400 `test_kda_v2_patches.py` | **345 passed** (every mode / setting == fast_kda, 50 repeats, busy stream, resume == fresh, engine entry); `test_lean_patches.py` with `KDA_V2=1` 53 passed |
| 0410 `test_sparse_v2_patches.py` | first run 27 passed, 6 failed: **all 6 `OutOfResources` (133,120 B of shared memory > 101,376) in bf16 cases**, none a bit difference: the test's `_v2` forced FP8's 3 stages on bf16 caches, which `sparse_v2.config` itself refuses. Test fixed (format default; unfit settings skipped): **30 passed, 3 skipped**; every FP8 (production) case passed both times. `test_b12x_attn_patches.py -k "one_pass or fp8"` 3 passed |
| 0370 `test_decode_overlap_patches.py` | 29 passed (incl. the affinity test on the Spark's cores); `test_batch_parallel_patches.py` with `DECODE_OVERLAP=1` 26 passed; `test_batch_sessions_patches.py` with it: 26 passed, 2 failed (`test_follower_replays_rank0_sessions`: the test records rank 0's `_share` calls only and replays them to a one-GPU follower; with `plan` the plan rides on the sampler exchange instead, so the replay is out of step: `parse_plan([1, 0])`. A harness limit; 0370's own lockstep test replays both kinds and passes, and the two-rank engine loads below are exact, with cancels) |
| 0380 `test_deep_verify_patches.py` | 4 passed + the 16-row child: 17 passed, 2 failed, both `assert deepest >= 9` in `test_gpu_drafted_replies_equal_serial` (the synthetic checkpoint never drafted deeper than 2: a coverage assertion; the replies equalled serial before it). Dense 1-16 == serial and long-context / MTP 1-16 == eager passed. The engine evidence is below (identical reply hashes with 16-row windows in use) |

Microbenches (one GPU each, prod stopped):

| kernel | today | new (best bit-identical setting) | a 512-row sub-block | projected prefill | plan gate |
| --- | --- | --- | --- | --- | --- |
| 0390 absorb + expand, 512 rows | 677 + 2,560 us | 346 + 886 us (final tiles; offline tiles: 399 + 1,543) | **0.38x** | +6.3% (43 us a token) | <= 0.5x: **pass** |
| 0390 at 1 / 8 rows (decode) | absorb 28 / 26, expand 114 / 105 us | 25 / 25, 68 / 68 us | 1.1x / 1.5-1.7x | decode +~1% | - |
| 0400 split (bv 32, 4 warps, maxnreg 168) | fast_kda 0.913 ms | 0.845 ms | 0.93x | +0.7% (4.5 us a token) | <= 0.85x: **fail** |
| 0400 fused (every setting) | 0.913 ms | 1.25-4.0 ms | 0.23-0.87x (slower) | negative | fail |
| 0410 (FP8, overlapping lists, 10.7k / 85.8k ctx) | one pass 2.41 / 9.23 / 36.3 ms (512 / 2,048 / 8,192 rows) | 2.03 / 7.54 / 29.7 ms (cfg 2,1,1; default 3,1,1 2-3% slower) | 0.82x | +1.3-1.6% (random lists at 85.8k: +2-3%) | <= 0.7x: **fail** |

The KDA and sparse kernels are far from their offline estimates (0400: 0.93x, estimated 0.85x split and 0.5-0.7x
fused; 0410: 0.82x, estimated 0.32-0.55x). Every timed setting was bitwise equal.

### 2. Engine A/B on the production config (per knob, then combined)

Every load: `config/prod.env` + overrides on b4 (`results/W10/load.sh`), then `run.sh` = exact, batchexact, a session
follow-up (1 conversation x 3 turns over ~61k tokens: turns 2-3 cached 60,864 with the cold request's reply sha, plus 4
batched == alone), `ab.py` 24.5k / 98k twice (cold prefill, reply sha), concurrent 1 / 4 streams x5 (aggregate tok/s,
the same 5 prompts every load).

| load | prefill 24.5k (2 runs) | 98k (2 runs) | vs P0 | reply sha | exact / batchexact / session | 1 stream median / 4 streams median | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0: b4, no new knob (control, window 1) | 1,501 / 1,500 | 1,504 / 1,502 | base | 8794a3463259cc2f | 10/10, 4/4, OK | 52.1 / 75.8 | == prod (W9 I 1,499 / 1,496) |
| **M: `MLA_EXPAND=v2`** | 1,611 / 1,604 | 1,608 / 1,599 | **+7.1% / +6.7%** | same | 10/10, 4/4, OK | 51.3 / 76.4 (per rep +1.7, +1.0, -1.5, +3.3, +1.0 / +1.0, +0.1, +1.6, +0.8, +1.2%) | **adopt** |
| K1: `KDA_V2=1` (split) | 1,518 / 1,518 | 1,517 / 1,518 | +1.2% / +1.0% | same | 10/10, 4/4, OK | 50.4 / 75.5 | < +2%: off |
| S: `SPARSE_V2=1` (window 2) | 1,498 / 1,503 | 1,502 / 1,504 | 0.0% / 0.0% | same | 10/10, 4/4, OK | 50.4 / 75.5 | < +3%: off |
| MSK: M + S + K1 | 1,631 / 1,632 | 1,626 / 1,635 | +8.7% / +8.5% (+1.5% over M) | same | 10/10, 4/4, OK | 52.6 / 76.8 | K and S each below their bar |
| M8: M + 8,192-row lone chunks (gates without MMLU) | 1,677 / 1,676 | 1,659 / 1,665 | +11.7% / +10.5% | same | 10/10, 4/4 | 51.8 / 77.0 | **stress 8.32 / 7.23 GiB: fail** |

KDA v2 fused (mode 2) was not loaded: every fused setting was slower than fast_kda in the bench.

**8,192-row chunks (item 3).** The prefill winner does not change memory, so the W9 finding stands: M8's 4 x ~250k
stress bottomed at **8.32 / 7.23 GiB** (head / worker; W9's H at 8,192: 8.44 / 7.21), under the 8 GiB floor on
the worker node, no OOM (0 / 0), longest decode gap 3.9 s. 4,096 stays. (8,192 would add another ~+4% on top of M.)

### 3. Decode (window 3, all on top of M; MA = M again as the same-window control)

Decode set a load (`dec.sh` / `deep.sh`): exact, batchexact, the W9 transcripts (6 prompts x greedy / sampled alone,
4 together; all == W9 load A's hashes in every load: 0390 keeps the bits), concurrent 1 / 4 streams x5; the 0380
loads also `glmbench --suites tf,tweet,kit,edit --reps 3 --long-tokens 512`.

| load | 1 stream per rep vs control (median) | 4 streams per rep vs control (median) | other | result |
| --- | --- | --- | --- | --- |
| D: `DECODE_OVERLAP=1` vs MA | +1.9 +1.2 +1.3 -0.5 +2.1% (53.0 -> 53.7, +1.3%) | +0.1 +0.7 0.0 +1.7 +2.6% (76.0 -> 77.3) | transcripts identical, exact 10/10, batchexact 4/4 | |
| D2 (repeat) vs MA2 (control repeat) | +1.8 +2.5 +1.9 +1.8 +1.7% (52.9 -> 53.9, +1.9%) | 0.0 +4.1 +1.6 -0.4 +1.2% | same; **cancel check** (3 decoding + 1 streamed client gone after 50 chunks) x2: the 3 == alone, request log `finish: cancelled`, next request fine | **adopt** |
| C: `CPU_PIN=auto` vs MA | +1.2 +0.4 +0.4 -1.0 +0.9% (+0.4%) | -0.4 +0.7 -0.4 +2.1 +0.3% | pinning as planned on both ranks (`tf-serve` cpu 19, RoCE proxy 18, `NCCL Progress` 16-17, rest 0-15; 40 samples at 0.5 s) | < +1%: off |
| MA2 vs MA (control noise) | +0.9 -0.6 -0.2 -1.1 +0.2% | +0.4 -0.7 -0.9 +1.3 +0.7% | | |
| R16: `MAX_DRAFT_ROWS=16` vs MA | +3.5 +0.2 -0.9 -0.3 +0.9% (53.0 -> 52.5) | 0.0 +0.5 -2.2 +1.2 +2.1% (76.0 -> 75.8) | calibration lists 16 windows (9-16: 75.2-101.6 ms, ~3.8 ms a row) | see below |

0380 glmbench cells (tok/s, MA -> R16): **edit-rename 89.8 -> 111.2 (+23.8%), edit-comments 81.8 -> 94.7 (+15.8%),
edit-print-to-log 91.1 -> 115.5 (+26.9%)**; code / chat T=1 +1.2 / +0.9%, code / chat T=0 +1.4 / -0.4%, sequence -0.2%,
code 512 -0.3%, json +0.3%, hashmap +1.9%, structured +0.9%, essay +1.6% (all within +-2%). **Every reply hash of
all 13 cells x reps is identical between MA and R16**, including the edit cells, where the 9-16-row windows ran, and the
sampled T=1 cells: the deeper windows keep the bits. Adopted (rule: edit >= +10%, everything else in noise).

0370 nsys (plan §4) was not captured: no window time left after the gates; the decode A/B carries the adoption.

### 4. Combined candidate FIN = prod + `MLA_EXPAND=v2` + `DECODE_OVERLAP=1` + `MAX_DRAFT_ROWS=16` (b4, 4,096 rows): full gates

`results/W10/gates.sh FIN` (W9's gate set) + `w4.sh` (needle, edit cells, cancel):

| gate | FIN | bar |
| --- | --- | --- |
| exact / batchexact, before and after the stress | 10/10, 4/4; 10/10, 4/4 | 10/10, 4/4 |
| reply sha, 24.5k / 98k x2 | 8794a3463259cc2f (all cells) | same |
| prefill 24.5k / 98k (tok/s) | 1,614 / 1,602; 1,577 / 1,606 (**+7.1% / +5.9%** vs P0; the 1,577 run is the low one) | - |
| decode 1 stream, per rep vs P0 / MA / MA2 (mean) | +3.7% / +1.1% / +1.3% (median 52.1; reps 71.1 / 49.8 / 52.1 / 97.5 / 43.2) | not lower |
| decode 4 streams, per rep vs P0 / MA / MA2 (mean) | +3.1% / +2.5% / +2.4% (median 78.1) | not lower |
| 4 x ~250k stress MemAvailable min, head / worker | **10.49 / 9.36 GiB** (W9 I, today's prod: 9.72 / 8.67); filled in 770 s, longest decode gap 3.9 s | >= 8 |
| OOM kills (dmesg, both nodes; after the stress and after the needle) | 0 / 0; 0 / 0 | 0 |
| MMLU-200, refusals | **88.0%** (176/200), 0/10 | >= 87% |
| needle, 314,249 tokens alone after the stress + MMLU | **found** cold and resumed (cold prefill 1,376 tok/s, W8 1,287; resume 314,240 cached, 0.19 s) | found |
| edit cells (glmbench `edit`, 3 reps) | 115.3 / 97.1 / 116.5 tok/s | - |
| cancel (3 decoding + 1 client gone after 50 chunks) | 3 == alone, `cancelled`, next request fine | - |
| /health | 340 requests, 0 errors | - |

**All gates pass.** One observation outside the gates: **MemAvailable during the needle** (sampled every 2 s,
after the stress and MMLU, with the pool holding the stress sessions) **bottomed at 8.55 / 7.39 GiB**, under 8 on
the worker node, with no OOM. A same-window control on today's production config (H2 = b2, W9 knobs, `w4b.sh`: stress then
the needle on a fresh start, without the gates before) gave 12.85 / 11.68 (stress) and 11.22 / 10.36 (needle), but
it is not a like-for-like baseline: W9 measured ~3 GiB of drift from the gates that precede FIN's stress (FIN's own
stress minimum, after them, was 10.49 / 9.36 and the highest yet), and the extra FIN-only memory is small (0380: ~115
MB a rank; 0390 / 0370: none). W8's needle after its stress (no MMLU between) bottomed at 9.28 / 8.35 with 8,192-row
buffers. Recorded as an open end: the needle-after-stress+MMLU minimum on the W9 config was never measured.

**Adopted:** `config/prod.env` -> `IMAGE=glm53-tensorfold:b4`, `GLM53_TF_MLA_EXPAND=v2`, `GLM53_TF_DECODE_OVERLAP=1`,
`GLM53_TF_MAX_DRAFT_ROWS=16` (4,096-row chunks, everything else as W9; the previous file is
`results/W10/prod.env.before-W10`). Not adopted, knobs off: `GLM53_TF_KDA_V2` (+1.0-1.2%), `GLM53_TF_SPARSE_V2` (+0%),
`GLM53_TF_CPU_PIN` (+0.4%), 8,192-row chunks (memory). Prod restarted on it at 04:25: ready in 23 s, canary ok (decode
80.1 tok/s; warm-up 16,384 prefill 9.90 s, was 10.6), both ranks print the 0390 / 0370 lines, drafter costs list 16
windows, RoCE connected (no marker), https `/v1/models` ok, `17*23` -> `391`, live `exact` 10/10, watchdog timer
active (its 04:26 run exited 0), lease refresher stopped, lease deleted. MemAvailable idle after start 16.84 / 15.49 GiB.
The NVMe session store starts cold (new image id).

Ops notes: window 2's restore ran twice (a stray `restore.sh` without a name before the named one: two prod starts,
02:26:50 and 02:27:32, both verified; `windows.log` line "w restored"). The 0370 nsys capture and the plan's arrival-latency
check were not run. `pgrep -f` inside the ssh commands matched the ssh's own shell again (harmless here: used for
listing only).
