# Results

> **Work in progress.** Measured on one pair of DGX Sparks, 2026-09-27/28, with the **abliterated** checkpoint below.
> The sections are in the order the work happened; the newest ones ("Stacked run and production config" for
> single-stream, "4 x 256k batch production" for the batch config) are the current state. The public repo carries
> the benchmark JSON of the runs in `results/` (E*, F*, L*, M*, P*, Q*, A1, B1, B2, S1, X1, Y1, Z1, Z2); logs, test
> output and some runs (B3, K0, K1, P1, T*) are summarized here only. Hosts in the JSON were normalized to
> `127.0.0.1`. Config names refer to the `config/*.env.example` files.

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
