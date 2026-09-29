Follow me on X for more updates: https://x.com/jayleaton

# GLM-5.3-Flash on TensorFold, 2x NVIDIA DGX Spark

Serve GLM-5.3-Flash (the abliterated EXL3 4-bit checkpoint
[`neko-legends/GLM-5.3-Flash-Uncensored-EXL3`](https://huggingface.co/neko-legends/GLM-5.3-Flash-Uncensored-EXL3))
across two NVIDIA DGX Sparks, tensor-parallel over the 200 Gb/s CX7 link, behind an OpenAI-compatible API. The
engine is [TensorFold](https://github.com/ashhart/TensorFold) (pinned, unmodified submodule) plus 57 patches applied
at image build: 4-bit non-expert weights, a latent (absorbed MLA) KV cache, fast chunked prefill, a multi-session
state cache with an NVMe tier, batching of up to 4 requests over a shared 1M-token KV pool, FP8 KV storage,
shared system-prompt reuse, a RoCE all-gather, fast restarts, deeper drafting and verify windows, and ops tooling.
Every patch is off by default; the configs in `config/` turn on the measured set, and
[`docs/PATCHES.md`](docs/PATCHES.md) records the ones that were measured and not adopted.

> **Work in progress.** This is an experimental setup, measured on one pair of Sparks. Knobs, defaults, APIs and
> numbers may change between commits. Read [Limits](#limits-and-negatives) before relying on it.

Weights attribution (the checkpoint's license requires it): the weights are by **Local Inference Lab, Inc.**
(<https://local-inference-lab.ai/>), upstream source
<https://huggingface.co/neko-legends/GLM-5.3-Flash-Uncensored-EXL3>, under the ShapleyMCG License 1.0. They are not
included here. See [Licensing](#licensing).

SPDX-License-Identifier: Apache-2.0 (this project's own code, scripts, benchmarks and docs; see [Licensing](#licensing)).

## Contents

- [Benchmarks](#benchmarks)
- [Real-agent use](#real-agent-use)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [Ways to run it](#ways-to-run-it)
- [Knobs](#knobs)
- [Limits and negatives](#limits-and-negatives)
- [Tests](#tests)
- [Layout](#layout)
- [Licensing](#licensing)
- [Credits](#credits)

## Benchmarks

All our numbers are on the **abliterated** checkpoint `neko-legends/GLM-5.3-Flash-Uncensored-EXL3` @ `07135ec0`, on
one pair of DGX Sparks (GB10, TP=2), 2026-09-27 to 2026-09-29. The production numbers are from the last test window
(W10, 2026-09-29). Full tables and methodology: [`docs/RESULTS.md`](docs/RESULTS.md) (sections W6-W10 for the current
production config); raw JSON and the window scripts in [`results/`](results/).

### (a) Ours vs the vLLM production kit, same weights, same client

The vLLM column is [Reederey87's kit](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark) @ `8e443d6` (a fork
of [MiaAI-Lab's kit](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)) as we ran it in production on
this pair: 1M context, FP8 KV, DFlash2 k=7 with adaptive k, fused EXL3 MoE kernels, **the same abliterated weights**.
Both stacks were measured with the same client (`bench/glmbench.py`), one stack at a time, nothing else on the GPUs.

Two TensorFold configurations:

| Config | File | What |
| --- | --- | --- |
| **Production** (4 requests, shared 1M-token pool) | [`config/prod.env.example`](config/prod.env.example) | 4 concurrent requests sharing one **1,048,576-token FP8 latent KV pool** (each request up to 1M tokens while the pool has room), sessions inside the batch slots plus an NVMe session tier, shared system-prompt reuse, row-split prefill in 4,096-row chunks for a lone request, the RoCE all-gather, 16-row verify windows |
| Single-stream (earlier, 2026-09-28) | [`config/prod-single.env.example`](config/prod-single.env.example) | one request at a time, 524,288-token context, bf16 latent KV, fast/lean prefill in 8192-row chunks, session store |

**Decode**, tok/s, single stream, thinking off (decode excludes prefill and time to first token). Production: W10
load R16, median of 3 (R16 is the production config without 0370's decode overlap, which adds another ~1-2%);
the other columns: median of 5.

| Cell | vLLM kit | **TF production** | vs vLLM | TF single-stream (09-28) |
| --- | ---: | ---: | ---: | ---: |
| tf code, sampled (T=1), 64 tok | 35.2 | **42.3** | 1.20x | 40.5 |
| tf chat, sampled (T=1), 64 tok | 27.1 | **41.0** | 1.51x | 37.5 |
| tf code, greedy, 64 tok | 41.9 | **77.6** | 1.85x | 65.8 |
| tf chat, greedy, 64 tok | 22.8 | **44.6** | 1.96x | 44.7 |
| kit hashmap (prose), 200 tok | 30.0 | **53.2** | 1.77x | 50.4 |
| kit structured, 200 tok | 72.7 | **100.6** | 1.38x | 98.1 |
| kit essay, 200 tok | 26.1 | **44.9** | 1.72x | 44.2 |
| tweet sequence, 512 tok | 67.5 | **93.0** | 1.38x | 93.3 |
| tweet code, 512 tok | 42.4 | **66.7** | 1.57x | 62.3 |
| tweet json, 512 tok | 50.4 | **74.8** | 1.48x | 71.7 |
| edit (rename / comments / print-to-log), 1024 tok | not run | **111.2 / 94.7 / 115.5** (final config: 115.3 / 97.1 / 116.5) | - | 81.8 / 78.2 / 82.6 |

The edit cells (the model rewrites a file it was given) benefit from prompt-lookup drafts (patch 0020) and, since
W10, from verify windows of up to 16 rows (patch 0380: +16-27% on these cells, every reply hash unchanged).

**Prefill and time to first token** (cold, unique prompt, no cache hit; kernels already compiled):

| Prompt | vLLM kit | **TF production** (alone) | TF single-stream (09-28) |
| --- | ---: | ---: | ---: |
| ~7k tokens | 1,340 tok/s | - | 1,162 tok/s |
| ~24.5k-28k tokens | 1,448 tok/s at 28k (TTFT 19.4 s) | **~1,607 tok/s** at 24.5k (1,614 / 1,602; TTFT ~13.4 s) | 1,266 tok/s at 28k (TTFT 22.2 s) |
| ~98k-112k tokens | - | **~1,600 tok/s** at 98k (1,577-1,606) | 1,238 tok/s at 112k (TTFT 90.5 s) |
| 314k tokens (needle, after the stress run) | - | 1,376 tok/s, found | - |
| TF vs vLLM at ~28k | | **~1.11x** (24.5k vs vLLM's 28k) | 0.87x |

**Multi-turn, concurrency, boot, quality**:

| | vLLM kit | **TF production** | TF single-stream (09-28) |
| --- | --- | --- | --- |
| context | 1M in one context | **4 concurrent requests sharing a 1,048,576-token KV pool, each request up to 1M** | 1 x 524k |
| concurrent streams 1 / 4, aggregate tok/s (median of 5) | batches up to 4; Reederey87 publishes 63.4-66.3 warm at 4 in flight | 52.1 / **78.1** (5 mixed prompts; 4-stream reps 70-80 across windows) | one at a time (queued) |
| switch back to a stored ~39k-token session | - | **0.42-0.45 s** from the NVMe session tier instead of a 31 s cold prefill, also after a server restart | 0.4-1.0 s (RAM store) |
| resume a 314k-token conversation | - | 0.19 s (314,240 tokens cached) | - |
| 4 new sessions at once over one ~18k-token system prompt (subagent burst) | - | **28.2 s** wall instead of 72.4 s (shared-prefix reuse, patch 0310) | - |
| decoders during a long prefill | - | longest decode gap 3.9 s (4 x ~250k stress) | queued behind it |
| restart to ready (prepared weight folders, patch 0140) | - | **22-23 s** | 34-37 s (490 s from the raw checkpoint) |
| drafted == serial, byte-identical (10 cases) | n/a | 10/10; batched == alone 4/4 | 10/10 |
| MMLU-200 (greedy, thinking off) / refusals (10 prompts) | - / 0/10 | **88.0%** / 0/10 | 89.5% / 0/10 |
| needle retrieval | - | found at 314k (cold and resumed); earlier at 358k | 10/10 at 28k |
| memory stress: 4 conversations grown to ~250k each, then a 32k turn beside 3 decoders | - | no OOM, no request errors; worst MemAvailable 10.49 / 9.36 GiB (head / worker) | - |
| RoCE all-gather instead of NCCL (patch 0230/0350, W9 A/B) | - | decode +10.8% (1 stream) / +4.3% (4 streams) median, transcripts byte-identical | - |

"Drafted == serial" means every drafted reply is the same bytes as the one-token-a-round reply of the same engine
and weights. "Batched == alone": a request served next to 3 others returns the same bytes as served alone. The
TF production column is W10's final config (`FIN`) unless the row says otherwise; the session-tier row is W4 and the
system-prompt row W8 (both features unchanged since).

### (b) MiaAI-Lab's and Reederey87's published numbers (different weights and settings)

These are the kits' own published figures, copied from their READMEs on 2026-09-28. They use the **base** (not
abliterated) weights `brandonmusic/GLM-5.3-Flash-tr3-4bpw` (MiaAI-Lab serves the byte-identical mirror
`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`), their own clients (sparkDash, `tests/bench_decode.py`) and their own
prompts, so they are **not** directly comparable to table (a).

[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks):

| Date | Settings | Result |
| --- | --- | --- |
| 2026-09-07 | cold prefill, E3 grouped MoE; 900k context, util 0.86, MNBT 7168, DFlash2 k=7, 4 seqs, thinking off | 1,492 / 1,554 / 1,428 / 1,587 / 1,562 / 1,517 tok/s at ~8k / 16k / 32k / 64k / 128k / 256k (TTFT 32k: 23.0 s, 128k: 84.0 s) |
| 2026-08-28 | decode, structured + code prompts, DFlash2 k=7, temp 0, thinking off, 400 tok, 1M context | x1: 62.9 tok/s (TTFT 719 ms); x2: 51.7 a stream, 103.3 aggregate; x4: 37.1 a stream, 146.5 aggregate |
| 2026-09-17 | decode, prose, adaptive k (EMA) + dense FP8 + cooperative MoE, 850k context | x1: 36.1 a stream; x4: 19.4 a stream, 75.3 aggregate |
| 2026-09-21 | custom qualification run (their "OFF" arm, 850k) | structured 78.6, code 53.5, prose 33.2 tok/s; cold 32k TTFT 27.7 s, cold 100.7k TTFT 85.6 s |

[Reederey87/glm53-flash-exl3-2x-dgx-spark](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark) (production
stack of 2026-09-20, 1M context):

| Metric | Published |
| --- | --- |
| prose decode (hashmap) / hard essay | ~33 tok/s (median 33.03) / ~26 tok/s (median 25.92) |
| structured decode | ~74 tok/s (median 74.19, acceptance 1.0) |
| cold prefill (2026-09-09 stack) | ~1,454 tok/s at 60k, ~1,408 tok/s at 240k |
| 4 in-flight, warm aggregate | 63.4-66.3 tok/s |

How to read the two tables together:

- **Our numbers are with an abliterated model.** The abliterated checkpoint keeps attention, the shared expert, the
  dense layers and the head in BF16; we re-quantize those to 4 bits at load (`q4mse`, patch 0001), which costs a
  little accuracy (13 of 200 MMLU answers change) and still reads more than a natively 4-bit layout. The MTP head
  and the DFlash2 drafter were trained against the base model, so draft acceptance on abliterated weights is likely
  lower. We expect base GLM-5.3-Flash weights (for example the MLX 4-bit checkpoint TensorFold's own recipe uses) to
  net further improvements; that is an expectation, not a measurement.
- On our pair, the vLLM kit on the abliterated weights measured hashmap 30.0 / structured 72.7 / essay 26.1 tok/s,
  close to Reederey87's published 33 / 74 / 26 on base weights.
- Cross-kit comparisons also differ in context length, KV format, drafter settings, prompts, clients and dates.

## Real-agent use

This has been used in [opencode](https://opencode.ai) for real agent workflows (multi-file edits, tool calls, long
sessions) and performed well with thinking at **high** reasoning effort, the default in the production configs
(`GLM53_TF_DEFAULT_EFFORT=high`).

| Check | Result |
| --- | --- |
| opencode tool-call harness (`bench/toolcall_harness.py`: opencode's tool set, 21 cases x 10 runs, streamed, T 0, thinking off) | **200/210 passed, 0 corrupted** calls (same with FP8 prefill on or off) |
| same harness, thinking on, single-stream production load (21 x 5) | 95/105 passed, 0 corrupted |
| same harness through the HTTPS reverse proxy in front of the API (21 x 2) | 38/42 passed, 0 corrupted |
| real-model API checks after each switch | `/v1/models`; a thinking reply returns `reasoning_content` + the answer (finish `stop`); a tool call returns `get_weather({"city":"Paris"})` with finish `tool_calls`; streamed == non-streamed; `stop` strings streamed and not |

"Corrupted" means a tool call with leaked GLM markup (`<arg_key>`, `<tool_call>`, `</think>`) or unparseable
arguments. Every failure is a case where the model made a different, reasonable call than the one the case
expects: `edit_file` reads the file before editing it (a `read` call where the case expects `edit`), and with
thinking on `multi_turn_chain` takes another step first.

opencode provider entry (`~/.config/opencode/opencode.json`), for the production configs (port 8000):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "glm-tf": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "GLM-5.3-Flash (TensorFold)",
      "options": { "baseURL": "http://127.0.0.1:8000/v1" },
      "models": {
        "GLM-5.3-Flash-EXL3": {
          "name": "GLM-5.3-Flash",
          "tool_call": true,
          "reasoning": true,
          "limit": { "context": 1048576, "output": 32768 }
        }
      }
    }
  }
}
```

Set `limit.context` to the server's `CONTEXT` (1048576 for the production config, 524288 for single-stream). In the
production config the four requests share one 1,048,576-token pool: one request can use all of it, but four long
ones together wait for pages or spill idle sessions to the store.

## Requirements

- Two DGX Sparks (GB10, 128 GB unified memory each) connected by a QSFP cable between their ConnectX-7 ports, with
  the link configured (an IP address on one CX7 netdev per node; the RDMA device visible in `ibv_devices`).
- Docker with the NVIDIA Container Toolkit on both nodes (stock DGX OS has both).
- Passwordless `ssh` from the head node to the worker, as a user that can run `docker` there. The batch config's
  memory gate also drops page caches with `sudo -n` on both nodes.
- The weights in each node's Hugging Face cache, same revision on both (the repo is gated: request access on its
  model card first):

  ```bash
  hf download neko-legends/GLM-5.3-Flash-Uncensored-EXL3 --revision 07135ec082f8f11f7a71e4244a4e5167a0f96277
  ```

- Optional: the DFlash2 drafter `incoai/GLM-5.3-Flash-DFlash2` (revision `7d74cdd`), in both caches. It is
  **CC BY-NC-ND 4.0 (non-commercial only)**; this repo never ships it. Without it, drafting uses the checkpoint's own
  MTP layer.
- Disk: the checkpoint, plus ~83 GB a node for the prepared weight folder (fast restarts) and up to 64 GiB a node
  for the NVMe session tier (`GLM53_TF_SESSION_DISK_GIB`).
- Network access at build time to pull `nvcr.io/nvidia/pytorch:26.07-py3`. At run time the container is offline.

## Quickstart

On the head node (rank 0):

```bash
git clone --recurse-submodules <this repo> glm53-tensorfold-spark
cd glm53-tensorfold-spark
cp config/prod.env.example config/prod.env        # production (4 requests, 1M pool); or prod-single.env.example
$EDITOR config/prod.env      # <worker-ssh>, <head-ip>, the two HF cache paths; check NCCL_SOCKET_IFNAME / NCCL_IB_HCA
export CONFIG=config/prod.env

scripts/serve.sh build       # build the image here, copy it to the worker (docker save | ssh docker load)
scripts/prepare.sh           # optional: write the prepared weight folders once (restarts then take ~35 s)
scripts/serve.sh start       # rank 1 on the worker, then rank 0 here; waits for /v1/models, canary, warm-up
scripts/serve.sh status
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "GLM-5.3-Flash-EXL3",
  "messages": [{"role": "user", "content": "Write a Python function that reverses a linked list."}],
  "max_tokens": 1024
}'
scripts/serve.sh stop
```

The first start compiles kernels into a Docker volume and calibrates the drafter costs; later starts reuse both. The
API has no authentication and binds to `127.0.0.1` by default; put a reverse proxy with auth in front of it before
exposing it.

## Ways to run it

[`docs/TRYING.md`](docs/TRYING.md) covers: the production config (4 requests, shared 1M pool), single-stream long
context (524k), 4 x 256k batch (FP8 KV), a safe
upstream-like baseline, per-request knob A/B (`tf_knobs`), draft policies (`model@policy`), reasoning effort,
sessions, fast boot, how to benchmark (`glmbench`, `multiturn`, `quality`, `toolcall_harness`), how to check
exactness, and how to roll back.

## Knobs

Every engine change is a patch in [`patches/`](patches/) with its own `GLM53_TF_*` knob, off by default (upstream
behaviour) unless stated. [`docs/PATCHES.md`](docs/PATCHES.md) documents each knob and why each patch keeps the
output exact; [`docs/CHANGES-SUMMARY.md`](docs/CHANGES-SUMMARY.md) lists every patch with its measured gain and
status (on / opt-in / rejected), ordered by impact.

Main groups:

| Area | Patches | Main knobs |
| --- | --- | --- |
| Weights | 0001 | `GLM53_TF_NONEXPERT=q4mse` |
| Drafting | 0010, 0020, 0070, 0071 | `GLM53_TF_AUTO_FDRAFTS=7`, `GLM53_TF_LOOKUP=1`, `GLM53_TF_CALIB`, `GLM53_TF_DEPTH=cost` |
| Drafting / verify | 0380 | `GLM53_TF_MAX_DRAFT_ROWS=16` (verify windows of up to 16 rows) |
| Long context / KV | 0050, 0060, 0065, 0220, 0290 | `GLM53_TF_LATENT_KV=1`, `GLM53_TF_KV_DTYPE=bf16\|fp8`, `GLM53_TF_KV_POOL_TOKENS=1048576` |
| Prefill | 0003-0006, 0080-0085, 0170, 0190, 0320, 0335, 0360, 0390 | `GLM53_TF_FAST_PREFILL=1`, `GLM53_TF_LEAN_PREFILL=1`, `GLM53_TF_PREFILL_ROWS=auto`, `GLM53_TF_FAST_EXPERTS=fat`, `GLM53_TF_MOE_GLUE=5`, `GLM53_TF_ATTN_BM32=1`, `GLM53_TF_MTP_PREFILL_CACHE=1`, `GLM53_TF_PREFILL_PP=1`, `GLM53_TF_SOLO_PIECE=4096`, `GLM53_TF_B12X=4`, `GLM53_TF_MLA_EXPAND=v2` |
| Sessions / batching | 0110, 0120, 0180, 0200, 0250, 0310 | `GLM53_TF_SESSION_GIB`, `GLM53_TF_BATCH=4`, `GLM53_TF_BATCH_SESSIONS=1`, `GLM53_TF_SESSION_DISK=/sessions`, `GLM53_TF_PREFIX_SHARE=1` |
| Communication / decode host work | 0230, 0350, 0370 | `GLM53_TF_COMM_BACKEND=roce` (NCCL fallback + failure marker), `GLM53_TF_DECODE_OVERLAP=1` |
| Per request | 0090-0093 | `"tf_knobs": {...}` in the request body |
| Serving / ops | 0002, 0140, 0150, 0160, 0210, 0300 | prepared folders, `/health`, `/metrics`, `reasoning_effort`, `stop`, prompt-token cache, request log (`GLM53_TF_REQUEST_LOG`, no text) |
| Measured, not adopted (off) | 0240 (bits 1-2), 0260, 0270, 0280, 0330, 0340, 0400, 0410, 0370's `GLM53_TF_CPU_PIN` | see [Limits](#limits-and-negatives) |

## Limits and negatives

| | TensorFold + patches (production) | vLLM kit |
| --- | --- | --- |
| Single-stream decode | 1.20-1.96x faster on every measured cell | baseline |
| Prefill | ~1,607 tok/s at 24.5k and ~1,600 at 98k, against 1,448 measured for the kit at 28k (~1.11x; not the same prompt length). MiaAI-Lab publishes 1,492-1,587 on base weights (table b) | measured 1,340-1,448 here |
| 4 concurrent streams | ~78 tok/s aggregate (median; 70-80 across runs) | Reederey87 publishes 63-66 warm; **MiaAI-Lab publishes 146.5 aggregate on 4-stream structured output** (base weights, their client), higher than anything we measured at 4 streams (`docs/RESEARCH-NIGHT.md` §5) |
| Context | 4 requests share one 1,048,576-token pool: a request can grow to 1M, but not four at once (admission waits or spills idle sessions to the store) | 850k-1M in one context |
| KV precision | **FP8** latent KV: greedy replies diverge from bf16 KV within the first 0-78 tokens on 15 of 20 prompts (quality checks above held; long-session recall checked by needle at 314k-358k only) | FP8 KV too |
| Memory margin | the worker node binds (2 GiB less memory). The 4 x 250k stress bottoms at 9.36 GiB MemAvailable there, but a 314k needle right after the stress and MMLU dipped to **7.39 GiB** (under our 8 GiB target, no OOM); 8,192-row prefill chunks (+~4% prefill) were rejected for memory (stress minimum 7.23 GiB) | - |
| API | no `logprobs`, `n > 1` rejected, no images; in single-stream mode a `stop` match ends the reply but the engine keeps decoding silently to EOS / `max_tokens` before the next queued request starts | full OpenAI surface of vLLM |
| Maturity | **work in progress**: one pair of Sparks, one checkpoint, three days of measurements | production kits with many contributors |

Other negatives and trade-offs, measured:

- **FP8 prefill (0083) was rejected**: +7-11% prefill, but greedy replies diverged from bf16 prefill on 25 of 30
  prompts; off.
- **Decode-step kernels (0130) regressed** on the real model (tf code greedy 64.3 -> 62.4 / 56.4 tok/s); off.
- **`hc_fused` (0190)** is bit-exact but 7-8x slower on GB10 (shared-memory limits); off. `mtp_window` (0190) cost
  decode after long prompts; off. (`attn_bm32` is on in production since W1: +5% prefill, same bits.)
- **Patches measured and not adopted** (they stay in the tree, off; `docs/PATCHES.md` and `docs/RESULTS.md` have the
  numbers): 0240 b12x bits 1-2 (slower than today's kernels; only bit 4 is used, via 0360), 0260 `once` expert
  kernel (never beats `fat`), 0270 `FAST_EXPERTS=auto` (-2% end to end despite faster isolated kernels), 0280
  batch round buckets (-5% at 4 streams), 0330 warp-specialized `tc` expert kernels (cfg 1/2 cannot launch on GB10,
  cfg 3 -4%), 0340 per-slot drafter choice (simulated -0.4%), 0400 KDA recurrence v2 (+1.0-1.2%, under its bar),
  0410 sparse attention v2 (+0%), 0370's CPU pinning (+0.4%), and 8,192-row lone chunks (memory, above).
- **The single-stream config died of unified-memory OOM** once, with a 12 GiB session store filling under agent
  traffic at 524k context. Its example config uses 6 GiB; the watchdog (`scripts/systemd/`) restarts a dead pair.
- `q4mse` is a re-quantization of the BF16 non-expert weights: not bit-identical to BF16 (13 of 200 MMLU answers
  differ; accuracy 87.0% -> 88.0%); exactness (drafted == serial) holds within each mode.
- The RoCE all-gather (0230/0350) is limited to 256 KiB a message: one unexplained 2 MiB mismatch was seen once in a
  harness run (above that limit). A run-time RoCE failure writes a marker and the next start uses NCCL.
- Only this checkpoint and this two-node topology have been tested.

## Tests

The patch tests run in the image on one GPU with TensorFold's synthetic checkpoint (no real weights):

```bash
docker run --rm --gpus all -e PYTHONDONTWRITEBYTECODE=1 -v $PWD:/work --entrypoint bash glm53-tensorfold:dev \
    -c "bash /work/scripts/run_tests_in_image.sh /work/results/tests -- tests/cuda/test_patches.py tests/test_glm_tool_calls.py"
```

Host-only (no GPU, no Docker): `python -m pytest -q tests/test_serve_ops.py tests/test_gpuwatch.py` (launcher, canary,
Xid parser, GPU clock watch); the other `tests/test_*.py` run against a patched tree (`PYTHONPATH=<tree>/src`), several
of them in Triton's CPU interpreter. Against a
running server: `python3 bench/glmbench.py --base http://127.0.0.1:8000 --model GLM-5.3-Flash-EXL3 --suites exact`
checks drafted == serial on the real model. Known GPU-test failures are listed in `docs/RESULTS.md` (for example the
engine-level FP8 KV tests do not run on the synthetic model).

Before publishing a fork: `scripts/check-public.sh` scans the tree for private IPs, hostnames, keys and tokens.

## Layout

| Path | What |
| --- | --- |
| `vendor/TensorFold` | TensorFold, pinned submodule (`2f8e514`, 0.3.4), unmodified |
| `patches/` | engine patches, applied in order at image build |
| `docker/` | Dockerfile, entrypoint, compose file |
| `scripts/` | `serve.sh` (build / start / stop / status / logs / canary / watchdog / gpucheck), `prepare.sh`, `gpuwatch.py` (GB10 clock / slow-state watch), `traffic-report.py` (request-log summary), systemd units, `check-public.sh` |
| `config/` | `*.env.example`: node, weights and serving settings |
| `bench/` | benchmark clients, MMLU-200 subset, tool-call harness, shared-prefix bench, draft-policy and lookup simulators |
| `tests/` | patch tests (GPU) and launcher tests (host) |
| `results/` | raw benchmark JSON and the test windows' scripts (W1-W10); logs omitted |
| `docs/` | results, patch notes, design and analysis notes |

## Licensing

| Part | License |
| --- | --- |
| This project's code, patches, scripts, benchmarks and docs | **Apache License 2.0** ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). Redistributions, modified or not, must keep the copyright line and the NOTICE attributions and state their changes. |
| TensorFold (`vendor/TensorFold`) | MIT, Copyright (c) 2026 TensorFold contributors; unmodified submodule, the patches are applied at build time. The TensorFold code the patches modify stays under its MIT License. Its third-party notices: `vendor/TensorFold/THIRD_PARTY_NOTICES.md`. |
| RoCE all-gather in `patches/0230`, fast-prefill kernels in `patches/0240` | adapted from / re-implementing [b12x](https://github.com/local-inference-lab/b12x) (Apache-2.0, Luke Alonso and the b12x contributors); details in [`NOTICE`](NOTICE). |
| Fat-expert MoE kernel structure in `patches/0170` | adapted from the Apache-2.0 [Reederey87 kit](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark) (code MiaAI-Lab contributed under MIT before 2026-09-07); its NOTICE is reproduced in [`NOTICE`](NOTICE). The BF16 KDA copy in the same patch re-implements an idea from MiaAI-Lab PR #233 without its code. |
| Docker base image | NVIDIA Deep Learning Container License (`nvcr.io/nvidia/pytorch:26.07-py3`) |
| Model weights (not included) | `neko-legends/GLM-5.3-Flash-Uncensored-EXL3`: ShapleyMCG License 1.0 per its model card (the Local Inference Lab Attribution License 1.0: MIT-like with a required attribution, given at the top of this README and in NOTICE). Its sources `orcarouter/GLM-5.3-Flash-Uncensored-FP8` and `zai-org/GLM-5.3-Flash` are MIT per their model cards. The weights are abliterated (refusals removed); you are responsible for how you use them. |
| DFlash2 drafter (not included) | `incoai/GLM-5.3-Flash-DFlash2`: **CC BY-NC-ND 4.0, non-commercial only**. Never bundled; download it yourself, or run without it (MTP drafts only). |

## Credits

- [Ash Hart / TensorFold](https://github.com/ashhart/TensorFold): the engine, kernels, drafting and server this
  project patches.
- [MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) and its contributors: the GLM-5.3-Flash
  2x DGX Spark vLLM kit, the fat-expert MoE design, and the ops ideas listed in `docs/MIA-AUDIT.md`.
- [Reederey87](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark): the production vLLM kit we measured
  against and the Apache-2.0 kernel code `patches/0170` adapts.
- [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) (Luke Alonso and contributors): the RoCE
  one-shot all-gather `patches/0230` ports and the kernel designs `patches/0240` / `0360` re-implement.
- [0xSero](https://huggingface.co/0xSero): GLM-5.3-Flash EXL3 builds and DGX Spark recipes.
- [neko-legends](https://huggingface.co/neko-legends) (abliterated EXL3 weights, under Local Inference Lab's
  ShapleyMCG license) and [orcarouter](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-FP8) (the
  uncensored FP8 source).
- [brandonmusic](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw): the TR3 4-bit EXL3 weights the other
  kits publish their numbers on.
- [turboderp / ExLlamaV3](https://github.com/turboderp-org/exllamav3): the EXL3 format.
- [incoai](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2): the DFlash2 drafter.
- [Z.ai](https://huggingface.co/zai-org/GLM-5.3-Flash): GLM-5.3-Flash.
- [Vontra](https://huggingface.co/Vontra): the MLX checkpoints TensorFold's GLM recipe uses.
- NVIDIA: the PyTorch container (`nvcr.io/nvidia/pytorch`).
- The [vLLM project](https://github.com/vllm-project/vllm).
