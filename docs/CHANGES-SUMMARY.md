# Changes summary

> **Work in progress.** Every number below comes from one pair of DGX Sparks and the abliterated checkpoint
> `neko-legends/GLM-5.3-Flash-Uncensored-EXL3`, over three days (2026-09-27 to 2026-09-29). Statuses and defaults may
> change. What changed since the initial public release: [Update 2026-09-29](#update-2026-09-29-patches-0230-0410-test-windows-w1-w10).

Every engine change is a patch against TensorFold `2f8e514` (0.3.4), applied at image build, with a `GLM53_TF_*` knob
that defaults to upstream behaviour. Details, knobs and exactness arguments: [`PATCHES.md`](PATCHES.md). Raw numbers:
[`RESULTS.md`](RESULTS.md) and `results/`.

**Gain** column: *measured* = an A/B on the real model on this pair (source run in brackets); *estimate* = from
kernel timings, a microbenchmark or arithmetic, not an end-to-end A/B. **Status**: *on* = set in both production
configs (`config/prod*.env.example`); *batch* = on in the batch configs (production and 4 x 256k) only; *single* = single-stream config only;
*opt-in* = off unless you set it; *rejected* = tried, measured, left off; *superseded* / *tool* as noted.

## Update 2026-09-29 (patches 0230-0410, test windows W1-W10)

Since the initial public release (patches 0001-0220, the 4 x 256k batch config), 20 patches were added (0230-0410,
57 in total) and ten GPU test windows run (W1-W10, `docs/RESULTS.md`; scripts and JSON in `results/W1`-`W10`). The
production config (`config/prod.env.example`) is now 4 concurrent requests over one shared 1,048,576-token FP8 KV
pool. Its headline numbers (W10): prefill ~1,607 tok/s at 24.5k and ~1,600 at 98k (was 1,154 / 1,127; the vLLM kit:
1,448 at 28k), single-stream decode 1.20-1.96x the vLLM kit on every cell (tf code greedy 77.6, kit structured 100.6),
~78 tok/s aggregate at 4 streams, restart in 22-23 s, exact 10/10, batchexact 4/4, MMLU-200 88.0%, refusals 0/10.

Adopted (on in `config/prod.env.example`):

| Patch | What | Gain (measured) | Window |
| --- | --- | --- | --- |
| 0190 update | `attn_bm32` (32-query latent tiles) + `GLM53_TF_MTP_PREFILL_CACHE=1` (the MTP head's prefill rows as cache writes only) | +11% prefill at 24.5k / 98k, same bits | W1 |
| 0250 `glm-session-nvme` | the session store's NVMe tier (`GLM53_TF_SESSION_DISK`): per-rank shards, SHA-256 per chunk, write-through, restart indexing | a 39k session back in 0.42-0.45 s instead of 31 s, also after a restart | W4 |
| 0290 `glm-kv-pool` | one paged latent KV pool shared by the 4 slots (`GLM53_TF_KV_POOL_TOKENS`); admission reserves pages, spills idle slots, or waits | 1M context a request (was 262k) at the same memory; prefill -0.5 to -2% | W6 |
| 0300 `glm-request-log` | one JSON line a request, no text (`GLM53_TF_REQUEST_LOG`); `scripts/traffic-report.py` | tool; no speed change | W8 |
| 0310 `glm-prefix-share` | snapshots at the end of the system prompt; new sessions resume there (`GLM53_TF_PREFIX_SHARE`) | 4-session burst over ~18k system prompt 28.2 s instead of 72.4 s; 2nd session TTFT 2.24 vs 3.25 s | W8 |
| 0320 `glm-prefill-row-split` | each rank runs the hyper-connections on half the rows (`GLM53_TF_PREFILL_PP`) | +7.3-8.1% prefill, same bits | W8 |
| 0335 `glm-solo-piece` | a lone request prefills in larger pieces (`GLM53_TF_SOLO_PIECE`) | +3-4% at 4,096 rows (+6% at 8,192, rejected for memory in W9) | W8, W9 |
| 0230 + 0350 `glm-roce-gather`, `glm-roce-loopback-fix` | one-shot RoCE all-gather ported from b12x (Apache-2.0), up to 256 KiB, NCCL fallback + failure marker (`GLM53_TF_COMM_BACKEND=roce`); 0350 explained W2's loopback failure (the harness) | decode +10.8% (1 stream) / +4.3% (4 streams) median, transcripts byte-identical | W9 |
| 0360 `glm-b12x-attn-fp8` | b12x bit 4 (one-pass sparse latent attention) fit for FP8 KV and sessions (`GLM53_TF_B12X=4`) | +6.3% / +6.7% prefill, same reply hashes | W9 |
| 0390 `glm-mla-expand-v2` | latent absorb / expand retiled, same bits (`GLM53_TF_MLA_EXPAND=v2`) | +7.1% / +6.7% prefill | W10 |
| 0370 `glm-decode-overlap-cpu-pin` | decode-round host work off the critical path (`GLM53_TF_DECODE_OVERLAP=1`) | decode +1.3-1.9% (1 stream), +1-3% (4 streams), transcripts identical | W10 |
| 0380 `glm-deep-verify` | verify windows of up to 16 rows (`GLM53_TF_MAX_DRAFT_ROWS=16`) | edit cells +16-27%, every other cell within +-2%, all reply hashes identical | W10 |

Measured and not adopted (the patches stay in the series, off by default):

| Patch | Result | Window |
| --- | --- | --- |
| 0240 `glm-b12x-prefill` bits 1-2 (b12x KDA, fused mhc) | KDA 0.94x, fused mhc 0.69-0.74x of today's kernels; only bit 4 helps (adopted via 0360) | W3 |
| 0260 `glm-expert-decode-once` | never beats `fat` (0.89-1.01x) | W2 |
| 0270 `glm-fast-experts-auto` | -2% end to end although fast2 wins the isolated kernel at <= 2,048 rows | W5 |
| 0280 `glm-batch-buckets` | 80-91% graph replays but -5% aggregate at 4 streams (padded rows cost ~1 ms each) | W5 |
| 0330 `glm-expert-tc` | cfg 1/2 cannot launch on GB10 (registers); cfg 3 -4% end to end | W8 |
| 0340 `glm-batch-adapt` | offline simulator: per-slot drafter choice -0.4%, serial rounds for weak slots -6.5%; not run on GPU end to end | offline, W9 (per-slot cost measured) |
| 0370 `GLM53_TF_CPU_PIN` | +0.4% | W10 |
| 0400 `glm-kda-v2` | split +1.0-1.2% (bar +2%); fused slower than today | W10 |
| 0410 `glm-sparse-v2` | +0% end to end (kernel 0.82x) | W10 |
| 8,192-row lone chunks | +~4% prefill, but the 4 x 250k stress minimum fell to 7.23 GiB on the worker (target >= 8) | W9, W10 |

Measurement-only work: `docs/PROFILE.md` (W7 nsys profile of prefill and decode), `docs/ROOFLINE.md` (first-principles
roofline and ranked gaps), `docs/RESEARCH-NIGHT.md` and `docs/PARADIGMS.md` (surveys), `docs/DECODE-PLAN.md` (a sourced plan toward +40% decode, not yet run), `docs/DEEP-VERIFY.md` /
`docs/ADAPTIVE-DRAFT.md` (draft simulators: `bench/draftsim.py`, `bench/lookupsim.py`), design notes `docs/KV-POOL.md`,
`PREFIX-SHARE.md`, `PREFILL-PP.md`, `EXPERT-TC.md`, `DECODE-OVERLAP.md`, `MLA-EXPAND.md`, `KDA-V2.md`, `SPARSE-V2.md`,
`ROCE-FIX.md`.

Launcher and ops: `scripts/gpuwatch.py` (GB10 clock / power-clamp / slow-state watch; `serve.sh gpucheck`, a
preflight gate, systemd units, `docs/OPS-GPUWATCH.md`), `scripts/traffic-report.py`, `serve.sh` mounts the NVMe
session tier (`HEAD_SESSIONS` / `WORKER_SESSIONS`) and takes `CPUSET`; the Dockerfile installs libibverbs if a base
image lacks it (0230). Memory notes: the worker node binds; a 314k needle right after the stress and MMLU dipped to
7.39 GiB MemAvailable on it (no OOM), recorded as an open end in `docs/RESULTS.md` W10.

## The journey (headline numbers over time)

| When | Stack | Decode: tf chat greedy / kit structured (tok/s) | Prefill 7k / 28k / 112k (tok/s) | Context | Other |
| --- | --- | ---: | --- | --- | --- |
| 09-27 morning | vLLM kit, same weights (baseline) | 22.8 / 72.7 | 1,340 / 1,448 / - | 1M | batches 4 |
| 09-27 morning | TensorFold upstream, bf16 non-experts | 28.7 / 65.1 | 256-266 | 2,051 (HTTP 400 past it) | ~6 min load, one request |
| 09-27 | + 0001-0004 (q4mse, tool calls, chunked prefill) | 44.3 / 85.9 | 404-420 | 32k | drafted == serial 10/10 |
| 09-27 | + 0005/0006/0010/0020/0050/0070/0071 | 44.7 / 101.2 | 533 / 500 / - | 32k | edit cells 79-92 |
| 09-27 | + 0060 latent KV, 262k load (L1, exact 1024-row chunks) | 42.5 / 97.8 | 625 / 660 / 582 | 262k | 524k and 1M load |
| 09-27 | + 0080/0081 fast prefill (F5) | 41.6 / 95.8 | 782 / 767 / 772 | 262k | warm follow-up 28k: 42 s -> 0.9 s |
| 09-27 | + 0082/0084 lean 8192 + overlap (F9) | - | 974 / 982 / 961 | 262k | (+ FP8 prefill: 1,024 / 1,034 / 1,013, rejected) |
| 09-27 night | stacked 0001-0180 (Load A) | 47.6 / 96.7 | 1,084-1,122 / 1,062 / 1,105 | 262k | session revisit 0.45-0.54 s; load 490 s |
| 09-27 23:57 | single-stream production (S1: fat experts, fast boot) | 44.7 / 98.1 | 1,162 / 1,209 / 1,162 | 524k | load 34-37 s |
| 09-28 01:55 | + 0190 `moe_glue=5`, 0210 (Z1) | - | - / 1,266 / 1,238 | 524k | follow-up turn 2.35 s |
| 09-28 11:00 | 4 x 256k batch, FP8 KV (K1/P1) | 41.2 / 95.7 | - / 1,154 (24.5k) / 1,127 (98k) | 4 x 262k | 72-77 tok/s aggregate at 4 streams |
| 09-28 12:55 | W1: + 0190 `attn_bm32` + MTP prefill cache rows | - | - / 1,283 (24.5k) / 1,258 (98k) | 4 x 262k | same bits |
| 09-28 13:44 | W4: + 0250 NVMe session tier | - | unchanged | 4 x 262k | 39k session back in 0.42-0.45 s instead of 31 s, also after a restart |
| 09-28 16:35 | W6: + 0290 shared KV pool | - | - / 1,252 / 1,250 | **4 requests, 1M pool, 1M a request** | needle at 358k found; 4 x 300k overfill stress passes |
| 09-28 19:15 | W8: + 0300 / 0310 / 0320 / 0335, 8,192-row lone chunks | - | - / 1,468-1,473 / 1,447-1,451 | same | 4-session burst over an ~18k system prompt 28 s instead of 72 s |
| 09-28 23:26 | W9: + 0230/0350 RoCE, 0360 b12x bit 4, back to 4,096-row chunks (memory) | - | - / 1,499 / 1,496 | same | decode +7% (1 stream) / +4% (4 streams); stress minimum 9.72 / 8.67 GiB |
| 09-29 04:25 | W10: + 0390 MLA expand v2, 0370 decode overlap, 0380 16-row verify (production now) | 44.6 / 100.6 | - / ~1,607 (24.5k) / ~1,600 (98k) | same | edit cells 111-116 tok/s; 4 streams ~78 aggregate; MMLU-200 88.0% |

## Engine patches, ordered by impact

| # | Patch | What it does | Gain | Status |
| ---: | --- | --- | --- | --- |
| 1 | 0001 `glm-exl3-nonexpert-q4` | stores the checkpoint's BF16 non-expert weights (attention, shared expert, dense, head) in 4 bits at load (`q4mse`) | measured: decode 1.2-1.9x vs upstream bf16 (tf chat greedy 28.7 -> 44.3); serial 17 -> 33 tok/s; 1-row verify 57 -> 30 ms; MMLU 87.0 -> 88.0% | on |
| 2 | 0080 `glm-fast-prefill` | prefill chunks through fused, non-row-invariant kernels at absolute multiples of a chunk grid; fast2 expert kernels | measured (with 0081): 7k / 28k / 112k 625 / 660 / 582 -> 782 / 767 / 772 (+25-33%) [L1 -> F5] | on |
| 3 | 0060 `glm-latent-kv` | caches the 512-wide MLA latent instead of per-head K/V (absorbed MLA) | measured: 390.75 -> 18.75 KB a token a rank (20.8x); makes 262k-1M contexts fit; decode unchanged | on |
| 4 | 0140 `glm-fast-boot` | prepared per-rank weight folders, O_DIRECT parallel reader, calibration cache, persistent JIT cache | measured: restart 6-8 min (490 s) -> 34-37 s | on |
| 5 | 0110 `glm-session-cache` | many sessions' states resident per rank (256-token pages, shared prefixes once, LRU) | measured: switching back to a ~37k session ~36 s (re-prefill) -> 0.38-1.01 s | single (6 GiB) |
| 6 | 0120 `glm-batch-v2` | 2-4 requests a round, each with its own caches, drafter, knobs; prompts prefill in pieces between rounds | measured: aggregate 72-79 tok/s at 4 streams vs 43-70 single; batched == alone 4/4 | batch |
| 7 | 0220 `glm-fp8-latent-kv` | latent KV rows as e4m3 + a power-of-two scale | measured: 13,616 -> 7,664 B a token a rank; 4 x 262k slots 13.26 -> 7.45 GiB a node; makes 4 x 256k fit with a store. Changes greedy replies (15/20 diverge); MMLU 88.0%, needle 9/9 + 3/3 | batch |
| 8 | 0010 `glm-auto-deep-dflash-drafts` | `auto` verifies up to 7 DFlash2 drafts a round (upstream 5) | measured: +13-20% on tf code greedy / structured / sequence / json, -4-5% on hashmap / tweet code [M-D1c -> D1b] | on |
| 9 | 0006 `glm-exl3-expert-loop` | grouped EXL3 expert kernel loops over member tiles instead of re-reading each expert | measured: prefill at 512-row chunks 476 / 422 / 419 -> 633 / 525 / 490 tok/s at 1.8k / 7k / 28k (+17-33%) [M-P1a -> P1b] | on |
| 10 | 0082 `glm-lean-prefill` | fast chunks of up to 8192 rows on 512-1024-row window buffers; experts run once a chunk | measured: 837 / 863 / 852 (1024) -> 913 / 920 / 898 (8192), +7-9% [F9] | on (2048 rows in batch) |
| 11 | 0081 `glm-fast-prefill-kernels` | three-kernel chunked KDA, tuned large-M matmuls, one-accumulator matmuls, row-tiled hc_pre | measured: KDA chain 13.76 -> 1.85 ms a chunk; +7-13% prefill from one-accumulator matmuls + hc_pre tiles (782 / 767 / 772 -> 837 / 863 / 852) [F5 -> F9] | on |
| 12 | 0084 `glm-prefill-overlap` | all-gathers on a comm stream behind the next sub-block; hyper-connections in L2 slabs | measured: +6-7% (913 / 920 / 898 -> 974 / 982 / 961) [F9]; same bits | on |
| 13 | 0190 `glm-prefill-glue` | parallel MoE grouping + in-place combine (`moe_glue=5`); one-kernel router, MTP prefill window, fused hc, 32-query tiles (opt-in) | measured: `moe_glue=5` +5-6% (28k 1,197 -> 1,266; 112k 1,177 -> 1,238), same bits [Z1] | on (`moe_glue=5`); rest opt-in, see rejected |
| 14 | 0065 `glm-1m-memory-indexer` | DFlash2 context and index keys in rings; blocked prefill index selection | measured: 29.5 -> 13.6 KB reserved a token a rank; 1.8k prefill 498 -> 649 (+30%), indexer at 28k 1.5 -> 0.5 s [L1 -> F5] | on |
| 15 | 0085 `glm-c-independent-prefill` | fast-prefill state independent of chunk size; snapshots on a 64-token grid; `prefill_rows=auto` | measured: follow-up turn TTFT 3.6-6.8 s -> 2.9 s (a next turn re-prefills < 64 old tokens) | on |
| 16 | 0170 `glm-mia-prefill` | "fat" routed-expert kernels (structure ported from the MiaAI-Lab / Reederey87 grouped fat MoE, our arithmetic); opt-in BF16 KDA projection copy | measured: fat +3-5% end to end, bitwise = fast2; KDA copy 1.45x on that matmul (~7% of a 28k prefill) for +3.26 GiB a rank | fat on; KDA copy opt-in (memory) |
| 17 | 0020 `glm-suffix-drafter` | prompt-lookup drafts: verify the tokens that followed an earlier occurrence of the current suffix | measured: edit cells +9-12% (81.4 / 75.5 / 81.8 -> 91.6 / 82.0 / 91.3); other cells unchanged [M-D1a -> D1b] | on |
| 18 | 0180 `glm-batch-sessions` | the session store behind the batch slots (restore into a free slot) | measured: slot revisit 2.5 s with 39.8k cached; a session resumed after a 5th evicted its slot (4 GiB store) | batch |
| 19 | 0200 `glm-batch-parallel` | graph capture after N sightings, KDA parity in graph keys, batched MTP drafts, per-row cost floor, short-prompt admission | measured as a set: 76-79 tok/s aggregate at 4 streams [Y1]; per-knob gains not separated | batch |
| 20 | 0003 `glm-prefill-rows` | prefill chunk size configurable past upstream's 64 rows (kernels tile past 128) | measured (with 0001/0004): prefill 256-266 -> 404-420 tok/s | on (`auto`) |
| 21 | 0004 `glm-sparse-long-prefill` | sparse-attention scratch sized per window; vectorized token selection | measured: prompts past 2,051 tokens work (upstream: HTTP 400 or truncated) | on |
| 22 | 0050 `glm-longctx-decode` | past 2,051 tokens: bounded indexer, CUDA graphs for 1-8-row steps | measured: prefill +2% (520 -> 533 at 7k) [M-D2a -> D2b]; decode graphs at long context | on |
| 23 | 0070 `glm-realistic-calibration` | draft cost calibration on real text (+ optional online refinement) | measured: online refinement tf code greedy 69.5 -> 72.1, other cells within +-3% [M-D1d -> D1e] | on |
| 24 | 0071 `glm-cost-derived-depth` | each round, the draft depth with the most expected tokens net of cost | measured: mixed: structured +3%, tweet code +9%, json +3%, edit -3%, tf code sampled -4% [M-D1b -> D1d] | on |
| 25 | 0002 `glm-tool-call-parser` | GLM `<arg_key>/<arg_value>` tool calls become OpenAI `tool_calls` | measured: tool-call harness 200/210, 0 corrupted | on |
| 26 | 0160 `glm-openai-compat` | `stop` strings, `reasoning` + `reasoning_content`, `n > 1` rejected | functional (verified through the API) | on |
| 27 | 0150 `glm-mia-wins` | `/health` (fatal errors, stalls), `/metrics`, JSON / SSE error responses, OpenAI `reasoning_effort`, default effort | functional; no speed change | on (effort high) |
| 28 | 0210 `glm-prompt-tokens` | tokenize once per request; reuse token ids across turns at special-token boundaries | estimate (host timing): ~34 -> ~1 ms a turn at 40k tokens, ~113 -> ~3 ms at 128k; ids unchanged | on (default) |
| 29 | 0090 `glm-request-knobs`, 0091-0093 | per-request `tf_knobs` for the speed knobs; fast prefill / FP8 prefill / overlap per request | none by itself: A/B without a ~6 min restart | on |
| 30 | 0005 `glm-prefill-profile` | per-component prefill timing with CUDA events (same bits) | tool | opt-in (`GLM53_TF_PROFILE=1`) |
| 31 | 0030 `glm-batch2` | first batching prototype (written against 0001-0020) | superseded by 0120 | superseded (kept in series) |
| 32 | 0040 `glm-comm-prefetch` | L2 prefetch of the next weights during all-gathers; NCCL protocol choice | estimate: -0.5 to -1.6 ms a decode step (+2-5%); never measured end to end | opt-in (`GLM53_TF_COMM`) |

## Rejected or left off, with the reason

| Patch / knob | Result | Why off |
| --- | --- | --- |
| 0083 FP8 prefill (`fp8_prefill`) | +7-11% prefill (28k 1,062 -> 1,177; 112k 1,105 -> 1,186) | greedy replies diverged from bf16 prefill within ~20 tokens on 25 of 30 prompts; a behaviour change for a single-digit gain |
| 0130 decode-step kernels (`DECODE_KERNELS=v2[,pdl]`) | slower on the real model: 1-row verify 31.8 -> 32.7 / 32.9 ms; tf code greedy 64.3 -> 62.4 / 56.4 tok/s | regression |
| 0190 `hc_fused` | bit-exact but 7-8x slower on GB10 (num_stages=1 to fit shared memory) | slower |
| 0190 `moe_glue=7` (one-kernel router) | 1,218 / 1,229 vs 1,266 / 1,238 for `moe_glue=5` | slower at 8192 rows |
| 0190 `attn_bm32` + `mtp_window` | up to 1,389 / 1,357 tok/s at 28k / 112k (+10%), exact 10/10 | rank 0 ran out of unified memory at 524k context + 12 GiB store; `mtp_window` cut decode after a 112k prompt 81 -> 56 tok/s. **Update (W1)**: `attn_bm32` is on in the batch / production configs (+5% prefill, memory unchanged); `mtp_window` stays off |
| 0190 `LATENT_TC` | bf16 tensor-core absorb / expand | changes replies |
| 0170 BF16 KDA projection copy | 1.45x on that matmul | +3.26 GiB a rank; unaffordable with the session store at 524k |
| Single-stream `SESSION_GIB=12` at 524k | worked for hours | the store filled under agent traffic and the pair died of unified-memory OOM; examples use 6 (single) / 2 (batch) |

## Launcher and operations changes (not engine patches)

| Change | What | Status |
| --- | --- | --- |
| `scripts/serve.sh` | build here and ship the image to the worker; start rank 1 then rank 0; wait for `/v1/models`; refuse while another CUDA process runs on either node; caller env overrides the config | on |
| post-load canary (`scripts/canary.py`) | 3 greedy probes: degenerate output or a dead drafter (tokens a round < 1.3) fail the start in `strict` | on (`warn`) |
| warm-up (`WARMUP_LENGTHS`) | prefills ~4k and ~16k prompts after load so the first real prompt does not compile kernels (first 7k prefill: 727 tok/s cold) | on |
| memory gate (`MEM_GATE_*`) | waits for MemFree on both nodes (drops page caches) before launch; needed for the 4th batch slot | batch |
| preflight, start lock, Xid scan, log rotation | config / ssh / image / RDMA / sysctl parity checks; one start at a time; NVIDIA Xid events on both nodes | opt-in |
| watchdog (`serve.sh watch`, systemd user units) | restarts both ranks after repeated `/health` failures; `KillMode=process` so the detached restart survives the oneshot unit (found in a heal test) | installed in production |
| `scripts/prepare.sh` | writes the prepared weight folders (0140) on both nodes in parallel | tool |
| entrypoint | removes stale torch-extension locks after an OOM-killed start (a start once waited 24 min on one) | on |
| kernel cache volume | torch extensions, Triton and the CUDA JIT cache persist across restarts | on |
| `docker/compose.yaml` | the same container per node via Docker Compose | alternative |
| config placeholder guard | `serve.sh` / `prepare.sh` refuse a config that still contains `<placeholders>` | on |
| `scripts/check-public.sh` | scans the tree for private IPs, hostnames, keys and tokens before publishing | tool |
| benchmarks (`bench/`) | `glmbench.py` (decode / prefill / exactness), `multiturn.py` (sessions, follow-ups, concurrency, stall, slots, memory stress), `quality.py` (MMLU-200 + refusals), `toolcall_harness.py`, `fp8ab.py` (reply agreement / needle A/B) | tool |
