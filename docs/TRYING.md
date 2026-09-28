# Ways to run and try it

> Work in progress: knobs and defaults may change. Everything here was run on one pair of DGX Sparks.

All commands run on the head node (rank 0) from the repo checkout. `scripts/serve.sh` reads `config/tensorfold.env`
unless `CONFIG=path` is set; a non-empty environment variable overrides the same key in the file for one start
(`CONTEXT=65536 scripts/serve.sh start`). Placeholders: `<worker-ssh>` (ssh target of the worker node, e.g.
`user@<worker CX7 address>`), `<head-ip>` (the head's address on the CX7 link), `<head HF cache>` / `<worker HF cache>`
(absolute paths of each node's `~/.cache/huggingface`).

Only one serving stack fits on the pair at a time: `scripts/serve.sh start` refuses to run while any other CUDA
process is up on either node. Stop vLLM (or anything else) first.

## 1. Pick a configuration

| Config | Copy from | Context | Concurrency | KV | Use it for |
| --- | --- | --- | --- | --- | --- |
| Single-stream, long context | `config/prod-single.env.example` | 524,288 | 1 (others queue) | bf16 latent | one user / one agent, the longest prompts, the fastest prefill (1,266 tok/s at 28k) |
| 4 x 256k batch | `config/prod.env.example` | 262,144 a slot | 4 | FP8 latent | several agents or sessions at once (72-77 tok/s aggregate at 4 streams) |
| 4 x 256k, bigger store | `config/prod-batch.env.example` | 262,144 a slot | 4 | FP8 latent | same, 4 GiB session store; only if your nodes have more free memory (worker hit 7.3 GiB in stress) |
| Safe / minimal (upstream-like) | `config/tensorfold.env.example` | 32,768 | 1 | per-head K/V (upstream) | a baseline with few patches active; check a problem against it |

```bash
cp config/prod.env.example config/prod.env
$EDITOR config/prod.env
CONFIG=config/prod.env scripts/serve.sh build       # image on the head node, copied to the worker
CONFIG=config/prod.env scripts/serve.sh start
CONFIG=config/prod.env scripts/serve.sh status      # logs 0 | logs 1 | stop | restart
```

### Single-stream (524k)

`config/prod-single.env.example`. One request runs at a time; the session store keeps other conversations'
states so switching back to one costs ~0.4-1 s instead of a re-prefill. Keep `GLM53_TF_SESSION_GIB` at 4-6: with
12 GiB at 524k the store filled under agent traffic and the pair died of unified-memory OOM. `stop` strings end the
reply, but the engine keeps decoding silently to EOS / `max_tokens` before the next queued request starts.

### 4 x 256k batch (FP8 KV)

`config/prod.env.example`. Four requests decode together; prompts prefill in 2048-token pieces between the others'
rounds, so a long prompt delays the decoders by ~2 s at most but takes longer itself (TTFT 45 s for ~40k beside 3
decoders). KV is stored as FP8 (`GLM53_TF_KV_DTYPE=fp8`): greedy replies differ from bf16 KV from the first tokens on;
quality checks held (MMLU-200 88.0%, refusals 0/10, needle 9/9 at 28k and 3/3 at 112k). To go back to bf16 KV
without a rebuild: `GLM53_TF_KV_DTYPE=bf16` with `GLM53_TF_SESSION_GIB=0` / `GLM53_TF_BATCH_SESSIONS=0`, or 3
slots, or the single-stream config. The memory gate (`MEM_GATE_GIB=108`, `MEM_GATE_DROP_CACHES=1`) needs
passwordless `sudo -n` on both nodes to drop page caches; without it the 4th slot may not fit at load.

### Safe / minimal

`config/tensorfold.env.example` turns on only the decode-side patches (4-bit non-expert weights `q4mse`, deeper
DFlash2 drafts, real-text draft calibration, prompt-lookup drafts, `prefill_rows=auto`) at `CONTEXT=32768`, and leaves
fast prefill, the latent / FP8 KV cache, sessions and batching off. For upstream TensorFold behaviour exactly, also
set `GLM53_TF_NONEXPERT=bf16`, `GLM53_TF_PREFILL_ROWS=64`, `GLM53_TF_AUTO_FDRAFTS=5`, `GLM53_TF_CALIB=random`
and `GLM53_TF_LOOKUP=0`, or build an image with only some patches:

```bash
docker build -f docker/Dockerfile --build-arg PATCHES="0001 0002 0003 0004" -t glm53-tensorfold:min .
IMAGE=glm53-tensorfold:min scripts/serve.sh start
```

(`serve.sh build` copies the image to the worker; with a manual `docker build`, build it on both nodes or copy it
with `docker save | ssh <worker-ssh> docker load`.)

## 2. Talk to it

```bash
B=http://127.0.0.1:8000; M=GLM-5.3-Flash-EXL3     # the prod configs; tensorfold.env.example: :8080, GLM-5.3-Flash-Uncensored
curl -s $B/v1/models
curl -s $B/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "'$M'", "messages": [{"role": "user", "content": "Summarize the CAP theorem in 3 bullets."}],
  "max_tokens": 2048, "stream": false
}'
```

- Streaming (`"stream": true`), tools (`tools`, `tool_choice`) and reasoning (`reasoning_content`, also as
  `reasoning`) follow the OpenAI API.
- The response has a `tensorfold` object (decode tok/s, TTFT, prefill seconds, the knobs used) and a `speculative`
  object (rounds, drafted and accepted tokens). `usage.prompt_tokens_details.cached_tokens` shows a session hit.
- `/health` and `/metrics` (Prometheus) on the same port (patch 0150).

### Reasoning effort

Thinking is on by default. In the prod configs `GLM53_TF_DEFAULT_EFFORT=high` and `GLM53_TF_EFFORT_FIELD=1`:

| Request | Effect |
| --- | --- |
| nothing | thinking on, effort high |
| `"reasoning_effort": "none"` or `"minimal"` | thinking off |
| `"reasoning_effort": "low"` | low effort (good for long structured output: tables, many numbers) |
| `"reasoning_effort": "medium"` / `"high"` | high |
| `"reasoning_effort": "max"` | max |
| `"chat_template_kwargs": {"enable_thinking": false}` | thinking off (always works, without the field mapping) |
| `"chat_template_kwargs": {"reasoning_effort": "low"}` | low effort (template-level) |

For structured output use thinking on at low effort rather than thinking off: the MiaAI-Lab kit measured 0/6 garbled
long tables that way against 5-6/6 with thinking off.

### Sessions

Nothing to do: every request resumes from the longest stored prefix of the same conversation (the stored states
are keyed by the token prefix). With the single-stream config the store holds `GLM53_TF_SESSION_GIB` of other
sessions; in the batch config each slot keeps its own conversation, and the 2 GiB store holds a few more (a 5th
session can evict the oldest). Check `cached_tokens` in `usage`. `"priority": "background"` (and opencode's
session-title requests, recognised automatically) waits behind foreground requests.

## 3. A/B a knob without a restart (`tf_knobs`)

Most speed knobs switch per request; the environment only sets their defaults. The response echoes the values
used in `tensorfold.tf_knobs`.

```bash
curl -s $B/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "'$M'", "messages": [{"role": "user", "content": "..."}], "max_tokens": 256,
  "tf_knobs": {"lookup": 0, "auto_fdrafts": 5, "depth": "threshold"}
}'
```

| Key | Values | Patch |
| --- | --- | --- |
| `lookup`, `lookup_min` | 0/1, 1-64 | 0020 prompt-lookup drafts |
| `auto_fdrafts` | 1-7 | 0010 DFlash2 depth of `auto` |
| `depth` | `cost`, `threshold` | 0071 |
| `calib_online` | 0/1 (refused in batch mode) | 0070 |
| `prefill_rows` | 1 to `GLM53_TF_PREFILL_ROWS_MAX`, or auto via the env | 0003 / 0085 |
| `expert_loop`, `longctx_graphs`, `profile` | 0/1 | 0006, 0050, 0005 |
| `fast_prefill`, `fp8_prefill`, `prefill_overlap` | 0/1 | 0091-0093 |
| `fat_experts` | 0/1 | 0170 |
| `moe_glue`, `mtp_window`, `hc_fused`, `attn_bm32` | see `docs/PATCHES.md` 0190 | 0190 |

Load-time settings (`GLM53_TF_NONEXPERT`, `GLM53_TF_LATENT_KV`, `GLM53_TF_KV_DTYPE`, `GLM53_TF_BATCH`, the prefill
buffer size, calibration) answer HTTP 400 if sent per request. Every bench script takes `--extra '{"tf_knobs": {...}}'`.

## 4. Draft policies (`model@policy`)

Append a policy to the model name, e.g. `"model": "GLM-5.3-Flash-EXL3@f7"`. `"draft": false` decodes that request
serially (one token a round, no drafts): the reference every drafted reply must equal.

| Policy | Meaning |
| --- | --- |
| `auto` | the default: per round, MTP (`c3:0.35`) or DFlash2 (`fc7:0.3` with patch 0010), whichever has committed more tokens per ms in this request; prompt-lookup drafts when they pay (0020); sampled requests use `a:0.6:0.85` |
| `N` | N MTP drafts a round |
| `a` / `a:LOW:HIGH` | 1-3 MTP drafts from the running acceptance |
| `cN:P` | up to N MTP drafts while the drafts' probability product stays >= P |
| `fN`, `fcN:P`, `fa:...` | the same with DFlash2 drafts (needs the drafter on both nodes) |
| `lN` / `lN:M` | prompt-lookup: up to N tokens that followed an earlier occurrence of the last M tokens |
| `o`, `om[N]`, `of[N]` | cost-derived depth (0071): each round the depth with the most expected tokens net of cost |

Measured (tok/s, single stream, same load): `@f7` is fastest on structured text (kit structured 103 vs 86 for
`auto` then) but slow on prose (essay 31 vs 42); `@c3:0.35` is best on sampled chat. `auto` is the balanced default.

## 5. Fast boot

`scripts/prepare.sh` writes each node's prepared weight folder once (split, re-quantized, tiled exactly as the load
builds them, ~83 GB a node); later starts read it with an O_DIRECT reader and reuse the cached calibration: ready in
~35 s instead of ~8 min. `serve.sh start` also writes the folder after a full load (`GLM53_TF_PREPARED_WRITE=1`).
`scripts/prepare.sh status` lists the folders; `--force` rewrites them. A folder is keyed by the checkpoint, the rank,
`GLM53_TF_NONEXPERT`, torch and the engine source, so a new image or weight format prepares again.

## 6. Benchmark it

```bash
B=http://127.0.0.1:8000; M=GLM-5.3-Flash-EXL3
# decode cells (tf / kit / tweet / edit) and prefill (ctx), median of 5
python3 bench/glmbench.py --base $B --model $M --suites tf,kit,tweet,edit --out results/mine-decode.json
python3 bench/glmbench.py --base $B --model $M --suites ctx --ctx 8000,32000,128000 --out results/mine-ctx.json
# sessions, follow-ups, concurrency, prefill stall, batch slots
python3 bench/multiturn.py --base $B --model $M --modes sessions,followup --doc 32000 --out results/mine-sess.json
python3 bench/multiturn.py --base $B --model $M --modes concurrent,stall,slots --streams 1,2,4 --out results/mine-conc.json
# quality: MMLU-200 + refusals
python3 bench/quality.py --base $B --model $M --label mine --out results/mine-quality.json
# opencode-shaped tool calls (21 cases; corruption = leaked GLM markup or bad JSON)
GLM_URL=$B/v1/chat/completions GLM_MODEL=$M python3 bench/toolcall_harness.py --reps 5 --out results/mine-tools.json
# A/B a knob on reply agreement and needle retrieval
python3 bench/fp8ab.py --base $B --model $M --modes agree,needle --knob fast_prefill
```

`glmbench.py` works against any OpenAI-compatible server (`--key` for a bearer token), so the same cells run against
vLLM for a fair comparison. Use one stack at a time and nothing else on the GPUs. The first long prompt after a load
compiles kernels; the configs' `WARMUP_LENGTHS="4096 16384"` pays that at start, otherwise run a warm-up first.

## 7. Check exactness

```bash
python3 bench/glmbench.py --base $B --model $M --suites exact              # drafted == serial, 10 cases
python3 bench/multiturn.py --base $B --model $M --modes batchexact         # batched == alone (batch config)
```

Both should report every case identical. They compare SHA-256 of replies decoded with drafts against `"draft": false`
(and 4 concurrent requests against the same requests alone). Settings that change the arithmetic (q4mse, fast
prefill, FP8 KV) change replies against other settings, but exactness holds within each setting.

## 8. Operations

```bash
scripts/serve.sh preflight   # config, ssh, image on both nodes, RDMA port ACTIVE, vm.min_free_kbytes parity
scripts/serve.sh canary      # the post-load probes on demand (fails on degenerate output or a dead drafter)
scripts/serve.sh xid 2h      # NVIDIA Xid events on both nodes
scripts/serve.sh watch       # watchdog loop; or the systemd user units in scripts/systemd/
```

The watchdog timer (`scripts/systemd/glm53-tf-watchdog.{service,timer}`, edit `WorkingDirectory` and `CONFIG`)
checks `/health` every minute and, with `WATCH_HEAL=1`, restarts both ranks after repeated failures. Watch
`MemAvailable` on the worker node during the first days of a new config; it is the binding node (2 GiB less memory).

## 9. Roll back

| To undo | Do |
| --- | --- |
| a per-request knob | drop it from the request; nothing persists |
| an env knob | remove it from the config and `scripts/serve.sh restart` (defaults are upstream behaviour) |
| FP8 KV | `GLM53_TF_KV_DTYPE=bf16` plus the memory changes in section 1 |
| batching | use `config/prod-single.env.example` |
| a patch | build with `--build-arg PATCHES="..."` listing the ones to keep, and `IMAGE=<tag>` |
| everything | `scripts/serve.sh stop`; start your previous stack (vLLM kit or upstream TensorFold). Old images stay tagged; `IMAGE=<old tag> scripts/serve.sh start` |

Prepared folders and the kernel cache volume (`glm53-tf-cache`) are safe to delete; the next start rebuilds them.
