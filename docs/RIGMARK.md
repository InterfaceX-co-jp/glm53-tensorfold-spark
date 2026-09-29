# RigMark: vLLM kit vs TensorFold on 2x DGX Spark

The goal is to benchmark both GLM-5.3-Flash stacks with [RigMark](https://github.com/alexellis/rigmark) (Alex Ellis,
MIT) using its **standard settings**, so our numbers can be compared with receipts other people publish. Both runs
use the same pair of Sparks and the same weights (`neko-legends/GLM-5.3-Flash-Uncensored-EXL3` @ `07135ec`, the
abliterated EXL3 4-bit checkpoint). The two runs:

1. **vLLM baseline**: the vLLM production kit on the head node (`~/glm53-exl3-2x-kit`, `local/prod-start.sh`).
2. **TensorFold production**: `config/prod.env` (image `glm53-tensorfold:b4`).

Both engines serve `GLM-5.3-Flash-EXL3` on `127.0.0.1:8000`. The client runs on the head node over loopback, so neither
a VPN nor a proxy is in the timing path.

Scripts: `scripts/rigmark/` (`run.sh`, `compare.sh`, `window.sh`, `install.sh`, `preflight.py`, `metadata.py`,
`rigmark.env`). Results: `results/rigmark/<engine>-<UTC date>/`.

## What RigMark is

- **Implementation.** Pure Python 3.10+ standard library: no pip packages, no venv, nothing compiled. The whole
  install is a git clone. It works on aarch64 with the Python 3.12 that the head node already has.
- **Licence.** MIT, (c) 2026 Alex Ellis, OpenFaaS Ltd.
- **Pinned revision.** `c5a0db01b054` (2026-09-14): protocol 1.1.0, prompt corpus 1.0.0
  (sha256 `0c3ac401…`). `rigmark compare` refuses receipts made at different benchmark revisions or from a dirty
  checkout, so `install.sh` pins this commit and checks that the tree is clean.
- **API.** The generic OpenAI-compatible API. It does not use llama.cpp or Ollama endpoints. The vLLM extensions
  are needed only for the prefill phase.

### What it measures (standard suite = the `bench.py` defaults)

| Phase | Requests | Reported |
| --- | --- | --- |
| Decode | 5 runs each of **code** (a Go token-bucket package plus tests), **prose** (a ~700-word memo) and **structured** (a 50-object JSON array). Chat, streamed, `max_tokens` 4096, `temperature` 0, `top_p` 1, `seed` 20260905. Every prompt starts with a deterministic nonce made from the protocol version and the comparison ID. | Chunk-timed decode estimate `(completion_tokens-1)/(last-first output event)`: this excludes prefill and counts reasoning tokens. Also TTFT, **time to last output**, wall time, median/min/max/p90, and a basic output gate per run (visible answer, `finish_reason` `stop`, `[DONE]`; for structured, the JSON must be exact). Structured is labelled a speculative-decoding *ceiling* |
| Prefill | 3 pairs each at **8,192 / 32,768 / 65,536** tokens. `/tokenize` builds an exact token-ID prompt with a nonce prefix. It is sent to `/v1/completions` with `max_tokens` 8 and `ignore_eos`, then the same IDs are sent again at once | Effective prefill `prompt_tokens / TTFT`, reported separately for "cold" (cache-busting) and "immediate replay". The server's `prompt_tokens` must equal the requested depth exactly |
| Concurrency | 3 rounds at **C1 / C2 / C4**. Each stream is a code prompt capped at **256 tokens**, and all streams start together after a barrier | Aggregate end-to-end tok/s (includes prefill and scheduling), per-stream decode and TTFT, and normal-stop / visible counts. With thinking on, most streams hit the cap during reasoning. That is expected: this is a capacity test, not finished work |

- **Results and sharing.** Each run writes a content-hashed JSON *receipt* (settings, appliance metadata, every raw
  row, the visible outputs, hashes of the reasoning) and a text *card* (`<label>.card.txt`) meant for screenshots.
  There is no leaderboard and no upload. You share the card and link the unedited JSON with the exact command
  (`#RigMark`). `rigmark compare [--card]` makes a matched A/B table or card. `rigmark audit-code` replays the
  model-written Go tests in a locked-down `golang` container. That is a self-consistency check, not a speed metric,
  and it is optional.
- **Metadata.** Metadata is required, and placeholders are refused. `scripts/rigmark/metadata.py` fills it from the
  live config: hardware, topology, link speed read from sysfs, model revision, quantisation, KV dtype,
  engine/kit revision, image id, drafter, context limit, scheduler, and competing traffic. It never writes paths,
  users, hostnames or tokens.

## Thinking / reasoning: the decision

RigMark defines no thinking-off profile on purpose. Its rule is to set the model's control explicitly with
`--extra-body` and keep it identical within a sweep. Every published GLM-5.3 receipt in the RigMark repo
(`results/reference/glm53-*.json`, both Alex's TP2 and TP4 runs) uses:

```json
{"chat_template_kwargs": {"reasoning_effort": "low"}}
```

**We use exactly this body for both engines** (`EXTRA_BODY` in `rigmark.env`). Reasons:

- **The two server defaults differ.** The vLLM kit sets `GLM53_DEFAULT_REASONING_EFFORT=max` through
  `--default-chat-template-kwargs`. TensorFold prod sets `GLM53_TF_DEFAULT_EFFORT=high`. Left to their defaults,
  the two engines would render different prompts and think for different lengths. The explicit per-request key
  overrides both. On vLLM, request `chat_template_kwargs` beat the server default. On TensorFold, `apply_effort`
  only uses `setdefault`, so the request's key wins.
- **Thinking stays on for both.** The kit's template turns thinking on when `enable_thinking` is undefined.
  TensorFold starts with `--thinking` (the default is True), so it renders `enable_thinking=True`.
- **The prompts are the same.** Offline check, 2026-09-29: the kit's `files/chat_template.jinja` (sha256
  `9df580ab…`) and the checkpoint's `chat_template.jinja` that TensorFold uses (`41cff9af…`) differ in general, but
  for RigMark's system+user messages with this body they render **byte-identical** prompts:
  `[gMASK]<sop><|system|>Reasoning Effort: Low<|system|>…<|user|>…<|assistant|><think>`.
- **Our receipts stay comparable with the published GLM-5.3 receipts.** They carry the same request body, and a
  low-effort run is how GLM stays inside the 4,096-token decode cap. A thinking-off run would be a separate,
  model-specific "ceiling" and would need its own comparison ID and label.

`preflight.py` confirms on each server that this body really produces reasoning (thinking on), a visible answer and
`finish_reason: stop`.

## Compatibility

Sources: RigMark's `bench.py` at `c5a0db0`. For TensorFold, the server code after all patches were applied to
`vendor/TensorFold` in a scratch tree (`tensorfold/cuda/server.py`, `families/glm5_next/cuda/app.py`). For vLLM, the
kit's `start.sh` / `.env`, read without changes.

| RigMark needs | vLLM kit | TensorFold prod |
| --- | --- | --- |
| `GET /v1/models` (only used with `--model auto`; we pass the model explicitly) | yes | yes |
| Streaming chat, `stream_options.include_usage` (final usage chunk), `[DONE]`, one choice at index 0 | yes | yes (0150/0160) |
| `temperature` 0 / `top_p` 1 / `seed` / `max_tokens` 4096 | yes | yes (greedy when temperature <= 0) |
| Reasoning in `delta.reasoning` / `delta.reasoning_content` | glm45 parser | yes, but **both** fields by default (see gap 2) |
| `n`, `logprobs`, `best_of`, `response_format`, `stop` | not sent | not sent (TensorFold refuses `n>1`; it has no `logprobs`, and RigMark needs neither) |
| Output in >= 2 SSE events per completion | yes | yes (one event per decode round) |
| 4 concurrent streams | `max-num-seqs 4` | `GLM53_TF_BATCH=4` |
| Context >= 65,544 tokens (64K prefill + 8) | 1,000,000 | 1,048,576 |
| `POST /tokenize {model, prompt, add_special_tokens:false}` -> `{"tokens": [...]}` | yes (vLLM built-in) | **no: 404** (gap 1) |
| `POST /v1/completions` with `"prompt": [token ids]`, `ignore_eos`, `add_special_tokens` | yes | **no**: `check()` / `run()` only accept a string prompt (gap 1). `ignore_eos` works |

### Gap 1 (blocks the prefill phase on TensorFold): `/tokenize` and token-ID prompts

Status: implemented as `patches/0490` (image b5); W13 (docs/RESULTS.md) ran the full standard suite on TensorFold
production with the prefill phase. The text below is the original analysis.

Without a fix, `run.sh` finds the gap in preflight and runs TensorFold with `--skip-prefill`. Decode and
concurrency still run exactly as standard. The receipt then has no prefill rows, and a strict `rigmark compare`
against the vLLM receipt fails on `settings.prefill_depths`. `compare.sh` falls back to `--allow-mismatch` and says so
at the top of the report. The fix below goes in the server, so the benchmark's standard settings stay unchanged.
It is described here and **not implemented**. Proposed as `patches/0470-glm-rigmark-compat.patch`, host-side
only, with no engine or kernel change:

1. **`/tokenize` route** (`tensorfold/cuda/server.py`, `make_handler.do_POST`, before the `/completions` suffix
   check). For path `/tokenize` or `/v1/tokenize`, parse the JSON. A string `prompt` gives
   `ids = app.tok.encode(prompt, add_special_tokens=bool(body.get("add_special_tokens", True))).ids`. GLM-5.3's
   `tokenizer.json` post-processor is plain `ByteLevel`, so the flag changes nothing. Optionally, `messages` gives
   the template rendering exactly as `run` does it. Return
   `200 {"count": len(ids), "max_model_len": getattr(app.engine, "limit", None), "tokens": ids, "token_strs": null}`,
   which is vLLM's shape. A non-string prompt returns 400. This runs on rank 0 only, uses no engine lock and no GPU,
   and nothing goes to rank 1.
2. **Token-ID prompts on `/v1/completions`.**
   - `App.check`: when `prompt` is a list, require one flat, non-empty list of `int` (not `bool`) with
     `0 <= id < tok.get_vocab_size(with_added_tokens=True)`, else 400. A list of several prompts returns 400
     ("one prompt a request").
   - `GlmApp.check`: when the prompt is a list, use `ids = list(prompt)` instead of `prompt_tokens.encode(text)`
     (no memo), then run the existing context check with `len(ids)`.
   - `App.run`: `prompt = self.given_ids(body["prompt"]) if not chat and isinstance(body.get("prompt"), list) else
     self.prompt_ids(text)`. The new hook `App.given_ids(ids)` returns the ids. `GlmApp.given_ids` also does what
     `prompt_ids` does for the 0300 request log (`self._rl.ticket = self.reqlog.begin(ids)`).
   - Rank sharing, the session store, the prefix share and TOKCACHE already work on ids and need no change.
     `usage.prompt_tokens = len(prompt)` is then exactly the requested depth, which RigMark checks.
3. **Tests** (host only, `tests/test_openai_compat.py` fake engine):
   - `/tokenize` equals `tok.encode`.
   - A token-ID stream reports `usage.prompt_tokens == len(ids)` and `ignore_eos` gives exactly `max_tokens`.
   - Bad ids return 400.
   - A 65,536-id prompt passes the context check.

   On the Sparks: `preflight.py` passes (exit 0), and the 8-token reply arrives in at least 2 SSE events.

The fix ships in a new image. Production then needs the image rebuilt, a gate run (the usual exact / batchexact /
reply-sha gates, because `check` / `run` change), and `IMAGE=` bumped in `config/prod.env`. With it, both
receipts carry the full standard suite, and `rigmark compare` / `--card` match strictly.

### Gap 2 (cosmetic): reasoning sent twice

**Resolved as a recommendation (2026-09-29, docs/REPLAY-TTFT.md §3):** the vLLM kit sends `delta.reasoning` only
(`--reasoning-parser glm45` on vLLM `0.1.dev20051`; every kit acceptance log prints `reasoning field: reasoning`;
Alex's receipts count each reasoning character once). Set `GLM53_TF_REASONING_FIELDS=reasoning` in
`config/prod.env` (restart needed; not changed yet) so TensorFold streams the same field; until then set it for a run
as below. The original analysis follows.

TensorFold's default `GLM53_TF_REASONING_FIELDS=both` puts the same text in `delta.reasoning` and
`delta.reasoning_content`. RigMark concatenates the two fields. As a result, the TensorFold receipt's
`reasoning_characters` and reasoning hash cover the reasoning **twice**. Token counts and timings are unaffected
(they come from `usage` and event times), and so are the gates. Options:

- Leave production as it is and read the reasoning-character column as 2x. `preflight.json` records
  `reasoning_sent_twice`, and `compare.sh` prints the note.
- Or restore TensorFold for the run with `GLM53_TF_REASONING_FIELDS=<the field vLLM sends>`. `serve.sh` passes any
  exported `GLM53_TF_*` to both ranks, so this needs no edit to prod.env. For example:
  `GLM53_TF_REASONING_FIELDS=reasoning scripts/rigmark/window.sh tf-up`. Check vLLM's field name in
  `results/rigmark/vllm-*/preflight.json` (`delta_keys`) first. The next watchdog heal restarts TensorFold without
  the setting, which is harmless.

### Server-side differences to disclose (appliance comparison, not engine-only)

- **Non-expert weights.** TensorFold re-quantizes the checkpoint's BF16 non-expert weights to 4-bit at load
  (`GLM53_TF_NONEXPERT=q4mse`). vLLM uses them as published. The expert weights are the same bits in both.
- **Drafting policy.**
  - Both use the DFlash2 drafter at the same revision.
  - vLLM: k=7 with adaptive k (`GLM53_ADAPTIVE_K=ema`).
  - TensorFold: MTP + DFlash2 with cost-derived depth, verify windows up to 16 rows, and suffix lookup.
- **KV cache.** Both use FP8 (vLLM packed `fp8_ds_mla`; TensorFold FP8 latent KV).
- **Context.** 1,000,000 (vLLM) vs 1,048,576 (TensorFold).
- **Prefix and replay caches.**
  - vLLM: its prefix cache with the kit's `GLM53_APC_NO_STORE=1` / fine-grained APC.
  - TensorFold: the session store (RAM + NVMe) and 0310 prefix marks.
  - Each engine serves the "immediate replay" with its own mechanism, and RigMark allows that.
- **Token definition.** Both use the same `tokenizer.json`, so tok/s ratios compare the same token units.

`metadata.py` writes these facts into each receipt.

### Comparability with other people's receipts

- **Our own sweep.** Both runs use the same pinned revision, protocol, prompts, comparison ID and request body, so
  `rigmark compare` accepts the pair strictly (with gap 1 fixed).
- **Alex's GLM-5.3 receipts.** Those are protocol 1.0 with other comparison IDs. Our runs match them on everything
  under the client's control: the same corpus, settings, request body, counts, lengths, depths and concurrency. So
  the numbers are comparable in the sense RigMark's `RESULTS.md` uses, but `rigmark compare` will not accept the
  pair strictly.

## Install on the head node

`scripts/rigmark/install.sh` clones RigMark to `~/rigmark` (next to this repo's copy), checks out the pin, and verifies a
clean tree and Python >= 3.10. The clone is about 1 MB. It needs no GPU, runs no daemons, touches no system Python
and leaves production alone, so it can run any time (it has not been run yet). If the head node cannot reach GitHub, run
`rsync -a ~/.cache/rigmark/ <head-ssh>:rigmark/` from the workstation (a clone at the pin).

The head node copy of this repo (`~/glm53-tensorfold-spark`) is not a git checkout. Copy the new files there
before the window (this only adds files):

```bash
# workstation, repo root
rsync -a scripts/rigmark docs/RIGMARK.md <head-ssh>:glm53-tensorfold-spark/ --relative
ssh <head-ssh> 'cd ~/glm53-tensorfold-spark && scripts/rigmark/install.sh'
```

## Runbook

Run every command on **head** in `~/glm53-tensorfold-spark`, inside `tmux` so an ssh drop does not kill a
step. Before starting:

- Nothing else may use the GPUs, so the other optimization run must be finished.
- The morning automations must be able to tolerate :8000 serving vLLM for about an hour. Both engines serve the
  same model id, so clients keep working, but responses change and there are gaps of a few minutes while the
  engines switch.

| # | Step | Command | Expected time |
| --- | --- | --- | --- |
| 0 | Copy scripts, install RigMark (any time before) | see above | 1 min |
| 1 | Open the window: lease + refresher (touch every 4 min, max 240 min), watchdog timer stopped. Refuses while TensorFold has requests in flight | `scripts/rigmark/window.sh open` | seconds |
| 2 | Stop TensorFold, clear leftover `glm53-tf` containers on both nodes, wait for **MemFree > 95 GiB on both nodes** (restore-prod.sh's `wait_mem`; drops page cache with `sudo -n` while waiting; after 15 min it goes on, and prod-start.sh re-gates at 90 GiB and retries), then `WORKER_SSH=<worker-ssh> local/prod-start.sh`, then wait until `/v1/models` lists the model (`owned_by: vllm`) | `scripts/rigmark/window.sh vllm-up` | settle 1-10 min + kit boot ~10-20 min (80 min worst case after a JIT-cache wipe; none expected, since the config shape is unchanged) |
| 3 | GPU sanity (clock / power clamp on either node) | `CONFIG=config/prod.env scripts/serve.sh gpucheck` | seconds |
| 4 | RigMark on vLLM: install check, idle check, preflight, metadata, standard suite | `scripts/rigmark/run.sh vllm` | ~15-30 min (decode ~10, prefill ~5, concurrency ~2) |
| 5 | Stop the vLLM kit (`./start.sh stop`), then `CONFIG=config/prod.env serve.sh start` (its own MemFree >= 108 GiB gate with drop_caches; it refuses while a CUDA process remains), canary, warm-up, then verify a 17*23 chat | `scripts/rigmark/window.sh tf-up` | ~3-10 min |
| 6 | Close the window: watchdog timer on, refresher killed, lease deleted. Refuses unless TensorFold serves the model | `scripts/rigmark/window.sh close` | seconds |
| 7 | GPU sanity again | `CONFIG=config/prod.env scripts/serve.sh gpucheck` | seconds |
| 8 | RigMark on TensorFold prod | `scripts/rigmark/run.sh tensorfold` | ~15-30 min |
| 9 | Side-by-side report | `scripts/rigmark/compare.sh` -> `results/rigmark/compare-<date>.md` | seconds |
| 10 | Copy the results to the workstation and commit | `rsync -a <head-ssh>:glm53-tensorfold-spark/results/rigmark/ results/rigmark/` | - |

**Production is not on TensorFold** during steps 2-5, about 30-60 min. The whole session takes about 1-1.5 h.

**Why the window closes before step 8.** Production is back on TensorFold and verified after step 5, so it is safer
to have the watchdog guard it again at once. Its ticks only read `/health` and `/metrics`, so they add no load.
`run.sh` refuses to start while `/health` reports requests in flight. If a user request arrives during the run,
the receipt's `competing_traffic` only reflects the check at the start, so run it when nobody uses the endpoint. The
alternative is to keep the window open through step 8 and close it after. The refresher lasts 240 min, and the
lease stays fresh while it runs.

**If something fails:**

- **vLLM does not come up in step 2.** Run `window.sh tf-up` and then `window.sh close`. Skip the vLLM run and
  retry another day.
- **TensorFold does not verify in step 5.** Do not close the window. Look at `results/rigmark/tf-start.log` and
  `CONFIG=config/prod.env scripts/serve.sh logs`. As a last resort, run `~/restore-prod.sh`, which starts
  TensorFold from prod.env and falls back to the vLLM kit. Then `window.sh close` (with `FORCE=1` if vLLM ended up
  serving).
- **You walk away.** If you stop refreshing, the lease is older than 20 min after the refresher's 240 min limit.
  `rearm-watchdog.sh` (every 5 min) then re-arms the watchdog, which heals TensorFold production on its own
  (`WATCH_HEAL=1`). While vLLM is serving, the watchdog stands down (both TensorFold ranks absent), so it will not
  fight a running vLLM.
- **Check the state** at any point: `scripts/rigmark/window.sh status` (lease age, timer, who owns :8000,
  containers, MemFree).

**Memory.**

- The vLLM kit claims about 104 GiB of unified memory per node at boot (`GPU_MEM_UTIL` 0.85 x 121.7 GiB,
  `--kv-cache-memory-bytes` 15.4 GB). Its pre-check reads CUDA free memory, which tracks **MemFree**, not
  MemAvailable. That is why `vllm-up` waits for MemFree > 95 GiB on both nodes, as restore-prod.sh does. Idle
  MemFree sits at 93-97 GiB because of page cache, so the drop_caches while waiting matters.
- TensorFold's prod.env gates at MemFree >= 108 GiB (`MEM_GATE_GIB`, drop_caches on, 300 s, then it starts anyway).
- Never start one stack while the other's containers still exist. `serve.sh start` refuses while any CUDA process
  runs. `prod-start.sh` does not check for TensorFold, which is why `vllm-up` stops TensorFold first.

## Files and outputs

`results/rigmark/<engine>-<UTC yyyymmdd-hhmmss>/` holds:

- `<label>.json`: the receipt.
- `<label>.card.txt`: the card.
- `<label>.json.sha256`
- `run.log`: RigMark's per-run lines.
- `command.txt`: the exact command, RigMark revision, start / end and rc.
- `metadata.json`
- `preflight.json`: gaps, reasoning field names, thinking on.
- `models.json`

`results/rigmark/windows.log` logs each window step. `vllm-start.log` and `tf-start.log` hold the start output.

Knobs (`scripts/rigmark/rigmark.env`; a caller export wins):

- `COMPARISON_ID`: one per sweep. Use a new one for a re-run, because TensorFold's NVMe session store survives
  restarts and could replay "cold" prompts.
- `EXTRA_BODY`
- `SKIP_PREFILL` (`auto` / `0` / `1`)
- `RIGMARK_DIR`
- `ALLOW_BUSY=1`
- `DRY_RUN=1`: prints the RigMark command only.
- Labels.

`run.sh` passes none of RigMark's count, length, depth or concurrency flags, so the suite is always the standard
one.

To publish (RigMark fair-use checklist):

- Publish the card, the unedited JSON and `command.txt`.
- Headline the medians with their ranges.
- Label structured output as a ceiling.
- Report cold prefill and replay separately.
- Say it is an appliance comparison (see the differences above).
- Optionally run `rigmark audit-code <receipt>`. It needs Docker and `golang:1.25` and runs CPU only. Do not run it
  on the Sparks while a server is serving.
