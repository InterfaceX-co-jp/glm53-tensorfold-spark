# GPU window plan: 4 threads x 256k context with FP8 latent KV (about 1 hour)

_This is the runbook used for the one-hour validation window of the 4 x 256k configuration (results in
`docs/RESULTS.md`, "4 x 256k batch production"). It is kept as a template for validating a config change on your own
pair. `config/prod.env` / `config/prod-batch.env` here are your copies of the `*.env.example` files._

Candidate: `config/prod-batch.env`. It runs 4 slots at `CONTEXT=262144` each, with `GLM53_TF_KV_DTYPE=fp8` (patches/0220)
and sessions inside the batch slots (0180, fixed 2026-09-28). The memory numbers are in `docs/MEMORY-4x256k.md`.
Commands run from the repo checkout on the head node. `B=http://127.0.0.1:8000`,
`M=GLM-5.3-Flash-EXL3`, `R=results/K1` (make the directory).

## Before the window (no GPU)

1. Sync the working tree (patches through 0220, tests, bench, config) to both nodes' mirrors.
2. Build the image on the head node and ship it to the worker (CPU and disk only, about 10-15 min):
   `CONFIG=config/prod-batch.env scripts/serve.sh build`. This produces `glm53-tensorfold:fp8kv`.
3. Save the bf16-KV reference replies from the running bf16 load. This is plain inference, about 5 min. The cleanest
   baseline is `config/prod.env`, but any latent-KV bf16 load with the same weights works: batching, context, lean
   block and chunk size do not change the bits.

   `python3 bench/fp8ab.py --base $B --model $M --modes replies --long 10 --tokens 256 --save results/K0/replies-bf16.json`

## In the window

| t (min) | Step | Pass gate |
| ---: | --- | --- |
| 0 | Stop production: `CONFIG=config/prod.env scripts/serve.sh stop`. Check `nvidia-smi` shows no process on either node. Note `free -g` and `dmesg -T \| tail`. | |
| 2 | **Tests, both nodes in parallel**, each in its own container (see "Test commands"). Head node: `test_fp8_kv_patches`, `test_latent_patches`, `test_glue_patches`, `test_1m_patches`. Worker node: `test_batch_sessions_patches`, `test_batch_parallel_patches`, `test_batch2_patches`, `test_session_patches`. About 8-10 min. | fp8_kv: all pass; note the printed `[fp8 kv]` errors. batch_sessions: all pass, otherwise set `GLM53_TF_BATCH_SESSIONS=0` and `LEAN_BLOCK=1024`. Everything else: no new failures against T4-T8 (latent_tc is a known failure). |
| 13 | **Start the candidate**: `CONFIG=config/prod-batch.env scripts/serve.sh start` (memory gate + drop_caches, canary, 4k/16k warm-up). Copy the boot lines of both ranks to `$R/boot-r{0,1}.log`. | `latent KV cache (fp8 rows): 7.4 KB a token a rank (262152 slots, 1.86 GB)`; `batching 4 requests: 3 extra sequence(s)` (~6.3 GB); session store 4.0 GiB; MemAvailable after warm-up >= ~17 GiB (r1) / ~19 GiB (r0). A value more than 1 GiB off means the memory table must be redone before the stress step. |
| 16 | Start a **memory log** on the head node and leave it running for the rest of the window: `while :; do echo "$(date +%T) $(grep -E 'MemAvailable\|MemFree' /proc/meminfo \| tr -s ' ' \| tr '\n' ' ') \| $(ssh $WORKER_SSH "grep -E 'MemAvailable\|MemFree' /proc/meminfo" \| tr -s ' ' \| tr '\n' ' ')"; sleep 5; done > $R/mem.log &` | |
| 17 | **Exact**: `python3 bench/glmbench.py --base $B --model $M --suites exact --out $R/exact.json`, then `python3 bench/multiturn.py --base $B --model $M --modes batchexact --out $R/batchexact.json` | drafted == serial 10/10; batched == alone 4/4 |
| 21 | **Concurrency 1 / 2 / 4, stall, slots**: `python3 bench/multiturn.py --base $B --model $M --modes concurrent,stall,slots --streams 1,2,4 --reps 2 --long-tokens 256 --doc 32000 --append 2000 --out $R/conc.json` | Aggregate at 4 streams >= Y1's 76-79 tok/s. Stall: longest decode gap <= ~2.5 s (Y1: 2.4 s at 2048-token pieces). Slots: every revisit shows `cached` > 0; a 5th session evicts a slot. |
| 28 | **Prompt speed** (one request at a time; batch pieces of 2048): `python3 bench/glmbench.py --base $B --model $M --suites ctx --ctx 28000,112000 --out $R/ctx.json`, then `python3 bench/multiturn.py --base $B --model $M --modes followup,sessions --doc 32000 --out $R/sess.json` | Record tok/s against production (1,266 / 1,238 at 28k / 112k) and Y1 (single request ~5% below single-stream). The follow-up resumes (cached ~ the previous turn); session revisits resume in < 1 s. |
| 33 | **Memory stress, the worst case** (about 15 min of prefill): `python3 bench/multiturn.py --base $B --model $M --modes stress --stress-target 250000 --stress-step 60000 --stress-final 32000 --long-tokens 512 --mem-hosts local,$WORKER_SSH --mem-log $R/stress-mem.log --out $R/stress.json`. This builds 4 conversations to ~250k tokens each (the 4th to ~216k), filling the 4 GiB store (FP8 pages; evictions start after ~560k tokens) and resuming in the slots. Then 3 of them decode 512-token replies while the 4th adds a 32k-token turn in 2048-token pieces. Afterwards check `dmesg -T \| grep -i 'out of memory'` on both nodes. | Script prints PASS: min MemAvailable >= 8 GiB on both nodes, overall and in the final phase. No NVRM OOM, both ranks still up (`/health`). Every turn after the first shows `cached` ~ the previous prompt. Record the final 32k TTFT and the decoders' longest gap. |
| 50 | **Optional, if time allows**: `attn_bm32` per request on this load. Run `python3 bench/glmbench.py ... --suites ctx --ctx 28000,112000 --extra '{"tf_knobs":{"attn_bm32":1}}' --out $R/ctx-bm32.json` and the exact suite with the same `--extra`. | Memory unchanged (no extra DRAM, see MEMORY-4x256k); >= +2% prefill and exact 10/10 means it can be turned on. |
| 55 | **Decide**. If every gate passed, the candidate can stay up for the quality pass below. Otherwise: `CONFIG=config/prod-batch.env scripts/serve.sh stop; CONFIG=config/prod.env scripts/serve.sh start` and verify through the API (models, one chat, one tool call). | |

If time is short, keep steps 0-33 and the decision. The stress step is the one this configuration exists for.

A/Bs that need their own restart (not in this hour):

- `LEAN_BLOCK=1024` vs `512`: memory doc estimate 2-4% of prefill. Only affordable without the store, or with
  `SESSION_GIB` <= 2.
- `BATCH_PIECE=4096` with `PREFILL_ROWS_MAX=4096`: a few % more prefill, +0.78 GiB, about 2x longer decode gaps.
- `mtp_window`: it cost decode after a 112k prompt in Z2 (81 -> 56 tok/s).

## Test commands

On each node, from its checkout of this repo, with
nothing else on the GPU:

```bash
# head node
docker run --rm --gpus all -e PYTHONDONTWRITEBYTECODE=1 -v $PWD:/work --entrypoint bash glm53-tensorfold:fp8kv \
  -c "bash /work/scripts/run_tests_in_image.sh /work/results/K1-tests -- tests/cuda/test_fp8_kv_patches.py \
      tests/cuda/test_latent_patches.py tests/cuda/test_glue_patches.py tests/cuda/test_1m_patches.py"
# worker node
docker run --rm --gpus all -e PYTHONDONTWRITEBYTECODE=1 -v $PWD:/work --entrypoint bash glm53-tensorfold:fp8kv \
  -c "bash /work/scripts/run_tests_in_image.sh /work/results/K1-tests-worker -- \
      tests/cuda/test_batch_sessions_patches.py tests/cuda/test_batch_parallel_patches.py \
      tests/cuda/test_batch2_patches.py tests/cuda/test_session_patches.py"
```

`SUMMARY` in each log directory has one line a file. To see the printed FP8 error figures:
`grep '\[fp8 kv\]' results/K1-tests/test_fp8_kv_patches.log`.

The host-only interpreter test runs anywhere without a GPU, in its own process:
`TRITON_INTERPRET=1 PYTHONPATH=<patched tree>/src pytest -q tests/test_fp8kv_interpreter.py`.

## Quality spot-check (on the FP8 load)

FP8 KV stores each cached latent with 4 significant bits and a per-token scale. It is new arithmetic, so greedy
replies will diverge from bf16 KV at some point. The question is whether they get worse. The bf16 reference numbers
are in RESULTS.md: MMLU-200 89.5%, refusals 0/10, needle 10/10 at 28k, tool calls clean.

1. **Same prompts, both formats** (~5 min):
   `python3 bench/fp8ab.py --base $B --model $M --modes replies --long 10 --tokens 256 --against results/K0/replies-bf16.json --save results/K1/replies-fp8.json`
   - Divergence in itself is expected: FP8 prefill diverged on 25/30 prompts.
   - Open the pairs in `results/K0/replies-bf16.json` and `results/K1/replies-fp8.json`. For the long (8k-28k) repo-text
     prompts, judge whether the fp8 reply is as correct and specific as the bf16 one.
2. **MMLU-200 + refusals** (~5 min): `python3 bench/quality.py --base $B --model $M --label fp8kv --out results/K1/Q-fp8kv.json`
   - Within ~1.5 points of 89.5% is run-to-run flip noise (13/200 answers differed between bf16 and q4mse).
   - Refusals should be 0/10.
3. **Long-context recall** (what FP8 KV could hurt most; ~10 min):
   `python3 bench/fp8ab.py --base $B --model $M --modes needle --needle-ctx 28000,112000,240000 --trials 5 --trials-long 3 --knob fast_prefill --values 1 --out results/K1/needle-fp8kv.json`
   - The target is every depth found at every length.
   - A miss at 240k only: repeat it on bf16 before blaming FP8, since the bf16 load was only measured at 28k.
4. **Your own sessions.** Use one real long agent session that grows past 100k tokens: tool calls, code edits,
   questions about early parts of the conversation. This is the workload the configuration is for. Watch for wrong
   recall of early details and for malformed tool calls.
5. **If FP8 looks worse**, fall back without a rebuild:
   - `GLM53_TF_KV_DTYPE=bf16` in `config/prod-batch.env`. The same image works.
   - That needs 5.8 GiB more a node at 4 x 262k, so pick one of:
     - `SESSION_GIB=0` / `BATCH_SESSIONS=0` with `LEAN_BLOCK=512`: worst case ~7.9 GiB on the worker node, just under the target;
     - 3 slots;
     - CONTEXT=196608.
