# Decode overlap and CPU pinning (patch 0370: RESEARCH-NIGHT N1a + N1b)

Written offline, without a GPU. Both knobs are off by default. Neither changes any math: the same kernels run on the
same inputs, and both ranks issue the same collectives in the same order. Replies are unchanged: drafted == serial,
batched == alone.

| knob | values | what it does |
| --- | --- | --- |
| `GLM53_TF_DECODE_OVERLAP` | `0` (default), `1` (all four parts), or a comma list of `sync,emit,plan,gil` | Takes host work off the decode round's critical path (§3). `plan` must be the same on both ranks (checked at load); the other parts are rank-local. |
| `GLM53_TF_SWITCH_US` | 100..5000 (default 500) | The Python thread switch interval used with `gil`. Python's default is 5,000. |
| `GLM53_TF_CPU_PIN` | `0` (default), `fast`, `1` / `auto`, or explicit roles `serve=19;roce=18;comm=16-17;rest=0-15;nice=0` | Places the host threads on GB10's fast cores (§5). |
| `GLM53_TF_CPU_PIN_EVERY` | seconds (default 30) | How often the round loop re-sorts the threads. |
| `CPUSET` / `HEAD_CPUSET` / `WORKER_CPUSET` (scripts/serve.sh) | a cpulist | `docker run --cpuset-cpus`. Empty by default. `5-9,15-19` is the container-level form of `fast`. |

Files:

- `patches/0370-glm-decode-overlap-cpu-pin.patch`: new `decode_overlap.py` and `cpupin.py`; changes to `batch.py`,
  `decode.py`, `engine.py` and `app.py`.
- `scripts/serve.sh`: `CPUSET`.
- Tests: `tests/cuda/test_decode_overlap_patches.py` and `tests/test_serve_ops.py::test_cpuset`.

## 1. What sits on the critical path today

Line numbers are from the tree through 0360. In production (`GLM53_TF_BATCH=4`), every request goes through
`Batcher`, including a request that is alone. Rank 0's round loop is the `glm-batch` thread. Rank 1's round loop is
the main thread in `follow`.

A decode round, in order:

1. **Plan share, before any GPU work of the round.**
   - `batch.py:1338` calls `_plan()`. This takes the lock and builds the lists. Then `batch.py:1460` runs
     `self.g._share(plan)`, and with the KV pool `batch.py:1463` shares the spills too.
   - Each `_share` (`engine.py:759-771`) is two NCCL all-gathers. Each is followed by a blocking readback
     (`got[0].item()` at `engine.py:766`, then `allv[:count].tolist()` at `engine.py:771`), plus a pageable
     `torch.tensor(values, device="cuda")` copy (`engine.py:767`).
   - In production that is **4 NCCL control all-gathers and 4 host-device syncs a round, with the GPU idle**. Rank 1
     does the mirror image (`batch.py:1364-1366`).
   - These stay on NCCL even with RoCE (0230 only moves the model's exchanges). W9 measured about 45-76 us for a
     graph all-gather on NCCL.
2. **Verify forward:** stage + graph replay (or eager) at `batch.py:1802`.
3. **Timing-only device sync** at `batch.py:1888`: `torch.cuda.synchronize()` before `sample_multi`. Nothing between
   it and the sampler needs it: `parts` and `specs` are host-only, and `RoundCosts.record` uses model costs, not a
   clock. The sampler's `.cpu()` at `batch.py:324` synchronizes anyway. This sync only keeps the host from launching
   the sampler's top-k, cat and all-gather while the forward's tail runs.
4. **Sampling** (`batch.py:1896`): top-k, one all-gather (RoCE in production), `.cpu()`, numpy draw. This is a hard
   sync, and it is needed: the host needs the tokens.
5. **Accept + commit, and emit** (`batch.py:1904`, `_emit` at `batch.py:1570`): `job.out.put` wakes the request's
   HTTP thread. That thread then runs `App.run.on_tokens` (`server.py:397-402`: `StreamDecoder.add`, which is two
   tokenizer decodes; `split_thinking` / `hide_tool_calls` over the reply text; `json.dumps`; the SSE write at
   `server.py:523`), holding the GIL. This happens **while the round loop drafts** (step 6). The drafting host work
   is sync-bound, so any GIL wait there leaves the GPU idle.
6. **Drafting** (`batch.py:1925` / `1930`): an MTP chain has one hard sync per draft (`sample_drafts` `.cpu()`,
   `batch.py:368`). DFlash2 has its `candidates` readbacks. These are needed: each draft needs the previous one.
7. At the request's end, the 0300 request log (`app.py:192`, `log.end`) and the final response run on the HTTP thread
   (GIL), once per request.

Without batching (`GLM53_TF_BATCH` unset), `auto_decode` has the same timing-only sync (`decode.py:847`). There it also
feeds 0070's online timer (`decode.py:850`). The HTTP callback runs **inside** the decode loop (`decode.py:888`).
`mtp_decode`, `dflash_decode` and `serial_decode` have the same sync (`decode.py:625`, `693`, `565`).

## 2. How big the addressable part is (W7 nsys, `results/W7/analysis/mix-r0.json` / `mix-r1.json`)

Computed from the per-round records that `analyze.py` wrote. A "round" is a `W7:verify` range. The "gap" is the time
between the end of one verify range and the start of the next: `_plan` + plan share + `_execute`'s preamble.

| | 1 stream r0 / r1 | 4 streams r0 / r1 |
| --- | --- | --- |
| gap between rounds, median (mean, p90) ms | **0.83 / 0.88** (0.97 / 0.99, 1.71) | **0.92 / 0.91** (1.08, 1.86) |
| GPU idle inside the round, median (mean) ms | 1.83 / 1.90 (4.56 / 4.44) | 3.81 / 3.52 (6.81 / 6.45) |
| ... of which inside the verify forward, median ms | 0.28 / 0.43 | 0.60 / 0.59 |
| ... of which sampling + drafting syncs, median ms | 1.56 / 1.49 | 3.23 / 2.98 |

- **Cross-check from the per-request stats** (`results/W7/reqA2.jsonl`, `reqB-dec.jsonl`): `decode_s / rounds -
  verify_ms - draft_ms` = 0.7-1.0 ms a round at 1 stream. That matches the gap.
- **The "~3.9 ms between single-stream rounds"** in PROFILE §5 (and repeated in ROOFLINE gap 5) is not reproduced by
  these files. The between-round host time is about 0.9 ms.
- **The mean idle is dominated by eager and capture rounds, not by host scheduling.**
  - 8 of the 106 single-stream rounds have 32-35 ms of idle each, and hold 55% of all idle. Their `W7:vfwd` host time
    is 70-105 ms, i.e. kernels launched eagerly or a graph being captured.
  - `round_kinds` shows the same: 23 eager + 9 capture of 107 rounds at 1 stream, and 101-151 eager + 11-21 capture
    of ~180 at 4 streams.
  - This is a separate lever (graph keys captured at warm-up, or a lower `GLM53_TF_BATCH_CAPTURE_AFTER`, as W5
    discussed). 0370 does not touch it.

So what host reordering can reach is the ~0.9 ms gap, the ~0.05-0.1 ms of the timing sync, and whatever GIL waits the
HTTP work adds during drafting. It cannot reach the drafting syncs (~1.5 ms at 1 stream) or the sampler's readback:
those need the device-side sampler / drafter (RESEARCH-NIGHT L5, ROOFLINE gap 5b).

## 3. What 0370 changes (`GLM53_TF_DECODE_OVERLAP`)

### `plan`: the next round's plan rides on the sampler's all-gather

- **When a plan rides.** In a verify round where every slot in flight decodes and nothing waits (empty queue, no
  prompt prefilling, not stopping), rank 0 decides the next round's plan (`Batcher._plan_ahead`). It does this right
  after launching this round's verify forward, while the GPU runs it. Such a plan can only contain cancels.
- **How it travels.** The plan's int32 words are appended to the candidates `sample_multi` already gathers
  (`rider`: n + 8 words, the same size on both ranks). They come back in rank 0's slice. Both ranks decode rank 0's
  words (`dover.rider_decode`), so rank 0 runs exactly what rank 1 received.
- **The next round.** It starts with `_take_ahead()`: no `_plan`, no `_share` (`_serve` / `follow`). Every other round
  (arrivals, admissions, spills, prefill pieces, stopping) plans at the top as before. Rank 0 signals that with a
  zero rider.
- **Cost.** In steady decode the rider removes all 4 control all-gathers and their 4 host syncs from every round, and
  moves the plan's Python into the forward's shadow. The price is 32-48 bytes more in an exchange that already happens
  (RoCE, ≤ 256 KiB).
- **Cancels.** A cancel is decided one round earlier than before. If that slot ends by itself in the round in
  between, `_execute` already skips it (`if self.seqs[s] is not None`). No admission can happen between the decision
  and its round, so a slot index cannot point at another request.
- **Arrivals.** A request that arrives while others decode can wait up to one round longer for admission (60-125 ms).
  This is the one user-visible effect.

### `sync`: no timing-only device sync after the verify forward

- `Batcher._verify` skips `torch.cuda.synchronize()`; the sampler's readback is the sync.
- In `auto_decode` / `mtp_decode` / `dflash_decode` / `serial_decode` the same (`decode._forward_sync`).
- 0070's online calibration (`e.calib`, rank 0, no batching) now reads the forward's time from two CUDA events
  (`ForwardTimer`) after the readback, instead of the host clock after the sync. That is GPU time from the forward's
  first to last kernel; the host clock measured the same thing plus the launch latency.
- The per-request `stages_ms` split between "forward" and "sample" moves; it is stats only.

### `emit`: HTTP work while the GPU verifies, not while the host drafts

- **Batching.** Rank 0 holds a round's tokens (`_held`). It hands them to the request threads right after the next
  round's verify forward has been launched (`_flush` in `_verify`), when the round loop is about to block on the GPU
  for 50-130 ms.
  - The held tokens also go out at a request's end (`_finish`, before the end marker), at the top of `_plan`, and on
    a loop failure.
  - Order within a request is kept. Nothing is lost (tested).
  - Cost: a streamed token reaches the client one round later (≤ 60-125 ms). The final tokens are not delayed.
- **Without batching.** The HTTP callback runs on an emitter thread (`Emitter`) instead of inside the decode loop.
  `generate` joins it before it returns, and the callback's exception is raised then.

### `gil`: `sys.setswitchinterval(GLM53_TF_SWITCH_US)` (500 us instead of 5 ms)

When the round loop's sync returns while an HTTP thread holds the GIL in pure Python, the loop waits at most 0.5 ms
instead of up to 5 ms. This bounds the tail. It does not help against a Rust/C call that holds the GIL (tokenizer
encode, numpy), which 0210's token cache keeps short.

### Expected savings (estimates; the GPU run decides)

| part | 1 stream, ms a round | 4 streams, ms a round |
| --- | --- | --- |
| `plan` (the ~0.85-0.9 ms gap down to ~0.2-0.3 ms: 4 NCCL all-gathers at ~45-76 us + 4 readbacks + 2 pageable H2D copies + `_plan`'s Python) | -0.5 to -0.7 | -0.5 to -0.7 in decode-only rounds; 0 in rounds with arrivals or pieces |
| `sync` (top-k / cat / gather launches overlap the forward's tail) | -0.05 to -0.1 | -0.05 to -0.1 |
| `emit` (HTTP GIL time moved from the drafting syncs to the forward wait) | 0 to -0.3 | -0.2 to -0.8 (4 HTTP threads) |
| `gil` | tail only | tail only |
| **total** | **-0.6 to -1.1 ms of ~52-57 ms: +1.1-2.0% decode** | **-0.7 to -1.5 ms of ~100-125 ms: +0.6-1.5%** |

This is below RESEARCH-NIGHT's +3-6% (N1b). That estimate took the between-round host time as 3.9 ms; the W7 files
say ~0.9 ms (§2). The larger in-round idle is sync-bound drafting and eager / capture rounds, which host scheduling
cannot move.

### Exactness and the lockstep protocol

- **Math.** Nothing in it changes: the same kernels, windows, sampler inputs and keyed draws.
- **Collective order.** It is identical on both ranks. Whether a round's plan rode is known on both ranks from the
  exchange itself: both ranks gather in every verify round (the rider is always there with `plan`), and both read
  rank 0's flag.
- **Load check.** `plan` is compared at load (`_gather_ints`); a mismatch refuses to start. The rider's size depends
  only on the slot count, which is compared too.
- **Tokens.** Rank 1 never sends a plan (zero rider, tested). `_emit` is a no-op on rank 1 (no `job.out`).
- **Watchdog / health.** `/health` progress marks come from the request thread's callback. With `emit` they arrive up
  to one round later, far inside `GLM53_TF_STALL_S`. `/metrics` counters are unchanged. The round loop's structure
  (a round per plan, the same `fair.after` bookkeeping) is unchanged.
- **Double-buffering the accept readback** (pinned memory + an event) was considered and not done. The next round's
  first launch (its drafting) depends on this round's keeps, so there is nothing to put between the readback and its
  use. The one readback-independent piece of work, the plan, is what `plan` moves.

## 4. Offline tests (`tests/cuda/test_decode_overlap_patches.py`, 29 tests, CPU)

- **Knob, rider, emitter, timer.** Parsing; rider encode / decode round trip and its failure message; `Emitter`
  order and exception at `close`; `ForwardTimer`'s CPU fallback.
- **0180's hostile fake model**, driven through real `Batcher._plan` / `_execute` / `_verify` / `_finish`:
  - 4 sessions x 14 turns over shared prefixes, fast and exact prefill, with every part of the knob alone and all
    together;
  - every reply and every slot's end state equal a fresh prefill + serial decoding;
  - 73 of 247 (exact) and 117 of 237 (fast) verify rounds planned the next round ahead.
- **Lockstep.** Rank 0's plan shares and sampler exchanges are recorded as one ordered stream and replayed through
  `follow` on a second batcher playing rank 1:
  - every message is of the kind rank 1 asks for at that point;
  - rank 1's riders are zeros;
  - both ranks end the same requests the same way (sha, keeps, cancelled), through the same rounds, with identical
    session stores and slot states;
  - fewer plan shares than without the knob;
  - a request cancelled mid-decode ends through a plan that rode, at the same round on both ranks, and the others'
    replies are unchanged.
- **`emit`.** A round's token is held until the next forward launch (or handed out at a request's end), in order,
  before the end marker. Nothing is lost.
- **`_plan_ahead`.** It refuses with a queue, a prompt prefilling, or while stopping, and includes cancels.
- **`auto_decode` on a fake engine with `sync`.** The same tokens, keeps and online-timer rows, and exactly one host
  sync less a round.
- **`cpupin`.**
  - GB10's cores from a fake sysfs / cpuinfo (the values read on both Sparks).
  - The auto / fast / explicit plans, a container with fewer cpus, and bad specs.
  - On this host with real affinity calls: `serving` pins the calling thread and names it `tf-serve`; a thread it
    starts inherits the core and `sweep` moves it to `rest`; an `NCCL`-named thread goes to `comm`; a thread sitting
    on the RoCE cpu is left there.
- **`serve.sh`.** `CPUSET` becomes `--cpuset-cpus` per node; it is absent when unset.

Existing suites on the tree with 0370, with the knob unset and with `GLM53_TF_DECODE_OVERLAP=1`:

- every `tests/cuda/test_*_patches.py` + `test_patches.py`: 494 passed, 604 skipped (GPU), 9 failed;
- the same 9 fail on the tree without 0370 (fp8 / glue Triton-on-CPU tests: environment);
- `tests/test_serve_ops.py` 36/36;
- the host-only `tests/test_*.py` failures (`test_fastk_interpreter`, `test_fastboot_prepared`: no safetensors, Triton
  interpreter) are identical without 0370.

## 5. N1a: host threads on the fast cores (`GLM53_TF_CPU_PIN`)

### Core map (read-only on both nodes, 2026-09-28)

| cpus | core | `CPU part` | max clock | `cpu_capacity` | L2 | L3 (shared by) |
| --- | --- | --- | --- | --- | --- | --- |
| 0-4 | Cortex-A725 | 0xd87 | 2.81 GHz | 718 | 512 KB | 8 MB (0-9) |
| 5-9 | Cortex-X925 | 0xd85 | 3.90 GHz | 997 | 2 MB | 8 MB (0-9) |
| 10-14 | Cortex-A725 | 0xd87 | 2.81 GHz | 731 | 512 KB | 16 MB (10-19) |
| 15-19 | Cortex-X925 | 0xd85 | 3.90 GHz | 1017 (19: 1024) | 2 MB | 16 MB (10-19) |

The best cores are 15-19: X925s with the 16 MB L3, and cpu 19 has the highest capacity.

### What production does now

- **cpusets.** The containers have no cpuset (`docker inspect`: `CpusetCpus` empty; `Cpus_allowed_list: 0-19`).
- **Thread names.** All 53 threads of the engine are named `tensorfold` except CUDA's and the TCPStore's, so `ps`
  cannot tell them apart.
- **Busy threads when idle.** The current container runs with `GLM53_TF_COMM_BACKEND=roce` (W9). Two threads are at
  ~100% CPU even with no request: on rank 0, 0230's busy-spinning proxy and one more. On rank 1 they are the main
  thread and one more; the main thread is `follow`, spinning in a CUDA synchronize.
- **Placement.** The two busy threads sat on X925s (6, 7, 8, 15) for 6 s of samples at 200 ms. Other threads moved
  between X925 and A725 (cpu 0, 10), including the main thread.
- **So the scheduler already keeps the spinning threads mostly on big cores,** as RESEARCH-NIGHT warned (0% is
  possible). The upside is:
  - no migration between the clusters (L3);
  - no chance of the round loop waking on an A725 after a sync;
  - the RoCE proxy on a core of its own.

### What 0370 does

- **`early()`** (`GlmEngine.__init__`, before NCCL creates its threads):
  - reads the plan;
  - sets `NCCL_SET_THREAD_NAME=1`, so NCCL's threads get `NCCL ...` names;
  - with RoCE, sets `GLM53_TF_ROCE_CPU` when unset, so 0230's proxy pins itself;
  - puts the main thread on `rest`. Threads made later inherit that, including the HTTP threads.
- **`serving()`.** Rank 0's batch thread and rank 1's `follow` pin themselves to `serve`, take the name `tf-serve`, and
  sort every other thread:
  - `NCCL*` to `comm`;
  - the thread on the RoCE cpu is left there;
  - everything else to `rest`.
- **Re-sort.** `tick()` repeats the sort every 30 s. A thread made by the round loop (say a pool worker) inherits its
  single core until then.
- **Without batching,** the request's thread pins itself to `serve` while it decodes.
- **HTTP threads** are named `tf-http`. `nice=N` lowers their priority, but the default is 0: they have cores of their
  own, and a descheduled thread that holds the GIL would stall the round loop (priority inversion). That is why "HTTP
  at lower priority" is not the default.
- **Plans on GB10.**
  - `auto` with RoCE: `serve 19, roce 18, comm 16-17, rest 0-15`.
  - `auto` without RoCE: `serve 19, comm 17-18, rest 0-16`.
  - `fast`: every thread on 5-9,15-19, nothing dedicated. That equals `CPUSET=5-9,15-19` in serve.sh; docker's
    `--cpuset-cpus` is not needed for the other modes.
  - A container that allows fewer cpus gets the best of what it allows (tested).
- **The explicit form overrides any role,** e.g. `rest=5-9,15` keeps the HTTP threads (GIL holders) off the A725s.
  That is worth one A/B cell: a GIL holder on a slow core holds the GIL longer.

Expected: +0 to +2% decode, mostly as less rank skew (PROFILE: ~3 ms each way, "jitter"). Unknown until measured.

## 6. GPU test plan (one window, ~2 h; build the image with 0370, knobs off = today's prod)

### 0. GPU unit tests, both nodes

- `tests/cuda/test_decode_overlap_patches.py` (CPU parts, and the affinity test on the Spark's real cores).
- `test_batch_parallel_patches.py` and `test_batch_sessions_patches.py` with `GLM53_TF_DECODE_OVERLAP=1` in the
  environment. Their GPU tests build real Batchers, so the rider goes through the real `sample_multi`, pinned staging
  and the one-GPU "two ranks" setup.
- Every existing GPU test must pass unchanged.

### 1. Loads

All loads are `config/prod.env` (W9: RoCE, b12x 4) plus:

| load | adds |
| --- | --- |
| A | nothing |
| B | `GLM53_TF_DECODE_OVERLAP=1` |
| C | `GLM53_TF_CPU_PIN=auto` |
| D | B + C |
| E (if C wins) | `GLM53_TF_CPU_PIN=auto` with `rest=5-9,15` |

### 2. Per load, in this order

1. `glmbench.py --suites exact` 10/10 and `multiturn.py --modes batchexact` 4/4.
2. The W9 transcripts (6 prompts x greedy / sampled seed 1234 alone, plus 4 together): byte-identical to A.
3. Decode:
   - `multiturn.py --modes concurrent --streams 1,4 --reps 5 --long-tokens 512` (median of 5);
   - `glmbench.py --suites tf` single-stream;
   - per request, from the stats: `decode_s / rounds - round_kinds.verify_ms - round_kinds.draft_ms` (the
     between-round time; expect ~0.9 -> ≤ 0.3 ms with `plan`).
4. Cancel: a streamed request whose client disconnects after ~50 tokens, with 3 others decoding. It must end with
   `cancelled` in the request log, and the other 3 replies must equal their `batchexact` references.
5. Arrival latency: queue a 5th request while 4 decode; its `queued_s` may grow by at most one round.

### 3. Pinning check (loads C / D)

During a 4-stream run, `ps -L -o tid,psr,pcpu,comm -p <pid>` every 100 ms for 30 s on both ranks (read-only). Expect:

- `tf-serve` on cpu 19 only;
- `NCCL ...` names on 16-17. If NCCL 2.30 ignores `NCCL_SET_THREAD_NAME`, they stay on `rest`: record that;
- the RoCE proxy on 18;
- nothing else on 16-19.

### 4. nsys: one capture per server lifetime (W7's rule)

Load D (or B if C is not adopted), with the W7 harness: a 256-token single-stream prose reply and a 4-stream
384-token set. Then `analyze.py` + `dec.py` + the gap computation of §2. Expect:

- the median gap between verify ranges 0.83 -> ≤ 0.3 ms;
- 4 fewer NCCL control all-gathers a steady-state round;
- median in-round idle unchanged or ~0.1 ms lower (the sync);
- `skew.py` rank skew lower with pinning, if pinning matters at all.

### 5. Adopt / reject

- **Adopt `GLM53_TF_DECODE_OVERLAP=1`** if exact / batchexact / transcripts are clean and 1-stream decode is ≥ +1%
  (median of 5) with 4 streams not worse. If `emit` alone hurts 4 streams (GIL during eager launches), adopt
  `sync,plan,gil`.
- **Adopt `GLM53_TF_CPU_PIN`** separately if it is ≥ +1% on either stream count, or the rank skew drops, with
  nothing worse.
- **Revert:** unset the knobs.
