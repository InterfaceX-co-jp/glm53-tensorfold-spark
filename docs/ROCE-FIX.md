# RoCE all-gather: the W2 loopback failure and patch 0350

Patch 0350 builds on 0230 (`GLM53_TF_COMM_BACKEND=roce`, still opt-in; production stays on NCCL). It was written
offline: no GPU, no RDMA device, nothing on the Sparks. Everything below that needs the hardware is a plan.

## What failed in W2

`results/W2/roce-loopback.log` is `tests/cuda/bench_roce.py loopback` on the head node, run over both CX7 functions with
the 16k size first. Rank 0 timed out at **sequence 312** on `roceP2p1s0f1`. The log line reads: "the flag HAS reached
this host's memory (the GPU did not observe it: sparkring #278's signature); doorbell 312, completed 311, this rank's
proxy posted up to 312".

## The cause: the harness, not the memory model

`loopback` drives **both** ranks from one process. It creates two `Runtime`s on one GPU and joins them through the
NIC's loopback. A rank's collective only completes once the peer's collective of the same sequence runs, because it
waits for the peer's flag. In 0230, `loopback` did this:

1. `both()` once (sequence 1), then `_time_eager(both, iters)`: 10 warm-ups plus `iters` (300) timed calls, which is
   sequences 2 to 311. Every call launches rank 0 and rank 1 together, so all of these complete.
2. `_graph(lambda: rts[0].gather(...))`. It runs `2 x ops` (180) eager warm-ups of **rank 0 alone** on a side stream,
   then `torch.cuda.synchronize()`. Rank 0's first warm-up is **sequence 312**, and it waits for rank 1's 312. The
   host only launches rank 1's warm-ups after `synchronize` returns, so rank 0 waits out its timeout (10 s), records
   the failure and poisons itself. Its remaining 179 warm-ups do nothing.
3. `_graph` for rank 1. Rank 1's warm-up of sequence 312 completes, because rank 0 had rung its doorbell before it
   waited, so rank 0's proxy had sent 312. Rank 1's proxy then writes flag 312 into rank 0's host memory, **about 10 s
   after rank 0 gave up**. Rank 1's 313 never comes, so rank 1 times out as well.
4. `for rt in rts: rt.check()` raises rank 0's error first. 0230's diagnosis read the flag word **at check time**,
   found 312 and called it "delivered but not seen".

So 312 = 1 + (10 + 300) + 1, and every other number in the log line follows: doorbell 312 (the poisoned rank rings
nothing more), completed 311, posted 312. `tests/test_roce_protocol_model.py` is an executable model of 0230's protocol
(kernel, doorbell, proxy with catch-up, RC queue pairs per HCA, per-HCA flags, two slots) driven by the old harness.
It reproduces the whole log line for one and two HCAs.

**Prediction for the GPU re-run** (W7): unchanged 0230 code fails at sequence `iters + 12`. That is **312** with the
default `--iters 300`, whichever HCA gets named and also with `GLM53_TF_ROCE_HCAS=1`; `--iters 100` gives 112. The same
image running 0350's `loopback` (the harness fix alone, with 0230's runtime) completes. A failure at another sequence,
or one that persists with the fixed harness, would point at a real transport problem (see "if it still fails").

### The memory-model audit (roce.cu / roce.cpp on GB10)

I checked each hypothesis against the code and the sm_121 PTX, compiled offline with nvcc 13.4:

| hypothesis | 0230 | verdict |
| --- | --- | --- |
| GPU polls the flag with a cached / non-volatile load | `ld.acquire.sys.global.u32` inside the loop (`asm volatile`, re-issued every iteration in the PTX) | correct |
| payload read through a stale cache | `ld.relaxed.sys` (v4 / u32 / u8) after the acquire; `bar.sync` orders the other threads of the block | correct |
| flag memory not mapped / not portable | `cudaHostAlloc(Mapped \| Portable)`, device pointer from `cudaHostGetDevicePointer`, the same region registered with `ibv_reg_mr` | correct |
| write-combined memory | not requested (`cudaHostAllocWriteCombined` absent); cached, coherent host memory | correct |
| NIC writes data and flag out of order | payload WR, then the 4-byte flag WR, on the **same RC queue pair** per HCA stripe; the kernel waits for every HCA's flag; the MR is registered **without** `IBV_ACCESS_RELAXED_ORDERING` | correct, provided the firmware does not force PCIe relaxed ordering (`PCI_WR_ORDERING`, see the plan) |
| doorbell seen before the staged payload | per-block `fence.sc.sys`, then a gpu-scope arrival atomic; the last block runs `fence.sc.sys`, then the doorbell store; the proxy uses an acquire load | correct |
| send slot overwritten before the NIC read it; receive slot overwritten before it was read | two slots; staging `seq + 2` needs the peer's `seq + 1`, which needs our flag `seq`, which the QP delivers after our payload `seq`; the model checks this under random schedules | correct (checked) |
| graph-captured kernels spin on a stale epoch | the epoch is in device memory, published with `st.release.gpu` by the last block, and read by the next launch | correct |

None of these explains the W2 failure; the harness does. 0350 still fixes two real defects that the investigation
exposed:

## Patch 0350

1. **A diagnosis that tells "late" from "not seen"** (`roce_watch.h`, `roce.cu`, `roce.py`).
   - The kernel stores its last read of the flag (`CTRL_ERR_SEEN`, a new control word).
   - Up to blocks x HCAs threads can time out in one launch (16 at the defaults). Only the first writes the record: it
     claims it with a compare-and-swap on the device poison word, so the record is never torn. The failure word is
     written last, after a system fence.
   - The epoch is published only when the device poison is clear.
   - The busy-spinning proxy thread polls a `FailWatch` on every iteration. That is one acquire load of the failure
     word, on the doorbell's cache line. When the word turns 1, the watch records the flag word as host memory holds
     it **at that moment**, then keeps watching and records how long after the failure the flag arrived, if it does.
     The test loop-back thread runs the same watch.
   - `roce.classify` turns the record into one of four verdicts:
     - `not_seen`: the flag was in host memory at the failure while the GPU read a stale value. This is a real
       visibility problem (sparkring #278).
     - `late`: the flag arrived N ms after the failure, so the peer was late or had not launched.
     - `never`: the flag did not arrive.
     - `unknown`: there is no watch record; only the check-time value is available, and the message says so.
   - The W2 failure classifies as `late`.
2. **The loopback harness** (`tests/cuda/bench_roce.py`). `_graph_pair` interleaves both ranks' warm-ups op by op,
   each rank on its own stream, then captures each rank's graph. Captures execute nothing, so capturing one rank at a
   time is safe. Graph replays of both ranks are enqueued together.
3. **A stress mode whose payload changes every op** (`bench_roce.py stress`). 0230's `soak` replays the same bytes
   every time, so a receive slot one or two sequences stale would still compare equal. In `stress`, every rank adds 1
   to its int32 payload on every op, the expected gathered words move with it, and mismatches are counted on the device
   and checked every `--check` ops (default 10,000). It runs eager and as replayed graphs of `--ops` (90), at 16k and
   128k, with `--iters` ops per size and mode (default 100,000). `--loop` runs it on one node with both ranks in the
   process; without `--loop` it runs across two nodes like `bench`.

Unchanged: the protocol, the wire format, `connect` / `probe` / fallback / marker. The torch extension is renamed
`tensorfold_glm_roce_v2`, so a cached 0230 build is never loaded.

## Offline results (2026-09-28)

| check | result |
| --- | --- |
| `tests/test_roce_protocol_model.py` (pure Python) | 90 passed. The old harness reproduces W2 exactly (seq `iters + 12`, doorbell, completed, posted, flag at check), 1 and 2 HCAs, `iters` 300 / 100 / 7. The watch says `late`. The fixed harness completes. 80 random schedules (1 and 2 HCAs, lagging proxies, host bursts) keep every invariant. 3 ranks OK. Controls: a flag posted on another QP than its payload is caught as a stale read, and a proxy lag bound of 1 is exceeded |
| `tests/test_roce_watch.py` (`roce_watch.h` with g++, `-Wall -Werror`) | 7 passed: idle, `not_seen`, `late` (~30 ms measured), `never`, an out-of-range record, 2,000 racing records read whole, `classify` |
| `tests/test_bench_roce_stress.py` (fake runtimes on the CPU) | 9 passed: a clean run passes; one stale peer slot (eager or inside a graph replay, either rank) counts all 4,096 words; a failed runtime stops the run |
| `tests/test_roce_logic.py` (0230 host tests on the 0350 tree) | 15 passed |
| `roce.cu` for sm_121 (nvcc 13.4) | 64 registers, 0 spills, 8 B stack (the wait's out-parameter) |
| `roce.cpp` (g++ C++20 against torch 2.14 / CUDA 13 / verbs headers) | syntax OK |
| 0001-0340 + 0350 + 0360 applied in order on `2f8e514` | clean |

## GPU plan (for the next window)

Run each step on the committed image with 0350; stop at the first failure. The logs go in `results/W<n>/`.

**0. Preflight (read-only).**
- `ibv_devinfo` on both nodes.
- `mlxconfig -d <dev> q | grep -i PCI_WR_ORDERING` for both functions. This must not be `force_relax` (1). If it is,
  data can overtake its flag, and `stress` is the test that would show it.
- `lspci -vvv -s <bdf> | grep -i RlxdOrd` for the record.

**1. Single-node loopback, the head node, production stopped (as W2).**
- `bench_roce.py loopback`, both functions. Expect `bits_equal` true for every size, and the eager-pair / graph
  latencies.
- `GLM53_TF_ROCE_HCAS=1` (the first function), then `GLM53_TF_ROCE_HCA=rocep1s0f1` and
  `GLM53_TF_ROCE_HCA=roceP2p1s0f1` alone.

**2. Single-node stress: 100k iterations, both functions.** Three runs:
- `bench_roce.py stress --loop` (striped);
- `GLM53_TF_ROCE_HCA=rocep1s0f1 bench_roce.py stress --loop`;
- `GLM53_TF_ROCE_HCA=roceP2p1s0f1 bench_roce.py stress --loop`.

Each run covers 16k and 128k, eager and graph: 100,000 ops per size and mode, `--check 10000`. Pass: every JSON line
has `mismatched_words` 0 and `failed` null, and `per_hca` shows writes on each function in use. About 1-3 min per run
at 10-30 us an op. On a timeout, the message now says `not_seen` / `late` / `never`.

**3. Two nodes, 0230's staged plan** (`--master $HEAD_IP`, rank 0 on the head node, rank 1 on the worker node, production stopped
on both):
- a. `bench`, default sizes 16k, 64k, 128k, 1m. Bits equal to NCCL, with eager / graph / step latency. Record the
  per-step saving line.
- b. `fault`. Rank 0 must raise within ~3 s with `never` (rank 1 never launched that op). Rank 1's next op completes,
  as the mode prints.
- c. `stress` (two nodes, 100k per size and mode), then `soak --minutes 30`.
- d. **Engine A/B**, one load each, same image, same env except `GLM53_TF_COMM_BACKEND`:
  - `nccl` vs `roce`: `exact` 10/10 on both;
  - greedy and sampled transcripts of the fixed prompts byte-identical between the two backends (an all-gather moves
    bytes, so the replies must not change);
  - decode tok/s at 1 stream and at 4 streams (the production batch), median of 5, thinking off, as in docs/RESULTS.md
    "Decode";
  - check `/metrics` for a RoCE failure or fallback line.

**Adoption rule.** Adopt RoCE when all of these hold: steps 1-3 are clean (no timeout, no mismatch, NCCL fallback
never taken), the transcripts are identical, and decode improves by >= 3% (single stream, or the 4-stream aggregate
production runs). Otherwise leave `GLM53_TF_COMM_BACKEND=nccl`. If adopted, set `GLM53_TF_ROCE_MARK` to the default
`/cache/roce-failed` (a run-time failure then pins the restart to NCCL), re-arm the watchdog, and soak for an hour
before leaving it unattended.

### If it still fails

- The fixed `loopback` or `stress` times out with `not_seen`. The GPU really did miss a flag that was in host memory
  (sparkring #278). Next: rerun with `GLM53_TF_ROCE_HCAS=1` on each function, then check whether one function or
  both is affected.
- `late` in the two-node runs. The peer was late: another process was on the GPU, or a peer was compiling. Raise
  `GLM53_TF_ROCE_TIMEOUT_S`, which is 120 s in the engine and 10 s in the bench.
- `never`, with the peer's proxy showing `ops_posted` past the sequence. RDMA writes were lost: look at CQ errors
  (`proxy_error`) and at the port counters (`ethtool -S`, `rdma statistic`).
- `mismatched_words` > 0. Data overtook the flag. Check `PCI_WR_ORDERING` first.

## Risks

- The diagnosis is only as good as the watch's timing. It records the host's view within microseconds of the failure
  word, and the failure word follows the kernel's last read by one fence. A flag arriving inside that window would be
  called `not_seen`. Over a 10-120 s timeout that is very unlikely, and the message gives the times.
- `stress` needs the payload to be a multiple of 4 bytes (it rounds the size down). Unaligned sizes are covered by
  `bench` and by the load-time probe.
- 0350 changes the kernel slightly: the CAS, one control word, and publishing the epoch on the device poison. The GPU
  unit tests (`tests/cuda/test_roce_patches.py`, now with `test_watch_records_the_failure_never` and
  `test_late_peer_is_late_not_unseen`) must pass before any RDMA run.

## W9 (2026-09-28): the plan run on the GPUs

Image `glm53-tensorfold:b2` (0230 + 0350). Logs: `results/W9/roce/` (one node), `results/W9/roce2/` (two nodes).
Summary and the engine A/B: docs/RESULTS.md "W9: batch 3".

- **0. Preflight.** `PCI_WR_ORDERING = per_mkey(0)` on all four CX7 functions of the head node (not `force_relax`); the MR is
  registered without relaxed ordering, so the NIC orders data before flag. `RlxdOrd+` in DevCtl only allows it.
- **GPU unit tests** (`test_roce_patches.py`): 25 passed.
- **W7's 1 MiB `bits_equal: false` was the harness.** `loopback` builds its inputs with `pattern()` on the default
  stream and gathers them on non-blocking side streams without waiting. `results/W9/roce_sizes.py`: with that race,
  first gathers differ in 0/20 trials at 16k-256k, but 19/20 at 512 KiB, 20/20 at 1-4 MiB, including the rank's OWN
  shard (copied straight from its input, never over the wire). With a synchronize in between: no difference in 20 trials at
  16k, 128k and 4 MiB and 320 trials each at 256 KiB-2 MiB (two gathers a trial). The fixed `loopback` gives `bits_equal: true` at 1 MiB; two nodes
  (`bench`, input on the same stream) are equal at 1 MiB too. One gather at 2 MiB (race mode, second gather of fully
  written inputs) differed once and never again in 360 more 2 MiB gathers; no detail was captured (above the engine's
  256 KiB limit; noted as unexplained).
- **Three more harness bugs, all fixed in `tests/cuda/bench_roce.py`** (none of these modes had run on a GPU before):
  1. `stress --loop`: the first op's mismatch check allocates, and a first-time device allocation can synchronize the
     device; with both ranks on ONE host thread that blocked the host behind rank 0's spinning gather before it
     launched rank 1's (10 s `late` timeout at sequence 1). The check's temporaries are now allocated before the run.
     Two nodes (one rank a process, as in the engine) cannot deadlock this way.
  2. `fault`: rank 1's "next exchange" was launched immediately, so it WAS the sequence rank 0 waited for (rank 0:
     "NO ERROR after 0.00 s"). Rank 1 now sleeps past rank 0's timeout: rank 0 raises after 3.00 s with `never`,
     rank 1's next exchange completes, as specified.
  3. `soak`: the graphs' input / output tensors were not kept alive (only the graph and z were), so the 16 KiB
     graph replayed into memory reused by the 128 KiB set ("DIFFERENT BYTES after 0 replays", every byte, both
     ranks). Kept now. And the ranks stopped on their own clocks (one rank started a replay the other never joined:
     120 s `never` at the end); rank 0's clock now decides over NCCL.
- **Stages 1-3 after the fixes**: loopback bits equal at 16k/64k/128k/1m on both functions, `HCAS=1`, and each
  function alone; `stress --loop` 100k ops a size and mode (16k, 128k; eager and graph) on both functions and each
  alone, plus 20k at 256k and 1m: 0 mismatched words, no failure; two-node `bench` bits equal at every size, `stress`
  100k a size and mode clean, `soak` 20 min (about 69.5M ops a rank, each replay compared, before the end-condition
  bug) plus 3 min after its fix (113,664 replays): clean.
