# PREFETCH-COMM: L2 prefetch in every decode path (E3) and the RoCE exchange's residual latency (E8), patch 0460

> Offline work (2026-09-29): no GPU, nothing on the Sparks. The measured inputs are W11 (docs/RESULTS.md "W11",
> `results/W11/tables.md`, `insitu-*.txt`, `probe-*.log`), W9 (RoCE bench) and the code of 0040 / 0230 / 0350 / 0370
> / 0440. Every gain below is an **estimate** from those numbers. Section 6 is the plan that turns them into
> measurements.
>
> "Exact" means what it means everywhere in this repo: drafted == serial, batched == alone and resumed == fresh hold,
> and the gathered bytes and every reduction order are unchanged. Patch 0460 changes no bit: its prefetch kernels only
> read weights, and its RoCE changes move the same bytes in the same order.

## 0. Bottom line

- **E3, the port, is real.** 0040's prefetch never ran where production decodes with more than one slot:
  - `batch.compute_multi` (the Batcher's multi-slot forward, `GLM53_TF_BATCH=4`) tagged no all-gather site and never
    joined a side stream.
  - A lone slot's rounds go through `Engine.forward`, i.e. `forward.compute`, which 0040 does cover. W11: 348 of 361
    single-stream rounds, "alone".
  - 0460 tags both paths identically. A CPU test checks it: compute_multi reaches the lone forward's sites in the
    same order, with the same rows.
- **E3, the extension.** 0460 adds two more sites:
  - `o`: after the attention projections, the output projection is prefetched while the KDA chains / DSA attention
    core run. This window grows with the slots.
  - `e`: after the router, the first tiles of the experts it picked, device-indexed. Off by default.

  0460 also changes the mechanism. `cp.async.bulk.prefetch.L2`, which ptxas accepts for sm_121, replaces 0040's
  Triton loads, which held half the SMs. And a 4-bit matrix whose launch is one wave gets the head of every
  (tile, K slice) chunk, not the first bytes of the tensor.
- **E8: the count cannot go down exactly, and the transport has little left.**
  - 100 / 115 exchanges a round (1 / 4 streams) is the minimum: 90 in the verify forward, 3 per MTP step, 1 sampler
    per verify. Every candidate fusion or deferral either changes bits or re-reads weights (section 3.1).
  - Of W11's 2.3 ms exposed exchange time at 1 stream, ~0.9 ms is waiting for the other rank (mean 23.1 us vs median
    14.5 us). That is compute jitter, which comm cannot remove.
  - The code-level cuts are all behind knobs: one HCA for small shards, release / acquire ordering in the kernel,
    inline posts for tiny payloads, lazy CQ reaping. Together they are worth ~0.1 ms a round.
  - The zero-copy idea is analysed and rejected (section 3.4). The new `GLM53_TF_ROCE_TRACE` timestamps measure the
    components on the GPU.
- **Estimate (mid):**
  - **1 stream:** -1.25 ms a round, **+2.3%**.
  - **4 streams:** -1.45 ms, **+1.2%**.
  - Range: +1.1% to +5% at 1 stream (section 4).
- **Knobs, all default off:**
  - `GLM53_TF_L2PF` (+ `_MB`, `_SITES`, `_EXPERT_KB`, `_ROWS`, `_CHUNK_KB`);
  - `GLM53_TF_ROCE_STRIPE_KB`, `_INLINE`, `_LAZY_CQ`, `_LEAN`, `_TRACE`.

## 1. The windows (W11, production config, rank 0)

| per round | 1 stream prose | 4 streams |
| --- | ---: | ---: |
| round (captured) | 54.6 ms, 2.42 tokens | 122.0 ms, 9.1 tokens |
| RoCE all-gathers, median (mean) | 100, 14.5 (23.1) us | 115, 20.5 (35.9) us |
| exchanges exposed | 2.3 ms | 4.2 ms |
| of it waiting for the peer: (mean - median) x count | ~0.86 ms | ~1.8 ms |
| hc (hc_pre / hc_post) | 1.45 ms | 1.8 ms |
| router / grouping / combine | 1.2 ms | 1.6 ms |
| KDA chains, conv, replay | 1.6 ms | 6.0 ms |
| DSA attention + indexer | 2.1 ms | 4.8 ms |
| small dense `_qmm` shapes below their read speed | (1, 32, 4) 55 calls x 28.5 us at 166 GB/s; (1, 64, 2) 77 x 18.8-34.5 us; (1, 64, 1) 68 x 4.4 us at 67 GB/s | similar |

During all of these, DRAM is mostly idle. W11's probe and in-situ numbers:

- The GB10 reads at **235 GB/s** when streaming.
- A single cold 1 / 2 / 4 MB launch reads at only 138 / 179 / 201 GB/s. Small matrices are latency-bound.
- The consumers after each window run below the plain-read speed of the same bytes:
  - KDA projection 197 vs 235 GB/s;
  - DSA o 189 vs 235;
  - the 2,048 x 4,096 shape (shared gate/up, DSA input projection) 166 vs 215.
- **GB10's L2 is 24.0 MB** (`probe-head.log`); 0460 caps a site at half of it.

## 2. E3: the design (patch 0460, `l2pf.py`, `l2pf.cu`, `l2pf.cpp`)

### 2.1 Sites

| site | fired | prefetched (in the order the next kernels read) | at 4 MiB, real shapes, a rank |
| --- | --- | --- | --- |
| `a` | at a layer's attention all-gather (`forward.gather`) | `ffn_hc.fn`, post norm; router, bias, shared expert gate/up (MoE) or the dense MLP's gate/up | MoE: fn 0.75 + router 2.25 + shared scales / biases 0.5 + 0.49 MiB of gate/up heads (128 chunks) |
| `f` | at a layer's FFN all-gather | the next layer's `attn_hc.fn`, input norm, KDA / DSA input projection; after the last layer: the final norm and head | to KDA: fn 0.75 + projection scales 1.54 + biases 1.54 + 0.17 MiB of words |
| `o` (new) | after the attention projections, before the core (`kda_block` / `dsa_block` / latent `dsa_block_latent` / `dsa_multi` / `batch._kda` / `batch._dsa`) | KDA: `o`; latent DSA: `kv_v` then `o` | KDA o: scales / biases 1.0 + 3.0 MiB of heads (128 chunks) |
| `e` (new, opt-in) | after the router (`moe_block`) | the first `EXPERT_KB` of each picked expert's gate and up trellis, from `qmm.Group` on the device | U 20 x 2 x 64 KiB = 2.5 MiB at 1 stream |
| `("mtp", a / f / o / e)` | the MTP head, alone and batched (`mtp_multi`) | as above; `f`: shared_head norm + the (0420) draft head | |

At 8 MiB the MoE `a` plan holds all of fn, the router and the shared expert (7.5 MiB), and the KDA `o` plan 8 MiB of
the 9.4 MB matrix.

**4-bit layout.** Q4 words are [N/64, K/64, 64, 8]. So one `qmm` program, a (64-column tile, K slice), streams one
contiguous chunk of K x 32 / SK bytes.

- **One-wave launches** (<= 192 programs: shared gate/up and DSA projection 128, KDA o 128, dense MLP gate/up 192)
  get the head of **every** chunk. The launch lasts as long as its slowest program, so prefetching the first bytes of
  the tensor would speed up only the first 10-20% of the programs.
- **Multi-wave launches** (KDA projection 788, DSA o 512, head 1,210) get their first bytes, the first wave's
  programs. Heads of later waves would be evicted by the earlier waves' stream.
- Scales and biases come first. Every program reads them, in scattered 128-byte pieces.

### 2.2 Mechanism

- **`bulk` (`=1`, default mode).** A 1-4 CTA kernel issues one `cp.async.bulk.prefetch.L2.global [addr], bytes` per
  32 KiB (`_CHUNK_KB`).
  - The bulk-copy unit streams the range into L2: no registers, no shared memory, no data back to the SM.
  - The kernel is ~20-30 instructions a thread and exits. It does not hold SMs the way 0040's loads did (24 CTAs
    looping over every word).
  - A table of chunk heads is spread over the threads.
- **`lines`.** One `prefetch.global.L2` per 128-byte line, the fallback if the bulk form turns out slower on GB10.
- **`touch`.** 0040's Triton loads over the same spans.
- **Where it runs.** Every launch goes on a side stream forked from the current stream at the site. It depends on the
  main stream's work so far and runs beside what follows, never in front of it. Every forward (`compute`,
  `compute_multi`, `mtp_compute`, `mtp_multi`) joins before it returns, so captured graphs hold no open fork.
  - An eager fork is never joined inside a capture (or the reverse). That could otherwise happen after a lean
    prefill that forked and never joined.
- **Expert site `e`.** A device-indexed kernel, grid min(48, MAXU):
  - block u reads `count` and `ids[u]` and prefetches `bases[m] + id x 2 MiB` for the gate and up trellis;
  - ids outside [0, E) are skipped.
- **Windows wider than `_ROWS`** (64) skip every site, so prefill chunks do nothing. The first `R <= 64` gate is in
  `gather`, which now passes R (0040's `Prefetcher.launch` takes it too, unused).
- **The EXL3 shared expert** is computed before the routed experts while any prefetcher is installed. This is 0040's
  order: disjoint inputs and outputs, same bits.
  - With 0130's `sharedout` it writes the shared slot of `ey` directly. The routed kernels (v1, v2 and 0440's)
    write only routed slots.
  - With `e` on, the picked experts' first tiles stream in during the shared expert, which reads at 166 GB/s: ~70
    GB/s of DRAM left idle for ~28 us.
- **`GLM53_TF_COMM=prefetch` (0040) and `GLM53_TF_L2PF`** refuse to load together.

### 2.3 Exactness

- **PTX check** (`tests/test_prefetch_comm_compile.py`): the prefetch kernels' only memory operations are `ld` (the
  table, the router's ids and count) and `cp.async.bulk.prefetch.L2.global` / `prefetch.global.L2`. There is no
  store, atomic or reduction.
- **Model kernels:** each runs with the same arguments on the same stream, in the same order, except the shared
  expert reorder above.
- **The link:** the all-gathers, their buffers and hc_post's rank-0-first sum are untouched.
- **PDL** (dv2, 0440): a fork is an event record in the stream, not a node between two programmatic kernels.
  Programmatic edges stay intact, and the fork's side node depends on the previous kernel's completion.

### 2.4 Interactions

- **0440** (streaming kernels, same weight tensors):
  - The plans still hit, since any prefetched line a kernel reads is a hit.
  - `DEC_PDL`'s own prologue prefetch covers each CTA's first item during the previous kernel's tail. `e` and `o`
    start earlier, one window before.
  - If E1 / E2 reach the roof, E3 still pays: it moves bytes out of the consumer into idle time. The latency-hiding
    half of E3 is what overlaps with 0440.
- **0370** (decode overlap): unaffected. Its next-round plan rides the sampler exchange, which is not a prefetch site.
- **Graph memory:** tables are ~0.3 MB in all. Nothing is allocated at launch.

## 3. E8: the exchange

### 3.1 Count: 100 / 115 is the minimum

| exchange | per | why it stays |
| --- | --- | --- |
| attention partial -> hc_post (45) and FFN partial -> hc_post (45) | verify forward | COMM-ANALYSIS 2(a): hc_post / hc_pre / RMSNorm / router are nonlinear in the gathered sum. Attention and FFN are sequential in this model (no parallel block), so neither can move past the other |
| MTP head: attention, MoE (2) + candidates (1) | MTP step | the next step's token depends on the candidates |
| sampler candidates (+0370's plan rider) | verify | already one exchange for every slot and row (0200 / 0370) |
| control (NCCL) | 2 a round, 0.02 ms | host-side, off the GPU's path |

The alternatives, each rejected:

- **Split the round into two micro-batches** so one's exchange hides behind the other's compute (DeepSeek-style).
  - Batched == alone makes it exact per row. But every weight is read twice: +2.75 GB dense and the experts'
    union split, i.e. **+12 ms a 4-stream round to hide ≤ 4.2 ms**.
- **Per-slot exchanges** (slot A attends while slot B's gather flies) multiply the count by the slots: a latency
  loss.
- **Replicating a block** (the 3 dense MLPs, the draft head's candidate exchange) re-reads 0.13 GB / ~40 MB a
  rank for 3 x 14.5 us / one exchange a step: a loss (COMM-ANALYSIS 2(a)).
- **Raise `GLM53_TF_ROCE_MAX_KB` to 512** for >16-row batched windows: W11 shows **no** model exchange on NCCL at 4
  streams (NCCL time is the 2 control exchanges only), so nothing to gain today.

### 3.2 One RoCE all-gather, component by component (from `roce.cu` / `roce.cpp`)

The estimates add up to W9's 11.7 us eager probe at 16 KiB and W11's 14.5 us in-situ median. `--trace` replaces them
with numbers (section 6).

| # | component | code | est. us (16-64 KiB) | 0460 cut |
| --- | --- | --- | ---: | --- |
| 1 | graph node dispatch | - | 0.5 (W11: 0.47-0.73) | - |
| 2 | stage the shard into pinned host memory | `copy_bytes` | 0.5-1.5 | - (3.4) |
| 3 | per-block `fence.sc.sys`, arrival atomic, last block's second `fence.sc.sys`, doorbell | step 2 | 1-2 | **`LEAN`**: one `atom.acq_rel.sys` arrival a block (none for 1 block) + a `st.release.sys` doorbell |
| 4 | proxy notices the doorbell (GPU store -> CPU cache line over C2C) | `main_loop` acquire load | 0.3-1 | - (pin the proxy: `GLM53_TF_ROCE_CPU`) |
| 5 | `ibv_post_send`, one per HCA (payload WR + inline flag WR) | `post_op` | 0.2-0.3 each | **`STRIPE_KB`**: one post below the threshold |
| 6 | NIC DMA-reads the payload (PCIe read round trip), wire, remote DMA write of payload + flag | - | 2-3 | **`INLINE`** skips the DMA read for stripes <= the QP's inline cap (sampler exchanges); **`STRIPE_KB`**: no max over two functions' paths |
| 7 | GPU sees the flag (`ld.acquire.sys` spin) | `wait_flag` | 0.5-1 | - |
| 8 | copy-out (peer slot with `ld.relaxed.sys`, local shard) | step 4 | 1-2 | - |
| 9 | tail: `fence.sc.gpu` x2, arrival, epoch | step 5 | 0.5-1 | **`LEAN`**: no fences (the next launch reads the epoch after a kernel boundary), one `atom.acq_rel.gpu` |
| - | proxy reaps completions right after posting | `post_op` | 0.2-0.4 (next op's path only) | **`LAZY_CQ`** |
| | **sum** | | **~8-13** | **-0.7 to -2.5** |
| + | waiting for the peer | | mean - median: 8.6 (1s), 15.4 (4s) | none (jitter of compute) |

**Protocol argument for the changes** (the send slot, the receive slot and the doorbell's catch-up are 0230's):

- **`STRIPE_KB`.** The HCA set of an op is a pure function of (padded bytes, seq), `roce_common.h one_hca /
  single_hca`, evaluated identically by the kernel (which flags to wait for), the proxy (which QPs to post on) and
  the test loop. Both ranks have equal shard sizes (an all-gather), and the threshold is in `Settings.agreed()`.
  - Each used HCA still carries its payload, then its flag, on one RC QP, so "flag after payload" holds per op.
    Slot reuse needs only per-op completion, not cross-op ordering.
  - Flags hold exact sequence numbers, so an unused HCA's stale flag can never equal a later seq.
- **`INLINE`.** The proxy reads the send slot after its acquire load of the doorbell, which the kernel published
  with a system release after the staging stores. So it copies the op's own bytes. At post time the slot cannot yet
  be restaged for seq + 2 (0230's two-slot argument).
- **`LEAN`.** The same happens-before edges:
  - staging stores -> `bar.sync` -> the block's release RMW -> the last block's acquiring RMW -> the release store
    of the doorbell -> the proxy's acquire load;
  - poison CAS -> `bar.sync` -> release arrival -> the last block's acquire -> its poison read.

  The classic path is unchanged and stays the default.

`tests/test_roce_protocol_model.py` (extended) runs the protocol model under random schedules with mixed sizes, both
thresholds and inline posts, and keeps every invariant. It also checks that the payload a proxy copies inline is its
op's own. Its controls time out: a kernel waiting for another HCA than the sender used, and ranks with different
thresholds.

### 3.3 Measuring it: `GLM53_TF_ROCE_TRACE=N`

- **Kernel side.** A device ring of N ops x 4 globaltimer stamps: start, doorbell, flags seen, end.
- **Proxy side.** CLOCK_REALTIME stamps: doorbell seen, posts returned.
- **`roce.trace_summary`** gives p50 / p90 of:
  - stage;
  - wait;
  - copy;
  - total;
  - notice (valid if the driver keeps globaltimer on the host clock: check that it is small and positive);
  - post;
  - across both ranks, **transport** (the smaller of the two ranks' waits per op, when neither waited for a late
    peer) and **skew**.
- **In the bench:** `bench_roce.py bench --trace 4096` prints this per size for the graph run.

### 3.4 Zero-copy (NIC DMA straight from the tensors): analysed, not built

- **Device allocations are not registrable.** `ibv_reg_mr` pins CPU pages (get_user_pages). The tensors are
  `cudaMalloc` memory, which on GB10 lives in the same LPDDR5x but is not mapped in the process's CPU page tables.
  Registering it needs a dma-buf export of GPU memory, i.e. GPUDirect RDMA, which DGX Spark does not support
  (PARADIGMS.md #15, 0230). Unified memory makes the pages physically shared, not NIC-registrable.
- **Moving the buffers to pinned host memory works mechanically but loses:**
  - Send side (NIC reads `b.part` directly). `b.part` is rewritten by the next block ~30 us later. No 0230 invariant
    says our NIC has finished reading it by then: the peer's flag says nothing about our own send. The kernel would
    have to wait for our own completion (CQE -> proxy -> flag: +1-3 us on the critical path) to save a 16-64 KiB
    staging copy (~0.5-1.5 us).
  - Receive side (the NIC writes into `b.gath`). The peer may deliver seq + 2 while hc_post still reads seq, which
    is safe only with two parity buffers. The consumers (hc_post, residual adds, samplers) are captured with fixed
    addresses, and the parity of an op inside a replayed graph depends on how many exchanges ran before it. So it
    needs parity-keyed graphs (twice the graphs) or parity-indexed pointers in the glue kernels.
- **Worth at most ~1 us an exchange (0.1 ms a round)** against a protocol and graph change. Not built.
- **The larger step, if E8 is ever pursued further:** a proxy-less post. The GPU writes the WQE into the (host)
  send queue and rings the NIC doorbell through the UAR page mapped with `cudaHostRegister(..., IoMemory)`, i.e.
  IBGDA over host memory. That removes components 4-5 (~1-1.5 us). It needs mlx5dv, the WQE format, and a
  mapping GB10 may not allow. First test: `cudaHostRegister` of an mlx5 UAR page.

## 4. Estimate (ms a round; W11 rounds 54.6 / 122.0 ms)

**Model for E3.** A site that prefetches B bytes saves `eta x B / 230 GB/s` of its consumer's time. eta is low 0.25,
mid 0.4, high 0.8. It covers L2 hits not being free and the consumers' latency-bound read rates, which work in
opposite directions. The windows (25-35 us at `a` / `f`, 20-90 us at `o`) fit 4 MiB at ~200 GB/s.

Sites a round:

- 1 stream: 135 in the verify forward + ~9 in MTP steps, at 4 MiB each (18 us DRAM-equivalent).
- 4 streams: the same count (sites are per layer, not per slot), with wider windows.
- `e`: B = U x 128 KiB, eta halved (it shares the shared expert's DRAM).

| item | 1 stream low / **mid** / high | 4 streams low / **mid** / high |
| --- | --- | --- |
| E3 `a,f,o` at 4 MiB | 0.65 / **1.05** / 2.1 | 0.65 / **1.05** / 2.1 (8 MiB may fit: up to ~3) |
| E3 `e` (opt-in) | 0 / **0.15** / 0.4 | 0 / **0.3** / 0.7 |
| E8 code knobs (`LEAN`, `STRIPE_KB=32`, `INLINE=256`, `LAZY_CQ`) | 0.07 / **0.1** / 0.25 | 0.08 / **0.12** / 0.3 |
| E8 proxy pinned to an idle X925 core (config: `GLM53_TF_ROCE_CPU`) | 0 / **?** / 0.3 | 0 / **?** / 0.5 |
| **0460 total** | **0.7 / 1.3 / 2.75 ms: +1.3% / +2.4% / +5.3%** | **0.7 / 1.5 / 3.1 ms: +0.6% / +1.2% / +2.6%** |

Against DECODE-PLAN:

- **E3 is higher** than its 0.3 / 0.7 / 1.5 (1 stream) and 0.3 / 0.8 / 1.5 (4 streams). The `o` site, the
  per-chunk heads for one-wave matrices and W11's slow small shapes add to it.
- **E8 is much lower** than its W11-revised 0.4 / 0.8 / 1.5 (1 stream) and 0.8 / 1.5 / 2.5 (4 streams). Those
  ranges assume the peer-wait skew (mean - median, 0.86 / 1.8 ms) can be cut. It is compute jitter between the
  ranks, which no transport change removes. The transport itself sits near its host-staged floor (section 3.2).
  What stays open for skew:
  - pinning the proxy (config);
  - making the ranks' kernels less variable, i.e. the E1 / E2 / E4 work;
  - a proxy-less post (section 3.4), ~0.1-0.15 ms a round.

## 5. Offline results

| check | result |
| --- | --- |
| `tests/test_prefetch_comm.py` (torch CPU + Triton 3.7.1 import) | 22 passed. Knob parsing; plans on fake weights (every span inside a weight, aligned, <= budget, order); per-chunk heads for one-wave Q4 launches, first bytes for multi-wave; expert plans; MTP keys; **compute_multi reaches the lone forward's sites a / o / e / f per layer in order, with the round's rows, and joins** (and fails with the site tags removed: checked); mtp_multi's head sites; hooks are no-ops without a 0460 prefetcher and 0040's still gets a / f only; the one-HCA rule; RoCE knobs and `agreed()`; `trace_summary` |
| `tests/test_roce_protocol_model.py` (extended) | 166 passed: 0350's 90 + 76 new (one-HCA x inline x thresholds under random schedules, 3 ranks, the queue-pair choice, two controls that time out) |
| `tests/test_prefetch_comm_compile.py` (nvcc 13.4, sm_121) | 4 passed. `l2pf.cu`: 4 kernels, 22-30 registers, no stack, no spills; only loads and prefetches in the PTX. `roce.cu`: 64 registers, 0 spills (as 0350); lean forms present, byte counts stored before the release doorbell. `l2pf.cpp` / `roce.cpp` pass `g++ -fsyntax-only` against torch 2.14 + CUDA 13.4 + verbs headers |
| regressions | `test_roce_logic.py`, `test_bench_roce_stress.py`, `test_roce_watch.py`, host parts of `test_batch_parallel_patches.py`, `test_decode_overlap_patches.py`, `test_batch_sessions_patches.py`, `test_draft_vocab_patches.py`: pass |
| series | 0001-0440 + 0460 apply in order on `2f8e514` (the Dockerfile's `git apply` loop); the tree equals the working tree the tests ran on |

## 6. GPU test plan (~70 min; production stopped on both nodes for steps 2-4)

Image: the committed series (0001-0440, 0460). Logs go to `results/W<n>/`.

**1. Unit tests, one GPU (~10 min).**
- `tests/cuda/test_prefetch_comm_patches.py`:
  - L2PF bulk / lines / touch with sites a, f, o, e: logits and hidden rows of windows 1-8 (graphs + eager) == off;
    every site launched in an eager forward; replies (serial, MTP, DFlash2, auto) == serial, greedy and sampled;
    no weight byte changed;
  - **the Batcher**: 4 slots with graph / pad / parity / batched MTP knobs, replies == serial and prefetch launches
    grow;
  - RoCE loop runtime with `STRIPE_KB` 0 / 32 x `LEAN` 0 / 1: every size and alignment eager and in graphs;
    ordered trace stamps.
- Regressions: `test_roce_patches.py` (extension now `tensorfold_glm_roce_v3`), `test_comm_patches.py` (0040),
  `test_batch_parallel_patches.py`, `test_decode_stream_patches.py -k pdl` (PDL with side-stream forks).

**2. RoCE bench, both nodes (~15 min).** `bench_roce.py bench --trace 4096 --sizes 16k,32k,64k,128k,256k`. Run it for
baseline, `LEAN=1`, `STRIPE_KB=32`, `STRIPE_KB=64`, `INLINE=256` (check `inline_cap` in the stats line) and all of
them with `LAZY_CQ=1`. Record, per size:
- the graph us;
- the trace components (stage / wait / copy / notice / post / transport / skew).

Keep a knob only if the graph latency drops by ≥ 0.3 us at 16-64 KiB with bits equal. `STRIPE_KB` is expected to
help at 16 KiB and hurt at 64 KiB: pick the threshold from the table.

**3. RoCE safety for the chosen knobs (~20 min).**
- `stress --loop` per function (`GLM53_TF_ROCE_HCA=`...) and striped, 16k / 32k / 128k.
- Two-node `stress` (100k a size and mode).
- `fault` (must still raise `never` in ~3 s).
- `soak --minutes 10`.
- Pass: 0 mismatched words, no failure.

**4. Engine A/B (~25 min), alternating loads, same image.**
- Control (prod env) vs `GLM53_TF_L2PF=1` (sites a,f,o, 4 MiB) vs `+ _SITES=a,f,o,e` vs `_MB=8`. Then the best L2PF
  plus the chosen RoCE knobs.
- Exactness on each: `exact` 10/10, `batchexact` 4/4, greedy + sampled transcripts of the fixed prompts
  byte-identical to control.
- Speed:
  - decode tok/s at 1 stream (prose and code, median of 5);
  - 4 streams (`multiturn.py --modes concurrent --streams 4 --reps 3`, two passes, as W11 `graphs.sh`);
  - the load line `l2pf: bulk, 4 MiB a site, ...`.
- Profile: one nsys capture (W11 `cap.sh`) with the best setting. The `segments_kernel` / `experts_kernel` should run
  beside `gather_kernel` / the KDA chains, and the in-situ times of `_qmm` (1,197,4), (1,32,4), (1,64,2) and
  `grouped_kernel` should drop against `insitu-prose-r0.txt`.

**Gates.**
- **L2PF:** adopt at ≥ +1.5% single stream, 4 streams not lower, every exactness check equal.
- **RoCE knobs:** step 3 clean, step 2 faster, decode not lower. Soak 1 h before leaving it unattended, as for 0350.
- **If `bulk` is not faster than `lines`** (the bulk unit may serialize), use `lines`. If neither beats `touch`,
  E3's mechanism is wrong and the doc's numbers go.

## 7. Risks

- **The bulk prefetch's real throughput on GB10 is unknown.** The kernel may not retire before the transfer
  finishes, which costs a side-stream CTA slot, not correctness. `lines` and `touch` are the fallbacks.
- **L2 pollution.** 4 MiB a site is evicted by the next expert stream anyway. The risk is evicting the consumers' own
  lines at 8 MiB with 4 slots. Sweep `_MB`.
- **Timing only, never bits.** Timing does move the peer skew; it cannot move bits.
- **`LEAN` relies on the PTX memory model's cumulativity through `bar.sync`**, the same property 0230's per-block
  fence relies on. The stress mode is the empirical check. Keep it off until step 3 has run 100k ops a size and mode
  on both functions.
- **`STRIPE_KB` is agreed at load; `INLINE` / `LAZY_CQ` / `LEAN` are local.** Ranks may differ in the local knobs
  without affecting the protocol.
