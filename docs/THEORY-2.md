# THEORY-2: what is left in the decode round, read from the traces (2026-09-29)

> Offline research. No GPU was used, nothing was started on the Sparks, and nothing was pushed. The only Spark access
> was a read-only `scp` of the four W11 nsys reports from the head node (`~/w11-traces/`, 214 MB) and a read-only listing
> of `results/W12/` there. The reports were exported to sqlite on this machine, never on a prod node.
>
> Inputs:
>
> - the W11 traces (1 stream: `cap-r0/r1`, 1,004 rounds a rank; 4 streams: `cap4-r0/r1`, 178 four-slot rounds);
> - `results/W11/` (tables, `probe-*.log`, `kbench.log`, `insitu-*.txt`, clocks) and `results/W12/` (A/B logs,
>   `kbench.json`), which has no write-up in RESULTS.md yet;
> - RESULTS W5 / W7 / W8 / W11, PROFILE, ROOFLINE, DECODE-PLAN, DECODE-KERNELS, GPU-ROUND, PREFETCH-COMM, SFXNZ-AUDIT.
>
> Every script is in `results/theory2/` with its output next to it (§8). Numbers marked *trace* are new measurements
> from the W11 traces. Numbers marked *est.* are estimates. Each estimate comes with the prototype that would test it
> in the server before any full build.

## 0. Bottom line

- **The round is one serial chain on one CUDA stream.** In both traces 99.97% of decode kernels run on stream 7, and
  two kernels are in flight for only 0.035 ms of a 53 ms round (*trace*).
  - At one stream, the chain verify → sample → draft → verify has no slack. Each step needs the previous step's
    final hidden rows or tokens.
  - The GPU is 96.6% busy, and its big kernels already stream at 205-217 GB/s against a 235 GB/s read ceiling.
  - So "more parallel execution" has little to win at one stream. Filling DRAM-idle windows was measured by 0460:
    +0.8%. At four streams the per-slot work is independent, so parallelism does have room there.
- **Three of the idle and waiting costs had not been attributed before (*trace*):**
  - **Verify graph launch.** At one stream, 1.16-1.23 ms of the 1.8 ms GPU idle a round is one gap at the start of
    the verify forward. The host is inside `cudaGraphLaunch` of the ~1,650-node verify graph (1,080-1,145 µs a call),
    and the graph's first kernel starts only when the call returns. This trace used `--cuda-graph-trace=node`, which
    inflates graph launches: the captured-minus-uncaptured round is +1.0-1.3 ms at one stream (graph-heavy) and +0.3%
    at four streams (eager). So the uncaptured size is somewhere between 0.1 and 1.1 ms, and a 10-minute probe
    settles it.
  - **Rank skew.** 1.26 / 1.49 ms a round (rank 0 / rank 1) at one stream, and 2.2 / 3.0 ms at four streams, is a
    rank waiting inside all-gathers for the other rank. The pure transport is only 10.8 µs an exchange (the shorter
    of the two ranks' gather times). The rest is run-to-run jitter of identical kernels: p5-p95 is 9-17% of the median
    for the same kernel on the same bytes, and the MoE blocks carry most of it. At four streams, rank 0 (the HTTP
    node) is also systematically late (62% of exchanges).
  - **Host-bound idle at four streams.** 3.9 ms a round. The sampling readback and numpy draw cost 2.3 ms and the
    per-slot accept / commit cost 1.3 ms. Eager launches cost 14.8 µs of host time each (2,099 a round, 31 ms of host
    time) but stay hidden behind the GPU.
- **Why the offline estimates failed (§2).** Each failure has a specific, checkable cause, and none of them is
  contention: decode runs one stream.
  - 0440 E2 was benchmarked on L2-resident small matrices: roofs of 311-1,041 GB/s against 235 from DRAM. On the cold
    large shapes the same bench already showed it 9-18% slower. Those cold rows predict an in-server -1.5 to -3%
    (measured -3.9%).
  - E1 assumed ~1 µs loaded latency (5 KB in flight an SM). Its own no-decode / no-mma probes ran at 140-160 GB/s, so
    the pipeline, not the ALU, was the limit.
  - 0460's ceiling is the L2-hot speed-up of small shapes, about 1.3x on ~5 ms, i.e. ~0.5 ms. It measured +0.8%.
  - 0450's -18% on greedy code has no trace behind it. The likely cause is capture rounds and drains on short,
    high-acceptance requests (§2).
- **No single remaining engineering item has ≥ 5% measured-grounded potential.**
  - The largest estimated item (a memory-pipeline rewrite of the expert / dense kernels, up to ~7%) sits behind a
    gate that two earlier designs failed.
  - The top of the ranked list is small, cheap and grounded: a config / policy change measured twice, a 10-minute
    launch probe, a fused layer-boundary kernel, per-slot batching at four streams, and rank-skew hygiene.
  - Items 1-6 stacked are about **+4-9% at one stream and +6-11% at four streams** (*est.*).
  - The large lever is still the drafters (DECODE-PLAN T1 / T2).

**Ranked list** (details and prototypes in §4; "1s" and "4s" = the 53.3 ms / 121.3 ms W11 rounds):

| # | idea | evidence | 1s | 4s | prototype (build / GPU) | full effort |
| ---: | --- | --- | ---: | ---: | --- | ---: |
| 1 | Batched graphs off only when ≥ 2 slots are active; adopt L2PF (0460) | G0 measured +2.6% (W11), +1.9% (W12) at 4s, but lone requests -2.7% (slots 1-3 ~-3%); L2PF +0.8% 1s, +2.4% lone slots, 4s flat | +0.8% | +2-2.6% | 0.5 d / 20 min | 0.5 d |
| 2 | Verify graph launch off the critical path (split the graph, or launch before the draft readback) | 1.1 ms a round of GPU idle inside `cudaGraphLaunch` (*trace*, nsys-inflated) | +0.2-2.2% | ~0 | 1 h / 10 min | 1-2 d |
| 3 | Fused layer-boundary kernel in CUDA (hc_post / hc_partial / hc_finish, router chain) | 90 hc boundaries x 15.6 µs + 43 router chains x 21 µs = 2.3 ms of latency-bound kernels (*trace*) | +1-2.6% | +0.5-1% | 1 d / 15 min | 4-6 d |
| 4 | Per-slot KDA chain / DSA attention in one launch (or 4 forked streams) at 4 streams | chain 130 launches x 32 µs on 32 of 48 SMs = 4.3 ms; `_lchunks` 44 x 50 µs = 2.2 ms (*trace*); 0450's kda part measured ~+0.7% | 0 | +1.2-2.1% | 1 d / 20 min | 2-4 d |
| 5 | Batched DFlash2 blocks across slots (F4) | 4.78 head reads a 4-stream round vs 2.59 at 1s (+1.8 ms); per-slot drafter kernels +5.5 ms (*trace*) | 0 | +1.2-2.5% | 1 d (count) / 0 | 3-5 d |
| 6 | Rank-skew hygiene: locked clocks, rank-0 host threads off the engine's cores | skew 1.3-1.5 ms (1s), 2.2-3.0 ms (4s); rank 0 late on 62% of 4s exchanges (*trace*) | 0-1% | 0-1.5% | 0 / 25 min | 0-0.5 d |
| 7 | E2 (0440 dense) only for small shapes, if it wins them cold | small shapes are 5.3 ms a 1s round; W12's small-shape wins were L2-hot | 0-2% | 0-1% | 10 min bench edit / 10 min | 0.5 d |
| 8 | Memory-pipeline rewrite of experts / dense (TMA bulk or deep register prefetch), gated by a Little's-law probe | old kernels 205-217 GB/s vs 235-248 plain read; 0440's ring capped at 158 GB/s with no decode (W12) | 0-7% | 0-5% | 1 d / 15 min | 7-12 d |
| 9 | Host-bound sampling / commit at 4 streams (C++ or a fixed 0450) | 3.6 ms host-bound idle (*trace*); 0450 recovered only +0.5-1.2% (W12), unexplained | 0 | 0-3% | 0 / 15 min nsys of GR1 | 3-6 d |
| 10 | Shared expert on a forked stream beside the router chain | ~20 µs overlappable a MoE layer x 42 (*est.* from the trace) | 0-1.6% | 0-0.7% | 1 d / 15 min | 1 d |
| 11 | KDA chain split over the value columns (all 48 SMs) | chain = 1 CTA an SM (1,024 threads x 61 registers), 16 SMs idle (*trace*) | +0.6% | +0.9% | 0.5 d / 10 min | 2 d |
| 12 | PDL on the Triton glue kernels | 0.73 → 0.42 µs a boundary (W11 probe), ~1,400 glue edges | +0.6-1.1% | +0.5% | 1 d / 15 min | 3 d |

## 1. What the W11 traces say (new attribution)

### 1.1 One stream, no concurrency (`crit.py` → `crit-1s.txt`, `crit-4s.txt`)

| per round (rank 0) | 1 stream (essay, 168 rounds) | 1 stream (code, 91) | 4 streams (178) |
| --- | ---: | ---: | ---: |
| wall / busy / idle ms | 53.50 / 51.71 / 1.79 | 64.67 / 62.92 / 1.75 | 117.86 / 111.99 / 5.87 |
| ≥ 2 kernels in flight, ms | **0.035** | 0.026 | **0.005** |
| CUDA streams with decode kernels | 1 (stream 7) | 1 | 1 |
| kernels (graph / eager) | 1,853 (1,734 / 119) | 1,880 (1,301 / 579) | 2,576 (476 / 2,099) |
| host µs a launch call, eager | 4.9 | 6.1 | **14.8** (31 ms of host time a round) |
| kernels < 3 µs / 3-10 / 10-50 / 50-200 / ≥ 200 µs (ms) | 1.13 / 3.07 / 7.05 / 15.25 / 25.25 | 1.20 / 3.09 / 7.86 / 13.56 / 37.24 | 1.94 / 3.12 / 14.81 / 15.07 / 77.04 |
| device gaps < 1.5 µs (count, ms) | 1,596, 0.21 | 1,683, 0.31 | 2,307, 0.64 |

The side streams that exist (0460's prefetch fork, the prefill overlap) do not run during these decode rounds. Every
kernel in a round waits for the previous one. The kernel-to-kernel gaps in graphs are 0.13 µs on average: launch
gaps are not where the time goes. What each boundary costs is inside the kernels: ramp, drain and the first
dependent load.

### 1.2 Where the 1-stream idle is (`crit.py` idle attribution, `between.py`)

Device idle inside the round windows, attributed by the launch of the kernel that ended the idle (rank 0, ms a round):

| cause | essay | code | 4 x prose (queued) |
| --- | ---: | ---: | ---: |
| **verify forward: waiting for `cudaGraphLaunch` to return** (one gap a round, > 20 µs) | **1.20** | 0.97 | 1.12 |
| verify forward: short dependency gaps | 0.18 | 0.27 | 0.17 |
| drafting: host-bound (MTP / DFlash2 decisions, their graph launches) | 0.17 | 0.20 | 0.19 |
| round other: host-bound (accept / commit launches; queued: + the `W7:plan` NCCL exchanges) | 0.14 | 0.19 | 0.27 |
| rest | 0.11 | 0.12 | 0.10 |
| **total** (`crit.py`; `idle_check.py` over all 1,004 rounds: 1.79 rank 0 / 1.86 rank 1) | 1.80 | 1.75 | 1.85 |

- **Graph launch cost by graph size.** 1,544 launches of < 100 nodes: 54 µs a call. 535 of 100-199 nodes (MTP
  steps): 384 µs, first kernel 508 µs after the call starts. 963 of 1,600-1,699 nodes (the verify): **1,127 µs, first
  kernel 1,105 µs after the call starts**. That is ~0.68 µs a node, and the GPU is idle for all of it.
- This is W11's "1.1 ms of idle inside the verify forward", now attributed.
- It is **inflated by the capture.** `--cuda-graph-trace=node` instruments every node. W11 measured the captured round
  +1.0-1.3 ms longer than uncaptured at one stream (1,700+ graph kernels a round) and only +0.3% at four streams
  (80-97% eager). That difference is about the size of this gap, so the uncaptured exposure could be as small as
  0.1-0.3 ms. The probe in §6 step 1a measures it directly.
- **Queued requests (segment 3, where one slot fit under nsys and four requests queued).** Every other round then runs
  `W7:plan` between rounds: four NCCL control all-gathers, ~0.1 ms each plus the syncs, 0.87 ms a round. It is
  outside W11's round windows, so W11's "host gap 0.00" did not see it. In production it happens only while requests
  queue.

### 1.3 Four streams: the host is on the critical path at the round's end

Idle at four streams is 5.9 ms a round. By the launch that ended it:

- 2.29 ms: `sample_multi`, host-bound (per-slot top-k, the readback, the numpy draw, then the next launches);
- 1.29 ms: round other, host-bound (per-slot accept / commit / replay launches);
- 1.0 ms: verify forward, short dependency gaps. Eager kernels come back to back, 180 gaps of 1.5-5 µs;
- 0.35 ms: verify, host-bound;
- 0.34 ms: drafting, host-bound.

Inside the verify forward the host is ahead of the GPU (56 ms of host time for 99 ms of device time), so Python's
per-layer loop is not on the critical path, even at 14.8 µs a launch. The host becomes critical only at the serial
decision points after the verify.

### 1.4 The two ranks (`skew.py` → `skew-*.txt`, `crit.py` skew, `jitter.py`)

The ranks run the same exchanges in lockstep (101,674 / 30,974 gathers, equal counts). Block durations are measured
on each rank's own clock.

| | 1 stream | 4 streams |
| --- | ---: | ---: |
| gather µs, rank 0 / rank 1: mean (median) | 23.4 (14.9) / 25.8 (17.1) | 32.5 (17.6) / 39.4 (22.9) |
| **transport** = the shorter of the pair: mean (median) | **10.8 (9.5)** | 13.7 (11.8) |
| skew = \|rank 0 - rank 1\|: mean, median, p90 µs | 27.6, 15.3, 50.4 | 44.5, 20.9, 96.0 |
| waiting beyond transport, ms a round, rank 0 / rank 1 | **1.26 / 1.49** | **2.16 / 2.96** |
| rank 1 arrives later (clock-aligned on gather ends, rolling offset) | 43% | 38% (rank 0 late 62%) |
| sd of the block-time difference: experts / dense / KDA / hc / router, µs | 46 / 22 / 8 / 7 / 4 | 87 / 27 / 22 / 9 / 6 |
| \|diff\| / block: MoE blocks, KDA blocks, dense-only blocks | 3.2%, 2.4%, 4.1% | 3.2%, 3.0%, 5.0% |

- **Identical work is not identical in time.** The R = 1 MTP expert calls (exactly 8 experts): p5-p95 is 10% of the
  median, p99 +45%. The head `_qmm` (178 MB): 6-9%, p99 +19%. KDA projection: 11%. `jitter.txt` has the rest.
- Independent jitter of ± a few percent on each of 100 barrier-separated blocks costs E\|a - b\| / 2 a barrier, i.e.
  ~1.4 ms a round. That is what the table shows.
- **There is a small systematic part.** Rank 0's head call is 2.5% slower (836 vs 815 µs), and its MoE blocks are
  +5 µs a block. At four streams rank 0 is late on 62% of exchanges.
  - Rank 0 hosts the HTTP server and the per-slot streaming.
  - On unified memory its host activity shares the LPDDR5x and the CPU cores with the engine's launch thread. That
    is a hypothesis, tested in §6 step 5.
- So "is either GPU idle while the other works?" Yes: each GPU idles ~1.3-1.5 ms a round (one stream) and 2.2-3.0 ms
  (four streams) inside gathers. Apart from that, both run the same work.
- **An asymmetric split cannot fix jitter.** It could fix only the systematic ~0.2-0.9 ms, at the cost of new shapes,
  new kernels and new bits (§5).

### 1.5 The latency-bound kernels (`smallk.py` → `smallk-*.txt`)

At one stream, the non-streaming kernels (everything except `grouped_kernel` / `_qmm` / `gather_kernel`) are
1,367 launches and 7.9 ms a round. The largest (median µs x calls a round):

| kernel | CTAs x threads | µs x calls | ms a round | what bounds it |
| --- | --- | --- | ---: | --- |
| `chain_kernel` (KDA recurrence) | 32 x 1,024 (61 regs: 1 CTA an SM) | 32.2 x 34 | 1.12 | 32 of 48 SMs, a sequential recurrence |
| `_expand2` (latent expand) | 128 x 128 | 71 x 12 | 0.86 | - |
| `_router_part` | 72 x 128 | 13.2 x 43 | 0.59 | a small GEMV + latency |
| `replay_layers_kernel` | 1,088 x 1,024 | 618 x 0.7 | 0.42 | 71 MB read + written on rejected windows |
| `_absorb2` | 256 x 128 | 31.8 x 12 | 0.38 | - |
| `_hc_partial` (4 grid sizes) | 32-96 x 128 | 8.0 x 90 | 0.72 | latency: a few dependent global round trips |
| `_hc_finish` | 2-5 x 256 | 5.3 x 90 | 0.46 | 2-5 CTAs: pure latency |
| `_hc_post` | 8-16 x 128 | 1.7 x 90 | 0.15 | |
| `_topk` / `_group` / `_router_sum` | 1-4 CTAs | 4.0 / 2.9 / 1.3 x 43 | 0.36 | 1-4 CTAs: pure latency |
| `rot_in` + gate/up + down epilogues | 144-2,880 x 32 | 4-10 x 43 each | ~0.9 | three launches a MoE layer (0440 folds them) |

Every layer boundary runs hc_post → gather → hc_partial → hc_finish, then norm → router_part → topk → group →
router_sum → rot_in at MoE layers. That is ~15.6 µs of hc and ~21 µs of router chain, on 1-96 CTAs, bound by
dependent-load latency. It is the megakernel argument, measured: what a fused boundary saves is these chains, not
launch gaps.

### 1.6 The per-slot cost at four streams, by kernel (`perslot.py`, `head4s.py`)

4-stream minus 1-stream, ms a round (per extra slot = / 3):

| kernel | 1s | 4s | Δ | what it is |
| --- | ---: | ---: | ---: | --- |
| `grouped_kernel` (verify) | 23.6 | 68.2 | +44.6 | expert union U 20 → 60: rows, not fixed cost |
| `chain_kernel` | 1.11 (34 calls) | 4.28 (130) | +3.17 | per-slot KDA chains, 32 CTAs each |
| `_qmm` in drafting / other | 2.73 | 6.90 | +4.17 | per-slot DFlash2 head / layers; **head 2.59 → 4.78 calls a round** |
| `_lchunks` (DSA attention) | 0.55 (11) | 2.35 (46) | +1.80 | per-slot attention, 24-60 CTAs a call |
| drafter experts (`grouped` / `grouped_loop` other) | 0 | 2.02 | +2.02 | per-slot DFlash2 / MTP expert reads |
| `gather_kernel` | 2.25 | 4.08 | +1.83 | larger exchanges + per-slot drafter exchanges |
| `replay_layers_kernel` | 0.37 | 1.23 | +0.85 | per-slot replays on rejected windows |
| everything else | | | ~+3 | |

The "6.6 ms a slot fixed cost" (W9) is, by kernel: per-slot drafting ~2 ms, KDA chain + replay ~1.3 ms, attention
~0.7 ms, idle (mostly host-bound) ~1.4 ms and exchanges ~0.6 ms. The expert-union growth is separate and much larger (+14.9 ms a
slot).

## 2. Why the offline estimates failed

| attempt | estimate | measured | cause, with the evidence |
| --- | --- | --- | --- |
| 0130 decode kernels | faster verify | 1-row verify 31.8 → 32.7 ms; tf code greedy -3% / -12% (PDL) | Never measured per kernel. PDL prefetches issued while the previous kernel was bandwidth-bound compete for DRAM (DECODE-KERNELS §1.3; inference, no trace) |
| 0270 fast2 experts | isolated kernel -2 ms a layer | -2% | Contention with the prefill overlap stream: fast2 holds 1 CTA an SM at 224 registers and competes worse beside overlapped kernels (RESULTS W5). Prefill only: decode has no second stream (§1.1) |
| 0280 batch buckets | +10% | -5% | The premise "eager rounds cost +10-25 ms" was false: at 4 slots the host enqueues faster than the GPU runs (confirmed here: 56 ms of host time for 99 ms of device time), and padded rows cost ~1 ms each |
| 0330 expert TC | faster experts | -4% | A 544-thread design exceeds the per-sub-partition register file (16,384): cannot launch. The warp-specialized cfg 3 lost |
| 0440 E1 (experts) | +2.3 ms, 215-225 GB/s | **127-163 GB/s over every config (best 160-163) vs old 201-218** (W12 kbench) | **Probe 1 (no decode) 156-160 GB/s and probe 2 (no mma) 140-159: the data movement itself caps it.** The design assumed ~1 µs loaded latency, so ~5 KB in flight an SM would do. By Little's law the ring's ~18 KB an SM at 158 GB/s implies ~5.5 µs effective latency. W11's plain-read probe reaches 235 with 16 warps an SM x 4 independent 16-byte loads (~32 KB an SM, grid 1 x 512 = 25% occupancy). Occupancy was never the limit; bytes in flight per warp before first use is |
| 0440 E2 (dense) | -2.1 ms (+4%) | **-3.9%** 1s (prose -5.9%), -3.1% 4s (W12) | **The bench's small shapes were L2-resident**: 3 copies < 24 MB L2, with roofs of 311 (4096 x 4096), 728-852 (8192 x 512 / 1536) and 1,041 GB/s (2048 x 4096), where DRAM gives 235. They showed 1.2-2.1x. The cold large shapes in the same bench were 0.82-0.91x: head 0.82, KDA projection 0.87, MLP gate/up 0.91, DSA o 0.90 (MLP down ~1.0). Weighting W11's in-situ time by the cold rows gives +0.8-1.6 ms a round (head +0.49, KDA projection +0.74, MLP / DSA o +0.25, `_reduce` removed -0.29, small shapes unknown cold, ±0.4), i.e. **-1.5 to -3.0% predicted vs -3.9% measured**. The bench predicted the sign and most of the size; the gate read the wrong rows |
| 0450 GPU round | 1s +1-2%, 4s +3-6% | 1s -2.3% (tf code greedy **-18%**, hashmap -8%, prose ~0), 4s +0.5% (resident) / +1.2% (+kda) | The 1s estimate targeted 0.7 ms of idle outside the verify. The trace shows 1.2 of the 1.8 ms idle is the verify graph launch (§1.2), which resident rounds still issue after the host's peek. At 4s the 3.6 ms host-bound pool is real, but most of it did not come back. No trace of a resident load exists, so the cause is unmeasured. The likely one for short high-acceptance requests: each (slots, rows, parity) key of the batcher path captures before it replays (`CAPTURE_AFTER=3`), and a capture costs a full extra forward. A 64-token request has ~12 rounds and the lone engine's pre-captured graphs are no longer used. Lookup-match drains (GPU-ROUND §2.2) are the other candidate. The W12 stats do not break resident rounds down into capture / replay / drain |
| 0460 L2 prefetch | +2.4% 1s | +0.8% 1s, +2.4% lone slots 1-3, -0.1% 4s (W12) | **The ceiling was the L2-hot speed of the consumers, not the DRAM time of the bytes.** Old `_qmm` L2-hot (W12 bench) vs cold in situ (W11): 2048 x 4096 221 vs 166 GB/s (1.33x). On the ~5 ms of small shapes a round, perfect prefetch buys ~0.5-1 ms; the large matrices do not fit L2. Measured 0.4 ms. The estimate's eta = 0.4 of the DRAM time over 135 sites assumed hits are nearly free for latency-bound consumers, and they are not |

**Common causes, to design against:**

1. **Bench state ≠ server state.**
   - L2 warmth: the bench's small matrices stayed in L2.
   - Clocks: the W11 probe ran at 2,418 MHz, production is capped at 2,250.
   - Isolation vs the prefill overlap stream.
   - Eager µs vs graph µs (DECODE-KERNELS §1.2).
2. **A latency model GB10 does not follow.** Designs sized with ~1 µs loaded latency undershoot the bytes in flight by
   ~5x. Measure the curve (§6 step 1b) before sizing any pipeline.
3. **Estimates from a trace whose own overhead sits in the attributed pool.** Node-level graph tracing adds ~1 ms a
   round of graph-launch time at one stream.
4. **End-to-end A/B without an in-situ capture of the candidate.**
   - W12 has no nsys of E2 / 0450 / 0460.
   - 0450's regression and E2's per-shape effect had to be reconstructed here.
   - Every candidate load gets one capture, even a short one.

## 3. The five questions

### 3.1 More parallel / non-blocking execution

- **The critical path at kernel granularity** is the verify forward, 100 exchange-separated blocks a round (template
  in `crit-1s.txt`). A KDA block runs 14 kernels, ~265 µs: KDA projection `_qmm` 147 µs, chain 30 µs, KDA o 57 µs,
  hc ~16 µs, gather ~10 µs. A MoE block runs 20 kernels, 730-940 µs: routed experts 640-840 µs, shared expert ~48,
  router ~24, hc ~15, gather. Then come sampling, drafting (4.1-4.3 ms of device time a round, host 4.8 ms) and the
  verify graph launch (§1.2).
- **What could run concurrently:**
  - *Attention vs MoE partials; KDA vs experts; router vs all-gather.* No. The model is sequential per layer: the MoE
    input is hc_post(attention partial sum), which is nonlinear (COMM-ANALYSIS 2(a)), and the next layer needs the
    MoE output.
  - *Shared expert vs router chain.* Yes. They are independent until the combine. ~20 µs a MoE layer is
    overlappable: -0.85 ms a round at most. The realistic figure is lower, because W7 measured a 3x overlap tax on
    kernels beside an all-gather. Item 10.
  - *Splitting experts across two streams.* No. `grouped_kernel` holds 3 CTAs an SM at 32 KB of shared memory each
    (96 of 99 KB), so a second stream's CTAs get SMs only in the tail. The plain-read probe shows one stream of
    independent loads at 16 warps an SM already reaches 235 GB/s. The lever is loads in flight inside one kernel
    (item 8), not a second stream.
  - *4 streams: the four slots' KDA chains, attention, replays and drafter passes are independent.* They run as 4-5
    serial launches of 1/2-2/3-wave grids. Items 4 and 5.
- **Overlapping the ranks' skew.** Nothing that is exact can start before the gathered sum. The only fillers are weight
  prefetch (0460, measured +0.8%) and the next layer's `f` / `a` sites, which 0460 already uses. Reducing the jitter
  (item 6) is the lever.
- **"Speculative drafting ahead"** (start round k+1's drafts before round k's verify is sampled).
  - Both drafters need what only the end of the verify gives: MTP needs the verified final hidden rows of the
    accepted tokens; DFlash2 needs the target taps through layer 42.
  - A pre-draft would have to run on the drafter's own states, EAGLE-style, on the "all kept" branch. Prose MTP keeps
    both drafts 43% of the time (W11 p_2). With the bonus token matching the drafter's next draft (~0.6), about 26%
    of rounds could skip their 4.1-4.3 ms drafting phase.
  - The pre-draft must run beside the verify, which is bandwidth-bound (96.6% busy). So its ~0.5 GB of MTP reads cost
    about their own time. Only the drafting's latency-bound half (~2 ms) can hide.
  - Upper bound ~0.26 x 2 ms ≈ 0.5 ms (1%), minus the always-paid extra drafting and the lower acceptance of
    self-fed drafts. It is exact, but it is net ~0 to negative, for a large change. **Not recommended.**

### 3.2 Both nodes, and the CPU

- **Neither GPU is ever idle while the other computes, except inside gathers.** That is 1.3-1.5 ms a round at one
  stream and 2.2-3.0 ms at four, from jitter (§1.4). An asymmetric column split could address only the systematic
  0.2-0.9 ms; §5 explains why it is not worth doing.
- **CPU on the critical path, counted:**
  - One stream: 1,853 kernels a round, 94% replayed in ~3 graph launches and 119 eager (4.9 µs a call).
    - Host-bound device idle: 0.3-0.4 ms a round (drafting decisions, accept / commit).
    - Plus the verify graph launch, 0.1-1.2 ms (§1.2). That is driver time in `cudaGraphLaunch`, not Python.
    - The host spends 85% of the round spinning in `sample_multi`'s readback.
  - Four streams: 2,576 kernels, 2,099 eager at 14.8 µs of host time each (Python + torch dispatch + Triton's launcher:
    31 ms a round), hidden behind 99 ms of verify.
    - Host-bound idle 3.9 ms a round: 2.3 sampling, 1.3 accept / commit, 0.3 drafting.
- **Moving the host loop to C++ matters only at the post-verify decision points.** At four streams that is ≤ 3.6 ms
  (≤ 3%); at one stream it is ≤ 0.4 ms. The per-layer forward loop is not on the critical path in any mode. Before
  writing C++, find out why 0450 (which moved exactly these points onto the device) got only +0.5-1.2% (item 9).

### 3.3 Lower-level rewrites: where they can beat Triton, and where they cannot

| kernel family | ms a round (1s / 4s) | above its floor | lower-level lever | why Triton cannot get there | expected | risk |
| --- | --- | --- | --- | --- | --- | --- |
| routed experts (`grouped_kernel`, CUDA already) | 27.8 / 75.5 | 4.5 / 7 ms | more independent loads in flight before first use: (a) probe-style 4-8 x `ld.global.nc.v4` into registers ahead of `decode_tile`, or (b) `cp.async.bulk` (TMA 1D) 4-16 KB chunks into a shared ring with mbarriers, one issuing lane a warp | Triton cannot express the EXL3 trellis decode (lane-shuffle codebook) or gathered 1D bulk copies with mbarrier rings on sm_121 | 205-214 → 225 GB/s: -2 to -2.7 ms 1s, -4 to -5 ms 4s | **high**: E1 (cp.async ring) and 0130 (1-deep register prefetch) both lost. Gate on §6 1b first |
| dense large `_qmm` (head, KDA projection, MLP, DSA o) | ~10.7 / ~13 | ~1.5 ms | same mechanism, plus a stream-K style tail (the KDA projection is 788 programs = 4.1 waves: the 0.1-wave tail runs at low parallelism) | Triton's pipeliner gives `num_stages` rings of `cp.async`; no bulk copies or cross-program tail balancing | 195-216 → 225: -0.8 to -1.2 ms | medium: 0440 E2 was 9-18% slower cold |
| layer boundary (hc x 3, router chain x 5, rot_in) | ~2.3 + 0.9 / ~3.4 | ~2 ms | one CUDA kernel a boundary: last-block reductions in a fixed order, no intermediate global round trips; later, the RoCE gather's flag wait inside the same persistent kernel | 0190's Triton `hc_fused` was bitwise but 7-8x slower (`num_stages=1`, multi-phase grid reduction) | -1.0 to -1.4 ms 1s | low-medium. Item 3 |
| KDA chain | 1.1 / 4.3 | ~0.5 / ~2.5 ms | split the value columns over 2 CTAs a head (48 SMs busy); one launch over a slot table at 4s | not a Triton kernel (`kda.cu`) | -0.3 / -1.1 to -2.5 ms | low. Items 4, 11 |
| whole-step megakernel (Hazy / MPK) | the ~17 ms above the floor | | persistent layer walker with in-kernel exchanges | | +5-10% for 25+ days | nothing supports MoE + KDA + DSA + mHC + EXL3. Item 3 is its first slice |

- **Why "latency at low occupancy" is the wrong frame.** The plain-read probe reaches 235 GB/s at 25% occupancy (1 x
  512 threads an SM) because each thread has 4 independent 16-byte loads outstanding. Our kernels do not starve for
  warps; they starve for independent bytes in flight per warp. The E1 ring (warp-private, 3 stages, ~18 KB an SM) is
  the counterexample that settles it: no ALU work, and still 158 GB/s.
- **TLB misses do not explain it either (§3.5).**
- **Where a host C++ loop matters:** §3.2. Nowhere inside the forward.

### 3.4 Throughput (4 streams)

Measured split of the per-slot cost: §1.6. Items, in value order:

- **Batched graphs only for lone rounds (item 1):** +1.9-2.6% measured.
- **Batched DFlash2 blocks (item 5):** 4.78 head reads a round vs 2.59 at 1s. One head pass and one drafter-layer pass
  for all f-slots in a round saves ~0.8 ms a head read avoided plus the per-slot layer reads: -1.5 to -3 ms (+1.2-2.5%).
  Exact with row-local layers and the `_dconv` block-boundary mask (GPU-ROUND §5).
- **Per-slot chain / attention in one launch (item 4):**
  - KDA chains: 4 x 32-CTA launches become 128 CTAs over 48 SMs, i.e. 3 waves instead of 4: -1.1 ms.
  - Attention: 44 `_lchunks` of 24-60 CTAs become ~11 launches at 1-1.3 waves: -1.1 ms.
  - W12's kda fold (chain_slots) accounts for ~+0.7%, consistent with the chain half.
  - A 1-day prototype that needs no kernel change: fork the per-slot launches onto 4 streams in the eager path. At 4s,
    60-97% of rounds are eager.
- **Host-bound sampling / commit (item 9):** ≤ 3.6 ms.
- **A single attention launch across slots** is item 4's attention half.

### 3.5 Anything else (≥ 5% was the bar: nothing qualifies alone)

- **TLB / page size.**
  - Weights are ordinary `cudaMalloc` memory (torch's allocator), not managed or host-registered.
  - W11's probe read gathered random experts over 2.1-13.5 GB footprints at 235-237 GB/s, the same as contiguous.
    That is 1,000-6,750 distinct 2 MB pages a round and far past any TLB reach, and it still ran at the ceiling.
  - W12's bench ran the old kernels over freshly allocated random weights at the in-situ speed.
  - So translation is not what separates the kernels (205-214) from the probe (235): **not a lever (≤ 1%)**.
  - Optional confirmation, 5 minutes: run the probe's read over the engine's own weight tensors in-process.
- **Clocks.**
  - Decode runs at 2,223-2,242 MHz and ~38 W (W11 `clocks-cap*.log`), far from any power limit, and the memory clock
    is fixed.
  - The ~8 ms of latency-bound kernels would scale partly with SM clock: 2,418 MHz is +7.5%, so about -0.5 ms (1%).
  - The 2,250 cap exists because of the prefill power-off. Do not raise it before that is understood.
  - A decode-only cap is possible but is not on this list.
  - The **clock lock** (min = max, item 6) is different: it targets jitter, not speed.
- **Unified-memory interference from the host.** A hypothesis for rank 0's systematic lateness (§1.4). It is tested
  in item 6, not assumed.
- **Queued-request plan path** (§1.2): 0.87 ms a round, only while requests queue. It rides NCCL. Moving it onto the
  RoCE rider as 0370 does for the steady state is worth doing only if production queues often.

## 4. Ranked ideas, with prototypes

Each prototype is ≤ 1 day to build and ≤ 30 minutes of GPU, and runs **in the server** (or with the server's exact
conditions: cold L2, 2,250 MHz, beside a streaming predecessor). Each has a gate that decides the full build.

**1. Lone-only batched graphs + L2PF.**
- Evidence:
  - W11 G0 +2.6% and W12 G0 +1.9% at 4s. W12 lone requests lose 2.7% under G0 (`slots-*.log`: slots 1-3
    about -3%, since they use the batcher path's graphs; slot 0 about -1.5%).
  - W12 L2PF: +0.8% 1s geo-mean, +2.4% lone slots, -0.1% 4s.
  - Control spread: geo-means ±0.1% (1s) and ±0.5% (4s); per cell ≤ 1.4%.
- Prototype: `GLM53_TF_BATCH_GRAPHS=lone` (graphs only for rounds with one active slot; ~20 lines in `batch.py`).
- Test: control vs lone+L2PF, full W12 `ab.sh` (exact, batchexact, transcripts, glmbench x3, conc x6, slots).
- Gate: 4s ≥ +1.5%, lone slots and 1s ≥ 0.
- Gain: +2-2.6% 4s, +0.8% 1s.

**2. Verify graph launch.**
- Probe (1 h to write, 10 min GPU; §6 1a): time `CUDAGraph.replay()` host-side and the first node's start (a
  globaltimer-stamp kernel as node 0) for 100 / 400 / 1,650-node graphs of production-like kernels. Run uncaptured,
  under `--cuda-graph-trace=graph`, and under `=node`.
- Gate: uncaptured first-node delay ≥ 0.4 ms at 1,650 nodes.
- Fix (1-2 d), either of:
  - capture the lone verify as 2-4 chained graphs (the first ~100 nodes launch in ~60 µs and run while the rest
    launch);
  - launch the verify before the draft readback from device-staged inputs (0450's `_stage_kernel`).
- Also re-check `cudaGraphUpload` at capture.
- Gain: 0.1-1.1 ms a 1s round (+0.2-2.2%).

**3. Fused layer boundary (CUDA).**
- Prototype (1 d): `hc_fused.cu` = hc_partial + hc_finish (+ hc_post of the next boundary), same per-element op order,
  fixed-order last-block reduction.
- Bench: inside a CUDA graph after a 50 MB streaming kernel (cold L2, a realistic predecessor tail), at 2,250 MHz
  locked, against the three Triton kernels. Bits equal on 1-16 rows.
- Gate: ≤ 50% of the three kernels' 15.6 µs. Then the router chain (router_part → topk → group → router_sum → rot_in)
  the same way.
- In-server: one nsys capture of a 1s cell with the knob. Per-boundary time must drop by ≥ 6 µs in situ.
- Gain: -1.0 to -1.4 ms 1s (+1-2.6%), -1 to -1.2 ms 4s.

**4. Per-slot KDA chain / attention at 4s.**
- Prototype (1 d): in `batch._kda` / `batch._dsa` eager rounds, fork slots 1-3 onto 3 side streams, join before the
  output projection. Same kernels, same inputs, so same bits.
- Test: 4s conc x6 + one nsys (4s) against control.
- Gate: chain + `_lchunks` family time ≥ 30% lower in situ, and aggregate ≥ +1%. If it passes, the durable form is one
  launch over a slot table (0450's `chain_slots` exists; attention needs a per-row sequence table).
- Gain: +1.2-2.1% 4s.

**5. Batched DFlash2 blocks (F4).**
- Prototype (1 d, no GPU): from the W11 4s trace and W12 conc stats, the distribution of f-slots a round (how often
  ≥ 2 slots draft DFlash2 in the same round), and the per-pass kernel time. This sizes the win before the 3-5 d build.
- Gate: ≥ 1.5 ms a round addressable.
- Gain: +1.2-2.5% 4s.

**6. Rank-skew hygiene.**
- Config only.
  - Load A: `nvidia-smi -lgc 2250,2250` on both nodes. This locks the frequency (today 300-2,250), so ranks that wait
    at 100 barriers a round do not re-ramp.
  - Load B: rank 0's HTTP / streaming threads pinned away from the engine thread and the RoCE proxy (0370's pin
    extended).
- Measure with `GLM53_TF_ROCE_TRACE` (0460, in image b5): per-op skew with no nsys.
- Gate: skew wait (mean - transport) -25%, or 1s / 4s ≥ +0.7%.
- Gain: 0-1% 1s, 0-1.5% 4s.

**7. E2 for small shapes only.**
- Bench edit (10 min): `COPIES` per shape so the rotation is ≥ 96 MB (4x L2), dense section only.
- Gate: new ≥ 1.10x old, cold, on shapes < 8 MB (KDA o, shared gate/up / down, DSA q_b / kv, index). Then add
  `GLM53_TF_DEC_QMM_MAX_MB` (0.5 d) and A/B.
- Gain: 0 to -1.1 ms 1s (+0-2%).

**8. Memory-pipeline rewrite, gated.**
- Prototype (1 d, §6 1b): `littles.cu`. Throughput vs bytes in flight an SM (4-64 KB) on the gathered expert layout,
  cold, 2,250 MHz, for four mechanisms:
  - `ld.global.nc.v4` x 1-8 independent loads into registers;
  - 0440's `cp.async` 16-byte warp ring (2-8 stages);
  - `cp.async.bulk` 1D, 4 / 8 / 16 KB, 2-4 stages, 1-2 CTAs an SM;
  - a pointer-chase for idle and loaded latency.
  Each alone and beside a 32-CTA latency-bound kernel.
- Gate: a mechanism that keeps a warp's 4 x 16-row trellis tiles fed at ≥ 230 GB/s with ≤ 112 registers. Only then
  build an EXL3 kernel around it with exl3.cu's per-warp K chains (same bits), then the dense twin.
- Gain: 0-7% 1s, 0-5% 4s. Two earlier designs failed this, so the probability is low.

**9. Explain 0450 before any C++ host loop.** 15 minutes of GPU: one nsys window on a GR1 load (tf code greedy x3 +
one 4s rep), with `Resident.drains` / captures dumped at the end. Gate: attribute ≥ 80% of the -18% (capture rounds,
drains, peeks). The host-side C++ port of `sample_multi` / accept / commit (3-6 d, ≤ 3%) is decided by that trace.

**10. Shared expert forked beside the router chain.**
- Prototype (1 d): event fork / join in `moe_block`, captured into the graph as a parallel branch. The shared expert
  output slot is disjoint, so combine order and bits are unchanged.
- Gate: in-situ MoE block -10 µs (nsys).
- Gain: 0-1.6% 1s.

**11. KDA chain over the value columns.**
- Prototype (0.5 d): `chain_kernel` with a dv split, 64 CTAs of 512 threads. Each state column's update is
  independent, so the op order per element is the same.
- Microbench in a graph, cold.
- Gate: ≤ 75% of 32 µs.
- Gain: +0.6% 1s, +0.9% 4s. It overlaps item 4 at 4s.

**12. PDL on Triton glue kernels.**
- Prototype (1 d): Triton's `launch_pdl` + `gdc_wait` on hc / norm / router kernels only, with 0440's late-writer
  race test.
- Gate: in-situ boundary -0.25 µs a kernel.
- Gain: +0.6-1.1% 1s.

## 5. Checked and not recommended

| idea | why not (numbers) |
| --- | --- |
| Speculative drafting ahead | Both drafters need the verify's final rows or taps. A self-fed pre-draft on the "all kept" branch skips drafting in ~26% of prose rounds. It runs beside a bandwidth-bound verify, so only ~2 ms of latency-bound drafting can hide: ≤ 1% before its costs. Net ~0 or negative |
| Experts split across two streams | The grouped kernel leaves 3 KB of 99 KB shared memory an SM, so the second stream waits for the tail. One stream of independent loads already reaches 235 (probe) |
| KDA ∥ experts, attention ∥ MoE, router ∥ all-gather | Data dependencies (sequential blocks, nonlinear hc_post on the gathered sum) |
| Asymmetric TP split between the nodes | Skew is mostly independent jitter (43-62% either way). The systematic part is 0.2-0.9 ms a round. A split changes shapes, kernels and bits (row-parallel partial sums) |
| Micro-batching two slot groups to hide exchanges | Re-reads every weight: +12 ms a 4s round to hide ≤ 4.2 ms (PREFETCH-COMM §3.1) |
| Hugepages / TLB work | Gathered 13.5 GB probe reads = contiguous at 235 GB/s (§3.5) |
| Raising the SM clock cap | ~1% at best; the cap guards the prefill power-off |
| Whole-step megakernel now | 25+ days, no MoE / KDA / DSA / EXL3 support anywhere. Item 3 is the incremental slice with its own gate |
| E1 as built (0440) | 127-163 GB/s (best config 160-163) vs the old 201-218 on every window; the no-decode probe is just as slow |

## 6. Next GPU prototype session (≤ 2 h; production down; W12 harness)

Image `b5` plus three small builds, prepared offline:

- the `BATCH_GRAPHS=lone` knob (item 1);
- `glprobe.py` (item 2);
- `littles.cu` (item 8, standalone, nvcc on the node).

Optionally `hc_fused.cu` (item 3) if it is ready. Everything runs under `timeout`, logs go to `results/W13/`, and the
lease / watchdog routine is the same as W11 / W12.

| step | time | what | decides |
| --- | ---: | --- | --- |
| 0 | 5 min | stop prod, lease, `nvidia-smi` clocks both nodes | |
| 1a | 10 min | `glprobe.py` on the worker node: graph launch host time and first-node delay for 100 / 400 / 1,650-node graphs, uncaptured and under nsys `graph` / `node` tracing; 2- and 4-way split variants | item 2 (≥ 0.4 ms uncaptured → build the split) |
| 1b | 15 min | `littles.cu` on the head node (the other GPU runs 1c at the same time): latency curve and four load mechanisms, cold, 2,250 MHz, alone and beside a latency kernel | item 8 (≥ 230 GB/s with a decode-compatible mechanism → design the kernel) |
| 1c | 10 min | `bench_decode_kernels.py --dense-only` with ≥ 96 MB rotation per shape | item 7 (≥ 1.10x cold on < 8 MB shapes → the size knob) |
| 1d | 10 min | (if built) `hc_fused` microbench: bits and µs vs the three Triton kernels in a graph behind a 50 MB stream | item 3 |
| 2 | 20 min | load **C** (control, prod env): W12 `ab.sh`, trimmed to exact, batchexact, glmbench tf / kit x3, conc 4s x6, slots | baseline |
| 3 | 20 min | load **L**: `BATCH_GRAPHS=lone` + `L2PF=1`: same set | item 1 |
| 4 | 20 min | load **K**: control + `-lgc 2250,2250` both nodes + rank-0 thread pinning + `GLM53_TF_ROCE_TRACE=4096`: glmbench tf / kit x3, conc x3, `roce.trace_summary` (transport / skew p50 / p90); compare with a 5-min `ROCE_TRACE` on C | item 6 |
| 5 | 15 min | load **GR1** + nsys, one window: tf code greedy x3, one 4s rep; dump resident counters | item 9 (why -18%) |
| 6 | 5 min | restore prod from `config/prod.env`, canary, https `/v1/models`, watchdog re-armed, lease deleted | |

Total ~2 h 10 min with boots; if time runs short, step 5 goes to the next window. Steps 1a-1d need no server and can
run on the two nodes in parallel.

**Prepared (offline, 2026-09-29):** `results/THEORY2-SESSION/` (`run.sh all` + README): image b6 = b5 + patches 0510
(`BATCH_GRAPHS=lone`, the verify graph in pieces `VERIFY_SPLIT`, the in-server launch probe `GRAPH_PROBE`, 0450's
drain report), 0520 (the fused hc boundary, docs/HC-FUSED.md) and 0530 (`CPU_PIN=http`, the RoCE trace dump for step
4's skew numbers); `probes/glprobe.py`, `probes/littles.cu`, `tests/cuda/bench_decode_cold.py`. The script adds a
trace control load (CT) and the split prototype (VS) to the loads above. After W12 (5c4da18) the control is the new
prod (L2PF=1 at 8 MiB, BATCH_CAPTURE_AFTER=8), so load L is `BATCH_GRAPHS=lone` alone and item 1's gate is re-based:
4s >= +0.7% over that control, lone slots >= -0.5%, 1s >= 0.

**Adopt / build rules:**

- Adopt L if its gate passes, and K if it passes (config only).
- Build items 2, 3, 7 and 8 only on their probe gates.
- Items 4 and 5 go to the next build batch regardless: their prototypes are in-server A/Bs in the following window.

## 7. Method rules (from §2)

1. **Microbench = server state:**
   - cold weights (rotation ≥ 4x L2, i.e. ≥ 96 MB a shape; assert the roof is ≤ 240 GB/s);
   - clocks locked at the production cap (2,250);
   - graph-launched, behind a realistic predecessor;
   - and, for prefill kernels, beside the overlap stream.
   Gates read the cold rows only.
2. **Size pipelines from a measured latency curve on GB10**, not from 1 µs.
3. **Every candidate load gets one short nsys capture** (graph-level tracing, not node-level, when timing launches).
   Per-family in-situ times are compared with the control's before any end-to-end conclusion.
4. **Estimate against attributed time only.** An idle or wait pool is spent once: say which gap an item removes
   (§1.2 / §1.3 give the gaps).
5. **Record per-round kinds** (capture / replay / eager / resident / drain) in every A/B's stats.

## 8. Files and reproduction

`results/theory2/` (scripts + their outputs):

| script | input | output | what |
| --- | --- | --- | --- |
| `w12_summary.py` | `results/W12/*.log` | `w12_summary.txt` | every W12 load vs the three same-window controls: 1s geo-means by class, 4s aggregate, lone slots, per cell |
| `w12_kbench.py` | `results/W12/kbench.json` | `w12_kbench.txt` | E1 / E2 old vs best new vs roof per window / shape; the roofs > 235 GB/s mark L2-resident rows |
| `crit.py` | W11 sqlite, both ranks | `crit-1s.*`, `crit-4s.*` | per segment: streams, concurrency, gaps, host-bound idle by phase, launch counts, kernel sizes, families, phase host vs device, exchange-block templates, clock-aligned skew |
| `idle_check.py` | sqlite | `idle_check.txt` | device idle inside the round windows on the full device timeline (kernels + memcpy / memset) |
| `between.py` | sqlite | `between.txt` | gaps between rounds, `W7:plan` / NCCL between rounds, vfwd start → first verify kernel, verify graph launch call |
| `skew.py` | both ranks | `skew-*.txt` | transport vs skew, block-time difference by block kind and family, within-rank spread of identical blocks |
| `jitter.py` | both ranks | `jitter.txt` | duration spread of identical-work kernels |
| `smallk.py` | sqlite | `smallk-*.txt` | the non-streaming kernel inventory (grid, block, registers, µs, calls) |
| `perslot.py` | 1s + 4s | `perslot.txt` | per-kernel count and ms, 4s minus 1s, by phase |
| `head4s.py` | 1s + 4s | `head4s.txt` | head `_qmm` calls a round and histogram |

Reproduce (on a workstation, not a prod node):

```
scp <head-ssh>:w11-traces/*.nsys-rep ~/tmp/w11-traces/
nsys export --type sqlite --output cap-r0.sqlite cap-r0.nsys-rep        # likewise cap-r1, cap4-r0, cap4-r1
python3 results/theory2/crit.py cap-r0.sqlite cap-r1.sqlite results/theory2/crit-1s.json > results/theory2/crit-1s.txt
python3 results/theory2/crit.py cap4-r0.sqlite cap4-r1.sqlite results/theory2/crit-4s.json > results/theory2/crit-4s.txt
python3 results/theory2/skew.py cap-r0.sqlite cap-r1.sqlite > results/theory2/skew-1s.txt
python3 results/theory2/w12_summary.py > results/theory2/w12_summary.txt       # etc.
```

The sqlite exports (0.1-0.2 GB each) and the reports stay out of git.
