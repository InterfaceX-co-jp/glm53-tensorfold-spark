# Routed experts at prefill piece sizes: patches/0330 (`tc` kernels) and 0335 (solo pieces)

Written offline (2026-09-28, no GPU, Sparks and production untouched). Both patches are off by default and give the same
bits as production. The kernels compile for sm_121 with nvcc 13.4, and the host-side tests pass. Nothing here has been
timed. The GPU test plan is at the end.

- **0330** `GLM53_TF_FAST_EXPERTS=tc`: a new fast-prefill expert kernel family (`exl3_tc.cu`). It uses fat's arithmetic
  bit for bit, with warp-specialized data movement.
- **0335** `GLM53_TF_SOLO_PIECE=N`: a request alone in the batch prefills in N-token pieces instead of 2,048. It falls
  back to 2,048 at the next piece boundary once another request is admitted. Optional `GLM53_TF_LEAN_LAZY_XU=1`
  allocates the lean set's unused Xu lazily, which saves memory.

## 1. Where the expert time goes

Per rank and MoE layer (288 experts, top 8, hidden 4,096, 1,024 of 2,048 expert width), per chunk of R rows (P = 8R
routed pairs), the kernels must at least:

- **DRAM:** read every weight once (1.81 GB), read Xg (P x 4096 fp16), write and read Xd (P x 1024 fp16), and write Y
  (P x 4096 fp32). The rate is 220 GB/s, which fast2 reaches at 1,024 rows.
- **Tensor cores:** do 2P x 12.6 M FLOP at the ~110 TFLOP/s measured `mma.sync` f16 peak.

Kernel times are W2/W5 `bench_experts.py`, gate/up + down, uniform routing, in ms:

| rows | bytes | DRAM floor | MMA floor | fast2 | fat (prod) | fat / floor | fat TF/s | fat us / token |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 2.05 GB | 9.3 | 1.9 | 9.4 (1.01x) | 11.6 | 1.25x | 17.8 | 11.3 |
| 2048 (prod piece) | 2.28 GB | 10.4 | 3.7 | 10.9 (1.05x) | 13.0 | 1.25x | 31.7 | 6.35 |
| 4096 | 2.75 GB | 12.5 | 7.5 | 16.9 (1.35x) | 16.0 | 1.28x | 51.5 | 3.91 |
| 8192 | 3.69 GB | 16.8 | 15.0 | 31.1 (1.85x) | 28.6 | 1.70x | 57.7 | 3.49 |

What the table says:

- **At <= 2,048 rows the experts are DRAM-bound, not tensor-core-bound.** fast2 is within 1-5% of the floor. So better
  MMA tiling cannot help there, which is why W2's no-decode and no-mma probes changed nothing (within 1%). fat, the
  production kernel, sits 25% above the floor, about 2.6 ms a layer. Any kernel can gain at most ~2.1-2.6 ms a layer
  over fat at 2,048 rows: 43 layers (42 + MTP) x ~2.3 ms is ~100 ms of a ~1.6 s piece, about **6%**.
- **At 8,192 rows both floors are ~15-17 ms and fat takes 28.6.** The no-mma probe saved 21-27% there (W2), so MMA is
  exposed: memory traffic and MMA do not overlap well. That is the tensor-core efficiency gap: 58 of ~110 TFLOP/s. A
  kernel that overlapped them fully would take ~18-20 ms, saving 8-10 ms a layer.
- **The expert cost per token falls 1.8x from 2,048 to 8,192 rows** (6.35 to 3.49 us a token and layer with fat), and
  86% of that fall has happened by 4,096. That is 0335's lever. It does not need a new kernel; 0330 makes it larger.

### Why fast2's isolated win did not survive end to end (W5)

Isolated, fast2 beat fat by ~2 ms a layer at 2,048 rows. On the production load it was 2.2% slower (auto, one rotated
input) and 4.6% slower (plain fast2, two rotations). The 2.4-point gap between those two is the extra rotation (rot_in
writes Xg and Xu). The rest needs an in-situ mechanism. Candidates, from the code:

1. **Static-stride scheduling vs ticket (leading hypothesis).** fast2's persistent grid hands CTA b the items b,
   b + grid, ... in advance. fat claims items with an `atomicAdd` ticket. With patches/0084's overlap, a second stream
   runs NCCL all-gather kernels and hc slabs beside the experts. The SMs those kernels share run their fast2 CTAs more
   slowly, and a static kernel ends with its slowest CTA; a ticket kernel moves the work to the free SMs. Nothing in
   the isolated bench shares an SM, so the bench cannot see this.
2. **Co-residency.** fast2 leaves more room on an SM (126 registers x 256 threads, 55 KB of shared memory) than fat does
   (2 CTAs x 256 x 121 registers fill the register file). So fast2 should if anything overlap better, and this does not
   explain the loss.
3. **L1/L2 behaviour.** fast2 loads trellis words with `__ldg` (L1-allocating); fat uses `cp.async.cg` (L2 only). This
   is second order.

`bench_experts.py --contend` (below) tests hypothesis 1 directly. It times every kernel while a side stream runs
DRAM-bound copies and a small GEMM that holds a few SMs. `tc static` (ticket off) vs `tc` (ticket on) is the controlled
A/B. If the static variant loses more under contention, fast2's end-to-end loss is explained. tc keeps the ticket, and
`GLM53_TF_TC_CTAS` can leave SMs free for the overlap stream.

## 2. What sm_121 (GB10) supports

Probed by assembling PTX with ptxas 13.4 for `sm_121` and `sm_121a`. torch's extension build targets plain `sm_121`
(`compute_121`), so only the first column matters for our kernels.

| feature | sm_121 | sm_121a | use here |
| --- | --- | --- | --- |
| `mma.sync` m16n8k16 f16 -> f32 (our instruction), m16n8k32 e4m3 | yes | yes | the only tensor-core MMA for our bits |
| `wgmma` (Hopper warpgroup MMA) | **no** | **no** | not available |
| `tcgen05` (Blackwell datacenter TMEM MMA) | **no** | **no** | not available |
| block-scaled `mma` (`kind::mxf8f6f4`, NVFP4) | no | yes | new bits anyway (PARADIGMS item 10) |
| `ldmatrix` / `stmatrix` | yes | yes | ldmatrix for the member rows (as fat) |
| `cp.async` (+ `cp.async.mbarrier.arrive.noinc`) | yes | yes | member-row gathers, completion on an mbarrier |
| `mbarrier` (init, arrive, expect_tx, try_wait.parity), `fence.mbarrier_init` | yes | yes | the stage ring and the item queue |
| `cp.async.bulk` global -> shared (complete_tx) | yes | yes | trellis words (1 KB contiguous a k tile and block) |
| TMA tensor copies (`cp.async.bulk.tensor`), `.multicast::cluster`, `tile::gather4` | assemble | assemble | not used (row gathers are by index; no tensor map needed) |
| clusters (`barrier.cluster`), `elect.sync`, `griddepcontrol` (PDL) | yes | yes | not used (PDL is a candidate to overlap rot_in1's tail) |
| `setmaxnreg` (warp-specialized register split) | **no** | yes | not available: producer and consumers get the same register cap |

Other limits: 48 SMs, 99 KB of shared memory a CTA (101,376 B), 100 KB an SM, and a 64K-register file.

Consequences:

- Nothing Hopper- or datacenter-Blackwell-class is available. The tensor-core path stays `mma.sync.m16n8k16`, which is
  also what keeps the bits.
- The gains must come from data movement (asynchronous copies, mbarrier pipelines, bigger tiles) and scheduling.
- A dequantized bf16 tile in shared memory (the "dequant once, reuse across row tiles" idea) was tried as 0080's v1: it
  was bound by shared-memory bandwidth. fast2/fat decode straight into the MMA A fragment, and decode costs 3-5%
  (probes). So decode reuse is not the lever.

## 3. patches/0330: the `tc` kernels

`GLM53_TF_FAST_EXPERTS=tc` is a fat-family mode. `tf_knobs.fat_experts=1` selects it, 0 selects fast2, and 2 selects
auto (fast2 inside `GLM53_TF_FAST2_ROWS`). When gate and up have different sign vectors, fat runs instead; the real
checkpoint has them equal.

Files:

- `exl3_tc.cu` and `exl3_tc.cpp`: their own extension, `tensorfold_glm_exl3_tc_v1`, built on first use, so the
  production build is untouched.
- `exl3_mm.py`: the mode and the dispatch, after `rot_in1`.

### Design

- **Producer / consumer warps.** A CTA has one producer warp and 16 consumer warps (544 threads, one CTA an SM).
  - The producer claims items by ticket. It gathers the item's member rows into a 3-stage cp.async ring (XOR-swizzled,
    fat's layout) and copies the item's trellis words with `cp.async.bulk`: one 1 KB copy per (matrix, column block,
    k tile).
  - A stage is complete when its **full** mbarrier sees 32 `cp.async.mbarrier.arrive.noinc` arrivals plus the bulk
    bytes (`expect_tx`).
  - A slot is refilled only after its **empty** mbarrier has one arrival from each consumer warp.
  - There is no `__syncthreads` in the K loop. fat's CTA-wide barrier each stage made every warp wait for the slowest
    one, with nothing loading meanwhile.
- **Consumers run fat's inner loop unchanged.** A warp owns 2 column tiles x 64 members. It decodes its trellis words
  into the m16n8k16 A fragment and ldmatrix's the member rows as B (the same chain of mma over ascending k tiles).
- **Wider items.** An item is (expert, pass of 64 x MH members, NBC adjacent 128-column blocks). The number of items
  in flight, and so the L2 traffic per FLOP, is set by MH and NBC:

  | cfg | gate/up | down | when (`cfg` 0 = by rows) | what it saves over fat |
  | --- | --- | --- | --- | --- |
  | 1 | 64 members x 2 blocks | 64 x 4 blocks | chunks < 4,096 rows | member rows read 1/2 (gate/up) or 1/4 (down) as often; at 2,048 rows an expert has ~57 members (one pass) |
  | 2 | 128 members x 1 block | 128 x 2 blocks | chunks >= 4,096 rows | each weight tile fetched once for 128 members instead of 64 |
  | 3 | 64 x 1 (fat's tile), 8 consumer warps, 2 CTAs an SM | 64 x 2 | A/B only | the pipeline alone, fat's footprint |

- **Epilogue.** It is fat's: fp32 rows through shared memory, then one warp a row with fwht_row and the same formulas.
  It has its own buffer, so the producer loads the next item's first stages while the consumers finish this item. Item
  info (rows, expert, count) goes through a 2-deep mbarrier queue, so the producer can run a whole item ahead.
- **Knobs.**
  - `GLM53_TF_TC_CFG="gu,dn"`: 0-3 each.
  - `GLM53_TF_TC_TICKET` (default 1; 0 = static stride, for the W5 A/B).
  - `GLM53_TF_TC_CTAS=N`: at most N CTAs, to leave SMs to the overlapped stream.
  - Timing probes (`probe=1` no decode, `2` no mma), as 0260's.

### Exactness

**Bit-identical to fat, fast2 and v1** by the same argument as 0170 and 0260. Each output element is the same chain of
m16n8k16 mma:

- the same A fragment (decode_tile is a pure function of the trellis word);
- the same fp16 B values (the stage holds the same rows);
- ascending k tiles from a zero accumulator, with all of K in one warp;
- then the same `fwht_row` butterflies and the same epilogue formulas.

The helpers are verbatim copies of `exl3_fast.cu`'s, and the test compares their text and the epilogue's arithmetic
lines. What changed only moves data: which CTA or warp computes an element, the item shapes and order, and the pipeline
depth.

So tc is **row-independent** (patches/0085: the member count only decides zero-filled B columns and idle warps) and
**deterministic** (the ticket only picks the CTA). It shares snapshots with fat, fast2, once and auto, and the
per-request knob can switch between them freely.

It does **not** change how exactness is defined. Decode and verify windows keep the row-invariant kernels. Fast
prefill's bits were already its own (0080's rule), and tc keeps exactly those bits.

A **non-bit-identical** variant was considered and not written: Y stored in bf16, which halves the 1.07 GB of Y at
8,192 rows and so ~5 ms of the DRAM floor. It would change fast-prefill bits. Every fast-prefill kernel (the chunk
size, lean, the snapshots shared by fat/fast2/once/auto/tc) would then have to use it. Decode would not, because
fast-prefill bits already differ from decode by design. It would also need its own snapshot tag, like FP8 prefill. Keep
it as a separate knob if tc proves memory-bound at 8,192 rows.

### Offline results

- **Compiles for sm_121.** nvcc 13.4 (`-gencode arch=compute_121,code=sm_121`, torch's half/bf16 defines), plus a host
  syntax check of the bindings.
  - Registers: `__maxnreg__` = 120 at 544 threads. `__launch_bounds__(544, 1)` would have capped at 96 and spilled
    the accumulators to the stack.
  - gate/up cfg 1/2: 120 registers, 0 spills. down cfg 1/2: 120, 8-12 B spilled (one or two registers). cfg 3:
    111-112 registers, 2 CTAs an SM.
  - Shared memory: 88.9 KB dynamic + 0.6-1.2 KB static (cfg 1/2), 40.4 KB (cfg 3).
- **Host tests** (`tests/cuda/test_expert_tc_patches.py`): 111 passed, 21 GPU tests skipped.
  - Helper text identical to fat's, and the epilogue arithmetic identical.
  - The env switch and the dispatch.
  - A Python model of the kernel's index arithmetic, for all 6 configurations and 1/3/8/32 column blocks:
    - items cover every (expert, pass, block group) once under stride and ticket walks;
    - every B value an mma receives is the right member row at the right k, through an emulated ldmatrix.x4 and the
      m16n8k16 fragment layout, with zero-filled rows past the count;
    - every A word is fat's trellis word;
    - the epilogue writes each output of each present block exactly once, from the right accumulator;
    - the budgets (shared memory, registers).
  - An mbarrier protocol simulation with random interleavings and random copy completion times: no deadlock, and no
    stage is read before it lands or refilled while read. Mutations that the simulation catches: a wrong full parity,
    a missing arrival, a missing `expect_tx`, a wrong empty parity, and one arrival too few on "empty".
- The existing host suites for once / auto / mia / batch2 / batch-parallel / sessions / buckets / kv-pool / lean /
  knobs still pass on the patched tree (162 passed).

### Expected gain (arithmetic, not timed)

- **2,048 rows (production).**
  - Target: fast2's DRAM efficiency (10.5-11.5 ms against fat's 13.0) and fat's robustness to a shared SM (ticket).
  - That is -1.5 to -2.5 ms a layer, **+4-7% prefill**, if it holds under contention. W5 is the warning that
    isolated wins can vanish.
  - The gate: tc beats fat by >= 10% in `--contend`, and end to end by >= 3%.
- **8,192 rows (only with 0335 or single-stream).**
  - Target: 20-24 ms against fat's 28.6, i.e. -5 to -9 ms a layer, about +5-8% on top of the solo pieces' own gain.
  - The risk: 0260's `once` also ran 16-warp CTAs over 128 members and was no faster than fat. The differences here
    are no CTA-wide barriers, a deeper and decoupled pipeline, and an overlapped epilogue. If tc cfg 2 ties fat, MMA
    issue itself is the limit and the rest of the kernel work is moot.

## 4. patches/0335: solo prefill pieces

**Change.**

- `GLM53_TF_SOLO_PIECE=N` (0 = off; values 1-63 are refused).
  - In `Batcher._piece`, a fast prefill piece is cut at N tokens when no other slot holds a request, and at
    `GLM53_TF_BATCH_PIECE` otherwise (`batchplan.solo_piece`).
  - The rule is re-evaluated at every piece boundary. A request admitted in this round's plan makes the next piece of
    the first request a normal one.
- The decision uses only `self.seqs`, which both ranks hold alike: cancels, admissions and finishes are applied from
  the same round plans. The value is checked equal on both ranks at load, in its own `_gather_ints`, to stay clear of
  0340's edits to the batch settings list.
- `job.stats["solo_pieces"]` counts the solo pieces, and the boot log prints the setting and the chunk cap.
- **The pieces become bigger chunks only if the lean set holds them.** `GLM53_TF_PREFILL_ROWS_MAX` must be >= N;
  production has 2,048. `pfgrid.chunk_rows` caps chunks at `prefill_max`. With ROWS_MAX < N a solo piece still runs,
  as several 2,048-row chunks in one `decode.prefill` call. That saves only the per-piece resume and snapshot overhead
  (~1%?), and the boot line says so.
- `GLM53_TF_LEAN_LAZY_XU=1` (optional, same bits) allocates the lean set's Xu at first use. The fat family (fat / once
  / auto / tc) never touches it on the real checkpoint (shared sign vector). A request with `fat_experts=0` (plain
  fast2) allocates it then.

**Exactness, verified in the code.**

- Every piece is `decode.prefill(e, prompt[:end], resume=seq.resume)`.
- `batchplan.piece_end` ends fast pieces on the 64-token snapshot grid (0180's `piece_grid`; with `prefill_rows=auto`
  the grid is 64). A piece that ends inside the prompt leaves its grid snapshot exactly at `end`, because the snapshot
  point S of `prompt[:end]` is `end`. So there is no tail-rule chunk.
- 0085 made every fast kernel row-independent, runs the KDA scan on absolute 64-row blocks, and runs lean sub-blocks
  at `pos + k x block`. So the state after the piece that ends at X is the state a fresh prefill holds at X, for any
  earlier cut and any chunk size. The expert kernels' own row-count switches (fat/tc configurations, `large` at 4,096)
  are bit-neutral.
- The last piece's snapshot (last multiple of 64, or the tail rule) depends only on the prompt, not on the piece
  sizes.
- Per-piece session marks (`e.checkpoints`) are still honoured inside the bigger pieces. Fewer piece-boundary
  snapshots exist, which changes what a later request can resume from, but not its bits (resumed == fresh).
- Replies are therefore byte-identical with and without the knob. The GPU tests check solo vs the lone engine, a
  newcomer mid-prefill, and the follower's replay.

**Memory, per rank (the binding node is the worker node).** The lean set is 397 KiB a row (`lean.py`), of which Xu is 72 KiB.

| setting | lean set | vs today (2,048, eager Xu) | chunk transients (router partials 9 KiB a row) | worst-case MemAvailable (W6 stress min 14.05 GiB) |
| --- | ---: | ---: | ---: | ---: |
| today: ROWS_MAX 2048 | 0.78 GiB | 0 | 18 MiB | 14.05 |
| ROWS_MAX 4096 | 1.55 | +0.78 | +18 MiB | ~13.2 |
| ROWS_MAX 4096, lazy Xu | 1.27 | +0.49 | +18 MiB | ~13.5 |
| ROWS_MAX 8192 | 3.10 | +2.33 | +54 MiB | ~11.6 |
| ROWS_MAX 8192, lazy Xu | 2.54 | +1.76 | +54 MiB | ~12.2 |

- **All fit the >= 8 GiB MemAvailable worst-case target** beside the 1M-token pool. The pool is preallocated, so a
  solo 8,192-row chunk runs only when the other slots are idle. The 4-slot worst case sees only the static lean-set
  increase.
- **The load-time slot rule** keeps `GLM53_TF_BATCH_RESERVE_GB` (11) free after each slot, plus the 2 GiB store. The
  lean set is allocated before the batcher, so it lowers `free` at the 4th slot's check. W6's idle MemAvailable after
  load was 17 GiB on the worker node, so the 4th slot's check (free >= ~13.3 GiB) had ~4 GiB to spare. That falls to ~3 GiB
  (4096) or ~1.5 GiB (8192 eager). Check the boot line "batching 4 requests" in the window. If only 3 slots are made, lower `GLM53_TF_BATCH_RESERVE_GB` by the
  lean-set delta or use lazy Xu.

**Expected gain (single long prompt, nothing else running).**

- **Kernels.** fat's cost per token and layer falls 6.35 to 3.91 us at 4,096 rows and to 3.49 us at 8,192. Over 43
  layers that is -105 / -123 us a token, of ~780 us a token today (1,280 tok/s).
- **In-engine transfer has been ~1/3 of the kernel prediction.** F9 predicted ~+10% for 2,048 to 8,192 and measured
  +3%.
- **Estimate:** +4-6% at 4,096 and +5-8% at 8,192 (24.5k: 1,280 to ~1,340-1,380 tok/s), plus ~1% from 4x fewer piece
  resumes. With 0330's tc at 8,192 rows, a further +3-6%.
- **Cost.** A request that arrives during a solo piece waits for it: up to ~3.2 s at 4,096 or ~6.5 s at 8,192, against
  ~1.6 s today. **Recommend 4,096 first:** 86% of the kernel gain, half the stall, a third of the memory.
- The W7 window (another agent) is sweeping the static version: `BATCH_PIECE=8192`, `ROWS_MAX=8192`, per-request
  `prefill_rows` 2,048/4,096/8,192, in results/W7 `ab-B.log`. Its numbers replace the estimate above. The chunk-size
  effect is exactly 0335's gain, because a lone request is solo.

## 5. GPU test plan (one window; production down ~45-60 min)

Build an image from the committed tree through 0335. Keep production's `kvpool` image for the revert. Run everything on
the head node under a lease, with production stopped.

1. **Kernel correctness first; a hang here must not wedge the node.** Every command runs under `timeout 900`.
   - `PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda timeout 900 pytest -q -x tests/cuda/test_expert_tc_patches.py -k "not engine"`
   - This covers bits vs fast2 and fat across 64-8,192 rows, cfgs 0-3, ticket on / off, the CTA cap, the odd
     shapes, row subsets and the probes.
   - Gate: all green. A timeout means an mbarrier protocol bug: stop and do not continue.
2. **Kernel bench, isolated and contended** (~10 min):
   - `timeout 1800 python tests/cuda/bench_experts.py 2048 4096 8192 --tc --contend --no-v1 --variants fast2,fat,tc`
   - Read three things:
     - `tc` vs `fat s3` and the floor column, both isolated and contended;
     - `tc static` vs `tc` under contention: the W5 hypothesis. If static loses more than 5% relative, it is confirmed;
     - `tc ctas 44/40` under contention.
   - Gate to continue: bits "same" everywhere, and tc >= 1.10x fat at 2,048 contended, or >= 1.15x at 8,192.
3. **Engine tests:**
   - `pytest -q tests/cuda/test_expert_tc_patches.py -k engine`
   - `pytest -q tests/cuda/test_solo_piece_patches.py`
   - `pytest -q tests/cuda/test_batch2_patches.py -k "pieces or fast_prefill or lean or follower"` (regressions)
4. **Load A: production plus tc.**
   - Settings: `GLM53_TF_FAST_EXPERTS=tc`, everything else production.
   - Run `results/W5/ab.py` per request with `fat_experts` 1 (tc) vs a second load with `fat` (or knob 0/1 on the tc
     load, to compare tc vs fast2): 24.5k and 98k, 2 reps.
   - Also run `--suites exact` 10/10 and the reply sha against production.
   - Gate: >= +3% prefill with the same sha. With `GLM53_TF_PROFILE=1`, compare `moe.routed` per chunk.
5. **Load B: solo pieces.**
   - Settings: `GLM53_TF_SOLO_PIECE=4096 GLM53_TF_PREFILL_ROWS_MAX=4096 GLM53_TF_LEAN_LAZY_XU=1`, with fat and then
     with tc if step 4 passed.
   - Boot log: 4 slots made, and the "SOLO_PIECE" line.
   - Single prompts of 24.5k and 98k: tok/s and `solo_pieces` in the stats.
   - Two-request stall: `multiturn.py --modes stall` (a newcomer during a long prefill) for its TTFT and decode gaps.
   - `batchexact` 4/4 and `exact` 10/10.
   - Memory: `results/W7/mem.sh` during the 4 x 300k stress (MemAvailable >= 8 GiB).
   - Repeat with 8,192 if W7 shows 8,192 clearly ahead of 4,096.
   - Gate: >= +4% single-stream prefill, a newcomer's TTFT <= +3.5 s, memory target met.
6. **Revert** to `config/prod.env` (image kvpool), run the canary, check https, and re-arm the watchdog.

## 6. Risks

- **An mbarrier protocol bug would hang the GPU**, not crash. The simulation covers the protocol logic but not PTX
  semantics. Run step 1 under `timeout` with production down. The kernel falls back to nothing: a hang needs a
  container restart, and possibly `nvidia-smi -r` or a reboot (the GB10 slow-state caveats apply).
- **One CTA of 544 threads and ~90 KB an SM leaves no room for the overlap stream's kernels on that SM.** This is the
  W5 mechanism in the other direction. `GLM53_TF_TC_CTAS` (e.g. 44) and cfg 3 (fat's footprint) are the fallbacks the
  bench measures.
- **`__maxnreg__` needs CUDA >= 12.4.** The 26.07 container has CUDA 13. The down kernels spill 8-12 bytes.
- **The first fast chunk with tc builds a new extension** (~1-2 min; cached in `/cache/torch_extensions`). Do the
  warm-up before timing.
- **0335:**
  - Newcomers wait up to one solo piece.
  - The 4th slot's load-time margin shrinks by the lean-set delta.
  - With ROWS_MAX < N it only merges pieces.
  - Solo pieces apply to fast prefill only; exact requests keep 2,048.
