# Pipeline-parallel prefill (patch 0320, `GLM53_TF_PREFILL_PP`): analysis and the row-split design

Written offline, 2026-09-28, on the production patch set through 0290 (`config/prod.env`: batch pieces of 2,048 rows,
`LEAN_BLOCK=512`, `PREFILL_OVERLAP=1`, bf16 gathers, MTP prefill cache rows). No GPU and no Spark were used. Link and
kernel numbers are the measured ones in `docs/RESULTS.md`, `docs/EXPERIMENTS.md` and `docs/PATCHES.md` (0084); every
gain below is arithmetic. `docs/PROFILE.md` (the W7 profiling window) did not exist yet when this was written; the
comm share used here is the measured `GLM53_TF_PROFILE` data (S1, results/S1/ctx-profiles.log), so re-check the
estimate against PROFILE.md once it lands (see "What to re-check").

## Short version

- **Comm is already hidden.** Patch 0084 (in production) already does option (b): each 512-row sub-block's exchange
  runs on a comm stream while the next sub-block computes. The measured exposed exchange time is **~11 ms of a 24.2 s
  28k prefill (0.05%)**. What communication still costs is contention: NCCL and its host-staged copies run beside the
  compute kernels (~0.9 s of 27 s, 3%, visible as the hc lap growing when overlap was switched on).
- **Bytes cannot be halved.** With two ranks, today's all-gather of bf16 partials already moves exactly what an
  all-reduce (reduce-scatter + all-gather) moves: each rank sends R x D once. Sequence-parallel RS/AG moves the same
  total. The next block's matmuls need every row on both ranks, so a second exchange is unavoidable.
- **(a) True PP with a second, PP-layout weight copy: impossible.** It needs ~152 GiB a node for weights alone (~175 GiB with everything else) on 121 GiB (119.7 on
  the worker node). PP instead of TP costs ~30-40% of single-stream decode.
- **(c) Expert parallelism: no gain, and no exact version at sane bytes.** Expert bytes and FLOPs are the same under
  TP and EP. Keeping TP's bits needs every routed pair's fp32 K-half output shipped: 8x the MoE bytes, about +29% time.
- **What TP really wastes is replicated row-wise work.** Every rank runs the hyper-connections (hc_post, and the next
  hc_pre with its 20 Sinkhorn steps and norm), the embedding, the taps and the final norm on all rows. That is ~8% of
  a prefill as pure hc work (~12% in the hc lap with contention), done twice. This is the part pipeline parallelism
  would not duplicate.
- **Chosen: the row split.** Implemented in `patches/0320-glm-prefill-row-split.patch`, off by default, same bits.
  Inside 0084's pipeline, each rank does the hyper-connections for half of each sub-block's rows:
  1. The partials are traded as row halves (NCCL send/recv). That is half the bytes on the critical path.
  2. hc_post and the next hc_pre run on each rank's own rows.
  3. The normed rows are shared back in place, 3-4 pieces before anyone reads them.
  Same bytes a rank in total, half the hc work.
- **Expected: +3 to +5% prefill tok/s** (1,255 -> ~1,295-1,320 at 24.5k-98k), untimed.

## 1. Where the time is (production shapes)

A batch piece is a lean chunk of 2,048 rows in 4 sub-blocks of 512 (`GLM53_TF_BATCH_PIECE=2048`,
`GLM53_TF_LEAN_BLOCK=512`). It takes ~1.63 s at the measured 1,255 tok/s (both ranks in lockstep).

The latest per-component profile is S1 (28,045 tokens, 8,192-row chunks, block 1,024, overlap on; 24.2 s GPU on rank
0). The shares carry over to 2,048-row pieces within a few points:

| lap | s | share | split across ranks today? |
| --- | ---: | ---: | --- |
| moe.routed (routed experts) | 5.84 | 24% | yes (TP: half of every expert) |
| dsa.sparse_attn + dsa.attn + dsa.indexer + dsa.proj + dsa.o_proj | 6.32 | 26% | heads split; latent / index keys replicated |
| **hc** (hc_post + hc_pre + taps + final norm) | **2.89** | **12%** | **no: every rank, every row** |
| kda.proj + kda.chain + kda.o_proj | 4.31 | 18% | yes (heads) |
| moe.router (router, top-k, grouping; before 0190's `moe_glue`) | 1.27 | 5% | no (replicated) |
| moe.shared + moe.combine + mlp.dense | 2.06 | 9% | yes |
| allgather (the waits the compute stream still saw) | **0.011** | **0.05%** | - |
| mtp.* (now cache rows only: 0190 `MTP_PREFILL_CACHE`) | ~1.3 | 5% (now ~1%) | - |

The hc lap was **2.0 s without the overlap and 2.9 s with it** (RESULTS, "Prefill work": fast lean 8192 vs 8192 +
fp8 + overlap). The exchanges disappeared from the timeline (3.0 s -> 0.0), and the NCCL kernels plus their copies
now run beside the hc kernels and slow them. So the hc lap is about 2.0 s of work (~8%) plus ~0.9 s of comm
contention (~3%).

## 2. Communication math (per 2,048-row piece, per rank)

- **Exchange sites:**
  - 45 attention outputs (34 KDA + 11 DSA) and 45 FFN outputs (3 dense + 42 MoE): 90 a sub-block.
  - x 4 sub-blocks = **360 all-gathers a piece**.
  - The MTP head makes none in prefill any more (cache rows only).
- **Bytes an all-gather:** a 512-row bf16 partial, 512 x 4,096 x 2 B = **4 MiB sent, 4 MiB received**.
- **A piece:** 360 x 4 MiB = **1.41 GiB (1.51 GB) each way** a rank, i.e. ~740 KB a token.
- **Link:**
  - 200 Gb/s line (25 GB/s).
  - 109 Gb/s measured per QP (13.6 GB/s, EXPERIMENTS appendix).
  - NCCL achieved ~10.5 GB/s on 8 MiB sub-block exchanges (0.8 ms a 1,024-row bf16 exchange, PATCHES 0084).
  - No GPUDirect RDMA: NCCL stages through host memory. On GB10 that is the same LPDDR5X, so each byte costs a few
    extra DRAM passes but no PCIe hop.
- **Wire time if nothing overlapped:** 1.51 GB / 10.5 GB/s = **144 ms a piece (8.8% of 1.63 s)**; 111 ms (6.8%) at
  13.6 GB/s. This is what 0084 hides.
- **Exposed today:** measured 0.05% (the `allgather` lap). By construction the only unhidden exchanges are these:
  - at each MoE layer, the last attention sub-block's exchange, which the routing needs for every row. It overlaps
    only the previous sub-block's hc post, which is about as long.
  - the first exchange of each chunk.
- **Hidden but not free:**
  - NCCL's CTAs on the 48 SMs (1-2 channels).
  - The host-staging copies: ~3 DRAM passes a byte each way, ~8.5 GiB a piece. That is ~37 ms of DRAM time at 230
    GB/s, 2.3%, competing with memory-bound kernels.
  - Rank skew (every exchange re-synchronizes the ranks).
  - Together these match the ~3% contention above.

**All-reduce vs all-gather vs RS/AG, two ranks.** Row r's reduced value needs both ranks' partials of row r.

- The all-gather of partials sends R x D a rank. NCCL's all-reduce of two ranks is a reduce-scatter then an
  all-gather: R/2 x D twice = R x D.
- A sequence-parallel RS then AG of the normed rows is also R/2 + R/2.
- So **no layout halves the bytes**: every rank's next matmul needs every row's input.
- What the split does change:
  - the exchange on the critical path (right after a partial is written) carries half the bytes;
  - the work between the two halves (the hyper-connections) is done once instead of twice.
- NCCL's own reduce-scatter would add in bf16 inside NCCL: new bits. The split does its reduction in hc_post, in fp32,
  rank 0 first, as today.

## 3. Options

### (a) True pipeline parallelism for prefill

The idea: each rank holds whole layers for its stage, the pieces are cut into micro-pieces, and only activations cross
the link, once per stage boundary.

**Memory with a second, PP-layout copy.**

| | GiB a node |
| --- | ---: |
| today: TP shard (82.2 GB of files) + loader overhead | 76.6 (78.1-78.3 resident) |
| a PP stage: half the layers whole (the full model ~150 GiB = 2 x 76.6 minus a few GiB of replicated tensors, halved) | ~75 |
| OS + services, CUDA context + NCCL, drafter | 5.2 + 2.1 + 0.8 |
| KV pool (1,048,576 tokens, FP8), window + lean buffers, session store | 7.4 + ~6 + 2 |
| **total** | **~175 GiB against 121 (119.7 on the worker node)** |

That is short by ~54 GiB; even without the KV pool and the session store it is ~45 GiB short, so it is impossible.

- Streaming the PP copy from NVMe per prefill (75 GiB at 3-6 GB/s = 12-25 s) is slower than the prefill itself.
- Keeping the experts TP and making only the non-experts PP gains nothing: the experts are 24% of the time and the
  largest weights, and the rest is the part that is already split.

**PP instead of TP (one layout).** The memory is the same.

- Prefill would lose the replicated work (the same ~10-13% the row split goes after) but gain:
  - a pipeline bubble: with 4 micro-pieces and 2 stages, (p - 1) / (m + p - 1) = 20% unless pieces stream
    back-to-back;
  - stage imbalance: the 11 DSA layers cost ~4x a KDA layer at long context and sit at fixed depths, so any split by
    layer count is unbalanced, and it drifts with context.
- Decode, the thing this deployment leads on, becomes sequential across the nodes: EXPERIMENTS section 3 estimates
  **42 ms vs 30 ms a one-row step (-30%)**, and only batching recovers it.

**Rejected.**

### (b) Keep TP weights, overlap comm with compute: already in production (0084), and what is left

- **Micro-batching a piece.** Done by 0084: a 2,048-row piece is 4 sub-blocks, and exchange k overlaps sub-block
  k+1's pre work (projections, KDA chain or attention, o_proj; or shared expert + combine). It uses two slots, a
  high-priority comm stream, and bf16 partials written directly. Exposed: 0.05%.
- **Overlapping the MoE dispatch.** There is no dispatch in TP: hc_pre's normed rows are on both ranks, and the
  experts' output is one exchange like any block's. The MoE boundary (routing needs every row) is the one structural
  bubble left. Its size is the last sub-block's exchange beyond the previous sub-block's hc: ~0-0.2 ms x 42 layers =
  **<= 8 ms a piece (0.5%)**.
  - Filling it with work would need to move the shared expert of sub-blocks 0-2 before the routing, or to split the
    routed experts. The first is exact, worth <= 0.5%, and is listed as a follow-up. The second reads every expert's
    weights twice.
- **RS/AG instead of all-gather to halve bytes.** Not a byte saving (section 2), but its sequence-parallel form, with
  the work between the two exchanges done on half the rows, is where the remaining gain is. That is the chosen design
  (section 4).
- **Deeper pipelining (2 exchanges in flight).** It can only absorb skew jitter. The measured waits are already 0.05%.

### (c) Expert parallelism (each rank holds 144 whole experts)

- **Memory and weight prep.**
  - The bytes are the same: 1.81 GB a MoE layer a rank either way. That is 288 experts x half (TP) or 144 x whole
    (EP).
  - The EXL3 layout would need new prepared folders (0140) and a new split in `split.py`. EXL3 tiles are 16 x 16, so
    whole experts are actually simpler to load than TP's column and row halves.
  - Keeping both layouts costs ~76 GB more NVMe a node.
- **Exactness.**
  - TP's down projection sums K = 2,048 as two halves, each rank's half in its own fp32 chain. Each rank's combine
    over the 9 slots is rounded to its bf16 partial, and hc_post adds the two partials.
  - EP computes the whole K on one rank: other roundings, other bits.
  - To keep the bits, the owner of expert e must produce both K-half outputs y_e^(0), y_e^(1) of every routed pair,
    and each rank must combine its half over all 9 slots of the row. Those slots live on both ranks.
  - So each rank receives ~4 remote pairs x 4,096 x 4 B = **64 KiB a row a MoE layer, against 8 KiB today (8x)**.
    That is +4.6 GiB a piece, about +470 ms at 10.5 GB/s (**+29% time, -22% tok/s**).
  - bf16 transfers would change the bits.
- **Gain even with new bits: none.** In a 2,048-row piece every expert is touched, so each rank reads the same bytes
  and does the same FLOPs under TP and EP. EP adds routing imbalance (hot experts). Decode reads +27% expert bytes on
  the busier rank at one row (EXPERIMENTS section 3).
- **All-to-all vs all-gather.** The EP combine is still one [R, D] exchange if the bits may change: the same bytes as
  today. Nothing saved.

**Rejected.**

### Ranking

| option | expected prefill gain | effort | exact | verdict |
| --- | ---: | ---: | --- | --- |
| (a) PP with a second weight copy | - | - | - | impossible (~175 GiB a node needed) |
| (a') PP instead of TP | ~+5-10% prefill, -30% decode | weeks | yes | rejected |
| (b) more overlap (MoE boundary fill, deeper pipeline) | <= 0.5% + jitter | 1 d | yes | follow-up |
| (c) expert parallelism | -22% (exact) / ~0 (new bits) | weeks | no, at sane bytes | rejected |
| **(b') row split of the replicated row-wise work (0320)** | **+3 to +5%** | 2-3 d + one GPU window | **yes** | **implemented, off** |

## 4. The chosen design: row-split hyper-connections in the pipelined chunk (`pfpp.py`)

**Ownership.** Each sub-block of r rows is split at `split(r)` = half rounded up to a multiple of 64 (at most r).
Rank 0 owns the first rows and rank 1 the rest; for r = 512 that is 256 / 256. The 64-alignment keeps every call on
the slab grid 0084 already verified. Chunks with fewer than `MIN_BLOCKS` = 2 sub-blocks keep 0084's path: with one
sub-block every piece drains, and two dependent exchanges would both be exposed. A 2,048-row batch piece has 4.

**Per exchange site** (one sub-block's partial):

1. The producer writes its partial for all r rows into `slot[i][rank]`, where `slot` is [2, rows, D] (the same memory
   as 0084's part + gather buffers).
2. `issue` makes one NCCL group of `ncclSend(slot[i][me, other's rows])` and
   `ncclRecv(slot[i][other, my rows])`, on the comm stream after the producer.
3. The post waits for it. `slot[i][:, my rows]` is then [rank 0's partial, rank 1's partial] of the rank's rows,
   which is exactly hc_post's `gathered` operand (rank stride = rows x D; rows contiguous).
4. hc_post plus the next hc_pre (or the taps and the final norm) run on the own rows only, in 0084's L2 slabs, fused
   where 0190's `hc_fused` applies.
5. `share` trades the own rows of what later pieces read from every row, in place, as one NCCL group:
   - normed rows (8 KiB a row) + their group sums (256 B a row);
   - after the last layer: the final-normed rows, their group sums and the DFlash2 taps.
6. The consumer, the next piece that reads those rows, waits for the share first (`need`). That is 3-4 pieces later,
   or the MoE routing (`need_all` after the drain).

**Order on the compute stream** (4 sub-blocks, one layer): `pre a0 | pre a1, post a0 [share a0] | pre a2, post a1
[share a1] | pre a3, post a2 | post a3 [share a3], need_all, routing | pre f0 (need f0: no-op) | pre f1, post f0 [share
f0] | ...`. Both ranks issue the same sends and receives in the same order from one comm stream.

**Hazards** (the same rules as 0084's `Pipe`, plus):

- A share's receive overwrites the other rank's rows of `lb.normed` / `lb.xs`. The previous readers of those rows
  (this sub-block's pre work) ran before the post that records the share's ready event.
- The next writer of the own rows (the next site's post of the same sub-block) runs after the consumer waited for
  the share.
- A slot's two halves are rewritten only after its swap was waited for.
- The other rank's rows of the residual streams `lb.x` go stale. Nothing reads `lb.x` after a chunk.

**Bits.**

- Every kernel is 0084's, on the same operands, on a row subset. The hyper-connection kernels, `stream_mean` and
  `rmsnorm` are row-independent: 0084 already cuts them into 64-row slabs and a partial last slab, and 0085 made every
  fast-chunk kernel independent of its call's row count.
- The exchanges are byte copies.
- hc_post adds `gathered[0]` (rank 0) then `gathered[1]`, as before.
- So every row's streams, normed rows, group sums, taps and final rows equal 0084's, and so do everything built from
  them (KDA / DSA / MoE inputs, caches, KDA state, the head's last row). Replies and snapshots are unchanged, shared
  with 0084 / 0082 / 0080, and resumed == fresh holds.

**What it changes.**

- The collectives: all-gathers become send/recv groups, and there are twice as many exchanges.
- Both ranks must agree on `GLM53_TF_PREFILL_PP` and on `GLM53_TF_PREFILL_OVERLAP`'s default. This is checked at load
  (`engine.py`, next to 0084's line). The per-request `prefill_overlap` knob (0093) travels in the header, so both
  ranks switch together.
- Memory: the swap slots are 16 MiB (0084's are 24 MiB, allocated only if a small chunk uses them), plus 0.5 MiB of
  final group sums at 2,048 rows, plus NCCL's p2p connection buffers (tens of MiB, allocated at the first swap).

**Communicator.** `comm.NCCL.swap(sends, recvs)` is `ncclGroupStart`, `ncclSend`/`ncclRecv` per tensor, then
`ncclGroupEnd`, through the existing ctypes handle, on the current stream. The RoCE communicator (0230) forwards to its
NCCL base. `comm.swap` falls back to a padded byte all-gather for communicators without `swap` (test stand-ins): the
same bits, more bytes.

## 5. Expected gain (arithmetic, not timed)

Per 2,048-row piece (1,632 ms):

| item | ms |
| --- | ---: |
| hc work today (~8% of the piece; the hc lap without overlap contention) | ~130 |
| saved: half of it on each rank | -65 |
| saved: half of the taps' stream_mean and the final norm (in the hc lap already) | (included) |
| lost: at each of 42 MoE boundaries the last sub-block's share is exposed (a 2 MiB swap + alpha ~0.25 ms), about what its halved hc saves (~0.29 ms) | +8 to +12 |
| lost: first and last share of the piece | +1 |
| lost: 360 more NCCL operations a piece (host enqueue ~40 us each, ahead of the GPU; proxy work) | ~0-5 |
| unchanged: NCCL / staging contention (~3%; same bytes) | 0 |
| **net** | **-45 to -60 ms: -3 to -4% time, +3 to +4% tok/s** |

- Upper end (+5%): the contention shrinks too. The critical-path exchange carries half the bytes, so it overlaps a
  shorter window, and the hc kernels it slowed now do half the rows.
- Lower end (~+2%): NCCL p2p over the host-staged net transport turns out slower than its collectives, or the
  doubled operation count adds per-op proxy latency that the pipeline cannot hide.
- 1,255 tok/s -> **~1,295-1,320 tok/s at 24.5k-98k.**
- Decode: unchanged (untouched).
- Prompts under 513 tokens (one sub-block): unchanged.
- Stacks with 0240's fused mhc if that is ever adopted (the split halves whatever hc costs), and with every expert or
  attention kernel change.

**Follow-ups on the same machinery** (exact, not implemented):

- **Router logits on the own rows**, sharing the [rows, 288] fp32 logits. The top-k and grouping still need every
  row. That is ~1.3% of a prefill after 0190, so <= +0.7%.
- **The shared expert of sub-blocks 0-2 before the routing**, filling the MoE boundary: <= +0.5%.
- **The replicated DSA latent / index-key projections and cache writes**, computed on the own rows and the rows
  shared: part of `dsa.proj` + `dsa.indexer`, <= +1-1.5%.
- Together with the split, the ceiling of "do every replicated row-wise thing once" is **~+6-8%**.

## 6. Tests

Offline, run for this patch (`PYTHONPATH=<tree>/src:<tree>/tests/cuda:tests/cuda pytest -q
tests/cuda/test_prefill_pp_patches.py`): **9 passed, 7 GPU tests skipped**. Also, with 0320 applied:
`test_overlap_patches`, `test_lean_patches`, `test_cindep_patches`, `test_fastpf_patches`, `test_roce_patches`,
`test_kv_pool_patches`, `test_knob_patches` and `test_glue_patches` host parts pass. One failure,
`test_glue_patches::test_latent_tc_tag_and_load_only`, fails the same way without 0320.

- **Two real processes over gloo** (CPU, `torch.distributed`), each running test_lean_patches' hash model as its rank:
  - The rank-specific output projections, down, experts and head make the partials differ between the ranks.
  - On each rank, the split chunk == 0084's pipelined chunk with the real all-gather, bit for bit: last-row logits,
    final-normed rows, taps, KDA states and conv windows, attention caches, and the own rows' streams.
  - Both ranks' final rows agree.
  - Coverage: 4 pipeline variants, R = 65 / 100 / 128 / 130 / 200 / 256 on 64-row sub-blocks (partial last
    sub-blocks, a rank owning 0 rows of one), two positions, EXL3 and MLX MoE.
  - Every chunk makes one share per sub-block and site (plus the embedding's), and one swap per exchange.
- The swap's rank layout, checked explicitly: `[0]` is rank 0's partial of the rank's rows, `[1]` rank 1's (4 row
  counts, overlapped and not).
- The padded all-gather fallback gives the same bits.
- **Controls, caught on both ranks:** a routing that does not wait for the shared rows, a swap that sends the wrong
  half, and a one-sub-block chunk that must not split.
- The piece order in one process: pre(k) before post(k - 1), every sub-block's rows waited for before the routing,
  nothing pending at the end.
- Knob, settings pair, split (64-aligned, disjoint, covering, 1-2,048 rows), `applies`.

GPU, in `tests/cuda/test_prefill_pp_patches.py`, not run:

- **One GPU:** every row-wise kernel the split calls on half a sub-block equals the rows of the whole-sub-block call,
  bit for bit, on real shapes (D 4,096, 4 streams, 20 Sinkhorn steps) at 100 / 512 / 1,024 rows, in a fast chunk.
  Kernels: hc_pre, hc_post on the slot's strided own-row view, `hc_post_pre` with `HC_FUSED` 0 and 3, `stream_mean`,
  `rmsnorm`.
- **Two engines on one GPU** (`PP_TWO_PROC=1`): rank 0 and rank 1 of the synthetic EXL3 checkpoint as two processes,
  with exchanges through a host-staged gloo communicator. Committed state and first token with the split == without
  it, on both ranks, 3-700 tokens (64-row sub-blocks, 256-row chunks).

## 7. GPU test plan (one window, both Sparks)

1. **Build.** Image with 0320 on the production set (`glm53-tensorfold:rowsplit`). In the image:
   - `pytest -q tests/cuda/test_prefill_pp_patches.py` (one GPU; then `PP_TWO_PROC=1` for the two-engine test);
   - regressions: `test_overlap_patches.py`, `test_lean_patches.py`, `test_cindep_patches.py`,
     `test_glue_patches.py`, `test_kv_pool_patches.py`.
   - Gate: all pass. The kernel test must be bitwise; if it is not for any kernel, stop: the split cannot be exact on
     this Triton.
2. **Load refusal.** Start with `GLM53_TF_PREFILL_PP=1` on rank 0 only: it must refuse at load with the settings
   message (no hang). Then both ranks.
3. **Bits** (prod config + `GLM53_TF_PREFILL_PP=1`):
   - ab.py reply sha == `8794a3463259cc2f` (W6);
   - `bench/glmbench.py --suites exact` 10/10;
   - `multiturn.py --modes batchexact` 4/4;
   - a session revisit resumes (0180 / 0250 snapshots written with PP=0 are read with PP=1: the same bits, the same
     compat hash, which does not include this knob);
   - a 98k prompt's reply hash equal PP 0 vs 1.
4. **Speed** (restart between, load-time knob):
   - cold prefill 24.5k and 98k, 2 runs each, PP 0 vs 1;
   - `GLM53_TF_PROFILE=1` both: the `hc` lap should drop ~45-50%, `allgather` stay < 0.5%;
   - one nsys window with W7's tooling: `ncclDevKernel_SendRecv` beside the compute kernels, and the share exchanges
     not delaying the next pre work.
   - Adopt at **>= +2% at both lengths** with identical hashes.
5. **Stall and memory:** the 4-stream stall mode (a 35k prefill beside 3 decoders), with decode gaps not longer.
   MemAvailable after load and at a 2,048-row piece within 0.1 GiB of PP=0 (NCCL p2p buffers appear at the first
   swap).
6. **If it is slower than expected:**
   - `NCCL_NCHANNELS_PER_NET_PEER=1/2/4` and `NCCL_P2P_NET_CHUNKSIZE` (the p2p path's own knobs;
     `NCCL_MAX_NCHANNELS` affects the collectives), and `GLM53_TF_PREFILL_HC_SLAB=256` (a whole own half in one slab);
   - a pair of pieces with PP 0 / 1 under nsys, to see whether the extra operations' proxy latency is exposed at the
     MoE boundary.

## What to re-check when docs/PROFILE.md lands

- **The comm share.** If W7 shows exposed comm well above the 0.05% measured in S1, the MoE-boundary fill (follow-up)
  moves up. Exposed means `allgather` or NCCL time on the compute stream's critical path, for example because
  2,048-row pieces with 512-row sub-blocks differ from S1's 8,192 / 1,024.
- **The hc share.** The split's gain scales with it. If the hc lap is under ~6% of the piece at production shapes,
  the expected gain drops to ~+2%.
- **NCCL SM / DRAM contention.** If it is large, the row split does not reduce it (same bytes). The levers then are
  `NCCL_MAX_NCHANNELS` / fewer NCCL CTAs, and 0230's RoCE path for large exchanges (host memory the CX7 writes into
  directly, no staging copy).
