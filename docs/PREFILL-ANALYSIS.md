# GLM-5.3-Flash prefill on TensorFold: where a 512-row chunk goes

Written offline from the kernels (TensorFold `2f8e514` + patches 0001-0004); no GPU was used. Every number
below is an estimate from FLOPs, bytes and serial steps unless it says "measured". `patches/0005` measures the
real split (use it before trusting the ranking in detail), and `patches/0006` removes the top bottleneck.

## Setting

- Measured: 2 ranks, q4mse non-experts, EXL3 routed experts, `GLM53_TF_PREFILL_ROWS=512`: ~404-420 tok/s flat
  from 1.8k to 28k tokens, i.e. **~1.25 s per 512-row chunk** (both ranks in lockstep). 64-row chunks with BF16
  non-experts: 266 tok/s (240 ms per 64-row chunk). vLLM on the same weights: 960 / 1340 / 1448 tok/s at
  1.8k / 7k / 28k.
- Per rank: 45 layers (34 KDA, 32 heads of 128; 11 DSA, 32 heads of 256), 42 MoE layers (288 experts, top 8 +
  shared, 1,024 of each expert's 2,048 width), hidden 4,096, plus the MTP layer that absorbs every prefill row.
- GB10: 48 SMs at 2.4-2.6 GHz, 273 GB/s LPDDR5X. BF16 `mma.sync` peak about 60 TFLOP/s dense with fp32 accumulate
  (1/16 of the 1 PFLOP sparse-FP4 figure, the RTX 5090 ratio). The recipe measures the EXL3 expert kernel at
  208-220 GB/s.

## Estimate per 512-row chunk (per rank)

| # | Component | Estimate | Share | What bounds it |
| --- | --- | ---: | ---: | --- |
| 1 | MoE routed experts (EXL3 grouped kernel, 42 layers + MTP) | 600-1000 ms | 50-75% | Each expert's weights are re-read from DRAM once per 16 members, and ~0.9M programs are launched per layer, almost all of them empty |
| 2 | Non-expert 4-bit matmuls (KDA/DSA projections, shared expert, dense MLP, head) | 120-180 ms | 10-15% | Tensor-core compute: 4.5 TFLOP at ~25-37 TFLOP/s in Triton |
| 3 | DSA sparse attention + indexer (only past 2,051 tokens) | 60-180 ms (under 10 ms below 2k) | 0-15% | Per-row gathers of 2,051 keys, and one useful row per 16-row mma tile |
| 4 | All-gathers (92 x 8 MB fp32 partials) | 30-40 ms | 3% | 200 Gb/s link, plus rank skew |
| 5 | MTP head absorb (one more DSA+MoE layer over all rows) | 20-30 ms | 2% | Same kernels as rows 1-2 |
| 6 | Hyper-connections (90 `hc_pre` + 90 `hc_post`) | 15-25 ms | 1-2% | L2 reads of `fn` (786 KB per row per call) |
| 7 | KDA recurrence (34 chains) | 10-17 ms | ~1% | 512 serial row steps per layer, latency-bound |
| 8 | lm_head over all 512 rows (only the last row is used) | ~10 ms | ~1% | Wasted compute |
| 9 | Router grouping (`glue._group`, one program) | ~8 ms | <1% | O(R) serial loop on one SM |

The totals, 0.9-1.5 s, bracket the measured 1.25 s. The 64-row measurement agrees: 64 rows touch at most ~240
distinct experts, which is about 1.5 GB a layer and about 230 ms at the peak read rate. That is essentially the
whole 240 ms chunk, so the experts dominated there too. Routing must also be local (consecutive tokens share
experts), or the 64-row number would be impossible.

### 1. Routed experts: the grouped EXL3 kernel does not amortize past 16 members

The weights are 288 x (2 x 4096 x 1024 + 1024 x 4096) x 0.5 B = **1.81 GB per layer per rank**, 76 GB for 42
layers. That matches the 88.6 GB a rank holds. A 512-row chunk makes 4,096 (row, expert) pairs per rank and
layer, 14.2 per expert on average, skewed in practice.

`exl3.cu grouped_kernel` has one program per (expert u, 128-column block, matrix x K split, 16-member tile). Its
grid is sized by the window's row count: `MT = ceil(R/16) = 32` member tiles, whatever the actual counts.

- **Weight re-reads.** Each non-empty member tile streams its expert's slice again. The member tiles of one expert
  sit in different `blockIdx.z` slices, 2,312 programs apart in launch order, so after ~150 MB of other weights
  the slice comes from DRAM again. Traffic is `sum_e ceil(n_e/16)` expert reads: between 288 and 256 + 288 = 544
  per layer, so **1.0-1.9x the layer (1.8-3.4 GB)**. At 210 GB/s that is 8.6-16 ms per layer. So cost per token
  hardly falls as rows grow. That is why 64 -> 512 rows gained only 1.55x, and why raising PREFILL_ROWS further
  would buy nothing for the experts.
- **Empty programs.** Gate/up launches 289 x 8 x (2 x 4 x 32) = 592k programs and down 289 x 32 x 32 = 296k. Only
  about 2-4% of them do work. The rest launch, read member codes, `__syncthreads` and exit, each holding 32 KB of
  static shared memory (at most 3 per SM). Estimate 1-4 ms per layer. At R = 2048 it would be 3.5M programs.
- **A-operand re-reads.** X (fp16, 4,096 pairs x 4,096 x 2 matrices = 67 MB) is read once per column block (8 for
  gate/up). The column blocks of one expert are far apart in launch order, so up to 0.5 GB per layer comes from
  DRAM. That is ~2 ms.
- Z partials (0.4 GB written and read) plus about 0.5M one-warp rotation and epilogue programs add ~2 ms.
- Tensor cores are not the issue. Per 16x16 weight tile a lane spends ~42 instructions on trellis decode against
  2 `mma`, so decode ALU is ~1.3 ms per layer per member-tile pass, under the memory time. It is a GEMV design
  whose m16 tile happens to be full at 14 rows. What it lacks is reuse of a decoded or fetched tile across more
  than 16 rows.

Per layer: ~14-25 ms. Times 43 layers: **600-1000 ms per chunk.**

### 2. Non-expert matmuls

About 4.37 G 4-bit params per rank: KDA 69 M x 34, DSA ~63 M x 11, dense MLP 0.23 G, shared experts 0.53 G,
head 0.32 G. At 512 rows that is 2 x 512 x 4.4 G = 4.5 TFLOP. `qmm._qmm` uses BM = 128 row tiles, the 4 row
tiles of a column block run concurrently, and weights come from L2, so it is compute-bound: 120-180 ms at
40-60% of peak. The head over all rows wastes ~0.33 TFLOP; prefill only needs the last row's logits (the MTP
head reads `fnormed`, not the logits). Row-exact tuning options that change no bits: wider BN, since columns
are independent, and more warps.

### 3. Attention and indexer

- Dense (context under 2,051): `_chunks` programs cover 16 rows x 1 head x 512 keys, so K/V is read once per
  16 rows, from L2 across the 32 row blocks. At 1.8k context that is ~15 GFLOP per layer, under 10 ms per chunk
  in total.
- Sparse (past 2,051): `_sparse_chunks` is **per row** (row, head, 512-token chunk). The query sits in row 0 of
  a 16-row mma tile, so 1/16 of the tensor work is useful. It gathers 2,051 keys and values per (row, head), about
  2 MB, which is 34 GB of gathers per layer, mostly L2 hits because neighbouring rows pick overlapping pools.
  Estimate 5-15 ms per layer, 60-180 ms per chunk. The flat measured tok/s from 1.8k to 28k points to the low
  end.
- Indexer `_scores` is per row over `capacity/4` pools, not the visible ones, followed by two `torch.sort` over
  [R, capacity/4]. That is a few ms per layer at long capacity.

Fix, if the profile shows it: batch rows that share a pool selection or a head into 16-row tiles. A row's sums
must keep the same K order. Upstream already does this for dense rows.

### 7. KDA recurrence

Yes, it is latency-bound. `kda.cu chain_kernel` runs one 1,024-thread block per head (32 blocks on 48 SMs), and
the R rows go strictly in sequence. Each row does:

- a dependent L2 load of the conv inputs;
- 5 `__syncthreads`;
- ~10 chained 5-step warp-shuffle reductions (norms, the delta-rule read, the output, the RMS norm);
- a second L2 round trip for the output gate.

That is ~1,500-2,500 cycles, or 0.6-1.0 us per row: 0.3-0.5 ms per layer at R = 512 and **10-17 ms per chunk,
about 1%**. A chunked (WY/UT) parallel scan would make this 3-10x faster, but it is not bit-exact against the
serial step. It computes the chunk's state update as matmuls, with a different fp32 summation order.

Using it for prefill only, with decode kept serial:
- (a) drafted == serial survives, because both run in decode.
- (c) only survives if sub-chunk boundaries are fixed to absolute positions (multiples of 64, say), not to the
  prefill chunk.
- (b) breaks. A snapshot taken after a reply holds reply positions built by serial decode steps. A fresh
  prefill of the same tokens builds them with the scan, and the bits differ. Keeping (b) would mean refusing
  resumes after replies, or re-prefilling the reply rows.

For about 1% of the time, that trade is not worth it.

### Other items

- All-gathers are 3%. They could overlap with the next layer's hc/proj work on a second stream, but only 30 ms
  is at stake.
- `_group` is O(R) serial on one SM and grows linearly with chunk size (~35 ms per 2,048 rows).
- rot_in and the epilogues launch one 32-thread program per (pair, 128-block). That is ~0.5M programs per layer
  at R = 512. They could be merged into warps per program, with the same arithmetic per row.

## patches/0005-glm-prefill-profile.patch (probe)

`GLM53_TF_PROFILE=1` makes `prefill` open a session. Probe sites in `forward.py`, `mtp.py` and `decode.py` record
one CUDA event each on the current stream, and the time between consecutive events goes to the named component:

- `kda.proj`, `kda.chain`, `kda.o_proj`
- `dsa.proj`, `dsa.attn`, `dsa.indexer`, `dsa.sparse_attn`, `dsa.o_proj`
- `mlp.dense`
- `moe.router`, `moe.routed`, `moe.shared`, `moe.combine`
- `allgather`, `hc`, `embed`, `head`
- `mtp.*` (the MTP absorb, same names)
- `drafter`, `commit`, `stage`, `resume`, `sample`

Everything, NCCL included, runs on one stream, so the intervals add up to the prefill's GPU time. The session
ends with one synchronize after the first token is sampled. Rank 0 prints a table (component, ms, ms per chunk,
%) and a `GLM53_TF_PROFILE {json}` line to stderr (docker logs). Unset, each site costs one `if profile.ON`.
There are no events or syncs, and CUDA graphs are untouched: sites are inactive outside `prefill`, and graph
replays run no Python. No tensor is read or written, so the bits are the same (tested). An all-gather's time
includes waiting for the other rank.

Ordering: `0040-glm-comm-prefetch` was regenerated on top of 0005, and it keeps the probe sites in its reordered
`moe_block`. Do not renumber or rebase 0005 without regenerating 0040. 0006 touches only the `exl3*` files and is
independent of both.

## patches/0006-glm-exl3-expert-loop.patch (the speedup)

- **What changes.** `exl3.cu` gets `grouped_loop_kernel`, one program per (column block, expert, matrix x split).
  It walks all of the expert's member tiles, MG at a time (default MG = 2, NT = 4). `exl3_mm.routed` uses it
  whenever the window can give an expert more than 16 members (`members.shape[1] > 16`: prefill chunks, MTP
  absorbs, backlogs). Decode and verify windows of up to 16 rows keep upstream's kernel.
- **Effect.**
  - Each weight slice comes from DRAM once per window. Later passes over the same 32 KB slice are back-to-back
    in the same block, so they hit L2.
  - No empty programs: 37k for gate/up and 18k for down, against 890k.
  - The column blocks of one expert are launched side by side, so X is fetched from DRAM once.
  - MG = 2 decodes each weight tile once for 32 rows.
- **Knobs.** `GLM53_TF_EXPERT_LOOP=0` restores upstream everywhere (for A/B timing).
  `GLM53_TF_EXPERT_LOOP_CFG=nt,mg` accepts `4,2` (default), `8,1`, `4,1` or `2,4`; none changes bits.
  The extension is renamed `tensorfold_glm_exl3_v2`, so the JIT cache rebuilds.

### Exactness argument (bit-identical to upstream, so guarantees (a), (b) and (c) stand as before)

1. **One item per output.** Every Z element Z[mat][split][pair p][column n] is produced by exactly one work
   item: (the expert of p, the column tile of n, mat, split, the member tile holding p).
2. **Same code on both paths.** Both kernels now run that item through the same device function,
   `member_tiles<NT, W, MG>`; upstream's kernel calls it with MG = 1 and its old NT, and its body is upstream's
   code moved.
3. **Same arithmetic for each accumulator.** For a given warp, row p and n-tile:
   - it starts from +0.0;
   - it runs one `mma.m16n8k16` per k tile over the fixed range `kt0(split, warp) .. + per_warp - 1`, in
     ascending order;
   - B is `decode_tile` of that (expert, k tile, n tile) word, which is integer ops plus one explicit `__hadd2`;
   - A holds row p's own fp16 inputs; other rows of the tile are other members or zeros.
4. **Warp reduction.** The warps' results are then added in shared memory in warp order 0..W-1.
5. **What stays fixed.** `W` and `SK`, which define the K partition and the add order, come from the unchanged
   `GATEUP_CFG`/`DOWN_CFG`.
6. **What varies, and why it does not matter:**
   - which block runs the item, and in what order;
   - how many items a block runs in sequence;
   - NT: which other column tiles share the block, each with its own accumulator and words; `per_warp` does not
     depend on NT;
   - MG: how many member tiles use one decoded B, which is a pure function of the word.
7. **Row independence.** An mma output row depends only on its own A row, B and its own C row. Upstream already
   relies on this: a row's tile-mates and its position in the tile differ between serial steps, verify windows
   and prefill chunks, and its tests check it.
8. **No contraction.** The only floating-point operations in `member_tiles` are the explicit `mma` asm and plain
   adds, so FMA contraction or different register allocation in the new kernel cannot change a result.
9. **No races.** Z writes are disjoint (a pair belongs to one expert, a column to one block, a split to one z)
   and there are no atomics. The epilogues are unchanged.

So every routed pair's output is bit-identical to upstream's at any window size. Serial decode, drafted decode,
resumed and fresh prefills, and any chunking all see the same expert bits as before.

### Expected speedup (to be confirmed with 0005)

- **At PREFILL_ROWS = 512:** expert DRAM weight traffic drops from 1.0-1.9x to 1.0x per layer, and the empty
  programs and X re-reads go away. Estimate ~10-12 ms per layer (8.6 ms of weights plus overheads) against
  14-25 ms: about 250-500 ms saved per chunk. That gives 1.25 s -> 0.8-1.0 s, **~510-640 tok/s (+25-55%)**.
- **The bigger lever:** with 0006, larger chunks finally amortize the experts. At PREFILL_ROWS = 2048 the experts
  cost ~43 x (8.6 + ~5-6 ms of L2 passes, Z, X and rotations) ≈ 0.6 s per 2,048 rows, against ~2-4 s today.
  Everything else scales with rows (~1.2-1.8 s). Estimate **~800-1100 tok/s**, and 1024 rows ~700-900 tok/s.
  Buffers cost about 5 MB per row per rank (2048 rows ≈ 10 GB), so check memory headroom at the context you
  serve.
- **Without 0006**, raising PREFILL_ROWS above 512 cannot help: expert cost is proportional to member tiles, and
  the empty-program count grows with R.
- After 0006 the ranking becomes: non-expert matmuls, then sparse attention at long context, the experts'
  remaining overheads (Z traffic, one-warp rotation and epilogue grids), then all-gathers.

## How to verify on a Spark

Rebuild the image (patches changed), then from the project checkout inside the container:

```bash
PYTHONPATH=/src/TensorFold/tests/cuda pytest -q tests/cuda/test_patches.py -k "profile or expert_loop"
PYTHONPATH=/src/TensorFold/tests/cuda pytest -q tests/cuda/test_patches.py
cd /src/TensorFold && pytest -q tests/cuda/test_glm_exl3.py tests/cuda/test_glm_engine.py tests/cuda/test_glm_kernels.py
```

Timing: serve with `GLM53_TF_PROFILE=1` and prompts of 1.8k / 7k / 28k tokens, for each of:

- `GLM53_TF_EXPERT_LOOP=0` and `1` at PREFILL_ROWS=512;
- loop on at PREFILL_ROWS=1024 and 2048;
- `GLM53_TF_EXPERT_LOOP_CFG` set to `4,2`, `8,1` and `2,4`.

Then check the full-model exactness as before: drafted vs serial SHA-256, and a resumed prompt vs a fresh one.

## patches/0080-0091: fast prefill (canonical chunk grid)

EXPERIMENTS.md's P2, implemented offline (`docs/PATCHES.md` has the details and the estimate table). With
`GLM53_TF_FAST_PREFILL=1` (or `tf_knobs.fast_prefill`), prefill chunks leave the row-invariant kernels: fused EXL3
expert GEMMs without split-K partials (0080), bf16 rank partials, the head on the last row only, and 0081's chunked
KDA scan and large-M matmuls. Chunks sit at absolute multiples of C (`prefill_rows` rounded down to 64) and only the
state at the prompt's last grid point is kept, which keeps drafted == serial and resumed == fresh (proof in
`fastpf.py`). The measured 1024-row profile (`results/M-P1c.profile.log`) puts the routed experts at 12.4-13.8 ms a
layer against a weight-read floor of ~8.4 ms; the ~3.7 ms a layer of Z partial traffic and the one-warp epilogue
grids are what the fused kernels remove. Estimate: 850-1,000 / 650-750 / 580-660 tok/s at 1.8k / 7k / 28k with
expanded KV, ~850-1,000 at 7k and 28k with the latent cache (0060), ~1,000-1,200 with 2048-row chunks. Not timed.

## patches/0082: lean prefill (fast chunks of 4,096-8,192 rows)

Written offline (no GPU). Memory numbers come from building the buffers on the meta device with the real config
(`hidden 4096`, 4 streams, 34 KDA + 11 DSA layers, 288 experts top 8, 2 ranks, latent KV, 5 DFlash2 taps). The
throughput numbers are arithmetic from the measured profile. Nothing here is timed.

**Why.** With fast prefill on 2 Sparks at 2048-row chunks we measured 770 tok/s at 32k and 667 at 128k (vLLM: 1,448 at
28k). A 900-row chunk (8k prompt) took 1,163 ms. Of that, `moe.routed` was 460 ms and ran at about the weight-read
floor: every expert is touched, ~1.8 GB a layer a rank. The only lever on that cost is to put more rows in each
chunk, which needs buffers for the rows. `forward.Buffers` costs ~3 MB a row a set at 1024 rows, and more per row
above that (the latent partials grow with the rows). There are two sets (model and MTP head), plus `State`'s KDA
replay scratch and projection rows.

**Change.** `GLM53_TF_LEAN_PREFILL=1` sizes the window buffers (both sets and `State`'s rows) at
`GLM53_TF_LEAN_BLOCK` rows (default 1024, a multiple of 64). Fast chunks of up to `GLM53_TF_PREFILL_ROWS_MAX` rows run
through `lean.py`:

- Every block runs in row sub-blocks on those buffers: attention (KDA/DSA), projections, shared expert, combine,
  all-gathers and hyper-connections. KDA carries its state and conv window from one sub-block to the next. DSA
  writes each sub-block's keys before it attends.
- Only the MoE routing and the routed experts run once over the whole chunk. This is where each expert's weights
  are read once per chunk.
- `LeanBuffers` keeps the rows that must span the chunk: the streams, the FFN's normed rows and mixes, the routing,
  the fused kernels' fp16 inputs and fp32 outputs, the final-normed rows (MTP absorb) and the DFlash2 taps.
- Nothing is sized [rows x something big] beyond that: no EXL3 split-K `z`, no qmm `sk`, no latent partials, no
  logits, and no KDA replay scratch at the chunk's size.

The bits equal patches/0080's fast chunk of the same rows (argument in `lean.py`; checked on a hash model on the host
and on the GPU by `tests/cuda/test_lean_patches.py`). So GLM53_TF_LEAN_PREFILL changes memory, not replies. The chunk
grid C (= `prefill_rows` rounded to 64) still defines a fast prefill's bits, and snapshots stay on the grid.

### Memory a rank

Every buffer that scales with rows: both `Buffers` sets, `State`'s KDA scratch and projection rows, the `fast_kda`
workspace, and the lean set. Capacity-scaled caches (docs/MEMORY-1M.md) are not included.

| config | window buffers | lean set | total | vs today's 2048-row fast config |
| --- | ---: | ---: | ---: | ---: |
| non-lean, `PREFILL_ROWS_MAX=1024` | 8.50 GiB | - | 8.50 GiB | -9.0 GiB |
| non-lean, `PREFILL_ROWS_MAX=2048` (today) | 17.50 GiB | - | 17.50 GiB | 0 |
| non-lean, 4096 / 8192 rows | 37.0 / 82.0 GiB | - | does not fit | |
| lean, block 1024, P = 2048 | 8.50 GiB | 0.78 GiB | 9.28 GiB | -8.2 GiB |
| lean, block 1024, **P = 4096** | 8.50 GiB | **1.55 GiB** | 10.05 GiB | -7.5 GiB |
| lean, block 1024, **P = 8192** | 8.50 GiB | **3.10 GiB** | 11.60 GiB | -5.9 GiB |
| lean, block 512, P = 4096 / 8192 | 4.19 GiB | 1.55 / 3.10 GiB | 5.74 / 7.29 GiB | -11.8 / -10.2 GiB |

The lean set at P rows is exactly linear, about 397 KiB a row. At P = 8192:

| part | MiB |
| --- | ---: |
| fused expert inputs Xg / Xu / Xd, fp16, [P x 9, 4096 / 4096 / 1024] | 1,296 |
| expert outputs `ey`, fp32, [P, 9, 4096] (what the per-sub-block combine reads) | 1,152 |
| DFlash2 taps, [P, 5 x 4096] bf16 (0 without the drafter) | 320 |
| residual streams, [P, 4 x 4096] bf16 | 256 |
| FFN normed rows, group sums, hyper-connection mixes | 67 |
| final-normed rows (MTP absorb input) | 64 |
| routing (router logits, picks, weights, member lists) | 19 |
| KDA carries (34 x 3 conv rows, one state) | 4 |

Transients are the same as at a 1024-row chunk: the latent sparse partials of one sub-block (320 MiB) and the selection
block (docs/MEMORY-1M.md). So 8,192-row chunks with 1,024-row sub-blocks need ~5.9 GiB less than the 2048-row config
running today. MemAvailable should go from ~12-14 GB to ~18-20 GB a node at the same context.

Still possible, not done:

- Expert-group streaming of Xg/Xu (bounded to ~16k pairs a group; each expert still in exactly one group) would save
  ~1 GiB at 8192 rows. It needs a small `exl3_fast.cu` change: the down kernel writing through a pair map.
- `ey` in bf16 would save 0.56 GiB, but changes bits.

### Expected prefill tok/s (not timed)

Model: a chunk of P rows costs E(P) + P x c(L).

- E is `moe.routed`, 460 ms at 900 rows (measured).
- c is everything else per token at context L. It comes from the 900-row profile at 8k ((1163 - 460) / 900 = 781
  us/token), and from the measured 2048-row rates at 32k and 128k once E(2048) is subtracted.

Two models for E:

- **A: weight-read bound.** E is constant per chunk, as the brief assumes.
- **B: the fused kernels' measured slope.** 14.5 -> 19.7 ms a layer from 1024 to 2048 rows (0080's timing table),
  i.e. +0.21 ms a row over 42 layers, so E(2048 / 4096 / 8192) = 705 / 1,142 / 2,015 ms. Each pass of 64 (gate/up) or
  128 (down) members decodes the expert's trellis tiles again, so the expert cost is not fully fixed per chunk.

Sub-blocks of 1024 rows add ~8 us/token against 2048-row chunks: non-expert weights and the MTP layer's experts are
read once per 1024 rows instead of 2048. That is included below.

| tok/s | 8k | 32k | 128k |
| --- | ---: | ---: | ---: |
| measured, fast, 2048-row chunks | (889-994 by the model) | 770 | 667 |
| lean, P = 4096 (B - A) | 936-1,109 | 806-837 | 694-717 |
| lean, P = 8192 (B - A) | 966-1,183 | 827-879 | 710-747 |
| limit as P -> infinity (A) | 1,280 | 931 | 785 |
| vLLM, same weights | 1,340 (7k) | 1,448 (28k) | - |

**Reading.**

- **Bigger chunks buy +5-15% at 32k-128k.** Model B is the likelier one: +7-8% at 32k, +4-7% at 128k.
- **At 2048-row chunks the experts are already only ~17-25% of a chunk.** The rest scales with rows: 0.8-1.3 ms a
  token, and it grows with context. In the 900-row profile those per-row costs are:

  | component | us/token |
  | --- | ---: |
  | sparse attention | 117 |
  | all-gathers | 106 |
  | `kda.proj` | 94 |
  | `hc` | 77 |
  | `dsa.o_proj` | 76 |
  | `kda.chain` | 63 |

  Even infinitely large chunks stop at 931 tok/s at 32k (model A). Reaching vLLM's 1,448 needs those per-row costs
  roughly halved: sparse attention (grows with context), the all-gathers (overlap: the pipelined half-chunks in
  0080's next steps), and the KDA projections / hyper-connections.
- **The expert kernel is now the lever inside big chunks.** More members per decoded tile (warps a block, or member
  tiles a warp) would move model B toward A, worth up to ~+5% at 8192 rows.
- **Multi-turn cost grows with C.** Under 0080's rule a turn re-prefills the old prompt's tail past the last grid
  point (on average C / 2 tokens) plus the old reply:

  | C | tokens re-prefilled a turn, on average | at ~1,000 tok/s |
  | ---: | ---: | ---: |
  | 2048 | ~1,000 | ~1 s |
  | 8192 | ~4,100 | ~4-5 s |

  A request with another `prefill_rows` cannot resume from the conversation's snapshots (a different grid), so C must
  stay fixed within a conversation. For agent loops that append a few thousand tokens a turn, C = 2048-4096 is
  likely better overall. C = 8192 pays off for long cold prompts (documents, first turns).

**Settings.** `GLM53_TF_FAST_PREFILL=1 GLM53_TF_LEAN_PREFILL=1 GLM53_TF_LEAN_BLOCK=1024 GLM53_TF_PREFILL_ROWS=4096
GLM53_TF_PREFILL_ROWS_MAX=8192` (both ranks). Pick the grid per request with `"tf_knobs": {"prefill_rows": 8192}`.
Exact requests (`fast_prefill: 0`) then run chunks of at most 1024 rows (same bits as any chunk size). For A/B,
`GLM53_TF_PROFILE=1` works unchanged (the same component names, laps per sub-block).

**Tests.**

```bash
PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda python -m pytest -q tests/cuda/test_lean_patches.py
```

The host-only part runs without a GPU: 29 tests, including the hash model and the fake-model rule. The GPU part has 13 more.

## patches/0084: the pipelined lean chunk (all-gathers hidden, hyper-connections in L2)

Written offline, nothing timed. This attacks two of the per-row costs above: all-gathers (106 us/token) and
hyper-connections (77). `docs/PATCHES.md` ("0084") has the design and the exactness argument. In short: same kernels,
same inputs, same collectives in the same order, so the same bits as 0082.

Where the 106 us of `allgather` went, per row and exchange site:
- `gather`'s two conversion copies: fp32 -> bf16 before (read 16 KB, write 8 KB) and bf16 -> fp32 after (read 16 KB,
  write 32 KB). That is 72 KB, and hc_post then read 32 KB of fp32. At 90 sites a token this is ~34 us/token of
  memory traffic that was lapped as `allgather`.
- The rest, ~70 us/token, is NCCL: about 0.8 ms a 1024-row sub-block a site, ~10 GB/s effective on the 25 GB/s link,
  plus the other rank's skew.

The pipeline removes the copies (partials written and read in the exchanged dtype). It runs exchange k on a comm
stream while sub-block k+1 does its pre work: 3-5 ms of projections / KDA / attention for an attention piece, ~1 ms of
shared expert + combine for a MoE piece. Exposed are:
- the MoE boundary: the routing needs every row, so the layer's last attention exchange overlaps only the previous
  sub-block's hc work;
- NCCL's blocks taking a few of the 48 SMs from the compute kernels;
- skew.

The hyper-connections move from DRAM to L2 for hc_pre's two reads of the rows hc_post just wrote: slabs of ~384 rows
(12 MB of streams).

| per token and rank | today (2048-row chunks, 32k) | with 0084 (estimate) |
| --- | ---: | ---: |
| all-gathers (lap) | 106 us | 10-30 us (the waits left) |
| hyper-connections (lap) | 77 us | 55-65 us |
| conversions moved out / NCCL SM contention | - | +10-25 us spread over the compute laps |
| total change | | -90 to -120 us |
| prefill tok/s at 32k, 2048-row chunks | 770 | ~830-850 |
| prefill tok/s at 32k, 8192-row lean chunks | 830-880 (est.) | ~910-970 |

What is left to reach vLLM's ~690 us/token is mostly sparse attention (117), `kda.proj` (94), `dsa.o_proj` (76) and
`kda.chain` (63).
