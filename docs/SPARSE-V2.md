# Sparse latent attention v2: patch 0410 (`GLM53_TF_SPARSE_V2=1`)

Offline work (2026-09-28). No GPU was used and production was not touched. Everything below that needs a GPU is in
the test plan (section 7) and has not run yet.

**Summary.**

- ROOFLINE gap 2 is the DSA sparse latent attention. Since W9 production's fast prefill runs it through the
  **one-pass kernel** (b12x bit 4, `b12x_attn._lsparse_one`, `GLM53_TF_B12X=4`): 5.0-5.4 ms per 1,024 rows x 2,051
  tokens, 1.71-1.81x faster than the chunked kernel, +6.3% / +6.7% prefill at 24.5k / 98k with the same reply sha
  (RESULTS W9 section 4). **0410 builds on that path**: it reproduces the one-pass kernel's bits and makes it faster.
- Patch 0410 adds `sparse_v2._lsparse_v2`, a **Gluon** kernel (Triton's explicit-layout dialect, in the image's
  Triton 3.7.1; checked on 3.8 too). With `GLM53_TF_SPARSE_V2=1` (load-time, default off),
  `b12x_attn.sparse_latent_one` runs it instead of `_lsparse_one`.
- **Same bits, per row, row-invariant.** Each row keeps its own tile sequence and online-softmax order. Every mma
  output keeps its k16 chain and operand placement. Both reductions run on the reference's own `#mma` layout. Every
  element-wise op is the same op on the same value.
- **Only data movement and scheduling change:**
  - pipelined `cp.async` gathers of the selected FP8 rows into a 3-slot ring, through the 0290 page table;
  - the FP8 tile is dequantized once into shared memory;
  - the scores are computed without the reference's duplicated warps (512 instead of 768 mma a tile);
  - half of q stays in registers for the whole row;
  - 33% less shared-memory traffic per tile.
- **Offline evidence** (section 5):
  - compiled IR: the same arithmetic op sequence on the same layouts, the same chains, the same PTX float
    instructions, under 3.7.1 and 3.8;
  - a CPU emulator runs v2's own source against the reference in Triton's interpreter, bit for bit, with its
    cp.async ring, wait groups and barriers modelled;
  - the reference kernel's PTX is unchanged by the patch.
- **Estimate** (section 6):
  - the kernel: 1.8-3.1x the one-pass kernel;
  - the stage: 53.8 -> ~18-30 us a token;
  - prefill: **+3.5-6%**, i.e. ~1,566 -> ~1,620-1,665 tok/s at 24.5k, and similar at 98k.
  The GPU bench decides where in that range it lands.

## 1. The kernel today

Two sparse kernels exist for fast-prefill rows past the dense limit (2,051 tokens). Each attends a row's 2,051
selected tokens, the DSA top-k: 512 pools x 4 tokens + 3.

| kernel | used by | a 512-row sub-block (FP8, W7 / W9) | TFLOP/s |
| --- | --- | ---: | ---: |
| chunked: `latent._lsparse_chunks` (grid R x 1 x 5, 512-token chunks, BM 32) + `_lsparse_merge` | production until W9; b12x bit 4 off | 3,693 + 883 us = **4.58 ms** | 15 |
| one pass: `b12x_attn._lsparse_one` (grid R, all 32 heads a program, one online softmax) | **production since W9** (`GLM53_TF_B12X=4`) | 2.50-2.70 ms (5.01-5.40 ms per 1,024 rows) | 25.5-27.5 |

A row is 32 heads x 2,051 keys x 512 x 4 = 134 MFLOP, so a 1,024-row call is 137.6 GFLOP. At the 110 TFLOP/s
`mma.sync` roof that is **1.25 ms**; the one-pass kernel takes 4x that.

**Heads-as-rows: already.** One program holds all 32 local heads of its row as the 32 M rows of the tensor-core
tile. Each gathered latent row serves all 32 heads, as keys and as values.

**What `_lsparse_one` compiles to** (Triton 3.7.1, sm_121; TTGIR + PTX + ptxas, `tests/test_sparse_v2_compile.py -s`):

- **Layout.**
  - Both dots and the softmax sit on `#mma = nvidia_mma<{2.0, warpsPerCTA [1, 8], instrShape [16, 8]}>`. Triton's
    chained-dot heuristic gives the QK dot the PV dot's layout.
  - The QK tile is [32 heads, 32 keys], but 8 warps x n8 = 64 columns. So **warps 4-7 recompute warps 0-3's scores**,
    and every warp reads all of q.
- **Per 32-key tile and CTA:**
  - **768 mma** (256 of them duplicates);
  - **~430 KB of shared-memory traffic**: 387 KB of it `ldmatrix`, of which q alone is 256 KB, plus a 32 KB store of
    the dequantized tile;
  - **nothing in flight**: the token ids, then the 16.9 KB of rows (`ld.global`, 16 B a thread), then use.
- **Occupancy.** 178 registers and 66 KB of shared memory: one 8-warp CTA an SM. No other CTA hides the two dependent
  round trips.
- **Measured.** 5.01 ms x 2.229 GHz / (1,024 rows x 64.1 tiles / 48 SMs) = **~8,170 clocks a tile**.
- **Static bound.** At 4 clocks an m16n8k16 an SM (1,024 bf16 FLOP/clk) and 128 B/clk of shared memory:
  - MMA: 3,072 clocks;
  - shared memory: 3,360 clocks.

  So ~4,800 clocks a tile are exposed latency. That matches W7's finding that the time does not move with context:
  it is latency-bound, not bandwidth-bound.

## 2. Design (`sparse_v2.py`)

A program is still one row with its 32 heads (grid (R,)): one online softmax over the row's list in KT = 32-token
tiles, the same tiles in the same order. Per tile:

```
wait for tile t's copies; barrier                       (tile landed everywhere; iteration t-1 done by every warp)
issue tile t+S-1's gathers (cp.async, 16 B, masked lanes zero-filled) into slot (t+S-1) % S; commit
load tile t+S's token ids (+ page-table lookup)         (plain loads, used next iteration)
barrier                                                  (before every operand load: see below)
raw FP8 tile (slot t % S, 16 B a thread) -> e4m3 -> f32 x 2^k -> bf16 -> bf16 tile in smem; barrier
scores = mma chain(q, tile^T) on [2, 4]                 (QREG: K 0-255 of q from registers, 256-511 from smem)
scores -> [1, 8] through the consumed raw slot; barrier
latent._ltile's softmax ops on #mma [1, 8]              (max, next_m, alpha, p; same ops, same order)
oa = o * alpha;  p (bf16) -> dot operand through the consumed slot; barrier
o = mma chain(p, tile, c = oa) on [1, 8];  l = l * alpha + sum(p)   (the sum on [1, 8])
end: o / l -> OUT
```

Settings (`GLM53_TF_SPARSE_V2_CFG=stages,qkl,qreg`; speed only, **every setting gives the same bits**, all tested):

| setting | shared memory | registers (3.7.1 / 3.8) | mma a tile | smem traffic a tile | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| reference `_lsparse_one` | 67,584 | 178 / 175 | 768 | ~430 KB | no pipelining |
| **3,1,1 (default FP8)**: 3 slots, QK on [2, 4], half of q in registers | 99,200 | 252-254 / 246-250 | 512 | ~286 KB | 2 tiles in flight |
| 2,1,1 | 82,688 | 254 / 252-255 | 512 | ~286 KB | 1 tile in flight |
| 2,1,0: q in smem | 99,072 | 172-174 / 172-178 | 512 | ~352 KB | |
| 2,0,0: QK on [1, 8] as the reference | 100,608 | 174-178 / 178 | 768 | ~475 KB | the pipelining alone |
| bf16 cache (default 2,0,0) | 100,352 | 170-174 / 176-184 | 768 | | a slot is the operand |

- GB10 has 101,376 B of shared memory a block. `config()` rejects settings that do not fit: 3 slots with q in shared
  memory need 115 KB.
- No setting spills (ptxas from each Triton).
- One 8-warp CTA an SM, as the reference.

**Barriers.** Five explicit barriers a tile, plus those of the two reductions:

1. After the wait (RAW on the landed slot; WAR on the slot about to be refilled and on the bf16 tile).
2. After the new copies.
   - This one is not needed for a hazard of its own. Triton's barrier analysis cannot tell ring slots apart (the
     slot index is dynamic), so it inserts one between the new copies and the first ring read anyway.
   - If Triton places it, it lands after q's operand loads. That leaves 256 registers live, and the bf16 kernel
     spilled 472 B.
   - Placed explicitly before every operand load, it costs nothing extra.
3. bf16 tile written.
4. Scores moved.
5. p moved.

The emulator's hazard model shows that barriers 1 and 3-5 each guard a real hazard: dropping any of them fails.

## 3. What decides the bits, and why v2 keeps them

1. **mma chains.**
   - Triton's MMAv2 lowering (`MMAv2.cpp`, 3.7.1 and 3.8) emits `for k: for m: for n: mma(..., c = fc[m, n])`. Each
     output element is therefore the chain of k16 steps over K in ascending order from the dot's c. The warp layout
     only decides which warp holds the element.
   - The K placement inside each m16n8k16 comes from `kWidth`. It is 2 on every dot operand of both kernels. That
     matters: 0360's W3 failure was `kWidth` 4.
   - QK starts from a zero constant in both kernels. v2 with QREG runs two dots, the second continuing the first's
     result: the same 32 steps.
   - PV starts from `o * alpha` in both. This is Triton's Combine fold of `o * alpha + dot`; v2 writes it that way.
2. **Reductions.**
   - `max` is exact in any order.
   - `sum` is not associative. `ReduceOpToLLVM` builds its tree (in-thread, lane butterfly, cross-warp) from the
     source layout, and the tree code changed between 3.7.1 and 3.8.
   - v2 therefore reduces on **the reference's own `#mma` [1, 8]**, not on [2, 4], so the tree is the reference's
     under any Triton. The scores cost one 4 KB move to get there.
3. **Element-wise ops.** Scale, selects, `maxnumf`, `exp`, the rescale, `l`'s mul + add (LLVM contracts the pair the
   same way: the PTX float counts match), `divf`, and the dequantization are the same ops on the same values in the
   same program order.
4. **Masked keys.** The reference loads zeros (`other=0`). v2's `cp.async` has a src-size operand of 0 for masked
   lanes, which zero-fills. The scale is zero too, so the dequantized value is +0 in both.

Row independence is unchanged: grid (R,), and a program reads only its row. The tile sequence depends on n, never on
R or the other rows. v2 has the one-pass kernel's bits, so it shares bit 4's snapshots and needs no new tag and no
rank check. `GLM53_TF_SPARSE_V2*` is left out of the NVMe compat hash, as 0400's `GLM53_TF_KDA_V2` is.

## 4. Sharing gathers between neighbouring rows: evaluated, not done

Adjacent query rows' top-2,051 sets overlap heavily (~97% in the bench's model).

**Could a row's gathered keys serve another row?**

- With bit identity, each row must still visit its own tiles: its own sorted list, cut at multiples of 32.
- A shared insertion shifts every later tile boundary. So two rows' tiles are equal only by coincidence.
- Sharing would need a union window in shared memory that both rows index into.
- Two rows a CTA do not fit GB10:
  - q for two rows is 64 KB;
  - a ~48-token union window is 48 KB of bf16;
  - the two o accumulators alone are 128 registers a thread.
- Splitting the heads would lose the per-row gather sharing across heads, which is the bigger win.

**What it would save.** Only SM<-L2 gather traffic: 16.9 KB a tile, ~6% of v2's shared-memory traffic. DRAM is
already deduplicated by L2, because the 48 SMs run 48 consecutive rows at once and their lists overlap: a new row
brings ~64 new tokens = 34 KB.

**Group-shared keys with a different tile order** (ROOFLINE's "second step") would be new arithmetic, needing a new
snapshot tag and the quality gate. That is out of scope for a same-bits patch.

## 5. Offline evidence

The tree is vendor/TensorFold at 2f8e514 plus every patch 0001-0400 plus 0410.

1. **`tests/test_sparse_v2_compile.py`** compiles for sm_121 without a GPU. It passes under 3.7.1 and 3.8, 36 tests.
   - **Layouts.**
     - The reference has exactly one `#mma`, [1, 8].
     - v2 contains that definition verbatim, plus [2, 4] for QK when QKL is 1.
     - Every operand is `kWidth = 2`.
   - **Arithmetic.** From the key loop to the end of the kernel, the float op sequence with resolved types is
     identical:
     - `fp_to_fp`, `mulf`, `truncf`;
     - the QK dot, compared by kWidth and total K;
     - `mulf`, `select`, `reduce(max)` on [1, 8], `cmpf`, `maxnumf`, `select`, `subf`, `exp`, and so on;
     - `mulf(o * alpha)`, `truncf(p)`, the PV dot, `mulf`, `reduce(sum)` on [1, 8], `addf`, `divf`.

     This holds for FP8 and bf16, contiguous and paged, and every setting.
   - **Chains.** QK's c is a zero constant; with QREG, the second QK dot's c is the first dot's result. PV's c is an
     `arith.mulf` of the carried o.
   - **PTX.**
     - Every float instruction class (fma/mul/add/sub/div/ex2/max/cvt/selp/setp, packed forms included) has
       **equal counts** in v2 and the reference.
     - mma is 64 (QKL 1) or 96 (QKL 0), against the reference's 96.
     - There is no `.rz` / `.ftz`.
     - The gathers are `cp.async.cg ... 16, src_size` (zero-fill).
   - **Fit.** Shared memory equals `smem_need()`; the default configurations do not spill.
   - The dispatch and knob parsing are tested too.
2. **`tests/test_sparse_v2_emulator.py`** runs on CPU (`TRITON_INTERPRET=1`), 25 tests.
   - **Method.**
     - v2's own Gluon source runs on `tests/sparse_v2_emu.py`, a numpy model of the Gluon ops it uses.
     - cp.async groups land at `wait_group`, and shared memory is modelled as bytes, with the slot reinterpretations
       aliasing as on the GPU.
     - A barrier / hazard model raises on:
       - a read of bytes in flight;
       - a read of bytes written since the last barrier;
       - a write to bytes read since the last barrier.
     - The reference runs in Triton's interpreter with the same arithmetic model:
       - bf16 nearest-even;
       - dots as an order-sensitive k16-block chain from c;
       - `_ltile`'s accumulator folded as on the GPU.
   - **Bit for bit, v2 == `_lsparse_one`:**
     - FP8 and bf16;
     - settings 3,1,1 / 2,1,1 / 2,1,0 / 2,0,0 / 4,1,1;
     - counts 0, 1, 31, 32, 33, 64, 65, 137, 300, and 2,051 / 2,050 / 1,900 / 1,024 / 5 at contexts 2,100 and 6,000;
     - paged pools (shuffled pages, NaN junk around) == the reference == contiguous;
     - rows alone, in subsets and with duplicates == the full launch;
     - FP8 == bf16 on the dequantized rows.
   - **Controls.**
     - An unfolded reference differs.
     - The chunked kernel differs.
     - v2 is within 1e-2 of float64.
     - A missing wait and each missing barrier raise.
     - Mutating the source fails the test: e.g. token ids a tile early, or the wrong ring slot.
3. **The reference is unchanged.** `_lsparse_one`'s PTX, debug info stripped, is identical before and after the
   patch: FP8 / bf16, contiguous / paged, hashes `eb58dc45c41f78ee`, `6c9277eb258fa8ad`, `98fe8ab56f3246eb`,
   `61fae683a8e8771f`. So with the knob off, production's bits are production's.
4. The existing 0360 tests still pass on the patched tree: `test_b12x_onepass_compile.py` 6/6 and
   `test_b12x_onepass_interpreter.py` 10/10.
5. **The GPU test's own code was dry-run on CPU.** `tests/cuda/test_sparse_v2_patches.py` was run with `cuda` -> `cpu`,
   v2 through the emulator and the reference in the interpreter. Seven of its cases passed:
   - paged FP8 / bf16;
   - FP8 == dequantized;
   - both extreme-logit cases;
   - the default setting at 2,051 tokens.

   So its input builders, the paged pool and its assertions work as written.

**Not proven offline:** that ptxas and the hardware treat the same PTX arithmetic identically in both kernels. They
do not reorder `mma` chains or IEEE ops, but only the GPU bitwise test (section 7, step 1) settles it.

## 6. Estimate

Per tile and CTA, from the loop PTX, at 4 clk an mma and 128 B/clk of shared memory:

| | mma | MMA clocks | smem traffic | smem clocks | static bound | measured |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reference | 768 | 3,072 | ~430 KB | ~3,360 | 3,360 | **~8,170** (W9 bench) |
| v2 3,1,1 | 512 | 2,048 | ~286 KB | ~2,240 | 2,240 | - |

- **Optimistic: 2,630-3,200 clocks a tile, 2.6-3.1x.** The latency is hidden (two tiles in flight), and v2 reaches
  70-85% of its binding pipe.
- **Conservative: ~4,500 clocks, 1.8x.** MMA and shared memory do not overlap at all (2,048 + 2,240), plus the barriers.
- **Where the time lands.** 1,024 rows: 5.01 ms -> 1.6-2.8 ms. A row and layer: 4.9 -> 1.6-2.7 us.
- **Saving a prefill token (11 DSA layers): 24-36 us** of ~638 (1,566 tok/s at 24.5k, W9). That is
  **+3.9-6.0%**, i.e. ~1,630-1,665 tok/s. Taking a margin for L2 misses gives the "+3.5-6%" above.
- **98k.** The same, since the stage does not depend on context. If some 98k rows missed L2, v2's in-flight gathers
  would hide more of it than today's kernel does.
- **Bandwidth check.**
  - At ~3,000 clocks a tile, 48 SMs request ~600 GB/s of rows, far above DRAM's 230 GB/s. This works only because L2
    serves the overlap of neighbouring rows' lists: W7's flat context curve says it does today.
  - With independent random lists over a cache larger than L2 (the bench's `--random` at 85.8k), both kernels meet
    the same DRAM floor: 1.08 MB a row at 230 GB/s = 4.7 us.
- **Decode / verify / MTP** keep the chunked kernel, so they are unaffected.

## 7. GPU test plan (one window, ~35-45 min, production down ~25)

Build `glm53-tensorfold:sv2` from the committed tree: production's patches plus 0410. The image and Triton (3.7.1)
are unchanged otherwise, and no build step is added: Gluon ships with Triton. Keep production's image for the revert.
Everything runs under `timeout`.

1. **Bitwise, one GPU** (production stopped, or on the worker node while the head node serves), ~2 min.
   - `timeout 900 pytest -q -s tests/cuda/test_sparse_v2_patches.py`
   - Covers v2 == one pass, bit for bit:
     - FP8 / bf16, contiguous / paged (NaN junk), all four settings, 3 seeds;
     - random and neighbour-overlapping lists; counts 0 .. 2,051;
     - rows of 1 .. 2,048-row launches; subsets / permutations / duplicates; deterministic;
     - extreme logits, signed zeros, zero rows;
     - FP8 == dequantized;
     - the knob's dispatch.
   - **Gate: all green.** One False means an arithmetic difference: keep the knob off and save the case.
   - Also rerun `pytest -q tests/cuda/test_b12x_attn_patches.py -k "one_pass or fp8"` (the reference unchanged).
2. **Microbench**, same GPU, ~5 min.
   - `timeout 1800 python tests/cuda/bench_sparse_v2.py 512 2048 8192`: contexts 10,700 and 85,800, overlapping lists.
   - Then the same with `--random`, and once with `--bf16`.
   - Read, per line:
     - chunked / one pass / each v2 setting in ms and TF/s;
     - `same bits True` on **every** line;
     - the best `GLM53_TF_SPARSE_V2_CFG` and its projected tok/s.
   - If another setting beats the default (3,1,1) by more than 3%, use it in step 3 (same bits by construction;
     step 1 covered it).
   - **Gate:** v2 <= 0.7x the one-pass time at 2,048 and 8,192 rows, 10.7k context (expected 0.35-0.55x).
3. **Load V: production + `GLM53_TF_SPARSE_V2=1`** (+ `GLM53_TF_SPARSE_V2_CFG` if step 2 chose one).
   - Start: `results/W9/load.sh V IMAGE=glm53-tensorfold:sv2 GLM53_TF_SPARSE_V2=1` (copied to `results/W10/`).
   - The boot log must show `sparse latent attention v2 for b12x bit 4 ... (patches/0410)` on rank 0, and no
     compile error on either rank.
   - Run W8's gate set (`run.sh V '{}'`):
     - exact 10/10;
     - batchexact 4/4;
     - `results/W5/ab.py` 24.5k / 98k twice;
     - concurrent 1 / 4 streams.
   - Gates:
     - **reply sha `8794a3463259cc2f`** at 24.5k and 98k (production's, with bit 4);
     - **exact 10/10**;
     - batchexact 4/4;
     - **prefill 24.5k / 98k >= +3%** over production's same-window numbers (W9 bit 4: 1,561-1,566 / 1,546-1,548
       tok/s);
     - decode unchanged.
   - **Sessions / resumed == fresh.** A 60k-token session follow-up (W9's `b12x_sessions.py`) must give `cached` > 0
     and the cold reply sha.
     - This also shows v2 resumes snapshots made by the one-pass kernel. Resume on load V a session saved on
       production's load: the NVMe tier's compat hash leaves `GLM53_TF_SPARSE_V2*` out, so the entry is visible.
   - **Control:** the same image with the knob off (`load.sh V0 IMAGE=glm53-tensorfold:sv2`) must equal production
     (PTX identical, section 5.3).
4. **Optional profile.** `GLM53_TF_PROFILE=1` on load V, one 24.5k prefill: `dsa.sparse_attn` a sub-block should drop
   from ~2.6 ms to ~0.9-1.4 ms.
5. **Adopt or revert.**
   - Adopt: `IMAGE=glm53-tensorfold:sv2`, `GLM53_TF_SPARSE_V2=1` (and the CFG, if any) in `config/prod.env`, with a
     RESULTS entry.
   - Otherwise production's config.
   - Either way: canary, https check, watchdog re-armed.

## 8. Files

- `patches/0410-glm-sparse-v2.patch`:
  - `sparse_v2.py` (new: the knob, `config`, `smem_need`, `_rows`, `_issue`, `_lsparse_v2`, `sparse_latent_v2`);
  - `b12x_attn.py`: `V2`, `enable_v2`, the dispatch in `sparse_latent_one` (`v2=` override);
  - `engine.py`: parse and print at load;
  - `sessdisk.py`: `GLM53_TF_SPARSE_V2` left out of the compat hash.
- Tests:
  - `tests/test_sparse_v2_compile.py`: compile, IR, PTX, fit;
  - `tests/test_sparse_v2_emulator.py` + `tests/sparse_v2_emu.py`: CPU emulator vs the interpreter;
  - `tests/cuda/test_sparse_v2_patches.py`: GPU bitwise;
  - `tests/cuda/bench_sparse_v2.py`: microbench.

## 9. Risks

- **Gluon is experimental.** Its API moved between 3.7 and 3.8: `mma_v2` and `async_copy` are unchanged, and the
  kernel compiles on both. A Triton upgrade must rerun the compile test, which compares against the reference as
  compiled by that Triton. A failure to import or compile Gluon fails the load (`enable_v2` imports at load), not a
  request.
- **Registers.** The default is at 246-254 registers. A different ptxas could spill: slower, same bits. Setting
  `2,1,0` (172 registers) is the fallback.
- **Rows past the head count or latent width** other than 32 x 512 fall back to the reference (checked in the
  dispatch). The MTP head, decode and verify never use either one-pass kernel.
- **v2 exists only for b12x bit 4.** When this was written, W9's window 3 (combined candidate F) was still running:
  RESULTS W9 has bit 4 adopted from window 2, and the F gates were pending. If bit 4 were ever reverted, v2 would
  serve only requests that ask for `tf_knobs.b12x` 4. The chunked kernel (bit 4 off) has other bits. A fused v2 of it
  is possible with the same method (per-chunk state reset every 16 tiles and `_lsparse_merge`'s element-wise merge in
  chunk order: the merge has no reductions), but it is not built.
- **If the bitwise test fails**, the difference is in hardware / ptxas treatment the offline checks cannot see. The
  knob stays off. The emulator and IR tests then point at data movement rather than arithmetic.
