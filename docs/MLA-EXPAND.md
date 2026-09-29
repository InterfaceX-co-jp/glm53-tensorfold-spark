# MLA latent expand / absorb on GB10: patch 0390 (`GLM53_TF_MLA_EXPAND=v2`)

Offline work (2026-09-28). No GPU was used and production was not touched. Everything below that needs a GPU is in
the test plan (section 6) and has not run yet.

**Summary.**

- Patch 0390 adds a retiled copy of the latent MLA **absorb** kernel (q' = W_k^T q) and **expand** kernel (o = W_v u).
  It is switched by a load-time knob, `GLM53_TF_MLA_EXPAND=v2`. Default `v1` is today's kernels.
- **The arithmetic is unchanged.** Each output element is the same single fp32 FMA chain as today:
  - same k order, same starting value +0.0;
  - same exactly dequantized weights;
  - same round-to-nearest-even bf16 output.
- **Only the tiling changes:**
  - a program handles 16 to 128 rows (today: 16), so the kv_b tile is dequantized once for all of them;
  - the dot steps are 16-wide (today: 64-wide, which makes expand spill);
  - the programs of one head are adjacent in the grid;
  - there are no spills.
- **So v2 should give v1's bits for every row, at any launch size.** Prefill, decode, verify and MTP steps can then
  share it, and snapshots, sessions and the exactness gates carry over unchanged.
- **Offline evidence** (section 4):
  - Triton's source and our compiled IR show that each output of both kernels is one FMA chain;
  - an interpreter model of that chain gives the same bits for v1 and v2 at every tested row count and tile;
  - v1's PTX is byte-identical before and after the patch.
- **Expected gain** (arithmetic, not timed): `_expand` 2,415 -> ~250-450 us and `_absorb` 676 -> ~200-300 us per
  512-row sub-block. That is **~50-57 us a token, about +7.5-8.5% prefill** (1,471 -> ~1,590-1,600 tok/s at 24.5k),
  plus a little decode.

## 1. The current kernels (v1: `latent._absorb`, `latent._expand`, patches/0060)

Shapes per rank: 32 local heads, head dim 256 (query and value), latent 512. kv_b is `q4mse` in production, a
`qmm.Q4` matrix of 4-bit groups of 64 along K with bf16 scale and bias per group and row.

- the key half, `kv_k`, is [8,192, 512];
- the value half, `kv_v`, is [8,192, 512].

| kernel | computes | grid (R rows) | a program |
| --- | --- | --- | --- |
| `_absorb` | QA[r, h, j] = bf16(sum_i q[r, h, i] W_k[256 h + i, j]), i = 0..255 | (R/16, 32 heads, 8 latent groups) | 16 rows x 64 latent columns; 4 steps of `acc = acc + tl.dot(x[16, 64], w[64, 64], ieee)` with the 64 x 64 kv_b tile dequantized in fp32 |
| `_expand` | OUT[r, n] = bf16(sum_j u[r, h(n), j] W_v[n, j]), j = 0..511 | (R/16, 128 column tiles) | 16 rows x 64 outputs; 8 steps of the same |

**Where they run.** `latent._project` calls `absorb`, and `latent._output` calls `expand`. Both are called for every
DSA layer, and for the MTP head's DSA layer, in:

- exact and fast prefill chunks;
- decode, verify and MTP draft steps (eager and inside CUDA graphs);
- batched rounds (`dsa_multi`).

Only FP8 fast chunks (0083) and `GLM53_TF_LATENT_TC=1` (0190, off: it changes replies) use the tensor-core
`absorb_tc` / `expand_tc` instead. **So prefill and decode/verify share these kernels.** Any change to their bits would
move every reply, which is why v2 must keep them.

**What Triton makes of them.** Checked on Triton 3.7.1 (the image's) and 3.8, sm_121:

1. **The accumulator is folded into the dot.**
   - Triton's Combine pass (`lib/Dialect/Triton/Transforms/Combine.cpp`, `CombineDotAddPattern`) rewrites
     `acc + tl.dot(x, w)` into `tl.dot(x, w, acc)`. It applies when the dot's own accumulator is zero, the dot has
     one use, and there is no imprecise-accumulation limit.
   - In our TTIR, the loop's single `tt.dot` takes the `scf.for` iter_arg as its `c` operand and is yielded directly.
     There is no fp32 add between the 64-wide groups.
2. **An `ieee` fp32 dot is an FMA chain.**
   - It never uses tensor cores: the TTGIR operands are `dot_op` encodings of a `blocked` parent, not `mma`.
   - It is lowered by `FMADotUtility.cpp` (`parametricConvertFMADot` + `GenericFMAVectorMultiplier`). For each
     output element, `accum = c; for k in 0..K-1: accum = llvm.fmuladd(a[k], b[k], accum)`.
   - That loop is over the dot's K in order, from its `c` operand. It does not depend on the tile shape, the layout
     or the warp count.
   - The LLIR shows one linear chain of 64 `fmuladd.v8f32` per group. In PTX, every fp32 operation is `fma.rn.f32`
     or `fma.rn.f32x2` (per-lane IEEE fma) or a conversion. There is no `add.f32` and no `mul.f32`.
3. **The weights are exact.**
   - `q * s + b` compiles to one `fma.rn.f32`.
   - The product q * s is exact anyway: q is below 16 (4 bits) and s is bf16 (8 significant bits). So "fused or not"
     gives one value.
   - The bf16 kv_b path is an exact widening.
4. **Output.** `acc.to(bf16)` is `cvt.rn.bf16.f32`.

So each output element is **one fp32 FMA chain over all K = 256 (absorb) or 512 (expand) inputs in ascending order,
from +0.0, then one round to bf16.** That is the specification v2 must meet.

**Why they are slow** (ptxas for sm_121a from Triton 3.7.1; `tests/test_mla_expand_compile.py`; W7 times):

| kernel | registers | spill | warps a CTA | FMA instructions in the loop | W7 time a 512-row call | TFLOP/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `_expand` (q4) | 255 | **452 B** | 4 | 18% of issue | **2,415 us** | 1.8 |
| `_absorb` (q4) | 96 | 0 | 4 | 20% of issue | 676 us | 6.4 |

- **The 64-deep dot is the main problem.** Triton's FMA lowering keeps a thread's whole A and B slices (all 64 k) in
  registers. With `_expand`'s fp32 u operand that needs 255 registers and still spills 452 bytes.
  - That leaves 8 warps an SM, and every k step waits on local memory.
- **Dequantization dominates the loop.** The 64 x 64 kv_b tile is dequantized again for every 16 rows:
  - 32 dequantizing FMAs a thread per 512 dot FMAs;
  - plus the word and scale loads, the shared-memory round trip of both operands and the barriers;
  - so FMAs are under a fifth of the issued instructions.
- `_absorb` has no spills and sits at about its issue bound: 20% of the 30.7 TFLOP/s FMA peak = 6.1. It is slow for
  the second reason only.

## 2. The new kernels (v2: `latent._absorb2`, `latent._expand2`)

v2 has the same functions (`latent.absorb`, `latent.expand`) and the same contract. With `EXPAND_V2` they launch the
new kernels.

```
_expand2: program (BN output rows n0.. of head h = n0 // 256, row tile of BMR rows)
  acc[BMR, BN] = 0
  for g in 0..7:                       # 64-wide latent groups, ascending
    for kk in 0..64/BK-1:              # BK-wide steps inside the group, ascending (static)
      x  = u[rows, h, 64 g + BK kk ..]                 fp32 [BMR, BK]
      wt = W_v[n0.., 64 g + BK kk ..] dequantized       fp32 [BN, BK]  (q * s + b, exactly _wtile's values)
      acc = tl.dot(x, wt^T, acc, input_precision="ieee")   # continues the chain
  OUT[rows, n0..] = bf16(acc)
_absorb2: program (head h and latent group g, row tile): the same over i = 0..255 in BK-wide steps
```

- **Same chain.** Each `tl.dot` continues the accumulator, and its FMA lowering walks its BK inputs in order. So an
  element's chain is the 256 or 512 FMAs in the same order from +0.0, however K is cut into dots and however M and N
  are tiled. The weights are `_wtile`'s values, loaded as a slab (`_wcols`, `_wrows`); the output rounding is the same.
- **Tiles.** The rows a program and the warps come from `latent.V2_TILES` by launch rows; BK = 16 (Triton 3.7 needs
  K >= 16 a dot). All fit without spills:

  | launch rows | expand tile | absorb tile | registers (3.7.1) |
  | --- | --- | --- | --- |
  | <= 16 (decode, verify, MTP) | 16 rows, 8 warps | 16 rows, 8 warps | 76 / 48 |
  | <= 32 | 32 rows, 8 warps | 32 rows, 8 warps | - |
  | <= 64 | 64 rows, 8 warps | 64 rows, 8 warps | - |
  | > 64 (prefill) | 64 rows, 8 warps (BN 64) | 128 rows, 8 warps | 128 / 128: 2 CTAs, 16 warps an SM |

  The table and `V2_BN` are speed knobs only; the tests show every entry gives the same bits.
  `tests/cuda/bench_mla_expand.py --sweep` measures the alternatives on the GPU and prints the best table.
- **Dequantization amortized.** A kv_b slab is dequantized once for 64-128 rows instead of 16 (4-8x less), 2 values
  a thread a step. At the prefill tiles the loop is **32-39% FMA instructions** (v1: 18-20%). Nearly all of them are
  packed `fma.rn.f32x2`, so FMAs are **64-77% of the issue slots**.
- **Grid order.** The head's programs are adjacent: expand's 4 column tiles of a head, absorb's 8 latent groups.
  They run together and read the same u / q rows from L2, not DRAM. u is 32 MB fp32 per 512 rows, so this matters.
- **No new state, memory or build step.** Triton kernels in `latent.py`, compiled on first use like v1. The decode
  graphs capture them after their eager warm-up step, as today.

**The knob.**

- `GLM53_TF_MLA_EXPAND=v1|v2`, read at import on each rank. Default `v1`; `0` / `off` / empty also mean v1, `1`
  means v2, anything else is an error.
- With v2, each rank prints `[tensorfold] latent absorb / expand: v2 kernels (patches/0390 ...)` at load.
- It is not a per-request knob, since there is nothing to choose if the bits are equal. There is no snapshot tag and
  no rank-agreement check, because the ranks' outputs do not depend on it. If the GPU test ever showed different
  bits, the knob must stay off; it would not be "new arithmetic".

## 3. Roofline estimate (per rank, per 512-row sub-block and DSA layer)

| | FLOPs | bytes (min) | FMA roof (30.7 TF/s) | DRAM roof (230 GB/s) | v1 (W7) | v2 estimate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| expand | 4.29 G | u 32 MB + out 8 MB + kv_b 2.1 MB | 140 us | 183 us (u from DRAM), ~50 us (u in L2) | 2,415 us | **250-450 us** |
| absorb | 4.29 G | q 8 MB + out 16 MB + kv_b 2.1 MB | 140 us | 114 us | 676 us | **200-300 us** |

- **Estimate basis.** The issue bound is 64-77% of FMA peak = 180-220 us a call. Take 50-70% of it, for
  shared-memory operand loads, barriers every 16 k and the tail of 5.3 waves (expand at 64 rows: 1,024 programs on 48
  SMs x 2).
- **Per token** of an 8,192-row chunk (11 DSA layers, 16 sub-blocks), v1 is 11 x 3,091 / 512 = **66 us a token**
  (ROOFLINE gap 3 counts 61 us plus o_proj). v2 is 11 x (450-750) / 512 = 10-16 us. **Saving: ~50-57 us a token of
  ~680 = +7.5-8.5% prefill.** That is 1,471 -> ~1,590-1,600 tok/s at 24.5k, similar at 98k (the stage does not
  depend on context).
- **Decode.**
  - v1 `_expand` takes 109 us a call at 1-8 rows (W7). It is latency-bound: spills, 4 warps, 8 dependent 64-deep
    steps.
  - v2's 16-row programs have no spills and 8 warps, over the same 128 programs.
  - Guess: 40-60 us a call, i.e. ~0.5-1 ms of a 57 ms round (+1-2% single-stream decode). Unmeasured; the bench
    prints R = 1 and 8.

## 4. Offline evidence

Tree: every patch 0001-0360 applied to vendor/TensorFold at 2f8e514, plus 0390. The patch also applies on top of the
in-progress 0380 tree (`git apply --check`).

1. **`tests/test_mla_expand_interpreter.py`** (Triton's CPU interpreter, `TRITON_INTERPRET=1`).
   - **Model.** The interpreter normally computes `tl.dot` with `np.matmul` (no defined order), skips the Combine
     pass and truncates fp32 -> bf16. The test replaces those with the GPU's semantics:
     - the per-element FMA chain over k from the dot's `c`, in exactly emulated fp32 FMA (product exact in float64,
       sum rounded to odd, then to fp32). `test_fma32_is_exact` checks the emulation against exact rational
       arithmetic;
     - nearest-even bf16.
   - **v1 as compiled.** v1 runs from its own source with `acc = acc + tl.dot(...)` rewritten to
     `acc = tl.dot(..., acc)` (the Combine fold). A control test shows the unfolded chain gives other bits, so the
     comparison is not vacuous.
   - **Checked, bit for bit:**
     - v2 == v1 == a plain numpy chain, q4 (min/max and mse) and bf16 kv_b, R = 1, 7, 16, 17, 64 (the real shapes)
       and R = 513, 2,048, 8,192 (2 heads, head dim 64, latent 128, to keep the emulation fast; row samples against
       numpy past 513);
     - every row equals the same row in 1-, 7-, 17-, 33- and 64-row launches;
     - every forced tile (16-128 rows, BK 16-64, BN 32-256) gives the same bits.
2. **`tests/test_mla_expand_compile.py`** (compile for sm_121, no GPU; passes under Triton 3.7.1 and 3.8).
   - v1's TTIR dot takes the loop-carried accumulator (the fold happened) and nothing adds after it.
   - v2's 64 / BK dots per group chain `c` -> result -> `c` from the iter_arg to the yield.
   - Both run on the FMA path (blocked `dot_op`, no `mma` / tf32).
   - PTX has only `fma.rn.f32(x2)` and conversions: no `add` / `mul` / `sub.f32`, no `fma.rz` / `ftz`, bf16
     conversion `cvt.rn`.
   - v2's FMA count a thread equals accumulator elements x K plus dequantization.
   - Under 3.7.1: no spills, and at most 128 registers at the prefill tiles.
3. **v1 unchanged:** the PTX of `_absorb` / `_expand` (q4 and bf16, Triton 3.7.1, debug info stripped) is identical
   before and after the patch (hashes `68ee65fc3d742cda`, `ba2aa45d2969967e`, `1b80c4d465ebf4a4`, `3839d77072418bb7`).
   So v1 still means production's bits.

What is **not** proven offline: that ptxas keeps each chain's order (it does not reassociate `fma.rn`, which is IEEE
and has no fast-math flags) and that the hardware FFMA2 rounds like FFMA (both are IEEE fma per lane). The GPU
bitwise test settles both.

## 5. Files

- `patches/0390-glm-mla-expand-v2.patch`: `latent.py` only. The knob, `v2_tiles` / `V2_TILES` / `V2_BN`, `_wcols`,
  `_wrows`, `_absorb2`, `_expand2`, and the dispatch in `absorb` / `expand`.
- Tests:
  - `tests/test_mla_expand_interpreter.py`: interpreter;
  - `tests/test_mla_expand_compile.py`: compile, IR and PTX checks;
  - `tests/cuda/test_mla_expand_patches.py`: GPU bitwise;
  - `tests/cuda/bench_mla_expand.py`: GPU microbench.

## 6. GPU test plan (one window, ~40-50 min, production down ~30)

Build `glm53-tensorfold:mla` from the committed tree (b2's patches + 0390) and load it on both nodes. Keep `b2` for
the revert. Everything runs under `timeout`.

1. **Bitwise, one GPU**, with production stopped or on the worker node while the head node serves. About 2 minutes.
   - `PYTHONPATH=/src/TensorFold/tests/cuda timeout 900 pytest -q tests/cuda/test_mla_expand_patches.py`
   - Covers v2 == v1 for q4 / q4mse / bf16 kv_b:
     - R = 1, 2, 3, 7, 8, 16, 17, 31, 64, 65, 127, 513, 2,048, 8,192, 3 seeds, one with extreme magnitudes and
       signed zeros;
     - rows of an 8,192 launch == the rows launched alone and in windows;
     - every tile / BN;
     - inside a CUDA graph.
   - The control (tensor-core kernels differ) must pass too.
   - **Gate: all green.** A single False means the FMA chain differs somewhere (e.g. a Triton version that splits k).
     Stop, keep the knob off, and save the failing case.
2. **Microbench**, same GPU, about 3-5 minutes.
   - `timeout 1200 python tests/cuda/bench_mla_expand.py 1 8 512 2048 8192 --sweep`
   - Read:
     - v1 and v2 us a call and TFLOP/s;
     - "same bits True" on every line;
     - the projected us a token;
     - the best tile table.
   - If the best table beats the default by more than 5%, put it in `latent.V2_TILES` (same bits by construction;
     re-run step 1).
   - **Gate:** at 512 rows, expand + absorb v2 <= 50% of v1 (expected ~15-25%).
3. **Engine tests (regressions)**, with `GLM53_TF_MLA_EXPAND=v2` in the environment:
   - `pytest -q tests/cuda/test_latent_patches.py`: drafted == serial, resumed == fresh, chunk sizes, batched ==
     alone, and the window-row == serial-row kernel test;
   - `pytest -q tests/cuda/test_kv_pool_patches.py tests/cuda/test_fp8_kv_patches.py -k latent`.
4. **Load M: production + `GLM53_TF_MLA_EXPAND=v2`**:
   - start with `results/W8/load.sh M IMAGE=glm53-tensorfold:mla GLM53_TF_MLA_EXPAND=v2`;
   - check the boot log has the v2 line on both ranks;
   - run `results/W8/run.sh M '{}'`: `exact` 10/10, `batchexact` 4/4, `results/W5/ab.py` 24.5k / 98k twice,
     concurrent 1 / 4 streams.
   - Gates:
     - **reply sha `8794a3463259cc2f`** (same as production);
     - exact 10/10;
     - batchexact 4/4;
     - prefill 24.5k / 98k **>= +5%** over production's 1,468-1,473 / 1,447-1,451 tok/s;
     - decode not lower.
   - Also a same-image control load with the knob off (`load.sh M0 IMAGE=glm53-tensorfold:mla`): off must equal
     production exactly, as the PTX check says.
   - Resumed == fresh is covered by exact / batchexact and the engine tests.
   - NVMe sessions (0250) do **not** cross from production to load M. Their compat hash covers the image id and every
     `GLM53_TF_*` knob not on its skip list, and `GLM53_TF_MLA_EXPAND` is deliberately left off that list (a new knob
     only costs a cold start). After adoption it could be added to `sessdisk.KNOB_SKIP`, since it never changes a
     stored bit. Not done in 0390.
5. **Optional profile:** `GLM53_TF_PROFILE=1` on load M, one 24.5k prefill. `dsa.o_proj` and `dsa.proj` should drop by
   ~2.4 ms and ~0.4 ms a sub-block and layer.
6. **Adopt or revert.**
   - Adopt: `IMAGE=glm53-tensorfold:mla` and `GLM53_TF_MLA_EXPAND=v2` in `config/prod.env`, with the RESULTS entry.
   - Otherwise revert to `config/prod.env` (b2).
   - Either way: canary, https check, watchdog re-armed.

## 7. Risks

- **Bits.** The whole case rests on "one FMA chain per element in k order", checked in Triton's source (3.7.1) and our
  IR. A future Triton that splits k inside an FMA dot, or uses packed k-pairs, would change both v1 and v2, but not
  necessarily in the same way.
  - `tests/test_mla_expand_compile.py` re-checks the IR on any Triton.
  - Step 1 re-checks the bits on the GPU.
- **Speed.** Triton's FMA-dot codegen caps the microtile at 4 x 4 a thread, so operands come from shared memory at
  roughly 1 load per 4-8 FMAs, and v2 may land at the low end of the estimate. The ceiling is a hand-written SIMT
  kernel (CUDA, 8 x 8 register tiles, the same explicit `__fmaf_rn` chain): perhaps another 1.5-2x on the stage, i.e.
  +1-2% prefill more. That would be a follow-up, and it needs the same bitwise gate.
- **Occupancy next to the overlap stream.** 0084's pipelined prefill runs other kernels beside these. v2's CTAs use
  16 KB of shared memory and 128 registers x 256 threads (half an SM each), about like v1's register footprint. The
  load A/B (step 4) is the real measure.
- **Graphs.** Decode graphs capture after the eager warm-up of each key. v2 has a different Triton kernel per tile
  entry: 1-16 rows share one, 17-32 another. It is compiled on the eager step, so capture never compiles. Checked in
  the GPU test (`test_v2_in_a_cuda_graph`).
