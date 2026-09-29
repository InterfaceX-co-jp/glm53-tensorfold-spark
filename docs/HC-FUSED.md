# Hyper-connection boundary in one CUDA kernel: patch 0520 (`GLM53_TF_HC_CUDA=1`)

Offline work (2026-09-29). No GPU was used, nothing ran on the Sparks. Everything that needs a GPU is in the plan
(section 6) and has not run yet. THEORY-2 §4 item 3, session step 1d.

**Summary.**

- **What is fused.** One layer boundary, `glue.hc_post(x, x, g, post, comb)` followed by the next `glue.hc_pre`, is
  three Triton kernels today: `_hc_post`, `_hc_partial` and `_hc_finish`. Patch 0520 runs them as one launch of
  `hc_fused.cu`. That covers 89 of a forward's 90 hc_pre calls:
  - each layer's attention → MLP boundary;
  - each layer's MLP → next-layer attention boundary. The kernel also computes the next layer's first hc_pre, and
    that layer then skips it.

  It applies in `forward.layer_forward` and `batch.compute_multi`, for windows of 1-16 rows (`GLM53_TF_HC_CUDA_ROWS`,
  at most 64). Everything else runs the Triton kernels as before: fast-prefill chunks, larger windows, layer 0's
  first hc_pre, and the last hc_post.
- **The bits are the Triton kernels' bits.** Every output is the same: the streams x, normed, xs, post, comb and the
  partial buffer. This is not "the same maths": the kernel reproduces the fp32 operation sequence that Triton 3.7.1
  and its ptxas actually execute on sm_121. That includes every reduction tree and every rounding / fusion choice,
  and ptxas's own contractions of Triton's non-rounding `mul` / `add`, which the PTX alone does not show (section 2).
- **Offline evidence** (section 4), 100 tests:
  - A CPU specification of the three kernels equals, bit for bit:
    - the Triton kernels' own PTX run in a PTX interpreter;
    - the CUDA kernel's own PTX in the same interpreter;
    - a lane-level numpy port of the CUDA kernel.
  - This holds on 7 input kinds, 1-16 rows and every row-group size.
  - Negative controls: 7 one-detail variants of the specification and 6 mutations of the CUDA source are all caught.
  - The interpreter's model of ptxas is checked against the SASS: FFMA / FADD / MUFU counts are exact for all three
    Triton kernels.
- **Knob:** off by default, and load-time only. Both ranks must agree. With the knob on, the engine also runs a
  bitwise self-check against the Triton kernels at load, and uses the kernel only if both ranks pass.
- **Expected gain** (not measured): ~7-9 µs a boundary instead of 15.6 µs, i.e. -0.6 to -0.8 ms of a 53 ms 1-stream
  round (**+1.1-1.5%**). THEORY-2 gave -1.0 to -1.4 ms for hc and the router chain together. The gate is fused ≤ 50%
  of the three kernels, cold, in a graph. My estimate lands at 45-60%, so **the gate is not a sure pass**.

## 1. The three kernels (production: D 4096, 4 streams, 16 K blocks, 20 Sinkhorn iterations, 2 ranks, fn bf16)

| kernel | grid × threads | per program |
| --- | --- | --- |
| `_hc_post` | (R, 4) × 128 | 1,024 columns × 4 streams: `X_s = bf16(fma(branch, post_s, fma(x3, c3s, fma(x2, c2s, fma(x0, c0s, x1·c1s)))))`, `branch = bf16(g0 + g1)` |
| `_hc_partial` | (R, 16) × 128 | K block b (1,024 of the 16,384 flattened columns), in 8 steps of 128 columns. Each step: a [32, 128] tile of fn × x' reduced over columns into 24 dots, plus the column squares, accumulated over the steps |
| `_hc_finish` | (R) × 256 | mixes from the 16 partials (in block order), rsqrt, pre / post / comb (sigmoids, 4×4 Sinkhorn × 20), the collapsed row `c = bf16(Σ p_s x_s)`, its RMSNorm (`y = bf16(w · bf16(c · rinv))`) and 64-column group sums |

W11 trace, 1 stream: 8.0 + 5.3 + 1.7 = **15.6 µs a boundary**, 90 a round. All three are latency-bound
(THEORY-2 §1.5).

## 2. What "the same bits" pins down

The specification is `tests/hc_fused_emu.py` (`hc_post_ref`, `hc_partial_ref`, `hc_finish_ref`). It was read from
three sources:

- **TTGIR:** the layouts, which fix the reduction trees;
- **PTX:** the operations;
- **SASS** from Triton's own ptxas (ptxas-blackwell 13.1): what ptxas did to the PTX.

It was then checked by running the kernels' PTX (section 4).

**Reduction trees.** All adds are IEEE, and fp add is commutative, so only the association tree matters.

- `_hc_partial`, dots. Per 128-column step, each thread multiplies 8 consecutive columns: `v = x1·w1`, then
  `fma(x0, w0, v)`, `fma(x2, w2, ·)` … `fma(x7, w7, ·)`. Then a butterfly over 16 lanes, xor 8, 4, 2, 1. Then
  `acc = ((+0 + T0) + T1) … + T7` over the steps.
- `_hc_partial`, squares. Per column k: `s = fma(x, x, s)` over the 8 steps from +0 (`fma.rn.f32.bf16`). Then a
  butterfly over the 32 columns of each warp, xor 16 … 1. Then `(W0 + W2) + (W1 + W3)`.
- `_hc_finish`, mixes: `((+0 + P0) + P1) … + P15` in block order.
- `_hc_finish`, logits: one-hot butterflies over 32 lanes.
- `_hc_finish`, Sinkhorn: `(c0 + c2) + (c1 + c3)` over j (lanes xor 2, 1) and over i (lanes xor 8, 4). Max via
  `max.f32`.
- `_hc_finish`, collapse:
  - 16 columns a thread (8 t … 8 t + 7 and 2,048 + 8 t …);
  - **per-slot fusion:** even slots compute `fma(p1, x1, p0·x0)`, odd slots `fma(p0, x0, p1·x1)`. This is LLVM's
    packed-FMA formation;
  - then `+ p2 x2`, `+ p3 x3` as fma.
- `_hc_finish`, squares of c: a 16-long chain per thread, then a butterfly over 32 lanes, then over the 8 warps:
  `((W0 + W4) + (W2 + W6)) + ((W1 + W5) + (W3 + W7))`.
- `_hc_finish`, group sums: 8 columns a thread in order (`add.rn.f32.bf16`), then a butterfly xor 4, 2, 1.

**The approximate units, as ptxas builds them** (read from the SASS):

- `tl.sqrt` = `sqrt.approx.ftz.f32` = one MUFU.SQRT.
- `tl.exp` = `ex2.approx.f32` of x·log2e. ptxas lowers it as: if x < -126, halve x, run MUFU.EX2, and square the
  result.
- `/` = `div.full.f32`. ptxas lowers it as:
  - if |b| > 2^126, scale both operands by 1/4;
  - if |b| < 2^-126, scale both by 2^24;
  - then MUFU.RCP(b′) · a′.
- A power-of-two constant divisor becomes an exact multiply.

**ptxas contractions that the PTX does not show.** Triton emits plain `mul.f32` / `add.f32`, which ptxas may fuse,
and it does:

| where | PTX | SASS | changes a value? |
| --- | --- | --- | --- |
| comb's first normalization `ce / rowsum + hc_eps`, and `pre = 1/(1+e) + hc_eps` | div.full, add | the division's final multiply fused with the add: `fma(rcp, a′, hc_eps)` | **yes** (control `no_contract` catches it) |
| `ss / (S D) + eps`, `Σc² / D + eps` | div.full by 2^14 / 2^12, add | `FFMA(x, 2^-14, eps)` | only for subnormal x·2^-k (the scaling is exact) |
| squares of the bf16-rounded collapse | odd squares `mul` + `add` | all FFMA | no (bf16² is exact in fp32) |

The partial's chain is value-neutral for the same reason: bf16 × bf16 products are exact. So its first-product
choice cannot matter. The collapse's per-slot choice does matter (fp32 × bf16). It is visible only through the bf16
rounding, so an adversarial input exists for it (`collapse` kind).

**Why the PTX is not enough, and how the gap is closed.** `ptxas_view()` rewrites Triton's PTX into what ptxas
executes:

- packed f32x2 split into lanes;
- dead lanes dropped;
- div.full expanded;
- a single-use non-rounding mul fused into its non-rounding add.

The FFMA / FADD / MUFU.RCP / EX2 / SQRT counts this predicts equal the SASS of all three kernels exactly:

| kernel | FFMA | FADD | RCP | note |
| --- | ---: | ---: | ---: | --- |
| `_hc_post` | 128 | 8 | 0 | |
| `_hc_partial` | 28 | 27 | 0 | |
| `_hc_finish` | 70 | 145 | 26 | Sinkhorn loop ×10 in SASS |

The plain PTX semantics give other bits (control test).

**Triton 3.8 is different.** Its kernels compile to other arithmetic (the partial's order and the contractions),
and the specification does not hold there. The compile test reports this under 3.8. The load-time self-check exists
for exactly that case: another Triton or ptxas in an image.

## 3. The kernel (`hc_fused.cu`)

**Grid.** 256-thread CTAs.

- 32 "step" CTAs a group of `rg` rows (`GLM53_TF_HC_CUDA_RG`, default 4): one per (quarter q of D, 128-column
  step t). Each step CTA:
  - loads its fn slice once (24 KB, 6 × 16 B a thread);
  - computes X′ for its 4 × 128 columns and rg rows with `_hc_post`'s expression, and stages it in shared memory and
    in a scratch copy XP;
  - computes, for its 4 K blocks (b = 4 s + q) and 24 mix rows, `_hc_partial`'s step value T[r, b, t, m] (the same
    8-column chain and 16-lane butterfly) into scratch TS;
  - fences, and takes a ticket on its group's counter.
- One "finisher" CTA a row (blockIdx after the step CTAs). It prefetches the norm weight, spins until its group's 32
  tickets are in (acquire), then:
  - `PART[b][m] = ((+0 + T0) … + T7)`: `_hc_partial`'s accumulator, over t in order;
  - the square sums from XP: the same per-column fma chains, the same warp butterfly and `(W0+W2)+(W1+W3)`;
  - `X := XP`. The streams are written only here: every step CTA of the group has read X by then (in place, as
    `hc_post(x, x, …)`);
  - `_hc_finish`, with its thread mapping (the same lanes, warps and exchanges);
  - the last finisher resets the tickets (graph replays).
- Deadlock-free: at most 64 finishers wait, and ≥ 144 CTAs are resident.

**Arithmetic.**

- Only `__fmaf_rn` / `__fmul_rn` / `__fadd_rn` / `__fsub_rn`, which ptxas never contracts. The compile test asserts
  that no non-`.rn` fp32 operation remains in the PTX, and that the SASS FFMA / FMUL counts equal the PTX's.
- The units are written as ptxas expands Triton's instructions: `rcp` / `sqrt` / `ex2` `.approx.ftz` (one MUFU each,
  checked in the SASS) inside the explicit div.full / ex2 scaling.
- Where ptxas fused Triton's multiply with an add, the kernel uses the fma.
- The mixed bf16 instructions are Triton's own: `fma.rn.f32.bf16`, `add.rn.f32.bf16`, `mul.bf16x2`.

**Fit** (nvcc 13.4, sm_121): 80 registers, no spills, 18.9 KB static shared memory, 3 CTAs an SM (144 on GB10).
One wave up to 16 rows at rg 4 (4 × 32 + 16 = 144).

**Knobs** (`hc_cuda.py`):

| knob | values | what it does |
| --- | --- | --- |
| `GLM53_TF_HC_CUDA` | 0 / 1 | the kernel. Load-time; both ranks must agree; used only after the self-check (below) |
| `GLM53_TF_HC_CUDA_ROWS` | 1-64, default 16 | largest window that uses it |
| `GLM53_TF_HC_CUDA_RG` | 1 / 2 / 4 / 8, default 4 | rows a group of step CTAs handles. Placement only |
| `GLM53_TF_HC_CUDA_PDL` | 0 / 1 | launch as a programmatic dependent: fn / norm-weight loads before `griddepcontrol.wait`. Timing only |

**Self-check.** With `GLM53_TF_HC_CUDA=1`, `Engine.__init__` (before calibration and any graph capture):

- compiles the extension (first start: ~1 min, then cached);
- runs `hc_cuda.self_check` on 1 / 3 / 16 random rows against the Triton kernels, every output bitwise;
- turns the kernel on only if both ranks pass. It prints `[tensorfold] hc boundaries (patches/0520): …`, or the
  failure.

A different Triton or ptxas in a future image therefore turns the knob off by itself, rather than changing bits.

**Call sites.**

- `forward.layer_forward`: the two boundaries. The next layer's first hc_pre is skipped only on a mark
  `(id(layer), rows)` that the very next call consumes. A stale mark (an interrupted forward) only costs a
  recomputation, because hc_pre recomputes from x.
- `batch.compute_multi`: the same.
- `engine.py`: the rank-agreement check and the self-check.

Off, the call sites do what they did: the same Triton calls with the same arguments in the same order (unit-tested).
The only addition is a module-attribute check. No edits touch `_compute` / `_forward` (0510's regions), and the patch
applies after 0510 and before 0530.

**Scratch.** Per `Buffers` (keyed by its hcpart storage): XP 2 MB, TS 768 KB, 65 tickets. It is allocated on the
first eager use. A capture never allocates: a graph captured before any eager boundary on its buffers keeps the
Triton kernels.

**Not changed:** the NVMe session compat hash. The new knob is in it, so the first start with it is a cold start,
as for 0390. It can join `sessdisk.KNOB_SKIP_PREFIX` once the GPU test passes.

## 4. Offline evidence

Tree: vendor/TensorFold + every patch through 0510, then 0520 (0530 after it also applies). Python 3.14, torch CPU,
Triton 3.7.1 (the image's) with its ptxas-blackwell 13.1 / nvdisasm, nvcc 13.4.

| test | what | result |
| --- | --- | --- |
| `tests/test_hc_fused.py` | knobs; fp helpers vs exact rationals; lane-level port of hc_fused.cu == specification (7 kinds × 1-16 rows × rg 1-8, random step-CTA orders); rows independent of the window; 7 negative controls each caught; `fits` rules; `layer_forward` / `compute_multi` call sequences off (untouched), on, on-but-declined, stale marks; bench gate helper | **67 passed** |
| `tests/test_hc_fused_compile.py` | Triton 3.7.1 compile (sm_121): TTGIR layouts, PTX inventory; **Triton PTX (finish through ptxas_view) == specification**, 7 kinds; plain-PTX control differs; **ptxas_view == SASS** (FFMA / FADD / MUFU exact, 3 kernels); hc_fused.cu: sm_121 fit (80 regs, 0 spill), strict PTX (only `.rn`, units `.approx.ftz`, no division), SASS units / FFMA / FMUL == PTX, **its PTX == specification** (10 cases: 1-16 rows, rg 1-8, pdl, 7 kinds; tickets back at 0); 6 source mutations each caught; full extension compile with torch headers | **33 passed** |

Run:

```
PYTHONPATH=<tree>/src:<triton 3.7.1> pytest -q tests/test_hc_fused.py                       # ~20 s
NVCC=.../nvcc PYTHONPATH=<tree>/src:<triton 3.7.1> pytest -q -s tests/test_hc_fused_compile.py   # ~90 s
```

The input kinds:

| kind | inputs |
| --- | --- |
| random | |
| zeros | signed zeros |
| extreme | logits around ±87-88: ex2's halving path and div.full's 1/4 path; underflowing Sinkhorn entries; huge streams |
| tiny | subnormals |
| nonfinite | inf / NaN |
| real | real-scale |
| collapse | adversarial for the per-slot fusion |

**The interpreter.** SIMT: numpy lanes, min-PC scheduling, shuffles, barriers, stmatrix, atomics, param space. It
places shared arrays adversarially, aligned only as declared, and faults on misaligned vector accesses. That check
found a real bug: the float4-read staging array had only 4-byte alignment, now `__align__(16)`.

The unit instructions are stand-ins that are deliberately not exact (the last bit is flipped for about half of the
inputs, subnormals flushed). Two programs agree under them only if they feed the same bits into the same unit.

**Not proven offline:**

- That the hardware MUFU / FHFMA / HMUL2 give equal results for equal inputs in both kernels. They are the same
  instructions, so this is expected.
- That the production image's Triton uses the same ptxas (ptxas-blackwell 13.1, bundled) as this analysis. The GPU
  test prints it, and the load-time self-check covers a mismatch.
- The memory ordering of the tickets. It is the standard fence + atomic / acquire pattern, and the finishers read
  scratch with `ld.global.cg`. The GPU test runs it, including 2,112 CTAs at 64 rows.
- Every speed number.

## 5. Estimate (not measured)

**Step CTAs.** fn is 768 KB from DRAM over 32 SMs, 24 KB each, all in flight at once: ~3.5-4 µs, then the dots.

**Finisher, after the last ticket:**

| stage | time |
| --- | --- |
| poll | ~0.5 µs |
| XP / TS from L2 | ~0.7 µs |
| partial sums + square trees | ~0.3 µs |
| Sinkhorn (40 dependent normalizations of shuffle + add + RCP; inherent to the arithmetic) | ~1.5-2 µs |
| collapse, norm, group sums | ~0.7 µs |

**Total ~7.5-9.5 µs** cold, against 15.6 µs: a ratio of 0.48-0.61. Saving per 1-stream round: 89-90 boundaries ×
6-8 µs = **0.55-0.75 ms of 53.3 ms (+1.0-1.4%)**. At 4 streams, ~0.5-0.6 ms of 121 ms (~+0.5%).

If the gate fails narrowly, the levers are:

1. `GLM53_TF_HC_CUDA_PDL=1` together with a RoCE gather kernel that releases its dependents early. The 3.5 µs fn
   fetch would then overlap the exchange; that needs a one-line `griddepcontrol.launch_dependents` in roce.cu.
2. More step CTAs: split the 24 mix rows in two for 64 CTAs; x′ is then recomputed twice, which is cheap.
3. rg 1-2 at small R.

Knobs 1 and 3 need no rebuild. The bench times rg 1 / 2 / 4 / 8 × PDL off / on and prints the best.

## 6. GPU plan (THEORY-2 §6 step 1d, ~10-15 min, one GPU, production down or on the idle node)

1. **Bits** (~3 min, first run compiles):

   ```
   PYTHONPATH=/src/TensorFold/src:/src/TensorFold/tests/cuda:/work/tests/cuda timeout 1200 \
       pytest -q -s tests/cuda/test_hc_fused_patches.py
   ```

   It covers:
   - fused == Triton on every output for 1-16 rows × 2 kinds, 5 edge kinds × 4 sizes, and 24 / 32 / 48 / 64 rows;
   - rg 1-8, seeds, a gathered row slice, world 1;
   - rows alone == in the window;
   - a CUDA graph ×3 plus a replay on changed inputs;
   - 2,112 CTAs;
   - PDL: the same bits, and a late-writer test (the partials NaN until ~0.2 ms after the kernel may start);
   - `self_check`.

   It prints the Triton / ptxas versions. **Gate: all green.** One red means stop and keep the knob off (the
   self-check would also refuse). Save the case.
2. **Microbench** (~3-4 min):

   ```
   nvidia-smi -lgc 2250,2250
   PYTHONPATH=... timeout 900 python tests/cuda/bench_hc_fused.py --json results/W13/hc_fused.json
   ```

   Each boundary runs in a CUDA graph after a 50 MB streaming kernel: cold L2 and a predecessor tail. It times the
   3 Triton kernels vs fused (rg × PDL) at 1-16, 32 and 64 rows, and prints:

   ```
   GATE item3 hc_fused: fused <= 50% of the 3 Triton kernels (cold, rows 1-16): PASS/FAIL (...)
   ```

   It also prints the estimated ms a round, and the RG / PDL to set. **Gate: PASS.**
3. **If it passes:** load A/B in the next window: control vs `GLM53_TF_HC_CUDA=1` (+ the bench's RG / PDL).
   - Check that the boot log shows the 0520 line on rank 0 and no self-check failure.
   - Run the W12 `ab.sh` set.
   - **Gates:** exact 10/10 and batchexact 4/4 with the production reply hash unchanged (the bits must not move);
     1s ≥ +0.8%; 4s not lower.
   - Take one nsys capture of a 1s cell: per-boundary hc time must drop by ≥ 6 µs in situ (THEORY-2 §4 item 3).
   - Then the router chain the same way (not started: section 7).

## 7. Not done

- **The router chain** (router_part → topk → group → router_sum → rot_in) was optional and not started.
- **Layer 0's first hc_pre and the last hc_post** stay Triton, 2 of ~180 launches a forward.
- **Prefill** (fast chunks: other kernels) is not covered. Exact chunks over 16 rows are not covered either (Triton
  is faster there).

## 8. Files

- `patches/0520-glm-hc-fused.patch`:
  - `cuda/hc_fused.cu`, `hc_fused.cpp`, `hc_cuda.py` (new);
  - call sites in `forward.py` (layer_forward), `batch.py` (compute_multi) and `engine.py` (knob agreement +
    self-check).
- `tests/hc_fused_emu.py`: specification, PTX interpreter, ptxas view, lane port, input kinds.
- `tests/test_hc_fused.py`, `tests/test_hc_fused_compile.py`: CPU tests.
- `tests/cuda/test_hc_fused_patches.py`, `tests/cuda/bench_hc_fused.py`: GPU tests.
