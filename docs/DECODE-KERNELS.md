# Decode kernels: streaming routed experts (E1) and dense 4-bit GEMV (E2), same bits (patches/0440, 2026-09-29)

> Offline work: no GPU was used and the Sparks were not touched. Inputs: DECODE-PLAN E1 / E2 / E4, ROOFLINE §2,
> PROFILE §5, RESULTS W7-W10, SFXNZ-AUDIT §3.2 / §4, and the W11 raw data in `results/W11/` (bandwidth probe
> `probe-{head,worker}.log`, kernel bench `kbench.log`, `ncu-kb-raw.csv`; the W11 write-up in RESULTS.md was not in yet).
> Every speed number for the new kernels below is an **estimate**; §8 is the GPU plan that turns them into
> measurements.

## 0. Bottom line

- **Patch 0440** adds two persistent streaming kernels behind knobs that are **off by default**:
  - `GLM53_TF_DEC_EXPERTS=1` (E1, `exl3_stream.cu`). Routed experts of every window whose member table has ≤ 64
    columns: decode, verify, MTP and DFlash2 windows, and 4-slot batched windows.
  - `GLM53_TF_DEC_QMM=1` (E2, `q4_stream.cu`). 4-bit dense matmuls of 1-64 rows.
  - `GLM53_TF_DEC_PDL=1` (the E4 piece). Launches both kernels as programmatic dependents.
- **Same bits as today's kernels, for every row count.** Every output element gets the instruction sequence the
  current kernel gives it:
  - E1: exl3.cu's warp K ranges, mma chain, warp order, split order and epilogue code;
  - E2: `_qmm`'s mma chain and fma epilogue as the image's Triton 3.7.1 compiles it, and `_reduce`'s slice order.

  So drafted == serial, batched == alone, resumed == fresh and prefill / decode lockstep hold unchanged, and the
  two ranks may even differ in the knobs. Checked offline three ways (§6):
  - a lane-level CPU port of each new kernel against a matrix-level model of the old one, with an order-sensitive
    mma model and negative controls: 67 tests pass;
  - PTX / TTIR structure of the references and of the new kernels;
  - a full sm_121 compile: no spills, 2-5 CTAs an SM.
- **One finding outside the patch.** Under **Triton 3.8** (not the image's 3.7.1), `qmm._qmm` leaves 16-36 of its
  epilogue product-adds a thread **unfused** (`mul` + `add` instead of `fma`). This hits 12288 x 4096 (dense MLP
  gate/up) and 77440 x 4096 (head) at every row bucket, plus 4096 x 128 at 64 rows. The unfused slots differ between
  the 16-, 32- and 64-row buckets, so **an image on Triton 3.8 would change today's bits and could break batched ==
  alone** for those matmuls.
  - E2 is refused automatically on such a Triton (`qmm_reference_fused`).
  - `tests/test_decode_kernels_compile.py` belongs in any image-upgrade checklist.
- **Estimate** against the W11 ceilings (a plain streaming read reaches 233-237 GB/s at expert-layer sizes, 230 GB/s
  over the verify's dense set):

| | today (DECODE-PLAN §1) | E1 | E2 | PDL (both) | total, mid (low / high) |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 stream, prose round | 54.4 ms, 44 tok/s | -2.3 ms | -2.1 ms | -0.4 ms | **-4.8 ms: 49.6 ms, +9.7%** (+5% / +16%) |
| 4 streams, round | 119 ms, 76.5 tok/s | -9.7 ms | -3.0 ms | -0.8 ms | **-13.5 ms: +12.8%** (+7% / +19%) |

- **The GPU window is ~45 minutes (§8).** Steps:
  1. kernel bitwise tests, including a PDL race test;
  2. `bench_decode_kernels.py`: old vs new µs and GB/s at 1 / 2 / 4 / 8 / 16 rows and 4-slot mixes, every
     configuration, a roof probe beside them;
  3. engine tests;
  4. a production A/B.

  **Gate:** bits equal everywhere, and 1 stream ≥ +3%.

## 1. Where the time is (W7 / W10 / W11)

### 1.1 Ceilings (W11 probe, `results/W11/probe-{head,worker}.log`, both nodes alike)

| read | GB/s |
| --- | ---: |
| One MoE layer's gate+up or down trellis bytes, U = 8-65 experts, one launch (gathered or contiguous) | 227-238 (median ~235) |
| A round's 84 expert launches back to back, U = 8-51 | 235-237 |
| Dense, one launch, cold: head 178 MB / KDA in 29 MB / MLP gate-up 28 MB / DSA o 19 MB | 238 / 235 / 235 / 235 |
| Dense, one launch: MLP down 14 MB / KDA o 9.4 MB / DSA q_b 7.1 MB / shared gate-up 4.7 MB | 229 / 228 / 221 / 215 |
| Dense, one launch: index q_b 3.5 MB / shared down 2.4 MB / DSA kv 2.4 MB / index k 0.37 MB / KDA f_b, g_b 0.29 MB | 200 / 186 / 186 / 150 / 111 |
| The verify's whole dense set, 304 launches back to back | 230 |
| Launch gaps | empty kernel in a graph: 0.47 µs; last block to next kernel's work: 0.73-0.79 µs (graph), **0.41-0.44 µs with PDL** (399 / 399 early starts on 1-4 MB kernels) |

### 1.2 Today's kernels (W11 `kbench.log`)

`kbench.log` is **eager and event-timed**, so every row carries a launch overhead, 5-15 µs a Python-launched
kernel. Two consequences:

- Its small kernels are overstated.
- SFXNZ-AUDIT §3.2 compares these eager times with Marlin's graph replays, which inflates its small-shape ratios
  (e.g. 1.75x on shared down).

The large kernels are representative.

- **Routed experts** (one layer, real shape):

| R rows / U experts | gate/up µs (GB/s) | down µs (GB/s) | rot_in + gate/up epilogue + down epilogue, µs (eager) | whole layer GB/s |
| --- | ---: | ---: | ---: | ---: |
| 1 / 8 | 162 (207) | 82 (204) | 9.8 + 9.6 + 9.3 | 200.5 |
| 2 / 12 | 240 (210) | 124 (203) | 29 | 204.8 |
| 3 / 17 | 333 (214) | 170 (210) | 28 | 208.9 |
| 4 / 21 | 410 (215) | 208 (212) | 28 | 208.0 |
| 8 / 40 | 760 (221) | 380 (221) | 29 | 217.4 |
| 16 / 65 | 1,219 (224) | 612 (223) | 41 | 216.8 |

  Reading the table:
  - The grouped kernel reaches 204-227 GB/s alone. It is weakest at the single-stream sizes (U = 8-21) that
    dominate a prose round.
  - The three side launches cost ~29 µs a layer eager: 1.2 ms a forward (the audit's figure; less in graphs).
  - In the W7 round the family ran at ~193 GB/s effective.
  - ncu (`ncu-kb-raw.csv`): `grouped_kernel<8, 4>` uses 108 registers and 32 KB of shared memory: 3 CTAs, 12
    warps an SM. **Waves:** 4 (gate/up) / 2 (down) at R = 1. **Warps active:** 25%.

- **Dense `_qmm` (+ `_reduce`), M = 1:**

| shape | µs (GB/s) |
| --- | ---: |
| head | 825 (216) |
| KDA in | 149 (195) |
| MLP gate/up | 148 (192) |
| DSA o | 105 (181) |
| MLP down | 89 (159) |
| KDA o | 57 (166) |
| DSA q_b | 40 (178) |
| shared gate/up | 35 (136) |
| index q_b | 29 (123) |
| shared down | 25 (94) |
| DSA kv | 25 (95) |
| index k | 25 (15) |
| KDA f_b / g_b | 18 (17) |

  Their gaps to the one-launch probe are large even where launch overhead is negligible: KDA o 166 vs 228, MLP
  down 159 vs 229, DSA o 181 vs 235.
  - ncu: 95-128 registers, 10-18 KB shared memory, 4-5 CTAs an SM.
  - Grids of 32-197 x SK programs, e.g. KDA o = 64 x 2 = 128 programs = 0.67 waves.
  - Each matmul with SK > 1 is two launches (`_qmm`, `_reduce`).

### 1.3 Why 0130 (the earlier decode-kernel attempt) regressed, and what 0440 does differently

0130 (`GLM53_TF_DECODE_KERNELS=v2`) measured:

- **1-row verify:** 31.8 → 32.7 ms (v2) and 32.9 ms (`v2,pdl`);
- **decode:** tf code greedy 64.3 → 62.4 → 56.4 tok/s (RESULTS, "Decode kernels (0130)").

It was only ever timed end to end, so the loss was never attributed. From its source (`decode_v2.py`, `exl3_dec.cu`)
and the numbers above, the likely causes (inferences, not measurements) are:

1. **The EXL3 kernel kept the non-persistent grid and got no deeper memory pipeline.**
   - What changed: the loads of the next k tile were issued before decoding the current one, in registers (142
     registers, still 3 CTAs an SM).
   - What did not: the bytes in flight were those of the old kernel. Wave quantization (4 / 2 waves at R = 1) and
     each program's ramp were unchanged.
   - It added a fence + atomic + serial epilogue on the last program of every (expert, block). At the end of a
     launch those epilogues run after the stream has drained.
2. **The split-K fixup was added to Triton `_qmm` programs as another round trip each:** partial store, barrier,
   atomic, then partial loads by the last program. It saved the `_reduce` launch but lengthened every program.
3. **PDL was applied to every glue kernel too**, with `gdc_wait` + `launch_dependents` at entry.
   - Each dependent program prefetched 16 KB of weights into L2. On multi-wave grids that is issued while the
     previous kernel is still bandwidth-bound: it competes for DRAM and pollutes L2, with no idle bandwidth to
     use.
   - `v2,pdl` lost another 10% on code greedy.
4. **Nothing was measured per kernel** before the A/B.

0440's design follows from these:

- Persistent grids sized to residency: one ramp and one tail a matrix, and dependents can only occupy slots in the
  tail.
- A shared-memory ring several k steps deep that crosses item boundaries.
- Epilogues that finish under other CTAs' streaming.
- PDL only on the two big kernels, prefetching only the first item of resident CTAs, as a separate knob.
- A microbench with bit checks and a roof probe beside it, before any end-to-end run.

## 2. Exactness: what "same bits" pins down

### 2.1 E1 reference: exl3.cu `grouped_kernel` (decode / verify windows) + epilogues

For output (member row p, column n) of matrix `mat` (gate, up or down), the reference computes, with
`exl3_mm.GATEUP_CFG = (8, 4, 4)` and `DOWN_CFG = (8, 4, 1)`:

- **Work item:** 16 member rows of one expert x one K split x NT column tiles.
- **Warp chains:** warp w (W = 4) runs k tiles `split * KT/SK + w * PW ... + PW - 1` in order, with PW = KT / (SK W)
  = 16 at the real shapes. Each warp does one `mma.m16n8k16.f16` a (k tile, n8 half) into accumulators that start
  at +0.0. The B fragments come from `decode_tile` (the trellis codebook; lane L decodes words L - 1 and L). The A
  fragments are the member rows' rotated fp16 inputs (0 for rows past the members).
- **Warp reduction:** `s = acc_0; s += acc_1; s += acc_2; s += acc_3`, stored to Z[mat][split][p][n] for live rows
  only.
- **Epilogues (per member row and 128-column Hadamard block):**
  - gate/up: `sg = 0.f; sg += Z[0][s]` over the splits in order, likewise `su`; then `fwht128`, the limited SwiGLU
    with bf16 roundings, `* suh_d`, `fwht128`, and `__float2half_rn` into Xd.
  - down: the same split sum, `fwht128`, `* HAD * svh_d` into Y.

NT is placement only (the per-element chain does not depend on it). W and SK are the arithmetic.

### 2.2 E2 reference: `qmm._qmm` + `_reduce`, as compiled

TTIR (every shape and bucket, `test_qmm_reference_sequence`):

```
p   = tt.dot(x[16..64 x 64], q^T[64 x 64], zeros)            one per 64-input group, from a zero accumulator
acc = (acc + p * s) + xs * b                                  acc carried by the loop, from zeros
```

- **The dot.** Triton's MMAv2 lowering chains the dot's k repetitions in ascending k from its accumulator
  (`MMAv2.cpp`: `for k < repK`), with canonical m16n8k16 bf16 fragments.
- **The epilogue, under the image's Triton 3.7.1.** Every one of these product-adds is contracted, on every
  per-rank shape and in every row bucket (16 / 32 / 64). Two `fma.rn.f32[x2]` an element and group, no
  `add` / `mul`, so:
  `acc = fma(xs, b, fma(p, s, acc))`.
- **The slices.** qmm.split_k's K slices are then added in slice order (`t = part_0; t = t + part_s`) and rounded
  once (`cvt.rn.bf16.f32`), or kept fp32.

**Triton 3.8.0** (the local venv) contracts only part of it:

| row bucket | shapes with unfused `mul` / `add` pairs (count a thread) |
| --- | --- |
| 16 | 12288 x 4096 (24), 77440 x 4096 (24) |
| 32 | 12288 x 4096 (36), 77440 x 4096 (36) |
| 64 | 4096 x 128 (36), 12288 x 4096 (36), 77440 x 4096 (36) |

- **The cause:** LLVM's packed `fma.rn.f32x2` formation pairs some slots with an unfused `add.f32x2` of a separate
  `mul`.
- **What it means:**
  - `_qmm`'s bits on 3.8 differ from 3.7.1's in those slots.
  - The unfused slots are a different (row, column) set in each bucket, so a row's bits can depend on the window
    size. That is today's exactness contract at risk on an upgrade, independent of 0440.
- **What 0440 does:** E2 reproduces the fused sequence and stays off unless `triton.__version__` is 3.7.x.
  `GLM53_TF_DEC_QMM=force` overrides this, for measurements only.

## 3. E1 design: `exl3_stream.cu`

### 3.1 The kernel

- **Grid and item walk.**
  - Grid: SMs x resident CTAs an SM, capped by the item count (`GLM53_TF_DEC_CTAS` caps it further).
  - Items: every (member tile, expert, matrix, split, column block) of the window, the expert-major order that
    `grouped_kernel` encodes in its grid. Item i runs on CTA i mod grid.
  - Items numbered `(pair, mat * SK + split, column block)`: CTAs that run together read neighbouring 512-byte runs
    of the same trellis rows (DRAM page locality) and the same member rows (L2).
  - Pairs: first every routed expert's first member tile, then the live extra member tiles of experts with 17+
    members, listed once a CTA from the members table. Only 4-slot / 17-64-row windows have any.
  - Dead items are never visited, and the shared expert (id E, last) is excluded.
- **Warp-private rings.**
  - CTA = 4 warps. Warp w runs chain w of every item its CTA takes: exl3.cu's K range for (split, w).
  - Each warp owns STAGES shared-memory slots. A slot holds one k step: NT trellis tiles (NT x 128 B, 16-byte
    `cp.async.cg`) + the item's live member rows (32 B a row, `cp.async.cg`; dead rows are never copied: an mma
    row is independent of the others and a dead row is never stored).
  - STAGES - 1 steps are in flight per warp **across item boundaries**. The loader cursor walks the item list
    ahead of the compute cursor, so the next item's first k tiles land while this item is being reduced.
  - One commit group a step, `cp.async.wait_group STAGES - 2`, `__syncwarp`. No CTA barrier in the k loop.
- **Fragments.**
  - The A block is stored with a 16-byte-half swizzle (bit 2 of the row), so the four fragment reads are
    conflict-free.
  - Words go through exl3.cu's `decode_tile` (verbatim, the lane - 1 shuffle included) into `mma16816` (verbatim).
- **End of an item.**
  - Warps 1-3 park their accumulators (live rows only) in shared memory. Warp 0 adds them to its own in warp order
    and stores Z's live rows.
  - A ticket on (member tile, expert, Hadamard block) after `__threadfence`. The last of the block's
    `mats * SK * 128 / (16 NT)` items runs exl3_dec.cu's `gateup_epilogue` / `down_epilogue` (verbatim) for the
    tile's rows, one warp a row, Z read with `ld.cg`.
  - The counters are left at zero. Which CTA runs an epilogue is scheduling; what it adds is fixed.
  - The next item's member rows are fetched while the reduction runs.
- **Launches: 3 instead of 5 a layer.** exl3.cu's `rot_in` (unchanged), gate/up (+ its epilogue), down (+ its
  epilogue).
- **PDL** (optional, sm_90+).
  - Each CTA prefetches its first item's trellis words into L2, then runs `griddepcontrol.wait` and
    `launch_dependents`.
  - Before the wait the only global operations are `ld.global.cg` of the grouping (to find the addresses) and the
    L2 prefetches. Nothing read before the wait is kept: the walk is rebuilt from fresh `ld.cg` reads after it. So
    no ordering among earlier kernels is assumed (the sfxnz PDL lesson, §5).
  - Asserted on the PTX by `test_decode_kernels_compile.py`.
- **Probes (timing only).** `probe=1`: no trellis decode. `probe=2`: no mma. These split data movement from ALU.

### 3.2 Configurations (sm_121, nvcc 13.4, `__launch_bounds__(128, 3)`; `test_decode_kernels_compile.py`)

`GLM53_TF_DEC_EXPERTS_CFG = nt_gu,stages_gu,nt_dn,stages_dn` (default 4,4,4,4). Every row has 0 B spilled.

| NT, STAGES | registers | shared memory (dynamic + static) | CTAs an SM (warps) | bytes in flight a warp |
| --- | ---: | ---: | ---: | ---: |
| 2, 4 | 100 / 105 | 20.0 KB | 4 (16) | 0.8 KB words + rows |
| 2, 6 | 100 / 105 | 26.0 KB | 3 (12) | 1.3 KB |
| 2, 8 | 100 / 105 | 32.0 KB | 3 (12) | 1.8 KB |
| **4, 4** | 112 | 30.0 KB | **3 (12)** | 1.5 KB |
| 4, 6 | 112 | 38.0 KB | 2 (8) | 2.5 KB |
| 8, 3 | 142 | 44.0 KB | 2 (8) | 2.0 KB |

- **Little's law.** 235 GB/s x ~1 µs loaded latency ≈ 235 KB in flight device-wide, i.e. ~5 KB an SM.
  - Every configuration keeps 12-24 KB an SM in flight, continuously, including across item ends.
  - The old kernel keeps ~12 KB an SM in flight only between its decode phases, then drains at each program's end.
- **Items at R = 1 (U = 8).** 1,024 gate/up items and 512 down items (NT 4) over 144 CTAs: ~7 / ~3.6 an SM slot.
  - A tail item runs with fewer competitors: each warp's 1.5 KB in flight sustains ~1.5 GB/s, so ~40 active CTAs
    still saturate DRAM.
  - The persistent tail is therefore a small fraction of one item (~20 µs at full share), not a wave.

### 3.3 Not done in 0440 (next steps, in order)

1. **Fuse rot_in** (9.8 µs eager; ~3-4 µs in a graph + its gap).
   - Two options: a phase 0 of the gate/up kernel (each warp rotates its own member rows' 256-wide k range into its
     ring from x and suh), or a grid-wide flag after a rot_in phase.
   - Cost of the first: ALU doubles on 16-row items. The second relies on all CTAs being co-resident (true for this
     grid).
   - PDL already overlaps rot_in's tail with the gate/up prologue.
   - Worth ~0.1-0.2 ms a forward.
2. **One launch a MoE layer** (gate/up → down chained with per-(expert, block) ready flags).
   - Down items wait for their expert's Xd blocks, while their trellis words are already streaming.
   - Removes one ramp / tail a layer: ~2-4 µs, ~0.1-0.2 ms a forward.
3. **A ticket-based item claim** (instead of static striping) if §8's `--contend` numbers show the static walk
   losing beside the overlap stream, as fast2 did in W5.

## 4. E2 design: `q4_stream.cu`

SFXNZ-AUDIT item 1 names Marlin's techniques. Which of them keep today's bits:

| Marlin technique | here |
| --- | --- |
| weights laid out for 128-bit loads | **Yes.** The stored layout already has a group's 64 columns x 8 words contiguous (2 KB): every copy is a 16-byte `cp.async.cg`; each lane then reads its column's 32 B as two `LDS.128`. No re-layout, no new weight format |
| dequantization in registers (lop3), no shared-memory tile | **Yes.** The nibbles go straight into the mma B fragment: `(v & 0xF) \| (v & 0xF0) << 12 \| 0x43004300`, then `sub.rn.bf16x2 0x4300` gives the exact bf16 of 0-15. `_qmm` today writes each q as a `st.shared.b16` (64 a thread and step) and re-reads it with `ldmatrix` |
| multi-stage cp.async pipeline | **Yes.** A CTA ring of 4-8 stages: a group's words, scales, biases, input rows (a 72-element row pitch: conflict-free fragment reads) and group sums (4-byte `cp.async`). It crosses item boundaries |
| all SMs busy on narrow shapes (Marlin stripes K) | **Partly.** A persistent grid over all SMs; items are (64-column tile, K slice) with qmm.split_k's own slices, the slices added by the last one to finish (ticket). Wide matrices (≥ 96 tiles) use whole-tile items, the slices added in registers in order (`GLM53_TF_DEC_QMM_SERIAL`). **Marlin's finer K striping adds partial sums in a new tree, which changes bits: not taken** |
| fixed-order reduction, no atomics on data | **Yes** (`_reduce`'s order; the only atomic is the ticket) |
| BF16 `mma.sync` | **Yes.** The same m16n8k16 bf16 as `_qmm` |
| Marlin's own weight formats (NVFP4, INT8, AWQ zero points), its fp16-only float-zero-point path | **No.** Not needed: the kernel reads q4mse as stored, with BF16 activations |

- **The kernel's contract.** Rows 1-64 in up to four 16-row tiles a CTA (the weights decoded once for all of them);
  N not a multiple of 64 handled (zero-filled scale copies past N); strided input and output rows.
- **PDL.** Each CTA prefetches its first item's words into L2, then waits. Nothing but that prefetch comes before
  `griddepcontrol.wait` (asserted on the PTX).
- **Prefill stays on today's kernels.** Fast chunks go through `FAST_MM` → `_fq4` before the hook. Windows past 64
  rows go to `_qmm`. So Marlin's prefill trap (a GEMV re-streaming weights per M block; the audit's 3.6x) does not
  arise.

### 4.1 Configurations (`GLM53_TF_DEC_QMM_CFG = groups_a_stage,stages`, default 1,4)

| row tiles, groups a stage, stages | registers | shared memory | CTAs an SM |
| --- | ---: | ---: | ---: |
| 1, 1, 4 (1-16 rows) | 68 | 18.7 KB | 5 |
| 1, 1, 6 | 68 | 28.0 KB | 3 |
| 1, 1, 8 | 68 | 37.4 KB | 2 |
| 1, 2, 4 | 72 | 37.4 KB | 2 |
| 2, 1, 4 (17-32 rows) | 116 | 28.2 KB | 3 |
| 4, 1, 4 (33-64 rows) | 166 | 47.1 KB | 2 |

## 5. E4: PDL and small-kernel fusion

- **PDL exists on sm_121.**
  - `griddepcontrol.wait` / `launch_dependents` assemble for `sm_121` (the compile test).
  - W11 measured the effect in graphs: the gap shrinks from 0.73-0.79 to 0.41-0.44 µs a kernel, and dependents
    start early (399 / 399) on 1-4 MB kernels.
- **What 0440 uses.** PDL on the two streaming kernels only (knob `GLM53_TF_DEC_PDL`).
  - The kernels before them (rot_in, glue, norms) are launched as today. A PDL kernel's successor is a normal
    launch, so it waits for completion as today.
  - That is ~400-500 launch edges a verify round.
- **The sfxnz caution** (SFXNZ-AUDIT item 9: their kit gates PDL off on sm_12x citing races with KDA state kernels).
  The race is a consumer reading its producer's output before `griddepcontrol.wait`. 0440 answers it three ways:
  - **Statically:** the PTX of every new kernel has no `cp.async`, store or plain global load before its first
    `griddepcontrol.wait`. E1 has only `ld.global.cg` of the grouping, used for prefetch addresses and never kept.
  - **Dynamically:** `test_pdl_consumers_wait_for_their_inputs` (GPU).
    - A late-writer kernel calls `launch_dependents` at entry, sleeps ~0.3 ms, then writes the consumer's inputs.
      Until then the inputs are poisoned with NaN.
    - The consumers: gate/up's Xg / Xu, down's Xd, and the dense kernel's x and group sums.
    - A side stream keeps DRAM busy throughout.
    - Outputs must equal the non-PDL outputs. A consumer that read early would see NaN.
  - **By scope:** no KDA, attention or glue kernel is PDL-launched by 0440.
- **Small-kernel fusion is not implemented in 0440.** Its value from PROFILE §5 / DECODE-PLAN §1:
  - hc + router / combine 2.7 ms, KDA 1.6, attention 2.1, norms 0.4 a round;
  - 1,846 kernels a round, i.e. ~1.4 ms of launch gaps (0.75 µs each).
- **The next steps, each behind its own bitwise test:**
  1. E1's rot_in fusion (§3.3).
  2. 0130's `hc` / `route` / `pairs` features, re-measured one at a time with this bench's method. They are
     bit-exact but were only measured together.
  3. PDL for glue kernels with the same static prologue check and late-writer test. Only kernels whose first
     dependent load follows the wait; for KDA state, chain and replay this has to be checked kernel by kernel.
     Worth ~0.35 µs x ~1,400 edges ≈ 0.5 ms a round.

## 6. Offline verification (done)

All on the stack 0001-0430 + 0440, the local Triton 3.7.1 (the image's version) and 3.8.0, nvcc 13.4. No GPU.

| test | what | result |
| --- | --- | --- |
| `tests/test_decode_kernels_emulator.py` (+ `tests/decode_kernels_emu.py`), E1, 10 windows | Lane-level port of `exl3_stream.cu` against a matrix-level model of exl3.cu's routed path. Setup: order-sensitive mma model; fp16 codebook decode, fwht, epilogues and rot_in in numpy; shared memory starting as NaN garbage; copies landing at random times before their wait; CTAs interleaved at random item by item. Windows: 1, 2, 3, 5, 8, 16 rows; 4-slot mixes 3+3+3+2 and 5+1+4+3; skewed 20 and 40 rows (2 and 3 member tiles); every allowed (NT, stages) of gate/up and down; grids of 1-50 CTAs. Plus: rows alone == rows in the window; one epilogue a (pair, block); tickets back to zero; the shared slot never written | Xd and Y **bit for bit** |
| same file, real K | gate/up K = 4,096 (SK 4), down K = 1,024 (SK 1), 16 k tiles a warp: Z against exl3.cu's model | bit for bit |
| same file, real shapes, index only | D 4,096 / NI 1,024, up to 289 experts, 1-64 rows, NT 2 / 4 / 8: every live (member tile, expert, matrix, split, block) exactly once; each warp's k range exl3.cu's; the grid bound | pass |
| same file, E2, 14 cases | Lane-level port of `q4_stream.cu` (both item modes) against `_qmm` + `_reduce`'s compiled sequence. Rows 1-64, K slices 1 / 2 / 4 / 8, N 72-320 (not multiples of 64), bf16 / fp32, every configuration; rows alone == in the window | bit for bit |
| same file, negative controls | Each must break the bits, and each does:<br>E1: one `cp.async` group too few waited, the A swizzle dropped, warps added in reverse, a live row not copied.<br>E2: one group too few waited, a k half moved, the two fmas swapped, k chunks reversed, slices added in reverse (split and whole-tile modes) | all 12 caught |
| same file, host side | Knobs, refusals, dispatch rules (≤ 64 member columns, lean scratch refused, split-K without a partial buffer refused on narrow matrices), the Triton guard, the NVMe compat hash untouched | pass |
| `tests/test_decode_kernels_compile.py` | Every instantiation compiles for sm_121: 0 spills, registers and shared memory as in §3.2 / §4.1, ≥ 2 CTAs an SM. PTX has the named instructions. The PDL prologues are clean. exl3_stream.cu's helpers equal exl3.cu's / exl3_dec.cu's text. `_qmm`'s TTIR / PTX sequence for 11 shapes x 3 buckets (3.7.1: fully fused; 3.8.0: prints the unfused set of §2.2) | pass (6 tests) |
| full extension compile | `exl3_stream.cu` / `q4_stream.cu` host + device with torch headers (CUDA context header swapped for the stream header, which lacks cuSPARSE locally), both `.cpp` syntax-checked | compile |

**Run:**

```
PYTHONPATH=<tree>/src pytest -q tests/test_decode_kernels_emulator.py
NVCC=... PYTHONPATH=<tree>/src pytest -q -s tests/test_decode_kernels_compile.py
```

~3 minutes, 67 tests.

**What offline cannot show:**

- The hardware mma's own summation. The model is order-sensitive, but it is not the tensor core. The GPU bitwise
  test decides.
- Whether Triton's canonical fragment k-mapping holds for `_qmm`'s ldmatrix-fed operands (it is the documented
  layout).
- Every speed number.

## 7. Estimates, with the W11 ceilings

### 7.1 E1

5.09 GB of routed-expert reads a 1-stream prose round (809.5 expert-layer reads) and 13.81 GB at 4 streams:
25.9 ms (1 stream) and 72.3 ms (4 streams) today.

| | per-kernel GB/s assumed | 1 stream (ms, Δ) | 4 streams (ms, Δ) |
| --- | --- | --- | --- |
| today | 204-224 alone, ~193-197 effective with side launches, ramps and tails | 25.9 | 72.3 |
| low | 210 (a small gain in the loop, epilogues fused) + rot_in 0.4 ms | 24.6 (-1.3) | 66.2 (-6.1) |
| **mid** | **220** (U = 8: 215, U ≥ 17: 222-225; ~94% of the probe) + rot_in 0.4 | **23.6 (-2.3)** | **62.6 (-9.7)** |
| high | 228 (97% of the probe; the round probe reaches 235-237) + rot_in 0.3 | 22.6 (-3.3) | 60.9 (-11.4) |

- **The bar.** The audit's bar (Marlin MoE, 213 GB/s in their step) sits between today and the low case.
- **Why the mid holds.** The persistent ring removes the costs today's kernel pays at U = 8-21 and that the probe
  does not: wave quantization, per-program ramps and drained tails, and the two epilogue launches with their
  serial latency.
- **Why it is not higher.** The kernel still decodes (probe 1 in §8 bounds that), reduces and writes Z. The probe
  does none of this.

### 7.2 E2

~3.1 GB of dense reads a 1-stream round (verify 2.37 GB + MTP / DFlash2 / extra heads): 16.0 ms (~190 GB/s).

- **Targets per shape:** ~95% of the one-launch probe for ≥ 7 MB shapes (220-226 GB/s), and 85-90% for 2-5 MB
  shapes. Those are latency-bound: the in-kernel reduction and a 4-8-deep pipeline help most there.
- **Mid:** 3.1 GB at ~217 GB/s = 14.3 ms (-1.7 ms), plus ~150 `_reduce` launches a round removed (~2 µs each in
  a graph, gap included: -0.3 ms). **-2.1 ms.**
- **Range:** low -1.2, high -2.8.
- **4 streams:** 23.0 ms today at 146 GB/s, where batched MTP / per-slot calls add launches. **-3.0 ms** mid.
- **Cross-check:** the audit's Marlin bar is -2.6 ms a verify forward + ~0.4 ms of drafting (-3.0). It is partly
  inflated by the eager-vs-graph comparison of §1.2.

### 7.3 PDL

~500 launch edges a 1-stream round through the two kernels, each gaining the measured 0.35 µs of gap plus the
first item's ramp (weights already in L2): 0.5-1.5 µs an edge.

| | 1 stream | 4 streams |
| --- | --- | --- |
| mid | -0.4 ms | -0.8 ms |
| range | 0 to -0.7 ms | - |

The low end is 0130's contention effect, which the per-CTA prefetch limit is meant to avoid.

### 7.4 Round

| | 1 stream, prose (54.4 ms, 2.4 tokens) | 4 streams (119 ms, 9.1 tokens) |
| --- | --- | --- |
| E1 + E2 + PDL, mid | 49.6 ms: **48.4 tok/s, +9.7%** | 105.5 ms: **86.3 tok/s, +12.8%** |
| low / high | +5% / +16% | +7% / +19% |

This is DECODE-PLAN's E1 + E2 + a third of E4, now tied to the measured ceilings.

## 8. GPU test plan (one window, ~45 minutes, one Spark for steps 1-3, both for step 4)

Build: the image with 0440; knobs off means no behaviour change. First run the offline suites inside the image,
where the 3.7.1 guard must say fused:

```
pytest -q tests/test_decode_kernels_emulator.py
pytest -q -s tests/test_decode_kernels_compile.py
```

1. **Bitwise, kernels (~10 min).**
   - Command: `PYTHONPATH=/src/TensorFold/tests/cuda pytest -q -s tests/cuda/test_decode_stream_patches.py -k "experts or qmm or pdl"`.
   - E1: on == off for 14 windows x 5 configurations x PDL x eager / graph, real and synthetic shapes; CTA caps;
     rows alone; repeatability; probes launch.
   - E2: every per-rank shape x 21 row counts x f32 / bf16 x configurations x item modes x PDL; strided out; graph;
     rows alone.
   - The PDL race test.
   - **Stop on any False.**
2. **Microbench (~8 min).**
   - Command: `python tests/cuda/bench_decode_kernels.py --json /work/w12-kbench.json`.
   - Covers old vs new µs / GB/s for: single stream R 1 / 2 / 4 / 8 / 16 (U 8 / 13 / 22 / 35 / 64); 4-slot mixes
     3+3+3+2 / 4x4 / 4x8 (U 51 / 60 / 110); every E1 configuration ± PDL; probes 1 / 2; the 11 dense shapes at
     M 1 / 2 / 4 / 8 / 16 / 11 / 44, every configuration x split / whole-tile ± PDL; a roof probe on the same bytes
     in the same process.
   - **Pick** `GLM53_TF_DEC_EXPERTS_CFG`, `_QMM_CFG`, `_QMM_SERIAL` from it.
   - **Gate:** every row `same bits`; best new ≥ 1.05x old on the U = 8-22 expert windows and on the five largest
     dense shapes. Otherwise stop and report the probe split: if probe 2 (no mma) ≈ new, the decode ALU is the
     limit; if probe 1 ≈ new, data movement is.
   - Add `--contend`-style runs if the chosen static walk is close to the tie line.
3. **Engines (~12 min).**
   - Command: `pytest -q tests/cuda/test_decode_stream_patches.py -k "windows or replies or resume"` (MLX and EXL3
     synthetic checkpoints).
   - Then the existing suites with the knobs on:
     `GLM53_TF_DEC_EXPERTS=1 GLM53_TF_DEC_QMM=1 pytest -q tests/cuda/test_decode_patches.py tests/cuda/test_batch_parallel_patches.py tests/cuda/test_deep_verify_patches.py`.
4. **Production A/B (~15-20 min, both Sparks, the W10 harness).**
   - Correctness: `exact` 10/10 and `batchexact` 4/4 with the knobs on; reply SHAs equal to knobs-off on the
     glmbench prompts.
   - Timing: canary + 1-stream prose / code cells + 4-stream set, alternating off / experts / experts+qmm /
     +pdl, 2 reps each.
   - **Adopt a knob if:** bits equal and 1 stream ≥ +3% (E1 + E2) with 4 streams not lower.
   - PDL separately: adopt if ≥ +1% and the race test passed.
   - Then one nsys capture (the W7 harness) for the new per-kernel table.

**Revert:** unset the knobs. Nothing else changes: no snapshot tag, and the NVMe compat hash skips the knobs.

## 9. Files

- `patches/0440-glm-decode-stream.patch`:
  - `exl3_stream.cu` / `.cpp` (E1);
  - `q4_stream.cu` / `.cpp` (E2);
  - `decode_stream.py` (knobs, dispatch, the Triton guard, `run_experts` / `run_qmm`, `using`);
  - hooks in `exl3_mm.routed` and `qmm.matmul`;
  - the import in `forward.py`;
  - the knobs skipped in `sessdisk.py`'s compat hash.
- Tests:
  - `tests/test_decode_kernels_emulator.py`, `tests/decode_kernels_emu.py`;
  - `tests/test_decode_kernels_compile.py`;
  - `tests/cuda/test_decode_stream_patches.py`;
  - `tests/cuda/bench_decode_kernels.py`.
