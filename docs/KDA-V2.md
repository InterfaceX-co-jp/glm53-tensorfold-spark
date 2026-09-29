# KDA prefill recurrence v2 on GB10: patch 0400 (`GLM53_TF_KDA_V2`)

Offline work (2026-09-28). No GPU was used and production was not touched. Everything below that needs a GPU is in
the test plan (section 7) and has not run yet.

**Summary.**

- Patch 0400 adds `kda_v2.py`, a drop-in for `fast_kda.kda_prefill_chunked` (the chunked KDA scan every fast prefill
  chunk runs, 34 layers). It is switched by a load-time knob, `GLM53_TF_KDA_V2`: `0` (default) is today's kernels,
  `1` = **split**, `2` = **fused**.
- **The arithmetic is unchanged.** Every floating-point operation of `fast_kda` is kept: the same values meet in the
  same order, with the same roundings. Only where intermediates live and which program does which piece of work
  change. So v2 gives `fast_kda`'s bits: every output row, the state after every call (every snapshot), at any call
  size. Resume == fresh, the 64-row grid and the session store carry over, and snapshots are shared (no new tag).
- **Where the time goes today** (W7 profile, one 512-row lean sub-block, 32 heads a rank): prep 568 us (256 programs,
  1 an SM), state 334 us (**64 programs** on 48 SMs), norm 41 us. That is ~150 MB of DRAM traffic a call, ~650 us at
  230 GB/s. The 46 MB fp32 workspace the prep writes and the state kernel reads back is most of it. The state kernel
  also reads its chunk's operands twice, because the two value blocks of a head run a wave apart.
- **Split (mode 1).** `fast_kda`'s prep unchanged, then a new scan kernel:
  - value blocks of 32 (128 programs);
  - the blocks of a head launched next to each other, so DRAM serves each chunk's W / Q~ / K^ / P once;
  - dots split into two 64-key halves (1 staging buffer live at a time: 24 KB of shared memory instead of 64 KB);
  - a register cap (`maxnreg` 168) that fits 3 programs an SM, so all 128 are resident.
  - Low risk. Estimated **-8..-12 us a token, about +1.2-1.8% prefill.**
- **Fused (mode 2).** One persistent kernel takes prep and scan-step work items by ticket:
  - the per-(chunk, head) workspace is a ring of 3 chunks (~14 MB) that stays in L2, instead of 46 MB a call through
    DRAM;
  - the last waves of prep and scan overlap;
  - a scan step waits on flags for its chunk's prep and for the previous step of its value block, and every wait
    points to an earlier ticket, so it cannot deadlock.
  - Estimated **-20..-35 us a token, about +3-5% prefill** (1,471 -> ~1,515-1,545 tok/s at 24.5k). This is the
    ROOFLINE's "+4-5%" idea.
- **Offline evidence** (section 5):
  - the helpers compile to `_kda_prep`'s PTX instruction for instruction;
  - every scan setting and both fused branches compile to the reference's float ops, dot chains and reduce layouts,
    under Triton 3.7.1 (the image's) and 3.8;
  - an interpreter model of the GPU's dot chains and of Triton's combine rewrite gives `fast_kda`'s bits for every
    mode, setting, length (1 to 8,192) and call split. Mutation controls show these tests would catch a wrong order.

## 1. Today's kernels (`fast_kda`, patches/0081 / 0085)

Per 64-row chunk c (absolute multiples of 64) and head h, with S the fp32 state [128 values, 128 keys]:

| kernel | grid (512 rows) | does | W7 time |
| --- | --- | --- | ---: |
| `_kda_prep` | (8 chunks, 32 heads) = 256, 8 warps | conv + SiLU (bf16), q / k L2 norms, decay, cumsum G, beta (bf16); A and P by 16-row blocks (tf32 dots); T = (I + A)^-1 (16 x 16 forward substitution + ieee dots); K^, Q~, W = T (beta K~), U = T (beta V), e^{G_C}: all into a 181 KB fp32 workspace per (chunk, head) | 568 us |
| `_kda_state` | (32 heads, 2 value blocks of 64) = **64**, 8 warps | chunks in order: E = U - W S^T, O = Q~ S^T + P E -> bf16, S <- S e^{G_C} + E^T K^ | 334 us |
| `_kda_norm` | (16 row blocks, 32 heads) | gated RMSNorm in place | 41 us |

943 us a sub-block x 34 layers / 512 rows = **62.6 us a token** (ROOFLINE rank 5: 63.5 now, roof 15.7).

**Bytes a 512-row call** (the roofline's 3.6 MB a token counts the workspace once; it goes through DRAM twice):

| traffic | MB |
| --- | ---: |
| prep reads q / k / v rows (bf16, + 3 conv rows) and the decay rows | ~17 |
| prep writes the workspace: 181 KB x 256 (A / T scratch included; dirty lines are written back) | 46 |
| state reads W, Q~, K^, P (112 KB a chunk-head) twice + U once: the block of head h >= 16 runs in the second wave | ~65 |
| state in / out, outputs (bf16) | ~8 |
| norm: outputs + gate in, outputs out | ~13 |
| **total** | **~150 (~650 us at 230 GB/s)** |

**Occupancy** (compiled for sm_121 with production's specializations; Triton 3.7.1 = the image's; `tests/test_kda_v2_compile.py -s`):

| kernel | registers | spill st/ld (B) | shared | programs an SM | programs a call |
| --- | ---: | ---: | ---: | ---: | ---: |
| `_kda_prep` (8 warps) | 255 | 240 / 360 | 40 KB | 1 | 256 (5.3 waves) |
| `_kda_state` bv 64 (8 warps) | 255 | 404 / 408 | 64 KB | 1 | 64 (2 rounds for 1.33) |
| v2 `_state2` bv 32, 4 warps, K split | 240 | 0 | 24 KB | 2 | 128 |
| same, `maxnreg` 168 (default) | 168 | 296 / 404 | 24 KB | **3** | 128 (all resident) |
| v2 `_state2` bv 16, 4 warps, K split | 166 | 0 | 20 KB | 3 | 256 |
| v2 `_fused` 64-row blocks, no K split (default) | 255 | 1,108 / 1,252 | 64 KB | 1 | 48 persistent |
| v2 `_fused` 32-row blocks, K split | 255 | 788 / 980 | 40 KB | 1 (2 with `maxnreg` 128: 5.4 KB spills) | 48 / 96 |

`num_stages` 2 would need 120-160 KB of shared memory (GB10: 99 KB a block), so the scan is not software-pipelined.

## 2. Why the value columns can be split, and what else is exact

Row j of S (value j, all 128 keys) only ever meets column j of U, E and O:

- E[:, j] = U[:, j] - W S[j, :]^T;
- O[:, j] = Q~ S[j, :]^T + P E[:, j];
- S[j, :] <- S[j, :] * e^{G_C} + E[:, j]^T K^.

The sums run over keys (W S^T, Q~ S^T: 128 terms) or over the chunk's rows (P E, E^T K^: 64 terms), never over values.
Nothing in the prep depends on the state. So a program holding 32 (or 16) value rows computes, per element, exactly
what a program holding 64 does. It needs the same dots over the same K in the same order: the delta rule mixes keys,
not values.

What makes each op the same bits (each checked on the compiled kernels, section 5):

| op | why it cannot change | check |
| --- | --- | --- |
| `tl.dot` (tf32) | On sm_12x a dot is an `mma.sync` m16n8k8 chain. The accumulator starts at the given value and is carried from one 8-wide k-step to the next in ascending K. An element depends on its row of A, its column of B, the start value and K's order, not on the tile's M / N or the warps. | every dot: mma v2 [16, 8], operand kWidth 1, tf32 (compile test) |
| K split (`KSPLIT`) | `dot(w2, s2^T, dot(w1, s1^T, 0))` is the chain of `dot(w, s^T, 0)`: the second dot starts from the first one's accumulator. | chains compared as (precision, accumulator source, total K) |
| `a + dot(...)` | Triton rewrites `addf(x, dot(a, b, 0))` into `dot(a, b, x)` (the compiled `_kda_state` has O = dot(P, E, acc = dot(Q~, S^T)) and S = dot(E^T, K^, acc = S * e^{G_C})). v2 keeps the same source form (no split) or writes that chain explicitly (split). | accumulator sources equal; interpreter models the rewrite |
| `tl.sum` / `tl.cumsum` | These are the only layout-ordered ops: q / k norms, the decay cumsum, 2 x 15 forward-substitution sums. They are all in the prep, which runs `fast_kda`'s code with its num_warps (8). | all 33 reduce / scan ops: the same full operand layout |
| fp32 stored and loaded back | The value is unchanged. v2 moves fp32 intermediates between registers, L2 and DRAM (the fused state passes step to step through `state_out`). | interpreter; the fused kernel's float ops = prep + one step |
| bf16 roundings | None added or removed: conv output, beta, O -> bf16 and the norm stay where `fast_kda` has them (`_kda_norm` is reused). | same ops in order |
| FMA contraction | LLVM fuses mul + add per instruction pair. Fusion or hoisting must not gain or lose an fma. | PTX fma / ex2 / rcp / sqrt / cvt / mma counts: fused == prep + step |
| register cap | `maxnreg` is a ptxas directive: the PTX differs only in the `.maxnreg` line. | compile test |

**Not exact, so not done:**

- Changing the prep's warps (layouts of its reductions).
- Storing W / U / Q~ / K^ in bf16 (`fast_kda.STORE_BF16`, which doubles the state error).
- Moving the prep into the scan program with other tiles. This is b12x's 16-row fused kernel, bit 2 of 0240: new
  arithmetic, and measured 0.94x of `fast_kda` anyway.

## 3. Design

### Mode 1: split (`GLM53_TF_KDA_V2=1`)

- `fast_kda._kda_prep` unchanged (same grid, same workspace).
- `_state2`: grid (value blocks, heads), so the 4 blocks of a head are adjacent in launch order and read each chunk's
  W / Q~ / K^ / P together. DRAM then serves each chunk once (~37 MB instead of ~65).
- Value blocks of `GLM53_TF_KDA_V2_BV` = 32 (128 programs), 4 warps.
- The K split loads each operand right before its dot, so one 16 KB staging buffer is live at a time: 24 KB of
  shared memory instead of 64.
- `GLM53_TF_KDA_V2_MAXNREG` = 168 fits 3 programs an SM, so all 128 are resident in one round.
- `_kda_norm` unchanged.

### Mode 2: fused (`GLM53_TF_KDA_V2=2`)

One persistent kernel, `_fused`: `GLM53_TF_KDA_V2_CTAS` programs (default one an SM, 48), 8 warps, 64-row value
blocks by default (`_FUSED_BV`).

**Work items, in ticket order** (an int32 counter, `atomic_add` acq_rel):

1. prep(0 .. LAG-1, every head);
2. for g = LAG .. NC-1: prep(g, every head), then the NVB x H scan steps of chunk g - LAG, with the blocks of a head
   adjacent;
3. the scan steps of the last LAG chunks.

`kda_v2.schedule` is the same order in Python.

**Flags.** An int32 control buffer, zeroed by the caller before each launch (one memset: graph-safe):

- prep(c, h) done;
- scan steps of (c, h) done (0..NVB);
- steps of value block (h, vb) done (0..NC).

A finished item does a CTA barrier, then one thread's `atom.release.gpu`.

**Waits.** A waiting item spins with one thread's `ld.acquire.gpu` (Triton lowers `atomic_add(p, 0, acquire)` to
it), broadcasts through shared memory, then a CTA barrier.

- A scan step (c, h, vb) waits for prep(c, h), and for step c - 1 of (h, vb). It then loads S from `state_in` (c = 0)
  or `state_out`, runs `_kda_state`'s loop body once, and stores S to `state_out`.
- A prep(c, h) with c >= RING waits for every scan step of (c - RING, h), the chunk whose ring slot it overwrites.

**No deadlock.** Every wait is on an item with an earlier ticket:

- a step's chunk prep is in its own or an earlier group;
- the previous step is one group earlier;
- the slot's last readers are in group c - RING + LAG <= c - 1, since RING >= LAG + 1 is enforced.

A ticket is only taken by a running program. So whatever the number of resident programs (NCCL on the comm stream,
`CTAS` > the SMs), the program being waited for is running and finishes. `test_fused_schedule_waits_point_backwards`
checks this for the Python order, and the interpreter runs the kernel's own decode.

**L2 residency.** The ring is RING x H slots of 181 KB. It is ~149 KB touched a slot, since A / T live in a
per-program 48 KB scratch: RING 3 = 14 MB, RING 2 = 9.5 MB of GB10's 24 MB L2. Slots are rewritten while hot, so
dirty lines seldom reach DRAM.

**L1.** L1 is not coherent across SMs, and a program can read a slot it read before, when it held chunk c - RING. So
every load of data another program wrote (the prep's own reads, scan operands, S) uses `.cg` (L2 only). This is a
cache hint, not arithmetic: the compile test compares the prep to `_kda_prep` with `CG` off, and the fused kernel's
float ops.

**Memory ordering.** This is the pattern CUTLASS semaphores and `decode_v2`'s tickets already use on this GPU:
producer barrier + release, consumer acquire + barrier.

### What flash-linear-attention does (fla-org/flash-linear-attention, main at 79d12e8, 2026-09-28)

`chunk_kda_fwd` runs these kernels:

- `kda_gate_chunk_cumsum`;
- `chunk_kda_fwd_intra`: Akk / Aqk, the solve, and w / u / qg / kg;
- `chunk_gated_delta_rule_fwd_h`: the recurrence, grid (V / BV, N x H) with BV 32 / 64;
- `chunk_gla_fwd_o_gk`: the outputs, parallel over chunks.

The recurrence kernel keeps the state as 64-key register blocks `b_h1..b_h4`, with the dots chained over them. It
stores h at every chunk and v_new, and on Blackwell it is pinned to 2 warps ("a Blackwell tl.dot recurrence race",
until Triton 3.8).

What 0400 takes and leaves:

- **Taken:** value blocks adjacent in the grid, 32-wide; the 64-key split of the state and the chained dots.
- **Left:**
  - the separate chunk-parallel output kernel: it needs h stored per chunk (64 KB a chunk-head more traffic);
  - bf16 operands for the state dots: other bits.
- The 2-warp note is why the GPU plan runs 50 repeated runs plus runs on a busy GPU, and a 2-warp setting.

## 4. Estimate (arithmetic, not timed)

- **Split.** The state kernel's DRAM drops from ~65 to ~37 MB, and all 128 programs are resident (today: 2 rounds,
  the second a third full). State 334 -> ~170-220 us, so the call is 943 -> ~780-830 us. That is **-7..-11 us a
  token** (x 34 / 512), +1.0-1.6% at 24.5k.
- **Fused.** DRAM a call drops ~150 -> ~40 MB (inputs, outputs, state, norm).
  - The prep (~50% of its time DRAM-bound today) runs from L2: ~55-70 us an item, vs ~106 today, 256 items / 48
    programs.
  - Scan steps from L2: ~8-12 us, 512 items / 48.
  - No tail waves.
  - ~350-500 us + norm 41 against 943: **-20..-35 us a token.** That is +3-5% prefill: 1,471 -> ~1,515-1,545 tok/s at
    24.5k, and about the same at 98k (the recurrence cost is context-independent).
- Uncertainties:
  - L2 contention: the ring vs prep's streaming inputs, and the 0084 all-gathers on the comm stream;
  - the fused kernel's spills (1.1 KB a thread);
  - the spin-waits.
  - The bench decides between split and fused, and their settings.
- Memory: the fused ring + scratch + control is ~20 MB a rank; split uses `fast_kda`'s workspace. Negligible.

## 5. Offline verification (done)

Trees: 0001-0360 and 0001-0390 plus 0400. Python 3.14, torch 2.14 CPU, Triton 3.8.0 and 3.7.1 (the image's),
ptxas 12.9 for sm_121a.

- **`tests/test_kda_v2_compile.py`: 17 passed on 3.7.1, 17 passed on 3.8.** No GPU needed. It checks:
  - `_prep_item` (on `_kda_prep`'s grid) compiles to `_kda_prep`'s PTX instruction for instruction;
  - `_state2` bv 64 / 8 warps has `_kda_state`'s float ops and PTX float-instruction counts;
  - `_state2` at bv 16 / 32 / 64, 2 / 4 / 8 warps, with and without the K split: same elementwise ops, same dot
    chains, mma v2 m16n8 with kWidth 1, no reductions;
  - `_fused` (NVB 2 / 4, K split on and off):
    - the prep branch has the reference's 33 reduce / scan ops with the same full layouts, the same elementwise ops
      and the same dot chains;
    - the scan branch is as above;
    - PTX fma / ex2 / rcp / sqrt / cvt / mma counts equal prep + step, so no FMA contraction is gained or lost. Only
      a mul count differs, 32, from the mma layout's 2x replication of a 64-row tile on 8 warps;
  - `maxnreg` only changes the `.maxnreg` line.
- **`tests/test_kda_v2_interpreter.py`** (Triton interpreter, 2 heads): **95 passed (~6 min, incl. 8,192 rows)** on 0001-0390 + 0400 (an earlier revision, before the register-cap knob, also 95 passed on 0001-0360 + 0400).
  - The interpreter is made GPU-like:
    - bf16 casts round to nearest even;
    - `tl.dot` is an emulated mma chain (tf32 operands, one fp32 rounding per k-step of 8; ieee: one fma per k);
    - Triton's combine rewrite is modelled.
  - All BITWISE:
    - modes 1 and 2, 11 settings, at 1 / 63 / 64 / 65 / 130 / 200 rows and aligned and unaligned starts, plus
      2,048 and 8,192 rows;
    - a 448-row prompt at 128 cut into 1-4 calls on the 64-row grid == one call, and each cut's state == `fast_kda`'s
      (the snapshots);
    - `state_out` aliasing `state_in`;
    - determinism; mode 0 == `fast_kda`;
    - the ticket order's waits all point backwards.
  - Inputs keep a large state (slow channels per channel, small v). With the old test inputs (per-element random
    decay, state gone in a few rows) a swapped dot order was absorbed and went unnoticed. It is caught now.
  - Controls (`test_control_the_checks_see_chain_order`):
    - the scan redone in numpy from the workspace equals the kernel;
    - with the key halves swapped, it does not;
    - without the combine rewrite, the K split differs from `fast_kda`.
  - Mutations run by hand:
    - swapped K halves: state differs;
    - the fused step reading `state_in` at every chunk: 12 failures.
- The patch applies on 0001-0390 (engine.py hunk offset from 0370 / 0380).
- **Not checkable offline:**
  - the tensor cores' internal rounding inside one mma (the model uses exact in-step sums; the same instruction per
    element either way);
  - real concurrency in the fused kernel (the interpreter runs programs one after another);
  - timing.

## 6. Knobs

| knob | default | meaning |
| --- | --- | --- |
| `GLM53_TF_KDA_V2` | `0` | `0` fast_kda; `1` / `split`; `2` / `fused` (load-time; checked at load; ranks need not agree: same bits) |
| `GLM53_TF_KDA_V2_BV` | `32` | split: value rows a scan program (16 / 32 / 64) |
| `GLM53_TF_KDA_V2_WARPS` | `4` | split: warps a scan program (2 / 4 / 8) |
| `GLM53_TF_KDA_V2_MAXNREG` | `168` | split: register cap (0 = none; 64..255) |
| `GLM53_TF_KDA_V2_KSPLIT` | `1` | both: the scan's dots over two 64-key halves |
| `GLM53_TF_KDA_V2_FUSED_BV` | `64` | fused: value rows a scan step (32 / 64) |
| `GLM53_TF_KDA_V2_CTAS` | `0` | fused: programs (0 = one an SM, two with a cap <= 128) |
| `GLM53_TF_KDA_V2_LAG` / `_RING` | `1` / `3` | fused: scan chunks behind the prep / ring slots (RING >= LAG + 1) |
| `GLM53_TF_KDA_V2_FUSED_MAXNREG` | `0` | fused: register cap |

The knobs are left out of the NVMe session compat hash (`sessdisk.KNOB_SKIP_PREFIX`), because they are speed only. They are
in the calibration key, because timings change. `save_replay`, a non-default precision / bv / warps argument and
|lower| > 5.33 go to `fast_kda`.

## 7. GPU test plan (one window, ~60-75 min, production down ~45 min)

Image: every patch through 0400 (e.g. `glm53-tensorfold:b3`). With `GLM53_TF_KDA_V2` unset it serves what b2 serves:
0400 only adds a module and a dispatch.

1. **Unit tests, one GPU, server stopped** (~10 min). Run
   `scripts/run_tests_in_image.sh results/W10/tests -- tests/cuda/test_kda_v2_patches.py` plus the regressions
   `test_fastk_patches.py test_cindep_patches.py test_lean_patches.py test_b12x_patches.py test_session_disk_patches.py`.
   - **Gate: `test_kda_v2_patches.py` all pass.** Every mode and setting must be bitwise == `fast_kda` at 1-8,192 rows,
     aligned / unaligned, normal and large-state inputs; 50 repeated runs plus 20 with a busy side stream; resume ==
     fresh with every piece's state == `fast_kda`'s; aliasing; the engine entry.
   - A setting that fails must not be used; if the split's default fails, the mode stays off.
   - Also run `tests/test_kda_v2_compile.py` in the image (its Triton): 17 passed.
2. **Microbench** (~10 min): `python tests/cuda/bench_kda_v2.py 512 1024 2048 8192`, then
   `bench_kda_v2.py 512 --sweep`.
   - Record fast_kda's prep / state / norm (expect ~0.57 / 0.33 / 0.04 ms at 512) and each setting's ms, speedup and
     bitwise flag.
   - The last lines give the fastest bit-identical settings as env lines.
   - **Go on if the best is <= 0.85x of fast_kda at 512 rows** (-8 us a token).
3. **Engine gates** with the bench's best settings: `results/W8/load.sh` with `GLM53_TF_KDA_V2=<mode> ...`, then
   `results/W8/run.sh`. **Gates:**
   - reply sha **8794a3463259cc2f** (ab.py, both prompts);
   - `glmbench.py --suites exact` **10/10**;
   - `multiturn.py --modes batchexact` 4/4;
   - resume == fresh: a session prefilled cold, then the same conversation resumed from a slot, RAM and NVMe snapshot,
     gives the same reply. Use 0180 / 0250's `test_batch_sessions_patches` / `test_session_disk_patches` engine tests
     with the knob set, or two identical 24.5k multiturn conversations, one after a restart;
   - `/health` clean, no hang (watch for a stuck fused kernel: the 0150 stall detector).
4. **Prefill A/B** (~15 min):
   - `ab.py` 24.5k / 98k, knob off, on, off, on (two runs each);
   - the concurrent 1 / 4 streams to see decode unchanged;
   - Worth adopting at **>= +2% at 24.5k** with every gate green. Expected: split +1-1.6%; fused +3-5%.
   - If both modes pass, adopt the faster.
5. Optional: nsys of one prefill with the knob on, to replace the estimate in ROOFLINE rank 5.
6. Restore production from `config/prod.env`, with the knob only if adopted. Canary, watchdog.

Revert: unset `GLM53_TF_KDA_V2` (same image, same bits; no session store is invalidated).

## 8. Risks

- **Bits on the GPU.** The argument rests on mma chains being element-local and on Triton's rewrites being the ones
  the compile test sees. Step 1 is the arbiter, on the image's Triton.
- **Fused-kernel liveness.**
  - The ticket argument covers residency.
  - A bug in the flags would hang a prefill. The repeated-run test and the watchdog / stall detector catch it; the
    revert is the knob.
- **FLA's Blackwell `tl.dot` recurrence race note** (Triton < 3.8). The image has 3.7.1 and our default scan uses 4
  or 8 warps. `fast_kda` already runs 8 warps in production, and the repeated-run and busy-GPU tests look for exactly
  this.
- **L2 contention** with the 0084 overlap may shrink the fused gain. RING 2 halves the ring.
