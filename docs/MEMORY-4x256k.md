# Memory at 4 slots x CONTEXT=262144 (FP8 latent KV)

Per node (= per rank: TP=2, one rank a node) for the production candidate `GLM53_TF_BATCH=4`, `CONTEXT=262144`
(capacity C = CONTEXT + 8 = 262,152 slots, `engine.py:200`), `GLM53_TF_KV_DTYPE=fp8` (patches/0220), latent KV,
q4mse, DFlash2 drafter, fast + lean prefill. Source: vendor/TensorFold 2f8e514 + patches 0001-0210 (line numbers
are in the patched tree, `src/tensorfold/families/glm5_next/cuda/`).

Where the numbers come from:

- **[code]**: tensor shapes in the code times the real config (`hidden` 4096, 32 local heads, `kv_lora` 512, 34 KDA
  and 11 DSA layers, 288 experts, top-8 + 1 shared, vocab 154,880 split in two).
- **[meas]**: boot lines (`[boot] ... MemFree / MemAvailable`, `fastboot.py:79-105`) and memory logs:
  - S1: production, 524k, rows 8192;
  - the 2026-09-28 09:23 run: bf16, BATCH=4 at 262k, rows 4096, block 1024, the 0200 on-set, no store;
  - Y1: the same kind of load (`results/Y1/mem.log`);
  - Load A / B1 / B2 on the head node (`/tmp/cycle*.log`).
- **[est]**: derived from the two, or not attributable from the code.

The worker node (rank 1) is the binding node. It has 2.0 GiB less MemTotal (119.7 vs 121.7 GiB) and ends a load 1.9 GiB
below rank 0.

## Bytes a token a rank

What each slot reserves for every position of its capacity. The latent rows are replicated on both ranks.

| # | Allocation (code) | bf16 | FP8, 516 B rows (unpadded) | **FP8 as patch 0220 stores it (528 B rows)** |
| ---: | --- | ---: | ---: | ---: |
| 1 | Latent KV, 11 DSA layers (`State.kc`, `forward.py:183`, `latent.py:97-103`) | 11,264 | 5,676 | 5,808 |
| 2 | MTP head latent (`State.mtp_kc`, `forward.py:191`) | 1,024 | 516 | 528 |
| 3 | MTP index keys + gates, 2 x [C, 128] bf16 (`forward.py:201`; the model layers' are 0065 rings) | 512 | 512 | 512 |
| 4 | Pool keys, 12 x [C/4 + 2, 128] bf16 (`forward.py:201`) | 768 | 768 | 768 |
| 5 | 0050 `LongScratch`, 2 sets x 8 rows x C/4 x 12 B (`decode.py:115-119`; once an engine, not a slot) | 48 | 48 | 48 |
| | **Slot 0 (rows 1-5)** | **13,616** | **7,520** | **7,664** |
| | Each extra slot (rows 1-4; `Buffers` are shared) | 13,568 | 7,472 | 7,616 |
| | Session store row (`sessions.layout`, `sessions.py:517-544`; the log says "13.25 KB a token") | 13,568 | 7,472 | 7,616 |

13,616 -> 7,520 B is the unpadded figure (512 e4m3 bytes + a 4-byte scale). Patch 0220 (`latent.py`, `ROW8 = 528`)
pads each row to 528 bytes so rows stay 16-byte aligned (128-bit loads in the sparse gather): **13,616 -> 7,664 B a
token a rank (0.56x)**. Over 4 slots at 262k the padding costs 0.14 GiB. The tables below were computed at 516 B; at
528 B subtract 0.14 GiB from every FP8 cell (4 slots) and count a full store as holding 1.9% fewer tokens. The
recommended cells then read 9.7 / 12.5 GiB (with a 4 GiB store) and 9.4 / 12.2 GiB (without).

Capacity-scaled caches for 4 slots at C = 262,152 **[code]**:

| | bf16 | FP8 | saved |
| --- | ---: | ---: | ---: |
| slot 0 + 3 extra slots | 13.26 GiB | 7.31 GiB (**7.45** at 528 B) | **5.95 GiB a node** (5.81 at 528 B) |
| a 256k context in the session store | 3.31 GiB | 1.82 GiB (1.86 at 528 B) | 1.8x more tokens a GiB |

## Everything else a node holds

| Item | Size | Source |
| --- | ---: | --- |
| OS, services, docker, non-reclaimable (MemTotal - MemAvailable at container start) | ~5.2 GiB (r1), ~5.2 (r0) | [meas] |
| CUDA context + NCCL init | 2.1 GiB | [meas] |
| Weights: 82.2 GB of files = 76.6 GiB, plus loader/allocator overhead | 78.1-78.3 GiB | [meas] |
| DFlash2 drafter weights (0.65 GiB) + its 4,096-slot ring (40 MiB, `dflash2.py:252-262`) | 0.7-0.8 GiB | [meas]/[code] |
| DFlash2 CUDA graphs; calibration | 0.5; 0.1-0.2 GiB | [meas] |
| `Buffers` x 2 (model + MTP head) at **LEAN_BLOCK** rows, not ROWS_MAX (`decode.py:103-112`): exl3 split-K `z` 1.13 MiB a row, qmm `sk` 0.5 MiB a row, latent scratch (`latent.py:141-162`: 0.55 GiB at 1024 rows, 0.25 at 512), logits 0.15 MiB a row, ... | 2.99 + 2.95 GiB at 1024; 1.46 + 1.44 at 512 | [code] |
| Slot 0 `State` rows at LEAN_BLOCK: KDA replay scratch (`kda.py:45-55`, 1.56 MiB a row) + projection rows (`forward.py:178`, 0.80 MiB a row) | 2.41 GiB at 1024; 1.21 at 512 | [code] |
| Window buffers in total (both sets + slot 0 rows + fast_kda workspace; `PREFILL-ANALYSIS.md` table) | **8.50 GiB at block 1024; 4.19 at 512** | [code] |
| Lean set at ROWS_MAX (`lean.py:93-145`, 397 KiB a row) | 3.10 / 1.55 / 0.78 GiB at 8192 / 4096 / 2048 | [code], matches the boot line |
| Per-slot fixed state: `rec` 136 MiB (2 x 34 x 32 x 128 x 128 fp32, `forward.py:176`), conv + ring tail 2.5 MiB, index rings 11 MiB (2,048 rows at block 1024, `sparse.py:35-39`), 8-row proj/replay rows 20 MiB (`batch.py:984-1000`), DFlash2 ring 40 MiB (`batch.py:533-545`), backlogs 1.5 MiB, MTP + drafter graphs ~16 MiB | **0.24 GiB a slot at load** | [code]; the graphs are [meas]: 3.587 vs 3.571 GiB estimated |
| Kept snapshots, up to 2 a slot: `rec` of the live parity 68 MiB + conv 2.5 + DFlash2 window 20 MiB (`decode.py:327-331`) | ~0.18 GiB a slot at runtime | [code] |
| So "~0.4 GB a slot" | 0.24 at load + 0.18 at runtime = 0.42 GiB: **confirmed** | |
| Batch round buffers `bproj` / `bkout` (32 rows, `batch.py:880-881`) | < 2 MiB | [code] |
| Engine main + MTP graphs, Triton/cuBLAS workspaces, allocator slack (the engine boot step minus the items above) | 0.5-1.4 GiB | [est] |
| Prefill transients, held by the caching allocator afterwards: latent sparse partials of one sub-block, 5 x r x 32 x 512 x 4 B (`latent.py:524-526`: 320 MiB at r = 1024, 160 at 512); blocked selection <= `GLM53_TF_SELECT_MB` 256 MiB + top-k (`sparse.py:416-438`); router partials 8 x R x 288 x 4 B (`glue.py:653`: 18 MiB at 2048 rows, 72 at 8192) | ~0.6 GiB at block 1024, ~0.45 at 512; bounded, not context-scaled | [code] |
| Session store: pages allocated lazily up to `GLM53_TF_SESSION_GIB` (`sessions.py:96`), while CUDA-free - n >= reserve (`sessions.py:714-721`; reserve = max(`SESSION_RESERVE_GIB` 2, `BATCH_ADMIT_GB`), `batch.py:922`) | 0 to SESSION_GIB | [code] |

Measured check of the engine step (bf16, 262k, rows 4096, block 1024; the 09:23 boot): 14.0 GiB on both ranks, against
3.32 + 8.50 + 1.55 + 0.15 = 13.5 GiB from the code. Its 3 extra slots: **10.76 GiB** (3.587 GiB a slot). After load,
MemFree / MemAvailable: r0 6.9 / 9.6, r1 5.9 / 7.7 GiB. After the canary warm-up: r0 5.9 / 8.6, r1 5.1 / 6.8 GiB.

## Q2: are the slots allocated up front?

**Yes.** At load, every slot's caches are allocated at full capacity with `torch.zeros`, so their pages are committed.
Nothing is sized lazily at admission.

- `Batcher.__init__` (`batch.py:856-869`) builds `State(w, e.st.capacity, ...)` for each extra slot (`batch.py:984-1000`).
- `State` allocates the latent caches at capacity (`forward.py:183`, `forward.py:191`), plus the MTP index keys and
  pool keys (`forward.py:201`).
- Each slot also gets its DFlash2 ring (`batch.py:538-539`).

**Load-time rule** (`batch.py:861`): an extra slot is added while
`free - per_slot - store_budget >= GLM53_TF_BATCH_RESERVE_GB` (default 4) on both ranks.

- `free` is `cudaMemGetInfo` free + allocator-cached (`batch.py:958-962`). On GB10 that is **MemFree**, not
  MemAvailable (`docs/MIA-AUDIT.md` item 5, `scripts/serve.sh:30-35`), so page cache counts against it.
- `per_slot` is `_slot_estimate` (`batch.py:964-982`): 3.571 GiB at 262k bf16, 2.08 GiB FP8.
- `store_budget` is `SESSION_GIB`, but only with `GLM53_TF_BATCH_SESSIONS=1` (`batch.py:850`). With `BATCH_SESSIONS=0`
  (the default), `SESSION_GIB` is ignored in batch mode and no store is built (`engine.py:397-400`; the 09:23 run logs
  exactly that).

**Admission** (`batch.py:1213`, `batchplan.py:274-278`): while other requests run, a request waits while
`free - (store budget - store used) < GLM53_TF_BATCH_ADMIT_GB` (1). The unused store budget is therefore held back at
admission. That is 0180's "set aside at load and at admission". The store itself grows only while CUDA-free stays above
its reserve.

Admission does not reserve prefill transients or graph growth; `ADMIT_GB` is the only margin. Neither check looks at
MemAvailable, and neither resizes a slot.

**"Only 1 slot fit at 262k"** (B1, the head node `/tmp/cycleB.log`): MemAvailable after load was 20 / 18 GiB, the same as
single-stream Load A, so no extra slot was made. The rule needs about 3.57 + 12 + 4 = 19.6 GiB of MemFree before the
first extra slot. That reproduces both B1 (1 slot at 262k) and B2 (2 slots at 131k, 19 / 17 GiB after load) when a
12 GiB store budget is counted, so B1/B2 very likely ran with `BATCH_SESSIONS=1`, `SESSION_GIB=12`. The 3.57 is
`_slot_estimate`, printed in GiB labelled "GB". Y1 had the store off and still needed `drop_caches` before 4 slots fit,
because page cache lowers MemFree.

## Q3: worst case, 4 slots at 256k, store full, a max-size prefill chunk

Model (rank 1, MemAvailable):

    worst = after-load - warm-up residue - dynamic - SESSION_GIB

- after-load: the 09:23 anchor, 7.7 GiB, plus the code deltas: FP8 +5.95; ROWS_MAX 2048 +0.78 or 8192 -1.55; LEAN_BLOCK
  512 +4.31.
- warm-up residue: 0.9 GiB [meas].
- dynamic: 4.0 GiB [meas]. Y1 fell from 6 to 2 GiB on r1 (7 to 4 on r0) during 10 minutes of 4 streams at 25-35k,
  with no store. From the code, ~1.5 GiB of it is prefill transients (0.6), snapshots (0.7) and a piece resume
  snapshot. The rest [est] is lazily captured multi-slot graphs (up to `GLM53_TF_BATCH_MAX_GRAPHS` 256) and host memory.
- Rank 0 ends 1.9 GiB higher.

In batch mode a prefill chunk is at most one piece: `GLM53_TF_BATCH_PIECE` = 2048 tokens with rows `auto`
(`batchplan.py:102-127`, `batch.py:1379-1382`, `pfgrid.chunk_rows`). So **ROWS_MAX above BATCH_PIECE is never used
under batching**: at ROWS_MAX 8192 the lean set holds 2.3 GiB it never uses. The max-size-chunk transients are
already inside "dynamic".

Rank 1 worst-case MemAvailable (GiB); target >= 8:

| KV | LEAN_BLOCK | ROWS_MAX | after load | store off / 0 | SESSION 4 | 6 | 8 | 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 1024 | 4096 (the 09:23 run) | 7.7 | 2.8 | -1.2 | - | - | - |
| bf16 | 512 | 2048 | 12.8 | 7.9 | 3.9 | 1.9 | -0.1 | -4.1 |
| FP8 | 1024 | 8192 | 12.1 | 7.2 | 3.2 | 1.2 | -0.8 | -4.8 |
| FP8 | 1024 | 4096 | 13.7 | 8.8 | 4.8 | 2.8 | 0.8 | -3.2 |
| **FP8** | **1024** | **2048** | 14.4 | **9.5** | 5.5 | 3.5 | 1.5 | -2.5 |
| FP8 | 512 | 4096 | 18.0 | 13.1 | 9.1 | 7.1 | 5.1 | 1.1 |
| **FP8** | **512** | **2048** | 18.7 | 13.8 | **9.8** | 7.8 | 5.8 | 1.8 |

A negative entry means the store cannot fill: it stops at its reserve, and prefill and graphs then run out of memory.
Rank 0 is +2.8 in every cell: it starts 1.9 higher and fell 1 GiB less in Y1 (FP8 / 512 / 2048 / SESSION 4: 12.6).

Recommendation:

- **With the store** (only once 0180's `GLM53_TF_BATCH_SESSIONS=1` passes its GPU tests): `GLM53_TF_KV_DTYPE=fp8`,
  `GLM53_TF_PREFILL_ROWS_MAX=2048` (= `BATCH_PIECE`: nothing lost), `GLM53_TF_LEAN_BLOCK=512`,
  `GLM53_TF_SESSION_GIB=4`. Worst case 9.8 / 12.6 GiB (r1 / r0; 9.7 / 12.5 with 0220's 528 B rows). A 4 GiB FP8 store holds ~570k tokens of pages, or ~8 sessions of
  32k (each ~0.5 GiB with its prompt snapshot and 2 marks of ~91 MiB).
- **Without it** (`BATCH_SESSIONS=0`: `SESSION_GIB` is ignored): FP8, `ROWS_MAX=2048`, `LEAN_BLOCK=1024` (no speed
  risk). Worst case 9.5 / 12.3 GiB (9.4 / 12.2 at 528 B).
- Block 512 costs an estimated 2-4% of prefill: non-expert weights are read per sub-block (+~16 us a token over 1024),
  and there are twice the all-gathers, half of them hidden by 0084. Exact chunks cap at 512 rows. Unmeasured.
- Keep `GLM53_TF_BATCH_PIECE=2048`. At 4096 (with ROWS_MAX 4096) prefill gains a few % (est.), for -0.78 GiB and
  ~2x longer decode stalls (Y1: 2.4 s a piece).

Guards:

- `GLM53_TF_BATCH_RESERVE_GB=11`. Load keeps free >= SESSION + 11 of MemFree, which covers the 4.9 GiB of warm-up +
  dynamic and leaves ~6 GiB MemFree (about 8 GiB MemAvailable). Expected load margin on r1: 19.0 - 2.1 - 4 - 11 = 1.9 GiB.
  If page cache costs the 4th slot, use 10.
- `GLM53_TF_BATCH_ADMIT_GB=2`: one admission's transients + snapshots are ~0.9 GiB; the default 1 is too thin on GB10.
- `GLM53_TF_SESSION_RESERVE_GIB=6`: the store never grows below 6 GiB CUDA-free.
- Start with `MEM_GATE_GIB` set and `MEM_GATE_DROP_CACHES=1`. Page cache (`Cached` is 4.2-4.7 GiB on both nodes now)
  comes out of MemFree, and on GB10 CUDA does not reclaim it in time.

## Q4: the 0190 knobs

- **`attn_bm32`: no extra DRAM.**
  - It only changes the tile `BMQ` (16 -> 32 queries) of the launches (`latent.py:571-573`): grid
    `cdiv(R x H, bm)` / `cdiv(H, bm)`.
  - The partials `po`/`pm`/`pl` are sized nch x R x H x L whatever the tile (`latent.py:147-149` for the dense scratch,
    `latent.py:524-526` for the per-call sparse partials: 320 MiB at a 1024-row sub-block, 160 at 512, the same as
    bm16).
  - The "extra scratch" is on-chip **shared memory** (the 32-query tile needs ~128 KB on the test shapes; GB10 allows
    ~101 KB a block), not DRAM. It adds a few MB of kernel binaries.
  - T8: `test_glue_patches` 79/80 pass; the one failure is `latent_tc`. Fits under the recommendation.
- **`mtp_window`: no extra memory.**
  - It zeroes the head's rows below `lo` in place (`pfglue.py:150-170`) and absorbs fewer rows (`pfglue.py:173-195`),
    so it does less work.
  - The non-memory issue is decode right after a 112k prompt with both knobs on: 56 tok/s against 81 (Z2). Check before
    enabling.
  - In batch mode each 2048-token piece keeps its own last W positions, so it saves less there.
- **The Z2 OOM** (rank 0 aborted during an `attn_bm32` repeat; NVRM OOM 01:49-01:50) was the configuration, not the
  knob.
  - At 524k, rank 1 idled at MemFree 13.4 GiB with an empty store (S1).
  - A 12 GiB store fills until only `SESSION_RESERVE_GIB` = 2 GiB of CUDA-free remains. The X1/Z2 benchmark's unique
    28k/112k prompts store ~1.4 GiB each plus marks.
  - That leaves ~2 GiB for prefill transients (~0.6), snapshots, graphs, host memory and page cache growth, which CUDA
    cannot reclaim in time.
  - The same production configuration (image z, 524k, `SESSION_GIB=12`) hit NVRM OOM again on 2026-09-28: the worker node
    07:00-07:01, the head node 07:04. Its containers ended at 07:04:44 (r0 exit 0, r1 137). Their logs were replaced by the
    09:23 run, so this is not confirmed from the engine log.

## What the GPU window must measure

1. An FP8 boot at 4x262k, rows 2048, block 512.
   - Expected: `latent KV cache (fp8 rows): 7.4 KB a token a rank` (7,616 B: slot 0 without the 0050 scratch, 528 B rows), and an engine step of ~7.4 GiB (bf16 /
     4096 / 1024: 14.0).
   - Expected: `batching 4 requests: 3 extra ... 6.3 GB` (10.76 bf16), and MemAvailable after load ~18.7 / 20.6 GiB
     (r1 / r0).
   - If any value is off by more than 1 GiB, recompute the table.
2. Prefill A/B of LEAN_BLOCK 512 against 1024 (28k / 112k single, and one 35k piece-wise prefill beside 3 decoders).
   Adopt 512 at <= 3% loss.
3. A 30-60 min soak with 4 streams, one slot past 200k and the others 30-130k, with the store on if 0180 is fixed.
   - Log MemFree / MemAvailable / Cached every 5 s on both nodes.
   - Confirm dynamic <= 4 GiB and no creep. The unattributed ~2.5 GiB is the graph pool: check it against the number
     of captured keys, or cap it with `GLM53_TF_BATCH_MAX_GRAPHS`.
   - Record the minimum during a 2048-row piece at full context.
4. `attn_bm32` / `mtp_window` on that load: the memory delta (expected 0) and decode after a long prompt with
   `mtp_window`.
5. Page cache during service. Whether `drop_caches` at start is needed for the 4th slot at `RESERVE_GB=11`.
