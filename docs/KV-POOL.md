# Shared latent KV pool (patches/0290, `GLM53_TF_KV_POOL_TOKENS`)

Status (2026-09-28): written and tested offline, no GPU. That means the Triton CPU interpreter, CPU torch, and real
sm_121 compiles of every touched kernel. The GPU tests and the window plan below are for the GPU owner. Off by default.
Production is unchanged.

## What it does

Today each batch slot allocates its full capacity (CONTEXT + 8 = 262,152 positions) of every capacity-sized cache at
load. At 7,616 B a token a rank (FP8 rows, `docs/MEMORY-4x256k.md` rows 1-4) that is 1.86 GiB a slot, and 7.44 GiB a
node for 4 slots. It is paid whether a slot holds 1k tokens or 262k, and no request can pass 262k.

With `GLM53_TF_KV_POOL_TOKENS=N`:

- Those caches of every slot live in one set of pool tensors of N rows, cut into pages of 256 tokens.
- Each slot maps logical page (position // 256) to a physical page through its own page table.
- A slot grows to the engine's capacity (`CONTEXT`, up to ~1M), as long as the pages of all slots fit in the pool.

Scope: this is the full first stage, not a subset. Every capacity-scaled per-slot cache is paged:

| cache (per slot, per rank) | bytes a token | today | with the pool |
| --- | ---: | --- | --- |
| `State.kc`: 11 DSA latent caches, 528-B FP8 rows (0220) | 5,808 | [C, 528] x 11 | paged, page = 256 rows |
| `State.mtp_kc`: the MTP head's latent cache | 528 | [C, 528] | paged |
| MTP head's index keys + gates (`State.index[-1]`), bf16 | 512 | 2 x [C, 128] | paged |
| pool keys, 12 x [C/4 + 2, 128] bf16 | 768 | 12 x [C/4+2, 128] | paged, page = 64 pool rows |
| model layers' index keys / gates (0065 rings) | 0 | ring, window-sized | unchanged (per slot) |
| KDA state, conv, DFlash2 ring, window rows | 0 | fixed a slot | unchanged |

Rows keep their format: 16-byte aligned 528-B FP8 rows (0220); a page of a latent cache is 256 x 528 B = 132 KiB.

## Knobs

| knob | default | meaning |
| --- | --- | --- |
| `GLM53_TF_KV_POOL_TOKENS` | `0` (off) | pool size in tokens, rounded up to whole pages. Needs `GLM53_TF_LATENT_KV=1`. Both ranks must agree (checked at load). |
| `GLM53_TF_KV_POOL_PAGE` | `256` | tokens a page: a power of two >= 256, so a session-store page (256) never straddles two pool pages. |
| `GLM53_TF_KV_POOL_SLACK` | `64` | tokens reserved past prompt + max_tokens (verify / pad windows and MTP drafts write up to ~16 past the committed end). |
| `GLM53_TF_KV_POOL_CHECK` | `0` | 1: after every batched round, check the null page is still all zeros (tests, the GPU window). This costs 16 small reductions a round. |
| `CONTEXT` (serve.sh `--context`) | 262144 | with the pool this is the per-slot maximum ("slot max context", e.g. 1048576). There is no separate `GLM53_TF_SLOT_MAX_CONTEXT`: the engine's capacity is already threaded from `CONTEXT` everywhere. |

`GLM53_TF_KV_POOL*` is excluded from 0250's compat hash, because it decides where rows live, never their bits. NVMe
sessions survive turning the pool on or off, or resizing it (same image).

## Design

### Pool, pages, tables (`kvpool.py`)

- `KvPool` (device side of `PoolBook`) is built in `decode.Engine` before any `State`, stored as `w.meta["kv_pool"]`.
  - Physical tensors, one per cache family (11 latent, MTP latent, MTP index k / g, 12 pool keys), are created with
    `torch.zeros` at `(npages + 1) x page` rows. They are committed at load, as today's caches are.
  - The extra page is the **null page**. Unmapped table entries point at it, so a stray read reads zeros, never another
    slot's rows.
- `SlotPages` holds one slot's table. It has a host list and a device int32 table of `ceil(capacity / page)` entries
  (4,097 at 1M: 16 KiB), plus a `quota` (the admission's reservation).
  - `ensure(end)` maps pages up to position `end` from the allocator.
  - `truncate(keep)` returns the pages past `keep`.
  - The allocator hands out the lowest free page first. Rank 1 performs the same operations in the same order, so its
    pool stays identical to rank 0's (it is not required for correctness, but it keeps divergence loud).
- `Paged` is the logical view the engine sees in place of a cache tensor. `shape[0]` is the slot's logical rows, and
  `dtype` / `device` / `data_ptr` are the physical tensor's.
  - Kernels take `(phys, table, shift)` (`kvpool.kernel_args`).
  - Host code slices it: a slice inside one page is a real view, as a slice of the contiguous cache is. A slice across
    pages raises, so a missed call site fails loudly instead of copying the wrong rows. `segments`, `gather`, `write`
    and `zero_rows` work across pages.

### Where rows are mapped (the "reserve / allocate" points)

| site | what |
| --- | --- |
| `forward.check_room` | every forward, verify window, MTP step, lean chunk and batched round calls it before staging: `ensure(pos + R)` (`mtp_len + n` for the head). A no-op except once every 256 tokens (then a tiny H2D copy of the new table entries). |
| `graphs.Graphs.__init__` | warm-up / capture steps write rows 0..R-1 without `stage`: those pages are mapped first. |
| `SessionStore.restore` / `prefetch` (0110/0180/0250) | the entry's positions are mapped before rows are copied or read in. |
| `pfglue.zero_mtp_rows` (0190 `mtp_window`, off in prod) | mapped, then zeroed page by page. |

Graph safety: the kernels read the table at run time. CUDA graphs (0050 long-context, 0120/0200 batched rounds, the
MTP graphs) hold the table's and the pool's addresses, which never change. Updating table entries between replays is
exactly what `ensure` / `truncate` do; `test_gpu_graph_replay_after_pages_move` checks a replay after every page moved.

### Every reader and writer (kernel indirection)

A logical row r becomes `table[r >> shift] << shift | (r & (page_rows - 1))`. This is `_prow` in `latent.py` and
`sparse.py`. It uses the same mask as the access it serves, with `PSH` constexpr; `PSH` = 0 (pool off) is pruned at
compile time.

| kernel | reads / writes | where the table enters |
| --- | --- | --- |
| `latent._lwrite`, `_lwrite8` | latent row writes (bf16 / FP8 + scale) | the row index, one scalar table load a row |
| `latent._lrows` (used by `_lchunks` dense, `_lsparse_chunks` sparse, `b12x_attn._lsparse_one`) | latent row gathers | each gathered row: TOK -> table -> row |
| `sparse._index_write` | index keys / gates (MTP head's: paged; model layers' rings: not) | the ring slot `(pos + r) % NR`, then the table |
| `sparse._pool_keys` | reads 4 key / gate rows, writes a pool key | keys / gates through their table (or ring), the pool key through the pool-key table (shift - 2) |
| `sparse._scores` (decode, 0050's device path), `_scores_rows` (0065 row blocks) | pool-key tiles of 64 | one table entry per 64-pool tile (a tile is exactly one 256-token page) |

The same list, host side:

- **Session store (0110/0180):** pages copied between a slot and the slabs. With pool page == store page this is one
  `index_copy_` / `index_select` per cache per run; with larger pool pages, one copy per store page. The tail past the
  last full page may span pages (a fast grid's `key_end` rounds up to the grid), so it goes through
  `Paged.gather` / `write`.
- **NVMe tier (0250):**
  - Packing and reading pages works one pool page per copy.
  - Tails are gathered on write.
  - On read, tails are staged in a temporary and written after the read is checked, so a failed read never
    half-writes the slot.
- **Snapshots:** `decode.Snapshot` never held attention rows (only KDA state, conv, pending MTP rows, the drafter
  window), so they are unchanged. `State.clone` refuses a pooled state (tests only; nothing in serving clones).
- **Load-time estimates:** `kv_bytes_per_token` counts a paged row. The batcher's per-slot estimate skips paged caches,
  because they are allocated once in the pool.

Not paged (not capacity-scaled): the model layers' 0065 index rings, 0050's `LongScratch` (scores / keys for `C/4`
pools per scratch row: 48 B per capacity token, **once an engine**), and the DFlash2 ring.

### Bit-exactness

The kernels compute the same arithmetic on the same row values in the same order. Only the addresses differ, and
masked lanes are masked for the table load as well. So outputs are byte-identical to the unpaged path. Evidence
(offline):

- **Interpreter:**
  - `tests/test_kvpool_interpreter.py`: every paged kernel equals the contiguous one bit for bit (`torch.equal`).
  - Setup: scrambled page tables, unmapped pages filled with NaN / 0xFF junk (any stray read poisons the result),
    windows straddling page bounds, bf16 and FP8, BM 16 / 32.
  - Kernels covered: `latent_write`, dense, sparse, b12x one-pass, `index_update` (full and ring), and all three
    selection paths.
  - Control: with the mapping planted wrong, 11 of 11 kernel tests fail.
- **Compiled:** `tests/kvpool_ptx.py` compiles the 16 touched kernel variants for sm_121 without a GPU. With the pool
  off, all 16 give PTX identical to the tree without 0290 (debug lines stripped), i.e. the pool-off path is the same
  machine code. Every paged variant compiles.
- **Engine logic:** `tests/cuda/test_kv_pool_patches.py` (host part, CPU) runs 0180's hostile fake model through the
  real `Batcher` and `SessionStore` (and 0250's `DiskTier`) with every slot on a shared paged pool. Every reply and every
  slot's whole state at the end of each request equals a fresh prefill + serial decode on contiguous caches. This covers
  exact / fast prefill, 256 / 512-token pages, roomy / tight pools (waits, spills, fragmented tables), and restores
  across slots and from disk. A 64-run multi-seed stress (2-4 slots, 5-10-page pools) was byte-exact every time, with
  the null page clean after every round and no page leaked.
- **GPU (to run):** the same checks on the real shapes with compiled kernels, and the engine with the pool vs without
  on the synthetic checkpoint (below).

## Pool exhaustion policy

Goals: never OOM, never corrupt, and an admitted request always finishes.

1. **Load:** the pool is allocated once (`torch.zeros`, committed), before the slots. With the pool the slot rule
   (`BATCH_RESERVE_GB`) only counts a slot's fixed state (~0.24 GiB at load), so the slots fit as before, or more
   easily.
2. **Admission** (rank 0, `Batcher._plan`; the decisions travel to rank 1 in the round plan). The request reserves
   `need = ceil((prompt + max_tokens + SLACK) / page)` pages, capped at the slot capacity. That is everything it can
   ever write.
   - `available = free pages - sum over running requests of (reserved - mapped)`.
   - If `need <= available`, it is admitted (reservation set in `_admit` on both ranks).
   - Otherwise **spill** idle slots, least recently admitted first, and the slot the request would resume in last
     (`batchplan.pool_spills`). A spill returns all of an idle slot's pages. Its sessions are already in the session
     store: 0180 saves every prompt, mark and exact-reply snapshot when it is made, and 0250's write-through tier also
     puts them on NVMe. So a later turn resumes from the store instead of from the slot. The slot's own kept
     snapshots and live-page map go with the pages. Spills run first in the round, before the admissions that need
     them (`Batcher._spill`, both ranks).
   - If even every idle slot's pages would not cover it, the request **waits** in the queue, FIFO: later small
     requests do not overtake it (`counts["pool_wait"]`). Running requests always finish (their pages are reserved),
     so it is eventually admitted.
   - A request that needs more than the whole pool is **refused** at once with a 400-style ValueError
     (`counts["pool_refused"]`) and does not block the queue.
3. **Before each prefill piece and each decode step:** `check_room` maps pages from the request's own reservation. It
   cannot fail. The one exception is a position past the reservation, which only a slack mistake could produce. That
   case takes only pages nobody reserved (counted in `SlotPages.over`), and raises `PoolExhausted` (a ValueError)
   rather than touch anything else.
4. **When a request ends** (finished, cancelled, background-preempted, or failed): the slot keeps the pages of its
   committed rows (what its kept snapshots and the store's live-page map refer to), returns the rest, and its
   reservation ends. An idle slot's pages are cached state, reclaimed by spills when needed.
5. **Admission within a slot:** the slot's pages past the resume point are returned before it is written. They are
   dead: the request rewrites them.

Why no corruption: a slot writes only rows below its mapped end (`ensure` precedes every write). A page is never on
two tables. Unmapped entries point at the null page, which `GLM53_TF_KV_POOL_CHECK=1` verifies every round.

Limits of this policy (stage 2 material):

- The reservation is up front. A client asking `max_tokens` = CONTEXT reserves the lot; the server's `MAX_TOKENS`
  (32,768 in prod) bounds the usual case.
- Running requests are not preempted for pages. Background requests step aside for slots (0120), not for pages.
- A slot's kept snapshot that the store skipped (RAM budget) with no NVMe tier is lost on spill. That costs a cold
  prefill later (slower, still exact).
- FIFO head-of-line: one huge waiting request holds back smaller ones behind it.

## Memory (per rank = per node; FP8 528-B rows)

| | tokens | cache bytes | other capacity-scaled |
| --- | ---: | ---: | --- |
| today: 4 slots x C = 262,152 | 4 x 262,152 | 4 x 1.859 = **7.438 GiB** | LongScratch 12 MiB |
| pool 1,048,576 (+ null page), CONTEXT 262,144 | any split of 1,048,576 | **7.439 GiB** | tables 64 KiB, LongScratch 12 MiB |
| pool 1,048,576, CONTEXT 1,048,576 | any split, one request up to 1M | 7.439 GiB | LongScratch **48 MiB** (+36), tables 64 KiB |

Per family at 1M: latent 5.67 GiB, MTP latent 0.52, MTP index keys / gates 0.50, pool keys 0.75.
1 GiB of pool = 140,985 tokens.

- **A pool of 1,048,576 tokens costs what today's 4 x 256k costs (+0.04 GiB with CONTEXT 1M).** Nothing else in
  `MEMORY-4x256k.md` changes, so MemAvailable after load, the warm-up residue and the dynamic 4 GiB carry over.
- **What fits:**
  - At equal memory: 1,048,576 tokens, the recommendation.
  - The binding node (the worker node) has an estimated long-uptime floor of ~9.3 GiB with the 2 GiB store (RESULTS, "4 x 256k
    batch production"; target >= 8). ~1.3 GiB more pool (~183k tokens, **~1.23M total**) would still clear 8, with no
    margin. Not recommended before the stress run below.
  - Or shrink the pool for headroom: 786,432 tokens (768k) frees 1.86 GiB, e.g. for `SESSION_GIB` 2 -> 4.
- bf16 KV would need 13.25 GiB for a 1M pool: FP8 (0220) is required.
- More slots become cheap: a slot is its fixed state only, 0.24 GiB at load + ~0.18 GiB of snapshots at runtime. 6
  slots is ~+0.8 GiB against 4 (`n * MAX_ROWS <= rows` allows up to 64 at `LEAN_BLOCK=512`). Unmeasured; round cost
  grows with rows.

## Expected performance cost

- **Sparse latent attention** (decode, verify, prefill past 2,051):
  - Each gathered row gets one extra dependent 4-byte load from a <= 16 KiB table (L1/L2-resident): TOK -> table ->
    528-B row, where it was TOK -> row.
  - Estimate: 0-3% of that kernel. It is ~9% of a 112k prefill (RESULTS W-profile: 9.7 s of ~110 s), so **+0-0.3% of
    prefill** and a similar share of a decode step.
  - If measured higher, stage 3 translates each row's token list once per step (`_expand`) instead of per head tile.
- **Dense attention:** a 32-row tile lies in one page. One table entry is broadcast; ~0.
- **Index scoring:** one entry per 64-pool tile (16 KiB of keys); ~0.
- **Writers:** one scalar table load a row; ~0.
- **Host:** `ensure` is a compare per staged window. A page is mapped every 256 tokens (one ~10-20 us H2D copy: ~8 per
  2,048-token piece, once every ~30-60 decode rounds); ~0.
- **Session restores (RAM):** one gather / scatter per cache per run, as before plus one small index copy. The NVMe
  loader copies a pool page at a time (more, smaller copies): maybe +10-20 ms on a 0.4 s 40k restore. Unmeasured.
- **Admission:** only when the pool is short do requests wait or idle slots get spilled. Today's equivalent was "the
  context does not fit": nothing ran.

## Risks

1. **GPU-only unknowns.** The paged kernels were compiled for sm_121 and interpreted on CPU, but never launched on a
   GB10. Launch-time specialization of `PT=None` has the same form as the compiled check. Watch for: register pressure
   in the FP8 sparse kernels (the extra int64 row math), and 0240's one-pass kernel (off in prod).
2. **A missed writer.** Any cache write without a preceding `ensure` would land in the null page. `KV_POOL_CHECK=1` in
   the window catches it (RuntimeError, `/health` fatal). All known write paths go through `check_room` / `Graphs` /
   the store; the GPU engine tests exercise prefill (exact, lean), decode, MTP, drafts, batching, graphs, restores and
   disk.
3. **Reservation behaviour.** `max_tokens` drives it. OpenAI clients that omit it get the server's 32k. A 1M prompt
   with a big `max_tokens` can be refused if prompt + max_tokens + 64 > pool.
4. **CONTEXT=1M side effects** (not the pool's):
   - more long-context graph buckets (9 vs 7; capped by `GLM53_TF_LONGCTX_MAX_GRAPHS`);
   - a new calibration identity (one re-measure);
   - decode past ~500k scores 128k+ pools a layer and step (slower decode at extreme lengths, as `MEMORY-1M.md`
     notes);
   - a 1M prefill takes ~14 min at ~1.25k tok/s, during which other slots decode at the fair share.
5. **Spill vs store budget:** with a 2 GiB RAM store and the NVMe tier on (prod), spilled sessions come back from
   NVMe (~0.4 s at 40k). Without the tier, only what the RAM store kept.

## Remaining stages

- **Stage 2, zero-copy sessions:** store entries point at pool pages with reference counts instead of slab copies. A
  restore becomes a table edit (no copy at all), and the RAM store and the slots share one budget: the pool.
  Copy-on-write for the partially filled last page.
- **Stage 3, lazy reservations and preemption:** reserve per piece / per decode chunk. When short, swap a running
  background request's pages to the store (or NVMe) and resume it later. Preempt background requests for pages.
- **Stage 4, only if the window shows > 1% on sparse attention:** translate each row's selected tokens to physical
  rows once per layer step and gather directly; batched page copies in the NVMe loader.

## GPU test plan

Before the window (no GPU; CPU and disk only):

1. Build the image on the head node and ship it to the worker node:
   `IMAGE=glm53-tensorfold:kvpool CONFIG=config/prod.env scripts/serve.sh build` (every patch through 0290).
2. PTX identity with the image's Triton, no `--gpus`:
   ```bash
   docker run --rm --entrypoint python3 -v $PWD/tests:/work/tests glm53-tensorfold:w5 /work/tests/kvpool_ptx.py > /tmp/ptx-w5.json
   docker run --rm --entrypoint python3 -v $PWD/tests:/work/tests -v /tmp:/host glm53-tensorfold:kvpool \
     /work/tests/kvpool_ptx.py --against /host/ptx-w5.json
   ```
   Gate: `16 of 16` identical (image w5 = 0001-0280).
3. Optional, on the running production (plain requests): baseline reply hashes and prefill speed.
   `python3 results/W5/ab.py results/W6/ab-prod.json 24500,98000 '{"prod":{}}'` (hashes in the `sha` field).

In the window. `R=results/W6`, `B=http://127.0.0.1:8000`, `M=GLM-5.3-Flash-EXL3`. A load helper like
`results/W5/load.sh` with `IMAGE=glm53-tensorfold:kvpool` starts `config/prod.env` plus overrides. Keep
`results/W5/mem.sh > $R/mem.log &` running from the first load.

| t (min) | step | pass gate |
| ---: | --- | --- |
| 0 | Stop production (`CONFIG=config/prod.env scripts/serve.sh stop`); lease; `nvidia-smi` clean on both nodes. | |
| 2 | **GPU tests, both nodes in parallel** (commands below). Head node: `test_kv_pool_patches.py`, `test_latent_patches.py`, `test_1m_patches.py`. Worker node (regressions with the pool off): `test_batch_sessions_patches.py`, `test_session_disk_patches.py`, `test_fp8_kv_patches.py`, `test_b12x_patches.py`, `test_glue_patches.py`. ~12 min. | kv_pool: every test passes. Others: the same counts as W2-W5 (known: `latent_tc`; fp8_kv's 6 fail / 16 error on the toy model's 128-wide latent; session_disk's 2 FP8 engine tests; b12x's 14). |
| 14 | **Load A**, pool only, same context: `load.sh poolA GLM53_TF_KV_POOL_TOKENS=1048576 GLM53_TF_KV_POOL_CHECK=1`. Copy both ranks' boot lines to `$R/boot-A-r{0,1}.log`. | `latent KV cache (fp8 rows): 7.4 KB a token a rank, in the KV pool: 1048576 tokens in pages of 256 (7.44 GB ...)`; `batching 4 requests: 3 extra sequence(s), ~0.8 GB` (was ~6.3: the slots are their fixed state only); MemAvailable after warm-up within 0.3 GiB of production's (K1: ~19.4 / 17.8). |
| 17 | **Same bits**: `python3 bench/glmbench.py --base $B --model $M --suites exact --out $R/exact.json`; `python3 bench/multiturn.py --base $B --model $M --modes batchexact --out $R/batchexact.json`; `python3 results/W5/ab.py $R/ab-poolA.json 24500,98000 '{"pool":{}}'`. | exact 10/10, batchexact 4/4, `sha` == step 3's production hashes (or W5's fat rows); decode after the prompt unchanged. |
| 24 | **Speed A/B**: `ab.py` again (2 runs each, 24.5k / 98k); `python3 bench/multiturn.py --base $B --model $M --modes concurrent --streams 1,4 --reps 3 --long-tokens 512 --out $R/conc-A.json`. | prefill within -1% of 1,283 / 1,258 tok/s; 4-stream aggregate within W5's spread (70-81 tok/s). |
| 34 | **Load B**, 1M slots: `load.sh poolB CONTEXT=1048576 GLM53_TF_KV_POOL_TOKENS=1048576 GLM53_TF_KV_POOL_CHECK=1`. | boot as A, `every slot up to 1048584`; MemAvailable within 0.1 GiB of A. |
| 37 | **Past 262k in one slot**: `python3 results/W5/ab.py $R/ab-400k.json 400000 '{"B":{}}'` (~5.5 min prefill); then `python3 bench/fp8ab.py --base $B --model $M --modes needle --needle-ctx 400000 --trials 2 --knob fast_prefill --values 1 --out $R/needle-400k.json` if time allows. | the request runs (it could not before); the response's `tensorfold.kv_pages` = 1,565 (400k + 256 + 64 tokens; ab.py does not save it: one curl); warm rerun `cached` ~ prompt; needle found. |
| 47 | **Pool pressure**: `python3 bench/multiturn.py --base $B --model $M --modes stress --stress-target 300000 --stress-step 100000 --stress-final 32000 --long-tokens 512 --mem-hosts local,$WORKER_SSH --mem-log $R/stress-mem.log --out $R/stress.json` (4 conversations x 300k = 1.2M > the 1M pool: admissions wait and idle slots spill; turns resume from the store / NVMe). ~18 min. Then `docker logs glm53-tf-r0 2>&1 \| grep -i 'KV pool'` and `dmesg -T \| grep -i 'out of memory'` on both nodes. | PASS from the script (min MemAvailable >= 8 GiB both nodes); no NVRM OOM; `/health` ok (the null-page check never fired); every request finishes; turns after the first show `cached` > 0 (from a slot, the store or disk). |
| 65 | **Restore production**: `CONFIG=config/prod.env scripts/serve.sh start` (image `sessdisk`), https check (models, a chat, a tool call), watchdog timer re-armed, lease deleted. | |

If time is short, keep 0-24 (same bits + speed with the pool on) and 65. That is the adoption gate for "pool 1M,
CONTEXT 262k", a no-behaviour-change switch. Steps 34-47 gate raising CONTEXT.

Test commands (step 2), from each node's mirror:

```bash
# head node
docker run --rm --gpus all -v glm53-tf-cache:/cache -e PYTHONDONTWRITEBYTECODE=1 -v $PWD/tests:/work/tests \
  -v $PWD/scripts:/work/scripts -v $PWD/results/W6:/work/out --entrypoint bash glm53-tensorfold:kvpool -c \
  "bash /work/scripts/run_tests_in_image.sh /work/out/tests-head -- tests/cuda/test_kv_pool_patches.py \
   tests/cuda/test_latent_patches.py tests/cuda/test_1m_patches.py"
# worker node
docker run --rm --gpus all -v glm53-tf-cache:/cache -e PYTHONDONTWRITEBYTECODE=1 -v $PWD/tests:/work/tests \
  -v $PWD/scripts:/work/scripts -v $PWD/results/W6:/work/out --entrypoint bash glm53-tensorfold:kvpool -c \
  "bash /work/scripts/run_tests_in_image.sh /work/out/tests-worker -- tests/cuda/test_batch_sessions_patches.py \
   tests/cuda/test_session_disk_patches.py tests/cuda/test_fp8_kv_patches.py tests/cuda/test_b12x_patches.py \
   tests/cuda/test_glue_patches.py"
```

GPU tests in `test_kv_pool_patches.py`:

| test | checks |
| --- | --- |
| `test_gpu_kernels_paged_equal_contiguous[bf16,fp8]` | real shapes (32 heads, latent 512): writes across a page bound, dense attention (windows across page bounds, BM 16 / 32), sparse (2,051 tokens a row, BM 16 / 32), b12x one-pass: paged == contiguous bit for bit, unmapped pages poisoned |
| `test_gpu_indexer_and_selection_paged_equal_contiguous` | `index_update` over windows of 1-2,048 rows (keys, gates, pool keys paged); `select_tokens` (sort and 0065 blocked paths), `select_tokens_dev` (0050) at 2.1-3.5k: equal |
| `test_gpu_graph_replay_after_pages_move` | a captured sparse step replayed after every logical page moved to another physical page: equal |
| `test_gpu_pool_prefill_state_equals_unpaged[256,512]` | synthetic checkpoint (32 index heads), 3,000-token prompt past the dense limit on a **fragmented** table: first token, latent rows, MTP latent rows, pool keys, MTP index keys, KDA state and conv equal the unpaged engine's; replies equal (greedy, sampled; serial, MTP, DFlash2, auto); resumed == fresh |
| `test_gpu_pool_bytes_and_accounting` | `kv_bytes_per_token` unchanged; pool bytes = rows x bytes a token |
| `test_gpu_batch_on_a_small_pool_equals_alone[greedy,sampled]` | 4 slots of 4k on a 6k pool with the store: 2 waves of 4 x 2.4k sessions; replies == alone; waits and spills happen; null page clean, nothing reserved at the end |
| `test_gpu_disk_sessions_into_pooled_slots` | 0250: tiny RAM store, sessions restored from NVMe into pooled slots, same replies, `cached` > 0 |

The host parts of the same file, `tests/test_kvpool_interpreter.py` and `tests/kvpool_ptx.py`, ran offline (results
above).
