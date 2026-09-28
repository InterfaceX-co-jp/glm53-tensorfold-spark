# Memory at CONTEXT=1000000 (and the prefill indexer)

Per rank, GLM-5.3-Flash (11 DSA layers + the MTP layer, 34 KDA layers, TP=2), `GLM53_TF_LATENT_KV=1`, the DFlash2
drafter loaded (`incoai/GLM-5.3-Flash-DFlash2`: 5 layers, 8 KV heads of 128, `sliding_window` 2048),
`GLM53_TF_PREFILL_ROWS_MAX=1024`. Capacity C = CONTEXT + 8 = 1,000,008 slots, R = 1,024 rows. Sizes come from
the code of every applied patch (0001-0091) and were checked by building `Buffers`/`State` on the meta device
with the real config.

## Why 1M did not load

Measured MemAvailable after load: 13.2 / 11.7 GB (rank 0 / rank 1) at CONTEXT=524288. Everything reserved per
capacity token before 0065 is **29,488 B a token a rank**, not the 18.75 KB of latent KV alone: the DFlash2
drafter reserved another 10 KB a token. Going from 524,296 to 1,000,008 slots adds 475,712 x 29,488 B = 14.0 GB,
more than rank 1's 11.7 GB: NVRM out of memory at load.

## Every capacity-scaled allocation

| # | Allocation (code) | Shape / rule | B / token | At C = 1M before | After 0065 |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | Latent KV, 11 DSA layers (`State.kc`, 0060) | 11 x [C, 512] bf16 (keys = values) | 11,264 | 10.49 GiB | 10.49 GiB |
| 2 | MTP head latent KV (`State.mtp_kc`, 0060) | [C, 512] bf16 | 1,024 | 0.95 GiB | 0.95 GiB |
| 3 | Indexer keys + gates, 11 layers (`State.index[i][0:2]`) | 2 x 11 x [C, 128] bf16 | 5,632 | **5.25 GiB** | **11 MiB** (ring of 2,048 rows) |
| 4 | Indexer keys + gates, MTP layer | 2 x [C, 128] bf16 | 512 | 0.48 GiB | 0.48 GiB (kept full) |
| 5 | Pool keys, 12 layers (`State.index[i][2]`) | 12 x [C/4 + 2, 128] bf16 | 768 | 0.72 GiB | 0.72 GiB |
| 6 | DFlash2 context K/V (`Drafter.kc/vc`, `dflash2.py`) | 5 layers x 2 x [4 KV heads, C + 8, 128] bf16 | 10,240 | **9.54 GiB** | **40 MiB** (ring of 4,096 slots) |
| 7 | 0050 `LongScratch` scores + keys (captured decode selection) | 2 buffer sets x 8 rows x C/4 x (4 + 8) B | 48 | 0.045 GiB | 0.045 GiB |
| 8 | 0060 latent attention scratch (`latent.Scratch`) | chunk partials sized by min(C, 2,051) + R | 0 | 0.55 GiB a set (rows-scaled) | same |
| 9 | Expanded attention scratch (`AttnScratch`, latent **off** only) | (C + R) / 512 chunks x R x 32 x 256 x 4 B | 64 K x R/512 | n/a with latent (128 GiB at 1M, 1,024 rows) | same |
| 10 | **Eager selection, transient** (`sparse.select_tokens`, every prefill chunk past 2,051) | scores [R, np] fp32 + stable sort (values, int64 indices, CUB buffers): ~36 B an entry; np = the 0050 bucket, up to C/4 | - | **~9.2 GB** peak at the end of a 1M prompt (1.2 GB at 112k), then held by the caching allocator | **<= 256 MiB** of int32 keys a block + top-k temporaries (~20 MB) |
| 11 | Latent sparse partials of a prefill chunk (`sparse_latent`, R > 8) | 5 chunks x R x 32 x 512 x 4 B | 0 | 320 MiB (rows-scaled, transient) | same |
| 12 | CUDA graphs | main/MTP graphs use static buffers; 0050's long graphs allocate nothing (`LongScratch`), one per (rows, parity, bucket) | 0 | - | - |
| | **Reserved per token** | rows 1-7 | **29,488** | **27.5 GiB** | **13,616 B: 12.7 GiB + 51 MiB of rings** |

Not capacity-scaled, for reference (R = 1,024 rows): each `Buffers` set is 2.92 GiB (EXL3 split-K scratch 1.28,
latent scratch 0.55, qmm split-K `sk` 0.50, logits 0.15, expert outputs 0.14, ...), and there are two (the model's
and the MTP head's); `State` adds the KDA replay scratch (1.60 GiB) and projection rows (0.82 GiB): **8.3 GiB a
rank at 1,024 rows**, ~9 MB a row. That is why CONTEXT=262144 with 2,048 rows had less headroom (9.8 GB) than
CONTEXT=524288 with 1,024 rows (11.7 GB).

## What 0065 changes

- **DFlash2 ring** (`GLM53_TF_DRAFTER_RING`, default 1): every drafter layer is a sliding-window layer (window
  2,047 keys), so position t lives in slot t mod 4,096 (`ring_slots`: window + block + one 64-key tile, rounded to
  a power of two). The attention kernel starts at the first 64-key tile any row can see (the skipped tiles were
  fully masked: exact no-ops of the online softmax), so drafts have the same bits; it also stops scanning the
  whole context each block pass (at 112k: 1,750 tiles a layer before, 33 now). A snapshot resumed after the
  drafter wrote more than a ring past it gets the positions it lost masked (drafts differ; replies never do).
- **Index ring** (`GLM53_TF_INDEX_RING`, default 1): the 11 model layers' index keys and gates are read only to
  build the pool key of a pool the current window completes, so a ring holding a window and the 3 rows before it
  gives the same pool keys. Pool keys (what scoring reads) stay full. The 3 committed rows of an incomplete pool
  are kept in a trailing slot of `State.conv` at every commit; snapshots already copy and restore `conv`, and
  `set_pos` writes the rows back on a restore, so resumed == fresh holds however far the reply went.
- **Blocked selection** (`GLM53_TF_SELECT=blocked`, default; `GLM53_TF_SELECT_MB`, default 256): prefill windows
  (more than 8 rows) score in row blocks within the budget, each block only up to its own last row's pools, and
  select without the full sort. Decode/verify/MTP windows (1-8 rows) and 0050's captured steps are unchanged.

## Expected MemAvailable at CONTEXT=1000000, 1,024 rows

Reserved per token before, at 524,296 slots: 15.46 GB. After, at 1,000,008 slots: 13.62 GB + 0.05 GB of rings.
From the measured 13.2 / 11.7 GB at 524k: **about 15.0 / 13.5 GB** (rank 0 / rank 1) after load at 1M, and a 1M
prompt's prefill no longer adds the ~9 GB sort transient (at most ~0.6 GB: a selection block, top-k temporaries
and the latent sparse partials). Target >= 8 GB: met with ~5 GB margin on rank 1.

Further headroom, not done here: the MTP head's `Buffers` set at fewer rows (the head already absorbs longer
windows in chunks of its buffer's rows): 2.2 GiB at 256 rows; KDA replay scratch only for verify windows (1.6
GiB); an fp8 latent (6.3 GB at 1M, a new configuration with a quality gate).

## The prefill indexer (0065)

Measured (latent on, 1,024-row chunks): indexer 1.5 s at 28k, 26.3 s at 112k (14% of the prefill), ~20 ms a
layer a chunk at 112k. Per layer and chunk it scored every row against every pool of a power-of-two bucket (one
program a (row, 64 pools), each loading its own copy of the 16 KB pool tile: R x np x 256 B of loads, ~8.6 GB a
layer at the end of 112k), then stable-sorted [R, bucket] fp32 with int64 indices.

Now (`sparse.select_pools_blocked`):

1. `_scores_rows`: a program serves 16 rows with one pool tile (16x fewer tile loads), running `_scores`' exact
   per-row code on the same shapes and warps ([32 heads, 128] x [128, 64] tensor-core dot, relu, head sum), so the
   scores are `_scores`' bits (checked at start on the device by `blocked_ok`, which falls back to the sorted path
   on a mismatch). A block scores only up to its last row's pools (no power-of-two padding: ~25% fewer on average).
   The score is written as an int32 key ordered like the stable descending sort (0050's key, upper half).
2. The 512th largest key t of each row with `torch.topk` on int32 (its value is exact whatever the algorithm),
   then `_gather_sel` lists, in pool order, every key > t and the lowest-index keys == t until 512: exactly the
   sorted path's set and tie rule (higher score, then lower pool; -0.0 == +0.0; NaN highest), already ascending.
3. fp8/bf16 score inputs: not used (the inputs are already bf16; fp8 would change scores and so the selection).

Estimate (unmeasured): the dot work is R x npool x 32 x 128 x 2 FLOP, ~117 GFLOP a layer a chunk at the 112k
run's mean 14k pools (~2 ms on GB10's tensor cores), plus the epilogue and the int32 top-k (~1-2 ms). About
**4-5 ms a layer a chunk instead of ~20 at 112k** (indexer 26.3 s -> ~5-7 s, TTFT 197 s -> ~178 s) and **~1.5 ms
instead of 4.5 at 28k** (1.5 s -> ~0.5 s). The work stays quadratic in the prompt (every row scores every earlier
pool), at a ~4-5x smaller constant.

## Tests

`tests/cuda/test_1m_patches.py` (GPU): row-block scores bit-equal to `_scores` (32 and 4 heads); blocked selection
equal to the sorted path (random, ties, -0.0, NaN; rows crossing 2,051; the 1,024-pool bucket boundary; up to
1,024 rows; 1M-token pool counts; small row blocks); decode windows keep the sorted path; selection scratch
bounded at 1,024 rows x 250,000 pools; index ring pool keys equal full caches; drafter ring block passes equal
the linear cache (wrapped, rewound); engine replies with 0065 on equal all-off, serial and drafted, 64- and
512-row chunks, 2,040-6,000-token prompts; the prompt snapshot resumed after a reply past both rings equals a
fresh prefill; memory accounting at a 1M capacity on the synthetic model.

    PYTHONPATH=/src/TensorFold/tests/cuda pytest -q tests/cuda/test_1m_patches.py
