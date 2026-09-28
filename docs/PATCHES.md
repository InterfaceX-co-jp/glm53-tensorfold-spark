# Patches to TensorFold

TensorFold is vendored as the git submodule `vendor/TensorFold`, pinned at `2f8e514` (0.3.4). The image build
(`docker/Dockerfile`) applies `patches/*.patch` in name order with `git apply`, then installs the patched tree.
The submodule stays at the pinned commit; the patches are the whole diff. Patches 0001, 0003 and 0004 touch
only the GLM (`glm5_next`) CUDA engine; 0002 touches the shared CUDA server's tool-call parsing.

Every knob defaults to upstream behaviour, so an image with the patches and no environment set serves exactly
what upstream serves (plus the GLM tool-call parser, which only adds a format upstream did not read).

| Patch | Knob | Default | Effect |
| --- | --- | --- | --- |
| 0001 `glm-exl3-nonexpert-q4` | `GLM53_TF_NONEXPERT=bf16\|q4\|q4mse` | `bf16` | EXL3 non-expert weights stored in 4 bits; decode 1.2-1.9x |
| 0002 `glm-tool-call-parser` | none | on | GLM `<arg_key>/<arg_value>` tool calls become OpenAI `tool_calls` |
| 0003 `glm-prefill-rows` | `GLM53_TF_PREFILL_ROWS=N` | `64` | prefill chunk size; kernels tile past 128 rows |
| 0004 `glm-sparse-long-prefill` | none | on | sparse attention (DSA) scratch sized per window; vectorized token selection |
| 0050 `glm-longctx-decode` | `GLM53_TF_LONGCTX_GRAPHS=0\|1` | `1` | past 2,051 tokens: indexer scores only existing pools (power-of-2 bucket); decode/verify/MTP steps of 1-8 rows replay CUDA graphs per (rows, parity, bucket), captured lazily. `0` = upstream path |
| 0060 `glm-latent-kv` | `GLM53_TF_LATENT_KV=0\|1` | `0` | DSA/MLA layers cache the 512-wide latent instead of per-head K/V (absorbed MLA): 18.75 KB a token a rank instead of 390.75 KB (20.8x), 1M context in ~19 GB; attention reads ~16-30x fewer bytes. New arithmetic (replies differ from `0` in the last bits), every exactness guarantee kept within `1` |
| 0065 `glm-1m-memory-indexer` | `GLM53_TF_SELECT=blocked\|sort`; `GLM53_TF_SELECT_MB=N`; `GLM53_TF_INDEX_RING=0\|1`; `GLM53_TF_DRAFTER_RING=0\|1` | `blocked`; `256`; `1`; `1` | 1M context: DFlash2 context and the model layers' index keys/gates in rings (29.5 -> 13.6 KB reserved a token a rank); prefill selection in row blocks with tile-shared scoring and a top-k threshold instead of a full sort (bounded scratch, ~4x faster indexer, estimated). Same bits: replies equal `sort`/`0`/`0` |
| 0080 `glm-fast-prefill` | `GLM53_TF_FAST_PREFILL=0\|1`; `GLM53_TF_FAST_GATHER=bf16\|fp32` | `0`; `bf16` | prefill chunks through kernels that need not be row-invariant (fused EXL3 expert GEMMs, bf16 rank partials, head on the last row; 0081's chunked KDA and large-M matmuls when present) at absolute multiples of a chunk grid C; only the state at the prompt's last grid point is kept. New arithmetic (replies differ from `0`), drafted == serial and resumed == fresh kept within `1` |
| 0082 `glm-lean-prefill` | `GLM53_TF_LEAN_PREFILL=0\|1`; `GLM53_TF_LEAN_BLOCK=N` | `0`; `1024` | fast chunks of up to `PREFILL_ROWS_MAX` rows (e.g. 8192) in sub-blocks of N rows on window buffers of N rows; the routed experts run once over the whole chunk. Same bits as 0080 at the same grid. 8192-row chunks need 11.6 GiB of row buffers a rank, against 17.5 for today's 2048-row config |
| 0083 `glm-fp8-prefill` | `GLM53_TF_FP8_PREFILL=0\|1`; `GLM53_TF_FP8_TILE=bm,bn,warps,stages` | `0`; `128,64,4,2` | fast prefill chunks only: the large non-expert matmuls on FP8 (e4m3) tensor cores with per-row power-of-two activation scales and the exact 4-bit weights; on the latent cache, absorb / expand on bf16 tensor cores and 32-query attention tiles. New arithmetic (fast replies differ from `0`); snapshots tagged C + 1; drafted == serial and resumed == fresh kept. Estimated -115 to -165 us a token |
| 0084 `glm-prefill-overlap` | `GLM53_TF_PREFILL_OVERLAP=0\|1\|gather,slab,direct`; `GLM53_TF_PREFILL_HC_SLAB=N` | `0`; auto (L2 / 2) | lean chunks pipelined: each sub-block's all-gather on a comm stream while the next sub-block computes; partials exchanged in the dtype they are written in (no conversion copies); hc_post + the next hc_pre per L2-sized row slab. Same kernels, same inputs, same collectives in the same order: same bits as 0082. Estimated -90 to -120 us a token (of ~1,200-1,300 at 32k) |
| 0085 `glm-c-independent-prefill` | `GLM53_TF_PREFILL_ROWS=auto`; `GLM53_TF_SNAPSHOT_GRID=N`; `GLM53_TF_SNAPSHOT_TAIL=N` | `64` (`config/` sets `auto`); `64`; `256` | a fast prefill's state no longer depends on its chunk size C (every fast kernel row-independent, the KDA scan on absolute 64-row blocks; the < 64-row qmm fallback removed from fast chunks): fast snapshots at the prompt's last multiple of 64, tagged by mode only, resumable by any C; `prefill_rows` "auto" picks C per prefill. A next turn re-prefills < 64 old tokens + the reply instead of up to C - 1. Fast replies change only for prompts whose last chunk had < 64 rows |
| 0090 `glm-request-knobs` | `"tf_knobs": {...}` per request; `GLM53_TF_PREFILL_ROWS_MAX=N` | env defaults; max = `PREFILL_ROWS` | the speed knobs of 0003/0005/0006/0010/0020/0050/0070/0071 switch per request on a loaded server (both ranks, via the request header); the env sets their defaults |
| 0091 `glm-fast-prefill-knob` | `"tf_knobs": {"fast_prefill": 0\|1}` | `GLM53_TF_FAST_PREFILL` | 0080's fast prefill switched per request; fast and exact snapshots never mix |
| 0092 `glm-fp8-prefill-knob` | `"tf_knobs": {"fp8_prefill": 0\|1}` | `GLM53_TF_FP8_PREFILL` | 0083 switched per request (in the header, so both ranks agree); FP8 and bf16 fast snapshots never mix; refused where FP8 dots are unavailable |
| 0093 `glm-prefill-overlap-knob` | `"tf_knobs": {"prefill_overlap": 0\|1}` | `GLM53_TF_PREFILL_OVERLAP` | 0084's pipeline switched per request (same bits either way; 1 = the env's variant, or `gather,slab`) |
| 0110 `glm-session-cache` | `GLM53_TF_SESSION_GIB=N`; `GLM53_TF_SESSION_EVERY=N`; `GLM53_TF_SESSION_FORK_MIN=N`; `GLM53_TF_SESSION_RESERVE_GIB=N` | `0` (off; `config/` sets 12); `16384`; `512`; `2` | many sessions resident per rank (snapshots + their attention rows in 256-token pages, shared prefixes stored once, LRU within the budget): a request resumes from the longest stored prefix by copying it into the live caches (~5 ms at 30k) instead of re-prefilling; marks at fork points and every N tokens; `usage.prompt_tokens_details.cached_tokens`. Same bits: resumed == fresh, drafted == serial |
| 0120 `glm-batch-v2` | `GLM53_TF_BATCH=N`; `GLM53_TF_BATCH_PIECE=N`; `GLM53_TF_BATCH_PREFILL_SHARE=F`; `GLM53_TF_BATCH_GRAPH_ROWS` / `_MAX_GRAPHS`; `GLM53_TF_BATCH_RESERVE_GB` / `_ADMIT_GB`; `"priority": "background"` | `1` (off); `2048`; `0.5`; `8` / `256`; `4` / `1` | 0030's batching on the current engine: 2-4 requests a round, each with its own latent KV / rings, KDA state, MTP, DFlash2 context, lookup, cost-derived depths and `tf_knobs`; prompts prefill in pieces through `decode.prefill` (exact / fast / lean / FP8) between the others' rounds; graphs per (slots, rows, dense / pool bucket); background requests step aside. Same bits: batched == alone == serial |
| 0140 `glm-fast-boot` | `GLM53_TF_PREPARED=DIR`; `GLM53_TF_PREPARED_WRITE=0\|1`; `GLM53_TF_PREPARED_VERIFY=full\|sample\|off`; `GLM53_TF_PREPARED_THREADS=N`; `GLM53_TF_CALIB=cached`; `GLM53_TF_CALIB_DIR`; `GLM53_TF_CLOCK_CAP`; `GLM53_TF_BOOT_WARMUP=0\|1` | unset (off; `serve.sh` sets `/prepared`, write 1, `cached`); `0`; `sample`; `8`; `real`; `/cache/calib`; none; `1` | restarts: each rank's weights (and the DFlash2 drafter's) read back from a prepared folder exactly as the load built them (split, q4mse, tiled, EXL3 words), with a parallel O_DIRECT reader into pinned buffers; the calibration table reused per (image, knobs, engine shape, GPUs, clock caps), rank 0 deciding for both; `[boot]` timeline lines. Same bits: prepared == load-time (tested), replies unchanged |
| 0150 `glm-mia-wins` | `GLM53_TF_HEALTH=basic\|strict`; `GLM53_TF_STALL_S=N`; `GLM53_TF_STALL_PREFILL_TPS=N`; `GLM53_TF_EFFORT_FIELD=0\|1`; `GLM53_TF_DEFAULT_EFFORT=low\|high\|max` | `basic`; `0` (off); `200`; `0`; unset | `/health` reports a fatal engine error, requests in flight and stalls (`strict`: 503, and new completions refused after a fatal error); `GET /metrics` (Prometheus counters); an engine error answers a JSON 500 / an SSE error event instead of dropping the connection; OpenAI `reasoning_effort` mapped onto the GLM template's thinking / effort, and a server default effort. Serving path unchanged; replies change only for requests the effort knobs rewrite |
| 0170 `glm-mia-prefill` | `GLM53_TF_FAST_EXPERTS=fat`; `"tf_knobs": {"fat_experts": 0\|1}`; `GLM53_TF_FAT_STAGES=3\|4`; `GLM53_TF_FAT_TICKET=0\|1`; `GLM53_TF_FAT_SHARED_X=auto\|0`; `GLM53_TF_KDA_PROJ_BF16=0\|1`; `GLM53_TF_KDA_BF16_TILE` | `fast2`; env; `3`; `1`; `auto`; `0`; table | fast chunks' routed experts through the `fat` kernels (MiaAI-Lab / Reederey87 E2/E3 data movement: trellis words in the cp.async ring, one rotated input for gate and up, swizzled 48 KB stages at 2 CTAs an SM, ticket scheduling) with fast2's arithmetic: **same bits** as fast2, row-independent. KDA input projection as a bf16 copy of the 4-bit weights in fast chunks (+98.25 MiB a KDA layer and rank; new arithmetic, row-independent, load-time only) |
| 0180 `glm-batch-sessions` | `GLM53_TF_BATCH_SESSIONS=0\|1` | `0` | with `GLM53_TF_BATCH` > 1 and `GLM53_TF_SESSION_GIB` > 0: 0110's store behind 0120's slots. An admission restores the longest stored prefix into a free slot (the one already holding most of its pages) and prefills only the suffix in pieces; marks, prompt and reply snapshots are saved from every slot; one index / budget / page pool, a live-page map per slot; the budget is set aside at load and at admission. Same bits: batched == alone == serial, resumed == fresh |
| 0190 `glm-prefill-glue` | `GLM53_TF_MOE_GLUE=0..7`; `GLM53_TF_MTP_PREFILL_WINDOW=N`; `GLM53_TF_HC_FUSED=0..3`; `GLM53_TF_ATTN_BM32=0\|1` (all per request: `tf_knobs.moe_glue` / `mtp_window` / `hc_fused` / `attn_bm32`); `GLM53_TF_LATENT_TC=0\|1` (load-time) | all `0` | prefill glue: MoE grouping by a parallel sort, the router in one kernel, the combine reading the shared expert in place (same bits); the MTP head run only on a prompt's last ~N positions (drafts may change, replies not); hc_post fused with the next hc_pre's dots + an unrolled finish (same bits, GPU-checked); 32-query latent attention tiles in bf16 fast chunks (same bits, GPU-checked); opt-in: latent absorb / expand on bf16 tensor cores (new arithmetic, own snapshot tag G + 2). Microbenchmark: `tests/cuda/bench_glue.py` |
| 0200 `glm-batch-parallel` | `GLM53_TF_BATCH_CAPTURE_AFTER=N`; `GLM53_TF_BATCH_PARITY_KEY=0\|1`; `GLM53_TF_BATCH_MTP=0\|1`; `GLM53_TF_BATCH_ROW_MS=F`; `GLM53_TF_BATCH_SHORT=N`; `GLM53_TF_BATCH_PAD=sizes` | `1`; `0`; `0`; `0`; `0`; off | with `GLM53_TF_BATCH` > 1: batched-round graph keys captured on their N-th sighting; KDA parities in the key instead of 71 MB state copies a slot and round; every slot's MTP chain drafted in one head pass a step; batch-aware cost depths price rows past the verify table at >= F ms; prompts of <= N tokens prefill in their admission round outside the fair share; windows padded to fewer graph keys. Per-request `round_kinds` stats. Same bits: replies == served alone |
| 0210 `glm-prompt-tokens` | `GLM53_TF_TOKCACHE=0\|1\|verify`; `GLM53_TF_TOKCACHE_ENTRIES=N` | `1`; `32` | host only (rank 0's HTTP app). A request's prompt is encoded once (the context check hands its ids to `run`; it was encoded twice), and the ids of the last N prompts are kept: a new prompt reuses the longest prefix it shares with one up to the end of a special token (`<\|user\|>`, `<\|assistant\|>`, `<\|observation\|>`, ...) and encodes only the rest. Exact: the tokenizer splits on added tokens before BPE (no normalizer, every added token plain; checked at start, off otherwise); `verify` also runs the full encode and logs + uses it on any difference. ~34 ms -> ~1 ms a turn at 40k tokens, ~113 ms -> ~3 ms at 128k, plus the second encode saved. Ids unchanged |
| 0220 `glm-fp8-latent-kv` | `GLM53_TF_KV_DTYPE=bf16\|fp8` (load-time, both ranks equal) | `bf16` | with `GLM53_TF_LATENT_KV=1`: the latent caches (11 DSA layers + the MTP head) store a token as 512 e4m3 values + one power-of-two fp32 scale (528 B rows, 16-byte aligned) instead of 1,024 B of bf16: **13,616 -> 7,664 B a token a rank** (4 x 262k slots: 13.26 -> 7.45 GiB a node). Rows quantized one at a time in `latent_write`; the dense / sparse latent kernels dequantize exactly into bf16 and run the bf16 arithmetic. KV storage only (prefill activations stay bf16). New arithmetic vs bf16 KV; within fp8, same bits: drafted == serial, resumed == fresh, batched == alone. Snapshots and session keys carry the format |

## 0001 — EXL3 non-expert weights in 4 bits

**Problem.** The EXL3 checkpoint stores the routed experts in 4-bit EXL3 (mcg codebook) but every other weight
in BF16: the KDA and DSA attention projections, the DSA indexer projections, the dense MLPs, the shared
experts and `lm_head`. Per rank that is 10.7 GB
read on every decode step, against 5.0 GB for the all-4-bit MLX checkpoint TensorFold was tuned on. One-row
verify took 57 ms.

**Change.** `GLM53_TF_NONEXPERT` controls how those BF16 matrices are stored at load time
(`families/glm5_next/cuda/weights.py`):

- `bf16`: upstream (`make_b16`).
- `q4`: the engine's existing MLX-style affine 4-bit format, groups of 64 along K, `q = round((w - min) / scale)`
  (`qmm.quantize4`, already used upstream for draft-only copies).
- `q4mse`: same format and same kernels, but each group's `[min, max]` range is shrunk about its centre by the
  factor from `{1.0, 0.97, ..., 0.76}` with the smallest squared reconstruction error (computed with the
  bf16-rounded scale and bias the kernel will actually use). Factor 1.0 is in the grid, so the error is never
  worse than `q4`. Load-time cost only.

The routed experts are untouched, and so are the router, norms, convolutions and the other small tensors. With a 4-bit `lm_head`, the separate 4-bit draft copy of the head is no longer
made (the head already is that copy).

`engine.py`: upstream maps the `auto` drafter policy on an EXL3 checkpoint to `EXL3_AUTO`, which was measured
with BF16 non-expert weights (where an MTP step is expensive). With 4-bit non-expert weights the MTP head costs
what it costs on the MLX checkpoint, so `auto` again chooses between MTP and DFlash2 the way it does there.

**Result.** One-row verify 57 ms -> 30 ms; serial decode 17 -> 33 tok/s; drafted decode 1.2-1.5x over upstream
on every cell (`docs/RESULTS.md`). MMLU-200: bf16 87.0%, q4mse 88.0%.

**Exactness.** Drafting and verification read the same stored weights, and the verify step decides every
emitted token, so a drafted reply is still byte-identical to the serial reply of the same configuration. q4mse
is not bit-identical to bf16: it is a different (re-quantized) model, which is why its quality is checked
separately (MMLU, refusals).

## 0002 — GLM tool-call parser

**Problem.** GLM-4.5 through 5.3 write tool calls as

```
<tool_call>bash<arg_key>command</arg_key><arg_value>ls -la</arg_value></tool_call>
```

Upstream's CUDA server only parsed the Qwen form (`<function=name><parameter=k>v</parameter></function>`), so
GLM tool calls came back as plain text and agents (opencode) could not use tools.

**Change.** `cuda/server.py` gains `_glm_call`, following vLLM's `glm47` parser: the name is the text before the
first `<arg_key>`; a value stays text when the tool's JSON schema types that parameter as `string`, and is
parsed as JSON otherwise (falling back to the text). A block whose name is empty or contains markup is left as
text. The Qwen format still parses.

**Exactness.** Output parsing only; the tokens are unchanged.

## 0003 — prefill chunk rows

**Problem.** Upstream prefills in fixed 64-row chunks. Every chunk reads every weight once, so on a
bandwidth-bound GPU larger chunks prefill faster. The attention and quantized matmul paths refused more than
128 rows.

**Change.** `GLM53_TF_PREFILL_ROWS=N` sets the chunk size (`engine.py`). `attention.py` accepts any row count
(the grid already tiles rows by the block size, and rows are independent) as long as the scratch holds the
window; `qmm.split_k` picks the 128-row tile for longer windows instead of raising. Costs about 5 MB of window
buffers per row on the EXL3 checkpoint.

**Exactness.** A row's result never depends on which chunk it is in, so the committed KV state has the same
bits for any chunk size; replies and resumed conversations match 64-row prefill bit for bit (tested).

## 0004 — sparse long-context prefill

**Problem.** Past 2,051 tokens GLM's DSA attention runs sparse (top-k pools). Upstream sized the sparse
kernels' partial buffers for 128 rows (the verify windows), so a prefill chunk longer than 128 rows past that
point wrote out of bounds: an illegal memory access at 8k context with big chunks. Token selection also
looped over rows in Python.

**Change.** `sparse.py`: the chunk partials (`po`, `pm`, `pl`) are sized for the window's `R` rows and the row
stride is passed to `_sparse_chunks` / `_sparse_merge`. `select_tokens` builds every row's list at once with
tensor ops (the selected pools' tokens ascending, then the visible tokens of the incomplete last pool; a dense
row gets count 0), producing the same lists as the loop.

**Exactness.** The row stride only places partials in memory; the arithmetic and its order are unchanged. The
vectorized selection is compared against upstream's loop, kept in the test as the reference.

## 0050 — long-context decode (graphs past 2,051 tokens, bounded indexer)

**Problem.** Graphs replayed only while `pos + R <= dense_limit`; every later step ran eagerly (~1,500 launches
from Python), and `select_tokens` scored and sorted `capacity / 4` pools in every DSA layer whatever the length.

**Change.** (a) `sparse.pool_bucket(pos + R, cap)`: the pools to score, `(pos + R) // 4` rounded up to a power of
two (>= 1024, <= capacity / 4); used by every eager selection (prefill chunks too). (b) `sparse.select_tokens_dev`:
the same `_scores` kernel over the bucket, then unique int64 keys (`_keys`: score order, ties to the lower pool,
-0.0 == +0.0, NaN highest), `torch.topk(sorted=False)` + `torch.sort` of the 512 indices, and `_expand` (tokens and
counts from the device position), all into `LongScratch` buffers sized at init (8 rows); `sparse_attention` takes
preallocated partials. `Engine.forward`/`.mtp` replay `Graphs.long[("main", R, parity, bucket)]` /
`[("mtp", n, bucket)]` when every row is past 2,050 and R <= 8; the first step of a key runs eagerly through the
same code (warm-up and result) and is then captured. Prefill chunks and mixed windows stay eager. A failed capture
logs once and falls back to eager. `GLM53_TF_LONGCTX_MAX_GRAPHS` (default 256) caps the lazily captured graphs.
Both ranks must agree on the knob (checked at start).

**Exactness.** Pools past `(pos + R) // 4` score -inf and sit after every kept pool in the stable descending
order, so the top 512 are unchanged; the keys are unique and order pools exactly as that sort does, so any top-k
over them returns the same set, listed ascending as before. Scores come from the same kernel. Everything is per
row, so verify rows keep the bits of serial steps. Tests: `tests/cuda/test_longctx_patches.py`.

## 0060 — latent (absorbed) MLA KV cache

**Problem.** The 11 DSA layers and the MTP layer cache decompressed per-head keys and values: 32 local heads x
(256 + 256) x 2 B = 32 KB a token a layer, **384 KB a token a rank** for 12 layers (plus 6.75 KB of indexer
caches). The ~35-40 GB free per rank hold ~100k tokens; 1M would need ~400 GB. Every decode row past 2,051 tokens
gathers 2,051 selected tokens x 32 KB = 67 MB per layer (0.74 GB over 11 layers, ~3 ms).

**Change.** `GLM53_TF_LATENT_KV=1` (new module `families/glm5_next/cuda/latent.py`). GLM-5.3 has no rotary part
in its heads (`qk_rope_head_dim = 0`), so `k_h = W_k,h c` and `v_h = W_v,h c` with `c = kv_a_layernorm(kv_a(x))`
(512 wide), and attention runs in latent space: `score = (W_k,h^T q_h) . c`, `o_h = W_v,h (sum p c / sum p)`.

- Cache: one bf16 latent row a token a layer (`State.kc[i]` is `[capacity, 512]`, `State.vc is State.kc`: keys =
  values; the MTP head's cache likewise). 12 x 1 KB + 6.75 KB indexer = **18.75 KB a token a rank (20.8x)**:
  100k tokens = 1.9 GB, 1M tokens = 19.2 GB a rank.
- `absorb`: `q'_h = bf16(W_k,h^T q_h)` from kv_b's key rows **as stored** (4-bit groups of 64, or BF16 on an EXL3
  checkpoint with `GLM53_TF_NONEXPERT=bf16`): each program dequantizes 64 x 64 tiles exactly in fp32 and multiplies
  in fp32 (IEEE, FMA). No absorbed copy is kept, so weight memory and the weight bytes a step reads are unchanged
  (the kernel replaces the key projection).
- Dense attention (`attention_latent`): 16 (row, head) queries a tensor-core tile against 32-row tiles of the
  latent, which serve as keys and values at once (MQA-style: a tile is read once for 16 heads), online softmax,
  512-key chunks fixed by absolute position, merged in chunk order into the fp32 normalized latent `u`.
- Sparse attention past 2,051 tokens (`sparse_latent`): the same tile over each row's own top-k tokens
  (`sparse.select_tokens`, and patches/0050's device-side `select_tokens_dev` in captured long-context steps),
  16 heads of the row a tile; preallocated partials for windows up to 8 rows (captured steps allocate nothing).
- `expand`: `o_h = bf16(W_v,h u_h)` from kv_b's value rows as stored (same fp32 dequant and FMA), then `o_proj` as
  before.
- Scratch: chunk partials are sized by the dense limit (dense attention never runs past it), not by the capacity:
  ~0.25 GB a buffer set at 512 prefill rows whatever the context (the expanded scratch grows with capacity:
  3.3 GB at 100k).
- Dispatch: one branch at the top of `forward.dsa_block` (serial, verify, prefill, MTP, graphs, 0050's long
  steps) and one at the top of `batch._dsa` (patches/0030 rounds). `decode.Engine` reads the knob before any
  `State`/`Buffers` and prints the latent KV bytes a token (rank 0); `GlmEngine` checks both ranks agree.

**Exactness.** The arithmetic differs from the expanded path (k and v are never rounded per head; q' is), so
replies differ from `GLM53_TF_LATENT_KV=0` in the last bits: a new engine configuration. Within it every
guarantee holds for the same reasons as before: each (row, head) is one row of a fixed 16-row tile whatever its
tile-mates; absorb/expand reduce in a fixed order per row; chunks are fixed by absolute key position and merged in
order; empty chunks and rows past a row's causal limit are exact no-ops; no atomics. So a verify row gets the bits
of the serial step (drafted == serial), prefill chunk size changes no bit, and a resumed prefill equals a fresh
one (snapshots still leave the attention caches in place). Accuracy against an fp32 reference on the real shapes
(CPU emulation, round-to-nearest bf16): latent 2.7e-3 / 4.3e-3 relative error vs expanded 2.8e-3 / 4.8e-3.

**Expected speed (arithmetic, unmeasured).** Past 2,051 tokens (8k and 32k alike: the selection is 2,051 tokens
either way) a verify row reads ~46 MB of latent (23 MB if the second head tile hits L2) instead of 0.74 GB:
**about -3 ms per window row** (R = 1: -3 ms of ~30 ms, +10%; R = 4: up to -12 ms) and -0.29 ms per MTP draft step.
Below 2,051 the saving grows with the context: -3.1 ms a window at 2,000 tokens. Prefill: attention FLOPs double
per head (512-wide dot products), and absorb/expand add ~0.1 TFLOP a 512-row chunk on FMA; estimated +1-2% of a
chunk, not measured.

**Memory at CONTEXT=32768, 1024-row prefill buffers (per node, arithmetic).** KV 12.21 -> 0.59 GiB; attention
scratch (two buffer sets: the model's and the MTP head's) 4.26 -> 1.10 GiB, because the expanded scratch grows 128 KB
per capacity token at 1024 rows and the latent one is capped by the dense limit. **Saving 14.8 GiB**: today's
9-11 GiB headroom becomes ~24-26 GiB. At 18.75 KB a token that would hold ~1.3-1.4M tokens; after the eager
selection's transient scores and sorts at 1024 rows (~6 KB per capacity token) and the DFlash2 drafter's own
capacity-sized cache (check its config), expect ~0.8-1M.

**Prefill (estimate, from the measured 28k profile: sparse attention 30% of a 1024-row chunk).** The expanded
sparse kernel puts one query in a 16-row tile (1/16 of each MMA used) and gathers 2,051 tokens x 32 KB per row per
layer (67 MB). The latent kernel is heads-as-rows: 16 heads of one row fill the tile, and one 32-row latent tile
load serves them all, so each row gathers 2 x 2,051 x 1 KB = 4.2 MB (16x fewer bytes) and issues ~8x less MMA
work. Taking a 5-10x faster sparse phase and +1-2% for absorb/expand: a 28k chunk costs 0.73-0.78 of today's,
so **prefill throughput +25-35% at 28k**. At 8k about the same per sparse row: every row past 2,051 selects
2,051 tokens whatever the context, and the dense rows below it also read 32x fewer bytes. So also +25-35% at 8k,
slightly less if the first 2k rows are a larger share of the chunk.

**Verified on the head node (GB10).** `tests/cuda/test_latent_patches.py` passes, together with `test_patches.py`,
`test_longctx_patches.py` and `test_knob_patches.py` (99 passed) on the full set 0001-0090. Kernel times per layer
(one rank's 32 heads; latent 512; sparse rows select 2,051 tokens of a 32k context; dense rows end at 2,051):

| Rows | Path | Expanded ms | Latent ms (BM = 16, default) | Latent ms (BM = 32) |
| ---: | --- | ---: | ---: | ---: |
| 1 | dense / sparse | 0.305 / 0.311 | 0.051 / 0.057 | 0.068 / 0.072 |
| 2 | dense / sparse | 0.307 / 0.577 | 0.054 / 0.058 | 0.068 / 0.072 |
| 4 | dense / sparse | 0.309 / 1.077 | 0.056 / 0.058 | 0.068 / 0.073 |
| 8 | dense / sparse | 0.313 / 1.931 | 0.091 / 0.103 | 0.070 / 0.076 |
| 1024 | dense / sparse | 4.29 / 58.7 | 9.36 / 11.2 | 7.53 / 8.49 |

Decode: -0.25 ms (1 row) to -1.8 ms (8 rows) of attention per layer past 2,051 tokens, about -3 ms to -20 ms a
window over 11 layers. Prefill: a 1024-row sparse chunk's attention drops 705 -> 135 ms over 12 layers. At the
measured 30% share that is ~0.76x the chunk time, **~+30% prefill throughput at 28k** (and similar at 8k). Dense
1024-row chunks (the first 2,051 tokens only) are slower, +5 ms a layer: twice the dot-product width, compute
bound. BM = 32 (32 heads a tile) gave the same bits on these inputs. It is faster from 8 rows up (-2.7 ms a layer
per 1024-row sparse chunk, ~2% of a chunk) but slower for 1-4-row decode windows (+0.015 ms a layer, +0.18 ms a
step), and it must be one fixed value because decode windows and prefill chunks compute the same positions. So the
default stays 16. The knob stays `0` by default until the `exact` suite passes on the real model.

## 0065 — 1M-token memory and the prefill indexer

**Problem.** CONTEXT=1000000 was OOM-killed at load: besides 0060's 18.75 KB a token, the DFlash2 drafter reserved
its context K/V for the whole capacity (10 KB a token a rank), so 29.5 KB a token was reserved. Past 2,051 tokens
every prefill chunk scored all its rows against a power-of-two bucket of pools, one program per (row, 64 pools)
each loading its own copy of the pool tile, and stable-sorted the [rows, bucket] scores (~9 GB of transient
scratch at the end of a 1M prompt, then held by the allocator); the indexer was 14% of a 112k prefill.

**Change.** (a) `dflash2.py`: every drafter layer is a sliding-window layer, so its context lives in a ring of
4,096 slots (`GLM53_TF_DRAFTER_RING`); the attention loop starts at the first 64-key tile any row sees. (b)
`forward.State`: the 11 model layers' index keys and gates in a ring of `index_ring_rows(rows)` rows
(`GLM53_TF_INDEX_RING`; the kernels address rows at position mod ring); pool keys and the MTP layer's caches stay
full. `commit` keeps the last 3 committed rows in a trailing slot of `conv` (which snapshots copy) and `set_pos`
writes them back on a restore. (c) `sparse.select_tokens`: windows of more than 8 rows go through
`select_pools_blocked` (`GLM53_TF_SELECT=blocked`, `GLM53_TF_SELECT_MB`): row blocks within the budget, each scoring
only up to its last row's pools; `_scores_rows` runs `_scores`' per-row code for 16 rows against one pool tile and
writes 0050's key (upper half, int32); the 512th largest key per row by `torch.topk`, then `_gather_sel` lists the
pools above it and the lowest-index ties, in pool order. Details and the memory table: `docs/MEMORY-1M.md`.

**Exactness.** Ring slots hold every position a kernel reads (window + 3 rows; window + block + a tile), and the
drafter's skipped tiles are fully masked no-ops of the online softmax, so every bit is unchanged; a snapshot
restore brings back the incomplete pool's rows. The blocked scores are `_scores`' bits (same per-row code and
shapes; checked on the device at start, `blocked_ok`, which falls back to the sort); pools past a block's last row
are -inf with higher indices than every kept pool; the threshold + lowest-index ties is exactly the first 512 of
the stable descending sort. Decode/verify/MTP windows and 0050's graphs are unchanged. One difference: a snapshot
resumed after the drafter wrote a whole ring past it masks the positions it lost, so drafts (never replies) differ
from a fresh prefill's. Tests: `tests/cuda/test_1m_patches.py`.

## 0080 — fast prefill (the chunk-grid rule, fused EXL3 experts)

**Problem.** Every prefill chunk ran the decode kernels, which are row-invariant (a row's bits never depend on its
chunk-mates). That is what drafted == serial and resumed == fresh rest on, and it rules out large-M GEMM tiles,
split-K choices by M, a chunked KDA scan and reduced-precision gathers. Measured at 1024-row chunks (patches/0005,
`results/M-P1c.profile.log`): 671 / 542 / 493 tok/s at 1.8k / 7k / 28k tokens; per chunk the routed experts take
520-580 ms (26-44%), sparse attention 450-610 ms past 2k, all-gathers 140-190 ms, the KDA chain 110-125 ms, the
4-bit projections and shared expert ~250-280 ms.

**The rule** (`fastpf.py` has it with the proof). Prefill bits need not equal decode bits:

- drafted == serial: both decodings of a request start from the same prefilled state, so any *deterministic*
  prefill keeps it (decode and verify windows stay on the row-invariant kernels);
- resumed == fresh: with C = `prefill_rows` rounded down to a multiple of 64, fast chunks sit at absolute multiples
  of C, and a fast prefill of n tokens keeps only the state at B = floor(n / C) * C (taken before the partial last
  chunk; the prompt's end state when n is on the grid; nothing when B = 0), tagged with C. The reply's rows are never
  kept (decode wrote them). A fast request resumes only from a snapshot of its own grid, and re-prefills from B: the
  old prompt's tail (< C tokens), the old reply and the new tokens. By induction over the full chunks before B, the
  state at B is what a fresh prefill of any prompt extending `ids[:B]` holds there (KDA state, conv window,
  attention / indexer / MTP / DFlash2 caches below B, the MTP head's pending row B-1 with `mtp_len = B - 1`); from B
  on both run the same chunks. The MTP head and DFlash2 stay on the row-invariant kernels.

**Change.**

- `fastpf.py`: the switch, the grid, the dispatch (`chunk(b)` marks a main-model chunk and installs
  `fast_qmm.matmul_prefill` behind `qmm.matmul`; `kda_chain` calls `fast_kda.kda_prefill_chunked`), with 0081's modules
  imported when present (a module that is present but broken is reported at load, not silently skipped).
- `exl3_fast.cu` (own extension, `tensorfold_glm_exl3_fast_v1`): two fused grouped GEMMs per MoE layer. A program is
  (128-column block = one Hadamard block, distinct expert); its 4 warps decode each 16x16 trellis tile of a K slab
  once into shared memory (as the lanes' mma B fragments), the next slab's words in flight, and every warp multiplies
  its own member tiles by the whole slab (64 members a pass for gate/up, 128 for down). One warp sums the full K of
  its rows (no split-K, no Z partials: ~0.8 GB a layer of Z traffic gone at 1024 rows), and the Hadamard output
  rotation, scales, GLM's limited SwiGLU and the down input rotation run on the mma accumulators (butterflies in
  `fwht128`'s order, checked bit-identical in a numpy emulation of the fragment layout); Xd and Y are stored once.
  It is row-independent and deterministic. `exl3.cu`'s `rot_in` still makes the fp16 rotated inputs.
- `forward.py`: `compute(..., fast=True, head=...)`: blocks read `b.fast`; a fast chunk's partials are all-gathered
  as bf16 (`GLM53_TF_FAST_GATHER=bf16`, both ranks add the same rounded partials rank 0 first); the final-normed rows
  are computed for every row (the MTP head reads them) but the head only for the last row of the last chunk.
- `decode.py`: `_prefill` cuts chunks on the grid, takes the grid snapshot (`Snapshot.grid`), and refuses a resume
  off the grid or from the other mode; `engine.py`: `_resume` matches the request's grid, `_run` keeps the grid
  snapshot and no reply snapshot in fast mode; both ranks check the fast settings equal at load; ignored with
  `GLM53_TF_BATCH` > 1 (the batch scheduler keeps exact reply snapshots).

**Cost of the rule.** A conversation's next turn re-prefills the old tail and reply through the fast path (e.g.
~500 + 300 tokens at C = 1024, ~0.8 s at 1,000 tok/s) where the exact mode resumes after the reply. That is why the
switch is also per request (0091). Replies depend on C: a fast request's tokens change with `prefill_rows`.

**Expected** (arithmetic from the measured 1024-row profile; nothing timed): experts -130 to -170 ms a chunk (Z
traffic and epilogue grids gone, weights read once at ~8.4 ms a layer), gathers -60 to -90 ms (bf16), head -11 ms;
with 0081, the KDA chain -90 to -105 ms and the projections -60 to -90 ms.

| Prefill tok/s | 1.8k | 7k | 28k |
| --- | ---: | ---: | ---: |
| measured, exact, 1024-row chunks | 671 | 542 | 493 |
| fast (0080 + 0081), 1024-row chunks, expanded KV | 850-1,000 | 650-750 | 580-660 |
| + latent KV (0060, sparse attention ~5x cheaper) | 850-1,000 | 850-1,000 | 850-1,000 |
| + 2048-row chunks (`PREFILL_ROWS_MAX=2048`, ~10 GB of buffers a rank; the experts' 76 GB read once per 2,048 tokens) | 1,050-1,200 | 1,000-1,150 | 1,000-1,100 |
| vLLM, same weights | 960 | 1,340 | 1,448 |

Next steps, in order of the remaining profile: fuse `rot_in` into the gate/up kernel (~590k one-warp programs and
134 MB a layer at 1024 rows); pipeline each chunk's all-gathers (two half-chunks, A's gather on a side stream while
B computes, layer by layer; needs per-half views of the buffers and the KDA/conv state handed from A to B); tune the
fused kernels' occupancy (237-242 registers: 2 blocks of 4 warps an SM; `__launch_bounds__(128, 3)` spills).

**Verified on GB10** (`tests/cuda/test_fastpf_patches.py`, all green; F3/F5 runs). Fused expert kernels timed on
the real per-rank shapes (288 experts, top 8, hidden 4096, 1024 of 2048 wide; random routing; one MoE layer,
`test_fast_expert_timing` and a profiler split):

| rows, routing | row-invariant (grouped loop + epilogues) | fused (rot_in + gate/up + down) |
| --- | ---: | ---: |
| 1024, uniform | 15.7-15.9 ms (rot_in 0.76, grouped 12.5, epilogues 2.5) | 14.5 ms (0.74 + 7.7 + 6.3); 13.9 with the down prefetch |
| 1024, skewed | 16.6-16.8 ms | 15.9-16.0 ms |
| 2048, uniform | 24.9-25.3 ms | 19.7 ms |
| 2048, skewed | 26.3-26.4 ms | 21.7 ms |

In the model (28k prompt, 1024-row chunks) `moe.routed` went 14.6 -> 13.7 s, and with 2048-row chunks 13.9 -> 12.4 s.
Nsight Compute on the gate/up kernel: 222 registers (2 blocks of 4 warps an SM, 17% occupancy), 25% of DRAM
bandwidth, 43% of the stalls at the slab barrier (warps without member tiles decode and wait). `__launch_bounds__(128,
3)` spills and is slower (8.4 / 8.0 ms). The down kernel loads its A fragments one k tile ahead (6.4 -> 5.8 ms, same
bits); the same in gate/up is slower (the registers). Next: spread the n tiles of a block over the warps (every warp
busy whatever the member count) with the Hadamard epilogue through shared memory.

**fast2 (the default since 2026-09-27; `GLM53_TF_FAST_EXPERTS=v1` keeps the kernels above).** Spreading the work over
every warp (a cp.async ring for the member rows, every warp decoding a share of each slab) gave only 1.2-1.8x: the
kernels became bound by shared memory. Each warp re-read every decoded 16 x 16 fragment (512 B) for its 16 member
rows, ~300 KB a 32-k stage and SM, as long as the stage's mma; removing the mma or the trellis decode changed
nothing. fast2 therefore computes Y^T = W^T X^T:

- A warp owns two 16-column tiles of the item's 128 columns (one matrix) for all of the item's members. It decodes
  its own trellis tiles straight into registers: a decoded tile, as `decode_tile` lays it out, is exactly an
  m16n8k16 A fragment. Decoded weights never touch shared memory, and each fragment feeds (members / 8) mma.
- Only the member rows are shared: a 3-deep cp.async ring of 4-k-tile stages (zero-filled past the last member),
  read with ldmatrix as B fragments that each feed 2 mma.
- Work items (expert, pass of 64 members, 128-column block) come from a one-block plan kernel (per-expert pass
  counts, prefix sum). A persistent grid walks them pass-major, with the column blocks of a pass together. No
  program is launched for an absent expert, and a skewed expert's passes spread over the SMs.
- Epilogue through shared memory: fp32 rows, then one warp a row runs the 128-point transforms (butterflies bit 0 to
  bit 6, fwht128's order) and v1's formulas element by element. Down on chunks of 4096+ rows uses 128-member items
  with 2-k-tile stages (same bits; measured faster there).

Same bits as v1: each element is the same chain of m16n8k16 products over ascending k tiles (swapping A and B gives
the same element bits), the same transforms and the same epilogue. Checked bit for bit on GB10 (1024-8192 rows,
uniform and skewed routing: `tests/cuda/test_fast_experts_patches.py` and the micro-benchmark). Row-independent and
deterministic: the tiling is fixed by the shapes.

| one MoE layer, per rank (ms) | v1 gate/up + down | fast2 gate/up + down | x | + rot_in (unchanged) |
| --- | ---: | ---: | ---: | ---: |
| 1024 rows, uniform / skewed | 13.5 / 14.2 | 9.7 / 10.5 | 1.39 / 1.35 | 0.8 |
| 2048, skewed | 18.8 | 13.6 | 1.38 | 1.5 |
| 4096, uniform / skewed | 28.5 / 29.6 | 17.7 / 19.8 | 1.61 / 1.50 | 2.9 |
| 8192, uniform / skewed | 54.7 / 53.8 | 32.1 / 33.6 | 1.70 / 1.60 | 6.4 |

At 8192 rows fast2 runs at ~50 TFLOP/s against the ~110 TFLOP/s measured mma.sync peak (f16 in, f32 accumulate) on GB10. `rot_in`
(6.4 ms, writing 1.2 GB of Xg/Xu) and down's fp32 Y (1.2 GB) are now a third of the layer. Next steps: fuse the
input rotation into gate/up's A loads (the transform runs on the 128-k blocks of a stage), and store Y narrower (new
bits).

## 0081 — fast prefill kernels (`fast_qmm`, `fast_kda`)

New files only (`families/glm5_next/cuda/fast_qmm.py`, `fast_kda.py`); nothing calls them until 0080's fast-prefill
path (`GLM53_TF_FAST_PREFILL`) try-imports them, so without 0080 the patch changes nothing.

- `fast_qmm.matmul_prefill(x, q, xs=None, *, out=None, f32=False, part=None, exact=True)`: `qmm.matmul`'s contract
  for Q4 and B16, for 64+ rows (fewer go to `qmm.matmul`). One program per BM x BN tile over all of K, M tiles
  consecutive so a column block's weights are read from DRAM about once; no split-K partial buffer and no reduce
  kernel. qmm's per-group arithmetic, and qmm's K slices kept as a summation order: **qmm's bits** (checked on GB10,
  `test_fast_qmm_bitwise`, every shape and every tile config of the sweep). Because of that the choice between the two
  is free, and `TUNED` holds the measured fastest per (N x K) and row bucket (512-1535 / 1536+ rows): a tile config,
  or `qmm.matmul` itself where its split-K kernels win.
- `fast_kda.kda_prefill_chunked(<kda.chain's arguments>, *, pos, save_replay=False)`: the KDA recurrence in 64-row
  chunks at absolute multiples of 64 (WY / UT transform, per-channel decay handled in 16-row blocks with a
  block-local reference so no exponential overflows), fp32 state, tf32 tensor-core products (`precision="tf32x3"` /
  `"ieee"` for more). Three kernels (FLA's layout): (1) one program per (64-row chunk, head), all parallel: prologue
  (conv, norms, gates), A, P, the UT transform T (the four 16 x 16 diagonal blocks solved together, the off-diagonal
  blocks by doubling), W = T(beta K~), U = T(beta V), Q~, K^ -- everything that does not need the state; (2) one
  program per (head, 64 value rows of the state) walking the chunks: E = U - W S^T, O = Q~ S^T + P E, S <- S
  diag(e^{G_C}) + E^T K^; (3) the gated RMSNorm per (32 rows, head). Scratch ~177 KB per (chunk, head), 94 MB at
  1024 rows (grown on demand). Same outputs and final state layout as `kda.chain`; not its bits.

**Measured on GB10** (`tests/cuda/test_fastk_patches.py -s`, 1024 rows):

| kernel | before (F2) | now |
| --- | ---: | ---: |
| KDA, 32 heads (`kda.chain` 3.70 ms) | 13.76 ms (one program a head, serial chunks) | 1.82-1.89 ms (prep 1.06, state 0.65, norm 0.08) |
| `fast_qmm` 12576x4096 / 4096x4096 / 2048x4096 | 3.02 / 1.02 / 0.46 | 2.54-2.71 / 0.80-0.82 / 0.42-0.48 |
| 8192x1536 / 8192x512 / 4096x8192 | 0.76 / 0.27 / 2.49 | 0.66 / 0.23 / 2.04 (qmm) |
| 12288x4096 / 4096x6144 / 4096x1024 | 2.23 / 1.47 / 0.24 | 1.82 (qmm) / 1.30-1.39 / 0.21 |
| 4096x128 / 160x4096 / 4096x1536 | 0.038 / 0.073 / 0.34 | 0.027 (qmm) / 0.045 (qmm) / 0.30 |

`fast_kda.STORE_BF16 = "wuqk"` hands W, U, Q~, K^ to kernel 2 as bf16 (1.50 ms) but doubles the state error against
the serial chain (1.1e-3 -> 2.3e-3; 3.9e-3 with slow decays), so it is off. Kernel 2 is bound by reading those
operands (128 KB a chunk and head); software-pipelining them does not fit the 101 KB of shared memory.

**Exactness.** `fast_qmm`: qmm's bits. `fast_kda`: new arithmetic, deterministic, and a prompt prefilled in calls cut
at multiples of 64 gets the same bits whatever the call sizes; against the serial chain on GB10 (1024 rows, tf32):
state 1.1e-3 relative (1.9e-3 with slow decays), outputs 5.7e-4 mean absolute, at most one bf16 step off at 2^-5.
The ieee variant matches the fp32 reference to 1e-4. `save_replay` writes the chain's replay inputs; a few conv
activations round to the neighbouring bf16 value (the fp32 conv sum's order differs), which the test allows.

`tests/test_fastk_interpreter.py` (the Triton CPU interpreter) skips itself where a GPU is visible: in the image the
interpreter misread qmm's inputs (errors of 1e7), and the GPU tests cover the same checks.

## 0082 — lean prefill (fast chunks of 4,096-8,192 rows)

**Problem.** Bigger fast chunks amortize the routed experts: once a chunk touches every expert, the fused kernels
read each one's weights once a chunk. But the window buffers cost ~3 MB a row a set (and grow faster than the rows
past 1024), there are two sets, and `State`'s KDA replay scratch and projection rows add more:

| rows | row-sized buffers a rank |
| ---: | ---: |
| 1024 | 8.5 GiB |
| 2048 | 17.5 GiB |
| 4096 | 37 GiB |
| 8192 | 82 GiB |

**Change.** `GLM53_TF_LEAN_PREFILL=1` (both ranks; checked at load with `GLM53_TF_LEAN_BLOCK`) keeps the window buffers
(both sets, `State`'s rows) at `GLM53_TF_LEAN_BLOCK` rows (default 1024, a multiple of 64). A fast chunk of up to
`GLM53_TF_PREFILL_ROWS_MAX` rows runs through `lean.py`:

- **Sub-blocks.** Every block runs in sub-blocks of the block's rows on those buffers: hc_pre/hc_post, KDA and DSA
  with their projections and o_proj all-gathers, the dense MLPs, the shared expert, the combine, the MoE all-gather
  and the final norm.
  - KDA: the first sub-block starts from the committed state and conv window. Each later one continues from the
    state its predecessor wrote (the layer's output buffer, through a copy) and the last 3 projection rows it saw
    (`LeanBuffers.tail`).
  - DSA: each sub-block runs as a window at `pos + a` (its keys and index keys written before it attends).
- **Whole chunk, once.** The MoE routing (router, top-k, grouping) and the routed experts (patches/0080's fused
  kernels) run once over all rows.
- **The lean set** (`LeanBuffers`, ~397 KiB a row on the real model) holds what spans the chunk: the streams, the FFN
  half's normed rows and mixes, the routing, Xg/Xu/Xd and `ey`, the final-normed rows (MTP absorb) and the DFlash2
  taps (`Engine.main_hidden` / `tap_rows` read them after a lean chunk).
- **Commit.** `lean.commit` installs the carried conv windows and flips the KDA buffers.
- **Engine.** With lean on, `Engine` splits its rows: buffers of the block, `prefill_max` = PREFILL_ROWS_MAX.
  `tf_knobs.prefill_rows` goes up to `prefill_max` (patches/0090 was regenerated for that: two lines read
  `e.prefill_max`). Exact chunks are capped at the buffers' rows (their bits do not depend on the chunk). With
  `GLM53_TF_BATCH` > 1 the lean set is dropped (fast prefill is off there).

**Exactness.** Bit-identical to 0080's fast chunk of the same rows, so replies do not depend on the switch or the
block. Everything run in sub-blocks is row-independent:

- `fast_qmm` (`matmul_fast`, FP8 too): one accumulator over K, row-independent, and the tile does not change bits.
  Its one row-count switch is `MIN_ROWS`: calls of fewer than 64 rows go to qmm (other bits). So `lean.compute` runs
  the chunk in `lean.chunk_rows(w, R)`: when the chunk has 64+ rows, a partial last sub-block of 1-63 rows still runs
  the fast kernels, as the whole-chunk path does. The head, one row in both paths, keeps qmm;
- glue: per row;
- attention and selection: per row and position;
- the bf16 gather: a copy;
- `fast_kda`: calls cut at multiples of 64 give the same bits given the entering state and conv window, and sub-blocks
  start at `pos + k x block`, multiples of 64.

The experts see exactly 0080's full-chunk call. 0080's grid rule is unchanged. Both ranks must agree on the lean
settings: they set the number of all-gathers.

**Memory and expected speed** (`docs/PREFILL-ANALYSIS.md`, "patches/0082").

| setting | memory a rank |
| --- | --- |
| block 1024 | 8.50 GiB of window buffers + the lean set: 0.78 / 1.55 / 3.10 GiB at 2048 / 4096 / 8192 rows |
| block 1024, 8192-row chunks | 11.6 GiB in total, 5.9 GiB less than the 2048-row config in use today |
| block 512 | 4.19 GiB of window buffers + the same lean set |

Estimated fast-prefill rates with 8192-row chunks (not timed):

| tok/s | 32k | 128k |
| --- | ---: | ---: |
| measured today, 2048-row chunks | 770 | 667 |
| estimated, 8192-row chunks | 830-880 | 710-750 |

The per-row costs (sparse attention, all-gathers, KDA projections, hyper-connections) are now most of a chunk. A
larger grid C also costs ~C / 2 re-prefilled tokens a conversation turn.

## 0083 — FP8 fast prefill (`fp8pf.py`, `fast_qmm.matmul_fp8`, latent tensor-core tiles)

Written offline (no GPU; numerics checked in Triton's CPU interpreter). Nothing here is timed.

**Problem.** In fast chunks the per-row costs now dominate (900-row profile at 8k, us a token a rank: sparse attention
117, all-gathers 106, `kda.proj` 94, `hc` 77, `dsa.o_proj` 76, `kda.chain` 63). The projections are compute-bound bf16
GEMMs (`kda.proj`'s 12576 x 4096 at 37-58 TFLOP/s of a ~60 bf16 peak), and GB10 (sm_121) runs e4m3 `mma` at about
twice the bf16 rate. On the latent cache, `absorb` / `expand` (in `dsa.proj` / `dsa.o_proj`) run as fp32 FMA-pipe dots
over 16-row tiles, dequantizing kv_b in fp32 for every 16 rows: ~8.6 GFLOP each a layer at 1024 rows with no tensor
cores. Sparse latent attention uses 16 (row, head) queries a tile, so each row gathers its 2,051 latent rows twice.

**Change.** `GLM53_TF_FP8_PREFILL=1` (default 0; per request: 0092) switches three things in the MAIN model's FAST
chunks only (`fp8pf.ON`, set by `decode._prefill`; decode, verify, the MTP head, DFlash2 and exact prefills never
see it):

- `fast_qmm.matmul_fp8` behind `matmul_fast` for matrices of at least `FP8_MIN_NK` = 4M weights and calls of 64+
  rows (every KDA / DSA / indexer-q_b / dense-MLP / shared-expert projection; f_b / g_b and the indexer's k|w stay
  bf16, and the head's single row stays on qmm):
  - `_fp8_rows`: each row of x gets its own power-of-two scale s_x, the smallest with amax / s_x <= 448, is rounded to
    e4m3 (RTNE), and its 64-input group sums of the ROUNDED values are kept (fp32). A power of two keeps x / s_x exact
    (no division; the host model reproduces the kernel's e4m3 values bit for bit) at no cost in relative precision.
  - `_f8q4`: a 4-bit value q (0..15) is an exact e4m3 number (`FP8_WENC = "cvt"`; or `"bits"`: the nibble read as an
    e4m3 bit pattern is exactly q x 2^-9, subnormals for q < 8, no conversion instruction; the test checks both give
    the same bits on the hardware). Per group g: P = x8 . q8 on the fp8 tensor cores from zero (at most 64 exact
    products, so the tensor cores' accumulator width does not matter), then `acc += P s[n,g] + X8[m,g] b[n,g]` in
    fp32 in group order, and `y = s_x acc`. That is exactly "x rounded to e4m3 per row, times the exact 4-bit
    weights"; the bias term on the rounded sums makes the error sum((x8 - x) w), not sum((x8 - x)(w - b)).
  - BF16 weights (`GLM53_TF_NONEXPERT=bf16`): per-output-channel e4m3 scales (amax / 448, computed once, kept on the
    matrix), fp32 accumulation in 64-input steps.
  - One tile for every shape and row count (`FP8_TILE` = 128 x 64, 4 warps, 2 stages since the GB10 sweep; `GLM53_TF_FP8_TILE` for the
    hardware sweep): 128 rows share each unpacked weight group, 128 columns share each e4m3 row tile.
- Latent cache (0060): `absorb_tc` / `expand_tc`, the same products on bf16 tensor cores over 64-row tiles (kv_b's
  tile dequantized exactly in fp32, then rounded to bf16; fp32 sums in the same K order).
- Latent attention: `FAST_BM` = 32 (row, head) queries a tile (all 32 local heads of a row: one gather of its selected
  latent rows instead of two). 0060 measured this at 11.2 -> 8.5 ms a layer (sparse) and 9.4 -> 7.5 (dense) for 1024
  rows, and it was kept at 16 only because decode windows must share the tile with prefill; fast chunks need not.
  `FAST_STAGES` (1) is the key loop's pipelining depth, for the hardware (no bit changes).

**Why not FP8 attention.** The latent sparse kernel runs at ~12-16 TFLOP/s on a 1024-row chunk (the MMA pipe is
~25% busy): it waits on the per-tile gathers (32 selected rows of 1 KB each, not pipelined) and the online softmax on
small tiles. FP8 Q'K and PV would halve the part that is not the bound, add e4m3 rounding to q', the latent and P, and
save no bytes (the cache stays bf16). The 32-query tile attacks the gathers instead. **Why not the routed experts
(yet).** `exl3_fast.cu` runs at ~15 TFLOP/s at 1024 rows: bound by trellis decode ALU and the slab barrier (43% of
stalls), not the `mma` rate, and a decoded EXL3 weight is an fp16 value with a full mantissa, so e4m3 would add a
second quantization comparable to the 3-4-bit one. Plan, once the kernel is MMA-bound (after 0080's "spread the n
tiles over the warps" and at 8192-row chunks, ~230 members an expert): `rot_in` writes Xg/Xu as e4m3 with a power-of-two
scale per pair (halving the 1.3 GB of Xg/Xu/Xd at 8192 rows), each decoded 16 x 16 trellis tile is converted once to
e4m3 with a fixed power-of-two scale per expert block (the trellis values are bounded), `mma.m16n8k32.e4m3`, and the
existing scale / Hadamard epilogue with the product of the two scales. Worth measuring only then.

**Exactness.** New arithmetic, like 0080: a fast prefill's tokens change with the switch. Every kernel above is
deterministic (no atomics, fixed order, one tile per shape) and row-independent (a row's scale is its own), so
`fastpf`'s argument holds as is: drafted == serial, resumed == fresh. Snapshots carry the tag C + 1 (`fp8pf.tag`;
C is a multiple of 64), so FP8 fast requests resume only from FP8 snapshots of their grid, and bf16 fast and exact
requests never from FP8 ones (`decode._prefill` refuses a mismatched resume, `engine._grid` picks the tag). Lean
chunks (0082) keep "same bits as the fast chunk" within FP8. Both ranks exchange `fp8pf.settings()` at load. The
response's stats show `fast_prefill: C` and `fp8_prefill: 1`.

**Quality.** e4m3 has 3 mantissa bits: ~2.5% rms relative error per rounded activation, so each projection output
carries ~2-3% relative noise (measured on random inputs with x30 outlier channels: 2.7% against fp32; the 4-bit
weights' own error is larger). Residual streams, norms, the KDA recurrence, softmax and the experts stay as before.
This is the arithmetic of W8A8-FP8 with dynamic per-token scales, which is typically within noise on MMLU-style
evaluations; check with MMLU (`fp8_prefill` 0 vs 1 on the same server) before making it a default. The synthetic-model
test prints the FP8 vs bf16 fast prefill distance of hidden rows, logits and KDA states.

**Expected** (per rank, us a token; arithmetic from the 900-row profile and 0081 / 0060's kernel timings):

| component | before | after | how |
| --- | ---: | ---: | --- |
| `kda.proj` | 94 | 55-65 | 12576 x 4096 at 1.5-1.8x, plus ~2 of row quantization |
| `kda.o_proj` (in the rest) | ~20 | 12-14 | 4096 x 4096 in fp8 |
| `dsa.o_proj` | 76 | 25-45 | expand on tensor cores (the FMA-pipe expand is most of it: the 4096 x 8192 matmul alone is ~13), o_proj in fp8 |
| `dsa.proj` (absorb, q_a / kv_a, q_b) | ? | -15 to -30 | absorb on tensor cores, the projections in fp8 |
| sparse attention | 117 | ~89 | 32-query tiles (-24% per kernel, measured in 0060) |
| shared expert, dense MLP, indexer q_b | ~30 | ~20 | fp8 |
| **total** | | **-115 to -165** | |

| prefill tok/s | 8k | 32k | 128k |
| --- | ---: | ---: | ---: |
| 2048-row chunks, measured / modelled today | 889-994 | 770 | 667 |
| + 0083 | 990-1,190 | 845-885 | 725-750 |
| lean 8192-row chunks (0082 estimate) | 966-1,183 | 827-879 | 710-747 |
| lean 8192 + 0083 | 1,090-1,470 | 915-1,030 | 775-850 |
| vLLM, same weights | 1,340 (7k) | 1,448 (28k) | - |

0084's -90 to -120 us a token (all-gathers, hc) is independent of these and adds on top. At 128k the context-growing
costs (indexer scores, sparse selection) dominate what is left.

**Verify on the Spark.** `tests/cuda/test_fp8_patches.py -s` (prints the fp8 vs bf16 timings per shape and M, a tile
sweep, absorb / expand, and the quality distances), then serve with `GLM53_TF_FAST_PREFILL=1 GLM53_TF_PROFILE=1` and
compare `"tf_knobs": {"fp8_prefill": 0}` against `1` on 8k / 32k / 128k prompts, and MMLU with each. If
`test_fp8_weight_encodings_and_tiles_same_bits` fails only for "bits", the tensor cores flush e4m3 subnormals: keep
"cvt" (the default). Set `FP8_TILE` to the fastest config of the sweep. Done (2026-09-27): 128,64,4,2 at every shape, M = 2048 and 8192 (1.0-1.27x the bf16 fast kernel; 128,128,8,3 was 0.8-0.9x, so before the switch the FP8 matmuls were a net loss and the FP8 gain came from absorb / expand). `test_fp8_patches.py` host-model tolerances: 6e-5 (Q4, measured 2.6-3.2e-5) and 5e-4 (BF16 weights, measured 1.9e-4): fp32 summation order only.

## 0084 — pipelined lean prefill (all-gathers hidden, hyper-connections in L2)

**Problem.** In the fast-prefill profile (per token and rank, 2048-row chunks) the all-gathers took 106 us and the
hyper-connections 77 us. Every exchange ran on the compute stream, and the next kernel waited for it. About a third
of the `allgather` time was not the link: `forward.gather` copied the fp32 partial to bf16 before the exchange and the
gathered bf16 back to fp32 after it (88 KB a row a site, 90 sites a token). The hyper-connection kernels are
memory-bound over the [rows, 4 x 4096] bf16 streams. hc_post reads and writes them, then the next hc_pre reads them
twice (its mixing dots, then the collapse). Each read came from DRAM, because a 1024-row sub-block of streams (32 MB)
does not fit in L2.

**Change** (`pfoverlap.py`; hooks in `forward.out_proj` / `gather`, `lean.py`, `glue.hc_post` / `combine`). With
`GLM53_TF_PREFILL_OVERLAP=1`, a lean chunk (0082) runs as a sequence of pieces (layer, half, sub-block). Each piece's
pre work ends in a rank partial, and its post work consumes the gathered partials.

- `gather`: piece k's all-gather goes on a comm stream. It is high priority and has two partial/gather slots with
  events. The compute stream runs piece k+1's pre work (hc-mixed inputs, projections, KDA chain or attention, o_proj;
  or shared expert + combine), then waits for exchange k and runs post(k). Rules:
  - A piece whose inputs come from the post just before it drains first: one-sub-block chunks, and the MoE routing,
    which reads every row.
  - A slot is rewritten only after its exchange was waited for. `Pipe` checks this.
  - Both ranks issue the same collectives in the same order from one stream.
- `direct` (always on in the pipeline): o_proj / down store the partial in the exchanged dtype (bf16 by default), the
  MoE combine stores bf16, and hc_post reads the gathered bf16 and widens it itself. That is 96 KB a row a site less
  traffic and 2 fewer kernels a site.
- `slab`: a piece's post runs hc_post and then the next hc_pre of the same rows, in slabs of
  `GLM53_TF_PREFILL_HC_SLAB` rows (default: half the L2 over 32 KB a row, 384 rows at 24 MB). The next hc_pre is the
  FFN's after attention, the next layer's attention's after the FFN, or the final norm and taps after the last layer.
  hc_pre's two reads then hit L2. Its outputs wait in the lean set's rows (`lb.normed` / `xs` / `post` / `comb`), which
  0082 already holds. DSA sub-blocks copy their 8 KB a row into the window buffers (11 layers); KDA reads the lean rows
  in place.

**Exactness.** The same kernels run on the same values; only the order and three store/load dtypes change:
- The o_proj / down matmuls and the combine now store bf16. That is the fp32 sum rounded to nearest even in the
  store, which is what `copy_` did.
- hc_post widens bf16 to fp32 itself. That is exact, so it sums the same fp32 values in the same order.
- Slabs are cut at multiples of 64 rows, and every glue / hc kernel is row-independent.

So the committed state and replies equal 0082's lean chunk bit for bit. The ranks need not even agree on the knob:
the collectives and their order are the same. Tests: `tests/cuda/test_overlap_patches.py`.

**Not done, and why.**
- Merging hc_post with the next hc_partial into one kernel would add little over the slabs: a few L2 re-reads and one
  launch a slab. It would also need fast_qmm's row-tiled dots and sum of squares reproduced bit for bit, which only
  running the same kernel guarantees (a Triton reduction's order follows the layout the compiler picks).
- Re-tiling hc_finish (per-row Sinkhorn over 4 x 4) would change its reduction order: new bits.
- Coalescing exchanges: a layer's MoE partials (or its attention partials) could go in one exchange of the whole
  chunk, exact since a gather is a copy. That saves only (sub-blocks - 1) x alpha a site, about 1-2 us a token, and
  leaves nothing to overlap the exchange with. Pipelining beats it.
- GLM53_TF_FAST_GATHER=fp32 works (fp32 slots, twice the wire bytes: overlap matters more there).
- 0040's L2 prefetch is skipped for pipelined exchanges: its plans are decode-shaped.

**Expected** (arithmetic, per token and rank, against the profile above; nothing timed):

| part | saves |
| --- | ---: |
| `direct`: 2 conversion passes + narrower partial writes / hc_post reads, 96 KB a row x 90 sites | 35-40 us |
| `gather`: the ~70 us of NCCL + wait left, minus the MoE boundary (the last attention exchange overlaps only the previous sub-block's post) and NCCL's SMs taken from the compute kernels | 40-60 us |
| `slab`: hc_pre's 2 x 32 KB a row from L2, 90 sites, minus slab tails | 15-22 us |
| total | ~90-120 us |

At 32k: 2048-row chunks 770 -> ~830-850 tok/s. 8192-row lean chunks (est. 830-880) -> ~910-970 tok/s.

**On the Sparks.**
- `GLM53_TF_PROFILE=1`: `allgather` is now the time the compute stream still waited for.
- Compare tok/s at 8k / 32k / 128k with `prefill_overlap` 0 vs 1 (per request, no restart), and each variant alone
  (`gather`, `slab`, `direct`).
- Try `NCCL_MAX_NCHANNELS=1` or `2` (fewer SMs taken by NCCL beside the compute kernels) and `GLM53_TF_PREFILL_HC_SLAB`
  256 / 384 / 512.
- Take an nsys trace of one chunk, to see `ncclDevKernel_AllGather` running beside the compute kernels.

## 0085 — chunk-size-independent fast prefill (`pfgrid.py`, 64-token snapshots, `prefill_rows=auto`)

Written offline (no GPU); the host tests ran, the GPU tests are for the Sparks.

**Problem.** 0080's grid rule put fast chunks and snapshots on absolute multiples of C and let a request resume
only from a snapshot of its own C, because the fast kernels were only required to be deterministic. At C = 8192 (the
fastest cold prefill, 1,034 tok/s at 28k) a follow-up turn re-prefilled up to C - 1 old tokens plus the reply:
3.6-6.8 s warm TTFT against ~0.7-1 s at C = 1024 (`docs/RESULTS.md`).

**Audit** (is a row's output independent of which rows share the call, and of how many?):

| kernel / op on the fast path | row-independent? | why |
| --- | --- | --- |
| `fast_qmm.matmul_fast` 4-bit (`_fq4`, one accumulator) | yes, but **not below 64 rows** (fixed) | tile by (N, K) only (`LOOSE` / `LOOSE_TUNED`, never by M), one fp32 accumulator walking K in group order, no split-K; `GROUP_M`'s grouped program order only reorders programs. `MIN_ROWS`: a call of < 64 rows went to qmm's split-K kernels (other bits), so a prompt's last chunk of 1-63 rows got other bits than the same rows inside a longer chunk |
| `fast_qmm` BF16 weights (`_fb16`) | yes (tile now fixed) | BM was picked by M (>= 512 rows); a tile never changes a row's operations (mma rows are independent, K order fixed by BK), but the choice is now by shape only |
| `matmul_fp8` (`_fp8_rows`, `_f8q4`, `_f8b16`) | yes, but not below 64 rows (fixed) | per-row power-of-two scale and group sums; `FP8_TILE` fixed for every shape and M; the per-channel weight scales are per weight. < 64 rows fell back to the bf16 kernel / qmm |
| `hc_partial` | yes | 64-row tiles, fixed K order, per-row sum of squares |
| latent `absorb_tc` / `expand_tc` | yes | `TC_ROWS` = 64 fixed, fixed K order |
| `exl3_fast.cu` gate/up and down | yes | a member tile's warp runs one ascending mma chain over all of K; the expert's member count only decides which pass / warp holds a pair (no K split by members, no reduction across members); epilogue per element; `rot_in` per pair. Keep it so in the "spread n tiles over warps" rework |
| router, top-k, grouping, combine, glue, hc_pre / hc_post, norms | yes | the row-invariant kernels, or per-row arithmetic |
| attention (dense / sparse, latent / expanded, `FAST_BM` 32 = one row's heads) | yes | the row-invariant kernels of every exact prefill; a row sees keys <= its position whatever the chunk (`nch`, `pool_bucket` only add masked work) |
| indexer blocked selection (0065, `GLM53_TF_SELECT_MB` row blocks) | yes | exact selection (== sort), scores per row |
| bf16 gathers, 0084 overlap, slabs | yes | copies; 0084 is bit-identical by construction |
| lean sub-blocks (0082) | yes | sub-blocks at `pos + k x block` (multiples of 64); `lean.chunk_rows` already forced the fast kernels for a < 64-row tail sub-block |
| `fast_kda` (3 kernels) | per 64-row block at ABSOLUTE multiples of 64 | `off = pos % 64`: a call starting mid-block would split the block into two partial ones (other bits); a block's bits depend on its rows and the fp32 state entering it, and a state handed between calls is the fp32 value the one-call scan keeps in registers. Every fast call now starts on a multiple of 64 (asserted in `fastpf.kda_chain`); the only partial block is the prompt's last one, the same in every schedule |
| MTP head absorb, DFlash2 taps, the head | yes | row-invariant kernels; the head runs on the one last row in every schedule (kept on qmm) |

**Change.**

- `fastpf.chunk(b, head)` / `fast_qmm.FAST_HEAD`: in a fast chunk every matmul but the head's runs the fast kernels
  whatever its row count (`min_rows` on `matmul_prefill` / `matmul_fp8`; direct calls keep `MIN_ROWS`). The one-accumulator
  BF16 tile is chosen by shape only. `fastpf.kda_chain` refuses a position off the 64 grid.
- `pfgrid.py` (new, no torch; the one place the grid / tag / chunk rules live, for 0110 and 0120 to call):
  `tag(fast, fp8)` = 0 / G / G + 1 (G = `GLM53_TF_SNAPSHOT_GRID`, default 64; never C), `grid_of`, `resumable`
  (same tag, position a multiple of 64), `parse_rows` ("auto" = 0), `chunk_rows` (auto: the rows to prefill rounded
  up to 64, at most the buffers), `plan(begin, n, C, marks)`: chunks of C from the resume point, cut at marks and at
  the snapshot point. The module docstring has the proof.
- Snapshot rule: one fast snapshot per prefill, at S = the prompt's last multiple of G (after the prefill when S = n,
  else before a chunk starting at S; the prefill cuts one there). Tail rule (`GLM53_TF_SNAPSHOT_TAIL`, default
  256): when the schedule's last chunk already starts within that many rows before S, snapshot there instead (saves
  an extra < 64-row chunk now; the next turn re-prefills at most tail + 63 more rows). The reply is still never kept
  in fast mode: its rows were written by the row-invariant decode kernels, which a fast prefill of the next prompt
  does not reproduce, so it is re-prefilled.
- `decode._prefill`: `pfgrid.plan` replaces the C grid; `e.checkpoints` marks (0110's) cut chunks and snapshot
  there (`e.mark_snaps`); `e.fast_rows` = the C used (`stats.fast_prefill`). `engine._grid` = `pfgrid.tag`, so
  `_resume` offers every fast snapshot of the request's mode whatever its C. `GLM53_TF_PREFILL_ROWS=auto` at load;
  both ranks exchange G and the tail with `fastpf.settings()`.
- 0090 (regenerated): `tf_knobs.prefill_rows` accepts `"auto"` (0 in the header; rank 1 applies it; the chunk size
  is then a pure function of the header's prompt and resume lengths and the load-checked buffers, so both ranks cut
  the same chunks); the echo says "auto"; `PREFILL_ROWS_MAX` defaults to 64 with `auto`. 0091-0093 apply unchanged.
- 0110: its `_prefill` loop hunk is gone (0085 does the marks); `sessions.snapshot_grid(tag)` already reads G from
  the tag, so fast pages are keyed by their 64-token block.
- Tests that check 0080's C-grid rule (hostile C-dependent fakes, snapshot positions) run with
  `GLM53_TF_SNAPSHOT_GRID` = C, which is exactly 0080's rule (`test_fastpf` / `test_lean` / `test_session` engines,
  `_FakeEngine.snap_grid`).

**Exactness.** Within fast mode: drafted == serial (deterministic prefill), resumed == fresh for ANY pair of chunk
sizes (and auto). A fast reply now depends only on the tokens, not on `prefill_rows`. Against 0084's bits: identical
except prompts whose last fast chunk had 1-63 rows (they now run the fast kernels there instead of qmm).

**Follow-up turn cost** (auto C, 8192 max; per-row ~1 ms and a tiny chunk ~0.2-0.3 s estimated from the 28k
profiles, nothing timed): a turn re-prefills the old prompt's last < 64 tokens (< 320 under the tail rule) + the reply
+ the new tokens in one chunk, plus a < 64-row tail chunk to put its own snapshot on the grid: ~0.8-1.1 s for a
300-token reply and a short new message, against 3.6-6.8 s at C = 8192 before; a cold 28k prefill gains one tail chunk
(+0.2-0.3 s, ~1%).

## 0090 — per-request knobs (`tf_knobs`)

**Problem.** A load takes ~6 minutes on the two Sparks, and every speed knob above was read from the environment
at load, so each A/B variant cost a restart.

**Change.** A request may carry `"tf_knobs": {...}` (any subset; `families/glm5_next/cuda/knobs.py`). The
GLM53_TF_* environment now only sets each knob's default; after the request the defaults apply again (also when it
fails). The response's `tensorfold.tf_knobs` (non-streamed body, and the last streamed chunk) echoes every knob's
value for that request.

| key | values | default from | per request? |
| --- | --- | --- | --- |
| `lookup` | 0, 1 | `GLM53_TF_LOOKUP` (1) | yes (auto/o's lookup gate) |
| `lookup_min` | 1-64 | `GLM53_TF_LOOKUP_MIN` (4) | yes |
| `auto_fdrafts` | 1-7 | `GLM53_TF_AUTO_FDRAFTS` (7) | yes |
| `expert_loop` | 0, 1 | `GLM53_TF_EXPERT_LOOP` (1) | yes (only windows > 16 rows use it: never graph-captured) |
| `prefill_rows` | 1 to `GLM53_TF_PREFILL_ROWS_MAX` | `GLM53_TF_PREFILL_ROWS` (64) | yes; the buffers are sized for the max (default = PREFILL_ROWS) |
| `calib_online` | 0, 1 | `GLM53_TF_CALIB_ONLINE` (0) | yes (rank 0's table travels in the header as before) |
| `longctx_graphs` | 0, 1 | `GLM53_TF_LONGCTX_GRAPHS` (1) | yes when loaded with 1 (the buffers exist); `1` refused on an engine loaded with 0 |
| `profile` | 0, 1 | `GLM53_TF_PROFILE` (0) | yes |
| `depth` | `"cost"`, `"threshold"` | `GLM53_TF_DEPTH` (threshold) | yes (plain `auto`; `o`/`om`/`of` policies are cost-derived anyway) |
| `nonexpert`, `latent_kv`, `comm`, `batch`, `prefill_rows_max`, `calib` | | | no: HTTP 400 naming the reason (weight format, cache layout, graph-captured comm / NCCL env, scheduler, buffer size, load-time calibration) |

Rank 0 validates the knobs in `App.check` (HTTP 400 before anything streams: unknown key, load-only key, range,
type, `GLM53_TF_BATCH` > 1). The header gains, after 0071's cost flag and before the policy code, a block
`[6, expert_loop, prefill_rows, longctx_graphs, profile, auto_fdrafts, calib_online]` holding rank 0's value of
every knob for the request (its defaults filled in); rank 1 sets exactly these, runs the request, and restores its
own. `lookup`/`lookup_min` travel in the policy code (0020's slots), `depth` as 0071's flag, `calib_online`'s
effect as 0070's table. Both ranks check `GLM53_TF_PREFILL_ROWS` and `GLM53_TF_PREFILL_ROWS_MAX` equal at start.

**Exactness.** Each knob only selects between paths already shown to give the same bits: chunk size (0003: rows
never depend on chunk-mates; resumes across chunk sizes match), `grouped_loop` vs grid (0006: same work item per
member tile), bounded/graph vs upstream selection (0050), timing-only probes (0005), and draft depth / lookup /
cost tables (0010/0020/0070/0071: they choose which drafts are verified; the verify window, keyed sampler and commit
decide every token, so drafted == serial). Rank 1 never reads its own environment for these: the values come from
rank 0's header, so both ranks run the same windows and collectives. Tests: `tests/cuda/test_knob_patches.py`.

## 0091 — per-request `fast_prefill` knob

`"tf_knobs": {"fast_prefill": 0|1}` (default `GLM53_TF_FAST_PREFILL`), in 0090's header block (now 7 knobs; both
ranks must run the same patch set). The request's grid is `fastpf.grid(prefill_rows)` of its own `prefill_rows`, and
`_resume` only offers snapshots of that grid (exact requests: grid 0), so the two modes never resume from each
other's states. Refused when the buffers hold fewer than 64 rows. Unlike every 0090 knob, this one changes a
request's tokens (the prefill's arithmetic); drafted == serial and resumed == fresh hold within each setting.

## 0092 — per-request `fp8_prefill` knob

`"tf_knobs": {"fp8_prefill": 0|1}` (default `GLM53_TF_FP8_PREFILL`), in 0090's header block after `fast_prefill` (8
knobs; both ranks must run the same patch set). No effect on a request with `fast_prefill: 0`. The request's snapshot
tag is `fp8pf.tag(grid(prefill_rows), fp8_prefill)`, so `_resume` offers FP8 requests only FP8 snapshots of their grid.
Refused (HTTP 400) where FP8 tensor-core dots are unavailable (`fp8pf.available()`). Like `fast_prefill`, it changes a
request's tokens; drafted == serial and resumed == fresh hold within each setting.

## 0093 — per-request `prefill_overlap` knob

`"tf_knobs": {"prefill_overlap": 0|1}` (default: `GLM53_TF_PREFILL_OVERLAP` on or off), in 0090's header block after
0092's `fp8_prefill` (9 knobs; both ranks must run the same patch set). `1` runs the environment's variant, or
`gather,slab` when the environment has none. Same bits either way (0084), so snapshots are shared across the setting.

## 0110 — multi-session state cache (`sessions.py`)

**Problem.** The engine kept the committed state of the last request only (the snapshots after its prompt and its
reply, `GlmEngine.cache`), so an agent switching between sessions (opencode's subagents, parallel tool loops)
re-prefilled the whole context on every switch: ~34 s at 30k tokens, ~115 s at 100k. The vLLM kit keeps ~14
conversations cached (97%+ prefix hits).

**Change.** `GLM53_TF_SESSION_GIB=N` (default 0: off, upstream behaviour; `config/tensorfold.env.example` and
`docker/compose.yaml` set 12) keeps a store of session entries per rank (`families/glm5_next/cuda/sessions.py`,
design in `docs/SESSIONS-DESIGN.md`):

- *Entry* = a `decode.Snapshot` (ids; KDA recurrent states + conv windows, which also hold 0065's index-ring tail;
  pending MTP rows; `mtp_len`, `drafter_end`, grid tag) + every per-position cache row below its length: DSA KV rows
  (latent or expanded), full-size index keys/gates (the MTP layer's; the model layers' only with
  `GLM53_TF_INDEX_RING=0`), pool keys, and the MTP head's rows below `mtp_len`. With DFlash2, the snapshot also keeps
  the drafter's context rows in its sliding window (`Snapshot.window`, ~20 MB), so a restored session drafts as
  before (0065's ring alone would mask what other sessions overwrote: drafts, not replies, would change).
- *Pages*: rows live in 256-token pages (16-page slabs per cache tensor), keyed by the tokens that determine their
  bits: exact rows by their prefix plus the next token (an MTP row reads it); fast rows (0080, grid C) by every token
  up to the end of their chunk; plus the tag and MTP validity. Equal keys share one page (reference counted), so a
  system prompt common to many subagents is stored once. The rest of an entry (last partial page, a reply's MTP
  backlog) is a private tail.
- *Restore* (copy-in): the entry's pages and tail are copied into the live caches at their positions, skipping
  pages the live caches already hold (tracked per page), plus the drafter window; `prefill(resume=snapshot)` then
  restores the KDA state as for any resume. A request uses the store only when it resumes more of the prompt than
  the live snapshots; `usage.prompt_tokens_details.cached_tokens` (and `tensorfold.cached`) is the resumed length.
- *Saves*: after the prefill, the prompt snapshot (exact: at the prompt end; fast: at its last grid point) and the
  marks; after the reply, the reply snapshot (exact only), exactly the snapshots the engine already keeps.
- *Marks*: the KDA state cannot be rebuilt from attention rows, so a prompt that forks from a stored one can only
  resume at a snapshot. Rank 0 asks the prefill for extra snapshots at the page (exact) / grid point (fast) at or
  before the longest common prefix with any entry (at least `GLM53_TF_SESSION_FORK_MIN` = 512 tokens past the resume
  point) and every `GLM53_TF_SESSION_EVERY` = 16,384 tokens; an exact chunk is cut there (no bit changes).
- *Eviction*: least recently used entry (restored, resumed from, saved again = used) until the new one fits; pages go
  with their last reference, empty slabs are released; an entry larger than the budget is not stored. Rank 0 also
  skips a save that would leave less than `GLM53_TF_SESSION_RESERVE_GIB` (2) of device memory.
- *Two ranks*: rank 0 plans (entry to restore, marks) and sends the plan after the prompt with a digest of its store;
  rank 1 checks the digest and follows. For each save rank 0 sends its decision (stored / skipped / duplicate, and the
  entries it evicted); rank 1 applies it and fails loudly if its store disagrees. Settings are checked equal at load.
  Off with `GLM53_TF_BATCH` > 1 (0030's slots keep their own states) unless `GLM53_TF_BATCH_SESSIONS=1` (0180).

**Memory a rank (real model, latent KV, index ring on).** 13.25 KB a token of pages (12 x 1 KB latent, the MTP
layer's index keys/gates 0.5 KB, 12 layers' pool keys 0.75 KB; 18.75 KB with `GLM53_TF_INDEX_RING=0`); ~94 MB a
snapshot (KDA 71.3 MB, conv 2.6 MB, DFlash2 window ~20 MB). A session with its prompt and reply snapshots:

| Tokens | Pages | + 2 snapshots | + marks (every 16k) | Sessions in 12 GiB (with marks, no sharing) |
| ---: | ---: | ---: | ---: | ---: |
| 8k | 0.11 GB | 0.30 GB | 0.30 GB | ~43 |
| 30k | 0.41 GB | 0.60 GB | 0.69 GB | ~19 |
| 100k | 1.36 GB | 1.55 GB | 2.11 GB | ~6 |

Expanded KV (0060 off) is 390 KB a token: the store works but holds ~30x fewer tokens.

**Switch latency (estimate).** A restore copies the entry's bytes once: 30k tokens ~0.5 GB, ~4-5 ms at ~110 GB/s of
device copy (100k: ~13 ms; 8k: ~2 ms), plus ~1 ms of host hashing; then the new turn's tokens prefill as usual. A
save after the prefill copies only new pages (a new 30k prompt: ~4 ms) and the snapshot.

**Exactness.** No new arithmetic. A page's key fixes every input its rows were computed from (exact rows are
row-invariant functions of their prefix; fast rows of their chunk and everything before it; 0080's lemma), so a
shared page has the bits the entry's own prefill wrote; tails, KDA states and drafter windows are copies of the
live state when the snapshot was taken. Snapshots are taken only where the engine already takes them (any position
exact, the grid fast; marks included), so restoring an entry gives the fresh prefill's state: resumed == fresh, and
drafted == serial as before. Tests: `tests/cuda/test_session_patches.py`.

## 0120 — batching on the current engine (`batch.py`, `batchplan.py`)

**Problem.** 0030 (`GLM53_TF_BATCH=N`) was written against 0001-0020. In batch mode DFlash2 ran as MTP drafts, the
lookup drafter and cost-derived depths were not used, `tf_knobs` were refused, fast / lean prefill were switched off,
windows over 4 rows and every round past 2,051 tokens ran eager, and an admitted prompt prefilled whole while the
others waited (a 100k prompt: minutes).

**Change** (design: `docs/BATCHING-DESIGN.md`, "v2 on the current engine").

- *Per sequence*: a `State` per slot (latent KV, index rings sized for prefill chunks), MTP graphs per slot (0050's
  long-context ones too), a DFlash2 context per slot (`_drafter_view`: shared weights, own context ring, positions
  and graphs), and `Stepper`, `decode.auto_decode`'s round cut at the forward, so every policy (MTP, DFlash2, `auto`,
  `lN` and auto's lookup gate, thresholds, `o` / `om` / `of`) drafts per request exactly as alone. Slots other than 0
  keep 8-row KDA window buffers and borrow slot 0's for a prefill.
- *Knobs*: a request's `tf_knobs` travel in its admission header (`batchplan.encode_header`); its prefill runs in
  `GlmEngine._knobs(values)`, its drafter with its `auto_fdrafts` / lookup / depth. Only `calib_online=1` is refused
  in batch mode (`knobs.parse`).
- *Prefill in pieces*: `GLM53_TF_BATCH_PIECE` tokens (fast: multiples of the chunk grid) a piece, each a
  `decode.prefill` resumed from the previous piece's snapshot, one piece a round, shortest remaining prompt first; while
  others decode, pieces take at most `GLM53_TF_BATCH_PREFILL_SHARE` of the time. Fast requests keep grid snapshots
  only, exact ones prompt and reply snapshots, per slot.
- *Rounds*: one forward over every decoding window; patches/0050's device-side selection per slot past 2,051 tokens
  (latent and expanded); one all-gather for every request's sampling candidates; CUDA graphs captured lazily per
  (slots, rows per slot, dense / pool bucket) up to 8-row windows and 256 graphs, KDA states normalized to buffer 0
  before a graphed round. Cost-derived depths price rows past the other slots' rows at the rounds' aggregate rate.
- *Server*: `"priority": "background"` or a session-title request waits behind foreground requests and steps aside
  (runs again later; its caller gets each token once) when one waits and no slot is free.
- *Admission control*: at load, slots are added while `GLM53_TF_BATCH_RESERVE_GB` stays free (both ranks agree on
  the count); at admission a request waits while less than `GLM53_TF_BATCH_ADMIT_GB` is free and others run.
- *Two ranks*: rank 0 plans each round (cancels, admissions with headers, the piece) as one int list
  (`batchplan.encode_plan`); everything else is computed alike from shared inputs. Settings checked equal at load.
- `engine.py`: the batcher is built after the knob defaults and before 0110's store (which stays off in batch mode);
  `generate` no longer refuses knobs. `app.py`: the background flag.

**Exactness.** A request's rows are its lone forward's bits (row-local kernels; per-slot kernels see only the slot's
state and rows), its prefill is `decode.prefill` (in pieces: resumed == fresh in every mode), its commits are
`forward.commit` on its own state, its sampling the keyed rule on the same candidates; drafts only propose. So each
batched reply equals the same request served alone and serial decoding, whatever shares its rounds; per-request knobs
change a request's tokens exactly as they do alone (fast / FP8 prefill), and never another request's. KDA parity
normalization moves the same values. Tests: `tests/cuda/test_batch2_patches.py` (replaces `test_batch_patches.py`).

**Expected** (arithmetic, BATCHING-DESIGN section 6 model): ~60 tok/s total at 2 sequences (1.3-1.35x one MTP
stream), ~80-90 at 4 (1.25-1.4x vLLM's 63-66). ~0.85 GB a rank per extra sequence at 32k, ~2.1 GB at 128k (latent KV).
Not run on a GPU yet.

## 0140 — fast restarts (`fastboot.py`)

**Problem.** A restart took 274-490 s (launcher to ready; 303-313 s in the current configuration), and almost all of
it rebuilt the same thing: each rank sliced its half out of the full 164 GB checkpoint through a file-backed mmap
(single thread, 1.4 ms and ~0.4 GB/s of share a tensor, over ~150k tensors, touching ~1/3 more bytes than the half),
re-quantized the BF16 non-experts (q4mse clip search), and 4-bit-quantized the drafter; then timed every verify
window and drafter again (patches/0070). The Spark's NVMe streams ~10 GB/s with O_DIRECT. `docs/BOOT.md` has the
timeline.

**Change.**

- `weights.load` / `Drafter.read_weights` (new, the drafter's old reading code) go through `fastboot.cached_tree`:
  with `GLM53_TF_PREPARED=DIR`, a valid folder `DIR/<model>-<rev>/<key>/rank<R>` is read back instead of building;
  otherwise the checkpoint path runs (and, with `GLM53_TF_PREPARED_WRITE=1`, the result is written for the next
  start). A folder holds the built objects as they are: every tensor's storage bytes (4 KiB aligned in one
  `data.bin`) with dtype, shape, stride and storage offset, and the dataclasses around them (`manifest.json`, which
  also has a SHA-256 per 64 MiB chunk). The key covers the checkpoint (revision, file names and sizes,
  `config.json`), the rank, `GLM53_TF_NONEXPERT`, torch's version, the device type, and the source of every
  function and class that builds the weights (`weights.py`, `split.py`, the quantizers, the EXL3 word layout, the
  drafter's reader), so a patch that changes how weights are built misses old folders. `scripts/prepare.sh` writes
  both ranks' folders once (`python -m tensorfold.families.glm5_next.cuda.fastboot prepare|verify|status`).
- Reader: 8 threads read 64 MiB chunks with O_DIRECT (buffered + `POSIX_FADV_DONTNEED` where O_DIRECT is refused)
  into pinned buffers and copy each chunk's pieces to the device on a stream per thread; no page cache (on GB10 it
  is GPU memory) and no file-backed mmap handed to CUDA. `GLM53_TF_PREPARED_VERIFY`: `sample` (default, every 16th
  chunk and the last), `full`, `off`; a mismatch falls back to the checkpoint.
- `GLM53_TF_CALIB=cached`: both ranks all-gather a digest of their identity (`GLM53_TF_IMAGE_ID`, every
  `GLM53_TF_*` knob but launch-only ones, context / capacity / drafter / policy / prefill rows, torch, GPU name,
  nvidia-smi's clocks and power limit, `GLM53_TF_CLOCK_CAP`); rank 0 looks up `calib-<both digests>.json` in
  `GLM53_TF_CALIB_DIR` (`/cache/calib`) and shares its bytes, so both ranks hold the same floats; a hit prefills the
  calibration prompt once (warm-up, `GLM53_TF_BOOT_WARMUP=0` skips it), a miss measures as `real` and stores the
  table. `real` always measures and refreshes the stored table (the forced re-measure).
- `[boot] rR +T s phase (own s) | MemFree, MemAvailable` lines from the engine (start, NCCL, weights, barrier,
  drafter, engine + graphs, drafter graphs, calibration, ready), counted from `serve.sh`'s launch
  (`GLM53_TF_LAUNCH_T0`) or the entrypoint (`GLM53_TF_T0`).
- Image / launcher: `CUDA_CACHE_PATH=/cache/nv/ComputeCache` (4 GiB) joins the torch-extension and Triton caches in
  the `/cache` volume; `serve.sh` starts both ranks at once, polls readiness every second (the worker over ssh
  every 10th), mounts `HEAD_PREPARED` / `WORKER_PREPARED` at `/prepared`.

**Exactness.** A prepared tensor is the built tensor's bytes and layout, so replies cannot change: tested bit for
bit against `load_checkpoint` for both ranks, every NONEXPERT mode and the drafter (`tests/test_fastboot_prepared.py`
on the CPU, `tests/cuda/test_boot_patches.py` on the GPU, where the load-time path is also checked deterministic),
and an engine on prepared weights replies as one built from the checkpoint. The cached calibration table is the
table a real calibration produced for the same key; costs only choose draft depths, so drafted == serial holds on it
(tested).

## 0150 — liveness, metrics and reasoning effort (ideas from the MiaAI-Lab kit, `health.py`)

See `docs/MIA-AUDIT.md` for the audit these come from.

**Liveness.** Upstream's `/health` answers `{"ok": true}` whatever happens. With two ranks, a CUDA error or an
out-of-memory in the middle of a request leaves rank 1 in a collective forever, and the next request hangs behind it
while `/health` still says ok (the vLLM kits saw the same: vLLM's `/health` stays 200 through a stuck NCCL collective
or a UVM livelock). `tensorfold/cuda/health.py` wraps the engine's `generate` (`Health.track`: every other attribute
reads and writes through, the signature is kept for `App.check`'s `draft` probe) and keeps:

- `fatal`: the first exception out of `generate` that is not a `ValueError` (a bad request);
- the requests in flight: prompt length, start, last token time; a request is `stalled` when its last token (or its
  start, before the first token) is older than `GLM53_TF_STALL_S` (0: never) plus, before the first token, its prompt
  at `GLM53_TF_STALL_PREFILL_TPS` tokens a second (200: a 1M prompt gets ~83 min);
- counters: requests, errors, refusals, prompt / cached / completion tokens, decode rounds and seconds, prefill
  seconds.

`GET /health` returns them (`ok`, `mode`, `uptime_s`, `inflight`, `oldest_s`, `idle_s`, `requests`, `errors`, `fatal`,
`stalled`). `GLM53_TF_HEALTH=basic` (default) always answers 200; `strict` answers 503 when `fatal` is set or a request
is stalled, and a new completion gets 503 at once after a fatal error. `GET /metrics` is Prometheus text
(`tensorfold_*_total` counters and gauges; (completion tokens - requests) / rounds over an interval is the drafter's
health). An exception in a request now answers a JSON 500 (400 for a `ValueError`) or, streaming, an SSE
`{"error": ...}` event and `[DONE]`, instead of a dropped connection. `scripts/serve.sh watch` restarts both ranks on
`strict`'s 503 and alerts on a low tokens-a-round rate.

**Reasoning effort.** GLM-5.3's template renders `Reasoning Effort: Low|High|Max` (Max unless
`chat_template_kwargs.reasoning_effort` says `low` or `high`). The MiaAI-Lab kit measured structured output (a long
numeric table): thinking off garbles 5-6 of 6 replies; thinking on at low effort garbles none (791/792 numbers
right). `GLM53_TF_EFFORT_FIELD=1` maps OpenAI's top-level `reasoning_effort` (`none`/`minimal`: thinking off; `low`:
Low; `medium`/`high`: High; `max`/`xhigh`: Max) onto `chat_template_kwargs`, only for keys the request did not set;
`GLM53_TF_DEFAULT_EFFORT` is the effort of a thinking request that names none. An unknown effort is a 400. Both run in
`GlmApp.check` (so the context check counts the rendered prompt) and `run` (idempotent).

**Exactness.** Nothing on the engine side changes: the wrapper only observes `generate`'s calls and callbacks. With
the effort knobs on, requests they rewrite render a different prompt (the point); off, nothing changes.

Tests (host only, no GPU): `tests/test_health.py` (modes, fatal vs `ValueError`, stall allowance before and after the
first token, counters without float rounding, the real handler: `/health`, `/metrics`, 400 / 500 / 503, the SSE error
event, the wrapper's transparency), `tests/test_effort.py` (mapping, request keys win, idempotent, default effort
only for thinking requests, env validation, the rendered effort line).

## 0170 — Mia's prefill wins, ported (`fat` expert kernels, KDA projection in bf16)

Written offline (no GPU): the host tests ran and the CUDA compiled for sm_120 (clang + ptxas, register counts
below); the GPU tests are for the Sparks.

**Sources and licenses.** MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks has been AGPL-3.0 since 2026-09-07; before
that date it was MIT. Its Apache-2.0 fork Reederey87/glm53-flash-exl3-2x-dgx-spark (`6c337d2`) carries the E2/E3
grouped fat-expert kernels (`overlay/exl3_fat_moe.cu`, `exl3_fat_gemm.cu`, ticket scheduler, `cp.async` pipeline).
Those files come from Mia's MIT-era kit, with the fork's own changes under Apache-2.0. The `fat` kernels take their
structure from that fork and keep our arithmetic (NOTICE). PR #233 (the KDA projection's bf16 copy) was merged
into the AGPL repository, so only the idea is used here, re-implemented without its code.

**(1) `GLM53_TF_FAST_EXPERTS=fat`** (default stays `fast2`; per request `"tf_knobs": {"fat_experts": 0|1}`, header
block after 0093's `prefill_overlap`, 10 knobs). `exl3_fast.cu` namespace `fat`: fast2's arithmetic with Mia's data
movement.

- **Trellis words through the `cp.async` ring.** fast2 loaded each warp's words with `__ldg` one stage ahead into
  registers. Now NSA - 1 stages of weights and member rows are in flight (Mia S2b: +39-41% kernel throughput for
  them), and no registers hold words.
- **One rotated input for gate and up** (Mia's "gate/up Hadamard reuse"). The checkpoint's gate and up input sign
  vectors are identical: checked on the head node, 18/18 sampled experts of `07135ec0` have equal `suh` bytes, and
  `exl3_mm.shared_suh` checks every layer on the device at first use. `rot_in1` writes Xg only (0.6 GB of rot_in's
  1.2 GB at 8192 rows), and a stage holds one matrix of rows: half the gather traffic and shared memory.
  `GLM53_TF_FAT_SHARED_X=0` turns it off.
- **Swizzled row stages, 2 CTAs an SM.** The row stages are XOR-swizzled (Mia's `fm_swz`) instead of padded, so a
  gate/up stage is 16 KB and 3 stages are 48 KB. Two CTAs of 8 warps then fit an SM, with
  `__launch_bounds__(256, 2)`. clang/ptxas sm_120: 128 registers, no spills (fast2's gate/up: 224 registers, one
  CTA of 8 warps an SM).
- **Ticket scheduling** (Mia S2a). Each CTA claims items with one `atomicAdd` instead of a static stride.
  `GLM53_TF_FAT_TICKET=0` restores the stride. The expert of an item is walked forward from the CTA's last one, so
  there is no 4 KB offsets table in shared memory.
- `GLM53_TF_FAT_STAGES=3|4` (default 3; 4 stages leave one CTA an SM).

**Not taken:** Mia's orientation (rows as the mma A operand) and its epilogue. The sorted fat-row buffer is not
needed: we gather member rows by index into the ring.

**Exactness.** fat == fast2 == v1 bit for bit by construction. Every output element is the same chain of m16n8k16
mma: the same decoded A fragment, the same B values, ascending k tiles. It also gets the same transforms and
epilogue code. Only the data movement changed, and stages, ticket and CTA count cannot change a bit. With the
shared input, this also needs `rot_in1` == `rot_in`'s first output (the same code; tested).
Row-independent (0085: the member count only decides which B columns are zero-filled) and deterministic (the ticket
only decides which CTA runs an item). Because the bits are the same, the per-request knob shares snapshots with
fast2.

**(2) `GLM53_TF_KDA_PROJ_BF16=1`** (default 0; load-time, in `fastpf.settings()`, so both ranks must agree). Each
KDA layer keeps a bf16 copy of its fused input projection [q|k|v|f_a|g_a|b], 12576 x 4096 a rank. The copy holds the
4-bit weights' own values (s * q + b in fp32, rounded once), not the checkpoint's original bf16. Fast chunks multiply
it with `fast_qmm`'s one-accumulator bf16 kernel (`_fb16`, tile by shape; `GLM53_TF_KDA_BF16_TILE=bm,bn,warps,stages`
to tune, same bits). Decode, verify, exact prefill, the MTP head and FP8 prefill keep the 4-bit weights.

- **Memory:** 98.25 MiB a KDA layer and rank. With GLM-5.3-Flash's 34 KDA layers (`config.json`: 34 `linear_attention` of 45) that is +3.26 GiB
  a rank, made at load after the prepared-folder read (not stored in it).
- **Row independence.** Mia switches at M > 512 (FP8-Marlin below). We do not: a row-count switch would give a
  prompt's short last chunk or a lean tail other bits than the same rows in a long chunk, which breaks 0085. Every
  fast-chunk call uses the copy, and there is no small-M penalty, because our 4-bit path is a Triton kernel and not
  Marlin.
- **Exactness.** New arithmetic: the weights are rounded to bf16 (relative 2^-9), so fast replies differ from `0`
  in the last bits. Deterministic and row-independent: drafted == serial and resumed == fresh hold within the
  setting. It is not a per-request knob, because snapshots are not tagged by it.

**Expected** (arithmetic, nothing timed):

- **fat.** fast2's gate/up is register-limited: 8 warps an SM, 25% of DRAM bandwidth, barrier stalls. Doubling the
  resident warps and moving the weight loads onto the async ring should give gate/up +15-35% and down +0-15%.
  rot_in1 halves the input rotation (-3 ms a layer at 8192 rows). Experts are ~37% of a 32k prefill, so the
  estimate is **-6 to -12% prefill time** (1,060 -> ~1,130-1,200 tok/s at 32k).
- **KDA bf16.** Mia's -11-12% came from FP8-Marlin being 3.1-3.7x slower than bf16 at large M. Our 4-bit path already
  runs at ~58 TF/s (1.82 ms at 1024 rows), so a bf16 kernel at 70-85 TF/s saves ~0.3-0.55 ms a 1024-row sub-block
  and KDA layer: **~1-2% prefill time** for 3.3 GiB. Keep it only if `test_kda_copy_timing` shows >= 1.3x.

## 0180 — sessions in batch mode (`GLM53_TF_BATCH_SESSIONS=1`)

Written offline (no GPU); the host tests ran (fake model through the real batcher and store), the GPU tests are for
the Sparks.

**Problem.** 0120 built the batcher before 0110's store and left the store off in batch mode, so production had to
choose between 4 concurrent requests and resuming cached sessions.

**Change** (opt-in: `GLM53_TF_BATCH_SESSIONS=1`, both ranks checked equal; default 0 keeps 0120's behaviour).

- `sessions.SessionStore.bind_slots(states)`: one store for every slot. The index, budget, page slabs and LRU are
  shared; each slot has its own live-page map (`on(slot)` switches the tensors and the map every save / restore /
  `begin` acts on). Slots of another cache layout are refused at load.
- *Admission* (rank 0, `Batcher._plan` → `_session_plan`): `SessionStore.plan` (the longest stored strict prefix of
  the request's prefill mode whose draft caches fit, and the marks). When it resumes more than the free slots' own
  snapshots, the request goes to the free slot already holding most of the entry's pages (`batchplan.place_entry`,
  then least recently admitted), and `_admit` copies the entry into that slot (restore = copy-in) and resumes the
  first piece from the entry's snapshot; the slot's own snapshots are dropped (its caches changed).
- *Pieces* (`_piece`): the store's marks are the prefill's checkpoints in every piece; mark snapshots (and a piece
  end that is a mark) are saved right after the piece, the prompt snapshot (exact: the prompt end; fast: its last
  grid point) after the last piece; the reply snapshot when a drafted exact request ends (`_finish`). The same
  snapshots the lone engine saves.
- *Fast pieces* (`batchplan.piece_end` / `piece_grid`): a stored fast snapshot or mark sits on the 64-token grid,
  possibly between two bounds of the request's chunk grid C (e.g. `tf_knobs.prefill_rows=1024`); its first piece
  runs to the next bound, the rest stay on the grid. Pieces end on a multiple of C and of the snapshot grid.
- *Eviction vs live slots*: a restore copies pages into the slot's own caches, and a snapshot's KDA state / conv /
  pending rows / drafter window are copies, so an eviction (entry or slab) never touches a running slot; the slot's
  live map only avoids re-copying pages whose key (hence bits) it already holds.
- *Two ranks*: each admission's session plan (entry id and length, marks, rank 0's store digest) is shared right
  after its prompt inside the round plan's messages; rank 1 checks the digest and restores the same entry. Every
  save's decision (stored / skipped / duplicate, evictions) is shared where the save happens, inside the round both
  ranks run in the same order (0110's mechanism); rank 1 applies it and fails loudly on a divergence.
- *Memory*: at load the batcher adds slots only while `GLM53_TF_BATCH_RESERVE_GB` AND the store's budget stay free
  (it prints when the budget costs a slot); at admission a request counts free memory less the store's unused budget
  (`batchplan.admit_free`); the store keeps at least `GLM53_TF_BATCH_ADMIT_GB` free when it grows.
- Stats: `cached` (as before), `restored` (entry id) and `sessions` per request.

**Exactness.** Nothing new: a restored slot holds the bits the lone engine's restore gives (the same copy, into
caches of the same layout), pieces are `decode.prefill` resumed from snapshots (resumed == fresh), and the saves are
the lone engine's. So batched == alone == serial, with or without the store.

**Deferred.** Restores copy (no paged kernels, as 0110); stored sessions are not preferred over a slot's own
snapshot of the same length (no copy either way); spill to NVMe; marks for pieces' own boundaries beyond the store's
marks (a piece of an exact prompt is not a store checkpoint unless it is a mark).

**Fixes (2026-09-28)** (the 4 GPU failures of `results/T4-worker/test_batch_sessions_patches.log`; written offline,
the CPU fake-model tests reproduce both and pass now, the GPU tests are still for the Sparks).

- *No cache reuse* (`test_concurrent_interleaved_sessions_equal_alone[4-*]`: the fork request got `cached` 0, not
  512). Rank 0 plans an admission's marks when it admits it, and `SessionIndex.marks` placed a fork mark only at the
  common prefix with a *stored* entry (`sessions.py` `d = self.lcp(prompt, blocks)`). With as many slots as sessions,
  all 4 sessions over the shared system prompt are admitted in one round, before anything is stored: no fork mark.
  Their later turns resume from their own reply snapshots, so the fork point stays behind `resume + fork_min` and is
  never marked. On 3 slots the 4th session waited for a slot, saw the stored prompts and took the mark, which is why
  only `[4-*]` failed. Fix: the prompts in flight (running slots and earlier admissions of the same round, not
  cancelled) count as fork partners too (`Batcher._plan` passes them to `_session_plan` → `SessionStore.plan(others=)`
  → `SessionIndex.marks(others=)`, `sessions.common_prefix`). The marks still travel in the admission's session plan,
  so rank 1 is unchanged. Partners admitted together may each take the same mark; the second save is a duplicate,
  and only its snapshot copy is wasted.
- *Follower replay* (`test_follower_replays_rank0_sessions[*]`: `assert [1, 0] is None`). `Batcher._save` went
  through `GlmEngine._store_save`, which picks the side of the save decision by `self.rank == 0`. Every other message
  of a batch round picks its side by role (`_plan` sends, `follow` receives). The test's second engine (one GPU
  playing both ranks) is rank 0, so its `follow` *sent* save decisions instead of replaying rank 0's. On real rank 1
  both rules agree, but the save was the only rank-keyed message in the round protocol. Fix: `Batcher.following`
  (set by `follow`); `_save` sends (`_share(save_all(...))`) or applies (`save_all(forced=_share(None))`) by that. The
  lone engine's `_store_save` is unchanged.
- Tests: `test_fake_follower_role_is_follow_not_rank` (a rank-0 follower batcher on the fake model) and
  `test_fake_concurrent_forks_leave_a_fork_mark[3|4-exact|fast]` (4 sessions submitted at once; on 3 slots the 4th
  now resumes at the mark too) fail before the fix and pass after it. The GPU tests were right and are unchanged.
- *Memory (unchanged, for the record)*: the store's whole `GLM53_TF_SESSION_GIB` is set aside at load (a slot is
  added only while free − per-slot − budget ≥ `GLM53_TF_BATCH_RESERVE_GB`) and at admission
  (`admit_free` = free − (budget − used) ≥ `GLM53_TF_BATCH_ADMIT_GB`). The store also keeps
  max(`GLM53_TF_SESSION_RESERVE_GIB`, `GLM53_TF_BATCH_ADMIT_GB`) free when it grows. Not in the budget: each slot's
  own ≤ 2 snapshots (0120's, which can outlive an evicted entry) and transient duplicate / skipped snapshots; the
  admission minimum covers them.

## 0190 — prefill glue (`pfglue.py`; `glue.py`, `latent.py`, `pfoverlap.py`, `lean.py`, `decode.py`)

Written offline (no GPU): host tests and Triton CPU-interpreter checks ran; the GPU tests and `tests/cuda/bench_glue.py`
are for the Sparks. Nothing here is timed.

**Problem.** The 28,045-token cold prompt (fast + lean 8192-row chunks, fast2, overlap, latent KV) takes 23.09 s
(1,154-1,209 tok/s; vLLM 1,448). Past the routed experts (4.7 s) the time is glue: `hc` 2.9 s, `moe.router` 1.3 s,
`moe.shared` + `moe.combine` 1.8 s, the MTP head ~1.3 s, sparse attention 3.0 s, `dsa.o_proj` 1.9 s.

**Change.** Five switches, each its own knob, default off. 1-4 travel per request in 0090's header (4 knobs appended
after 0170's `fat_experts`; both ranks must run the same patch set).

| knob | what | bits |
| --- | --- | --- |
| `moe_glue` bit 1 | windows of 64+ rows group their (row, slot) picks by a stable sort + scatter (`glue.group_sorted`) instead of `_group`, one program walking R x 9 picks twice and R member columns twice (O(R) serial: ~1 us a row a MoE layer, most of `moe.router`) | the same integers (ids, count, every member cell), element for element |
| `moe_glue` bit 2 | the router's 8 K-slice partials summed in registers (`_router_fused`), no [8, R, 288] fp32 buffer (150 MB a layer at 8192 rows) | each slice the same dot chain, slices added in order: same bits (GPU test) |
| `moe_glue` bit 4 | lean chunks' combine reads the shared expert's fp32 rows where its matmul wrote them (`_combine_s`), no copy into `ey` | the same fp32 adds in the same order |
| `mtp_window` = N | a prefill of n tokens runs the MTP head only from lo = (n - N) rounded down to 64; below lo its caches (latent / K,V, index keys and gates, the pools they complete) are zeroed and `mtp_len` advances | main model untouched; drafts may change, replies never (verify + keyed sampler decide every token); deterministic (zeros whatever ran before); lo depends on n only, not on C |
| `hc_fused` bit 1 | in 0084's pipelined slabs, hc_post and the next hc_pre's 24 mixing dots + square sums in one kernel (`_hc_post_part`): the new bf16 tiles go from registers into the dots instead of being re-read, one launch fewer | `_hc_post`'s expressions operand for operand, `_hc_partial_mm`'s tiles / dot chain / square sums; the GPU test decides (a Triton reduction's order follows the compiler's layout) |
| `hc_fused` bit 2 | `_hc_finish_u`: hc_pre's finish with its 16 partial loads unrolled (all in flight; the finish was 16 dependent L2 round trips a row) | same adds in the same order; fast chunks only (decode keeps `_hc_finish`) |
| `attn_bm32` | bf16 fast chunks use the 32-query latent attention tile (FP8 chunks already do; 0060: -24% a layer at 1024 sparse rows, same bits as 16 on the real shapes) | same bits (GPU test); prefill need not match decode anyway (fastpf), only row independence, and a tile is one row's heads |
| `GLM53_TF_LATENT_TC=1` (load-time) | bf16 fast chunks run 0083's tensor-core `absorb_tc` / `expand_tc` instead of the fp32 FMA-pipe absorb / expand (most of `dsa.o_proj`, part of `dsa.proj`) | NEW arithmetic (q, u and kv_b's tile rounded to bf16 at the dot; bf16-matmul precision, not e4m3): fast replies change; deterministic, row-independent; snapshots tagged G + 2 (`pfgrid.TC`); refused per request |

**Exactness.** Items with "same bits" leave every committed state and reply unchanged, so snapshots and the session store
are shared across the switch. The ones that depend on Triton's lowering (router fusion, hc fusion, 32-query tiles) are
checked bit for bit by `tests/cuda/test_glue_patches.py` and `bench_glue.py` prints "bitwise True/False" per kernel: a
knob whose line says False must stay off (its per-request switch would otherwise mix snapshots of two arithmetics).
The MTP window changes only the MTP head's rows: replies are the full head's (tested with MTP, DFlash2 and auto drafts,
greedy and sampled, fresh and resumed); a resumed prefill keeps the head rows the earlier request wrote, so its drafts
(not replies) can differ from a fresh one's, as 0065 allows for the DFlash2 ring. Batch mode prefills in pieces (each a
prefix), so each piece keeps its own last N positions: a smaller saving, no correctness issue.

**Expected** (arithmetic from the measured profile; ranges; not timed):

| item | 28k (23.09 s) | 112k (95.8 s) |
| --- | ---: | ---: |
| `moe_glue` = 7 (grouping ~-1.0 s, router partials ~-0.1 s, shared copy ~-0.2 s) | -1.0 to -1.3 s | -4.0 to -5.4 s |
| `mtp_window` = 4096 (MTP head ~85% / 96% skipped) | -1.0 to -1.1 s | -4.5 to -4.8 s |
| `hc_fused` = 3 (the hc lap is ~70 us a token without gather waits, ~25-30 of it hc_post's DRAM floor) | -0.2 to -0.6 s | -0.8 to -2.4 s |
| `attn_bm32` (sparse -24%, dense -20%) | -0.6 to -0.8 s | -2.5 to -3.2 s |
| same-bits + drafts-only total | -2.8 to -3.8 s: ~1,380-1,450 tok/s | -11.8 to -15.8 s: ~1,330-1,400 tok/s |
| opt-in `GLM53_TF_LATENT_TC` (`dsa.o_proj` 1.9 -> ~0.8 s, `dsa.proj` -0.3 s) | -1.1 to -1.4 s more: ~1,460-1,570 tok/s | -4.5 to -5.5 s more: ~1,410-1,500 tok/s |

Watch the MTP window's effect on decode (tokens a round with MTP / auto drafts on long prompts): the head's own attention
no longer sees the prompt's early part. Remaining headroom, not attacked: `kda.proj` runs at ~52 TFLOP/s (12576 x 4096
at 8192 rows; ~110 measured mma.sync peak): a better-tiled kernel could save ~0.5 s at 28k; `kda.chain`'s state kernel
is bound by reading W / U / Q / K in fp32 (bf16 operands were rejected for accuracy in 0081); `moe.combine` already
reads `ey` at ~215 GB/s (only a bf16 `ey` or a combine fused into the down epilogue would cut it: new bits / a fixed-
order cross-expert reduction); the hc lap's gather waits (~30 us a token at 28k) are exposed NCCL time, not hc.

**GPU round 1 (T7, image z) and fixes.** `moe_glue` bitwise everywhere; `moe_glue=5` (grouping + shared in place) is
+5-6% prefill and in production. The one-kernel router (bit 2) is bitwise but slower at 8192 rows (2.2 vs 1.3 ms: 576
long programs against 4,608 short ones), so leave bit 2 off. `hc_fused` bit 1 could not launch: with `num_stages=3`
Triton 3.7.1 staged the 4 stream tiles, both partial tiles and 4 fn tiles a step in shared memory (128 KB > the 99 KB
a block on sm_121; reproduced offline by compiling for sm_121 with 16-byte-aligned pointers, 8 KB with
`num_stages=1`, now the default `glue.HC_FUSED_STAGES`); a kernel that still cannot launch now falls back to the
separate kernels (same bits) and says so once. The offline TTGIR shows the fused kernel's square sums reduce in the
same blocked layout ([1, 8] x [4, 8] x [4, 1]) as `_hc_partial_mm`'s. `hc_fused` bit 2 (`_hc_finish_u`) passed
bitwise. `attn_bm32`: the kernel test passed bit for bit; its engine test failed only because it also turned on the
fused hc (now a separate step). `mtp_window`: replies equal with and without the window (passed); two test bugs
failed it: the follow-up needed a drafted snapshot (the cold serial run leaves snapshots without MTP rows, so nothing
resumed), and on the latent engine the "main state" compared included the MTP layer's own indexer caches, which the
window zeroes by design. `GLM53_TF_LATENT_TC` stays off (it changes replies). `bench_glue.py` now reports a failing section
and keeps going.

**On the Sparks.** `python tests/cuda/bench_glue.py` first (a minute: old vs new kernels at 1024 / 4096 / 8192 rows,
with a bitwise check each), then `pytest -q -s tests/cuda/test_glue_patches.py`, then per-request A/B on one load:
`"tf_knobs": {"moe_glue": 7}`, `{"mtp_window": 4096}`, `{"hc_fused": 3}` (with `prefill_overlap` on), `{"attn_bm32": 1}`,
each against 0, with `GLM53_TF_PROFILE=1`. `GLM53_TF_LATENT_TC=1` needs a restart and the `exact` suite / MMLU.

## 0200 — batched parallelism (`batch.py`, `batchplan.py`)

**Measured (0120, `GLM53_TF_BATCH=4`, `results/B2`) and what it means.** 4 streams: 61-74 tok/s aggregate, per
stream 27-64; 2 streams 56-60. Per stream the batched rate is 0.6-0.7x the same prompt's lone rate (lone, `B1`:
46 / 47 / 66 / 98 tok/s for the four bench prompts; batched 27-33 / 34 / 42 / 62-64): the spread is the prompts'
own draft acceptance (the 98 tok/s prompt commits ~2x the tokens a round), not the scheduler; every active slot
already gets one window every round. The aggregate is below the sum of the per-stream rates because the streams
barely overlap: the four prompts' first tokens are staggered (one piece a round, then a fair-share wait between
pieces) and the fastest stream ends early. `results/B1` is the unbatched server: its stall run's 105.8 s TTFT is
three 1,024-token replies served first (~75 s) plus the 31 s prefill; batched (`B2`, one decoder) the same prompt
took 57.7 s (the 0.5 share), longest decode gap 3.2 s (a fast piece rounds up to the chunk grid).

Model of a batched round (two Sparks: verify 31/39/45/51/56/62/68/74 ms at 1-8 rows, past 8 ~6-7 ms a row; MTP
draft 2.04 ms + 1.68 a chained draft per slot; sampling + commit ~0.5 ms a slot; parity copy 0.6 ms a slot): 4 x
~3.5 rows = 14 rows: verify ~113 ms, MTP drafting 4 x 5.4 = 21.6 ms, parity copies 2.4 ms, host ~2 ms: ~139 ms
for ~8.8 tokens (63 tok/s; 70-80 with one high-acceptance stream), when every round replays a graph. A round that
meets a new (slots, rows, modes) key runs eagerly (+10-25 ms) and then captures (+capture, instantiation); with
2-4 slots of 1-8 rows most keys are met once, and the 256-graph cap fills with them.

**Knobs** (each off by default = 0120's behaviour; both ranks must agree, checked at load):

- `GLM53_TF_BATCH_CAPTURE_AFTER=N` (1): a key is captured on its N-th sighting (`batchplan.Sightings`); before that
  its rounds run eagerly through the same capturable code (same bits). Keys met once no longer pay a capture and
  the graph cap is kept for keys that recur. Suggested 3.
- `GLM53_TF_BATCH_PARITY_KEY=1`: each slot's KDA parity goes into the key instead of `_parity0`'s copy of a state
  left in buffer 1 (71 MB a slot, every round: every commit flips the parity). The graph bakes in `rec[cur]` /
  `rec[1 - cur]` as the engine's own (rows, parity) graphs do. Keys at most double. -0.6 ms a slot and round.
- `GLM53_TF_BATCH_MTP=1`: `MtpChains` drafts every slot's MTP chain together: one head pass absorbs every slot's
  backlog (`mtp_multi`: row-local kernels over all rows, the DSA cache write / attention per slot on its own head
  cache through `_dsa(..., caches=)`), then one pass per chained step over the slots still drafting. Each chain
  follows `decode.draft` step for step (own sampling, confidence or cost-depth stop, positions, head cache), and
  each row has the bits of the slot's own head pass, so the drafts are the ones it drafts alone. An MTP step reads
  ~270 MB a rank, ~165 MB of it the vocabulary head: 4 slots read it once instead of 4 times. The drafts of a pass
  are sampled with one all-gather (`sample_drafts`: `sample_rows`' candidates, draw and probability per row), so a
  4-slot chain step has one hard sync instead of four. Eager passes (the per-slot head graphs are kept for a single
  MTP slot). Expected -8 to -12 ms a 4-slot round (21.6 -> ~10 ms), -4 ms at 2 slots.
- `GLM53_TF_BATCH_ROW_MS=F` (0): batch-aware cost-derived depths (`BatchDepth`, 0120) extend the verify table past
  8 rows at no less than F ms a row. 0120 extends it along the table's upper-half slope (5.75 ms on the table
  above), below what rows of other sequences cost (6-10 ms): 3-4 sequences drafted too deep. Suggested 6.5.
  Changes only drafts (never bits).
- `GLM53_TF_BATCH_SHORT=N` (0): every prompt with at most N tokens left prefills in the round it is admitted,
  several in one round, outside the fair share (`batchplan.pick_pieces`); long prompts keep
  `GLM53_TF_BATCH_PREFILL_SHARE`. Four simultaneous short prompts get their first tokens in one round instead of
  one piece + one share-wait apart. Suggested 1024 (a piece of that size is ~0.4-1 s).
- `GLM53_TF_BATCH_PAD=2,4,8` (off): each slot's window padded to the next listed size with its last token (fewer
  keys). Padded rows are verified like drafts that can never be accepted (the accept rule reads only the real
  drafts) and `commit` gets the padded row count (KDA replay of the kept prefix), so every kept bit is the
  unpadded window's. A padded row costs ~6 ms: only worth it when `round_kinds` shows mostly eager rounds.
- Stats: each request's `round_kinds`: rounds by kind (`alone`, `graph`, `eager`, `capture`), `pad_rows`,
  `mtp_batched`, and the wall ms of its verify rounds (`verify_ms`), of its drafting (`draft_ms`) and of other
  requests' prefill pieces it waited through (`piece_ms`). `bench/multiturn.py --modes concurrent` prints TTFTs,
  tokens a round, ms a round (+ piece seconds) and the round kinds per stream.

**Exactness.** CAPTURE_AFTER, PARITY_KEY: which of two identical code paths (captured or not, parity baked in or
normalized) runs. PAD: row independence, and the padded rows are never kept (the same argument as rejected drafts).
MTP, ROW_MS: drafts only. SHORT: which pieces run when (resumed == fresh). Each batched reply stays byte-identical to
the same request served alone.

**Expected** (the model above; steady state, all slots decoding, graphs hit): 4 streams ~72-80 tok/s (0120's
model 63-70 steady state; +10-17%: MTP -10 ms, parity -2.4 ms, ROW_MS 6.5 ms moves 4-slot windows from ~3.5 to
~2.8 rows); a typical stream ~18-20 tok/s, a high-acceptance one ~35-40. 2 streams ~58-62 (+8%). The measured
aggregate gains more where 0120 lost rounds to captures and stagger (the concurrent bench's `round_kinds` shows
how many). The ceiling is structural: a verify row costs ~6-7 ms (its routed experts; rows of different sequences
share few of 288), so 4 x 3 rows cost ~2.3x one window.

**Prefill under load.** TTFT ~= alone / share (time slicing; `B2`: 57.7 s = 1.86 x 31 s at 0.5). share 0.7: ~44 s
with the decoders at ~30% speed during it; 0.8: ~39 s. The longest decode gap is one piece: a fast piece rounds up
to the chunk grid (auto rows up to 8192: ~3 s); `"tf_knobs": {"prefill_rows": 2048}` on the long request or a
smaller `GLM53_TF_BATCH_PIECE` with a smaller grid trades ~5-10% prefill speed for ~1 s gaps.

**Not done, and why.**

- *Mixed rounds (prefill piece + decode rows in one forward, vLLM style).* The gain in vLLM comes from decode rows
  riding on the prefill's weight reads. Here a decode row's cost is its routed experts (~6 ms of expert reads a
  row), and the fast prefill kernels (fast2 / fat experts, bf16 gathers, chunked KDA) are not the row-invariant
  decode kernels: decode rows through them would change their bits (replies != served alone). Kept exact, decode
  rows need their own grouped expert launch (their expert reads are not shared), their own fp32 gathers and their
  per-slot KDA / attention: what is shared is the dense 4-bit matmuls (fast_qmm has qmm's bits), ~10-12 ms once
  per piece of 0.4-7 s: <1-3%. Meanwhile each decoder advances one window per piece (a round as long as the piece)
  unless pieces shrink, and pieces under ~1k rows lose prefill speed (every piece reads all experts once, ~0.3 s).
  The exact alternative (prefill pieces on the row-invariant kernels, expert reads shared with the decode rows)
  prefills ~2x slower (500-670 vs ~1,100 tok/s) and differs from the request's fast prefill alone. Time slicing
  with fast pieces dominates both.
- *One launch per layer for per-slot KDA / attention / indexer* (kernels taking per-sequence row offsets and state
  pointers, graphs keyed by total rows): the KDA chain is a CUDA extension (`kda.cu`), untestable offline here.
  With CAPTURE_AFTER / PARITY_KEY the key churn it would remove is mostly handled; what remains is ~(N-1) x 45
  layers x 3-5 small nodes and the chain's 32-block grids (~2-5 ms a 4-slot round, 2-4%). Design:
  `chain_kernel` grid (heads, sequences) with a device table (row offset, rows, rec in/out, conv, scratch) over
  one `[slots, 2, layers, H, 128, 128]` state tensor; `kv_write` / attention / indexer with a per-row sequence id
  -> (cache base, pos); graphs keyed by (T, modes). 3-4 days with a GPU.

**GPU window: quick check (in this order).**

1. Tests: `PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q tests/cuda/test_batch_parallel_patches.py`
   then `test_batch2_patches.py` and `test_batch_sessions_patches.py` (knobs off: 0120 / 0180 unchanged).
2. Worth it? Same load twice (`GLM53_TF_BATCH=4`), knobs off, then on
   (`GLM53_TF_BATCH_CAPTURE_AFTER=3 GLM53_TF_BATCH_PARITY_KEY=1 GLM53_TF_BATCH_MTP=1 GLM53_TF_BATCH_ROW_MS=6.5
   GLM53_TF_BATCH_SHORT=1024`, passed through `scripts/serve.sh`'s environment):
   `python3 bench/multiturn.py --base http://127.0.0.1:8080 --model GLM-5.3-Flash-Uncensored --modes
   batchexact,concurrent --streams 1,2,4 --reps 2 --long-tokens 256 --out results/B3/<off|on>.json` (~3 min).
   Look at: `batchexact` all true; aggregate at 2 / 4 streams; per stream `ms/round` (verify + own drafting) and
   `rounds {...}` (with knobs off, a large `capture` + `eager` share confirms the key churn; on, mostly `graph`);
   `ttft` spread at 4 streams (SHORT). Worth keeping if 4 streams gain >= ~8% with batchexact true.
3. Prefill under load: `--modes stall --streams 4 --long-tokens 1024 --doc 28000` with
   `GLM53_TF_BATCH_PREFILL_SHARE=0.5` and `0.7` (TTFT vs the decoders' tok/s).
4. Optional A/B: `GLM53_TF_BATCH_PAD=4,8` (only if step 2 still shows many eager rounds).

Tests: `tests/cuda/test_batch_parallel_patches.py`.

## 0220 — FP8 latent KV cache (`GLM53_TF_KV_DTYPE=fp8`)

**Problem.** 4 concurrent threads x 256k context need 4 slots of capacity-sized caches: 13,616 B a token a rank in
bf16 (docs/MEMORY-1M.md), 13.26 GiB a node for 4 x 262,152 slots, and with the session store and prefill buffers that
left MemAvailable at 2-4 GiB under load (results/Y1). The latent rows are 90% of it (12 x 1,024 B).

**Change.** `GLM53_TF_KV_DTYPE=bf16|fp8` (`latent.kv_dtype`, load-time; fp8 needs `GLM53_TF_LATENT_KV=1`;
`GlmEngine` checks both ranks agree; `decode.Engine` sets `w.meta["kv_fp8"]` before any `State`).

- Layout: `latent.caches` allocates each latent cache (the 11 DSA layers' and the MTP head's; keys = values as before)
  as uint8 `[capacity, ROW8 = 528]`: bytes 0-511 the e4m3 values, 512-515 the fp32 scale, 516-527 padding so rows stay
  16-byte aligned (128-bit loads in the sparse gather; unpadded 516 B would save 144 B a token). Every consumer that
  copies rows generically (sessions' pages, batch slots, snapshots, pfglue's MTP-window zeroing, `State.clone`)
  moves the scale with its row; an all-zero row reads as zeros. The indexer's keys, gates and pool keys stay bf16:
  they decide the token selection.
- Write (`_lwrite8`, from `latent_write`): each row on its own. s = 2^e, the smallest power of two with amax / s <= 448
  (from amax's exponent and mantissa bits: no log2, exact); y = x / s (exact); y rounded to the e4m3 grid to nearest
  even in fp32 bit arithmetic (3 mantissa bits; below 2^-6 the fixed 2^-9 step via +- 1.5 x 2^14), sign kept (-0.0
  too), then converted (an exact relabelling, whatever the backend's cvt rounding). Equal to torch's
  `float8_e4m3fn` conversion byte for byte (`quantize_rows_reference`; checked on every finite bf16 value <= 448).
  Triton's CPU interpreter drops the carry when its own conversion rounds across a power of two (7.84 -> 4.0),
  another reason not to rely on it.
- Read (`_lrows`, in `_lchunks` and `_lsparse_chunks`, `FP8` constexpr): e4m3 -> fp32 x s -> bf16. With a
  power-of-two scale this is exact, so the tile fed to the dots is exactly the bf16 tile of the dequantized row and
  the rest of 0060's arithmetic (16- and 32-query tiles, online softmax, chunk merge) is unchanged: an FP8 cache
  gives the bits of a bf16 cache holding `dequantize_rows` of it (tested).
- Tags: `decode.Snapshot.kv` (0 / 1) set by `take_snapshot`; `restore` refuses the other format. `sessions.KV_TAG`
  (b"" for bf16, so earlier keys are unchanged; b"kv:fp8") is hashed into every entry and page key. The fastboot
  calibration key includes every `GLM53_TF_*`, so an fp8 load calibrates its own table. `knobs.LOAD_ONLY` lists
  `kv_dtype`. batch.py needs no change: its rounds call `latent._attend` per slot and size slots from the tensors.

**Exactness.** Quantization is per row and depends only on the row (the same latent gives the same bytes whatever
the window, chunk, slot or position), reads are exact, and the attention arithmetic is 0060's. So every 0060 / 0085
guarantee holds within the fp8 configuration: verify rows == serial rows (drafted == serial), chunk size changes no
bit, resumed == fresh, batched == alone. Against bf16 KV the stored latent loses precision (e4m3: relative step 2^-3
to 2^-4, rms error ~2.6% a value; half a step at most), so replies differ: a new configuration with a quality gate.

**Memory.** 12 x 528 + 512 (MTP index keys + gates) + 768 (pool keys) + 48 (0050 scratch) = **7,664 B a token a rank**
(bf16: 13,616; 0.56x). At CONTEXT=262144: 1.86 GiB a slot instead of 3.31; 4 slots 7.45 GiB instead of 13.26
(docs/MEMORY-4x256k.md). Decode past 2,051 tokens reads half the latent bytes per selected token.

**Tests.** `tests/cuda/test_fp8_kv_patches.py` (GPU) and `tests/test_fp8kv_interpreter.py` (host, Triton's CPU
interpreter; run it in its own pytest process: another module importing Triton first disables the interpreter).

## Tests

| File | What it checks |
| --- | --- |
| `tests/cuda/test_patches.py` | on TensorFold's synthetic GLM checkpoint (one GPU playing rank 0 of two): non-expert weights really are 4-bit; q4 drafted == serial (sampled and greedy); 64- vs 512-row prefill give the same reply and the same resume; the MSE clip is never worse than min/max; vectorized sparse selection equals the per-row loop at the dense/sparse boundary; a 3,000-token prompt (past the 2,051 dense limit) gives the same reply with 64- and 512-row chunks and with drafting |
| `tests/cuda/test_longctx_patches.py` | 0050: bounded and device selection == unbounded (random, tied, signed-zero scores); the device selection replays in a graph across its bucket; long-context graph steps (rows 1-8, both parities, MTP 1-8) == upstream eager logits at 2.5k-7k tokens (contexts 4,096 / 8,192); replies with graphs == `GLM53_TF_LONGCTX_GRAPHS=0`, serial and drafted (8 policies), greedy and sampled, including a prompt crossing 2,051 during decode; resume after a long reply == fresh prefill |
| `tests/cuda/test_latent_patches.py` | 0060: latent vs expanded kernels and an fp32 reference on the real shapes (32 heads, 256, latent 512; 4-bit and BF16 kv_b); window rows == serial rows bit for bit (dense, sparse, graph-style chunk counts); scratch independent of capacity; with `GLM53_TF_LATENT_KV=1`: latent caches and bytes a token, drafted == serial (11 policies), resumed == fresh, 64 vs 512-row prefill same state, BF16 kv_b (EXL3), a 3,000-token prompt past the dense limit (chunks, drafting, resume), batched == alone; latent vs expanded prefill within bf16 tolerance |
| `tests/cuda/test_1m_patches.py` | 0065: row-block scores == `_scores` bits; blocked selection == sorted (random, ties, -0.0, NaN, dense-limit rows, bucket boundary, up to 1,024 rows, 1M pool counts, small blocks); bounded selection scratch; index ring pool keys == full; DFlash2 ring == linear (wrapped, rewound); replies with 0065 == all-off (serial, drafted, 64/512 rows); resume behind both rings == fresh; memory accounting at a 1M capacity |
| `tests/cuda/test_knob_patches.py` | 0090: host-only validation (unknown / load-only / out-of-range / typed knobs, batch mode), header block round trip, the server's 400 message; on the synthetic EXL3 checkpoint: 14 knob sets == serial (greedy, sampled, drafted and serial, fresh and resumed across chunk sizes), knobs revert after each request (also a failing one) and are echoed, invalid ones fail before rank 1 hears, lookup/depth/auto_fdrafts reach the header and bound the windows, a replaying follower runs rank 0's knobs and windows with other defaults of its own, `longctx_graphs` 0 vs 1 per request past 2,051 tokens |
| `tests/cuda/test_batch2_patches.py` | 0120 (replaces 0030's `test_batch_patches.py`). Host only: request headers and round plans round trip; piece bounds (exact, fast grid); the prefill time share; piece order, admission order, which background request steps aside; the title heuristic; the round cost model; knobs in batch mode. GPU (synthetic EXL3 checkpoint): per-slot states / DFlash2 contexts / graphs; batched rows == lone rows (logits, MTP rows, taps; 2-4 sequences; graphs first round and replay, eager); 2 and 4 requests == serial for every policy (MTP, DFlash2, auto, o / om / of, thresholds, serial, lookup), sampled and greedy, graphs and eager; uneven lengths and queued requests; per-slot prefix reuse; a 700-token prompt in pieces while another decodes; fast and lean prefill admissions == the lone fast / lean engine (grid snapshots only, resume, exact requests never resume fast snapshots); per-request knobs per sequence == lone with the same knobs; latent and expanded KV past 2,051 tokens (pieces across the limit, a crossing window, bucket-keyed graphs, resume); a client gone mid-reply; a background request stepping aside; concurrent streaming callers; slots trimmed to memory; the MLX checkpoint; a follower replaying rank 0's plans makes the same decisions |
| `tests/cuda/test_fastk_patches.py` | 0081: fast_qmm vs qmm on the per-rank GLM shapes, M = 256/1024/1500 (close, deterministic, bitwise), BF16, strided rows, timings; fast_kda vs `kda.chain` (32 heads, aligned/unaligned pos, tf32 tolerance), the fp32 variant vs the fp32 reference, determinism, 64-aligned split == one call, `save_replay` feeds `kda.replay`, timing |
| `tests/test_fastk_interpreter.py` | 0081 in Triton's CPU interpreter (no GPU): the same checks on small shapes, tf32 operand rounding emulated |
| `tests/cuda/test_fp8_kv_patches.py` | 0220 (GPU): the FP8 writer == torch's e4m3 conversion (every finite bf16 value <= 448, random rows over 11 decades, zeros / -0.0), rows independent, error <= half an e4m3 step; FP8 dense / sparse latent attention (BM 16 / 32) == the bf16 kernels on the dequantized rows bit for bit; absorb -> attend -> expand vs the fp32 expanded reference (prints bf16 vs fp8 error); window rows == serial rows; the knob's validation. Engine with fp8 KV: layout and bytes a token, drafted == serial (11 policies), resumed == fresh, 64 / 512-row chunks same state, fp8 vs bf16 KV prefill within tolerance (quality hook), snapshots never cross formats, 3,000 tokens past the dense limit (chunks, drafting, 0050 graphs, resume), batched == alone (2 slots; 4 slots with 0200's on-set), sessions resume FP8 rows (keys carry the format), 0180 batch sessions == alone |
| `tests/test_fp8kv_interpreter.py` | 0220 in Triton's CPU interpreter: writer == torch conversion (exhaustive bf16), rows independent, fp8 attention == bf16 on dequantized rows (dense, sparse, BM 16 / 32); the knob, session key tag, snapshot format check |
| `tests/cuda/test_fastpf_patches.py` | 0080/0091. Host only: the grid, the switches, the knob, and the rule through the engine's real prefill / snapshot / resume code on a fake model whose chunks depend on their start and length (random conversations, serial and MTP-drafted, fast and exact). GPU: the fused expert kernels vs a float64 reference and the row-invariant path (synthetic and real shapes, multi-pass experts), deterministic and row-independent; fast engine: deterministic state, drafted == serial (7 policies), resumed == fresh (within the last chunk, across grid points, after a reply, chains, on the grid, short prompts), only grid snapshots kept, fast and exact snapshots never mix, off-grid resumes refused; the same with adversarial chunk-dependent stand-ins for 0081 (and the control that they do depend on the chunks); fast vs exact prefill within bf16 tolerance; a 3,000-token prompt on the latent cache past the dense limit |
| `tests/cuda/test_lean_patches.py` | 0082. Host only: the lean orchestration on a hash model (every kernel exact integer arithmetic with the real row / position / KDA-state / cache dependencies) equals 0080's fast chunk bit for bit (lengths 1-256 around 64/128-row sub-blocks, two positions, commit included, EXL3 and MLX MoE), the routed experts run once a layer, a planted carry bug is caught; a last sub-block of 1 / 17 / 63 rows runs the fast matmuls like the whole chunk (with fast_qmm's < 64-row qmm fallback emulated; control without the fix); the fast rule with lean chunks that depend on their sub-blocks and a grid 4x the buffers (fake model, real prefill / snapshot / resume code); the lean set's bytes on the real shapes (linear, ~397 KiB a row). GPU: lean (64-row buffers, 256-row chunks) vs non-lean fast engine: same committed state (3-700 tokens), same replies (7 policies, greedy and sampled), drafted == serial, resumed == fresh (chains too), deterministic, knobs (prefill_rows up to the lean max, exact requests clamped), memory; 3,000 tokens on the latent cache past the dense limit |
| `tests/cuda/test_fp8_patches.py` | 0083/0092. Host only: the switch, the tag, the knob; the fast rule with FP8 and bf16 fast requests mixed in random conversations on a fake model whose fast chunks depend on the mode (real prefill / snapshot / resume code). Kernels (GPU on the real shapes at M = 1024 / 2048 / 8192; Triton's interpreter on small shapes without a GPU): `matmul_fp8` == its host model to fp32 summation order, vs fp32 within the e4m3 bound, deterministic, row-independent (slices, permutations, the first 1024 rows alone), strided rows, zero rows; "cvt" == "bits" weight encoding and every tile == the same bits; small-shape / short-call fallbacks; BF16 weights; tensor-core absorb / expand vs a float64 reference; 32- vs 16-query latent attention. GPU engine (every matmul in fp8): runs and differs from bf16 fast (control), deterministic, drafted == serial (7 policies), resumed == fresh (chains too), FP8 / bf16 / exact snapshots never mix, lean == non-lean within FP8, 3,000 tokens on the latent cache; quality bound vs bf16 fast; timing prints (fp8 vs bf16 per shape and M, tile sweep, absorb / expand) |
| `tests/cuda/test_overlap_patches.py` | 0084/0093. Host only: on 0082's hash model, every pipeline variant (direct, gather, slab, all; slabs of 64 and whole) equals the unpipelined lean chunk bit for bit (lengths 1-256, two positions, blocks 64/128, EXL3 and MLX MoE, fp32 gathers, one rank). A deferred exchange runs only when waited for. Planted bugs are caught: a post that skips its exchange, a missing drain, routing before the last attention post. The pipelined order is checked, and the collectives equal 0082's in size and order. Also: knob parsing, slab rows, the 0093 knob. GPU kernels: hc_post on bf16 partials (and row slices) == on the fp32 copy; bf16 combine == fp32 combine rounded; the fast matmul's bf16 store == fp32 store rounded (o_proj / down shapes, Q4 and BF16, 1-2048 rows); an hc timing print on real shapes. GPU engine (64-row buffers, 256-row chunks), variants switched on one engine: committed state == unpipelined (3-700 tokens), also with an exchange stand-in that sleeps on the comm stream; fp32 gathers; replies (5 policies, greedy and sampled); resumed == fresh; deterministic; the per-request knob; 3,000 tokens on the latent cache |
| `tests/cuda/test_cindep_patches.py` | 0085. Host only: `pfgrid` (tags, resumable, rows parsing, auto rows, plans: snapshot point, cut, tail rule, marks; with the grid at C exactly 0080's rule; 2,000 random plans); the engine's real `_run` / `_prefill` / snapshot / resume code on a fake whose fast rows depend on their whole 64-row block, the call's offset in it and the mode (not on C): 60-turn random conversations with a random C per request (auto included), FP8 / bf16 mixed, marks, serial and MTP-drafted, every reply and state == a fresh prefill with another C, follow-ups resume at the old prompt's last 64-point (or within the tail); the control (a C-dependent fake) is caught; misaligned resumes and fast KDA calls refused; `fastpf.chunk` keeps only the head on qmm. GPU kernels, bit for bit: `matmul_fast` (4-bit, BF16, bf16 / FP8 prefill, real shapes) on 1-300-row slices and permutations, head's 1-row call == qmm; `hc_partial`; `absorb_tc` / `expand_tc`; the fused experts (row subsets, permutations, skewed routing past a pass); `fast_kda` split anywhere on 64 == one call. GPU engine: committed state identical for C = 64 / 1024 / 4096 / 8192, non-lean and lean (256 sub-blocks), overlap on / off, bf16 and FP8, 3-1,000 tokens and 3,000 / 9,000 tokens on the latent cache with 32 index heads; resumed from the 64-grid snapshot with another C (64 / 1024 / auto) == fresh with a third (MTP, DFlash2, auto drafts; FP8; chains); drafted == serial with auto; the auto knob and `GLM53_TF_PREFILL_ROWS=auto` |
| `tests/cuda/test_mia_prefill_patches.py` | 0170. Host only: the knob in the header block, `kda_proj_bf16` refused per request, `fastpf.settings` / parsing / the memory estimate (98.25 MiB a layer), the copy's dispatch (fast chunks only, FP8 off), a Python model of the fat kernel's index arithmetic (swizzled stages read back what was written, conflict-free phases; the weight ring hands each warp fast2's words; the expert walk == fast2's binary search; every item claimed once). GPU kernels: fat == fast2 bit for bit (Xd, Y; shared and own inputs; 3 / 4 stages; ticket on / off; 300 / 1024 / 4160 rows, uniform and skewed); `rot_in1` == `rot_in`; row subsets and permutations; timings of fat vs fast2 and of the KDA copy vs 4-bit with a tile sweep (`-s`); the copy row-independent (1-63-row and unaligned slices, permutations) and within 5e-3 of the 4-bit path. GPU engine (gate / up sign vectors made equal): committed state fat == fast2 (lean and not, 3-1,000 tokens); with the KDA copies the state is the same for C = 64 / 1024 / 8192, fat or not; drafted == serial and resumed == fresh (fat, KDA copies); a fast2 request resumes a fat request's snapshot |
| `tests/cuda/test_session_patches.py` | 0110. Host only: page keys and counts (exact / fast / MTP), tail bytes; longest fitting prefix, common prefix, marks; shared pages with reference counts, LRU eviction within the budget, slabs reused and released, oversize entries skipped; rank 1 replaying rank 0's decisions (300 random saves) ends with the same store, divergence detected; the plan message. With torch on any device: the tensor store on fake caches whose rows are functions of their prefix (random forks, follow-ups, junk past every request, evictions): every restore gives the entry's rows and drafter window. GPU: sessions A B A C B A interleaved (shared system prompt, fork mark) == fresh prefill + serial, pages shared and not recopied; eviction under a tiny budget and an oversize entry stay exact; mixed drafter policies; 2,300-token system prompt past the dense limit on the latent cache, exact and fast; rank 1 following rank 0's messages (same restores, saves, evictions, replies; divergence refused) |
| `tests/cuda/test_batch_sessions_patches.py` | 0180. Host only: piece bounds for fast resumes between chunk-grid bounds, the piece grid, entry placement, admission memory, the switch. Torch on the CPU: the store bound to several slots (save on one slot, restore into another, per-slot live maps, eviction after a restore leaves the slot's copy, other layouts refused); the real `Batcher._plan` / `_execute` / `_admit` / `_piece` / `_finish` / `follow` with the real store on a hostile fake model, 4 sessions over a shared system prompt on 3 slots, exact and fast, roomy and tiny budgets: every reply and the slot's whole state at the end == fresh prefill + serial decode; a replaying follower ends with the same store; the 2026-09-28 fixes (a follower that says rank 0 replays save decisions by role; sessions admitted together leave a fork mark). GPU: off by default; 4 concurrent sessions on 3 / 4 slots == alone (mixed policies, sampled and greedy), fork mark, shared pages; tiny budget; fast prefill with resumes between chunk bounds; rank 1 replaying plans, session plans and save decisions |
| `tests/cuda/test_glue_patches.py` | 0190. Host only: the knobs in the header, the MTP window's start rule, `group_sorted` == a Python model of `_group` (every cell, ids past the count untouched), the switch rules, the MTP absorb with a window on a fake state (zeroed rows / index keys / pools, `mtp_len`, drafted rows dropped), the TC tag and load-only refusal. GPU kernels: `group_sorted` == `_group` (64-8192 rows, uniform / skewed / local), `select` on == off, `_router_fused` == partials + sum (and row slices), `_combine_s` == copy + `_combine`, fused hc == separate kernels bit for bit (1-1000 rows, bf16 / fp32 partials, 1 / 2 ranks, modes 1-3, slab slices), fused tiles of 16 / 32 rows reported, 32- vs 16-query latent attention bit for bit (dense, sparse). GPU engine: state with `moe_glue` on == off (non-lean, lean, 3-1000 tokens), every same-bits knob on at C = 1024 / 8192 (pipelined or not) == all off at C = 64, `hc_fused` 1-3 in the pipeline, `attn_bm32` on the latent cache past the dense limit; MTP window: main state unchanged, head rows zero below lo, history-independent, replies == window off == serial (MTP, DFlash2, auto; greedy, sampled), resumed == fresh; knobs echoed and restored; `GLM53_TF_LATENT_TC`: C-independent, differs from the FMA path (control), drafted == serial, resumed == fresh |
| `tests/cuda/test_batch_parallel_patches.py` | 0200. Host only: `pick_pieces`, `pad_mask` / `padded`, `Sightings`, plans with several pieces, the row-cost floor in `verify_ms` / `RoundCosts`. Torch on the CPU (0180's hostile fake model, the real `_plan` / `_execute` / `_piece` / `_verify` / `follow`): short prompts prefill several a round and every reply and slot state == fresh prefill + serial decode, a follower replays the same rounds; a padding forward keeps replies and states exact (offsets, commit rows); `MtpChains` on a fake head == `decode.draft` alone (drafts, head-cache length, chained entries, optimizer calls; confidence and cost-depth stops, several rounds); `sample_drafts` == `sample_rows` row by row (greedy / sampled, with / without probability, 1 and 2 ranks). GPU: padded batched rows == lone rows (eager, capture on the 2nd sighting, replay); 4 requests with every knob == serial (mixed policies, sampled / greedy), parity-keyed graphs; short prompts share their admission round; one MTP head pass over 2-4 slots == each slot's own; batched MTP drafting == serial with per-slot drafting's keeps |
| `tests/test_health.py`, `tests/test_effort.py` | 0150, host only: `/health` modes (fatal, stalls with the prefill allowance), `/metrics` counters, JSON 500 / SSE error events, 503 refusals in `strict`, the engine wrapper's transparency; `reasoning_effort` mapping and the default effort |
| `tests/test_serve_ops.py` | host only (fake docker / ssh / nvidia-smi / journalctl, a fake OpenAI server): `serve.sh` start (preflight, memory gate, canary warn / strict, retries, log rotation, NCCL passthrough, the start lock), parallel stop, `xid`, `watch` (absent, loading grace, bad ticks, alert, heal, unreachable worker, drafter rate alert); `canary.py` (degenerate replies, dead drafter, warmup); `xid.py` classes |
| `tests/test_glm_tool_calls.py` | GLM tool calls: schema-typed values, a string that looks like JSON stays text, two calls plus an unknown tool, a call without arguments, the Qwen format still parses |
| `/src/TensorFold/tests/cuda/test_glm_*.py` | upstream's own GLM CUDA tests, run against the patched tree |

Run inside the image (needs one GPU; does not load the real checkpoint):

```bash
docker run --rm --gpus all -e PYTHONDONTWRITEBYTECODE=1 -v $PWD/tests:/work/tests --entrypoint bash \
    glm53-tensorfold:dev -c "pip install -q pytest; cd /work && \
    PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda python -m pytest -q \
    tests/cuda/test_patches.py tests/test_glm_tool_calls.py /src/TensorFold/tests/cuda/test_glm_*.py"
```

Expected: 50 passed.

On the real model, `bench/glmbench.py --suites exact` checks drafted == serial end to end (10/10 byte-identical
in every TensorFold run so far).

## Rebasing on a newer TensorFold

```bash
git -C vendor/TensorFold fetch && git -C vendor/TensorFold checkout <new-rev>
for p in patches/*.patch; do git -C vendor/TensorFold apply --check "../../$p" || echo "conflict: $p"; done
```

Refresh a conflicting patch against the new tree, rebuild the image, and re-run the tests and the `exact` suite.
