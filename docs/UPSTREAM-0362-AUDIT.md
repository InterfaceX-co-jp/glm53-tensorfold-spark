# Upstream TensorFold 0.3.6.2 against our GLM engine (0.3.4 + patches 0001-0490), 2026-09-29

> Offline audit: no GPU, the Sparks were not touched, the submodule pin (`2f8e514`, 0.3.4) is unchanged. Upstream was
> read in a separate clone (`~/.cache/tf-upstream` at `71377a5`, 0.3.6.2, 70 commits ahead). Every speed number for
> an upstream change below is an **estimate** from our W11 measurements (`docs/RESULTS.md` W11, `docs/DECODE-PLAN.md`,
> `docs/DECODE-KERNELS.md`); `results/upstream-0362/estimate.py` holds the arithmetic, and
> `results/upstream-0362/rebase-probe.txt` holds the result of applying our patches to 0.3.6.2 mechanically.

## 0. Bottom line

- **Cruz's EXL3 work (b16fb91, e9780ed, 38235a9, 1bbd2dd, 5b4b343, d7d18e0, 654e4ad) is not on the GLM path in
  0.3.6.2.** It lives in a new shared module, `src/tensorfold/cuda/exl3/` (format, row-invariant dense linear, grouped
  routed experts for every codebook and width). Only `qwen3_5` (Qwen3.8-27B) and `qwen4_exp` (Flash Next) import it.
  `families/glm5_next` still runs its own `cuda/exl3.cu` and still declares
  `EXL3_VARIANT = {"bits": 4, "codebook": "mcg", "scope": "glm53_routed_experts_only"}`. A rebase would bring us none of
  it unless we wire it in ourselves.
- **Its two ideas that can apply to our checkpoint (4-bit mcg routed experts, K2 = 8):**
  1. the next k step's trellis words loaded into registers while the current one decodes (`PF = 1` in
     `experts_grouped.cuh`, the "next k step in flight" of 38235a9);
  2. a reduction buffer sized to the live rows (38235a9, on the linear only).

  Both **keep today's bits**: the per-element mma chain, the warp order and the split order are unchanged. Upstream
  has a GPU test at the real GLM shapes (288 experts, D 4096, NI 1024 / 2048, R 1-16) asserting
  `torch.equal` against GLM's `exl3_mm.routed`. **Both are already contained in our 0440 E1 (`exl3_stream.cu`)**,
  in a stronger form: warp-private `cp.async` rings three k steps deep that cross item boundaries, a persistent grid,
  and only live rows parked. So they are useful as a cheap, low-risk fallback arm for the 0440 GPU window, not as a
  new lever.
- **The EXL3 dense linear does not apply.** Our non-expert weights are BF16 in the checkpoint and re-quantized at load
  to q4mse (0001), served by the Triton `_qmm`. Nothing in our model is dense EXL3.
- **`--kv-dtype` (int8 / int4) is Flash Next only.** `serve_options.check` refuses anything but bf16 for a family
  without `CUDA_KV_DTYPES`, and glm5_next declares none. Our 0220 FP8 latent cache is already smaller than int8 per-head
  K/V would be.
- **Estimated gain on our system from Cruz's changes:** **0% prefill**. Decode, if we port the two ideas onto today's
  `grouped_kernel` (patch 0500), mid +1-2% at 1 stream and ~+0.5% at 4 streams (range 0 to +3.4%). With 0440 E1
  adopted, **0%**: E1 already does both. Upstream's own measurements put this kernel at 204-217 GB/s at one row, which
  is where ours already runs in situ (205-214).
- **The bigger upstream decode item is not Cruz's.** ashhart's 0.3.5 shared CUDA "lane matmul"
  (`cuda/kernels/qmm.cu`) replaced GLM's Triton `_qmm` for 4-bit dense decode. It is Marlin-like: a 4-stage `cp.async`
  ring, register dequant, and split-K summed in slice order inside a thread-block cluster (no `_reduce` launch). It
  keeps `_qmm`'s arithmetic form, `acc = fma(xs, b, fma(P, s, acc))`, with the same `split_k` table. That is our 0440
  E2's design space, minus the persistent grid. **Estimate: dense 175 -> ~205 GB/s, -2.6 ms a 1-stream round (+5%).**
  It arrives only with a rebase, and it brings a new packed weight layout that our `_fq4` prefill kernels, 0440 E2,
  0470 and the 0140 prepared cache all read in the old form.
- **Recommendation:**
  1. **Next GPU window: no upstream code.** Run the 0440 plan as written (DECODE-KERNELS §8) and add two cheap bench
     arms: "0500-lite" (PF = 1 + sized reduction on `grouped_kernel`) beside E1, and upstream's lane matmul
     (`qmm.cu`, built standalone from the clone) beside E2. That tells us whether a rebase buys anything on dense
     decode that E2 does not.
  2. **After RigMark: rebase (option b) only for the server and ops fixes and to stop drifting.** It is a multi-day
     port (§4), not a speed win.
  3. **Reply to Cruz:** thanks; his kernels do not reach GLM's path; we will benchmark the ideas on our 4-bit experts
     in the next window (§5).

## 1. Upstream changes that touch code our GLM path uses

Import trace: `tensorfold/cuda/server.py`, the server our two-Spark path runs, imports these:

- `server.{cancellation, errors, http.Server, messages, tool_policy, tools, text}`;
- `engine.call_gate`;
- `cuda/reply_text`.

It does **not** use `server/app.py`, `scheduler`, `checkpoints`, `prompt_memory`, `memory_budget`, `admission` or
`request_options`. Those are reached only through `cli._serve_mlx` (the Mac / lane server).

| commit(s) | author | what it does | on our GLM path? |
| --- | --- | --- | --- |
| b16fb91, e9780ed, 38235a9, 1bbd2dd, 654e4ad | vcruz305 | New `cuda/exl3/` module: format and decode for every codebook and width; row-invariant dense linear (38235a9: coalesced low-bit word runs, next k step in flight, half-depth reduction buffer); grouped routed experts for any codebook and mixed widths, with device grouping, fused down epilogue + combine, graph-safe (§2) | **No.** Imported only by qwen3_5 / qwen4_exp. glm5_next keeps its own `exl3.cu` and its 4-bit-mcg-only `EXL3_VARIANT` |
| 5b4b343, d7d18e0, e5d3a2a, 57028f3 | vcruz305 | Qwen3.8-27B and Flash Next on EXL3 packs; stop at the end of the turn from `generation_config.json` | No (other families) |
| 549b7a7, 22649b7, 6ed268b, b8612e4, 90aaff7, 67b8c61, 5f4e940 | vcruz305, ashhart | `--kv-dtype int8 / int4` (fp16 scale per 32 values) for Flash Next's per-head K/V; `serve_options.check` refuses it for any family without `CUDA_KV_DTYPES` | **No.** glm5_next declares none, so GLM on CUDA refuses int8 / int4. Our 0220 FP8 latent cache is the GLM equivalent, and smaller |
| 163da97, c96eaae | vcruz305 | `--mtp-confidence` on the CUDA serve path | No (Flash Next engine only; GLM has its own 0071 cost-derived depth) |
| e6be5ff | vcruz305 | `return_token_ids`: the reply's token ids in the response's `tensorfold` block (streamed: final chunk) | **Yes** (`cuda/server.py`, 6 lines). Useful for our exact / RigMark checks. Trivial cherry-pick |
| a3274f1 (+ b41a6b6, 85653c7, 2fbebf7) | ashhart | 0.3.5 CUDA engines. For GLM: `comm.py` moves to `tensorfold/cuda/comm.py` (store key `tf_nccl_uid`, 600 s timeout); `qmm.py` sends Q4 to the shared lane matmul with a **new packed layout** `[N/64, KG, 8, 32, gs/32]` (ours `[N/64, K/64, 64, 8]`); 4-bit prompt matmuls on `qmm_prefill` (new arithmetic); `exl3_mm.routed` takes a `cuda.experts.Plan` (items of <= 16 pairs); `DrafterChoice` moves to `drafter_choice.py`; startup admission via `cuda/capacity.admit` + `geometry.mla_geometry`; every docstring cut to one line | **Yes**, the bulk of the rebase conflicts (§3) |
| d6ce24f | taussoe | DSA index scores for any index-head count | Yes, but a no-op for GLM-5.3 (32 x 128) |
| fb985b8, 279d8f7 | taussoe, ashhart | **Latent MLA cache, on by default** (`TF_GLM_LATENT=1`). Details in the next list. Plus `tool_choice: "required"` / a named tool enforced (`cuda/server._call_gate`, `engine/call_gate.generate_gated`) | **Yes.** It is our 0060 again, a day later, without our FP8 / pool / 1M / sessions layers (§3) |
| 24afe5e, 740811a, ce09822, 9270aeb, 9a1e6f1 | nood-co1 | The CUDA server stops a request when its client has gone or its per-token callback fails. `cancelled=` is checked every round, queued requests of departed clients are dropped, and the callback never raises into the engine (it used to desync rank 1) | **Yes.** A real robustness fix. Our 0160 / 0370 cancel on disconnect in the batch path, but the "callback never raises into the engine" part is worth taking |
| 4fb2528, 1fc5915 | jeidbugs404, ashhart | GLM `<arg_key>/<arg_value>` tool-call parser, one for both servers (`server/tools.parse_glm_tool_call_block` via `cuda/reply_text`) | **Yes. Supersedes our 0002.** One difference: an untyped parameter stays text upstream, where ours tries JSON first |
| 675d4c2 | gilby | Spill evicted conversation prefixes to disk (`--spill-gib`), `--checkpoint-slots` | **No:** MLX / lane server only (`server/checkpoints.py` via `_serve_mlx`). Our 0110 / 0250 / 0310 are the CUDA equivalent |
| 6f474d1, 222fe5f, 70a6113, 153f488, 923c1f0 | ashhart, Chedrian07 | Memory refusal messages naming `TENSORFOLD_MEMORY_LIMIT_GB`; Flash Next host-memory admission | **No:** MLX server (`server/admission`, `memory_budget`) |
| 8c8dcec | chris247474 | Request `reasoning_effort`, typed Qwen tool parameters | No (MLX path). Our 0150 effort mapping stays |
| 34bae79 | ashhart | 0.3.6.1: extensions build for the GPU present (NVIDIA containers name sm_80+) | Yes (`cuda/build.py`). Harmless for us: our image builds for sm_121 only (0330 note) |
| 5d5140e, eab4963, 9a82b16, 72e6809, fcca783, 0b77e6d, dc207e6, 4bb7786, 1f84d15 | several | GLM-5.3-Flash on Apple Silicon (MLX), mixed-bit MLX checkpoints | No (Mac) |

The upstream latent MLA cache (fb985b8 / 279d8f7) in more detail:

- **Layout.** GLM's attention has no rotary part, so the cache holds only the 512-wide latent. It is a flat
  contiguous `(capacity, 512)` bf16 per attention layer, plus the MTP layer. Index keys are `(cap, 128)` bf16 plus
  `(cap/4 + 2, 128)` pool keys. That is **19,200 B a token and rank, the same as our 0060**. There are no pages, no
  pool, no FP8, and a full copy on each rank.
- **Sparse path.** Absorbed q and expanded v (bf16 `AbsorbW` for EXL3); radix-select of the top-512 pools over a
  power-of-2 bucket (minimum 1024); sparse CUDA graphs captured eagerly per bucket; `kda.cu`, a three-kernel prompt
  chain with the same bits.
- **A kept-conversation store.** `TF_GLM_CACHE_GIB=3`, 8 entries.
- **The "256k on two Sparks" rule.**
  - budget = MemAvailable - max(4 GiB, 10% of RAM);
  - binary-search the largest window where weights + `mla_geometry.bytes_at(window + MAX_ROWS)` + DFlash2 fit;
  - take the minimum over the two ranks, capped at the native `max_position_embeddings`;
  - refuse an explicit `--context` above it.
- **Measured upstream.** 93.7 GiB peak at 262,144 tokens. MLX checkpoint only: "EXL3 long context TBD".

Other upstream commits (Gemma 4, Qwen 3.6 / 27B concurrency, Mac memory, MLX prefill) do not touch our files.

## 2. The EXL3 kernel changes against our kernels

### 2.1 What upstream's EXL3 kernels are

- **`cuda/exl3/experts_grouped.cuh` + `experts.cu` (1bbd2dd).** The docs say it plainly: "the structure of GLM-5.3's
  own expert kernel (`families/glm5_next/cuda/exl3.cu`) generalized off the 4-bit-`mcg` case".
  - Same grid, (expert, n block, mat x split x member tile), same 4 warps with fixed k ranges, the same
    `red[W][16][NT*16]` warp-order reduction, the same epilogues. Plus a `group_kernel`, and `down_combine` (the down
    epilogue and the combine in one launch).
  - What is new: a per-expert K2 (1-8 bits in half-bit steps) through pointer tables, and the three codebooks.
  - At K2 = 8 it keeps our lane decode verbatim (`__funnelshift_r` on words L-1 / L). The GLM tile settings are
    `GLM_GATEUP = (8, 4, 4, 1)` and `GLM_DOWN = (8, 4, 1, 1)`: NT, W and SK as ours, plus **PF = 1**.
  - Upstream's measurements (MiMo 2.5 bpw, mixed 2-5 bit, one GB10): 179.6-209.7 GB/s at one row (195.6-217.2 in a
    graph), 210-226 at 8 rows. The authors' own conclusion is "this work does not beat `exl3_moe_coop` on uniform
    layers, it removes the mixed-K penalty". **Our layers are uniform 4-bit, where the mixed-K penalty does not
    exist.**
- **`cuda/exl3/linear.cu` (e9780ed, then 38235a9).** The dense EXL3 linear: rot_in, then a program per (128-column
  block, K split), WK warps with fixed k runs, split sums in split order by the last program to arrive. 38235a9 adds
  three things:
  - (a) at 1-2 bits, a warp loads its k step's words as one coalesced run and lanes take theirs by shuffle;
  - (b) up to 6 bits, the next k step's words are loaded while the current one decodes;
  - (c) the reduction buffer is sized `WK * min(M, 8)` rows instead of 16.

  Measured in the commit: 2-bit 76-157 -> 186-220 GB/s, 3-5 bit +10-15%, 4-bit Qwen `in_proj_qkv` 196 -> 216 GB/s.
- **`cuda/exl3/prefill.py`.** Dense EXL3 prompt matmuls: W_q decoded once a chunk, then a fixed-tile fp16 Triton GEMM.
- **`cuda/exl3/format.py`, `decode.cuh`, `inspect.py` (b16fb91).** The format for every codebook and width, a numpy
  reference, and header-only decode. No speed content.

### 2.2 Technique by technique, against ours

"prod" = today's `grouped_kernel` (exl3.cu, 0.3.4 + 0006 `grouped_loop` for > 16-member windows) and the Triton
`_qmm`. "0440" = E1 `exl3_stream.cu` / E2 `q4_stream.cu` (built, off, not yet GPU-measured).

| technique (upstream source) | prod | 0440 | same bits for us? | worth on our system |
| --- | --- | --- | --- | --- |
| Tile decoded straight into mma.m16n8k16 B fragments, lane L from words L-1 / L (both) | **yes**: upstream's K2 = 8 path *is* our `decode_tile` | yes, verbatim | yes (identical code) | 0: already ours |
| Any codebook (3inst / mcg / mul1), any width 1-8, mixed widths per expert, pointer tables (1bbd2dd) | no (4-bit mcg only) | no | n/a | 0: our checkpoint is uniform 4-bit mcg. Would matter only for a mixed-width GLM pack (none exists; see §2.5) |
| Warp-coalesced run load + shuffle for 1-2-bit words (38235a9 a) | n/a: at 4 bits each lane already loads one 32-bit word of a 128 B tile, coalesced | n/a | n/a | 0 |
| **Next k step in flight** (38235a9 b; `PF = 1` in the grouped experts) | **no**: `words[NT]` loaded at the top of each k step, then decoded | **yes, deeper**: E1 keeps STAGES-1 = 3 k steps a warp in flight in a shared-memory `cp.async` ring, across item boundaries, on a persistent grid | **yes**: only load timing moves; the mma chain, warp order and split order are unchanged. Upstream's `test_glm_shaped_bit_identical` asserts it at the real shapes | prod + PF: mid -0.6 ms a 1-stream round (+1.2%), range 0 to +2.3%. **With E1: 0** |
| **Reduction buffer sized to live rows** (38235a9 c; linear only, not in upstream's grouped experts) | **no**: static 32 KB `red`, the reason `grouped_kernel` sits at 3 CTAs an SM (ncu, W11) | **yes**: E1 parks only live rows, 30 KB with the ring | yes: same warp-order sum, fewer rows stored | with PF: mid -1.0 ms (+1.9%), range +0.5 to +3.4% (3 -> 4 CTAs an SM; registers then limit). **With E1: 0** |
| Split-K finished by the last program to arrive, atomic ticket (linear) | experts: separate epilogue launches; dense: separate `_reduce` launch | **yes**: E1 tickets per (member tile, expert, Hadamard block), E2 per tile | yes (fixed slice order) | already in 0440 |
| Device grouping in one block (`group_kernel`) | 0190 parallel-sort grouping (`MOE_GLUE=5`) | same | yes (grouping never changes a row's arithmetic) | 0 |
| Down epilogue + combine in one launch (`down_combine`) | no: `down_epilogue`, then the 0190 in-place combine | no: E1 fuses the down epilogue, the combine stays separate | yes: upstream asserts `torch.equal` against `glue.combine`. Our 0190 in-place combine is also same-bits as `glue.combine` | ~42 launches a round: ~0.1-0.15 ms (0.2-0.3%). A later E1 follow-up: fuse the combine into E1's down-epilogue ticket |
| Graph-capturable, no allocation per call | yes | yes | - | 0 |
| Dense EXL3 linear / EXL3 prefill GEMM (e9780ed, `prefill.py`) | n/a: our dense weights are q4mse | n/a | n/a | 0 |

### 2.3 Not Cruz's, but the one upstream decode kernel that would reach us: the shared 4-bit lane matmul

`src/tensorfold/cuda/kernels/qmm.cu` (a3274f1, 0.3.5, with an optional pipelined epilogue; 2fbebf7 touched it again). Since 0.3.5,
`glm5_next/cuda/qmm.matmul` sends every Q4 matrix there instead of the Triton `_qmm`.

- **Design.**
  - Weights are re-packed at load into a lane layout `[n/64][k/64][8][32][2]`, so a thread's nibbles are one
    register.
  - A 4-stage `cp.async` ring carries inputs (swizzled, `ldmatrix`), weights, scales, biases and group sums.
  - Nibbles go straight into bf16 B fragments; `mma0` starts each group's P from zero.
  - The epilogue is `acc = __fmaf_rn(xs, b, __fmaf_rn(P, s, acc))`.
  - K slices form a thread-block cluster: slice 0 adds the peers' partials from DSMEM in slice order (at most 8
    slices; more fall back to `reduce_kernel`).
  - Non-persistent, 16 / 32 / 64-row buckets.
- **Against ours.**
  - It is the E2 technique set minus two things: E2's persistent grid, and E2's ring that crosses item boundaries.
    Instead of E2's last-arriving ticket it uses cluster DSMEM for split-K.
  - Its arithmetic is exactly the form our E2 had to reproduce: the Triton 3.7.1 fused `fma(xs, b, fma(p, s, acc))`,
    the same `SHAPE_SK` / `split_k` table, and slices in order.
  - **Same bits as today's `_qmm` is plausible but not proven.** Upstream's test (`tests/cuda/test_qmm.py::
    test_27b_triton_bits`) checks it against the 27B's Triton lane matmul, not GLM's `_qmm`. One GPU test settles it:
    our `test_decode_stream_patches.py` E2 cases, with the lane matmul as the "new" side.
- **Estimate.** Dense 175 GB/s in situ (W11) -> low 190 / mid 205 / high 215, plus the 224 `_reduce` launches gone:
  **-1.5 / -2.6 / -3.2 ms a 1-stream prose round (+5.1% mid)**, and -3.3 ms (+2.8%) at 4 streams. E2's mid is
  -3.3 ms (+6.7%) on the same W11 baseline. They overlap: take one, not both.
- **Cost.**
  - It needs the new packed layout. Our `_fq4` fast-prefill kernels (0081 / 0083 / 0170), `q4_stream.cu` (0440 E2),
    0470 (q8 non-experts, in progress) and the 0140 prepared rank folders all read the 0.3.4 layout
    `words [N/64][K/64][64][8]`.
  - Holding both layouts costs the whole q4 non-expert set twice. That does not fit our memory margin (4 x 250k
    stress minimum 9.4-10.5 GiB against the 8 GiB floor).
  - So it comes with a rebase (and a port of `_fq4` to the lane layout), or we test it standalone as a bench arm only.

### 2.4 Decode and prefill estimates (W11 baseline; `results/upstream-0362/estimate.py`)

W11 round (uncaptured):

- 1 stream prose: 53.3 ms / 2.42 tokens = 45.4 tok/s. Experts 5.47 GB at 205 GB/s (27.8 ms), dense 2.75 GB at
  175 GB/s (16.1 ms).
- 1 stream code: 63.6 ms / 4.16 tokens.
- 4 streams: 121.3 ms / 9.1 tokens. Experts 16.1 GB at 220 GB/s, dense 3.51 GB at 169 GB/s.

Ceilings: 233-238 GB/s (experts), 230.5 GB/s (dense).

| change | bits | 1 stream prose, ms (low / mid / high) | mid tok/s | 4 streams, mid | prefill |
| --- | --- | --- | --- | --- | --- |
| U1: upstream grouped experts at GLM settings (PF = 1), drop-in | same | -0.0 / -0.6 / -1.2 | +1.2% | +0.3% | 0 |
| U2: U1 + live-row reduction buffer (the 0500-lite port) | same | -0.3 / -1.0 / -1.8 | **+1.9%** | +0.5% | 0 |
| E1 (0440, ours) | same | -1.0 / -2.2 / -3.1 | +4.3% | +2.3% | 0 |
| L1: upstream lane matmul (rebase only) | same (to verify) | -1.5 / -2.6 / -3.2 | **+5.1%** | +2.8% | 0 (decode kernel) |
| E2 (0440, ours) | same | -1.9 / -3.3 / -3.8 | +6.7% | +3.2% | 0 |

Notes on the table:

- **The 4-stream rows are smaller than DECODE-KERNELS §7.** That section assumed ~193-197 GB/s effective experts at
  4 streams. W11 measured 208-220 in situ, which leaves less room (DECODE-PLAN already says so).
- **U1 / U2 are not additive with E1.** E1 already contains both.
- **Prefill is ~0 for every EXL3 change.** Our prefill is MMA-bound at 2-8k-row chunks: fat experts 77 us MMA roof
  against 73 us DRAM (ROOFLINE), `_fq4` 38-57 TFLOP/s. Nothing upstream changed in the GLM EXL3 expert prefill.
- **Upstream's new prefill kernels are deliberately new arithmetic.** These are the shared `qmm_prefill.cu`,
  `qmm_prefill8.cu` (FP8) and `experts_prefill.cu`: `w = bf16(fma(q, s, b))` then one fp32 chain over K,
  "chunk-invariant bits, not decode's (replies prefill again)". Our fast prefill keeps decode's per-group arithmetic
  (`_fq4`), so prefill and decode stay in lockstep and resumed == fresh without a re-prefill. Adopting upstream's
  model would change our reply bits and our session / prefix-share contract. **Not recommended.** ROOFLINE item 3's
  same-bits `_fq4` rewrite remains the prefill lever.

### 2.5 Exactness

- **Everything in §2.2 that applies to us keeps today's bits by construction:**
  - PF prefetch and the sized reduction change when words arrive and how many rows are parked, not what is added in
    which order.
  - Upstream's GLM-parity GPU test at the real shapes is the same check our 0440 tests make.
  - So drafted == serial, batched == alone and resumed == fresh hold, and reply SHA `8794a3463259cc2f` should not move.
- **The lane matmul (§2.3) is the only candidate whose bits are not yet proven equal to ours.** If a GPU test shows a
  difference, it is still row-invariant (upstream tests that), so the exactness contract holds, but the reply SHA
  moves. A rebase would then need a new baseline SHA, MMLU-200 and refusals again.
- **One outside item to remember (DECODE-KERNELS §2.2).** Under Triton 3.8, `_qmm` leaves some epilogue product-adds
  unfused, bucket-dependently. Upstream 0.3.6.2's GLM dense decode no longer depends on Triton's contraction (the lane
  matmul spells out `__fmaf_rn`). A small point in the rebase's favour: it removes that upgrade hazard.
- **Mixed-width EXL3 GLM packs.** If someone publishes a GLM-5.3-Flash EXL3 pack with mixed expert widths (e.g.
  3/4/5-bit), Cruz's grouped kernel is exactly what GLM would need. Neither our engine nor upstream's glm5_next can
  load one today.

## 3. Overlap and conflict map (our patches against 0.3.6.2)

### 3.1 Mechanical probe (`results/upstream-0362/rebase_probe.sh`, log `rebase-probe.txt`)

| stage | result |
| --- | --- |
| 1: the series 0001-0490 (incl. the uncommitted 0470) on 2f8e514, Dockerfile method | all apply |
| 2: each patch alone on 71377a5, `git apply --3way` | **every patch but 0081 conflicts** (later ones partly cascade on files earlier ones create) |
| 3: keep-going with `--reject`, rejects classified as caused by upstream changes vs by our own earlier rejects | clean: 0081, 0200, 0335, 0340, 0350, 0360, 0400, 0410. Most upstream-caused rejects: 0050 (22), 0080 (21), 0005 (15), 0490 (13), 0290 (11), 0420 (11) |
| 4: the squashed series (2f8e514..tip) merged onto 71377a5 with `git merge-tree` | **19 files, about 100 conflict regions**: forward.py 21, server.py 11, decode.py 11, engine.py 9, sparse.py / latent.py (add/add) / glue.py / mtp.py 5 each, weights.py 6, qmm.py 4, graphs.py 4, app.py / dflash2.py / exl3.cu 3 each, exl3_mm.py 2, attention.py, pyproject, THIRD_PARTY_NOTICES, the recipe 1 each |

Many conflicts are cosmetic: upstream cut every docstring to one line, which moves context under our hunks. The
squash merge is therefore the practical route (§4 b).

### 3.2 Five semantic breaks a textual merge would hide

1. **`comm.py` moved** to `tensorfold/cuda/comm.py`. Nine of our patches import `from . import comm as comm_mod`
   or `from .comm import fast_gather`, and 0040 puts `from .overlap import nccl_env` inside comm.py.
   - Fix: a re-export shim in `glm5_next/cuda/comm.py` (~1 h).
2. **The Q4 packed layout changed** (§2.3). Everything that reads our layout would compute garbage silently:
   - `_fq4` (0081), `_qmm2` (0130), `q4_stream` (0440), `l2pf` (0460);
   - the q4mse loader (0001), 0470, the 0140 prepared folders, 0420's draft-head rows.
   - Fix: **keep our `qmm.py`** (Triton `_qmm`, old layout) in the rebase. The lane matmul becomes a separate,
     measured follow-up (§4 b step 6).
3. **Startup admission.** `cuda/capacity.admit` in `GlmEngine.__init__` knows nothing of FP8 rows, rings, batch
   slots, the 0290 pool or 1M contexts, and caps the window at the native context. It would refuse or mis-size
   `CONTEXT=1048576`.
   - Fix: replace or extend it with our pool / slot rule.
4. **Two conversation caches.** Upstream's kept-conversation store (`TF_GLM_CACHE_GIB=3`) would run beside our
   0110 / 0180 / 0250 / 0310 session store, and cost 3 GiB we do not have.
   - Fix: disable it.
5. **`latent.py` add/add.** Both sides created the file.
   - Fix: keep ours (0060 + 0065 + 0220 + 0290 + 0390 + 0460) and map `TF_GLM_LATENT` onto `GLM53_TF_LATENT_KV`.

### 3.3 Per patch

Legend:

- **rej**: hunks rejected because of upstream changes (stage 3). A second number in brackets counts hunks rejected
  only because an earlier patch of ours was rejected.
- **Relation**: superseded (upstream does it), overlap (upstream does part or a variant), complementary (upstream
  has nothing like it), or break (§3.2).
- **Effort**: port hours, excluding GPU validation.

| patch | rej | relation to 0.3.6.2 | effort |
| --- | ---: | --- | --- |
| 0001 q4mse non-experts | 6 | complementary; break 2 (layout) | 2-3 h |
| **0002 tool-call parser** | 3 | **superseded** by 4fb2528 / 1fc5915. Check the untyped-parameter difference with our clients, then drop | 0.5 h |
| 0003 prefill rows | 2 | partly superseded (engine `prefill_rows` argument) | 1 h |
| 0004 sparse long prefill | 3 | mostly superseded by fb985b8's sparse.py | 1 h or drop |
| 0005 prefill profile | 15 | overlaps upstream `prof.py` | 2-3 h or drop |
| 0006 expert loop | 8 | `Plan` API (items, counts, members) instead of uids / ucount | 2-3 h |
| 0010 / 0020 / 0030 | 3 / 6 / 4 | complementary; 0030 maps onto upstream's `engine.concurrent` hook | 1-2 h each |
| 0040 comm prefetch | 10 | break 1 (comm move) | 2-3 h |
| **0050 long-ctx decode** | 22 | **largely superseded** (upstream: eager per-bucket sparse graphs; ours: lazy capture, 1-16 rows, per parity). Keep ours | 4-6 h, hard |
| **0060 latent KV** | 7 + add/add | **functionally superseded** by fb985b8 (same 19,200 B a token), but it is the base of our 0065 / 0220 / 0290 / 0390 / 0460 stack. Keep ours (break 5) | 6-10 h, hard |
| 0065 1M memory + indexer | 7 | complementary (rings, blocked select; upstream caps at native) | 3-4 h |
| 0070 / 0071 calibration, depth | 4 / 9 | `DrafterChoice` moved to `drafter_choice.py` | 1-2 h / 2-4 h |
| 0080 fast prefill | 21 | complementary (no fast EXL3 prefill upstream); break 2 via `_fq4` | 4-6 h, hard |
| 0081 fast kernels | 0 | clean | trivial |
| 0082-0085 lean / FP8 / overlap / chunk-independent | 9 / 6 (+11) / 6 / 10 | complementary | 2-4 h each |
| 0090-0093 request knobs | 9, 2-3 each | complementary | 4-6 h together |
| 0110 session cache | 10 | **supersedes** upstream's kept store (break 4) | 4-6 h |
| 0120 / 0180 / 0200 batching | 4 / 2 / 0 | complementary: our slots, pool and sessions go beyond upstream's `engine.concurrent` round sharing | 2-3 h / 1 h / trivial |
| 0130 decode step | 10 | break 2 (`_qmm2`), Plan API | 3-4 h |
| 0140 fast boot | 10 | breaks 2 + 3 (prepared layout, admission) | 3-4 h |
| 0150 health, metrics, effort | 8 | complementary; its cancel pieces overlap 24afe5e | 3-4 h |
| 0160 OpenAI compat | 6 | complementary; overlaps upstream's `server/messages` / `errors` | 2-3 h |
| 0170 / 0190 Mia prefill, glue | 5 / 10 (+5) | complementary | 1-2 h / 3-4 h |
| 0210 prompt tokens | 7 | half superseded (`PreparedRequest`); our token cache remains | 1.5-2 h |
| **0220 FP8 latent KV** | 4 (+10) | complementary: **upstream has no FP8 latent; `--kv-dtype` does not reach GLM** | 1-2 h |
| 0230 / 0350 / 0460 RoCE, L2 prefetch | 6 / 0 / 6 (+3) | break 1 (comm move) | 2-3 h / trivial / 2-3 h |
| 0240 / 0250 / 0260 / 0270 / 0280 | 7 / 4 / 2 / 5 / 2 | complementary; 0250 is not superseded by 675d4c2 (Mac only) | 0.5-3 h each |
| **0290 KV pool** | 11 (+14) | complementary; break 3: must replace upstream's admission | 6-10 h, hard |
| 0300 / 0310 / 0320 / 0330 | 6 / 1 / 0 (comm) / 3 | complementary | 0.5-2 h each |
| 0335 / 0340 / 0360 / 0400 / 0410 | 0 | clean | trivial |
| 0370 / 0380 | 5 / 3 | complementary | 1-2 h each |
| 0390 MLA expand v2 | 0 (+4, latent.py) | trivial if our latent.py is kept | trivial |
| 0420 / 0430 | 11 / 5 | complementary; 0420 break 2 (draft-head rows) | 2-4 h / 1-2 h |
| 0440 decode stream | 5 | Plan API in `exl3_mm.routed`; E2 reads our layout (fine if our qmm.py is kept) | 2 h |
| 0470 q8 non-experts (uncommitted) | 9 (+6) | break 2 | 3-4 h |
| 0490 API context | 13 | complementary; upstream's server restructure (`server/*` helpers) | 2-3 h |

Summary of the map:

- **Superseded:** 0002 (drop); 0004 mostly. 0050 / 0060 are superseded in design, but ours stay as the base.
- **Upstream supersedes nothing else of ours.** Nothing upstream for GLM on CUDA matches: FP8 latent, the KV pool, 1M,
  our slot batching, sessions / NVMe / prefix share, RoCE, fast / lean / b12x prefill, deep verify or the draft
  vocabulary.
- **Worth adopting from upstream:** 24afe5e (disconnect / callback robustness), the `tool_choice: "required"` call
  gate, `return_token_ids`, and the shared GLM tool-call parser.

## 4. Options, effort and risk

### (a) Port only the EXL3 kernel ideas onto 0.3.4, for the next GPU window. Recommended, small.

There is nothing to cherry-pick verbatim. Upstream's grouped kernel needs its own `prepare` / pointer tables /
`Plan`, and glm5_next never calls it. The two portable ideas become one small patch, plus a bench arm.

- **Patch 0500 `glm-exl3-grouped-pf`** (knob `GLM53_TF_EXL3_PF=0|1`, off).
  - In `exl3.cu`'s `grouped_kernel` (and 0006's `grouped_loop`), load the next k step's `NT` words into registers
    while the current step decodes. This is `PF = 1`: +8 registers, 108 -> ~116.
  - Size the warp-reduction buffer to the live rows as dynamic shared memory (R <= 4: ~8 KB instead of 32 KB), so the
    kernel goes from 3 to 4 CTAs an SM.
  - Same per-element mma chain, warp order and split order: **same bits**. No new weight format, no host changes
    beyond the launch.
  - Tests:
    - add the arm to `tests/cuda/test_decode_stream_patches.py`: on == off for the 14 windows x eager / graph, rows
      alone, 4-slot mixes;
    - port upstream's `test_glm_shaped_bit_identical` shapes;
    - a compile test for registers, spills and CTAs an SM on sm_121;
    - the arm in `bench_decode_kernels.py` beside old / E1 and the roof probe.
  - **Effort: ~0.5-1 day offline, ~10 min of the 0440 GPU window.**
  - **Expected: mid +1-2% at 1 stream (0 to +3.4%), ~+0.5% at 4 streams.**
  - Only worth adopting if E1 misses its gate: E1 contains both ideas.
- **Bench-only arm: upstream's lane matmul.**
  - A standalone script that builds `cuda/kernels/qmm.{cpp,cu}` + `qmm_frag.cuh` (+ the two prefill sources its
    binding links) from the clone with `torch.utils.cpp_extension.load`.
  - It packs the same random MLX words both ways and times `_qmm` + `_reduce` / E2 / the lane matmul on our 11
    per-rank dense shapes at M 1-16. It checks `torch.equal` of the lane matmul against `_qmm`, which answers the
    open bits question of §2.3.
  - **Effort: 2-3 h offline, ~5 min GPU.**
  - This decides whether the rebase should ever switch GLM's dense decode to the lane matmul, or keep E2.
- **Tiny server cherry-picks** onto 0.3.4 `cuda/server.py`, if wanted before the rebase:
  - e6be5ff `return_token_ids` (6 lines): 0.5 h, host-side only.
  - The engine-callback half of 24afe5e: 2-3 h, because it touches our 0150 / 0160 / 0490 hunks.
  - Neither changes bits. `return_token_ids` would let the RigMark / exact harnesses hash ids through the OpenAI
    path.

GPU window order (inside DECODE-KERNELS §8, +15 min):

1. Bitwise: E1, E2 and 0500 (+ the lane-matmul `torch.equal` check).
2. Microbench: old / 0500 / E1 experts; `_qmm` / E2 / lane matmul dense.
3. Engines.
4. Production A/B of the winners only.

Gates are unchanged: bits equal everywhere, 1 stream >= +3% for E1 + E2; 0500 alone >= +1% if E1 fails.

### (b) Full rebase to 0.3.6.2 (or the 0.3.6.x current then), after RigMark as decided

- **Why do it.**
  - Upstream fixes on our path: disconnect / callback robustness, the `tool_choice` gate, the shared parser,
    `return_token_ids`, the build fix.
  - Staying mergeable with a fast-moving upstream (70 commits in 2 days).
  - Easier upstreaming of our work: FP8 latent KV, RoCE, streaming kernels, the KV pool.
- **Why it is not a speed item.**
  - Nothing in 0.3.6.2 makes GLM decode or prefill faster *for our configuration*. The latent cache equals our 0060;
    Cruz's EXL3 work does not reach GLM.
  - The one candidate, the lane matmul, is better measured first via (a)'s bench arm.
- **Effort** (from §3.3):
  - Patch by patch: ~110-130 h.
  - **Squash merge + the five semantic fixes: ~40-60 h** (~1-1.5 weeks).
  - Either way, **15-25 h of GPU validation**: `exact` / `batchexact`, reply SHA, MMLU-200, refusals, the 4 x 250k
    stress, needle 314k, prefill and decode A/B against b4.
- **Risk.**
  - Medium-high. The conflicts sit in `forward.py`, `decode.py`, `engine.py`, `latent.py` and `server.py`: the
    exactness-critical core.
  - Upstream's admission and kept store are on by default and would silently change memory.
  - If the lane matmul is taken, the reply SHA changes (a new baseline).
  - Mitigation: keep our `qmm.py`, `latent.py` and admission, so the rebased image serves the same bits as b4.
    Treat any SHA change as a bug unless it is explained.
- **Plan.**
  1. **Branch.** A new submodule branch at 71377a5 (the pin in the superproject changes only after the gates pass).
     Squash our series (`git merge-tree` result as the start), resolve the ~100 regions, and regenerate
     `patches/*.patch` against the new base per original patch (the Dockerfile keeps applying files in order).
  2. **Semantic fixes (§3.2).**
     - The comm shim.
     - Keep our `qmm.py` / Q4 layout. Bump the 0140 prepared-folder and 0250 NVMe compat hashes anyway, since the
       base changed.
     - Replace `capacity.admit` for GLM with our slot / pool rule.
     - `TF_GLM_CACHE_GIB=0` (upstream's kept store off).
     - Keep our `latent.py`, with `TF_GLM_LATENT` mapped to `GLM53_TF_LATENT_KV`.
  3. **Drop or shrink superseded patches.**
     - Drop 0002: take upstream's parser, and add a JSON-first fallback for untyped parameters if our clients need
       it.
     - Fold 0004 / part of 0003 / part of 0210 into upstream's versions.
     - Keep 0050 / 0060 as ours.
  4. **Port in dependency order.**
     - Kernels and weights: 0001, 0006, 0080-0085, 0130, 0170, 0260-0270, 0330, 0440.
     - KV: 0060, 0065, 0220, 0290, 0390, 0410, 0460.
     - Engine / drafting: 0010, 0020, 0070, 0071, 0380, 0420, 0430.
     - Batching / sessions: 0030, 0120, 0180, 0200, 0250, 0280, 0300, 0310, 0335, 0340.
     - Comm: 0040, 0230, 0320, 0350.
     - Server: 0150, 0160, 0210, 0490 on top of upstream's 24afe5e / call gate / `return_token_ids`.
  5. **Offline suites**, then one GPU day with the W10 gate set. **Adopt only if the reply SHA equals b4** (or every
     difference is explained) and speed is within +-2%.
  6. **Follow-up, a separate knob.** The lane matmul for GLM dense decode, if (a)'s bench showed it matching or
     beating E2 with equal bits. Otherwise keep E2.

### Recommendation

1. Do (a) now: 0500 + the lane-matmul bench arm, added to the already-planned 0440 window.
2. Do (b) after RigMark, as a hygiene / robustness rebase with our kernels and KV stack kept. Budget ~1.5 weeks of
   offline work plus ~1 GPU day.
3. Expect no decode or prefill speed-up from the rebase itself. The speed levers remain 0440 E1 / E2 and the
   DECODE-PLAN items.

## 5. Reply to Cruz

A short reply to the upstream author was drafted from sections 0-4 (his kernels do not reach GLM's path in
0.3.6.2; we bench the two ideas on our 4-bit experts in the next window; the full rebase comes after RigMark). The
draft itself is not reproduced here.
