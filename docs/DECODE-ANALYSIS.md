# GLM-5.3-Flash decode on 2x GB10: where a drafted round's time goes

Offline analysis of `vendor/TensorFold` @ 2f8e514 + patches 0001-0004 (decode path:
`src/tensorfold/families/glm5_next/cuda/{decode,mtp,dflash2,engine,graphs}.py`), using the costs the engine
calibrated at load on the real model with our patches (EXL3, `GLM53_TF_NONEXPERT=q4mse`):

| Piece | ms |
| --- | ---: |
| Verify window, 1..8 rows | 30.0, 40.0, 44.7, 49.5, 54.2, 58.9, 63.7, 68.4 |
| MTP draft (absorb + sample) / each further chained draft | 2.04 / +1.68 |
| DFlash2 block (one pass, up to 7 candidates, host chain) | 3.88 |
| Serial decoding | 33 tok/s = 30.3 ms a token |

Measured `auto` medians: code sampled 44.9, chat sampled 41.7, code greedy 61.0, chat greedy 44.3; kit
(greedy, 200 tokens) hashmap 49.1, structured 85.9, essay 43.9 tok/s.

## 1. Anatomy of a round

Every round is: draft (MTP chain or DFlash2 block), stage + verify window (CUDA graph for 1..6 rows, eager for 7
and 8), sample every row on the host, commit (KDA replay of the kept prefix + conv shift), then the drafters take
the kept rows.

### Host-device synchronizations per round

"Hard" = the host blocks until the GPU queue drains, so the GPU idles for the host work that follows until the next
launch. "Event" = `b.staged.synchronize()` in `stage`/`mtp_stage`; it waits on the previous staging copy, which
has always finished by then (a hard sync came earlier), so it costs a few µs.

| Where | Call | Kind | Per round |
| --- | --- | --- | --- |
| `decode.draft` -> `sample_rows` | topk, eager all-gather, `g.cpu()` then numpy lexsort (+ `_probability` for `cN:P`) | hard | 1 per MTP draft (d) |
| `decode.draft` -> `Engine.mtp` -> `mtp_stage` | pinned `ids_host` write, H2D copy, `hin` copy, graph replay | event | 1 per MTP step |
| `mtp_decode`/`auto_decode` after `e.forward` | `torch.cuda.synchronize()` (only for stage timing) | hard | 1 |
| `sample_rows` on the verify logits | topk [R, 77,440] fp32, eager all-gather, `.cpu()`, numpy `choose_rows` | hard | 1 |
| `forward.stage` | pinned write, H2D copy of R ids | event | 1 |
| `forward.commit` | `replay_layers` (keep < R), `_conv_shift`, `pos_dev.fill_` | async | 0 |
| `Drafter.candidates` | `packed[:, :depth].cpu()`, then `proj[:depth].cpu()` | hard x2 | 2 per DFlash2 block |
| `Drafter.chain` | numpy selector edges (`succ[tok] @ (pred[prev] * proj)`), float64 Gumbel for sampled | host | per DFlash2 block |
| `auto_decode` backlog | `m_rows` copy, `tap_rows` cat, `add_taps` graph replay | async | 0 |

Hard syncs per round: MTP round with d drafts = d + 2; DFlash2 round = 4. The GPU sits idle after each for the
host work up to the next launch.

### What a hard sync costs on GB10

- Anchor from measurements: serial decoding is 30.3 ms a token against a 30.0 ms one-row window, so a verify
  step's host tail (explicit sync, topk + all-gather launch, D2H, numpy draw, commit launches, next stage) is about
  **0.3 ms**.
- An MTP step reads about 270 MB a rank (its half of the 154,880-row head ~160-170 MB, the MoE layer ~64 MB, DSA
  ~35 MB, eh_proj ~4 MB). At the ~220-230 GB/s the kernels reach, that is ~1.2-1.3 ms of GPU work. The measured
  chained draft is 1.68 ms, so the host bubble per MTP draft (topk/all-gather launch, D2H, numpy, staging, graph
  launch) is **~0.3-0.4 ms**. The first draft (2.04 ms) also pays the absorb staging and a cold graph launch.
- A bare D2H round trip plus Python on the Grace cores is ~20-50 µs. Most of each bubble is Python and numpy
  around it (lexsort, `_probability`, list building, tensor allocs for `torch.cat`/`empty`) and relaunching.
- All-gathers: 90 a forward, ~2.4 ms of a window (captured in the graphs, ~27 µs each); ~3 per MTP step
  (~0.1 ms); ~11 per DFlash2 block (~0.3 ms). Plus one eager all-gather per sample (~20-50 µs over the link).

## 2. Per-round time budget

| Round | Draft | of which host bubbles | Verify | Host sample + commit | Round | Max tokens | Ceiling tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Serial (no draft) | - | - | 30.0 | 0.3 | 30.3 | 1 | 33.0 |
| MTP, 1 draft (`a` low) | 2.04 | ~0.4 | 40.0 | 0.3 | 42.3 | 2 | 47.2 |
| MTP, 2 drafts | 3.72 | ~0.8 | 44.7 | 0.3 | 48.7 | 3 | 61.6 |
| MTP, 3 drafts (`c3`, `a` high) | 5.40 | ~1.1 | 49.5 | 0.3 | 55.2 | 4 | 72.5 |
| DFlash2, 5 drafts (`fc5`, upstream cap) | 3.88 | ~0.3 | 58.9 | 0.3 | 63.1 | 6 | 95.1 |
| DFlash2, 7 drafts (`fc7`) | 3.88 | ~0.3 | 68.4 | 0.3 | 72.6 | 8 | 110.2 |

Shares in a typical 3-draft MTP round: verify 90% (experts ~45%, dense 4-bit matmuls ~30%, all-gathers ~4%,
hyper-connections and KDA ~7%), MTP GPU work 7.5%, host bubbles in drafting ~2%, verify-side host ~0.5%.

Marginal prices that decide depth: row 2 of a window costs 10 ms and every row from the 3rd on ~4.7 ms. A
further DFlash2 draft costs only its verify row (4.7 ms; the block computes 7 positions anyway), a further MTP
draft 4.7 + 1.68 = 6.4 ms. Draft i pays when P(drafts 1..i all accepted) > marginal / (ms per committed token):
at 60 tok/s that is 0.28 (DFlash2) and 0.38 (MTP); at 86 tok/s 0.40 and 0.55.

## 3. Candidates

**(a) GPU-side draft picking / one graph for the MTP chain.** Removes d - 1 of the d hard syncs of an MTP chain
(the last readback stays, the host needs the drafts to stage the window). Saves ~0.3-0.4 ms per extra chained
draft: 0 at d = 1, ~0.35 ms at d = 2, ~0.7 ms at d = 3, i.e. **0.6-1.3% of a round**. Costs: a GPU
implementation of the lexsort/top-k/top-p/splitmix64 Gumbel draw (it only has to match the host rule for
acceptance, not for exactness), device-side `mtp_pos_dev` advance inside a new graph, and it cannot serve `cN:P`
(the greedy MTP arm): its early stop is a host decision, and running the full chain wastes ~1.3 ms of GPU per step
that would have been skipped. Dropping the explicit `torch.cuda.synchronize()` after the verify forward (the
sample's `.cpu()` syncs anyway) would overlap ~0.05-0.1 ms of launches (0.1-0.2%). Real but small.

**(b) Tree verification with a second first-position candidate: not feasible here, and it would not pay.**
KDA is a recurrent state. `kda.chain` runs a window's rows in order from the committed state (conv window + delta
rule), `kda.replay_layers` rebuilds the state for a kept *prefix*, and `_conv_shift` takes rows `keep..keep+2`
of one chain. A sibling row needs the state after row 0, not after its left neighbour, so trees need a new kernel
in `kda.cu` (fork/restore state per branch, parent-indexed conv windows, path replay at commit) plus a
tree mask in DSA. `Buffers.parents()` exists but nothing uses it. That is kernel work outside this patch's
scope, and `kda.cu` is what the prefill work touches. Economics: the second candidate only adds a token when
the first draft misses and the second is right: 4-30% misses times maybe 30-50% right is 0.01-0.15 tokens a round.
Its row costs 4.7-10 ms, and 4.7 ms is worth ~0.28 tokens at 60 tok/s. Negative even with exact tree kernels.

**(c) A smaller head for drafts (frequency-ranked top N).** The head is about half an MTP step (~0.8 ms). A 32k
subset would cut it to ~0.17 ms: -1.9 ms on a 3-draft MTP round (3.5%), -0.6 ms on a DFlash2 block (1%). Every
reply token outside the subset kills the chain at that position. The first-32k-ids head missed 8.7% of tokens; a
frequency-ranked subset needs a token-frequency table built from a corpus through the tokenizer (for example
the model's own replies across the kit, tf and tool-call suites), which cannot be built or checked offline.
Expect +1-3% on the MTP-heavy (sampled) cells if misses stay under ~2%. **Best next step for the sampled
cells**, after a frequency pass on hardware.

**(d) Draft depth in `auto`: chosen.** Upstream `auto` caps DFlash2 at 5 drafts (`DepthPolicy(5, fixed=True,
confidence=0.3)`) although a block proposes 7 positions for the same 3.88 ms. On confident stretches the round
is capped at 6 tokens: structured text sits at 85.9 tok/s against a 95.1 ceiling. With 7 drafts the ceiling is
110.2. Fixed-depth DFlash2 throughput with per-draft acceptance r (verify + block + 0.3 ms host):

| r | 5 drafts | 6 drafts | 7 drafts |
| ---: | ---: | ---: | ---: |
| 0.80 | 58.5 | 58.2 | 57.3 |
| 0.90 | 74.3 | 76.9 | 78.5 |
| 0.95 | 84.0 | 88.9 | 92.7 |
| 0.99 | 92.8 | 100.1 | 106.4 |

The chain's probability gate (stop once the product of the picks' probabilities < 0.3) keeps weak chains short,
so drafts 6 and 7 enter only on stretches like the lower rows. The sampled `a:0.6:0.85` policy's MTP cap of 3 is
also below the optimum for r >= 0.85 (4 drafts: 57.6 -> 60.1 at r = 0.85). It is left alone: chained MTP drafts
decay faster with depth than a constant r, and 6.4 ms a draft makes misjudging costly without data.

## 4. The patch: `patches/0010-glm-auto-deep-dflash-drafts.patch`

- `engine.py`: `auto_f_most()` reads `GLM53_TF_AUTO_FDRAFTS` (default **7**, 1..7; 5 is upstream). `auto`'s DFlash2
  arm is `DepthPolicy(self.f_most, fixed=True, confidence=0.3)`. The value joins the settings both ranks compare
  at start, so ranks with different values refuse to start instead of drafting differently.
- `DFLASH_POLICY`/`EXL3_AUTO` (`fc5:0.3`, the no-MTP / BF16-EXL3 paths) are unchanged, so upstream's
  `test_cuda_cli.py` still holds. `GRAPH_ROWS` stays 1..6: the recipe's direct timings of eager 7- and 8-row
  windows (65.6, 69.2 ms) sit on the fitted line, so graphing them would buy little for more capture memory.
- `docker/compose.yaml`, `scripts/serve.sh` and `config/tensorfold.env.example` pass the knob to both ranks.

**Exactness.** The knob changes only how many DFlash2 candidates a round sends to verification. Every emitted
token is still `sample_rows` of the verify window's row at its absolute position (argmax of logit/T plus
Gumbel(seed, pos, id) over top-k/top-p, ties by id). A draft is kept only while it equals that sample, and the
first mismatch's row supplies the token (`auto_decode`). Windows of 7 and 8 rows are within `MAX_ROWS = 8`, the
width upstream sizes every buffer for. The recipe validated 2- to 8-row windows bit-identical to serial steps
(87/87 rows). Row-invariant kernels make each row's bits independent of the window width. `commit` replays exactly
the kept prefix. Both ranks compute the same drafts from the same gathered candidates, and the start-up check
guarantees the same cap. So replies are byte-identical to serial decoding for any value 1..7. Only speed moves.

**Expected gain** (to be confirmed on hardware; drafts cannot change output, so the only risk is speed):

- structured (greedy): 85.9 -> **~99 tok/s (+15%)** if it keeps its 90% share of the ceiling (95.1 -> 110.2).
- code greedy (61.0; about half its rounds on DFlash2): +0-5%, from rounds that hit the cap of 5 at product >= 0.3.
- chat greedy, hashmap, essay: about +-1%. The 6th and 7th drafts are admitted by the same rule as the 5th at the same
  4.7 ms price. The rule would lose only if DFlash2's probabilities were overconfident at deep positions.
- sampled cells: unchanged (sampled `auto` drafts with MTP only).
- `DrafterChoice` already prices 7- and 8-row windows from the calibrated costs, so it stays consistent.

**Hardware A/B for the GPU owner** (no rebuild needed for the first line; per-request specs exist upstream):

```bash
# DFlash2-only depth, per request: kit suite with model@fc5:0.3 vs model@fc7:0.3
python3 bench/glmbench.py --base http://127.0.0.1:8080 --model 'GLM-5.3-Flash-Uncensored@fc5:0.3' --suites kit,tf
python3 bench/glmbench.py --base http://127.0.0.1:8080 --model 'GLM-5.3-Flash-Uncensored@fc7:0.3' --suites kit,tf
# the default: restart both ranks with GLM53_TF_AUTO_FDRAFTS=5 (upstream) vs 7, then --suites tf,kit,exact
```

Keep 7 as the default if kit/structured gains and the greedy chat and prose cells hold within noise. Otherwise
set 6, or 5 for upstream behaviour.

**Tests** (in the image, one GPU):

```bash
PYTHONPATH=/src/TensorFold/tests/cuda pytest -q tests/cuda/test_decode_patches.py
```

`tests/cuda/test_decode_patches.py` checks: the knob's default and bounds; every policy (`auto`, `auto:1:1:0`,
`auto:1:2:0`, `f7`, `fc7:0.3`, `fc5:0.3`, `7`, `c3:0.35`, `a:0.6:0.85`, `2`) against serial decoding, sampled and
greedy, with 8-row windows seen for `f7`/`7`; `auto` at knob 5 (windows <= 6 rows) and at 7 (<= 8), both equal
to serial and to each other; 7- and 8-row windows against serial steps; resume after deep rounds equal to a
fresh prefill and to serial.

## 5. Next, in order of expected gain

1. Frequency-ranked draft head (c): +1-3% on sampled cells, needs a frequency pass on hardware.
2. One graph for fixed-depth MTP chains with GPU picking (a): +0.6-1.3% on `a:` rounds.
3. Drop the timing-only `torch.cuda.synchronize()` after verify forwards: ~+0.2%, trivial, same bits.
4. The sampled `a` policy: a 4th draft when the running acceptance is >= ~0.92, once measured.
