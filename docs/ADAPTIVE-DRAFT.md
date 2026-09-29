# Adaptive per-slot drafting in batched decode (patches/0340) — evaluation, model, GPU plan

Offline work, 2026-09-28. No GPU was used. The inputs are the concurrent-stream logs of W1 and W5
(`results/W1/conc.json`, `results/W5/conc-{off,nographs,b48}.json`: 11 four-stream runs, with per stream the drafter
of every round, the rows it kept, and verify / draft ms). Simulator: `bench/draftsim.py`. Its outputs:
`results/sim0340/`. Patch: `patches/0340-glm-batch-adapt.patch` (`GLM53_TF_BATCH_ADAPT`, off by default). Tests:
`tests/cuda/test_adapt_patches.py`.

## Summary

- **Draft length is already adaptive per slot.** 0071's cost-derived depths (`depth.py`) choose each draft from the
  drafter's own probability of that draft. A per-slot, per-position calibration corrects that probability, and a
  draft joins the window only when its expected tokens beat the shared rounds' aggregate rate × its marginal row
  cost (0120's `BatchDepth`, 0200's `GLM53_TF_BATCH_ROW_MS`). A per-slot EMA of acceptance would carry less
  information than this, so it was not built as a separate thing.
- **Simulated headroom for draft policy is small, and most of it is out of reach.** The simulator runs the engine's
  own decision code (`depth`, `batchplan`, 0340's `adapt`) round by round on stochastic streams fitted to the
  recorded per-stream, per-drafter acceptance. Results for the 4-stream aggregate against today's policy, over 11
  runs × 12 seeds:

| policy | concurrent bench (finite) | steady serving (60 s) | verdict |
| --- | ---: | ---: | --- |
| today (`ROW_MS=6.5`, `DrafterChoice`, cost depths) | 81.8 tok/s (measured 74.1) | 96.8 | |
| `GLM53_TF_BATCH_ROW_MS` unset (the table's own 5.75 ms slope) | +0.5% (-0.7 .. +1.3) | +1.0% | noise; config-only A/B |
| `ROW_MS=8` | -1.4% | -2.6% | no |
| **0340: shared-round drafter chosen at the margin (`ADAPT=1`)** | -0.4% (-2.2 .. +0.8) | -0.3% | ≈ 0 |
| 0340 with probes every 16 | -0.4% | 0.0% | ≈ 0 |
| 0340 + serial rounds (skip drafting when it does not pay, `ADAPT_SERIAL=1`) | **-6.5%** | **-4.3%** | harmful |
| 0340 + serial rounds only when clearly worse (margin 0.5 tokens) | -0.6% | -0.3% | ≈ 0 (almost never serial) |
| bound: an oracle that drafts exactly the drafts that will be kept | +20.8% | +14.7% | not reachable |

  The findings hold under every sensitivity variant (`results/sim0340/`): flat vs informative drafter
  probabilities, row cost with or without expert sharing, per-slot overhead of 4 or 7 ms, and the raw (censored)
  acceptance. Across all of them, 0340 without serial rounds lands between -0.4% and +1.0%, and serial rounds cost
  4-7%.
- **What the patch is.** 0340 implements the best of the evaluated ideas: per-slot drafter choice in shared rounds,
  priced at the round's margin, plus an optional serial arm. Replies stay exact by construction and the code is
  tested. It is off by default. The expected gain is about 0 (within ±1%), so it is **not recommended for
  adoption**. It only earns a GPU slot as a cheap side-A/B in a window that already runs the concurrent bench
  (plan below).
- **Where 4-stream throughput actually is** (from the fitted round model, same simulator):
  - A decoding slot costs ~7 ms of verify time a round on top of its rows. This is inferred from the recorded verify
    ms per round and is not measured directly (see the plan). Halving it gives **+6%** aggregate (81.8 → 86.8
    simulated).
  - Batching the per-slot DFlash2 block passes, the way 0200 batched MTP, gives **+2%**.
  - The oracle's +15-21% measures what better draft *prediction* would be worth, meaning drafters that are right
    more often or know when they are wrong. No depth or drafter-choice rule can buy it: the rules are already close
    to the best a policy can do with the information it has.

## 1. What decides a slot's drafts today

Each round (`batch.Batcher._verify`), for every decoding slot:

1. **Drafter.** `decode.DrafterChoice` (auto, greedy with DFlash2 loaded) takes the arm with the higher committed
   tokens per *lone-model* ms over its last 6 rounds of that arm. Every 8 rounds it runs one round of the other
   arm. Sampled `auto` requests use MTP only.
2. **Depth.** `BatchDepth` (a `depth.DepthOptimizer`) prices rows with `RoundCosts.table(others)`: the slot's window
   costs past the other slots' rows, at no less than `ROW_MS` ms a row. It uses the shared rate λ (`RoundCosts.rate`:
   committed tokens per model ms of the last 16 shared rounds).
   - MTP: draft j joins while q1..qj ≥ λ × its row's marginal ms, and the chain continues while the next draft's
     expected token pays for an MTP step and its row.
   - DFlash2: k = argmax E(k) − λ·verify[k].
   - q comes from the drafter's own probability × a per-position correction (`depth.Calibration`).
3. The first draft of an MTP or DFlash2 round is always verified.

So "per-slot adaptive length maximizing accepted tokens per ms of round time" is the existing rule: the renewal
criterion (a row pays if its expected tokens / its ms > the aggregate rate). It uses per-draft confidence, not just
an average. The recorded tokens a round of 2.0-7.0 are the streams' own acceptance (a repetitive "sequence" prompt
keeps ~7.5 a DFlash2 round, chat prompts ~2.3), not mis-sized windows.

## 2. Round cost model

Inputs:
- the verify table V(R) = 31/39/45/51/56/62/68/74 ms for 1..8 rows (two Sparks, docs/PATCHES.md 0200);
- W5's ~6-7 ms a real row;
- the recorded per-stream verify ms a round: 94-106 ms for slow streams, 125-144 ms for streams alive while the fast
  one runs;
- draft ms a round per slot: 3.9-5.3 ms (median ~4.5).

A true-cost model for the simulator (`draftsim.py`, not the engine's pricing):

    round = V(R_total) [past 8 rows: row_ms x (1 - 8/288)^(R - 8) a row]  +  slot_ms x (slots - 1)
          + per slot: DFlash2 block 3.9 + 0.6 host  |  batched MTP: 2.3 + 1.9 a chained step + 0.5 a slot and step

- **Expert sharing.** A row reads 8 of 288 routed experts a layer. With uniform routing, a row joining N rows
  finds a fraction (1 − 8/288)^N of its experts not loaded yet: 0.80 at N = 8, 0.71 at 12, 0.57 at 20. Real routing
  is skewed, so sharing is somewhat higher. So yes, the marginal row cost falls as a round fills: from ~6 ms toward
  ~4 ms at 20 rows. Within one sequence's 8-row window the table's slope (~6 ms) already includes this.
- **Fit.** After de-censoring the acceptance (section 3), a grid fit of the recorded per-stream verify ms a round
  (rmse ~4 ms) puts the true row cost at 5.5-7 ms and the per-slot overhead at 5-8 ms. The two trade off along a
  ridge: the data cannot separate them. Defaults: row 6.0 ms with sharing, slot 7.0 ms.
  - A slot's fixed share is plausible. Every slot runs its own KDA chain, attention and indexer launches for 45
    layers, a KDA replay in `commit` for a partly kept window, and its own accept / commit host path. But this is an
    inference; the GPU plan measures it.
- **Absolute error.** The simulator is ~10% optimistic in absolute terms (81.8 simulated vs 74.1 measured, mean of
  the 11 runs). It does not model prefill staggering, eager / capture rounds, the per-round plan exchange, or 0280's
  padded rows in the b48 runs. Comparisons between policies are what it is for.

What this means for the ideas:
- A slot's serial round is not cheap: ~7 ms (slot) + ~6 ms (row).
- A drafted round adds a drafter pass (~4.5 ms) plus ~6 ms a further row.
- Slow chat streams keep their first draft ~80-85% of the time (de-censored), so their drafts pay at the shared
  rate λ ≈ 0.07 tokens/ms.

## 3. The simulator (`bench/draftsim.py`)

- **Streams.** Per recorded stream and drafter: the survival P(L ≥ j), j = 1..7, of L = leading drafts kept, from
  the recorded keeps.
  - Rounds cut short by the chosen depth count as misses, so the raw survival under-reads acceptance.
  - `decensor` fits a per-stream, per-drafter miss scale (a_j' = 1 − (1 − a_j)·β) until the simulated *baseline*
    keeps what the recorded rounds kept (mean keep per round of each arm). The fitted β are 0.1-0.8.
  - After the fit, the simulated tokens a round match the recorded ones per stream: e.g. W1 rep 0 recorded 4.15 /
    2.41 / 2.50 / 7.00, simulated 4.12 / 2.52 / 2.84 / 6.77. Verify ms a round also match: recorded 125 / 104 / 106
    / 139, simulated 126 / 103 / 110 / 140.
- **Drafter probabilities.** Each draft's chance to be kept, π_j, is drawn from Beta(mean a_j, concentration 3). The
  draft is kept with probability π_j, and the drafter reports π_j (`--signal info`: calibrated and informative), or
  every draft reports a_j (`flat`).
- **Policies.** The engine's classes, imported from a patched tree: `depth.DepthOptimizer` as `BatchDepth`,
  `batchplan.RoundCosts`, and `adapt.SlotChoice`. Also a line-for-line copy of `decode.DrafterChoice` (decode.py
  imports torch); a test checks the copy against the real one.
- **Metrics.**
  - Finite: 4 streams × 512 tokens, aggregate = tokens / makespan. This is the concurrent bench, including its tail
    where 1-3 streams remain.
  - Steady: each ended stream is replaced by a new request of the same profile; aggregate over 60 s.
- **Oracle.** It drafts exactly min(L, 7) (at least 1). This is an upper bound for any depth rule with today's
  drafters.

Commands (repo root):

    python3 bench/draftsim.py --src <patched tree>/src results/W1/conc.json results/W5/conc-{off,nographs,b48}.json \
        --seeds 12 --variants base,base:row=0,adapt:serial=0,adapt,oracle [--steady 60] [--signal flat] [--no-share] \
        [--raw] [--true-row 7 --slot-ms 4]

Limitations:
- Rounds are independent per stream (no streaks).
- Both drafters' L are drawn independently.
- Lookup rounds are counted as DFlash2.
- Greedy only: the recorded runs are greedy. Sampled `auto` requests draft MTP only, so there is no drafter choice
  for them.

## 4. The ideas, one by one

1. **Per-slot adaptive draft length from acceptance (EMA), maximizing accepted tokens per ms of round time.** This
   exists (section 1) in a stronger per-draft form. The pricing parameter that remains is `ROW_MS`. Unset (5.75 ms
   slope) simulates +0.5% finite / +1.0% steady (+1.4-1.5% with flat probabilities); 8 simulates -1.4 / -2.6%. The
   true marginal row, ~5-6 ms with expert sharing, sits just below 6.5. A config-only A/B, noise-level.
2. **Shared expert-load awareness** (prefer drafts whose rows hit already-loaded experts). Not buildable cheaply,
   and it would not pay:
   - A draft row's experts are chosen by the router inside the verify forward, from a hidden state that depends on
     the window's earlier rows. Nothing before the forward knows them; the drafter's own hidden states route
     differently.
   - The saving is bounded by the sharing fraction: at 12-20 rows, 20-45% of one row's expert reads.
   - It would only change which of the already-chosen rows run. Depth is chosen by expected tokens, and a cheaper
     row changes the decision only at the margin.
   - The modelled effect of sharing on today's rule is inside the `ROW_MS` sweep above (≤ 1%).
3. **Drafter choice per slot (MTP vs DFlash2) by acceptance.** Built: `adapt.SlotChoice`.
   - The rule: surplus over a serial round = (keep − 1) − λ × (rows 2..R on the slot's table + drafter ms). It is
     averaged over the arm's last 8 shared rounds, re-priced at the current λ, and each drafting arm is probed every
     8 shared rounds.
   - It simulates at about 0. The slow streams' two drafters are worth the same: recorded MTP 1.8-2.6 vs DFlash2
     2.1-3.1 tokens a round, with MTP rounds shorter. The fast streams already run DFlash2 ~87% of the time; the 13%
     MTP probes cost them little.
4. **Skip drafting for slots with persistently low acceptance.** Built as 0340's serial arm (`ADAPT_SERIAL=1`). It
   loses 4-7%:
   - A serial round still pays the slot's fixed cost and one row (~13 ms).
   - Skipping saves only the drafter pass and the draft rows, while the slow streams' first draft is kept ~80-85% of
     the time.
   - In the finite bench it also slows exactly the streams that set the makespan: their per-stream rate drops 3-8%.
     Their users would see that too.
   - With a 0.5-token margin it almost never fires and is neutral.

## 5. Patch 0340 (`GLM53_TF_BATCH_ADAPT`, off by default)

`adapt.py` (new, no torch) plus small hooks in `batch.py`.

- **Knobs.** Load-time; both ranks must agree, and they are compared with the other batch settings at start.
  - `GLM53_TF_BATCH_ADAPT=0|1` (0).
  - `GLM53_TF_BATCH_ADAPT_SERIAL=0|1` (0): a serial round is a choice too.
  - `GLM53_TF_BATCH_ADAPT_EVERY=N` (8): probe interval, in shared rounds.
  - `GLM53_TF_BATCH_ADAPT_WINDOW=N` (8): shared rounds in each arm's surplus.
- **Scope.** Requests with cost-derived depths only (`o`, `om`, `of`, and `auto` with `GLM53_TF_DEPTH=cost` as in
  production). The arms are those the request may use: "mf" under `DrafterChoice`, else its fixed arm. Lookup rounds
  are untouched: the lookup's own gate still runs first.
- **Alone** (one decoding slot, `RoundCosts.rate()` None): the slot's own `DrafterChoice` decides exactly as
  without the patch. So single-stream decoding is unchanged: same code path, same picks.
- **Shared.** `SlotChoice.pick(λ)`: every arm once, then the highest mean surplus (serial = 0, only if allowed),
  with the stalest arm probed after EVERY rounds. A serial round is a window of the pending token alone (arm "s"): no
  MTP chain, no DFlash2 block, and the drafters' backlogs grow as in any round of the other arm. `BatchDepth` records
  it as a one-row window; lookup and the base choice never see it.
- **Rank consistency.** Every input is the same on both ranks: committed keeps, the load-time cost table, the
  round's windows, and λ from `RoundCosts` (model ms, no clock).
- **Stats.** Per request, `adapt: {serial, probes}`; `drafters` shows "s" for serial rounds.

**Exactness** (why any per-slot window length, including none, keeps replies byte-identical):
- A slot's token at position p is `choose_rows` / greedy argmax of that row's own logits, keyed by (seed, p). This
  holds in `sample_multi` for every slot of the round.
- The logits row does not depend on the window's length or on other slots' rows. The kernels are row-independent,
  which is what drafted == serial and 0200's padded rows already rely on; they are GPU-tested bit for bit.
- The accept rule keeps rows up to the first draft that differs from the sampled token. `commit(R, keep)` keeps
  exactly those (KDA replay for keep < R).
- Drafts and their count therefore only decide how many positions a round confirms, never which token a position
  gets.
- The CPU tests exercise this on the real `Stepper` / `Batcher._verify` with a hostile hash model. They include a
  mode that forces MTP / DFlash2 / serial at random per slot and round, with requests arriving so that rounds are
  alone and shared. Every reply equals serial decoding, and a follower batcher replaying rank 0's plans decides the
  same arms and keeps.

## 6. GPU test plan (one window, ~50 min, prod down)

Image: through 0340 (plus whatever else the window carries). Loads from `config/prod.env` with overrides, as in W5
(`results/W5/load.sh`).

1. **Tests** (the worker node, prod may stay up if memory allows, as in W5): `tests/cuda/test_adapt_patches.py` (4 GPU cases:
   4 requests `o` / `of` / `om` with 0340 + serial on, chosen and forced arms, greedy and sampled == serial),
   `test_batch_parallel_patches.py`, `test_depth_patches.py`. Gate: all pass.
2. **Measure the round model first** (production image and knobs, ~10 min). `bench/multiturn.py --modes concurrent
   --streams 1,2,4 --reps 3 --long-tokens 512`, with `--extra '{"draft": false}'` (serial: exactly 1 row a slot),
   then without.
   - From the serial runs' `ms/round`: V(1) alone and 2 / 4 slots × 1 row, which separates the per-slot fixed cost
     from the row cost. Expected: ~31 / ~44 / ~70 ms if the fit's slot 7 + row 6 holds; ~31 / ~38 / ~50 if rows and
     slots are cheaper.
   - This decides whether the per-slot cost (+6% if halved) is the next target.
   - Record `draft_ms` / `verify_ms` per round.
3. **A/B, 3 loads × (concurrent `--streams 1,4 --reps 5 --long-tokens 512`, `batchexact`, `exact`), ~10 min each:**
   - A: production (`ROW_MS=6.5`);
   - B: `GLM53_TF_BATCH_ADAPT=1` (serial off);
   - C: `GLM53_TF_BATCH_ROW_MS` unset (config-only; the simulator's best, +0.5-1.5%).

   Serial rounds (`ADAPT_SERIAL=1`) are not worth a load: the simulator says -4 to -7%. Record per stream
   `drafters`, `keeps` and `adapt` (B) so the simulator can be re-fitted.
4. **Gates.** Adopt only if:
   - the 4-stream aggregate mean is ≥ +5% over 5 reps (the rep-to-rep spread is 70-81);
   - single stream is unchanged within noise (B: identical code path when alone, so replies and rates should match
     A);
   - `batchexact` is 4/4 and `exact` is 10/10.

   Expected: neither B nor C passes, so production stays unchanged.
5. **Memory.** Nothing allocates (host-side decisions only); a MemAvailable check after the runs, as usual.

## 7. Levers ranked for 4-stream throughput (for the next windows)

| lever | simulated aggregate | note |
| --- | ---: | --- |
| per-slot fixed cost ~7 ms → ~3.5 ms (fused per-slot KDA / attention / commit across slots: 0200's "one launch per layer", a batched KDA replay) | +6% | measure first (plan step 2); 0200 put the launch part at 2-4% |
| batched DFlash2 blocks (one drafter pass for all DFlash2 slots, as 0200's `BATCH_MTP` for MTP) | +2% | ~1-1.5 d; drafts only, exact by construction |
| `ROW_MS` unset | +0.5-1.5% | config only |
| 0340 drafter choice at the margin | ≈ 0 | this patch |
| better drafts (oracle gap) | up to +15-21% | drafter quality / self-knowledge; not a scheduling problem |
