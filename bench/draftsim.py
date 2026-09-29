"""Offline simulator of batched decode rounds (patches/0340, docs/ADAPTIVE-DRAFT.md). No GPU, no torch.

It replays the concurrent-stream runs of results/W1 and results/W5 (``multiturn.py --modes concurrent``: per stream,
the drafter of every round and the rows it kept) as stochastic streams and runs the ENGINE'S OWN decision code on
them, round by round: ``depth.DepthOptimizer`` as ``batch.BatchDepth`` (batch-aware cost depths, the shared rate of
``batchplan.RoundCosts`` with ``GLM53_TF_BATCH_ROW_MS``), a copy of ``decode.DrafterChoice`` (decode.py imports torch;
the copy is line for line, ``tests/cuda/test_adapt_patches.py`` checks it against the real one where torch exists)
and, with ``--adapt``, 0340's ``adapt.SlotChoice``. Time comes from a separate "true" cost model (below), so a policy
that prices rounds wrongly pays for it.

Streams. For each recorded stream and drafter (MTP "m", DFlash2 "f"): the survival P(L >= j), j = 1..7, of L = the
number of leading drafts a round would keep, taken from the recorded keeps (keep - 1 drafts kept). A recorded round
that stopped short because its depth was cut, not because a draft missed, counts as a miss: the survival is a LOWER
bound on what deeper drafting would keep (conservative for policies that draft deeper, generous for shallower
ones). Each simulated round draws its L from the stream's survival for the arm it runs, independently.

The drafters' own probabilities (what cost depths read, draft by draft): ``--signal info`` draws each draft's
probability from Beta(5, 1.5) when the draft would be kept and Beta(1.5, 2) when not (informative, like a real
drafter's confidence), ``flat`` gives every draft 0.7 (the optimizer then learns only the stationary acceptance).

True cost (ms, two Sparks; docs/RESULTS.md W1 / W5, docs/PATCHES.md 0200): a forward of R rows over n slots
= V(R) (31/39/45/51/56/62/68/74 for 1..8, then ``--true-row`` ms a row, default 6.5) + ``--slot-ms`` (1.0) per slot
past the first (per-slot sampling, accept, commit, KDA / attention launches); a DFlash2 block 3.9 ms + 0.6 host a
slot; MTP: alone 2.04 ms + 1.68 a chained step, batched (``GLM53_TF_BATCH_MTP``) one pass of 2.3 ms + 1.9 a chained
step for all MTP slots together, + 0.5 ms host a slot and step. Calibration check: the baseline's simulated aggregate
and tokens a round next to the recorded ones (printed).

Usage (from the repo root, with a patched tree):

    python3 bench/draftsim.py --src /tmp/tf/src results/W1/conc.json results/W5/conc-off.json --seeds 20

patches/0380 (deeper verify windows, docs/DEEP-VERIFY.md): variants take ``rows=N`` (the window cap,
GLM53_TF_MAX_DRAFT_ROWS; the engine's depth code then may verify up to N - 1 drafts), ``block=N`` (the DFlash2 block,
GLM53_TF_DFLASH_BLOCK: picks for N - 1 positions a pass, ``--block-ms`` a pass past 8) and ``tail=F`` (a draft's chance
to be kept at positions past the 7 recorded ones: the stream's conditional acceptance at position 7 times F per
further position; 1.0 = as sure as the 7th, 0 = never). A lone window past 8 rows costs ``--deep-row`` ms a row (the
engine's calibrated table is taken to be the truth there); a shared round past 8 rows as before. ``--alone``
simulates every recorded stream on its own and reports the gain per stream class (prose / code-like / repetitive, by
recorded tokens a round). The tree's modules are imported with GLM53_TF_MAX_DRAFT_ROWS=16 (unless set), so the
per-position calibration has room for 15 positions; 8-row variants behave exactly as without it.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics
import sys
from pathlib import Path

V = [31.0, 39.0, 45.0, 51.0, 56.0, 62.0, 68.0, 74.0]
COSTS = {"verify": list(V), "mtp": 2.04, "mtp_step": 1.68, "mtp_row": 0.1, "block": 3.88, "taps_row": 0.05}
BACKLOG = 32
KAPPA = 3.0               # the drafters' per-draft acceptance spread (``--signal info``)
DEEP_ROW = 5.5            # patches/0380: ms a lone window's row past 8 (docs/DEEP-VERIFY.md: 4.7-6.5)
BLOCK_MS = 3.9            # a DFlash2 block pass (8 rows)
BLOCK16_MS = 4.4          # a 16-row block pass (GLM53_TF_DFLASH_BLOCK=16; an estimate: +8 rows through 5 small layers)


def costs_for(rows=8, deep_row=DEEP_ROW):
    """The engine's cost table for a window cap of ``rows``: the 8-row table, then ``deep_row`` ms a row (what the
    load-time calibration of patches/0380 measures past 8 rows on consecutive tokens)."""

    c = dict(COSTS)
    c["verify"] = list(V) + [V[-1] + deep_row * r for r in range(1, max(rows, 8) - len(V) + 1)]
    return c


def load(paths):
    """-> [(label, measured aggregate tok/s, [stream: {"surv": {arm: [P(L>=1..7)]}, "tpr", "rounds", "mix"}])]"""

    runs = []
    for p in paths:
        d = json.load(open(p))
        for c in d.get("concurrent", []):
            streams = []
            for s, arms in zip(c["stats"], c["drafters"]):
                by = collections.defaultdict(list)
                for a, k in zip(arms, s["keeps"]):
                    by["f" if a == "l" else a].append(k)
                surv, rkeep = {}, {}
                for a in "mf":
                    ks = by.get(a) or by.get("f" if a == "m" else "m")
                    surv[a] = [sum(1 for k in ks if k - 1 >= j) / len(ks) for j in range(1, 8)]
                    rkeep[a] = (sum(ks) / len(ks), len(by.get(a, ())))
                tpr = (sum(s["keeps"]) - 1) / len(s["keeps"])
                streams.append(dict(raw=dict(surv), beta={"m": 1.0, "f": 1.0}, rkeep=rkeep, vms=s["round_kinds"]["verify_ms"] / len(s["keeps"]), surv=surv, tpr=tpr, rounds=len(s["keeps"]),
                                    mix=arms.count("m") / max(len(arms), 1), tokens=sum(s["keeps"]), cls=stream_class(tpr)))
            runs.append((f"{Path(p).parent.name}/{Path(p).stem} {c['streams']}x rep{c['rep']}", c["aggregate_tps"],
                         streams))
    return runs


def stream_class(tpr):
    """patches/0380: prose / chat (< 3.5 tokens a round), code-like (3.5-6), repetitive (> 6), as docs/RESEARCH-NIGHT.md
    §1 splits the recorded streams."""

    return "repetitive" if tpr > 6 else "code" if tpr > 3.5 else "prose"


def set_beta(stream, arm, beta):
    """Scale the arm's miss chances: a_j' = 1 - (1 - a_j) x beta at every position (beta < 1: more kept)."""

    stream["beta"][arm] = beta
    prev, out, run = 1.0, [], 1.0
    for s in stream["raw"][arm]:
        a = s / prev if prev > 0 else 0.0
        prev = s
        run *= min(max(1 - (1 - a) * beta, 0.0), 0.995)
        out.append(run)
    stream["surv"][arm] = out


def decensor(mods, runs, sim_kw, *, iters=8, seeds=4, log=None):
    """Fit each stream's per-arm miss scale so the BASELINE policy, simulated, keeps what the recorded rounds kept
    (mean keep per round of each arm): the recorded survival is censored by the depths the engine chose (a round cut
    short counts as a miss), so the raw survival under-reads acceptance."""

    for label, measured, streams in runs:
        for it in range(iters):
            res = [Sim(mods, streams, parse_variant("base"), seed=1000 + sd, **sim_kw).run() for sd in range(seeds)]
            for j, st in enumerate(streams):
                for a in "mf":
                    rec, n = st["rkeep"][a]
                    got = [r["karm"][j][a] for r in res if r["karm"][j][a] is not None]
                    if n < 5 or not got:
                        continue
                    sim = statistics.mean(got)
                    ratio = max(sim - 1, 0.05) / max(rec - 1, 0.05)
                    set_beta(st, a, min(max(st["beta"][a] * ratio ** 1.5, 0.02), 2.0))
        if log:
            log(f"  {label}: miss scale " + ", ".join(f"m {st['beta']['m']:.2f} f {st['beta']['f']:.2f}"
                                                     for st in streams))


# -- decode.DrafterChoice, line for line (decode.py imports torch) -------------------------------------------------
def _other(arm):
    return "m" if arm == "f" else "f"


class DrafterChoice:
    def __init__(self, costs, *, first, explore=2, every=8, margin=0.03, window=6):
        self.costs = costs
        self.first = first
        self.explore, self.every, self.margin, self.window = explore, every, margin, window
        self.rounds = []
        self.choice = first
        self.run = 0

    def cost(self, arm, rows, steps, backlog):
        c = self.costs
        verify = c["verify"][min(rows, len(c["verify"])) - 1]
        if arm == "m":
            return verify + c["mtp"] + c["mtp_step"] * max(steps - 1, 0) + c["mtp_row"] * max(backlog - 1, 0)
        return verify + c["block"] + c["taps_row"] * backlog

    def rate(self, arm):
        rs = [r for r in self.rounds if r[0] == arm][-self.window:]
        return sum(r[1] for r in rs) / sum(r[2] for r in rs) if rs else None

    def pick(self):
        n = len(self.rounds)
        if n < self.explore:
            return self.first
        if n < 2 * self.explore:
            return _other(self.first)
        cur = self.choice
        rc, ro = self.rate(cur), self.rate(_other(cur))
        need = 1.0 if n == 2 * self.explore else 1.0 + self.margin
        if ro is not None and (rc is None or ro > rc * need):
            self.choice = cur = _other(cur)
            self.run = 0
        if self.every and self.run >= self.every:
            self.run = 0
            return _other(cur)
        self.run += 1
        return cur

    def record(self, arm, rows, steps, backlog, keep):
        self.rounds.append((arm, keep, self.cost(arm, rows, steps, backlog)))


# -- the simulation --------------------------------------------------------------------------------------------------
class Sim:
    """One simulated run. ``spec``: the policy (``parse_variant``). ``horizon_ms`` None: every stream runs its
    ``tokens`` and the run ends with the last (the concurrent bench: aggregate = tokens / makespan); else a stream
    that ends is replaced at once by a new request of the same profile (fresh choice / optimizer state) and the
    aggregate is the tokens committed within the horizon (steady serving)."""

    def __init__(self, mods, streams, spec, *, tokens, seed, signal, true_row, slot_ms, horizon_ms=None, share=False,
                 deep_row=DEEP_ROW, block_ms=BLOCK16_MS):
        self.depth, self.batchplan, self.adapt_mod = mods
        self.rng = random.Random(seed)
        self.streams, self.spec = streams, spec
        self.tokens, self.signal, self.horizon = tokens, signal, horizon_ms
        self.true_row, self.slot_ms, self.batch_mtp = true_row, slot_ms, spec.get("batch_mtp", True)
        self.share = share
        # patches/0380: the window cap, the DFlash2 block, the acceptance past the recorded positions
        self.rows, self.block = int(spec.get("rows", 8)), int(spec.get("block", 8))
        self.tail, self.deep_row = float(spec.get("tail", 1.0)), float(deep_row)
        self.block_ms = BLOCK_MS if self.block <= 8 else float(block_ms)
        self.most = {"m": self.rows - 1, "f": min(self.block, self.rows) - 1}
        self.costs = costs_for(self.rows, self.deep_row)
        self.rc = self.batchplan.RoundCosts(self.costs, spec.get("row_ms", 6.5))
        rc, depth = self.rc, self.depth

        class BatchDepth(depth.DepthOptimizer):
            def rate(self_):
                shared = rc.rate()
                return shared if shared is not None else depth.DepthOptimizer.rate(self_)

        self.BatchDepth = BatchDepth
        self.slots = [self._new(j) for j in range(len(streams))]
        self.finished = []

    def _new(self, j):
        s = self.streams[j]
        base = DrafterChoice(self.costs, first="f")             # greedy: DFlash2 first (``Stepper``)
        sc = None
        if self.spec.get("adapt"):
            sc = self.adapt_mod.SlotChoice(self.costs, "mf", base, every=self.spec.get("every", 8),
                                           window=self.spec.get("window", 8), serial=self.spec.get("serial", True),
                                           margin=self.spec.get("margin", 0) / 10)
        opt = self.BatchDepth(self.costs, most_m=self.most["m"], most_f=self.most["f"])
        return dict(prof=j, surv=s["surv"], out=1, opt=opt, choice=base,
                    sc=sc, m_back=1, n_f=0, arms=collections.Counter(), rounds=0, t0=0.0, done_t=None, rows=1,
                    vms=0.0, karm={"m": [0, 0], "f": [0, 0]})

    def _draw(self, sl, arm):
        """-> (L, [the drafter's probability of each of its picks]: 7, or the arm's cap with patches/0380). Draft j's
        chance to be kept (given drafts before it were) is pi_j ~ Beta with the stream's conditional acceptance a_j as
        its mean and concentration ``KAPPA`` (``info``: a calibrated, informative drafter reports pi_j) or exactly a_j
        (``flat``). Past the 7 recorded positions a_j = a_7 x tail^(j - 7)."""

        surv = sl["surv"][arm]
        conds, prev = [], 1.0
        for s in surv:
            conds.append(s / prev if prev > 0 else 0.0)
            prev = s
        for j in range(len(conds), self.most[arm]):
            conds.append(conds[-1] * self.tail if conds else 0.0)
        L, probs, alive = 0, [], True
        for c in conds[:max(self.most[arm], 7)]:
            a = min(max(c, 0.02), 0.995)
            if self.signal == "flat":
                pi = a
            else:
                pi = min(max(self.rng.betavariate(a * KAPPA, (1 - a) * KAPPA), 1e-4), 1 - 1e-4)
            probs.append(pi)
            if alive and self.rng.random() < pi:
                L += 1
            else:
                alive = False
        return L, probs

    def _verify_true(self, rows, n):
        if rows <= len(V):
            ms = V[rows - 1]
        elif n == 1:              # patches/0380: a lone deep window (consecutive tokens of one sequence)
            ms = V[-1] + self.deep_row * (rows - len(V))
        elif self.share:          # a row past the table reads the experts no earlier row of the round read
            ms = V[-1] + sum(self.true_row * (1 - 8 / 288) ** (r - len(V)) for r in range(len(V), rows))
        else:
            ms = V[-1] + self.true_row * (rows - len(V))
        return ms + self.slot_ms * max(n - 1, 0)

    def _propose(self, sl, others, rate):
        opt = sl["opt"]
        opt.verify = self.rc.table(others)
        room = self.tokens - sl["out"]
        arm = sl["sc"].pick(rate) if sl["sc"] is not None else sl["choice"].pick()
        fixed = self.spec.get("fixed")
        L, probs = self._draw(sl, arm) if arm in "mf" else (0, [])
        steps = backlog = 0
        if arm == "m":
            backlog, sl["m_back"] = sl["m_back"], 0
        elif arm == "f":
            backlog, sl["n_f"] = sl["n_f"], 0
        if arm in "mf" and self.spec.get("oracle"):      # upper bound: the depth that keeps every draft
            k = max(1, min(L, room, self.most[arm]))
            steps = k if arm == "m" else 0
        elif arm in "mf" and fixed:                      # a fixed cap on top of the cost depths
            k = self._cost_depth(sl, arm, probs, min(room, fixed))
            steps = sl.pop("_steps", 0)
        elif arm in "mf":
            k = self._cost_depth(sl, arm, probs, room)
            steps = sl.pop("_steps", 0)
        else:
            k = 0
        return arm, k, L, steps, backlog

    def _cost_depth(self, sl, arm, probs, room):
        opt = sl["opt"]
        if arm == "m":
            opt.mtp_begin()
            k = chained = 0
            count = max(1, min(opt.most_m, room))
            for j in range(count):
                take, more = opt.mtp_next(j, probs[j], count)
                if not take:
                    break
                k += 1
                if not more:
                    break
                chained += 1
            sl["_steps"] = 1 + chained
            return k
        return opt.f_depth(probs[:max(1, min(opt.most_f, room))])

    def run(self):
        t = 0.0
        while True:
            live = [i for i, sl in enumerate(self.slots) if sl["out"] < self.tokens]
            if not live or (self.horizon is not None and t >= self.horizon):
                break
            shared = len(live) > 1
            rate = self.rc.rate() if shared else None
            plan = {}
            for i in live:
                others = sum(self.slots[o]["rows"] for o in live if o != i) if shared else 0
                plan[i] = self._propose(self.slots[i], others, rate)
            rows = sum(1 + plan[i][1] for i in live)
            ms = self._verify_true(rows, len(live))
            for i in live:
                self.slots[i]["vms"] += ms
            m_slots = [i for i in live if plan[i][0] == "m"]
            together = self.batch_mtp and len(m_slots) > 1
            f_slots = [i for i in live if plan[i][0] == "f"]
            if self.spec.get("fb") and f_slots:          # what-if: every slot's DFlash2 block in one pass
                ms += self.block_ms + 1.0 * (len(f_slots) - 1) + 0.6 * len(f_slots)
            for i in live:
                arm, k, L, steps, backlog = plan[i]
                if arm == "f" and not self.spec.get("fb"):
                    ms += self.block_ms + 0.6
                elif arm == "m" and not together:
                    ms += 2.04 + 1.68 * (steps - 1) + 0.5 * steps
            if together:
                top = max(plan[i][3] for i in m_slots)
                ms += 2.3 + 1.9 * (top - 1) + 0.5 * sum(plan[i][3] for i in m_slots)
            t += ms
            parts, tokens = [], 0
            for i in live:                               # ``Stepper.accept`` / ``Batcher._verify``
                sl = self.slots[i]
                arm, k, L, steps, backlog = plan[i]
                keep = min(min(L, k) + 1, self.tokens - sl["out"])
                R = 1 + k
                sl.setdefault("keeps", collections.Counter())[keep] += 1
                tokens += keep
                sl["out"] += keep
                sl["rounds"] += 1
                sl["arms"][arm] += 1
                if arm in sl["karm"]:
                    sl["karm"][arm][0] += keep
                    sl["karm"][arm][1] += 1
                sl["m_back"] = keep if sl["m_back"] + keep > BACKLOG else sl["m_back"] + keep
                sl["n_f"] = keep if sl["n_f"] + keep > BACKLOG else sl["n_f"] + keep
                if sl["sc"] is not None:
                    sl["sc"].record(arm, R, steps, backlog, keep, sl["opt"].verify, shared)
                elif arm in "mf":
                    sl["choice"].record(arm, R, steps, backlog, keep)
                if arm in "mf":
                    sl["opt"].record(arm, R, steps, backlog, keep)
                else:
                    sl["opt"].rounds.append((1, self.costs["verify"][0]))
                    del sl["opt"].rounds[:-self.depth.RATE_WINDOW]
                parts.append((arm, R, steps, backlog))
                sl["rows"] = R
                if sl["out"] >= self.tokens:
                    sl["done_t"] = t
                    self.finished.append(sl)
                    if self.horizon is not None:        # steady: a new request of the same profile takes the slot
                        self.slots[i] = self._new(sl["prof"])
                        self.slots[i]["t0"] = t
            self.rc.record(parts, tokens)
        every = self.finished + [sl for sl in self.slots if sl not in self.finished]
        total = sum(sl["out"] for sl in every) if self.horizon is None else \
            sum(sl["out"] for sl in self.finished) + sum(sl["out"] for sl in self.slots if sl["done_t"] is None)
        by = collections.defaultdict(list)
        for sl in every:
            by[sl["prof"]].append(sl)
        n = len(self.streams)
        agg = lambda f: [sum(f(sl) for sl in by[j]) / max(sum(sl["rounds"] for sl in by[j]), 1) for j in range(n)]  # noqa: E731
        lat = [statistics.mean(sl["out"] / (sl["done_t"] - sl["t0"]) * 1e3 for sl in by[j] if sl["done_t"])
               if any(sl["done_t"] for sl in by[j]) else float("nan") for j in range(n)]
        vms = [sum(sl["vms"] for sl in by[j]) / max(sum(sl["rounds"] for sl in by[j]), 1) for j in range(n)]
        karm = [{a: (sum(sl["karm"][a][0] for sl in by[j]) / sum(sl["karm"][a][1] for sl in by[j])
                     if sum(sl["karm"][a][1] for sl in by[j]) else None) for a in "mf"} for j in range(n)]
        keeps = collections.Counter()
        for sl in every:
            keeps.update(sl.get("keeps", ()))
        return dict(keeps=keeps, karm=karm, vms=vms, agg=total / t * 1e3, tpr=agg(lambda sl: sl["out"] - 1), serial=agg(lambda sl: sl["arms"]["s"]),
                    mix=agg(lambda sl: sl["arms"]["m"]), per=lat, t=t)


def parse_variant(text):
    """``base:fb=1`` (what-if: batched DFlash2 blocks, 3.9 ms + 1.0 a further slot), ``base``, ``base:row=8``, ``adapt``, ``adapt:serial=0:every=16``, ``adapt:margin=3`` (tenths of a token), ``oracle``, ``base:fixed=3``, ``base:mtp=0``;
    patches/0380: ``base:rows=16``, ``base:rows=16:block=16:tail=0.9``, ``oracle:rows=16:block=16``."""

    name, *opts = text.split(":")
    spec = {"adapt": name == "adapt", "oracle": name == "oracle"}
    for o in opts:
        k, v = o.split("=")
        k = {"row": "row_ms", "mtp": "batch_mtp"}.get(k, k)
        spec[k] = float(v) if k in ("row_ms", "tail") else bool(int(v)) if k in ("serial", "batch_mtp") else int(v)
    if not 8 <= spec.get("rows", 8) <= 16 or not 2 <= spec.get("block", 8) <= max(spec.get("rows", 8), 8):
        raise ValueError(f"variant {text!r}: rows 8 to 16, block 2 to rows")
    return spec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--src", default=None, help="a patched TensorFold tree's src/ (default: the installed one)")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--signal", default="info", choices=("info", "flat"))
    ap.add_argument("--true-row", type=float, default=6.0)
    ap.add_argument("--slot-ms", type=float, default=7.0)
    ap.add_argument("--no-share", dest="share", action="store_false", help="true row cost past the table decays as rows share experts: "
                    "--true-row x (1 - 8/288)^(rows past 8) (uniform routing, top 8 of 288)")
    ap.add_argument("--steady", type=float, default=0.0, help="seconds of steady serving (0: the finite bench)")
    ap.add_argument("--variants", default="base,adapt,adapt:serial=0,oracle",
                    help="comma list: base[:row=F][:fixed=N][:mtp=0|1], adapt[:serial=0|1][:every=N][:window=N]"
                         "[:row=F], oracle")
    ap.add_argument("--streams", default="all", help="all, or 'low' (drop streams with > 4 tokens a round)")
    ap.add_argument("--quiet", action="store_true", help="only the summary")
    ap.add_argument("--raw", action="store_true", help="use the recorded (censored) survival as it is, no fit")
    ap.add_argument("--json", default=None)
    ap.add_argument("--alone", action="store_true", help="patches/0380: every recorded stream simulated on its own; "
                    "gains per stream class")
    ap.add_argument("--deep-row", type=float, default=DEEP_ROW, help="ms a lone window's row past 8 (0380)")
    ap.add_argument("--block-ms", type=float, default=BLOCK16_MS, help="ms a DFlash2 block pass of more than 8 rows")
    a = ap.parse_args()
    if a.src:
        sys.path.insert(0, a.src)
    os.environ.setdefault("GLM53_TF_MAX_DRAFT_ROWS", "16")      # patches/0380: room for 15 calibrated positions
    from tensorfold.families.glm5_next.cuda import adapt, batchplan, depth

    mods = (depth, batchplan, adapt)
    runs = load(a.files)
    variants = [(v, parse_variant(v)) for v in a.variants.split(",")]
    if any(spec.get("rows", 8) - 1 > getattr(depth, "MAX_DRAFTS", 7) for _, spec in variants):
        print("note: this tree's depth.py calibrates 7 positions (no patches/0380): positions past 7 share one "
              "correction", file=sys.stderr)
    horizon = a.steady * 1e3 if a.steady > 0 else None
    if a.alone:
        runs = [(f"{l} s{j}", None, [st]) for l, m, sts in runs for j, st in enumerate(sts)]
    else:
        runs = [r for r in runs if len(r[2]) >= 2]
    if a.streams == "low":
        runs = [(l, m, [s for s in st if s["tpr"] <= 4.0]) for l, m, st in runs]
    sim_kw = dict(tokens=a.tokens, signal=a.signal, true_row=a.true_row, slot_ms=a.slot_ms, share=a.share,
                  deep_row=a.deep_row, block_ms=a.block_ms)
    if not a.raw:
        print("fitting each stream's acceptance to its recorded keeps (baseline policy, finite runs):")
        decensor(mods, runs, sim_kw, log=print)
        print()
    out = {}
    print(f"signal={a.signal} true row={a.true_row} ms, slot overhead={a.slot_ms} ms, {a.seeds} seeds, "
          f"{a.tokens} tokens a stream, {'steady %.0f s' % a.steady if horizon else 'finite (makespan)'}\n")
    summary = collections.defaultdict(list)
    by_class = collections.defaultdict(lambda: collections.defaultdict(list))     # patches/0380 (--alone)
    keep_hist = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for label, measured, streams in runs:
        if len(streams) < 2 and not a.alone:
            continue
        if not a.quiet and measured is None:
            print(f"{label} ({streams[0]['cls']}): recorded tok/round {streams[0]['tpr']:.2f}")
        elif not a.quiet:
            print(f"{label}: measured aggregate {measured:.1f} tok/s; recorded tok/round "
                  f"{', '.join(f'{s['tpr']:.2f}' for s in streams)}; verify ms/round "
                  f"{', '.join(f'{s['vms']:.0f}' for s in streams)}")
        for name, spec in variants:
            res = [Sim(mods, streams, spec, seed=sd, horizon_ms=horizon, **sim_kw).run() for sd in range(a.seeds)]
            agg = statistics.mean(r["agg"] for r in res)
            col = lambda key: [statistics.mean(r[key][i] for r in res) for i in range(len(streams))]  # noqa: E731
            tpr, ser, mix, per, vms = col("tpr"), col("serial"), col("mix"), col("per"), col("vms")
            summary[name].append(agg)
            if a.alone:
                by_class[streams[0]["cls"]][name].append(agg)
                for r in res:
                    keep_hist[streams[0]["cls"]][name].update(r["keeps"])
            out.setdefault(label, {})[name] = dict(agg=agg, tpr=tpr, serial=ser, mtp=mix, per_stream=per)
            if not a.quiet:
                print(f"  {name:24s} aggregate {agg:6.1f}  tok/round {', '.join(f'{x:.2f}' for x in tpr)}"
                      f"  per stream {', '.join(f'{x:.1f}' for x in per)}"
                      f"  verify ms/round {', '.join(f'{x:.0f}' for x in vms)}"
                      f"  serial {', '.join(f'{x:.0%}' for x in ser)}  mtp {', '.join(f'{x:.0%}' for x in mix)}")
        if not a.quiet:
            print()
    first = variants[0][0]
    print(f"mean over {len(summary[first])} runs (vs {first}):")
    for name, v in summary.items():
        g = [x / b - 1 for x, b in zip(v, summary[first])]
        print(f"  {name:24s} {statistics.mean(v):6.1f} tok/s  {statistics.mean(g):+.1%}  (runs {min(g):+.1%} .. "
              f"{max(g):+.1%})")
    if a.alone:                                   # patches/0380: per stream class
        print(f"\nalone, by stream class (mean tok/s over its streams; gain vs {first}; share of rounds keeping 9+ "
              "tokens):")
        for cls in ("prose", "code", "repetitive"):
            if cls not in by_class:
                continue
            base = by_class[cls][first]
            print(f"  {cls} ({len(base)} streams)")
            for name, v in by_class[cls].items():
                g = [x / b - 1 for x, b in zip(v, base)]
                h = keep_hist[cls][name]
                deep = sum(c for k, c in h.items() if k > 8) / max(sum(h.values()), 1)
                print(f"    {name:34s} {statistics.mean(v):6.1f} tok/s  {statistics.mean(g):+.1%}  (streams "
                      f"{min(g):+.1%} .. {max(g):+.1%})  rounds keeping 9+: {deep:.0%}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
