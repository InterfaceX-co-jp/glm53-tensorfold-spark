"""Prompt-lookup drafts (patches/0020, ``glm5_next/cuda/lookup.py``): the matcher, the per-round decision and a
host-only simulation of the verify loop. No torch, no GPU.

Run against the patched tree: PYTHONPATH=<tree>/src pytest -q tests/test_glm_lookup.py
"""

from __future__ import annotations

import random

import pytest

from tensorfold.families.glm5_next.cuda.lookup import (BANDS, LOOKUP_KIND, Lookup, SuffixIndex, best_depth,
                                                        lookup_for, parse_policy, round_ms, with_lookup)

# the real model's verify windows of 1 to 8 rows (ms) and drafter costs, as the engine's calibration reports them
COSTS = {"verify": [30.0, 40.0, 44.7, 49.5, 54.2, 58.9, 63.7, 68.4], "mtp": 2.0, "mtp_step": 2.0, "mtp_row": 0.1,
         "block": 3.9, "taps_row": 0.05}


def brute_match(t: list[int], n: int) -> tuple[int, int]:
    """Reference: the longest backward match of the history's end at an earlier end position (ties: latest)."""

    L = len(t)
    best = (0, -1)
    for end in range(L - 1, n - 1, -1):
        length = 0
        while length < end and t[end - 1 - length] == t[L - 1 - length]:
            length += 1
        if length >= n and length > best[0]:
            best = (length, end)
    return best


# -- matcher --------------------------------------------------------------------------------------------------------
def test_continues_earlier_occurrence():
    idx = SuffixIndex(3)
    idx.extend([9, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3])
    assert idx.match() == (3, 4)
    assert idx.propose(4) == ([4, 5, 6, 7], 3)


def test_no_match_on_novel_text_and_no_self_match():
    idx = SuffixIndex(2)
    idx.extend(range(50))
    assert idx.match() == (0, -1) and idx.propose(7) == ([], 0)
    idx = SuffixIndex(2)
    idx.extend([5, 6])                        # the suffix itself is not an earlier occurrence
    assert idx.match() == (0, -1)


def test_longest_wins_then_most_recent():
    # [7 1 2] then [1 2] later: the older occurrence matches 3 tokens back, the newer only 2
    idx = SuffixIndex(2)
    idx.extend([7, 1, 2, 40, 0, 1, 2, 50, 7, 1, 2])
    assert idx.propose(1) == ([40], 3)
    # equal lengths: the most recent occurrence
    idx = SuffixIndex(2)
    idx.extend([1, 2, 40, 1, 2, 50, 1, 2])
    assert idx.propose(1) == ([50], 2)


def test_min_match_threshold():
    idx = SuffixIndex(2)
    idx.extend([1, 2, 3, 4, 9, 9, 3, 4])
    assert idx.propose(3, min_match=3) == ([], 2)
    assert idx.propose(3, min_match=2) == ([9, 9, 3], 2)


def test_overlapping_match_continues_periodically():
    idx = SuffixIndex(2)
    idx.extend([0, 1, 2, 3, 1, 2, 3, 1, 2])
    drafts, length = idx.propose(7)
    assert drafts == [3, 1, 2, 3, 1, 2, 3] and length == 5
    idx = SuffixIndex(1)
    idx.extend([4, 4])                        # a run of one token continues as that token
    assert idx.propose(5)[0] == [4] * 5


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_incremental_index_matches_brute_force(n):
    rng = random.Random(n)
    idx = SuffixIndex(n, candidates=10**9, extend_to=10**9)
    t: list[int] = []
    for step in range(400):
        new = [rng.randrange(4) for _ in range(rng.randrange(1, 9))]
        t += new
        idx.extend(new)
        got, want = idx.match(), brute_match(t, n)
        assert got == want, (step, got, want)


def test_candidates_and_extension_are_bounded():
    idx = SuffixIndex(2, candidates=3, extend_to=5)
    idx.extend([1, 2] * 200)
    length, end = idx.match()
    assert length == 5 and end == len(idx) - 2         # the most recent candidate, measured up to 5


def test_next_turn_reuses_the_index_and_proposes_the_same():
    rng = random.Random(9)
    block = [rng.randrange(30) for _ in range(40)]
    first = Lookup(block, COSTS, min_match=2, gated=False)
    out = [block[0]]
    for t in block[1:12]:
        first.plan(out, 50)
        out.append(t)
    first.plan(out, 50)
    history = list(first.index.tokens)                          # the prompt, then the reply so far
    nxt = history + [rng.randrange(30) for _ in range(9)] + block[:5]
    warm = Lookup(nxt, COSTS, min_match=2, gated=False)
    assert warm.index is first.index                            # the next turn extends the last history
    cold = Lookup([7777] + nxt, COSTS, min_match=2, gated=False)  # not an extension: a fresh index
    assert cold.index is not first.index
    for extra in ([], [block[5]], [block[5], block[6], 3]):
        a, b = warm.plan([1] + extra, 50), cold.plan([1] + extra, 50)
        assert a == b and warm.index.match()[0] == cold.index.match()[0]


def test_live_lookups_never_share_history():
    prompt = [1, 2, 3, 4, 5, 6, 7, 8] * 3
    a = Lookup(prompt, COSTS, min_match=2, gated=False)
    b = Lookup(prompt, COSTS, min_match=2, gated=False)       # takes the shared index over
    assert b.plan([1, 2], 9) == [3, 4, 5, 6, 7, 8, 1]           # at most 7 drafts
    assert a.plan([1, 9], 9) == []                            # a's own history: ... 8 1 9, never seen
    assert a.index is not b.index and a.index.tokens == prompt + [1, 9]
    assert b.plan([1, 2, 3], 3) == [4, 5, 6]


# -- policy codes ---------------------------------------------------------------------------------------------------
def test_policy_codes():
    assert parse_policy("l7") == [LOOKUP_KIND, 7, 3, 0]
    assert parse_policy("l4:8") == [LOOKUP_KIND, 4, 8, 0]
    assert parse_policy("auto") is None and parse_policy("c3:0.35") is None
    for bad in ("l0", "l8", "l", "lx", "l3:0", "l3:65", "l3:2:1"):
        with pytest.raises(ValueError):
            parse_policy(bad)


def test_engine_encodes_lookup_specs():
    from tensorfold.families.glm5_next.cuda.engine import encode_policy

    assert encode_policy("l7") == [LOOKUP_KIND, 7, 3, 0] and encode_policy("l2:6") == [LOOKUP_KIND, 2, 6, 0]
    assert encode_policy("auto") == [4, 2, 8, 30000]                   # unchanged
    for bad in ("fl3", "l9"):
        with pytest.raises(ValueError):
            encode_policy(bad)


def test_with_lookup_settings_travel_in_the_code(monkeypatch):
    dflash = [13, 5, 300000, 0]
    monkeypatch.delenv("GLM53_TF_LOOKUP", raising=False)
    monkeypatch.delenv("GLM53_TF_LOOKUP_MIN", raising=False)
    assert with_lookup([4, 2, 8, 30000], True, dflash) == [4, 2, 8, 30000, 1, 4]
    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")
    monkeypatch.setenv("GLM53_TF_LOOKUP_MIN", "6")
    assert with_lookup([5, 1, 1, 0], True, dflash) == [5, 1, 1, 0, 0, 6]
    assert with_lookup([LOOKUP_KIND, 7, 3, 0], True, dflash) == [LOOKUP_KIND, 7, 3, 0]
    assert with_lookup([LOOKUP_KIND, 7, 3, 0], False, dflash) == dflash      # no MTP head: DFlash2 alone
    assert with_lookup([3, 3, 350000, 0], True, dflash) == [3, 3, 350000, 0]
    assert with_lookup([13, 5, 300000, 0], True, dflash) == [13, 5, 300000, 0]
    monkeypatch.setenv("GLM53_TF_LOOKUP_MIN", "0")
    with pytest.raises(ValueError):
        with_lookup([4, 2, 8, 30000], True, dflash)
    # rank 1 builds the drafter from the code alone
    assert lookup_for([4, 2, 8, 30000, 0, 4], [1], COSTS) is None
    assert lookup_for([4, 2, 8, 30000], [1], COSTS) is None
    assert lookup_for([3, 3, 350000, 0], [1], COSTS) is None
    lk = lookup_for([4, 2, 8, 30000, 1, 5], [1], COSTS)
    assert lk.gated and lk.min_match == 5 and lk.most == 7
    lk = lookup_for([LOOKUP_KIND, 3, 2, 0], [1], COSTS)
    assert not lk.gated and lk.min_match == 2 and lk.most == 3


# -- the decision ---------------------------------------------------------------------------------------------------
def test_best_depth_on_measured_costs():
    assert best_depth(1.0, COSTS, 7) == (7, pytest.approx(8 / 68.4))
    assert best_depth(0.0, COSTS, 7)[0] == 1
    assert best_depth(0.9, COSTS, 3)[0] == 3                   # capped by the drafts on offer
    # at the prior for long copies a full window pays; for short matches a lookup round loses to MTP's prior
    lk = Lookup([], COSTS)
    assert best_depth(BANDS[-1][1], COSTS, 7)[0] == 7
    assert best_depth(BANDS[0][1], COSTS, 7)[1] < lk.alt_rate()


def test_round_ms_prices_like_drafter_choice():
    assert round_ms(COSTS, "l", 8, 0, 0) == 68.4
    assert round_ms(COSTS, "m", 4, 3, 5) == pytest.approx(49.5 + 2.0 + 2 * 2.0 + 4 * 0.1)
    assert round_ms(COSTS, "f", 8, 0, 3) == pytest.approx(68.4 + 3.9 + 3 * 0.05)


def test_forced_plan_room_and_eos():
    lk = Lookup([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2], COSTS, most=7, min_match=2, gated=False, eos=(6,))
    assert lk.plan([3], room=10) == [4, 5]                    # cut before the end-of-sequence token
    lk = Lookup([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2], COSTS, most=7, min_match=2, gated=False, eos=(6,),
                stop_eos=False)
    assert lk.plan([3], room=4) == [4, 5, 6, 7]               # never more than the room left
    assert lk.plan([3], room=0) == []


def test_gate_learns_from_rounds():
    block = list(range(100, 140))
    lk = Lookup(block + block, COSTS, min_match=4, gated=True)
    out = [block[0]]                                            # the text repeats: a match of 41 tokens
    drafts = lk.plan(out, 50)
    assert drafts == block[1:8]                                 # the prior for a long copy verifies 7
    lk.record("l", 8, 0, 0, 1)                                  # ... and none was kept, twice
    lk.record("l", 8, 0, 0, 1)
    assert lk.p(lk.band) < BANDS[-1][1]
    for _ in range(4):
        lk.record("m", 3, 2, 1, 3)                              # MTP rounds keep 3 tokens for ~46 ms
    assert lk.plan(out, 50) == []                               # lookup no longer pays: MTP drafts


def test_long_match_is_probed_after_skips():
    block = list(range(100, 140))
    lk = Lookup(block + block, COSTS, min_match=4, gated=True)
    for _ in range(4):
        lk.record("f", 8, 0, 1, 8)                              # DFlash2 keeps 8 tokens a round: hard to beat
    out = [block[0]]
    got = [lk.plan(out, 50) for _ in range(9)]
    assert got[:7] == [[]] * 7 and got[7] == block[1:8] and got[8] == []


def test_short_matches_wait_for_evidence():
    lk = Lookup([1, 2, 3, 4, 77, 9, 9, 9], COSTS, min_match=4, gated=True)
    assert lk.plan([1, 2, 3, 4], 50) == []                      # 4 matching tokens: the cautious prior says no
    for _ in range(6):
        lk.record("m", 2, 1, 1, 1)                              # the MTP rounds keep nothing (novel text)
    assert lk.plan([1, 2, 3, 4], 50) == [77, 9]                 # now a short lookup beats them


# -- the verify loop, simulated -------------------------------------------------------------------------------------
class Model:
    """A deterministic 'model' that copies: after the last ``order`` tokens it emits what followed their latest
    earlier occurrence (an induction head), else a hash of them; at a ``noise`` share of positions (a hash of the
    position) it emits something else. Hashes of int tuples do not depend on PYTHONHASHSEED."""

    def __init__(self, seed: int, order: int = 6, noise: float = 0.0, vocab: int = 5000) -> None:
        self.seed, self.order, self.noise, self.vocab = seed, order, noise, vocab

    def next(self, h: list[int]) -> int:
        L, n = len(h), self.order
        if self.noise and hash((self.seed, L)) % 1000 < self.noise * 1000:
            return hash((self.seed, L, 7919)) % self.vocab
        tail = h[-n:]
        for s in range(L - n - 1, -1, -1):
            if h[s:s + n] == tail:
                return h[s + n]
        return hash((self.seed, *tail)) % self.vocab


def serial(model: Model, prompt: list[int], count: int) -> list[int]:
    h = list(prompt)
    out = []
    for _ in range(count):
        out.append(model.next(h))
        h.append(out[-1])
    return out


def drafted(model: Model, prompt: list[int], count: int, lookup: Lookup | None, alt_keep: int = 2):
    """``decode.auto_decode``'s accept rule with lookup rounds and a stand-in MTP arm that keeps ``alt_keep``
    tokens a round; returns (tokens, model ms, arms)."""

    out = [model.next(list(prompt))]
    ms = 0.0
    arms = ""
    while len(out) < count:
        room = count - len(out)
        look = lookup.plan(out, room) if lookup is not None else None
        h = list(prompt) + out
        if look:
            arm, drafts = "l", look
        else:
            arm, drafts = "m", serial(model, h, min(alt_keep - 1, room))       # always-right MTP stand-in
        R = 1 + len(drafts)
        sampled, hh = [], list(h)
        for r in range(R):                                     # the window: row r sees the drafts before it
            sampled.append(model.next(hh))
            if r < len(drafts):
                hh.append(drafts[r])
        keep = 1
        for i, d in enumerate(drafts):
            if sampled[i] != d:
                break
            keep += 1
        out.extend(sampled[:keep])
        ms += round_ms(COSTS, arm, R, max(len(drafts), 1), 1)
        if lookup is not None:
            lookup.record(arm, R, max(len(drafts), 1), 1, keep)
        arms += arm
    return out[:count], ms, arms


@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize("noise", [0.0, 0.05, 0.5])
def test_simulated_replies_equal_serial(gated, noise):
    rng = random.Random(7)
    block = [rng.randrange(5000) for _ in range(60)]
    prompt = block * 3 + block[:10]
    model = Model(1, noise=noise)
    want = serial(model, prompt, 150)
    lk = Lookup(prompt, COSTS, min_match=3 if not gated else 4, gated=gated)
    got, _, _ = drafted(model, prompt, 150, lk)
    assert got == want


def test_simulated_speedup_on_repeats_and_no_loss_on_novel_text():
    rng = random.Random(3)
    # repetitive: the model copies (its next token depends on the last 6), and the prompt holds the block 3 times
    block = [rng.randrange(5000) for _ in range(80)]
    prompt = block * 3 + block[:8]
    model = Model(2)
    copy = serial(model, prompt, 200)
    base, base_ms, _ = drafted(model, prompt, 200, None)
    got, ms, arms = drafted(model, prompt, 200, Lookup(prompt, COSTS, min_match=4))
    assert got == base == copy
    assert base_ms / ms > 2.0, (base_ms, ms, arms)             # ~7 tokens a 68 ms window against 2 a 42 ms one
    # novel: no span repeats, so the gate never spends a window on the lookup
    novel = [rng.randrange(5000) for _ in range(300)]
    model = Model(3, order=64)
    base, base_ms, _ = drafted(model, novel, 120, None)
    got, ms, arms = drafted(model, novel, 120, Lookup(novel, COSTS, min_match=4))
    assert got == base and "l" not in arms and ms == base_ms
