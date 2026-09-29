#!/usr/bin/env python3
"""Offline replay of prompt-lookup drafts on real token streams: 8- vs 16-row windows (patches/0380) and a
cross-request suffix index (N3, SuffixDecoding-style). No GPU, no torch. docs/DEEP-VERIFY.md has the results.

What it replays. Each request is (history, reply): the reply's tokens are what serial decoding produced, so a lookup
draft is kept exactly when it equals the next reply token. Every round runs the ENGINE'S OWN lookup code from a patched
tree (``lookup.Lookup`` with ``auto``'s gate and ``depth.DepthOptimizer``'s cost depths, patches/0020 / 0071): the
lookup proposes from the request's history (prompt, then every committed token), the gate decides whether the round is
a lookup round and how many drafts it verifies. A round the gate leaves to the model's drafters is simulated: its
committed tokens are drawn from the recorded DFlash2 keeps of the stream class the reply position belongs to (``prose``
for reasoning / text, ``code`` for tool-call arguments and code; results/*/conc*.json, docs/RESEARCH-NIGHT.md §1),
independently of the text (so copy regions the gate declines are priced as ordinary rounds: see the caveat below).

Time: a round costs its verify window (the two-Spark table, then ``--deep-row`` ms a row past 8: what the load-time
calibration measures), plus a DFlash2 block (3.9 ms) and 0.6 ms host for drafter rounds, plus ``--host-ms`` between
rounds (PROFILE.md §5: ~3.9 ms of plan share and emit a single-stream round).

Sources (``--source``):
  opencode:DB   agent turns from an opencode database (local, read only; nothing of the text leaves this process):
                every assistant step served by a GLM-5.3-Flash provider (``--providers``) is a request whose history
                is the session so far (user text, earlier replies without their reasoning, tool calls, tool outputs)
                and whose reply is its reasoning + text + tool calls, tokenized with the GLM tokenizer (``--tokenizer``)
                in the chat template's shape.
  synthetic     the bench cells that can be written down exactly: ``count`` (count 1..200), ``primes`` (the first 150
                primes), ``json`` (25 employee records), ``edit`` (a Python file of this repo re-emitted with a small
                edit, glmbench's edit suite).

Variants: ``8`` (today: windows up to 8 rows), ``16`` (GLM53_TF_MAX_DRAFT_ROWS=16), ``16g`` (16 rows and a global
suffix index over every earlier reply of the corpus, time-ordered, consulted when it matches a longer suffix than the
request's own history: N3), ``8g``.

Caveat: drafter rounds are drawn independently of the text, so where the gate declines a copy the model's own
drafters would do better than the draw; gains of lookup variants are upper-leaning. Absolute tok/s are indicative;
the comparisons between variants are what the tool is for.

    python3 bench/lookupsim.py --src /tmp/tf/src --tokenizer <dir with tokenizer.json> --source synthetic
    python3 bench/lookupsim.py --src /tmp/tf/src --tokenizer <dir> --source opencode:~/.local/share/opencode/opencode.db
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sqlite3
import statistics
import sys
import zlib
from pathlib import Path

V = [31.0, 39.0, 45.0, 51.0, 56.0, 62.0, 68.0, 74.0]
BASE = {"mtp": 2.04, "mtp_step": 1.68, "mtp_row": 0.1, "block": 3.88, "taps_row": 0.05}
# recorded DFlash2 keeps (tokens committed a round, 1..8) by stream class: results/W1, W5, W6, W8 concurrent runs
KEEPS = {"prose": [0.251, 0.349, 0.164, 0.117, 0.053, 0.040, 0.008, 0.017],
         "code": [0.091, 0.169, 0.128, 0.060, 0.051, 0.091, 0.066, 0.343],
         "repetitive": [0.016, 0.014, 0.016, 0.001, 0.001, 0.115, 0.013, 0.824]}
SEP = -1                  # between replies in the global index: no match crosses it


def costs_for(rows: int, deep_row: float) -> dict:
    c = dict(BASE)
    c["verify"] = list(V) + [V[-1] + deep_row * r for r in range(1, rows - len(V) + 1)]
    return c


# -- the corpus ------------------------------------------------------------------------------------------------------
class Tok:
    def __init__(self, path: str) -> None:
        from tokenizers import Tokenizer

        p = Path(path)
        self.t = Tokenizer.from_file(str(p / "tokenizer.json" if p.is_dir() else p))
        self.cache: dict[str, list[int]] = {}

    def __call__(self, text: str) -> list[int]:
        got = self.cache.get(text)
        if got is None:
            got = self.t.encode(text, add_special_tokens=False).ids
            if len(text) < 4096:
                self.cache[text] = got
        return got


def _tool_text(name: str, args: dict) -> str:
    """A tool call as GLM emits it (``<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>``)."""

    body = "".join(f"<arg_key>{k}</arg_key><arg_value>{v if isinstance(v, str) else json.dumps(v)}</arg_value>"
                   for k, v in args.items())
    return f"<tool_call>{name}{body}</tool_call>"


def opencode_requests(db: str, tok: Tok, providers: set[str], limit: int = 0):
    """-> [(session, [history token segments], [(reply tokens, class)])] in time order, GLM-served steps only."""

    con = sqlite3.connect(f"file:{os.path.expanduser(db)}?mode=ro", uri=True)
    msgs = con.execute("select id, session_id, time_created, data from message order by session_id, time_created, id")
    by_session = collections.defaultdict(list)
    for mid, sid, t, data in msgs:
        by_session[sid].append((mid, t, json.loads(data)))
    parts = collections.defaultdict(list)
    for mid, data in con.execute("select message_id, data from part order by message_id, id"):
        parts[mid].append(json.loads(data))
    out = []
    for sid, ms in by_session.items():
        hist: list[list[int]] = []
        for mid, t, m in ms:
            role = m.get("role")
            ps = parts.get(mid, [])
            if role == "user":
                text = "".join(p.get("text", "") for p in ps if p.get("type") == "text")
                hist.append(tok("<|user|>\n" + text))
                continue
            if role != "assistant":
                continue
            reply: list[tuple[list[int], str]] = [(tok("<|assistant|>"), "prose")]
            reasoning = "".join(p.get("text", "") for p in ps if p.get("type") == "reasoning")
            reply.append((tok("<think>" + reasoning + "</think>"), "prose"))
            for p in ps:
                if p.get("type") == "text":
                    reply.append((tok("\n" + p.get("text", "")), "prose"))
                elif p.get("type") == "tool":
                    st = p.get("state", {})
                    reply.append((tok(_tool_text(p.get("tool", ""), st.get("input") or {})), "code"))
            # the next step's history holds this step as emitted (GLM keeps an agent turn's reasoning between tool
            # calls; earlier user turns' reasoning would be dropped: a small overstatement of the history)
            later = [seg for seg, _ in reply]
            served = m.get("providerID") in providers and "glm" in str(m.get("modelID", "")).lower()
            if served:
                out.append((f"{sid}:{mid}", t, [list(h) for h in hist], reply))
            hist.extend(later)
            for p in ps:                           # the tool outputs are the next request's prompt
                if p.get("type") == "tool":
                    o = (p.get("state") or {}).get("output")
                    if isinstance(o, str) and o:
                        hist.append(tok("<|observation|>\n<tool_response>" + o[:200000] + "</tool_response>"))
            if limit and len(out) >= limit:
                break
    out.sort(key=lambda r: r[1])
    return [(sid, hist, reply) for sid, _, hist, reply in out]


def synthetic_requests(tok: Tok, repo: Path):
    rng = random.Random(0)
    reqs = []
    count = " ".join(str(i) for i in range(1, 201))
    reqs.append(("count", [tok("<|user|>\nCount from 1 to 200. Output only the numbers, separated by spaces. No other "
                               "text.<|assistant|>\n</think>")], [(tok(count), "repetitive")]))
    primes, n = [], 2
    while len(primes) < 150:
        if all(n % p for p in primes if p * p <= n):
            primes.append(n)
        n += 1
    reqs.append(("primes", [tok("<|user|>\nList the first 150 prime numbers in order, separated by commas. Output only "
                                "the numbers.<|assistant|>\n</think>")], [(tok(", ".join(map(str, primes))), "repetitive")]))
    first = "James Mary Robert Patricia John Jennifer Michael Linda David Elizabeth William Barbara".split()
    last = "Smith Johnson Williams Brown Jones Garcia Miller Davis Rodriguez Martinez Hernandez Lopez".split()
    depts = [("Engineering", ["Software Engineer", "Senior Engineer", "Staff Engineer"], ["Python", "Go", "Kubernetes"]),
             ("Marketing", ["Marketing Manager", "Content Strategist"], ["SEO", "Copywriting", "Analytics"]),
             ("Sales", ["Account Executive", "Sales Manager"], ["Negotiation", "CRM", "Forecasting"]),
             ("Finance", ["Financial Analyst", "Controller"], ["Excel", "Modeling", "GAAP"])]
    recs = []
    for i in range(1, 26):
        f, l = rng.choice(first), rng.choice(last)
        d, titles, skills = rng.choice(depts)
        recs.append({"id": i, "first_name": f, "last_name": l, "email": f"{f.lower()}.{l.lower()}@example.com",
                     "department": d, "title": rng.choice(titles), "salary": rng.randrange(50, 180) * 1000,
                     "start_date": f"20{rng.randrange(10, 24)}-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}",
                     "skills": rng.sample(skills, 2)})
    reqs.append(("json", [tok("<|user|>\nOutput a JSON array of 25 fictional employee records. Output only JSON."
                              "<|assistant|>\n</think>")], [(tok(json.dumps(recs, indent=2)), "code")]))
    for name in ("bench/multiturn.py", "scripts/traffic-report.py", "bench/prefixshare.py"):
        p = repo / name
        if not p.exists():
            continue
        src = p.read_text()[:12000]
        lines = src.splitlines(keepends=True)
        j = len(lines) // 2
        edited = "".join(lines[:j] + ["    # (edited)\n"] + lines[j:])
        reqs.append((f"edit:{name}", [tok(f"<|user|>\nHere is a Python file:\n\n```python\n{src}\n```\n\nAdd a comment "
                                          "in the middle. Output only the code.<|assistant|>\n</think>")],
                     [(tok("```python\n" + edited + "```"), "code")]))
    return reqs


# -- the replay ---------------------------------------------------------------------------------------------------------
class _Both:
    """The request's own index with a global one next to it: ``propose`` returns the longer match's copy."""

    def __init__(self, own, glob) -> None:
        self.own, self.glob = own, glob

    def __getattr__(self, name):
        return getattr(self.own, name)

    def propose(self, count: int, min_match: int = 1):
        drafts, length = self.own.propose(count, min_match)
        g_len, g_end = match_external(self.glob, self.own.tokens, self.own.n) if self.glob is not None else (0, -1)
        if g_end >= 0 and g_len > length and g_len >= min_match and count > 0:
            t = self.glob.tokens
            copy = []
            for i in range(count):
                if g_end + i >= len(t) or t[g_end + i] == SEP:
                    break
                copy.append(t[g_end + i])
            if copy:
                self.hits += 1
                return copy, g_len
        return drafts, length


def match_external(index, q: list[int], n: int, extend_to: int = 64) -> tuple[int, int]:
    """The longest match of ``q``'s suffix in ``index``'s tokens (``SuffixIndex.match`` for a query from elsewhere)."""

    if len(q) < n or len(index.tokens) <= n:
        return 0, -1
    index._index()
    ends = index.ends.get(tuple(q[-n:]))
    if not ends:
        return 0, -1
    t = index.tokens
    best_len, best_end = 0, -1
    for end in reversed(ends[-index.candidates:]):
        length = n
        limit = min(extend_to, end, len(q))
        while length < limit and t[end - 1 - length] == q[-1 - length] and t[end - 1 - length] != SEP:
            length += 1
        if length > best_len:
            best_len, best_end = length, end
            if length >= extend_to:
                break
    return best_len, best_end


class Sessions:
    """Each session's prompt index, extended from one request to the next (a request's history is the previous
    request's history, its reply and the new tool outputs / user text), so a prompt is indexed once, not once per
    variant and seed; ``truncate`` undoes what a replay appended."""

    def __init__(self, lookup, n: int) -> None:
        self.lookup, self.n = lookup, n
        self.idx: dict = {}

    def get(self, session: str, prompt: list[int]):
        idx = self.idx.get(session)
        if idx is None or len(idx.tokens) > len(prompt) or prompt[:len(idx.tokens)] != idx.tokens:
            idx = self.lookup.SuffixIndex(self.n)
        idx.extend(prompt[len(idx.tokens):])
        idx._index()
        self.idx[session] = idx
        return idx


def truncate(idx, length: int) -> None:
    """``idx`` back to its first ``length`` tokens (the n-grams past them leave ``ends``, newest first)."""

    t, n = idx.tokens, idx.n
    for end in range(idx.indexed - 1, max(length, n) - 1, -1):
        key = tuple(t[end - n:end])
        lst = idx.ends[key]
        assert lst[-1] == end
        lst.pop()
        if not lst:
            del idx.ends[key]
    del t[length:]
    idx.indexed = min(idx.indexed, length)


def _split(reply):
    toks, cls = [], []
    for seg, c in reply:
        toks.extend(seg)
        cls.extend([c] * len(seg))
    return toks, cls


def replay_all(mods, reqs, variants, seeds, *, deep_row: float, host_ms: float, min_match: int = 4):
    """-> {(variant, seed): [per request: tokens, ms, rounds, lookup rounds, lookup tokens kept, keeps, global hits]}.
    Requests in time order; the global index (N3) holds every earlier reply."""

    lookup, depth = mods
    n_gram = min(min_match, 8)
    sessions = Sessions(lookup, n_gram)
    g_index = lookup.SuffixIndex(n_gram)
    res = {(v, sd): [] for v in variants for sd in range(seeds)}
    real_reuse = lookup.reuse_index
    try:
        for name, hist, reply in reqs:
            toks, cls = _split(reply)
            if len(toks) < 2:
                continue
            prompt = [t for seg in hist for t in seg]
            idx = sessions.get(name.split(":")[0], prompt)
            base_len = len(idx.tokens)
            lookup.reuse_index = lambda n, p, idx=idx: idx
            for v in variants:
                rows, glob = int(v.rstrip("g")), v.endswith("g")
                costs = costs_for(rows, deep_row)
                for sd in range(seeds):
                    res[(v, sd)].append(_one(lookup, depth, name, prompt, toks, cls, costs, rows,
                                             g_index if glob else None, sd, host_ms, min_match))
                    truncate(idx, base_len)
            g_index.extend(toks + [SEP])
    finally:
        lookup.reuse_index = real_reuse
    return res


def _one(lookup, depth, name, prompt, toks, cls, costs, rows, g_index, seed, host_ms, min_match):
    lk = lookup.Lookup(prompt, costs, most=lookup.MAX_DRAFTS, min_match=min_match, gated=True)
    opt = depth.DepthOptimizer(costs, most_m=rows - 1, most_f=7)
    lk.opt = opt
    both = None
    if g_index is not None:
        both = _Both(lk.index, g_index)
        both.hits = 0
        lk.index = both                        # everything but ``propose`` goes to the request's own index
    out = [toks[0]]
    ms, rounds, l_rounds, l_kept = 0.0, 0, 0, 0
    keeps = collections.Counter()
    n = len(toks)
    while len(out) < n:
        room = n - len(out)
        drafts = lk.plan(out, room)
        pos = len(out)
        if drafts:
            keep = 1
            for i, d in enumerate(drafts):
                if pos + i >= n or toks[pos + i] != d:
                    break
                keep += 1
            keep = min(keep, room)
            R = 1 + len(drafts)
            arm, cost = "l", costs["verify"][R - 1]
            l_rounds += 1
            l_kept += keep - 1
        else:                                  # the same draw at a position in every variant (common numbers)
            w = KEEPS[cls[pos]]
            keep = min(random.Random(zlib.crc32(f"{seed}:{name}:{pos}".encode())).choices(range(1, 9), w)[0], room)
            R = min(keep + 1, 8)
            arm, cost = "f", costs["verify"][R - 1] + BASE["block"] + 0.6
        ms += cost + host_ms
        lk.record(arm, R, 0, 0, keep)
        if arm == "l":
            opt.record("l", R, 0, 0, keep)
        else:
            opt.rounds.append((keep, cost))
            del opt.rounds[:-depth.RATE_WINDOW]
        keeps[keep] += 1
        out.extend(toks[pos:pos + keep])
        rounds += 1
    return dict(name=name, tokens=n, ms=ms, rounds=rounds, l_rounds=l_rounds, l_kept=l_kept, keeps=keeps,
                hits=both.hits if both is not None else 0)


def potential(mods, reqs, *, min_match: int = 4, most: int = 15):
    """Gate-free: walk each reply as if every position with a match of ``min_match``+ tokens ran a lookup round of
    ``most`` drafts (else one token); -> {(global index?, class): [reply tokens, tokens committed in lookup rounds,
    in rounds keeping 9+ tokens (past today's window), rounds keeping 9+]}."""

    lookup, _ = mods
    n_gram = min(min_match, 8)
    sessions = Sessions(lookup, n_gram)
    g_index = lookup.SuffixIndex(n_gram)
    out = collections.defaultdict(lambda: [0, 0, 0, 0])
    for name, hist, reply in reqs:
        toks, cls = _split(reply)
        if len(toks) < 2:
            continue
        prompt = [t for seg in hist for t in seg]
        idx = sessions.get(name.split(":")[0], prompt)
        base_len = len(idx.tokens)
        for glob in (False, True):
            src = _Both(idx, g_index) if glob else idx
            src.hits = 0
            pos, n = 1, len(toks)
            idx.extend(toks[:1])
            for c in cls:
                out[(glob, c)][0] += 1
            while pos < n:
                drafts, _ = src.propose(most, min_match)
                run = 0
                for d in drafts:
                    if pos + run >= n or toks[pos + run] != d:
                        break
                    run += 1
                step = min(run + 1 if drafts else 1, n - pos)
                if drafts and run:
                    o = out[(glob, cls[pos])]
                    o[1] += step
                    if step > 8:
                        o[2] += step
                        o[3] += 1
                idx.extend(toks[pos:pos + step])
                pos += step
            truncate(idx, base_len)
        g_index.extend(toks + [SEP])
    return out


def summarize(label: str, res: list[dict], base: list[dict] | None) -> dict:
    tok = sum(r["tokens"] for r in res)
    ms = sum(r["ms"] for r in res)
    rounds = sum(r["rounds"] for r in res)
    keeps = collections.Counter()
    for r in res:
        keeps.update(r["keeps"])
    deep = sum(c for k, c in keeps.items() if k > 8) / max(rounds, 1)
    lr = sum(r["l_rounds"] for r in res)
    out = dict(label=label, tokens=tok, tok_s=tok / ms * 1e3, tpr=tok / rounds, lookup_rounds=lr / max(rounds, 1),
               lookup_tokens=sum(r["l_kept"] for r in res) / max(tok, 1), deep_rounds=deep,
               global_hits=sum(r["hits"] for r in res))
    if base is not None:
        bt = sum(r["ms"] for r in base)
        out["gain"] = bt / ms - 1
        per = [b["ms"] / r["ms"] - 1 for r, b in zip(res, base)]
        out["per_request"] = (min(per), statistics.median(per), max(per))
        out["requests_gaining_5pct"] = sum(1 for g in per if g >= 0.05) / len(per)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="a patched TensorFold tree's src/ (patches through 0380)")
    ap.add_argument("--tokenizer", required=True, help="the GLM-5.3 tokenizer.json (or its folder)")
    ap.add_argument("--source", default="synthetic", help="synthetic | opencode:PATH")
    ap.add_argument("--providers", default="dgxspark,zai-coding-plan", help="opencode providers serving GLM-5.3-Flash")
    ap.add_argument("--variants", default="8,16,8g,16g")
    ap.add_argument("--deep-row", type=float, default=5.5)
    ap.add_argument("--host-ms", type=float, default=3.9)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="at most this many requests (opencode)")
    ap.add_argument("--by-request", action="store_true")
    ap.add_argument("--potential", action="store_true", help="also the gate-free copy potential, per token class")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    os.environ["GLM53_TF_MAX_DRAFT_ROWS"] = "16"
    sys.path.insert(0, a.src)
    from tensorfold.families.glm5_next.cuda import depth, lookup

    if lookup.MAX_DRAFTS < 15:
        sys.exit("this tree has no patches/0380 (lookup.MAX_DRAFTS is 7)")
    tok = Tok(a.tokenizer)
    repo = Path(__file__).resolve().parents[1]
    if a.source == "synthetic":
        reqs = synthetic_requests(tok, repo)
    elif a.source.startswith("opencode:"):
        reqs = opencode_requests(a.source.split(":", 1)[1], tok, set(a.providers.split(",")), a.limit)
    else:
        sys.exit(f"--source {a.source!r}")
    ntok = sum(len(s) for _, _, rep in reqs for s, _ in rep)
    print(f"{len(reqs)} requests, {ntok} reply tokens; deep row {a.deep_row} ms, host {a.host_ms} ms a round, "
          f"{a.seeds} seeds")
    variants = a.variants.split(",")
    got = replay_all((lookup, depth), reqs, variants, a.seeds, deep_row=a.deep_row, host_ms=a.host_ms)
    table = {v: [got[(v, sd)] for sd in range(a.seeds)] for v in variants}
    first = a.variants.split(",")[0]
    summary = {}
    for v, runs in table.items():
        sums = [summarize(v, r, b) for r, b in zip(runs, table[first])]
        s = {k: statistics.mean(x[k] for x in sums) for k in ("tok_s", "tpr", "lookup_rounds", "lookup_tokens",
                                                               "deep_rounds", "global_hits")}
        if v != first:
            s["gain"] = statistics.mean(x["gain"] for x in sums)
            s["requests_gaining_5pct"] = statistics.mean(x["requests_gaining_5pct"] for x in sums)
            s["per_request_median"] = statistics.mean(x["per_request"][1] for x in sums)
        summary[v] = s
        print(f"  {v:4s} {s['tok_s']:6.1f} tok/s  {s['tpr']:.2f} tok/round  lookup rounds {s['lookup_rounds']:.1%}, "
              f"lookup-kept tokens {s['lookup_tokens']:.1%}, rounds keeping 9+ {s['deep_rounds']:.1%}"
              + (f", global hits {s['global_hits']:.0f}" if v.endswith("g") else "")
              + (f"  gain {s['gain']:+.1%} (median request {s['per_request_median']:+.1%}, requests >= +5%: "
                 f"{s['requests_gaining_5pct']:.0%})" if "gain" in s else ""))
    if a.potential:
        pot = potential((lookup, depth), reqs)
        for glob in (False, True):
            print(f"  gate-free copy potential ({'request + global index' if glob else 'request index'}; share of "
                  "reply tokens):", flush=True)
            for (g, c), (n, kept, deep, dr) in sorted(pot.items()):
                if g == glob:
                    print(f"    {c:10s} {n:8d} tokens  committed by lookup rounds {kept / n:6.1%}  of them in rounds "
                          f"of 9+ tokens {deep / n:6.1%} ({dr} rounds)")
    if a.by_request:
        for i, r in enumerate(table[first][0]):
            cells = "  ".join(f"{v} {table[v][0][i]['tokens'] / table[v][0][i]['ms'] * 1e3:6.1f}" for v in table)
            print(f"    {r['name'][:40]:40s} {r['tokens']:6d} tok  {cells}")
    if a.json:
        Path(a.json).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
