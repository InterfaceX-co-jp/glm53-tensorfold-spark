#!/usr/bin/env python3
"""Offline coverage study for a trimmed draft vocabulary (patches/0420, DECODE-PLAN E7, FR-Spec style), and the
ranking file ``GLM53_TF_DRAFT_VOCAB`` reads. No GPU, no torch. docs/DRAFT-VOCAB.md has the results.

The drafters (MTP head, DFlash2) score only listed token ids; a reply token outside the list can never be drafted,
so a chain of accepted drafts ends at the first such token. Verification is unchanged (the target samples over the
whole vocabulary), so replies are too; only acceptance moves.

Cost is per rank. Each rank holds half of the head (ids < 77,440 on rank 0), both ranks exchange their candidates
every draft step, so a draft step takes as long as the larger half. The list is therefore built per rank: the M
most frequent ids of each half (M = N / 2). A global top-N list costs its larger half (in this corpus almost all of
it, rank 1 holds rare ids and the specials) and is also reported, at equal cost.

Ranking: frequency in GLM-5.3-Flash replies, then frequency in every other text of the database (user turns, tool
outputs, other models' replies: the ``prior``), then id (BPE merge order). Measured held-out: the list is built on
one part of the corpus (by session hash, or by time) and measured on the other part's GLM replies.

Reported per list size:
  coverage      share of reply tokens (by class) that are listed; ``+ctx``: listed or already in the request's own
                history (an upper bound for a per-request extension, not built)
  loss          tokens a drafter round keeps, full vs trimmed: each round's keep is drawn from the recorded keeps
                (DFlash2 by stream class; MTP a = 0.74 / 0.45 / 0.22) and cut at the first unlisted token among its
                accepted drafts (the round's last token is the target's own and needs no listing). Draws ignore the
                text, so unlisted (rare) tokens count as drafted as often as common ones: upper-leaning
  net           tokens / ms against the full head, with the head's saved ms in the round (head GEMV at --gbps), with
                and without the fallback rule (a request drafts over the full head while more than ``--fb-rate`` of
                its last ``--fb-window`` committed tokens, prompt first, are unlisted)

Classes: ``prose`` (reasoning, reply text outside code fences), ``code`` (fenced code; ``content`` / ``oldString``
/ ``newString`` of write / edit calls), ``tool`` (the rest of a tool call), ``cjk`` (a stand-in: CJK text from
local gettext catalogs and zh / ja / ko READMEs, not GLM output). Only aggregates are printed or stored.

    python3 bench/draftvocab.py --tokenizer <dir with tokenizer.json> --json results/draftvocab/study.json \\
        --write-list <tree>/src/tensorfold/families/glm5_next/cuda/draft_vocab.txt
"""

from __future__ import annotations

import argparse
import collections
import gettext
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
from pathlib import Path

VOCAB = 154880                 # the head's rows (tokenizer 154,856 + padding)
HALF = VOCAB // 2              # rank 0's rows: ids < 77,440
TOKENS = 154856
PER_RANK = (8192, 12288, 16384, 24576, 32768)       # M; N = 2M = 16k / 24k / 32k / 48k / 64k
CLASSES = ("prose", "code", "tool", "cjk")
EOS = {"user": 154827, "observation": 154829}
# recorded DFlash2 keeps (tokens committed a round, 1..8) by stream class (bench/lookupsim.py, docs/RESEARCH-NIGHT §1)
KEEPS = {"prose": [0.251, 0.349, 0.164, 0.117, 0.053, 0.040, 0.008, 0.017],
         "code": [0.091, 0.169, 0.128, 0.060, 0.051, 0.091, 0.066, 0.343]}
MTP_CUM = (0.74, 0.45, 0.22)   # MTP: P(drafts 1..j all accepted) (DECODE-PLAN T1: E[T] 2.41)
# round model (bench/lookupsim.py; DECODE-PLAN §1): verify windows of 1..8 rows (ms), drafter and host costs
V = [31.0, 39.0, 45.0, 51.0, 56.0, 62.0, 68.0, 74.0]
HOST = 3.9
MTP_STEP, MTP_ABSORB, BLOCK = 1.68, 2.04, 3.88
CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯＀-￯　-〿]")
CODE_ARGS = {"content", "oldString", "newString", "new_string", "old_string", "patch"}


# -- the corpus ------------------------------------------------------------------------------------------------------
class Tok:
    def __init__(self, path: str) -> None:
        from tokenizers import Tokenizer

        p = Path(path)
        self.t = Tokenizer.from_file(str(p / "tokenizer.json" if p.is_dir() else p))

    def ids(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.t.encode_batch(texts, add_special_tokens=False)]

    def spans(self, pieces: list[tuple[str, str]]) -> tuple[list[int], list[str]]:
        """Pieces (text, class) as one string, tokenized once: (ids, each token's class by its first character)."""

        text, owner, at = "", [], 0
        for s, c in pieces:
            text += s
            owner.append((at, at + len(s), c))
            at += len(s)
        enc = self.t.encode(text, add_special_tokens=False)
        classes, j = [], 0
        for (a, _b) in enc.offsets:
            while j + 1 < len(owner) and a >= owner[j][1]:
                j += 1
            classes.append(owner[j][2])
        return list(enc.ids), classes


def _text_pieces(text: str) -> list[tuple[str, str]]:
    """Reply text split at ``` fences: fenced blocks are code, the rest prose."""

    out, code = [], False
    for part in re.split(r"(```)", text):
        if part == "```":
            out.append((part, "code"))
            code = not code
        elif part:
            out.append((part, "code" if code else "prose"))
    return out


def _tool_pieces(name: str, args: dict) -> list[tuple[str, str]]:
    """A tool call as GLM emits it: markup, name, keys and short values are ``tool``; written file content is ``code``."""

    out = [("<tool_call>" + name, "tool")]
    for k, v in args.items():
        out.append((f"<arg_key>{k}</arg_key><arg_value>", "tool"))
        out.append((v if isinstance(v, str) else json.dumps(v, ensure_ascii=False), "code" if k in CODE_ARGS else "tool"))
        out.append(("</arg_value>", "tool"))
    out.append(("</tool_call>", "tool"))
    return out


def load_sessions(db: str, tok: Tok, providers: set[str]) -> list[dict]:
    """Every message of the database in time order as events {session, time, glm, ids, classes}: GLM-served
    assistant steps as replies (classes per token), everything else (user text, other models' steps, tool outputs,
    which are the next request's prompt) as context text (classes None)."""

    con = sqlite3.connect(f"file:{os.path.expanduser(db)}?mode=ro", uri=True)
    parts = collections.defaultdict(list)
    for mid, data in con.execute("select message_id, data from part order by message_id, id"):
        parts[mid].append(json.loads(data))
    events, plain = [], []
    for mid, sid, t, data in con.execute("select id, session_id, time_created, data from message order by time_created, id"):
        m = json.loads(data)
        ps = parts.get(mid, [])
        role = m.get("role")
        if role == "user":
            plain.append(len(events))
            events.append(dict(session=sid, time=t, glm=False, text="<|user|>\n" + "".join(
                p.get("text", "") for p in ps if p.get("type") == "text")))
            continue
        if role != "assistant":
            continue
        glm = (not providers or m.get("providerID") in providers) and "glm" in str(m.get("modelID", "")).lower()
        pieces = [("".join(p.get("text", "") for p in ps if p.get("type") == "reasoning"), "prose"), ("</think>", "prose")]
        called = False
        for p in ps:
            if p.get("type") == "text" and p.get("text"):
                pieces += _text_pieces("\n" + p["text"])
            elif p.get("type") == "tool":
                called = True
                pieces += _tool_pieces(p.get("tool", ""), (p.get("state") or {}).get("input") or {})
        if glm:
            ids, classes = tok.spans([(s, c) for s, c in pieces if s])
            ids.append(EOS["observation" if called else "user"])
            classes.append(classes[-1] if classes else "prose")
            events.append(dict(session=sid, time=t, glm=True, ids=ids, classes=classes))
        else:
            plain.append(len(events))
            events.append(dict(session=sid, time=t, glm=False, text="<|assistant|>" + "".join(s for s, _ in pieces)))
        for p in ps:                               # tool outputs: the next request's prompt
            if p.get("type") == "tool":
                o = (p.get("state") or {}).get("output")
                if isinstance(o, str) and o:
                    plain.append(len(events))
                    events.append(dict(session=sid, time=t, glm=False,
                                       text="<|observation|>\n<tool_response>" + o[:200000] + "</tool_response>"))
    for i, ids in zip(plain, tok.ids([events[i].pop("text") for i in plain])):
        events[i]["ids"], events[i]["classes"] = ids, None
    return events


def cjk_texts(roots: list[str], limit_chars: int = 3_000_000) -> list[str]:
    """CJK text from local files: gettext catalogs' translations (zh / ja / ko) and CJK READMEs / markdown."""

    texts, total = [], 0

    def add(s: str) -> bool:
        nonlocal total
        if len(CJK_RE.findall(s)) >= max(4, len(s) // 5):
            texts.append(s)
            total += len(s)
        return total < limit_chars

    for root in roots:
        for p in sorted(Path(root).rglob("*")) if Path(root).is_dir() else [Path(root)]:
            try:
                if p.suffix == ".mo" and re.search(r"/(zh|ja|ko)[^/]*/LC_MESSAGES/", str(p)):
                    with open(p, "rb") as f:
                        cat = gettext.GNUTranslations(f)._catalog          # noqa: SLF001 - read only
                    if not add("\n".join(v for k, v in cat.items() if k and isinstance(v, str))):
                        return texts
                elif p.suffix == ".md" and re.search(r"(zh|ja|ko|cn|chinese|japanese)", p.name, re.I):
                    if not add(p.read_text(errors="ignore")):
                        return texts
            except (OSError, ValueError, UnicodeError):
                continue
    return texts


# -- the list -------------------------------------------------------------------------------------------------------
def ranking(glm: collections.Counter, prior: collections.Counter) -> list[int]:
    """Every id with a count: GLM replies' frequency, then the prior's, then id (unlisted ids follow by id)."""

    seen = set(glm) | set(prior)
    return sorted((i for i in seen if i < VOCAB), key=lambda i: (-glm[i], -prior[i], i))


def per_rank(rank_order: list[int], m: int) -> set[int]:
    """The first ``m`` ids of each rank's half in ``rank_order``, then that half's other ids by id (the engine's rule)."""

    out: set[int] = set()
    for lo, hi in ((0, HALF), (HALF, VOCAB)):
        mine = [i for i in rank_order if lo <= i < hi][:m]
        taken = set(mine)
        j = lo
        while len(mine) < m:
            if j not in taken and j < min(hi, TOKENS):
                mine.append(j)
            j += 1
        out |= set(mine)
    return out


def global_top(rank_order: list[int], n: int) -> set[int]:
    return set(rank_order[:n])


def cost_rows(keep: set[int]) -> int:
    r0 = sum(1 for i in keep if i < HALF)
    return -(-max(r0, len(keep) - r0) // 64) * 64


# -- measurements ----------------------------------------------------------------------------------------------------
def coverage(replies: list[dict], keep: set[int], ctx: dict | None = None) -> dict:
    hit, tot = collections.Counter(), collections.Counter()
    for k, r in enumerate(replies):
        extra = ctx[k] if ctx is not None else ()
        for i, c in zip(r["ids"], r["classes"]):
            tot[c] += 1
            hit[c] += i in keep or i in extra
    out = {c: hit[c] / tot[c] for c in tot}
    out["all"] = sum(hit.values()) / max(1, sum(tot.values()))
    return out


def _draw(rng: random.Random, arm: str, cls: str) -> int:
    if arm == "mtp":
        u = rng.random()
        return 1 + sum(1 for p in MTP_CUM if u < p)
    ks = KEEPS["code" if cls in ("code", "tool") else "prose"]
    return rng.choices(range(1, len(ks) + 1), ks)[0]


def round_ms(arm: str, cls: str) -> tuple[float, int]:
    """(ms of a typical round, head evaluations in it): MTP 3 drafts (absorb + 2 chained steps, 4-row window);
    DFlash2 one block (7 head rows in one pass) and the window the class usually verifies (prose 4, code 8 rows)."""

    if arm == "mtp":
        return V[3] + MTP_ABSORB + 2 * MTP_STEP + HOST, 3
    rows = 8 if cls in ("code", "tool") else 4
    return V[rows - 1] + BLOCK + HOST, 1


def simulate(streams: list[tuple[list[int], list[str], list[int]]], keep: set[int] | None, arm: str, save_ms: float,
             seeds: int, fb_rate: float = 1.0, fb_window: int = 256) -> dict:
    """Per class: rounds, tokens, ms with the full head (``keep`` None) or the list, and the fallback rule.
    Stream: (reply ids, classes, the request's last prompt tokens)."""

    acc = collections.defaultdict(lambda: [0, 0.0, 0.0, 0])     # class -> rounds, tokens, ms, fallback rounds
    for seed in range(seeds):
        rng = random.Random(seed)
        for ids, classes, prompt in streams:
            recent = collections.deque((t not in keep for t in prompt[-fb_window:]) if keep else (), maxlen=fb_window)
            misses = sum(recent)
            p = 0
            while p < len(ids):
                cls = classes[p]
                k = min(_draw(rng, arm, cls), len(ids) - p)
                ms, evals = round_ms(arm, cls)
                full = keep is None or (len(recent) and misses > fb_rate * len(recent))
                if not full:
                    ms -= evals * save_ms
                    for j in range(k - 1):               # accepted drafts: positions p .. p + k - 2
                        if ids[p + j] not in keep:
                            k = j + 1                    # the target emits this token itself
                            break
                a = acc[cls]
                a[0] += 1
                a[1] += k
                a[2] += ms
                a[3] += bool(full and keep is not None)
                if keep is not None:
                    for t in ids[p:p + k]:
                        if len(recent) == recent.maxlen:
                            misses -= recent[0]
                        recent.append(t not in keep)
                        misses += t not in keep
                p += k
    return {c: dict(rounds=v[0], tpr=v[1] / v[0], tok_ms=v[1] / v[2], fallback=v[3] / v[0]) for c, v in acc.items()}


def head_mb(rows: int, hidden: int = 4096) -> float:
    """A 4-bit group-64 head of ``rows`` rows: words + bf16 scales and biases, MB."""

    return rows * hidden * 0.5 / 1e6 + rows * (hidden // 64) * 4 / 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--source", default="opencode:~/.local/share/opencode/opencode.db")
    ap.add_argument("--providers", default="",
                    help="comma-separated opencode provider ids whose GLM replies count (default: every provider)")
    ap.add_argument("--cjk", default="/usr/share/locale,/usr/share/doc",
                    help="comma-separated roots for CJK text (gettext catalogs, CJK markdown)")
    ap.add_argument("--per-rank", default=",".join(map(str, PER_RANK)), help="rows a rank (M; N = 2M)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gbps", type=float, default=200.0, help="the head GEMV's DRAM rate (qmm: 190-215 GB/s)")
    ap.add_argument("--fb-rate", type=float, default=0.03)
    ap.add_argument("--fb-window", type=int, default=256)
    ap.add_argument("--write-list", default=None, help="write the ranking built on the whole corpus")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    tok = Tok(a.tokenizer)
    if not a.source.startswith("opencode:"):
        sys.exit("--source opencode:PATH")
    events = load_sessions(a.source.split(":", 1)[1], tok, {p for p in a.providers.split(",") if p})
    reps = [e for e in events if e["glm"]]
    cjk = [dict(session="cjk", time=0, glm=True, ids=i, classes=c)
           for i, c in (tok.spans([(s, "cjk")]) for s in cjk_texts([r for r in a.cjk.split(",") if r]))]
    Ms = [int(s) for s in a.per_rank.split(",")]
    n_tok = collections.Counter(c for r in reps for c in r["classes"])
    prior_all = sum(len(e["ids"]) for e in events if not e["glm"])
    print(f"{len(reps)} GLM-served steps, {sum(n_tok.values())} reply tokens "
          f"({', '.join(f'{c} {n_tok[c]}' for c in ('prose', 'code', 'tool'))}) in "
          f"{len({r['session'] for r in reps})} sessions; prior: {prior_all} tokens of other text in "
          f"{len({e['session'] for e in events})} sessions; CJK stand-in {sum(len(r['ids']) for r in cjk)} tokens")

    # each GLM reply's history (every earlier event of its session) as a set, and its last 256 prompt tokens
    hist: dict[str, set] = collections.defaultdict(set)
    tail: dict[str, list] = collections.defaultdict(list)
    ctx, prompt = {}, {}
    for e in events:
        if e["glm"]:
            ctx[id(e)] = set(hist[e["session"]])
            prompt[id(e)] = list(tail[e["session"]][-a.fb_window:])
        hist[e["session"]].update(e["ids"])
        tail[e["session"]] = (tail[e["session"]] + e["ids"])[-a.fb_window:]

    def hsh(s: str) -> int:
        return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)

    cut = reps[len(reps) * 3 // 4]["time"]
    splits = {
        "session": (lambda e: hsh(e["session"]) % 4 != 0),
        "time": (lambda e: e["time"] < cut),
        "in-sample": (lambda e: True),
    }
    result: dict = {"steps": len(reps), "tokens": dict(n_tok), "prior_tokens": prior_all,
                    "cjk_tokens": sum(len(r["ids"]) for r in cjk), "per_rank": Ms, "coverage": {}, "net": {}}
    for name, train_of in splits.items():
        train = [e for e in events if train_of(e)]
        test = [r for r in reps if not train_of(r)] if name != "in-sample" else reps
        glm = collections.Counter(i for e in train if e["glm"] for i in e["ids"])
        prior = collections.Counter(i for e in train if not e["glm"] for i in e["ids"])
        order = ranking(glm, prior)
        order_glm_only = ranking(glm, collections.Counter())
        tctx = [ctx[id(r)] for r in test]
        print(f"\ncoverage, {name} split: built on {sum(1 for e in train if e['glm'])} GLM steps "
              f"({len(glm)} distinct ids) + prior ({len(prior)} distinct ids), measured on {len(test)} steps")
        print(f"  {'M/rank':>7} {'N':>6} {'list':14s} " + " ".join(f"{c:>7s}" for c in CLASSES + ("all",))
              + "  cost rows")
        for m in Ms:
            rows = {"per-rank": per_rank(order, m), "per-rank GLM": per_rank(order_glm_only, m),
                    "global 2M": global_top(order, 2 * m)}
            for label, keep in rows.items():
                cov = coverage(test, keep)
                cov["cjk"] = coverage(cjk, keep)["cjk"]
                result["coverage"][f"{name}/{m}/{label}"] = dict(cov, cost_rows=cost_rows(keep))
                print(f"  {m:7d} {2 * m:6d} {label:14s} " + " ".join(f"{cov.get(c, float('nan')):7.2%}"
                                                                       for c in CLASSES + ("all",))
                      + f"  {cost_rows(keep):6d}")
            cov = coverage(test, rows["per-rank"], tctx)
            result["coverage"][f"{name}/{m}/per-rank+ctx"] = cov
            print(f"  {m:7d} {2 * m:6d} {'per-rank +ctx':14s} " + " ".join(f"{cov.get(c, float('nan')):7.2%}"
                                                                            for c in CLASSES[:3]) + "       -"
                  + f" {cov['all']:7.2%}")
    # acceptance and net: session split held out, list built on the rest; the CJK stand-in as prose streams
    train_of = splits["session"]
    train = [e for e in events if train_of(e)]
    order = ranking(collections.Counter(i for e in train if e["glm"] for i in e["ids"]),
                    collections.Counter(i for e in train if not e["glm"] for i in e["ids"]))
    test = [r for r in reps if not train_of(r)]
    streams = [(r["ids"], r["classes"], prompt[id(r)]) for r in test] + [(r["ids"], r["classes"], r["ids"][:0])
                                                                          for r in cjk]
    full_mb = head_mb(HALF)
    print(f"\nhead a rank: full {HALF} rows {full_mb:.1f} MB = {full_mb / a.gbps:.3f} ms at {a.gbps:.0f} GB/s")
    print("drafter rounds, session split held out: tokens a round full -> trimmed, net tokens/ms vs full "
          f"(fallback: > {a.fb_rate:.0%} of the last {a.fb_window} tokens unlisted -> full head)")
    for arm in ("mtp", "dflash"):
        base = simulate(streams, None, arm, 0.0, a.seeds)
        for m in Ms:
            keep = per_rank(order, m)
            save = (full_mb - head_mb(cost_rows(keep))) / a.gbps
            nofb = simulate(streams, keep, arm, save, a.seeds)
            fb = simulate(streams, keep, arm, save, a.seeds, a.fb_rate, a.fb_window)
            cells = []
            for c in CLASSES:
                if c not in base:
                    continue
                b, x, y = base[c], nofb[c], fb[c]
                result["net"][f"{arm}/{m}/{c}"] = dict(tpr_full=b["tpr"], tpr_trim=x["tpr"],
                                                       net=x["tok_ms"] / b["tok_ms"] - 1,
                                                       net_fallback=y["tok_ms"] / b["tok_ms"] - 1,
                                                       fallback_rounds=y["fallback"], save_ms=save)
                cells.append(f"{c} {b['tpr']:.2f}->{x['tpr']:.2f} net {x['tok_ms'] / b['tok_ms'] - 1:+.1%} "
                             f"fb {y['tok_ms'] / b['tok_ms'] - 1:+.1%} ({y['fallback']:.0%} full)")
            print(f"  {arm:6s} M {m:5d} (saves {save:.3f} ms an eval)  " + "  ".join(cells))
    if a.write_list:
        glm = collections.Counter(i for r in reps for i in r["ids"])
        prior = collections.Counter(i for e in events if not e["glm"] for i in e["ids"])
        order = ranking(glm, prior)
        hdr = ("# GLM-5.3-Flash draft vocabulary ranking (patches/0420, GLM53_TF_DRAFT_VOCAB=<N>: each rank drafts over "
               "the first N / 2 ids of its half of this list, then that half's other ids by id). 16 ids a line.\n"
               f"# {len(order)} ids, most frequent first: frequency in {len(reps)} GLM-5.3-Flash agent steps "
               f"({sum(n_tok.values())} reply tokens), then in {prior_all} tokens of the other text of the local opencode "
               "database, then id. Written by bench/draftvocab.py (docs/DRAFT-VOCAB.md).\n")
        lines = [" ".join(str(i) for i in order[j:j + 16]) for j in range(0, len(order), 16)]
        Path(a.write_list).write_text(hdr + "\n".join(lines) + "\n")
        print(f"\nwrote {a.write_list}: {len(order)} ids ({sum(1 for i in glm)} seen in GLM replies)")
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
