#!/usr/bin/env python3
"""KL divergence and top-1 agreement of a candidate weight configuration against a reference one, teacher-forced over
a fixed ~200k-token corpus: the standard quantization-quality measurement (llama.cpp's ``--kl-divergence``), far
more sensitive than MMLU. TensorFold serves no logprobs, so the distributions come from patches/0430's draft dump
(GLM53_TF_DRAFT_DUMP): for every prefill row the engine writes the target's top-32 next-token log-probabilities
over the whole vocabulary (both ranks' halves merged). Both configurations prefill the SAME token ids (the same
texts through the same tokenizer; checked here by hashing each document's ids), so row i of a document in the two
dumps is the same position with the same context, and the two distributions are compared per position.

Steps:

  corpus   fetch a fixed mixture into bench/data/divergence/corpus.jsonl (cached; ~200k tokens by default), from
           permissive / attribution-only datasets through the HF datasets-server rows API (stdlib only):
             wiki   wikimedia/wikipedia 20231101.en        CC BY-SA 4.0 / GFDL (cached for evaluation, not shipped)
             code   bigcode/humanevalpack (6 languages)    MIT
             chat   HuggingFaceH4/ultrachat_200k test_sft  MIT (multi-turn, sent as chat: the template's tokens too)
             math   openai/gsm8k test                      MIT (worked solutions, as chat)
  run      send every document to a server started with the dump on (below): text as a raw completion, chats
           through the chat template (thinking off), max_tokens 1, one at a time; writes <out>/run.json (the
           order, the server's prompt_tokens for each document).
  compare  --ref <dump dir(s)> --cand <dump dir(s)>: per position KL(P_ref || P_cand), top-1 agreement, the true
           next token's log-probability under both, and their distributions (overall, per source, per document),
           with 95% intervals from a bootstrap over documents.

Server settings for a run (both configurations identical except the weight knobs under test):

    GLM53_TF_DRAFT_DUMP=/sessions/div-<label>  GLM53_TF_DRAFT_DUMP_TAPS=0  GLM53_TF_DRAFT_DUMP_WHAT=prefill
    GLM53_TF_SESSION_GIB=0  GLM53_TF_BATCH_SESSIONS=0  GLM53_TF_PREFIX_SHARE=0     (every document prefilled whole)
    GLM53_TF_KV_POOL_TOKENS=131072  CONTEXT=131072                                  (fits bf16, see QUALITY-PLAN)

(8,388 B a dumped token without taps: ~1.7 GB a configuration for 200k tokens.) An A/A pair (the same
configuration twice) must give KL 0 and agreement 1 exactly: the engine is deterministic.

The KL estimator. Each side has the top 32 tokens' log-probabilities and nothing else. Over the reference's top-32
set S plus one "rest" bucket: KL = sum_{i in S} p_i log(p_i / q_i) + p_rest log(p_rest / q_rest). q_i is the
candidate's log-probability when i is in its top 32; otherwise it is only known to be at most the candidate's 32nd
value and at most its tail mass, and that bound is used (an upper bound on q_i: the term is under-estimated).
Merging outcomes never increases a KL divergence, so this is a lower bound of the full-vocabulary KL, and it is
exact up to the tail when both top-32 lists cover nearly all the mass (GLM-5.3-Flash's top 32 hold > 99% of the
probability at most positions; the report gives the share of positions with a reference token missing from the
candidate's list and the mean reference tail mass).

    python3 bench/divergence.py corpus
    python3 bench/divergence.py run --base http://127.0.0.1:8000 --model GLM-5.3-Flash-EXL3 --out results/Q8/div-bf16
    python3 bench/divergence.py compare --ref /sessions/div-bf16/* --cand /sessions/div-q4mse/* --out results/Q8/kl-q4mse.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "divergence"
ROWS_API = "https://datasets-server.huggingface.co/rows"

# (source, tokens wanted, chars a token (estimate for the budget only; ``run`` reports the real counts))
MIX = {"wiki": (0.40, 4.2), "code": (0.25, 3.0), "chat": (0.25, 4.0), "math": (0.10, 3.4)}
LICENSES = {"wiki": "CC BY-SA 4.0 / GFDL (wikimedia/wikipedia 20231101.en)",
            "code": "MIT (bigcode/humanevalpack)",
            "chat": "MIT (HuggingFaceH4/ultrachat_200k)",
            "math": "MIT (openai/gsm8k)"}


# -- corpus ------------------------------------------------------------------------------------------------------------
def _get_json(url: str, tries: int = 10) -> dict:
    import urllib.error

    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.loads(r.read())
        except Exception as exc:  # noqa: BLE001 - rate limits / transient errors: back off and retry
            if i == tries - 1:
                raise
            wait = min(120.0, 5.0 * 2 ** i)
            if isinstance(exc, urllib.error.HTTPError) and exc.headers.get("Retry-After", "").isdigit():
                wait = max(wait, float(exc.headers["Retry-After"]))
            print(f"  {type(exc).__name__}: {exc}; retry in {wait:.0f}s", file=sys.stderr, flush=True)
            time.sleep(wait)
    raise AssertionError


def rows(dataset: str, config: str, split: str, offset: int, length: int) -> tuple[list[dict], int]:
    q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset,
                                "length": length})
    d = _get_json(f"{ROWS_API}?{q}")
    time.sleep(0.5)
    return [r["row"] for r in d["rows"]], int(d["num_rows_total"])


def _wiki(budget: int) -> list[dict]:
    out, have, i = [], 0, 0
    total = rows("wikimedia/wikipedia", "20231101.en", "train", 0, 1)[1]
    while have < budget:
        i += 1
        off = (i * 1_000_003) % total                 # fixed, spread over the dump
        for r in rows("wikimedia/wikipedia", "20231101.en", "train", off, 4)[0]:
            text = r["text"].strip()
            if len(text) < 3000:
                continue
            text = f"{r['title']}\n\n{text[:14000]}"
            out.append({"source": "wiki", "kind": "text", "text": text, "ref": r["url"]})
            have += len(text)
            break
    return out


def _code(budget: int) -> list[dict]:
    out, have = [], 0
    langs = ["python", "js", "java", "go", "cpp", "rust"]
    per = budget // len(langs)
    for lang in langs:
        got, off = 0, 0
        while got < per:
            batch, total = rows("bigcode/humanevalpack", lang, "test", off, 20)
            if not batch:
                break
            for r in batch:
                text = (r.get("prompt") or "") + (r.get("canonical_solution") or "") + "\n" + (r.get("test") or "")
                out.append({"source": "code", "kind": "text", "text": text, "ref": f"{lang}:{r['task_id']}"})
                got += len(text)
                if got >= per:
                    break
            off += 20 * 7                               # every 7th page: spread over the 164 tasks
            if off >= total:
                off = off % total + 1
        have += got
    return out


def _chat(budget: int) -> list[dict]:
    out, have, off = [], 0, 0
    while have < budget:
        batch, total = rows("HuggingFaceH4/ultrachat_200k", "default", "test_sft", off, 10)
        for r in batch:
            msgs = [{"role": m["role"], "content": m["content"]} for m in r["messages"]][:8]
            while msgs and msgs[-1]["role"] != "assistant":
                msgs.pop()
            size = sum(len(m["content"]) for m in msgs)
            if not msgs or size > 16000:
                continue
            out.append({"source": "chat", "kind": "chat", "messages": msgs, "ref": r["prompt_id"]})
            have += size
            if have >= budget:
                break
        off = (off + 997) % total
    return out


def _math(budget: int) -> list[dict]:
    out, have, off = [], 0, 0
    while have < budget:
        batch, total = rows("openai/gsm8k", "main", "test", off, 5)
        if not batch:
            off = 0
            continue
        msgs = []
        for r in batch:
            msgs += [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["answer"]}]
        size = sum(len(m["content"]) for m in msgs)
        out.append({"source": "math", "kind": "chat", "messages": msgs, "ref": f"gsm8k test {off}-{off + 4}"})
        have += size
        off = (off + 53) % max(1, total - 5)
    return out


def corpus(a) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    path = Path(a.path)
    if path.exists() and not a.force:
        docs = [json.loads(line) for line in path.open()]
        print(f"{path}: cached, {len(docs)} documents")
        return
    docs = []
    for src, (share, cpt) in MIX.items():
        budget = int(a.tokens * share * cpt)
        got = {"wiki": _wiki, "code": _code, "chat": _chat, "math": _math}[src](budget)
        print(f"  {src}: {len(got)} documents, ~{sum(len(d.get('text') or '') + sum(len(m['content']) for m in d.get('messages') or []) for d in got) / cpt:,.0f} tokens",
              flush=True)
        docs += got
    with path.open("w") as f:
        for i, d in enumerate(docs):
            d["id"] = i
            d["license"] = LICENSES[d["source"]]
            f.write(json.dumps(d) + "\n")
    (DATA / "SOURCES.md").write_text("bench/divergence.py corpus: " + "; ".join(f"{k}: {v}" for k, v in LICENSES.items())
                                     + ". Evaluation cache only; not redistributed.\n")
    print(f"{path}: {len(docs)} documents")


# -- run -------------------------------------------------------------------------------------------------------------
def _post(base: str, path: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run(a) -> None:
    docs = [json.loads(line) for line in Path(a.corpus).open()]
    if a.limit:
        docs = docs[: a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"label": a.label or out.name, "base": a.base, "model": a.model, "corpus": str(a.corpus),
           "corpus_sha": hashlib.sha256(Path(a.corpus).read_bytes()).hexdigest()[:16], "docs": []}
    t0 = time.time()
    tokens = 0
    for d in docs:
        if d["kind"] == "chat":
            body = {"model": a.model, "messages": d["messages"], "max_tokens": 1, "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False}}
            got = _post(a.base, "/v1/chat/completions", body, a.timeout)
        else:
            body = {"model": a.model, "prompt": d["text"], "max_tokens": 1, "temperature": 0}
            got = _post(a.base, "/v1/completions", body, a.timeout)
        n = int((got.get("usage") or {}).get("prompt_tokens") or 0)
        tokens += n
        rec["docs"].append({"id": d["id"], "source": d["source"], "prompt_tokens": n, "t": round(time.time(), 3)})
        if len(rec["docs"]) % 20 == 0:
            print(f"  {len(rec['docs'])}/{len(docs)} documents, {tokens:,} tokens, {time.time() - t0:.0f} s", flush=True)
    rec["tokens"] = tokens
    rec["wall_s"] = round(time.time() - t0, 1)
    (out / "run.json").write_text(json.dumps(rec, indent=1))
    print(f"{len(docs)} documents, {tokens:,} prompt tokens in {rec['wall_s']} s -> {out / 'run.json'}")


# -- compare -----------------------------------------------------------------------------------------------------------
def _docs(dirs: list[str]):
    """The dump's documents (train/dumpdata.build_docs), keyed by the hash of their token ids."""

    sys.path.insert(0, str(HERE.parent / "train"))
    import numpy as np
    from dumpdata import build_docs

    out = {}
    for doc in build_docs(dirs, min_tokens=2):
        if any(seg["kind"] != "p" for _, seg in doc.parts):
            continue                                   # decode rows (a server dumping decode too): prefill only
        toks = doc.tokens()
        key = hashlib.blake2b(np.ascontiguousarray(toks, dtype="<i4").tobytes(), digest_size=16).hexdigest()
        order = max(seg["seg"] + 1e-6 * seg["start"] for _, seg in doc.parts)
        out[key] = (doc, toks, order)
    return out


def _rows(doc, name: str):
    import numpy as np

    parts = []
    for dd, seg in doc.parts:
        parts.append(np.asarray(dd.raw(seg, name)))
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


def position_stats(lp_r, ids_r, lp_c, ids_c, nxt=None):
    """Per position (numpy, [T, k]): KL(ref || cand) over the ref's top-k + a tail bucket, top-1 agreement, whether
    every ref top-k id is in the cand's top k, the ref's tail mass, and the log-probabilities of the true next token
    (``nxt`` [T], -1 where unknown) under both (NaN where it is outside a top-k)."""

    import numpy as np

    lp_r = lp_r.astype(np.float64)
    lp_c = lp_c.astype(np.float64)
    T, k = lp_r.shape
    p = np.exp(lp_r)
    # candidate log-probs of the ref's ids: exact where listed, else bounded by its k-th value and its tail mass
    match = ids_r[:, :, None] == ids_c[:, None, :]                    # [T, k, k]
    found = match.any(-1)
    val = np.where(match, lp_c[:, None, :], -np.inf).max(-1)
    tail_c = np.clip(1.0 - np.exp(lp_c).sum(-1), 1e-12, 1.0)
    bound = np.minimum(lp_c[:, -1], np.log(tail_c))
    lq = np.where(found, val, bound[:, None])
    q = np.exp(lq)
    p_rest = np.clip(1.0 - p.sum(-1), 0.0, 1.0)
    q_rest = np.clip(1.0 - q.sum(-1), 1e-12, 1.0)
    kl = (p * (lp_r - lq)).sum(-1) + np.where(p_rest > 0, p_rest * (np.log(np.maximum(p_rest, 1e-300)) - np.log(q_rest)), 0.0)
    kl = np.maximum(kl, 0.0)
    top1 = ids_r[:, 0] == ids_c[:, 0]
    out = {"kl": kl, "top1": top1, "covered": found.all(-1), "tail_ref": p_rest}
    if nxt is not None:
        def lp_of(lp, ids):
            m = ids == nxt[:, None]
            return np.where(m.any(-1), np.where(m, lp, -np.inf).max(-1), np.nan)
        out["nll_ref"] = -lp_of(lp_r, ids_r)
        out["nll_cand"] = -lp_of(lp_c, ids_c)
    return out


def _bootstrap(per_doc_sum, per_doc_n, reps=2000, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    s, n = np.asarray(per_doc_sum, dtype=np.float64), np.asarray(per_doc_n, dtype=np.float64)
    idx = rng.integers(0, len(s), size=(reps, len(s)))
    means = s[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(stats: list[dict], label: str) -> dict:
    import numpy as np

    kl = np.concatenate([s["kl"] for s in stats])
    top1 = np.concatenate([s["top1"] for s in stats])
    cov = np.concatenate([s["covered"] for s in stats])
    tail = np.concatenate([s["tail_ref"] for s in stats])
    nr = np.concatenate([s["nll_ref"] for s in stats])
    nc = np.concatenate([s["nll_cand"] for s in stats])
    both = ~np.isnan(nr) & ~np.isnan(nc)
    pct = {f"p{q}": float(np.percentile(kl, q)) for q in (50, 90, 95, 99, 99.9)}
    edges = [0, 1e-5, 1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.3, 1, 3, float("inf")]
    hist = {f"{edges[i]:g}-{edges[i + 1]:g}": int(((kl >= edges[i]) & (kl < edges[i + 1])).sum())
            for i in range(len(edges) - 1)}
    return {
        "label": label, "positions": int(kl.size), "documents": len(stats),
        "kl_mean": float(kl.mean()), "kl_mean_ci95": _bootstrap([s["kl"].sum() for s in stats], [s["kl"].size for s in stats]),
        "kl_median": pct["p50"], **{f"kl_{k}": v for k, v in pct.items() if k != "p50"}, "kl_max": float(kl.max()),
        "top1_agreement": float(top1.mean()),
        "top1_ci95": _bootstrap([s["top1"].sum() for s in stats], [s["top1"].size for s in stats]),
        "ref_top32_all_in_cand": float(cov.mean()), "ref_tail_mass_mean": float(tail.mean()),
        "true_token_positions": int(both.sum()),
        "nll_ref_mean": float(nr[both].mean()) if both.any() else None,
        "nll_cand_mean": float(nc[both].mean()) if both.any() else None,
        "ppl_ratio_cand_over_ref": float(math.exp(nc[both].mean() - nr[both].mean())) if both.any() else None,
        "kl_histogram": hist,
    }


def compare(a) -> None:
    import numpy as np

    ref, cand = _docs(a.ref), _docs(a.cand)
    common = sorted(set(ref) & set(cand), key=lambda k: ref[k][2])
    print(f"ref {len(ref)} documents, cand {len(cand)}, common (same token ids) {len(common)}", flush=True)
    if not common:
        raise SystemExit("no document with the same token ids in both dumps")
    sources = {}
    if a.run:
        run_docs = json.loads(Path(a.run).read_text())["docs"]
        ordered = sorted(ref.items(), key=lambda kv: kv[1][2])
        if len(ordered) == len(run_docs) and all(kv[1][0].n == d["prompt_tokens"] for kv, d in zip(ordered, run_docs)):
            sources = {kv[0]: d["source"] for kv, d in zip(ordered, run_docs)}
        else:
            print("  (run.json does not line up with the reference dump's documents: no per-source breakdown)")
    stats, by_src, per_doc = [], {}, []
    for key in common:
        dr, toks, _ = ref[key]
        dc = cand[key][0]
        lp_r, ids_r = _rows(dr, "lp"), _rows(dr, "ids")
        lp_c, ids_c = _rows(dc, "lp"), _rows(dc, "ids")
        nxt = np.full(len(toks), -1, dtype=np.int64)
        nxt[:-1] = toks[1:]
        s = position_stats(lp_r, ids_r, lp_c, ids_c, nxt)
        stats.append(s)
        src = sources.get(key, "all")
        by_src.setdefault(src, []).append(s)
        per_doc.append({"tokens": int(len(toks)), "source": src, "kl_mean": float(s["kl"].mean()),
                        "top1": float(s["top1"].mean())})
    res = {"overall": summarize(stats, a.label), "sources": {k: summarize(v, k) for k, v in sorted(by_src.items())},
           "documents": per_doc, "ref": a.ref, "cand": a.cand}
    o = res["overall"]
    print(f"{o['label']}: {o['positions']:,} positions in {o['documents']} documents")
    print(f"  KL mean {o['kl_mean']:.5f} (95% CI {o['kl_mean_ci95'][0]:.5f}-{o['kl_mean_ci95'][1]:.5f}), median "
          f"{o['kl_median']:.5f}, p99 {o['kl_p99']:.4f}, p99.9 {o['kl_p99.9']:.4f}, max {o['kl_max']:.3f}")
    print(f"  top-1 agreement {o['top1_agreement']:.4%} (95% CI {o['top1_ci95'][0]:.4%}-{o['top1_ci95'][1]:.4%})")
    if o["ppl_ratio_cand_over_ref"] is not None:
        print(f"  true next token (in both top-32, {o['true_token_positions']:,} positions): NLL ref "
              f"{o['nll_ref_mean']:.4f}, cand {o['nll_cand_mean']:.4f}, perplexity ratio {o['ppl_ratio_cand_over_ref']:.4f}")
    print(f"  estimator: ref top-32 all in cand's {o['ref_top32_all_in_cand']:.2%} of positions; ref tail mass "
          f"{o['ref_tail_mass_mean']:.4f}")
    print("  KL histogram: " + ", ".join(f"[{k}) {v}" for k, v in o["kl_histogram"].items()))
    for k, v in res["sources"].items():
        print(f"  {k:<5} {v['positions']:>8,} positions: KL mean {v['kl_mean']:.5f}, p99 {v['kl_p99']:.4f}, "
              f"top-1 {v['top1_agreement']:.4%}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("corpus")
    c.add_argument("--tokens", type=int, default=200_000)
    c.add_argument("--path", default=str(DATA / "corpus.jsonl"))
    c.add_argument("--force", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--base", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--label", default="")
    r.add_argument("--corpus", default=str(DATA / "corpus.jsonl"))
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--timeout", type=float, default=900)
    m = sub.add_parser("compare")
    m.add_argument("--ref", nargs="+", required=True, help="the reference configuration's dump folder(s)")
    m.add_argument("--cand", nargs="+", required=True, help="the candidate's dump folder(s)")
    m.add_argument("--run", default="", help="the reference's run.json (per-source breakdown)")
    m.add_argument("--label", default="cand vs ref")
    m.add_argument("--out", default="")
    a = p.parse_args()
    {"corpus": corpus, "run": run, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
