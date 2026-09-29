#!/usr/bin/env python3
"""Full MMLU (57 subjects, 14,042 test questions) against a GLM-5.3-Flash endpoint, in quality.py's format.

The prompt is bench/quality.py's MMLU-200 prompt (a user turn: "The following is a multiple choice question about
<subject>. ... Answer with the letter of the correct option only."), chat completions, thinking off, greedy
(temperature 0), max_tokens 8, and the reply's first standalone A-D letter is the answer (none: wrong). So MMLU-200
is a stratified sample of exactly this run's items (bench/data/mmlu200.jsonl), and the two agree up to sampling.

--shots 5 adds the subject's 5 dev questions first, as earlier user / assistant turns (the assistant answering with
the letter only), so the few-shot block is a prefix every question of a subject shares (sessions can resume it).
--shots-format inline puts them in the one user turn instead, as "Answer: X" lines (the original MMLU layout).

Data: HF ``cais/mmlu`` (MIT), config ``all``, splits ``test`` (14,042) and ``dev`` (285), fetched once through the
HF datasets-server rows API (stdlib only, no ``datasets``) into bench/data/mmlu/{test,dev}.jsonl.

Runs are resumable: every answer is appended to <out>/answers.jsonl as it arrives; a restart skips the questions
already there (same shots / format; a mismatch refuses to mix). --limit / --subjects for smoke runs.

Output (<out>/summary.json and the console): overall (micro) accuracy with a 95% Wilson interval, the macro
average over subjects, per-category (STEM / humanities / social sciences / other) and per-subject accuracy, the
number of replies without a letter, wall time and questions a second.

    python3 bench/mmlu_full.py fetch
    python3 bench/mmlu_full.py run --base http://127.0.0.1:8000 --model GLM-5.3-Flash-EXL3 --out results/Q8/mmlu-q4mse
    python3 bench/mmlu_full.py run ... --shots 5 --out results/Q8/mmlu5-q4mse
    python3 bench/mmlu_full.py compare results/Q8/mmlu-q4mse results/Q8/mmlu-q8       # paired: McNemar, per subject

Time at our speeds (docs/QUALITY-PLAN.md section 4): MMLU-200 + the refusal probe took 130-185 s at concurrency 1
(~0.55-0.75 s a question). 0-shot at concurrency 4 (4 batch slots): ~0.2-0.3 s a question aggregate, so ~50-70 min
for 14,042; 5-shot prompts are ~6x longer (~900 tokens mean): ~2-4 h, less when the shared few-shot prefix resumes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LETTERS = "ABCD"
DATA = Path(__file__).parent / "data" / "mmlu"
ROWS_API = "https://datasets-server.huggingface.co/rows"
ANSWER_RE = re.compile(r"\b([ABCD])\b")

# hendrycks/test categories.py: subject -> subcategory -> category
SUBCATEGORY = {
    "abstract_algebra": "math", "anatomy": "health", "astronomy": "physics", "business_ethics": "business",
    "clinical_knowledge": "health", "college_biology": "biology", "college_chemistry": "chemistry",
    "college_computer_science": "computer science", "college_mathematics": "math", "college_medicine": "health",
    "college_physics": "physics", "computer_security": "computer science", "conceptual_physics": "physics",
    "econometrics": "economics", "electrical_engineering": "engineering", "elementary_mathematics": "math",
    "formal_logic": "philosophy", "global_facts": "other", "high_school_biology": "biology",
    "high_school_chemistry": "chemistry", "high_school_computer_science": "computer science",
    "high_school_european_history": "history", "high_school_geography": "geography",
    "high_school_government_and_politics": "politics", "high_school_macroeconomics": "economics",
    "high_school_mathematics": "math", "high_school_microeconomics": "economics", "high_school_physics": "physics",
    "high_school_psychology": "psychology", "high_school_statistics": "math", "high_school_us_history": "history",
    "high_school_world_history": "history", "human_aging": "health", "human_sexuality": "culture",
    "international_law": "law", "jurisprudence": "law", "logical_fallacies": "philosophy",
    "machine_learning": "computer science", "management": "business", "marketing": "business",
    "medical_genetics": "health", "miscellaneous": "other", "moral_disputes": "philosophy",
    "moral_scenarios": "philosophy", "nutrition": "health", "philosophy": "philosophy", "prehistory": "history",
    "professional_accounting": "other", "professional_law": "law", "professional_medicine": "health",
    "professional_psychology": "psychology", "public_relations": "politics", "security_studies": "politics",
    "sociology": "culture", "us_foreign_policy": "politics", "virology": "health", "world_religions": "philosophy",
}
CATEGORY = {
    "STEM": {"physics", "chemistry", "biology", "computer science", "math", "engineering"},
    "humanities": {"history", "philosophy", "law"},
    "social sciences": {"politics", "culture", "economics", "geography", "psychology"},
    "other": {"other", "business", "health"},
}


def category(subject: str) -> str:
    sub = SUBCATEGORY.get(subject, "other")
    return next(c for c, subs in CATEGORY.items() if sub in subs)


# -- data ------------------------------------------------------------------------------------------------------------
def _get_json(url: str, tries: int = 10) -> dict:
    """GET JSON, backing off on rate limits (the datasets-server answers 429 to fast anonymous paging)."""

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


def _parquet_rows(dataset: str, config: str, split: str) -> list[dict] | None:
    """The split's parquet file from the dataset repo when pyarrow is installed (one request), else None."""

    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    import io

    url = f"https://huggingface.co/datasets/{dataset}/resolve/main/{config}/{split}-00000-of-00001.parquet"
    with urllib.request.urlopen(url, timeout=300) as r:
        return pq.read_table(io.BytesIO(r.read())).to_pylist()


def fetch_rows(dataset: str, config: str, split: str, page: int = 100, pause: float = 0.5) -> list[dict]:
    """Every row of a split: the parquet file with pyarrow, else the datasets-server rows API (stdlib only, paced)."""

    got = _parquet_rows(dataset, config, split)
    if got is not None:
        return got
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset,
                                    "length": page})
        d = _get_json(f"{ROWS_API}?{q}")
        total = d["num_rows_total"]
        got = [r["row"] for r in d["rows"]]
        if not got:
            break
        rows += got
        offset += len(got)
        print(f"\r  {dataset} {split}: {offset:,} / {total:,}", end="", file=sys.stderr, flush=True)
        time.sleep(pause)
    print(file=sys.stderr)
    return rows


def fetch(force: bool = False) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for split, n in (("test", 14042), ("dev", 285)):
        path = DATA / f"{split}.jsonl"
        if path.exists() and not force:
            print(f"{path}: cached ({sum(1 for _ in path.open())} rows)")
            continue
        rows = fetch_rows("cais/mmlu", "all", split)
        if len(rows) != n:
            raise SystemExit(f"cais/mmlu {split}: {len(rows)} rows, expected {n}")
        with path.open("w") as f:
            for i, r in enumerate(rows):
                f.write(json.dumps({"i": i, "subject": r["subject"], "question": r["question"],
                                    "choices": r["choices"], "answer": int(r["answer"])}) + "\n")
        print(f"{path}: {len(rows)} rows")
    (DATA / "SOURCE.md").write_text("cais/mmlu (config all), MIT licence, https://huggingface.co/datasets/cais/mmlu; "
                                     "fetched by bench/mmlu_full.py through the HF datasets-server rows API.\n")


def load(split: str) -> list[dict]:
    path = DATA / f"{split}.jsonl"
    if not path.exists():
        fetch()
    return [json.loads(line) for line in path.open() if line.strip()]


# -- prompts ---------------------------------------------------------------------------------------------------------
def question_text(q: dict) -> str:
    """bench/quality.py's MMLU prompt, character for character."""

    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(q["choices"]))
    return (f"The following is a multiple choice question about {q['subject'].replace('_', ' ')}.\n\n"
            f"{q['question']}\n{opts}\n\nAnswer with the letter of the correct option only.")


def messages(q: dict, dev: list[dict], shots: int, fmt: str) -> list[dict]:
    examples = dev[:shots]
    if not examples:
        return [{"role": "user", "content": question_text(q)}]
    if fmt == "chat":
        out = []
        for e in examples:
            out += [{"role": "user", "content": question_text(e)}, {"role": "assistant", "content": LETTERS[e["answer"]]}]
        return out + [{"role": "user", "content": question_text(q)}]
    subj = q["subject"].replace("_", " ")
    blocks = []
    for e in examples + [q]:
        opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(e["choices"]))
        blocks.append(f"{e['question']}\n{opts}\nAnswer:" + (f" {LETTERS[e['answer']]}" if e is not q else ""))
    text = (f"The following are multiple choice questions (with answers) about {subj}.\n\n" + "\n\n".join(blocks)
            + "\n\nAnswer with the letter of the correct option only.")
    return [{"role": "user", "content": text}]


def chat(base: str, model: str, msgs: list[dict], max_tokens: int, extra: dict, timeout: float) -> str:
    body = {"model": model, "messages": msgs, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    body.update(extra)
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"].get("content") or ""


# -- statistics ------------------------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def summarize(rows: list[dict], meta: dict) -> dict:
    by_subj: dict[str, list[int]] = {}
    for r in rows:
        s = by_subj.setdefault(r["subject"], [0, 0])
        s[0] += int(r["correct"])
        s[1] += 1
    k, n = sum(v[0] for v in by_subj.values()), sum(v[1] for v in by_subj.values())
    lo, hi = wilson(k, n)
    cats: dict[str, list[int]] = {}
    for subj, (a, b) in by_subj.items():
        c = cats.setdefault(category(subj), [0, 0])
        c[0] += a
        c[1] += b
    macro = sum(a / b for a, b in by_subj.values()) / max(1, len(by_subj))
    return {**meta, "n": n, "correct": k, "accuracy": k / max(1, n), "ci95": [lo, hi],
            "macro_accuracy": macro, "no_letter": sum(1 for r in rows if r["got"] < 0),
            "categories": {c: {"accuracy": a / b, "n": b, "ci95": list(wilson(a, b))} for c, (a, b) in sorted(cats.items())},
            "subjects": {s: {"accuracy": a / b, "n": b, "ci95": list(wilson(a, b))} for s, (a, b) in sorted(by_subj.items())}}


def print_summary(s: dict) -> None:
    print(f"MMLU {s.get('label') or ''} ({s['shots']}-shot {s['shots_format']}): {s['accuracy']:.4f} "
          f"({s['correct']:,}/{s['n']:,}), 95% CI {s['ci95'][0]:.4f}-{s['ci95'][1]:.4f}; macro over "
          f"{len(s['subjects'])} subjects {s['macro_accuracy']:.4f}; no letter in {s['no_letter']} replies")
    for c, v in s["categories"].items():
        print(f"  {c:<16} {v['accuracy']:.4f}  (n {v['n']:,}, {v['ci95'][0]:.3f}-{v['ci95'][1]:.3f})")
    worst = sorted(s["subjects"].items(), key=lambda kv: kv[1]["accuracy"])
    print("  lowest subjects: " + ", ".join(f"{k} {v['accuracy']:.3f}" for k, v in worst[:8]))


# -- run -------------------------------------------------------------------------------------------------------------
def run(a) -> None:
    test, dev = load("test"), load("dev")
    dev_by: dict[str, list[dict]] = {}
    for d in dev:
        dev_by.setdefault(d["subject"], []).append(d)
    items = test
    if a.subjects:
        want = set(a.subjects.split(","))
        items = [q for q in items if q["subject"] in want]
    if a.limit:
        items = items[: a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"label": a.label or out.name, "base": a.base, "model": a.model, "shots": a.shots,
            "shots_format": a.shots_format, "max_tokens": a.max_tokens, "extra": json.loads(a.extra),
            "prompt": "bench/quality.py MMLU prompt, thinking off, greedy, first A-D letter"}
    mpath = out / "meta.json"
    if mpath.exists():
        old = json.loads(mpath.read_text())
        for k in ("shots", "shots_format", "max_tokens", "extra"):
            if old.get(k) != meta[k]:
                raise SystemExit(f"{out}: an earlier run used {k}={old.get(k)!r}, this one {meta[k]!r}: use another --out")
    mpath.write_text(json.dumps(meta, indent=1))
    apath = out / "answers.jsonl"
    done: dict[int, dict] = {}
    if apath.exists():
        for line in apath.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                                  # a torn last line
            done[r["i"]] = r
    todo = [q for q in items if q["i"] not in done]
    print(f"{len(items):,} questions, {len(done):,} already answered, {len(todo):,} to go "
          f"({a.shots}-shot, concurrency {a.concurrency})", flush=True)
    lock = threading.Lock()
    f = apath.open("a", buffering=1)
    t0 = time.time()
    count = [0, 0]
    extra = meta["extra"]

    def one(q: dict) -> None:
        msgs = messages(q, dev_by.get(q["subject"], []), a.shots, a.shots_format)
        for attempt in range(4):
            try:
                reply = chat(a.base, a.model, msgs, a.max_tokens, extra, a.timeout)
                break
            except Exception as exc:  # noqa: BLE001 - a transient server error: retry, then record it
                if attempt == 3:
                    reply = f"<error {type(exc).__name__}: {exc}>"
                time.sleep(2 * (attempt + 1))
        m = ANSWER_RE.search(reply)
        got = LETTERS.index(m.group(1)) if m else -1
        rec = {"i": q["i"], "subject": q["subject"], "answer": q["answer"], "got": got,
               "correct": got == q["answer"], "reply": reply[:64]}
        with lock:
            f.write(json.dumps(rec) + "\n")
            count[0] += 1
            count[1] += rec["correct"]
            if count[0] % 200 == 0 or count[0] == len(todo):
                el = time.time() - t0
                rate = count[0] / el
                print(f"  {count[0]:,}/{len(todo):,}  {count[1] / count[0]:.4f} so far  {rate:.2f} q/s  "
                      f"eta {(len(todo) - count[0]) / max(rate, 1e-9) / 60:.0f} min", flush=True)

    with ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(one, todo))
    f.close()
    rows = {}
    for line in apath.open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[r["i"]] = r
    keep = [rows[q["i"]] for q in items if q["i"] in rows]
    s = summarize(keep, meta)
    s["wall_s_this_run"] = round(time.time() - t0, 1)
    s["q_per_s_this_run"] = round(len(todo) / max(time.time() - t0, 1e-9), 3)
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print_summary(s)


def compare(a) -> None:
    """Paired comparison of two runs over the questions both answered: the accuracy difference with a 95% interval
    (normal approximation for paired proportions, from the discordant counts), McNemar's exact test, and the
    subjects that moved most."""

    def rows(d):
        out = {}
        for line in (Path(d) / "answers.jsonl").open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[r["i"]] = r
        return out

    A, B = rows(a.a), rows(a.b)
    common = sorted(set(A) & set(B))
    n = len(common)
    b10 = sum(1 for i in common if A[i]["correct"] and not B[i]["correct"])
    b01 = sum(1 for i in common if B[i]["correct"] and not A[i]["correct"])
    same_answer = sum(1 for i in common if A[i]["got"] == B[i]["got"])
    acc_a = sum(A[i]["correct"] for i in common) / max(1, n)
    acc_b = sum(B[i]["correct"] for i in common) / max(1, n)
    d = acc_b - acc_a
    se = math.sqrt(max(b10 + b01 - (b01 - b10) ** 2 / max(1, n), 0)) / max(1, n)
    k, m = min(b10, b01), b10 + b01
    p = min(1.0, 2 * sum(math.comb(m, j) for j in range(k + 1)) / 2 ** m) if m else 1.0
    print(f"{n:,} common questions: A {acc_a:.4f}, B {acc_b:.4f}, B - A {d:+.4f} (95% CI {d - 1.96 * se:+.4f} to "
          f"{d + 1.96 * se:+.4f}); same letter on {same_answer / max(1, n):.2%}; A right / B wrong {b10}, "
          f"B right / A wrong {b01}; McNemar exact p = {p:.3g}")
    per: dict[str, list[int]] = {}
    for i in common:
        s = per.setdefault(A[i]["subject"], [0, 0, 0])
        s[0] += A[i]["correct"]
        s[1] += B[i]["correct"]
        s[2] += 1
    moved = sorted(per.items(), key=lambda kv: abs(kv[1][1] - kv[1][0]), reverse=True)
    print("  largest subject moves (B - A, questions): " + ", ".join(f"{s} {v[1] - v[0]:+d}/{v[2]}" for s, v in moved[:10]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download cais/mmlu test + dev into bench/data/mmlu/")
    f.add_argument("--force", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--base", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True, help="folder: answers.jsonl (resumable), meta.json, summary.json")
    r.add_argument("--label", default="")
    r.add_argument("--shots", type=int, default=0, choices=range(0, 6))
    r.add_argument("--shots-format", default="chat", choices=("chat", "inline"))
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--max-tokens", type=int, default=8)
    r.add_argument("--timeout", type=float, default=900)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--subjects", default="")
    r.add_argument("--extra", default="{}", help="JSON merged into every request body (e.g. tf_knobs)")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    s = sub.add_parser("summary", help="reprint a finished run's summary")
    s.add_argument("out")
    a = p.parse_args()
    if a.cmd == "fetch":
        fetch(a.force)
    elif a.cmd == "run":
        run(a)
    elif a.cmd == "compare":
        compare(a)
    else:
        print_summary(json.loads((Path(a.out) / "summary.json").read_text()))


if __name__ == "__main__":
    main()
