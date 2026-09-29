#!/usr/bin/env python3
"""The shipped draft vocabulary ranking (patches/0420, ``draft_vocab.txt``; ``GLM53_TF_DRAFT_VOCAB=<N>``) built from
public, permissively licensed text only. No GPU, no torch; needs ``tokenizers`` and ``pyarrow``. docs/DRAFT-VOCAB.md has the study behind the list.

The drafters (MTP head, DFlash2) score only listed token ids; each rank takes the first N / 2 ids of its half of
the list (ids < 77,440 on rank 0). What they draft is an agentic coding assistant's reply: prose, fenced code,
tool calls (CJK text is left to the fallback rule). The public stand-in for such replies:

  reply-like    OpenAssistant oasst1 (Apache-2.0), revision fdf72ae0827c1cda404aff25b6603abec9e3399b,
                2023-04-12_oasst_ready.messages.jsonl.gz: English assistant turns (prose, markdown, fenced code)
                SWE-Gym/OpenHands-SFT-Trajectories (MIT), revision 4aaa5a4a4b5861f4799d2336908760c190ac3b17,
                data/train.success.oss-00000-of-00001.parquet: a coding agent's assistant turns, its
                ``<function=...>`` calls rewritten in GLM's tool-call format
                CPython 3.14.7 (PSF-2.0), Python-3.14.7.tar.xz: Lib/**/*.py without tests, Doc/**/*.rst
                denoland/std (MIT) at f834d0223364361169314833e3c7a8f62ce11d58: *.ts, *.md
                BurntSushi/ripgrep 14.1.1 (MIT OR Unlicense): *.rs, *.md, *.toml, *.sh
  prior         the other turns: oasst1 prompter turns (all languages) and non-English assistant turns, SWE-Gym
                user turns and tool outputs

Every download is pinned by URL and checked by sha256. Ranking: the chat / tool-call format tokens (FORMAT below,
which a reply emits and the drafter must be able to propose), then frequency in the reply-like text, then in the
prior, then id (BPE merge order). Other special tokens are left out. Tokenized with add_special_tokens=False.

    python3 bench/draftvocab_public.py --tokenizer <dir with tokenizer.json> --cache ~/.cache/draftvocab \\
        --write-list <tree>/src/tensorfold/families/glm5_next/cuda/draft_vocab.txt

A ranking of your own traffic (bench/draftvocab.py --write-list) can replace it at run time:
``GLM53_TF_DRAFT_VOCAB=<path>`` or ``<path>:N``.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import io
import json
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

VOCAB = 154880                 # the head's rows (tokenizer 154,856 + padding)
HALF = VOCAB // 2              # rank 0's rows: ids < 77,440
TOKENS = 154856
SPECIAL_LO = 154820            # <|endoftext|> .. <|video|>: the tokenizer's special tokens
# GLM chat / tool-call format tokens a reply emits (ordinary ids to the drafter), listed first:
FORMAT = (
    154847,  # <arg_key>
    154848,  # </arg_key>
    154849,  # <arg_value>
    154850,  # </arg_value>
    154843,  # <tool_call>
    154844,  # </tool_call>
    154842,  # </think>
    154829,  # <|observation|>  (ends a reply that called a tool)
    154827,  # <|user|>         (ends a reply that did not)
    154845,  # <tool_response>
    154846,  # </tool_response>
    154828,  # <|assistant|>
)

OASST = ("https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/fdf72ae0827c1cda404aff25b6603abec9e3399b/"
         "2023-04-12_oasst_ready.messages.jsonl.gz",
         "286a6e9a5a413b3272ae9c0b5a20d327983dea1c24342ae28cb244a6da65185c")
SWEGYM = ("https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories/resolve/"
          "4aaa5a4a4b5861f4799d2336908760c190ac3b17/data/train.success.oss-00000-of-00001.parquet",
          "ea4bf37de020e165c5210bedddeef523d8834a89a35a8c65fec24f76f0eae4f1")
FUNC_RE = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.S)
# (name, url, sha256, license, member filter)
CODE = (
    ("CPython 3.14.7", "https://www.python.org/ftp/python/3.14.7/Python-3.14.7.tar.xz",
     "3b48dac8fb59f62eaa67ac83c1eb12bda1b7a08406dd286e252c11a66be27f81", "PSF-2.0",
     re.compile(r"^[^/]+/(Lib/(?!(?:.*/)?(?:test|tests|idle_test)/).*\.py|Doc/.*\.rst)$")),
    ("denoland/std f834d02", "https://codeload.github.com/denoland/std/tar.gz/f834d0223364361169314833e3c7a8f62ce11d58",
     None, "MIT", re.compile(r"^[^/]+/.*\.(ts|md)$")),
    ("ripgrep 14.1.1", "https://codeload.github.com/BurntSushi/ripgrep/tar.gz/refs/tags/14.1.1",
     None, "MIT OR Unlicense", re.compile(r"^[^/]+/.*\.(rs|md|toml|sh)$")),
)


def fetch(url: str, sha: str | None, cache: Path) -> bytes:
    """``url``'s bytes, cached; checked against ``sha`` (GitHub archives are pinned by commit / tag instead: their
    compression is not guaranteed byte-stable)."""

    cache.mkdir(parents=True, exist_ok=True)
    p = cache / hashlib.sha256(url.encode()).hexdigest()[:16]
    if not p.exists():
        with urllib.request.urlopen(url) as r:
            p.write_bytes(r.read())
    data = p.read_bytes()
    if sha and hashlib.sha256(data).hexdigest() != sha:
        sys.exit(f"{url}: sha256 mismatch (cached as {p})")
    return data


def oasst(cache: Path) -> tuple[list[str], list[str]]:
    """(English assistant turns, every other turn)."""

    reply, rest = [], []
    for line in gzip.decompress(fetch(*OASST, cache)).splitlines():
        m = json.loads(line)
        (reply if m.get("role") == "assistant" and m.get("lang") == "en" else rest).append(m.get("text") or "")
    return reply, rest


def glm_calls(text: str) -> str:
    """An OpenHands assistant turn with its ``<function=...>`` calls in GLM's tool-call format."""

    def call(m: re.Match) -> str:
        args = "".join(f"<arg_key>{k}</arg_key><arg_value>{v.strip(chr(10))}</arg_value>"
                       for k, v in PARAM_RE.findall(m.group(2)))
        return f"<tool_call>{m.group(1)}{args}</tool_call>"

    return FUNC_RE.sub(call, text)


def swegym(cache: Path) -> tuple[list[str], list[str]]:
    """(assistant turns with GLM-format tool calls, user turns and tool outputs); system prompts are left out."""

    import pyarrow.parquet as pq

    reply, rest = [], []
    for r in pq.read_table(io.BytesIO(fetch(*SWEGYM, cache))).column("messages").to_pylist():
        for m in r:
            if m["role"] == "assistant":
                reply.append(glm_calls(m["content"] or ""))
            elif m["role"] == "user":
                rest.append(m["content"] or "")
    return reply, rest


def code(cache: Path) -> list[tuple[str, list[str]]]:
    out = []
    for name, url, sha, _lic, pick in CODE:
        tf = tarfile.open(fileobj=io.BytesIO(fetch(url, sha, cache)))
        texts = []
        for m in sorted(tf.getmembers(), key=lambda m: m.name):
            if m.isfile() and pick.match(m.name):
                texts.append(tf.extractfile(m).read().decode("utf-8", errors="ignore"))
        out.append((name, texts))
    return out


def ranking(reply: collections.Counter, prior: collections.Counter) -> list[int]:
    """FORMAT, then every other ordinary id with a count: reply-like frequency, then the prior's, then id."""

    seen = (set(reply) | set(prior)) - set(FORMAT)
    return list(FORMAT) + sorted((i for i in seen if i < SPECIAL_LO), key=lambda i: (-reply[i], -prior[i], i))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json or its directory (GLM-5.3-Flash)")
    ap.add_argument("--cache", default="~/.cache/draftvocab_public", help="downloads")
    ap.add_argument("--write-list", required=True)
    a = ap.parse_args()
    from tokenizers import Tokenizer

    p = Path(a.tokenizer)
    tok = Tokenizer.from_file(str(p / "tokenizer.json" if p.is_dir() else p))
    if tok.get_vocab_size() != TOKENS:
        sys.exit(f"{a.tokenizer}: {tok.get_vocab_size()} tokens, expected GLM-5.3-Flash's {TOKENS}")
    for i in FORMAT:
        assert tok.id_to_token(i).startswith("<"), (i, tok.id_to_token(i))
    cache = Path(a.cache).expanduser()

    def count(texts: list[str]) -> collections.Counter:
        c: collections.Counter = collections.Counter()
        for j in range(0, len(texts), 1024):
            for e in tok.encode_batch(texts[j:j + 1024], add_special_tokens=False):
                c.update(e.ids)
        return c

    reply_t, rest_t = oasst(cache)
    reply, prior = count(reply_t), count(rest_t)
    print(f"oasst1: {len(reply_t)} English assistant turns, {sum(reply.values())} tokens; "
          f"{len(rest_t)} other turns, {sum(prior.values())} tokens")
    reply_t, rest_t = swegym(cache)
    c, o = count(reply_t), count(rest_t)
    print(f"SWE-Gym: {len(reply_t)} assistant turns, {sum(c.values())} tokens; "
          f"{len(rest_t)} user turns / tool outputs, {sum(o.values())} tokens")
    reply += c
    prior += o
    for name, texts in code(cache):
        c = count(texts)
        print(f"{name}: {len(texts)} files, {sum(c.values())} tokens")
        reply += c
    n_reply, n_prior = sum(reply.values()), sum(prior.values())
    print(f"reply-like {n_reply} tokens, prior {n_prior} tokens")
    order = ranking(reply, prior)
    hdr = ("# GLM-5.3-Flash draft vocabulary ranking (patches/0420, GLM53_TF_DRAFT_VOCAB=<N>: each rank drafts over "
           "the first N / 2 ids of its half of this list, then that half's other ids by id). 16 ids a line.\n"
           f"# {len(order)} ids: {len(FORMAT)} chat / tool-call format tokens, then most frequent first: frequency in "
           f"{n_reply} tokens of public reply-like text (OpenAssistant oasst1 English assistant turns, Apache-2.0; "
           "SWE-Gym OpenHands trajectories' assistant turns, MIT; CPython 3.14.7 Lib + Doc, PSF-2.0; denoland/std "
           f"f834d02, MIT; ripgrep 14.1.1, MIT), then in {n_prior} tokens of those datasets' other turns, then id. "
           "Written by bench/draftvocab_public.py.\n")
    lines = [" ".join(str(i) for i in order[j:j + 16]) for j in range(0, len(order), 16)]
    Path(a.write_list).write_text(hdr + "\n".join(lines) + "\n")
    print(f"wrote {a.write_list}: {len(order)} ids ({sum(1 for i in order if i < HALF)} below {HALF}, "
          f"{sum(1 for i in reply if i < SPECIAL_LO)} seen in reply-like text)")


if __name__ == "__main__":
    main()
