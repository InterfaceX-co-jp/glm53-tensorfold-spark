#!/usr/bin/env python3
"""Reading patches/0430's draft dumps (GLM53_TF_DRAFT_DUMP) as training documents.

A dump folder (``<root>/<host>-<pid>-<start>/``) holds ``meta.json``, ``index.jsonl`` (one segment a line) and
``shard-*.bin``. A segment is a run of consecutive positions of one sequence: a prefill chunk (kind ``p``) or up to 256
decode rows of one batch slot (kind ``d``, tokens the target sampled itself). Per position: the input token, the main
model's final-normed row (``hidden``, the MTP head's input), the DFlash2 taps (``taps``, bf16 or e4m3 + ``tap_scale``)
and the target's top-k next-token log-probabilities (``lp`` float16, ``ids`` int32; row i describes position i + 1).

Segments link by the hash of the token ids before them (``h0``) and through them (``h1``): a document is a path from
the empty prefix, following h1 -> h0 with contiguous positions. Documents share prefixes (a system prompt, earlier
turns); every segment's rows are "owned" by the first document that reaches it, so a loss over owned rows counts
each dumped row once. Orphan decode runs (``h0`` null: the dump was armed mid-request) are dropped.

    python3 train/dumpdata.py stats /sessions/draftdata/*          # tokens by kind, documents, lengths, bytes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EMPTY = hashlib.blake2b(b"", digest_size=16).hexdigest()


def _torch():
    import torch

    return torch


class DumpDir:
    def __init__(self, path: str | Path) -> None:
        self.dir = Path(path)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.segs: list[dict] = []
        with open(self.dir / "index.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.segs.append(json.loads(line))
                    except json.JSONDecodeError:
                        break                                 # a torn last line (the writer was killed)
        self._maps: dict[str, np.memmap] = {}

    def raw(self, seg: dict, name: str) -> np.ndarray | None:
        a = seg["arrays"].get(name)
        if a is None:
            return None
        off, dtype, shape = a
        m = self._maps.get(seg["shard"])
        if m is None:
            m = self._maps[seg["shard"]] = np.memmap(self.dir / seg["shard"], dtype=np.uint8, mode="r")
        npdt = {"bfloat16": np.uint16, "float8_e4m3fn": np.uint8}.get(dtype) or np.dtype(dtype)
        n = int(np.prod(shape)) * np.dtype(npdt).itemsize
        return np.frombuffer(m[off:off + n], dtype=npdt).reshape(shape)

    def tensor(self, seg: dict, name: str):
        """A segment's array as a CPU torch tensor (bf16 and e4m3 restored)."""

        torch = _torch()
        a = self.raw(seg, name)
        if a is None:
            return None
        dtype = seg["arrays"][name][1]
        t = torch.from_numpy(np.array(a))
        if dtype == "bfloat16":
            return t.view(torch.int16).view(torch.bfloat16)
        if dtype == "float8_e4m3fn":
            return t.view(torch.float8_e4m3fn)
        return t


@dataclass
class Doc:
    """A path of segments from position 0. ``owned[i]``: segment i's rows count for this document's loss."""

    parts: list[tuple[DumpDir, dict]]
    owned: list[bool]
    key: str = ""
    starts: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.starts = [seg["start"] for _, seg in self.parts]
        self.key = self.parts[-1][1]["h1"] or ""

    @property
    def n(self) -> int:
        _, last = self.parts[-1]
        return last["start"] + last["n"]

    def kinds(self) -> np.ndarray:
        """Per position: 1 decode row (the target's own sample), 0 prefill row."""

        out = np.zeros(self.n, dtype=np.uint8)
        for _, seg in self.parts:
            if seg["kind"] == "d":
                out[seg["start"]:seg["start"] + seg["n"]] = 1
        return out

    def owned_mask(self) -> np.ndarray:
        out = np.zeros(self.n, dtype=bool)
        for (_, seg), own in zip(self.parts, self.owned):
            if own:
                out[seg["start"]:seg["start"] + seg["n"]] = True
        return out

    def get(self, name: str, a: int, b: int):
        """Rows [a, b) of an array over the document (CPU torch tensor)."""

        torch = _torch()
        chunks = []
        for dd, seg in self.parts:
            s0, s1 = seg["start"], seg["start"] + seg["n"]
            if s1 <= a or s0 >= b:
                continue
            t = dd.tensor(seg, name)
            if t is None:
                raise KeyError(f"segment {seg['seg']} of {dd.dir} has no {name!r}")
            chunks.append(t[max(a, s0) - s0:min(b, s1) - s0])
        return torch.cat(chunks) if len(chunks) > 1 else chunks[0]

    def tap_layers(self) -> list[int]:
        return list(self.parts[0][0].meta.get("tap_layers") or [])

    def taps(self, a: int, b: int, layers=None):
        """Taps rows [a, b) as bf16 [b - a, T * D] (e4m3 rows dequantized with their per-tap scales). ``layers``: a
        subset of the dumped tap layers, in that order (a 9-tap DFlash2-G dump serves incoai's 5-tap drafter)."""

        torch = _torch()
        t = self.get("taps", a, b)
        have = self.tap_layers()
        T = max(1, len(have))
        n = t.shape[0]
        if t.dtype == torch.float8_e4m3fn:
            s = self.get("tap_scale", a, b).float()
            t = (t.float().view(n, T, -1) * s[..., None]).to(torch.bfloat16)
        else:
            t = t.view(n, T, -1)
        if layers is not None and list(layers) != have:
            missing = [x for x in layers if x not in have]
            if missing:
                raise KeyError(f"taps {missing} were not dumped (dumped: {have})")
            t = t[:, [have.index(x) for x in layers]]
        return t.reshape(n, -1)

    def tokens(self) -> np.ndarray:
        out = np.empty(self.n, dtype=np.int32)
        for dd, seg in self.parts:
            out[seg["start"]:seg["start"] + seg["n"]] = dd.raw(seg, "tokens")
        return out


def build_docs(dirs: list[str | Path], *, min_tokens: int = 16) -> list[Doc]:
    """Every root-to-leaf path of linked segments over the dump folders (duplicates dropped)."""

    dumps = [DumpDir(d) for d in dirs]
    seen: set[tuple] = set()
    by_h0: dict[tuple[str, int], list[tuple[DumpDir, dict]]] = defaultdict(list)
    for dd in dumps:
        for seg in dd.segs:
            if seg.get("h0") is None or seg.get("h1") is None:
                continue                                 # orphans: no known prefix
            key = (seg["h0"], seg["h1"], seg["start"], seg["n"])
            if key in seen:
                continue
            seen.add(key)
            by_h0[(seg["h0"], seg["start"])].append((dd, seg))
    docs: list[Doc] = []
    owner: set[tuple] = set()
    stack = [[p] for p in by_h0.get((EMPTY, 0), [])]
    while stack:
        path = stack.pop()
        _, last = path[-1]
        kids = by_h0.get((last["h1"], last["start"] + last["n"]), [])
        if kids:
            for k in kids:
                stack.append(path + [k])
            continue
        if last["start"] + last["n"] < min_tokens:
            continue
        owned = []
        for _, seg in path:
            key = (seg["h0"], seg["h1"], seg["start"])
            owned.append(key not in owner)
            owner.add(key)
        docs.append(Doc(path, owned))
    return docs


def is_eval(doc: Doc, permille: int = 20) -> bool:
    """Held-out split by the document's first segment (a conversation's turns stay on one side)."""

    first = doc.parts[0][1]["h1"] or ""
    return int(first[:8] or "0", 16) % 1000 < permille


def special_ids(tokenizer_json: str | Path) -> dict[str, int]:
    tj = json.loads(Path(tokenizer_json).read_text())
    return {t["content"]: int(t["id"]) for t in tj.get("added_tokens", [])}


def assistant_mask(tokens: np.ndarray, sp: dict[str, int]) -> np.ndarray:
    """Positions inside assistant turns (after ``<|assistant|>``, up to the next role token): the text the target
    writes, whoever wrote this copy of it."""

    start = sp.get("<|assistant|>")
    ends = {sp[t] for t in ("<|user|>", "<|observation|>", "<|system|>") if t in sp}
    out = np.zeros(len(tokens), dtype=bool)
    on = False
    for i, t in enumerate(tokens.tolist()):
        if t == start:
            on = True
            continue
        if t in ends:
            on = False
        out[i] = on
    return out


def stats(dirs: list[str]) -> dict:
    docs = build_docs(dirs)
    by_kind = defaultdict(int)
    nbytes = 0
    for d in dirs:
        dd = DumpDir(d)
        for seg in dd.segs:
            by_kind[seg["kind"]] += seg["n"]
        nbytes += sum(p.stat().st_size for p in Path(d).glob("shard-*.bin"))
    lens = sorted(doc.n for doc in docs)
    owned = sum(int(doc.owned_mask().sum()) for doc in docs)
    return {"segments_tokens": dict(by_kind), "docs": len(docs), "owned_rows": owned, "bytes": nbytes,
            "doc_len_p50": lens[len(lens) // 2] if lens else 0, "doc_len_max": lens[-1] if lens else 0,
            "docs_past_2051": sum(1 for n in lens if n > 2051),
            "eval_docs": sum(1 for doc in docs if is_eval(doc))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["stats"])
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    json.dump(stats(a.dirs), sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
