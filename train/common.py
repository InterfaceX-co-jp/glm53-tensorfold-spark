"""Pieces the two distillation trainers and the acceptance evaluator share: the loss against the target's dumped
top-k distribution, window sampling over dump documents, the LR schedule and run bookkeeping."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dumpdata import Doc, assistant_mask


def kl_topk(logits: torch.Tensor, lp: torch.Tensor, ids: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    """Per row KL(p_target || q_draft) over the target's top-k tokens plus one "everything else" bucket.

    ``logits`` [R, V] (the draft's, full vocabulary), ``lp`` [R, k] the target's log-probabilities of ``ids`` [R, k]
    (temperature 1 in the dump; ``temp`` rescales both). The bucket keeps the loss a proper divergence when the
    top-k does not hold all of the target's mass (k = 32 holds > 0.99 on most rows). -> [R] fp32."""

    lq = F.log_softmax(logits.float() / temp, dim=-1)
    lp = lp.float()
    if temp != 1.0:
        lp = F.log_softmax(lp / temp, dim=-1)          # the top-k renormalized at that temperature
    p = lp.exp()
    q_top = lq.gather(1, ids.long())
    kl = (p * (lp - q_top)).sum(-1)
    p_rest = (1.0 - p.sum(-1)).clamp_min(0.0)
    q_rest = (1.0 - q_top.exp().sum(-1)).clamp_min(1e-9)
    return kl + torch.where(p_rest > 1e-6, p_rest * (torch.log(p_rest.clamp_min(1e-9)) - torch.log(q_rest)),
                            torch.zeros_like(kl))


def ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float(), target.long(), reduction="none")


class DocTable:
    """Per document: tokens, kinds (1 = the target's own decode rows), owned rows and an optional assistant mask;
    windows are sampled by the number of loss rows they hold."""

    def __init__(self, docs: list[Doc], *, kinds: str = "d,p", special: dict | None = None,
                 assistant_only: bool = False) -> None:
        self.docs = docs
        want = {"d": 1, "p": 0}
        allowed = {want[k] for k in kinds.split(",") if k}
        self.tokens, self.loss = [], []
        for d in docs:
            tok = d.tokens()
            m = d.owned_mask() & np.isin(d.kinds(), list(allowed))
            if assistant_only and special:
                m &= assistant_mask(tok, special) | (d.kinds() == 1)
            self.tokens.append(tok)
            self.loss.append(m)
        self.weight = np.array([m.sum() for m in self.loss], dtype=np.float64)
        self.total = float(self.weight.sum())

    def sample(self, rng: random.Random, window: int, need_after: int) -> tuple[int, int, int]:
        """(doc, start, stop): a window of anchor rows [start, stop) holding loss rows, with ``need_after`` rows of
        the document after it when possible."""

        if self.total <= 0:
            raise ValueError("no loss rows in these documents (check --kinds / --assistant-only)")
        i = int(np.searchsorted(np.cumsum(self.weight), rng.random() * self.total, side="right"))
        i = min(i, len(self.docs) - 1)
        m = self.loss[i]
        rows = np.flatnonzero(m)
        centre = int(rows[rng.randrange(len(rows))])
        n = len(self.tokens[i])
        start = max(0, min(centre - rng.randrange(window), n - 1 - need_after - 1))
        stop = min(n - 1 - need_after, start + window)
        if stop <= start:
            start, stop = 0, max(1, n - 1 - need_after)
        return i, start, stop


def cosine_lr(step: int, total: int, base: float, warmup: int, floor: float = 0.1) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t)))


class Log:
    def __init__(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        self.f = open(out / "log.jsonl", "a", buffering=1)
        self.t0 = time.time()

    def __call__(self, **kv) -> None:
        kv["elapsed_s"] = round(time.time() - self.t0, 1)
        line = json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in kv.items()})
        self.f.write(line + "\n")
        print(line, flush=True)
