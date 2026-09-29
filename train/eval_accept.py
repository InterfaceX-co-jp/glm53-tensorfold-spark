#!/usr/bin/env python3
"""Offline acceptance by draft position on held-out dump documents, before any engine test (docs/DRAFTER-TRAINING.md).

MTP (``--mtp``): the reference head (train/glmref.py, the engine's arithmetic and its q4mse weights) runs the
teacher-forced chain of ``--steps`` drafts from every anchor row. DFlash2 (``--dflash``): blocks at anchors every
``--stride`` rows, the engine's chain rule (candidates + selector edge). Per draft position k, cumulative acceptance
a_k = P(drafts 1..k all accepted), three ways:

- ``greedy``: draft == the target's argmax at that position (the dump's top-1): what greedy serving keeps;
- ``data``: draft == the token the document has there (for the target's own decode rows: its sample);
- ``T=<t>``: the expected acceptance of sampled serving at temperature t (top-p 1). The engine samples a draft with the
  same keyed Gumbel noise as the target's sample at that position, so a draft is kept when both argmaxes of
  (log-prob / t + noise) agree; estimated with ``--samples`` noise draws over the target's top-k and the draft's
  top-32 (the chain's probability is the product over positions, each conditioned on the true prefix).

Plus E[tokens a round] = 1 + sum_k a_k for a fixed-depth chain, split by row kind (``d`` the target's own samples,
``p`` prompt text) and by position (below / past the 2,051-token dense limit, where the engine's DSA selection
starts). The engine's own baseline to reproduce with the stock head: a = 0.74 / 0.45 / 0.22 (docs/DECODE-PLAN.md).

    python3 train/eval_accept.py --model $MODEL --dumps /data/dump/* --mtp [--mtp-weights runs/mtp1/export] \\
        --max-docs 200 --out runs/mtp1/eval.json
    python3 train/eval_accept.py --model $MODEL --dumps /data/dump/* --dflash $DRAFTER --out eval-f.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import glmref as R  # noqa: E402
from dumpdata import build_docs, is_eval  # noqa: E402

TEMPS = (0.6, 1.0)


class Tally:
    def __init__(self, steps: int) -> None:
        self.steps = steps
        self.sums = defaultdict(lambda: np.zeros(steps))
        self.n = defaultdict(int)

    def add(self, bucket: str, how: str, cum: np.ndarray, valid: np.ndarray) -> None:
        """cum [rows, steps] cumulative acceptance (0/1 or expected), valid [rows] rows with every position."""

        if valid.any():
            self.sums[(bucket, how)] += cum[valid].sum(0)
            self.n[(bucket, how)] += int(valid.sum())

    def report(self) -> dict:
        out = {}
        for (bucket, how), s in sorted(self.sums.items()):
            n = self.n[(bucket, how)]
            a = s / max(n, 1)
            out.setdefault(bucket, {})[how] = {"rows": n, "a": [round(float(x), 4) for x in a],
                                               "tokens_per_round": round(1 + float(a.sum()), 4)}
        return out


def coupled(lp_t: torch.Tensor, ids_t: torch.Tensor, lq_t: torch.Tensor, top_v: torch.Tensor, top_i: torch.Tensor,
            temp: float, samples: int, gen: torch.Generator, vocab: int = 154880) -> torch.Tensor:
    """P(argmax of the target's (lp / t + G) == argmax of the draft's (lq / t + G)), G shared per token id.
    lp_t / ids_t [R, k]: target top-k; lq_t [R, k]: the draft's log-probs at those ids; top_v / top_i [R, m]: the
    draft's own top-m. -> [R]."""

    R_, k = ids_t.shape
    m = top_i.shape[1]
    ids = torch.cat([ids_t, top_i], dim=1)                          # union with duplicates (masked below)
    dup = (top_i[:, :, None] == ids_t[:, None, :]).any(-1)          # draft ids already in the target's top-k
    # the target's mass outside its top-k, spread evenly over the rest of the vocabulary (V = the draft's logits')
    rest = torch.log((1 - lp_t.float().exp().sum(-1)).clamp_min(1e-9)) - np.log(max(vocab - k, 1))
    lt = torch.cat([lp_t.float(), rest[:, None].expand(R_, m)], dim=1)
    lq = torch.cat([lq_t.float(), top_v.float()], dim=1)
    lq[:, k:][dup] = float("-inf")
    lt[:, k:][dup] = float("-inf")
    hit = torch.zeros(R_, device=ids.device)
    for _ in range(samples):
        g = -torch.log(-torch.log(torch.rand(ids.shape, generator=gen, device="cpu").to(ids.device).clamp(1e-12, 1)))
        hit += ((lt / temp + g).argmax(-1) == (lq / temp + g).argmax(-1)).float()
    return hit / samples


def eval_mtp(m: R.MTP, docs, *, steps: int = 3, window: int = 1024, ctx: int = 1024, max_rows: int = 200_000,
             samples: int = 64, temps=TEMPS, sparse_docs: int = 0, seed: int = 0) -> dict:
    """Teacher-forced chain acceptance over ``docs`` (windows of ``window`` anchors after ``ctx`` context rows; a
    document of at most ``sparse_docs`` rows runs whole from position 0 with DSA selection, the engine's exact view)."""

    dev = next(m.buffers()).device
    tally = Tally(steps)
    gen = torch.Generator().manual_seed(seed)
    done = 0
    m.eval()
    for doc in docs:
        n = doc.n
        tok = torch.from_numpy(doc.tokens()).long()
        kinds = doc.kinds()
        whole = 0 < n <= sparse_docs
        spans = [(0, n - 1)] if whole else [(s, min(n - 1, s + window)) for s in range(0, n - 1, window)]
        for s, e in spans:
            c0 = 0 if whole else max(0, s - ctx)
            hidden = doc.get("hidden", c0, e).to(dev)
            tokens = tok[c0:e + 1].to(dev)
            lp = doc.get("lp", s, n).to(dev)
            ids = doc.get("ids", s, n).to(dev)
            A = e - s

            def reduce(k, lg, s=s, lp=lp, ids=ids):
                rows = torch.arange(lg.shape[0], device=dev)
                tr = rows + k                                            # target row - s
                ok = tr < lp.shape[0]
                tr = tr.clamp(max=lp.shape[0] - 1)
                lq = torch.log_softmax(lg.float(), -1)
                tv, ti = torch.topk(lq, 32, dim=-1)
                return {"arg": ti[:, 0], "ok": ok, "lq_t": lq.gather(1, ids[tr].long()), "tv": tv, "ti": ti,
                        "lp_t": lp[tr].float(), "ids_t": ids[tr].long(), "tr": tr, "V": lg.shape[1]}

            with torch.no_grad():
                outs = m.chain(hidden, tokens, steps, first=c0, ctx=s - c0, sparse=whole, reduce=reduce)
            acc = {h: np.zeros((A, steps)) for h in ["greedy", "data"] + [f"T={t}" for t in temps]}
            valid = np.ones(A, dtype=bool)
            for k, o in enumerate(outs, start=1):
                nk = o["arg"].shape[0]
                ok = o["ok"].cpu().numpy()
                valid[:nk] &= ok
                valid[nk:] = False
                arg = o["arg"].cpu()
                tr = o["tr"].cpu()
                acc["greedy"][:nk, k - 1] = (arg == o["ids_t"][:, 0].cpu()).numpy()
                data_next = tok[(s + tr + 1).clamp(max=n - 1)]
                acc["data"][:nk, k - 1] = (arg == data_next).numpy()
                for t in temps:
                    acc[f"T={t}"][:nk, k - 1] = coupled(o["lp_t"], o["ids_t"], o["lq_t"], o["tv"], o["ti"], t,
                                                        samples, gen, o["V"]).cpu().numpy()
            pos = np.arange(s, e)
            for how, a in acc.items():
                cum = np.cumprod(a, axis=1)
                for bucket, sel in (("all", np.ones(A, bool)), ("kind_d", kinds[s:e] == 1),
                                    ("kind_p", kinds[s:e] == 0), ("pos<2051", pos + steps < 2051),
                                    ("pos>=2051", pos >= 2051)):
                    tally.add(bucket, how, cum, valid & sel)
            done += A
            if done >= max_rows:
                return tally.report()
    return tally.report()


def eval_dflash(m: R.DFlash, docs, *, stride: int = 16, ctx: int = 2048, batch: int = 64, max_blocks: int = 20_000,
                selector: bool = True) -> dict:
    dev = next(m.buffers()).device
    B = m.block
    tally = Tally(B - 1)
    done = 0
    for doc in docs:
        n = doc.n
        tok = torch.from_numpy(doc.tokens()).long()
        kinds = doc.kinds()
        anchors_all = list(range(1, n - B, stride))
        for i in range(0, len(anchors_all), batch):
            anc = anchors_all[i:i + batch]
            c0 = max(0, anc[0] - ctx)
            c1 = anc[-1]
            with torch.no_grad():
                cx = m.context(doc.taps(c0, c1, list(m.taps)).to(dev) if hasattr(doc, "tap_layers")
                               else doc.taps(c0, c1).to(dev))
                a_loc = torch.tensor(anc, device=dev) - c0
                h, lg = m.blocks(cx, a_loc, tok[anc].to(dev), first=c0)
            ids = doc.get("ids", anc[0], anc[-1] + B)
            A = len(anc)
            greedy = np.zeros((A, B - 1))
            data = np.zeros((A, B - 1))
            for r, a in enumerate(anc):
                drafts = m.chain(h[r], lg[r], int(tok[a]), B - 1) if selector else \
                    lg[r].float().argmax(-1).tolist()
                tgt = ids[a - anc[0]:a - anc[0] + B - 1, 0].tolist()     # row a + j - 1 predicts position a + j
                greedy[r] = [float(x == y) for x, y in zip(drafts, tgt)]
                data[r] = [float(x == y) for x, y in zip(drafts, tok[a + 1:a + B].tolist())]
            sel_d = kinds[anc] == 1
            for how, arr in (("greedy", greedy), ("data", data)):
                cum = np.cumprod(arr, axis=1)
                for bucket, sel in (("all", np.ones(A, bool)), ("kind_d", sel_d), ("kind_p", ~sel_d)):
                    tally.add(bucket, how, cum, sel)
            done += A
            if done >= max_blocks:
                return tally.report()
    return tally.report()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the target checkpoint folder (config, embed, head, MTP layer)")
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--split", choices=["eval", "all"], default="eval")
    ap.add_argument("--max-docs", type=int, default=200)
    ap.add_argument("--max-rows", type=int, default=200_000)
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--mtp-weights", default="", help="an export folder (GLM53_TF_MTP_WEIGHTS format)")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--sparse-docs", type=int, default=0, help="run documents up to this long whole, with DSA")
    ap.add_argument("--nonexpert", default="q4mse", help="the engine's GLM53_TF_NONEXPERT")
    ap.add_argument("--kv", default="fp8", help="the engine's GLM53_TF_KV_DTYPE")
    ap.add_argument("--expert-cache", default="")
    ap.add_argument("--dflash", default="", help="a DFlash2 folder")
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    docs = build_docs(a.dumps)
    if a.split == "eval":
        docs = [d for d in docs if is_eval(d)]
    docs = docs[:a.max_docs]
    res = {"docs": len(docs), "rows": int(sum(d.n for d in docs))}
    if a.mtp:
        m = R.build_mtp(a.model, device=a.device, mode=a.nonexpert, kv_fp8=a.kv == "fp8",
                        override=a.mtp_weights or None, expert_cache=a.expert_cache or None)
        res["mtp"] = eval_mtp(m, docs, steps=a.steps, window=a.window, ctx=a.ctx, max_rows=a.max_rows,
                              samples=a.samples, sparse_docs=a.sparse_docs)
        res["mtp_weights"] = a.mtp_weights or "checkpoint"
        del m
    if a.dflash:
        f = R.load_dflash(a.dflash, a.model, device=a.device, head_mode=a.nonexpert)
        res["dflash"] = eval_dflash(f, docs, stride=a.stride)
        res["dflash_dir"] = a.dflash
    text = json.dumps(res, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
