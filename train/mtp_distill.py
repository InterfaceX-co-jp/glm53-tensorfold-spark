#!/usr/bin/env python3
"""T1: self-distil GLM-5.3-Flash's MTP head on the target's own dumped rows, FastMTP-style (docs/DRAFTER-TRAINING.md).

Recipe:
- the head drafts ``--steps`` (3) positions by recurring on itself with shared weights (training-time test: step k
  reads step k-1's shared_head.norm output and attends to the chain's own keys, ``glmref.MTP.chain``), teacher-forced
  on the document's tokens;
- loss at step k: KL(target || head) against the target's dumped top-32 distribution (plus a rest bucket), weight
  ``--pos-weights`` (FastMTP's exponentially decaying 0.6^(k-1), normalized: 0.51 / 0.31 / 0.18), plus ``--ce`` x
  cross-entropy on the document's token;
- trained: attention, eh_proj, norms, router and the shared expert (``--trainable``); frozen: the 288 routed experts
  (EXL3, dequantized), the embedding and the target's head (as the engine stores them: q4mse);
- quantization-aware (``--qat``, default on): the forward multiplies by the q4mse copy the engine will make of the
  exported BF16 weights (straight-through gradient), so the checkpoint that trains is the one that serves;
- windows of ``--window`` anchor rows after ``--ctx`` context rows (dense attention: the engine's exact rule while a
  document is at most 2,051 tokens; a window of a longer document sees its last ctx + window rows instead of DSA's
  selection, the one approximation, measured by ``eval_accept.py --sparse-docs``);
- AdamW (fp32 master weights), LR ``--lr`` with warmup and cosine to 10%, gradient clip 1.0.

Output ``--out``: ``log.jsonl``, ``ckpt.pt`` (resumable), ``export-<step>/`` and ``export/`` (the BF16 tensors under the
checkpoint's names: serve with GLM53_TF_MTP_WEIGHTS=<out>/export, patches/0430), ``eval-<step>.json``.

One DGX Spark (GB10, 121 GiB unified) with the engine stopped: ~23 GB of frozen weights (experts 14.5 GB bf16, embed
and head 2.5 GB, the rest of the layer), ~3 GB of optimizer state, ~15-30 GB of activations at 2 x 1,024 anchors x 3
steps. Throughput estimate: docs/DRAFTER-TRAINING.md §5.

    python3 train/mtp_distill.py --model $MODEL --dumps /data/dump/* --out /data/runs/mtp1 \\
        --expert-cache /data/mtp-experts.pt --max-steps 6000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import glmref as R  # noqa: E402
from common import DocTable, Log, ce, cosine_lr, kl_topk  # noqa: E402
from dumpdata import build_docs, is_eval, special_ids  # noqa: E402


def window_loss(m: R.MTP, doc, tok: torch.Tensor, loss_rows, kinds, s: int, e: int, ctx: int, steps: int,
                weights: list[float], ce_w: float, dev) -> tuple[torch.Tensor, dict]:
    """The chain over anchors [s, e) with ``ctx`` context rows before them."""

    n = doc.n
    c0 = max(0, s - ctx)
    hidden = doc.get("hidden", c0, e).to(dev)
    tokens = tok[c0:e + 1].to(dev)
    lp = doc.get("lp", s, min(n, e + steps)).to(dev)
    ids = doc.get("ids", s, min(n, e + steps)).to(dev)
    outs = m.chain(hidden, tokens, steps, first=c0, ctx=s - c0, sparse=False)
    total = torch.zeros((), device=dev)
    info = {}
    for k, lg in enumerate(outs, start=1):
        j = torch.arange(lg.shape[0], device=dev)
        tr = j + k                                              # target row - s
        ok = tr < lp.shape[0]
        absr = (s + tr).clamp(max=n - 1).cpu()
        mask = ok.cpu() & torch.from_numpy(loss_rows[absr.numpy()]) & (s + tr + 1 < n).cpu()
        mask = mask.to(dev)
        if not bool(mask.any()):
            continue
        sel = mask.nonzero().squeeze(1)
        trs = tr[sel]
        l_kl = kl_topk(lg[sel], lp[trs], ids[trs])
        loss = l_kl.mean()
        if ce_w:
            nxt = tok[(s + trs + 1).cpu()].to(dev)
            loss = loss + ce_w * ce(lg[sel], nxt).mean()
        total = total + weights[k - 1] * loss
        with torch.no_grad():
            info[f"kl{k}"] = float(l_kl.mean())
            info[f"top1_{k}"] = float((lg[sel].float().argmax(-1) == ids[trs][:, 0]).float().mean())
    return total, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="", help="start from an earlier export (GLM53_TF_MTP_WEIGHTS format)")
    ap.add_argument("--steps", type=int, default=3, help="chained draft positions trained (FastMTP: 3)")
    ap.add_argument("--pos-weights", default="0.51,0.31,0.18")
    ap.add_argument("--ce", type=float, default=0.1, help="weight of cross-entropy on the document's token")
    ap.add_argument("--trainable", default=",".join(R.TRAINABLE_DEFAULT))
    ap.add_argument("--qat", type=int, default=1, help="train through the engine's q4mse copy (straight-through)")
    ap.add_argument("--nonexpert", default="q4mse")
    ap.add_argument("--kv", default="fp8")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--accum", type=int, default=2, help="windows per optimizer step")
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--kinds", default="d,p", help="rows that carry loss: d (target's own samples), p (prompt text)")
    ap.add_argument("--assistant-only", action="store_true", help="prompt rows only inside assistant turns")
    ap.add_argument("--tokenizer", default="", help="tokenizer.json (for --assistant-only); default: the model's")
    ap.add_argument("--expert-cache", default="")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-docs", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    out = Path(a.out)
    log = Log(out)
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    groups = [g for g in a.trainable.split(",") if g]
    mode = a.nonexpert if a.qat else "bf16"
    m = R.build_mtp(a.model, device=dev, mode=mode, kv_fp8=a.kv == "fp8", trainable=groups,
                    override=a.init or None, expert_cache=a.expert_cache or None)
    params = R.master_fp32(m)
    log(event="model", trainable=sum(p.numel() for p in params), groups=groups, mode=mode)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd)
    step = 0
    ck = out / "ckpt.pt"
    if ck.exists():
        state = torch.load(ck, map_location=dev)
        for p, v in zip(params, state["params"]):
            p.data.copy_(v)
        opt.load_state_dict(state["opt"])
        step = state["step"]
        rng.setstate(state["rng"])
        log(event="resume", step=step)

    docs = build_docs(a.dumps)
    train_docs = [d for d in docs if not is_eval(d)]
    eval_docs = [d for d in docs if is_eval(d)][:a.eval_docs]
    tj = Path(a.tokenizer) if a.tokenizer else Path(a.model) / "tokenizer.json"
    sp = special_ids(tj) if a.assistant_only and tj.exists() else None
    table = DocTable(train_docs, kinds=a.kinds, special=sp, assistant_only=a.assistant_only)
    log(event="data", docs=len(docs), train_docs=len(train_docs), eval_docs=len(eval_docs),
        loss_rows=int(table.total))
    weights = [float(x) for x in a.pos_weights.split(",")][:a.steps]
    toks = [torch.from_numpy(t).long() for t in table.tokens]

    def evaluate(tag: str) -> None:
        import eval_accept as E

        m.eval()
        rep = E.eval_mtp(m, eval_docs, steps=a.steps, window=a.window, ctx=a.ctx, max_rows=50_000, samples=32)
        (out / f"eval-{tag}.json").write_text(json.dumps(rep, indent=1))
        g = rep.get("all", {}).get("greedy", {})
        log(event="eval", step=step, greedy=g.get("a"), tokens_per_round=g.get("tokens_per_round"),
            t1=rep.get("all", {}).get("T=1.0", {}).get("a"))
        m.train()

    def save(final: bool = False) -> None:
        torch.save({"params": [p.detach() for p in params], "opt": opt.state_dict(), "step": step,
                    "rng": rng.getstate(), "args": vars(a)}, ck)
        meta = {"step": step, "args": vars(a), "base": str(a.model), "time": time.time()}
        R.export_mtp(m, out / ("export" if final else f"export-{step}"), groups, meta)

    if step == 0 and eval_docs:
        evaluate("base")
    m.train()
    t_last, rows_since = time.time(), 0
    while step < a.max_steps:
        lr = cosine_lr(step, a.max_steps, a.lr, a.warmup)
        for gr in opt.param_groups:
            gr["lr"] = lr
        opt.zero_grad(set_to_none=True)
        info_all = {}
        for _ in range(a.accum):
            i, s, e = table.sample(rng, a.window, a.steps)
            loss, info = window_loss(m, table.docs[i], toks[i], table.loss[i], None, s, e, a.ctx, a.steps, weights,
                                     a.ce, dev)
            (loss / a.accum).backward()
            rows_since += e - s
            for k, v in info.items():
                info_all[k] = info_all.get(k, 0.0) + v / a.accum
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if step % 20 == 0:
            dt = time.time() - t_last
            log(event="train", step=step, lr=lr, grad_norm=float(gn), anchors_per_s=round(rows_since / dt, 1),
                **info_all)
            t_last, rows_since = time.time(), 0
        if a.eval_every and step % a.eval_every == 0 and eval_docs:
            evaluate(str(step))
        if a.save_every and step % a.save_every == 0:
            save()
    save(final=True)
    if eval_docs:
        evaluate("final")


if __name__ == "__main__":
    main()
