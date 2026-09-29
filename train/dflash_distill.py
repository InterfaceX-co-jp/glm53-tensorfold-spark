#!/usr/bin/env python3
"""T2: re-fit a DFlash2 block drafter to the abliterated target on its own dumped taps (docs/DRAFTER-TRAINING.md).

Recipe (DFlash / DSpark-style on-policy distillation):
- start weights ``--drafter``: a DFlash2 folder. For anything shared, start from canada-quant/GLM-5.3-Flash-DFlash2-G
  (Apache-2.0; 8 layers, 9 taps, full attention, a learnt mask embedding): incoai/GLM-5.3-Flash-DFlash2 is
  CC BY-NC-ND 4.0 (private use only, no derivative may be shared). The dump's taps must be the drafter's tap layers
  (GLM53_TF_DRAFT_DUMP_TAP_LAYERS when they are not the served drafter's);
- blocks at ``--anchors`` random positions of a window: [t_a, mask x (B - 1)] at positions a.., context = the taps of
  positions < a (hidden_norm(fc(taps)), per layer keys / values); row j predicts position a + j;
- loss: KL(target || drafter) against the dumped top-32 (plus a rest bucket) and ``--ce`` x cross-entropy on the
  document's token, row j weighted exp(-(j - 1) / ``--gamma``) (DFlash's decay: early rows decide the acceptance);
- every drafter weight trains except the candidate selector (codebooks and hidden projection): the engine's chain
  adds its edge score to the logits unchanged (a jointly trained selector / stop head is T3, not here);
- quantization-aware (``--qat``, default on): the engine quantizes the drafter to 4 bits at load
  (``dflash2._quantize4``); the forward uses that copy with a straight-through gradient;
- AdamW (fp32 master weights), LR ``--lr``, warmup + cosine, clip 1.0.

Output ``--out``: ``log.jsonl``, ``ckpt.pt``, ``export-<step>/`` and ``export/`` (the drafter folder in its own layout:
serve with DRAFTER=<out>/export when the engine loads that architecture, i.e. incoai's layout today),
``eval-<step>.json``.

Memory on one Spark: incoai's layout 1.1 B parameters (DFlash2-G 1.84 B): weights + fp32 master + AdamW ~16 B a
parameter = 18-30 GB, plus the frozen target head and embedding (2.5 GB) and activations.

    python3 train/dflash_distill.py --model $MODEL --drafter /models/GLM-5.3-Flash-DFlash2-G --dumps /data/dump/* \\
        --out /data/runs/f1 --max-steps 20000
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
from dumpdata import build_docs, is_eval  # noqa: E402


def window_loss(m: R.DFlash, doc, tok: torch.Tensor, loss_rows, s: int, e: int, anchors: int, ctx: int,
                gamma: float, ce_w: float, rng: random.Random, dev) -> tuple[torch.Tensor, dict]:
    B = m.block
    n = doc.n
    hi = min(e, n - B)
    lo = max(1, s)
    if hi <= lo:
        return torch.zeros((), device=dev, requires_grad=True), {}
    anc = sorted(rng.sample(range(lo, hi), min(anchors, hi - lo)))
    c0 = max(0, anc[0] - ctx)
    cx = m.context(doc.taps(c0, anc[-1], m.taps).to(dev))
    a_loc = torch.tensor(anc, device=dev) - c0
    h, lg = m.blocks(cx, a_loc, tok[anc].to(dev), first=c0)
    lp = doc.get("lp", anc[0], anc[-1] + B - 1).to(dev)
    ids = doc.get("ids", anc[0], anc[-1] + B - 1).to(dev)
    rows = torch.tensor(anc, device=dev)[:, None] - anc[0] + torch.arange(B - 1, device=dev)[None]   # row a + j - 1
    mask = torch.from_numpy(loss_rows[(rows + anc[0]).cpu().numpy()]).to(dev)
    wj = torch.exp(-torch.arange(B - 1, device=dev, dtype=torch.float32) / gamma)[None].expand_as(rows)
    sel = mask.reshape(-1)
    if not bool(sel.any()):
        return torch.zeros((), device=dev, requires_grad=True), {}
    flat = lg.reshape(-1, lg.shape[-1])[sel]
    r = rows.reshape(-1)[sel]
    w = wj.reshape(-1)[sel]
    l_kl = kl_topk(flat, lp[r], ids[r])
    loss = (w * l_kl).sum() / w.sum()
    if ce_w:
        nxt = tok[(r + anc[0] + 1).cpu()].to(dev)
        loss = loss + ce_w * (w * ce(flat, nxt)).sum() / w.sum()
    with torch.no_grad():
        top1 = (lg.float().argmax(-1) == ids[rows][..., 0]).float()
        info = {"kl": float(l_kl.mean()), "top1_row1": float(top1[:, 0].mean()),
                "top1_row4": float(top1[:, min(3, B - 2)].mean())}
    return loss, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the target checkpoint (embedding and head)")
    ap.add_argument("--drafter", required=True, help="the DFlash2 folder to start from")
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchors", type=int, default=128, help="blocks a window")
    ap.add_argument("--window", type=int, default=2048, help="rows the anchors are drawn from")
    ap.add_argument("--ctx", type=int, default=2048, help="context rows before the first anchor")
    ap.add_argument("--gamma", type=float, default=7.0)
    ap.add_argument("--ce", type=float, default=0.1)
    ap.add_argument("--qat", type=int, default=1)
    ap.add_argument("--head-mode", default="q4mse", help="the engine's head (GLM53_TF_NONEXPERT)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--kinds", default="d,p")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-docs", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    out = Path(a.out)
    log = Log(out)
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    m = R.load_dflash(a.drafter, a.model, device=dev, mode="q4" if a.qat else "bf16", trainable=True,
                      head_mode=a.head_mode)
    for p in (m.hproj,):
        p.requires_grad_(False)
    params = R.master_fp32(m)
    log(event="model", trainable=sum(p.numel() for p in params), taps=list(m.taps), block=m.block,
        window=m.window)
    docs = build_docs(a.dumps)
    have = docs[0].tap_layers() if docs else []
    if not set(m.taps) <= set(have):
        raise SystemExit(f"the dump's taps {have} do not hold the drafter's {list(m.taps)}: dump with "
                         f"GLM53_TF_DRAFT_DUMP_TAP_LAYERS={','.join(map(str, m.taps))} (or a superset)")
    train_docs = [d for d in docs if not is_eval(d)]
    eval_docs = [d for d in docs if is_eval(d)][:a.eval_docs]
    table = DocTable(train_docs, kinds=a.kinds)
    toks = [torch.from_numpy(t).long() for t in table.tokens]
    log(event="data", docs=len(docs), train_docs=len(train_docs), eval_docs=len(eval_docs),
        loss_rows=int(table.total))
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
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

    def evaluate(tag: str) -> None:
        import eval_accept as E

        m.eval()
        rep = E.eval_dflash(m, eval_docs, stride=32, max_blocks=4000)
        (out / f"eval-{tag}.json").write_text(json.dumps(rep, indent=1))
        g = rep.get("all", {}).get("greedy", {})
        log(event="eval", step=step, greedy=g.get("a"), tau=g.get("tokens_per_round"))
        m.train()

    def save(final: bool = False) -> None:
        torch.save({"params": [p.detach() for p in params], "opt": opt.state_dict(), "step": step,
                    "rng": rng.getstate(), "args": vars(a)}, ck)
        R.export_dflash(m, a.drafter, out / ("export" if final else f"export-{step}"),
                        {"step": step, "args": vars(a), "start": str(a.drafter), "time": time.time()})

    if step == 0 and eval_docs:
        evaluate("base")
    m.train()
    t_last, blocks = time.time(), 0
    while step < a.max_steps:
        lr = cosine_lr(step, a.max_steps, a.lr, a.warmup)
        for gr in opt.param_groups:
            gr["lr"] = lr
        opt.zero_grad(set_to_none=True)
        info_all = {}
        for _ in range(a.accum):
            i, s, e = table.sample(rng, a.window, m.block)
            loss, info = window_loss(m, table.docs[i], toks[i], table.loss[i], s, e, a.anchors, a.ctx, a.gamma,
                                     a.ce, rng, dev)
            (loss / a.accum).backward()
            blocks += a.anchors
            for k, v in info.items():
                info_all[k] = info_all.get(k, 0.0) + v / a.accum
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if step % 20 == 0:
            dt = time.time() - t_last
            log(event="train", step=step, lr=lr, grad_norm=float(gn), blocks_per_s=round(blocks / dt, 1), **info_all)
            t_last, blocks = time.time(), 0
        if a.eval_every and step % a.eval_every == 0 and eval_docs:
            evaluate(str(step))
        if a.save_every and step % a.save_every == 0:
            save()
    save(final=True)
    if eval_docs:
        evaluate("final")


if __name__ == "__main__":
    main()
