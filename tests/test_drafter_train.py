"""End to end on CPU with a tiny synthetic model: a patches/0430 dump (through ``dump.Dumper`` and a fake engine whose
target is the tiny model's own head), then ``train/mtp_distill.py`` and ``train/dflash_distill.py`` for a few steps
each (loss goes down, exports load back through GLM53_TF_MTP_WEIGHTS's reader / the DFlash2 loader), then
``train/eval_accept.py`` on the exports.

    TF_SRC=<patched tree>/src python -m pytest -q tests/test_drafter_train.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train"))
sys.path.insert(0, str(REPO / "tests"))
import glmref as R  # noqa: E402
from test_drafter_ref import TINY, _tiny_dflash, _tiny_mtp, _tiny_tensors_of  # noqa: E402

TF_SRC = Path(os.environ.get("TF_SRC", "/src/TensorFold/src"))
DUMP_PY = TF_SRC / "tensorfold/families/glm5_next/cuda/dump.py"
D, V = 128, 96


def _write_model(path: Path, m: R.MTP) -> None:
    """The tiny head as a BF16 checkpoint folder (non-EXL3 experts, layer 2 = the MTP layer)."""

    pre = R.mtp_prefix(m.cfg)
    t = {pre + R.MTP_NAMES[k]: v.detach().to(torch.bfloat16) if v.is_floating_point() else v
         for k, v in _tiny_tensors_of(m).items() if k in R.MTP_NAMES}
    t[pre + R.MTP_NAMES["router_bias"]] = m.bias.float()
    g = torch.Generator().manual_seed(21)
    for e in range(m.cfg.experts):
        t[pre + f"mlp.experts.{e}.gate_proj.weight"] = m.e_gate[e].t().contiguous()
        t[pre + f"mlp.experts.{e}.up_proj.weight"] = m.e_up[e].t().contiguous()
        t[pre + f"mlp.experts.{e}.down_proj.weight"] = m.e_down[e].t().contiguous()
    t[R.PREFIX + "embed_tokens.weight"] = m.embed
    t["lm_head.weight"] = (torch.randn(V, D, generator=g) * 0.08).to(torch.bfloat16)
    path.mkdir(parents=True, exist_ok=True)
    R.write_safetensors(path / "model.safetensors", t)
    (path / "config.json").write_text(json.dumps(TINY))


def _dump(tmp: Path, target_head: torch.Tensor, n_docs: int = 6) -> Path:
    if not DUMP_PY.exists():
        pytest.skip("TF_SRC: a patched TensorFold source tree (patches/0430)")
    os.environ.update(GLM53_TF_DRAFT_DUMP=str(tmp / "dump"), GLM53_TF_DRAFT_DUMP_TOPK="8",
                      GLM53_TF_DRAFT_DUMP_TAPS="bf16")
    spec = importlib.util.spec_from_file_location("dump_e2e", DUMP_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = SimpleNamespace(hidden=D, vocab=V, layers=2, eos=(95,))
    w = SimpleNamespace(cfg=cfg, rank=0, world=1, comm=None, vocab_offset=0, meta={})
    d = mod.Dumper(w, str(tmp / "dump"), (0, 1))
    d._head_rows = lambda h: d._local_topk((h.float() @ target_head.float().t()).to(torch.bfloat16))
    d.armed = True
    g = torch.Generator().manual_seed(22)
    for k in range(n_docs):
        n = 60 + 7 * k
        # hidden rows with structure (a smooth walk), so the head has something to learn
        h = torch.cumsum(torch.randn(n, D, generator=g) * 0.3, 0).to(torch.bfloat16)
        taps = torch.randn(n, 2 * D, generator=g).to(torch.bfloat16)
        toks = torch.randint(0, V, (n,), generator=g).tolist()
        e = SimpleNamespace(st=object(), main_hidden=lambda sl, h=h: h[sl], tap_rows=lambda r, t=taps: t[:r])
        d.prefill_chunk(e, toks, 0, n, 0)
    d.close()
    return next((tmp / "dump").iterdir())


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=REPO, capture_output=True, text=True, timeout=900)


def test_mtp_and_dflash_training_end_to_end(tmp_path):
    m = _tiny_mtp(7)
    model = tmp_path / "model"
    _write_model(model, m)
    ck = R.Checkpoint(model)
    head = R.load_head(ck, "q4mse")
    dump = _dump(tmp_path, head)

    out = tmp_path / "mtp"
    r = _run(["train/mtp_distill.py", "--model", str(model), "--dumps", str(dump), "--out", str(out),
              "--device", "cpu", "--max-steps", "160", "--window", "32", "--ctx", "16", "--lr", "3e-3",
              "--warmup", "2", "--eval-every", "0", "--save-every", "20", "--accum", "1"])
    assert r.returncode == 0, r.stderr[-3000:]
    logs = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
    kl = [x["kl1"] for x in logs if x.get("event") == "train" and "kl1" in x]
    assert len(kl) >= 8 and np.mean(kl[-3:]) < np.mean(kl[:2])            # (one random window a step: noisy)
    exp = out / "export"
    names = R.Checkpoint(exp).names()
    assert R.mtp_prefix(m.cfg) + "eh_proj.weight" in names
    m2 = R.build_mtp(model, device="cpu", override=exp)
    assert not torch.equal(m2.eh.weight, m.eh.weight)

    r = _run(["train/eval_accept.py", "--model", str(model), "--dumps", str(dump), "--split", "all", "--mtp",
              "--mtp-weights", str(exp), "--device", "cpu", "--window", "32", "--ctx", "16", "--samples", "4",
              "--out", str(tmp_path / "eval.json")])
    assert r.returncode == 0, r.stderr[-3000:]
    rep = json.loads((tmp_path / "eval.json").read_text())
    assert len(rep["mtp"]["all"]["greedy"]["a"]) == 3

    # DFlash2: a drafter folder in the engine's layout, trained a few steps, exported, reloaded
    f = _tiny_dflash(window=6)
    dr = tmp_path / "drafter"
    R.export_dflash(f, _seed_drafter(tmp_path / "seed", f), dr)
    outf = tmp_path / "f"
    r = _run(["train/dflash_distill.py", "--model", str(model), "--drafter", str(dr), "--dumps", str(dump),
              "--out", str(outf), "--device", "cpu", "--max-steps", "100", "--anchors", "8", "--window", "40",
              "--ctx", "8", "--lr", "3e-3", "--warmup", "2", "--eval-every", "0", "--save-every", "0",
              "--accum", "1"])
    assert r.returncode == 0, r.stderr[-3000:]
    logs = [json.loads(x) for x in (outf / "log.jsonl").read_text().splitlines()]
    kl = [x["kl"] for x in logs if x.get("event") == "train" and "kl" in x]
    assert np.mean(kl[-2:]) < np.mean(kl[:2])
    f2 = R.load_dflash(outf / "export", model, device="cpu")
    assert f2.block == f.block and len(f2.layers) == len(f.layers)


def _seed_drafter(path: Path, f: R.DFlash) -> Path:
    """A drafter folder whose tensors are ``f``'s (config + model.safetensors: the layout DRAFTER= loads)."""

    path.mkdir(parents=True, exist_ok=True)
    t = {"fc.weight": f.fc.weight, "hidden_norm.weight": f.hidden_norm, "norm.weight": f.norm,
         "candidate_selector.hidden_projection.weight": f.hproj,
         "candidate_selector.predecessor_codebook": f.pred.to(torch.bfloat16),
         "candidate_selector.successor_codebook": f.succ.to(torch.bfloat16)}
    R.write_safetensors(path / "model.safetensors", {k: v.detach().to(torch.bfloat16) for k, v in t.items()})
    (path / "config.json").write_text(json.dumps(f.cfg))
    return path
