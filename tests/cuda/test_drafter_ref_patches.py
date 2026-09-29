"""patches/0430 and the drafter reference (train/glmref.py) against the engine, on a GPU.

Synthetic (one GPU, upstream's tiny EXL3 checkpoint and one-layer DFlash2 drafter from test_glm_engine.py, the engine
as rank 0 of two with no communicator: rank 0's own partials, which the reference reproduces with ``shard=(0, 2)``):

- GLM53_TF_MTP_WEIGHTS (``mtpw``): an export of the reference loads into the engine's MTP layer, quantized as the
  checkpoint's own tensors, nothing else changed;
- MTP: the engine's ``mtp_forward`` over n rows at the head's first position, then two chained steps, against
  ``glmref.MTP.step`` (logits close, argmax equal on >= 97% of rows; q4mse non-expert weights, EXL3 experts through
  the engine's reference decoder, bf16 latent rows: the synthetic latent is 128 wide and FP8 rows need 512, so the
  FP8 path is covered by tests/test_drafter_ref.py against patches/0220's own reference);
- DFlash2: a block's candidates after context taps (engine ``Drafter``) against ``glmref.DFlash.blocks``;
- the dump (GLM53_TF_DRAFT_DUMP, a batch engine): replies byte-identical with the dump on and off; the records of a
  greedy request assemble into one document (prompt rows kind p, reply rows kind d) whose dumped top-1 at every decode
  row is the token the engine emitted next.

Real model (the GPU phase; skipped unless DRAFT_TEST_MODEL and DRAFT_TEST_DUMPS are set): the reference MTP head on
dumped held-out documents reproduces the engine's measured MTP acceptance (DRAFT_TEST_BASELINE, default
0.74,0.45,0.22, +-DRAFT_TEST_TOL 0.08) with the checkpoint's own head, the "data" definition (draft == the token the
engine then emitted) on decode rows.

    docker run --rm --gpus all -v $PWD/tests:/work/tests -v $PWD/train:/work/train --entrypoint bash IMAGE -c \
      "pip install -q pytest; cd /work && PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda \
       python -m pytest -q -s tests/cuda/test_drafter_ref_patches.py"

(``train/`` must sit next to ``tests/``: the reference is imported from ../../train.)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train"))
import glmref as R  # noqa: E402

os.environ.setdefault("GLM53_EXL3_PY", str(Path(__import__("tensorfold").__file__).parent
                                           / "families/glm5_next/cuda/exl3.py"))


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_ref")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


def _weights(path, monkeypatch):
    from tensorfold.families.glm5_next.cuda import weights

    monkeypatch.setattr(weights, "NONEXPERT", "q4mse")
    monkeypatch.setenv("GLM53_TF_LATENT_KV", "1")
    monkeypatch.setenv("GLM53_TF_KV_DTYPE", "bf16")
    monkeypatch.delenv("GLM53_TF_PREPARED", raising=False)
    w = weights.load_checkpoint(path / "model", rank=0)
    w.comm = None
    return w


def test_mtp_reference_matches_engine(ckpt, monkeypatch):
    from tensorfold.families.glm5_next.cuda.decode import Engine
    from tensorfold.families.glm5_next.cuda.mtp import mtp_forward

    w = _weights(ckpt, monkeypatch)
    e = Engine(w, capacity=512, max_rows=8, prefill_rows=64)
    ref = R.build_mtp(ckpt / "model", device="cuda", mode="q4mse", kv_fp8=False, shard=(0, 2),
                      expert_dtype=torch.float32)
    g = torch.Generator(device="cuda").manual_seed(0)
    n = 8
    D = w.cfg.hidden
    hidden = (torch.randn(n, D, device="cuda", generator=g)).to(torch.bfloat16)
    toks = torch.randint(0, 1000, (n + 3,), device="cuda", generator=g)
    st, b = e.st, e.mbuf
    st.set_mtp_len(0)
    lg_e = mtp_forward(w, st, b, toks[1:n + 1].tolist(), hidden, last_only=False).clone()
    out_e = b.fnormed[n - 1:n].clone()
    st.set_mtp_len(n)
    lg_r, out_r, keys = ref.step(hidden, toks[1:n + 1], None, 0, zero_first=True)
    _close(lg_e, lg_r, "step 1")
    # two chained steps from the last row, each on the engine's own previous output
    prev_e, prev_r = out_e, out_r[n - 1:n]
    for k in range(2):
        t = [int(toks[n + 1 + k])]
        lg_e = mtp_forward(w, st, b, t, prev_e, last_only=True).clone()
        prev_e = b.fnormed[0:1].clone()
        st.set_mtp_len(st.mtp_len + 1)
        lg_r, prev_r, keys = ref.step(prev_r, toks[n + 1 + k:n + 2 + k], keys, n + k)
        _close(lg_e, lg_r, f"chained step {k + 2}")


def _close(a: torch.Tensor, b: torch.Tensor, what: str) -> None:
    a, b = a.float(), b.float()
    scale = a.abs().max().clamp_min(1e-3)
    err = float((a - b).abs().max() / scale)
    agree = float((a.argmax(-1) == b.argmax(-1)).float().mean())
    print(f"[ref] {what}: max |diff| / max |x| = {err:.4f}, argmax agreement {agree:.3f}")
    assert err < 0.03, what
    assert agree >= 0.97 or a.shape[0] == 1 and agree == 1.0, what


def test_mtp_weights_override_loads_the_export(ckpt, monkeypatch, tmp_path):
    """GLM53_TF_MTP_WEIGHTS: an export of the reference head with changed eh_proj / o_proj / shared expert is what the
    engine then stores (rank 0's part, quantized as the checkpoint's own: q4mse), every other tensor unchanged."""

    from tensorfold.families.glm5_next.cuda import mtpw, qmm, weights

    monkeypatch.setattr(weights, "NONEXPERT", "q4mse")
    monkeypatch.delenv("GLM53_TF_PREPARED", raising=False)
    ref = R.build_mtp(ckpt / "model", device="cuda", mode="q4mse", kv_fp8=False)
    g = torch.Generator(device="cuda").manual_seed(4)
    with torch.no_grad():
        for q in (ref.eh, ref.o, ref.s_gu, ref.s_down):
            q.weight.add_(torch.randn(q.weight.shape, device="cuda", generator=g).to(q.weight.dtype) * 0.01)
    R.export_mtp(ref, tmp_path / "export", ["eh", "o", "shared"])
    monkeypatch.setenv("GLM53_TF_MTP_WEIGHTS", str(tmp_path / "export"))
    w = mtpw.load(ckpt / "model", rank=0)
    monkeypatch.delenv("GLM53_TF_MTP_WEIGHTS")
    base = weights.load_checkpoint(ckpt / "model", rank=0)
    want = R.dequant4(*R.quantize4(ref.eh.weight.detach(), mse=True))
    assert torch.equal(qmm.dequantize_q4(w.mtp.eh), want)
    half = ref.o.weight.shape[1] // 2                     # o_proj: rank 0 holds the first half of its inputs
    assert torch.equal(qmm.dequantize_q4(w.mtp.layer.dsa.o), R.dequant4(*R.quantize4(ref.o.weight[:, :half].detach(),
                                                                                      mse=True)))
    assert torch.equal(qmm.dequantize_q4(w.mtp.layer.dsa.q_b), qmm.dequantize_q4(base.mtp.layer.dsa.q_b))
    assert not torch.equal(qmm.dequantize_q4(w.mtp.eh), qmm.dequantize_q4(base.mtp.eh))
    assert torch.equal(qmm.dequantize_q4(w.layers[1].dsa.o), qmm.dequantize_q4(base.layers[1].dsa.o))


def test_dflash_reference_matches_engine(ckpt, monkeypatch):
    from tensorfold.families.glm5_next.cuda.dflash2 import Drafter

    w = _weights(ckpt, monkeypatch)
    d = Drafter(ckpt / "dflash2", w, capacity=256)
    ref = R.load_dflash(ckpt / "dflash2", ckpt / "model", device="cuda", head_mode="q4mse", shard=(0, 2))
    g = torch.Generator(device="cuda").manual_seed(1)
    n = 40
    taps = torch.randn(n, len(d.tap_layers) * w.cfg.hidden, device="cuda", generator=g).to(torch.bfloat16)
    d.reset()
    d.add_taps(taps)
    pending = 17
    toks, vals, _ = d.candidates(pending, d.block - 1)
    with torch.no_grad():
        h, lg = ref.blocks(ref.context(taps), torch.tensor([n], device="cuda"), torch.tensor([pending], device="cuda"))
    rv, ri = torch.topk(lg[0].float(), d.top_k, dim=-1)
    top1 = float((ri[:, 0].cpu().numpy() == toks[:, 0]).mean())
    err = float(np.abs(rv[:, 0].cpu().numpy() - vals[:, 0]).max() / max(1e-3, np.abs(vals[:, 0]).max()))
    print(f"[ref] DFlash2 block: top-1 agreement {top1:.3f}, top-1 logit rel diff {err:.4f}")
    assert top1 >= 6 / 7 and err < 0.05


def test_dump_leaves_replies_alone_and_lines_up(ckpt, monkeypatch, tmp_path):
    from test_glm_engine import _TwoCopies

    from tensorfold.families.glm5_next.cuda import dump, weights
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    import dumpdata

    monkeypatch.setattr(weights, "NONEXPERT", "q4mse")
    monkeypatch.setattr(dump, "ON", True)
    for k, v in {"GLM53_TF_DRAFT_DUMP": str(tmp_path), "GLM53_TF_BATCH": "2", "GLM53_TF_LOOKUP": "0",
                 "GLM53_TF_BATCH_RESERVE_GB": "0.25", "GLM53_TF_BATCH_ADMIT_GB": "0",
                 "GLM53_TF_NONEXPERT": "q4mse"}.items():
        monkeypatch.setenv(k, v)
    eng = GlmEngine(ckpt / "model", rank=0, master="", port=0, drafter=ckpt / "dflash2", comm=_TwoCopies())
    try:
        assert dump.DUMP is not None and dump.DUMP.armed
        prompt = list(np.random.default_rng(3).integers(0, 1000, size=45))

        def gen():
            out = []
            eng.request.policy, eng.request.stop_eos = None, False
            eng.generate(list(prompt), 40, None, lambda new: out.extend(new))
            return out

        with_dump = gen()
        keep, dump.DUMP = dump.DUMP, None
        without = gen()
        dump.DUMP = keep
        assert with_dump == without
        keep.close()
    finally:
        eng.batch.stop()
    dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    docs = dumpdata.build_docs(dirs, min_tokens=1)
    full = [d for d in docs if d.tokens().tolist()[:45] == prompt and d.n > 45]
    assert full, [d.n for d in docs]
    doc = max(full, key=lambda d: d.n)
    tok = doc.tokens().tolist()
    assert tok == prompt + with_dump[:doc.n - 45]
    kinds = doc.kinds()
    assert kinds[:45].sum() == 0 and kinds[45:].all()
    ids = doc.get("ids", 45, doc.n)[:, 0].tolist()
    nxt = tok[46:] + [with_dump[doc.n - 45]] if doc.n - 45 < len(with_dump) else tok[46:]
    assert ids[:len(nxt)] == nxt                      # a greedy reply: each decode row's top-1 is the next token
    assert doc.get("hidden", 0, doc.n).shape == (doc.n, 512)
    assert doc.taps(0, doc.n).shape[1] == 2 * 512


@pytest.mark.skipif(not (os.environ.get("DRAFT_TEST_MODEL") and os.environ.get("DRAFT_TEST_DUMPS")),
                    reason="the real model: DRAFT_TEST_MODEL=<checkpoint> DRAFT_TEST_DUMPS=<dump dirs, colon-separated>")
def test_reference_reproduces_engine_mtp_acceptance():
    import eval_accept as E
    from dumpdata import build_docs, is_eval

    docs = [d for d in build_docs(os.environ["DRAFT_TEST_DUMPS"].split(":")) if is_eval(d, 100)]
    assert docs, "no held-out documents in the dumps"
    m = R.build_mtp(os.environ["DRAFT_TEST_MODEL"], device="cuda", mode=os.environ.get("GLM53_TF_NONEXPERT", "q4mse"),
                    kv_fp8=os.environ.get("GLM53_TF_KV_DTYPE", "fp8") == "fp8",
                    expert_cache=os.environ.get("DRAFT_TEST_EXPERTS") or None)
    rep = E.eval_mtp(m, docs[:int(os.environ.get("DRAFT_TEST_DOCS", "60"))], steps=3, window=1024, ctx=1024,
                     max_rows=int(os.environ.get("DRAFT_TEST_ROWS", "60000")), samples=32,
                     sparse_docs=int(os.environ.get("DRAFT_TEST_SPARSE_DOCS", "16384")))
    print(json.dumps(rep, indent=1))
    base = [float(x) for x in os.environ.get("DRAFT_TEST_BASELINE", "0.74,0.45,0.22").split(",")]
    tol = float(os.environ.get("DRAFT_TEST_TOL", "0.08"))
    got = rep["kind_d"]["data"]["a"]
    for k, (g, b) in enumerate(zip(got, base), start=1):
        assert abs(g - b) <= tol, f"position {k}: reference {g:.3f} vs engine {b:.3f} (tolerance {tol})"
