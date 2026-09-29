"""bench/divergence.py on the CPU: the top-k KL estimator against the full-vocabulary KL (exact when k covers the
vocabulary, a lower bound otherwise, 0 for identical inputs), and ``compare`` end to end over synthetic
patches/0430 dump folders written by the engine's own writer (``dump._Writer``): documents matched by their token
ids whatever order and chunking the two dumps used, A/A == 0, a perturbed candidate > 0.

    PYTHONPATH=<patched tree>/src pytest -q tests/test_divergence.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("divergence", ROOT / "bench" / "divergence.py")
div = importlib.util.module_from_spec(spec)
spec.loader.exec_module(div)


def _dist(rng, T, V, temp=1.0):
    z = rng.standard_normal((T, V)) * 3 / temp
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def _topk(lp, k):
    ids = np.argsort(-lp, axis=-1)[:, :k]
    return np.take_along_axis(lp, ids, -1), ids.astype(np.int32)


def test_exact_when_k_is_the_vocabulary():
    rng = np.random.default_rng(0)
    a = _dist(rng, 50, 40)
    b = a + rng.standard_normal(a.shape) * 0.1
    b = b - np.log(np.exp(b).sum(-1, keepdims=True))
    full = (np.exp(a) * (a - b)).sum(-1)
    s = div.position_stats(*_topk(a, 40), *_topk(b, 40))
    assert np.allclose(s["kl"], full, atol=1e-9)
    assert s["covered"].all()


def test_lower_bound_and_zero():
    rng = np.random.default_rng(1)
    a = _dist(rng, 200, 500)
    b = a + rng.standard_normal(a.shape) * 0.3
    b = b - np.log(np.exp(b).sum(-1, keepdims=True))
    full = (np.exp(a) * (a - b)).sum(-1)
    s = div.position_stats(*_topk(a, 32), *_topk(b, 32))
    assert (s["kl"] <= full + 1e-9).all()
    assert s["kl"].mean() > 0.5 * full.mean()
    z = div.position_stats(*_topk(a, 32), *_topk(a, 32))
    assert (z["kl"] == 0).all() and z["top1"].all()


def test_true_token_logprob():
    lp = np.log(np.array([[0.7, 0.2, 0.1]]))
    ids = np.array([[5, 9, 2]], dtype=np.int32)
    s = div.position_stats(lp, ids, lp, ids, np.array([9]))
    assert np.isclose(s["nll_ref"][0], -np.log(0.2))
    s = div.position_stats(lp, ids, lp, ids, np.array([7]))
    assert np.isnan(s["nll_ref"][0])


def _write_dump(root: Path, docs, perturb: float, chunk: int, seed: int, order):
    """A dump folder as the engine's writer makes it: every document's prefill chunks (kind p), linked by hashes."""

    dump = pytest.importorskip("tensorfold.families.glm5_next.cuda.dump")
    w = dump._Writer(root, {"version": 1, "topk": 32})
    rng = np.random.default_rng(seed)
    for di in order:
        toks, lp = docs[di]
        if perturb:
            lp = lp + rng.standard_normal(lp.shape) * perturb
            lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
        h = dump._hash_ids([])
        for a in range(0, len(toks), chunk):
            b = min(a + chunk, len(toks))
            h0 = h.hexdigest()
            h.update(np.asarray(toks[a:b], dtype="<i4").tobytes())
            top, ids = _topk(lp[a:b], 32)
            w.put({"kind": "p", "start": a, "n": b - a, "h0": h0, "h1": h.hexdigest(), "t": 0.0, "tap_layers": [],
                   "arrays": {"tokens": np.asarray(toks[a:b], dtype="<i4"), "lp": top.astype(np.float16), "ids": ids},
                   "dtypes": {}})
    w.close()
    return root


def test_compare_end_to_end(tmp_path, capsys):
    rng = np.random.default_rng(3)
    docs = []
    for n in (70, 130, 45):
        toks = rng.integers(0, 300, size=n).astype(np.int32)
        docs.append((toks, _dist(rng, n, 300)))
    ref = _write_dump(tmp_path / "ref", docs, 0.0, 64, 0, [0, 1, 2])
    same = _write_dump(tmp_path / "same", docs, 0.0, 32, 0, [2, 0, 1])          # other chunking and order
    cand = _write_dump(tmp_path / "cand", docs, 0.2, 48, 1, [1, 2, 0])
    run = tmp_path / "run.json"
    run.write_text(json.dumps({"docs": [{"id": i, "source": s, "prompt_tokens": len(docs[i][0])}
                                        for i, s in enumerate(["wiki", "code", "chat"])]}))

    class A:
        pass

    for other, label in ((same, "aa"), (cand, "cand")):
        a = A()
        a.ref, a.cand, a.run, a.label, a.out = [str(ref)], [str(other)], str(run), label, str(tmp_path / f"{label}.json")
        div.compare(a)
        res = json.loads((tmp_path / f"{label}.json").read_text())
        o = res["overall"]
        assert o["documents"] == 3 and o["positions"] == 245
        assert set(res["sources"]) == {"wiki", "code", "chat"}
        if label == "aa":
            assert o["kl_mean"] == 0 and o["kl_max"] == 0 and o["top1_agreement"] == 1
        else:
            assert o["kl_mean"] > 0.001 and o["top1_agreement"] < 1
    assert "common (same token ids) 3" in capsys.readouterr().out
