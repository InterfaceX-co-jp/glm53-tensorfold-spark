"""patches/0500: image rows through ``glue.embed`` in Triton's CPU interpreter, no GPU.

- Inside ``vision.active(table)`` the embedding writes the table's row in every hyper-connection stream for each
  virtual id and the vocabulary row for every other id, for the BF16 table (EXL3 checkpoints) and the 4-bit table
  (MLX checkpoints), for the main forward (4 copies) and the MTP head's rows (1 copy); the kernel never reads past
  the vocabulary (virtual ids are clamped before it runs).
- Outside the block, or while a CUDA graph is being captured, ``glue.embed`` is exactly what it was.

Run: TRITON_INTERPRET=1 PYTHONPATH=<patched TensorFold>/src pytest -q tests/test_vision_interpreter.py
(the module sets TRITON_INTERPRET itself when it is imported first).
"""

from __future__ import annotations

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
glue = pytest.importorskip("tensorfold.families.glm5_next.cuda.glue")
vision = pytest.importorskip("tensorfold.families.glm5_next.cuda.vision")
vp = pytest.importorskip("tensorfold.families.glm5_next.cuda.vision_prep")

if os.environ.get("TRITON_INTERPRET") != "1" or type(glue._embed_b16).__name__ != "InterpretedFunction":
    pytest.skip("needs Triton's interpreter (TRITON_INTERPRET=1 before triton is imported)", allow_module_level=True)
if torch.cuda.is_available():
    pytest.skip("CPU-interpreter checks: run where no GPU is visible", allow_module_level=True)

V, D = 40, 128


def table_for(vids, d=D, seed=0):
    g = torch.Generator().manual_seed(seed)
    keys = sorted(vids)
    rows = torch.randn(len(keys), d, generator=g).to(torch.bfloat16)
    return vision.Table(torch.tensor(keys, dtype=torch.int32), rows), dict(zip(keys, rows))


def q4_table(seed=1):
    g = torch.Generator().manual_seed(seed)
    words = torch.randint(-2 ** 31, 2 ** 31 - 1, (V, D // 8), generator=g, dtype=torch.int64).to(torch.int32)
    scales = torch.rand(V, D // 64, generator=g).to(torch.bfloat16)
    biases = torch.randn(V, D // 64, generator=g).to(torch.bfloat16)
    return words, scales, biases


@pytest.mark.parametrize("q4", [False, True])
@pytest.mark.parametrize("copies", [4, 1])
def test_image_rows_replace_virtual_ids(q4, copies):
    emb = q4_table() if q4 else torch.randn(V, D).to(torch.bfloat16)
    vids = vp.derive(b"x", 5)
    t, by_id = table_for(vids)
    ids = torch.tensor([3, vids[0], 17, vids[3], vids[4], 0, vids[1], V - 1, vids[2]], dtype=torch.int32)
    plain_ids = torch.where(ids >= vp.VBASE, torch.zeros_like(ids), ids)
    ref = torch.empty(len(ids), copies * D, dtype=torch.bfloat16)
    glue.embed(plain_ids, emb, D, copies, ref)                           # the vocabulary rows (row 0 for images)
    out = torch.full_like(ref, 5.0)
    with vision.active(t):
        glue.embed(ids, emb, D, copies, out)
    o, r = out.view(-1, copies, D), ref.view(-1, copies, D)
    for i, tok in enumerate(ids.tolist()):
        for c in range(copies):
            want = by_id[tok] if tok >= vp.VBASE else r[i, c]
            assert torch.equal(o[i, c], want), (i, tok, c)


def test_nothing_changes_outside_or_while_capturing(monkeypatch):
    emb = torch.randn(V, D).to(torch.bfloat16)
    ids = torch.tensor([1, 2, 39], dtype=torch.int32)
    a = torch.empty(3, 4 * D, dtype=torch.bfloat16)
    glue.embed(ids, emb, D, 4, a)
    t, _ = table_for(vp.derive(b"y", 2))
    b = torch.empty_like(a)
    with vision.active(t):
        glue.embed(ids, emb, D, 4, b)                                    # a table, no virtual ids: same rows
    assert torch.equal(a, b)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with vision.active(t):
        assert vision.embed_ids(ids) == (None, ids)                      # graphs never get the substitution
