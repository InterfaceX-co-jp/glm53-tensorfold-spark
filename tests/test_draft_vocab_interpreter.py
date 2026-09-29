"""patches/0420 (GLM53_TF_DRAFT_VOCAB) in Triton's CPU interpreter: the trimmed draft head (``draftvocab.take_rows``)
through the real 4-bit matmul (``qmm.matmul``) gives the full head's listed columns bit for bit, for 1-16 rows, when
the two shapes take the same K split (``qmm.split_k``; production: 1 at 12,288+ listed rows a rank and at 77,440).

Run: TRITON_INTERPRET=1 PYTHONPATH=<patched TensorFold>/src pytest -q tests/test_draft_vocab_interpreter.py (~1 min).
"""

from __future__ import annotations

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
qmm = pytest.importorskip("tensorfold.families.glm5_next.cuda.qmm")
dvm = pytest.importorskip("tensorfold.families.glm5_next.cuda.draftvocab")

INTERP = os.environ.get("TRITON_INTERPRET") == "1" and type(qmm._qmm).__name__ == "InterpretedFunction"
interp = pytest.mark.skipif(not INTERP or torch.cuda.is_available(),
                            reason="Triton's CPU interpreter, no GPU (tests/cuda/test_draft_vocab_patches.py on GPUs)")


@interp
@pytest.mark.parametrize("n,k,m", [(640, 512, 128), (1280, 256, 192), (256, 4096, 64)])
def test_trimmed_head_equals_full_columns(n, k, m):
    g = torch.Generator().manual_seed(n + k)
    q = qmm.quantize4((torch.randn((n, k), generator=g) * 0.05).to(torch.bfloat16), mse=True)
    rows = torch.randperm(n, generator=g)[:m]
    t = dvm.take_rows(q, rows)
    assert qmm.split_k(n, k) == qmm.split_k(m, k)
    for r in (1, 3, 7, 16):
        x = (torch.randn((r, k), generator=g)).to(torch.bfloat16)
        xs = qmm.group_sums(x)
        full = qmm.matmul(x, q, xs)
        trim = qmm.matmul(x, t, xs)
        assert trim.shape == (r, m)
        assert torch.equal(trim, full[:, rows]), (n, k, m, r)
