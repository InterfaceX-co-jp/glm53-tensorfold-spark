"""patches/0080, fast2 expert kernels (exl3_fast.cu, the default): bit for bit equal to the v1 kernels
(``GLM53_TF_FAST_EXPERTS=v1``, read once a process, so v1 runs in a subprocess), on the real per-rank shapes with
uniform and skewed routing, including a chunk of 4096+ rows (down's large-chunk configuration); row-independent."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

D, NI, E, TOP, LIMIT = 4096, 1024, 288, 8, 10.0


def _run(rows: int, kind: str, sub=None):
    """Y [rows * (TOP + 1), D] of one MoE layer through the fast kernels (Xd too)."""

    sys.path.insert(0, str(Path(__file__).parent))
    from test_fastpf_patches import _fast_group
    from test_patches import _exl3_layer

    from tensorfold.families.glm5_next.cuda import exl3_mm

    ex = _exl3_layer(D, NI, E, seed=5)
    x = (torch.randn((rows, D), generator=torch.Generator().manual_seed(6)) * 0.5).to(torch.bfloat16)
    g = torch.Generator().manual_seed(7)
    w = torch.ones(E) if kind == "uniform" else 1.0 / torch.arange(1, E + 1).float() ** 0.8
    picks = torch.full((rows, TOP + 1), E, dtype=torch.int32)
    picks[:, :TOP] = torch.multinomial(w.expand(rows, E), TOP, replacement=False, generator=g).int()
    if sub is not None:
        x, picks = x[sub], picks[sub]
    n = x.shape[0]
    scratch = exl3_mm.Scratch(n, TOP + 1, D, NI, "cuda")
    y = torch.zeros((n * (TOP + 1), D), dtype=torch.float32, device="cuda")
    exl3_mm.routed(x.cuda(), picks.cuda(), _fast_group(picks, E), ex, scratch, y, n, LIMIT, fast=True)
    torch.cuda.synchronize()
    return y.view(n, TOP + 1, D)[:, :TOP].cpu(), scratch.xd.view(n, TOP + 1, NI)[:, :TOP].cpu()


@gpu
@pytest.mark.parametrize("rows,kind", [(300, "skewed"), (1024, "uniform"), (4160, "skewed")])
def test_fast2_equals_v1(rows, kind, tmp_path):
    out = tmp_path / "v1.pt"
    code = (f"import sys; sys.path[:0] = {[str(Path(__file__).parent)]!r}; import torch; "
            f"from test_fast_experts_patches import _run; torch.save(_run({rows}, {kind!r}), {str(out)!r})")
    env = dict(os.environ, GLM53_TF_FAST_EXPERTS="v1")
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
    y1, xd1 = torch.load(out)
    y2, xd2 = _run(rows, kind)
    assert torch.isfinite(y2).all() and y2.abs().max() > 0
    assert torch.equal(xd2, xd1)
    assert torch.equal(y2, y1)


@gpu
def test_fast2_row_independent():
    y, xd = _run(700, "skewed")
    for sub in ([5], list(range(64, 200)), list(range(699, -1, -3))):
        ys, xds = _run(700, "skewed", sub=torch.tensor(sub))
        assert torch.equal(ys, y[sub]) and torch.equal(xds, xd[sub]), sub[:3]
