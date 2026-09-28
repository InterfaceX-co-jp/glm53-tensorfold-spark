"""Diagnostic: which prefill chunkings agree past the 2,051-token dense limit (synthetic EXL3 checkpoint)."""
import sys, tempfile
from pathlib import Path
import numpy as np, pytest, torch
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda import weights
from test_glm_engine import _TwoCopies, _checkpoint, _drafter, _generate

nonexpert = sys.argv[1] if len(sys.argv) > 1 else "bf16"
size = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
path = Path(tempfile.mkdtemp())
_checkpoint(path / "model", exl3=True); _drafter(path / "dflash2")
from test_patches import _index_heads_32
_index_heads_32(path / "model")
weights.NONEXPERT = nonexpert
from tensorfold.families.glm5_next.cuda.engine import GlmEngine
import os
prompt = list(np.random.default_rng(31).integers(0, 1000, size=size))
res = {}
for rows in [int(x) for x in os.environ.get("ROWS", "16,64,128,512").split(",")]:
    print("rows", rows, flush=True)
    os.environ["GLM53_TF_PREFILL_ROWS"] = str(rows)
    e = GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", context=4096, comm=_TwoCopies())
    for name, s in (("greedy", None), ("sampled", Sampling(1234, 1.0, 20, 0.95))):
        res[(rows, name, "serial")] = _generate(e, prompt, s, draft=False, tokens=24)[0]
        res[(rows, name, "drafted")] = _generate(e, prompt, s, tokens=24)[0]
    del e; torch.cuda.empty_cache()
for name in ("greedy", "sampled"):
    ref = res[(64, name, "serial")]
    for k, v in res.items():
        if k[1] == name:
            first = next((i for i, (x, y) in enumerate(zip(ref, v)) if x != y), None)
            print(nonexpert, size, k, "same as 64/serial" if v == ref else f"DIFF at {first}")
