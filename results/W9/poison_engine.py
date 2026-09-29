"""W9 0310 IMA hypothesis: an engine's construction (Graphs warm-up) reads an uninitialized device buffer as an index.
Poison device memory (fill ~GIB GiB with a byte, free it back to the driver), then build the tests' toy engines as
test_prefix_share_patches' test 29 does (ref: batch 1, no store; then eng: batch 4, share on, piece 128), and
generate once. Run under CUDA_LAUNCH_BLOCKING=1 so a fault names its kernel. Usage: poison_engine.py BYTE GIB ROUNDS"""
import sys, os, tempfile, pathlib
import torch
sys.path[:0] = ["/src/TensorFold/tests/cuda", "/work/tests/cuda"]
byte, gib, rounds = int(sys.argv[1], 0), float(sys.argv[2]), int(sys.argv[3])
from test_glm_engine import _checkpoint, _drafter
from test_batch_sessions_patches import _engine as build, _free
path = pathlib.Path(tempfile.mkdtemp())
_checkpoint(path / "model", exl3=True); _drafter(path / "dflash2")
def poison():
    blocks = []
    n = int(gib * (1 << 30)) // (256 << 20)
    for _ in range(n):
        blocks.append(torch.full((256 << 20,), byte, dtype=torch.uint8, device="cuda"))
    torch.cuda.synchronize(); del blocks; torch.cuda.empty_cache()
os.environ.update(GLM53_TF_PREFIX_SHARE="1", GLM53_TF_PREFIX_SHARE_TOKENS="1010", GLM53_TF_PREFIX_SHARE_MIN="256",
                  GLM53_TF_PREFIX_SHARE_WAIT="1")
for r in range(rounds):
    poison()
    ref = build(path, batch=1, gib=0, on=False)
    torch.cuda.synchronize()
    print(f"round {r}: ref engine built after poison 0x{byte:02x}", flush=True)
    poison()
    eng = build(path, batch=4, piece=128, fork=64)
    torch.cuda.synchronize()
    print(f"round {r}: share engine built", flush=True)
    _free(ref, eng); del ref, eng
    import gc; gc.collect(); torch.cuda.empty_cache()
print("POISON OK", flush=True)
