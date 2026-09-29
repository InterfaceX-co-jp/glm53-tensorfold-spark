import os, sys
os.environ["TRITON_INTERPRET"] = "0"
import numpy as np, torch, pytest
import test_gpu_round_patches as T
from tensorfold.families.glm5_next.cuda import decode, gpusample, comm as comm_mod
from tensorfold.engine.exact_sampling import choose_rows

def test_dbg(tmp_path):
    from test_glm_engine import _checkpoint, _drafter
    from test_batch2_patches import _prompt, _sampling
    from test_batch_parallel_patches import _serial
    _checkpoint(tmp_path / "model", exl3=True); _drafter(tmp_path / "dflash2")
    e = T._gpu_engine(tmp_path, "sample")
    orig = decode._sample_device
    calls = []
    def spy(w, vals, ids, positions, sampling, k, world, probs):
        print("spy: vals", vals.dtype, tuple(vals.shape), "ids", ids.dtype, tuple(ids.shape), "k", k, "world", world,
              "comm", type(w.comm).__name__, "offset", w.vocab_offset, flush=True)
        packed = torch.cat([vals, ids.view(torch.float32)], dim=1).contiguous().view(-1)
        got = torch.empty((world * packed.numel(),), dtype=torch.float32, device=vals.device)
        comm_mod.fast_gather(w.comm, packed, got)
        g = got.view(world, -1).cpu()
        v = g[:, :k].numpy(); i = g[:, k:2 * k].contiguous().view(torch.int32).numpy()
        print("  gathered ids rank0", i[0][:6], "rank1", i[1][:6] if world > 1 else None, "local ids", ids[0][:6].tolist(), flush=True)
        out = orig(w, vals, ids, positions, sampling, k, world, probs)
        print("  device ->", out, flush=True)
        calls.append(out)
        return out
    decode._sample_device = spy
    s = _sampling(True)
    p = _prompt(470, 30)
    got = _serial(e, p, s, 3)
    print("serial with knob:", got, flush=True)
    assert calls
