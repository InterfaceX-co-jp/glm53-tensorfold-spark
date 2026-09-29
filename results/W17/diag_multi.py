"""W17 diagnostic: test_gpu_group_equals_alone's second wave resumed 0 tokens (slot 0). Print every member's stats in
both waves with the knob on and off (same batch engine otherwise), no cached assert; replies vs alone reported."""
import numpy as np
import pytest
import torch
from test_multi_prefill_patches import gckpt, _gpu_engine, _alone  # noqa: F401


@pytest.mark.parametrize("multi", [True, False], ids=["multi", "nomulti"])
def test_diag(gckpt, multi):
    lone = _gpu_engine(gckpt, batch=1, multi=False, overlap="0")
    eng = _gpu_engine(gckpt, batch=4, multi=multi, overlap="0")
    rng = np.random.default_rng(17)
    prompts = [[int(t) for t in rng.integers(0, 1000, size=n)] for n in (130, 128, 300, 70)]
    policies = ["2", "auto", "f3", "2"]
    for wave in range(3):
        got = eng.batch.generate_batch([dict(prompt=p, max_tokens=16, sampling=None, policy=pol)
                                        for p, pol in zip(prompts, policies)])
        for i, (p, (reply, stats)) in enumerate(zip(prompts, got)):
            same = reply == _alone(lone, p, None, 16)
            print(f"DIAG multi={multi} wave {wave} member {i} n={len(p)} same={same} "
                  f"{ {k: stats.get(k) for k in ('cached', 'slot', 'multi_prefill', 'cache_src', 'pieces')} }", flush=True)
        prompts = [p + r + [5, 6, 7] for p, (r, _) in zip(prompts, got)]
    eng.batch.stop()
    torch.cuda.empty_cache()
