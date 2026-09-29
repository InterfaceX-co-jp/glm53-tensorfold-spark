import pytest
from test_kv_pool_patches import long_ckpt, _engine, gpu  # noqa: F401


@gpu
@pytest.mark.parametrize("pool", [0, 8192])
@pytest.mark.parametrize("extend", [False, True])
def test_debug_disk(long_ckpt, tmp_path, pool, extend):
    import numpy as np
    from test_glm_engine import _generate
    ref = _engine(long_ckpt)
    eng = _engine(long_ckpt, pool=pool, batch=3, gib=0.05, sessions_on=True, disk=tmp_path)
    rng = np.random.default_rng(51)
    prompts = [[int(t) for t in rng.integers(0, 1000, size=n)] for n in (2200, 2300, 2400, 2500)]
    for rnd in range(2):
        if rnd and extend:
            prompts = [p + [1] * 40 for p in prompts]
        got = eng.batch.generate_batch([dict(prompt=p, max_tokens=12, sampling=None) for p in prompts])
        for i, (p, (reply, stats)) in enumerate(zip(prompts, got)):
            ok = reply == _generate(ref, p, None, draft=False, tokens=12)[0]
            print("DBG", pool, extend, rnd, i, ok, {k: v for k, v in stats.items() if k in ("kv_pages", "cached", "slot", "restored_disk")}, flush=True)
        print("DBG counts", dict(eng.batch.counts), flush=True)
    eng.batch.stop()
