"""W12 probe: is test_four_requests_equal_serial's failure the harness or 0450? The same test with parts '0' (knob off
on both engines) and with 'sample'."""
import os, sys
os.environ["TRITON_INTERPRET"] = "0"
sys.path[:0] = ["/work/tests/cuda", "/work/tests"]
import pytest, torch
import test_gpu_round_patches as T

@pytest.fixture(scope="module")
def gckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter
    path = tmp_path_factory.mktemp("glm_gpuround")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path

@pytest.mark.parametrize("parts", ["0", "sample"])
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
def test_probe(gckpt, parts, greedy):
    from test_batch2_patches import _prompt, _sampling
    from test_batch_parallel_patches import _serial
    ref = T._gpu_engine(gckpt, "0")
    sampling = _sampling(greedy)
    prompts = [_prompt(470 + i, 30 + 17 * i) for i in range(4)]
    want = [_serial(ref, p, sampling, 64) for p in prompts]
    want2 = [_serial(ref, p, sampling, 64) for p in prompts]
    assert want == want2, "serial not repeatable on one engine"
    del ref; torch.cuda.empty_cache()
    e = T._gpu_engine(gckpt, parts, batch=4)
    ser = [_serial(e, p, sampling, 64) for p in prompts]
    print("serial on the batch engine == ref serial:", ser == want, [a[:6] for a in ser], [a[:6] for a in want])
    got = e.batch.generate_batch([dict(prompt=p, max_tokens=64, sampling=sampling, policy=pol) for p, pol in zip(prompts, ("o", "o", "of", "om"))])
    print("batched == ref serial:", [t for t, _ in got] == want, "batched == own serial:", [t for t, _ in got] == ser)
    assert [t for t, _ in got] == want
