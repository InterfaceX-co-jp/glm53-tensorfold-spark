"""patches/0500 on a GPU (TensorFold's synthetic checkpoint, one GPU playing rank 0 of two), plus the real tower when
$GLM53_TF_MODEL points at the GLM-5.3-Flash checkpoint.

The substitution is exact everywhere: a vision table whose rows are the embedding rows of real tokens T makes a
prompt ``text + virtual ids`` reply exactly as ``text + T`` does, serial and drafted (MTP absorb included), greedy
and sampled (same seed), with 1-row, 16-row and default prefill chunks (image rows across chunk boundaries and in
the last small chunk), fast prefill on and off, resumed == fresh (the prompt's own snapshot and a longer turn). A
table with other rows (another image, same text) replies differently and never resumes past its first row.
``glue.embed`` with the compiled kernels == the interpreter's rule (tests/test_vision_interpreter.py); inside a
CUDA graph capture nothing is substituted.

With $GLM53_TF_MODEL: the checkpoint's tower (``Tower.load``) on a 1024 x 768 image: 1,036 rows, deterministic
(two runs bit-identical), bf16 against an fp32 run of the same tower within 8% relative (W14: bf16 itself gives ~6%),
latency and peak memory printed (docs/VISION.md, GPU plan step 5).

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q -s tests/cuda/test_vision_patches.py
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.glm5_next.cuda import glue, vision, vision_prep as vp  # noqa: E402
from test_glm_engine import _TwoCopies, _checkpoint, _generate  # noqa: E402


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    path = tmp_path_factory.mktemp("glm_vision")
    _checkpoint(path)
    return GlmEngine(path, rank=0, master="", port=0, comm=_TwoCopies())


def rows_of(engine, toks):
    """The embedding rows the engine gives these tokens (one stream copy)."""

    D = engine.w.cfg.hidden
    out = torch.empty((len(toks), D), dtype=torch.bfloat16, device="cuda")
    glue.embed(torch.tensor(toks, dtype=torch.int32, device="cuda"), engine.w.embed, D, 1, out)
    return out


def with_table(engine, vids, rows):
    order = sorted(range(len(vids)), key=vids.__getitem__)
    table = vision.Table(torch.tensor([vids[i] for i in order], dtype=torch.int32, device="cuda"),
                         rows[torch.tensor(order, device="cuda")])
    engine.vision = SimpleNamespace(table=lambda req: table, last={"images": 1})
    engine.request.vision = SimpleNamespace(images=[])
    return table


def gen(engine, prompt, sampling, **kw):
    kw.setdefault("tokens", 20)
    engine.request.knobs = kw.pop("knobs", None)
    return _generate(engine, prompt, sampling, **kw)


SAMPLED = Sampling(4321, 1.0, 20, 0.95)


@pytest.mark.parametrize("sampling", [SAMPLED, None], ids=["sampled", "greedy"])
@pytest.mark.parametrize("where", ["middle", "end"])
def test_image_rows_equal_their_tokens(engine, sampling, where):
    rng = np.random.default_rng(11)
    text = list(rng.integers(0, 1000, size=40))
    T = list(rng.integers(0, 1000, size=37))                     # the "image": 37 rows
    vids = vp.derive(b"tokens-as-image", len(T))
    tail = [5, 6, 7] if where == "end" else list(rng.integers(0, 1000, size=30))
    real, virt = text + T + tail, text + vids + tail
    try:
        engine.cache = []
        ref, _ = gen(engine, real, sampling, draft=False)
        for knobs in (None, {"prefill_rows": 1}, {"prefill_rows": 16}, {"fast_prefill": 1}):
            try:
                engine.parse_knobs(knobs)
            except ValueError:                                   # e.g. no fast kernels in this build
                continue
            with_table(engine, vids, rows_of(engine, T))
            engine.cache = []
            engine.request.vision = SimpleNamespace(images=[])
            got, _ = gen(engine, virt, sampling, draft=False, knobs=knobs)
            if knobs and knobs.get("fast_prefill"):
                engine.cache = []
                want, _ = gen(engine, real, sampling, draft=False, knobs=knobs)
            else:
                want = ref
            assert got == want, knobs
            for policy in (None, "2", "c3:0.35"):
                engine.cache = []
                drafted, _ = gen(engine, virt, sampling, policy=policy, knobs=knobs)
                assert drafted == want, (knobs, policy)
    finally:
        engine.vision = None
        engine.request.vision = None
        engine.request.knobs = None


def test_resumed_equals_fresh_and_images_never_share(engine):
    rng = np.random.default_rng(12)
    text = list(rng.integers(0, 1000, size=70))
    A, B = list(rng.integers(0, 1000, size=33)), list(rng.integers(0, 1000, size=33))
    va, vb = vp.derive(b"A", 33), vp.derive(b"B", 33)
    # W15: 30 tail tokens (was 9) so the prompt's last 64-grid point (128) lies past the image under patches/0540's
    # rule (GLM53_TF_SNAPSHOT_BEFORE_END: the snapshot strictly before the end; 112 tokens put it at 64, in the text)
    tail = list(rng.integers(0, 1000, size=30))
    try:
        engine.cache = []
        with_table(engine, va, rows_of(engine, A))
        fresh, s1 = gen(engine, text + va + tail, None)
        with_table(engine, va, rows_of(engine, A))
        again, s2 = gen(engine, text + va + tail + fresh[:5], None)          # the next turn resumes past the image
        assert s2["cached"] >= len(text) + 33
        engine.cache = []
        with_table(engine, va, rows_of(engine, A))
        cold, _ = gen(engine, text + va + tail + fresh[:5], None)
        assert again == cold
        # the other image with the same text: resumes at most the text, replies as its own tokens do
        with_table(engine, vb, rows_of(engine, B))
        other, s3 = gen(engine, text + vb + tail, None)
        assert s3["cached"] <= len(text)
        engine.cache = []
        want_b, _ = gen(engine, text + B + tail, None, draft=False)
        assert other == want_b and other != fresh
    finally:
        engine.vision = None
        engine.request.vision = None


def test_prompt_with_rows_but_no_images_is_refused(engine):
    engine.vision = None
    engine.request.vision = None
    with pytest.raises(ValueError, match="GLM53_TF_VISION"):
        gen(engine, [1, 2, *vp.derive(b"x", 3), 4], None)


def test_embed_compiled_and_graph_capture(engine):
    D = engine.w.cfg.hidden
    vids = vp.derive(b"g", 4)
    rows = torch.randn(4, D, device="cuda").to(torch.bfloat16)
    table = vision.Table(torch.tensor(sorted(vids), dtype=torch.int32, device="cuda"),
                         rows[torch.tensor(sorted(range(4), key=vids.__getitem__), device="cuda")])
    ids = torch.tensor([3, vids[2], 9, vids[0]], dtype=torch.int32, device="cuda")
    out = torch.empty((4, 4 * D), dtype=torch.bfloat16, device="cuda")
    with vision.active(table):
        glue.embed(ids, engine.w.embed, D, 4, out)
    o = out.view(4, 4, D)
    assert all(torch.equal(o[1, c], rows[2]) and torch.equal(o[3, c], rows[0]) for c in range(4))
    base = rows_of(engine, [3, 9])
    assert torch.equal(o[0, 2], base[0]) and torch.equal(o[2, 1], base[1])
    # during a capture the embedding is the plain kernel (graphs only ever hold decode rows)
    plain = torch.tensor([3, 9, 11, 12], dtype=torch.int32, device="cuda")
    g = torch.cuda.CUDAGraph()
    out2 = torch.empty_like(out)
    with vision.active(table):
        glue.embed(plain, engine.w.embed, D, 4, out2)             # warm-up outside the capture
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            glue.embed(plain, engine.w.embed, D, 4, out2)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out2.view(4, 4, D)[:, 0], rows_of(engine, [3, 9, 11, 12]))


# -- the real tower -------------------------------------------------------------------------------------------------
MODEL = os.environ.get("GLM53_TF_MODEL", "")


@pytest.mark.skipif(not MODEL or not os.path.exists(os.path.join(MODEL, "config.json")),
                    reason="GLM53_TF_MODEL: the GLM-5.3-Flash checkpoint folder")
def test_real_tower_screenshot():
    from PIL import Image

    s = vp.Settings.read(MODEL)
    img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (768, 1024, 3), dtype=np.uint8))
    prep = vp.preprocess(img, s)
    assert prep.tokens == 1036 and prep.grid == (1, 56, 74)
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    tw = vision.Tower.load(MODEL)
    print(f"\n[vision] tower {tw.nbytes / 1e9:.3f} GB")
    a = tw.encode(prep.pixels, prep.grid)
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        b = tw.encode(prep.pixels, prep.grid)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    assert a.shape == (1036, 4096) and torch.equal(a, b)
    peak = (torch.cuda.max_memory_allocated() - base) / 1e9
    f = vision.Tower.flops(prep.grid, s.vision_config)
    print(f"[vision] 1024x768: {min(ts) * 1e3:.1f} ms ({f / min(ts) / 1e12:.1f} TFLOP/s), peak +{peak:.2f} GB")
    big = vp.preprocess(Image.fromarray(np.random.default_rng(1).integers(0, 256, (2160, 3840, 3), dtype=np.uint8)), s)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tw.encode(big.pixels, big.grid)
    torch.cuda.synchronize()
    print(f"[vision] 3840x2160 ({big.tokens} rows): {(time.perf_counter() - t0) * 1e3:.0f} ms, peak "
          f"+{(torch.cuda.max_memory_allocated() - base) / 1e9:.2f} GB")
    f32 = vision.Tower.load(MODEL, dtype=torch.float32)
    ref = f32.encode(prep.pixels, prep.grid).float()
    rel = ((a.float() - ref).norm() / ref.norm()).item()
    print(f"[vision] bf16 vs fp32 tower: {rel:.4f} relative")
    # 8%, not 2%: bf16 itself is that far from fp32 on this tower. docs/RESULTS.md W14 (head, transformers 5.17,
    # same inputs): transformers' own Glm5NextVisionModel bf16 vs fp32 is 0.060 on this random image (ours 0.059),
    # while our fp32 vs theirs is 0.004; the old 2% bound was below what bf16 gives and failed on a correct tower.
    assert rel < 8e-2
