"""patches/0400 (``kda_v2``, GLM53_TF_KDA_V2) on a GPU: BIT-IDENTICAL to ``fast_kda`` on the real shape (32 heads).

Every comparison is ``torch.equal`` on the raw bits (outputs as int16, state as int32):
- every mode and setting (split: value rows 16 / 32 / 64, 2 / 4 / 8 warps, K split on / off, register caps; fused:
  32 / 64 value rows, K split, lag / ring, 1 / 2 / 8 / all programs) == ``fast_kda.kda_prefill_chunked`` at 1 / 63 /
  64 / 65 / 512 / 2,048 / 8,192 rows, aligned and unaligned starts, with the production-like and a large-state input;
- repeated runs are identical (the fused kernel's programs race for tickets: 50 runs, and runs with the GPU shared
  with a busy stream), and ``state_out`` may alias ``state_in``;
- resume == fresh: a prompt cut on the 64-row grid into calls of any sizes (the lean sub-blocks, 0335's solo pieces)
  == one call, and each cut's state (a snapshot) == ``fast_kda``'s at that cut;
- through the engine's entry (``fastpf.kda_chain``) with GLM53_TF_KDA_V2 set: same bits as unset.
A failure here means the mode must stay off (the design says same bits; this file is where the GPU agrees or not).

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda pytest -q -s tests/cuda/test_kda_v2_patches.py (~5 min)
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.glm5_next.cuda import fast_kda, fastpf, kda_v2  # noqa: E402

DEV = "cuda"
H = 32
C = 3 * H * 128
B_OFF = C + 256
WIDTH = B_OFF + H


def _inputs(rows, seed=0, big_state=False):
    g = torch.Generator(device=DEV).manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, device=DEV)                          # noqa: E731
    p = r(rows, WIDTH)
    a = r(rows, H * 128)
    if big_state:                    # slow channels keep a large state, small v: last bits of the scan matter
        slow = torch.rand(1, H * 128, generator=g, device=DEV) < 0.75
        a = torch.where(slow, a * 0.5 - 9.0, a * 0.5 + 12.0)
        p[:, 2 * H * 128:3 * H * 128] *= 0.05
    return dict(p=p.bfloat16(), a=a.bfloat16(), g=r(rows, H * 128).bfloat16(), conv_state=r(3, C).bfloat16(),
                conv_w=(r(C, 4) * 0.5).bfloat16(), state_in=r(H, 128, 128) * (3.0 if big_state else 0.1),
                a_log=r(H) * 0.3, dt_bias=r(H * 128) * 0.3, norm_w=(1 + 0.1 * r(128)).bfloat16(), eps=1e-6,
                lower=-5.0)


def _args(d, rows):
    return (d["p"], B_OFF, d["a"], d["g"], d["conv_state"], d["conv_w"], d["state_in"], d["a_log"], d["dt_bias"],
            d["norm_w"], d["eps"], d["lower"], rows)


def _old(d, rows, pos):
    so = torch.empty_like(d["state_in"])
    o = fast_kda.kda_prefill_chunked(*_args(d, rows), None, so, pos=pos)
    return o.clone(), so


def _cfg(**kw):
    cfg = dict(bv=32, warps=4, maxnreg=168, ksplit=1, ctas=0, lag=1, ring=3, fbv=64, fmaxnreg=0)
    cfg.update(kw)
    return cfg


def _new(d, rows, pos, mode, **kw):
    so = torch.empty_like(d["state_in"])
    o = kda_v2.kda_prefill_v2(*_args(d, rows), None, so, pos=pos, v2=mode, cfg=_cfg(**kw))
    return o.clone(), so


def _same(x, y, what=""):
    (o1, s1), (o2, s2) = x, y
    no = int((o1.view(torch.int16) != o2.view(torch.int16)).sum())
    ns = int((s1.view(torch.int32) != s2.view(torch.int32)).sum())
    assert no == 0 and ns == 0, f"{what}: {no} output / {ns} state elements differ"


SPLIT = [dict(bv=32, warps=4, ksplit=1), dict(bv=16, warps=4, ksplit=1), dict(bv=32, warps=8, ksplit=1),
         dict(bv=64, warps=8, ksplit=0), dict(bv=32, warps=4, ksplit=0), dict(bv=64, warps=4, ksplit=1),
         dict(bv=32, warps=2, ksplit=1), dict(bv=32, warps=4, ksplit=1, maxnreg=0),
         dict(bv=16, warps=4, ksplit=1, maxnreg=128)]
FUSED = [dict(fbv=64, ksplit=0), dict(fbv=64, ksplit=1), dict(fbv=32, ksplit=1), dict(fbv=32, ksplit=0),
         dict(fbv=64, ksplit=0, lag=1, ring=2), dict(fbv=64, ksplit=0, lag=2, ring=4), dict(fbv=64, ctas=1),
         dict(fbv=64, ctas=2), dict(fbv=32, ctas=8), dict(fbv=64, ctas=96),
         dict(fbv=32, ksplit=1, fmaxnreg=128)]
VARIANTS = [(1, kw) for kw in SPLIT] + [(2, kw) for kw in FUSED]
IDS = [f"m{m}-" + "-".join(f"{k}{v}" for k, v in kw.items()) for m, kw in VARIANTS]


@pytest.mark.parametrize("big", [False, True])
@pytest.mark.parametrize("rows,pos", [(1, 0), (63, 64), (64, 0), (65, 128), (512, 0), (512, 4096), (1000, 37),
                                      (2048, 0)])
@pytest.mark.parametrize("mode,kw", VARIANTS, ids=IDS)
def test_same_bits_as_fast_kda(mode, kw, rows, pos, big):
    d = _inputs(rows, seed=rows + pos, big_state=big)
    _same(_new(d, rows, pos, mode, **kw), _old(d, rows, pos), f"mode {mode} {kw} rows {rows} pos {pos}")


@pytest.mark.parametrize("mode,kw", [(1, dict(bv=32, warps=4, ksplit=1)), (1, dict(bv=16, warps=4, ksplit=1)),
                                     (2, dict(fbv=64, ksplit=0)), (2, dict(fbv=32, ksplit=1))])
def test_same_bits_8192(mode, kw):
    rows = 8192
    for big in (False, True):
        d = _inputs(rows, seed=99, big_state=big)
        _same(_new(d, rows, 0, mode, **kw), _old(d, rows, 0), f"8192 big={big}")


@pytest.mark.parametrize("mode,kw", [(2, dict(fbv=64, ksplit=0)), (2, dict(fbv=32, ksplit=1, ring=2)),
                                     (1, dict(bv=32, warps=4, ksplit=1))])
def test_repeated_runs_identical(mode, kw):
    """The fused kernel's programs take tickets in a racy order and wait on flags: 50 runs, then 20 more while
    another stream keeps the GPU busy (fewer resident programs, other interleavings)."""

    rows = 2048
    d = _inputs(rows, seed=5, big_state=True)
    ref = _old(d, rows, 0)
    for n in range(50):
        _same(_new(d, rows, 0, mode, **kw), ref, f"run {n}")
    side = torch.cuda.Stream()
    x = torch.randn(4096, 4096, device=DEV)
    for n in range(20):
        with torch.cuda.stream(side):
            for _ in range(3):
                x = (x @ x).clamp_(-1, 1)
        _same(_new(d, rows, 0, mode, **kw), ref, f"busy run {n}")
    torch.cuda.synchronize()


def _rest(d, cut, state):
    return dict(d, p=d["p"][cut:], a=d["a"][cut:], g=d["g"][cut:], conv_state=d["p"][cut - 3:cut, :C].contiguous(),
                state_in=state)


@pytest.mark.parametrize("mode,kw", [(1, dict(bv=32, warps=4, ksplit=1)), (2, dict(fbv=64, ksplit=0)),
                                     (2, dict(fbv=32, ksplit=1))])
@pytest.mark.parametrize("cuts", [[512], [64, 576], [1024, 1536, 3072], [8192 - 64]])
def test_resume_equals_fresh(mode, kw, cuts):
    rows, pos0 = 8192, 2048
    d = _inputs(rows, seed=7, big_state=True)
    whole = _new(d, rows, pos0, mode, **kw)
    _same(whole, _old(d, rows, pos0), "whole")
    outs, st, prev = [], d["state_in"], 0
    for cut in cuts + [rows]:
        e = _rest(d, prev, st) if prev else d
        o, st = _new(e, cut - prev, pos0 + prev, mode, **kw)
        _same((o, st), _old(e, cut - prev, pos0 + prev), f"piece ending at {cut}")      # the snapshot at the cut
        outs.append(o)
        prev = cut
    assert torch.equal(torch.cat(outs).view(torch.int16), whole[0].view(torch.int16))
    assert torch.equal(st.view(torch.int32), whole[1].view(torch.int32))


@pytest.mark.parametrize("mode", [1, 2])
def test_state_out_aliases_state_in(mode):
    rows = 1000
    d = _inputs(rows, seed=3, big_state=True)
    ref = _old(d, rows, 64)
    s = d["state_in"].clone()
    o = kda_v2.kda_prefill_v2(*_args(dict(d, state_in=s), rows), None, s, pos=64, v2=mode, cfg=_cfg())
    _same((o, s), ref)


@pytest.mark.parametrize("env", ["1", "2", "split", "fused"])
def test_engine_entry(monkeypatch, env):
    rows = 512
    d = _inputs(rows, seed=13)
    so0, so1 = torch.empty_like(d["state_in"]), torch.empty_like(d["state_in"])
    monkeypatch.delenv("GLM53_TF_KDA_V2", raising=False)
    o0 = fastpf.kda_chain(*_args(d, rows), None, so0, pos=1024).clone()
    monkeypatch.setenv("GLM53_TF_KDA_V2", env)
    o1 = fastpf.kda_chain(*_args(d, rows), None, so1, pos=1024).clone()
    _same((o1, so1), (o0, so0))
