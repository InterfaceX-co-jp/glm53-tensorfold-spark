"""Our engine patches on TensorFold's synthetic GLM checkpoint (one GPU playing rank 0 of two).

patches/0001  EXL3 non-expert weights in 4 bits (GLM53_TF_NONEXPERT=q4 | q4mse): the weights really are 4-bit,
              drafted replies still equal serial ones, and plain ``auto`` chooses between drafters again.
patches/0003  GLM53_TF_PREFILL_ROWS: a prompt prefilled in 512-row chunks (several 128-row tiles a matmul)
              leaves the same state as 64-row chunks, so replies and resumes match bit for bit.
patches/0004  sparse attention past 2,051 tokens with long prefill chunks.
patches/0005  GLM53_TF_PROFILE: the prefill timing probe changes no bit and reports every component.
patches/0006  the looped EXL3 expert kernel (one program per expert, n block and split for prefill windows) gives
              the Z bits of upstream's per-member-tile grid: kernel level, and the whole prefilled state.

Run inside the image: PYTHONPATH=/src/TensorFold/tests/cuda pytest -q tests/cuda/test_patches.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.glm5_next.cuda import qmm, weights  # noqa: E402
from test_glm_engine import _TwoCopies, _checkpoint, _drafter, _generate  # noqa: E402

SAMPLINGS = [Sampling(1234, 1.0, 20, 0.95), None]
IDS = ["sampled", "greedy"]


def _engine(path, monkeypatch, nonexpert: str, rows: int):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    monkeypatch.setattr(weights, "NONEXPERT", nonexpert)
    monkeypatch.setenv("GLM53_TF_NONEXPERT", nonexpert)
    monkeypatch.setenv("GLM53_TF_PREFILL_ROWS", str(rows))
    return GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", comm=_TwoCopies())


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm_patch")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


@pytest.fixture(scope="module")
def mp():
    m = pytest.MonkeyPatch()
    yield m
    m.undo()


@pytest.fixture(scope="module")
def e64(ckpt, mp):
    return _engine(ckpt, mp, "q4mse", 64)


@pytest.fixture(scope="module")
def e512(ckpt, mp):
    return _engine(ckpt, mp, "q4mse", 512)


def test_nonexpert_weights_are_4bit(e512):
    from tensorfold.families.glm5_next.cuda.engine import EXL3_AUTO, encode_policy

    layer = e512.w.layers[0]
    assert isinstance(layer.kda.proj, qmm.Q4) and isinstance(layer.kda.o, qmm.Q4)
    assert isinstance(e512.w.layers[1].dsa.kv_k, qmm.Q4) and isinstance(e512.w.layers[1].moe.shared.gu, qmm.Q4)
    assert isinstance(e512.w.head, qmm.Q4) and e512.w.draft_head is None
    assert e512.e.prefill_rows == 512
    assert e512._effective(encode_policy("auto")) != encode_policy(EXL3_AUTO)


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
def test_q4_drafted_equals_serial(e512, sampling):
    prompt = list(np.random.default_rng(8).integers(0, 1000, size=45))
    serial, _ = _generate(e512, prompt, sampling, draft=False, tokens=32)
    for policy in (None, "auto", "auto:1:1:0", "fc5:0.3", "2", "c3:0.35", "a:0.6:0.85"):
        drafted, _ = _generate(e512, prompt, sampling, policy=policy, tokens=32)
        assert drafted == serial, policy


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
def test_long_prefill_chunks_match(e64, e512, sampling):
    """A 1,300-token prompt: 21 chunks of 64 rows or 3 of 512 (tiled by 128) give the same reply."""

    prompt = list(np.random.default_rng(11).integers(0, 1000, size=1300))
    a, _ = _generate(e64, prompt, sampling, draft=False, tokens=24)
    b, _ = _generate(e512, prompt, sampling, draft=False, tokens=24)
    c, _ = _generate(e512, prompt, sampling, tokens=24)
    assert a == b == c


def test_long_prefill_resume_matches_fresh(e512):
    sampling = Sampling(21, 1.0, 20, 0.95)
    rng = np.random.default_rng(22)
    first = list(rng.integers(0, 1000, size=700))
    reply, _ = _generate(e512, first, sampling, tokens=20)
    after = first + reply + list(rng.integers(0, 1000, size=600))
    warm, stats = _generate(e512, after, sampling)
    assert stats["cached"] >= len(first) + len(reply) - 1
    _generate(e512, list(rng.integers(0, 1000, size=9)), sampling)
    cold, stats = _generate(e512, after, sampling)
    assert stats["cached"] == 0 and warm == cold


def test_mse_clip_never_worse_than_minmax():
    torch.manual_seed(0)
    w = (torch.randn(256, 1024, device="cuda") * torch.rand(256, 1, device="cuda")).to(torch.bfloat16)
    w[:, ::97] *= 8                                  # outliers, where clipping pays

    def err(q):
        x = torch.eye(1024, device="cuda", dtype=torch.bfloat16)
        deq = torch.cat([qmm.matmul(x[i:i + 128], q) for i in range(0, 1024, 128)]).T.float()
        return ((deq - w.float()) ** 2).sum().item()

    plain, mse = err(qmm.quantize4(w)), err(qmm.quantize4(w, mse=True))
    assert mse <= plain * 1.0001


# -- patches/0004: sparse attention past 2,051 tokens with long prefill chunks ------------------------------------

def _select_reference(pools: torch.Tensor, pos: int, R: int):
    """Upstream's per-row loop (TensorFold 0.3.4 sparse.select_tokens), kept as the reference."""

    from tensorfold.families.glm5_next.cuda.sparse import POOL, TOPK_POOLS

    width = TOPK_POOLS * POOL + POOL - 1
    tokens = torch.full((R, width), -1, dtype=torch.int32, device=pools.device)
    counts = []
    for r in range(R):
        q = pos + r
        npool = (q + 1) // POOL
        if npool <= TOPK_POOLS:
            counts.append(0)
            continue
        body = (pools[r, :, None] * POOL + torch.arange(POOL, device=pools.device)).reshape(-1)
        tail = torch.arange(npool * POOL, q + 1, device=pools.device)
        row = torch.cat([body, tail])
        tokens[r, :row.numel()] = row.to(torch.int32)
        counts.append(row.numel())
    return tokens, torch.tensor(counts, dtype=torch.int32, device=pools.device)


@pytest.mark.parametrize("pos,R", [(2040, 64), (2045, 512), (5000, 300), (2051, 1)])
def test_vectorized_selection_equals_loop(monkeypatch, pos, R):
    from tensorfold.families.glm5_next.cuda import sparse

    np_max = (pos + R) // 4 + 8
    g = torch.Generator(device="cuda").manual_seed(pos + R)
    scores = torch.rand((R, np_max), device="cuda", generator=g)
    q = pos + torch.arange(R, device="cuda")
    scores = torch.where(torch.arange(np_max, device="cuda")[None, :] < ((q + 1) // 4)[:, None], scores,
                         torch.full_like(scores, float("-inf")))
    class _K:                      # stands in for the scoring kernel: hands back the fixed scores
        def __getitem__(self, grid):
            return lambda qi, wts, ws, pk, out, *rest, **kw: out.copy_(scores)

    monkeypatch.setattr(sparse, "_scores", _K())
    qi = torch.zeros((R, 32 * 128), dtype=torch.bfloat16, device="cuda")
    wts = torch.zeros((R, 32), dtype=torch.bfloat16, device="cuda")
    tokens, counts = sparse.select_tokens(qi, wts, None, pos, R, np_max, None)
    order = torch.sort(scores, dim=1, descending=True, stable=True).indices[:, :sparse.TOPK_POOLS]
    ref_t, ref_c = _select_reference(torch.sort(order, dim=1).values, pos, R)
    assert torch.equal(tokens, ref_t) and torch.equal(counts, ref_c)


def _index_heads_32(model: Path) -> None:
    """Upstream's synthetic checkpoint has 2 index heads, but the index scoring kernel is built for the real
    model's 32 (``sparse._scores``: H=32), so past 2,051 tokens it would read past the synthetic buffers. Widen
    the indexer's weights_proj and wq_b to 32 heads so the long-context path runs on the real shapes."""

    import json

    from safetensors.torch import load_file, save_file

    f = model / "model-00001-of-00001.safetensors"
    t = load_file(str(f))
    g = torch.Generator().manual_seed(7)
    for name in list(t):
        if name.endswith("indexer.weights_proj.weight"):
            t[name] = (torch.randn((32, t[name].shape[1]), generator=g) * 0.05).to(t[name].dtype)
        elif name.endswith("indexer.wq_b.weight"):
            t[name] = (torch.randn((32 * 128, t[name].shape[1]), generator=g) * 0.02).to(t[name].dtype)
    save_file(t, str(f), metadata={"format": "mlx"})
    cfg = json.loads((model / "config.json").read_text())
    cfg["text_config"]["index_n_heads"] = 32
    (model / "config.json").write_text(json.dumps(cfg))


@pytest.fixture(scope="module")
def long_ckpt(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm_long")
    _checkpoint(path / "model", exl3=True)
    _index_heads_32(path / "model")
    _drafter(path / "dflash2")
    return path


def _long_engine(path, monkeypatch, rows: int):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    monkeypatch.setattr(weights, "NONEXPERT", "q4mse")
    monkeypatch.setenv("GLM53_TF_PREFILL_ROWS", str(rows))
    return GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", context=4096,
                     comm=_TwoCopies())


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
def test_sparse_long_prefill_chunks_match(long_ckpt, sampling):
    """A 3,000-token prompt crosses the 2,051-token dense limit: 64-row and 512-row prefill chunks, and drafted
    decoding, give the same reply."""

    m = pytest.MonkeyPatch()
    try:
        prompt = list(np.random.default_rng(31).integers(0, 1000, size=3000))
        a, _ = _generate(_long_engine(long_ckpt, m, 64), prompt, sampling, draft=False, tokens=24)
        e = _long_engine(long_ckpt, m, 512)
        b, _ = _generate(e, prompt, sampling, draft=False, tokens=24)
        c, _ = _generate(e, prompt, sampling, tokens=24)
        assert a == b == c
    finally:
        m.undo()


# -- helpers for 0005 / 0006: the committed state after a prefill, bit for bit ----------------------------------

def _prefill_state(eng, prompt) -> list[torch.Tensor]:
    """Prefill ``prompt`` from scratch on ``eng``'s engine (MTP head absorbing, no DFlash2) and return the first
    token and every piece of committed state: KDA states and conv windows, the attention caches, the MTP head's
    cache, and the hidden row the first draft reads."""

    from tensorfold.families.glm5_next.cuda.decode import prefill

    e = eng.e
    first = prefill(e, list(prompt), None, mtp=True)
    st = e.st
    n = len(prompt)
    out = [torch.tensor([first]), st.rec[st.cur[0]].clone(), st.conv.clone(), e.last_hidden.clone()]
    out += [k[:n].clone() for k in st.kc] + [v[:n].clone() for v in st.vc]
    out += [st.mtp_kc[:st.mtp_len].clone(), st.mtp_vc[:st.mtp_len].clone()]
    eng.cache = []            # this prefill overwrote the caches the engine's kept snapshots point at
    return out


def _same(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    return len(a) == len(b) and all(x.shape == y.shape and torch.equal(x, y) for x, y in zip(a, b))


# -- patches/0005: GLM53_TF_PROFILE ------------------------------------------------------------------------------

def test_profile_probe_changes_no_bits_and_reports(e512, capfd):
    import json

    from tensorfold.families.glm5_next.cuda import profile

    prompt = list(np.random.default_rng(41).integers(0, 1000, size=700))
    old = profile.ON
    try:
        profile.ON = False
        ref = _prefill_state(e512, prompt)
        assert not profile.P.active and profile.P.marks == []
        profile.ON = True
        got = _prefill_state(e512, prompt)
    finally:
        profile.ON = old
    assert not profile.P.active
    assert _same(ref, got)
    err = capfd.readouterr().err
    lines = [l for l in err.splitlines() if l.startswith("GLM53_TF_PROFILE ")]
    assert lines, err[-2000:]
    rep = json.loads(lines[-1][len("GLM53_TF_PROFILE "):])
    assert rep["tokens"] == 700 and rep["cached"] == 0 and rep["rows"] == 700 and rep["chunks"] == 2
    ms = rep["ms"]
    for name in ("embed", "hc", "kda.proj", "kda.chain", "kda.o_proj", "dsa.proj", "dsa.attn", "dsa.o_proj",
                 "mlp.dense", "moe.router", "moe.routed", "moe.shared", "moe.combine", "allgather", "head", "stage",
                 "commit", "sample", "mtp.dsa.attn", "mtp.moe.routed", "mtp.head"):
        assert name in ms, (name, sorted(ms))
    assert all(v >= 0 for v in ms.values())
    assert abs(sum(ms.values()) - rep["gpu_ms"]) < 0.05 + 1e-3 * rep["gpu_ms"]
    assert "[GLM53_TF_PROFILE] prefill 700 tokens" in err


# -- patches/0006: looped EXL3 expert kernel for prefill windows -------------------------------------------------

def _exl3_layer(D: int, NI: int, E: int, seed: int):
    from tensorfold.families.glm5_next.cuda import exl3_mm

    rng = np.random.default_rng(seed)

    def trellis(k, n):
        return torch.from_numpy(rng.integers(-2**15, 2**15, size=(k // 16, n // 16, 64)).astype(np.int16))

    def scales(n, sc):
        return torch.from_numpy((rng.standard_normal(n) * sc).astype(np.float16))

    ex = [(trellis(D, NI), trellis(D, NI), trellis(NI, D), scales(D, 0.02), scales(D, 0.02), scales(NI, 0.5),
           scales(NI, 0.5), scales(NI, 0.05), scales(D, 0.2)) for _ in range(E)]
    cols = list(zip(*ex))
    wds = lambda ts: torch.stack([exl3_mm.words(t) for t in ts]).cuda()        # noqa: E731
    hs = lambda ts: torch.stack(list(ts)).cuda()                               # noqa: E731
    return exl3_mm.Exl3Experts(wds(cols[0]), wds(cols[1]), wds(cols[2]), hs(cols[3]), hs(cols[4]), hs(cols[5]),
                               hs(cols[6]), hs(cols[7]), hs(cols[8]), E, NI, D)


def _exl3_group(picks: torch.Tensor):
    """The engine's grouping (``glue.select``) of a window's picks [n, slots] (the last slot the shared expert,
    id E): distinct experts ascending, each one's (row * 32 + slot) members in row order, one spare entry."""

    from types import SimpleNamespace

    n, slots = picks.shape
    used = sorted({int(e) for e in picks.flatten()})
    ids = torch.zeros((len(used) + 1,), dtype=torch.int32)
    members = torch.full((len(used) + 1, n), -1, dtype=torch.int32)
    for u, e in enumerate(used):
        ids[u] = e
        j = 0
        for r in range(n):
            for s in range(slots):
                if int(picks[r, s]) == e:
                    members[u, j] = r * 32 + s
                    j += 1
    return SimpleNamespace(ids=ids.cuda(), count=torch.tensor([len(used)], dtype=torch.int32).cuda(),
                           members=members.cuda())


def test_expert_loop_kernel_bits(monkeypatch):
    """150 rows over 4 routed experts (up to ~80 members each, 5 member tiles; expert 3 exactly 32): the looped
    kernel in every tile setting gives upstream's per-member-tile bits, and a row gives the same bits alone (the
    upstream kernel, one member tile) or inside any window."""

    from tensorfold.families.glm5_next.cuda import exl3_mm

    D, NI, E, SLOTS, R, LIMIT = 512, 128, 4, 3, 150, 10.0
    ex = _exl3_layer(D, NI, E, seed=17)
    rng = np.random.default_rng(18)
    picks = torch.full((R, SLOTS), E, dtype=torch.int32)
    for r in range(R):
        pair = [3, int(rng.integers(0, 3))] if r < 32 else list(rng.choice(3, size=2, replace=False))
        picks[r, 0], picks[r, 1] = pair[0], pair[1]
    x = (torch.randn((R, D), generator=torch.Generator().manual_seed(19)) * 0.5).to(torch.bfloat16).cuda()

    def run(rows, loop, cfg=(4, 2)):
        monkeypatch.setattr(exl3_mm, "LOOP", loop)
        monkeypatch.setattr(exl3_mm, "LOOP_CFG", cfg)
        n = len(rows)
        p = picks[rows]
        scratch = exl3_mm.Scratch(n, SLOTS, D, NI, "cuda")
        y = torch.zeros((n * SLOTS, D), dtype=torch.float32, device="cuda")
        exl3_mm.routed(x[rows].contiguous(), p.cuda(), _exl3_group(p), ex, scratch, y, n, LIMIT)
        torch.cuda.synchronize()
        return y.view(n, SLOTS, D)[:, :SLOTS - 1].cpu()

    every = list(range(R))
    ref = run(every, False)
    assert torch.isfinite(ref).all() and ref.abs().max() > 0
    for cfg in ((4, 2), (8, 1), (4, 1), (2, 4)):
        assert torch.equal(run(every, True, cfg), ref), cfg
    for r in (0, 31, 32, 77, 149):                   # alone: members width 1, upstream's kernel
        assert torch.equal(run([r], True)[0], ref[r]), r
    sub = list(range(100, 140))                     # another window (40 rows, looped) gives the same rows
    assert torch.equal(run(sub, True), ref[100:140])
    rev = every[::-1]                               # other tile-mates and tile positions for every row
    assert torch.equal(run(rev, True), ref[rev])


def test_expert_loop_prefill_state_bits(e64, e512, monkeypatch):
    """A 1,300-token prompt (8 synthetic experts, top 2: 16 to 128 members an expert per chunk): the prefilled
    state with the looped kernel equals upstream's grid bit for bit, in 64-row and 512-row chunks, and the two
    chunkings equal each other."""

    from tensorfold.families.glm5_next.cuda import exl3_mm

    prompt = list(np.random.default_rng(51).integers(0, 1000, size=1300))
    states = {}
    for name, eng in (("64", e64), ("512", e512)):
        monkeypatch.setattr(exl3_mm, "LOOP", False)
        ref = _prefill_state(eng, prompt)
        monkeypatch.setattr(exl3_mm, "LOOP", True)
        got = _prefill_state(eng, prompt)
        assert _same(ref, got), name
        states[name] = got
    assert _same(states["64"], states["512"])


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
def test_expert_loop_replies_and_resume(e512, sampling):
    """With the looped kernel on (the default): drafted replies equal serial ones after a long prefill, and a
    prompt resumed from the kept reply equals a fresh prefill."""

    from tensorfold.families.glm5_next.cuda import exl3_mm

    assert exl3_mm.LOOP
    rng = np.random.default_rng(61)
    first = list(rng.integers(0, 1000, size=900))
    serial, _ = _generate(e512, first, sampling, draft=False, tokens=24)
    drafted, _ = _generate(e512, first, sampling, tokens=24)
    assert drafted == serial
    after = first + drafted + list(rng.integers(0, 1000, size=700))
    warm, stats = _generate(e512, after, sampling, tokens=16)
    assert stats["cached"] >= len(first) + len(drafted) - 1
    cold, stats = _generate(e512, after, sampling, draft=False, tokens=16)
    assert stats["cached"] == 0 and warm == cold
