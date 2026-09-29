"""patches/0500: the vision tower, the image-row table and the rank exchange (``vision.py``), on CPU.

- ``Tower.encode`` == transformers' ``Glm5NextVisionModel`` (a tiny random config with GLM-5.3-Flash's structure:
  patch 14, temporal 2, merge 2, axial 2D RoPE, q/k norms, clamped SwiGLU with a limit small enough to clip,
  the Conv2d merge and the merger) in float32 to 1e-5 relative, eager and SDPA attention, grids from 2 x 2 to
  12 x 10 and non-square; in bf16 to bf16 rounding; row chunking of the MLP changes nothing.
- ``Tower.load`` from a sharded safetensors checkpoint (index + shards; only ``model.visual.*`` read), refusing a
  checkpoint without a tower; ``Encoder.table``: ids sorted with their rows, an image twice encoded once, the cache.
- ``Table.fixup``: only rows whose id is in the table change, in every stream copy; ``clamp``; ``active`` nests.
- ``exchange`` between two ranks in threads over a fake all-gather: rank 1 receives rank 0's rows bit for bit (in
  chunks), a prompt without image rows exchanges nothing, mismatched ids fail loudly on both sides.
- The FLOP count used in docs/VISION.md.

Run: PYTHONPATH=<patched TensorFold>/src pytest -q tests/test_vision.py
"""

from __future__ import annotations

import json
import threading

import pytest

torch = pytest.importorskip("torch")
vision = pytest.importorskip("tensorfold.families.glm5_next.cuda.vision")
vp = pytest.importorskip("tensorfold.families.glm5_next.cuda.vision_prep")

CFG = dict(depth=2, hidden_size=64, num_heads=4, intermediate_size=128, out_hidden_size=96,
           projection_intermediate_size=160, patch_size=14, temporal_patch_size=2, spatial_merge_size=2,
           swiglu_limit=1.0, rms_norm_eps=1e-5, hidden_act="silu", attention_bias=True, in_channels=3)
GRIDS = [(2, 2), (4, 6), (8, 2), (6, 10), (12, 10)]


def hf_model(impl="sdpa", seed=0):
    mod = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    conf = pytest.importorskip("transformers.models.glm5_next.configuration_glm5_next")
    torch.manual_seed(seed)
    cfg = conf.Glm5NextVisionConfig(**CFG)
    cfg._attn_implementation = impl
    m = mod.Glm5NextVisionModel(cfg).float().eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.5)
    return m, cfg


def state(m):
    return {"model.visual." + k: v.detach().clone() for k, v in m.state_dict().items()}


@pytest.mark.parametrize("impl", ["eager", "sdpa"])
def test_tower_equals_transformers_fp32(impl):
    m, cfg = hf_model(impl)
    tw = vision.Tower(cfg.to_dict(), state(m), device="cpu", dtype=torch.float32)
    for gh, gw in GRIDS:
        px = torch.randn(gh * gw, 1176)
        with torch.no_grad():
            ref = m(px, grid_thw=torch.tensor([[1, gh, gw]])).pooler_output
        ours = tw.encode(px, (1, gh, gw))
        assert ours.shape == (gh * gw // 4, CFG["out_hidden_size"])
        assert (ours - ref).abs().max().item() <= 1e-5 * max(1.0, ref.abs().max().item())


def test_clamps_are_exercised():
    """The SwiGLU clamps must bite in the test config, or the parity above would not cover them."""

    m, cfg = hf_model()
    tw = vision.Tower(cfg.to_dict(), state(m), device="cpu", dtype=torch.float32)
    x = torch.randn(24, 64)
    blk = tw.blocks[0]
    gate = torch.nn.functional.linear(tw._rms(x, blk["norm2.weight"]), blk["mlp.gate_proj.weight"],
                                      blk["mlp.gate_proj.bias"])
    assert (gate > CFG["swiglu_limit"]).any() and (gate < -CFG["swiglu_limit"]).any()


def test_tower_bf16_close_to_transformers_bf16():
    m, cfg = hf_model()
    tw = vision.Tower(cfg.to_dict(), state(m), device="cpu", dtype=torch.bfloat16)
    mb = m.to(torch.bfloat16)
    for gh, gw in GRIDS[:3]:
        px = torch.randn(gh * gw, 1176)
        with torch.no_grad():
            ref = mb(px.to(torch.bfloat16), grid_thw=torch.tensor([[1, gh, gw]])).pooler_output.float()
        ours = tw.encode(px, (1, gh, gw)).float()
        rel = ((ours - ref).norm() / ref.norm()).item()
        assert ours.dtype == torch.float32 and rel < 3e-2, rel


def test_mlp_chunking_is_exact():
    m, cfg = hf_model()
    tw = vision.Tower(cfg.to_dict(), state(m), device="cpu", dtype=torch.float32)
    h = torch.randn(50, 64)
    blk = tw.blocks[1]
    whole = tw._mlp_rows(h, blk, chunk=1 << 20)
    assert torch.equal(torch.cat([tw._mlp_rows(h[a:a + 7], blk, chunk=1 << 20) for a in range(0, 50, 7)]),
                       tw._mlp_rows(h, blk, chunk=7))
    assert (whole - tw._mlp_rows(h, blk, chunk=7)).abs().max() < 1e-5


def test_bad_grid_refused():
    m, cfg = hf_model()
    tw = vision.Tower(cfg.to_dict(), state(m), device="cpu", dtype=torch.float32)
    with pytest.raises(ValueError):
        tw.encode(torch.randn(12, 1176), (1, 3, 4))
    with pytest.raises(ValueError):
        tw.encode(torch.randn(10, 1176), (1, 4, 4))


def write_checkpoint(d, m, shards=2):
    from safetensors.torch import save_file

    sd = state(m)
    names = sorted(sd)
    wmap = {}
    for i in range(shards):
        part = {k: sd[k].contiguous() for k in names[i::shards]}
        f = f"model-{i + 1:05d}.safetensors"
        save_file(part, str(d / f))
        wmap.update({k: f for k in part})
    other = {"model.language_model.norm.weight": torch.ones(8)}
    save_file(other, str(d / "model-00009.safetensors"))
    wmap["model.language_model.norm.weight"] = "model-00009.safetensors"
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wmap}))
    (d / "config.json").write_text(json.dumps({"vision_config": dict(CFG, model_type="glm5_next_vision"),
                                               "image_token_id": 154854}))


def test_load_and_encoder_table(tmp_path, monkeypatch):
    pytest.importorskip("safetensors")
    m, cfg = hf_model()
    write_checkpoint(tmp_path, m)
    tw = vision.Tower.load(tmp_path, device="cpu", dtype=torch.float32)
    px = torch.randn(24, 1176)
    with torch.no_grad():
        ref = m(px, grid_thw=torch.tensor([[1, 4, 6]])).pooler_output
    assert (tw.encode(px, (1, 4, 6)) - ref).abs().max() < 1e-4
    assert tw.nbytes > 0

    enc = vision.Encoder.__new__(vision.Encoder)
    enc.tower, enc.cfg = tw, CFG
    enc.cache, enc.cache_bytes, enc.cache_cap, enc.lock, enc.last = {}, 0, 1 << 30, threading.Lock(), {}
    enc.stream, enc.run_lock = None, threading.Lock()
    import collections

    enc.cache = collections.OrderedDict()
    a = vp.Prepared(px, (1, 4, 6), 6, b"A", (1, 1), vp.derive(b"A", 6))
    pb = torch.randn(16, 1176)
    b = vp.Prepared(pb, (1, 4, 4), 4, b"B", (1, 1), vp.derive(b"B", 4))
    t = enc.table(vp.Request([a, b, a]))
    assert enc.last["images"] == 3 and enc.last["encoded"] == 2 and t.n == 10
    keys = t.keys.tolist()
    assert keys == sorted(keys) and set(keys) == set(a.vids) | set(b.vids)
    rows_a, rows_b = tw.encode(px, (1, 4, 6)), tw.encode(pb, (1, 4, 4))
    for k, vid in enumerate(a.vids):
        assert torch.equal(t.rows[keys.index(vid)], rows_a[k])
    for k, vid in enumerate(b.vids):
        assert torch.equal(t.rows[keys.index(vid)], rows_b[k])
    enc.table(vp.Request([b]))
    assert enc.last["encoded"] == 0                                  # from the cache
    enc.cache_cap = 0
    enc.cache.clear()
    enc.table(vp.Request([b]))
    assert enc.last["encoded"] == 1 and not enc.cache


def test_load_refuses_a_text_only_checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {}}))
    with pytest.raises(ValueError, match="no vision"):
        vision.Tower.load(tmp_path, device="cpu")


def test_table_fixup_and_clamp():
    V = vp.VBASE
    keys = torch.tensor([V + 9, V + 2, V + 5], dtype=torch.int32)
    order = torch.argsort(keys)
    rows = torch.randn(3, 8).to(torch.bfloat16)
    t = vision.Table(keys[order], rows[order])
    ids = torch.tensor([1, V + 5, 7, V + 9, V + 2, V + 3], dtype=torch.int32)
    assert t.clamp(ids).tolist() == [1, 0, 7, 0, 0, 0]
    out = torch.full((6, 3 * 8), 7.0, dtype=torch.bfloat16)
    t.fixup(ids, out, 3, 8)
    o = out.view(6, 3, 8)
    for r, k in [(1, 2), (3, 0), (4, 1)]:
        assert all(torch.equal(o[r, c], rows[k]) for c in range(3))
    for r in (0, 2, 5):                               # vocabulary ids and an id missing from the table: untouched
        assert (o[r] == 7.0).all()


def test_active_nests_and_resets():
    t1 = vision.Table(torch.tensor([vp.VBASE], dtype=torch.int32), torch.zeros(1, 4))
    t2 = vision.Table(torch.tensor([vp.VBASE + 1], dtype=torch.int32), torch.zeros(1, 4))
    assert vision.ACTIVE is None
    with vision.active(t1):
        assert vision.ACTIVE is t1
        with vision.active(None):
            assert vision.ACTIVE is t1
        with vision.active(t2):
            assert vision.ACTIVE is t2
        assert vision.ACTIVE is t1
    assert vision.ACTIVE is None
    with pytest.raises(RuntimeError):
        with vision.active(t1):
            raise RuntimeError("x")
    assert vision.ACTIVE is None
    ids = torch.tensor([3], dtype=torch.int32)
    assert vision.embed_ids(ids) == (None, ids)


# -- two ranks ---------------------------------------------------------------------------------------------------
class Pair:
    """An all-gather between two threads (rank order), as the engine's comm does it."""

    def __init__(self):
        self.bar = threading.Barrier(2)
        self.slots = [None, None]
        self.calls = 0

    def comm(self, rank):
        pair = self

        class C:
            def all_gather(self, send, recv):
                pair.slots[rank] = send.clone()
                pair.bar.wait()
                recv.copy_(torch.cat(pair.slots))
                pair.bar.wait()
                if rank == 0:
                    pair.calls += 1
        return C()


class Rank:
    def __init__(self, rank, comm):
        self.rank, self.comm, self.torch = rank, comm, torch

    def _share(self, values):                         # GlmEngine._share on CPU
        n = torch.tensor([len(values) if self.rank == 0 else 0], dtype=torch.int32)
        got = torch.empty((2,), dtype=torch.int32)
        self.comm.all_gather(n, got)
        count = int(got[0].item())
        buf = torch.tensor(values, dtype=torch.int32) if self.rank == 0 else torch.zeros((count,), dtype=torch.int32)
        allv = torch.empty((2 * count,), dtype=torch.int32)
        self.comm.all_gather(buf, allv)
        return [int(v) for v in allv[:count].tolist()]


def run_pair(prompt0, prompt1, table):
    pair = Pair()
    out, errs = [None, None], []

    def go(rank, prompt):
        try:
            out[rank] = vision.exchange(Rank(rank, pair.comm(rank)), prompt, table if rank == 0 else None)
        except BaseException as exc:                  # noqa: BLE001
            errs.append((rank, exc))
            pair.bar.abort()

    th = [threading.Thread(target=go, args=(r, p)) for r, p in ((0, prompt0), (1, prompt1))]
    for t in th:
        t.start()
    for t in th:
        t.join(20)
    return out, errs, pair


def test_exchange_ships_rows_bit_for_bit(monkeypatch):
    monkeypatch.setattr(vision, "DEVICE", "cpu")
    monkeypatch.setattr(vision, "CHUNK_ROWS", 3)
    vids = vp.derive(b"img", 7)
    keys = sorted(vids)
    rows = torch.randn(7, 64).to(torch.bfloat16)
    rows[0, 0] = float("nan")                          # any bit pattern travels
    table = vision.Table(torch.tensor(keys, dtype=torch.int32), rows)
    prompt = [5, 6, *vids, 7]
    (t0, t1), errs, pair = run_pair(prompt, prompt, table)
    assert not errs and t0 is table
    assert t1.keys.tolist() == keys
    assert torch.equal(t1.rows.view(torch.int16), rows.view(torch.int16))
    assert pair.calls == 4 + 3                         # two int shares (2 gathers each), then 3 chunks of rows


def test_no_image_rows_exchange_nothing(monkeypatch):
    monkeypatch.setattr(vision, "DEVICE", "cpu")
    (t0, t1), errs, pair = run_pair([1, 2, 3], [1, 2, 3], None)
    assert not errs and t0 is None and t1 is None and pair.calls == 0


def test_mismatch_fails(monkeypatch):
    monkeypatch.setattr(vision, "DEVICE", "cpu")
    vids = vp.derive(b"img", 4)
    table = vision.Table(torch.tensor(sorted(vids), dtype=torch.int32), torch.zeros(4, 8, dtype=torch.bfloat16))
    _, errs, _ = run_pair([1, *vids], [1, *vids[:3], vids[3] + 1], table)
    assert any(r == 1 and "does not match" in str(e) for r, e in errs)
    _, errs, _ = run_pair([1, *vids[:3]], [1, *vids[:3]], table)
    assert any(r == 0 and "does not match" in str(e) for r, e in errs)


def test_flops_of_a_screenshot():
    real = {"depth": 24, "hidden_size": 1024, "intermediate_size": 4096, "out_hidden_size": 4096,
            "projection_intermediate_size": 10240, "spatial_merge_size": 2, "patch_size": 14, "temporal_patch_size": 2}
    f = vision.Tower.flops((1, 56, 74), real)          # 1024 x 768 -> 56 x 74 patches, 1,036 rows
    assert 5.0e12 < f < 5.8e12
    f_max = vision.Tower.flops((1, 154, 206), real)    # the 8,000-token budget: 7,931 rows
    assert 1.2e14 < f_max < 1.4e14
