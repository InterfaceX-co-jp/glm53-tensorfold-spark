"""patches/0460, the parts that need no GPU (docs/PREFETCH-COMM.md):

- ``GLM53_TF_L2PF*`` parsing; the per-site plans on fake weights (CPU tensors): every span lies inside a weight, in
  the order the next kernels read them, 16-byte aligned, a multiple of 16 bytes, at most the depth; the ``o`` plan
  (latent: ``kv_v`` then ``o``); the MTP head's keys; the expert plans (one EXL3 expert apart, first bytes);
- **the port**: with every kernel stubbed out and a recording prefetcher installed, ``batch.compute_multi`` reaches
  exactly the prefetch sites of the lone ``forward.compute`` (a, o, e, f per layer, in the same order, with the
  round's rows) and joins the side stream before it returns -- 0040 tagged no site there; the batched MTP head pass
  (``batch.mtp_multi``) reaches the head's sites; off, nothing is launched;
- 0040's ``overlap.Prefetcher.launch`` takes the row count the gather now passes;
- RoCE (0460): the one-HCA rule, the agreed settings, the knob parsing, ``trace_summary``.

The protocol-level checks of the RoCE changes are in tests/test_roce_protocol_model.py (extended), the PTX checks in
tests/test_prefetch_comm_compile.py, the GPU checks in tests/cuda/test_prefetch_comm_patches.py.

Run: PYTHONPATH=<patched tree>/src:<triton> pytest -q tests/test_prefetch_comm.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.glm5_next.cuda import l2pf, roce  # noqa: E402


# -- knobs --------------------------------------------------------------------------------------------------------------
def test_settings_parsing():
    assert not l2pf.settings({}).on
    for off in ("0", "off", "", "none"):
        assert not l2pf.settings({"GLM53_TF_L2PF": off}).on
    s = l2pf.settings({"GLM53_TF_L2PF": "1"})
    assert (s.mode, s.mb, s.sites, s.expert_kb, s.rows, s.chunk_kb) == ("bulk", 4.0, ("a", "f", "o"), 64, 64, 32)
    s = l2pf.settings({"GLM53_TF_L2PF": "lines", "GLM53_TF_L2PF_MB": "3.5", "GLM53_TF_L2PF_SITES": "e, a,e",
                       "GLM53_TF_L2PF_EXPERT_KB": "128", "GLM53_TF_L2PF_ROWS": "16", "GLM53_TF_L2PF_CHUNK_KB": "8"})
    assert (s.mode, s.mb, s.sites, s.expert_kb, s.rows, s.chunk_kb) == ("lines", 3.5, ("e", "a"), 128, 16, 8)
    assert l2pf.settings({"GLM53_TF_L2PF": "touch"}).mode == "touch"
    for bad in ({"GLM53_TF_L2PF": "yes"}, {"GLM53_TF_L2PF": "1", "GLM53_TF_L2PF_MB": "0"},
                {"GLM53_TF_L2PF": "1", "GLM53_TF_L2PF_MB": "20"}, {"GLM53_TF_L2PF": "1", "GLM53_TF_L2PF_SITES": "a,x"},
                {"GLM53_TF_L2PF": "1", "GLM53_TF_L2PF_ROWS": "0"}, {"GLM53_TF_L2PF": "1", "GLM53_TF_L2PF_MB": "x"}):
        with pytest.raises(ValueError, match="GLM53_TF_L2PF"):
            l2pf.settings(bad)


# -- fake weights -------------------------------------------------------------------------------------------------------
def _t(n: int, dtype=torch.int32) -> torch.Tensor:
    return torch.zeros(n, dtype=dtype)


def _q4(n: int, k: int) -> SimpleNamespace:
    """qmm.Q4's fields: words [N/64, K/64, 64, 8] int32 (flat here), scales / biases [K/64, N] bf16."""

    return SimpleNamespace(weight=_t(-(-n // 64) * 64 * k // 8), scales=_t(k // 64 * n, torch.bfloat16),
                           biases=_t(k // 64 * n, torch.bfloat16), n=n, k=k)


def _hc() -> SimpleNamespace:
    return SimpleNamespace(fn=_t(3000, torch.bfloat16), base=_t(24, torch.float32), scale=_t(3, torch.float32))


class _Exl3:                         # exl3_mm.Exl3Experts' fields the plans read
    def __init__(self, E: int = 12) -> None:
        self.gt = _t(E * 4 * 4 * 32 * 8).view(E, 4 * 8, 4, 32)      # 16 KiB an expert
        self.ut = _t(E * 4 * 4 * 32 * 8).view(E, 4 * 8, 4, 32)
        self.dt = _t(E * 4 * 4 * 32 * 8).view(E, 4, 4 * 8, 32)


def _layer(i: int, kind: str, mlp: str, *, plain: bool = False) -> SimpleNamespace:
    L = SimpleNamespace(index=i, kind=kind, attn_hc=None if plain else _hc(), ffn_hc=None if plain else _hc(),
                        in_norm=_t(64, torch.bfloat16), post_norm=_t(64, torch.bfloat16), kda=None, dsa=None,
                        mlp=None, moe=None)
    if kind == "kda":
        L.kda = SimpleNamespace(proj=_q4(1536, 512), fb=_q4(64, 128), gb=_q4(64, 128), o=_q4(512, 512), fa_off=0,
                                ga_off=0, b_off=0, conv=None, a_log=None, dt_bias=None, norm=None, heads=2)
    else:
        L.dsa = SimpleNamespace(proj=_q4(1024, 512), q_norm=None, kv_norm=None, q_b=_q4(64, 512), kv_k=_q4(512, 128),
                                kv_v=_q4(512, 256), o=_q4(512, 1024), heads=2, index=None)
    if mlp == "dense":
        L.mlp = SimpleNamespace(gu=_q4(1024, 512), down=_q4(512, 512), width=64)
    else:
        L.moe = SimpleNamespace(router=_t(9000, torch.bfloat16), bias=_t(12, torch.float32), experts=_Exl3(),
                                shared=SimpleNamespace(gu=_q4(256, 512), down=_q4(512, 128), width=64))
    return L


def _weights(latent_kv: bool = True) -> SimpleNamespace:
    layers = [_layer(0, "kda", "dense"), _layer(1, "dsa", "moe"), _layer(2, "kda", "moe")]
    mtp = SimpleNamespace(enorm=_t(64, torch.bfloat16), hnorm=_t(64, torch.bfloat16), eh=_q4(512, 1024),
                          norm=_t(64, torch.bfloat16), layer=_layer(3, "dsa", "moe", plain=True))
    return SimpleNamespace(layers=layers, mtp=mtp, norm=_t(64, torch.bfloat16), head=_q4(4096, 512), draft_head=None,
                           meta={"latent_kv": latent_kv}, world=2, comm=object(), device="cpu")


def _ranges(tensors) -> list[tuple[int, int]]:
    return [(t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in tensors]


def _inside(sp, tensors) -> bool:
    return any(a <= sp.addr and sp.addr + sp.nbytes <= b for a, b in _ranges(tensors))


def _all_tensors(w) -> list:
    out = []

    def add(x):
        if isinstance(x, torch.Tensor):
            out.append(x)
        elif isinstance(x, (SimpleNamespace, _Exl3)):
            for v in vars(x).values():
                add(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                add(v)

    add(w.layers)
    add(w.mtp)
    add([w.norm, w.head])
    return out


@pytest.mark.parametrize("budget", [16, 1000, 64 * 1024, 1 << 22])
@pytest.mark.parametrize("latent_kv", [True, False])
def test_plans_read_only_weights_within_the_budget(budget, latent_kv):
    w = _weights(latent_kv)
    got = l2pf.plans(w, budget, l2pf.SITES, latent_kv=latent_kv)
    weights = _all_tensors(w)
    keys = {(i, k) for i in (0, 1, 2) for k in "afo"} | {("mtp", k) for k in "afo"}
    assert set(got) == keys
    for k, spans in got.items():
        assert spans and sum(s.nbytes for s in spans) <= max(budget, 16)
        for s in spans:
            assert s.addr % 16 == 0 and s.nbytes % 16 == 0 and s.nbytes > 0
            assert _inside(s, weights) and _inside(s, [s.tensor]), k


def test_plan_orders():
    w = _weights(True)
    big = 1 << 30
    got = l2pf.plans(w, big, l2pf.SITES, latent_kv=True)
    L0, L1, L2 = w.layers

    def tensors(k):
        out = []
        for s in got[k]:
            if not out or out[-1] is not s.tensor:
                out.append(s.tensor)
        return out

    def q(m):
        return [m.scales, m.biases, m.weight]

    same = lambda a, b: len(a) == len(b) and all(x is y for x, y in zip(a, b))       # noqa: E731
    # a: the FFN's first reads (dense MLP / router + shared gate/up), f: the next layer's attention input
    assert same(tensors((0, "a")), [L0.ffn_hc.fn, L0.post_norm] + q(L0.mlp.gu))
    assert same(tensors((1, "a")), [L1.ffn_hc.fn, L1.post_norm, L1.moe.router, L1.moe.bias] + q(L1.moe.shared.gu))
    assert same(tensors((0, "f")), [L1.attn_hc.fn, L1.in_norm] + q(L1.dsa.proj))
    assert same(tensors((2, "f")), [w.norm] + q(w.head))
    # o: the attention output side (latent DSA: the value expansion first)
    assert same(tensors((0, "o")), q(L0.kda.o))
    assert same(tensors((1, "o")), q(L1.dsa.kv_v) + q(L1.dsa.o))
    assert same(tensors(("mtp", "o")), q(w.mtp.layer.dsa.kv_v) + q(w.mtp.layer.dsa.o))
    assert same(tensors(("mtp", "a")), [w.mtp.layer.post_norm, w.mtp.layer.moe.router, w.mtp.layer.moe.bias] +
                q(w.mtp.layer.moe.shared.gu))
    lat = l2pf.plans(w, big, ("o",), latent_kv=False)
    assert same([s.tensor for s in lat[(1, "o")]][:2], q(L1.dsa.o)[:2]) and set(lat) == {(0, "o"), (1, "o"),
                                                                                          (2, "o"), ("mtp", "o")}


def test_expert_plans():
    w = _weights()
    got = l2pf.expert_plans(w, 4096)
    moes = [L for L in w.layers + [w.mtp.layer] if L.moe is not None]
    assert set(got) == {id(L.moe.experts) for L in moes}
    for L in moes:
        ex = L.moe.experts
        bases, stride, n, count = got[id(ex)]
        assert bases == [ex.gt.data_ptr(), ex.ut.data_ptr()] and stride == 16 * 1024 and n == 4096 and count == 12
    assert all(v[2] == 16 * 1024 for v in l2pf.expert_plans(w, 1 << 30).values())     # capped at one expert


def test_keys():
    w = _weights()
    assert l2pf.key(w, w.layers[1], "o") == (1, "o")
    assert l2pf.key(w, w.mtp.layer, "o") == ("mtp", "o")
    w.mtp.layer.index = 1                                   # an MTP index equal to a main layer's never collides
    assert l2pf.key(w, w.mtp.layer, "e") == ("mtp", "e")


# -- the port: every forward path reaches the same sites -----------------------------------------------------------------
class Stub:
    """Anything: every attribute, call and index is a Stub (kernels stubbed out)."""

    shape = (64, 64)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return Stub()

    def __call__(self, *a, **k):
        return Stub()

    def __getitem__(self, k):
        return Stub()

    def __setitem__(self, k, v):
        pass

    def __bool__(self):
        return True

    def __pow__(self, o):
        return 1.0

    def __rsub__(self, o):
        return 0

    def __sub__(self, o):
        return 0


class Rec:
    L2PF = True

    def __init__(self) -> None:
        self.log: list = []

    def launch(self, site, R=None):
        self.log.append(("launch", site, R))

    def experts(self, layer, grp, R):
        self.log.append(("experts", layer.index if layer.index != 3 else "mtp", R))

    def join(self):
        self.log.append(("join",))


def _stub_engine(monkeypatch, latent_kv: bool):
    from tensorfold.families.glm5_next.cuda import batch, exl3_mm, forward, latent

    for mod in (forward, batch):
        for name in ("glue", "qmm", "kda_mod", "dv2", "sparse"):
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, Stub())
        for name in ("attention", "kv_write", "fast_gather"):
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, Stub())
    monkeypatch.setattr(forward, "fastpf", Stub())
    monkeypatch.setattr(batch, "check_room", lambda *a, **k: None)
    monkeypatch.setattr(batch.draftvocab, "head_for", lambda *a, **k: Stub())
    monkeypatch.setattr(exl3_mm, "routed", lambda *a, **k: None)
    monkeypatch.setattr(latent, "on", lambda w: latent_kv)
    for name in ("_project", "_index_proj", "_attend", "qmm", "expand", "expand_tc"):
        monkeypatch.setattr(latent, name, Stub())
    monkeypatch.setattr(latent, "tc", lambda b: False)
    cfg = SimpleNamespace(hidden=64, streams=4, eps=1e-6, hc_eps=1e-6, hc_iters=2, limit=7.0, top_k=8, experts=12,
                          routed_scale=1.0, norm_topk=True, q_lora=32, qk_dim=16, v_dim=16, dense_limit=2050,
                          lower=None, index_dim=16)
    w = _weights(latent_kv)
    w.cfg = cfg
    w.embed = Stub()
    rec = Rec()
    w.meta["prefetch"] = rec
    b = Stub()
    b.site = None
    b.defer = None
    b.fast = False
    b.route_src = None
    b.tap_at = {}
    b.world = 2
    grp = SimpleNamespace(ids=Stub(), count=Stub(), members=Stub())
    b.group = lambda R: grp
    return w, b, rec


def _state():
    st = Stub()
    st.index = None
    st.kda_index = {0: 0, 2: 1}
    st.dsa_index = {1: 0}
    st.mtp_len = 5
    st.pos = 7
    st.cur = [0, 0]
    return st


def _sites(log):
    return [e for e in log if e[0] != "join"]


@pytest.mark.parametrize("latent_kv", [True, False])
def test_compute_multi_reaches_the_lone_forwards_sites(monkeypatch, latent_kv):
    from tensorfold.families.glm5_next.cuda import batch, forward

    w, b, rec = _stub_engine(monkeypatch, latent_kv)
    R = 5
    forward.compute(w, _state(), b, R)
    lone = list(rec.log)
    rec.log.clear()
    batch.compute_multi(w, [_state(), _state()], b, [2, 3], npbs=[0, 0])
    multi = list(rec.log)
    # a, o, f per layer; e per MoE layer; in forward order
    want = []
    for L in w.layers:
        want.append(("launch", (L.index, "o"), R))
        want.append(("launch", (L.index, "a"), R))
        if L.moe is not None:
            want.append(("experts", L.index, R))
        want.append(("launch", (L.index, "f"), R))
    assert _sites(lone) == want
    assert _sites(multi) == want                         # 0040 tagged none of these in compute_multi
    assert lone[-1] == ("join",) and multi[-1] == ("join",)
    rec.log.clear()
    batch.compute_multi(w, [_state()], b, [4], logits=False)
    assert rec.log[-1] == ("join",) and len(_sites(rec.log)) == len(want)


def test_mtp_multi_reaches_the_heads_sites(monkeypatch):
    from tensorfold.families.glm5_next.cuda import batch

    w, b, rec = _stub_engine(monkeypatch, True)
    b.rows = 64
    batch.mtp_multi(w, [_state(), _state()], b, [[1, 2], [3]], [Stub(), Stub()])
    assert _sites(rec.log) == [("launch", ("mtp", "o"), 3), ("launch", ("mtp", "a"), 3), ("experts", "mtp", 3),
                               ("launch", ("mtp", "f"), 3)]
    assert rec.log[-1] == ("join",)


def test_hooks_are_noops_without_a_0460_prefetcher(monkeypatch):
    from tensorfold.families.glm5_next.cuda import batch

    w, b, rec = _stub_engine(monkeypatch, True)
    w.meta["prefetch"] = None
    batch.compute_multi(w, [_state()], b, [3], npbs=[0])
    assert rec.log == []
    w.meta["prefetch"] = SimpleNamespace(launch=lambda *a: rec.log.append(a), join=lambda: None)   # 0040's (no L2PF)
    batch.compute_multi(w, [_state()], b, [3], npbs=[0])
    assert [site for site, _ in rec.log] == [(L.index, k) for L in w.layers for k in "af"]          # a / f only


def test_install_refuses_both_prefetchers(monkeypatch):
    monkeypatch.setenv("GLM53_TF_L2PF", "1")
    w = _weights()
    w.meta["prefetch"] = object()                         # 0040's GLM53_TF_COMM=prefetch installed first
    with pytest.raises(ValueError, match="exclude each other"):
        l2pf.install(w)
    monkeypatch.setenv("GLM53_TF_L2PF", "0")
    assert l2pf.install(w) is None


def test_overlap_prefetcher_takes_rows():
    import inspect

    from tensorfold.families.glm5_next.cuda import overlap

    assert list(inspect.signature(overlap.Prefetcher.launch).parameters) == ["self", "site", "R"]


# -- RoCE ---------------------------------------------------------------------------------------------------------------
def test_one_hca_rule():
    assert not roce.one_hca(16 * 1024, 0, 2)               # 0: always striped (0350)
    assert roce.one_hca(16 * 1024, 32 * 1024, 2)
    assert not roce.one_hca(32 * 1024, 32 * 1024, 2)       # at the threshold: striped
    assert not roce.one_hca(16, 32 * 1024, 1)              # one HCA anyway
    assert roce.stripe_min(roce.Settings(stripe_kb=48)) == 48 * 1024


def test_roce_knobs(monkeypatch):
    for k in ("GLM53_TF_ROCE_STRIPE_KB", "GLM53_TF_ROCE_INLINE", "GLM53_TF_ROCE_LAZY_CQ", "GLM53_TF_ROCE_LEAN",
              "GLM53_TF_ROCE_TRACE"):
        monkeypatch.delenv(k, raising=False)
    s = roce.settings()
    assert (s.stripe_kb, s.inline_bytes, s.lazy_cq, s.lean, s.trace) == (0, 0, False, False, 0)
    base = s.agreed()
    monkeypatch.setenv("GLM53_TF_ROCE_STRIPE_KB", "64")
    monkeypatch.setenv("GLM53_TF_ROCE_INLINE", "512")
    monkeypatch.setenv("GLM53_TF_ROCE_LAZY_CQ", "1")
    monkeypatch.setenv("GLM53_TF_ROCE_LEAN", "1")
    monkeypatch.setenv("GLM53_TF_ROCE_TRACE", "4096")
    s = roce.settings()
    assert (s.stripe_kb, s.inline_bytes, s.lazy_cq, s.lean, s.trace) == (64, 512, True, True, 4096)
    # only the stripe threshold changes what crosses the link: the ranks must agree on it, not on the local knobs
    assert s.agreed()[:-1] == base[:-1] and s.agreed()[-1] == 64 and base[-1] == 0
    monkeypatch.setenv("GLM53_TF_ROCE_TRACE", "1000")
    with pytest.raises(ValueError, match="power of two"):
        roce.settings()


def test_trace_summary():
    rows, peer = [], []
    for q in range(1, 101):
        t = q * 100_000
        rows.append({"seq": q, "start": t, "bell": t + 2_000, "flag": t + 2_000 + 9_000 + (q % 2) * 4_000,
                     "end": t + 14_000 + (q % 2) * 4_000, "seen": t + 2_700, "posted": t + 3_100})
        peer.append({"seq": q, "start": t, "bell": t + 3_000, "flag": t + 3_000 + 9_000 + (1 - q % 2) * 4_000,
                     "end": 0})
    rows.append({"seq": 101, "start": 0, "bell": 0, "flag": 0, "end": 0})       # not traced: skipped
    got = roce.trace_summary(rows, peer)
    assert got["stage"] == {"n": 100, "p50_us": 2.0, "p90_us": 2.0}
    assert got["wait"]["p50_us"] in (9.0, 13.0) and got["wait"]["p90_us"] == 13.0
    assert got["copy"]["p50_us"] == 3.0 and got["notice"]["p50_us"] == 0.7 and got["post"]["p50_us"] == 0.4
    assert got["transport"] == {"n": 100, "p50_us": 9.0, "p90_us": 9.0}
    assert got["skew"] == {"n": 100, "p50_us": 4.0, "p90_us": 4.0}
    assert "transport" not in roce.trace_summary(rows)


def test_q4_heads_interleave_small_matrices():
    """A 4-bit matrix launched as one wave (<= ONE_WAVE programs): scales, biases, then the head of every (64-column
    tile, K slice) chunk, evenly (every program starts on hits); a multi-wave one: its first bytes."""

    from tensorfold.families.glm5_next.cuda import qmm

    small = _q4(2048, 4096)                                 # the shared expert's gate/up, DSA's input projection
    sk = qmm.split_k(2048, 4096)
    assert 32 * sk <= l2pf.ONE_WAVE
    spans = l2pf.take([small], 1 << 20)
    assert spans[0].tensor is small.scales and spans[1].tensor is small.biases
    heads = spans[2:]
    chunks = 32 * sk
    size = small.weight.numel() * 4 // chunks
    assert len(heads) == chunks and len({h.nbytes for h in heads}) == 1 and heads[0].nbytes % 16 == 0
    assert [h.addr - small.weight.data_ptr() for h in heads] == [c * size for c in range(chunks)]
    assert sum(sp.nbytes for sp in spans) <= 1 << 20
    big = _q4(12576, 4096)                                  # the KDA input projection (~29 MB, 197 x 4 programs)
    spans = l2pf.take([big], 4 << 20)                        # scales and biases are 1.6 MB each here
    assert [sp.tensor is t for sp, t in zip(spans, (big.scales, big.biases, big.weight))] == [True] * 3
    assert len(spans) == 3 and spans[2].addr == big.weight.data_ptr()
    tiny = l2pf.take([small], 4096 + 2 * small.scales.numel() * 2)    # heads under 512 B: first bytes instead
    assert tiny[-1].addr == small.weight.data_ptr() and len(tiny) == 3
