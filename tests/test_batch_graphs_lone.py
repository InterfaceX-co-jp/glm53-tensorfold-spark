"""patches/0510, the parts that need no GPU (THEORY-2 ideas 1 and 2):

- ``GLM53_TF_BATCH_GRAPHS=1|0|lone`` parsing (``batchplan.graph_policy``) and the per-round rule (``graphs_for``);
- ``batch.Batcher._forward`` on a fake batcher with ``compute_multi`` / ``stage_multi`` stubbed and CUDA graphs faked:
  under ``lone`` a round of 2+ slots takes the eager path exactly as ``GLM53_TF_BATCH_GRAPHS=0`` does (same call,
  same arguments, nothing captured, no replay), a round of one slot (other than slot 0) captures / replays as ``1``
  does, and ``1`` is unchanged; slot 0 alone still runs the engine's own graphs;
- 0450's resident rounds and batched MTP head passes follow the same rule (``resident.Resident.forward`` / ``_pass``
  checked on fakes);
- ``GLM53_TF_VERIFY_SPLIT`` / ``GLM53_TF_GRAPH_PROBE`` parsing, ``vsplit.split_bounds`` (every layer once, in order,
  the first piece's size), and **the split's call sequence**: with every kernel wrapper stubbed by a recorder,
  ``vsplit.compute_pieces`` called in order records exactly the calls of ``forward._compute`` (embed, each layer, the
  taps' stream means, the final stream mean, norm, head, the prefetch join), plus one extra ``_join`` at each inner
  piece boundary (a wait only);
- ``vsplit.capture`` / ``Split.replay`` on fake CUDA graphs (one capture a piece, in order, the probe's stamp as the
  first node of the first piece only; replay in capture order), and the probe's delay arithmetic.

GPU checks (same bits split vs unsplit, the probe on a real graph) are in the W13 session (results/THEORY2-SESSION).

Run: PYTHONPATH=<patched tree>/src pytest -q tests/test_batch_graphs_lone.py
"""

from __future__ import annotations

import collections
import contextlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.glm5_next.cuda import batch, batchplan, forward, glue, qmm, vsplit  # noqa: E402


# -- knobs --------------------------------------------------------------------------------------------------------------
def test_graph_policy():
    for v in (None, "", "1", "on", "all", " 1 "):
        assert batchplan.graph_policy(v) == batchplan.GRAPHS_ALL
    for v in ("0", "off", "none"):
        assert batchplan.graph_policy(v) == batchplan.GRAPHS_OFF
    for v in ("lone", "LONE", " lone", "single"):
        assert batchplan.graph_policy(v) == batchplan.GRAPHS_LONE
    for bad in ("2", "yes please", "multi"):
        with pytest.raises(ValueError, match="GLM53_TF_BATCH_GRAPHS"):
            batchplan.graph_policy(bad)


def test_graphs_for():
    for n in (1, 2, 3, 4):
        assert batchplan.graphs_for(batchplan.GRAPHS_ALL, n)
        assert not batchplan.graphs_for(batchplan.GRAPHS_OFF, n)
        assert batchplan.graphs_for(batchplan.GRAPHS_LONE, n) == (n == 1)


# -- the batched round's dispatch ---------------------------------------------------------------------------------------
class _FakeGraph:
    captured: list = []

    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


@contextlib.contextmanager
def _fake_capture(graph, **kw):
    _FakeGraph.captured.append(graph)
    yield


def _batcher(policy: int, n: int = 4):
    st = [SimpleNamespace(cur=[0], pos=100) for _ in range(n)]
    logits = torch.zeros((64, 8))
    e = SimpleNamespace(rows_from=None, buf=SimpleNamespace(logits=logits, route_src=None),
                        forward=lambda win: ("engine graphs", list(win)))
    fake = SimpleNamespace(
        g=SimpleNamespace(e=e, w=SimpleNamespace()), last_rows=[], counts=collections.Counter(), states=st,
        _mode=lambda s, R: 0, pad=0, buckets=0, tie=False, graph_rows=16, parity_key=True, multi={},
        max_graphs=256, sightings=batchplan.Sightings(1), pool=None, use_graphs=policy != batchplan.GRAPHS_OFF,
        graph_policy=policy, last_kind=None)
    fake._on = lambda slot: contextlib.nullcontext()
    fake.graphs_on = lambda k: batch.Batcher.graphs_on(fake, k)
    fake._parity0 = lambda s: None
    return fake


@pytest.fixture
def stubs(monkeypatch):
    calls = []
    monkeypatch.setattr(batch, "stage_multi", lambda w, sts, b, windows: sum(len(x) for x in windows))
    monkeypatch.setattr(batch, "compute_multi", lambda w, sts, b, Rs, **kw: calls.append((tuple(Rs), sorted(kw))))
    monkeypatch.setattr(batch, "chunks_for", lambda st, R: 1)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", _FakeGraph)
    monkeypatch.setattr(torch.cuda, "graph", _fake_capture)
    _FakeGraph.captured = []
    return calls


def _run(fake, active, rows):
    windows = [[1] * r for r in rows]
    return batch.Batcher._forward(fake, active, windows)


def test_lone_multi_slot_rounds_run_eagerly_like_graphs_off(stubs):
    lone, off = _batcher(batchplan.GRAPHS_LONE), _batcher(batchplan.GRAPHS_OFF)
    for _ in range(3):
        _run(lone, [0, 1], [3, 2])
        _run(lone, [0, 1, 2, 3], [1, 1, 4, 2])
    got = list(stubs)
    stubs.clear()
    for _ in range(3):
        _run(off, [0, 1], [3, 2])
        _run(off, [0, 1, 2, 3], [1, 1, 4, 2])
    assert got == stubs                                   # same calls, same keyword sets (nchs / host_pos: eager)
    assert all("nchs" in kw and "host_pos" in kw for _, kw in got)
    assert not lone.multi and not _FakeGraph.captured and lone.counts["eager"] == 6 and not lone.counts["graph"]


def test_lone_single_slot_rounds_capture_and_replay(stubs):
    lone = _batcher(batchplan.GRAPHS_LONE)
    _run(lone, [2], [3])                                  # first sight: eager through the capturable code + capture
    assert lone.counts["capture"] == 1 and len(lone.multi) == 1 and len(_FakeGraph.captured) == 1
    assert stubs[-1][1] == ["npbs"]                       # the capturable call (npbs=modes), not the eager one
    _run(lone, [2], [3])
    g = next(iter(lone.multi.values()))
    assert g.replays == 1 and lone.counts["graph"] == 1
    assert _run(lone, [0], [4])[0] == "engine graphs"     # slot 0 alone: the engine's own graphs, as before


def test_all_policy_unchanged(stubs):
    a = _batcher(batchplan.GRAPHS_ALL)
    _run(a, [0, 1], [3, 2])
    _run(a, [0, 1], [3, 2])
    assert a.counts["capture"] == 1 and a.counts["graph"] == 1


def test_resident_rounds_follow_the_policy(stubs, monkeypatch):
    resident = pytest.importorskip("tensorfold.families.glm5_next.cuda.resident")
    ran = []
    monkeypatch.setattr(batch, "compute_multi", lambda w, sts, b, Rs, **kw: ran.append(tuple(Rs)))
    for policy, active, captures in ((batchplan.GRAPHS_LONE, [0, 1], 0), (batchplan.GRAPHS_LONE, [3], 1),
                                     (batchplan.GRAPHS_ALL, [0, 1], 1), (batchplan.GRAPHS_OFF, [3], 0)):
        bat = _batcher(policy)
        ops = SimpleNamespace(bat=bat, w=SimpleNamespace(), e=bat.g.e,
                              res=SimpleNamespace(kda=False, st=SimpleNamespace(SRC=None)))
        Rs = [2] * len(active)
        for _ in range(2):
            resident.RealOps.forward(ops, active, Rs, [0] * len(active))
        assert bat.counts["capture"] == captures and len(bat.multi) == captures
        assert bat.counts["graph"] == captures                    # the second round replays what the first captured
        assert bat.counts["eager"] == 2 - captures
    for policy, items, captures in ((batchplan.GRAPHS_LONE, [0, 2], 0), (batchplan.GRAPHS_LONE, [2], 1),
                                    (batchplan.GRAPHS_ALL, [0, 2], 1)):
        bat = _batcher(policy)                                     # the batched MTP head pass: first sight only
        ops = SimpleNamespace(bat=bat, _head_pass=lambda *a: "out")
        assert resident.RealOps._pass(ops, items, None, [1] * len(items), None, 0, None, False, [0]) == "out"
        assert len(bat.multi) == captures


# -- the verify split ---------------------------------------------------------------------------------------------------
def test_split_env():
    assert vsplit.split_env({}) == 1
    for v in ("0", "1", "off", ""):
        assert vsplit.split_env({"GLM53_TF_VERIFY_SPLIT": v}) == 1
    assert vsplit.split_env({"GLM53_TF_VERIFY_SPLIT": "4"}) == 4
    for bad in ("9", "x", "-2"):
        with pytest.raises(ValueError, match="GLM53_TF_VERIFY_SPLIT"):
            vsplit.split_env({"GLM53_TF_VERIFY_SPLIT": bad})
    assert vsplit.probe_env({}) == 0 and vsplit.probe_env({"GLM53_TF_GRAPH_PROBE": "200"}) == 200
    with pytest.raises(ValueError, match="GLM53_TF_GRAPH_PROBE"):
        vsplit.probe_env({"GLM53_TF_GRAPH_PROBE": "-1"})


@pytest.mark.parametrize("layers", [1, 2, 7, 43, 78])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 8])
def test_split_bounds(layers, n):
    for first in (None, 1, 2, 5, 100):
        b = vsplit.split_bounds(layers, n, first)
        assert b[0][0] == 0 and b[-1][1] == layers and len(b) == min(n, layers)
        assert all(lo < hi for lo, hi in b) and all(b[i][1] == b[i + 1][0] for i in range(len(b) - 1))
        if first is not None and len(b) > 1:
            assert b[0][1] == max(1, min(first, layers - (len(b) - 1)))
    if n <= layers:
        sizes = [hi - lo for lo, hi in vsplit.split_bounds(layers, n)]
        assert max(sizes) - min(sizes) <= 1 + layers % n


def _model(n_layers=6, taps=None):
    layers = [SimpleNamespace(index=i) for i in range(n_layers)]
    cfg = SimpleNamespace(hidden=8, streams=4, eps=1e-6)
    w = SimpleNamespace(cfg=cfg, layers=layers, embed="E", norm="N", head="H", meta={})
    R = 3
    t = {k: torch.zeros((16, 8)) for k in ("ids", "x", "hidden", "fnormed", "fxs", "logits", "tap0", "tap1")}
    b = SimpleNamespace(ids=t["ids"], x=t["x"], hidden=t["hidden"], fnormed=t["fnormed"], fxs=t["fxs"],
                        logits=t["logits"], sk="SK", tap_at=taps or {}, taps={0: t["tap0"], 1: t["tap1"]})
    return w, SimpleNamespace(), b, R


@pytest.fixture
def rec(monkeypatch):
    log = []
    monkeypatch.setattr(glue, "embed", lambda *a, **k: log.append(("embed",)))
    monkeypatch.setattr(glue, "stream_mean", lambda x, out: log.append(("mean", out.data_ptr())))
    monkeypatch.setattr(glue, "rmsnorm", lambda *a, **k: log.append(("norm",)))
    monkeypatch.setattr(qmm, "matmul", lambda *a, **k: log.append(("head",)))
    monkeypatch.setattr(forward, "layer_forward",
                        lambda layer, w, st, b, R, nch=None, host_pos=None, npb=None: log.append(("layer", layer.index,
                                                                                               R, nch, host_pos, npb)))
    monkeypatch.setattr(forward, "_join", lambda w: log.append(("join",)))
    return log


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("npb", [None, 2])
def test_pieces_launch_what_compute_launches(rec, n, npb):
    w, st, b, R = _model(6, taps={1: (0,), 4: (1, 0)})
    forward._compute(w, st, b, R, npb=npb)
    ref = list(rec)
    rec.clear()
    pieces = vsplit.compute_pieces(w, st, b, R, n, npb=npb)
    assert len(pieces) == min(n, 6)
    for p in pieces:
        p()
    got = list(rec)
    inner = [i for i, c in enumerate(got) if c == ("join",)]
    assert len(inner) == len(pieces)                      # one join a piece (the last is _compute's own)
    assert [c for c in got if c != ("join",)] == [c for c in ref if c != ("join",)]
    assert got[-1] == ("join",) and ref[-1] == ("join",)


def test_capture_and_replay_order(monkeypatch):
    order = []

    class G:
        def __init__(self):
            self.id = len(order)

        def replay(self):
            order.append(("replay", self.id))

    caps = []

    @contextlib.contextmanager
    def cap(g, **kw):
        caps.append(("begin", kw.get("capture_error_mode")))
        yield
        caps.append(("end",))

    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: (order.append(("new",)), G())[1])
    monkeypatch.setattr(torch.cuda, "graph", cap)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: caps.append(("sync",)))
    ran = []
    stamp = SimpleNamespace(stamp_node=lambda: ran.append("stamp"))
    s = vsplit.capture([lambda i=i: ran.append(i) for i in range(3)], "pool", key="k", probe=None,
                       error_mode="thread_local")
    assert ran == [0, 1, 2] and len(s.graphs) == 3
    assert caps == [("begin", "thread_local"), ("end",), ("sync",), ("begin", "thread_local"), ("end",), ("sync",),
                    ("begin", "thread_local"), ("end",)]
    ran.clear()
    vsplit.capture([lambda i=i: ran.append(i) for i in range(2)], "pool", probe=stamp)
    assert ran == ["stamp", 0, 1]                         # the stamp: node 0 of the first piece only
    order.clear()
    s.replay()
    assert [o[0] for o in order] == ["replay"] * 3 and [o[1] for o in order] == sorted(o[1] for o in order)


def test_probe_arithmetic():
    assert vsplit.delays_us([1000, 5000], [401000, 5600], [100000, 0]) == [300.0, 0.6]
    xs = list(range(1, 101))
    assert vsplit.pct(xs, 0.5) in (50, 51) and vsplit.pct(xs, 0.9) in (90, 91) and vsplit.pct(xs, 0.0) == 1
    assert vsplit.pct([], 0.5) != vsplit.pct([], 0.5)     # nan
    assert "4 pieces" in vsplit.describe(4, 0) and "probe every 100" in vsplit.describe(1, 100)


def test_stamp_kernel_compiles_for_sm121():
    """The probe's stamp kernel: PTX for GB10 reads %globaltimer and stores it (no GPU needed)."""

    triton = pytest.importorskip("triton")
    try:
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
    except ImportError:
        pytest.skip("this Triton has no offline compile API")
    fn = vsplit._stamp_fn()
    k = triton.compile(ASTSource(fn=fn, signature={"OUT": "*i64", "IDX": "i32"}, constexprs={}),
                       target=GPUTarget("cuda", 121, 32))
    ptx = k.asm["ptx"]
    assert "%globaltimer" in ptx and "st.global" in ptx
