"""patches/0470 on the CPU, no GPU: GLM53_TF_NONEXPERT=q8 and GLM53_TF_NONEXPERT_MAP (``weights.parse_map`` /
``precision`` / ``precision_key``), what ``weights.load_checkpoint`` stores per module class on the synthetic EXL3
checkpoint of upstream's engine test, and the prepared folders (patches/0140) keyed by the resolved precision: a map
that changes a class is another folder, a map that changes nothing keys the same folder as before, and a mixed tree
reads back bit for bit.

    TF_TREE=<tree> PYTHONPATH=<tree>/src pytest -q tests/test_nonexpert_map.py
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.glm5_next.cuda import fastboot, qmm, weights  # noqa: E402

if not hasattr(weights, "parse_map"):
    pytest.skip("needs patches/0470", allow_module_level=True)

from test_fastboot_prepared import same, small_chunks, synthetic  # noqa: E402,F401  (fixtures)

USER_EXAMPLE = "o_proj=q8,shared_down=q8,kda_out=q8,lm_head=bf16,default=q4mse"


@pytest.fixture
def mode(monkeypatch):
    def set_(nonexpert: str, text: str = ""):
        monkeypatch.setattr(weights, "NONEXPERT", nonexpert)
        monkeypatch.setattr(weights, "NONEXPERT_MAP", weights.parse_map(text))
    return set_


# -- parsing -------------------------------------------------------------------------------------------------------
def test_parse_valid():
    m = weights.parse_map(" O_PROJ = q8 ; shared_down=Q8,, lm_head=bf16,default=q4mse, mtp.default=q4 ")
    assert m == {"o_proj": "q8", "shared_down": "q8", "lm_head": "bf16", "default": "q4mse", "mtp.default": "q4"}
    assert weights.parse_map("") == {} and weights.parse_map(" , ") == {}
    assert weights.parse_map(USER_EXAMPLE)["kda_out"] == "q8"


@pytest.mark.parametrize("text,msg", [
    ("o_proj", "not key=mode"),
    ("oproj=q8", "unknown key"),
    ("o_proj=q5", "must be one of"),
    ("o_proj=q8,o_proj=bf16", "given twice"),
    ("mtp.kda_out=q8", "MTP layer has no"),
    ("mtp.lm_head=q8", "MTP layer has no"),
    ("o_proj=q8,out=bf16", "groups disagree"),          # kda_out / dsa_out: two groups, one level, two modes
    ("mtp.down=q8,mtp.mlp=q4", "groups disagree"),
])
def test_parse_errors(text, msg):
    with pytest.raises(ValueError, match=msg):
        weights.parse_map(text)


def test_groups_agreeing_are_fine():
    weights.parse_map("o_proj=q8,out=q8,mtp.o_proj=bf16")  # out and o_proj agree; mtp. is another level
    # kda_out / dsa_out are in attn (q4mse) and in o_proj (q8) at one level: an error until named alone
    with pytest.raises(ValueError, match="groups disagree"):
        weights.parse_map("o_proj=q8,attn=q4mse")
    with pytest.raises(ValueError, match="groups disagree on dsa_out"):
        weights.parse_map("o_proj=q8,attn=q4mse,kda_out=q8")
    m = weights.parse_map("o_proj=q8,attn=q4mse,kda_out=q8,dsa_out=q8")
    assert m["kda_out"] == m["dsa_out"] == "q8"


def test_classes_and_groups_cover_the_model():
    assert set(weights.MTP_CLASSES) <= set(weights.CLASSES)
    for g, members in weights.GROUPS.items():
        assert set(members) <= set(weights.CLASSES), g
    assert set(weights.GROUPS["out"]) == {"kda_out", "dsa_out", "dense_down", "shared_down", "lm_head"}


# -- resolution ----------------------------------------------------------------------------------------------------
def test_resolution_user_example(mode):
    mode("q4mse", USER_EXAMPLE)
    p = weights.precision
    assert p("kda_out") == p("dsa_out") == p("shared_down") == "q8"
    assert p("lm_head") == "bf16"
    for c in ("kda_proj", "kda_fb", "kda_gb", "dsa_proj", "dsa_qb", "dsa_kvb", "idx_proj", "idx_qb", "dense_gu",
              "dense_down", "shared_gu", "mtp_eh"):
        assert p(c) == "q4mse", c
    # the MTP layer inherits the unprefixed keys
    assert p("dsa_out", mtp=True) == p("shared_down", mtp=True) == "q8"
    assert p("dsa_qb", mtp=True) == "q4mse"
    assert p("") == "q4mse"


def test_resolution_order(mode):
    mode("q4", "default=q4mse,mtp.default=bf16,down=q8,mtp.down=q4,shared_down=bf16,mtp.shared_down=q8")
    p = weights.precision
    assert p("kda_proj") == "q4mse"                    # default beats GLM53_TF_NONEXPERT
    assert p("dsa_qb", mtp=True) == "bf16"             # mtp.default beats default in the MTP layer
    assert p("dense_down") == "q8"                     # group beats default
    assert p("shared_down") == "bf16"                  # class beats group
    assert p("shared_down", mtp=True) == "q8"          # mtp.class beats class
    mode("q4", "mtp.down=q4,shared_down=bf16")
    assert p("shared_down", mtp=True) == "bf16"        # class beats mtp.group
    mode("q8")
    assert all(p(c) == "q8" for c in weights.CLASSES) and p("dsa_out", mtp=True) == "q8"
    with pytest.raises(KeyError):
        weights.precision("router")


def test_precision_key(mode):
    mode("q4mse")
    assert weights.precision_key() == "q4mse"
    mode("q4mse", "default=q4mse,o_proj=q4mse")        # changes nothing: the same key (and prepared folder)
    assert weights.precision_key() == "q4mse"
    mode("bf16", "default=q8")
    assert weights.precision_key() == "q8"
    mode("q4mse", USER_EXAMPLE)
    k1 = weights.precision_key()
    assert k1.startswith("map:") and "kda_out=q8" in k1 and "lm_head=bf16" in k1 and "mtp.dsa_out=q8" in k1
    mode("q4mse", "lm_head=bf16,default=q4mse,kda_out=q8,shared_down=q8,o_proj=q8")   # same table, other spelling
    assert weights.precision_key() == k1
    mode("q4mse", USER_EXAMPLE + ",mtp.dsa_out=q4mse")
    assert weights.precision_key() != k1


# -- what the load stores ----------------------------------------------------------------------------------------------
def _kinds(w):
    """(scope, class) -> stored type name, from a loaded tree."""

    out = {}

    def put(key, q):
        t = type(q).__name__
        assert out.setdefault(key, t) == t, key
    for lw in w.layers:
        if lw.kda is not None:
            put("kda_proj", lw.kda.proj), put("kda_fb", lw.kda.fb), put("kda_gb", lw.kda.gb)
            put("kda_out", lw.kda.o)
        if lw.dsa is not None:
            put("dsa_proj", lw.dsa.proj), put("dsa_qb", lw.dsa.q_b), put("dsa_kvb", lw.dsa.kv_k)
            put("dsa_kvb", lw.dsa.kv_v), put("dsa_out", lw.dsa.o)
            put("idx_proj", lw.dsa.index.kw), put("idx_qb", lw.dsa.index.qb)
        if lw.mlp is not None:
            put("dense_gu", lw.mlp.gu), put("dense_down", lw.mlp.down)
        if lw.moe is not None:
            put("shared_gu", lw.moe.shared.gu), put("shared_down", lw.moe.shared.down)
    put("lm_head", w.head)
    m = w.mtp
    put("mtp.mtp_eh", m.eh)
    d = m.layer.dsa
    put("mtp.dsa_proj", d.proj), put("mtp.dsa_qb", d.q_b), put("mtp.dsa_kvb", d.kv_k), put("mtp.dsa_out", d.o)
    put("mtp.idx_proj", d.index.kw), put("mtp.idx_qb", d.index.qb)
    put("mtp.shared_gu", m.layer.moe.shared.gu), put("mtp.shared_down", m.layer.moe.shared.down)
    return out


TYPE = {"bf16": "B16", "q4": "Q4", "q4mse": "Q4", "q8": "Q8"}


@pytest.mark.parametrize("text", [USER_EXAMPLE, "default=q8,mtp.default=q4mse,kv_b=bf16",
                                  "out=q8,mtp.dsa_out=bf16,indexer=q4"])
def test_load_stores_each_class_as_mapped(synthetic, mode, text):
    ckpt, _ = synthetic
    mode("q4mse", text)
    w = weights.load_checkpoint(ckpt, rank=1, device="cpu")
    got = _kinds(w)
    table = weights.precision_table()
    table["mtp.mtp_eh"] = table.pop("mtp.mtp_eh", weights.precision("mtp_eh", mtp=True))
    for key, t in got.items():
        want = table.get(key) or weights.precision(key)
        assert t == TYPE[want], (key, t, want)
    # an 8-bit or BF16 head gets the 4-bit draft copy; a 4-bit head is its own
    assert (w.draft_head is None) == isinstance(w.head, qmm.Q4)
    if w.draft_head is not None:
        assert isinstance(w.draft_head, qmm.Q4) and w.draft_head.n == w.head.n
    assert w.nbytes() > 0


def test_q8_values_match_checkpoint(synthetic, mode):
    """Every 8-bit matrix dequantizes to within half a step of the checkpoint's BF16 weights (the q4mse ones do not)."""

    ckpt, _ = synthetic
    mode("q8")
    w8 = weights.load_checkpoint(ckpt, rank=0, device="cpu")
    mode("bf16")
    wb = weights.load_checkpoint(ckpt, rank=0, device="cpu")
    pairs = [(w8.head, wb.head), (w8.layers[0].kda.proj, wb.layers[0].kda.proj), (w8.layers[1].dsa.o, wb.layers[1].dsa.o),
             (w8.mtp.eh, wb.mtp.eh), (w8.layers[1].dsa.kv_k, wb.layers[1].dsa.kv_k)]
    for q, b in pairs:
        assert isinstance(q, qmm.Q8) and isinstance(b, qmm.B16)
        ref = b.weight.float()
        s = q.scales.t().float().repeat_interleave(q.gs, -1)
        assert ((qmm.dequantize_q8(q) - ref).abs() <= s / 2 + 1e-12).all()


# -- prepared folders ------------------------------------------------------------------------------------------------
def test_prepared_keyed_by_precision(synthetic, tmp_path, monkeypatch, mode, small_chunks):
    ckpt, _ = synthetic
    monkeypatch.setenv("GLM53_TF_PREPARED", str(tmp_path))
    monkeypatch.setenv("GLM53_TF_PREPARED_WRITE", "1")
    monkeypatch.setenv("GLM53_TF_PREPARED_VERIFY", "full")
    folders = {}
    for name, (ne, text) in {"q4mse": ("q4mse", ""), "q8": ("q8", ""), "bf16": ("bf16", ""),
                             "mixed": ("q4mse", USER_EXAMPLE), "mixed_mtp": ("q4mse", USER_EXAMPLE + ",mtp.default=q8"),
                             "noop": ("q4mse", "default=q4mse")}.items():
        mode(ne, text)
        folders[name] = fastboot.folder(tmp_path, fastboot.weights_key(ckpt, 0, device="cpu"))
    assert folders["noop"] == folders["q4mse"]
    distinct = [folders[k] for k in ("q4mse", "q8", "bf16", "mixed", "mixed_mtp")]
    assert len(set(distinct)) == len(distinct)
    # a mixed tree writes, validates and reads back bit for bit
    mode("q4mse", USER_EXAMPLE)
    ref = weights.load_checkpoint(ckpt, rank=0, device="cpu")
    first = weights.load(ckpt, rank=0, device="cpu")
    same(ref, first)
    key = fastboot.weights_key(ckpt, 0, device="cpu")
    assert key["nonexpert"] == weights.precision_key()
    assert fastboot.valid(fastboot.folder(tmp_path, key), key) == (True, "ok")
    monkeypatch.setattr(weights, "load_checkpoint", lambda *a, **k: pytest.fail("built instead of read"))
    again = weights.load(ckpt, rank=0, device="cpu")
    same(ref, again)
    assert isinstance(again.layers[0].kda.o, qmm.Q8) and isinstance(again.head, qmm.B16)
