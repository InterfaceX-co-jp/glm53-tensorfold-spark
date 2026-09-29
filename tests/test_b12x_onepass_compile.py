"""patches/0360: the one-pass sparse latent attention kernel (``tf_knobs.b12x`` bit 4) compiled for sm_121 (GB10)
WITHOUT a GPU, with the Triton the tree runs under (the image's 3.7.1 or 3.8; both checked 2026-09-28):

- FP8 cache with 0360's opaque tile (``OPQ``): both ``tt.dot`` ops have exactly the bf16 kernel's operand and
  accumulator encodings (``kWidth`` 2, the same ``#mma``), and the key loop runs the same compute ops in the same
  order (only how the transposed tile reaches the first dot's operand differs: data movement, no arithmetic) -- so
  the FP8 kernel is the bf16 kernel on the dequantized rows, bit for bit, whatever the tensor cores' internal order;
- control: 0240's FP8 kernel (no ``OPQ``) got ``kWidth`` 4 (Triton's layout for operands upcast from 8-bit data),
  which places K elements differently inside each m16n8k16 MMA: the W3 GPU failure of "FP8 == dequantized";
- paged (patches/0290) variants too; ``sparse_latent_one`` passes ``OPQ`` for FP8 caches only (fake launcher).

Not the interpreter: run in its own process without TRITON_INTERPRET.
    PYTHONPATH=<patched tree>/src pytest -q tests/test_b12x_onepass_compile.py      (~1 minute)
"""

from __future__ import annotations

import os
import re

import pytest

if os.environ.get("TRITON_INTERPRET") == "1":
    pytest.skip("compiles kernels: run without TRITON_INTERPRET", allow_module_level=True)

triton = pytest.importorskip("triton")
torch = pytest.importorskip("torch")
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

from tensorfold.families.glm5_next.cuda import b12x_attn  # noqa: E402

TGT = GPUTarget("cuda", 121, 32)
MOVES = {"ttg.local_alloc", "ttg.local_load", "ttg.convert_layout", "tt.trans", "ttg.memdesc_trans"}


def _compile(fp8: bool, opq: bool, paged: bool = False, bmq: int = 32):
    sig = {"QA": "*bf16", "LC": "*u8" if fp8 else "*bf16", "LCS": "*fp32" if fp8 else "*bf16", "TOK": "*i32",
           "CNT": "*i32", "OUT": "*fp32"}
    cst = {"W": 2051, "H": 32, "L": 512, "KTS": 32, "SCALE": 0.0625, "BMQ": bmq, "FP8": fp8,
           "RB": 528 if fp8 else 512, "SB": 132 if fp8 else 0, "SA": 128, "PSH": 8 if paged else 0, "OPQ": opq}
    if paged:
        sig["PT"] = "*i32"
    else:
        cst["PT"] = None
    for k in cst:
        sig[k] = "constexpr"
    try:
        return triton.compile(ASTSource(fn=b12x_attn._lsparse_one, signature=sig, constexprs=cst), target=TGT,
                              options={"num_warps": b12x_attn.WARPS, "num_stages": b12x_attn.STAGES})
    except Exception as exc:  # noqa: BLE001
        if "ptxas" in str(exc).lower() or "not found" in str(exc).lower():
            pytest.skip(f"cannot compile for sm_121 here: {exc}")
        raise


def _dots(ttgir: str) -> list[str]:
    out = []
    for line in ttgir.splitlines():
        if "tt.dot " in line:
            sig = line.strip().split(" : ", 1)[1]
            out.append(re.sub(r" loc\(.*\)$", "", re.sub(r"%[\w#]+", "%", sig)))
    return out


def _loop_ops(ttgir: str) -> list[str]:
    lines = ttgir.splitlines()
    i0 = next(i for i, x in enumerate(lines) if "scf.for" in x)
    i1 = next(i for i in range(i0, len(lines)) if "scf.yield" in lines[i])
    ops = [re.sub(r"%[\w#]+", "%", re.sub(r"^\s*(%\S+ = )?", "", x)).split(" ")[0] for x in lines[i0:i1]]
    start = next(i for i, o in enumerate(ops) if o == "ttg.local_alloc")       # from the tile on
    return [o for o in ops[start:] if o not in MOVES]


@pytest.fixture(scope="module")
def ttgir():
    return {k: _compile(*k).asm["ttgir"] for k in
            [(False, False), (True, True), (True, False), (False, False, True), (True, True, True)]}


def test_fp8_dots_are_the_bf16_dots(ttgir):
    b, f = ttgir[(False, False)], ttgir[(True, True)]
    assert len(_dots(b)) == 2 and _dots(f) == _dots(b)
    assert set(re.findall(r"kWidth = \d+", f)) == {"kWidth = 2"}
    assert re.findall(r"#mma = .*", f) == re.findall(r"#mma = .*", b)


def test_fp8_loop_computes_like_bf16(ttgir):
    assert _loop_ops(ttgir[(True, True)]) == _loop_ops(ttgir[(False, False)])


def test_paged_fp8_dots_are_the_bf16_dots(ttgir):
    assert _dots(ttgir[(True, True, True)]) == _dots(ttgir[(False, False, True)])


def test_control_0240_fp8_layout_differs(ttgir):
    """Without the opaque tile Triton picks the 8-bit upcast layout (the W3 FP8 == dequantized failure)."""

    f0 = ttgir[(True, False)]
    assert "kWidth = 4" in f0 and _dots(f0) != _dots(ttgir[(False, False)])


def test_fp8_bm16_dots_are_the_bf16_dots():
    assert _dots(_compile(True, True, bmq=16).asm["ttgir"]) == _dots(_compile(False, False, bmq=16).asm["ttgir"])


def test_sparse_latent_one_sets_opq_for_fp8_only(monkeypatch):
    """The host wrapper: OPQ for FP8 caches (compiled), never for bf16; the grid is (R, H / BM) whatever R."""

    from tensorfold.families.glm5_next.cuda import latent

    seen = []

    class Fake:
        def __getitem__(self, grid):
            def launch(*args, **kw):
                seen.append((grid, kw["OPQ"], kw["FP8"], kw["BMQ"]))
            return launch

    monkeypatch.setattr(b12x_attn, "_lsparse_one", Fake())
    for fp8 in (False, True):
        for R in (1, 7, 300):
            lat = torch.zeros(64, 512, dtype=torch.bfloat16)
            lc = latent.quantize_rows_reference(lat) if fp8 else lat
            qa = torch.zeros(R, 32, 512, dtype=torch.bfloat16)
            b12x_attn.sparse_latent_one(qa, lc, torch.zeros(R, 16, dtype=torch.int32),
                                        torch.ones(R, dtype=torch.int32), torch.zeros(R, 32, 512), 0.06)
    assert seen == [((R, 1), fp8, fp8, 32) for fp8 in (False, True) for R in (1, 7, 300)]
