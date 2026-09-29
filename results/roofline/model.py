#!/usr/bin/env python3
"""First-principles roofline of GLM-5.3-Flash (EXL3 experts, q4mse non-experts) on 2x DGX Spark GB10, TP=2.

  python3 results/roofline/model.py            (no GPU; reads results/W7/analysis/*.json, prints the tables used
                                                 in docs/ROOFLINE.md)

Everything is PER RANK (= per node): TP=2 splits every big matrix in two; the indexer, q_a / kv_a, f_a / g_a, the
router, the hyper-connection weights and eh_proj are replicated (vendor/TensorFold .../cuda/split.py), the head is
split by vocab (docs/MEMORY-4x256k.md).  Shapes: config.json + safetensors headers of the production snapshot
(neko-legends/GLM-5.3-Flash-Uncensored-EXL3 @ 07135ec0).

Roofs (per node): DRAM 230 GB/s attainable (273 nominal; the best kernels here reach 220-233), bf16/f16 mma.sync
with fp32 accumulate 110 TFLOP/s (measured, docs/EXPERT-TC.md; 48 SMs x 1,024 FLOP/clk x ~2.5 GHz = 123 spec),
e4m3 / int8 mma.sync 2x that, FP32 FMA (CUDA cores) 48 x 128 x 2 x 2.5 GHz = 30.7 TFLOP/s.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
W7 = os.path.join(HERE, "..", "W7", "analysis")

BW = 230e9          # B/s attainable DRAM
MMA = 110e12        # bf16/f16 mma.sync, fp32 accumulate
FMA = 30.7e12       # fp32 CUDA-core FMA
Q4 = 4.5 / 8        # q4mse: 4-bit + bf16 scale and bias per 64 weights = 4.5 bits a weight
EXL3 = 4.0 / 8      # EXL3 trellis, 4.0 bits a weight

H, E_I, D_I, V = 4096, 1024, 6144, 154_880   # hidden, routed/shared expert width a rank, dense MLP width a rank
N_KDA, N_DSA, N_MOE, N_DENSE = 34, 11, 42, 3
HEADS = 32                                     # attention heads a rank (64 / 2); KDA head_dim 128, DSA latent 512
TOPK, LAT = 2048, 512

M = 1e6
# ---- parameters a rank (millions) --------------------------------------------------------------------------------
kda_l = 3 * 8192 * H / 2 + H * 8192 / 2 + 2 * 8192 * 128 / 2 + 64 * H / 2 + 2 * 128 * H   # q,k,v | o | f_b,g_b | b | f_a,g_a
dsa_qkv = 1536 * H + 16384 * 1536 / 2 + 512 * H + (4096 * 1536 + 128 * H + 32 * H + 128 * H)  # q_a | q_b | kv_a | indexer
kv_b = 32768 * 512 / 2
dsa_o = H * 16384 / 2
shared_l = 3 * H * 2048 / 2
dense_l = 3 * H * 12288 / 2
head = V * H / 2
expert = 3 * H * 2048 / 2                      # one routed expert, this rank's half
router_l, hc_l = 288 * H, 2 * 24 * 16384

P = {
    "kda": N_KDA * kda_l, "dsa": N_DSA * (dsa_qkv + kv_b + dsa_o), "shared": N_MOE * shared_l,
    "dense": N_DENSE * dense_l, "head": head,
}
NE_BYTES = sum(P.values()) * Q4                                  # q4mse weights a rank
SMALL_BYTES = (N_MOE * router_l + 45 * hc_l) * 2                 # bf16 router + hc weights (replicated)
EXPERT_BYTES = expert * EXL3                                     # 6.29 MB
MOE_BYTES = N_MOE * 288 * EXPERT_BYTES


def us(x):
    return x * 1e6


# ---- prefill: FLOPs and minimum bytes a token a rank, at 8,192-row chunks / 512-row lean sub-blocks -----------------
def prefill_stages(ctx_avg):
    """(name, flops, flop_roof, bytes) a token a rank. ctx_avg: mean context over the prompt's rows."""
    KB = 1024
    idx_keys = ctx_avg / 4                                       # index_kpool 4
    st = []
    # routed: 8 picks x gate/up/down; weights once a chunk; Xg (rot_in write + read), Xd write + read, Y fp32 write
    st.append(("Routed experts (fat + rot_in + plan)", N_MOE * 8 * 2 * expert, MMA,
               MOE_BYTES / 8192 + N_MOE * (16 + 8 * 2 * 2 * E_I / KB + 8 * 4 * H / KB) * KB))
    # router dots (fp32 on the FMA pipe) + combine: read Y (8 x fp32 rows) + shared fp32 row, write the bf16 partial
    st.append(("MoE router / grouping + combine", 2 * N_MOE * router_l, FMA,
               N_MOE * (8 * 4 * H + 4 * H + 2 * H + 4 * 288 * 8)))
    st.append(("Shared expert + dense MLP (q4 GEMMs)", 2 * (P["shared"] + P["dense"]), MMA,
               (P["shared"] + P["dense"]) * Q4 / 512 + (N_MOE + N_DENSE) * 6 * 2 * H))
    st.append(("KDA projections (q4 GEMMs)", 2 * P["kda"], MMA,
               P["kda"] * Q4 / 512 + N_KDA * 2 * (H + 12576 + 8192 + H)))
    # chunked gated delta rule, C = 64, dk = dv = 128: ~184k FLOP a head and token (intra-chunk A, UT solve, W, U,
    # Q S, Q K^T, P (U - W S), state update); operands q, k, v, g (bf16) + W, U fp32 intermediates + out
    st.append(("KDA chunked recurrence", N_KDA * HEADS * 184e3, FMA,
               N_KDA * HEADS * 128 * (4 * 2 + 2 * 4 * 2 + 2)))
    absorb = 2 * HEADS * 256 * LAT
    st.append(("DSA q/kv proj + absorb", N_DSA * (2 * dsa_qkv + absorb), MMA,
               N_DSA * dsa_qkv * Q4 / 512 + N_DSA * 2 * (H + 1536 + 16384 / 2 + LAT + HEADS * LAT)))
    st.append(("DSA indexer + top-k", N_DSA * 2 * 32 * 128 * idx_keys, MMA,
               N_DSA * (idx_keys * 256 / 512 + 2 * 4 * idx_keys)))
    sel = min(TOPK, ctx_avg)
    st.append(("DSA sparse + dense attention", N_DSA * HEADS * sel * 2 * 2 * LAT, MMA,
               N_DSA * (2 * HEADS * LAT * 2 + (min(ctx_avg, 1e9) * 528) / 512)))   # q' + out; latent KV once a sub-block
    st.append(("MLA latent expand + o_proj", N_DSA * (absorb + 2 * dsa_o), MMA,
               N_DSA * (dsa_o * Q4 / 512 + 4 * HEADS * LAT + 2 * 16384 / 2 + 4 * H)))
    # hc_post + hc_pre a site: streams 4 x bf16 read, partials 2 x bf16, streams write, hc_pre re-read + normed x;
    # 90 sites; PREFILL_PP=1: each rank does half the rows
    st.append(("Hyper-connections (hc_post / hc_pre / Sinkhorn)", 90 * 2 * 24 * 4 * H / 2, FMA,
               90 * (4 * 2 * H * 2 + 2 * 2 * H + 4 * 2 * H + 2 * H) / 2))
    st.append(("MTP cache rows + DFlash2 taps", 2 * (512 * H + 1536 * H + 5 * H * H / 2), MMA, 0.3e6))
    st.append(("NCCL exposed + memcpy + host gaps + other", 0.0, MMA, 0.0))
    return st


# W7 24.5k rank 0 partition (results/W7/analysis/pf24-r0.json) -> stages
MAP = [
    ("Routed experts (fat + rot_in + plan)", ["moe.routed"]),
    ("MoE router / grouping + combine", ["moe.router", "moe.combine"]),
    ("Shared expert + dense MLP (q4 GEMMs)", ["moe.shared", "mlp.dense"]),
    ("KDA projections (q4 GEMMs)", ["kda.proj", "kda.o_proj"]),
    ("KDA chunked recurrence", ["kda.chain"]),
    ("DSA q/kv proj + absorb", ["dsa.proj"]),
    ("DSA indexer + top-k", ["dsa.indexer"]),
    ("DSA sparse + dense attention", ["dsa.sparse_attn", "dsa.attn"]),
    ("MLA latent expand + o_proj", ["dsa.o_proj"]),
    ("Hyper-connections (hc_post / hc_pre / Sinkhorn)", ["hc"]),
    ("MTP cache rows + DFlash2 taps", ["mtp.misc", "mtp.dsa.proj", "mtp.dsa.indexer", "mtp.other", "drafter"]),
]
# W7 -> production today (W8 load E: 8,192-row solo chunks + 0320 row split), us a token at 24.5k:
#   capture: W7 traced 17,323 ms for 21,464 tokens (1,239 tok/s); the same config without nsys (W8 load A) 1,261 tok/s
#   chunk 2,048 -> 8,192 (W8 D vs A, 1,261 -> 1,342.5 tok/s): -48.1 us, of it ~3.7 us idle (8 fewer piece boundaries
#     x ~10 ms) and the rest routed experts (the only chunk-size-dependent stage: weights amortized over 4x rows)
#   0320 row split (W8 E vs D, 1,342.5 -> 1,470.9): -65.0 us; hc_post / hc_pre work halves (-23 us) and most of the
#     all-gather overlap tax, which sat on hc_post (W7: 139 us alone vs 413 us beside an all-gather), goes with it;
#     attributed -57 us to hc, -8 us to the shared expert (its _fq4 also ran beside the all-gathers)
ADJ = {"Routed experts (fat + rot_in + plan)": -44.4, "Hyper-connections (hc_post / hc_pre / Sinkhorn)": -57.0,
       "Shared expert + dense MLP (q4 GEMMs)": -8.0, "NCCL exposed + memcpy + host gaps + other": -3.7}
TOK24 = 21_464
PROD_TOKS = 1_470.9


def prefill():
    a = json.load(open(os.path.join(W7, "pf24-r0.json")))["segments"][0]["prefill"]
    part, wall = a["partition_ms"], a["wall_ms"]
    cap = (1e6 / 1261.2) / (wall * 1e3 / TOK24)                 # W7 traced -> uncaptured
    meas, used = {}, set()
    for name, keys in MAP:
        meas[name] = sum(part.get(k, 0.0) for k in keys)
        used.update(keys)
    meas["NCCL exposed + memcpy + host gaps + other"] = sum(v for k, v in part.items() if k not in used)
    st = prefill_stages(ctx_avg=TOK24 / 2)
    rows = []
    for name, fl, roof_rate, by in st:
        w7_us = meas[name] * 1e3 / TOK24
        now = w7_us * cap + ADJ.get(name, 0.0)
        t_c, t_m = fl / roof_rate, by / BW
        roof = max(t_c, t_m)
        rows.append(dict(stage=name, w7_ms=meas[name], w7_us=w7_us, now_us=now, now_ms_chunk=now * 8192 / 1e3,
                         gflop=fl / 1e9, mb=by / 1e6, tflops=fl / (now * 1e-6) / 1e12 if now > 0 else 0,
                         gbs=by / (now * 1e-6) / 1e9 if now > 0 else 0, roof_us=us(roof),
                         bound="compute" if t_c >= t_m else "memory", pct=100 * us(roof) / now if now > 0 else 0,
                         gap_us=now - us(roof), gap_ms_chunk=(now - us(roof)) * 8192 / 1e3))
    return rows, wall, cap


def decode():
    """Per round bytes and floors (per rank). U(R) from experts_union.py (grid.x = 8R + 1, R = 1 calibration)."""
    d1 = json.load(open(os.path.join(W7, "dec1-r0.json")))["mean"]
    d4 = json.load(open(os.path.join(W7, "dec4-r0.json")))["mean"]
    mtp_step = (dsa_qkv + kv_b + dsa_o + shared_l + 2 * H * H) * Q4 + 8 * EXPERT_BYTES + head * Q4 + router_l * 2                                 # one MTP draft step (1 row)
    kda_state = N_KDA * HEADS * 128 * 128 * 4                       # fp32 recurrent state a rank
    out = {"ne_bytes": NE_BYTES, "small_bytes": SMALL_BYTES, "expert_bytes": EXPERT_BYTES, "mtp_step": mtp_step,
           "kda_state": kda_state, "moe_bytes": MOE_BYTES, "dec1": d1, "dec4": d4}
    return out


if __name__ == "__main__":
    rows, wall, cap = prefill()
    print(f"q4mse non-expert bytes a rank {NE_BYTES / 1e9:.3f} GB (kda {P['kda'] * Q4 / 1e9:.3f}, dsa "
          f"{P['dsa'] * Q4 / 1e9:.3f}, shared {P['shared'] * Q4 / 1e9:.3f}, dense {P['dense'] * Q4 / 1e9:.3f}, head "
          f"{P['head'] * Q4 / 1e9:.3f}); router+hc {SMALL_BYTES / 1e9:.3f} GB; expert {EXPERT_BYTES / 1e6:.2f} MB; "
          f"all experts {MOE_BYTES / 1e9:.1f} GB")
    print(f"W7 24.5k traced wall {wall:.0f} ms = {wall * 1e3 / TOK24:.1f} us/token; capture factor {cap:.3f}")
    tot = {k: sum(r[k] for r in rows) for k in ("w7_us", "now_us", "gflop", "mb", "roof_us", "gap_us")}
    print(f"{'stage':50s} {'W7 us':>7s} {'now us':>7s} {'ms/8k':>7s} {'GFLOP':>6s} {'MB':>6s} {'TF/s':>6s} {'GB/s':>6s}"
          f" {'roof':>6s} {'bound':>7s} {'%roof':>6s} {'gap us':>7s} {'gap ms/8k':>9s}")
    for r in sorted(rows, key=lambda r: -r["gap_us"]):
        print(f"{r['stage'][:50]:50s} {r['w7_us']:7.1f} {r['now_us']:7.1f} {r['now_ms_chunk']:7.0f} {r['gflop']:6.3f}"
              f" {r['mb']:6.2f} {r['tflops']:6.1f} {r['gbs']:6.1f} {r['roof_us']:6.1f} {r['bound']:>7s}"
              f" {r['pct']:6.1f} {r['gap_us']:7.1f} {r['gap_ms_chunk']:9.0f}")
    print(f"{'total':50s} {tot['w7_us']:7.1f} {tot['now_us']:7.1f} {tot['now_us'] * 8.192:7.0f} {tot['gflop']:6.2f}"
          f" {tot['mb']:6.1f} {'':6s} {'':6s} {tot['roof_us']:6.1f} {'':7s} {100 * tot['roof_us'] / tot['now_us']:6.1f}"
          f" {tot['gap_us']:7.1f} {tot['gap_us'] * 8.192:9.0f}")
    print(f"prod now {1e6 / tot['now_us']:.0f} tok/s (measured {PROD_TOKS}); sum-of-roofs ceiling "
          f"{1e6 / tot['roof_us']:.0f} tok/s; pure compute {tot['gflop'] * 1e9 / MMA * 1e6:.0f} us "
          f"-> {MMA / (tot['gflop'] * 1e9):.0f} tok/s; pure bytes {tot['mb'] * 1e6 / BW * 1e6:.0f} us")
    for ctx in (TOK24 / 2, 85_781 / 2):
        st = prefill_stages(ctx)
        print(f"ctx_avg {ctx:.0f}: GFLOP a token a rank {sum(s[1] for s in st) / 1e9:.2f}, indexer "
              f"{st[6][1] / 1e9:.3f}, attention {st[7][1] / 1e9:.3f}")
    d = decode()
    print(f"decode: MTP step {d['mtp_step'] / 1e6:.0f} MB, KDA state {d['kda_state'] / 1e6:.0f} MB a rank")
    for R, U in ((1, 8), (3, 17.0), (8, 33.0), (8, 40.0)):
        ex = U * EXPERT_BYTES * N_MOE
        kv = R * N_DSA * TOPK * 528
        tot_b = ex + NE_BYTES + SMALL_BYTES + 2 * d["kda_state"] + kv
        print(f"verify R={R} U={U}: experts {ex / 1e9:.2f} GB + q4 {NE_BYTES / 1e9:.2f} + small {SMALL_BYTES / 1e9:.2f}"
              f" + KDA state r/w {2 * d['kda_state'] / 1e9:.2f} + KV {kv / 1e9:.3f} = {tot_b / 1e9:.2f} GB -> "
              f"{tot_b / BW * 1e3:.1f} ms at 230 GB/s")
