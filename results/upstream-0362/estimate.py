#!/usr/bin/env python3
"""Decode-round estimates for the upstream 0.3.6.2 kernel ideas on our GLM engine (docs/UPSTREAM-0362-AUDIT.md §2).

Inputs are the W11 in-situ measurements (docs/RESULTS.md W11 §2-§4): per-round trellis / q4 bytes, the kernels'
in-situ GB/s and the round times. Output: ms saved a round and the decode speed-up for each scenario, low / mid / high.
Offline arithmetic only; no scenario here has been measured.
"""

# W11 §2 (rank 0, uncaptured round ms; bytes a round and rank; in-situ GB/s)
CELLS = {
    #             round ms, tokens, expert GB, expert GB/s, expert ms, dense GB, dense GB/s, dense ms
    "1 stream prose": (53.3, 2.42, 5.47, 205, 27.8, 2.75, 175, 16.1),
    "1 stream code": (63.6, 4.16, 7.46, 211, 36.8, 2.78, 173, 16.5),
    "4 streams": (121.3, 9.1, 16.1, 220, 75.5, 3.51, 169, 21.7),
}

# grouped_kernel GB/s after the change (low, mid, high); None = unchanged. Ceiling 233-238 (W11 §4).
SCEN_EXPERTS = {
    # U1: upstream's K2=8 grouped kernel (GLM tile setting + PF=1 register prefetch of the next k step), same bits.
    #     Upstream's own numbers at 1 row are 204-217 GB/s (MiMo, mixed 2-4 bit), i.e. where ours already is; 0130's
    #     register prefetch did not show a gain end to end. Low = no gain.
    "U1 upstream grouped PF=1": {"1 stream prose": (205, 210, 215), "1 stream code": (211, 214, 218),
                                 "4 streams": (220, 221, 224)},
    # U2: U1 + the live-row-sized reduction buffer (38235a9's idea on the linear): 32 KB static -> dynamic,
    #     3 -> 4 CTAs an SM (register-limited at 108 regs). Same bits.
    "U2 PF=1 + sized reduction": {"1 stream prose": (207, 213, 220), "1 stream code": (212, 216, 222),
                                  "4 streams": (220, 222, 226)},
    # E1 (0440, ours; docs/DECODE-KERNELS.md §7.1): 210 / 220 / 228, epilogue launches fused (-0.4 ms side).
    "E1 0440 exl3_stream": {"1 stream prose": (210, 220, 228), "1 stream code": (214, 222, 229),
                            "4 streams": (222, 226, 230)},
}
SIDE_SAVED = {"E1 0440 exl3_stream": 0.4}   # rot_in / epilogue launches fused into the streaming kernels (1 stream)

# dense q4 GB/s after the change (low, mid, high)
SCEN_DENSE = {
    # L1: upstream's shared CUDA lane matmul (cuda/kernels/qmm.cu, 0.3.5, not an EXL3 change): 4-stage cp.async
    #     ring, register dequant, split-K summed in slice order inside a thread-block cluster (no _reduce launch).
    #     Non-persistent grid. Arrives with a rebase; same bits as our Triton _qmm is plausible, not proven.
    "L1 upstream lane matmul": {"1 stream prose": (190, 205, 215), "1 stream code": (188, 203, 213),
                                "4 streams": (180, 195, 205)},
    # E2 (0440, ours; DECODE-KERNELS §7.2): persistent, ring across items, ticketed split-K.
    "E2 0440 q4_stream": {"1 stream prose": (195, 217, 225), "1 stream code": (193, 215, 223),
                          "4 streams": (185, 200, 210)},
}
REDUCE_SAVED = 0.29   # W11: 224 _reduce launches a 1-stream round at 1.1 us + gaps; both L1 and E2 remove them


def saved(gb, old_ms, old_bw, new_bw):
    return gb / old_bw * 1e3 - gb / new_bw * 1e3 if new_bw else 0.0


def main():
    print("ms saved a round (low / mid / high) and decode speed-up at mid, from the W11 round\n")
    for group, scen, idx in (("experts", SCEN_EXPERTS, 2), ("dense", SCEN_DENSE, 5)):
        for name, per in scen.items():
            print(f"{name}:")
            for cell, row in CELLS.items():
                round_ms, tok, gb, bw = row[0], row[1], row[idx], row[idx + 1]
                extra = SIDE_SAVED.get(name, 0.0) if group == "experts" else REDUCE_SAVED
                if cell == "4 streams":
                    extra *= 2
                ds = [saved(gb, None, bw, nb) + extra for nb in per[cell]]
                mid = ds[1]
                print(f"  {cell:15s} -{ds[0]:.1f} / -{mid:.1f} / -{ds[2]:.1f} ms"
                      f"  -> {round_ms:.1f} -> {round_ms - mid:.1f} ms, {tok / round_ms * 1e3:.1f} -> "
                      f"{tok / (round_ms - mid) * 1e3:.1f} tok/s ({round_ms / (round_ms - mid) - 1:+.1%})")
            print()
    print("Prefill: none of the upstream EXL3 changes touch our prefill path (fat / fast2 experts and _fq4 are compute-")
    print("bound at 2-8k-row chunks; upstream's GLM prefill kernels are new arithmetic, not decode's): 0% expected.")


if __name__ == "__main__":
    main()
