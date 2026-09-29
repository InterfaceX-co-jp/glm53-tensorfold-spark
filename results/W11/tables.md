## Round breakdown, rank 0 (mean over rounds; rank 1 wall in the last row)

| | 1 stream, prose (chat 256 + essay 384 + chat 256, greedy) | 1 stream, code (LRU module 384, greedy) | 4 streams, prose (4 x 384, greedy) |
| --- | ---: | ---: | ---: |
| rounds (count) | 361 | 91 | 162 |
| **round wall, ms (under capture)** | 54.59 | 64.67 | 121.97 |
| verify rows (largest window a round) | 3.52 | 5.10 | 11.76 |
| routed experts, ms (grouped_kernel + epilogues / rot_in) | 27.81 | 36.79 | 75.52 |
|   grouped_kernel ms | 26.77 | 35.36 | 73.11 |
|   distinct experts a verify layer U | 20.09 | 27.79 | 59.94 |
|   expert-layer reads a round | 876 | 1189 | 2555 |
|   trellis GB a round (rank) | 5.47 | 7.46 | 16.11 |
| grouped_kernel GB/s | 205 | 211 | 220 |
| dense q4 GEMV ms (_qmm + _reduce + _swiglu) | 16.05 | 16.55 | 21.74 |
|   q4 GB a round (rank) | 2.75 | 2.78 | 3.51 |
| _qmm + _reduce GB/s | 175 | 173 | 169 |
| RoCE all-gathers a round | 100.1 | 101.4 | 115.2 |
| RoCE all-gather us, median (mean) | 14.5 (23.1) | 17.3 (27.2) | 20.5 (35.9) |
| RoCE all-gather kernel ms | 2.31 | 2.76 | 4.15 |
| NCCL (control exchanges: 2 a round) ms | 0.02 | 0.04 | 0.08 |
| exchanges exposed (nothing else running), ms | 2.32 | 2.81 | 4.23 |
| DSA attention + indexer ms | 2.12 | 2.09 | 4.76 |
| KDA ms | 1.59 | 1.68 | 5.99 |
| hc ms | 1.45 | 1.50 | 1.83 |
| router / grouping / combine ms | 1.18 | 1.22 | 1.61 |
| norms / elementwise / memcpy / sampling ms | 0.29 | 0.33 | 1.16 |
| GPU busy (union) ms | 52.79 | 62.95 | 116.83 |
| **GPU idle inside the round, ms** | 1.81 | 1.72 | 5.14 |
|   of it inside the verify forward's host range | 1.09 | 0.87 | 2.52 |
| host gap between rounds, ms | 0.00 | 0.01 | 0.01 |
| kernels a round | 1853 | 1879 | 2621 |
| by launch phase: verify forward ms | 47.96 | 57.77 | 102.93 |
| by launch phase: drafting (MTP / DFlash2) ms | 4.32 | 4.42 | 11.40 |
| by launch phase: sampling ms | 0.08 | 0.08 | 0.57 |
| by launch phase: round other ms | 0.43 | 0.67 | 1.43 |
| host: drafting ranges (propose + mtp_chains + mtp_multi) ms | 4.99 | 5.33 | 16.69 |
| graph-replayed kernel ms | 49.26 | 38.32 | 10.80 |
| rounds mostly graph-replayed | 348 / 361 | 62 / 91 | 8 / 162 |
| rank 1 round wall ms | 54.60 | 64.68 | 121.97 |

## Against the floor (probe ceiling: 235 GB/s expert reads, 230.5 GB/s the dense set)

| workload | round ms | experts ms / floor | dense ms / floor | exchanges ms (exposed) / floor | small kernels ms / floor | idle ms | floor ms | % of floor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 stream, prose (chat 256 + essay 384 + chat 256, greedy) | 54.6 | 27.8 / 23.3 | 16.1 / 11.9 | 2.3 / 0.3 | 6.6 / 1.7 | 1.8 | 37.2 | 68% |
| 1 stream, code (LRU module 384, greedy) | 64.7 | 36.8 / 31.7 | 16.5 / 12.1 | 2.8 / 0.3 | 6.8 / 1.7 | 1.7 | 45.8 | 71% |
| 4 streams, prose (4 x 384, greedy) | 122.0 | 75.5 / 68.5 | 21.7 / 15.2 | 4.2 / 0.3 | 15.3 / 4.9 | 5.1 | 89.0 | 73% |

## Capture overhead: the same requests uncaptured (engine round_kinds: verify_ms + draft_ms a round)

| requests | uncaptured ms a round | captured | tokens a round | decode_s / rounds uncaptured |
| --- | ---: | ---: | ---: | ---: |
| prose chat + essay (load 1) | 53.3 | 54.3 (+2.0%) | 2.42 | 53.3 |
| code-lru (load 1) | 63.6 | 64.7 (+1.8%) | 4.16 | 63.6 |
| 4 x prose (load 2, 4 slots) | 109.2 | 109.6 (+0.3%) | 2.28 | 121.3 |
| chat (load 2) | 56.9 | 55.9 (+-1.9%) | 2.55 | 57.0 |

| load | pass | aggregate tok/s, reps | mean | request-rounds graph / eager / capture / alone (%) | ms a round (engine) | batchexact |
| --- | --- | --- | ---: | --- | ---: | --- |
| DEF | a | 83.3, 75.7, 76.7 | 78.6 | 11 / 80 / 7 / 2 | 113.8 | batched == alone: 4/4 [True, True, True, True] |
| DEF | b | 83.0, 74.6, 77.3 | 78.3 | 22 / 66 / 11 / 0 | 114.3 |  |
| G0 | a | 84.4, 77.2, 79.2 | 80.3 | 0 / 97 / 0 / 3 | 109.8 | batched == alone: 4/4 [True, True, True, True] |
| G0 | b | 84.0, 76.9, 79.4 | 80.1 | 0 / 97 / 0 / 3 | 109.9 |  |
| CA8 | a | 84.2, 76.4, 78.2 | 79.6 | 2 / 97 / 1 / 0 | 110.5 | batched == alone: 4/4 [True, True, True, True] |
| CA8 | b | 83.9, 77.0, 79.4 | 80.1 | 7 / 92 / 1 / 0 | 111.4 |  |
| DEF2 | a | 82.1, 74.6, 77.2 | 78.0 | 13 / 80 / 7 / 0 | 113.4 | batched == alone: 4/4 [True, True, True, True] |
| DEF2 | b | 82.9, 74.3, 76.4 | 77.8 | 28 / 60 / 12 / 0 | 115.2 |  |

DEF: mean of 6 reps 78.44 tok/s (+0.3% vs DEF+DEF2)
G0: mean of 6 reps 80.18 tok/s (+2.6% vs DEF+DEF2)
CA8: mean of 6 reps 79.86 tok/s (+2.2% vs DEF+DEF2)
DEF2: mean of 6 reps 77.91 tok/s (-0.3% vs DEF+DEF2)
