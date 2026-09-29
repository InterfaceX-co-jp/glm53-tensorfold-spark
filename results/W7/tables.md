### Prefill: whole prompts (nsys, both ranks)

#### 21,464-token prompt ('24.5k' cell) and 85,781-token prompt ('98k' cell); partition of the wall time

| component | 24.5k r0 ms | 24.5k r0 % | 24.5k r1 ms | 24.5k r1 % | 98k r0 ms | 98k r0 % | 98k r1 ms | 98k r1 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Routed experts (fat kernels, rot_in, plan) | 4,637.2 | 26.8 | 4,643.8 | 26.8 | 17,182.5 | 24.7 | 17,226.8 | 24.8 |
| MoE router / grouping + combine | 876.4 | 5.1 | 884.2 | 5.1 | 3,417.4 | 4.9 | 3,479.0 | 5.0 |
| Shared expert + dense MLP GEMMs | 996.9 | 5.8 | 989.3 | 5.7 | 3,983.3 | 5.7 | 3,802.3 | 5.5 |
| KDA: projections (q4 GEMMs) | 1,997.3 | 11.5 | 2,023.1 | 11.7 | 7,951.6 | 11.4 | 8,048.0 | 11.6 |
| KDA: chunked recurrence | 1,387.7 | 8.0 | 1,386.8 | 8.0 | 5,594.7 | 8.1 | 5,507.2 | 7.9 |
| DSA/MLA: q/kv proj + absorb | 603.1 | 3.5 | 606.8 | 3.5 | 2,400.5 | 3.5 | 2,419.2 | 3.5 |
| DSA/MLA: indexer + top-k | 283.7 | 1.6 | 291.0 | 1.7 | 2,709.6 | 3.9 | 2,737.2 | 3.9 |
| DSA/MLA: sparse + dense attention | 2,094.0 | 12.1 | 2,091.4 | 12.1 | 8,513.0 | 12.3 | 8,517.2 | 12.3 |
| DSA/MLA: latent expand + o_proj | 1,499.4 | 8.7 | 1,512.7 | 8.7 | 6,012.2 | 8.7 | 6,049.5 | 8.7 |
| Hyper-connections (hc_post / pre / Sinkhorn) | 2,248.5 | 13.0 | 2,187.4 | 12.6 | 8,965.1 | 12.9 | 8,938.4 | 12.9 |
| MTP prefill cache rows | 63.3 | 0.4 | 60.3 | 0.3 | 248.6 | 0.4 | 245.8 | 0.4 |
| DFlash2 drafter taps | 171.6 | 1.0 | 166.8 | 1.0 | 659.6 | 0.9 | 671.0 | 1.0 |
| NCCL exposed (nothing else running) | 144.0 | 0.8 | 148.7 | 0.9 | 538.6 | 0.8 | 565.0 | 0.8 |
| memcpy / memset only | 68.3 | 0.4 | 70.6 | 0.4 | 276.6 | 0.4 | 267.3 | 0.4 |
| GPU idle (host / Python gaps) | 213.1 | 1.2 | 220.7 | 1.3 | 915.2 | 1.3 | 896.3 | 1.3 |
| other (head, sample, stage, commit, piece setup) | 38.4 | 0.2 | 38.9 | 0.2 | 94.4 | 0.1 | 92.6 | 0.1 |
| **wall** | **17,323.0** | 100 | **17,322.6** | 100 | **69,463.0** | 100 | **69,463.0** | 100 |

#### One 2,048-row piece: piece 6/11 of the 24.5k prompt (context 10,240) and piece 41/42 of the 98k prompt (context 81,920)

| component | 24.5k r0 ms | 24.5k r0 % | 24.5k r1 ms | 24.5k r1 % | 98k r0 ms | 98k r0 % | 98k r1 ms | 98k r1 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Routed experts (fat kernels, rot_in, plan) | 393.0 | 24.8 | 394.8 | 24.9 | 405.5 | 23.9 | 401.9 | 23.7 |
| MoE router / grouping + combine | 83.1 | 5.2 | 84.0 | 5.3 | 82.4 | 4.9 | 83.0 | 4.9 |
| Shared expert + dense MLP GEMMs | 91.7 | 5.8 | 90.1 | 5.7 | 92.8 | 5.5 | 93.9 | 5.5 |
| KDA: projections (q4 GEMMs) | 186.7 | 11.8 | 191.1 | 12.1 | 189.5 | 11.2 | 191.4 | 11.3 |
| KDA: chunked recurrence | 132.2 | 8.3 | 133.5 | 8.4 | 134.1 | 7.9 | 133.0 | 7.9 |
| DSA/MLA: q/kv proj + absorb | 56.2 | 3.5 | 57.1 | 3.6 | 57.1 | 3.4 | 57.2 | 3.4 |
| DSA/MLA: indexer + top-k | 30.4 | 1.9 | 30.3 | 1.9 | 120.2 | 7.1 | 119.6 | 7.1 |
| DSA/MLA: sparse + dense attention | 203.4 | 12.8 | 204.8 | 12.9 | 204.2 | 12.0 | 203.3 | 12.0 |
| DSA/MLA: latent expand + o_proj | 141.3 | 8.9 | 143.7 | 9.1 | 142.9 | 8.4 | 142.5 | 8.4 |
| Hyper-connections (hc_post / pre / Sinkhorn) | 216.2 | 13.6 | 203.8 | 12.9 | 213.2 | 12.6 | 215.9 | 12.7 |
| MTP prefill cache rows | 5.7 | 0.4 | 6.0 | 0.4 | 6.4 | 0.4 | 5.6 | 0.3 |
| DFlash2 drafter taps | 16.0 | 1.0 | 15.9 | 1.0 | 15.7 | 0.9 | 15.0 | 0.9 |
| NCCL exposed (nothing else running) | 5.9 | 0.4 | 6.6 | 0.4 | 6.0 | 0.4 | 8.1 | 0.5 |
| memcpy / memset only | 6.1 | 0.4 | 5.8 | 0.4 | 6.1 | 0.4 | 5.9 | 0.3 |
| GPU idle (host / Python gaps) | 9.7 | 0.6 | 9.8 | 0.6 | 10.9 | 0.6 | 10.9 | 0.6 |
| other (head, sample, stage, commit, piece setup) | 1.4 | 0.1 | 1.6 | 0.1 | 1.5 | 0.1 | 1.5 | 0.1 |
| **wall** | **1,584.5** | 100 | **1,584.8** | 100 | **1,694.3** | 100 | **1,693.9** | 100 |

#### NCCL all-gathers in prefill (all are `ncclDevKernel_AllGather_RING_LL`, bf16 partials of 512-row sub-blocks)

| window | count | kernel ms | overlapped by compute ms | exposed ms | exposed % of wall | overlap tax ms | tax % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24.5k r0 | 4,038 | 2,302.9 | 2,158.9 | 144.0 | 0.8 | 1,495.6 | 8.6 |
| 24.5k r1 | 4,038 | 2,249.7 | 2,101.0 | 148.7 | 0.9 | 1,428.4 | 8.2 |
| 98k r0 | 15,541 | 9,187.8 | 8,649.2 | 538.6 | 0.8 | 6,030.5 | 8.7 |
| 98k r1 | 15,541 | 9,127.7 | 8,562.8 | 565.0 | 0.8 | 5,843.5 | 8.4 |

Idle split (ms): 24.5k r0: inside pieces 108, between pieces 105; 24.5k r1: inside pieces 111, between pieces 109; 98k r0: inside pieces 445, between pieces 470; 98k r1: inside pieces 440, between pieces 456

### Decode rounds (nsys, kernel time per round, mean over rounds)

| kernel family | 1 stream r0 ms | % | 1 stream r1 ms | % | 4 streams r0 ms | % | 4 streams r1 ms | % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| routed experts (exl3 grouped) | 26.43 | 44.4 | 26.32 | 44.3 | 72.31 | 57.6 | 71.77 | 57.2 |
| dense q4 GEMMs (proj, shared expert, head, drafter) | 16.29 | 27.4 | 16.33 | 27.5 | 22.98 | 18.3 | 23.02 | 18.3 |
| NCCL | 4.77 | 8.0 | 4.89 | 8.2 | 7.67 | 6.1 | 8.34 | 6.6 |
| DSA/MLA attention + indexer | 2.60 | 4.4 | 2.64 | 4.4 | 5.21 | 4.2 | 5.24 | 4.2 |
| KDA (chain, conv, replay) | 1.60 | 2.7 | 1.61 | 2.7 | 5.53 | 4.4 | 5.60 | 4.5 |
| hyper-connections | 1.51 | 2.5 | 1.52 | 2.6 | 1.76 | 1.4 | 1.75 | 1.4 |
| MoE router / grouping / combine | 1.25 | 2.1 | 1.26 | 2.1 | 1.58 | 1.3 | 1.58 | 1.3 |
| norms / elementwise / other | 0.28 | 0.5 | 0.28 | 0.5 | 0.72 | 0.6 | 0.74 | 0.6 |
| memcpy / memset | 0.10 | 0.2 | 0.10 | 0.2 | 0.45 | 0.4 | 0.52 | 0.4 |
| GPU busy (union) | 54.82 | 92.2 | 54.94 | 92.4 | 118.22 | 94.2 | 118.55 | 94.5 |
| NCCL exposed | 4.77 | 8.0 | 4.89 | 8.2 | 7.67 | 6.1 | 8.34 | 6.6 |
| GPU idle | 4.65 | 7.8 | 4.51 | 7.6 | 7.29 | 5.8 | 6.96 | 5.5 |
| **round wall (under capture)** | 59.47 | 100.0 | 59.45 | 100.0 | 125.51 | 100.0 | 125.51 | 100.0 |
| all-gathers a round | 100 |  | 100 |  | 120 |  | 120 |  |
| kernels a round | 1846 |  | 1846 |  | 2672 |  | 2673 |  |

By phase (kernel ms a round, by where the kernel was launched):

| phase | 1 stream r0 | 1 stream r1 | 4 streams r0 | 4 streams r1 |
| --- | ---: | ---: | ---: | ---: |
| verify forward | 49.20 | 49.05 | 102.42 | 102.36 |
| drafting (MTP / DFlash2) | 4.80 | 4.82 | 13.65 | 13.85 |
| round other | 0.44 | 0.44 | 1.06 | 1.06 |
| sampling | 0.30 | 0.55 | 0.64 | 0.77 |
