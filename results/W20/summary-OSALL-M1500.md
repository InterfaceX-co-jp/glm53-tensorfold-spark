== OSALL
  exact 10/10, 10/10 | batchexact [['4/4'], ['4/4']] | transcripts together == alone: {'1-s': True, '3-s': True, '0-g': True, '2-g': True}
  reply sha ['8794a3463259cc2f'] OK
  prefill 24500: 1,658, 1,663, 1,657, 1,658  (min 1,657, mean 1,659)
  prefill 98000: 1,653, 1,656, 1,652, 1,654  (min 1,652, mean 1,654)
  glmbench 1 stream geomean vs OSALL +0.00%; greedy hashes 11/11, all 13/13
  4 streams mean of 6 88.58 (sd 3.39) [92.5, 84.6, 87.8, 93.5, 85.0, 88.1]
  mmlu200 accuracy 0.880 (176/200);   refusals 0/10
  N1 drafted == serial: 6/6 | N1 batched == alone: 6/6
  needle 314267 tokens: cold found True prefill 221.9578 s (1415.9 tok/s); resend found True cached 314240
  OSALL: MemAvailable min by phase (head / worker GiB)
    pre       15.60 /  15.10
    ab        14.35 /  13.79
    ab2       13.80 /  13.32
    n1        13.34 /  12.77
    stress    10.72 /  10.77
    mmlu      11.25 /  11.19
    needle    11.38 /  11.19
    ab3       12.19 /  11.73
    ab4       11.31 /  10.84
    end       11.62 /  11.16
    stress+mmlu+needle min 10.72 / 10.77  -> PASS (>= 8 GiB)
    stress min 10.72 / 10.77
    needle min 11.38 / 11.19
    whole load min 10.72 / 10.77
    MemFree steps >= 1 GiB (r0): mmlu 1, needle 1
    MemFree steps >= 1 GiB (r1): mmlu 1, needle 1
  oom / nvrm lines after: 0 0 nvrm-nomem 0 0 (before: 0 0 nvrm-nomem 0 0)
== M1500
  exact 10/10, 0/10 | batchexact [['4/4'], []] | transcripts together == alone: {'1-s': True, '3-s': True, '0-g': True, '2-g': True}
  reply sha ['8794a3463259cc2f'] OK
  prefill 24500: 1,658  (min 1,658, mean 1,658)
  prefill 98000: 1,648  (min 1,648, mean 1,648)
  glmbench 1 stream geomean vs OSALL -0.16%; greedy hashes 11/11, all 13/13
  4 streams mean of 6 87.83 (sd 3.58) [91.8, 84.3, 86.7, 93.3, 83.7, 87.2]

| metric | OSALL | M1500 |
| --- | --- | --- |
| idle MemAvailable min r0 / r1 (GiB) | 17.93 / 17.69 | 19.45 / 19.06 |
| idle MemFree mean r0 / r1 (GiB) | 15.51 / 16.49 | 17.18 / 17.93 |
| stress min MemAvailable r0 / r1 | 10.72 / 10.77 | None |
| needle min MemAvailable r0 / r1 | 10.72 / 10.77 | None |
| stress+mmlu+needle min r0 / r1 | 10.72 / 10.77 | None |
| polkitd RSS r0 (pre) | 0.01 GiB | 0.01 GiB |
| polkitd RSS r1 (pre) | 0.01 GiB | 0.01 GiB |
| services running r0 / r1 | 34 / 33 | 34 / 33 |
| 1s: skew mean / p90 (us) | None / None | 20.85 / 38.14 |
| 1s: beyond transport r0+r1 (us/exch) | None | 20.85 |
| 1s: rank 0 late (%) | None | 57 |
| 4s: skew mean / p90 (us) | 39.77 / 88.13 | 43.59 / 97.31 |
| 4s: beyond transport r0+r1 (us/exch) | 39.77 | 43.59 |
| 4s: rank 0 late (%) | 50 | 67 |
| transport p50 1s / 4s (us) | None / 4.51 | 2.37 / 2.94 |
| 1s decode: ctx switches/s r0 / r1 | 30064 / 29604 | 31521 / 29760 |
| 1s decode: top-15 IRQs/s r0 / r1 | 15730 / 17772 | 16550 / 18257 |
| 4s decode: ctx switches/s r0 / r1 | 9336 / 6582 | 7006 / 6172 |
| 4s decode: busy % X925 r0 / r1 | 20.57 / 20.61 | 20.5 / 20.53 |
| 4s decode: busy % A725 r0 / r1 | 1.83 / 0.83 | 1.44 / 1.25 |
| boot -> rank 0 start (s) | 750 | None |
| boot -> serving (s) | 756 | None |

(decode / prefill / 4-stream / MMLU / exactness: summ.py block above; glmbench geomean is vs the first name)
