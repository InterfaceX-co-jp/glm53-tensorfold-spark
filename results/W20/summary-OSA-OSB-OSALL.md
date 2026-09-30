== OSA
  exact 10/10, 10/10 | batchexact [['4/4'], ['4/4']] | transcripts together == alone: {'1-s': True, '3-s': True, '0-g': True, '2-g': True}
  reply sha ['8794a3463259cc2f'] OK
  prefill 24500: 1,644, 1,642, 1,647, 1,646  (min 1,642, mean 1,645)
  prefill 98000: 1,635, 1,636, 1,637, 1,638  (min 1,635, mean 1,637)
  glmbench 1 stream geomean vs OSA +0.00%; greedy hashes 11/11, all 13/13
  4 streams mean of 6 84.95 (sd 2.78) [87.3, 81.6, 84.5, 89.7, 82.5, 84.1]
  mmlu200 accuracy 0.880 (176/200);   refusals 0/10
  N1 drafted == serial: 6/6 | N1 batched == alone: 6/6
  needle 314284 tokens: cold found True prefill 224.6866 s (1398.8 tok/s); resend found True cached 314240
  OSA: MemAvailable min by phase (head / worker GiB)
    pre       12.91 /  12.26
    ab        11.42 /  10.84
    ab2       10.94 /  10.35
    n1        10.40 /   9.77
    stress     8.48 /   7.92
    mmlu       8.88 /   8.37
    needle     8.80 /   8.45
    ab3        9.44 /   8.90
    ab4        8.60 /   8.07
    stress+mmlu+needle min 8.48 / 7.92  -> FAIL (>= 8 GiB)
    stress min 8.48 / 7.92
    needle min 8.80 / 8.45
    whole load min 8.48 / 7.92
    MemFree steps >= 1 GiB (r0): stress 4, mmlu 1, needle 1
    MemFree steps >= 1 GiB (r1): stress 4, mmlu 1, needle 1
  oom / nvrm lines after: 0 0 nvrm-nomem 0 0 (before: 0 0 nvrm-nomem 0 0)
== OSB
  exact 10/10, 10/10 | batchexact [['4/4'], ['4/4']] | transcripts together == alone: {'1-s': True, '3-s': True, '0-g': True, '2-g': True}
  reply sha ['8794a3463259cc2f'] OK
  prefill 24500: 1,653, 1,650, 1,648, 1,648  (min 1,648, mean 1,650)
  prefill 98000: 1,644, 1,646, 1,641, 1,648  (min 1,641, mean 1,645)
  glmbench 1 stream geomean vs OSA -0.12%; greedy hashes 11/11, all 13/13
  4 streams mean of 6 84.93 (sd 3.04) [86.6, 81.5, 85.0, 90.2, 81.3, 85.0]
  mmlu200 accuracy 0.880 (176/200);   refusals 0/10
  N1 drafted == serial: 6/6 | N1 batched == alone: 6/6
  needle 314328 tokens: cold found True prefill 224.1355 s (1402.4 tok/s); resend found True cached 314304
  OSB: MemAvailable min by phase (head / worker GiB)
    pre       15.77 /  15.35
    ab        14.37 /  13.79
    ab2       13.84 /  13.36
    n1        13.25 /  12.79
    stress    11.18 /  10.90
    mmlu      11.52 /  11.11
    needle    11.62 /  11.27
    ab3       12.11 /  11.72
    ab4       11.22 /  10.91
    stress+mmlu+needle min 11.18 / 10.90  -> PASS (>= 8 GiB)
    stress min 11.18 / 10.90
    needle min 11.62 / 11.27
    whole load min 11.18 / 10.90
    MemFree steps >= 1 GiB (r0): mmlu 1, needle 1
    MemFree steps >= 1 GiB (r1): stress 1, mmlu 1, needle 1
  oom / nvrm lines after: 0 0 nvrm-nomem 0 0 (before: 0 0 nvrm-nomem 0 0)
== OSALL
  exact 10/10, 10/10 | batchexact [['4/4'], ['4/4']] | transcripts together == alone: {'1-s': True, '3-s': True, '0-g': True, '2-g': True}
  reply sha ['8794a3463259cc2f'] OK
  prefill 24500: 1,658, 1,663, 1,657, 1,658  (min 1,657, mean 1,659)
  prefill 98000: 1,653, 1,656, 1,652, 1,654  (min 1,652, mean 1,654)
  glmbench 1 stream geomean vs OSA +4.79%; greedy hashes 11/11, all 13/13
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

| metric | OSA | OSB | OSALL |
| --- | --- | --- | --- |
| idle MemAvailable min r0 / r1 (GiB) | 16.54 / 15.89 | 19.30 / 18.89 | 17.93 / 17.69 |
| idle MemFree mean r0 / r1 (GiB) | 14.57 / 15.18 | 17.51 / 18.20 | 15.51 / 16.49 |
| stress min MemAvailable r0 / r1 | 8.48 / 7.92 | 11.18 / 10.90 | 10.72 / 10.77 |
| needle min MemAvailable r0 / r1 | 8.48 / 7.92 | 11.18 / 10.90 | 10.72 / 10.77 |
| stress+mmlu+needle min r0 / r1 | 8.48 / 7.92 | 11.18 / 10.90 | 10.72 / 10.77 |
| polkitd RSS r0 (pre) | 2.77 GiB | 0.01 GiB | 0.01 GiB |
| polkitd RSS r1 (pre) | 3.01 GiB | 0.01 GiB | 0.01 GiB |
| services running r0 / r1 | 44 / 45 | 44 / 45 | 34 / 33 |
| 1s: skew mean / p90 (us) | 26.72 / 44.86 | 22.95 / 42.05 | None / None |
| 1s: beyond transport r0+r1 (us/exch) | 26.72 | 22.94 | None |
| 1s: rank 0 late (%) | 64 | 62 | None |
| 4s: skew mean / p90 (us) | 36.15 / 73.57 | 36.34 / 66.08 | 39.77 / 88.13 |
| 4s: beyond transport r0+r1 (us/exch) | 36.15 | 36.34 | 39.77 |
| 4s: rank 0 late (%) | 62 | 50 | 50 |
| transport p50 1s / 4s (us) | 2.37 / 4.06 | 2.21 / 3.04 | None / 4.51 |
| 1s decode: ctx switches/s r0 / r1 | 30468 / 29973 | 30718 / 29851 | 30064 / 29604 |
| 1s decode: top-15 IRQs/s r0 / r1 | 16259 / 18302 | 16465 / 18223 | 15730 / 17772 |
| 4s decode: ctx switches/s r0 / r1 | 7116 / 5824 | 6954 / 5954 | 9336 / 6582 |
| 4s decode: busy % X925 r0 / r1 | 21.13 / 20.85 | 21.1 / 21.03 | 20.57 / 20.61 |
| 4s decode: busy % A725 r0 / r1 | 1.01 / 0.64 | 1.03 / 0.61 | 1.83 / 0.83 |
| boot -> rank 0 start (s) | None | None | 750 |
| boot -> serving (s) | None | None | 756 |

(decode / prefill / 4-stream / MMLU / exactness: summ.py block above; glmbench geomean is vs the first name)
