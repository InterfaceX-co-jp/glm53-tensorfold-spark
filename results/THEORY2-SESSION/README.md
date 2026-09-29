# THEORY2-SESSION (W13): the GPU session of docs/THEORY-2.md §6

Everything the ≤ 2 h prototype window needs, prepared offline (no GPU was used to build it). One script,
`run.sh`, runs it on the head node: no-server probes on both nodes in parallel, then the server loads, then prod restored.
Logs go to `results/W13/` (`session.log` first, `summary.txt` at the end).

## What is new for it

| item (THEORY-2 §4) | built | where |
| --- | --- | --- |
| 1 lone-only batched graphs | `GLM53_TF_BATCH_GRAPHS=lone` (patch 0510, off by default) | `patches/0510-glm-batch-graphs-lone.patch`, `tests/test_batch_graphs_lone.py`, `tests/cuda/test_batch_graphs_lone_patches.py` |
| 2 verify graph launch | `probes/glprobe.py` (synthetic 100 / 400 / 1,650-node graphs, split 2 / 4 / 8, no per-node tracing); in-server `GLM53_TF_GRAPH_PROBE=N` (the real verify graph) and the fix prototype `GLM53_TF_VERIFY_SPLIT=N` (0510) | `probes/glprobe.py`, 0510 `vsplit.py` |
| 3 fused layer boundary | `GLM53_TF_HC_CUDA=1` (patch 0520, off; load-time self-check vs Triton; docs/HC-FUSED.md). Not in a server load here: bits + microbench first (gate ≤ 50% of the three kernels); an in-server load `GLM53_TF_HC_CUDA=1` + nsys comes after the gate passes | `patches/0520-glm-hc-fused.patch`, `tests/cuda/test_hc_fused_patches.py`, `tests/cuda/bench_hc_fused.py` |
| 6 rank-skew hygiene | `GLM53_TF_CPU_PIN=http` (rank 0's HTTP threads off the engine's cores) + `GLM53_TF_ROCE_TRACE_DUMP` (0460's per-exchange trace from a running server) (patch 0530) | `patches/0530-glm-http-pin.patch`, `rocetrace.py` |
| 7 E2 small shapes cold | `tests/cuda/bench_decode_cold.py` (rotation ≥ 96 MB, and behind a 64 MB streaming predecessor; per shape old vs new, same-bits check) | |
| 8 memory pipeline gate | `probes/littles.cu` + `littles.sh` (pointer chase idle / loaded; throughput vs bytes in flight for ld.global.nc.v4, cp.async ring, cp.async.bulk; alone and beside a latency kernel) | |
| 9 explain 0450's -18% | load GR: `GLM53_TF_GPU_ROUND=resident` + one graph-level nsys window + `GLM53_TF_RESIDENT_REPORT` (0510) | `req.py`, `nsys/` |

All knobs are off by default: an image with 0510 / 0520 / 0530 and no new env serves exactly what b5 serves.

## Before the window (prod keeps serving)

1. Put this commit on the head node (the repo copy is not a clone; nothing is pushed). From the workstation:

   ```
   cd <your checkout of this repo>
   git archive HEAD patches/0510-glm-batch-graphs-lone.patch patches/0520-glm-hc-fused.patch \
       patches/0530-glm-http-pin.patch results/THEORY2-SESSION docs/THEORY-2.md docs/PATCHES.md docs/HC-FUSED.md \
       tests/test_batch_graphs_lone.py tests/test_http_pin.py tests/test_theory2_probes.py tests/test_hc_fused.py \
       tests/test_hc_fused_compile.py tests/hc_fused_emu.py tests/cuda/test_batch_graphs_lone_patches.py tests/cuda/bench_decode_cold.py \
       tests/cuda/test_hc_fused_patches.py tests/cuda/bench_hc_fused.py \
     | ssh <head-ssh> 'cd ~/glm53-tensorfold-spark && tar xf -'
   ```

   Only these paths: the copy on the head node has its own `config/prod.env` (the restore and the control C use it; as of
   W12, 5c4da18: b5 + L2PF=1, L2PF_MB=8, BATCH_CAPTURE_AFTER=8) and W12's files. `bench_decode_cold.py` imports
   `tests/cuda/bench_decode_kernels.py` with W12's `load_inline` fix (committed in 5c4da18; check it is on the head node).
2. On the head node: `bash results/THEORY2-SESSION/run.sh build` (docker build of `glm53-tensorfold:b6` from every patch,
   shipped to the worker node; check `results/W13/build.log` lists `applying patches/0510..0530`), then
   `bash results/THEORY2-SESSION/run.sh sync` (tests, patches and this directory to the worker node's `~/...` copy).
   The first start of b6 re-measures the calibration (~100 s); prepared weight folders are reused.

## The window

```
cd ~/glm53-tensorfold-spark
tmux new -s w13 'bash results/THEORY2-SESSION/run.sh all 2>&1 | tee -a results/W13/run.out'
```

| phase | time | what | decides |
| --- | ---: | --- | --- |
| open | 2 min | wait for 0 requests in flight (FORCE=1 overrides), lease + refresher (tied to this script's pid), watchdog timer stopped, prod stopped, GPUs empty on both nodes, clocks logged | |
| probes | ~25 min | clocks `-lgc 2250,2250` both nodes; **worker**: glprobe plain / nsys `graph` / nsys `node`, the 0510 GPU test, 0520 bits + microbench; **head** (same time): CPU suites in the image, littles, bench_decode_cold `--mode both`; clocks back to `300,2250` | items 2, 3, 7, 8 (GATE lines in `probes-*/SUMMARY` and logs) |
| load C | ~13 min | control = prod env as of W12 (5c4da18: L2PF=1, L2PF_MB=8, BATCH_CAPTURE_AFTER=8) on b6 — full set | baseline |
| load L | ~13 min | `BATCH_GRAPHS=lone` on top of prod — full set (optional `L1`: + `BATCH_CAPTURE_AFTER=1`, add it to LOADS) | item 1, re-based on W12: prod's CAPTURE_AFTER=8 already took +2.0% at 4s and G0 cost lone slots 1-3 -3.1%, so: 4s ≥ +0.7% over C (beyond the ±0.5% spread), lone slots ≥ -0.5%, 1s ≥ 0 |
| load CT | ~7 min | control + `ROCE_TRACE=4096` + dump + `GRAPH_PROBE=100` — glmbench tf,kit x1, conc x3 | the skew baseline; the REAL verify graph's launch exposure (`lines-CT-r0.txt`) |
| load K | ~14 min | `-lgc 2250,2250` + `CPU_PIN=http` + trace + dump — full set; thread affinities in `threads-K-r0.txt` | item 6: skew wait -25% (`rocetrace-K.txt`) or 1s / 4s ≥ +0.7% |
| load VS | ~13 min | `VERIFY_SPLIT=4` + `GRAPH_PROBE=100` — full set | item 2's fix: first-node delay down, 1s ≥ +0.4%, same hashes |
| load H | ~13 min | only if the probes' 0520 bits test and `GATE item3` passed: `HC_CUDA=1` — full set | item 3 in situ: 1s ≥ +1%, same hashes |
| load GR | ~10 min | `GPU_ROUND=resident` + `RESIDENT_REPORT=10`, nsys (graph-level) window: tf code greedy x3 + one 4-stream rep (the same uncaptured first: `gr-ctl.log`) | item 9: attribute ≥ 80% of the -18% |
| restore | 5 min | prod from `config/prod.env` (canary), local + https `/v1/models`, 17*23 == 391, clocks `300,2250`, watchdog timer on, refresher killed, lease deleted | |

"Full set" (`ab.sh TAG full`) = warm-up, **exact 10/10**, **batchexact 4/4**, transcripts (== C's), ab.py 24.5k /
98k once (**reply sha 8794a3463259cc2f**, prefill), glmbench **tf,kit,edit x3** (every reply hash compared with C's in
`summary.py`), **concurrent 4 streams x3 twice (6 reps)**, lone requests over slots 0-3, /health, engine error lines.

Order and time: `SESSION_MAX_MIN` (150) skips later loads when over; GR is skipped first (it needs ~20 min) and goes
to the next window, as §6 says. `LOADS="C L K"` runs a subset; `NO_RESTORE=1` leaves the window open.

Single phases (the lease refresher then runs for `WINDOW_MAX_MIN` without a pid tie; close with `restore`):
`run.sh open`, `run.sh probes`, `run.sh loads C L`, `run.sh restore`, `run.sh status`, `run.sh summary`.

## Rules the script enforces

- The lease `~/.test-window-lease` is touched every 4 min while `run.sh all` lives; if it dies, the lease
  goes stale and the re-armed watchdog heals prod (rigmark rules: stale after 20 min).
- `glm53-tf-watchdog.timer` is stopped only inside the window and started by `restore`, which also runs on any exit or
  signal of `run.sh all` (EXIT trap).
- Restore starts prod from `config/prod.env` with no IMAGE / knob overrides (`env -u IMAGE`), retries once, and
  re-arms the watchdog even if verification fails (then it says so loudly: check `scripts/serve.sh logs 0`).
- Every probe container is `w13-*` under `timeout`; leftovers are removed at restore.
- nsys reports stay on the nodes (`/var/tmp/w13/out/`, `results/W13/probes-worker/*.nsys-rep` on the worker node); export them
  on a workstation, never on a prod node (THEORY-2 §8).

## Reading the results

- `results/W13/summary.txt`: every load vs C (1-stream geo-mean, code-like, prose, sampled, 4-stream aggregate over 6
  reps, lone slots), the bit gates (exact, batchexact, ab sha, hashes == C) and the GATE lines for L / K / VS.
- `probes-head/SUMMARY`, `probes-worker/SUMMARY` + `littles.log`, `cold.log`, `glp-*.log`, `hcbench.log`: GATE lines
  for items 8, 7, 2, 3.
- `rocetrace-CT.txt` / `rocetrace-K.txt`: transport / skew p50 / p90 and waiting beyond transport per rank.
- `lines-CT-r0.txt` / `lines-VS-r0.txt`: graph probe lines (host us inside the verify graph's replay and the first
  node's delay, unsplit vs 4 pieces).
- `gr-*.log` + `lines-GR-*.txt`: resident per-request round kinds and drain reasons.

Adopt / build rules (THEORY-2 §6): adopt L and K (config only) if their gates pass; build items 2, 3, 7, 8 only on
their probe gates.
