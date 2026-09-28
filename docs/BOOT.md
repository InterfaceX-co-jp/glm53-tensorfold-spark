# Boot time: where a restart goes, and patches/0140

A restart of the two ranks (`scripts/serve.sh start`, launcher to `/v1/models` answering) took **303-313 s** in
the current configuration (Load A: q4mse non-experts, latent KV, 262k context, fast/lean prefill,
`GLM53_TF_CALIB=real`, DFlash2, 12 GiB session store); 274-490 s across the configurations of 2026-09-27
(`/tmp/tf/tf-start*.log`, `/tmp/loadA.log`, `/tmp/serve-*.log` on the head node). The goal, borrowed from MiaAI-Lab's PR
#230 for the vLLM kit (restart 259 s -> 122 s), is ~100 s or less.

## What the logs say (before)

The server printed almost nothing between `loading ...` and the calibration result, so the phases below come from:

- **Measured**: launcher totals (above); `serving ... loaded in 371.0s` (rank 0, CLI start to HTTP, the older
  32k-context configuration, `/tmp/tf/matrix-load-r0.log`); imports in the image (torch + triton 1.0 s, the GLM
  engine modules load lazily); the NVMe with O_DIRECT (worker, one reader 9.7-11.6 GB/s, 4 and 8 readers
  ~10.4-11 GB/s); and **the current read path on the CPU side alone**: `split.RankReader` (file-backed `np.memmap`,
  one thread, a copy a tensor) over three checkpoint files cold: 5,424 tensors, 2.86 GB of rank share from 5.70 GB
  of files in 7.4 s, i.e. **0.77 GB/s of checkpoint, 0.38 GB/s of share, 1.37 ms a tensor**. The checkpoint is
  164 GB in 92 files, 150,226 tensors.
- **Estimated**: everything else (marked ~), from the code path and the kernels' measured speeds. The `[boot]`
  lines patches/0140 adds replace these with measurements on the next start (below).

| Phase (a rank) | Before | Where the time goes |
| --- | --- | --- |
| launcher | ~3-8 s + up to 10 s polling | `docker rm -f` of both ranks, then rank 1 over ssh, then rank 0, one after the other; readiness polled every 10 s (5 s lost on average) |
| container start + imports | ~2-4 s | `docker run --gpus`, entrypoint, `import torch, triton` (1.0 s warm), CLI |
| NCCL init | ~2-5 s | TCPStore rendezvous, `ncclCommInitRank` over the CX7 link |
| **weights** | **~200-230 s** | `RankReader` slices every tensor out of the full checkpoint through an mmap: ~164 GB of pages touched a rank at 0.77 GB/s (the `dim1` split of the gate/up trellises and the column split read whole pages for half their bytes: ~1/3 more than the 82 GB share), ~150k Python-level gets at ~1.4 ms; pageable host-to-device copies; q4mse clip search of ~5.4 G BF16 non-expert parameters on the GPU (~10-20 s); the page cache it fills is GPU memory on GB10 (MemFree) |
| barrier | 0-20 s | the faster rank waits for the slower |
| DFlash2 drafter | ~3-6 s | 2.3 GB BF16 checkpoint through `safe_open`, rank slices, 4-bit quantization |
| engine: caches, buffers, graphs | ~10-25 s | KV / latent / ring caches, 8192-row prefill buffers, torch-extension loads (ninja no-op), Triton kernels from the cache, 12 main + 4 MTP CUDA graphs (each 2 warm runs + capture) |
| DFlash2 graphs | ~1-3 s | 9 captures |
| **calibration** (`real`) | **~15-30 s** | tokenize, prefill ~250 tokens, 24 greedy tokens, then 3 spans x (prefill + 9 turns x 13 timed pieces) |
| session store, batcher | ~1-3 s | 12 GiB store |
| **total** | **~303-313 s** | |

Not a factor here, unlike the vLLM kit: the image's torch has SASS for sm_120 (runs on GB10's sm_121) and Triton
compiles for sm_121, so the driver JIT cache (`~/.nv/ComputeCache`, lost with every `docker rm`) holds little; the
Triton cache (2 GB) and the torch extensions already persist in the `glm53-tf-cache` volume; imports are 1 s. The
64 KiB-page wedge Mia hit (cuMemcpyHtoDAsync from file-backed mmaps) does not apply: both Sparks run 4 KiB pages
(7.0.0-1019-nvidia), and `RankReader` copies off the mmap before `.to(device)`.

## What patches/0140 changes

| Phase | After (expected) | Change |
| --- | --- | --- |
| launcher | ~2 s + <= 1 s polling | both ranks started at once; readiness every 1 s (the worker's container over ssh every 10 s); stop was already parallel (`stop_both`) |
| container + imports | ~2-4 s | unchanged (`CUDA_CACHE_PATH=/cache/nv/ComputeCache`, 4 GiB, persists the driver JIT cache anyway) |
| NCCL init | ~2-5 s | unchanged |
| **weights** | **~9-15 s** | prepared rank folder: the built weights (split, q4mse, tiled, EXL3 words) as stored bytes, ~82 GB a rank read by 8 threads in 64 MiB O_DIRECT chunks into pinned buffers, async copies to the device; no page cache, no conversion. Bit-identical to the load-time path (tested) |
| barrier | ~0-3 s | both ranks now read at disk speed |
| DFlash2 drafter | < 1 s | prepared too (~0.4 GB a rank) |
| engine + graphs | ~10-25 s | unchanged (the `[boot]` line will say whether it is worth a next step) |
| DFlash2 graphs | ~1-3 s | unchanged |
| **calibration** | **~1-2 s** | `GLM53_TF_CALIB=cached`: rank 0 reads the table stored for this pair of ranks (image, knobs, engine shape, GPUs, clock caps) and shares it; one prefill of the calibration prompt as warm-up |
| session store | ~1-3 s | unchanged |
| **total** | **~40-70 s** | first start after a change of image / NONEXPERT / checkpoint: the old path, plus ~20-40 s writing the folder (`GLM53_TF_PREPARED_WRITE=1`), unless `scripts/prepare.sh` ran first |

The reader's target is >= 4.5 GB/s a rank (Mia's drive tops out at ~4.9 GB/s single-reader); this NVMe does ~10
GB/s with O_DIRECT, so the weights phase should be ~10 s. `tests/cuda/test_boot_patches.py::test_reader_throughput`
measures it with 1, 4, 8 and 16 threads.

### Next candidates, once the `[boot]` lines are in

- **engine + graphs**: 16 main/MTP graphs each run the step twice before capture; one warm run is enough after the
  first graph of a row count. Graph capture itself cannot be cached across processes.
- The calibration warm-up prefill (~1 s) can go (`GLM53_TF_BOOT_WARMUP=0`) if the first request's latency does not
  mind the kernel loads.
- Buffered checkpoint reads for the fallback path (a start with no valid folder): a parallel reader over
  `RankReader` would cut the first start too.

## Using it

One-time, with the server stopped (needs each node's GPU for a few minutes; same `IMAGE` and `GLM53_TF_NONEXPERT`
as serving, from `config/tensorfold.env` or exports):

```
scripts/serve.sh stop
GLM53_TF_NONEXPERT=q4mse IMAGE=glm53-tensorfold:<tag> scripts/prepare.sh      # both nodes in parallel
scripts/prepare.sh status
scripts/serve.sh start                                                          # reads the folders
```

Folders: `HEAD_PREPARED` / `WORKER_PREPARED` (default `<HF cache>/../glm53-tf/prepared`, e.g.
`~/.cache/glm53-tf/prepared` next to `~/.cache/huggingface` on each node), then
`<model>-<rev>/<key hash>/rank<R>/{manifest.json,data.bin}` and `drafter-<model>-<rev>/...`; the two newest keys a
rank are kept. Without `prepare.sh`, `serve.sh` (GLM53_TF_PREPARED_WRITE=1) writes a missing folder after the first
start that had to build it.

- `GLM53_TF_PREPARED=` (empty) / `0`: never read or write folders (the old path).
- `GLM53_TF_PREPARED_VERIFY=full|sample|off` (default `sample`); `python -m tensorfold.families.glm5_next.cuda.fastboot
  verify <folder>` checks every chunk offline.
- `GLM53_TF_CALIB=real`: measure again (and refresh the stored table); `cached` is `serve.sh`'s default.
  `GLM53_TF_CLOCK_CAP=<MHz>`: set it when clocks are locked (`nvidia-smi -lgc`), which nvidia-smi's queries do not
  show, so a table measured at another cap is not reused.

Reading a start:

```
docker logs glm53-tf-r0 2>&1 | grep '^\[boot\]'
[boot] r0 +1.3s container started
[boot] r0 +4.0s start: container, imports, CLI (4.0s) | MemFree ...
[boot] r0 +7.1s NCCL init (both ranks up) (3.1s) | ...
[boot] r0 +18.9s weights: 82.0 GB on the GPU (11.8s) | ...
...
```

## Tests

- Host, no torch: `PYTHONPATH=<tree>/src pytest -q tests/test_fastboot_logic.py` (manifest validity and
  invalidation, folder layout, the reader's chunk plan, calibration keys, storage and rank 0 deciding for both
  ranks with two fake ranks on threads).
- Host, CPU torch: `TF_TREE=<tree> PYTHONPATH=<tree>/src pytest -q tests/test_fastboot_prepared.py` (prepared ==
  `load_checkpoint` bit for bit on the synthetic EXL3 checkpoint, both ranks, bf16 / q4 / q4mse, the drafter; a
  changed key misses; a corrupted chunk falls back to the checkpoint; O_DIRECT and buffered).
- GPU, in the image: `PYTHONPATH=/src/TensorFold/tests/cuda pytest -q -s tests/cuda/test_boot_patches.py` (the
  same round trip on the device, determinism of the load-time path, an engine on prepared weights replies the same,
  the calibration cache end to end, reader GB/s).
