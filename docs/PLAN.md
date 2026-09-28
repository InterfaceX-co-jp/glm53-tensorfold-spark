# Plan — GLM-5.3-Flash (abliterated) on TensorFold, 2× DGX Spark

## Goal

Serve our abliterated GLM-5.3-Flash (`neko-legends/GLM-5.3-Flash-Uncensored-EXL3` @ `07135ec0`, the weights the
production vLLM kit runs today) on TensorFold's CUDA engine across the two Sparks, and beat the production
single-stream decode by a wide margin **without changing the output** (drafted == serial, byte-identical).

Targets, single stream, thinking off, same client (`bench/glmbench.py`):

| Cell | vLLM kit (prod) | Target |
| --- | --- | --- |
| prose (kit `hashmap` / `essay`, 200 tok) | ~33 / ~26 tok/s (kit README, 2026-09-20) | ≥ 1.5× |
| structured (kit `structured`, 200 tok) | ~74 tok/s at 7/7 acceptance | ≥ parity |
| TensorFold cells (64 tok, code/chat × sampled/greedy) | measured here | ≥ 2× |

Non-goals for phase 1: 1M context, 4 concurrent sequences, prefix caching across sessions. The vLLM kit keeps
those; TensorFold is one request at a time with a short validated context (see Risks).

## Phases

1. **Baseline A** — production vLLM kit (`glm53-selfbuild:e3-pipeline-f1s8-reuse`) on the abliterated EXL3,
   benched with `glmbench.py --suites tf,tweet,kit,ctx`.
2. **Baseline B** — TensorFold 0.3.4 (`vendor/TensorFold` @ `2f8e514`), unmodified behaviour
   (`GLM53_TF_NONEXPERT=bf16`), same weights, DFlash2 `7d74cdd`; same suites + `exact`.
3. **Optimize** — `patches/0001`: store the EXL3 checkpoint's BF16 non-expert weights in 4-bit at load
   (`q4`, `q4mse`), EXL3 experts untouched; MLX-style `auto` drafter choice. Then per-policy sweeps.
4. **Quality gate** — teacher-forced top-1 agreement / NLL vs the BF16 path; refusal behaviour (abliteration
   preserved); tool-call harness from an earlier private benchmark set.
5. **Package** — Docker image + `scripts/serve.sh` + docs, ready to publish as a public project.

## Risks / open questions

- TensorFold measured GLM contexts only up to 2,051 tokens; past that DSA runs sparse top-k eagerly and was
  checked only on a truncated model. Must measure long-context correctness and speed on the full model.
- One request at a time; no cross-session prefix cache (only the last prompt/reply is resumable).
- Memory: EXL3 layout holds ~88.6 GB/rank in BF16 mode; KV at ~0.4 MB/token/rank caps the context.
- DFlash2 drafter licence is CC BY-NC-ND 4.0 — the public image must not bundle it.
