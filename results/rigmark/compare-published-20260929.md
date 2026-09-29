| metric | TensorFold (ours) | vLLM TP2 k=7 (Alex, `glm53-libert-nvfp4-tp2-low`) | vLLM TP2 adaptive (Alex, `glm53-libert-nvfp4-tp2-adaptive-low`) | ours / k=7 | ours / adaptive |
|---|---:|---:|---:|---:|---:|
| code decode tok/s, median | 68.6 | 44.0 | 42.6 | 1.56x | 1.61x |
| code decode, 5-run range | 66.5-69.3 | 31.6-47.7 | 39.0-44.7 | - | - |
| code time to last output s, median (lower is better) | 26.3 | 46.6 | 50.2 | 0.57x | 0.52x |
| code TTFT s, median (lower is better) | 0.5 | 0.6 | 0.6 | 0.84x | 0.86x |
| code completion tokens, median | 1,790.0 | 2,116.0 | 2,053.0 | 0.85x | 0.87x |
| code reasoning chars, median (ours halved: sent twice) | 0.0 | 0.0 | 0.0 | - | - |
| code basic gate | 5/5 | 5/5 | 5/5 | - | - |
| prose decode tok/s, median | 43.2 | 18.9 | 22.2 | 2.29x | 1.95x |
| prose decode, 5-run range | 43.0-43.5 | 17.8-19.3 | 21.8-22.3 | - | - |
| prose time to last output s, median (lower is better) | 24.1 | 55.7 | 45.4 | 0.43x | 0.53x |
| prose TTFT s, median (lower is better) | 0.4 | 0.5 | 0.5 | 0.83x | 0.86x |
| prose completion tokens, median | 1,018.0 | 1,023.0 | 990.0 | 1.00x | 1.03x |
| prose reasoning chars, median (ours halved: sent twice) | 60.0 | 60.0 | 60.0 | 1.00x | 1.00x |
| prose basic gate | 5/5 | 5/5 | 5/5 | - | - |
| structured (ceiling) decode tok/s, median | 88.2 | 64.9 | 54.6 | 1.36x | 1.61x |
| structured (ceiling) decode, 5-run range | 87.3-89.1 | 63.9-66.8 | 53.0-55.7 | - | - |
| structured (ceiling) time to last output s, median (lower is better) | 8.2 | 7.3 | 8.7 | 1.13x | 0.95x |
| structured (ceiling) TTFT s, median (lower is better) | 0.5 | 0.5 | 0.5 | 0.98x | 0.97x |
| structured (ceiling) completion tokens, median | 687.0 | 441.0 | 441.0 | 1.56x | 1.56x |
| structured (ceiling) reasoning chars, median (ours halved: sent twice) | 0.0 | 18.0 | 18.0 | 0.00x | 0.00x |
| structured (ceiling) basic gate | 5/5 | 5/5 | 5/5 | - | - |
| 8K prefill cold tok/s, median | 1,597.8 | 1,812.9 | 1,834.7 | 0.88x | 0.87x |
| 8K immediate replay tok/s, median | 1,597.2 | 1,811.6 | 1,830.9 | 0.88x | 0.87x |
| 32K prefill cold tok/s, median | 1,641.2 | 1,907.9 | 1,898.1 | 0.86x | 0.86x |
| 32K immediate replay tok/s, median | 3,318.7 | 11,045.9 | 11,338.9 | 0.30x | 0.29x |
| 64K prefill cold tok/s, median | 1,621.0 | 1,921.6 | 1,905.3 | 0.84x | 0.85x |
| 64K immediate replay tok/s, median | 6,303.9 | 11,363.5 | 11,463.9 | 0.55x | 0.55x |
| C1 aggregate end-to-end tok/s, median | 54.3 | 31.6 | 31.2 | 1.72x | 1.74x |
| C1 per-stream decode tok/s, median | 60.5 | 33.9 | 33.1 | 1.78x | 1.83x |
| C1 per-stream TTFT s, median (lower is better) | 0.5 | 0.6 | 0.5 | 0.80x | 0.98x |
| C2 aggregate end-to-end tok/s, median | 65.5 | 42.0 | 42.8 | 1.56x | 1.53x |
| C2 per-stream decode tok/s, median | 37.5 | 23.9 | 24.9 | 1.57x | 1.51x |
| C2 per-stream TTFT s, median (lower is better) | 0.9 | 0.7 | 1.1 | 1.37x | 0.86x |
| C4 aggregate end-to-end tok/s, median | 82.2 | 66.1 | 61.1 | 1.24x | 1.34x |
| C4 per-stream decode tok/s, median | 24.6 | 18.7 | 16.4 | 1.32x | 1.50x |
| C4 per-stream TTFT s, median (lower is better) | 1.9 | 0.8 | 0.8 | 2.34x | 2.46x |

| appliance | TensorFold (ours) | vLLM TP2 k=7 (Alex, `glm53-libert-nvfp4-tp2-low`) | vLLM TP2 adaptive (Alex, `glm53-libert-nvfp4-tp2-adaptive-low`) |
|---|---|---|---|
| protocol / rigmark rev / comparison id | 1.1.0 / c5a0db01b054 / 2026-09-glm53-exl3-2xspark-vllm-vs-tensorfold-v1 | 1.0.0 / 046e92cbe941 / 2026-09-05-rigmark-standard-v2 | 1.0.0 / d8353e93b274 / 2026-09-05-glm-tp2-tp4-rigmark-v1 |
| model | neko-legends/GLM-5.3-Flash-Uncensored-EXL3 (GLM-5.3-Flash, abliterated, EXL3 4-bit) | LibertAIDAI/GLM-5.3-Flash-NVFP4 | LibertAIDAI/GLM-5.3-Flash-NVFP4 |
| quantisation | EXL3 4-bit experts as published; non-expert weights re-quantized to 4-bit at load (q4mse, from the checkpoint's BF16) | ModelOpt NVFP4 | ModelOpt NVFP4 |
| kv_cache_dtype | FP8 latent (MLA) KV, shared pool of 1048576 tokens | FP8 E4M3 | FP8 E4M3 |
| drafter | incoai/GLM-5.3-Flash-DFlash2@7d74cdd881ed + MTP head, cost-derived depth (cost), up to 7 drafts, verify windows up to 16 rows, suffix lookup on; greedy verification | incoai/GLM-5.3-Flash-DFlash2 | incoai/GLM-5.3-Flash-DFlash2 |
| speculative_tokens | - | 7 | 5 |
| speculative_policy | - | - | adaptive per-request k=3 or k=5, acceptance EMA |
| context_limit | 1048576 | 262144 | 262144 |
| serving_engine | TensorFold 2f8e514 + glm53-tensorfold-spark engine patches (repo unknown (not a git checkout); config/prod.env), OpenAI server from tensorfold.cuda.server | vLLM 0.1.dev20051+g487ecf187 | vLLM 0.1.dev20051+g487ecf187 |
| topology | TP2 across two nodes, direct-connect ConnectX-7 RoCE, no switch | TP2, switchless direct-link RoCE | TP2, switchless direct-link RoCE |
| max_sequences | - | 6 | 6 |
| scheduler | 4 batch slots; prefill pieces 2048 rows in a batch / 4096 alone; batch MTP 1; session store 2 GiB RAM + NVMe; prefix share 1; all-gather roce | - | - |
