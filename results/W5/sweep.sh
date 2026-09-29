#!/usr/bin/env bash
# W5 task A: fast2 vs fat (and once) per row count, one GPU (the head node, prod stopped). Rows 1-40: decode-sized windows
# (decode never runs these kernels -- fast chunks only -- listed for completeness); 64-2048: batch pieces / tails.
cd "$(dirname "$0")/../.."
docker run --rm --name w5-sweep --gpus all -v glm53-tf-cache:/cache -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD/tests:/work/tests" --entrypoint bash glm53-tensorfold:w5 -c \
  "pip install -q pytest >/dev/null 2>&1; cd /work && PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda python tests/cuda/bench_experts.py 1 8 16 40 64 128 256 512 1024 1536 2048 4096 8192 --no-v1" \
  > results/W5/bench_experts.log 2>&1
echo "RC=$?" >> results/W5/bench_experts.log
