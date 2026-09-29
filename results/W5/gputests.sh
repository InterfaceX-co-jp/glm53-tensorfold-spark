#!/usr/bin/env bash
# W5: GPU tests of 0270 / 0280 (+ the batching regression files) in the w5 image, one GPU.
cd "$(dirname "$0")/../.."
docker run --rm --name w5-tests --gpus all -v glm53-tf-cache:/cache -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD/tests:/work/tests" -v "$PWD/scripts:/work/scripts" -v "$PWD/results/W5:/work/out" --entrypoint bash \
  glm53-tensorfold:w5 -c "bash /work/scripts/run_tests_in_image.sh /work/out/tests -- $*"
echo "RC=$?"
