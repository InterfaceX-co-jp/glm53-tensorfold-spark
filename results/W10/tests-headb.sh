#!/usr/bin/env bash
# W10: after tests-node.sh head, rerun 0410's GPU bits test with the bf16 harness fix (the test's _v2 forced FP8's
# 3 stages on bf16 caches: OutOfResources 133,120 > 101,376 B, not a bit difference)
cd $HOME/glm53-tensorfold-spark
while pgrep -f "tests-node.sh head" >/dev/null; do sleep 5; done
O=results/W10/tests-head; t0=$(date +%s)
docker run --rm --name w10-sparse-bits2 --gpus all -v glm53-tf-cache:/cache -e PYTHONDONTWRITEBYTECODE=1 -v $PWD/tests:/work/tests --entrypoint bash glm53-tensorfold:b4 -c \
 "pip install -q pytest >/dev/null 2>&1; cd /work && PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda timeout -k 30 900 python -m pytest -q -p no:cacheprovider -rs tests/cuda/test_sparse_v2_patches.py" > $O/sparse-bits2.log 2>&1
echo "$(date +%T) sparse-bits2 rc=$? $(( $(date +%s) - t0 ))s :: $(grep -E 'passed|failed|error' $O/sparse-bits2.log | tail -1 | cut -c1-200)" | tee -a $O/SUMMARY
