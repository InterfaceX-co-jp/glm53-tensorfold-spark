#!/usr/bin/env bash
# W14: TensorFold's synthetic-engine GPU tests on the worker node in glm53-tensorfold:b6 (regression of the engine paths 0500
# touches), prod stopped. Log: $HOME/w14-tests/
O=$HOME/w14-tests; mkdir -p $O
t0=$(date +%s)
docker run --rm --name w14-engine --gpus all --network host --ipc host -e PYTHONDONTWRITEBYTECODE=1 -e TRITON_INTERPRET=0 \
  --entrypoint bash glm53-tensorfold:b6 -c "pip install -q pytest >/dev/null 2>&1; cd /src/TensorFold && PYTHONPATH=/src/TensorFold/src:/src/TensorFold/tests/cuda timeout -k 30 1800 python -m pytest -q -p no:cacheprovider -rs tests/cuda/test_glm_engine.py" > $O/gpu-engine.log 2>&1
echo "$(date +%T) gpu-engine rc=$? $(( $(date +%s) - t0 ))s :: $(grep -E 'passed|failed|error' $O/gpu-engine.log | tail -1)" | tee -a $O/SUMMARY
