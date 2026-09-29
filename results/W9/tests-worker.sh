#!/usr/bin/env bash
# W9 GPU tests on the worker node (prod stopped): (1) 0310 test_prefix_share_patches.py x5 whole file (the W8 IMA), then the
# 0360 GPU plan steps 1-2. Logs: $HOME/glm53-tensorfold-spark/results/W9/tests-worker/
cd $HOME/glm53-tensorfold-spark
O=results/W9/tests-worker; mkdir -p $O
run() { # name timeout "docker env args" cmd...
  local n=$1 t=$2 e=$3; shift 3; local t0=$(date +%s)
  docker run --rm --name w9-$n --gpus all -v glm53-tf-cache:/cache -e PYTHONDONTWRITEBYTECODE=1 $e -v $PWD/tests:/work/tests \
    -v $PWD/scripts:/work/scripts --entrypoint bash glm53-tensorfold:b2 -c \
    "pip install -q pytest >/dev/null 2>&1; nvidia-smi --query-gpu=memory.used,clocks.sm --format=csv,noheader; cd /work && PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda timeout -k 30 $t $*" > $O/$n.log 2>&1
  local rc=$?; docker rm -f w9-$n >/dev/null 2>&1
  echo "$(date +%T) $n rc=$rc $(( $(date +%s) - t0 ))s :: $(grep -E 'passed|failed|error' $O/$n.log | tail -1 | cut -c1-200)" | tee -a $O/SUMMARY
  return $rc
}
P="python -m pytest -q -p no:cacheprovider"
fails=0
for i in 1 2 3 4 5 6; do
  run prefix-$i 1200 "" $P tests/cuda/test_prefix_share_patches.py || fails=$((fails+1))
done
echo "prefix: $fails of 6 runs failed" | tee -a $O/SUMMARY
if [[ $fails -gt 0 ]]; then
  run prefix-clb 2400 "-e CUDA_LAUNCH_BLOCKING=1" $P tests/cuda/test_prefix_share_patches.py
fi
run b12x-attn 1800 "" $P -s tests/cuda/test_b12x_attn_patches.py
run b12x-0240 1800 "" $P -s tests/cuda/test_b12x_patches.py -k "'attn or latent_sparse or snapshots or knob'"
run bench-b12x 900 "" python tests/cuda/bench_b12x.py
echo "worker tests done $(date +%T)" | tee -a $O/SUMMARY
