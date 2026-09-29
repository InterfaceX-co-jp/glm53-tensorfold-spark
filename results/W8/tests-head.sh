#!/usr/bin/env bash
# W8 GPU tests on the head node (prod stopped): 0330 tc kernel bits under timeout FIRST (a hang must not wedge the node),
# then the expert bench (isolated + contended), then tc's engine tests. Logs: results/W8/tests-head/
cd $HOME/glm53-tensorfold-spark
O=results/W8/tests-head; mkdir -p $O
run() { # name timeout cmd...
  local n=$1 t=$2; shift 2; local t0=$(date +%s)
  docker run --rm --name w8-$n --gpus all -v glm53-tf-cache:/cache -e PYTHONDONTWRITEBYTECODE=1 -v $PWD/tests:/work/tests \
    -v $PWD/scripts:/work/scripts --entrypoint bash glm53-tensorfold:b2 -c \
    "pip install -q pytest >/dev/null 2>&1; cd /work && PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda timeout -k 30 $t $*" > $O/$n.log 2>&1
  local rc=$?; docker rm -f w8-$n >/dev/null 2>&1
  echo "$n rc=$rc $(( $(date +%s) - t0 ))s :: $(grep -E 'passed|failed|error' $O/$n.log | tail -1)" | tee -a $O/SUMMARY
  return $rc
}
run tc-kernels 900 python -m pytest -q -p no:cacheprovider -x tests/cuda/test_expert_tc_patches.py -k "'not engine'" || { echo "STOP: tc kernel tests failed or timed out" | tee -a $O/SUMMARY; exit 1; }
run bench-experts 1800 python tests/cuda/bench_experts.py 2048 4096 8192 --tc --contend --no-v1
run tc-engine 900 python -m pytest -q -p no:cacheprovider tests/cuda/test_expert_tc_patches.py -k engine
echo "head tests done $(date +%T)" | tee -a $O/SUMMARY
