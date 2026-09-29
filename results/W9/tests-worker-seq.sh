#!/usr/bin/env bash
# W9: replay W8's the worker node order: 0320 two-engine test at a08d7d6 (fails in capture: two processes on one GPU), then the
# 0310 file in a fresh container, x3 -- does the IMA follow a crashed two-process run (GPU state)?
cd $HOME/glm53-tensorfold-spark
O=results/W9/tests-worker; mkdir -p $O
while docker ps --format '{{.Names}}' | grep -q '^w9-prefix-memcheck$'; do sleep 10; done
run() { local n=$1 t=$2 e=$3 T=$4; shift 4; local t0=$(date +%s)
  docker run --rm --name w9-$n --gpus all -v glm53-tf-cache:/cache -e PYTHONDONTWRITEBYTECODE=1 $e -v $T:/work/tests \
    -v $PWD/scripts:/work/scripts --entrypoint bash glm53-tensorfold:b2 -c \
    "pip install -q pytest >/dev/null 2>&1; cd /work && PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda timeout -k 30 $t $*" > $O/$n.log 2>&1
  local rc=$?; docker rm -f w9-$n >/dev/null 2>&1
  echo "$(date +%T) $n rc=$rc $(( $(date +%s) - t0 ))s :: $(grep -E 'passed|failed|error' $O/$n.log | tail -1 | cut -c1-200)" | tee -a $O/SUMMARY
}
P="python -m pytest -q -p no:cacheprovider"
for i in 1 2 3; do
  run seq$i-twoproc-old 1200 "-e PP_TWO_PROC=1" /root/w9old/tests $P tests/cuda/test_prefill_pp_patches.py -k two_engines
  run seq$i-prefix 1200 "" $PWD/tests $P tests/cuda/test_prefix_share_patches.py
done
echo "seq done $(date +%T)" | tee -a $O/SUMMARY
