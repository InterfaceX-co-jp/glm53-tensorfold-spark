#!/usr/bin/env bash
# W9: 0310 test file with a COLD Triton cache (TRITON_CACHE_DIR in the container's /tmp: every kernel compiles in the
# test process, like W8's first run, 70 s instead of 13 s), x3; then the GPU part under compute-sanitizer memcheck.
cd $HOME/glm53-tensorfold-spark
O=results/W9/tests-worker; mkdir -p $O
run() { local n=$1 t=$2 e=$3; shift 3; local t0=$(date +%s)
  docker run --rm --name w9-$n --gpus all -v glm53-tf-cache:/cache -e PYTHONDONTWRITEBYTECODE=1 $e -v $PWD/tests:/work/tests \
    -v $PWD/scripts:/work/scripts --entrypoint bash glm53-tensorfold:b2 -c \
    "pip install -q pytest >/dev/null 2>&1; cd /work && PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda timeout -k 30 $t $*" > $O/$n.log 2>&1
  local rc=$?; docker rm -f w9-$n >/dev/null 2>&1
  echo "$(date +%T) $n rc=$rc $(( $(date +%s) - t0 ))s :: $(grep -E 'passed|failed|error|ERROR SUMMARY' $O/$n.log | tail -1 | cut -c1-200)" | tee -a $O/SUMMARY
}
P="python -m pytest -q -p no:cacheprovider"
for i in 1 2 3; do run prefix-cold-$i 1200 "-e TRITON_CACHE_DIR=/tmp/tcold" $P tests/cuda/test_prefix_share_patches.py; done
run prefix-memcheck 2400 "" compute-sanitizer --tool memcheck --error-exitcode 9 --print-limit 20 $P tests/cuda/test_prefix_share_patches.py -k gpu
echo "cold done $(date +%T)" | tee -a $O/SUMMARY
