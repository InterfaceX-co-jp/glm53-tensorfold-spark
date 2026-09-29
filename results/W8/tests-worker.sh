#!/usr/bin/env bash
# W8 GPU tests on the worker node (prod stopped): 0320 row split (incl. the one-GPU row-split kernel identity test and the
# two-engine test), 0310 prefix share, 0335 solo pieces, 0300 request log, regressions. Logs: $HOME/glm53-tensorfold-spark/results/W8/tests-worker/
cd $HOME/glm53-tensorfold-spark
O=results/W8/tests-worker; mkdir -p $O
run() { # name timeout env cmd...
  local n=$1 t=$2 e=$3; shift 3; local t0=$(date +%s)
  docker run --rm --name w8-$n --gpus all -v glm53-tf-cache:/cache -e PYTHONDONTWRITEBYTECODE=1 $e -v $PWD/tests:/work/tests \
    -v $PWD/scripts:/work/scripts --entrypoint bash glm53-tensorfold:b2 -c \
    "pip install -q pytest >/dev/null 2>&1; cd /work && PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda timeout -k 30 $t $*" > $O/$n.log 2>&1
  local rc=$?; docker rm -f w8-$n >/dev/null 2>&1
  echo "$n rc=$rc $(( $(date +%s) - t0 ))s :: $(grep -E 'passed|failed|error' $O/$n.log | tail -1)" | tee -a $O/SUMMARY
}
P="python -m pytest -q -p no:cacheprovider"
run pp 1200 "" $P tests/cuda/test_prefill_pp_patches.py
run pp-twoproc 1200 "-e PP_TWO_PROC=1" $P tests/cuda/test_prefill_pp_patches.py -k two_engines
run prefix 1200 "" $P tests/cuda/test_prefix_share_patches.py
run solo 1200 "" $P tests/cuda/test_solo_piece_patches.py
run reqlog 600 "" $P tests/test_request_log.py
run batch2 1200 "" $P tests/cuda/test_batch2_patches.py -k "'pieces or fast_prefill or lean or follower'"
run overlap 1200 "" $P tests/cuda/test_overlap_patches.py
run kvpool 1200 "" $P tests/cuda/test_kv_pool_patches.py
echo "worker tests done $(date +%T)" | tee -a $O/SUMMARY
