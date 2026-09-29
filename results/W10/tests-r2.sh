#!/usr/bin/env bash
# W10 round 2 on the final b4 (0410 engine.py syntax fix, 0390 4-warp tiles): tests-r2.sh head|worker
# hang). tests-node.sh head|worker. Logs: results/W10/tests-<node>/ (SUMMARY: one line a step)
node=$1
if [[ $node == head ]]; then cd $HOME/glm53-tensorfold-spark; else cd $HOME/glm53-tensorfold-spark; fi
O=results/W10/tests-$node-r2; mkdir -p $O
run() { # name timeout "docker env args" cmd...
  local n=$1 t=$2 e=$3; shift 3; local t0=$(date +%s)
  docker run --rm --name w10-$n --gpus all -v glm53-tf-cache:/cache -e PYTHONDONTWRITEBYTECODE=1 $e -v $PWD/tests:/work/tests \
    -v $PWD/scripts:/work/scripts --entrypoint bash glm53-tensorfold:b4 -c \
    "pip install -q pytest >/dev/null 2>&1; nvidia-smi --query-gpu=memory.used,clocks.sm --format=csv,noheader; cd /work && PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda timeout -k 30 $t $*" > $O/$n.log 2>&1
  local rc=$?; docker rm -f w10-$n >/dev/null 2>&1
  echo "$(date +%T) $n rc=$rc $(( $(date +%s) - t0 ))s :: $(grep -E 'passed|failed|error' $O/$n.log | tail -1 | cut -c1-200)" | tee -a $O/SUMMARY
  return $rc
}
P="python -m pytest -q -p no:cacheprovider"
echo "$(date +%T) start $node" | tee -a $O/SUMMARY
if [[ $node == head ]]; then
  run mla-bits 900 "" $P tests/cuda/test_mla_expand_patches.py
  run mla-bench 900 "" python tests/cuda/bench_mla_expand.py 1 8 512 2048 8192
  run sparse-bits 900 "" $P -rs tests/cuda/test_sparse_v2_patches.py
  run mla-latent-v2 1500 "-e GLM53_TF_MLA_EXPAND=v2" $P tests/cuda/test_latent_patches.py
  run mla-kvfp8-v2 900 "-e GLM53_TF_MLA_EXPAND=v2" $P tests/cuda/test_kv_pool_patches.py tests/cuda/test_fp8_kv_patches.py -k latent
  run mla-kvfp8-v1 900 "" $P tests/cuda/test_fp8_kv_patches.py -k close_to_expanded
else
  run overlap 600 "" $P tests/cuda/test_decode_overlap_patches.py
  run deep 1500 "" $P tests/cuda/test_deep_verify_patches.py
  run bpar-ov 1200 "-e GLM53_TF_DECODE_OVERLAP=1" $P tests/cuda/test_batch_parallel_patches.py
  run bsess-ov 1200 "-e GLM53_TF_DECODE_OVERLAP=1" $P tests/cuda/test_batch_sessions_patches.py
  run lean-kda1 900 "-e GLM53_TF_KDA_V2=1" $P tests/cuda/test_lean_patches.py
fi
echo "$(date +%T) $node r2 done" | tee -a $O/SUMMARY
