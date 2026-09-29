#!/usr/bin/env bash
# W12 0460 RoCE knobs (docs/PREFETCH-COMM.md §6 steps 2-3), two nodes, prod stopped: bench --trace per knob set, then
# stress (100k a size and mode, two nodes) and --loop stress per function for the sets named in STRESS. Rank 0 here,
# rank 1 on the worker node ($HOME/glm53-tensorfold-spark). roce.sh [bench|stress]
cd $HOME/glm53-tensorfold-spark
O=results/W12/roce; mkdir -p $O
IMG=glm53-tensorfold:b5; W=$WORKER_SSH
dargs="--rm --gpus all --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK --network host --ipc host -v glm53-tf-cache:/cache -e PYTHONDONTWRITEBYTECODE=1 -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e NCCL_IB_HCA=rocep1s0f1"
pair() { # name timeout "env args" args...
  local n=$1 t=$2 e=$3; shift 3; local t0=$(date +%s)
  local cmd="cd /work && PYTHONPATH=/src/TensorFold/src:/src/TensorFold/tests/cuda:/work/tests/cuda python tests/cuda/bench_roce.py $* --master <head-ip>"
  ssh -o BatchMode=yes $W "docker rm -f w12-r1 >/dev/null 2>&1; timeout -k 30 $t docker run --name w12-r1 $dargs $e -v $HOME/glm53-tensorfold-spark/tests:/work/tests --entrypoint bash $IMG -c '$cmd --rank 1'" > $O/$n-r1.log 2>&1 &
  local sp=$!
  docker rm -f w12-r0 >/dev/null 2>&1
  timeout -k 30 $t docker run --name w12-r0 $dargs $e -v $PWD/tests:/work/tests --entrypoint bash $IMG -c "$cmd --rank 0" > $O/$n-r0.log 2>&1
  local rc0=$?; wait $sp; local rc1=$?
  echo "$(date +%T) $n rc0=$rc0 rc1=$rc1 $(( $(date +%s) - t0 ))s :: $(grep -E 'stress (ok|FAILED)|soak ok|DIFF|Error|saved|mismatch' $O/$n-r0.log | tail -1 | cut -c1-200)" | tee -a $O/SUMMARY
  [[ $rc0 == 0 && $rc1 == 0 ]]
}
if [[ ${1:-bench} == bench ]]; then
  S="--sizes 16k,32k,64k,128k,256k --trace 4096"
  pair base 600 "" bench $S
  pair lean 600 "-e GLM53_TF_ROCE_LEAN=1" bench $S
  pair stripe32 600 "-e GLM53_TF_ROCE_STRIPE_KB=32" bench $S
  pair stripe64 600 "-e GLM53_TF_ROCE_STRIPE_KB=64" bench $S
  pair inline256 600 "-e GLM53_TF_ROCE_INLINE=256" bench $S
  pair lazycq 600 "-e GLM53_TF_ROCE_LAZY_CQ=1" bench $S
  pair all 600 "-e GLM53_TF_ROCE_LEAN=1 -e GLM53_TF_ROCE_STRIPE_KB=32 -e GLM53_TF_ROCE_INLINE=256 -e GLM53_TF_ROCE_LAZY_CQ=1" bench $S
  pair base2 600 "" bench $S
else
  E="${STRESS_ENV:-}"
  pair stress 1200 "$E" stress --sizes 16k,32k,128k
  pair fault 300 "$E" fault
fi
echo "roce $1 done $(date +%T)" | tee -a $O/SUMMARY
