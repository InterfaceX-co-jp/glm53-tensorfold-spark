#!/usr/bin/env bash
# W9 RoCE stage 3 (two nodes, docs/ROCE-FIX.md): bench, fault, stress (100k a size and mode), soak. Rank 0 here
# (the head node), rank 1 on the worker node (its repo copy $HOME/glm53-tensorfold-spark). Stops at the first failure.
cd $HOME/glm53-tensorfold-spark
O=results/W9/roce2; mkdir -p $O
IMG=glm53-tensorfold:b2; W=$WORKER_SSH
dargs="--rm --gpus all --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK --network host --ipc host -v glm53-tf-cache:/cache -e PYTHONDONTWRITEBYTECODE=1 -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e NCCL_IB_HCA=rocep1s0f1"
pair() { # name timeout args...
  local n=$1 t=$2; shift 2; local t0=$(date +%s)
  local cmd="cd /work && PYTHONPATH=/src/TensorFold/src:/src/TensorFold/tests/cuda:/work/tests/cuda python tests/cuda/bench_roce.py $* --master $HEAD_IP"
  ssh -o BatchMode=yes $W "docker rm -f w9-r1 >/dev/null 2>&1; timeout -k 30 $t docker run --name w9-r1 $dargs -v $HOME/glm53-tensorfold-spark/tests:/work/tests --entrypoint bash $IMG -c '$cmd --rank 1'" > $O/$n-r1.log 2>&1 &
  local sp=$!
  docker rm -f w9-r0 >/dev/null 2>&1
  timeout -k 30 $t docker run --name w9-r0 $dargs -v $PWD/tests:/work/tests --entrypoint bash $IMG -c "$cmd --rank 0" > $O/$n-r0.log 2>&1
  local rc0=$?; wait $sp; local rc1=$?
  echo "$(date +%T) $n rc0=$rc0 rc1=$rc1 $(( $(date +%s) - t0 ))s :: $(grep -E 'stress (ok|FAILED)|soak ok|raised|DIFF|Error|saved' $O/$n-r0.log | tail -1 | cut -c1-220)" | tee -a $O/SUMMARY
  [[ $rc0 == 0 && $rc1 == 0 ]]
}
pair bench 600 bench || { echo STOP bench | tee -a $O/SUMMARY; exit 1; }
pair fault 300 fault; grep -h "raised\|completed\|NO ERROR" $O/fault-r*.log | cut -c1-250 | tee -a $O/SUMMARY
pair stress 900 stress || { echo STOP stress | tee -a $O/SUMMARY; exit 1; }
pair soak $(( ${SOAK_MIN:-30} * 60 + 300 )) soak --minutes ${SOAK_MIN:-30} || { echo STOP soak | tee -a $O/SUMMARY; exit 1; }
echo "roce two-node done $(date +%T)" | tee -a $O/SUMMARY
