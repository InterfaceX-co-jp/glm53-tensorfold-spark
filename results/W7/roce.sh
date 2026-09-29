#!/usr/bin/env bash
# W7: rerun W2's 0230 RoCE single-node loopback (tests/cuda/bench_roce.py loopback, default sizes) on the head node, prod down.
# roce.sh LOG [K=V ...] (e.g. GLM53_TF_ROCE_HCAS=1)
cd $HOME/glm53-tensorfold-spark
log=$1; shift; envs=""; for kv in "$@"; do envs="$envs -e $kv"; done
docker rm -f w7-roce >/dev/null 2>&1
timeout 600 docker run --rm --name w7-roce --gpus all --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK \
  --network host --ipc host -v glm53-tf-cache:/cache -e PYTHONDONTWRITEBYTECODE=1 -e NCCL_SOCKET_IFNAME=enp1s0f1np1 \
  -e NCCL_IB_HCA=rocep1s0f1 $envs -v "$PWD/tests:/work/tests" \
  --entrypoint bash glm53-tensorfold:kvpool -c "cd /work && PYTHONPATH=/src/TensorFold/src:/src/TensorFold/tests/cuda:/work/tests/cuda python tests/cuda/bench_roce.py loopback; echo RC=\$?" \
  > "$log" 2>&1
tail -5 "$log"
