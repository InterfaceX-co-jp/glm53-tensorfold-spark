#!/usr/bin/env bash
# rrun.sh NAME LOG CMD... : run a command in the w2 image with RDMA access, detached
name=$1; log=$2; shift 2
cd "$(dirname "$0")/../.."
docker rm -f "$name" >/dev/null 2>&1
setsid nohup docker run --rm --name "$name" --gpus all --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK \
  --network host --ipc host -v glm53-tf-cache:/cache -e PYTHONDONTWRITEBYTECODE=1 -e NCCL_SOCKET_IFNAME=enp1s0f1np1 \
  -e NCCL_IB_HCA=rocep1s0f1 $(env | grep -E '^GLM53_TF_ROCE' | sed 's/^/-e /') -v "$PWD/tests:/work/tests" \
  --entrypoint bash glm53-tensorfold:w2 -c "cd /work && PYTHONPATH=/src/TensorFold/src:/src/TensorFold/tests/cuda:/work/tests/cuda $*; echo RC=\$?" \
  > "$log" 2>&1 < /dev/null &
