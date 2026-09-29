#!/usr/bin/env bash
# W9 RoCE stages 0-2 on the head node (docs/ROCE-FIX.md), prod stopped: preflight, GPU unit tests, the 1 MiB size / sync
# diagnosis, loopback (both functions, 1 HCA, each function), stress --loop 100k x3. Stops at the first failure.
cd $HOME/glm53-tensorfold-spark
O=results/W9/roce; mkdir -p $O
IMG=${IMG:-glm53-tensorfold:b2}
dr() { # name timeout "env args" cmd...
  local n=$1 t=$2 e=$3; shift 3; local t0=$(date +%s)
  docker rm -f w9-$n >/dev/null 2>&1
  timeout -k 30 $t docker run --rm --name w9-$n --gpus all --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK \
    --network host --ipc host -v glm53-tf-cache:/cache -e PYTHONDONTWRITEBYTECODE=1 -e NCCL_SOCKET_IFNAME=enp1s0f1np1 \
    -e NCCL_IB_HCA=rocep1s0f1 $e -v "$PWD/tests:/work/tests" -v "$PWD/results/W9:/work/w9" \
    --entrypoint bash $IMG -c "pip install -q pytest >/dev/null 2>&1; cd /work && PYTHONPATH=/src/TensorFold/src:/src/TensorFold/tests/cuda:/work/tests/cuda $*" > $O/$n.log 2>&1
  local rc=$?; docker rm -f w9-$n >/dev/null 2>&1
  echo "$(date +%T) $n rc=$rc $(( $(date +%s) - t0 ))s :: $(grep -E 'passed|failed|VERDICT|stress (ok|FAILED)|Error|bits_equal\": false' $O/$n.log | tail -1 | cut -c1-200)" | tee -a $O/SUMMARY
  return $rc
}
{ echo "== ibv_devinfo"; ibv_devinfo; for d in 0000:01:00.0 0000:01:00.1 0002:01:00.0 0002:01:00.1; do
    echo "== $d"; sudo -n mstconfig -d $d q 2>&1 | grep -iE "PCI_WR_ORDERING|RELAXED|Device type|FW" ; sudo -n lspci -vvv -s $d 2>&1 | grep -iE "RlxdOrd|MaxPayload|MaxReadReq"; done; } > $O/preflight-head.log 2>&1
grep -iE "PCI_WR_ORDERING" $O/preflight-head.log | sort | uniq -c | tee -a $O/SUMMARY
docker run --rm -v glm53-tf-cache:/cache --entrypoint ls $IMG -la /cache/roce-failed 2>&1 | tail -1 | tee -a $O/SUMMARY
dr unit 1200 "" python -m pytest -q -p no:cacheprovider tests/cuda/test_roce_patches.py || { echo STOP unit | tee -a $O/SUMMARY; exit 1; }
dr sizes 1200 "" python w9/roce_sizes.py 16,128,256,512,1024,2048,4096 20
dr loopback 600 "" python tests/cuda/bench_roce.py loopback || { echo STOP loopback | tee -a $O/SUMMARY; exit 1; }
dr loopback-hcas1 600 "-e GLM53_TF_ROCE_HCAS=1" python tests/cuda/bench_roce.py loopback || { echo STOP | tee -a $O/SUMMARY; exit 1; }
dr loopback-f0 600 "-e GLM53_TF_ROCE_HCA=rocep1s0f1" python tests/cuda/bench_roce.py loopback || { echo STOP | tee -a $O/SUMMARY; exit 1; }
dr loopback-f1 600 "-e GLM53_TF_ROCE_HCA=roceP2p1s0f1" python tests/cuda/bench_roce.py loopback || { echo STOP | tee -a $O/SUMMARY; exit 1; }
dr stress-both 900 "" python tests/cuda/bench_roce.py stress --loop || { echo STOP | tee -a $O/SUMMARY; exit 1; }
dr stress-f0 900 "-e GLM53_TF_ROCE_HCA=rocep1s0f1" python tests/cuda/bench_roce.py stress --loop || { echo STOP | tee -a $O/SUMMARY; exit 1; }
dr stress-f1 900 "-e GLM53_TF_ROCE_HCA=roceP2p1s0f1" python tests/cuda/bench_roce.py stress --loop || { echo STOP | tee -a $O/SUMMARY; exit 1; }
dr stress-256k 900 "" python tests/cuda/bench_roce.py stress --loop --sizes 256k,1m --iters 20000 || { echo STOP | tee -a $O/SUMMARY; exit 1; }
echo "roce single-node done $(date +%T)" | tee -a $O/SUMMARY
dr sizes-sync 1500 "" python w9/roce_sizes.py 256,512,1024,2048 300 sync
dr sizes-race2 900 "" python w9/roce_sizes.py 1024,2048 60 race
echo "roce single-node extra done $(date +%T)" | tee -a $O/SUMMARY
