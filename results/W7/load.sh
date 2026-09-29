#!/usr/bin/env bash
# W7: start a test load from config/prod.env (image kvpool) with overrides: load.sh NAME nsys|plain [K=V ...]
# nsys: results/W7/serve-w7.sh with the nsys shim (idle session "w7"), SYS_ADMIN, W7 profile.py (both ranks print,
# NVTX marks); /tmp/w7 on both nodes holds entry.sh, bin/, profile.py and out/.
cd $HOME/glm53-tensorfold-spark
name=$1; mode=$2; shift 2
for kv in "$@"; do export "$kv"; done
export IMAGE=${IMAGE:-glm53-tensorfold:kvpool}
S=scripts/serve.sh
if [[ $mode == nsys ]]; then
  S=results/W7/serve-w7.sh
  export W7_DOCKER="--cap-add SYS_ADMIN -v /tmp/w7:/w7 -v /tmp/w7/profile.py:/usr/local/lib/python3.12/dist-packages/tensorfold/families/glm5_next/cuda/profile.py:ro --entrypoint /w7/entry.sh -e W7_NSYS=1 -e W7_NVTX=1"
fi
CONFIG=config/prod.env scripts/serve.sh stop >/dev/null 2>&1
CONFIG=config/prod.env timeout 1200 $S start > "results/W7/start-$name.log" 2>&1
echo "start rc=$? $(date +%T)"; tail -3 "results/W7/start-$name.log"
docker logs glm53-tf-r0 > "results/W7/boot-$name-r0.log" 2>&1
ssh -o BatchMode=yes $WORKER_SSH docker logs glm53-tf-r1 > "results/W7/boot-$name-r1.log" 2>&1
grep -iE 'KV pool|batching [0-9]+ requests|every slot|W7|prefill pieces' "results/W7/boot-$name-r0.log" | head
