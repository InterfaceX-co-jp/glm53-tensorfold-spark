#!/usr/bin/env bash
# W11: start a test load from config/prod.env (image b4) with overrides: load.sh NAME nsys|plain [K=V ...]
# nsys: scripts/serve.sh with ${W11_DOCKER} added to both ranks' docker run (serve-w11.sh, generated here): the nsys
# shim (idle session "w11"), SYS_ADMIN (RmProfilingAdminOnly), the W11 profile.py copy (NVTX ranges) and /var/tmp/w11
# (entry.sh, bin/, profile.py, out/) on both nodes.
cd $HOME/glm53-tensorfold-spark
R=results/W11; name=$1; mode=$2; shift 2
for kv in "$@"; do export "$kv"; done
export IMAGE=${IMAGE:-glm53-tensorfold:b4}
echo "load $name ($mode): IMAGE=$IMAGE $* ($(date +%T))" | tee -a $R/loads.log
S=scripts/serve.sh
if [[ $mode == nsys ]]; then
  S=$R/serve-w11.sh
  python3 - <<'PY'
s = open("scripts/serve.sh").read()
a = 'cd "$(dirname "${BASH_SOURCE[0]}")/.."'
b = '-e GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" "$IMAGE"'
assert s.count(a) == 1 and s.count(b) == 1
s = s.replace(a, 'cd $HOME/glm53-tensorfold-spark   # W11 copy').replace(
    b, '-e GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" ${W11_DOCKER:-} "$IMAGE"   # W11')
open("results/W11/serve-w11.sh", "w").write(s)
PY
  chmod +x $S
  grep -c 'W11' $S
  mkdir -p /var/tmp/w11/out /var/tmp/w11/bin
  cp $R/entry.sh $R/profile.py /var/tmp/w11/; cp $R/bin/tensorfold /var/tmp/w11/bin/
  ssh -o BatchMode=yes $WORKER_SSH mkdir -p /var/tmp/w11/out /var/tmp/w11/bin
  scp -q $R/entry.sh $R/profile.py $WORKER_SSH:/var/tmp/w11/; scp -q $R/bin/tensorfold $WORKER_SSH:/var/tmp/w11/bin/
  export W11_DOCKER="--cap-add SYS_ADMIN -v /var/tmp/w11:/w11 -v /var/tmp/w11/profile.py:/usr/local/lib/python3.12/dist-packages/tensorfold/families/glm5_next/cuda/profile.py:ro --entrypoint /w11/entry.sh -e W11_NSYS=1 -e W7_NVTX=1"
fi
CONFIG=config/prod.env scripts/serve.sh stop >/dev/null 2>&1
CONFIG=config/prod.env timeout 1200 $S start > "$R/start-$name.log" 2>&1
echo "start rc=$? $(date +%T)" | tee -a $R/loads.log; tail -3 "$R/start-$name.log"
docker logs glm53-tf-r0 > "$R/boot-$name-r0.log" 2>&1
ssh -o BatchMode=yes $WORKER_SSH docker logs glm53-tf-r1 > "$R/boot-$name-r1.log" 2>&1
grep -iE 'roce conn|W7|nsys|decode overlap|absorb / expand|drafter costs|graphs' "$R/boot-$name-r0.log" | cut -c1-220 | head -12
