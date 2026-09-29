#!/usr/bin/env bash
# W6: start a test load from config/prod.env on image kvpool with extra overrides: load.sh NAME [K=V ...]
cd "$(dirname "$0")/../.."
name=$1; shift
for kv in "$@"; do export "$kv"; done
export IMAGE=${IMAGE:-glm53-tensorfold:kvpool}
CONFIG=config/prod.env scripts/serve.sh stop >/dev/null 2>&1
CONFIG=config/prod.env timeout 1200 scripts/serve.sh start > "results/W6/start-$name.log" 2>&1
echo "start rc=$? $(date +%T)"; tail -3 "results/W6/start-$name.log"
docker logs glm53-tf-r0 > "results/W6/boot-$name-r0.log" 2>&1
ssh -o BatchMode=yes $WORKER_SSH docker logs glm53-tf-r1 > "results/W6/boot-$name-r1.log" 2>&1
grep -iE 'KV pool|latent KV cache|batching [0-9]+ requests|every slot' "results/W6/boot-$name-r0.log" | head
