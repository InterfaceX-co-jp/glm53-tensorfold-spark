#!/usr/bin/env bash
# W8: start a test load from config/prod.env with overrides: load.sh NAME [K=V ...]; IMAGE defaults to b2 (0001-0360)
cd $HOME/glm53-tensorfold-spark
name=$1; shift
for kv in "$@"; do export "$kv"; done
export IMAGE=${IMAGE:-glm53-tensorfold:b2}
echo "load $name: IMAGE=$IMAGE $* ($(date +%T))" | tee -a results/W8/loads.log
CONFIG=config/prod.env scripts/serve.sh stop >/dev/null 2>&1
CONFIG=config/prod.env timeout 1200 scripts/serve.sh start > "results/W8/start-$name.log" 2>&1
echo "start rc=$? $(date +%T)" | tee -a results/W8/loads.log; tail -3 "results/W8/start-$name.log"
docker logs glm53-tf-r0 > "results/W8/boot-$name-r0.log" 2>&1
ssh -o BatchMode=yes $WORKER_SSH docker logs glm53-tf-r1 > "results/W8/boot-$name-r1.log" 2>&1
grep -iE 'batching [0-9]+ requests|lean prefill|request log|shared prefixes|row-split|PREFILL_PP|SOLO|solo|expert|tc ' "results/W8/boot-$name-r0.log" | head -20
