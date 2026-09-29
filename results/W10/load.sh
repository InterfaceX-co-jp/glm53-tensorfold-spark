#!/usr/bin/env bash
# W10: start a test load from config/prod.env with overrides: load.sh NAME [K=V ...]; IMAGE defaults to b4
cd $HOME/glm53-tensorfold-spark
R=results/W10; name=$1; shift
for kv in "$@"; do export "$kv"; done
export IMAGE=${IMAGE:-glm53-tensorfold:b4}
echo "load $name: IMAGE=$IMAGE $* ($(date +%T))" | tee -a $R/loads.log
CONFIG=config/prod.env scripts/serve.sh stop >/dev/null 2>&1
CONFIG=config/prod.env timeout 1200 scripts/serve.sh start > "$R/start-$name.log" 2>&1
echo "start rc=$? $(date +%T)" | tee -a $R/loads.log; tail -3 "$R/start-$name.log"
docker logs glm53-tf-r0 > "$R/boot-$name-r0.log" 2>&1
ssh -o BatchMode=yes $WORKER_SSH docker logs glm53-tf-r1 > "$R/boot-$name-r1.log" 2>&1
grep -iE 'roce conn|b12x|patches/03[7-9]0|patches/04[01]0|kda v2|sparse latent attention v2|absorb / expand|decode overlap|cpu pin|drafter costs|lean' "$R/boot-$name-r0.log" | cut -c1-220 | head -14
grep -iE 'patches/03[7-9]0|patches/04[01]0|kda v2|absorb / expand|error|Traceback' "$R/boot-$name-r1.log" | cut -c1-220 | head -6
