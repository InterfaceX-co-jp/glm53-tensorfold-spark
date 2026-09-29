#!/usr/bin/env bash
# W14: start a test load from config/prod.env with overrides: load.sh NAME [K=V ...]; IMAGE defaults to b6
cd $HOME/glm53-tensorfold-spark
R=results/W14; name=$1; shift
for kv in "$@"; do export "$kv"; done
export IMAGE=${IMAGE:-glm53-tensorfold:b6}
echo "load $name: IMAGE=$IMAGE $* ($(date +%T))" | tee -a $R/loads.log
CONFIG=config/prod.env scripts/serve.sh stop >/dev/null 2>&1
CONFIG=config/prod.env timeout 1800 scripts/serve.sh start > "$R/start-$name.log" 2>&1
rc=$?
echo "start rc=$rc $(date +%T)" | tee -a $R/loads.log; tail -3 "$R/start-$name.log"
docker logs glm53-tf-r0 > "$R/boot-$name-r0.log" 2>&1
ssh -o BatchMode=yes $WORKER_SSH docker logs glm53-tf-r1 > "$R/boot-$name-r1.log" 2>&1
grep -iE 'roce conn|roce:|context:|vision|l2pf|drafter costs|batching|session store' "$R/boot-$name-r0.log" | cut -c1-240 | head -16
grep -iE 'vision|l2pf|error|Traceback' "$R/boot-$name-r1.log" | cut -c1-240 | head -6
exit $rc
