#!/usr/bin/env bash
# W6 load B (pool 1,048,576, CONTEXT 1,048,576): one ~400k prompt with a needle, then the 4 x 300k pool-pressure stress
cd "$(dirname "$0")/../.."
B=http://127.0.0.1:8000; M=GLM-5.3-Flash-EXL3; R=results/W6
python3 $R/needle.py $R/needle-400k.json 400000 0.4 > $R/needle-400k.log 2>&1
curl -s $B/health > $R/health-B1.json
python3 bench/multiturn.py --base $B --model $M --modes stress --stress-target 300000 --stress-step 100000 --stress-final 32000 \
  --long-tokens 512 --mem-hosts local,$WORKER_SSH --mem-log $R/stress-mem.log --out $R/stress.json > $R/stress.log 2>&1
curl -s $B/health > $R/health-B2.json
docker logs glm53-tf-r0 2>&1 | grep -iE 'KV pool|null page' > $R/kvpool-r0.log
echo DONE $(date +%T) >> $R/runB.done
