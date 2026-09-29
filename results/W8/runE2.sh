#!/usr/bin/env bash
# W8 load E (combined candidate), worst case + quality: 4 x ~250k stress (pool under pressure), then a ~350k needle
# alone (solo 8,192-row chunks while the pool holds the stress sessions: spills), MMLU-200, exact / batchexact again.
cd $HOME/glm53-tensorfold-spark
B=http://127.0.0.1:8000; M=GLM-5.3-Flash-EXL3; R=results/W8
echo "E2 start $(date +%T)"
python3 bench/multiturn.py --base $B --model $M --modes stress --stress-target 250000 --stress-step 100000 --stress-final 32000 \
  --long-tokens 512 --mem-hosts local,$WORKER_SSH --mem-log $R/stress-mem.log --out $R/stress-E.json > $R/stress-E.log 2>&1
tail -3 $R/stress-E.log; curl -s $B/health > $R/health-E-stress.json
python3 results/W6/needle.py $R/needle-350k.json 350000 0.4 > $R/needle-350k.log 2>&1
cat $R/needle-350k.log | cut -c1-300
python3 bench/quality.py --base $B --model $M --label E --out $R/quality-E.json > $R/quality-E.log 2>&1
tail -3 $R/quality-E.log
python3 bench/glmbench.py --base $B --model $M --suites exact --out $R/exact-E2.json > $R/exact-E2.log 2>&1
python3 bench/multiturn.py --base $B --model $M --modes batchexact --out $R/batchexact-E2.json > $R/batchexact-E2.log 2>&1
echo "exact: $(tail -1 $R/exact-E2.log | grep -o true | wc -l)/10"; grep -E 'batched == alone' $R/batchexact-E2.log
curl -s $B/health > $R/health-E2.json
for h in local $WORKER_SSH; do if [[ $h == local ]]; then dmesg -T 2>/dev/null | grep -ciE 'out of memory|oom-kill'; else ssh -o BatchMode=yes $h "dmesg -T | grep -ciE 'out of memory|oom-kill'"; fi; done > $R/oom-count-E.txt 2>&1
echo "E2 done $(date +%T)"
