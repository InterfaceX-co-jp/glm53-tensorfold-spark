#!/usr/bin/env bash
# W9 window 3: combined gates on the adoption candidate G = prod config + the overrides given (w3.sh K=V ...):
# exact 10/10, batchexact 4/4, ab.py 24.5k / 98k x2 (reply sha 8794a3463259cc2f, prefill), decode 1 / 4 streams x5,
# 4 x 250k stress (MemAvailable >= 8 GiB both nodes), MMLU-200 >= 87%, exact / batchexact again, OOM count.
cd $HOME/glm53-tensorfold-spark
B=http://127.0.0.1:8000; M=GLM-5.3-Flash-EXL3; R=results/W9
rmmark() { for h in local $WORKER_SSH; do c="docker run --rm -v glm53-tf-cache:/cache --entrypoint bash glm53-tensorfold:b2 -c 'cat /cache/roce-failed 2>/dev/null; rm -f /cache/roce-failed'"; if [ $h = local ]; then eval "$c"; else ssh -o BatchMode=yes $h "$c"; fi; done; }
rmmark > $R/roce-marks-before-G.txt 2>&1
bash $R/mem.sh > $R/mem-w3g.log 2>&1 & echo $! > $R/mem.pid
bash $R/load.sh G "$@"
echo "G start $(date +%T)"
python3 results/W7/req.py prefill $R/warm-G.jsonl 12000 8 '{}' > /dev/null 2>&1
python3 bench/glmbench.py --base $B --model $M --suites exact --out $R/exact-G.json > $R/exact-G.log 2>&1
python3 bench/multiturn.py --base $B --model $M --modes batchexact --out $R/batchexact-G.json > $R/batchexact-G.log 2>&1
echo "exact: $(tail -1 $R/exact-G.log | grep -o true | wc -l)/10"; grep -E 'batched == alone' $R/batchexact-G.log
for i in 1 2; do python3 results/W5/ab.py $R/ab-G-$i.json 24500,98000 '{"prod":{}}' > $R/ab-G-$i.log 2>&1; cut -c1-110,200- $R/ab-G-$i.log; done
python3 bench/multiturn.py --base $B --model $M --modes concurrent --streams 1,4 --reps 5 --long-tokens 512 --out $R/conc-G.json > $R/conc-G.log 2>&1
grep -E 'streams rep' $R/conc-G.log | cut -c1-60
python3 bench/multiturn.py --base $B --model $M --modes stress --stress-target 250000 --stress-step 100000 --stress-final 32000 \
  --long-tokens 512 --mem-hosts local,$WORKER_SSH --mem-log $R/stress-mem-G.log --out $R/stress-G.json > $R/stress-G.log 2>&1
tail -3 $R/stress-G.log; curl -s $B/health > $R/health-G-stress.json
python3 bench/quality.py --base $B --model $M --label G --out $R/quality-G.json > $R/quality-G.log 2>&1
tail -3 $R/quality-G.log
python3 bench/glmbench.py --base $B --model $M --suites exact --out $R/exact-G2.json > $R/exact-G2.log 2>&1
python3 bench/multiturn.py --base $B --model $M --modes batchexact --out $R/batchexact-G2.json > $R/batchexact-G2.log 2>&1
echo "exact again: $(tail -1 $R/exact-G2.log | grep -o true | wc -l)/10"; grep -E 'batched == alone' $R/batchexact-G2.log
curl -s $B/health > $R/health-G.json; docker logs glm53-tf-r0 2>&1 | grep -iE "roce.*(timed out|failed|fallback)" | head -3
for h in local $WORKER_SSH; do if [[ $h == local ]]; then dmesg -T 2>/dev/null | grep -ciE 'out of memory|oom-kill'; else ssh -o BatchMode=yes $h "dmesg -T | grep -ciE 'out of memory|oom-kill'"; fi; done > $R/oom-count-G.txt 2>&1
kill $(cat $R/mem.pid) 2>/dev/null; rm -f $R/mem.pid
echo "G done $(date +%T)"
