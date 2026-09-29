#!/usr/bin/env bash
# W9 window 3: combined gates on the adoption candidate F = prod config + the overrides given (w3.sh K=V ...):
# exact 10/10, batchexact 4/4, ab.py 24.5k / 98k x2 (reply sha 8794a3463259cc2f, prefill), decode 1 / 4 streams x5,
# 4 x 250k stress (MemAvailable >= 8 GiB both nodes), MMLU-200 >= 87%, exact / batchexact again, OOM count.
cd $HOME/glm53-tensorfold-spark
B=http://127.0.0.1:8000; M=GLM-5.3-Flash-EXL3; R=results/W9
rmmark() { for h in local $WORKER_SSH; do c="docker run --rm -v glm53-tf-cache:/cache --entrypoint bash glm53-tensorfold:b2 -c 'cat /cache/roce-failed 2>/dev/null; rm -f /cache/roce-failed'"; if [ $h = local ]; then eval "$c"; else ssh -o BatchMode=yes $h "$c"; fi; done; }
rmmark > $R/roce-marks-before-F.txt 2>&1
bash $R/mem.sh > $R/mem-w3.log 2>&1 & echo $! > $R/mem.pid
bash $R/load.sh F "$@"
echo "F start $(date +%T)"
python3 results/W7/req.py prefill $R/warm-F.jsonl 12000 8 '{}' > /dev/null 2>&1
python3 bench/glmbench.py --base $B --model $M --suites exact --out $R/exact-F.json > $R/exact-F.log 2>&1
python3 bench/multiturn.py --base $B --model $M --modes batchexact --out $R/batchexact-F.json > $R/batchexact-F.log 2>&1
echo "exact: $(tail -1 $R/exact-F.log | grep -o true | wc -l)/10"; grep -E 'batched == alone' $R/batchexact-F.log
for i in 1 2; do python3 results/W5/ab.py $R/ab-F-$i.json 24500,98000 '{"prod":{}}' > $R/ab-F-$i.log 2>&1; cut -c1-110,200- $R/ab-F-$i.log; done
python3 bench/multiturn.py --base $B --model $M --modes concurrent --streams 1,4 --reps 5 --long-tokens 512 --out $R/conc-F.json > $R/conc-F.log 2>&1
grep -E 'streams rep' $R/conc-F.log | cut -c1-60
python3 bench/multiturn.py --base $B --model $M --modes stress --stress-target 250000 --stress-step 100000 --stress-final 32000 \
  --long-tokens 512 --mem-hosts local,$WORKER_SSH --mem-log $R/stress-mem-F.log --out $R/stress-F.json > $R/stress-F.log 2>&1
tail -3 $R/stress-F.log; curl -s $B/health > $R/health-F-stress.json
python3 bench/quality.py --base $B --model $M --label F --out $R/quality-F.json > $R/quality-F.log 2>&1
tail -3 $R/quality-F.log
python3 bench/glmbench.py --base $B --model $M --suites exact --out $R/exact-F2.json > $R/exact-F2.log 2>&1
python3 bench/multiturn.py --base $B --model $M --modes batchexact --out $R/batchexact-F2.json > $R/batchexact-F2.log 2>&1
echo "exact again: $(tail -1 $R/exact-F2.log | grep -o true | wc -l)/10"; grep -E 'batched == alone' $R/batchexact-F2.log
curl -s $B/health > $R/health-F.json; docker logs glm53-tf-r0 2>&1 | grep -iE "roce.*(timed out|failed|fallback)" | head -3
for h in local $WORKER_SSH; do if [[ $h == local ]]; then dmesg -T 2>/dev/null | grep -ciE 'out of memory|oom-kill'; else ssh -o BatchMode=yes $h "dmesg -T | grep -ciE 'out of memory|oom-kill'"; fi; done > $R/oom-count-F.txt 2>&1
kill $(cat $R/mem.pid) 2>/dev/null; rm -f $R/mem.pid
echo "F done $(date +%T)"
