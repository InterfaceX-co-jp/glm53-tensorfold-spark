#!/usr/bin/env bash
# W10 window 4: the combined candidate FIN = prod config + the overrides given (w4.sh K=V ...): gates.sh (exact 10/10,
# batchexact 4/4, ab.py 24.5k / 98k x2 with the reply sha, decode 1 / 4 streams x5, 4 x 250k stress MemAvailable >= 8 GiB
# on both nodes, MMLU-200 >= 87%, exact / batchexact again, OOM count), then a ~314k-token needle alone, the edit cells, the cancel check
cd $HOME/glm53-tensorfold-spark
R=results/W10
bash $R/gates.sh FIN "$@"
bash $R/mem.sh > $R/mem-FIN-needle.log 2>&1 & mp=$!
python3 results/W6/needle.py $R/needle-FIN.json 350000 0.4 > $R/needle-FIN.log 2>&1
kill $mp; cut -c1-300 $R/needle-FIN.log | tail -5
for h in local $WORKER_SSH; do if [[ $h == local ]]; then dmesg -T 2>/dev/null | grep -ciE 'out of memory|oom-kill'; else ssh -o BatchMode=yes $h "dmesg -T | grep -ciE 'out of memory|oom-kill'"; fi; done > $R/oom-count-FIN-needle.txt 2>&1
curl -s http://127.0.0.1:8000/health > $R/health-FIN-needle.json
python3 bench/glmbench.py --base http://127.0.0.1:8000 --model GLM-5.3-Flash-EXL3 --suites edit --label FIN --out $R/glmbench-FIN.json > $R/glmbench-FIN.log 2>&1
grep -E "^  edit" $R/glmbench-FIN.log
python3 $R/cancel.py $R/cancel-FIN.json 2>&1 | tail -1
curl -s http://127.0.0.1:8000/health > $R/health-FIN-end.json; cat $R/health-FIN-end.json; echo
echo "w4 done $(date +%T)"
