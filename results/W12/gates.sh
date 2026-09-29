#!/usr/bin/env bash
# W12: full gates on a candidate: gates.sh NAME [K=V ...] = load, ab.sh set (exact 10/10, batchexact 4/4, transcripts,
# reply sha, prefill once, glmbench 1 stream, 4 streams x6, slots), ab.py 24.5k / 98k again (prefill), N1 long exactness,
# 4 x 250k stress (MemAvailable >= 8 GiB both nodes), MMLU-200 >= 87%, exact / batchexact again, needle ~314k alone,
# OOM counts, /health.
cd $HOME/glm53-tensorfold-spark
B=http://127.0.0.1:8000; M=GLM-5.3-Flash-EXL3; R=results/W12
N=$1; shift
rmmark() { for h in local $WORKER_SSH; do c="docker run --rm -v glm53-tf-cache:/cache --entrypoint bash glm53-tensorfold:b5 -c 'cat /cache/roce-failed 2>/dev/null; rm -f /cache/roce-failed'"; if [ $h = local ]; then eval "$c"; else ssh -o BatchMode=yes $h "$c"; fi; done; }
rmmark > $R/roce-marks-before-$N.txt 2>&1
bash $R/load.sh $N "$@" || { echo "LOAD $N FAILED"; exit 1; }
bash $R/mem.sh > $R/mem-$N.log 2>&1 & echo $! > $R/mem.pid
bash $R/ab.sh $N
python3 results/W5/ab.py $R/ab-$N-2.json 24500,98000 '{"prod":{}}' > $R/ab-$N-2.log 2>&1; cut -c1-110,200- $R/ab-$N-2.log | tail -2
python3 bench/longexact.py --corpus $R/n1-corpus.txt --out $R/n1-$N.json > $R/n1-$N.log 2>&1; grep '^N1' $R/n1-$N.log
python3 bench/multiturn.py --base $B --model $M --modes stress --stress-target 250000 --stress-step 100000 --stress-final 32000 \
  --long-tokens 512 --mem-hosts local,$WORKER_SSH --mem-log $R/stress-mem-$N.log --out $R/stress-$N.json > $R/stress-$N.log 2>&1
tail -3 $R/stress-$N.log; curl -s $B/health > $R/health-$N-stress.json
python3 bench/quality.py --base $B --model $M --label $N --out $R/quality-$N.json > $R/quality-$N.log 2>&1
tail -3 $R/quality-$N.log
python3 bench/glmbench.py --base $B --model $M --suites exact --out $R/exact-${N}2.json > $R/exact-${N}2.log 2>&1
python3 bench/multiturn.py --base $B --model $M --modes batchexact --out $R/batchexact-${N}2.json > $R/batchexact-${N}2.log 2>&1
echo "exact again: $(tail -1 $R/exact-${N}2.log | grep -o true | wc -l)/10"; grep -E 'batched == alone' $R/batchexact-${N}2.log
python3 results/W6/needle.py $R/needle-$N.json 350000 0.4 > $R/needle-$N.log 2>&1
cut -c1-300 $R/needle-$N.log | tail -4
curl -s $B/health > $R/health-$N-end.json; cat $R/health-$N-end.json; echo
docker logs glm53-tf-r0 2>&1 | grep -iE "roce.*(timed out|failed|fallback)" | head -3
for h in local $WORKER_SSH; do if [[ $h == local ]]; then dmesg -T 2>/dev/null | grep -ciE 'out of memory|oom-kill'; else ssh -o BatchMode=yes $h "dmesg -T | grep -ciE 'out of memory|oom-kill'"; fi; done > $R/oom-count-$N.txt 2>&1
cat $R/oom-count-$N.txt | tr '\n' ' '; echo "(oom lines before: $(cat $R/oom-before.txt 2>/dev/null | tr '\n' ' '))"
kill $(cat $R/mem.pid) 2>/dev/null; rm -f $R/mem.pid
awk 'NR>1{if(a==""||$2<a)a=$2; if(b==""||$3<b)b=$3} END{print "MemAvailable min over the gates: " a " / " b " GiB"}' $R/mem-$N.log
echo "$N gates done $(date +%T)"
