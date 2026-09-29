#!/usr/bin/env bash
# W10 decode set (0370): dec.sh TAG [pin] = exact, batchexact, W9 transcripts (== W9 load A's hashes), concurrent 1 / 4
# streams x5; with "pin": thread placement samples on both ranks during a 4-stream run
cd $HOME/glm53-tensorfold-spark
B=http://127.0.0.1:8000; M=GLM-5.3-Flash-EXL3; R=results/W10; tag=$1
echo "dec $tag start $(date +%T)"
python3 results/W7/req.py prefill $R/warm-$tag.jsonl 12000 8 '{}' > /dev/null 2>&1
python3 bench/glmbench.py --base $B --model $M --suites exact --out $R/exact-$tag.json > $R/exact-$tag.log 2>&1
python3 bench/multiturn.py --base $B --model $M --modes batchexact --out $R/batchexact-$tag.json > $R/batchexact-$tag.log 2>&1
echo "exact: $(tail -1 $R/exact-$tag.log | grep -o true | wc -l)/10"; grep -E 'batched == alone' $R/batchexact-$tag.log
python3 $R/transcripts.py $R/transcripts-$tag.json > $R/transcripts-$tag.log 2>&1
python3 - $R/transcripts-$tag.log results/W9/transcripts-A.log <<'PY'
import sys; a, b = [open(f).read().splitlines() for f in sys.argv[1:3]]
print("transcripts alone == W9 A:", a[0] == b[0], "|", a[-1])
PY
if [[ "${2:-}" == pin ]]; then
  ( sleep 20; for i in $(seq 1 40); do
      p0=$(docker inspect -f '{{.State.Pid}}' glm53-tf-r0); ps -L -o tid=,psr=,pcpu=,comm= -p $p0 | awk '$3 > 5' | sed "s/^/$(date +%T.%N | cut -c1-12) r0 /"
      ssh -o BatchMode=yes $WORKER_SSH 'p=$(docker inspect -f "{{.State.Pid}}" glm53-tf-r1); ps -L -o tid=,psr=,pcpu=,comm= -p $p | awk "\$3 > 5"' | sed "s/^/$(date +%T) r1 /"
      sleep 0.5; done ) > $R/pin-$tag.log 2>&1 &
fi
python3 bench/multiturn.py --base $B --model $M --modes concurrent --streams 1,4 --reps 5 --long-tokens 512 --out $R/conc-$tag.json > $R/conc-$tag.log 2>&1
grep -E 'streams rep' $R/conc-$tag.log | cut -c1-60
wait
curl -s $B/health > $R/health-$tag.json; cat $R/health-$tag.json; echo
echo "dec $tag done $(date +%T)"
