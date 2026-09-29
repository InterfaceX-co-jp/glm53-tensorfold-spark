#!/usr/bin/env bash
# W9 0360 load (GLM53_TF_B12X=4 default): exact, batchexact, sessions + batch with b12x 4, prefill A/B b0 vs b4 x3, decode
cd $HOME/glm53-tensorfold-spark
B=http://127.0.0.1:8000; M=GLM-5.3-Flash-EXL3; R=results/W9; tag=${1:-B}
echo "run $tag start $(date +%T)"
S='{"b0":{"b12x":0},"b4":{"b12x":4}}'
python3 results/W7/req.py prefill $R/warm-$tag.jsonl 12000 8 '{"b12x":4}' > /dev/null 2>&1
python3 results/W7/req.py prefill $R/warm-$tag.jsonl 9000 8 '{"b12x":0}' > /dev/null 2>&1
python3 bench/glmbench.py --base $B --model $M --suites exact --out $R/exact-$tag.json > $R/exact-$tag.log 2>&1
python3 bench/multiturn.py --base $B --model $M --modes batchexact --out $R/batchexact-$tag.json > $R/batchexact-$tag.log 2>&1
echo "exact: $(tail -1 $R/exact-$tag.log | head -c 300)"; grep -iE 'identical|batchexact|[0-9]+/[0-9]+' $R/batchexact-$tag.log | tail -2
python3 $R/b12x_sessions.py $R/b12x-sessions-$tag.json 4 > $R/b12x-sessions-$tag.log 2>&1; tail -3 $R/b12x-sessions-$tag.log
for i in 1 2 3; do python3 results/W5/ab.py $R/ab-$tag-$i.json 24500,98000 "$S" > $R/ab-$tag-$i.log 2>&1; cut -c1-110,200- $R/ab-$tag-$i.log; done
python3 bench/multiturn.py --base $B --model $M --modes concurrent --streams 1,4 --reps 3 --long-tokens 512 --out $R/conc-$tag.json > $R/conc-$tag.log 2>&1
grep -E 'streams rep' $R/conc-$tag.log | cut -c1-70
curl -s $B/health > $R/health-$tag.json
echo "run $tag done $(date +%T)"
