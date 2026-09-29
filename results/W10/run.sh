#!/usr/bin/env bash
# W10 per-load A/B set: run.sh TAG [nosess] = warm-up, exact 10/10, batchexact 4/4, session follow-up resumed == cold
# (1 conversation x 3 turns, ~61k tokens), ab.py 24.5k / 98k x2 (reply sha 8794a3463259cc2f, prefill), concurrent 1 / 4 streams x5
cd $HOME/glm53-tensorfold-spark
B=http://127.0.0.1:8000; M=GLM-5.3-Flash-EXL3; R=results/W10; tag=$1
echo "run $tag start $(date +%T)"
python3 results/W7/req.py prefill $R/warm-$tag.jsonl 12000 8 '{}' > /dev/null 2>&1
python3 bench/glmbench.py --base $B --model $M --suites exact --out $R/exact-$tag.json > $R/exact-$tag.log 2>&1
python3 bench/multiturn.py --base $B --model $M --modes batchexact --out $R/batchexact-$tag.json > $R/batchexact-$tag.log 2>&1
echo "exact: $(tail -1 $R/exact-$tag.log | grep -o true | wc -l)/10"; grep -E 'batched == alone' $R/batchexact-$tag.log
if [[ "${2:-}" != nosess ]]; then NCONV=1 python3 $R/sessions.py $R/sessions-$tag.json > $R/sessions-$tag.log 2>&1; tail -2 $R/sessions-$tag.log; fi
for i in 1 2; do python3 results/W5/ab.py $R/ab-$tag-$i.json 24500,98000 '{"prod":{}}' > $R/ab-$tag-$i.log 2>&1; cut -c1-110,200- $R/ab-$tag-$i.log; done
python3 bench/multiturn.py --base $B --model $M --modes concurrent --streams 1,4 --reps 5 --long-tokens 512 --out $R/conc-$tag.json > $R/conc-$tag.log 2>&1
grep -E 'streams rep' $R/conc-$tag.log | cut -c1-60
curl -s $B/health > $R/health-$tag.json; cat $R/health-$tag.json; echo
echo "run $tag done $(date +%T)"
