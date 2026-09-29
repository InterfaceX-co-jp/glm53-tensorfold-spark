#!/usr/bin/env bash
# W10 0380 per-load set: deep.sh TAG = glmbench tf,tweet,kit,edit x3 (512 tokens), exact, batchexact, concurrent 1/4 x3
cd $HOME/glm53-tensorfold-spark
B=http://127.0.0.1:8000; M=GLM-5.3-Flash-EXL3; R=results/W10; tag=$1
echo "deep $tag start $(date +%T)"
python3 results/W7/req.py prefill $R/warm-$tag.jsonl 12000 8 '{}' > /dev/null 2>&1
grep -E "drafter costs" $R/boot-$tag-r0.log | cut -c1-400
python3 bench/glmbench.py --base $B --model $M --suites tf,tweet,kit,edit --reps 3 --long-tokens 512 --label $tag --out $R/glmbench-$tag.json > $R/glmbench-$tag.log 2>&1
tail -4 $R/glmbench-$tag.log | cut -c1-600
python3 bench/glmbench.py --base $B --model $M --suites exact --out $R/exact-$tag.json > $R/exact-$tag.log 2>&1
python3 bench/multiturn.py --base $B --model $M --modes batchexact --out $R/batchexact-$tag.json > $R/batchexact-$tag.log 2>&1
echo "exact: $(tail -1 $R/exact-$tag.log | grep -o true | wc -l)/10"; grep -E 'batched == alone' $R/batchexact-$tag.log
python3 bench/multiturn.py --base $B --model $M --modes concurrent --streams 1,4 --reps 5 --long-tokens 512 --out $R/conc-$tag.json > $R/conc-$tag.log 2>&1
grep -E 'streams rep' $R/conc-$tag.log | cut -c1-60
curl -s $B/health > $R/health-$tag.json; cat $R/health-$tag.json; echo
echo "deep $tag done $(date +%T)"
