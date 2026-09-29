#!/usr/bin/env bash
# W11 graph-policy set a load: graphs.sh TAG = warm-up, batchexact, concurrent 4 streams x3 (512 tokens, W10's prompts),
# then concurrent 4 streams x3 again (the same load, graph keys warm) and /health
cd $HOME/glm53-tensorfold-spark
B=http://127.0.0.1:8000; M=GLM-5.3-Flash-EXL3; R=results/W11; tag=$1
echo "graphs $tag start $(date +%T)"
python3 results/W7/req.py prefill $R/warm-$tag.jsonl 12000 8 '{}' > /dev/null 2>&1
python3 bench/multiturn.py --base $B --model $M --modes batchexact --out $R/batchexact-$tag.json > $R/batchexact-$tag.log 2>&1
grep -E 'batched == alone' $R/batchexact-$tag.log
for pass in a b; do
  python3 bench/multiturn.py --base $B --model $M --modes concurrent --streams 4 --reps 3 --long-tokens 512 --out $R/conc-$tag-$pass.json > $R/conc-$tag-$pass.log 2>&1
  grep -E 'streams rep' $R/conc-$tag-$pass.log | cut -c1-70
done
curl -s $B/health > $R/health-$tag.json; cat $R/health-$tag.json; echo
echo "graphs $tag done $(date +%T)"
