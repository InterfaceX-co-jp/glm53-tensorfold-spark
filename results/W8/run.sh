#!/usr/bin/env bash
# W8: the per-load gate set: run.sh TAG 'KNOBSETS-JSON' [skip-exact]
#   exact 10/10, batchexact 4/4, ab.py (24.5k / 98k, two runs, reply sha), concurrent 1 / 4 streams x 3 reps
cd $HOME/glm53-tensorfold-spark
B=http://127.0.0.1:8000; M=GLM-5.3-Flash-EXL3; R=results/W8; tag=$1; sets=$2
echo "run $tag start $(date +%T)"
python3 results/W7/req.py prefill $R/warm-$tag.jsonl 12000 8 '{}' > /dev/null 2>&1   # warm-up (Triton / extension builds)
python3 -c "import json,sys;[print(json.dumps(v)) for v in json.loads(sys.argv[1]).values()]" "$sets" | while read -r k; do
  python3 results/W7/req.py prefill $R/warm-$tag.jsonl 9000 8 "$k" > /dev/null 2>&1
done
if [[ "${3:-}" != skip-exact ]]; then
  python3 bench/glmbench.py --base $B --model $M --suites exact --out $R/exact-$tag.json > $R/exact-$tag.log 2>&1
  python3 bench/multiturn.py --base $B --model $M --modes batchexact --out $R/batchexact-$tag.json > $R/batchexact-$tag.log 2>&1
  echo "exact: $(tail -1 $R/exact-$tag.log | head -c 400)"; grep -iE 'identical|batchexact|[0-9]+/[0-9]+' $R/batchexact-$tag.log | tail -2
fi
python3 results/W5/ab.py $R/ab-$tag-1.json 24500,98000 "$sets" > $R/ab-$tag-1.log 2>&1
python3 results/W5/ab.py $R/ab-$tag-2.json 24500,98000 "$sets" > $R/ab-$tag-2.log 2>&1
cat $R/ab-$tag-1.log $R/ab-$tag-2.log | cut -c1-110,200-
python3 bench/multiturn.py --base $B --model $M --modes concurrent --streams 1,4 --reps 3 --long-tokens 512 --out $R/conc-$tag.json > $R/conc-$tag.log 2>&1
grep -E 'streams rep' $R/conc-$tag.log | cut -c1-60
curl -s $B/health > $R/health-$tag.json
echo "run $tag done $(date +%T)"
