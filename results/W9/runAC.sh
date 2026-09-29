#!/usr/bin/env bash
# W9 per-load set for the RoCE A/B (A = nccl, C = roce) + item 5 on A: runAC.sh TAG [perslot]
cd $HOME/glm53-tensorfold-spark
B=http://127.0.0.1:8000; M=GLM-5.3-Flash-EXL3; R=results/W9; tag=$1
echo "run $tag start $(date +%T)"
python3 results/W7/req.py prefill $R/warm-$tag.jsonl 12000 8 '{}' > /dev/null 2>&1
python3 $R/transcripts.py $R/transcripts-$tag.json > $R/transcripts-$tag.log 2>&1; cat $R/transcripts-$tag.log
python3 bench/glmbench.py --base $B --model $M --suites exact --out $R/exact-$tag.json > $R/exact-$tag.log 2>&1
python3 bench/multiturn.py --base $B --model $M --modes batchexact --out $R/batchexact-$tag.json > $R/batchexact-$tag.log 2>&1
echo "exact: $(tail -1 $R/exact-$tag.log | head -c 300)"; grep -iE 'identical|batchexact|[0-9]+/[0-9]+' $R/batchexact-$tag.log | tail -2
python3 bench/multiturn.py --base $B --model $M --modes concurrent --streams 1,4 --reps 5 --long-tokens 512 --out $R/conc-$tag.json > $R/conc-$tag.log 2>&1
grep -E 'streams rep' $R/conc-$tag.log | cut -c1-70
if [[ "${2:-}" == perslot ]]; then
  python3 bench/multiturn.py --base $B --model $M --modes concurrent --streams 1,2,4 --reps 3 --long-tokens 512 --extra '{"draft": false}' --out $R/serial-$tag.json > $R/serial-$tag.log 2>&1
  grep -E 'streams rep' $R/serial-$tag.log | cut -c1-200
fi
curl -s $B/metrics > $R/metrics-$tag.txt; grep -iE 'roce|fallback|comm' $R/metrics-$tag.txt | head
curl -s $B/health > $R/health-$tag.json
echo "run $tag done $(date +%T)"
