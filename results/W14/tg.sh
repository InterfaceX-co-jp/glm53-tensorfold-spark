#!/usr/bin/env bash
# W14 text gates on a load: tg.sh TAG = warm-up, exact 10/10, batchexact 4/4, W9 transcripts (== W9 load A), ab.py 24.5k /
# 98k (reply sha 8794a3463259cc2f), glmbench tf,tweet,kit,edit x1 (reply hashes, compared across loads), /health
cd $HOME/glm53-tensorfold-spark
B=http://127.0.0.1:8000; M=GLM-5.3-Flash-EXL3; R=results/W14; tag=$1
echo "tg $tag start $(date +%T)"
python3 results/W7/req.py prefill $R/warm-$tag.jsonl 12000 8 '{}' > /dev/null 2>&1
python3 bench/glmbench.py --base $B --model $M --suites exact --out $R/exact-$tag.json > $R/exact-$tag.log 2>&1
python3 bench/multiturn.py --base $B --model $M --modes batchexact --out $R/batchexact-$tag.json > $R/batchexact-$tag.log 2>&1
echo "exact: $(tail -1 $R/exact-$tag.log | grep -o true | wc -l)/10"; grep -E 'batched == alone' $R/batchexact-$tag.log
python3 results/W10/transcripts.py $R/transcripts-$tag.json > $R/transcripts-$tag.log 2>&1
python3 - $R/transcripts-$tag.log results/W9/transcripts-A.log <<'PY'
import sys; a, b = [open(f).read().splitlines() for f in sys.argv[1:3]]
print("transcripts alone == W9 A:", bool(a) and a[0] == b[0], "|", a[-1] if a else "NONE")
PY
python3 results/W5/ab.py $R/ab-$tag.json 24500,98000 '{"prod":{}}' > $R/ab-$tag.log 2>&1; cut -c1-110,200- $R/ab-$tag.log | tail -2
python3 bench/glmbench.py --base $B --model $M --suites tf,tweet,kit,edit --reps 1 --long-tokens 512 --label $tag --out $R/glmbench-$tag.json > $R/glmbench-$tag.log 2>&1
grep -E 'median' $R/glmbench-$tag.log | cut -c1-80 | head -3
curl -s $B/health > $R/health-$tag.json; cat $R/health-$tag.json; echo
docker logs glm53-tf-r0 2>&1 | grep -ciE 'traceback|error' | sed 's/^/r0 error lines: /'
echo "tg $tag done $(date +%T)"
