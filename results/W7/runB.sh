#!/usr/bin/env bash
# W7 load B: piece sweep for one active request (PIECE / PREFILL_ROWS_MAX from load.sh), per-request prefill_rows
cd $HOME/glm53-tensorfold-spark
R=results/W7; tag=$1; sets=$2
echo "warm-up $(date +%T)"
python3 -c "import json,sys;[print(json.dumps(v)) for v in json.loads(sys.argv[1]).values()]" "$sets" | while read -r k; do
  python3 $R/req.py prefill $R/warm-$tag.jsonl 12000 8 "$k" 2>&1 | tail -1
done
python3 results/W5/ab.py $R/ab-$tag.json 24500,98000 "$sets" 256 > $R/ab-$tag.log 2>&1
echo "ab done $(date +%T)"; cat $R/ab-$tag.log
