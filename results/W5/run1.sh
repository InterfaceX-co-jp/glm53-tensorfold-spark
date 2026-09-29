#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
B=http://127.0.0.1:8000; M=GLM-5.3-Flash-EXL3; O=results/W5
python3 $O/ab.py $O/ab1.json 28000,112000,28000 '{"fat":{"fat_experts":1},"auto":{"fat_experts":2},"fast2":{"fat_experts":0}}' 256 > $O/ab1.log 2>&1
python3 bench/glmbench.py --base $B --model $M --suites tf,exact --out $O/tf-exact-auto.json > $O/tf-exact-auto.log 2>&1
python3 bench/multiturn.py --base $B --model $M --modes batchexact,concurrent --streams 4 --reps 3 --long-tokens 512 --out $O/conc-off.json > $O/conc-off.log 2>&1
python3 bench/quality.py --base $B --model $M --out $O/mmlu-auto.json > $O/mmlu-auto.log 2>&1
echo DONE $(date +%T) >> $O/ab1.log
