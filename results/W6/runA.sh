#!/usr/bin/env bash
# W6 load A (pool 1,048,576, CONTEXT 262144): same bits, then speed
cd "$(dirname "$0")/../.."
B=http://127.0.0.1:8000; M=GLM-5.3-Flash-EXL3; R=results/W6
python3 bench/glmbench.py --base $B --model $M --suites exact --out $R/exact.json > $R/exact.log 2>&1
python3 bench/multiturn.py --base $B --model $M --modes batchexact --out $R/batchexact.json > $R/batchexact.log 2>&1
python3 results/W5/ab.py $R/ab-poolA.json 24500,98000 '{"pool":{}}' > $R/ab-poolA.log 2>&1
python3 results/W5/ab.py $R/ab-poolA2.json 24500,98000 '{"pool":{}}' > $R/ab-poolA2.log 2>&1
python3 bench/multiturn.py --base $B --model $M --modes concurrent --streams 1,4 --reps 3 --long-tokens 512 --out $R/conc-A.json > $R/conc-A.log 2>&1
curl -s $B/health > $R/health-A.json
echo DONE $(date +%T) >> $R/runA.done
