#!/usr/bin/env bash
# run2.sh TAG: concurrency + exactness on the current load
cd "$(dirname "$0")/../.."
B=http://127.0.0.1:8000; M=GLM-5.3-Flash-EXL3; O=results/W5; T=$1
python3 bench/multiturn.py --base $B --model $M --modes batchexact,concurrent --streams 4 --reps 3 --long-tokens 512 --out $O/conc-$T.json > $O/conc-$T.log 2>&1
python3 bench/glmbench.py --base $B --model $M --suites exact --out $O/exact-$T.json > $O/exact-$T.log 2>&1
echo DONE $(date +%T) >> $O/conc-$T.log
