#!/usr/bin/env bash
# W9: light traffic during the first hour on RoCE: every 5 min one concurrent rep at 1 and 4 streams (256 tokens) and
# the exact suite once at the start and the end
cd $HOME/glm53-tensorfold-spark
B=http://127.0.0.1:8000; M=GLM-5.3-Flash-EXL3; R=results/W9
python3 bench/glmbench.py --base $B --model $M --suites exact --out $R/exact-prod1.json > $R/exact-prod1.log 2>&1
echo "$(date +%T) exact $(tail -1 $R/exact-prod1.log | grep -o true | wc -l)/10"
for i in $(seq 1 11); do
  python3 bench/multiturn.py --base $B --model $M --modes concurrent --streams 1,4 --reps 1 --long-tokens 256 > $R/prod-traffic-$i.log 2>&1
  echo "$(date +%T) $(grep -E 'streams rep' $R/prod-traffic-$i.log | cut -c1-45 | tr '\n' ' ')"
  sleep 270
done
python3 bench/glmbench.py --base $B --model $M --suites exact --out $R/exact-prod2.json > $R/exact-prod2.log 2>&1
echo "$(date +%T) exact $(tail -1 $R/exact-prod2.log | grep -o true | wc -l)/10"
