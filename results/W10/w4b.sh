#!/usr/bin/env bash
# W10 window 4b: the needle-after-stress memory baseline on the unchanged production config (H2 = b2, W9 knobs):
# 4 x 250k stress then the ~314k needle alone, MemAvailable sampled (FIN's needle dipped to 7.39 GiB on the worker node)
cd $HOME/glm53-tensorfold-spark
B=http://127.0.0.1:8000; M=GLM-5.3-Flash-EXL3; R=results/W10
bash $R/load.sh H2 IMAGE=glm53-tensorfold:b2
python3 bench/multiturn.py --base $B --model $M --modes stress --stress-target 250000 --stress-step 100000 --stress-final 32000 \
  --long-tokens 512 --mem-hosts local,$WORKER_SSH --mem-log $R/stress-mem-H2.log --out $R/stress-H2.json > $R/stress-H2.log 2>&1
tail -1 $R/stress-H2.log
bash $R/mem.sh > $R/mem-H2-needle.log 2>&1 & mp=$!
python3 results/W6/needle.py $R/needle-H2.json 350000 0.4 > $R/needle-H2.log 2>&1
kill $mp; cut -c1-200 $R/needle-H2.log | tail -3
echo "needle min the head node $(sort -k2 -n $R/mem-H2-needle.log | head -1 | cut -d' ' -f2) the worker node $(sort -k3 -n $R/mem-H2-needle.log | head -1 | cut -d' ' -f3)"
echo "w4b done $(date +%T)"
