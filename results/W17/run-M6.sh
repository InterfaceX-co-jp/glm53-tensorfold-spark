#!/usr/bin/env bash
# W17 isolation: 0560 alone = b9 + GLM53_TF_MULTI_PREFILL=1 with 0550 off (ADMIT_MEM=free, SELECT_SCRATCH=off,
# ALLOC_TRIM_GB=0), through B9 / C7 exact sequence (same PRE, gates.sh)
cd $HOME/glm53-tensorfold-spark
R=results/W17
echo "=== M6 ($(date +%T))"
bash $R/meminfo.sh > $R/meminfo-M6.log 2>&1 & mp=$!
PRE="python3 $R/mpf.py group $R/mpf-M6.json W17m 10 > $R/mpf-M6.log 2>&1; tail -3 $R/mpf-M6.log | cut -c1-300; \
python3 $R/c4.py $R/c4-M6.json 3 > $R/c4-M6.log 2>&1; grep SUMMARY $R/c4-M6.log; \
C4_THINK=0 python3 $R/c4.py $R/c4-M6-nothink.json 3 > $R/c4-M6-nothink.log 2>&1; grep SUMMARY $R/c4-M6-nothink.log" \
  bash $R/gates.sh M6 GLM53_TF_MULTI_PREFILL=1 GLM53_TF_ADMIT_MEM=free GLM53_TF_SELECT_SCRATCH=off GLM53_TF_ALLOC_TRIM_GB=0 || { echo "M6 FAILED"; kill $mp; exit 1; }
kill $mp
echo "nvrm lines worker: $(ssh -o BatchMode=yes $WORKER_SSH "dmesg -T | grep -ciE \"out of memory\"")"
echo "=== M6 done ($(date +%T))"
