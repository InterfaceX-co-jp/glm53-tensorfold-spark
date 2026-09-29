#!/usr/bin/env bash
# W17 candidate window (window-start.sh already ran; GPU tests done): run-window.sh B9|B9b|S
#   B9  = config/prod.env + IMAGE b9 (0550 defaults) + GLM53_TF_MULTI_PREFILL=1: PRE = mpf.py group (grouped == alone on
#         the real model), c4.py thinking low and off (C2 / C4 first tokens); then W15's gates.sh sequence (ab.sh set,
#         ab.py again, N1, 4 x 250k stress, MMLU-200, exact / batchexact again, needle ~314k alone; MemAvailable every 2 s
#         with phase marks, meminfo breakdown beside it)
#   B9b = the same config restarted: mpf.py alone (the grouped cold replies vs cold alone on a fresh server), replay
#         check (token-id prompts, n - 64), c4.py again
#   S   = on the running server: s900.py (~900k needle beside 3 busy slots) with MemAvailable sampling
# A failed load restores prod at once.
cd $HOME/glm53-tensorfold-spark
R=results/W17
CAND="GLM53_TF_MULTI_PREFILL=1"
step() { echo "=== $* ($(date +%T))"; }
oom() { echo "$(dmesg -T 2>/dev/null | grep -ciE 'out of memory|oom-kill') $(ssh -o BatchMode=yes $WORKER_SSH "dmesg -T | grep -ciE 'out of memory|oom-kill'")"; }
case $1 in
C7)
  # control: production b7 (config/prod.env unchanged) through B9's exact sequence (same PRE, same gates)
  step C7
  bash $R/meminfo.sh > $R/meminfo-C7.log 2>&1 & echo $! > $R/meminfo.pid
  PRE="python3 $R/mpf.py group $R/mpf-C7.json W17c 10 > $R/mpf-C7.log 2>&1; tail -3 $R/mpf-C7.log | cut -c1-300; \
python3 $R/c4.py $R/c4-C7.json 3 > $R/c4-C7.log 2>&1; grep SUMMARY $R/c4-C7.log; \
C4_THINK=0 python3 $R/c4.py $R/c4-C7-nothink.json 3 > $R/c4-C7-nothink.log 2>&1; grep SUMMARY $R/c4-C7-nothink.log" \
    bash $R/gates.sh C7 IMAGE=glm53-tensorfold:b7 || { echo "C7 FAILED"; kill $(cat $R/meminfo.pid); bash $R/restore.sh w17-fail; exit 1; }
  kill $(cat $R/meminfo.pid) 2>/dev/null
  docker logs glm53-tf-r0 > $R/run-C7-r0.log 2>&1 ;;
B9)
  echo "$(oom)" > $R/oom-before.txt
  step B9
  bash $R/meminfo.sh > $R/meminfo-B9.log 2>&1 & echo $! > $R/meminfo.pid
  PRE="python3 $R/mpf.py group $R/mpf-B9.json W17a 10 > $R/mpf-B9.log 2>&1; tail -30 $R/mpf-B9.log | cut -c1-300; \
python3 $R/c4.py $R/c4-B9.json 3 > $R/c4-B9.log 2>&1; grep SUMMARY $R/c4-B9.log; \
C4_THINK=0 python3 $R/c4.py $R/c4-B9-nothink.json 3 > $R/c4-B9-nothink.log 2>&1; grep SUMMARY $R/c4-B9-nothink.log" \
    bash $R/gates.sh B9 $CAND || { echo "B9 FAILED"; kill $(cat $R/meminfo.pid); bash $R/restore.sh w17-fail; exit 1; }
  kill $(cat $R/meminfo.pid) 2>/dev/null
  echo "oom lines after B9: $(oom) (before: $(cat $R/oom-before.txt))"
  docker logs glm53-tf-r0 > $R/run-B9-r0.log 2>&1; ssh -o BatchMode=yes $WORKER_SSH docker logs glm53-tf-r1 > $R/run-B9-r1.log 2>&1
  grep -ciE 'traceback|error' $R/run-B9-r0.log | sed 's/^/r0 error lines: /'
  grep -iE 'admission waits|trim|scratch' $R/run-B9-r0.log | tail -5 | cut -c1-240 ;;
B9b)
  step B9b
  bash $R/load.sh B9b $CAND || { echo "B9b LOAD FAILED"; bash $R/restore.sh w17-fail; exit 1; }
  python3 $R/mpf.py alone $R/mpf-alone-B9b.json $R/mpf-B9.json > $R/mpf-alone-B9b.log 2>&1; tail -2 $R/mpf-alone-B9b.log
  python3 $R/replay.py tokens $R/replay-B9b.json 8192,32768,65536 2 > $R/replay-B9b.log 2>&1; grep SUMMARY $R/replay-B9b.log | cut -c1-260
  python3 $R/c4.py $R/c4-B9b.json 3 > $R/c4-B9b.log 2>&1; grep SUMMARY $R/c4-B9b.log
  # B9's first ab.py after the PRE bursts was slow (1,454 / 1,550): the same order again, then once more
  for k in 1 2; do python3 results/W5/ab.py $R/ab-B9b-$k.json 24500,98000 '{"prod":{}}' > $R/ab-B9b-$k.log 2>&1; cut -c1-110 $R/ab-B9b-$k.log | tail -2; done
  docker logs glm53-tf-r0 2>&1 | grep -iE 'trim|admission wait' | tail -5 | cut -c1-240
  curl -s http://127.0.0.1:8000/health; echo ;;
S)
  step S
  bash $R/meminfo.sh > $R/meminfo-S.log 2>&1 & mp=$!
  timeout 5400 python3 $R/s900.py $R/s900.json 1000000 3 > $R/s900.log 2>&1; echo "s900 rc=$?"; cat $R/s900.log | cut -c1-500
  kill $mp
  awk -F'|' 'NR>1{split($2,a," "); split($3,b," "); if(x==""||a[1]<x)x=a[1]; if(y==""||b[1]<y)y=b[1]} END{print "S MemAvailable min: " x " / " y " GiB"}' $R/meminfo-S.log
  echo "oom lines: $(oom) (before: $(cat $R/oom-before.txt))"
  docker logs --since 90m glm53-tf-r0 2>&1 | grep -iE 'traceback|error|admission waits|trim' | tail -8 | cut -c1-240 ;;
esac
step "$1 done"
