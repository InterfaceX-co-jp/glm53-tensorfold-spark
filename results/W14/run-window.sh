#!/usr/bin/env bash
# W14 window, in order (window-start.sh already ran: lease, watchdog timer stopped, prod stopped):
#   GPU tests (head) + engine regressions (worker) -> V0 (b6, vision off): text gates -> V1 (b6, GLM53_TF_VISION=1):
#   text gates, image checks, cache checks, TTFT, 4 x 250k stress + a post-stress image + needle with MemAvailable
#   sampled -> V1R (restart, same config): NVMe restore of the 3-turn conversation -> V1F (vision on, another session
#   compat dir: MAX_IMAGES=7): B fresh -> restore.sh (prod b5 from config/prod.env, verified, watchdog on, lease gone)
cd $HOME/glm53-tensorfold-spark
B=http://127.0.0.1:8000; M=GLM-5.3-Flash-EXL3; R=results/W14
W="ssh -o BatchMode=yes $WORKER_SSH"
step() { echo "=== $* ($(date +%T))"; }
oom() { echo "$(dmesg -T 2>/dev/null | grep -ciE 'out of memory|oom-kill') $($W "dmesg -T | grep -ciE 'out of memory|oom-kill'")"; }
echo "oom lines before: $(oom)" > $R/oom-before.txt
if [[ "${SKIP_TESTS:-0}" != 1 ]]; then
  step tests
  $W "bash $HOME/w14-tests-worker.sh" > $R/tests-worker.out 2>&1 &
  wpid=$!
  bash $R/tests-gpu.sh
  wait $wpid; cat $R/tests-worker.out
fi
step V0; bash $R/load.sh V0 || { echo "V0 LOAD FAILED"; bash $R/restore.sh w14b; exit 1; }
bash $R/tg.sh V0
step V1; bash $R/load.sh V1 GLM53_TF_VISION=1 || { echo "V1 LOAD FAILED"; bash $R/restore.sh w14b; exit 1; }
bash $R/tg.sh V1
python3 $R/vis.py correct $R/correct-V1.json > $R/correct-V1.log 2>&1; tail -45 $R/correct-V1.log | cut -c1-260
python3 $R/vis.py cache $R/cache-V1.json > $R/cache-V1.log 2>&1; cut -c1-400 $R/cache-V1.log
python3 $R/vis.py ttft $R/ttft-V1.json > $R/ttft-V1.log 2>&1; cat $R/ttft-V1.log
curl -s $B/health > $R/health-V1-pre-stress.json
bash $R/mem.sh > $R/mem-V1.log 2>&1 & echo $! > $R/mem.pid
echo "stress start $(date +%T)" >> $R/mem-marks.txt
python3 bench/multiturn.py --base $B --model $M --modes stress --stress-target 250000 --stress-step 100000 --stress-final 32000 \
  --long-tokens 512 --mem-hosts local,$WORKER_SSH --mem-log $R/stress-mem-V1.log --out $R/stress-V1.json > $R/stress-V1.log 2>&1
echo "stress end $(date +%T)" >> $R/mem-marks.txt; tail -3 $R/stress-V1.log
python3 $R/vis.py probe $R/probe-V1.json > $R/probe-V1.log 2>&1; cat $R/probe-V1.log | cut -c1-300
echo "needle start $(date +%T)" >> $R/mem-marks.txt
python3 results/W6/needle.py $R/needle-V1.json 350000 0.4 > $R/needle-V1.log 2>&1; cut -c1-300 $R/needle-V1.log | tail -4
echo "needle end $(date +%T)" >> $R/mem-marks.txt
kill $(cat $R/mem.pid) 2>/dev/null; rm -f $R/mem.pid
awk 'NR>1{if(a==""||$2<a)a=$2; if(b==""||$3<b)b=$3} END{print "MemAvailable min V1 (stress+probe+needle): " a " / " b " GiB"}' $R/mem-V1.log
curl -s $B/health > $R/health-V1-end.json; cat $R/health-V1-end.json; echo
echo "oom lines after V1: $(oom) (before: $(cat $R/oom-before.txt))"
step V1R; bash $R/load.sh V1R GLM53_TF_VISION=1 && python3 $R/vis.py replay $R/replay-V1R.json $R/cache-V1.json > $R/replay-V1R.log 2>&1; cut -c1-500 $R/replay-V1R.log
step V1F; bash $R/load.sh V1F GLM53_TF_VISION=1 GLM53_TF_VISION_MAX_IMAGES=7 && python3 $R/vis.py fresh $R/fresh-V1F.json $R/cache-V1.json > $R/fresh-V1F.log 2>&1; cat $R/fresh-V1F.log
curl -s $B/health > $R/health-V1F.json; cat $R/health-V1F.json; echo
step restore; bash $R/restore.sh w14b
step done
