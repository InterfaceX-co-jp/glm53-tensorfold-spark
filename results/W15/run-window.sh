#!/usr/bin/env bash
# W15 window, in order (window-start.sh already ran: lease + refresher, watchdog timer stopped, prod stopped):
#   GPU tests (head) + engine regressions (worker) -> B7 = config/prod.env + IMAGE b7 + vision (64 / 64 MB caches) +
#   REASONING_FIELDS=reasoning: vision checks (PRE, so the tower's caches are warm), then W12's gate sequence verbatim
#   (ab.sh, ab.py again, N1, 4 x 250k stress, MMLU-200, exact / batchexact again, needle ~314k; MemAvailable every 2 s,
#   phase marks), a post-needle image probe -> B7b (restart, same config): replay (token ids, RigMark's shape), chat
#   regenerate, C2 / C4 first tokens, image cache check again.
# Leaves B7b serving and the window open (lease refresher on, watchdog off): the caller adopts (prod.env -> restore.sh)
# or restores b5 (restore.sh with prod.env unchanged). A failed load restores b5 at once.
cd $HOME/glm53-tensorfold-spark
B=http://127.0.0.1:8000; M=GLM-5.3-Flash-EXL3; R=results/W15
W="ssh -o BatchMode=yes $WORKER_SSH"
step() { echo "=== $* ($(date +%T))"; }
oom() { echo "$(dmesg -T 2>/dev/null | grep -ciE 'out of memory|oom-kill') $($W "dmesg -T | grep -ciE 'out of memory|oom-kill'")"; }
echo "$(oom)" > $R/oom-before.txt
CAND="GLM53_TF_VISION=1 GLM53_TF_VISION_CACHE_MB=64 GLM53_TF_VISION_PREP_MB=64 GLM53_TF_REASONING_FIELDS=reasoning"
if [[ "${SKIP_TESTS:-0}" != 1 ]]; then
  step tests
  $W "bash $HOME/w15-tests.sh worker" > $R/tests-worker.out 2>&1 &
  wpid=$!
  bash $R/tests.sh gpu
  wait $wpid; cat $R/tests-worker.out
fi
step B7
PRE="python3 $R/vis.py correct $R/correct-B7.json > $R/correct-B7.log 2>&1; tail -40 $R/correct-B7.log | cut -c1-260; \
python3 $R/vis.py cache $R/cache-B7.json > $R/cache-B7.log 2>&1; cut -c1-400 $R/cache-B7.log" \
  bash $R/gates.sh B7 $CAND || { echo "B7 FAILED"; bash $R/restore.sh w15-fail; exit 1; }
python3 $R/vis.py probe $R/probe-B7.json > $R/probe-B7.log 2>&1; cut -c1-300 $R/probe-B7.log
echo "oom lines after B7: $(oom) (before: $(cat $R/oom-before.txt))"
docker logs glm53-tf-r0 > $R/run-B7-r0.log 2>&1; $W docker logs glm53-tf-r1 > $R/run-B7-r1.log 2>&1
step B7b
bash $R/load.sh B7b $CAND || { echo "B7b LOAD FAILED"; bash $R/restore.sh w15-fail; exit 1; }
python3 $R/replay.py tokens $R/replay-B7b.json > $R/replay-B7b.log 2>&1; grep SUMMARY $R/replay-B7b.log
python3 $R/replay.py chat $R/regen-B7b.json > $R/regen-B7b.log 2>&1; cat $R/regen-B7b.log
python3 $R/c4.py $R/c4-B7b.json 3 > $R/c4-B7b.log 2>&1; cat $R/c4-B7b.log
python3 $R/vis.py cache $R/cache-B7b.json > $R/cache-B7b.log 2>&1; cut -c1-400 $R/cache-B7b.log
curl -s $B/health > $R/health-B7b.json; cat $R/health-B7b.json; echo
docker logs glm53-tf-r0 2>&1 | grep -ciE 'traceback|error' | sed 's/^/r0 error lines: /'
step done
