#!/usr/bin/env bash
# W16 Part B (docs/DRAFTER-SEARCH.md §6): drafter arms on prod (config/prod.env, image b7), one DRAFTER= start each,
# inside the THEORY2-SESSION window (run.sh all's AFTER_LOADS hook: lease refresher + watchdog handled there, restore
# follows). Per arm: start (canary from prod.env), warm-up, acceptfix (untruncated fixed-depth: "4" MTP control on the
# baseline arm only, "f7" DFlash2 7-block every arm), accept (production policy, greedy + sampled), glmbench
# tf,kit,edit x3; the modal arm also exact 10/10 + batchexact 4/4. Compare: results/W16/dscmp.py.
#   ARMS="inco-7d7 modal inco-bf5 [stock]"   DS_DEADLINE=<epoch s>: no arm starts after it
cd $HOME/glm53-tensorfold-spark
source results/THEORY2-SESSION/lib.sh      # log, wssh, health_ok, $B, $M (R = results/W16)
D=results/W16/DS; mkdir -p $D
S=/root/.cache/huggingface/hub
declare -A DR=(
  [inco-7d7]=$S/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/7d74cdd881ed7e32c31175984a67823127b66cfe
  [modal]=$S/models--modal-labs--GLM-5.3-Flash-DFlash/snapshots/dae6d319510889a7f1ac5242d5b7fb2d5d95eb05
  [inco-bf5]=$S/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a
  [stock]=$S/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/7d74cdd881ed7e32c31175984a67823127b66cfe
)
STOCK_MODEL=$S/models--brandonmusic--GLM-5.3-Flash-tr3-4bpw/snapshots/1ae6d70430a12d762917786696db06a7b4f9bbae
# W16 schedule change (22:50): wrap up by ~00:30; modal vs current first, bf5 only if it can start by 23:58, no stock arm
ARMS="inco-7d7 modal inco-bf5"
DS_DEADLINE=$(date -d "2026-09-29 23:58" +%s)
Q="python3 results/W16/w16req.py"
dropcaches() { sync; sudo -n sh -c 'echo 1 > /proc/sys/vm/drop_caches' 2>/dev/null; wssh "sync; echo 1 > /proc/sys/vm/drop_caches" 2>/dev/null
  log "drop_caches: MemFree head $(awk '/MemFree/{printf "%.1f", $2/2^20}' /proc/meminfo) / worker $(wssh "awk '/MemFree/{printf \"%.1f\", \$2/2^20}' /proc/meminfo") GiB"; }
# Part A's control C came up with 1 batch slot (rank 1 short of free memory right after b8's docker load: page cache),
# so its 4-stream / slot numbers are not a control: C2 = the same control again, caches dropped first, full set
if [[ "${C2_RERUN:-0}" == 1 ]]; then   # (W16: folded into the inco-7d7 arm, same load)
  CONFIG=config/prod.env scripts/serve.sh stop > /dev/null 2>&1; dropcaches
  bash results/THEORY2-SESSION/load.sh C2 plain IMAGE=glm53-tensorfold:b7 && health_ok && bash results/THEORY2-SESSION/ab.sh C2 full
  log "C2 slots: $(grep -hE 'batching [0-9] requests|only .* fit' $R/boot-C2-r0.log | cut -c1-120)"
fi
for arm in $ARMS; do
  if (( DS_DEADLINE > 0 && $(date +%s) > DS_DEADLINE )); then log "DS: past the deadline, skipping $arm"; continue; fi
  log "DS arm $arm: DRAFTER=${DR[$arm]}"
  extra=()
  # stock base (a different checkpoint; measurement only): no prepared-folder write (the folders keep 2 keys a rank)
  [[ $arm == stock ]] && extra=(MODEL_PATH=$STOCK_MODEL GLM53_TF_PREPARED_WRITE=0)
  CONFIG=config/prod.env scripts/serve.sh stop > /dev/null 2>&1; dropcaches
  env -u IMAGE "${extra[@]}" DRAFTER="${DR[$arm]}" CONFIG=config/prod.env timeout ${DS_START_TIMEOUT:-1800} scripts/serve.sh start > $D/start-$arm.log 2>&1
  rc=$?
  docker logs glm53-tf-r0 > $D/boot-$arm-r0.log 2>&1
  log "DS $arm start rc=$rc: $(grep -iE 'canary' $D/start-$arm.log | tail -2 | tr '\n' ' ' | cut -c1-300)"
  health_ok || { log "DS $arm: not healthy, next arm"; continue; }
  $Q one $D/warm-$arm.jsonl chat 128 > /dev/null 2>&1
  if [[ $arm == inco-7d7 ]]; then   # this arm IS the control (prod.env, b7, incoai 7d74cdd): Part A's C2 full set here
    log "C2 (= arm inco-7d7) slots: $(grep -hE 'batching [0-9] requests|only .* fit' $D/boot-$arm-r0.log | cut -c1-100)"
    echo "load C2 (plain): IMAGE=glm53-tensorfold:b7 (DS arm inco-7d7) ($(date +%T))" >> $R/loads.log
    cp $D/boot-$arm-r0.log $R/boot-C2-r0.log
    bash results/THEORY2-SESSION/ab.sh C2 full
    cp $R/glmbench-C2.json $D/glmbench-$arm.json
  fi
  if [[ $arm == inco-7d7 ]]; then pol=4,f7; else pol=f7; fi
  POLICIES=$pol $Q acceptfix $D/acceptfix-$arm.jsonl > $D/acceptfix-$arm.log 2>&1
  [[ $arm == stock ]] || $Q accept $D/accept-$arm.jsonl > $D/accept-$arm.log 2>&1
  python3 results/W11/accept.py $D/acceptfix-$arm.jsonl --json $D/acceptfix-$arm.json > $D/acceptfix-$arm.txt 2>&1
  [[ -f $D/accept-$arm.jsonl ]] && python3 results/W11/accept.py $D/accept-$arm.jsonl --json $D/accept-$arm.json > $D/accept-$arm.txt 2>&1
  if [[ $arm != stock && $arm != inco-7d7 ]]; then
    python3 bench/glmbench.py --base $B --model $M --suites tf,kit,edit --reps 3 --long-tokens 512 --label "ds-$arm" \
        --out $D/glmbench-$arm.json > $D/glmbench-$arm.log 2>&1
  fi
  if [[ $arm == modal ]]; then
    python3 bench/glmbench.py --base $B --model $M --suites exact --out $D/exact-$arm.json > $D/exact-$arm.log 2>&1
    python3 bench/multiturn.py --base $B --model $M --modes batchexact --out $D/batchexact-$arm.json > $D/batchexact-$arm.log 2>&1
    log "DS $arm exact: $(grep -c 'identical=True' $D/exact-$arm.log)/$(grep -c 'identical=' $D/exact-$arm.log) | $(grep -E 'batched == alone' $D/batchexact-$arm.log | tail -1)"
  fi
  docker logs glm53-tf-r0 2>&1 | grep -ciE 'traceback' > $D/tracebacks-$arm.txt
  log "DS $arm done: $(grep -E '^(prose|code|agent) +all +DFlash2' $D/acceptfix-$arm.txt | tr -s ' ' | tr '\n' ';' | cut -c1-400)"
done
python3 results/W16/dscmp.py $D > $D/dscmp.txt 2>&1
cat $D/dscmp.txt | tee -a $R/session.log >&2
# item 8 again (the probes' littles run failed at cudaFuncSetAttribute: bulk_kernel's 128 B static smem + the opt-in
# dynamic size > the limit; fixed in probes/littles.cu), prod stopped, clocks locked as in the probes phase
if [[ "${LITTLES_RERUN:-1}" == 1 ]]; then
  log "littles re-run (fixed smem attribute)"
  CONFIG=config/prod.env scripts/serve.sh stop > /dev/null 2>&1
  set_clocks "$LOCK_CLOCKS"
  mv results/W16/probes-head/littles.log results/W16/probes-head/littles-failed.log 2>/dev/null
  LITTLES_ARGS=--quick IMAGE=glm53-tensorfold:b8 bash results/THEORY2-SESSION/probes-node.sh head littles > results/W16/probes-head-littles.out 2>&1
  set_clocks "$PROD_CLOCKS"
  grep GATE results/W16/probes-head/littles.log | tee -a $R/session.log >&2
fi
# before run.sh's restore: page cache out on both nodes, so prod's batch slots fit (C: 1 slot after a big file read)
CONFIG=config/prod.env scripts/serve.sh stop > /dev/null 2>&1; dropcaches
