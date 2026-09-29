#!/usr/bin/env bash
# W7 load A (prod config under an idle nsys session, GLM53_TF_PROFILE=1, W7 NVTX marks): control, then captures
cd $HOME/glm53-tensorfold-spark
R=results/W7; C=$R/nsysctl.sh; J=$R/reqA.jsonl
waitrep() { # wait until both ranks' report for $1 exists and no nsys import runs
  for i in $(seq 1 180); do
    a=$(ls /tmp/w7/out/$1-r0.nsys-rep 2>/dev/null); b=$(ssh -o BatchMode=yes $WORKER_SSH ls /tmp/w7/out/$1-r1.nsys-rep 2>/dev/null)
    if [[ -n $a && -n $b ]] && ! pgrep -f QdstrmImporter >/dev/null && ! ssh -o BatchMode=yes $WORKER_SSH pgrep -f QdstrmImporter >/dev/null; then echo "report $1 ready $(date +%T)"; return; fi
    sleep 5
  done; echo "report $1 TIMEOUT"; }
echo "control $(date +%T)"
python3 $R/req.py prefill $J 24500 16 '{"profile":0}'
python3 $R/req.py prefill $J 24500 16 '{"profile":1}'
for cap in pf24:24500 pf98:98000; do
  name=${cap%%:*}; n=${cap##*:}
  $C start $name; sleep 1
  python3 $R/req.py prefill $J $n 16 '{"profile":1}'
  sleep 1; $C stop; waitrep $name
done
$C start dec1; sleep 1; python3 $R/req.py decode $J 1 256; sleep 1; $C stop; waitrep dec1
$C start dec4; sleep 1; python3 $R/req.py decode $J 4 384; sleep 1; $C stop; waitrep dec4
echo "DONE $(date +%T)"
