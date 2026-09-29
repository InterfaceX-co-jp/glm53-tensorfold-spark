#!/usr/bin/env bash
# W11 capture (load "nsys"): warm-up, the same requests uncaptured (ctl.jsonl), then ONE nsys window on both ranks
# (cap.jsonl): prose chat 256, prose essay 384, code 384, 4 x prose 384 concurrent; 4 s apart (segments). Greedy.
cd $HOME/glm53-tensorfold-spark
R=results/W11; C=$R/nsysctl.sh; Q="python3 $R/w11req.py"
set -x
$Q one $R/warm.jsonl chat 64
$Q conc $R/warm.jsonl 4 64
for f in ctl cap; do
  J=$R/$f.jsonl
  if [[ $f == cap ]]; then $C start cap; sleep 2; fi
  $Q one $J chat 256; sleep 4
  $Q one $J essay 384; sleep 4
  $Q one $J code-lru 384; sleep 4
  $Q conc $J 4 384; sleep 2
  if [[ $f == cap ]]; then $C stop; fi
  sleep 4
done
for i in $(seq 1 180); do
  a=$(ls /var/tmp/w11/out/cap-r0.nsys-rep 2>/dev/null); b=$(ssh -o BatchMode=yes $WORKER_SSH ls /var/tmp/w11/out/cap-r1.nsys-rep 2>/dev/null)
  if [[ -n $a && -n $b ]] && ! pgrep -f QdstrmImporter >/dev/null && ! ssh -o BatchMode=yes $WORKER_SSH pgrep -f QdstrmImporter >/dev/null; then echo "report cap ready $(date +%T)"; break; fi
  sleep 5
done
ls -la /var/tmp/w11/out; ssh -o BatchMode=yes $WORKER_SSH ls -la /var/tmp/w11/out
echo "DONE $(date +%T)"
