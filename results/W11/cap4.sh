#!/usr/bin/env bash
# W11 capture 2 (load "nsys4": 4 slots; the first nsys load fit only 1): warm-up, the same requests uncaptured
# (ctl4.jsonl), then ONE nsys window on both ranks (cap4.jsonl): 4 x prose 384 concurrent, 4 s, 1 x prose chat 256. Greedy.
cd $HOME/glm53-tensorfold-spark
R=results/W11; C=$R/nsysctl.sh; Q="python3 $R/w11req.py"
$Q one $R/warm4.jsonl chat 64
$Q conc $R/warm4.jsonl 4 64
for f in ctl4 cap4; do
  J=$R/$f.jsonl
  if [[ $f == cap4 ]]; then $C start cap4; sleep 2; fi
  $Q conc $J 4 384; sleep 4
  $Q one $J chat 256; sleep 2
  if [[ $f == cap4 ]]; then $C stop; fi
  sleep 4
done
for i in $(seq 1 180); do
  a=$(ls /var/tmp/w11/out/cap4-r0.nsys-rep 2>/dev/null); b=$(ssh -o BatchMode=yes $WORKER_SSH ls /var/tmp/w11/out/cap4-r1.nsys-rep 2>/dev/null)
  if [[ -n $a && -n $b ]] && ! pgrep -f QdstrmImporter >/dev/null && ! ssh -o BatchMode=yes $WORKER_SSH pgrep -f QdstrmImporter >/dev/null; then echo "report cap4 ready $(date +%T)"; break; fi
  sleep 5
done
ls -la /var/tmp/w11/out
echo "DONE $(date +%T)"
