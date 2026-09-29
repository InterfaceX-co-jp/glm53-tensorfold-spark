#!/usr/bin/env bash
# W7 load A2: ONE nsys capture (a second start/stop cycle in the same session lost the nsys agent and took the
# server down in A1): the 98k prompt, one single-stream decode, one 4-stream decode, then stop
cd $HOME/glm53-tensorfold-spark
R=results/W7; C=$R/nsysctl.sh; J=$R/reqA2.jsonl
$C start mix; sleep 1
python3 $R/req.py prefill $J 98000 16 '{"profile":1}'
sleep 1; python3 $R/req.py decode $J 1 256
sleep 1; python3 $R/req.py decode $J 4 384
sleep 1; $C stop
for i in $(seq 1 180); do
  a=$(ls /tmp/w7/out/mix-r0.nsys-rep 2>/dev/null); b=$(ssh -o BatchMode=yes $WORKER_SSH ls /tmp/w7/out/mix-r1.nsys-rep 2>/dev/null)
  if [[ -n $a && -n $b ]] && ! pgrep -f QdstrmImporter >/dev/null; then echo "report mix ready $(date +%T)"; break; fi
  sleep 5
done
echo "DONE $(date +%T)"
