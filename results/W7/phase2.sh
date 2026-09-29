#!/usr/bin/env bash
# W7 phase 2: after load A2's capture -> save logs, stop, RoCE loopback rerun, load B (piece 8192, rows max 8192),
# prefill sweep per request, overlap A/B, plain decode (single / 4 streams), exact suite
cd $HOME/glm53-tensorfold-spark
R=results/W7
while ! grep -q DONE $R/runA2.log 2>/dev/null; do sleep 5; done
docker logs glm53-tf-r0 > $R/run-A2-r0.log 2>&1
ssh -o BatchMode=yes $WORKER_SSH docker logs glm53-tf-r1 > $R/run-A2-r1.log 2>&1
CONFIG=config/prod.env scripts/serve.sh stop > /dev/null 2>&1
echo "A2 stopped $(date +%T)"
bash $R/roce.sh $R/roce-loopback.log
echo "roce done $(date +%T)"
bash $R/load.sh B plain GLM53_TF_BATCH_PIECE=8192 GLM53_TF_PREFILL_ROWS_MAX=8192 > $R/loadB.out 2>&1
echo "B loaded $(date +%T)"; tail -n 4 $R/loadB.out
bash $R/runB.sh B '{"r2048":{"prefill_rows":2048},"r4096":{"prefill_rows":4096},"r8192":{"prefill_rows":8192},"r2048_ov0":{"prefill_rows":2048,"prefill_overlap":0}}'
python3 $R/req.py decode $R/reqB-dec.jsonl 1 256
python3 $R/req.py decode $R/reqB-dec.jsonl 1 256
python3 $R/req.py decode $R/reqB-dec.jsonl 4 384
python3 bench/glmbench.py --base http://127.0.0.1:8000 --model GLM-5.3-Flash-EXL3 --suites exact --out $R/exact-B.json > $R/exact-B.log 2>&1
tail -n 3 $R/exact-B.log
docker logs glm53-tf-r0 > $R/run-B-r0.log 2>&1
ssh -o BatchMode=yes $WORKER_SSH docker logs glm53-tf-r1 > $R/run-B-r1.log 2>&1
echo "PHASE2 DONE $(date +%T)"
