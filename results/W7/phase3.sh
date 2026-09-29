#!/usr/bin/env bash
# W7 phase 3: after phase 2 -> stop B, RoCE loopback with the offline 0350 harness (interleaved warm-ups; snapshot of
# tests/cuda/bench_roce.py at /tmp/w7/bench_roce_0350.py, runtime = the image's 0230 roce.py), load C (piece 4096,
# rows max 4096), prefill at 24.5k / 98k. Prod is restored by hand afterwards.
cd $HOME/glm53-tensorfold-spark
R=results/W7
while ! grep -q "PHASE2 DONE" $R/phase2.log 2>/dev/null; do sleep 5; done
CONFIG=config/prod.env scripts/serve.sh stop > /dev/null 2>&1
echo "B stopped $(date +%T)"
docker rm -f w7-roce >/dev/null 2>&1
timeout 300 docker run --rm --name w7-roce --gpus all --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK \
  --network host --ipc host -v glm53-tf-cache:/cache -e PYTHONDONTWRITEBYTECODE=1 -e NCCL_SOCKET_IFNAME=enp1s0f1np1 \
  -e NCCL_IB_HCA=rocep1s0f1 -v "$PWD/tests:/work/tests" -v /tmp/w7/bench_roce_0350.py:/work/bench_roce_0350.py:ro \
  --entrypoint bash glm53-tensorfold:kvpool -c "cd /work && PYTHONPATH=/src/TensorFold/src:/src/TensorFold/tests/cuda:/work/tests/cuda python bench_roce_0350.py loopback; echo RC=\$?" \
  > $R/roce-loopback-0350harness.log 2>&1
echo "roce 0350 harness done $(date +%T)"; tail -n 4 $R/roce-loopback-0350harness.log
bash $R/load.sh C plain GLM53_TF_BATCH_PIECE=4096 GLM53_TF_PREFILL_ROWS_MAX=4096 > $R/loadC.out 2>&1
echo "C loaded $(date +%T)"; tail -n 3 $R/loadC.out
python3 $R/req.py prefill $R/warm-C.jsonl 12000 8 '{"prefill_rows":4096}' | tail -n 1
python3 results/W5/ab.py $R/ab-C.json 24500,98000 '{"c4096":{"prefill_rows":4096}}' 256 > $R/ab-C.log 2>&1
cat $R/ab-C.log
docker logs glm53-tf-r0 > $R/run-C-r0.log 2>&1
echo "PHASE3 DONE $(date +%T)"
