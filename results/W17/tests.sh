#!/usr/bin/env bash
# W17 GPU tests in glm53-tensorfold:b9 (prod stopped), one node each, every run under timeout:
#   tests.sh head | worker      REPO = the node's repo copy (tests/ mounted at /work/tests)  -> REPO/results/W17/tests-NODE/
#   tests.sh probe01 | probe02  bench/pagecache_probe.py (0550 step P) on a prepared weight file, page cache dropped first
node=$1
IMG=glm53-tensorfold:b9
P="python -m pytest -q -p no:cacheprovider -rs"
if [[ $node == worker || $node == probe02 ]]; then REPO=$HOME/glm53-tensorfold-spark; HF=/root/.cache/huggingface; else REPO=$HOME/glm53-tensorfold-spark; HF=$HOME/.cache/huggingface; fi
cd $REPO; O=results/W17/tests-$node; mkdir -p $O
run() { # name timeout "env args" cmd...
  local n=$1 t=$2 e=$3; shift 3; local t0=$(date +%s)
  docker run --rm --name w17-$n --gpus all --network host --ipc host -e PYTHONDONTWRITEBYTECODE=1 -e TRITON_INTERPRET=0 $e \
    -v $HF:/root/.cache/huggingface:ro -v $REPO/tests:/work/tests -v $REPO/bench:/work/bench -v /tmp/w17-pc:/pc \
    --entrypoint bash $IMG -c "pip install -q pytest >/dev/null 2>&1; cd /work && PYTHONPATH=/src/TensorFold/src:/src/TensorFold/tests/cuda:/work/tests/cuda:/work/tests timeout -k 30 $t $*" > $O/$n.log 2>&1
  local rc=$?; docker rm -f w17-$n >/dev/null 2>&1
  echo "$(date +%T) $node $n rc=$rc $(( $(date +%s) - t0 ))s :: $(grep -E 'passed|failed|error|cache_reclaimed|VERDICT' $O/$n.log | tail -1 | cut -c1-220)" | tee -a $O/SUMMARY
}
dropc() { sync; if [[ $node == worker || $node == probe02 ]]; then echo 3 > /proc/sys/vm/drop_caches; else sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'; fi; }
echo "$(date +%T) start $node" | tee -a $O/SUMMARY
case $node in
probe01|probe02)
  f=$(find $(dirname $HF)/glm53-tf/prepared -name data.bin -size +20G | head -1); mkdir -p /tmp/w17-pc
  echo "file $f $(du -h $f | cut -f1)" | tee -a $O/SUMMARY
  dropc
  run pcprobe 900 "-v $f:/pcf/data.bin:ro" python3 /work/bench/pagecache_probe.py /pc/out.json /pcf/data.bin 20 6 1 8
  cat /tmp/w17-pc/out.json > $O/pcprobe.json 2>&1
  dropc ;;
head)
  run memsafe 1800 "" $P tests/cuda/test_memory_safety_patches.py
  run multi 2400 "" $P -s tests/cuda/test_multi_prefill_patches.py
  run lean 900 "" $P tests/cuda/test_lean_patches.py
  run overlap 900 "" $P tests/cuda/test_overlap_patches.py
  run prefill-pp 1200 "" $P tests/cuda/test_prefill_pp_patches.py
  run replay 900 "" $P tests/cuda/test_replay_ttft_patches.py ;;
worker)
  run multi 2400 "" $P -s tests/cuda/test_multi_prefill_patches.py
  run memsafe 1800 "" $P tests/cuda/test_memory_safety_patches.py
  run 1m 900 "" $P tests/cuda/test_1m_patches.py
  run batch-sessions 1800 "" $P tests/cuda/test_batch_sessions_patches.py
  run session-disk 1200 "" $P tests/cuda/test_session_disk_patches.py
  run fastpf 1200 "" $P tests/cuda/test_fastpf_patches.py
  run cindep 1200 "" $P tests/cuda/test_cindep_patches.py
  run engine 1800 "" bash -c "'cd /src/TensorFold && $P tests/cuda/test_glm_engine.py'" ;;
esac
echo "$(date +%T) done $node" | tee -a $O/SUMMARY
