#!/usr/bin/env bash
# W12 step 1: offline suites in the image + GPU kernel tests / microbenches of 0440 (head) and 0450 / 0460 (worker) in
# glm53-tensorfold:b5 (prod stopped), every step under timeout (new kernels can hang). tests-node.sh head|worker [steps]
# Logs: results/W12/tests-<node>/ (SUMMARY: one line a step)
node=$1; only=${2:-}
if [[ $node == head ]]; then cd $HOME/glm53-tensorfold-spark; else cd $HOME/glm53-tensorfold-spark; fi
O=results/W12/tests-$node; mkdir -p $O
IMG=glm53-tensorfold:b5
run() { # name timeout "docker env args" cmd...
  local n=$1 t=$2 e=$3; shift 3
  if [[ -n "$only" && " $only " != *" $n "* ]]; then return 0; fi
  local t0=$(date +%s)
  docker run --rm --name w12-$n --gpus all --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK --network host \
    --ipc host -v glm53-tf-cache:/cache -e PYTHONDONTWRITEBYTECODE=1 -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e NCCL_IB_HCA=rocep1s0f1 \
    $e -v $PWD/tests:/work/tests -v $PWD/scripts:/work/scripts -v $PWD/results/W12:/work/w12 --entrypoint bash $IMG -c \
    "pip install -q pytest >/dev/null 2>&1; nvidia-smi --query-gpu=memory.used,clocks.sm --format=csv,noheader; cd /work && PYTHONPATH=/src/TensorFold/src:/src/TensorFold/tests/cuda:/work/tests/cuda:/work/tests timeout -k 30 $t $*" > $O/$n.log 2>&1
  local rc=$?; docker rm -f w12-$n >/dev/null 2>&1
  echo "$(date +%T) $n rc=$rc $(( $(date +%s) - t0 ))s :: $(grep -E 'passed|failed|error|Error' $O/$n.log | tail -1 | cut -c1-200)" | tee -a $O/SUMMARY
  return $rc
}
P="python -m pytest -q -p no:cacheprovider -rs"
echo "$(date +%T) start $node" | tee -a $O/SUMMARY
if [[ $node == head && "${3:-}" != all ]]; then
  # offline suites in the image (0440 guard must say fused on Triton 3.7.1; 0490 host tests)
  run off-emu 900 "" $P tests/test_decode_kernels_emulator.py
  run off-compile 900 "" $P -s tests/test_decode_kernels_compile.py
  run off-api 600 "" $P tests/test_api_context.py tests/test_prompt_tokens.py
  # 0440 step 1: bitwise kernels + the PDL race test
  run e-bits 1800 "" $P -s tests/cuda/test_decode_stream_patches.py -k "'experts or qmm or pdl'"
  # 0440 step 2: microbench
  run e-bench 1800 "" python tests/cuda/bench_decode_kernels.py --json /work/w12/kbench.json
  # 0440 step 3: engines
  run e-engines 1800 "" $P -s tests/cuda/test_decode_stream_patches.py -k "'windows or replies or resume or timing'"
  run e-decode-on 1800 "-e GLM53_TF_DEC_EXPERTS=1 -e GLM53_TF_DEC_QMM=1" $P tests/cuda/test_decode_patches.py
  run e-bpar-on 1800 "-e GLM53_TF_DEC_EXPERTS=1 -e GLM53_TF_DEC_QMM=1" $P tests/cuda/test_batch_parallel_patches.py
  run e-deep-on 1800 "-e GLM53_TF_DEC_EXPERTS=1 -e GLM53_TF_DEC_QMM=1" $P tests/cuda/test_deep_verify_patches.py
else
  run off-pf 900 "" $P tests/test_prefetch_comm.py tests/test_prefetch_comm_compile.py tests/test_roce_protocol_model.py
  run off-gr 900 "" $P tests/test_gpu_sampler_interpreter.py tests/test_gpu_round_compile.py
  # 0450 step 0
  run g-unit 3600 "" $P -v -s -o faulthandler_timeout=900 tests/cuda/test_gpu_round_patches.py
  # the test module sets TRITON_INTERPRET=1 by default (CPU interpreter): the GPU half needs it off
  run g-unit0 3600 "-e TRITON_INTERPRET=0" $P -s -o faulthandler_timeout=900 tests/cuda/test_gpu_round_patches.py
  run g-resident 1800 "" $P tests/cuda/test_gpu_round_resident.py
  # 10^8 keyed draws + 2.5 x 10^9 libm values vs the node's numpy (self_check's halves at scale: w12/sampler.py)
  run g-sampler 5400 "" python w12/sampler.py 100 1048576 1048576
  run g-bpar 1800 "-e GLM53_TF_GPU_ROUND=1" $P tests/cuda/test_batch_parallel_patches.py
  run g-bsess 1800 "-e GLM53_TF_GPU_ROUND=1" $P tests/cuda/test_batch_sessions_patches.py
  run g-overlap 1200 "-e GLM53_TF_GPU_ROUND=1 -e GLM53_TF_DECODE_OVERLAP=1" $P tests/cuda/test_decode_overlap_patches.py
  # 0460 step 1
  run p-unit 1800 "" $P -s tests/cuda/test_prefetch_comm_patches.py
  run p-roce 1200 "" $P tests/cuda/test_roce_patches.py
  run p-comm 1200 "" $P tests/cuda/test_comm_patches.py
  run p-bpar 1800 "-e GLM53_TF_L2PF=1" $P tests/cuda/test_batch_parallel_patches.py
  run p-pdl 900 "-e GLM53_TF_L2PF=1" $P tests/cuda/test_decode_stream_patches.py -k pdl
fi
echo "$(date +%T) $node tests done" | tee -a $O/SUMMARY
