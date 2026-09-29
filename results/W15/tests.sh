#!/usr/bin/env bash
# W15 tests in glm53-tensorfold:b7. tests.sh cpu|gpu|gpu2|worker|workerb
#   cpu    (prod may serve: no --gpus) the 0540 / 0500 host and fake-model suites on CPU        -> results/W15/tests-cpu/
#   gpu    (prod stopped, head) 0540's tests on the GPU, 0500's GPU test with the real tower (8% bound) and on the
#          production MTP prefill path, 0540's plan regressions (batch sessions, fastpf, sessions, cindep, overlap)
#                                                                                                 -> results/W15/tests-gpu/
#   worker  (prod stopped, on the worker node as $HOME/w15-tests.sh) TensorFold's test_glm_engine.py (0500's engine regressions)
#          + test_replay_ttft_patches.py                                                          -> $HOME/w15-tests/
mode=$1
IMG=glm53-tensorfold:b7
SNAP=/root/.cache/huggingface/hub/models--neko-legends--GLM-5.3-Flash-Uncensored-EXL3/snapshots/07135ec082f8f11f7a71e4244a4e5167a0f96277
P="python -m pytest -q -p no:cacheprovider -rs"
if [[ $mode == worker ]]; then
  O=$HOME/w15-tests; mkdir -p $O; W=$HOME/w15-work
  run() { local n=$1 t=$2; shift 2; local t0=$(date +%s)
    docker run --rm --name w15-$n --gpus all --network host --ipc host -e PYTHONDONTWRITEBYTECODE=1 -e TRITON_INTERPRET=0 \
      -v $W/tests:/work/tests --entrypoint bash $IMG -c "pip install -q pytest >/dev/null 2>&1; cd /src/TensorFold && PYTHONPATH=/src/TensorFold/src:/src/TensorFold/tests/cuda:/work/tests/cuda:/work/tests timeout -k 30 $t $*" > $O/$n.log 2>&1
    echo "$(date +%T) $n rc=$? $(( $(date +%s) - t0 ))s :: $(grep -E 'passed|failed|error' $O/$n.log | tail -1 | cut -c1-200)" | tee -a $O/SUMMARY; }
  echo "$(date +%T) start" | tee -a $O/SUMMARY
  run gpu-engine 1800 $P tests/cuda/test_glm_engine.py
  run gpu-replay 1800 $P /work/tests/cuda/test_replay_ttft_patches.py
  echo "$(date +%T) done" | tee -a $O/SUMMARY
  exit 0
fi
if [[ $mode == workerb ]]; then
  # W15 rerun: TensorFold's engine test with 0540's updated expectation (patches/0540's test hunk, mounted: b7's copy
  # predates it), knob on (default) and off
  O=$HOME/w15-tests; W=$HOME/w15-work
  run() { local n=$1 t=$2 e=$3; shift 3; local t0=$(date +%s)
    docker run --rm --name w15-$n --gpus all --network host --ipc host -e PYTHONDONTWRITEBYTECODE=1 -e TRITON_INTERPRET=0 $e \
      -v $W/test_glm_engine.py:/src/TensorFold/tests/cuda/test_glm_engine.py:ro --entrypoint bash $IMG -c "pip install -q pytest >/dev/null 2>&1; cd /src/TensorFold && PYTHONPATH=/src/TensorFold/src:/src/TensorFold/tests/cuda timeout -k 30 $t $*" > $O/$n.log 2>&1
    echo "$(date +%T) $n rc=$? $(( $(date +%s) - t0 ))s :: $(grep -E 'passed|failed|error' $O/$n.log | tail -1 | cut -c1-200)" | tee -a $O/SUMMARY; }
  echo "$(date +%T) start workerb" | tee -a $O/SUMMARY
  run gpu-engine2 1800 "" $P tests/cuda/test_glm_engine.py
  run gpu-engine2-off 1800 "-e GLM53_TF_SNAPSHOT_BEFORE_END=0" $P tests/cuda/test_glm_engine.py
  echo "$(date +%T) done" | tee -a $O/SUMMARY
  exit 0
fi
cd $HOME/glm53-tensorfold-spark
O=results/W15/tests-$mode; mkdir -p $O
G="--gpus all"; [[ $mode == cpu ]] && G=""
run() { # name timeout "env args" cmd...
  local n=$1 t=$2 e=$3; shift 3
  local t0=$(date +%s)
  docker run --rm --name w15-$n $G --network host --ipc host -v glm53-tf-cache:/cache -e PYTHONDONTWRITEBYTECODE=1 \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface:ro $e \
    -v $PWD/tests:/work/tests -v $PWD/scripts:/work/scripts --entrypoint bash $IMG -c \
    "pip install -q pytest >/dev/null 2>&1; nvidia-smi --query-gpu=memory.used,clocks.sm --format=csv,noheader 2>/dev/null; cd /work && PYTHONPATH=/src/TensorFold/src:/src/TensorFold/tests/cuda:/work/tests/cuda:/work/tests timeout -k 30 $t $*" > $O/$n.log 2>&1
  local rc=$?; docker rm -f w15-$n >/dev/null 2>&1
  echo "$(date +%T) $n rc=$rc $(( $(date +%s) - t0 ))s :: $(grep -E 'passed|failed|error|Error' $O/$n.log | tail -1 | cut -c1-200)" | tee -a $O/SUMMARY
}
echo "$(date +%T) start $mode" | tee -a $O/SUMMARY
if [[ $mode == cpu ]]; then
  run replay 1800 "" $P tests/cuda/test_replay_ttft_patches.py
  run session 1800 "" $P tests/cuda/test_session_patches.py
  run overlap 1800 "" $P tests/cuda/test_decode_overlap_patches.py
  run host-vision 900 "" $P tests/test_vision_prep.py tests/test_vision.py tests/test_vision_server.py
  run host-api 600 "" $P tests/test_api_context.py tests/test_prompt_tokens.py tests/test_openai_compat.py
elif [[ $mode == gpu2 ]]; then
  # W15 rerun after the test updates for patches/0540's snapshot rule (fastpf, session budget, vision tail), knob on
  # (default) and, for the files whose expectations depend on it, off
  run vision 2400 "-e GLM53_TF_MODEL=$SNAP -e TRITON_INTERPRET=0" $P -s tests/cuda/test_vision_patches.py
  run vision-latent 2400 "-e GLM53_TF_LATENT_KV=1 -e GLM53_TF_MTP_PREFILL_CACHE=1 -e TRITON_INTERPRET=0" $P -s tests/cuda/test_vision_patches.py -k "'not real_tower'"
  run fastpf 2400 "-e TRITON_INTERPRET=0" $P tests/cuda/test_fastpf_patches.py
  run fastpf-off 2400 "-e TRITON_INTERPRET=0 -e GLM53_TF_SNAPSHOT_BEFORE_END=0" $P tests/cuda/test_fastpf_patches.py
  run session 1800 "-e TRITON_INTERPRET=0" $P tests/cuda/test_session_patches.py
  run session-off 1800 "-e TRITON_INTERPRET=0 -e GLM53_TF_SNAPSHOT_BEFORE_END=0" $P tests/cuda/test_session_patches.py
  run vision-off 2400 "-e TRITON_INTERPRET=0 -e GLM53_TF_SNAPSHOT_BEFORE_END=0" $P -s tests/cuda/test_vision_patches.py -k "'not real_tower'"
else
  run vision 2400 "-e GLM53_TF_MODEL=$SNAP -e TRITON_INTERPRET=0" $P -s tests/cuda/test_vision_patches.py
  run vision-latent 2400 "-e GLM53_TF_LATENT_KV=1 -e GLM53_TF_MTP_PREFILL_CACHE=1 -e TRITON_INTERPRET=0" $P -s tests/cuda/test_vision_patches.py -k "'not real_tower'"
  run replay 1800 "-e TRITON_INTERPRET=0" $P tests/cuda/test_replay_ttft_patches.py
  run batch-sessions 2400 "-e TRITON_INTERPRET=0" $P tests/cuda/test_batch_sessions_patches.py
  run fastpf 2400 "-e TRITON_INTERPRET=0" $P tests/cuda/test_fastpf_patches.py
  run session 1800 "-e TRITON_INTERPRET=0" $P tests/cuda/test_session_patches.py
  run cindep 1800 "-e TRITON_INTERPRET=0" $P tests/cuda/test_cindep_patches.py
  run overlap 1800 "-e TRITON_INTERPRET=0" $P tests/cuda/test_decode_overlap_patches.py
fi
echo "$(date +%T) done $mode" | tee -a $O/SUMMARY
