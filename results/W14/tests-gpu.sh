#!/usr/bin/env bash
# W14 step 1 (prod stopped): 0500 GPU tests in glm53-tensorfold:b6 on the head node (real tower via GLM53_TF_MODEL) and the
# host vision tests in the image (torchvision / torch of the image give the same processor bits). Logs: results/W14/tests/
cd $HOME/glm53-tensorfold-spark
O=results/W14/tests; mkdir -p $O
IMG=glm53-tensorfold:b6
SNAP=/root/.cache/huggingface/hub/models--neko-legends--GLM-5.3-Flash-Uncensored-EXL3/snapshots/07135ec082f8f11f7a71e4244a4e5167a0f96277
run() { # name timeout "env args" cmd...
  local n=$1 t=$2 e=$3; shift 3
  local t0=$(date +%s)
  docker run --rm --name w14-$n --gpus all --network host --ipc host -v glm53-tf-cache:/cache -e PYTHONDONTWRITEBYTECODE=1 \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface:ro $e \
    -v $PWD/tests:/work/tests -v $PWD/scripts:/work/scripts --entrypoint bash $IMG -c \
    "pip install -q pytest >/dev/null 2>&1; nvidia-smi --query-gpu=memory.used,clocks.sm --format=csv,noheader; cd /work && PYTHONPATH=/src/TensorFold/src:/src/TensorFold/tests/cuda:/work/tests/cuda:/work/tests timeout -k 30 $t $*" > $O/$n.log 2>&1
  local rc=$?; docker rm -f w14-$n >/dev/null 2>&1
  echo "$(date +%T) $n rc=$rc $(( $(date +%s) - t0 ))s :: $(grep -E 'passed|failed|error|Error' $O/$n.log | tail -1 | cut -c1-200)" | tee -a $O/SUMMARY
}
P="python -m pytest -q -p no:cacheprovider -rs"
echo "$(date +%T) start" | tee -a $O/SUMMARY
mkdir -p results/W14/tower
timeout 900 $HOME/venvs/vllm-exl3/bin/python results/W14/hf_tower.py $HOME/.cache/huggingface/hub/models--neko-legends--GLM-5.3-Flash-Uncensored-EXL3/snapshots/07135ec082f8f11f7a71e4244a4e5167a0f96277 results/W14/tower results/W14/img/desk_1920x1080.png > $O/hf-tower.log 2>&1
echo "$(date +%T) hf-tower rc=$? :: $(tail -1 $O/hf-tower.log)" | tee -a $O/SUMMARY
run gpu-vision2 2400 "-e GLM53_TF_MODEL=$SNAP -e TRITON_INTERPRET=0" $P -s tests/cuda/test_vision_patches.py
# the production MTP prefill path (latent KV + GLM53_TF_MTP_PREFILL_CACHE=1: cache_absorb, eager) on the synthetic engine
run gpu-vision-latent 2400 "-e GLM53_TF_LATENT_KV=1 -e GLM53_TF_KV_DTYPE=fp8 -e GLM53_TF_MTP_PREFILL_CACHE=1 -e TRITON_INTERPRET=0" $P -s tests/cuda/test_vision_patches.py -k "'not real_tower'"
# the tower against transformers' Glm5NextVisionModel (bf16 / fp32) on the same inputs: hf_tower.py ran on the host venv
run tower-ours 900 "-v $PWD/results/W14:/work/w14" python /work/w14/tower_ours.py $SNAP /work/w14/tower /work/w14/img/desk_1920x1080.png
echo "$(date +%T) done" | tee -a $O/SUMMARY
