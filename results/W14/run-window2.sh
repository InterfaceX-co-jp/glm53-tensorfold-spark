#!/usr/bin/env bash
# W14 window 2 (window-start.sh already ran): the latent-KV synthetic GPU test (bf16 latent + MTP_PREFILL_CACHE, the
# production MTP prefill path), then D1 (vision on, a fresh session compat dir via MAX_IMAGES=6): B2 fresh (1,000+ text
# tokens before the image), the long image conversation (>= GLM53_TF_SESSION_DISK_MIN), MAX_PIXELS; D2 (MAX_IMAGES=5,
# another fresh dir): A2 then B2 (B2 resumes the text only, reply == fresh); D3 (= D1's config, restart): the long
# conversation restored from NVMe, identical; restore.sh
cd $HOME/glm53-tensorfold-spark
R=results/W14; O=$R/tests
step() { echo "=== $* ($(date +%T))"; }
step latent
docker run --rm --name w14-latent --gpus all --network host --ipc host -v glm53-tf-cache:/cache -e PYTHONDONTWRITEBYTECODE=1 \
  -e GLM53_TF_LATENT_KV=1 -e GLM53_TF_MTP_PREFILL_CACHE=1 -e TRITON_INTERPRET=0 -v $PWD/tests:/work/tests --entrypoint bash glm53-tensorfold:b6 -c \
  "pip install -q pytest >/dev/null 2>&1; cd /work && PYTHONPATH=/src/TensorFold/src:/src/TensorFold/tests/cuda:/work/tests/cuda:/work/tests timeout -k 30 900 python -m pytest -q -s -rs -p no:cacheprovider tests/cuda/test_vision_patches.py -k 'not real_tower'" > $O/gpu-vision-latent-bf16.log 2>&1
echo "latent rc=$? :: $(grep -E 'passed|failed|error' $O/gpu-vision-latent-bf16.log | tail -1)"
step D1; bash $R/load.sh D1 GLM53_TF_VISION=1 GLM53_TF_VISION_MAX_IMAGES=6 || { bash $R/restore.sh w14c; exit 1; }
python3 $R/vis.py pairfresh $R/pair2-fresh.json; python3 $R/vis.py disk1 $R/disk1.json; python3 $R/vis.py pixels $R/pixels.json
sleep 15
step D2; bash $R/load.sh D2 GLM53_TF_VISION=1 GLM53_TF_VISION_MAX_IMAGES=5 || { bash $R/restore.sh w14c; exit 1; }
python3 $R/vis.py pairab $R/pair2-ab.json $R/pair2-fresh.json
step D3; bash $R/load.sh D3 GLM53_TF_VISION=1 GLM53_TF_VISION_MAX_IMAGES=6 || { bash $R/restore.sh w14c; exit 1; }
python3 $R/vis.py disk2 $R/disk2.json $R/disk1.json
docker logs glm53-tf-r0 2>&1 | grep -iE "session.*disk|nvme|sessdisk" | head -5
step restore; bash $R/restore.sh w14c
step done
