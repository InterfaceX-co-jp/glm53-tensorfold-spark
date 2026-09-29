#!/usr/bin/env bash
# W9 0310 IMA reproducer: a FRESH cache volume (empty /cache: torch extensions, Triton, calibration all built in
# the run), then the 0310 file. ima-exp.sh NAME REPO OLDTESTS twoproc|alone [extra docker env]
# twoproc: first the 0320 two-engine test at a08d7d6 (W8's order), in the same fresh volume.
n=$1 repo=$2 old=$3 mode=$4 env=${5:-}
cd $repo; O=results/W9/ima; mkdir -p $O
vol=w9c-$n; docker volume rm -f $vol >/dev/null 2>&1; docker volume create $vol >/dev/null
# CLONE=1: a copy of the warm glm53-tf-cache (Triton, calibration, extensions) minus the kda / exl3 extension builds,
# which the first container then rebuilds (what seq1 on the head node did before its IMA)
if [[ "${CLONE:-0}" == 1 ]]; then
  docker run --rm -v glm53-tf-cache:/src:ro -v $vol:/cache --entrypoint bash glm53-tensorfold:b2 -c \
    "cp -a /src/. /cache/ && rm -rf /cache/torch_extensions/tensorfold_glm_kda_v1 /cache/torch_extensions/tensorfold_glm_exl3_v2 /cache/roce-failed"
fi
run() { local nm=$1 t=$2 e=$3 T=$4; shift 4; local t0=$(date +%s)
  docker run --rm --name w9-$nm --gpus all -v $vol:/cache -e PYTHONDONTWRITEBYTECODE=1 $e -v $T:/work/tests \
    --entrypoint bash glm53-tensorfold:b2 -c \
    "pip install -q pytest >/dev/null 2>&1; cd /work && PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda timeout -k 30 $t $*" > $O/$nm.log 2>&1
  local rc=$?; docker rm -f w9-$nm >/dev/null 2>&1
  echo "$(date +%T) $nm rc=$rc $(( $(date +%s) - t0 ))s :: $(grep -E 'passed|failed|error|ERROR SUMMARY' $O/$nm.log | tail -1 | cut -c1-200)" | tee -a $O/SUMMARY
}
P="python -m pytest -q -p no:cacheprovider"
[[ $mode == twoproc ]] && run $n-twoproc 1500 "-e PP_TWO_PROC=1" $old $P tests/cuda/test_prefill_pp_patches.py -k two_engines
run $n-prefix 2400 "$env" $repo/tests $P tests/cuda/test_prefix_share_patches.py
docker run --rm -v $vol:/cache --entrypoint bash glm53-tensorfold:b2 -c "find /cache -maxdepth 2 | grep -v triton/ | sort" > $O/$n-cache.txt
echo "$n done $(date +%T)" | tee -a $O/SUMMARY
