#!/usr/bin/env bash
# W9: poison_engine.py in the b2 image: poison.sh NAME BYTE GIB ROUNDS [docker env]
cd ${REPO:-$HOME/glm53-tensorfold-spark}; O=results/W9/ima; mkdir -p $O
n=$1; t0=$(date +%s)
docker run --rm --name w9-$n --gpus all -v glm53-tf-cache:/cache -e PYTHONDONTWRITEBYTECODE=1 ${5:-} -v $PWD/tests:/work/tests -v $PWD/results/W9:/work/w9 \
  --entrypoint bash glm53-tensorfold:b2 -c "pip install -q pytest >/dev/null 2>&1; cd /work && timeout -k 30 900 python w9/poison_engine.py $2 $3 $4" > $O/$n.log 2>&1
echo "$(date +%T) $n rc=$? $(( $(date +%s) - t0 ))s :: $(grep -E 'POISON OK|Error|illegal' $O/$n.log | tail -1 | cut -c1-200)" | tee -a $O/SUMMARY
