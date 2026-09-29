#!/usr/bin/env bash
# W11 window 2, after the captures: plain loads (no nsys) for the 4-stream graph policy and the acceptance baseline.
#   DEF  = prod (CAPTURE_AFTER=3)  -> graphs.sh DEF, then the acceptance set (w11req.py accept)
#   G0   = GLM53_TF_BATCH_GRAPHS=0 -> graphs.sh G0
#   CA8  = GLM53_TF_BATCH_CAPTURE_AFTER=8 -> graphs.sh CA8
#   DEF2 = prod again (same-window control) -> graphs.sh DEF2
cd $HOME/glm53-tensorfold-spark
R=results/W11
run() { name=$1; shift; bash $R/load.sh $name plain "$@"; grep -E "batching|only .* fit" $R/boot-$name-r0.log; bash $R/graphs.sh $name; }
run DEF
python3 $R/w11req.py accept $R/accept-DEF.jsonl 512 > $R/accept-DEF.log 2>&1; echo "accept done $(date +%T)"
run G0 GLM53_TF_BATCH_GRAPHS=0
run CA8 GLM53_TF_BATCH_CAPTURE_AFTER=8
run DEF2
echo "AB DONE $(date +%T)"
