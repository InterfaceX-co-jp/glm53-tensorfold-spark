#!/usr/bin/env bash
# W9 window 4: I = prod + RoCE + b12x bit 4 + 4,096-row chunks for lone requests (W8's documented memory step back),
# full gates; then H = prod unchanged, the same sequence without MMLU (today's baseline for the 4 x 250k minimum).
cd $HOME/glm53-tensorfold-spark
R=results/W9
bash $R/gates.sh I GLM53_TF_COMM_BACKEND=roce GLM53_TF_ROCE_MARK=/cache/roce-failed GLM53_TF_B12X=4 GLM53_TF_PREFILL_ROWS_MAX=4096 GLM53_TF_SOLO_PIECE=4096 > $R/gates-I.out 2>&1
NOMMLU=1 bash $R/gates.sh H > $R/gates-H.out 2>&1
echo "w4 done $(date +%T)" | tee -a $R/loads.log
