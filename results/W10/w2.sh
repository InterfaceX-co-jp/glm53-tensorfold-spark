#!/usr/bin/env bash
# W10 window 2: S = 0410 alone, MSK = 0390 + 0410 + 0400 split together (per-load set), then M8 = 0390 with 8,192-row
# lone chunks through the gates without MMLU (the 4 x 250k stress decides whether memory allows 8,192)
cd $HOME/glm53-tensorfold-spark
R=results/W10
bash $R/loads.sh "S GLM53_TF_SPARSE_V2=1" "MSK GLM53_TF_MLA_EXPAND=v2 GLM53_TF_SPARSE_V2=1 GLM53_TF_KDA_V2=1"
NOMMLU=1 bash $R/gates.sh M8 GLM53_TF_MLA_EXPAND=v2 GLM53_TF_PREFILL_ROWS_MAX=8192 GLM53_TF_SOLO_PIECE=8192
echo "w2 done $(date +%T)"
