#!/usr/bin/env bash
# W10 window 3b: after w3.sh, D2 = 0390 + DECODE_OVERLAP=1 again (repeat of the decode set) + the cancel / disconnect check,
# then MA2 = 0390 alone repeated (decode control for D2, same order effects)
cd $HOME/glm53-tensorfold-spark
R=results/W10
while pgrep -f "results/W10/w3.sh" >/dev/null; do sleep 5; done
bash $R/load.sh D2 GLM53_TF_MLA_EXPAND=v2 GLM53_TF_DECODE_OVERLAP=1
python3 $R/cancel.py $R/cancel-D2.json 2>&1 | tail -2
bash $R/dec.sh D2
python3 $R/cancel.py $R/cancel-D2b.json 2>&1 | tail -2
bash $R/load.sh MA2 GLM53_TF_MLA_EXPAND=v2
bash $R/dec.sh MA2
echo "w3b done $(date +%T)"
