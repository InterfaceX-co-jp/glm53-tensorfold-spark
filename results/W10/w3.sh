#!/usr/bin/env bash
# W10 window 3 (decode, on top of 0390 v2): MA = control (deep.sh: glmbench tf,tweet,kit,edit x3 + exact + batchexact +
# concurrent), D = + DECODE_OVERLAP=1 (dec.sh), C = + CPU_PIN=auto (dec.sh pin), R16 = + MAX_DRAFT_ROWS=16 (deep.sh)
cd $HOME/glm53-tensorfold-spark
R=results/W10; MV=GLM53_TF_MLA_EXPAND=v2
one() { local kind=$1 name=$2; shift 2; bash $R/load.sh $name "$@" || true
  curl -s -m 5 http://127.0.0.1:8000/health | grep -q '"ok": true' || { echo "LOAD $name NOT HEALTHY"; return 1; }
  if [[ $kind == deep ]]; then bash $R/deep.sh $name; else bash $R/dec.sh $name $kind; fi; }
one deep MA $MV
one x D $MV GLM53_TF_DECODE_OVERLAP=1
one pin C $MV GLM53_TF_CPU_PIN=auto
one deep R16 $MV GLM53_TF_MAX_DRAFT_ROWS=16
echo "w3 done $(date +%T)"
