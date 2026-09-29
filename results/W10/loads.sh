#!/usr/bin/env bash
# W10: a sequence of loads, each = load.sh + run.sh: loads.sh "NAME K=V ..." "NAME2 ..." (a load that fails to start stops the list)
cd $HOME/glm53-tensorfold-spark
R=results/W10
for spec in "$@"; do
  set -- $spec; name=$1; shift
  bash $R/load.sh $name "$@" || true
  if ! curl -s -m 5 http://127.0.0.1:8000/health | grep -q '"ok": true'; then echo "LOAD $name NOT HEALTHY: stop"; exit 1; fi
  bash $R/run.sh $name
done
echo "loads done $(date +%T)"
