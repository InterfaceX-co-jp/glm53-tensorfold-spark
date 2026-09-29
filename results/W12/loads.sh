#!/usr/bin/env bash
# W12: a sequence of loads, each = load.sh + ab.sh (+ api.py and N1 on loads named C*): loads.sh "NAME K=V ..." ...
# A load that fails to start or is unhealthy stops the list.
cd $HOME/glm53-tensorfold-spark
R=results/W12
for spec in "$@"; do
  set -- $spec; name=$1; shift
  bash $R/load.sh $name "$@" || true
  if ! curl -s -m 5 http://127.0.0.1:8000/health | grep -q '"ok": true'; then echo "LOAD $name NOT HEALTHY: stop"; exit 1; fi
  if [[ $name == C* ]]; then python3 $R/api.py $R/api-$name.json > $R/api-$name.log 2>&1; cat $R/api-$name.log; fi
  bash $R/ab.sh $name
  if [[ $name == C || $name == N1* ]]; then
    python3 bench/longexact.py --corpus $R/n1-corpus.txt --out $R/n1-$name.json > $R/n1-$name.log 2>&1; grep '^N1' $R/n1-$name.log
  fi
done
echo "loads done $(date +%T)"
