#!/usr/bin/env bash
# W10: after the running loads.sh, start another list if the clock is before DEADLINE (HH:MM): then.sh HH:MM OUT spec...
cd $HOME/glm53-tensorfold-spark
dl=$1; out=$2; shift 2
while pgrep -f "results/W10/loads.sh" >/dev/null; do sleep 5; done
if [[ "$(date +%H:%M)" < "$dl" ]]; then bash results/W10/loads.sh "$@" > results/W10/$out 2>&1; else echo "skipped: past $dl" > results/W10/$out; fi
