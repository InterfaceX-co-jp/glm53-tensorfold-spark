#!/usr/bin/env bash
# W9: the first hour of production on RoCE (docs/ROCE-FIX.md adoption rule): every 5 min health, errors, RoCE failure /
# fallback lines on both ranks, the marker, MemAvailable both nodes. Read-only.
cd $HOME/glm53-tensorfold-spark
for i in $(seq 1 13); do
  h=$(curl -s -m 10 http://127.0.0.1:8000/health | head -c 200)
  e0=$(docker logs glm53-tf-r0 2>&1 | grep -ciE "roce.*(timed out|failed|fallback|poison)|RoceError")
  e1=$(ssh -o BatchMode=yes $WORKER_SSH 'docker logs glm53-tf-r1 2>&1 | grep -ciE "roce.*(timed out|failed|fallback|poison)|RoceError"')
  mk=$(docker exec glm53-tf-r0 ls /cache/roce-failed 2>/dev/null || echo none)
  m0=$(awk '/MemAvailable/{printf "%.2f",$2/1048576}' /proc/meminfo)
  m1=$(ssh -o BatchMode=yes $WORKER_SSH "awk '/MemAvailable/{printf \"%.2f\",\$2/1048576}' /proc/meminfo")
  echo "$(date +%T) health $h | roce errors r0 $e0 r1 $e1 | marker $mk | MemAvailable $m0 / $m1"
  [ $i -lt 13 ] && sleep 300
done
