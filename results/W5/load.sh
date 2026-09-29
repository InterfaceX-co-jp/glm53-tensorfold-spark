#!/usr/bin/env bash
# W5: start a test load from config/prod.env on image w5 with extra GLM53_TF_* overrides: load.sh NAME [K=V ...]
cd "$(dirname "$0")/../.."
name=$1; shift
for kv in "$@"; do export "$kv"; done
export IMAGE=glm53-tensorfold:w5
CONFIG=config/prod.env scripts/serve.sh stop >/dev/null 2>&1
CONFIG=config/prod.env timeout 1200 scripts/serve.sh start > "results/W5/start-$name.log" 2>&1
echo "start rc=$? $(date +%T)"; tail -3 "results/W5/start-$name.log"
