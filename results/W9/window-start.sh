#!/usr/bin/env bash
# W9: open a test window on the head node: lease + refresher, watchdog timer stopped, prod stopped. window-start.sh NAME
cd $HOME/glm53-tensorfold-spark
R=results/W9; n=${1:-w}
touch $HOME/.test-window-lease
nohup bash $R/lease.sh > /dev/null 2>&1 &
echo $! > $R/lease.pid
systemctl --user stop glm53-tf-watchdog.timer
systemctl --user is-active glm53-tf-watchdog.timer || true
CONFIG=config/prod.env scripts/serve.sh stop > $R/stop-$n.log 2>&1
echo "$n start $(date +%T) lease pid $(cat $R/lease.pid)" | tee -a $R/windows.log
docker ps --format '{{.Names}} {{.Image}}'; ssh -o BatchMode=yes $WORKER_SSH "docker ps --format '{{.Names}} {{.Image}}'"
