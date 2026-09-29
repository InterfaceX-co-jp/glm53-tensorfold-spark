#!/usr/bin/env bash
# W11: open a window: lease + refresher (pid in lease.pid), watchdog timer stopped
cd $HOME/glm53-tensorfold-spark
R=results/W11
touch $HOME/.test-window-lease
setsid nohup bash $R/lease.sh > /dev/null 2>&1 < /dev/null & echo $! > $R/lease.pid
systemctl --user stop glm53-tf-watchdog.timer
echo "window start $(date +%T) lease pid $(cat $R/lease.pid) watchdog $(systemctl --user is-active glm53-tf-watchdog.timer)" | tee -a $R/windows.log
