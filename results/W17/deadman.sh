#!/usr/bin/env bash
# W17 failsafe: at 05:10 (+07), if the test-window lease still exists, restore prod (restore.sh deadman)
cd $HOME/glm53-tensorfold-spark
while [ "$(date +%H%M)" != "0510" ]; do sleep 20; done
[ -f $HOME/.test-window-lease ] && bash results/W17/restore.sh deadman > results/W17/deadman.out 2>&1
