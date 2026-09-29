#!/usr/bin/env bash
# W16 window: THEORY2-SESSION (Part A) then the drafter arms (Part B) inside it, then restore (run.sh all EXIT trap).
cd $HOME/glm53-tensorfold-spark
export WINDOW_MAX_MIN=330 SESSION_MAX_MIN=150 LOADS="C L CT K VS H GR"
export AFTER_LOADS="DS_DEADLINE=1790709600 ARMS=\"inco-7d7 modal inco-bf5 stock\" bash results/W16/drafters.sh"
exec bash results/THEORY2-SESSION/run.sh all
