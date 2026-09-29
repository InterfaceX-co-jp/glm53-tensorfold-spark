#!/usr/bin/env bash
# refresh the test-window lease every 5 min while it exists, for at most 75 min (deleting the lease ends it)
L=$HOME/.test-window-lease
for i in $(seq 1 15); do [ -f "$L" ] || exit 0; touch "$L"; sleep 300; done
