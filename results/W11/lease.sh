#!/usr/bin/env bash
# W11: refresh the test-window lease every 4 min while it exists, for at most 6 h (deleting the lease ends it)
L=$HOME/.test-window-lease
for i in $(seq 1 90); do [ -f "$L" ] || exit 0; touch "$L"; sleep 240; done
