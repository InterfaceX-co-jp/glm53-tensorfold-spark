#!/usr/bin/env bash
# refresh the test-window lease every 5 min while it exists, for at most 90 min (W9 copy of W8) (deleting the lease ends it)
L=$HOME/.test-window-lease
for i in $(seq 1 18); do [ -f "$L" ] || exit 0; touch "$L"; sleep 300; done
