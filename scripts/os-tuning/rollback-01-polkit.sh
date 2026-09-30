#!/usr/bin/env bash
# rollback 01: remove the polkit memory cap (the leaked memory itself is not "restored").
source "$(dirname "$0")/lib.sh"; need_root
run rm -f /etc/systemd/system/polkit.service.d/90-spark-memory.conf
run rmdir /etc/systemd/system/polkit.service.d 2>/dev/null
run systemctl daemon-reload
run systemctl restart polkit
rm -f "$(state_file 01-polkit)"
log "polkit drop-in removed"
