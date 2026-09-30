#!/usr/bin/env bash
# rollback 06: journald back to the distribution defaults.
source "$(dirname "$0")/lib.sh"; need_root
run rm -f /etc/systemd/journald.conf.d/90-spark.conf
run systemctl restart systemd-journald
log "journald defaults restored"
