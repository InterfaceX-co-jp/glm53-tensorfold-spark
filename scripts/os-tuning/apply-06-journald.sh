#!/usr/bin/env bash
# 06: cap the persistent journal at 1 GiB (docs/OS-TUNING.md, "06 journald"). Disk only; journald's RAM is page cache.
# Frequent SSH sessions (e.g. a dashboard polling over Tailscale SSH) can write thousands of lines a minute.
#   sudo scripts/os-tuning/apply-06-journald.sh        rollback: rollback-06-journald.sh
source "$(dirname "$0")/lib.sh"; need_root
d=/etc/systemd/journald.conf.d; run mkdir -p $d
if [[ "$DRY_RUN" == 1 ]]; then echo "DRY: write $d/90-spark.conf"; else cat > $d/90-spark.conf <<'CONF'
# spark os-tuning 06 (docs/OS-TUNING.md)
[Journal]
SystemMaxUse=1G
SystemMaxFileSize=128M
CONF
fi
run systemctl restart systemd-journald
run journalctl --vacuum-size=1G
