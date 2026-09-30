#!/usr/bin/env bash
# 01: reclaim polkitd's leaked memory and cap it so it cannot grow back (docs/OS-TUNING.md, "01 polkit").
#   sudo scripts/os-tuning/apply-01-polkit.sh        rollback: rollback-01-polkit.sh
# Drop-in: MemoryMax=512M (a fresh polkitd uses ~10 MiB), no swap, restart on failure. A cgroup OOM kill of polkitd is
# harmless: it is D-Bus activated and restarted; clients (logind, NetworkManager, a desktop shell) retry.
# Restarting polkitd frees the memory at once; it drops pending interactive authorisations (none on a headless box).
# Safe while serving: the inference engine does not use polkit.
source "$(dirname "$0")/lib.sh"; need_root
f=$(state_file 01-polkit)
[[ -f $f ]] || { echo "rss_kib $(ps -o rss= -C polkitd | head -1)" > "$f"; }
d=/etc/systemd/system/polkit.service.d
run mkdir -p $d
if [[ "$DRY_RUN" == 1 ]]; then echo "DRY: write $d/90-spark-memory.conf"; else cat > $d/90-spark-memory.conf <<'CONF'
# spark os-tuning 01 (docs/OS-TUNING.md): polkitd leaks under session churn; cap it and let systemd restart it
[Service]
MemoryMax=512M
MemorySwapMax=0
Restart=on-failure
RestartSec=2
CONF
fi
# measure BEFORE daemon-reload: the reload applies MemoryMax to the running polkitd at once (the kernel reclaims it
# down to 512M before the restart), so a later reading under-reports what the restart frees
before=$(awk '/MemAvailable/{printf "%.2f",$2/1048576}' /proc/meminfo)
log "restart polkit (rss before: $(ps -o rss= -C polkitd | awk '{printf "%.2f GiB",$1/1048576}'), swap $(awk '/VmSwap/{printf "%.2f GiB",$2/1048576}' /proc/"$(pgrep -xo polkitd)"/status 2>/dev/null))"
run systemctl daemon-reload
run systemctl restart polkit
sleep 2
log "MemAvailable $before -> $(awk '/MemAvailable/{printf "%.2f",$2/1048576}' /proc/meminfo) GiB; polkitd rss now $(ps -o rss= -C polkitd | awk '{printf "%.0f MiB",$1/1024}')"
