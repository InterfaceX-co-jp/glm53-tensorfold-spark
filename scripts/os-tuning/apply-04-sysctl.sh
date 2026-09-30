#!/usr/bin/env bash
# 04: memory-writeback and scheduler sysctls (docs/OS-TUNING.md, "04 sysctl"). Live now and at boot (/etc/sysctl.d).
#   sudo scripts/os-tuning/apply-04-sysctl.sh        rollback: rollback-04-sysctl.sh
# - dirty limits in bytes: the default ratios (20 % / 10 % of dirtyable memory) allow tens of GiB of dirty page cache
#   on an idle 128 GiB box (e.g. while writing prepared weights) and several GiB while serving. Fixed caps (256 MiB
#   background / 1 GiB hard) make writeback early and bounded.
# - vm.stat_interval 10: the per-cpu vmstat fold runs every 10 s instead of 1 s (fewer kworker wakeups on every core).
# - kernel.sched_autogroup_enabled 0: no per-session task groups (a server, not a desktop).
# Not changed: swappiness (swapping out idle daemons raises MemAvailable), numa_balancing, watchdogs.
source "$(dirname "$0")/lib.sh"; need_root
keys=(vm.dirty_background_bytes vm.dirty_bytes vm.dirty_background_ratio vm.dirty_ratio vm.stat_interval kernel.sched_autogroup_enabled)
f=$(state_file 04-sysctl)
[[ -f $f ]] || for k in "${keys[@]}"; do echo "$k $(sysctl -n "$k")"; done > "$f"
conf=/etc/sysctl.d/90-spark-os-tuning.conf
if [[ "$DRY_RUN" == 1 ]]; then echo "DRY: write $conf"; else cat > $conf <<'CONF'
# spark os-tuning 04 (docs/OS-TUNING.md). Rollback: scripts/os-tuning/rollback-04-sysctl.sh
vm.dirty_background_bytes = 268435456
vm.dirty_bytes = 1073741824
vm.stat_interval = 10
kernel.sched_autogroup_enabled = 0
CONF
fi
run sysctl -p $conf
