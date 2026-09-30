#!/usr/bin/env bash
# The recommended set in order (docs/OS-TUNING.md): 01 polkit, 02 headless (next boot), 03 services tier A,
# 04 sysctl, 05 cpu/irq, 06 journald. Not 07 (CX7 MTU, optional) and not TIER=B / CPUIDLE=1 unless exported.
#   sudo scripts/os-tuning/apply-all.sh        (one node; run on the worker first, then the head)
#   sudo DRY_RUN=1 scripts/os-tuning/apply-all.sh   preview
source "$(dirname "$0")/lib.sh"; need_root
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$STATE_DIR"; WAKE_S=5 bash "$here/snapshot.sh" "$STATE_DIR/snapshot-before-$(date +%Y%m%d-%H%M%S).txt"
for s in 01-polkit 02-headless 03-services 04-sysctl 05-cpu-irq 06-journald; do
    log "=== apply-$s"; bash "$here/apply-$s.sh" || die "apply-$s failed: fix it or run rollback-all.sh"
done
assert_network_kept
WAKE_S=5 bash "$here/snapshot.sh" "$STATE_DIR/snapshot-after-$(date +%Y%m%d-%H%M%S).txt"
log "applied; reboot this node in a planned window to finish the headless switch (docs/OS-TUNING.md, reboot test)"
