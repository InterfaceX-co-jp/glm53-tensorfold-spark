#!/usr/bin/env bash
# rollback 04: remove the sysctl file and put the recorded values back (ratios after bytes: setting one zeroes the other).
source "$(dirname "$0")/lib.sh"; need_root
f=$(state_file 04-sysctl); [[ -f $f ]] || die "no $f"
run rm -f /etc/sysctl.d/90-spark-os-tuning.conf
for k in vm.stat_interval kernel.sched_autogroup_enabled vm.dirty_background_ratio vm.dirty_ratio; do
    v=$(awk -v k="$k" '$1==k{print $2}' "$f"); [[ -n "$v" ]] && run sysctl -w "$k=$v"
done
for k in vm.dirty_background_bytes vm.dirty_bytes; do   # only if they were in use (non-zero) before
    v=$(awk -v k="$k" '$1==k{print $2}' "$f"); [[ -n "$v" && "$v" != 0 ]] && run sysctl -w "$k=$v"
done
run rm -f "$f"
sysctl vm.dirty_ratio vm.dirty_background_ratio vm.dirty_bytes vm.stat_interval kernel.sched_autogroup_enabled
