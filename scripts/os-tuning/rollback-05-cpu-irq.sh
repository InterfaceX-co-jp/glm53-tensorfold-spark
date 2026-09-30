#!/usr/bin/env bash
# rollback 05: drop-ins removed, service / slice affinity back to all cpus, IRQ affinities and cpuidle restored.
source "$(dirname "$0")/lib.sh"; need_root
f=$(state_file 05-cpu-irq); [[ -f $f ]] || die "no $f"
all="0-$(( $(nproc --all) - 1 ))"
run systemctl disable --now spark-os-tuning.service 2>/dev/null
run rm -f /etc/systemd/system/spark-os-tuning.service /usr/local/sbin/spark-os-tuning-runtime.sh /etc/default/spark-os-tuning
for c in /etc/systemd/system/*.service.d/90-spark-cpu.conf; do
    [[ -f $c ]] || continue
    s=$(basename "$(dirname "$c")" .d); run rm -f "$c"; run rmdir "$(dirname "$c")" 2>/dev/null
    for p in $(cat "/sys/fs/cgroup/system.slice/$s/cgroup.procs" 2>/dev/null); do run taskset -a -pc "$all" "$p" >/dev/null 2>&1; done
done
run rm -f /etc/systemd/system/user.slice.d/90-spark-cpu.conf; run rmdir /etc/systemd/system/user.slice.d 2>/dev/null
run systemctl daemon-reload
run systemctl set-property --runtime user.slice AllowedCPUs=
while read -r k a b; do
    case $k in
        default_smp_affinity) [[ "$DRY_RUN" == 1 ]] || echo "$a" > /proc/irq/default_smp_affinity ;;
        irq) [[ -n "$b" && -w /proc/irq/$a/smp_affinity_list ]] && { [[ "$DRY_RUN" == 1 ]] || echo "$b" > "/proc/irq/$a/smp_affinity_list" 2>/dev/null; } ;;
    esac
done < "$f"
for s in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do [[ "$DRY_RUN" == 1 ]] || echo 0 > "$s"; done
run rm -f "$f"
log "cpu / irq tuning rolled back"
