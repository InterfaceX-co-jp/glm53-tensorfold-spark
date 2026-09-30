#!/usr/bin/env bash
# 05: keep OS noise off the engine's cores (docs/OS-TUNING.md, "05 CPU / IRQ"). Housekeeping = the 10 Cortex-A725
# cores of a GB10 (0-4,10-14); the engine keeps the 10 Cortex-X925 cores (5-9,15-19).
#   sudo scripts/os-tuning/apply-05-cpu-irq.sh              IRQs + service affinity (now + at every boot)
#   sudo CPUIDLE=1 scripts/os-tuning/apply-05-cpu-irq.sh    + deepest idle state (LPI-3) off on the X925s (experiment)
#   rollback: rollback-05-cpu-irq.sh
# What it does:
#  1. /etc/systemd/system/<svc>.service.d/90-spark-cpu.conf: CPUAffinity=<housekeeping> for chatty system services
#     (polkit, dbus, logind, journald, NetworkManager, tailscaled, sshd, ...), applied live with taskset -a.
#     NOT docker/containerd: containers inherit containerd-shim's affinity, and without a container cpuset the engine
#     would land on the A725s.
#  2. user.slice AllowedCPUs=<housekeeping>: ssh sessions, benchmark clients, user units.
#  3. every movable IRQ + the default IRQ affinity -> housekeeping cores, persisted by spark-os-tuning.service.
#     (irqbalance is not installed on a stock Spark; if you run it, it will move them back.)
source "$(dirname "$0")/lib.sh"; need_root
here=$(cd "$(dirname "$0")" && pwd)
SVCS=(polkit dbus systemd-logind systemd-journald systemd-udevd NetworkManager wpa_supplicant tailscaled ssh rsyslog snapd fwupd
      udisks2 cron anacron earlyoom systemd-resolved systemd-networkd systemd-timesyncd lldpd rasdaemon smartmontools
      multipathd nvidia-dgx-telemetry dgx-dashboard dgx-dashboard-admin accounts-daemon upower ModemManager avahi-daemon)
f=$(state_file 05-cpu-irq)
if [[ ! -f $f ]]; then
    { echo "default_smp_affinity $(cat /proc/irq/default_smp_affinity)"
      for d in /proc/irq/[0-9]*; do echo "irq ${d##*/} $(cat "$d"/smp_affinity_list 2>/dev/null)"; done; } > "$f"
fi
aff=${HOUSEKEEPING_CPUS//,/ }
for s in "${SVCS[@]}"; do
    unit_exists "$s.service" || continue
    d=/etc/systemd/system/$s.service.d; run mkdir -p "$d"
    if [[ "$DRY_RUN" == 1 ]]; then echo "DRY: $d/90-spark-cpu.conf"; else
        printf '# spark os-tuning 05 (docs/OS-TUNING.md)\n[Service]\nCPUAffinity=%s\n' "$aff" > "$d/90-spark-cpu.conf"; fi
    for p in $(systemctl show -p MainPID --value "$s.service" 2>/dev/null) $(cat "/sys/fs/cgroup/system.slice/$s.service/cgroup.procs" 2>/dev/null); do
        [[ "$p" =~ ^[1-9][0-9]*$ ]] && run taskset -a -pc "$HOUSEKEEPING_CPUS" "$p" >/dev/null 2>&1
    done
done
d=/etc/systemd/system/user.slice.d; run mkdir -p $d
[[ "$DRY_RUN" == 1 ]] || printf '# spark os-tuning 05\n[Slice]\nAllowedCPUs=%s\n' "$HOUSEKEEPING_CPUS" > $d/90-spark-cpu.conf
run systemctl daemon-reload
run systemctl set-property --runtime user.slice AllowedCPUs="$HOUSEKEEPING_CPUS"
# boot-time IRQ / cpuidle part
run install -m 0755 "$here/runtime.sh" /usr/local/sbin/spark-os-tuning-runtime.sh
[[ "$DRY_RUN" == 1 ]] || printf 'HOUSEKEEPING_CPUS=%s\nENGINE_CPUS=%s\nCPUIDLE=%s\n' "$HOUSEKEEPING_CPUS" "$ENGINE_CPUS" "${CPUIDLE:-0}" > /etc/default/spark-os-tuning
if [[ "$DRY_RUN" == 1 ]]; then echo "DRY: write spark-os-tuning.service"; else cat > /etc/systemd/system/spark-os-tuning.service <<'UNIT'
[Unit]
Description=spark os-tuning 05: IRQs to the housekeeping cores (+ optional cpuidle), docs/OS-TUNING.md
After=network-online.target NetworkManager.service
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/sbin/spark-os-tuning-runtime.sh

[Install]
WantedBy=multi-user.target
UNIT
fi
run systemctl daemon-reload
run systemctl enable --now spark-os-tuning.service
log "done: $(journalctl -u spark-os-tuning -n 2 --no-pager -o cat 2>/dev/null | tr '\n' ' ')"
