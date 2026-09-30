#!/usr/bin/env bash
# 03: disable services / timers a headless inference pair does not use (docs/OS-TUNING.md, "03 services").
#   sudo scripts/os-tuning/apply-03-services.sh            tier A (safe on both nodes)
#   sudo TIER=B scripts/os-tuning/apply-03-services.sh     + tier B (rsyslog, lldpd, sysstat, apt timers)
#   EXTRA_KEEP="unit1 unit2"                               never disable these even if listed
#   rollback: rollback-03-services.sh (re-enables / restarts exactly what was enabled / running before)
# NEVER touched: ssh, tailscaled, NetworkManager, wpa_supplicant, systemd-networkd/resolved, docker/containerd,
# nvidia-* / dgx-* units, rasdaemon, smartmontools, earlyoom, cron/anacron, logrotate, fstrim, user units.
source "$(dirname "$0")/lib.sh"; need_root
assert_network_kept
TIER=${TIER:-A}
A=(
  cups.service cups.socket cups.path cups-browsed.service          # printing
  avahi-daemon.service avahi-daemon.socket                         # mDNS/.local (reach the nodes by IP / DNS instead)
  bluetooth.service                                                # + rfkill block below
  ModemManager.service                                             # no WWAN hardware
  fwupd.service fwupd-refresh.timer                                # fwupdmgr still works by hand (D-Bus activation)
  snapd.service snapd.socket snapd.seeded.service snapd.snap-repair.timer   # stock image: desktop snaps only
  udisks2.service upower.service switcheroo-control.service colord.service accounts-daemon.service rtkit-daemon.service
  apport.service apport-autoreport.timer apport-autoreport.path apport-forward.socket whoopsie.service kerneloops.service
  motd-news.timer ua-timer.timer update-notifier-download.timer update-notifier-motd.timer man-db.timer
)
B=(
  rsyslog.service                                                  # journald keeps the logs
  lldpd.service                                                    # LLDP on a point-to-point cable
  sysstat.service sysstat-collect.timer sysstat-summary.timer
  apt-daily.timer apt-daily-upgrade.timer                          # manual updates only
)
units=("${A[@]}"); [[ "$TIER" == B ]] && units+=("${B[@]}")
# multipathd only if nothing is on a multipath device (root is on NVMe on a stock Spark)
if ! lsblk -no TYPE 2>/dev/null | grep -q mpath; then units+=(multipathd.service multipathd.socket); else log "keep multipathd: an mpath device exists"; fi
if [[ -n "${EXTRA_KEEP:-}" ]]; then
    keep=" $EXTRA_KEEP "; kept=()
    for u in "${units[@]}"; do [[ "$keep" == *" $u "* || "$keep" == *" ${u%.*} "* ]] || kept+=("$u"); done
    units=("${kept[@]}")
fi
f=$(state_file 03-services)
record_units "$f" "${units[@]}"
# freeze snap refreshes before snapd goes (manual: snap refresh --unhold; snap refresh)
if command -v snap >/dev/null && systemctl is-active -q snapd; then
    grep -q '^snap-hold' "$f" 2>/dev/null || echo "snap-hold set" >> "$f"
    run snap refresh --hold >/dev/null 2>&1 && log "snap refreshes held"
fi
# Bluetooth radio off (soft block); remember the previous state
if command -v rfkill >/dev/null; then
    grep -q '^rfkill-bluetooth' "$f" 2>/dev/null || echo "rfkill-bluetooth $(rfkill -no SOFT -r list bluetooth 2>/dev/null | head -1)" >> "$f"
    run rfkill block bluetooth && log "bluetooth radio blocked"
fi
before=$(awk '/MemAvailable/{printf "%.2f",$2/1048576}' /proc/meminfo)
disable_units "${units[@]}"
assert_network_kept
log "MemAvailable $before -> $(awk '/MemAvailable/{printf "%.2f",$2/1048576}' /proc/meminfo) GiB"
