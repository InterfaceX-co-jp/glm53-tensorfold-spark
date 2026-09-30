#!/usr/bin/env bash
# 02: boot to multi-user.target with the display manager off (docs/OS-TUNING.md, "02 headless").
#   sudo scripts/os-tuning/apply-02-headless.sh          default target + gdm disabled; takes effect at the next boot
#   sudo NOW=1 scripts/os-tuning/apply-02-headless.sh    also stops gdm now (ends any graphical session)
#   rollback: rollback-02-headless.sh
# Pre-checks (refuses on failure): the default route is up; every Wi-Fi profile (if any) is a system connection
# (no connection.permissions, PSK stored in the profile: psk-flags 0, autoconnect on), so NetworkManager can join it at
# boot without a desktop keyring; with LINGER_USER=<user> (head), linger is on for the owner of the user units.
source "$(dirname "$0")/lib.sh"; need_root
assert_network_kept
bad=0
while IFS=: read -r uuid type; do
    [[ "$type" == 802-11-wireless ]] || continue
    perm=$(nmcli -g connection.permissions con show "$uuid" 2>/dev/null)
    flags=$(nmcli -g 802-11-wireless-security.psk-flags con show "$uuid" 2>/dev/null)
    auto=$(nmcli -g connection.autoconnect con show "$uuid" 2>/dev/null)
    name=$(nmcli -g connection.id con show "$uuid" 2>/dev/null)
    log "wifi profile '$name': permissions='${perm}' psk-flags='${flags}' autoconnect=${auto}"
    if [[ -n "$perm" || ( -n "$flags" && "$flags" != 0 && "$flags" != "0 (none)" ) || "$auto" != yes ]]; then
        log "FIX FIRST: nmcli con mod $uuid connection.permissions '' 802-11-wireless-security.psk-flags 0 802-11-wireless-security.psk '<PSK>' connection.autoconnect yes"
        bad=1
    fi
done < <(nmcli -t -f UUID,TYPE con show)
[[ $bad == 0 ]] || die "a Wi-Fi profile depends on a desktop user; fix it (command above) and re-run"
if [[ -n "${LINGER_USER:-}" ]]; then
    [[ -f /var/lib/systemd/linger/$LINGER_USER ]] || die "linger is off for $LINGER_USER: sudo loginctl enable-linger $LINGER_USER (user units must run without a login)"
    log "linger on for $LINGER_USER"
fi
f=$(state_file 02-headless)
d=$(state_file 02-headless-default); [[ -f $d ]] || systemctl get-default > "$d"
record_units "$f" gdm.service gdm3.service gnome-remote-desktop.service
log "set-default multi-user.target"
run systemctl set-default multi-user.target
disable_units gdm.service gnome-remote-desktop.service
if [[ "${NOW:-0}" == 1 ]]; then
    log "stopping gdm now (ends the graphical session)"
    run systemctl stop gdm.service
    sleep 3; assert_network_kept
else
    log "gdm stays up until the next boot (NOW=1 stops it now)"
fi
