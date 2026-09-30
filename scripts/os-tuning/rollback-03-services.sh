#!/usr/bin/env bash
# rollback 03: re-enable and restart exactly the units apply-03 found enabled / running; unblock Bluetooth; unhold snaps.
source "$(dirname "$0")/lib.sh"; need_root
f=$(state_file 03-services)
[[ -f $f ]] || die "no $f"
grep -vE '^(snap-hold|rfkill-bluetooth) ' "$f" > "$f.units"
restore_units "$f.units"
if grep -q '^rfkill-bluetooth unblocked' "$f"; then run rfkill unblock bluetooth; log "bluetooth unblocked"; fi
if grep -q '^snap-hold' "$f"; then run snap refresh --unhold >/dev/null 2>&1 && log "snap refreshes unheld"; fi
run rm -f "$f" "$f.units"
assert_network_kept
