#!/usr/bin/env bash
# rollback 02: back to the recorded default target (usually graphical.target) with gdm (START=1 also starts gdm now).
source "$(dirname "$0")/lib.sh"; need_root
f=$(state_file 02-headless); d=$(state_file 02-headless-default)
def=$(cat "$d" 2>/dev/null); def=${def:-graphical.target}
log "set-default $def"; run systemctl set-default "$def"
if [[ "${START:-0}" == 1 ]]; then restore_units "$f"
else [[ -f $f ]] && while read -r u en _; do [[ "$en" == enabled ]] && run systemctl enable "$u"; done < "$f"; fi
run rm -f "$f" "$d"
log "graphical boot restored (active from the next boot, or now with START=1)"
