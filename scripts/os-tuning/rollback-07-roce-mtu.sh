#!/usr/bin/env bash
# rollback 07: CX7 MTU back to the recorded values and the netplan file back from its backup.
source "$(dirname "$0")/lib.sh"; need_root
f=$(state_file 07-roce-mtu); [[ -f $f ]] || die "no $f"
server_running_here && die "a rank container is running here: scripts/serve.sh stop first (both nodes)"
while read -r i m; do
    [[ "$i" == netplan ]] && continue
    log "$i mtu -> $m"; run ip link set dev "$i" mtu "$m"
done < "$f"
y=$(awk '$1=="netplan"{print $2}' "$f" | tail -1)
if [[ -n "$y" && -f $y.pre-spark ]]; then run cp -p "$y.pre-spark" "$y"; run rm -f "$y.pre-spark"; run netplan generate; log "netplan restored"; fi
run rm -f "$f"
