#!/usr/bin/env bash
# 07 (optional): jumbo MTU on the two CX7 link ports, so RoCE's active_mtu goes 1024 -> 4096
# (docs/OS-TUNING.md, "07 CX7 MTU"). Measured ~neutral for speed and costs ~0.75-1.4 GiB per node; your call.
# Must be done on BOTH nodes with the rank containers stopped (queue pairs pick the MTU at creation).
# The uplink is not touched.
#   sudo CX7_IFS="<if1> <if2>" scripts/os-tuning/apply-07-roce-mtu.sh      runtime only (gone after reboot)
#   sudo PERSIST=1 NETPLAN_FILE=/etc/netplan/<your-cx7>.yaml ...           + "mtu: 9000" in that netplan file
#                                                                          (netplan generate, no netplan apply)
#   PEER_IPS="<peer-cx7-ip-1> <peer-cx7-ip-2>"   optional: printed as a ping check
#   rollback: rollback-07-roce-mtu.sh
source "$(dirname "$0")/lib.sh"; need_root
[[ -n "${CX7_IFS:-}" ]] || die "set CX7_IFS to the CX7 link interfaces (see: ip -br link; rdma link)"
read -ra IFS_CX7 <<< "$CX7_IFS"; MTU=${MTU:-9000}
server_running_here && die "a rank container is running here: scripts/serve.sh stop first (both nodes)"
f=$(state_file 07-roce-mtu)
[[ -f $f ]] || for i in "${IFS_CX7[@]}"; do echo "$i $(cat "/sys/class/net/$i/mtu")"; done > "$f"
for i in "${IFS_CX7[@]}"; do log "$i mtu -> $MTU"; run ip link set dev "$i" mtu "$MTU"; done
if [[ "${PERSIST:-0}" == 1 ]]; then
    y=${NETPLAN_FILE:?set NETPLAN_FILE to the netplan file that defines the CX7 interfaces}
    [[ -f $y ]] || die "no $y"
    echo "netplan $y" >> "$f"
    [[ -f $y.pre-spark ]] || run cp -p "$y" "$y.pre-spark"
    # adds / replaces "mtu:" under each listed interface of a standard netplan layout (ethernets: <if>: at 4 spaces)
    [[ "$DRY_RUN" == 1 ]] || python3 - "$y" "$MTU" "${IFS_CX7[@]}" <<'PY'
import sys, re
path, mtu, ifs = sys.argv[1], sys.argv[2], sys.argv[3:]
out, cur = [], None
for line in open(path).read().splitlines():
    m = re.match(r"^    (\S+):\s*$", line)
    if m: cur = m.group(1)
    if cur in ifs and re.match(r"^      mtu:", line): continue
    out.append(line)
    if m and cur in ifs: out.append(f"      mtu: {mtu}")
open(path, "w").write("\n".join(out) + "\n")
PY
    run netplan generate
    log "persisted in $y (backup $y.pre-spark); active at the next boot / NM reload"
fi
for d in /sys/class/infiniband/*; do
    n=$(basename "$d"); st=$(cat "$d"/ports/1/state 2>/dev/null)
    [[ "$st" == *ACTIVE* ]] && log "$n active_mtu: $(ibv_devinfo -d "$n" 2>/dev/null | awk '/active_mtu/{print $2,$3}')"
done
for p in ${PEER_IPS:-}; do log "check: ping -M do -s $((MTU - 28)) -c 3 $p"; done
