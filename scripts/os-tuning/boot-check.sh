#!/usr/bin/env bash
# Boot-to-serving checklist after a (test) reboot or power loss; read-only. Run on the head; it checks the worker over
# ssh. Exit 0 only if every check passed.   (docs/OS-TUNING.md, "Reboot and power-loss recovery")
#   WORKER_SSH=<user>@<worker-cx7-ip> scripts/os-tuning/boot-check.sh [--wait MIN]
#     --wait MIN   first poll the local server's /v1/models for up to MIN minutes
# Environment (all optional except WORKER_SSH):
#   WORKER_SSH       ssh target of the worker (the same value scripts/serve.sh uses)
#   PORT             the server's local port (default 8000)
#   UPLINK_IF        the interface that must carry the default route (default: not checked, any default route is ok)
#   CX7_IFS          "<if1> <if2>": link interfaces that must have an IPv4 address
#   USER_TIMERS      "a.timer b.timer": systemd --user timers on the head that must be active (e.g. the watchdog)
#   LINGER_USER      user that must have linger on (head)
#   PUBLIC_URL       e.g. https://<head>.<your-tailnet>/v1/models: also checked if set
#   OLD_CONTAINERS   regex of containers that must NOT be running after boot (old stacks with a restart policy)
#   RANK0 / RANK1    rank container names (default glm53-tf-r0 / glm53-tf-r1)
W=${WORKER_SSH:?set WORKER_SSH=<user>@<worker>}; PORT=${PORT:-8000}
RANK0=${RANK0:-glm53-tf-r0}; RANK1=${RANK1:-glm53-tf-r1}
fails=0
ok()  { printf '  %-50s OK   %s\n' "$1" "${2:-}"; }
bad() { printf '  %-50s FAIL %s\n' "$1" "${2:-}"; fails=$((fails+1)); }
chk() { local name=$1; shift; local out; if out=$("$@" 2>&1); then ok "$name" "$(echo "$out" | head -1 | cut -c1-70)"; else bad "$name" "$(echo "$out" | head -1 | cut -c1-70)"; fi; }
if [[ "${1:-}" == --wait ]]; then
    end=$(( $(date +%s) + ${2:-30} * 60 ))
    until curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null || (( $(date +%s) > end )); do sleep 15; done
fi
node_checks() { # runs on a node (locally or via ssh), prints "name|rc|detail" lines
printf 'UPLINK_IF=%q; CX7_IFS=%q; OLD_CONTAINERS=%q\n' "${UPLINK_IF:-}" "${CX7_IFS:-}" "${OLD_CONTAINERS:-}"
cat <<'NODE'
t() { local n=$1; shift; local o; o=$(eval "$@" 2>&1); echo "$n|$?|$(echo "$o" | head -1 | cut -c1-80)"; }
t "uptime / boot"                   "uptime -p"
t "default target = multi-user"     "[ \$(systemctl get-default) = multi-user.target ] && systemctl get-default"
t "no display manager running"      "! systemctl is-active -q gdm && echo gdm inactive"
t "system state (no failed units)"  "s=\$(systemctl is-system-running); systemctl --failed --no-legend | awk '{print \$2}' | tr '\n' ' '; [ \"\$s\" = running ]"
if [ -n "$UPLINK_IF" ]; then
t "default route on $UPLINK_IF"     "ip -4 route show default | grep -q \"dev $UPLINK_IF\" && echo $UPLINK_IF"
else
t "default route present"           "ip -4 route show default | grep -q . && ip -4 route show default | awk '{print \"dev\", \$5; exit}'"
fi
t "gateway reachable"               "ping -c 2 -W 2 \$(ip -4 route show default | awk '{print \$3; exit}') >/dev/null && echo ok"
t "DNS"                             "getent hosts github.com >/dev/null && echo ok"
if systemctl is-enabled -q tailscaled 2>/dev/null; then
t "tailscaled running + online"     "systemctl is-active -q tailscaled && tailscale status --self --peers=false >/dev/null && echo online"
fi
for i in $CX7_IFS; do
t "CX7 $i has IPv4"                 "ip -br -4 addr show $i | grep -q '[0-9]/' && echo \"mtu \$(cat /sys/class/net/$i/mtu)\""
done
t "RoCE ports ACTIVE"               "n=0; for d in /sys/class/infiniband/*; do grep -q ACTIVE \$d/ports/1/state && n=\$((n+1)); done; [ \$n -ge 1 ] && echo \"\$n active\""
t "docker active"                   "systemctl is-active docker"
t "GPU visible, persistence on"     "nvidia-smi --query-gpu=name,persistence_mode --format=csv,noheader"
t "os-tuning runtime unit"          "systemctl is-active spark-os-tuning.service"
t "polkitd small (< 256 MiB)"       "r=\$(ps -o rss= -C polkitd | head -1); [ \${r:-0} -lt 262144 ] && echo \$((\${r:-0}/1024)) MiB"
if [ -n "$OLD_CONTAINERS" ]; then
t "no old stack auto-started"       "! docker ps --format '{{.Names}}' | grep -qE '$OLD_CONTAINERS' && echo none"
fi
t "MemAvailable GiB"                "awk '/MemAvailable/{printf \"%.1f\n\",\$2/1048576}' /proc/meminfo"
NODE
}
report() { while IFS='|' read -r n rc d; do [[ -z "$n" ]] && continue; if [[ "$rc" == 0 ]]; then ok "$n" "$d"; else bad "$n" "$d"; fi; done; }
echo "== head"; bash -c "$(node_checks)" | report
echo "== head-only"
[[ -n "${LINGER_USER:-}" ]] && chk "linger for $LINGER_USER" bash -c "[ \"\$(loginctl show-user '$LINGER_USER' -p Linger --value)\" = yes ] && echo yes"
for tm in ${USER_TIMERS:-}; do chk "user timer $tm active" systemctl --user is-active "$tm"; done
chk "worker ssh (BatchMode)"               ssh -o BatchMode=yes -o ConnectTimeout=5 "$W" true
chk "rank 0 container running"             docker inspect -f '{{.State.Status}} since {{.State.StartedAt}}' "$RANK0"
chk ":$PORT /v1/models"                    curl -sf -m 10 "http://127.0.0.1:$PORT/v1/models"
[[ -n "${PUBLIC_URL:-}" ]] && chk "public endpoint"  curl -sf -m 20 "$PUBLIC_URL"
echo "== worker"; ssh -o BatchMode=yes -o ConnectTimeout=5 "$W" "bash -s" < <(node_checks) 2>/dev/null | report
chk "rank 1 container running"             ssh -o BatchMode=yes "$W" "docker inspect -f '{{.State.Status}} since {{.State.StartedAt}}' $RANK1"
echo "== boot to serving (head)"
bt=$(awk '/^btime/{print $2}' /proc/stat)
st=$(docker inspect -f '{{.State.StartedAt}}' "$RANK0" 2>/dev/null); st_s=$(date -d "$st" +%s 2>/dev/null || echo 0)
sv=$(docker logs -t "$RANK0" 2>&1 | grep -m1 -E 'serving ' | awk '{print $1}'); sv_s=$(date -d "$sv" +%s 2>/dev/null || echo 0)
echo "  kernel boot -> rank 0 container start: $(( st_s > 0 ? st_s - bt : -1 )) s"
echo "  kernel boot -> 'serving' (HTTP up):     $(( sv_s > 0 ? sv_s - bt : -1 )) s"
systemd-analyze 2>/dev/null | head -1 | sed 's/^/  /'
echo "== $fails check(s) failed"
exit $(( fails > 0 ))
