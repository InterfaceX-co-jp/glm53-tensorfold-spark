#!/usr/bin/env bash
# W16: start a test load from config/prod.env (IMAGE default b8) with overrides, both ranks:
#   load.sh NAME plain|nsys [K=V ...]
# nsys: scripts/serve.sh with ${W16_DOCKER} added to both ranks' docker run (serve-w16.sh, generated here): the nsys shim
# (idle session "w16", graph-level tracing), SYS_ADMIN (RmProfilingAdminOnly), W11's NVTX profile.py copy (results/W11/
# profile.py: phase ranges for rounds / verify / drafting; timing only) and /var/tmp/w16 (entry.sh, bin/, profile.py,
# out/) on both nodes. Boot logs -> $R/boot-NAME-r{0,1}.log. Exit status = serve.sh start's.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
cd "$REPO"
name=$1; mode=$2; shift 2
# W16 (schedule change at 22:50: wrap up by ~00:30): GR (nsys) goes to a later window
if [[ $name == GR && "${W16_SKIP_GR:-1}" == 1 ]]; then log "load GR skipped (W16 schedule: next window)"; exit 1; fi
for kv in "$@"; do export "$kv"; done
export IMAGE
log "load $name ($mode): IMAGE=$IMAGE $*"
echo "load $name ($mode): IMAGE=$IMAGE $* ($(date +%T))" >> "$R/loads.log"
S=scripts/serve.sh
if [[ $mode == nsys ]]; then
    S=$R/serve-w16.sh
    python3 - "$S" "$REPO" <<'PY'
import sys
s = open("scripts/serve.sh").read()
a = 'cd "$(dirname "${BASH_SOURCE[0]}")/.."'
b = '-e GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" "$IMAGE"'
assert s.count(a) == 1 and s.count(b) == 1, "scripts/serve.sh changed: update load.sh's nsys injection"
s = s.replace(a, f'cd {sys.argv[2]}   # W16 copy').replace(
    b, '-e GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" ${W16_DOCKER:-} "$IMAGE"   # W16')
open(sys.argv[1], "w").write(s)
PY
    chmod +x "$S"
    mkdir -p /var/tmp/w16/out /var/tmp/w16/bin
    cp "$T2/nsys/entry.sh" results/W11/profile.py /var/tmp/w16/ && cp "$T2/nsys/bin/tensorfold" /var/tmp/w16/bin/
    wssh "mkdir -p /var/tmp/w16/out /var/tmp/w16/bin"
    scp -q "$T2/nsys/entry.sh" results/W11/profile.py "$W:/var/tmp/w16/" && scp -q "$T2/nsys/bin/tensorfold" "$W:/var/tmp/w16/bin/"
    PP=/usr/local/lib/python3.12/dist-packages/tensorfold/families/glm5_next/cuda/profile.py
    export W16_DOCKER="--cap-add SYS_ADMIN -v /var/tmp/w16:/w16 -v /var/tmp/w16/profile.py:$PP:ro --entrypoint /w16/entry.sh -e W16_NSYS=1 -e W7_NVTX=1 -e W16_GRAPH_TRACE=${W16_GRAPH_TRACE:-graph}"
fi
CONFIG=config/prod.env scripts/serve.sh stop > /dev/null 2>&1
CONFIG=config/prod.env timeout "${START_TIMEOUT:-1800}" $S start > "$R/start-$name.log" 2>&1
rc=$?
echo "start rc=$rc $(date +%T)" >> "$R/loads.log"
log "load $name start rc=$rc: $(tail -1 "$R/start-$name.log" | cut -c1-160)"
docker logs glm53-tf-r0 > "$R/boot-$name-r0.log" 2>&1
wssh docker logs glm53-tf-r1 > "$R/boot-$name-r1.log" 2>&1
grep -iE 'roce:|context:|decode overlap|l2pf|batch graphs|0510|0520|0530|graph probe|pieces|CPU pinning|trace dump|gpu round|resident|hc fused|Traceback' \
    "$R/boot-$name-r0.log" | cut -c1-220 | head -16 | tee -a "$R/session.log" >&2
grep -iE '0510|0520|0530|CPU pinning|trace dump|error|Traceback' "$R/boot-$name-r1.log" | cut -c1-220 | head -6 | tee -a "$R/session.log" >&2
exit $rc
