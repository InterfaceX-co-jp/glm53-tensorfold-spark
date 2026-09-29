#!/usr/bin/env bash
# W13: start a test load from config/prod.env (IMAGE default b6) with overrides, both ranks:
#   load.sh NAME plain|nsys [K=V ...]
# nsys: scripts/serve.sh with ${W13_DOCKER} added to both ranks' docker run (serve-w13.sh, generated here): the nsys shim
# (idle session "w13", graph-level tracing), SYS_ADMIN (RmProfilingAdminOnly), W11's NVTX profile.py copy (results/W11/
# profile.py: phase ranges for rounds / verify / drafting; timing only) and /var/tmp/w13 (entry.sh, bin/, profile.py,
# out/) on both nodes. Boot logs -> $R/boot-NAME-r{0,1}.log. Exit status = serve.sh start's.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
cd "$REPO"
name=$1; mode=$2; shift 2
for kv in "$@"; do export "$kv"; done
export IMAGE
log "load $name ($mode): IMAGE=$IMAGE $*"
echo "load $name ($mode): IMAGE=$IMAGE $* ($(date +%T))" >> "$R/loads.log"
S=scripts/serve.sh
if [[ $mode == nsys ]]; then
    S=$R/serve-w13.sh
    python3 - "$S" "$REPO" <<'PY'
import sys
s = open("scripts/serve.sh").read()
a = 'cd "$(dirname "${BASH_SOURCE[0]}")/.."'
b = '-e GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" "$IMAGE"'
assert s.count(a) == 1 and s.count(b) == 1, "scripts/serve.sh changed: update load.sh's nsys injection"
s = s.replace(a, f'cd {sys.argv[2]}   # W13 copy').replace(
    b, '-e GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" ${W13_DOCKER:-} "$IMAGE"   # W13')
open(sys.argv[1], "w").write(s)
PY
    chmod +x "$S"
    mkdir -p /var/tmp/w13/out /var/tmp/w13/bin
    cp "$T2/nsys/entry.sh" results/W11/profile.py /var/tmp/w13/ && cp "$T2/nsys/bin/tensorfold" /var/tmp/w13/bin/
    wssh "mkdir -p /var/tmp/w13/out /var/tmp/w13/bin"
    scp -q "$T2/nsys/entry.sh" results/W11/profile.py "$W:/var/tmp/w13/" && scp -q "$T2/nsys/bin/tensorfold" "$W:/var/tmp/w13/bin/"
    PP=/usr/local/lib/python3.12/dist-packages/tensorfold/families/glm5_next/cuda/profile.py
    export W13_DOCKER="--cap-add SYS_ADMIN -v /var/tmp/w13:/w13 -v /var/tmp/w13/profile.py:$PP:ro --entrypoint /w13/entry.sh -e W13_NSYS=1 -e W7_NVTX=1 -e W13_GRAPH_TRACE=${W13_GRAPH_TRACE:-graph}"
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
