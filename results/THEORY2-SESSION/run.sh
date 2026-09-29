#!/usr/bin/env bash
# THEORY2-SESSION = W13: the GPU session of docs/THEORY-2.md §6 in one script. Run on the head node from the repo copy
# ($HOME/glm53-tensorfold-spark), as head, inside tmux/nohup. Logs: results/W13/ (session.log first).
#
#   run.sh build          BEFORE the window (prod keeps serving): image $IMAGE (b6 = b5 + 0510 / 0520 / 0530) built
#                         here from patches/ and shipped to the worker node; then `run.sh sync`
#   run.sh sync           tests/, patches/, results/THEORY2-SESSION/ -> the worker node's copy (tar over ssh)
#   run.sh all            the whole window: open -> probes -> loads -> restore (restore also on any exit / signal)
#   run.sh open | probes | loads [NAMES] | restore | status | summary     single phases (open first; restore last)
#
# Window rules (W11 / W12 / rigmark): lease $HOME/.test-window-lease touched every 4 min by a refresher that
# dies with this script (so a dead session leaves a stale lease and the re-armed watchdog heals prod), for at most
# WINDOW_MAX_MIN; glm53-tf-watchdog.timer stopped while open; restore = prod from config/prod.env (serve.sh start runs
# the canary), local + https /v1/models, 17*23 == 391, clocks back to -lgc 300,2250, watchdog timer started, refresher
# killed, lease deleted. Env: IMAGE, LOADS (default "C L CT K VS H GR"; H only if 0520's probes passed; L1 optional), SESSION_MAX_MIN (150: later loads skipped when
# over; GR goes first), VS_SPLIT (4), GR_MODE (resident), FORCE=1 (stop prod with requests in flight), NO_RESTORE=1.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
cd "$REPO"
WINDOW_MAX_MIN=${WINDOW_MAX_MIN:-240}
SESSION_MAX_MIN=${SESSION_MAX_MIN:-150}
LOADS=${LOADS:-C L CT K VS H GR}
STATE=$R/.state; mkdir -p "$STATE"
T0=$(date +%s)
SESS_DIR0=$HOME/.cache/glm53-tf/sessions            # serve.sh HEAD_SESSIONS (HEAD_HF/../glm53-tf/sessions)

timer() { systemctl --user is-active glm53-tf-watchdog.timer 2>/dev/null || true; }
lease_age() { [[ -f "$LEASE" ]] && echo $(( $(date +%s) - $(stat -c %Y "$LEASE") )) || echo none; }
elapsed_min() { echo $(( ($(date +%s) - T0) / 60 )); }

# -- window -----------------------------------------------------------------------------------------------------------
open_window() {
    docker image inspect "$IMAGE" > /dev/null 2>&1 || { log "no image $IMAGE on the head node: run.sh build first"; return 1; }
    wssh docker image inspect "$IMAGE" > /dev/null 2>&1 || { log "no image $IMAGE on the worker node: run.sh build ships it"; return 1; }
    local inflight
    for i in $(seq 1 30); do
        inflight=$(curl -s -m 5 "$B/health" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("inflight", 0))' 2>/dev/null || echo 0)
        [[ "$inflight" == 0 || "${FORCE:-0}" == 1 ]] && break
        log "production has $inflight request(s) in flight: waiting ($i/30)"; sleep 10
    done
    [[ "$inflight" == 0 || "${FORCE:-0}" == 1 ]] || { log "requests still in flight: not opening (FORCE=1 to override)"; return 1; }
    touch "$LEASE"
    [[ -f $STATE/lease.pid ]] && kill "$(cat $STATE/lease.pid)" 2>/dev/null
    local owner=${SESSION_PID:-0}              # run.sh all: its pid (the refresher dies with it); single phases: 0
    nohup bash -c "for i in \$(seq 1 $(( WINDOW_MAX_MIN / 4 ))); do [ -f '$LEASE' ] || exit 0; [ $owner = 0 ] || kill -0 $owner 2>/dev/null || exit 0; touch '$LEASE'; sleep 240; done" \
        > /dev/null 2>&1 &
    echo $! > "$STATE/lease.pid"
    systemctl --user stop glm53-tf-watchdog.timer
    log "window open: lease refresher pid $(cat $STATE/lease.pid) (tied to pid $owner, max $WINDOW_MAX_MIN min); watchdog timer $(timer)"
    health_ok && log "prod before: $(curl -s -m 10 $B/v1/models | head -c 200)"
    CONFIG=config/prod.env scripts/serve.sh stop > "$R/stop-prod.log" 2>&1
    clear_w13
    for i in $(seq 1 30); do
        [[ -z "$(gpu_apps)" && -z "$(wssh nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]] && break
        sleep 5
    done
    log "prod stopped; GPU apps head: [$(gpu_apps | tr '\n' ' ')] worker: [$(wssh nvidia-smi --query-compute-apps=pid,name --format=csv,noheader | tr '\n' ' ')]"
    docker ps --format '{{.Names}} {{.Image}}' >> "$R/session.log"; wssh "docker ps --format '{{.Names}} {{.Image}}'" >> "$R/session.log"
    clocks before
}

restore() {
    [[ -f $STATE/restored ]] && return 0
    log "restore: prod from config/prod.env"
    clear_w13
    set_clocks "$PROD_CLOCKS"
    local ok=0 answer https rc
    for attempt in 1 2; do
        CONFIG=config/prod.env scripts/serve.sh stop > /dev/null 2>&1
        env -u IMAGE CONFIG=config/prod.env timeout 1500 scripts/serve.sh start > "$R/start-prod-$attempt.log" 2>&1
        rc=$?
        grep -iE "canary|ready" "$R/start-prod-$attempt.log" | tail -3 | tee -a "$R/session.log" >&2
        answer=$(curl -s -m 180 "$B/v1/chat/completions" -H 'Content-Type: application/json' \
            -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23? Answer with the number only.\"}],\"max_tokens\":64,\"reasoning_effort\":\"none\"}" \
            | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null)
        https=$(curl -s -m 20 "$HTTPS_URL" | grep -c "$M")
        log "restore attempt $attempt: start rc=$rc, canary: $(grep -c '\[canary\] ok' "$R/start-prod-$attempt.log") ok line(s), local /v1/models $(curl -s -m 10 $B/v1/models | grep -c "$M"), https $https, 17*23 -> ${answer:-none}"
        if [[ $rc == 0 && "$answer" == *391* && $https -ge 1 ]]; then ok=1; break; fi
    done
    systemctl --user start glm53-tf-watchdog.timer
    [[ -f $STATE/lease.pid ]] && kill "$(cat $STATE/lease.pid)" 2>/dev/null; rm -f "$STATE/lease.pid"
    rm -f "$LEASE"
    clocks after
    if [[ $ok == 1 ]]; then
        log "RESTORED: prod verified; watchdog timer $(timer); lease $(lease_age)"
    else
        log "PROD NOT VERIFIED after 2 starts: watchdog timer $(timer) (it heals prod); CHECK NOW (scripts/serve.sh logs 0)"
    fi
    touch "$STATE/restored"
}

status() {
    log "status: lease age $(lease_age) s, watchdog timer $(timer), :8000 $(curl -s -m 5 $B/v1/models | head -c 120), elapsed $(elapsed_min) min"
    docker ps --format '{{.Names}} {{.Image}} {{.Status}}'; wssh "docker ps --format '{{.Names}} {{.Image}} {{.Status}}'"
}

# -- step 1: no-server probes on both nodes at once -------------------------------------------------------------------
probes() {
    set_clocks "$LOCK_CLOCKS"
    clocks probes
    log "probes: worker (glprobe x3, 0510 GPU test, hc_fused) and the head node (CPU suites, littles, cold dense) in parallel"
    wssh "IMAGE=$IMAGE bash $REPO2/results/THEORY2-SESSION/probes-node.sh worker" > "$R/probes-worker.out" 2>&1 &
    local p2=$!
    IMAGE=$IMAGE bash "$T2/probes-node.sh" head > "$R/probes-head.out" 2>&1
    wait $p2
    mkdir -p "$R/probes-worker"
    scp -q -r "$W:$REPO2/results/W13/probes-worker/." "$R/probes-worker/" 2>/dev/null
    rm -f "$R"/probes-worker/*.nsys-rep 2>/dev/null   # (reports stay on the worker node: $REPO2/results/W13/probes-worker/)
    set_clocks "$PROD_CLOCKS"
    log "probes done:"; cat "$R/probes-head/SUMMARY" "$R/probes-worker/SUMMARY" 2>/dev/null | tee -a "$R/session.log" >&2
}

# -- loads ------------------------------------------------------------------------------------------------------------
collect_trace() { # collect_trace NAME: the two ranks' RoCE dumps + graph probe / resident lines, then the dumps removed
    local n=$1
    docker exec glm53-tf-r0 sh -c "cat /sessions/w13-roce-$n-r0.jsonl 2>/dev/null; rm -f /sessions/w13-roce-$n-r0.jsonl" > "$R/roce-$n-r0.jsonl"
    wssh "docker exec glm53-tf-r1 sh -c 'cat /sessions/w13-roce-$n-r1.jsonl 2>/dev/null; rm -f /sessions/w13-roce-$n-r1.jsonl'" > "$R/roce-$n-r1.jsonl"
    docker logs glm53-tf-r0 2>&1 | grep -E 'graph probe|resident rounds|trace dump' > "$R/lines-$n-r0.txt"
    wssh docker logs glm53-tf-r1 2>&1 | grep -E 'graph probe|resident rounds|trace dump' > "$R/lines-$n-r1.txt"
    log "$n: RoCE dumps $(wc -l < "$R/roce-$n-r0.jsonl") / $(wc -l < "$R/roce-$n-r1.jsonl") lines; probe lines $(wc -l < "$R/lines-$n-r0.txt")"
}

TRACE="GLM53_TF_ROCE_TRACE=4096 GLM53_TF_ROCE_TRACE_EVERY=65536"
one_load() {
    local n=$1
    case $n in
    C|C2)  bash "$T2/load.sh" "$n" plain && health_ok && bash "$T2/ab.sh" "$n" full ;;
    # prod (W12, 5c4da18) already has L2PF=1 / L2PF_MB=8 / BATCH_CAPTURE_AFTER=8: L adds only the graph policy;
    # L1 (optional) also captures lone keys on first sight (the 8 sightings delay only lone slots 1-3 under lone)
    L)     bash "$T2/load.sh" L plain GLM53_TF_BATCH_GRAPHS=lone && health_ok && bash "$T2/ab.sh" L full ;;
    L1)    bash "$T2/load.sh" L1 plain GLM53_TF_BATCH_GRAPHS=lone GLM53_TF_BATCH_CAPTURE_AFTER=1 && health_ok \
               && bash "$T2/ab.sh" L1 full ;;
    CT)    bash "$T2/load.sh" CT plain $TRACE GLM53_TF_ROCE_TRACE_DUMP=/sessions/w13-roce-CT GLM53_TF_GRAPH_PROBE=100 \
               && health_ok && bash "$T2/ab.sh" CT short
           collect_trace CT
           python3 "$T2/rocetrace.py" "$R/roce-CT" --json "$R/rocetrace-CT.json" > "$R/rocetrace-CT.txt" 2>&1 ;;
    K)     set_clocks "$LOCK_CLOCKS"; clocks K
           bash "$T2/load.sh" K plain $TRACE GLM53_TF_ROCE_TRACE_DUMP=/sessions/w13-roce-K GLM53_TF_CPU_PIN=http \
               && health_ok && { docker exec glm53-tf-r0 sh -c 'ps -L -o tid,psr,comm -p 1 2>/dev/null; for t in /proc/[0-9]*/task/*; do echo "$(cat $t/comm) $(grep Cpus_allowed_list $t/status | cut -f2)"; done | sort | uniq -c' > "$R/threads-K-r0.txt" 2>&1
                                 bash "$T2/ab.sh" K full; }
           collect_trace K
           python3 "$T2/rocetrace.py" "$R/roce-K" --vs "$R/roce-CT" --json "$R/rocetrace-K.json" > "$R/rocetrace-K.txt" 2>&1
           grep GATE "$R/rocetrace-K.txt" | tee -a "$R/session.log" >&2
           set_clocks "$PROD_CLOCKS" ;;
    VS)    bash "$T2/load.sh" VS plain GLM53_TF_VERIFY_SPLIT=${VS_SPLIT:-4} GLM53_TF_GRAPH_PROBE=100 \
               && health_ok && bash "$T2/ab.sh" VS full
           collect_trace VS ;;
    H)     if ! grep -q 'GATE item3.*PASS' "$R/probes-worker/hcbench.log" 2>/dev/null \
              || ! grep -q 'rc=0' <(grep ' t520 ' "$R/probes-worker/SUMMARY" 2>/dev/null); then
               log "H skipped: 0520's GPU bits test or microbench gate did not pass (probes-worker)"; return 0
           fi
           bash "$T2/load.sh" H plain GLM53_TF_HC_CUDA=1 && health_ok && bash "$T2/ab.sh" H full ;;
    GR)    gr ;;
    *)     log "unknown load $n" ;;
    esac
}

gr() { # item 9: GPU_ROUND=resident + ONE nsys window (graph-level) on greedy tf code x3 + one 4-stream rep
    local Q="python3 $T2/req.py"
    bash "$T2/load.sh" GR nsys GLM53_TF_GPU_ROUND=${GR_MODE:-resident} GLM53_TF_RESIDENT_REPORT=10 && health_ok || return 1
    $Q tfcode "$R/gr-warm.jsonl" 1 > /dev/null 2>&1; $Q conc "$R/gr-warm.jsonl" 4 64 > /dev/null 2>&1
    $Q tfcode "$R/gr-ctl.jsonl" 3 > "$R/gr-ctl.log" 2>&1; $Q conc "$R/gr-ctl.jsonl" 4 384 >> "$R/gr-ctl.log" 2>&1
    docker exec glm53-tf-r0 nsys start --session=w13 --sample=none --cpuctxsw=none --force-overwrite=true -o /w13/out/gr-r0 &
    wssh docker exec glm53-tf-r1 nsys start --session=w13 --sample=none --cpuctxsw=none --force-overwrite=true -o /w13/out/gr-r1 &
    wait; sleep 2
    $Q tfcode "$R/gr-cap.jsonl" 3 > "$R/gr-cap.log" 2>&1; sleep 2
    $Q conc "$R/gr-cap.jsonl" 4 384 >> "$R/gr-cap.log" 2>&1; sleep 2
    docker exec glm53-tf-r0 nsys stop --session=w13 &
    wssh docker exec glm53-tf-r1 nsys stop --session=w13 &
    wait
    for i in $(seq 1 120); do
        if [[ -f /var/tmp/w13/out/gr-r0.nsys-rep ]] && wssh test -f /var/tmp/w13/out/gr-r1.nsys-rep \
           && ! pgrep -f QdstrmImporter > /dev/null && ! wssh pgrep -f QdstrmImporter > /dev/null; then break; fi
        sleep 5
    done
    collect_trace GR
    log "GR: reports $(ls -la /var/tmp/w13/out/gr-r0.nsys-rep 2>&1 | awk '{print $5, $NF}') / worker $(wssh ls -la /var/tmp/w13/out/gr-r1.nsys-rep 2>&1 | awk '{print $5, $NF}') (export offline, not on a prod node)"
    cat "$R/gr-ctl.log" "$R/gr-cap.log" | tee -a "$R/session.log" >&2
}

loads() {
    local order=("$@")
    for n in "${order[@]}"; do
        if (( $(elapsed_min) >= SESSION_MAX_MIN )); then log "time: $(elapsed_min) min >= $SESSION_MAX_MIN, skipping $n"; continue; fi
        if [[ $n == GR ]] && (( $(elapsed_min) + 20 > SESSION_MAX_MIN )); then log "time: GR needs ~20 min, skipped (next window)"; continue; fi
        log "== load $n ($(elapsed_min) min in)"
        one_load "$n" || log "load $n FAILED or unhealthy (see start-$n.log / boot-$n-r0.log); next"
        if [[ $n == C ]] && ! [[ -f "$R/glmbench-C.json" ]]; then log "control C has no results: stopping the loads"; return 1; fi
    done
    python3 "$T2/summary.py" "$R" > "$R/summary.txt" 2>&1; grep -E 'GATE|^C |^L |^K |^VS ' "$R/summary.txt" | tee -a "$R/session.log" >&2
}

case "${1:-status}" in
build)
    log "build $IMAGE on the head node (prod keeps serving) and ship it to the worker node"
    ls patches/05[1-3]0-*.patch | tee -a "$R/session.log"
    IMAGE=$IMAGE scripts/serve.sh build > "$R/build.log" 2>&1; rc=$?
    log "build rc=$rc: $(grep -E 'applying patches/05|tensorfold .* torch' "$R/build.log" | tail -4 | tr '\n' ' ')"
    wssh docker image inspect "$IMAGE" --format '{{.Id}}' | tee -a "$R/session.log" ;;
sync)
    tar cf - tests patches results/THEORY2-SESSION | wssh "mkdir -p $REPO2 && cd $REPO2 && tar xf -" && log "synced tests/ patches/ results/THEORY2-SESSION/ to the worker node:$REPO2" ;;
open) rm -f "$STATE/restored"; open_window ;;
probes) probes ;;
loads) shift; loads ${@:-$LOADS} ;;
restore) restore ;;
status) status ;;
summary) python3 "$T2/summary.py" "$R" ;;
all)
    echo $$ > "$STATE/session.pid"; rm -f "$STATE/restored"
    SESSION_PID=$$
    [[ "${NO_RESTORE:-0}" == 1 ]] || trap 'restore' EXIT
    trap 'log "signal: restoring"; exit 1' INT TERM HUP
    log "W13 session: image $IMAGE, loads [$LOADS], max $SESSION_MAX_MIN min"
    open_window || exit 1
    probes
    loads $LOADS
    log "session done after $(elapsed_min) min (restore follows)" ;;
*) sed -n '2,20p' "$0" ;;
esac
