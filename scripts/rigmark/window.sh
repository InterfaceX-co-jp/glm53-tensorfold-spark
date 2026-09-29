#!/usr/bin/env bash
# Test-window helper for the RigMark sweep (docs/RIGMARK.md). Run on the head node (the head Spark), from this repo's copy.
#
#   window.sh open      lease + refresher, glm53-tf-watchdog.timer stopped (TensorFold keeps serving)
#   window.sh vllm-up   stop TensorFold, wait for MemFree > 95 GiB on both nodes, start the vLLM kit, wait healthy
#   window.sh tf-up     stop the vLLM kit, start TensorFold prod (config/prod.env), verify
#   window.sh close     only with TensorFold prod healthy: watchdog timer on, refresher killed, lease deleted
#   window.sh status    lease age, timer, who serves :8000, containers, MemFree
#
# Rules it enforces (restore-prod.sh / rearm-watchdog.sh): while the watchdog timer is stopped the lease
# ~/.test-window-lease must be touched at least every 15 min (the refresher touches it every 4 min, for at most
# WINDOW_MAX_MIN, default 240); a lease older than 20 min lets rearm-watchdog.sh re-arm the watchdog, which then heals
# production itself. Every step is logged to results/rigmark/windows.log.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
KIT="${VLLM_KIT_DIR:-$HOME/glm53-exl3-2x-kit}"
W="${WORKER_SSH:-$(sed -n 's/^WORKER_SSH=//p' "$repo/${CONFIG:-config/prod.env}" 2>/dev/null | tr -d '"' | tail -1)}"   # the worker's ssh target
LEASE="${LEASE:-$HOME/.test-window-lease}"
BASE="${BASE:-http://127.0.0.1:8000}"
MODEL="${MODEL:-GLM-5.3-Flash-EXL3}"
NEED_GIB="${NEED_GIB:-95}"                 # restore-prod.sh wait_mem: MemFree > 95 GiB on both nodes before the kit
MEM_WAIT_S="${MEM_WAIT_S:-900}"
VLLM_READY_S="${VLLM_READY_S:-4800}"       # the kit's READY_TIMEOUT (a cold JIT after a cache wipe is slow)
WINDOW_MAX_MIN="${WINDOW_MAX_MIN:-240}"
STATE="${XDG_STATE_HOME:-$HOME/.local/state}/glm53-rigmark"
LOGDIR="$repo/results/rigmark"
mkdir -p "$STATE" "$LOGDIR"
log() { echo "[window $(date +%T)] $*" | tee -a "$LOGDIR/windows.log" >&2; }
wssh() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$W" "$@"; }
owner() { # "owned_by:id" of each served model (vllm:... or tensorfold:...), or none
    curl -sf -m 10 "$BASE/v1/models" | python3 -c 'import json,sys
print(",".join(m.get("owned_by", "?") + ":" + m["id"] for m in json.load(sys.stdin)["data"]))' 2>/dev/null || echo none
}
memfree() { awk '/^MemFree:/ {printf "%d", $2/1048576}' /proc/meminfo; }
memfree_w() { wssh "awk '/^MemFree:/ {printf \"%d\", \$2/1048576}' /proc/meminfo" 2>/dev/null || echo 0; }
lease_age() { [[ -f "$LEASE" ]] && echo $(( $(date +%s) - $(stat -c %Y "$LEASE") )) || echo none; }
timer() { systemctl --user is-active glm53-tf-watchdog.timer 2>/dev/null || true; }
need_window() {
    [[ "$(timer)" != active ]] || { log "watchdog timer is active: run 'window.sh open' first"; exit 1; }
    local age; age=$(lease_age)
    [[ "$age" != none && "$age" -lt 900 ]] || { log "lease missing or stale ($age s): run 'window.sh open' again"; exit 1; }
}
clear_tf() { # leftover TensorFold containers on both nodes (restore-prod.sh clear_tf)
    local ids
    ids=$(docker ps -a --format '{{.ID}} {{.Image}} {{.Names}}' | awk '$2 ~ /^glm53-tensorfold/ || $3 ~ /^glm53-tf/ {print $1}')
    [[ -z "$ids" ]] || docker rm -f $ids > /dev/null
    wssh 'ids=$(docker ps -a --format "{{.ID}} {{.Image}} {{.Names}}" | awk "\$2 ~ /^glm53-tensorfold/ || \$3 ~ /^glm53-tf/ {print \$1}"); [ -z "$ids" ] || docker rm -f $ids >/dev/null; true'
}

case "${1:-status}" in
open)
    [[ -d "$KIT" ]] || { log "no vLLM kit at $KIT: run this on the head node"; exit 1; }
    inflight=$(curl -sf -m 10 "$BASE/health" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("inflight", 0))' 2>/dev/null || echo "?")
    if [[ "$inflight" != 0 && "${FORCE:-0}" != 1 ]]; then
        log "production has requests in flight ($inflight): wait, or FORCE=1"; exit 1
    fi
    touch "$LEASE"
    if [[ -f "$STATE/lease.pid" ]] && kill -0 "$(cat "$STATE/lease.pid")" 2>/dev/null; then
        log "lease refresher already running (pid $(cat "$STATE/lease.pid"))"
    else
        nohup bash -c "for i in \$(seq 1 $(( WINDOW_MAX_MIN / 4 ))); do [ -f '$LEASE' ] || exit 0; touch '$LEASE'; sleep 240; done" \
            > /dev/null 2>&1 &
        echo $! > "$STATE/lease.pid"
    fi
    systemctl --user stop glm53-tf-watchdog.timer
    log "window open: lease refresher pid $(cat "$STATE/lease.pid") (max ${WINDOW_MAX_MIN} min), watchdog timer $(timer); :8000 $(owner)"
    ;;
vllm-up)
    need_window
    log "stopping TensorFold"
    ( cd "$repo" && CONFIG=config/prod.env scripts/serve.sh stop ) >> "$LOGDIR/windows.log" 2>&1
    clear_tf
    t0=$(date +%s)
    while :; do
        h=$(memfree); w=$(memfree_w)
        (( h > NEED_GIB && w > NEED_GIB )) && { log "MemFree $h / $w GiB (head / worker) > $NEED_GIB"; break; }
        if (( $(date +%s) - t0 >= MEM_WAIT_S )); then
            log "MemFree $h / $w GiB still <= $NEED_GIB after ${MEM_WAIT_S}s; starting anyway (prod-start.sh re-gates at 90 and retries)"; break
        fi
        sync; sudo -n sh -c 'echo 1 > /proc/sys/vm/drop_caches' 2>/dev/null || true          # page cache only
        wssh "sync; sudo -n sh -c 'echo 1 > /proc/sys/vm/drop_caches' 2>/dev/null || true" || true
        sleep 10
    done
    log "starting the vLLM kit (local/prod-start.sh); log: results/rigmark/vllm-start.log"
    ( cd "$KIT" && WORKER_SSH="$W" local/prod-start.sh ) > "$LOGDIR/vllm-start.log" 2>&1
    rc=$?
    t0=$(date +%s)
    until owner | grep -q ":$MODEL"; do
        (( $(date +%s) - t0 < VLLM_READY_S )) || { log "vLLM not serving $MODEL (prod-start rc=$rc); see vllm-start.log. Restore: window.sh tf-up"; exit 1; }
        sleep 15
    done
    log "vLLM up (prod-start rc=$rc): :8000 $(owner); next: scripts/rigmark/run.sh vllm"
    ;;
tf-up)
    need_window
    log "stopping the vLLM kit"
    ( cd "$KIT" && ./start.sh stop ) >> "$LOGDIR/windows.log" 2>&1
    clear_tf
    # serve.sh gates on MemFree >= MEM_GATE_GIB (prod.env: 108, dropping page cache) and refuses while a CUDA process runs
    log "starting TensorFold prod (config/prod.env)"
    ( cd "$repo" && CONFIG=config/prod.env timeout 1500 scripts/serve.sh start ) > "$LOGDIR/tf-start.log" 2>&1
    rc=$?
    grep -iE "canary|ready|warm" "$LOGDIR/tf-start.log" | tail -4 | tee -a "$LOGDIR/windows.log" >&2
    answer=$(curl -s -m 180 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23? Answer with the number only.\"}],\"max_tokens\":64,\"reasoning_effort\":\"none\"}" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null)
    log "TensorFold start rc=$rc; :8000 $(owner); health $(curl -s -m 10 "$BASE/health" | head -c 200); 17*23 -> ${answer:-no answer}"
    [[ $rc == 0 && "$answer" == *391* ]] || { log "TensorFold NOT verified: fix before 'close' (restore-prod.sh falls back to vLLM)"; exit 1; }
    log "production verified; next: window.sh close, then scripts/rigmark/run.sh tensorfold"
    ;;
close)
    if ! owner | grep -q "tensorfold:$MODEL" && [[ "${FORCE:-0}" != 1 ]]; then
        log "TensorFold is not serving $MODEL on :8000 ($(owner)): run 'window.sh tf-up' first (or FORCE=1)"; exit 1
    fi
    systemctl --user start glm53-tf-watchdog.timer
    [[ -f "$STATE/lease.pid" ]] && kill "$(cat "$STATE/lease.pid")" 2>/dev/null; rm -f "$STATE/lease.pid"
    rm -f "$LEASE"
    log "window closed: watchdog timer $(timer), lease $(lease_age); :8000 $(owner)"
    ;;
status)
    echo "lease: age $(lease_age) s; refresher pid $(cat "$STATE/lease.pid" 2>/dev/null || echo none); watchdog timer: $(timer)"
    echo ":8000 $(owner)"
    echo "MemFree head $(memfree) GiB, worker $(memfree_w) GiB"
    echo "head:   $(docker ps --format '{{.Names}}' | tr '\n' ' ')"
    echo "worker: $(wssh "docker ps --format '{{.Names}}'" 2>/dev/null | tr '\n' ' ')"
    ;;
*)
    sed -n '2,15p' "$0"; exit 2 ;;
esac
