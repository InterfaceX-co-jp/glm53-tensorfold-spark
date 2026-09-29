#!/usr/bin/env bash
# W7 (results/W7): scripts/serve.sh with ${W7_DOCKER} (extra docker run args: nsys shim, SYS_ADMIN, profile.py mount) on both ranks
# Start / stop / inspect the two TensorFold ranks. Run on the head Spark (rank 0).
#
#   scripts/serve.sh build      build the image here and ship it to the worker
#   scripts/serve.sh start      preflight, memory gate, rank 1 on the worker, then rank 0 here; waits for /v1/models,
#                               then runs the canary (scripts/canary.py)
#   scripts/serve.sh restart    stop, then start
#   scripts/serve.sh stop | status | logs [0|1] | canary | xid [SINCE] | watch [--once] | preflight
#   scripts/serve.sh gpucheck   GB10 clock / power / slow-state check of both nodes (scripts/gpuwatch.py check);
#                               non-zero when a node is degraded or was recently slow: gate benchmark runs on it
#
# Configuration: config/tensorfold.env (override with CONFIG=path).
set -euo pipefail
cd $HOME/glm53-tensorfold-spark   # W7 copy: absolute repo path
CONFIG="${CONFIG:-config/tensorfold.env}"
# A non-empty caller export wins over the same key in the config file:
#   CONTEXT=32768 GLM53_TF_NONEXPERT=q4mse scripts/serve.sh start
caller_env=$(env | grep -E '^(HEAD_PREPARED|WORKER_PREPARED|HEAD_SESSIONS|WORKER_SESSIONS|CONTEXT|MTP_DRAFTS|NO_DRAFTS|IMAGE|PORT|HOST|DRAFTER|MODEL_PATH|EXTRA_ARGS|SERVED_NAME|MAX_TOKENS|CANARY[A-Z_]*|MEM_GATE_[A-Z_]+|START_ATTEMPTS|READY_TIMEOUT|LOG_MAX_[A-Z]+|WATCH_[A-Z_]+|GPUWATCH_[A-Z_]+|WARMUP_LENGTHS|PREFLIGHT|NCCL_PASSTHROUGH|GLM53_TF_[A-Z0-9_]+)=.' || true)
# shellcheck disable=SC1090
set -a; source "$CONFIG"; set +a
while IFS= read -r kv; do [[ -n "$kv" ]] && export "${kv?}"; done <<< "$caller_env"

NAME="${NAME:-glm53-tf}"
IMAGE="${IMAGE:-glm53-tensorfold:dev}"
PORT="${PORT:-8080}"
BASE="http://127.0.0.1:$PORT"
# Every ops feature below is off unless set (config or environment); `start` / `stop` / `status` behave as before.
# post-load canary (scripts/canary.py): off | warn (log a failure, keep serving) | strict (stop both ranks, exit 1)
CANARY="${CANARY:-off}"
CANARY_MIN_TPR="${CANARY_MIN_TPR:-1.3}"       # least tokens a decode round over the probes (drafter alive)
CANARY_MIN_TPS="${CANARY_MIN_TPS:-0}"         # least decode tok/s over the probes (0: no floor)
# wait before launch until MemFree >= MEM_GATE_GIB on both nodes (0: off), at most MEM_GATE_TIMEOUT s, then go on.
# MemFree, not MemAvailable: on the Spark's unified memory CUDA cannot use reclaimable page cache right away, and
# MemAvailable counts it (the vLLM kit measured MemAvailable 111.9 GiB green-lighting starts the engine then refused)
MEM_GATE_GIB="${MEM_GATE_GIB:-0}"
MEM_GATE_TIMEOUT="${MEM_GATE_TIMEOUT:-600}"
MEM_GATE_DROP_CACHES="${MEM_GATE_DROP_CACHES:-0}"   # 1: while waiting, `sync; echo 1 > drop_caches` (sudo -n) on both nodes
WARMUP_LENGTHS="${WARMUP_LENGTHS:-}"          # e.g. "4096 16384": after the canary, prefill prompts of about these lengths
START_ATTEMPTS="${START_ATTEMPTS:-1}"         # a failed start (a rank exited, not ready, strict canary) tears down and retries
READY_TIMEOUT="${READY_TIMEOUT:-0}"           # seconds to wait for /v1/models (0: no limit; first start compiles kernels)
LOG_MAX_SIZE="${LOG_MAX_SIZE:-}"              # e.g. 200m: docker json-file log rotation per container (empty: docker's default)
LOG_MAX_FILE="${LOG_MAX_FILE:-3}"
PREFLIGHT="${PREFLIGHT:-off}"                 # before start: off | warn (log problems) | strict (refuse to start on one)
NCCL_PASSTHROUGH="${NCCL_PASSTHROUGH:-0}"     # 1: every other NCCL_* variable set here reaches both ranks
# preflight's GPU state check (scripts/gpuwatch.py check, docs/OPS-GPUWATCH.md): off | on (a degraded node -- clock or
# power clamp, a step-time regression past the transient window -- is a preflight problem; warnings are logged) |
# strict (a warning is a problem too: a recent slow state, an idle clock asymmetry; use for benchmark windows)
GPUWATCH_PREFLIGHT="${GPUWATCH_PREFLIGHT:-on}"
STATE_DIR="${STATE_DIR:-${XDG_STATE_HOME:-${HOME:-/tmp}/.local/state}/glm53-tf}"
# patches/0140: prepared rank folders (scripts/prepare.sh), next to each node's HF cache unless set
HEAD_PREPARED="${HEAD_PREPARED:-${HEAD_HF%/*}/glm53-tf/prepared}"
WORKER_PREPARED="${WORKER_PREPARED:-${WORKER_HF%/*}/glm53-tf/prepared}"
# patches/0250: the session store's NVMe tier, mounted at /sessions in both ranks (used when GLM53_TF_SESSION_DISK=/sessions;
# up to GLM53_TF_SESSION_DISK_GIB a node, default 64), next to each node's HF cache unless set
HEAD_SESSIONS="${HEAD_SESSIONS:-${HEAD_HF%/*}/glm53-tf/sessions}"
WORKER_SESSIONS="${WORKER_SESSIONS:-${WORKER_HF%/*}/glm53-tf/sessions}"
IMAGE_ID="${IMAGE_ID:-}"
log() { echo "[glm53-tf] $*"; }
wssh() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$@"; }

run_args() { # $1 = rank, $2 = host HF cache dir
    local rank=$1 hf=$2 prep sess
    # patches/0140: this node's prepared rank folders (scripts/prepare.sh) at /prepared; the image id and the launch
    # time key the calibration cache and start the [boot] timeline
    if [[ "$rank" == 0 ]]; then prep="$HEAD_PREPARED"; else prep="$WORKER_PREPARED"; fi
    if [[ "$rank" == 0 ]]; then sess="$HEAD_SESSIONS"; else sess="$WORKER_SESSIONS"; fi   # patches/0250
    echo --name "$NAME-r$rank" -d --gpus all --ipc=host --network host \
        -v "$prep:/prepared" -e GLM53_TF_PREPARED=/prepared -e GLM53_TF_PREPARED_WRITE="${GLM53_TF_PREPARED_WRITE:-1}" \
        -v "$sess:/sessions" \
        -e GLM53_TF_CALIB="${GLM53_TF_CALIB:-cached}" -e GLM53_TF_IMAGE_ID="$IMAGE_ID" \
        -e GLM53_TF_LAUNCH_T0="$(date +%s.%N)" \
        --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK \
        ${LOG_MAX_SIZE:+--log-opt max-size="$LOG_MAX_SIZE" --log-opt max-file="$LOG_MAX_FILE"} \
        -v "$hf:/root/.cache/huggingface" -v "$NAME-cache:/cache" \
        -e RANK="$rank" -e MASTER="$HEAD_IP" -e MASTER_PORT="${MASTER_PORT:-29551}" \
        -e MODEL_PATH="$MODEL_PATH" -e DRAFTER="${DRAFTER:-}" -e CONTEXT="${CONTEXT:-}" \
        -e SERVED_NAME="${SERVED_NAME:-}" -e MAX_TOKENS="${MAX_TOKENS:-}" -e MTP_DRAFTS="${MTP_DRAFTS:-}" \
        -e NO_DRAFTS="${NO_DRAFTS:-0}" -e HOST="${HOST:-127.0.0.1}" -e PORT="$PORT" \
        -e EXTRA_ARGS="${EXTRA_ARGS:-}" -e GLM53_TF_NONEXPERT="${GLM53_TF_NONEXPERT:-bf16}" \
        -e GLM53_TF_PREFILL_ROWS="${GLM53_TF_PREFILL_ROWS:-auto}" -e GLM53_TF_AUTO_FDRAFTS="${GLM53_TF_AUTO_FDRAFTS:-7}" \
        -e GLM53_TF_PROFILE="${GLM53_TF_PROFILE:-0}" -e GLM53_TF_EXPERT_LOOP="${GLM53_TF_EXPERT_LOOP:-1}" \
        -e GLM53_TF_EXPERT_LOOP_CFG="${GLM53_TF_EXPERT_LOOP_CFG:-4,2}" \
        $(env | grep -E '^GLM53_TF_[A-Z0-9_]+=' | sed 's/^/-e /' | tr '\n' ' ') \
        $([[ "$NCCL_PASSTHROUGH" == 1 ]] && env | grep -E '^NCCL_[A-Z0-9_]+=' | grep -vE '^NCCL_(SOCKET_IFNAME|IB_HCA|PASSTHROUGH)=' | sed 's/^/-e /' | tr '\n' ' ') \
        -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" -e NCCL_IB_HCA="$NCCL_IB_HCA" \
        -e GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" ${W7_DOCKER:-} "$IMAGE"
}

gpu_busy() { # a foreign CUDA process on either node means the other stack is still up
    # grep reads everything (not -q): with pipefail, -q's early exit SIGPIPEs the writer and a busy node read as free
    { nvidia-smi --query-compute-apps=pid --format=csv,noheader; wssh nvidia-smi --query-compute-apps=pid --format=csv,noheader; } \
        | grep '[0-9]' >/dev/null
}

running() { # $1 = rank: prints true | false | absent | unreachable (the worker's ssh failed)
    local out rc
    if [[ "$1" == 0 ]]; then
        out=$(docker inspect -f '{{.State.Running}}' "$NAME-r0" 2>/dev/null) || out=absent
    else
        out=$(wssh docker inspect -f "'{{.State.Running}}'" "$NAME-r1" 2>/dev/null) && rc=0 || rc=$?
        if [[ $rc == 255 ]]; then out=unreachable; elif [[ $rc != 0 ]]; then out=absent; fi
    fi
    echo "${out:-absent}"
}

# MemFree in GiB: head, worker (the worker's is empty if ssh fails)
memfree() { awk '/^MemFree:/ {printf "%d", $2/1048576}' /proc/meminfo; }
memfree_worker() { wssh "awk '/^MemFree:/ {printf \"%d\", \$2/1048576}' /proc/meminfo" 2>/dev/null || true; }

mem_gate() {
    [[ "$MEM_GATE_GIB" -gt 0 ]] || return 0
    local t0 h w
    t0=$(date +%s)
    while :; do
        h=$(memfree); w=$(memfree_worker)
        if [[ "${h:-0}" -ge "$MEM_GATE_GIB" && "${w:-0}" -ge "$MEM_GATE_GIB" ]]; then
            log "MemFree ${h} / ${w} GiB (head / worker) >= $MEM_GATE_GIB"; return 0
        fi
        if (( $(date +%s) - t0 >= MEM_GATE_TIMEOUT )); then
            log "MemFree ${h} / ${w:-?} GiB still under $MEM_GATE_GIB after ${MEM_GATE_TIMEOUT}s; starting anyway"
            return 0
        fi
        log "waiting for memory: MemFree ${h} / ${w:-?} GiB (head / worker), want $MEM_GATE_GIB"
        if [[ "$MEM_GATE_DROP_CACHES" == 1 ]]; then   # page cache only (1), never swap: other processes own it
            sync; sudo -n sh -c 'echo 1 > /proc/sys/vm/drop_caches' 2>/dev/null || log "drop_caches: sudo -n refused on the head"
            wssh "sync; sudo -n sh -c 'echo 1 > /proc/sys/vm/drop_caches'" 2>/dev/null || log "drop_caches: refused on the worker"
        fi
        sleep 10
    done
}

preflight() { # read-only checks; non-zero on a problem a retry cannot fix
    local bad=0 st wst h w
    for v in WORKER_SSH HEAD_IP NCCL_SOCKET_IFNAME NCCL_IB_HCA MODEL_PATH HEAD_HF WORKER_HF; do
        [[ -n "${!v:-}" ]] || { log "preflight: $v is not set in $CONFIG"; bad=1; }
    done
    [[ $bad == 0 ]] || return 1
    wssh true || { log "preflight: cannot ssh to $WORKER_SSH"; return 1; }
    docker image inspect "$IMAGE" >/dev/null 2>&1 || { log "preflight: no image $IMAGE here (scripts/serve.sh build)"; bad=1; }
    wssh docker image inspect "$IMAGE" >/dev/null 2>&1 || { log "preflight: no image $IMAGE on the worker"; bad=1; }
    # the RDMA port of the link must be up on both nodes (a down port shows as an NCCL timeout minutes into the load)
    st=$(cat "/sys/class/infiniband/$NCCL_IB_HCA/ports/1/state" 2>/dev/null || echo "")
    wst=$(wssh cat "/sys/class/infiniband/$NCCL_IB_HCA/ports/1/state" 2>/dev/null || echo "")
    for pair in "head:$st" "worker:$wst"; do
        case "${pair#*:}" in
            *ACTIVE*) ;;
            "") log "preflight: warning: cannot read $NCCL_IB_HCA port state on the ${pair%%:*}" ;;
            *) log "preflight: $NCCL_IB_HCA port 1 on the ${pair%%:*} is '${pair#*:}', not ACTIVE"; bad=1 ;;
        esac
    done
    # vm.min_free_kbytes is taken from the GPU's share of unified memory; a mismatch gives the ranks different room
    h=$(cat /proc/sys/vm/min_free_kbytes 2>/dev/null || echo "?"); w=$(wssh cat /proc/sys/vm/min_free_kbytes 2>/dev/null || echo "?")
    [[ "$h" == "$w" ]] || log "preflight: warning: vm.min_free_kbytes differs (head $h, worker $w)"
    # the same driver on both nodes; the vLLM kit holds 580.x (590.x: a CUDA-graph deadlock reported on GB10)
    h=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
    w=$(wssh nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
    [[ "$h" == "$w" ]] || log "preflight: warning: NVIDIA driver differs (head ${h:-?}, worker ${w:-?})"
    [[ "$h $w" != *590.* ]] || log "preflight: warning: driver 590.x (a GB10 CUDA-graph deadlock is reported on 590; 580.x is the known-good line)"
    gpu_state "$GPUWATCH_PREFLIGHT" || bad=1
    return $bad
}

gpu_state() { # $1 = off | on | strict: the GB10 clock / power / slow-state check of both nodes; non-zero on a problem
    [[ "$1" != off && -f scripts/gpuwatch.py ]] || return 0
    local out rc=0 args=(check --state-dir "$STATE_DIR" --worker "$WORKER_SSH" --port "$PORT")
    [[ "$1" == strict ]] && args+=(--strict)
    out=$(python3 scripts/gpuwatch.py "${args[@]}" 2>&1) || rc=$?
    [[ -z "$out" ]] || while IFS= read -r l; do log "preflight: $l"; done <<< "$out"
    case $rc in
        0) return 0 ;;
        1) if [[ "$1" == strict ]]; then log "preflight: GPU warning above (GPUWATCH_PREFLIGHT=strict)"; return 1; fi
           log "preflight: warning: GPU state (above)"; return 0 ;;
        2) log "preflight: a node's GPU is degraded (docs/OPS-GPUWATCH.md: power-drain a clamped node)"; return 1 ;;
        *) log "preflight: warning: cannot check the GPU state (rc $rc)"; return 0 ;;
    esac
}

stop_both() { # both ranks at once
    docker rm -f "$NAME-r0" >/dev/null 2>&1 &
    local p0=$!
    wssh docker rm -f "$NAME-r1" >/dev/null 2>&1 &
    local p1=$!
    wait "$p0" || true; wait "$p1" || true
}

run_canary() { # 0: pass or off; 1: failed
    [[ "$CANARY" == off ]] && return 0
    local args=(--base "$BASE" --min-tpr "$CANARY_MIN_TPR" --min-tps "$CANARY_MIN_TPS")
    [[ -n "${SERVED_NAME:-}" ]] && args+=(--model "$SERVED_NAME")
    [[ "${NO_DRAFTS:-0}" == 1 ]] && args+=(--no-draft)
    [[ -n "$WARMUP_LENGTHS" ]] && args+=(--warm-lengths "$WARMUP_LENGTHS")
    if python3 scripts/canary.py "${args[@]}"; then return 0; fi
    if [[ "$CANARY" == strict ]]; then return 1; fi
    log "canary failed (CANARY=warn: still serving; CANARY=strict fails the start)"
    return 0
}

start_once() { # 0: ready; 1: failed (containers left for the caller to inspect or remove); 2: the canary failed
    stop_both
    mem_gate
    IMAGE_ID=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo "$IMAGE")   # same id after save | load
    # both ranks at once (rank 1 waits for rank 0's TCP store either way); patches/0140
    # shellcheck disable=SC2046
    wssh "mkdir -p '$WORKER_PREPARED' '$WORKER_SESSIONS' && docker run $(run_args 1 "$WORKER_HF")" >/dev/null &
    local wpid=$!
    mkdir -p "$HEAD_PREPARED" "$HEAD_SESSIONS"
    # shellcheck disable=SC2046
    docker run $(run_args 0 "$HEAD_HF") >/dev/null
    wait "$wpid" || { log "rank 1 did not start"; return 1; }
    log "waiting for rank 0 on :$PORT (first start compiles the kernels)"
    local t0 w_fail=0 r1 tick=0
    t0=$(date +%s)
    until curl -sf -m 5 "$BASE/v1/models" >/dev/null; do
        [[ "$(running 0)" == true ]] || { log "rank 0 exited"; docker logs --tail 40 "$NAME-r0" 2>&1; return 1; }
        tick=$((tick + 1))
        # 1 s polling (patches/0140); the worker over ssh every 10th tick
        if (( tick % 10 )); then sleep 1; continue; fi
        r1=$(running 1)
        case "$r1" in
            true) w_fail=0 ;;
            unreachable) w_fail=$((w_fail + 1)); log "worker unreachable over ssh ($w_fail)" ;;
            *) log "rank 1 exited"; wssh docker logs --tail 40 "$NAME-r1" 2>&1; return 1 ;;
        esac
        [[ $w_fail -lt 3 ]] || { log "worker unreachable 3 times in a row"; return 1; }
        (( READY_TIMEOUT <= 0 || $(date +%s) - t0 < READY_TIMEOUT )) || { log "not ready after ${READY_TIMEOUT}s"; return 1; }
        sleep 1
    done
    log "ready after $(( $(date +%s) - t0 )) s"
    curl -s "$BASE/v1/models"; echo
    run_canary || { log "canary failed (CANARY=strict)"; return 2; }
}

run_preflight() { # PREFLIGHT=off: skip; warn: log and go on; strict: non-zero on a problem
    case "$PREFLIGHT" in
        off) return 0 ;;
        strict) preflight ;;
        *) preflight || log "preflight: problems above (PREFLIGHT=warn: starting anyway)"; return 0 ;;
    esac
}

take_lock() { # one start at a time; the watchdog stands down while it is held. Best effort: no lock if unavailable
    [[ "${GLM53_TF_LOCKED:-0}" == 1 ]] && return 0
    GLM53_TF_LOCKED=1
    command -v flock >/dev/null 2>&1 && mkdir -p "$STATE_DIR" 2>/dev/null && { exec 8>"$STATE_DIR/lock"; } 2>/dev/null \
        || { log "no start lock ($STATE_DIR not writable or no flock); going on"; return 0; }
    flock -n 8 || { log "another start (or a watchdog heal) is running"; exit 1; }
}

cmd_start() {
    take_lock
    if gpu_busy; then log "a CUDA process is running on a node; stop the other stack first"; exit 1; fi
    run_preflight || { log "preflight failed; not starting (PREFLIGHT=strict)"; exit 1; }
    local attempt=1
    while :; do
        local rc=0
        start_once || rc=$?
        [[ $rc != 0 ]] || return 0
        if (( attempt >= START_ATTEMPTS )); then
            if [[ $rc == 2 ]]; then     # a degenerate engine must not keep serving
                log "stopping both ranks: they loaded but failed the canary (last logs in $STATE_DIR/canary-fail-r{0,1}.log)"
                docker logs --tail 200 "$NAME-r0" > "$STATE_DIR/canary-fail-r0.log" 2>&1 || true
                wssh docker logs --tail 200 "$NAME-r1" > "$STATE_DIR/canary-fail-r1.log" 2>&1 || true
                stop_both
            else
                log "start failed (attempt $attempt of $START_ATTEMPTS); leaving the containers for 'logs', 'stop' removes them"
            fi
            exit 1
        fi
        log "start failed (attempt $attempt of $START_ATTEMPTS); tearing down and retrying"
        stop_both
        run_preflight || { log "preflight failed; not retrying (PREFLIGHT=strict)"; exit 1; }
        attempt=$((attempt + 1))
    done
}

kernel_log() { # $1 = since (journalctl syntax); the head's kernel log, else dmesg
    journalctl -k -q --no-pager --since "$1" 2>/dev/null || dmesg 2>/dev/null || true
}

cmd_xid() { # Xid events on both nodes since $1 (default: the last hour); non-zero if a fatal one was seen
    local since="${1:--1h}" rc=0
    kernel_log "$since" | python3 scripts/xid.py --label "head: " || rc=1
    wssh "journalctl -k -q --no-pager --since '$since' 2>/dev/null || dmesg 2>/dev/null || true" \
        | python3 scripts/xid.py --label "worker: " || rc=1
    return $rc
}

# -- watchdog --------------------------------------------------------------------------------------------------
# One tick: decide whether the pair is healthy. Stands down while a start runs (lock), when both containers are
# absent (stopped on purpose) and while rank 0 is younger than WATCH_GRACE. A tick is bad when a rank exited, or
# /health fails (connection refused, or 503 with GLM53_TF_HEALTH=strict: a fatal engine error or a stall).
# WATCH_FAILS bad ticks in a row heal (WATCH_HEAL=1: stop + start in the background) at most once every
# WATCH_MIN_HEAL s. A fatal Xid never heals (an off-the-bus GPU needs a power cycle): it is logged, and alerted.
WATCH_GRACE="${WATCH_GRACE:-1800}"
WATCH_FAILS="${WATCH_FAILS:-3}"
WATCH_HEAL="${WATCH_HEAL:-0}"
WATCH_MIN_HEAL="${WATCH_MIN_HEAL:-1800}"
WATCH_INTERVAL="${WATCH_INTERVAL:-60}"
WATCH_ALERT="${WATCH_ALERT:-}"                # a command run with the message as $1 (e.g. a notify script)
# drafter health from /metrics (patches/0150) between checks: alert (never heal) when the decode tokens a round over at
# least WATCH_TPR_ROUNDS rounds fall under WATCH_MIN_TPR (0: off). A dead/mismatched drafter decodes 1 token a round.
WATCH_MIN_TPR="${WATCH_MIN_TPR:-0}"
WATCH_TPR_ROUNDS="${WATCH_TPR_ROUNDS:-500}"

metric() { # $1 = metrics text, $2 = name: the value summed over label sets (0 when absent)
    awk -v n="$2" '$1 ~ "^"n"([{]|$)" {s += $NF} END {printf "%.0f", s + 0}' <<< "$1"
}

tpr_check() { # compare /metrics with the last stored sample; alert on a low tokens-a-round rate
    [[ "$WATCH_MIN_TPR" != 0 ]] || return 0
    local m tok rounds reqs f="$STATE_DIR/tpr" pt pr pq dt dr dq
    m=$(curl -s -m 10 "$BASE/metrics") || return 0
    [[ "$m" == *tensorfold_decode_rounds_total* ]] || return 0
    tok=$(metric "$m" tensorfold_completion_tokens_total); rounds=$(metric "$m" tensorfold_decode_rounds_total)
    reqs=$(metric "$m" tensorfold_requests_total)
    read -r pt pr pq < "$f" 2>/dev/null || { echo "$tok $rounds $reqs" > "$f"; return 0; }
    dt=$((tok - pt)); dr=$((rounds - pr)); dq=$((reqs - pq))
    if (( dt < 0 || dr < 0 || dq < 0 )); then echo "$tok $rounds $reqs" > "$f"; return 0; fi   # restarted: reseed
    (( dr >= WATCH_TPR_ROUNDS )) || return 0          # not enough rounds yet: keep the old sample
    echo "$tok $rounds $reqs" > "$f"
    # the first token of each request comes from the prefill, not a round
    if awk -v t="$dt" -v q="$dq" -v r="$dr" -v m="$WATCH_MIN_TPR" 'BEGIN {exit !((t - q) / r < m)}'; then
        alert "drafter: $(awk -v t="$dt" -v q="$dq" -v r="$dr" 'BEGIN {printf "%.2f", (t - q) / r}') tokens a round over $dr rounds < $WATCH_MIN_TPR"
    fi
}

alert() { log "ALERT: $*"; [[ -z "$WATCH_ALERT" ]] || "$WATCH_ALERT" "$*" || true; }

watch_tick() {
    mkdir -p "$STATE_DIR"
    local fails_f="$STATE_DIR/fails" heal_f="$STATE_DIR/last_heal" xid_f="$STATE_DIR/xid_since"
    local fails r0 r1 age code body now
    fails=$(cat "$fails_f" 2>/dev/null || echo 0)
    now=$(date +%s)
    # a start (by hand or a heal) holds the lock for its whole run
    if ! flock -n "$STATE_DIR/lock" true; then log "watch: a start is in progress; standing down"; return 0; fi
    # Xid events since the last tick (both nodes), alert only
    local since
    since=$(cat "$xid_f" 2>/dev/null || echo "-${WATCH_INTERVAL}s")
    date '+%F %T' > "$xid_f"
    local xids
    if ! xids=$(cmd_xid "$since" 2>&1); then alert "fatal Xid: $xids"; elif [[ -n "$xids" ]]; then log "watch: $xids"; fi
    r0=$(running 0); r1=$(running 1)
    if [[ "$r0" == absent && "$r1" == absent ]]; then
        echo 0 > "$fails_f"; log "watch: both ranks absent (stopped on purpose); standing down"; return 0
    fi
    local bad=""
    if [[ "$r0" != true ]]; then bad="rank 0 $r0"
    elif [[ "$r1" == unreachable ]]; then log "watch: worker unreachable over ssh (not counted)"
    elif [[ "$r1" != true ]]; then bad="rank 1 $r1"
    else
        age=$(( now - $(date -d "$(docker inspect -f '{{.State.StartedAt}}' "$NAME-r0")" +%s) ))
        body=$(curl -s -m 10 -w '\n%{http_code}' "$BASE/health" || true)
        code=${body##*$'\n'}; body=${body%$'\n'*}
        if [[ "$code" == 200 ]]; then :
        elif [[ "$code" == 000 && $age -lt $WATCH_GRACE ]]; then log "watch: loading (${age}s)"; echo 0 > "$fails_f"; return 0
        else bad="/health $code ${body:0:300}"
        fi
    fi
    if [[ -z "$bad" ]]; then echo 0 > "$fails_f"; tpr_check; return 0; fi
    fails=$((fails + 1)); echo "$fails" > "$fails_f"
    log "watch: bad tick $fails of $WATCH_FAILS: $bad"
    (( fails >= WATCH_FAILS )) || return 0
    local last
    last=$(cat "$heal_f" 2>/dev/null || echo 0)
    if [[ "$WATCH_HEAL" != 1 ]]; then alert "unhealthy ($bad); WATCH_HEAL=0, not restarting"; return 1; fi
    if (( now - last < WATCH_MIN_HEAL )); then alert "unhealthy ($bad); healed $((now - last))s ago, waiting"; return 1; fi
    echo "$now" > "$heal_f"; echo 0 > "$fails_f"
    alert "unhealthy ($bad); restarting both ranks"
    # detached: a start takes minutes; the lock makes later ticks stand down until it is done
    setsid env GLM53_TF_LOCKED=1 flock "$STATE_DIR/lock" "$0" restart >>"$STATE_DIR/heal.log" 2>&1 < /dev/null &
    return 1
}

case "${1:-}" in
build)
    docker build -f docker/Dockerfile -t "$IMAGE" .
    log "shipping $IMAGE to $WORKER_SSH"
    docker save "$IMAGE" | wssh docker load
    ;;
start)
    cmd_start ;;
restart)
    take_lock; stop_both; log "stopped"; cmd_start ;;
stop)
    stop_both
    log "stopped"
    ;;
status)
    docker ps -a --filter "name=$NAME" --format '{{.Names}} {{.Status}}'
    wssh "docker ps -a --filter name=$NAME --format '{{.Names}} {{.Status}}'"   # one string: the remote shell re-splits args
    curl -s "$BASE/v1/models" || true; echo
    curl -s "$BASE/health" || true; echo
    ;;
logs)
    if [[ "${2:-0}" == 1 ]]; then wssh docker logs --tail "${TAIL:-80}" "$NAME-r1"; else docker logs --tail "${TAIL:-80}" "$NAME-r0"; fi
    ;;
canary)
    CANARY=strict run_canary ;;
xid)
    cmd_xid "${2:--1h}" ;;
preflight)
    preflight && log "preflight ok" ;;
gpucheck)   # for benchmark runs: strict unless GPUWATCH_PREFLIGHT says otherwise; exits 1 on any problem
    mode="${GPUWATCH_PREFLIGHT:-on}"; [[ "$mode" == on ]] && mode=strict
    gpu_state "$mode" && log "gpucheck ok" ;;
watch)
    if [[ "${2:-}" == --once ]]; then watch_tick; exit $?; fi
    while :; do watch_tick || true; sleep "$WATCH_INTERVAL"; done
    ;;
*)
    sed -n '2,12p' "$0"; exit 2 ;;
esac
