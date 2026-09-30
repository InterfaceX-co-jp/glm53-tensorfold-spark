#!/usr/bin/env bash
# BEFORE / AFTER measurement of the OS changes (docs/OS-TUNING.md, "Measure it on your rig"). Changes nothing on the OS.
# Run on the head, from the repo root, against a running server; use the same image and config for every run.
#
#   WORKER_SSH=<user>@<worker> MODEL=<served-name> scripts/os-tuning/measure.sh run BEFORE
#   (apply the OS changes on both nodes, reboot worker first, then head, start the server again)
#   WORKER_SSH=... MODEL=... scripts/os-tuning/measure.sh run AFTER
#   scripts/os-tuning/measure.sh summary BEFORE AFTER
#
# One `run`:
#   1. OS snapshot of both nodes (snapshot.sh, 5 s wakeup sample)
#   2. 60 s idle: MemAvailable / MemFree min + mean on both nodes
#   3. decode, 1 stream: bench/glmbench.py --suites tf x REPS1 (30 s wakeup + mpstat sample on both nodes meanwhile)
#   4. decode, 4 streams: bench/multiturn.py --modes concurrent --streams 4 x REPS4 (same sampling)
#   5. STRESS=1: bench/multiturn.py --modes stress (4 x ~250k conversations; MemAvailable minimum on both nodes)
#   6. OS snapshot of both nodes again
# Environment: BASE (default http://127.0.0.1:${PORT:-8000}), MODEL (served model name), WORKER_SSH, OUT (default
# results/os-tuning), REPS1 (3), REPS4 (6), STRESS (0), MEM_MIN (8.0 GiB, the stress gate).
set -uo pipefail
R=${OUT:-results/os-tuning}; H=$(cd "$(dirname "$0")" && pwd)
BASE=${BASE:-http://127.0.0.1:${PORT:-8000}}
mkdir -p "$R"
log() { echo "[$(date +%T)] $*" | tee -a "$R/measure.log"; }
wssh() { ssh -o BatchMode=yes "$W" "$@"; }

osnap() { # $1 = tag
    sudo -n env WAKE_S=5 bash "$H/snapshot.sh" > "$R/os-$1-head.txt" 2>&1 || WAKE_S=5 bash "$H/snapshot.sh" > "$R/os-$1-head.txt" 2>&1
    wssh "WAKE_S=5 bash -s" < "$H/snapshot.sh" > "$R/os-$1-worker.txt" 2>&1
}

idle() { # $1 = N: 60 s, every 2 s, both nodes
    local i a b rd="awk '/^MemFree/{f=\$2}/^MemAvailable/{v=\$2}END{printf \"%.2f %.2f\",v/1048576,f/1048576}' /proc/meminfo"
    for i in $(seq 1 30); do
        a=$(bash -c "$rd"); b=$(wssh "$rd")
        echo "$(date +%T) $a $b"; sleep 2
    done > "$R/idle-$1.log"
    awk '{for(i=2;i<=5;i++){s[i]+=$i; if(!(i in m)||$i<m[i])m[i]=$i}} END{n=NR;
        printf "idle_memavail_head_min=%.2f\nidle_memavail_head_mean=%.2f\nidle_memfree_head_mean=%.2f\n", m[2], s[2]/n, s[3]/n;
        printf "idle_memavail_worker_min=%.2f\nidle_memavail_worker_mean=%.2f\nidle_memfree_worker_mean=%.2f\n", m[4], s[4]/n, s[5]/n}' \
        "$R/idle-$1.log" > "$R/idle-$1.txt"
    log "idle: $(tr '\n' ' ' < "$R/idle-$1.txt")"
}

sample() { # $1 = tag: 30 s wakeups / interrupts / per-cpu load on both nodes, in parallel
    local ps=()
    (sudo -n env WAKE_S=30 bash "$H/snapshot.sh" > "$R/wake-$1-head.txt" 2>&1) & ps+=($!)
    (wssh "WAKE_S=30 bash -s" < "$H/snapshot.sh" > "$R/wake-$1-worker.txt" 2>&1) & ps+=($!)
    (mpstat -P ALL 30 1 > "$R/mpstat-$1-head.txt" 2>&1) & ps+=($!)
    (wssh "mpstat -P ALL 30 1" > "$R/mpstat-$1-worker.txt" 2>&1) & ps+=($!)
    wait "${ps[@]}"
}

run() {
    local N=$1 p
    : "${W:?}" "${MODEL:?set MODEL to the served model name}"
    curl -sf -m 10 "$BASE/v1/models" >/dev/null || { log "no server at $BASE"; exit 1; }
    log "=== $N start (BASE=$BASE MODEL=$MODEL)"
    osnap "$N-pre"
    idle "$N"
    python3 bench/glmbench.py --base "$BASE" --model "$MODEL" --suites tf --reps "${REPS1:-3}" --label "$N-1s" --out "$R/decode1-$N.json" > "$R/decode1-$N.log" 2>&1 & p=$!
    sleep 20; sample "$N-1s"; wait $p
    python3 bench/multiturn.py --base "$BASE" --model "$MODEL" --modes concurrent --streams 4 --reps "${REPS4:-6}" --long-tokens 512 --out "$R/decode4-$N.json" > "$R/decode4-$N.log" 2>&1 & p=$!
    sleep 20; sample "$N-4s"; wait $p
    if [[ "${STRESS:-0}" == 1 ]]; then
        log "$N: stress"
        python3 bench/multiturn.py --base "$BASE" --model "$MODEL" --modes stress --mem-hosts "local,$W" \
            --mem-log "$R/stress-mem-$N.log" --mem-min "${MEM_MIN:-8.0}" --out "$R/stress-$N.json" > "$R/stress-$N.log" 2>&1
        log "$N: stress rc=$? ($(grep -iE 'min' "$R/stress-$N.log" | tail -3 | tr '\n' ' ' | cut -c1-200))"
    fi
    osnap "$N-end"
    log "=== $N done"
}

summary() { # $@ = run names, first = BEFORE
    python3 - "$R" "$@" <<'PY'
import json, os, re, statistics, sys
R, names = sys.argv[1], sys.argv[2:]
def txt(p):
    try: return open(os.path.join(R, p)).read()
    except OSError: return ""
def kv(p): return {k: v for k, v in (l.split("=", 1) for l in txt(p).splitlines() if "=" in l)}
def medians(n):  # every "median X" in the 1-stream log
    return [float(x) for x in re.findall(r"median ([\d.]+)", txt(f"decode1-{n}.log"))]
def geo(a, b):
    r = [y / x for x, y in zip(a, b) if x and y]
    return round((statistics.geometric_mean(r) - 1) * 100, 2) if r else None
def agg4(n):
    try: d = json.load(open(os.path.join(R, f"decode4-{n}.json")))
    except (OSError, ValueError): return None
    vals = re.findall(r'"aggregate_tps":\s*([\d.]+)', json.dumps(d))
    return round(statistics.mean(map(float, vals)), 2) if vals else None
def snap(n, node, pat):
    m = re.search(pat, txt(f"os-{n}-{node}.txt"), re.M); return m.group(1) if m else None
def nsvc(n, node):
    v = snap(n + "-pre", node, r"## running services\n(.*)"); return len(v.split()) if v else None
def ctx(n, node):
    m = re.search(r"system context switches/s: (\d+)", txt(f"wake-{n}-{node}.txt")); return m.group(1) if m else None
def stress(n):
    t = txt(f"stress-mem-{n}.log")
    return None if not t else "see " + f"stress-{n}.log"
RSS = r"rss ([\d.]+ GiB)"
base = medians(names[0]) if names else []
print("| metric | " + " | ".join(names) + " |")
print("| --- | " + " | ".join("---" for _ in names) + " |")
rows = {
  "idle MemAvailable min head / worker (GiB)": lambda n: f"{kv(f'idle-{n}.txt').get('idle_memavail_head_min')} / {kv(f'idle-{n}.txt').get('idle_memavail_worker_min')}",
  "polkitd RSS head / worker (pre)": lambda n: f"{snap(n + '-pre', 'head', RSS)} / {snap(n + '-pre', 'worker', RSS)}",
  "services running head / worker": lambda n: f"{nsvc(n, 'head')} / {nsvc(n, 'worker')}",
  f"1 stream geomean vs {names[0] if names else '-'} (%)": lambda n: geo(base, medians(n)),
  "4 streams aggregate tok/s (mean)": agg4,
  "1s decode ctx switches/s head / worker": lambda n: f"{ctx(n + '-1s', 'head')} / {ctx(n + '-1s', 'worker')}",
  "4s decode ctx switches/s head / worker": lambda n: f"{ctx(n + '-4s', 'head')} / {ctx(n + '-4s', 'worker')}",
  "stress": stress,
}
for k, f in rows.items():
    print(f"| {k} | " + " | ".join(str(f(n)) for n in names) + " |")
PY
}

W=${WORKER_SSH:-}
case "${1:-}" in
    run) run "${2:?name}" ;;
    summary) shift; summary "$@" | tee "$R/summary-$(IFS=-; echo "$*").md" ;;
    *) sed -n '2,9p' "$0"; exit 2 ;;
esac
