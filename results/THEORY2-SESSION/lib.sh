# THEORY2-SESSION (W16) shared settings and helpers; sourced by run.sh, load.sh, ab.sh. Runs on the head node (rank 0).
# Every path can be overridden from the environment.
REPO=${REPO:-$HOME/glm53-tensorfold-spark}
REPO2=${REPO2:-$HOME/glm53-tensorfold-spark}                 # the worker node copy
W=${WORKER_SSH:?set WORKER_SSH (the worker's ssh target, as in config/prod.env)}                         # worker over the CX7 link (as W11 / W12)
R=${W16:-$REPO/results/W16}                                  # every log of the session
T2=$REPO/results/THEORY2-SESSION
IMAGE=${IMAGE:-glm53-tensorfold:b8}                          # b7 (prod) + 0510 / 0520 / 0530 (all knobs off by default)
B=${BASE:-http://127.0.0.1:8000}
M=${MODEL:-GLM-5.3-Flash-EXL3}
LEASE=${LEASE:-$HOME/.test-window-lease}
HTTPS_URL=${HTTPS_URL:-https://<head-host>/v1/models}
PROD_CLOCKS=${PROD_CLOCKS:-300,2250}                         # dgx-gpu-clock-cap.service's -lgc
LOCK_CLOCKS=${LOCK_CLOCKS:-2250,2250}                        # THEORY-2 §7: probes at the production cap, locked
DARGS="--gpus all --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK --network host --ipc host \
-v glm53-tf-cache:/cache -e PYTHONDONTWRITEBYTECODE=1 -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e NCCL_IB_HCA=rocep1s0f1"
PYP="PYTHONPATH=/src/TensorFold/src:/src/TensorFold/tests/cuda:/work/tests/cuda:/work/tests"

mkdir -p "$R"
log() { echo "[w16 $(date +%T)] $*" | tee -a "$R/session.log" >&2; }
wssh() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$W" "$@"; }
health_ok() { curl -s -m 5 "$B/health" | grep -q '"ok": true'; }
gpu_apps() { nvidia-smi --query-compute-apps=pid,name --format=csv,noheader 2>/dev/null; }
clocks() { # clocks NAME: SM / memory clocks and the application limits on both nodes into $R/clocks-NAME.log
    { echo "== head $(date +%T)"; nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu --format=csv
      nvidia-smi -q -d CLOCK | grep -A3 -iE 'locked|applications clocks' | head -20
      echo "== worker"; wssh "nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu --format=csv; nvidia-smi -q -d CLOCK | grep -A3 -iE 'locked|applications clocks' | head -20"
    } > "$R/clocks-$1.log" 2>&1
}
set_clocks() { # set_clocks MIN,MAX on both nodes (head needs passwordless sudo; worker is root)
    sudo -n nvidia-smi -lgc "$1" > /dev/null 2>&1 || log "WARNING: head nvidia-smi -lgc $1 failed (sudo -n?)"
    wssh "nvidia-smi -lgc $1" > /dev/null 2>&1 || log "WARNING: worker nvidia-smi -lgc $1 failed"
    log "clocks -lgc $1 on both nodes"
}
clear_w16() { # leftover w16-* probe containers on both nodes
    docker ps -a --format '{{.Names}}' | grep -E '^w16-' | xargs -r docker rm -f > /dev/null 2>&1
    wssh "docker ps -a --format '{{.Names}}' | grep -E '^w16-' | xargs -r docker rm -f" > /dev/null 2>&1
    true
}
