# shellcheck shell=bash
# Shared helpers for scripts/os-tuning/*.sh (docs/OS-TUNING.md). Sourced, never run directly.
#
# Every apply-NN script records the state it changes in $STATE_DIR/NN-*.state BEFORE changing anything, and the
# matching rollback-NN script restores exactly that recorded state (a unit that was disabled before stays disabled).
# Run on the node itself, as root. DRY_RUN=1 prints the commands instead of running them.
#
# Environment:
#   STATE_DIR           where the saved state lives (default /var/lib/spark-os-tuning)
#   DRY_RUN=1           preview only
#   HOUSEKEEPING_CPUS   cores for OS noise; default = the 10 Cortex-A725 cores of a GB10 (0-4,10-14)
#   ENGINE_CPUS         cores left to the engine; default = the 10 Cortex-X925 cores of a GB10 (5-9,15-19)
#   NODE_ROLE           head | worker (only used by checks that differ per node; unset = no role-specific checks)
#   LINGER_USER         (head) the user that owns the systemd user units (watchdog etc.); apply-02 checks linger for it
set -uo pipefail

STATE_DIR="${STATE_DIR:-/var/lib/spark-os-tuning}"
DRY_RUN="${DRY_RUN:-0}"
HOUSEKEEPING_CPUS="${HOUSEKEEPING_CPUS:-0-4,10-14}"
ENGINE_CPUS="${ENGINE_CPUS:-5-9,15-19}"
NODE_ROLE="${NODE_ROLE:-}"
SERVER_CONTAINER_RE="${SERVER_CONTAINER_RE:-^glm53-tf-r}"   # the rank containers scripts/serve.sh starts

log() { echo "[$(date '+%F %T')] $(hostname -s) $*"; }
die() { log "ERROR: $*"; exit 1; }
run() { if [[ "$DRY_RUN" == 1 ]]; then echo "DRY: $*"; else "$@"; fi; }

need_root() { [[ $EUID -eq 0 ]] || die "run as root (sudo $0)"; }

state_file() { mkdir -p "$STATE_DIR"; echo "$STATE_DIR/$1.state"; }

# unit_exists UNIT: the unit file is known to systemd
unit_exists() { systemctl list-unit-files --no-legend "$1" 2>/dev/null | grep -q .; }

# record_units STATEFILE UNIT...: "unit enabled-state active-state" per existing unit (appends; first record wins)
record_units() {
    local f=$1; shift
    local u
    for u in "$@"; do
        unit_exists "$u" || continue
        grep -q "^$u " "$f" 2>/dev/null && continue
        echo "$u $(systemctl is-enabled "$u" 2>/dev/null | head -1) $(systemctl is-active "$u" 2>/dev/null | head -1)" >> "$f"
    done
}

# disable_units UNIT...: disable + stop units that exist; never masks
disable_units() {
    local u
    for u in "$@"; do
        unit_exists "$u" || { log "skip $u (not installed)"; continue; }
        log "disable --now $u"
        run systemctl disable --now "$u" 2>&1 | grep -v '^Removed' || true
    done
}

# restore_units STATEFILE: re-enable / restart what the state file says was enabled / active
restore_units() {
    local f=$1 u en ac
    [[ -f "$f" ]] || die "no state file $f (nothing was applied on this node?)"
    while read -r u en ac; do
        [[ -n "$u" ]] || continue
        case "$en" in
            enabled|enabled-runtime) log "enable $u"; run systemctl enable "$u" >/dev/null 2>&1 ;;
            masked) run systemctl mask "$u" >/dev/null 2>&1 ;;
            *) : ;;   # static / disabled / indirect: leave as the package has it
        esac
        if [[ "$ac" == active ]]; then log "start $u"; run systemctl start "$u" || log "WARN: $u did not start"; fi
    done < "$f"
}

# the interface that carries the default route (your uplink; these scripts never touch it)
default_if() { ip -4 route show default 2>/dev/null | awk '{for(i=1;i<NF;i++) if($i=="dev"){print $(i+1); exit}}'; }

assert_network_kept() { # refuse to continue if the default route, NetworkManager or (if installed) Tailscale is gone
    local ifc; ifc=$(default_if)
    [[ -n "$ifc" ]] || die "no default route - stop and roll back"
    systemctl is-active -q NetworkManager || die "NetworkManager is not active - stop and roll back"
    if unit_exists tailscaled.service && systemctl is-enabled -q tailscaled 2>/dev/null; then
        systemctl is-active -q tailscaled || die "tailscaled is not active - stop and roll back"
    fi
    log "network ok: default via $ifc, NetworkManager active"
}

server_running_here() { docker ps --format '{{.Names}}' 2>/dev/null | grep -qE "$SERVER_CONTAINER_RE"; }
