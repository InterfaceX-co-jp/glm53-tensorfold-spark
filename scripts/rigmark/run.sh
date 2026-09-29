#!/usr/bin/env bash
# Run RigMark's standard suite against one engine and keep everything under results/rigmark/<engine>-<date>/.
#
#   scripts/rigmark/run.sh vllm|tensorfold [BASE_URL] [MODEL]
#       BASE_URL  default http://127.0.0.1:8000 (run on the head node: loopback, no VPN / proxy in the path)
#       MODEL     default GLM-5.3-Flash-EXL3 (both the vLLM kit and TensorFold prod serve this id)
#
# Steps: install / verify the pinned RigMark (install.sh) -> idle check (no request in flight) -> preflight.py (every
# request shape the suite sends; a few seconds of GPU) -> metadata.py (live config) -> `rigmark run` with the standard
# settings (no count / length / depth / concurrency flags) and the sweep's EXTRA_BODY -> card + checksums.
# Knobs (scripts/rigmark/rigmark.env, a caller export wins): COMPARISON_ID, EXTRA_BODY, SKIP_PREFILL=auto|0|1,
# RIGMARK_DIR, OUT_ROOT, ALLOW_BUSY=1 (run although requests are in flight; disclosed in the metadata),
# SKIP_PREFLIGHT=1, DRY_RUN=1 (print the rigmark command, run nothing against the server).
# Does not start, stop or restart any server: docs/RIGMARK.md (window.sh) does that.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
# shellcheck disable=SC1091
source "$here/rigmark.env"

engine="${1:-}"
case "$engine" in
    vllm) label="$LABEL_VLLM" ;;
    tensorfold) label="$LABEL_TENSORFOLD" ;;
    *) sed -n '2,15p' "$0"; exit 2 ;;
esac
base="${2:-http://127.0.0.1:8000}"; base="${base%/}"; base="${base%/v1}"
model="${3:-GLM-5.3-Flash-EXL3}"
say() { echo "[rigmark-run] $*" >&2; }

rigmark_dir="$("$here/install.sh")"
stamp="$(date -u +%Y%m%d-%H%M%S)"
out="$repo/$OUT_ROOT/$engine-$stamp"
mkdir -p "$out"
say "engine $engine, $base, model $model -> ${out#"$repo"/}"

# --- quiet endpoint ------------------------------------------------------------------------------------------------
curl -sf -m 10 "$base/v1/models" > "$out/models.json" || { say "no answer from $base/v1/models"; exit 1; }
busy=""
if [[ "$engine" == tensorfold ]]; then
    inflight=$(curl -sf -m 10 "$base/health" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("inflight", 0))' || echo "?")
    [[ "$inflight" == 0 ]] || busy="TensorFold /health inflight=$inflight"
else
    running=$(curl -sf -m 10 "$base/metrics" | awk '/^vllm:num_requests_(running|waiting)[{ ]/ {s += $NF} END {print s + 0}' || echo "?")
    [[ "$running" == 0 ]] || busy="vLLM num_requests_running+waiting=$running"
fi
traffic="none: endpoint idle at start (checked), test window held, loopback client"
if [[ -n "$busy" ]]; then
    [[ "${ALLOW_BUSY:-0}" == 1 ]] || { say "endpoint busy ($busy); wait, or ALLOW_BUSY=1 to run and disclose it"; exit 1; }
    traffic="NOT idle at start: $busy (run anyway, ALLOW_BUSY=1)"
fi

# --- preflight -----------------------------------------------------------------------------------------------------
skip_prefill=()
if [[ "${SKIP_PREFLIGHT:-0}" != 1 && "${DRY_RUN:-0}" != 1 ]]; then
    rc=0
    python3 "$here/preflight.py" --base-url "$base" --model "$model" --extra-body "$EXTRA_BODY" \
        --out "$out/preflight.json" > /dev/null || rc=$?
    python3 -c 'import json,sys; [print("[rigmark-run] preflight:", g, file=sys.stderr) for g in json.load(open(sys.argv[1]))["gaps"]]' "$out/preflight.json"
    if [[ $rc == 1 ]]; then say "preflight: decode requests fail; RigMark cannot run (see $out/preflight.json)"; exit 1; fi
    if [[ $rc == 3 ]]; then
        case "$SKIP_PREFILL" in
            auto|1) say "preflight: no /tokenize or token-id completions: running with --skip-prefill (prefill rows absent; compare.sh will note the mismatch)"
                    skip_prefill=(--skip-prefill) ;;
            *) say "preflight: prefill path unsupported and SKIP_PREFILL=0"; exit 1 ;;
        esac
    fi
fi
[[ "$SKIP_PREFILL" == 1 ]] && skip_prefill=(--skip-prefill)

# --- metadata ------------------------------------------------------------------------------------------------------
python3 "$here/metadata.py" "$engine" --out "$out/metadata.json" --base-url "$base" \
    --competing-traffic "$traffic" > /dev/null

# --- run -----------------------------------------------------------------------------------------------------------
cmd=(python3 "$rigmark_dir/rigmark" run --base-url "$base" --model "$model" --label "$label"
     --comparison-id "$COMPARISON_ID" --metadata "$out/metadata.json" --extra-body "$EXTRA_BODY"
     --output "$out/$label.json" "${skip_prefill[@]}")
{
    printf '%q ' "${cmd[@]}"; echo
    echo "# rigmark $(git -C "$rigmark_dir" rev-parse HEAD), started $(date -u +%FT%TZ) on $(uname -m)"
} > "$out/command.txt"
if [[ "${DRY_RUN:-0}" == 1 ]]; then cat "$out/command.txt"; exit 0; fi
say "running the standard suite (about 20-30 min); log: ${out#"$repo"/}/run.log"
t0=$(date +%s)
set +e
( cd "$rigmark_dir" && NO_COLOR=1 "${cmd[@]}" ) 2>&1 | tee "$out/run.log"
rc=${PIPESTATUS[0]}
set -e
echo "# finished $(date -u +%FT%TZ), rc=$rc, $(( $(date +%s) - t0 )) s" >> "$out/command.txt"
if [[ $rc != 0 ]]; then say "rigmark exited $rc; partial receipt (if any) kept in ${out#"$repo"/}"; exit "$rc"; fi
( cd "$out" && sha256sum "$label.json" > "$label.json.sha256" )
say "done: ${out#"$repo"/}/$label.json (+ .card.txt); compare with scripts/rigmark/compare.sh"
