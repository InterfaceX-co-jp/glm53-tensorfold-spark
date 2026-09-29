#!/usr/bin/env bash
# Side-by-side markdown of the vLLM and TensorFold receipts -> results/rigmark/compare-<date>.md (and stdout).
#
#   scripts/rigmark/compare.sh [VLLM_DIR_OR_JSON TENSORFOLD_DIR_OR_JSON]
#       default: the newest results/rigmark/vllm-* and results/rigmark/tensorfold-* directories with a receipt
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
# shellcheck disable=SC1091
source "$here/rigmark.env"
newest() { # newest $OUT_ROOT/$1-* directory holding a finished receipt (a .sha256 written by run.sh)
    local d
    for d in $(ls -1dt "$repo/$OUT_ROOT/$1"-* 2>/dev/null); do
        compgen -G "$d/*.json.sha256" > /dev/null && { echo "$d"; return 0; }
    done
    echo "no finished $1 run under $OUT_ROOT" >&2; return 1
}
left="${1:-$(newest vllm)}"
right="${2:-$(newest tensorfold)}"
rigmark_dir="$("$here/install.sh" --check)"
out="$repo/$OUT_ROOT/compare-$(date -u +%Y%m%d-%H%M%S).md"
python3 "$here/compare.py" "$left" "$right" --rigmark "$rigmark_dir" --out "$out"
echo "[rigmark-compare] wrote ${out#"$repo"/}" >&2
