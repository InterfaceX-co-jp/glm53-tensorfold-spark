#!/usr/bin/env bash
# Install or locate RigMark at the pinned revision (scripts/rigmark/rigmark.env). Prints the checkout's path.
#
#   scripts/rigmark/install.sh            clone into RIGMARK_DIR if missing, check out RIGMARK_REV, verify
#   scripts/rigmark/install.sh --check    verify only (no clone / fetch); non-zero if missing or wrong
#
# RigMark is pure Python 3.10+ standard library (no pip, no venv, no compiled parts): on the head node (aarch64, Python
# 3.12) a git clone is the whole install, ~1 MB, no GPU, no daemons. Default RIGMARK_DIR: ~/rigmark on the Spark
# head (next to ~/glm53-tensorfold-spark), else ~/.cache/rigmark.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$here/rigmark.env"
if [[ -z "${RIGMARK_DIR:-}" ]]; then
    if [[ -d "$HOME/glm53-tensorfold-spark" ]]; then RIGMARK_DIR="$HOME/rigmark"; else RIGMARK_DIR="$HOME/.cache/rigmark"; fi
fi
check_only=0; [[ "${1:-}" == --check ]] && check_only=1
say() { echo "[rigmark-install] $*" >&2; }

python3 - <<'EOF' || { say "python3 >= 3.10 is required"; exit 1; }
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
EOF

if [[ ! -d "$RIGMARK_DIR/.git" ]]; then
    [[ $check_only == 0 ]] || { say "no checkout at $RIGMARK_DIR"; exit 1; }
    say "cloning $RIGMARK_URL into $RIGMARK_DIR"
    git clone -q "$RIGMARK_URL" "$RIGMARK_DIR"
fi
if [[ "$(git -C "$RIGMARK_DIR" rev-parse HEAD)" != "$RIGMARK_REV" ]]; then
    [[ $check_only == 0 ]] || { say "$RIGMARK_DIR is not at $RIGMARK_REV"; exit 1; }
    git -C "$RIGMARK_DIR" cat-file -e "$RIGMARK_REV^{commit}" 2>/dev/null || git -C "$RIGMARK_DIR" fetch -q origin
    git -C "$RIGMARK_DIR" -c advice.detachedHead=false checkout -q "$RIGMARK_REV"
fi
# A dirty benchmark tree is recorded in every receipt and blocks a strict `rigmark compare` against a clean one.
dirty=$(git -C "$RIGMARK_DIR" status --porcelain=v1 --untracked-files=all)
if [[ -n "$dirty" ]]; then
    say "$RIGMARK_DIR has local changes; receipts would be marked dirty:"; echo "$dirty" >&2; exit 1
fi
say "rigmark $(git -C "$RIGMARK_DIR" rev-parse --short=12 HEAD) clean at $RIGMARK_DIR"
echo "$RIGMARK_DIR"
