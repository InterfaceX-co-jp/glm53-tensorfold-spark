#!/usr/bin/env bash
# Inside the image: run each given test file in its own pytest process; one summary line a file.
# Usage: run_tests_in_image.sh <logdir> [ENV=VAL ...] -- file...
set -u
log=$1; shift
envs=()
while [[ $# -gt 0 && "$1" != "--" ]]; do envs+=("$1"); shift; done
shift
pip install -q pytest >/dev/null 2>&1
mkdir -p "$log"
cd /work
for f in "$@"; do
    tag=$(basename "$f" .py)
    [[ ${#envs[@]} -gt 0 ]] && tag="$tag.$(echo "${envs[*]}" | tr ' =,/' '____')"
    t0=$(date +%s)
    env "${envs[@]}" PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda python -m pytest -q -p no:cacheprovider -s "$f" > "$log/$tag.log" 2>&1
    rc=$?
    summary=$(grep -E "^(=+ )?[0-9]+ (passed|failed|error)|passed|failed|no tests ran|error" "$log/$tag.log" | tail -1)
    echo "$tag rc=$rc $(( $(date +%s) - t0 ))s :: $summary" | tee -a "$log/SUMMARY"
done
