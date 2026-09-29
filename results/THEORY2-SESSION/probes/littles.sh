#!/usr/bin/env bash
# THEORY-2 item 8 / §6 step 1b: build and run littles.cu (Little's-law / latency probe) on one GB10.
#
#   bash littles.sh OUT_DIR [--quick] [extra littles flags, e.g. --only bulk --gb 8 --no-beside]
#
# Meant to run inside the production image (nvcr.io/nvidia/pytorch:26.07 based, nvcc on PATH), e.g.
#   docker run --rm --gpus all -v $PWD/results/THEORY2-SESSION:/t2 -v $OUT:/out --entrypoint bash IMAGE \
#     -c 'bash /t2/probes/littles.sh /out [--quick]'
# The source is read from this script's directory (may be read-only); the binary and all outputs go to OUT_DIR:
#   littles.build.log  nvcc / ptxas -v (registers, spills) and which target was used
#   littles.log        the human-readable run (tables + the final GATE line)
#   littles.json       machine-readable results
#   littles.smi.txt    nvidia-smi before / after
# Exit status: the probe's (124 if the internal timeout fired). The caller should still wrap this in `timeout`.
set -u
OUT=${1:?usage: littles.sh OUT_DIR [--quick] [littles flags]}
shift
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC="$HERE/littles.cu"
mkdir -p "$OUT"
BIN="$OUT/littles"
QUICK=0
for a in "$@"; do [ "$a" = "--quick" ] && QUICK=1; done
LIMIT=$([ $QUICK = 1 ] && echo 300 || echo 900)   # seconds: 5 min quick, 15 min full

NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
{
    echo "# $(date -Is) host $(hostname) src $SRC"
    "$NVCC" --version | tail -2
} > "$OUT/littles.build.log" 2>&1

# sm_121 native first; sm_120 cubins run on sm_121 (same major); compute_90 PTX is JIT-compiled as a last resort.
built=""
for gen in "-arch=sm_121" "-arch=sm_120" "-gencode arch=compute_90,code=compute_90"; do
    echo "## nvcc -O3 $gen -lineinfo -Xptxas -v" >> "$OUT/littles.build.log"
    # shellcheck disable=SC2086
    if "$NVCC" -O3 $gen -lineinfo -Xptxas -v -o "$BIN" "$SRC" >> "$OUT/littles.build.log" 2>&1; then
        built="$gen"
        break
    fi
done
if [ -z "$built" ]; then
    echo "littles.sh: build failed (see $OUT/littles.build.log)" >&2
    tail -20 "$OUT/littles.build.log" >&2
    exit 3
fi
echo "littles.sh: built with $built" | tee -a "$OUT/littles.build.log"
if grep "spill stores" "$OUT/littles.build.log" | grep -qv " 0 bytes spill stores, 0 bytes spill loads"; then
    echo "littles.sh: WARNING: register spills (see littles.build.log)"
else
    echo "littles.sh: 0 spills in every kernel"
fi

{
    echo "# before $(date -Is)"
    nvidia-smi 2>&1 || true
    nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu,pstate --format=csv 2>&1 || true
} > "$OUT/littles.smi.txt"

if command -v timeout > /dev/null; then TO=(timeout "$LIMIT"); else TO=(); fi
"${TO[@]}" "$BIN" --json "$OUT/littles.json" "$@" 2>&1 | tee "$OUT/littles.log"
rc=${PIPESTATUS[0]}

{
    echo "# after $(date -Is) rc $rc"
    nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu,pstate --format=csv 2>&1 || true
} >> "$OUT/littles.smi.txt"
grep -E "^GATE item8" "$OUT/littles.log" || echo "GATE item8: no result (rc $rc)"
exit "$rc"
