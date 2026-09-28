#!/usr/bin/env bash
# Container entrypoint: one TensorFold rank. All configuration comes from the environment
# (scripts/serve.sh passes it from config/*.env).
set -euo pipefail
# patches/0140: the [boot] timeline counts from the launcher's docker run (GLM53_TF_LAUNCH_T0) or from here
export GLM53_TF_T0="${GLM53_TF_T0:-$(date +%s.%N)}"
if [[ -n "${GLM53_TF_LAUNCH_T0:-}" ]]; then
    echo "[boot] r${RANK:-?} +$(awk -v a="$GLM53_TF_T0" -v b="$GLM53_TF_LAUNCH_T0" 'BEGIN {printf "%.1f", a - b}')s container started"
fi
mkdir -p "${CUDA_CACHE_PATH:-/cache/nv/ComputeCache}" "${TRITON_CACHE_DIR:-/cache/triton}" 2>/dev/null || true

: "${RANK:?RANK=0|1}"
: "${MASTER:?MASTER=<rank 0 address on the CX7 link>}"
: "${MODEL_PATH:?MODEL_PATH=<checkpoint dir inside the container>}"

# A start killed mid-compile (e.g. OOM) leaves torch.utils.cpp_extension's FileBaton `lock` in the cache volume,
# and the next start waits on it forever. This container is the only one compiling into its volume at startup
# (the server holds the GPU; tests run with --entrypoint bash and don't come through here), so any lock present
# now is stale.
if [[ -n "${TORCH_EXTENSIONS_DIR:-}" && -d "$TORCH_EXTENSIONS_DIR" ]]; then
    while IFS= read -r -d '' lock; do
        echo "[glm53-tf] removing stale torch extension lock $lock"
        rm -f -- "$lock"
    done < <(find "$TORCH_EXTENSIONS_DIR" -maxdepth 3 -type f -name lock -print0)
fi

args=(serve "$MODEL_PATH" --tp 2 --rank "$RANK" --master "$MASTER" --master-port "${MASTER_PORT:-29551}"
      --no-update-check)
[[ -n "${DRAFTER:-}" ]] && args+=(--drafter "$DRAFTER")
[[ -n "${CONTEXT:-}" ]] && args+=(--context "$CONTEXT")
[[ -n "${SERVED_NAME:-}" ]] && args+=(--name "$SERVED_NAME")
[[ -n "${MAX_TOKENS:-}" ]] && args+=(--max-tokens "$MAX_TOKENS")
[[ -n "${MTP_DRAFTS:-}" ]] && args+=(--mtp-drafts "$MTP_DRAFTS")
[[ "${NO_DRAFTS:-0}" == 1 ]] && args+=(--no-drafts)
if [[ "$RANK" == 0 ]]; then
    args+=(--host "${HOST:-127.0.0.1}" --port "${PORT:-8080}")
fi
# shellcheck disable=SC2206
args+=(${EXTRA_ARGS:-})

echo "[glm53-tf] rank $RANK: tensorfold ${args[*]}"
exec tensorfold "${args[@]}"
