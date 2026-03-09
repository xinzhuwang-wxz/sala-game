#!/usr/bin/env bash

echo "[prepare_env] start $(date '+%F %T')"

# ---- install custom sglang (editable) ----
uv pip install --no-deps -e ./sglang/python

# ---- Official container defaults ----
# Applied when SGLANG_SERVER_ARGS is empty/unset.
# Replicating here because we override the container entrypoint.
_DEFAULT_ARGS="--disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 32768"

# ---- GPTQ mode ----
# Set ENABLE_GPTQ=1 to activate W4A16 quantization (preprocess_model.py does the actual quant).
# In GPTQ mode we must:
#   1. Patch hardcoded bfloat16 -> float16 in minicpm backend files (our editable copy)
#   2. Add GPTQ-specific server args
if [ "${ENABLE_GPTQ:-0}" = "1" ]; then
    echo "[prepare_env] GPTQ mode enabled — patching bfloat16->float16 in minicpm backends"
    BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sglang/python/sglang/srt/layers/attention"

    for pyfile in \
        "${BACKEND_DIR}/minicpm_backend.py" \
        "${BACKEND_DIR}/minicpm_sparse_utils.py"; do
        if [ -f "$pyfile" ]; then
            sed -i 's/torch\.bfloat16/torch.float16/g' "$pyfile"
            sed -i 's/"bfloat16"/"float16"/g'          "$pyfile"
            echo "[prepare_env] patched $(basename "$pyfile")"
        fi
    done
    # Remove stale __pycache__ so Python picks up the patched files
    find "${BACKEND_DIR}/__pycache__" \
        \( -name "minicpm_backend*" -o -name "minicpm_sparse_utils*" \) \
        -delete 2>/dev/null || true

    _GPTQ_ARGS="--quantization gptq --dtype float16 --disable-cuda-graph --dense-as-sparse --skip-server-warmup"
    _DEFAULT_ARGS="${_DEFAULT_ARGS} ${_GPTQ_ARGS}"
fi

if [ -z "${SGLANG_SERVER_ARGS:-}" ]; then
    export SGLANG_SERVER_ARGS="${_DEFAULT_ARGS} --log-level info"
else
    export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS} --log-level info"
fi

echo "[prepare_env] ENABLE_GPTQ=${ENABLE_GPTQ:-0}"
echo "[prepare_env] SGLANG_SERVER_ARGS=${SGLANG_SERVER_ARGS}"
echo "[prepare_env] done"
