#!/usr/bin/env bash
# Local trigger: kick off an HF Job that validates upstream SakanaAI Doc-to-LoRA
# on a cloud GPU. The job entry point is `validate_sakana_upstream.sh`,
# which is passed inline via `bash -c "$(cat ...)"`.
#
# Usage:
#   ./run_validate_sakana.sh
#
# Override defaults via env vars:
#   HF_JOB_IMAGE=...          docker image (default: pytorch cuda12.4 devel)
#   HF_JOB_FLAVOR=...         hardware flavor (default: a10g-small)
#   HF_JOB_TIMEOUT=...        max duration   (default: 2h)
#   HF_JOB_DETACH=1           run detached and print the job id

set -euo pipefail

cd "$(dirname "$0")"

SCRIPT="validate_sakana_upstream.sh"
IMAGE="${HF_JOB_IMAGE:-pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel}"
FLAVOR="${HF_JOB_FLAVOR:-a10g-small}"
TIMEOUT="${HF_JOB_TIMEOUT:-2h}"

echo "[run] image   = $IMAGE"
echo "[run] flavor  = $FLAVOR"
echo "[run] timeout = $TIMEOUT"
echo "[run] script  = $SCRIPT"

if ! command -v hf >/dev/null 2>&1; then
    echo "[run] error: hf CLI not found. Install with:"
    echo "  pip install -U 'huggingface_hub[cli]'"
    exit 1
fi

if [[ ! -f "$SCRIPT" ]]; then
    echo "[run] error: entry script not found: $SCRIPT"
    exit 1
fi

DETACH_FLAG=()
if [[ "${HF_JOB_DETACH:-0}" == "1" ]]; then
    DETACH_FLAG=(--detach)
fi

# `-s HF_TOKEN` (no value) tells `hf jobs run` to forward the local user's
# HF token into the job as a secret env var. Required for gated Gemma-2.
hf jobs run \
    --flavor "$FLAVOR" \
    --timeout "$TIMEOUT" \
    -s HF_TOKEN \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    "${DETACH_FLAG[@]}" \
    "$IMAGE" \
    bash -c "$(cat "$SCRIPT")"
