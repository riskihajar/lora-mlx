#!/usr/bin/env bash
# Local trigger: kick off the Pasal.id Sakana D2L cloud eval job.
#
# Usage:
#   ./run_eval_pasalid.sh
#
# Tunables (env vars):
#   HF_JOB_FLAVOR        default: a10g-small
#   HF_JOB_TIMEOUT       default: 2h
#   HF_JOB_DETACH=1      run detached and print job id
#   PASALID_INPUT_REPO   default: riskihajar/pasalid-d2l-eval-input
#   PASALID_OUTPUT_REPO  default: riskihajar/pasalid-d2l-eval-output
#   PASALID_SPLIT        default: test_seen,test_unseen
#   PASALID_MAX_NEW      default: 256
#   PASALID_RUN_TAG      default: $(date +%Y%m%d_%H%M%S)

set -euo pipefail

cd "$(dirname "$0")"

SCRIPT="eval_pasalid_d2l.sh"
IMAGE="${HF_JOB_IMAGE:-pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel}"
FLAVOR="${HF_JOB_FLAVOR:-a10g-small}"
TIMEOUT="${HF_JOB_TIMEOUT:-2h}"

if [[ ! -f "$SCRIPT" ]]; then
    echo "[run] error: entry script not found: $SCRIPT"
    exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
    echo "[run] error: hf CLI not found. Install with:"
    echo "  pip install -U 'huggingface_hub[cli]'"
    exit 1
fi

DETACH_FLAG=()
if [[ "${HF_JOB_DETACH:-0}" == "1" ]]; then
    DETACH_FLAG=(--detach)
fi

echo "[run] image       = $IMAGE"
echo "[run] flavor      = $FLAVOR"
echo "[run] timeout     = $TIMEOUT"
echo "[run] script      = $SCRIPT"
echo "[run] input_repo  = ${PASALID_INPUT_REPO:-riskihajar/pasalid-d2l-eval-input}"
echo "[run] output_repo = ${PASALID_OUTPUT_REPO:-riskihajar/pasalid-d2l-eval-output}"
echo "[run] split       = ${PASALID_SPLIT:-test_seen,test_unseen}"
echo "[run] max_new     = ${PASALID_MAX_NEW:-256}"

ENV_FLAGS=(
    -e "HF_HUB_ENABLE_HF_TRANSFER=1"
    -e "PASALID_INPUT_REPO=${PASALID_INPUT_REPO:-riskihajar/pasalid-d2l-eval-input}"
    -e "PASALID_OUTPUT_REPO=${PASALID_OUTPUT_REPO:-riskihajar/pasalid-d2l-eval-output}"
    -e "PASALID_SPLIT=${PASALID_SPLIT:-test_seen,test_unseen}"
    -e "PASALID_MAX_NEW=${PASALID_MAX_NEW:-256}"
)
if [[ -n "${PASALID_RUN_TAG:-}" ]]; then
    ENV_FLAGS+=(-e "PASALID_RUN_TAG=$PASALID_RUN_TAG")
fi

hf jobs run \
    --flavor "$FLAVOR" \
    --timeout "$TIMEOUT" \
    -s HF_TOKEN \
    "${ENV_FLAGS[@]}" \
    "${DETACH_FLAG[@]}" \
    "$IMAGE" \
    bash -c "$(cat "$SCRIPT")"
