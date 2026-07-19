#!/bin/bash
# CCC batch job: evaluate the trained Granite DSpark drafter's speculative-decoding
# acceptance against the Granite-4.1-8b target. Single GPU on a reserved node.
set -euo pipefail

REPO=${REPO:-/dccstor/knewedge/galbloch/DeepSpec}
ENVDIR=${ENVDIR:-/dccstor/knewedge/galbloch/envs/granite}
PY="${ENVDIR}/bin/python"
CKPT_ROOT=${CKPT_ROOT:-/dccstor/knewedge/galbloch/granite_ckpt/checkpoints/deepspec/dspark_block7_granite_4_1_8b}
DRAFT=${DRAFT:-${CKPT_ROOT}/step_latest}
TARGET=${TARGET:-ibm-granite/granite-4.1-8b}

export HF_HOME=${HF_HOME:-/dccstor/knewedge/galbloch/.cache/hf}
export TMPDIR=${TMPDIR:-/dccstor/knewedge/galbloch/tmp}
HF_TOKEN_FILE=${HF_TOKEN_FILE:-/dccstor/knewedge/galbloch/.hf_token}
[ -f "${HF_TOKEN_FILE}" ] && export HF_TOKEN="$(cat "${HF_TOKEN_FILE}")"
export TOKENIZERS_PARALLELISM=false
cd "$REPO"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

echo "=== Eval Granite DSpark drafter ==="
echo "target=${TARGET}"
echo "draft =${DRAFT}"
test -e "${DRAFT}" || { echo "ERROR: draft checkpoint missing: ${DRAFT}" >&2; exit 1; }

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES:-0} "$PY" eval.py \
    --target_name_or_path "${TARGET}" \
    --draft_name_or_path "${DRAFT}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-512}"

echo "===JOB_COMPLETE==="
