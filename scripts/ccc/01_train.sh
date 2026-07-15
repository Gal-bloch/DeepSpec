#!/bin/bash
# CCC batch job: train the Granite DSpark draft model on 2x A100-80GB.
# train.py spawns one worker per visible GPU; CUDA_VISIBLE_DEVICES=0,1 -> 2 ranks.
set -euo pipefail

REPO=${REPO:-/dccstor/knewedge/galbloch/DeepSpec}
ENVDIR=${ENVDIR:-/dccstor/knewedge/galbloch/envs/granite}
PY="${ENVDIR}/bin/python"
CACHE=${CACHE:-/dccstor/knewedge/galbloch/granite_cache/granite_4_1_8b_target_cache}
CONFIG=config/dspark/dspark_granite_4_1_8b.py

# Keep checkpoints/tensorboard + HF cache on GPFS, not the near-full home quota.
export HOME_CKPT=${HOME_CKPT:-/dccstor/knewedge/galbloch/granite_ckpt}
export HF_HOME=${HF_HOME:-/dccstor/knewedge/galbloch/.cache/hf}
export TMPDIR=${TMPDIR:-/dccstor/knewedge/galbloch/tmp}
mkdir -p "${HF_HOME}" "${TMPDIR}"
cd "$REPO"
# Repo is not pip-installed; ensure `import deepspec` resolves from repo root.
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

echo "=== Training Granite DSpark draft (2x A100) ==="
echo "cache=${CACHE}"
test -d "${CACHE}" || { echo "ERROR: target cache missing: ${CACHE}" >&2; exit 1; }

# The config writes checkpoints under ~/checkpoints/<project>/<exp>; redirect
# that root onto GPFS via BASE_CKPT_DIR/BASE_TB_DIR overrides.
export DEEPSPEC_BASE_CKPT_DIR="${HOME_CKPT}/checkpoints"
export DEEPSPEC_BASE_TB_DIR="${HOME_CKPT}/tensorboard"
mkdir -p "${DEEPSPEC_BASE_CKPT_DIR}" "${DEEPSPEC_BASE_TB_DIR}"

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES:-0} "$PY" train.py \
    --config "${CONFIG}" \
    --opts "data.target_cache_path=${CACHE}"

echo "Checkpoints:"; ls -1 "${DEEPSPEC_BASE_CKPT_DIR}"/deepspec/dspark_block7_granite_4_1_8b/ 2>/dev/null || true
echo "===JOB_COMPLETE==="
