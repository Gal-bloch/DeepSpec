#!/bin/bash
# CCC batch job: train the Granite DSpark draft model on 2x A100-80GB.
# train.py spawns one worker per visible GPU; CUDA_VISIBLE_DEVICES=0,1 -> 2 ranks.
set -euo pipefail

REPO=${REPO:-/dccstor/galbloch/DeepSpec}
CACHE=${CACHE:-/dccstor/galbloch/granite_cache/granite_4_1_8b_target_cache}
CONFIG=config/dspark/dspark_granite_4_1_8b.py

# Keep checkpoints/tensorboard on GPFS, not the 25 GB home quota.
export HOME_CKPT=${HOME_CKPT:-/dccstor/galbloch/granite_ckpt}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate granite
cd "$REPO"

echo "=== Training Granite DSpark draft (2x A100) ==="
echo "cache=${CACHE}"
test -d "${CACHE}" || { echo "ERROR: target cache missing: ${CACHE}" >&2; exit 1; }

# The config writes checkpoints under ~/checkpoints/<project>/<exp>; redirect
# that root onto GPFS via BASE_CKPT_DIR/BASE_TB_DIR overrides.
export DEEPSPEC_BASE_CKPT_DIR="${HOME_CKPT}/checkpoints"
export DEEPSPEC_BASE_TB_DIR="${HOME_CKPT}/tensorboard"
mkdir -p "${DEEPSPEC_BASE_CKPT_DIR}" "${DEEPSPEC_BASE_TB_DIR}"

CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --config "${CONFIG}" \
    --opts "data.target_cache_path=${CACHE}"

echo "Checkpoints:"; ls -1 "${DEEPSPEC_BASE_CKPT_DIR}"/deepspec/dspark_block7_granite_4_1_8b/ 2>/dev/null || true
echo "===JOB_COMPLETE==="
