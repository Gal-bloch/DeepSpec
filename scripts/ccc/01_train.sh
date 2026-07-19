#!/bin/bash
# CCC batch job: train the Granite DSpark draft model on a single node with NUM_GPUS
# GPUs (default 8). train.py spawns one worker per VISIBLE GPU, so the GPU count is
# controlled purely by CUDA_VISIBLE_DEVICES / NUM_GPUS below.
#
# RESUME SAFETY: checkpoints are per-rank and ckpt_manager asserts that world_size,
# local_batch_size and rank all match on resume. You MUST resume with the SAME
# NUM_GPUS you started with, or the job aborts. Pick a GPU count and keep it.
set -euo pipefail

# ---- allocation-parameterized paths (override per allocation) ----------------
SCRATCH=${SCRATCH:-/dccstor/knewedge/galbloch}
REPO=${REPO:-${SCRATCH}/DeepSpec}
ENVDIR=${ENVDIR:-${SCRATCH}/envs/granite}
PY="${ENVDIR}/bin/python"
CACHE=${CACHE:-${SCRATCH}/granite_cache/granite_4_1_8b_target_cache}
CKPT_ROOT=${CKPT_ROOT:-${SCRATCH}/granite_ckpt}
CONFIG=${CONFIG:-config/dspark/dspark_granite_4_1_8b.py}
NUM_GPUS=${NUM_GPUS:-8}

export HF_HOME=${HF_HOME:-${SCRATCH}/.cache/hf}
export TMPDIR=${TMPDIR:-${SCRATCH}/tmp}
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}" "${TMPDIR}"
HF_TOKEN_FILE=${HF_TOKEN_FILE:-${SCRATCH}/.hf_token}
[ -f "${HF_TOKEN_FILE}" ] && export HF_TOKEN="$(cat "${HF_TOKEN_FILE}")"

cd "$REPO"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

# Checkpoints/tensorboard on GPFS scratch (not the small home quota).
export DEEPSPEC_BASE_CKPT_DIR="${CKPT_ROOT}/checkpoints"
export DEEPSPEC_BASE_TB_DIR="${CKPT_ROOT}/tensorboard"
mkdir -p "${DEEPSPEC_BASE_CKPT_DIR}" "${DEEPSPEC_BASE_TB_DIR}"

# Expose exactly NUM_GPUS devices -> spawn NUM_GPUS ranks.
export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS-1)))"

echo "=== Training Granite DSpark draft: NUM_GPUS=${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}) ==="
echo "repo=${REPO} env=${ENVDIR} cache=${CACHE}"
echo "ckpt_root=${DEEPSPEC_BASE_CKPT_DIR}"
test -d "${CACHE}" || { echo "ERROR: target cache missing: ${CACHE}" >&2; exit 1; }

# Resume awareness: if a step_latest already exists for this exp, the job WILL resume
# and MUST use the same NUM_GPUS as when it was created.
EXP=$("$PY" -c "from deepspec.utils import load_config;print(load_config('${CONFIG}').exp_name)" 2>/dev/null || echo "?")
LATEST="${DEEPSPEC_BASE_CKPT_DIR}/deepspec/${EXP}/step_latest"
if [ -e "${LATEST}" ]; then
  echo "NOTE: ${LATEST} exists -> this run will RESUME. Ensure NUM_GPUS=${NUM_GPUS} matches the original run."
fi

# num_workers=0 is enforced here too (belt-and-suspenders vs the config default) to
# avoid the CUDAPrefetcher + persistent-workers dataloader deadlock seen in the PoC.
"$PY" train.py \
    --config "${CONFIG}" \
    --opts "data.target_cache_path=${CACHE}" \
    --opts "data.num_workers=0"

echo "Checkpoints:"; ls -1 "${DEEPSPEC_BASE_CKPT_DIR}/deepspec/${EXP}/" 2>/dev/null || true
echo "===JOB_COMPLETE==="
