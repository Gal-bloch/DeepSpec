#!/bin/bash
# Submit Granite DSpark CCC jobs via LSF bsub.
#   bash scripts/ccc/submit.sh data    # data pipeline (1 node, 2 GPUs for cache)
#   bash scripts/ccc/submit.sh train   # draft training (2 A100-80GB)
# Run this FROM a ccc-login node (inside the repo on GPFS).
set -euo pipefail

STAGE=${1:?usage: submit.sh {data|train}}
REPO=${REPO:-/dccstor/knewedge/galbloch/DeepSpec}
GMODEL=${GMODEL:-NVIDIAA100_SXM4_80GB}
OUT="${HOME}/%J.stdout"
ERR="${HOME}/%J.stderr"

# Export REPO/CACHE into the job environment so the scripts pick them up.
export REPO
export CACHE=${CACHE:-/dccstor/knewedge/galbloch/granite_cache/granite_4_1_8b_target_cache}
export HOME_CKPT=${HOME_CKPT:-/dccstor/knewedge/galbloch/granite_ckpt}

case "$STAGE" in
  data)
    SCRIPT="${REPO}/scripts/ccc/00_data.sh"
    # Serving + cache build; 2 GPUs, generous memory + walltime for regen.
    bsub -M 204800 -hl -n 8 -R "span[hosts=1]" \
         -gpu "num=2:gmodel=${GMODEL}" \
         -W 1440 \
         -o "$OUT" -e "$ERR" \
         -env "all" \
         bash "$SCRIPT"
    ;;
  train)
    SCRIPT="${REPO}/scripts/ccc/01_train.sh"
    bsub -M 204800 -hl -n 8 -R "span[hosts=1]" \
         -gpu "num=2:gmodel=${GMODEL}" \
         -W 1440 \
         -o "$OUT" -e "$ERR" \
         -env "all" \
         bash "$SCRIPT"
    ;;
  *)
    echo "unknown stage: $STAGE (want data|train)" >&2; exit 1;;
esac

echo "Submitted ${STAGE}. Watch with: bjobs ; tail -f ${HOME}/<jobid>.stdout"
echo "Job prints ===JOB_COMPLETE=== on success."
