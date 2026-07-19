#!/bin/bash
# Submit Granite DSpark CCC jobs via LSF bsub. Parameterized for any single-node
# multi-GPU allocation (not just the NCU-reserved node).
#
#   bash scripts/ccc/submit.sh smoke   # pre-flight 8-GPU smoke + resume test (do FIRST)
#   bash scripts/ccc/submit.sh data    # full-data target-cache build
#   bash scripts/ccc/submit.sh train   # full drafter training
#
# Allocation knobs (env): NUM_GPUS (default 8), GMODEL, RESERVATION (optional -U),
#   QUEUE (optional -q), WALLTIME minutes (default 10080 = 7d), NCORES, SCRATCH.
# NOTE: no -M/-hl — a hard memory limit produced a bogus 200 TB MEMLIMIT that made
# GPU jobs un-schedulable on CCC. Let LSF use node defaults.
set -euo pipefail

STAGE=${1:?usage: submit.sh {smoke|data|train}}
SCRATCH=${SCRATCH:-/dccstor/knewedge/galbloch}
REPO=${REPO:-${SCRATCH}/DeepSpec}
NUM_GPUS=${NUM_GPUS:-8}
GROUP=${GROUP:-grp_ai_compiler_design}  # bv (BlueVela) esub requires -G <group>
GMODEL=${GMODEL:-NVIDIAH10080GBHBM3}
NCORES=${NCORES:-$((NUM_GPUS > 4 ? NUM_GPUS : 4))}
WALLTIME=${WALLTIME:-10080}
OUT="${HOME}/%J.stdout"; ERR="${HOME}/%J.stderr"

# Forward everything the job scripts read.
export SCRATCH REPO NUM_GPUS
export CACHE=${CACHE:-${SCRATCH}/granite_cache/granite_4_1_8b_target_cache}
export CKPT_ROOT=${CKPT_ROOT:-${SCRATCH}/granite_ckpt}

# Optional reservation / queue (only added if set).
RES_FLAG=(); [ -n "${RESERVATION:-}" ] && RES_FLAG=(-U "${RESERVATION}")
Q_FLAG=(); [ -n "${QUEUE:-}" ] && Q_FLAG=(-q "${QUEUE}")

case "$STAGE" in
  smoke) SCRIPT="${REPO}/scripts/ccc/00b_smoke.sh"; W=${SMOKE_WALL:-90} ;;
  data)  SCRIPT="${REPO}/scripts/ccc/00_data.sh";  W="${WALLTIME}" ;;
  train) SCRIPT="${REPO}/scripts/ccc/01_train.sh"; W="${WALLTIME}" ;;
  *) echo "unknown stage: $STAGE (want smoke|data|train)" >&2; exit 1 ;;
esac

set -x
bsub -G "${GROUP}" -n "${NCORES}" -R "span[hosts=1]" \
     -gpu "num=${NUM_GPUS}:gmodel=${GMODEL}" \
     -W "${W}" \
     "${RES_FLAG[@]}" "${Q_FLAG[@]}" \
     -o "$OUT" -e "$ERR" -env "all" \
     bash "$SCRIPT"
set +x

echo "Submitted ${STAGE} (NUM_GPUS=${NUM_GPUS}, gmodel=${GMODEL}, wall=${W}m${RESERVATION:+, -U ${RESERVATION}})."
echo "Watch: bjobs ; tail -f ${HOME}/<jobid>.stdout   (job prints ===JOB_COMPLETE===)"
