#!/bin/bash
# Run the minimal acceptance probe on a reserved GPU node.
set -euo pipefail
REPO=${REPO:-/dccstor/knewedge/galbloch/DeepSpec}
ENVDIR=${ENVDIR:-/dccstor/knewedge/galbloch/envs/granite}
PY="${ENVDIR}/bin/python"
export HF_HOME=/dccstor/knewedge/galbloch/.cache/hf
export TMPDIR=/dccstor/knewedge/galbloch/tmp
export HF_TOKEN="$(cat /dccstor/knewedge/galbloch/.hf_token)"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
cd "$REPO"; export PYTHONPATH="$REPO"
DRAFT=/dccstor/knewedge/galbloch/granite_ckpt/checkpoints/deepspec/dspark_block7_granite_4_1_8b/step_latest
echo "probe start $(date +%T) host $(hostname)"
# dump stacks after 240s if it hangs, so we get a real traceback
CUDA_VISIBLE_DEVICES=0 "$PY" -X faulthandler -c "
import faulthandler; faulthandler.dump_traceback_later(240, exit=False)
import runpy, sys
sys.argv=['probe_accept.py','--target','ibm-granite/granite-4.1-8b','--draft','${DRAFT}','--task','gsm8k','--max-samples','8','--max-new-tokens','256']
runpy.run_path('scripts/ccc/probe_accept.py', run_name='__main__')
"
echo "===JOB_COMPLETE==="
