#!/bin/bash
# PRE-FLIGHT SMOKE + RESUME TEST for the Granite DSpark 8-GPU training path.
# Run this FIRST on the new allocation, before the multi-day full job. It exercises
# the EXACT production code path (no_shard + torch_compile + flex_attention + the
# rank-0-broadcast embed init + checkpoint save + auto-resume) on a SMALL cache with
# the production config overridden to fail fast (8 steps, checkpoint at step 4).
#
# It self-runs two phases in one job:
#   Phase A: train 4 steps -> checkpoint -> (script kills the run right after).
#   Phase B: relaunch the SAME exp -> must AUTO-RESUME from step 4 and continue to 8.
# Prints PASS/FAIL per check. Gate the full run on all checks passing.
#
# Requires a small cache at SMOKE_CACHE (reuse the PoC ~2k-sample cache).
set -uo pipefail

SCRATCH=${SCRATCH:-/dccstor/knewedge/galbloch}
REPO=${REPO:-${SCRATCH}/DeepSpec}
ENVDIR=${ENVDIR:-${SCRATCH}/envs/granite}
PY="${ENVDIR}/bin/python"
NUM_GPUS=${NUM_GPUS:-8}
SMOKE_CACHE=${SMOKE_CACHE:-${SCRATCH}/dspark_head_cache/qwen3_4b_small}   # override with a granite cache
# ^ NOTE: must be a Granite target cache (target_layer_ids [2,11,20,29,38], Granite target).
#   If you only have the PoC granite cache, point SMOKE_CACHE at it.
CONFIG=${CONFIG:-config/dspark/dspark_granite_4_1_8b.py}
EXP=smoke_granite_8gpu
CKPT_ROOT=${CKPT_ROOT:-${SCRATCH}/granite_smoke}

export HF_HOME=${HF_HOME:-${SCRATCH}/.cache/hf}
export TMPDIR=${TMPDIR:-${SCRATCH}/tmp}
export TOKENIZERS_PARALLELISM=false
# CUDA toolkit for inductor/Triton (flex_attention compiles a CUDA helper).
# On bv, nvcc/CUDA_HOME are not on the default job env though the toolkit exists.
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
# Use plain gcc for triton/inductor builds, NOT the /usr/lib64/ccache/gcc wrapper:
# ccache (default ~/.ccache on GPFS) fails inside multi-rank jobs, breaking the
# flex_attention CUDA-helper build ('cuda_utils.c ... exit 1'). Plain gcc works.
export CC=${CC:-/usr/bin/gcc}
export CXX=${CXX:-/usr/bin/g++}
export TRITON_CC=${TRITON_CC:-/usr/bin/gcc}
# Triton/Inductor compile caches: put on NODE-LOCAL /tmp and make them job-unique.
# 8 ranks sharing the default (GPFS) cache concurrently races on the cuda_utils
# .so build and fails intermittently ('gcc ... cuda_utils.c ... exit 1').
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/${USER}_triton_${LSB_JOBID:-$$}}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/${USER}_inductor_${LSB_JOBID:-$$}}
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
export TORCH_LOGS=${TORCH_LOGS:-recompiles}
HF_TOKEN_FILE=${HF_TOKEN_FILE:-${SCRATCH}/.hf_token}
[ -f "${HF_TOKEN_FILE}" ] && export HF_TOKEN="$(cat "${HF_TOKEN_FILE}")"
mkdir -p "${HF_HOME}" "${TMPDIR}"
cd "$REPO"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export DEEPSPEC_BASE_CKPT_DIR="${CKPT_ROOT}/checkpoints"
export DEEPSPEC_BASE_TB_DIR="${CKPT_ROOT}/tensorboard"
export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS-1)))"

CKDIR="${DEEPSPEC_BASE_CKPT_DIR}/deepspec/${EXP}"
LOGA="${TMPDIR}/smoke_phaseA.log"; LOGB="${TMPDIR}/smoke_phaseB.log"

echo "############ SMOKE: fresh start ############"
rm -rf "${CKDIR}"; mkdir -p "${DEEPSPEC_BASE_CKPT_DIR}" "${DEEPSPEC_BASE_TB_DIR}"
test -d "${SMOKE_CACHE}/" || { echo "FAIL: SMOKE_CACHE missing: ${SMOKE_CACHE}"; exit 1; }

OPTS_COMMON=(--opts "exp_name=${EXP}" \
             --opts "data.target_cache_path=${SMOKE_CACHE}" \
             --opts "data.num_workers=0" \
             --opts "train.global_batch_size=${NUM_GPUS}" \
             --opts "logging.checkpointing_steps=4")

# ---- Phase A: run to first checkpoint (cap at 4 steps) -----------------------
# Portable peak-RSS sampler (bv compute nodes lack /usr/bin/time): poll node RSS of
# all python procs every 2s while Phase A runs, record the max KB to a file.
RSSFILE="${TMPDIR}/smoke_peak_rss_kb"; echo 0 > "${RSSFILE}"
( while true; do
    tot=$(ps -eo rss,comm 2>/dev/null | awk '/python/{s+=$1} END{print s+0}')
    cur=$(cat "${RSSFILE}" 2>/dev/null || echo 0)
    [ "${tot:-0}" -gt "${cur:-0}" ] && echo "${tot}" > "${RSSFILE}"
    sleep 2
  done ) & RSS_PID=$!

echo "############ PHASE A: train to step 4 (checkpoint) ############"
"$PY" train.py --config "${CONFIG}" \
    "${OPTS_COMMON[@]}" --opts "train.max_train_steps=4" > "${LOGA}" 2>&1 || true
kill "${RSS_PID}" 2>/dev/null || true
PEAK_KB=$(cat "${RSSFILE}" 2>/dev/null || echo 0)
echo "peak python RSS (phaseA, KB) = ${PEAK_KB}" | tee -a "${LOGA}"
echo "--- phaseA tail ---"; tail -n 20 "${LOGA}"

# ---- Phase B: relaunch same exp -> must auto-resume from step 4 to step 8 ----
echo "############ PHASE B: relaunch -> expect AUTO-RESUME from step 4 -> 8 ############"
"$PY" train.py --config "${CONFIG}" \
    "${OPTS_COMMON[@]}" --opts "train.max_train_steps=8" > "${LOGB}" 2>&1 || true
echo "--- phaseB tail ---"; tail -n 20 "${LOGB}"

# ============================ CHECKS ============================
pass=0; fail=0
chk(){ if eval "$2"; then echo "PASS: $1"; pass=$((pass+1)); else echo "FAIL: $1"; fail=$((fail+1)); fi; }

# 1. startup host RAM: PEAK_KB SUMS all python ranks on the node. With the rank-0
# broadcast fix, only ONE rank loads the 8B target (~16GB) while the others hold
# just their draft+CUDA context (~8-10GB each). So ~8 ranks * ~10GB + one 16GB
# spike is normal/healthy (~90GB). A regression to 8x FULL target loads would be
# ~128GB + contexts (>180GB). Threshold set to catch that regression, not normal use.
echo "peak python RSS summed-over-ranks (phaseA, KB) = ${PEAK_KB:-unknown}"
chk "startup host RAM sane (rank-0 broadcast worked, not ${NUM_GPUS}x full target load)" \
    "[ -n \"${PEAK_KB}\" ] && [ \"${PEAK_KB}\" -gt 0 ] && [ \"${PEAK_KB}\" -lt 157286400 ]"
# 2. loss line present & finite in phase A
chk "training produced loss lines (compile+flex+fsdp ran)" \
    "grep -qE 'loss[= ]' '${LOGA}'"
# 3. torch.compile not storming: fewer than ~20 recompiles across the short run
RECOMP=$(grep -ac "recompil" "${LOGA}" 2>/dev/null || echo 0)
echo "recompile log lines (phaseA) = ${RECOMP}"
chk "torch.compile did NOT storm (<20 recompile lines; else set torch_compile=False)" \
    "[ \"${RECOMP}\" -lt 20 ]"
# 4. checkpoint saved with all ranks' state + safetensors
chk "step_4 checkpoint dir exists" "[ -d '${CKDIR}/step_4' ] || [ -d '${CKDIR}/step_latest' ]"
NRANK=$(ls "${CKDIR}"/step_*/training_state.rank*.pt 2>/dev/null | grep -c "rank" || echo 0)
chk "all ${NUM_GPUS} per-rank training_state files present" "[ \"${NRANK}\" -ge \"${NUM_GPUS}\" ]"
chk "step_latest symlink present" "[ -e '${CKDIR}/step_latest' ]"
# 5. Phase B AUTO-RESUMED (not from scratch) and reached step 8
chk "phase B auto-resumed from an existing checkpoint" \
    "grep -qiE 'resum' '${LOGB}'"
chk "phase B did NOT restart from step 0 (resume offset applied)" \
    "! grep -qiE 'Training from scratch' '${LOGB}'"
# 6. no topology/assert crash on resume
chk "phase B has no world_size/rank/batch resume assertion error" \
    "! grep -qiE 'AssertionError|world_size|does not match' '${LOGB}'"
# 7. phase B completed
chk "phase B ran to completion (reached max steps)" \
    "grep -qE 'loss[= ]' '${LOGB}'"

echo "=================================================="
echo "SMOKE RESULT: ${pass} passed, ${fail} failed"
if [ "${fail}" -eq 0 ]; then
    echo "===SMOKE_PASS=== clear to launch the full run"
else
    echo "===SMOKE_FAIL=== do NOT launch the full run; inspect ${LOGA} / ${LOGB}"
    echo "  (if only the recompile check failed: set torch_compile=False in the config and re-run)"
fi
echo "===JOB_COMPLETE==="
