#!/bin/bash
# CCC batch job: Granite DSpark data pipeline (full-data capable).
#   Step 1: get open-perfectblend prompts (FULL dataset or a streamed subset).
#   Step 2: regenerate assistant answers with the Granite-8B TARGET (DSpark-faithful),
#           via a vLLM (preferred) or SGLang OpenAI-compatible server. Optional.
#   Step 3: build the target hidden-state cache with the real Granite model.
#
# KEY SWITCHES (env):
#   FULL_DATA=1  (default) -> full open-perfectblend via download_and_split.py.
#                0         -> streamed subset of SAMPLE_SIZE rows (PoC path).
#   REGEN=1      (default) -> DSpark-faithful: regenerate answers with the target.
#                0         -> train on the dataset's ORIGINAL answers (fallback if no
#                             inference server can be built; not fully DSpark-faithful).
#   SERVE_ENGINE=vllm (default) | sglang  -> which OpenAI-compatible server to use.
#   NUM_GPUS (default 8) -> GPUs for serving + the cache build.
# The DSpark paper regenerates answers with the target, so REGEN=1 is the faithful
# path. SGLang failed to build on CCC (outlines_core/Rust); vLLM usually builds, so it
# is preferred. If neither works, set REGEN=0 to unblock (documented tradeoff).
set -uo pipefail

SCRATCH=${SCRATCH:-/dccstor/knewedge/galbloch}
REPO=${REPO:-${SCRATCH}/DeepSpec}
ENVDIR=${ENVDIR:-${SCRATCH}/envs/granite}
PY="${ENVDIR}/bin/python"
CACHE=${CACHE:-${SCRATCH}/granite_cache/granite_4_1_8b_target_cache}
CONFIG=${CONFIG:-config/dspark/dspark_granite_4_1_8b.py}
MODEL=${MODEL:-ibm-granite/granite-4.1-8b}
NUM_GPUS=${NUM_GPUS:-8}

FULL_DATA=${FULL_DATA:-1}
REGEN=${REGEN:-1}
SERVE_ENGINE=${SERVE_ENGINE:-vllm}
SAMPLE_SIZE=${SAMPLE_SIZE:-2200}          # only used when FULL_DATA=0
# Full-data cache is tens of TB. Guard well above the PoC's 40 GB; override per FS.
MIN_FREE_MB=${MIN_FREE_MB:-5000000}       # ~5 TB floor by default
SERVE_PORT=${SERVE_PORT:-30000}

TRAIN_SPLIT=${TRAIN_SPLIT:-train_datasets/perfectblend_train.jsonl}
REGEN_OUT=${REGEN_OUT:-train_datasets/granite_4_1_8b/perfectblend_train_regen.jsonl}

export HF_HOME=${HF_HOME:-${SCRATCH}/.cache/hf}
export TMPDIR=${TMPDIR:-${SCRATCH}/tmp}
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}" "${TMPDIR}"
HF_TOKEN_FILE=${HF_TOKEN_FILE:-${SCRATCH}/.hf_token}
[ -f "${HF_TOKEN_FILE}" ] && export HF_TOKEN="$(cat "${HF_TOKEN_FILE}")"
cd "$REPO"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

# ---- Step 1: prompts ---------------------------------------------------------
if [ -s "${TRAIN_SPLIT}" ]; then
    echo "=== Step 1/3: train split present (${TRAIN_SPLIT}, $(wc -l < "${TRAIN_SPLIT}") rows) — skipping ==="
elif [ "${FULL_DATA}" = "1" ]; then
    echo "=== Step 1/3: download FULL open-perfectblend ==="
    "$PY" scripts/data/download_and_split.py \
        --dataset-name mlabonne/open-perfectblend \
        --test-size 0.05 \
        --train-output-path "${TRAIN_SPLIT}" \
        --test-output-dir eval_datasets \
        --skip-existing || { echo "ERROR: download_and_split failed" >&2; exit 1; }
else
    echo "=== Step 1/3: stream-subset open-perfectblend (max-rows=${SAMPLE_SIZE}) ==="
    "$PY" scripts/ccc/prep_subset.py \
        --dataset-name mlabonne/open-perfectblend \
        --max-rows "${SAMPLE_SIZE}" --eval-frac 0.05 \
        --train-out "${TRAIN_SPLIT}" \
        --eval-out eval_datasets/perfectblend.jsonl || { echo "ERROR: prep_subset failed" >&2; exit 1; }
fi

# ---- Step 2: regenerate answers with the target (DSpark-faithful) ------------
CACHE_INPUT="${TRAIN_SPLIT}"
if [ "${REGEN}" = "1" ]; then
    echo "=== Step 2/3: serve ${MODEL} (${SERVE_ENGINE}) + regenerate answers ==="
    mkdir -p "$(dirname "${REGEN_OUT}")" logs/serve_granite
    SERVE_CVD="$(seq -s, 0 $((NUM_GPUS-1)))"
    if [ "${SERVE_ENGINE}" = "vllm" ]; then
        # vLLM OpenAI-compatible server; tensor-parallel across the visible GPUs.
        CUDA_VISIBLE_DEVICES="${SERVE_CVD}" "${ENVDIR}/bin/python" -m vllm.entrypoints.openai.api_server \
            --model "${MODEL}" --port "${SERVE_PORT}" \
            --tensor-parallel-size "${NUM_GPUS}" --dtype bfloat16 \
            > logs/serve_granite/vllm.log 2>&1 &
    else
        CUDA_VISIBLE_DEVICES="${SERVE_CVD}" "${ENVDIR}/bin/sglang" serve \
            --model-path "${MODEL}" --host 127.0.0.1 --port "${SERVE_PORT}" \
            --dtype bfloat16 --mem-fraction-static 0.9 \
            > logs/serve_granite/sglang.log 2>&1 &
    fi
    SERVE_PID=$!
    trap 'kill "${SERVE_PID}" 2>/dev/null || true' EXIT
    echo "Waiting for ${SERVE_ENGINE} on :${SERVE_PORT} ..."
    ready=0
    for i in $(seq 1 180); do
        if curl -sf "http://127.0.0.1:${SERVE_PORT}/v1/models" >/dev/null 2>&1 \
           || curl -sf "http://127.0.0.1:${SERVE_PORT}/health" >/dev/null 2>&1; then
            ready=1; echo "server ready after ~${i}0s"; break
        fi
        if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
            echo "ERROR: ${SERVE_ENGINE} server died — see logs/serve_granite/. Set REGEN=0 to train on original answers, or fix the server." >&2
            exit 1
        fi
        sleep 10
    done
    [ "${ready}" = "1" ] || { echo "ERROR: server did not become ready" >&2; exit 1; }

    "$PY" scripts/data/generate_train_data.py \
        --model "${MODEL}" \
        --server-address "127.0.0.1:${SERVE_PORT}" \
        --concurrency 32 \
        --temperature 0.0 --top-p 1.0 --top-k -1 --min-p 0 \
        --max-tokens 4096 --resume \
        --input-file-path "${TRAIN_SPLIT}" \
        --output-file-path "${REGEN_OUT}" || { echo "ERROR: regen failed" >&2; exit 1; }
    kill "${SERVE_PID}" 2>/dev/null || true; trap - EXIT; sleep 15
    CACHE_INPUT="${REGEN_OUT}"
else
    echo "=== Step 2/3: REGEN=0 — using ORIGINAL open-perfectblend answers (not DSpark-faithful) ==="
fi

# ---- Step 3: build the target hidden-state cache -----------------------------
echo "=== Step 3/3: build target cache from ${CACHE_INPUT} -> ${CACHE} ==="
mkdir -p "${CACHE}"
free_mb=$(df -Pm "${CACHE}" | awk 'NR==2{print $4}')
echo "Free space on target fileset: ${free_mb} MB (guard: ${MIN_FREE_MB} MB)"
if [ "${free_mb}" -lt "${MIN_FREE_MB}" ]; then
    echo "ERROR: not enough free space for the full cache (need >= ${MIN_FREE_MB} MB). Full open-perfectblend is tens of TB; point CACHE at a large fileset or lower MIN_FREE_MB deliberately." >&2
    exit 1
fi
# num-workers 0 + TOKENIZERS_PARALLELISM=false: avoid the fork+tokenizer dataloader
# deadlock. Build shards across all visible GPUs (builder shards by rank).
CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS-1)))" "$PY" scripts/data/prepare_target_cache.py \
    --config "${CONFIG}" \
    --train-data-path "${CACHE_INPUT}" \
    --output-dir "${CACHE}" \
    --local-batch-size "${CACHE_LOCAL_BATCH:-8}" \
    --num-workers "${CACHE_NUM_WORKERS:-0}" || { echo "ERROR: cache build failed" >&2; exit 1; }

echo "Cache size:"; du -sh "${CACHE}" || true
echo "Free space after build:"; df -h "${CACHE}" | tail -1
echo "===JOB_COMPLETE==="
