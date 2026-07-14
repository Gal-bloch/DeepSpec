#!/bin/bash
# CCC batch job: Granite DSpark data pipeline.
#   Step 1: download + subset open-perfectblend.
#   Step 2: (optional) regenerate answers with Granite-8B via SGLang.
#   Step 3: build the target cache with the real Granite model.
#
# SKIP_REGEN=1 (default) trains on the dataset's original answers and skips the
# SGLang step entirely — right for a proof-of-concept, and avoids the sglang
# install (outlines_core needs a Rust toolchain to build on CCC). Set
# SKIP_REGEN=0 only if sglang is installed and you want Granite-style answers.
set -euo pipefail

REPO=${REPO:-/dccstor/knewedge/galbloch/DeepSpec}
ENVDIR=${ENVDIR:-/dccstor/knewedge/galbloch/envs/granite}
PY="${ENVDIR}/bin/python"
CACHE=${CACHE:-/dccstor/knewedge/galbloch/granite_cache/granite_4_1_8b_target_cache}
CONFIG=config/dspark/dspark_granite_4_1_8b.py
MODEL=ibm-granite/granite-4.1-8b

# PoC sized for ~100 GB cache on the near-full /dccstor/knewedge fileset:
# ~2k train samples after the 5% eval split. Keeps all 5 target layers.
SAMPLE_SIZE=${SAMPLE_SIZE:-2200}
MIN_FREE_MB=${MIN_FREE_MB:-40000}
SKIP_REGEN=${SKIP_REGEN:-1}

TRAIN_SPLIT=train_datasets/perfectblend_train.jsonl
REGEN=train_datasets/granite_4_1_8b/perfectblend_train_regen.jsonl
SGLANG_PORT=30000
SGLANG_NCCL_PORT=31000

# Keep HF downloads on GPFS, not home.
export HF_HOME=${HF_HOME:-/dccstor/knewedge/galbloch/.cache/hf}
export TMPDIR=${TMPDIR:-/dccstor/knewedge/galbloch/tmp}
mkdir -p "${HF_HOME}" "${TMPDIR}"
cd "$REPO"

echo "=== Step 1/3: stream-subset open-perfectblend (max-rows=${SAMPLE_SIZE}) ==="
# Stream only the rows we need instead of downloading the full multi-GB dataset
# (upstream download_and_split.py pulls everything, then .select()s — too slow
# and wasteful for a ~2k-row PoC).
if [ -s "${TRAIN_SPLIT}" ]; then
    echo "train split already present: ${TRAIN_SPLIT} ($(wc -l < "${TRAIN_SPLIT}") rows) — skipping"
else
    "$PY" scripts/ccc/prep_subset.py \
        --dataset-name mlabonne/open-perfectblend \
        --max-rows "${SAMPLE_SIZE}" \
        --eval-frac 0.05 \
        --train-out "${TRAIN_SPLIT}" \
        --eval-out eval_datasets/perfectblend.jsonl
fi

if [ "${SKIP_REGEN}" = "1" ]; then
    echo "=== Step 2/3: SKIPPED (SKIP_REGEN=1) — caching original dataset answers ==="
    CACHE_INPUT="${TRAIN_SPLIT}"
else
    echo "=== Step 2/3: serve Granite-8B (SGLang) + regenerate answers ==="
    mkdir -p "$(dirname "${REGEN}")" logs/sglang_granite_4_1_8b
    CUDA_VISIBLE_DEVICES=0 "${ENVDIR}/bin/sglang" serve \
        --model-path "${MODEL}" \
        --host 127.0.0.1 --port "${SGLANG_PORT}" --nccl-port "${SGLANG_NCCL_PORT}" \
        --dtype bfloat16 --mem-fraction-static 0.9 \
        > logs/sglang_granite_4_1_8b/worker.log 2>&1 &
    SGLANG_PID=$!
    trap 'kill "${SGLANG_PID}" 2>/dev/null || true' EXIT
    echo "Waiting for SGLang to become ready..."
    for i in $(seq 1 120); do
        if curl -sf "http://127.0.0.1:${SGLANG_PORT}/health" >/dev/null 2>&1; then
            echo "SGLang ready after ${i}0s"; break
        fi
        sleep 10
    done
    "$PY" scripts/data/generate_train_data.py \
        --model "${MODEL}" \
        --server-address "127.0.0.1:${SGLANG_PORT}" \
        --concurrency 32 \
        --temperature 0.0 --top-p 1.0 --top-k -1 --min-p 0 \
        --max-tokens 2048 --resume \
        --input-file-path "${TRAIN_SPLIT}" \
        --output-file-path "${REGEN}"
    kill "${SGLANG_PID}" 2>/dev/null || true
    trap - EXIT
    sleep 15
    CACHE_INPUT="${REGEN}"
fi

echo "=== Step 3/3: build target cache from ${CACHE_INPUT} -> ${CACHE} ==="
mkdir -p "${CACHE}"
free_mb=$(df -Pm "${CACHE}" | awk 'NR==2{print $4}')
echo "Free space on target fileset: ${free_mb} MB (guard: ${MIN_FREE_MB} MB)"
if [ "${free_mb}" -lt "${MIN_FREE_MB}" ]; then
    echo "ERROR: not enough free space to safely build the cache." >&2
    exit 1
fi
CUDA_VISIBLE_DEVICES=0,1 "$PY" scripts/data/prepare_target_cache.py \
    --config "${CONFIG}" \
    --train-data-path "${CACHE_INPUT}" \
    --output-dir "${CACHE}" \
    --local-batch-size 8

echo "Cache size:"; du -sh "${CACHE}" || true
echo "Free space after build:"; df -h "${CACHE}" | tail -1
echo "===JOB_COMPLETE==="
