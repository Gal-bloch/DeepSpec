#!/bin/bash
# CCC batch job: Granite DSpark data pipeline (download subset -> regen -> cache).
# Runs on a single A100 node. Serves Granite-8B with SGLang locally, regenerates
# a ~10k-sample subset of open-perfectblend, then builds the target cache.
set -euo pipefail

REPO=${REPO:-/dccstor/knewedge/galbloch/DeepSpec}
CACHE=${CACHE:-/dccstor/knewedge/galbloch/granite_cache/granite_4_1_8b_target_cache}
CONFIG=config/dspark/dspark_granite_4_1_8b.py
MODEL=ibm-granite/granite-4.1-8b

# PoC sized for ~100 GB cache on the near-full /dccstor/knewedge fileset:
# ~2k train samples after the 5% eval split. Keeps all 5 target layers.
SAMPLE_SIZE=${SAMPLE_SIZE:-2200}
# Abort the cache build if free space on the target fileset drops below this
# (MB) so we never fill a shared near-full fileset.
MIN_FREE_MB=${MIN_FREE_MB:-40000}
TRAIN_SPLIT=train_datasets/perfectblend_train.jsonl
REGEN=train_datasets/granite_4_1_8b/perfectblend_train_regen.jsonl

# SGLang serving (single GPU for the PoC subset).
SGLANG_PORT=30000
SGLANG_NCCL_PORT=31000

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate granite
cd "$REPO"

echo "=== Step 1/3: download + subset open-perfectblend (sample-size=${SAMPLE_SIZE}) ==="
python scripts/data/download_and_split.py \
    --dataset-name mlabonne/open-perfectblend \
    --sample-size "${SAMPLE_SIZE}" \
    --test-size 0.05 \
    --train-output-path "${TRAIN_SPLIT}" \
    --test-output-dir eval_datasets \
    --skip-existing

mkdir -p "$(dirname "${REGEN}")"

echo "=== Step 2/3: serve Granite-8B (SGLang) + regenerate answers ==="
mkdir -p logs/sglang_granite_4_1_8b
CUDA_VISIBLE_DEVICES=0 sglang serve \
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

# Greedy decoding for clean, deterministic target answers (PoC). Granite has no
# thinking mode to disable.
python scripts/data/generate_train_data.py \
    --model "${MODEL}" \
    --server-address "127.0.0.1:${SGLANG_PORT}" \
    --concurrency 32 \
    --temperature 0.0 \
    --top-p 1.0 --top-k -1 --min-p 0 \
    --max-tokens 2048 \
    --resume \
    --input-file-path "${TRAIN_SPLIT}" \
    --output-file-path "${REGEN}"

echo "Stopping SGLang before cache build (frees the GPU)."
kill "${SGLANG_PID}" 2>/dev/null || true
trap - EXIT
sleep 15

echo "=== Step 3/3: build target cache -> ${CACHE} ==="
mkdir -p "${CACHE}"
free_mb=$(df -Pm "${CACHE}" | awk 'NR==2{print $4}')
echo "Free space on target fileset: ${free_mb} MB (guard: ${MIN_FREE_MB} MB)"
if [ "${free_mb}" -lt "${MIN_FREE_MB}" ]; then
    echo "ERROR: not enough free space to safely build the cache." >&2
    exit 1
fi
CUDA_VISIBLE_DEVICES=0,1 python scripts/data/prepare_target_cache.py \
    --config "${CONFIG}" \
    --train-data-path "${REGEN}" \
    --output-dir "${CACHE}" \
    --local-batch-size 8

echo "Cache size:"; du -sh "${CACHE}" || true
echo "Free space after build:"; df -h "${CACHE}" | tail -1
echo "===JOB_COMPLETE==="
