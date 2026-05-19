#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
NUM_GPUS="${NUM_GPUS:-1}"
GEN_LENGTHS="${GEN_LENGTHS:-128 256}"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval_results/sudoku}"
GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))

mkdir -p "${OUTPUT_ROOT}"

for gen_length in ${GEN_LENGTHS}; do
  batch_size=16
  master_port=$((12000 + RANDOM % 20000))

  CUDA_VISIBLE_DEVICES="${GPU_LIST}" python -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${master_port}" \
    eval.py \
    --dataset sudoku \
    --batch_size "${batch_size}" \
    --gen_length "${gen_length}" \
    --output_dir "${OUTPUT_ROOT}/gl${gen_length}" \
    --model_path "${MODEL_PATH}"
done

python eval/parse_and_get_acc.py --directory "${OUTPUT_ROOT}"
