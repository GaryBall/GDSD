#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${1:-1}"
DATASET="countdown"
RUN_NAME="${DATASET}_gdsd_tlc_demo"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Dream-org/Dream-v0-Instruct-7B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"

mkdir -p "${OUTPUT_ROOT}/${RUN_NAME}"

accelerate launch \
  --config_file recipes/accelerate_configs/zero2.yaml \
  --num_processes "${NUM_GPUS}" \
  gdsd/gdsd_train.py \
  --config recipes/train_gdsd.yaml \
  --dataset_name "${DATASET}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --run_name "${RUN_NAME}" \
  --rl_loss_type gdsd_tlc \
  --output_dir "${OUTPUT_ROOT}/${RUN_NAME}/checkpoints"
