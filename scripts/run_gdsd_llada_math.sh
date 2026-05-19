#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${1:-${NUM_GPUS:-8}}"
WORLD_SIZE="${WORLD_SIZE:-${NUM_GPUS}}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"

DATASET="math_xml"
NUM_ITER=8
TRAIN_BS=8
GEN_BS=8
SAMPLE_STEPS=2
GRAD_ACCUM=4
MAX_COMPLETION_LENGTH=256
LEARNING_RATE=1e-5
BETA_KL_DIV=1e-4
PSI=10.0
NUM_GEN=$((WORLD_SIZE * 2))

RUN_NAME="${DATASET}_gdsd_llada_mu${NUM_ITER}_cl${MAX_COMPLETION_LENGTH}_lr${LEARNING_RATE}_kl${BETA_KL_DIV}_psi${PSI}_mc${SAMPLE_STEPS}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}/checkpoints"
mkdir -p "${OUTPUT_DIR}"

accelerate launch \
  --config_file recipes/accelerate_configs/zero2.yaml \
  --num_processes "${NUM_GPUS}" \
  gdsd/gdsd_train.py \
  --config recipes/train_gdsd.yaml \
  --dataset_name "${DATASET}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --run_name "${RUN_NAME}" \
  --num_iterations "${NUM_ITER}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --per_device_train_batch_size "${TRAIN_BS}" \
  --generation_batch_size "${GEN_BS}" \
  --num_generations "${NUM_GEN}" \
  --num_mc "${SAMPLE_STEPS}" \
  --rl_loss_type gdsd \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --max_prompt_length 400 \
  --beta "${BETA_KL_DIV}" \
  --psi "${PSI}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay 0.01 \
  --lr_scheduler_type constant_with_warmup \
  --output_dir "${OUTPUT_DIR}" \
  --save_total_limit 200 \
  --save_steps 100 \
  --diffusion_steps 256 \
  --report_to none
