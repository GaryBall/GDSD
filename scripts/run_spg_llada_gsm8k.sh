#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${1:-${NUM_GPUS:-8}}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"

DATASET="gsm8k_xml"
NUM_ITER=4
NUM_GENERATIONS=6
TRAIN_BS=10
GRAD_ACCUM=2
MAX_COMPLETION_LENGTH=256
MAX_PROMPT_LENGTH=200
LEARNING_RATE=3e-6
WEIGHT_DECAY=0.1
FORWARD_TYPE="block_random"
NUM_T=2
LOGP_ESTIMATION="mix"
MIX_WEIGHT=0.5
EUBO_BETA=1.5

RUN_NAME="gsm8k_spg_mix_beta${EUBO_BETA}_weight${MIX_WEIGHT}_iter${NUM_ITER}"
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
  --output_dir "${OUTPUT_DIR}" \
  --rl_loss_type spg \
  --num_iterations "${NUM_ITER}" \
  --num_generations "${NUM_GENERATIONS}" \
  --per_device_train_batch_size "${TRAIN_BS}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --max_grad_norm 0.2 \
  --warmup_ratio 0.0001 \
  --lr_scheduler_type constant_with_warmup \
  --beta 0.0 \
  --forward_type "${FORWARD_TYPE}" \
  --num_t "${NUM_T}" \
  --min_t 0 \
  --max_t 1 \
  --logp_estimation "${LOGP_ESTIMATION}" \
  --mix_weight "${MIX_WEIGHT}" \
  --eubo_beta "${EUBO_BETA}" \
  --sync_ref_model true \
  --ref_model_sync_steps 64 \
  --save_steps 100 \
  --save_total_limit 500 \
  --logging_steps 1 \
  --num_train_epochs 10 \
  --diffusion_steps 256 \
  --report_to none
