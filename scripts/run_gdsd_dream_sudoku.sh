#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${1:-${NUM_GPUS:-8}}"
WORLD_SIZE="${WORLD_SIZE:-${NUM_GPUS}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"

DATASET="sudoku"
ALGO="gdsd_tlc"          # "gdsd_tlc" (with TLC) or "gdsd" (ablation baseline, no TLC)
MODEL="dream"
NUM_ITER=1
TRAIN_BS=8
GEN_BS=6
SAMPLE_STEPS=2
GRAD_ACCUM=4
MAX_COMPLETION_LENGTH=256
LEARNING_RATE=1e-5
BETA_KL_DIV=1e-3
PSI=1.0
TLC_TOPK=8
REMASKING="random"
NUM_GEN=$((WORLD_SIZE * 2))

# Trust region on the sequence-level log-ratio. GDSD keeps the recipe default, while GDSD-TLC
# needs a wider one: the TLC centering constant enters the log-ratio multiplied by the
# completion length, so it contributes common-mode drift that a 4e-4 nat clamp would treat as
# a real policy change and zero the MSE gradient. Note the clamp is inactive when NUM_ITER=1,
# since old and current log-probs then come from the same forward pass and the ratio is 0.
if [[ "$ALGO" == "gdsd_tlc" ]]; then
  EPSILON=0.2
else
  EPSILON=0.0004
fi

RUN_NAME="${DATASET}_${ALGO}_${MODEL}_mu${NUM_ITER}_cl${MAX_COMPLETION_LENGTH}_lr${LEARNING_RATE}_kl${BETA_KL_DIV}_psi${PSI}_tk${TLC_TOPK}_mc${SAMPLE_STEPS}"

if [[ "$MODEL" == "dream" ]]; then
  MODEL_NAME_OR_PATH="Dream-org/Dream-v0-Instruct-7B"
elif [[ "$MODEL" == "llada" ]]; then
  MODEL_NAME_OR_PATH="GSAI-ML/LLaDA-8B-Instruct"
else
  echo "Unknown MODEL: $MODEL" >&2
  exit 1
fi

OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}/checkpoints"
mkdir -p "${OUTPUT_DIR}"

# Only matters when NUM_ITER > 1: keeps the old-log-prob forward on the same batch shape as
# the compute_loss forward, so the sequence log-ratio is not dominated by numerical noise.
export GDSD_ELBO_ITER_CHUNK_SIZE=1

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
  --rl_loss_type "${ALGO}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --max_prompt_length 400 \
  --beta "${BETA_KL_DIV}" \
  --psi "${PSI}" \
  --tlc_topk "${TLC_TOPK}" \
  --epsilon "${EPSILON}" \
  --remasking "${REMASKING}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay 0.01 \
  --lr_scheduler_type constant_with_warmup \
  --output_dir "${OUTPUT_DIR}" \
  --save_total_limit 200 \
  --save_steps 100 \
  --diffusion_steps 128 \
  --report_to none
