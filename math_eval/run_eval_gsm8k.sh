#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
NUM_GPUS="${NUM_GPUS:-1}"
GEN_LENGTHS="${GEN_LENGTHS:-128 256 512}"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval_results/gsm8k}"
GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))

mkdir -p "${OUTPUT_ROOT}"

for gen_length in ${GEN_LENGTHS}; do
  if [[ "${gen_length}" -eq 512 ]]; then
    batch_size=8
  else
    batch_size=16
  fi

  master_port=$((12000 + RANDOM % 20000))
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" python -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${master_port}" \
    eval.py \
    --dataset gsm8k \
    --batch_size "${batch_size}" \
    --gen_length "${gen_length}" \
    --diffusion_steps 128 \
    --output_dir "${OUTPUT_ROOT}/gl${gen_length}" \
    --model_path "${MODEL_PATH}"
done

python eval/parse_and_get_acc.py --directory "${OUTPUT_ROOT}"
