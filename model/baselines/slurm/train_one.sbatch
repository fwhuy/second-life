#!/usr/bin/env bash
# This file is launched by submit_baselines.sh, which supplies account/GPU args.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

CONFIGS=(
  "model/baselines/configs/resnet18.yaml"
  "model/baselines/configs/naive_mlp.yaml"
  "model/baselines/configs/swin_b.yaml"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID} config=${CONFIG}"
echo "host=$(hostname)"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not set; W&B logging will be disabled."
  WANDB_FLAG="--no-wandb"
else
  WANDB_FLAG=""
fi

python model/baselines/train_baselines.py --config "${CONFIG}" ${WANDB_FLAG}
