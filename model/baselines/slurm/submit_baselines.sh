#!/usr/bin/env bash

set -euo pipefail

GPU_TYPE="${1:-a100}"
case "${GPU_TYPE}" in
  l40s|a100|h100) ;;
  *)
    echo "usage: $0 [l40s|a100|h100]" >&2
    exit 2
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

mkdir -p model/baselines/logs

sbatch \
  --job-name=second-life-baselines \
  --account=torch_pr_63_tandon_advanced \
  --partition=gpu \
  --gres="gpu:${GPU_TYPE}:1" \
  --cpus-per-task=16 \
  --mem=96G \
  --time=24:00:00 \
  --array=0-2 \
  --output=model/baselines/logs/%x_%A_%a.out \
  --error=model/baselines/logs/%x_%A_%a.err \
  --export=ALL \
  model/baselines/slurm/train_one.sbatch
