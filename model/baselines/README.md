# Baseline Training

This directory contains the three requested 40-epoch baselines:

- `resnet18` - ImageNet-pretrained ResNet-18.
- `naive_mlp` - a simple flatten-then-MLP image classifier.
- `swin_b` - ImageNet-pretrained Swin-B Transformer.

All configs use batch size 64 and log `train/loss`, `val/loss`, `val/acc`, and
`val/macro_f1` to Weights & Biases when `WANDB_API_KEY` is available.

## Local Smoke Test

```bash
python model/baselines/train_baselines.py \
  --config model/baselines/configs/resnet18.yaml \
  --epochs 1 \
  --no-wandb
```

## Slurm Submission

Set the W&B key in the shell before submitting. Do not commit it.

```bash
export WANDB_API_KEY=...
model/baselines/slurm/submit_baselines.sh a100
```

The first positional argument is the GPU type. Use `l40s`, `a100`, or `h100`.
Jobs are submitted with account `torch_pr_63_tandon_advanced`.
