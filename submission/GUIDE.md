# Current-model reproduction guide

Run commands from `submission/code` unless stated otherwise.

## 1. Environment

Python 3.10+ and an NVIDIA GPU are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Verify the recorded artifacts

From the repository root:

```bash
python submission/code/verify_artifacts.py
```

This validates the metrics schemas, expected model identities, class orders,
parameter counts, checkpoint locations, and SHA-256 hashes without loading the
models into GPU memory.

## 3. Train ConvNeXt V2-Tiny

```bash
python train_convnextv2.py --no-upload
```

The trainer downloads the OMA waste-management dataset and official UCI
RealWaste archive, maps them to six shared classes, removes perceptual
duplicates, creates a group-disjoint fold, and trains the 384px
`convnextv2_tiny.fcmae_ft_in22k_in1k_384` model.

Outputs:

- `best_convnextv2.pt`
- `best_convnextv2_metadata.json`

The recorded run reached 98.30% validation accuracy at epoch 37. It has no
separate held-out test result; do not describe the validation score as test
accuracy.

## 4. Train Swin-B

Place the ten-class dataset at:

```text
submission/code/standardized_256/
  battery/
  biological/
  cardboard/
  clothes/
  glass/
  metal/
  paper/
  plastic/
  shoes/
  trash/
```

Then run:

```bash
python train_swin_b.py
```

Outputs:

- `best_swin_b.pt`
- `swin_b_results.json`

The recorded run reached 97.01% validation, 97.61% held-out test, and 97.82%
test accuracy with TTA. Swin-B exceeds the course's 30M-parameter cap and is
included explicitly as the transformer comparison model.

## 5. Run the current website

From the repository root:

```bash
cd website
./start.sh
```

The website loads the canonical checkpoints directly from:

```text
model/convnextv2_tiny_cnn/results/best_convnextv2.pt
model/Swin B Transformer/best_swin_b.pt
```

The public interface always displays the six shared waste labels. For Swin-B,
the four non-shared logits are omitted and the six retained logits are
renormalized.
