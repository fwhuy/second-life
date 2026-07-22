# ResNet-50 hyperparameter tuning

This package compares a small, controlled set of ResNet-50 fine-tuning recipes
on validation fold 0. It never evaluates the quarantined test set.

## Experiments

Each experiment changes one setting from the locked baseline:

| Experiment | Change |
|---|---|
| `baseline_reference` | Original recipe |
| `backbone_lr_low` | Backbone LR: `1e-5` to `3e-6` |
| `backbone_lr_high` | Backbone LR: `1e-5` to `3e-5` |
| `freeze_2_epochs` | Frozen-head phase: 5 to 2 epochs |
| `label_smoothing_005` | Label smoothing: 0.10 to 0.05 |
| `ema_decay_0999` | EMA decay: 0.9998 to 0.999 |

## Run

From the `model` directory, activate its training environment and list the
experiments:

```bash
source .venv/bin/activate
python hyperparameter_tuning/run_tuning.py --list
```

Run one experiment first:

```bash
python hyperparameter_tuning/run_tuning.py --run backbone_lr_low
```

Or run the complete sequence:

```bash
python hyperparameter_tuning/run_tuning.py --run all
```

Interrupted runs resume from their `last.pth` checkpoint. Results are written
to `hyperparameter_tuning/results.csv`.

Each invocation also produces:

- `EXPERIMENT_LOG.md`: append-only settings, status, metrics, and notes.
- `logs/<experiment>.log`: complete training console output for diagnosis.
- `../experiments.csv`: the main pipeline's existing append-only audit trail.

Add a human note to the log with:

```bash
python hyperparameter_tuning/run_tuning.py --run backbone_lr_low \
  --note "Testing whether a more conservative backbone update reduces overfitting"
```

## Selection rule

The locked baseline validation result is 404/434 (93.09%) single-view and
412/434 (94.93%) with four-view TTA. Treat small changes as noise. Only accept
a recipe if it gains at least 7 correct validation images (about 1.5%), does
not reduce macro-F1, and does not seriously harm minority-class recall.

Do not repeatedly evaluate candidates on the test set. That would tune the
project to its test data and make the reported test accuracy unreliable.
