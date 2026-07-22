"""5-fold CV orchestration → final ensemble + honest error bars (M2.5).

Trains one model per fold (each fold is validation exactly once), reports
mean ± std of fold val accuracies, and writes an ensemble manifest that
evaluate.py / demo can load. Each fold resumes from its own checkpoint, so a
crashed overnight run continues with the same command.

Usage: python -m src.kfold --config configs/<winner>.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .data import N_FOLDS
from .train import train_run
from .utils import REPO_ROOT, load_config, log_experiment


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    results = []
    for fold in range(N_FOLDS):
        print(f"\n===== fold {fold}/{N_FOLDS - 1} =====")
        results.append(train_run(cfg, val_fold=fold, resume=True))

    accs = np.array([r["best_val_acc"] for r in results])
    macro_f1s = np.array([r["macro_f1"] for r in results])
    recall_labels = sorted(results[0]["per_class_recall"])
    mean_recalls = {
        label: round(float(np.mean([r["per_class_recall"][label] for r in results])), 4)
        for label in recall_labels
    }
    rel_ckpts = [Path(r["checkpoint"]).resolve().relative_to(REPO_ROOT).as_posix()
                 for r in results]

    manifest_path = REPO_ROOT / "checkpoints" / cfg["run_name"] / "ensemble.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_name": cfg["run_name"],
        "checkpoints": rel_ckpts,
        "fold_val_accs": [round(float(a), 4) for a in accs],
        "fold_macro_f1s": [round(float(score), 4) for score in macro_f1s],
        "mean_val_acc": round(float(accs.mean()), 4),
        "std_val_acc": round(float(accs.std(ddof=1)), 4),
        "mean_macro_f1": round(float(macro_f1s.mean()), 4),
        "std_macro_f1": round(float(macro_f1s.std(ddof=1)), 4),
        "mean_per_class_recall": mean_recalls,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n5-fold macro-F1: {macro_f1s.mean():.4f} ± {macro_f1s.std(ddof=1):.4f}  "
          f"(folds: {[f'{score:.4f}' for score in macro_f1s]})")
    print(f"5-fold val accuracy: {accs.mean():.4f} ± {accs.std(ddof=1):.4f}")
    print(f"mean per-class recall: {mean_recalls}")
    print(f"ensemble manifest → {manifest_path}")
    print(f"evaluate with: python -m src.evaluate --checkpoint "
          f"{manifest_path.relative_to(REPO_ROOT)} --tta")

    log_experiment(cfg, stage="kfold",
                   metrics={"best_val_acc": float(accs.mean()),
                            "macro_f1": float(macro_f1s.mean()),
                            "per_class_recall": mean_recalls, "epochs_ran": ""},
                   notes=(f"macro-F1 mean±std = {macro_f1s.mean():.4f}±"
                          f"{macro_f1s.std(ddof=1):.4f}; accuracy mean±std = "
                          f"{accs.mean():.4f}±{accs.std(ddof=1):.4f}"))


if __name__ == "__main__":
    main()
