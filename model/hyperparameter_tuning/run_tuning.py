"""Run controlled ResNet-50 tuning experiments on validation fold 0 only."""

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml


MODEL_ROOT = Path(__file__).resolve().parent.parent
TUNING_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = TUNING_ROOT / "configs"
RESULTS_PATH = TUNING_ROOT / "results.csv"
EXPERIMENT_LOG = TUNING_ROOT / "EXPERIMENT_LOG.md"
CONSOLE_LOG_DIR = TUNING_ROOT / "logs"
EXPERIMENTS = [
    "baseline_reference",
    "backbone_lr_low",
    "backbone_lr_high",
    "freeze_2_epochs",
    "label_smoothing_005",
    "ema_decay_0999",
    "trash_aug_boost",
]


def checkpoint_metrics(name: str):
    path = MODEL_ROOT / "checkpoints" / f"tune_{name}" / "fold0" / "best.pth"
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    metrics = state.get("metrics", {})
    return {
        "experiment": name,
        "val_accuracy": metrics.get("acc", ""),
        "macro_f1": metrics.get("macro_f1", ""),
        "epoch": state.get("epoch", ""),
        "checkpoint": str(path.relative_to(MODEL_ROOT)),
    }


def write_summary():
    rows = [row for name in EXPERIMENTS if (row := checkpoint_metrics(name))]
    fields = ["experiment", "val_accuracy", "macro_f1", "epoch", "checkpoint"]
    with RESULTS_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if rows:
        print(f"\n{'experiment':24s} {'val_acc':>9s} {'macro_f1':>9s}")
        for row in sorted(rows, key=lambda x: float(x["val_accuracy"]), reverse=True):
            print(f"{row['experiment']:24s} {float(row['val_accuracy']):9.4f} "
                  f"{float(row['macro_f1']):9.4f}")
    print(f"\nSummary: {RESULTS_PATH}")


def append_experiment_log(name: str, status: str, config: dict, metrics=None, note=""):
    """Append a human-readable audit entry; never rewrite earlier entries."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metrics = metrics or {}
    entry = [
        f"\n## {now} — `{name}` — {status}\n",
        f"- Model: `{config.get('model')}`\n",
        f"- Validation fold: `0` (test set untouched)\n",
        f"- Image size / batch size: `{config.get('img_size')}` / `{config.get('batch_size')}`\n",
        f"- Head LR / backbone LR: `{config.get('lr_head')}` / `{config.get('lr_backbone')}`\n",
        f"- Head epochs / fine-tune epochs: `{config.get('epochs_head')}` / `{config.get('epochs_ft')}`\n",
        f"- Weight decay / label smoothing: `{config.get('weight_decay')}` / `{config.get('label_smoothing')}`\n",
        f"- EMA / decay: `{config.get('ema')}` / `{config.get('ema_decay')}`\n",
    ]
    if metrics:
        entry.extend([
            f"- Best validation accuracy: `{metrics.get('val_accuracy')}`\n",
            f"- Macro-F1: `{metrics.get('macro_f1')}`\n",
            f"- Best epoch: `{metrics.get('epoch')}`\n",
            f"- Checkpoint: `{metrics.get('checkpoint')}`\n",
        ])
    if note:
        entry.append(f"- Note: {note}\n")
    with EXPERIMENT_LOG.open("a") as handle:
        handle.writelines(entry)


def run(name: str, note: str = ""):
    config = CONFIG_DIR / f"{name}.yaml"
    config_values = yaml.safe_load(config.read_text())
    CONSOLE_LOG_DIR.mkdir(exist_ok=True)
    console_log = CONSOLE_LOG_DIR / f"{name}.log"
    print(f"\n===== {name} =====", flush=True)
    append_experiment_log(name, "STARTED", config_values, note=note)
    command = [sys.executable, "-m", "src.train", "--config", str(config),
               "--fold", "0", "--resume"]
    with console_log.open("a") as output:
        output.write(f"\n===== invocation {datetime.now(timezone.utc).isoformat()} =====\n")
        process = subprocess.Popen(
            command, cwd=MODEL_ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            output.write(line)
        return_code = process.wait()
    if return_code:
        append_experiment_log(
            name, f"FAILED (exit {return_code})", config_values,
            note=f"{note} Full output: `{console_log.relative_to(TUNING_ROOT)}`".strip())
        raise subprocess.CalledProcessError(return_code, command)
    metrics = checkpoint_metrics(name)
    append_experiment_log(
        name, "COMPLETED", config_values, metrics,
        note=f"{note} Full output: `{console_log.relative_to(TUNING_ROOT)}`".strip())
    write_summary()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--run", choices=["all", *EXPERIMENTS])
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--note", default="", help="note saved in EXPERIMENT_LOG.md")
    args = parser.parse_args()
    if args.list:
        print("\n".join(EXPERIMENTS))
    if args.summary:
        write_summary()
    if args.run == "all":
        for name in EXPERIMENTS:
            run(name, note=args.note)
    elif args.run:
        run(args.run, note=args.note)
    if not (args.list or args.summary or args.run):
        parser.print_help()


if __name__ == "__main__":
    main()
