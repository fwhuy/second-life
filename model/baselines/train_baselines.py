"""Train baseline classifiers for Second Life AI.

Examples:
  python model/baselines/train_baselines.py --config model/baselines/configs/resnet18.yaml
  python model/baselines/train_baselines.py --config model/baselines/configs/naive_mlp.yaml --no-wandb
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms as T


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "model" / "baselines" / "runs"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_workers(requested) -> int:
    if requested != "auto":
        return int(requested)
    return max(0, min(16, (os.cpu_count() or 4) - 1))


class SplitDataset(Dataset):
    def __init__(self, rows: list[dict], data_root: Path, classes: list[str], transform):
        self.paths = [data_root / row["path"] for row in rows]
        class_to_idx = {name: i for i, name in enumerate(classes)}
        self.targets = torch.tensor([class_to_idx[row["label"]] for row in rows], dtype=torch.long)
        self.num_classes = len(classes)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), self.targets[index]


class NaiveMLP(nn.Module):
    def __init__(self, img_size: int, num_classes: int, hidden_dim: int, dropout: float):
        super().__init__()
        in_dim = 3 * img_size * img_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_transform(img_size: int):
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.65, 1.0)),
        T.RandomHorizontalFlip(),
        T.TrivialAugmentWide(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.15),
    ])


def eval_transform(img_size: int):
    return T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_frames(cfg: dict) -> tuple[list[dict], list[dict]]:
    splits = REPO_ROOT / cfg["splits_csv"]
    with open(splits, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    val_fold = int(cfg.get("val_fold", 0))
    train = [row for row in rows if int(row["fold"]) != val_fold]
    val = [row for row in rows if int(row["fold"]) == val_fold]
    if {row["group"] for row in train}.intersection({row["group"] for row in val}):
        raise RuntimeError(f"group leakage detected for val_fold={val_fold}")
    return train, val


def make_loaders(cfg: dict):
    classes = cfg["classes"]
    data_root = REPO_ROOT / cfg["data_root"]
    train_rows, val_rows = load_frames(cfg)
    missing = [
        str(data_root / row["path"])
        for row in train_rows[:100]
        if not (data_root / row["path"]).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Training images are missing. First missing example: " + missing[0]
        )

    train_ds = SplitDataset(train_rows, data_root, classes, train_transform(cfg["img_size"]))
    val_ds = SplitDataset(val_rows, data_root, classes, eval_transform(cfg["img_size"]))

    counts = np.array([sum(row["label"] == klass for row in train_rows) for klass in classes], dtype=np.float64)
    weights = counts.sum() / (len(classes) * counts)
    sample_weights = torch.tensor([weights[int(target)] for target in train_ds.targets], dtype=torch.double)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    workers = resolve_workers(cfg.get("num_workers", "auto"))
    common = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        common["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        sampler=sampler,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        **common,
    )
    return train_loader, val_loader


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    scores = []
    for class_idx in range(num_classes):
        tp = np.logical_and(y_true == class_idx, y_pred == class_idx).sum()
        fp = np.logical_and(y_true != class_idx, y_pred == class_idx).sum()
        fn = np.logical_and(y_true == class_idx, y_pred != class_idx).sum()
        denom = (2 * tp) + fp + fn
        scores.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(np.mean(scores))


def build_model(cfg: dict) -> nn.Module:
    num_classes = len(cfg["classes"])
    pretrained = bool(cfg.get("pretrained", True))
    name = cfg["model"]
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "swin_b":
        weights = models.Swin_B_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.swin_b(weights=weights)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model
    if name == "naive_mlp":
        return NaiveMLP(
            img_size=cfg["img_size"],
            num_classes=num_classes,
            hidden_dim=int(cfg.get("hidden_dim", 512)),
            dropout=float(cfg.get("dropout", 0.3)),
        )
    raise ValueError(f"unknown model baseline: {name}")


def configure_amp(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return None, None
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    props = torch.cuda.get_device_properties(0)
    dtype = torch.bfloat16 if props.major >= 8 and torch.cuda.is_bf16_supported() else torch.float16
    scaler = None if dtype is torch.bfloat16 else torch.amp.GradScaler("cuda")
    print(f"gpu={props.name} vram={props.total_memory / 1024**3:.0f}GB amp={dtype}")
    return dtype, scaler


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device, amp_dtype, scaler):
    model.train()
    total_loss = 0.0
    total_seen = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(images)
            loss = criterion(logits, targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * targets.numel()
        total_seen += targets.numel()
    return total_loss / max(total_seen, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_dtype):
    model.eval()
    total_loss = 0.0
    total_seen = 0
    preds = []
    targets_all = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(images)
            loss = criterion(logits, targets)
        total_loss += loss.item() * targets.numel()
        total_seen += targets.numel()
        preds.append(logits.argmax(1).cpu().numpy())
        targets_all.append(targets.cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets_all)
    return {
        "loss": total_loss / max(total_seen, 1),
        "acc": float((y_pred == y_true).mean()),
        "macro_f1": macro_f1_score(y_true, y_pred, loader.dataset.num_classes),
    }


def init_wandb(cfg: dict, disabled: bool):
    if disabled:
        return None
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY is not set; continuing without W&B logging.")
        return None
    import wandb

    return wandb.init(
        project=cfg.get("wandb_project", "second-life-baselines"),
        group=cfg.get("wandb_group"),
        name=cfg["run_name"],
        config={k: v for k, v in cfg.items() if not str(k).startswith("_")},
    )


def save_history(history: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="train")
    ax.plot([row["epoch"] for row in history], [row["val_loss"] for row in history], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(out_dir.name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=160)
    plt.close(fig)


def train(cfg: dict, *, no_wandb: bool = False) -> dict:
    seed_everything(int(cfg.get("seed", 42)))
    device = get_device()
    train_loader, val_loader = make_loaders(cfg)
    amp_dtype, scaler = configure_amp(device, bool(cfg.get("amp", True)))

    model = build_model(cfg).to(device)
    if device.type == "cuda" and cfg["model"] != "naive_mlp":
        model = model.to(memory_format=torch.channels_last)
    params = sum(p.numel() for p in model.parameters())
    print(f"run={cfg['run_name']} model={cfg['model']} params={params / 1e6:.2f}M device={device}")

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.0)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg["epochs"]),
        eta_min=float(cfg.get("min_lr", 1.0e-6)),
    )

    run = init_wandb(cfg, no_wandb)
    out_dir = OUT_ROOT / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    best_acc = -math.inf
    history = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, amp_dtype, scaler)
        val = evaluate(model, val_loader, criterion, device, amp_dtype)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_macro_f1": val["macro_f1"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{cfg['epochs']} train_loss={train_loss:.4f} "
            f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f} "
            f"macro_f1={val['macro_f1']:.4f}"
        )
        if run is not None:
            run.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val["loss"],
                "val/acc": val["acc"],
                "val/macro_f1": val["macro_f1"],
                "lr": row["lr"],
            }, step=epoch)
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                    "metrics": val,
                    "classes": cfg["classes"],
                },
                out_dir / "best.pt",
            )
        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, out_dir / "last.pt")
        save_history(history, out_dir)

    summary = {"best_val_acc": best_acc, "epochs": int(cfg["epochs"]), "params": params}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if run is not None:
        run.summary.update(summary)
        run.finish()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, help="override config epochs, useful for smoke tests")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    train(cfg, no_wandb=args.no_wandb)


if __name__ == "__main__":
    main()
