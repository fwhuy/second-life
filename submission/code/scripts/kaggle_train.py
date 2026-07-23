"""Single-file overnight training run for Kaggle. Big corpus, 6 classes, under the cap.

Beats a bigger model with a smaller one by using more data, not more parameters:
ConvNeXt V2-Tiny (27.9M, IN-22k pretrained) instead of ResNet-50 (23.6M, IN-1k),
trained on ~19.5k images pooled from four public waste datasets instead of
TrashNet's 2527. Stays under the 30M pretrained-backbone competition cap.

Kaggle setup:
  Notebook > Settings > Accelerator: GPU T4 x2 (or P100) · Internet: ON
  !pip install -q timm datasets
  !python kaggle_train.py            # writes /kaggle/working/best_convnextv2.pt

Everything is checkpointed each epoch, and TIME_BUDGET_H stops cleanly before
Kaggle's session limit, so an overnight run always leaves a usable .pt behind.
"""

import hashlib
import io
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# ─── Config ───────────────────────────────────────────────────────────────────
SEED = 42
MODEL = "convnextv2_tiny.fcmae_ft_in22k_in1k"   # 27.9M params, under the 30M cap
MAX_PARAMS = 30_000_000
IMG_SIZE = 224
BATCH_SIZE = 48
EPOCHS_HEAD, EPOCHS_FT = 2, 28
LR_HEAD, LR_BACKBONE = 1e-3, 2e-5
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
MIXUP, CUTMIX = 0.2, 1.0
EMA_DECAY = 0.9998
TIME_BUDGET_H = 8.0          # stop before Kaggle kills the session
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

OUT = Path("/kaggle/working" if Path("/kaggle/working").exists() else ".")
CKPT = OUT / "best_convnextv2.pt"

# Same four HF sources and label mapping as scripts/download_unified_datasets.py,
# so this corpus matches the one the local pipeline was built around.
SOURCES = [
    ("garythung/trashnet", "trashnet"),
    ("omasteam/waste-garbage-management-dataset", "garbage_classification"),
    ("shahzaibvohra/realwaste", "realwaste"),
    ("steveharianto/waste-garbage-management-dataset", "garbage_v2"),
]
MAPPING = {
    "cardboard": "cardboard", "glass": "glass", "metal": "metal", "aluminium": "metal",
    "paper": "paper", "plastic": "plastic", "hard plastic": "plastic",
    "soft plastics": "plastic", "trash": "trash", "miscellaneous trash": "trash",
    # Classes with no six-class home (battery, clothes, shoes, biological, …) are
    # dropped rather than forced into `trash`, which would poison that class.
}

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Data ─────────────────────────────────────────────────────────────────────
def load_corpus():
    """Pool the sources, map to six classes, and deduplicate by pixel hash.

    Deduplication is the whole ballgame: these datasets overlap heavily (87.8% of
    TrashNet reappears inside garbage_classification). Without this, the same
    photo lands in train and val and the score becomes meaningless — a 99% that
    does not survive contact with a real photo. `group` ties every copy of an
    image together so the splitter keeps them on the same side.
    """
    from datasets import load_dataset

    records, groups = [], {}
    for repo_id, source in SOURCES:
        print(f"loading {repo_id} ...", flush=True)
        try:
            ds = load_dataset(repo_id, split="train")
        except Exception as exc:                       # a source may be gated/renamed
            print(f"  SKIPPED {repo_id}: {type(exc).__name__} {exc}")
            continue
        names = ds.features["label"].names
        for i, row in enumerate(ds):
            label = MAPPING.get(names[row["label"]].strip().lower())
            if label is None:
                continue
            img = row["image"].convert("RGB")
            digest = hashlib.sha256(img.resize((64, 64)).tobytes()).hexdigest()
            gid = groups.setdefault(digest, len(groups))
            records.append({"img": img, "label": CLASSES.index(label),
                            "group": gid, "digest": digest, "source": source})
        print(f"  {source}: kept {sum(r['source'] == source for r in records)}")

    seen, unique = set(), []
    for r in records:
        if r["digest"] in seen:
            continue
        seen.add(r["digest"])
        unique.append(r)
    print(f"\ncorpus: {len(records)} raw -> {len(unique)} unique "
          f"({len(records) - len(unique)} exact duplicates removed)")
    return unique


class WasteDataset(Dataset):
    def __init__(self, records, transform):
        self.records, self.transform = records, transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        return self.transform(r["img"]), r["label"]


MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
train_tf = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0)),
    T.RandomHorizontalFlip(),
    T.TrivialAugmentWide(),
    T.ToTensor(), T.Normalize(MEAN, STD),
    T.RandomErasing(p=0.25, scale=(0.02, 0.2)),
])
eval_tf = T.Compose([
    T.Resize(int(IMG_SIZE * 1.14)), T.CenterCrop(IMG_SIZE),
    T.ToTensor(), T.Normalize(MEAN, STD),
])


# ─── Train ────────────────────────────────────────────────────────────────────
def build():
    model = timm.create_model(MODEL, pretrained=True, num_classes=len(CLASSES))
    n = sum(p.numel() for p in model.parameters())
    if n > MAX_PARAMS:
        raise SystemExit(f"{MODEL} has {n/1e6:.1f}M params, over the {MAX_PARAMS/1e6:.0f}M cap")
    print(f"model={MODEL} params={n/1e6:.1f}M (cap {MAX_PARAMS/1e6:.0f}M)")
    return model.to(DEVICE)


def phase_optimizer(model, phase):
    head = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    for p in backbone:
        p.requires_grad = phase == "ft"
    groups = [{"params": head, "lr": LR_HEAD}]
    if phase == "ft":
        groups.append({"params": backbone, "lr": LR_BACKBONE})
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    per_class = torch.zeros(len(CLASSES), 2)
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item(); total += y.numel()
        for c in range(len(CLASSES)):
            m = y == c
            per_class[c, 0] += (pred[m] == c).sum().item(); per_class[c, 1] += m.sum().item()
    return correct / max(total, 1), per_class


def main():
    records = load_corpus()
    y = np.array([r["label"] for r in records])
    g = np.array([r["group"] for r in records])
    # Group-aware so no image (or duplicate of it) spans the split.
    tr_idx, va_idx = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(records, y, g))
    train_rec = [records[i] for i in tr_idx]
    val_rec = [records[i] for i in va_idx]
    assert not (set(g[tr_idx]) & set(g[va_idx])), "group leaked across split"
    print(f"train {len(train_rec)} | val {len(val_rec)} | groups disjoint OK")

    train_loader = DataLoader(WasteDataset(train_rec, train_tf), BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(WasteDataset(val_rec, eval_tf), BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = build()
    ema = timm.utils.ModelEmaV2(model, decay=EMA_DECAY)
    from timm.data import Mixup
    from timm.loss import SoftTargetCrossEntropy
    mixup = Mixup(mixup_alpha=MIXUP, cutmix_alpha=CUTMIX,
                  label_smoothing=LABEL_SMOOTHING, num_classes=len(CLASSES))
    criterion = SoftTargetCrossEntropy()
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")

    total_epochs = EPOCHS_HEAD + EPOCHS_FT
    best, phase, opt, sched = 0.0, None, None, None
    started = time.time()

    for epoch in range(total_epochs):
        want = "head" if epoch < EPOCHS_HEAD else "ft"
        if want != phase:
            phase = want
            opt = phase_optimizer(model, phase)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, total_epochs - epoch))
            print(f"--- phase {phase} ---")

        model.train()
        running = 0.0
        for x, y_ in train_loader:
            x, y_ = x.to(DEVICE, non_blocking=True), y_.to(DEVICE, non_blocking=True)
            x, target = mixup(x, y_)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                loss = criterion(model(x), target)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); ema.update(model)
            running += loss.item() * y_.size(0)
        sched.step()

        acc, per_class = evaluate(ema.module, val_loader)
        elapsed = (time.time() - started) / 3600
        print(f"epoch {epoch:2d} [{phase}] loss {running/len(train_rec):.4f} "
              f"val_acc {acc:.4f} (best {max(best, acc):.4f}) · {elapsed:.2f}h", flush=True)

        if acc > best:
            best = acc
            torch.save({k: v.cpu().clone() for k, v in ema.module.state_dict().items()}, CKPT)
            print(f"  saved {CKPT} @ {acc:.4f}")

        if elapsed > TIME_BUDGET_H:
            print(f"time budget {TIME_BUDGET_H}h reached — stopping cleanly")
            break

    print(f"\nbest val_acc {best:.4f} -> {CKPT}")
    for c, name in enumerate(CLASSES):
        correct, total = per_class[c]
        print(f"  {name:<11} {correct/max(total,1)*100:5.1f}%  ({int(correct)}/{int(total)})")


if __name__ == "__main__":
    main()
