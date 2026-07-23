"""
Swin-B Training with Aggressive Augmentation — Targeting 98-99%
================================================================
Uses Swin-B (larger model) + heavy augmentation pipeline:
  - Color: brightness, contrast, saturation, hue, grayscale
  - Geometric: rotation, affine, perspective, elastic
  - Degradation: blur, noise, erasing
  - CutMix + Mixup
  - Test-Time Augmentation (TTA) at evaluation
"""

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms import autoaugment, functional as TF
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.model_selection import train_test_split
import time
import json
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# ─── Config ───────────────────────────────────────────────────────────────────
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 24  # smaller batch for larger model
EPOCHS = 50
LR = 1e-4
WARMUP_EPOCHS = 5
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CUTMIX_ALPHA = 1.0
MIXUP_ALPHA = 0.8
LABEL_SMOOTHING = 0.1
TTA_VIEWS = 12

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "standardized_256")
CLASS_NAMES = sorted(os.listdir(DATA_ROOT))
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ─── Dataset ──────────────────────────────────────────────────────────────────
class GarbageDatasetFull(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def collect_samples(data_root):
    samples = []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        label = CLASS_TO_IDX[class_name]
        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                samples.append((fpath, label))
    return samples


# ─── Aggressive Augmentation Pipeline ────────────────────────────────────────
class AggressiveAugment:
    """Heavy augmentation: color, geometric, degradation."""
    def __init__(self, img_size=224):
        self.img_size = img_size

    def __call__(self, img):
        # Random resize + crop
        img = TF.resize(img, (self.img_size + 32, self.img_size + 32))
        i, j, h, w = transforms.RandomCrop.get_params(img, (self.img_size, self.img_size))
        img = TF.crop(img, i, j, h, w)

        # Geometric transforms
        if random.random() < 0.5:
            img = TF.hflip(img)
        if random.random() < 0.3:
            img = TF.vflip(img)
        if random.random() < 0.5:
            angle = random.uniform(-20, 20)
            img = TF.rotate(img, angle)
        if random.random() < 0.3:
            # Random affine (scale, translate, shear)
            img = TF.affine(img, angle=0, translate=(10, 10),
                           scale=random.uniform(0.9, 1.1),
                           shear=random.uniform(-10, 10))
        if random.random() < 0.2:
            # Random perspective
            img = TF.perspective(img,
                startpoints=[[0, 0], [self.img_size-1, 0],
                            [self.img_size-1, self.img_size-1], [0, self.img_size-1]],
                endpoints=[[5, 5], [self.img_size-5, 3],
                          [self.img_size-3, self.img_size-5], [3, self.img_size-3]])

        # Color/Chroma transforms
        if random.random() < 0.8:
            img = transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                         saturation=0.3, hue=0.15)(img)
        if random.random() < 0.2:
            img = TF.rgb_to_grayscale(img)
            img = TF.rgb_to_grayscale(img, num_output_channels=3)
        if random.random() < 0.3:
            # Random channel shuffle
            channels = list(img.split())
            random.shuffle(channels)
            img = Image.merge('RGB', channels)

        # Degradation transforms
        if random.random() < 0.2:
            # Gaussian blur
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        if random.random() < 0.15:
            # Sharpen
            img = img.filter(ImageFilter.SHARPEN)
        if random.random() < 0.1:
            # Edge enhance (like dilation effect)
            img = img.filter(ImageFilter.EDGE_ENHANCE)

        # Convert to tensor
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # Random erasing
        if random.random() < 0.3:
            img = transforms.RandomErasing(p=1.0, scale=(0.02, 0.25))(img)

        return img


train_transform = AggressiveAugment(IMG_SIZE)

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─── TTA Transforms ──────────────────────────────────────────────────────────
def get_tta_transforms():
    """Multiple views for test-time augmentation."""
    base_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tta_list = [
        # Original
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                           transforms.ToTensor(), base_norm]),
        # Horizontal flip
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                           transforms.RandomHorizontalFlip(p=1.0),
                           transforms.ToTensor(), base_norm]),
        # Vertical flip
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                           transforms.RandomVerticalFlip(p=1.0),
                           transforms.ToTensor(), base_norm]),
        # Center crop
        transforms.Compose([transforms.Resize((IMG_SIZE+32, IMG_SIZE+32)),
                           transforms.CenterCrop(IMG_SIZE),
                           transforms.ToTensor(), base_norm]),
        # Five crops
        transforms.Compose([transforms.Resize((IMG_SIZE+32, IMG_SIZE+32)),
                           transforms.FiveCrop(IMG_SIZE),
                           transforms.Lambda(lambda crops: crops[0]),
                           transforms.ToTensor(), base_norm]),
        transforms.Compose([transforms.Resize((IMG_SIZE+32, IMG_SIZE+32)),
                           transforms.FiveCrop(IMG_SIZE),
                           transforms.Lambda(lambda crops: crops[1]),
                           transforms.ToTensor(), base_norm]),
        transforms.Compose([transforms.Resize((IMG_SIZE+32, IMG_SIZE+32)),
                           transforms.FiveCrop(IMG_SIZE),
                           transforms.Lambda(lambda crops: crops[2]),
                           transforms.ToTensor(), base_norm]),
        transforms.Compose([transforms.Resize((IMG_SIZE+32, IMG_SIZE+32)),
                           transforms.FiveCrop(IMG_SIZE),
                           transforms.Lambda(lambda crops: crops[3]),
                           transforms.ToTensor(), base_norm]),
        transforms.Compose([transforms.Resize((IMG_SIZE+32, IMG_SIZE+32)),
                           transforms.FiveCrop(IMG_SIZE),
                           transforms.Lambda(lambda crops: crops[4]),
                           transforms.ToTensor(), base_norm]),
        # Color jitter
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                           transforms.ColorJitter(brightness=0.2),
                           transforms.ToTensor(), base_norm]),
        # Slight rotation
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                           transforms.RandomRotation((5, 5)),
                           transforms.ToTensor(), base_norm]),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                           transforms.RandomRotation((-5, -5)),
                           transforms.ToTensor(), base_norm]),
    ]
    return tta_list[:TTA_VIEWS]

# ─── CutMix & Mixup ──────────────────────────────────────────────────────────
def cutmix(images, labels, alpha=CUTMIX_ALPHA):
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    W, H = images.size(2), images.size(3)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    images[:, :, x1:x2, y1:y2] = images[index, :, x1:x2, y1:y2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return images, labels, labels[index], lam


def mixup(images, labels, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)
    images = lam * images + (1 - lam) * images[index]
    return images, labels, labels[index], lam


# ─── Swin-B Model ────────────────────────────────────────────────────────────
class SwinBClassifier(nn.Module):
    """Swin-B with pretrained weights."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.swin = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
        self.swin.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.swin.head.in_features, num_classes)
        )

    def forward(self, x):
        return self.swin(x)


# ─── Training ─────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        r = random.random()
        if r < 0.35:
            images, targets_a, targets_b, lam = cutmix(images, labels)
            logits = model(images)
            loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)
        elif r < 0.7:
            images, targets_a, targets_b, lam = mixup(images, labels)
            logits = model(images)
            loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += len(labels)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return correct / total, all_preds, all_labels


@torch.no_grad()
def evaluate_tta(model, dataset, tta_transforms):
    """Test-time augmentation."""
    model.eval()
    all_correct, all_total = 0, 0
    for idx in range(len(dataset)):
        path, label = dataset.samples[idx]
        img = Image.open(path).convert("RGB")
        avg_logits = None
        for tf in tta_transforms:
            img_t = tf(img).unsqueeze(0).to(DEVICE)
            logits = model(img_t)
            avg_logits = logits if avg_logits is None else avg_logits + logits
        pred = avg_logits.argmax(1).item()
        if pred == label:
            all_correct += 1
        all_total += 1
    return all_correct / all_total


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  SWIN-B + AGGRESSIVE AUGMENTATION — TARGET: 98-99%")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LR}")
    print(f"  Augmentation: Color + Geometric + Degradation + CutMix/Mixup")
    print(f"  TTA views: {TTA_VIEWS}")
    print()

    # Collect samples
    all_samples = collect_samples(DATA_ROOT)
    print(f"  Total images: {len(all_samples)}")

    # Same split as before
    labels_arr = [s[1] for s in all_samples]
    train_samples, temp_samples, train_labels, temp_labels = train_test_split(
        all_samples, labels_arr, test_size=0.30, random_state=SEED, stratify=labels_arr)
    val_samples, test_samples, _, _ = train_test_split(
        temp_samples, temp_labels, test_size=0.50, random_state=SEED, stratify=temp_labels)

    print(f"  Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    train_ds = GarbageDatasetFull(train_samples, transform=train_transform)
    val_ds = GarbageDatasetFull(val_samples, transform=val_transform)
    test_ds = GarbageDatasetFull(test_samples, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=False)

    # Model
    model = SwinBClassifier(NUM_CLASSES).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {params:,} (trainable: {trainable:,})")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Training
    print(f"\n{'─' * 70}")
    print("  TRAINING Swin-B")
    print(f"{'─' * 70}")

    best_val_acc = 0
    best_state = None

    for epoch in range(EPOCHS):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_acc, _, _ = evaluate(model, val_loader)
        scheduler.step()
        elapsed = time.time() - t0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "best_swin_b.pt"))

        lr_now = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == EPOCHS - 1:
            print(f"  Ep {epoch+1:2d}/{EPOCHS} | lr={lr_now:.5f} | "
                  f"Loss: {train_loss:.4f} | Train: {train_acc:.4f} | "
                  f"Val: {val_acc:.4f} (best: {best_val_acc:.4f}) | {elapsed:.0f}s")

    # Load best and evaluate
    model.load_state_dict(best_state)
    model = model.to(DEVICE)

    # Standard test
    test_acc, preds, labels = evaluate(model, test_loader)
    print(f"\n  Standard test accuracy: {test_acc:.4f}")

    # TTA test
    print(f"  Running TTA ({TTA_VIEWS} views)...")
    tta_transforms = get_tta_transforms()
    tta_acc = evaluate_tta(model, test_ds, tta_transforms)
    print(f"  TTA test accuracy:      {tta_acc:.4f}")
    print(f"  TTA improvement:        {tta_acc - test_acc:+.4f}")

    # Per-class accuracy with TTA
    print(f"\n  Per-Class Accuracy (TTA):")
    model.eval()
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    for idx in range(len(test_ds)):
        path, label = test_ds.samples[idx]
        img = Image.open(path).convert("RGB")
        avg_logits = None
        for tf in tta_transforms:
            img_t = tf(img).unsqueeze(0).to(DEVICE)
            logits = model(img_t)
            avg_logits = logits if avg_logits is None else avg_logits + logits
        pred = avg_logits.argmax(1).item()
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1

    for i, cls_name in enumerate(CLASS_NAMES):
        acc = class_correct[i] / max(class_total[i], 1)
        print(f"    {cls_name:<12}: {acc:.4f} ({class_correct[i]}/{class_total[i]})")

    # Final comparison
    print(f"\n{'=' * 70}")
    print("  FINAL COMPARISON")
    print(f"{'=' * 70}")
    print(f"\n  {'Model':<35} | {'Val':>7} | {'Test':>7} | {'TTA':>7}")
    print(f"  {'─'*35}-+-{'─'*7}-+-{'─'*7}-+-{'─'*7}")
    print(f"  {'CNN-only (ResNet-50)':<35} | {'96.0%':>7} | {'95.8%':>7} | {'—':>7}")
    print(f"  {'Swin-Tiny':<35} | {'95.9%':>7} | {'96.2%':>7} | {'—':>7}")
    print(f"  {'Swin-B + Aug + TTA':<35} | {best_val_acc*100:>6.1f}% | {test_acc*100:>6.1f}% | {tta_acc*100:>6.1f}%")

    print(f"\n  Target: 99.0%")
    print(f"  Gap to target: {0.99 - tta_acc:+.4f}")
    if tta_acc >= 0.99:
        print(f"  ✓ TARGET ACHIEVED!")
    else:
        print(f"  → Consider: ensemble, more epochs, or data cleanup")

    print("=" * 70)

    # Save results
    results = {
        "swin_b": {
            "val_acc": best_val_acc,
            "test_acc": test_acc,
            "tta_acc": tta_acc,
            "params": params,
        },
        "per_class_tta": {CLASS_NAMES[i]: class_correct[i] / max(class_total[i], 1)
                          for i in range(NUM_CLASSES)},
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "swin_b_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
