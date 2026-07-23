"""
Garbage vs Non-Garbage Classifier — Multi-Signal OOD Detection
===============================================================
Combines a fine-tuned garbage Swin-B with the original ImageNet Swin-B
to detect whether an image is garbage or a non-garbage object.

Five complementary OOD signals are fused:
  1. Energy-based score (Liu et al. 2020)
  2. Max Softmax Probability (Hendrycks & Gimpel 2017)
  3. Mahalanobis distance in feature space (Lee et al. 2018)
  4. ImageNet model confidence ratio
  5. Feature-space nearest-neighbor distance to garbage prototypes

The Mahalanobis + feature-distance signals are the strongest because
they operate in the 1024-dim feature space rather than the compressed
10-dim logit space.

Usage:
  from garbage_vs_nongarbage import GarbageOrNotClassifier

  clf = GarbageOrNotClassifier(checkpoint_path="best_swin_b.pt")
  clf.calibrate_features(garbage_data_dir="path/to/garbage/images")

  result = clf.classify("photo.jpg")
  # → {"is_garbage": True, "garbage_class": "plastic", "score": 0.87, ...}
"""

import os
import math
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union
from collections import OrderedDict
from pathlib import Path
from scipy.spatial.distance import mahalanobis as mahalanobis_distance_raw


# ═══════════════════════════════════════════════════════════════════════════════
# Model Definitions
# ═══════════════════════════════════════════════════════════════════════════════

class SwinB_Garbage(nn.Module):
    """Fine-tuned garbage classifier with feature extraction."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.swin = models.swin_b(weights=None)
        self.swin.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.swin.head.in_features, num_classes)
        )

    def forward(self, x, return_features: bool = False):
        x = self.swin.features(x)
        x = self.swin.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.swin.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.swin.head(features)
        if return_features:
            return logits, features
        return logits


class SwinB_ImageNet(nn.Module):
    """Original ImageNet-1k Swin-B with feature extraction."""
    def __init__(self):
        super().__init__()
        self.swin = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)

    def forward(self, x, return_features: bool = False):
        x = self.swin.features(x)
        x = self.swin.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.swin.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.swin.head(features)
        if return_features:
            return logits, features
        return logits


# ═══════════════════════════════════════════════════════════════════════════════
# ImageNet Class Names
# ═══════════════════════════════════════════════════════════════════════════════

def load_imagenet_class_names() -> List[str]:
    import urllib.request
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "imagenet_classes.txt")
    if not os.path.exists(cache_path):
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        print(f"  Downloading ImageNet class names...")
        urllib.request.urlretrieve(url, cache_path)
    with open(cache_path) as f:
        return [line.strip() for line in f.readlines()]


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def preprocess(image: Union[str, Image.Image]) -> torch.Tensor:
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    else:
        img = image.convert("RGB")
    return base_transform(img).unsqueeze(0)


def preprocess_batch(paths: List[str], device: torch.device) -> torch.Tensor:
    """Batch preprocess multiple image paths into one tensor."""
    tensors = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            tensors.append(base_transform(img))
        except Exception:
            tensors.append(torch.zeros(3, 224, 224))
    return torch.stack(tensors).to(device)


# ═══════════════════════════════════════════════════════════════════════════════
# OOD Score Functions
# ═══════════════════════════════════════════════════════════════════════════════

def energy_score(logits: torch.Tensor, T: float = 1.0) -> np.ndarray:
    """Energy-based OOD score. Lower = in-distribution."""
    return (-T * torch.logsumexp(logits / T, dim=-1)).detach().cpu().numpy()


def msp_score(logits: torch.Tensor) -> np.ndarray:
    """Maximum Softmax Probability."""
    return F.softmax(logits, dim=-1).max(dim=-1).values.detach().cpu().numpy()


def predictive_entropy(logits: torch.Tensor) -> np.ndarray:
    """Entropy of predictive distribution. Higher = more uncertain."""
    probs = F.softmax(logits, dim=-1)
    return -(probs * torch.log(probs + 1e-10)).sum(dim=-1).detach().cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# Mahalanobis OOD Detector (Lee et al. 2018)
# ═══════════════════════════════════════════════════════════════════════════════

class MahalanobisDetector:
    """Class-conditional Gaussian OOD detector in feature space.

    Fits a multivariate Gaussian per class on the penultimate-layer features.
    At test time, the minimum Mahalanobis distance to any class centroid
    determines if a sample is in-distribution.
    """
    def __init__(self):
        self.means: Dict[int, np.ndarray] = {}       # class_idx → (D,) mean
        self.precisions: Dict[int, np.ndarray] = {}  # class_idx → (D, D) precision
        self.feature_dim: int = 0
        self.threshold: Optional[float] = None
        self.class_names: List[str] = []

    def fit(self, features: np.ndarray, labels: np.ndarray,
            class_names: List[str], shrinkage: float = 0.01):
        """Fit class-conditional Gaussians with shrinkage regularization.

        Parameters
        ----------
        features : (N, D) array of penultimate features
        labels : (N,) array of integer class labels
        class_names : list of class name strings
        shrinkage : covariance shrinkage factor for regularization
        """
        self.feature_dim = features.shape[1]
        self.class_names = list(class_names)
        unique_labels = np.unique(labels)

        for lbl in unique_labels:
            mask = labels == lbl
            class_feats = features[mask]
            n = class_feats.shape[0]

            mean = class_feats.mean(axis=0)
            # Shrinkage covariance estimate
            cov = np.cov(class_feats, rowvar=False)
            cov = (1 - shrinkage) * cov + shrinkage * np.eye(self.feature_dim)
            # Add small diagonal for numerical stability
            cov += 1e-6 * np.eye(self.feature_dim)
            precision = np.linalg.inv(cov)

            self.means[int(lbl)] = mean
            self.precisions[int(lbl)] = precision

        # Compute threshold: 95th percentile of in-distribution distances
        all_dists = []
        for lbl in unique_labels:
            mask = labels == lbl
            class_feats = features[mask]
            mean = self.means[int(lbl)]
            prec = self.precisions[int(lbl)]
            for feat in class_feats:
                diff = feat - mean
                d = diff @ prec @ diff
                all_dists.append(d)
        self.threshold = float(np.percentile(all_dists, 95))

        return self

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        """Return minimum Mahalanobis distance for each sample.
        Lower = more like training data (garbage)."""
        n = features.shape[0]
        min_dists = np.full(n, np.inf)
        for lbl, mean in self.means.items():
            prec = self.precisions[lbl]
            diff = features - mean  # (N, D)
            # Vectorized Mahalanobis: (N,D) @ (D,D) @ (N,D)ᵀ → diag
            dists = np.einsum('nd,de,ne->n', diff, prec, diff)
            min_dists = np.minimum(min_dists, dists)
        return min_dists

    def is_ood(self, features: np.ndarray) -> np.ndarray:
        """Return boolean array: True if OOD (non-garbage)."""
        if self.threshold is None:
            raise RuntimeError("Must fit() before calling is_ood()")
        return self.score_samples(features) > self.threshold


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Prototype Bank (nearest-neighbor OOD)
# ═══════════════════════════════════════════════════════════════════════════════

class FeaturePrototypeBank:
    """Stores prototype features from garbage training images.
    At test time, the minimum cosine distance to any prototype indicates OOD.
    """
    def __init__(self):
        self.prototypes: np.ndarray = None       # (P, D)
        self.labels: np.ndarray = None           # (P,)
        self.threshold: Optional[float] = None

    def build(self, features: np.ndarray, labels: np.ndarray, n_per_class: int = 50):
        """Build prototype bank by sampling n_per_class exemplars per class."""
        prototypes = []
        proto_labels = []
        for lbl in np.unique(labels):
            class_feats = features[labels == lbl]
            n = min(n_per_class, len(class_feats))
            idx = np.random.choice(len(class_feats), n, replace=False)
            prototypes.append(class_feats[idx])
            proto_labels.extend([int(lbl)] * n)
        self.prototypes = np.vstack(prototypes)
        self.prototypes = self.prototypes / (np.linalg.norm(
            self.prototypes, axis=1, keepdims=True) + 1e-10)
        self.labels = np.array(proto_labels)

        # Threshold: 95th percentile of training cosine distances
        all_cos_dists = []
        for i in range(0, len(self.prototypes), 50):
            batch = self.prototypes[i:i+50]
            sim = batch @ self.prototypes.T
            # Exclude self-match
            for j in range(len(batch)):
                row = sim[j]
                row[np.argmax(row)] = -np.inf
                all_cos_dists.append(1.0 - row.max())
        self.threshold = float(np.percentile(all_cos_dists, 95))
        return self

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        """Cosine distance to nearest prototype. Higher = more OOD."""
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)
        sim = features_norm @ self.prototypes.T  # (N, P)
        return 1.0 - sim.max(axis=1)  # cosine distance

    def is_ood(self, features: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Must build() before calling is_ood()")
        return self.score_samples(features) > self.threshold


# ═══════════════════════════════════════════════════════════════════════════════
# Main Classifier
# ═══════════════════════════════════════════════════════════════════════════════

class GarbageOrNotClassifier:
    """Multi-signal garbage vs non-garbage classifier.

    Fuses 5 OOD detection signals:
      Signal 1 — Mahalanobis distance (feature space): strongest signal
      Signal 2 — Prototype cosine distance (feature space)
      Signal 3 — Energy score (logit space)
      Signal 4 — MSP / confidence (logit space)
      Signal 5 — ImageNet model confidence ratio (dual-model)
    """

    def __init__(
        self,
        checkpoint_path: str,
        garbage_class_names: List[str] = None,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Default 10-class garbage names (must match training order)
        if garbage_class_names is None:
            garbage_class_names = [
                "battery", "biological", "cardboard", "clothes",
                "glass", "metal", "paper", "plastic", "shoes", "trash"
            ]
        self.garbage_class_names = list(garbage_class_names)
        self.num_classes = len(self.garbage_class_names)

        # ── Load models ──
        print(f"  Loading garbage model from {checkpoint_path}...")
        self.garbage_model = SwinB_Garbage(self.num_classes)
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.garbage_model.load_state_dict(state)
        self.garbage_model.to(self.device)
        self.garbage_model.eval()

        print(f"  Loading ImageNet model...")
        self.imagenet_model = SwinB_ImageNet()
        self.imagenet_model.to(self.device)
        self.imagenet_model.eval()
        self.imagenet_class_names = load_imagenet_class_names()

        # ── OOD detectors ──
        self.mahalanobis_detector: Optional[MahalanobisDetector] = None
        self.prototype_bank: Optional[FeaturePrototypeBank] = None

        # ── Calibrated thresholds ──
        self._energy_threshold: Optional[float] = None
        self._msp_threshold: Optional[float] = None

        # ── Thresholds for the fusion ──
        # Each signal produces a normalized score in [0, 1] where >0.5 → non-garbage
        # We use a QUORUM-based fusion:
        #   - Feature-space signals (mahalanobis, prototype) are STRONG for
        #     identifying clear non-garbage, but can be overzealous on
        #     unusual-looking garbage
        #   - Logit-space signals (energy, MSP) are more conservative and
        #     better at recognizing legitimate garbage
        #   - So: if BOTH feature-space signals strongly say OOD, that's
        #     a clear non-garbage. Otherwise, defer to logit-space signals.
        self.signal_weights = {
            "mahalanobis": 0.25,
            "prototype":   0.15,
            "energy":      0.25,
            "msp":         0.20,
            "imagenet":    0.15,
        }

        # Decision boundary on the fused score
        self.decision_threshold = 0.50  # >0.5 → non-garbage

        # Feature-space override: if BOTH mahalanobis AND prototype
        # exceed this threshold, immediately classify as non-garbage
        self.feature_space_override = 0.85

        print(f"  ✓ Ready ({self.num_classes} garbage classes, 1000 ImageNet classes)")
        print(f"  Device: {self.device}")

    # ── Feature extraction ───────────────────────────────────────────────

    @torch.no_grad()
    def extract_features(self, images: Union[str, List[str], torch.Tensor],
                          batch_size: int = 32) -> np.ndarray:
        """Extract 1024-dim penultimate features from the garbage model."""
        if isinstance(images, torch.Tensor):
            tensors = images
        elif isinstance(images, str):
            tensors = preprocess(images).to(self.device)
        else:
            # List of paths
            all_feats = []
            for i in range(0, len(images), batch_size):
                batch_paths = images[i:i+batch_size]
                batch_t = preprocess_batch(batch_paths, self.device)
                _, feats = self.garbage_model(batch_t, return_features=True)
                all_feats.append(feats.cpu().numpy())
            return np.vstack(all_feats)

        _, features = self.garbage_model(tensors, return_features=True)
        return features.cpu().numpy()

    # ── Calibration ──────────────────────────────────────────────────────

    def calibrate_features(
        self,
        garbage_data_dir: str,
        n_samples: int = 500,
        batch_size: int = 32,
    ) -> Dict:
        """Extract features from garbage images and fit OOD detectors.

        This is the key step — it learns what garbage features look like
        so that non-garbage features can be detected as anomalous.

        Parameters
        ----------
        garbage_data_dir : str
            Directory with class subdirectories containing garbage images.
        n_samples : int
            Max number of images to use for calibration.
        batch_size : int
            Batch size for feature extraction.

        Returns
        -------
        Dict with calibration statistics.
        """
        # ── Check for cached calibration ──
        cache_dir = Path(garbage_data_dir).parent
        cache_path = cache_dir / ".garbage_ood_calibration.npz"

        if cache_path.exists():
            print(f"\n  Loading cached calibration from {cache_path}...")
            cached = np.load(cache_path, allow_pickle=True)
            features = cached["features"]
            labels_arr = cached["labels"]
            print(f"  Loaded {len(features)} precomputed features ({features.shape[1]}D)")

            # Re-fit detectors from cached features
            self.mahalanobis_detector = MahalanobisDetector().fit(
                features, labels_arr, self.garbage_class_names
            )
            self.prototype_bank = FeaturePrototypeBank().build(features, labels_arr)

            # Recompute logit thresholds from cached features
            mahal_dists = self.mahalanobis_detector.score_samples(features)
            mahal_thresh = self.mahalanobis_detector.threshold
            proto_dists = self.prototype_bank.score_samples(features)
            proto_thresh = self.prototype_bank.threshold

            # For energy/MSP we need logits, not features. Recompute from cached paths.
            cached_paths = list(cached["paths"])
            all_energies = []
            all_msps = []
            for i in range(0, len(cached_paths), batch_size):
                batch_paths = cached_paths[i:i+batch_size]
                batch_t = preprocess_batch(batch_paths, self.device)
                logits = self.garbage_model(batch_t)
                all_energies.extend(energy_score(logits).tolist())
                all_msps.extend(msp_score(logits).tolist())

            self._energy_threshold = float(np.percentile(all_energies, 95))
            self._msp_threshold = float(np.percentile(all_msps, 5))

            result = {
                "n_images": len(cached_paths),
                "feature_dim": features.shape[1],
                "from_cache": True,
                "mahalanobis": {
                    "threshold": float(mahal_thresh),
                    "train_mean": float(mahal_dists.mean()),
                    "train_std": float(mahal_dists.std()),
                    "train_p95": float(np.percentile(mahal_dists, 95)),
                },
                "prototype": {
                    "threshold": float(proto_thresh),
                    "train_mean": float(proto_dists.mean()),
                    "train_std": float(proto_dists.std()),
                    "train_p95": float(np.percentile(proto_dists, 95)),
                },
                "energy": {
                    "threshold": self._energy_threshold,
                    "train_mean": float(np.mean(all_energies)),
                    "train_std": float(np.std(all_energies)),
                },
                "msp": {
                    "threshold": self._msp_threshold,
                    "train_mean": float(np.mean(all_msps)),
                    "train_std": float(np.std(all_msps)),
                },
            }
            self._calibration = result
            print(f"  ✓ Loaded from cache")
            print(f"    Mahalanobis threshold: {mahal_thresh:.2f}")
            print(f"    Prototype threshold:   {proto_thresh:.4f}")
            print(f"    Energy threshold:      {self._energy_threshold:.3f}")
            print(f"    MSP threshold:         {self._msp_threshold:.4f}")
            return result

        print(f"\n{'─' * 55}")
        print(f"  CALIBRATING FEATURE-SPACE OOD DETECTORS")
        print(f"{'─' * 55}")

        # ── Collect garbage image paths and labels ──
        t0 = time.time()
        paths = []
        labels = []
        for cls_dir in sorted(Path(garbage_data_dir).iterdir()):
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name
            if cls_name not in self.garbage_class_names:
                continue
            cls_idx = self.garbage_class_names.index(cls_name)
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                    paths.append(str(img_path))
                    labels.append(cls_idx)

        # Subsample if needed
        if len(paths) > n_samples:
            indices = np.random.choice(len(paths), n_samples, replace=False)
            paths = [paths[i] for i in indices]
            labels = [labels[i] for i in indices]

        print(f"  Using {len(paths)} garbage images from {garbage_data_dir}")

        # ── Extract features ──
        print(f"  Extracting features (batch_size={batch_size})...")
        all_features = []
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i+batch_size]
            feats = self.extract_features(batch, batch_size=batch_size)
            all_features.append(feats)
        features = np.vstack(all_features)
        labels_arr = np.array(labels)

        # ── Fit Mahalanobis detector ──
        print(f"  Fitting Mahalanobis detector...")
        self.mahalanobis_detector = MahalanobisDetector().fit(
            features, labels_arr, self.garbage_class_names
        )
        mahal_dists = self.mahalanobis_detector.score_samples(features)
        mahal_thresh = self.mahalanobis_detector.threshold

        # ── Build prototype bank ──
        print(f"  Building prototype bank...")
        self.prototype_bank = FeaturePrototypeBank().build(features, labels_arr)
        proto_dists = self.prototype_bank.score_samples(features)
        proto_thresh = self.prototype_bank.threshold

        # ── Calibrate energy and MSP from logits ──
        print(f"  Calibrating logit-space thresholds...")
        all_energies = []
        all_msps = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i+batch_size]
            batch_t = preprocess_batch(batch_paths, self.device)
            logits = self.garbage_model(batch_t)
            all_energies.extend(energy_score(logits).tolist())
            all_msps.extend(msp_score(logits).tolist())

        self._energy_threshold = float(np.percentile(all_energies, 95))
        self._msp_threshold = float(np.percentile(all_msps, 5))

        elapsed = time.time() - t0

        # ── Compute normalized scores on training data ──
        # We want the calibrated signals to produce scores in [0,1]
        # where 0 = definitely garbage, 1 = definitely non-garbage
        # Calibrate the scaling factors so 95th percentile of training → score ≈ 0.3
        result = {
            "n_images": len(paths),
            "feature_dim": features.shape[1],
            "time_seconds": round(elapsed, 1),
            "mahalanobis": {
                "threshold": float(mahal_thresh),
                "train_mean": float(mahal_dists.mean()),
                "train_std": float(mahal_dists.std()),
                "train_p95": float(np.percentile(mahal_dists, 95)),
            },
            "prototype": {
                "threshold": float(proto_thresh),
                "train_mean": float(proto_dists.mean()),
                "train_std": float(proto_dists.std()),
                "train_p95": float(np.percentile(proto_dists, 95)),
            },
            "energy": {
                "threshold": self._energy_threshold,
                "train_mean": float(np.mean(all_energies)),
                "train_std": float(np.std(all_energies)),
            },
            "msp": {
                "threshold": self._msp_threshold,
                "train_mean": float(np.mean(all_msps)),
                "train_std": float(np.std(all_msps)),
            },
        }

        # ── Cache features for faster reload ──
        cache_path = Path(garbage_data_dir).parent / ".garbage_ood_calibration.npz"
        np.savez(
            cache_path,
            features=features,
            labels=labels_arr,
            paths=np.array(paths),
        )
        print(f"  Cached features to {cache_path}")

        self._calibration = result
        print(f"  ✓ Calibrated in {elapsed:.1f}s")
        print(f"    Mahalanobis threshold: {mahal_thresh:.2f}")
        print(f"    Prototype threshold:   {proto_thresh:.4f}")
        print(f"    Energy threshold:      {self._energy_threshold:.3f}")
        print(f"    MSP threshold:         {self._msp_threshold:.4f}")

        return result

    # ── Single-image classification ─────────────────────────────────────

    @torch.no_grad()
    def classify(self, image: Union[str, Image.Image]) -> Dict:
        """Classify a single image as garbage or non-garbage.

        Returns dict with keys:
          - is_garbage: bool
          - garbage_class: str or None
          - score: float — fused OOD score (0=garbage, 1=non-garbage)
          - confidence: float — garbage model max softmax
          - individual_scores: Dict[str, float] — per-signal normalized scores
          - top_garbage: List[Tuple[str, float]]
          - top_imagenet: List[Tuple[str, float]]
          - method: str — how the decision was made
        """
        img_tensor = preprocess(image).to(self.device)

        # ── Forward passes ──
        garbage_logits, garbage_features = self.garbage_model(
            img_tensor, return_features=True
        )
        imagenet_logits, imagenet_features = self.imagenet_model(
            img_tensor, return_features=True
        )

        garbage_probs = F.softmax(garbage_logits, dim=-1)
        imagenet_probs = F.softmax(imagenet_logits, dim=-1)

        # ── Top predictions ──
        top_gc_probs, top_gc_idx = torch.topk(garbage_probs, k=3, dim=-1)
        top_garbage = [
            (self.garbage_class_names[idx.item()], prob.item())
            for idx, prob in zip(top_gc_idx[0], top_gc_probs[0])
        ]

        top_in_probs, top_in_idx = torch.topk(imagenet_probs, k=3, dim=-1)
        top_imagenet = [
            (self.imagenet_class_names[idx.item()], prob.item())
            for idx, prob in zip(top_in_idx[0], top_in_probs[0])
        ]

        # ── Raw scores ──
        raw_energy = float(energy_score(garbage_logits)[0])
        raw_msp = float(msp_score(garbage_logits)[0])
        raw_imagenet_msp = float(msp_score(imagenet_logits)[0])
        feat_np = garbage_features.cpu().numpy()

        # ── Normalize each signal to [0, 1] where >0.5 → non-garbage ──

        # Signal 1: Mahalanobis distance
        if self.mahalanobis_detector is not None:
            raw_mahal = float(self.mahalanobis_detector.score_samples(feat_np)[0])
            thresh = self.mahalanobis_detector.threshold
            # Normalize: score = sigmoid((d - threshold) / scale)
            # At threshold: score ≈ 0.50; at 2*threshold: score ≈ 0.88
            score_mahal = float(1.0 / (1.0 + np.exp(-3.0 * (raw_mahal / max(thresh, 0.01) - 1.0))))
        else:
            raw_mahal = None
            score_mahal = 0.5  # neutral if uncalibrated

        # Signal 2: Prototype distance
        if self.prototype_bank is not None:
            raw_proto = float(self.prototype_bank.score_samples(feat_np)[0])
            thresh = self.prototype_bank.threshold
            score_proto = float(1.0 / (1.0 + np.exp(-5.0 * (raw_proto / max(thresh, 0.001) - 1.0))))
        else:
            raw_proto = None
            score_proto = 0.5

        # Signal 3: Energy
        if self._energy_threshold is not None:
            # Above threshold → OOD. Scale so threshold → 0.5
            diff = raw_energy - self._energy_threshold
            score_energy = float(1.0 / (1.0 + np.exp(-diff)))
        else:
            # Default: energy > -3.0 → suspicious
            score_energy = float(1.0 / (1.0 + np.exp(-(raw_energy + 3.0))))

        # Signal 4: MSP (inverted — low MSP → OOD)
        if self._msp_threshold is not None:
            diff = self._msp_threshold - raw_msp  # positive when below threshold
            score_msp = float(1.0 / (1.0 + np.exp(-5.0 * diff)))
        else:
            # Default: MSP < 0.30 → suspicious
            score_msp = float(1.0 / (1.0 + np.exp(-5.0 * (0.30 - raw_msp))))

        # Signal 5: ImageNet confidence ratio
        # If ImageNet is confident AND garbage model is less confident → non-garbage
        garbage_entropy = float(predictive_entropy(garbage_logits)[0])
        imagenet_entropy = float(predictive_entropy(imagenet_logits)[0])
        # Entropy ratio: if ImageNet is more certain (lower entropy) than garbage model
        entropy_ratio = imagenet_entropy / max(garbage_entropy, 0.01)
        # Also: raw ImageNet confidence
        score_imagenet = float(1.0 / (1.0 + np.exp(-3.0 * (raw_imagenet_msp - 0.5))))

        individual_scores = {
            "mahalanobis": score_mahal,
            "prototype": score_proto,
            "energy": score_energy,
            "msp": score_msp,
            "imagenet": score_imagenet,
        }

        # ── Fuse signals ──
        # Quorum approach: feature-space signals identify clear non-garbage,
        # logit-space signals handle the ambiguous cases
        weights = dict(self.signal_weights)
        if self.mahalanobis_detector is None:
            weights["mahalanobis"] = 0
            weights["prototype"] = 0
            weights["energy"] = 0.35
            weights["msp"] = 0.35
            weights["imagenet"] = 0.30

        total_w = sum(weights.values())
        fused_score = sum(
            individual_scores[k] * weights[k] / total_w for k in weights
        )

        # ── Decision with quorum logic ──
        method = "fused"

        # Feature-space quorum: if BOTH mahalanobis AND prototype are strongly
        # OOD (>0.85), this is very likely non-garbage.  But if they disagree
        # (one high, one low), trust the logit-space signals.
        feature_ood_quorum = (
            self.mahalanobis_detector is not None
            and score_mahal > self.feature_space_override
            and score_proto > self.feature_space_override
        )

        # Feature-space in-distribution quorum: if BOTH are low (<0.30),
        # this is very likely garbage, regardless of logit-space uncertainty
        feature_id_quorum = (
            self.mahalanobis_detector is not None
            and score_mahal < 0.30
            and score_proto < 0.30
        )

        if feature_id_quorum:
            # Feature space strongly says garbage — override everything else
            is_garbage = True
            method = "feature_space_id_quorum"
        elif feature_ood_quorum:
            # Feature space strongly says non-garbage
            is_garbage = False
            method = "feature_space_ood_quorum"
        elif raw_msp > 0.95:
            # Extremely confident garbage prediction
            is_garbage = True
            method = "very_high_confidence"
        elif raw_imagenet_msp > 0.80 and raw_msp < 0.40:
            # ImageNet is very confident about a known object
            is_garbage = False
            method = "imagenet_override"
        elif 0.48 <= fused_score <= 0.55 and raw_msp > 0.70:
            # Borderline score but garbage model is reasonably confident.
            # This handles unusual-looking garbage (e.g., crumpled trash,
            # oddly-lit paper) that confuses feature-space detectors.
            is_garbage = True
            method = "borderline_garbage_model"
        else:
            # Default: use the fused score
            is_garbage = fused_score <= self.decision_threshold

        return {
            "is_garbage": is_garbage,
            "garbage_class": top_garbage[0][0] if is_garbage else None,
            "score": round(fused_score, 4),          # fused OOD score
            "confidence": round(raw_msp, 4),          # garbage model confidence
            "energy": round(raw_energy, 4),
            "mahalanobis": round(raw_mahal, 2) if raw_mahal is not None else None,
            "prototype_dist": round(raw_proto, 4) if raw_proto is not None else None,
            "entropy_ratio": round(float(entropy_ratio), 4),
            "individual_scores": {k: round(v, 4) for k, v in individual_scores.items()},
            "top_garbage": top_garbage,
            "top_imagenet": top_imagenet,
            "method": method,
        }

    # ── Batch classification ─────────────────────────────────────────────

    def classify_batch(self, images: List[Union[str, Image.Image]]) -> List[Dict]:
        return [self.classify(img) for img in images]

    # ── Pretty printing ───────────────────────────────────────────────────

    def describe(self, result: Dict) -> str:
        verdict = "🗑️  GARBAGE" if result["is_garbage"] else "✅ NOT GARBAGE"
        lines = [
            f"{'─' * 55}",
            f"  Verdict: {verdict}  (method: {result['method']})",
            f"  Fused OOD score: {result['score']:.4f}  "
            f"(>0.5 → non-garbage)",
        ]
        if result["garbage_class"]:
            lines.append(f"  Garbage class: {result['garbage_class']}")
        lines += [
            f"  Garbage confidence: {result['confidence']:.4f}",
            f"  Energy: {result['energy']:.4f}",
            f"  Mahalanobis: {result['mahalanobis']}",
            f"  Prototype dist: {result['prototype_dist']}",
            f"",
            f"  Individual signal scores (→1 = non-garbage):",
        ]
        for k, v in result["individual_scores"].items():
            bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
            lines.append(f"    {k:<14} [{bar}] {v:.4f}")
        lines += [
            f"",
            f"  Top-3 garbage:",
        ]
        for cls, prob in result["top_garbage"]:
            lines.append(f"    {cls:<16} {prob:.4f}")
        lines += [f"  Top-3 ImageNet:"]
        for cls, prob in result["top_imagenet"]:
            lines.append(f"    {cls:<16} {prob:.4f}")
        lines.append(f"{'─' * 55}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Garbage vs Non-Garbage — Multi-Signal OOD Classifier"
    )
    parser.add_argument("image", nargs="*", help="Image file(s) to classify")
    parser.add_argument("--checkpoint", default="best_swin_b.pt",
                        help="Garbage Swin-B checkpoint")
    parser.add_argument("--classes", nargs="*",
                        default=["battery","biological","cardboard","clothes",
                                 "glass","metal","paper","plastic","shoes","trash"])
    parser.add_argument("--calibrate", type=str, default=None,
                        help="Path to garbage image dir for calibration")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    clf = GarbageOrNotClassifier(
        checkpoint_path=args.checkpoint,
        garbage_class_names=args.classes,
    )

    if args.calibrate:
        clf.calibrate_features(args.calibrate)

    if args.image:
        results = clf.classify_batch(args.image)
        if args.json:
            print(json.dumps([{
                "image": img,
                "is_garbage": r["is_garbage"],
                "garbage_class": r["garbage_class"],
                "score": r["score"],
                "confidence": r["confidence"],
                "top_garbage": [(c, round(p,4)) for c,p in r["top_garbage"]],
                "top_imagenet": [(c, round(p,4)) for c,p in r["top_imagenet"]],
            } for img, r in zip(args.image, results)], indent=2))
        else:
            for img, r in zip(args.image, results):
                print(f"\n📷 {img}")
                print(clf.describe(r))
    else:
        print("\n  Usage examples:")
        print("    # Calibrate on garbage dataset:")
        print("    python garbage_vs_nongarbage.py --calibrate <garbage_dir>")
        print()
        print("    # Classify images:")
        print("    python garbage_vs_nongarbage.py photo1.jpg photo2.jpg")
        print()
        print("    # JSON output:")
        print("    python garbage_vs_nongarbage.py --json photo.jpg")
