"""Second Life AI — out-of-domain guard (garbage vs non-garbage).

A serve-time port of model/"Swin B Transformer"/garbage_vs_nongarbage.py,
kept self-contained in website/ so the deployed app has no dependency on the training
repo. Answers "is this image trash at all, or some random non-waste object (a cat, a
landscape)?" by fusing five out-of-distribution signals over the Swin-B's feature space:

  1. Mahalanobis distance to per-class Gaussians (Lee et al. 2018)   — feature space
  2. Prototype nearest-neighbour cosine distance                      — feature space
  3. Energy score (Liu et al. 2020)                                   — logit space
  4. Max-softmax probability (Hendrycks & Gimpel 2017)               — logit space
  5. Dual-model: a parallel ImageNet Swin-B's confidence             — cross-model

The two feature-space signals are the strongest but need calibration: a one-time pass
over real garbage images fits the Gaussians and prototype bank. That calibration is
cached to an .npz (features + labels); this module fits the detectors from that cache
and never needs the garbage dataset at serve time. Uncalibrated, it auto-falls back to
the three logit/cross-model signals (weaker, but still functional), mirroring the
original's behaviour.

The guard reuses the already-loaded garbage Swin-B for features and lazily loads the
ImageNet Swin-B on first use, so enabling it costs one extra ~87M model in memory.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models as tvm

# The Swin-B's ten training classes, in label order (sorted os.listdir), matching
# app.py's SWIN_CLASSES. Calibration labels are indexed against this list.
SWIN_CLASSES = ["battery", "biological", "cardboard", "clothes", "glass",
                "metal", "paper", "plastic", "shoes", "trash"]


# ─── Feature extraction over a torchvision Swin module ────────────────────────

def swin_forward(swin, x):
    """Return (logits, penultimate 1024-d features) for a torchvision swin_b.

    Mirrors garbage_vs_nongarbage.py's SwinB_Garbage.forward: features → norm →
    NHWC->NCHW → avgpool → flatten → head. Works for both the garbage model (10-class
    head) and the stock ImageNet model (1000-class head)."""
    f = swin.features(x)
    f = swin.norm(f)
    f = f.permute(0, 3, 1, 2)
    f = swin.avgpool(f)
    feats = torch.flatten(f, 1)
    logits = swin.head(feats)
    return logits, feats


# ─── OOD score functions (logit space) ───────────────────────────────────────

def energy_score(logits: torch.Tensor, T: float = 1.0) -> np.ndarray:
    """Energy-based OOD score. Higher = more out-of-distribution."""
    return (-T * torch.logsumexp(logits / T, dim=-1)).detach().cpu().numpy()


def msp_score(logits: torch.Tensor) -> np.ndarray:
    """Maximum softmax probability. Lower = more out-of-distribution."""
    return F.softmax(logits, dim=-1).max(dim=-1).values.detach().cpu().numpy()


def predictive_entropy(logits: torch.Tensor) -> np.ndarray:
    probs = F.softmax(logits, dim=-1)
    return -(probs * torch.log(probs + 1e-10)).sum(dim=-1).detach().cpu().numpy()


# ─── Mahalanobis detector (Lee et al. 2018) ──────────────────────────────────

class MahalanobisDetector:
    """Class-conditional Gaussian OOD detector in feature space."""

    def __init__(self):
        self.means: Dict[int, np.ndarray] = {}
        self.precisions: Dict[int, np.ndarray] = {}
        self.feature_dim: int = 0
        self.threshold: Optional[float] = None

    def fit(self, features: np.ndarray, labels: np.ndarray, shrinkage: float = 0.01):
        self.feature_dim = features.shape[1]
        for lbl in np.unique(labels):
            class_feats = features[labels == lbl]
            mean = class_feats.mean(axis=0)
            cov = np.cov(class_feats, rowvar=False)
            cov = (1 - shrinkage) * cov + shrinkage * np.eye(self.feature_dim)
            cov += 1e-6 * np.eye(self.feature_dim)
            self.means[int(lbl)] = mean
            self.precisions[int(lbl)] = np.linalg.inv(cov)

        dists = []
        for lbl in np.unique(labels):
            class_feats = features[labels == lbl]
            mean, prec = self.means[int(lbl)], self.precisions[int(lbl)]
            for feat in class_feats:
                diff = feat - mean
                dists.append(diff @ prec @ diff)
        self.threshold = float(np.percentile(dists, 95))
        return self

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        """Minimum Mahalanobis distance to any class centroid. Lower = in-distribution."""
        min_dists = np.full(features.shape[0], np.inf)
        for lbl, mean in self.means.items():
            diff = features - mean
            dists = np.einsum('nd,de,ne->n', diff, self.precisions[lbl], diff)
            min_dists = np.minimum(min_dists, dists)
        return min_dists


# ─── Prototype bank (nearest-neighbour OOD) ──────────────────────────────────

class FeaturePrototypeBank:
    def __init__(self):
        self.prototypes: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None

    def build(self, features: np.ndarray, labels: np.ndarray, n_per_class: int = 50):
        protos = []
        for lbl in np.unique(labels):
            class_feats = features[labels == lbl]
            n = min(n_per_class, len(class_feats))
            idx = np.random.choice(len(class_feats), n, replace=False)
            protos.append(class_feats[idx])
        self.prototypes = np.vstack(protos)
        self.prototypes /= (np.linalg.norm(self.prototypes, axis=1, keepdims=True) + 1e-10)

        dists = []
        for i in range(0, len(self.prototypes), 50):
            batch = self.prototypes[i:i + 50]
            sim = batch @ self.prototypes.T
            for j in range(len(batch)):
                row = sim[j].copy()
                row[np.argmax(row)] = -np.inf  # exclude self-match
                dists.append(1.0 - row.max())
        self.threshold = float(np.percentile(dists, 95))
        return self

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        """Cosine distance to nearest prototype. Higher = more OOD."""
        fn = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)
        return 1.0 - (fn @ self.prototypes.T).max(axis=1)


# ─── ImageNet class names (for the dual-model signal's top-3 readout) ─────────

def _load_imagenet_class_names() -> List[str]:
    cache = Path(__file__).resolve().parent.parent / "model" / \
        "Swin B Transformer" / "imagenet_classes.txt"
    if cache.exists():
        return [ln.strip() for ln in cache.read_text().splitlines() if ln.strip()]
    return [str(i) for i in range(1000)]  # fall back to indices


# ─── The guard ───────────────────────────────────────────────────────────────

class OODGuard:
    """Multi-signal garbage vs non-garbage guard, fusion + thresholds ported verbatim
    from garbage_vs_nongarbage.py's GarbageOrNotClassifier."""

    def __init__(self, garbage_net, device: torch.device,
                 class_names: List[str] = SWIN_CLASSES):
        self.garbage_swin = garbage_net.swin  # reuse the already-loaded Swin-B
        self.device = device
        self.class_names = list(class_names)

        self.imagenet_swin = None                 # lazy — loaded on first classify()
        self.imagenet_class_names: List[str] = []

        self.mahalanobis: Optional[MahalanobisDetector] = None
        self.prototype: Optional[FeaturePrototypeBank] = None
        self._energy_threshold: Optional[float] = None
        self._msp_threshold: Optional[float] = None
        self.calibrated = False
        self.calib_info: Dict = {}

        # Fusion parameters — identical to the original.
        self.signal_weights = {"mahalanobis": 0.25, "prototype": 0.15,
                               "energy": 0.25, "msp": 0.20, "imagenet": 0.15}
        self.decision_threshold = 0.50
        self.feature_space_override = 0.85

    # ── calibration cache ────────────────────────────────────────────────────

    def load_cache(self, npz_path: Path) -> bool:
        """Fit the feature-space detectors from a precomputed .npz (features + labels).

        Energy/MSP thresholds are recomputed by applying the Swin head to the cached
        features — no images or dataset needed. Returns True if calibrated."""
        if not Path(npz_path).exists():
            print(f"[OOD guard] no calibration cache at {Path(npz_path).name} — "
                  f"running UNCALIBRATED (energy+msp+imagenet only)")
            return False
        data = np.load(npz_path, allow_pickle=True)
        features, labels = data["features"], data["labels"]

        self.mahalanobis = MahalanobisDetector().fit(features, labels)
        self.prototype = FeaturePrototypeBank().build(features, labels)

        feats_t = torch.from_numpy(features).float().to(self.device)
        with torch.no_grad():
            logits = self.garbage_swin.head(feats_t)
        self._energy_threshold = float(np.percentile(energy_score(logits), 95))
        self._msp_threshold = float(np.percentile(msp_score(logits), 5))

        self.calibrated = True
        self.calib_info = {
            "n_features": int(features.shape[0]), "feature_dim": int(features.shape[1]),
            "classes_present": sorted({int(x) for x in np.unique(labels)}),
            "mahalanobis_threshold": round(float(self.mahalanobis.threshold), 2),
            "prototype_threshold": round(float(self.prototype.threshold), 4),
            "energy_threshold": round(self._energy_threshold, 3),
            "msp_threshold": round(self._msp_threshold, 4),
        }
        print(f"[OOD guard] calibrated from {Path(npz_path).name}: "
              f"{self.calib_info['n_features']} feats, "
              f"maha_thr={self.calib_info['mahalanobis_threshold']}, "
              f"proto_thr={self.calib_info['prototype_threshold']}")
        return True

    # ── dual model ───────────────────────────────────────────────────────────

    def _ensure_imagenet(self):
        if self.imagenet_swin is None:
            print("[OOD guard] loading ImageNet Swin-B for the dual-model signal…")
            m = tvm.swin_b(weights=tvm.Swin_B_Weights.IMAGENET1K_V1)
            self.imagenet_swin = m.to(self.device).eval()
            self.imagenet_class_names = _load_imagenet_class_names()

    # ── classify ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def classify(self, img_tensor: torch.Tensor) -> Dict:
        """Verdict for one preprocessed image tensor (1x3x224x224). Fusion logic and
        every threshold are ported verbatim from GarbageOrNotClassifier.classify."""
        self._ensure_imagenet()
        img_tensor = img_tensor.to(self.device)

        garbage_logits, garbage_features = swin_forward(self.garbage_swin, img_tensor)
        imagenet_logits, _ = swin_forward(self.imagenet_swin, img_tensor)

        garbage_probs = F.softmax(garbage_logits, dim=-1)
        top_gc_probs, top_gc_idx = torch.topk(garbage_probs, k=3, dim=-1)
        top_garbage = [(self.class_names[i.item()], round(p.item(), 4))
                       for i, p in zip(top_gc_idx[0], top_gc_probs[0])]

        imagenet_probs = F.softmax(imagenet_logits, dim=-1)
        top_in_probs, top_in_idx = torch.topk(imagenet_probs, k=3, dim=-1)
        names = self.imagenet_class_names
        top_imagenet = [(names[i.item()] if i.item() < len(names) else str(i.item()),
                         round(p.item(), 4))
                        for i, p in zip(top_in_idx[0], top_in_probs[0])]

        raw_energy = float(energy_score(garbage_logits)[0])
        raw_msp = float(msp_score(garbage_logits)[0])
        raw_imagenet_msp = float(msp_score(imagenet_logits)[0])
        feat_np = garbage_features.cpu().numpy()

        # Signal 1: Mahalanobis
        if self.mahalanobis is not None:
            raw_mahal = float(self.mahalanobis.score_samples(feat_np)[0])
            thr = self.mahalanobis.threshold
            score_mahal = float(1.0 / (1.0 + np.exp(-3.0 * (raw_mahal / max(thr, 0.01) - 1.0))))
        else:
            raw_mahal, score_mahal = None, 0.5

        # Signal 2: Prototype
        if self.prototype is not None:
            raw_proto = float(self.prototype.score_samples(feat_np)[0])
            thr = self.prototype.threshold
            score_proto = float(1.0 / (1.0 + np.exp(-5.0 * (raw_proto / max(thr, 0.001) - 1.0))))
        else:
            raw_proto, score_proto = None, 0.5

        # Signal 3: Energy
        if self._energy_threshold is not None:
            score_energy = float(1.0 / (1.0 + np.exp(-(raw_energy - self._energy_threshold))))
        else:
            score_energy = float(1.0 / (1.0 + np.exp(-(raw_energy + 3.0))))

        # Signal 4: MSP (low MSP → OOD)
        if self._msp_threshold is not None:
            score_msp = float(1.0 / (1.0 + np.exp(-5.0 * (self._msp_threshold - raw_msp))))
        else:
            score_msp = float(1.0 / (1.0 + np.exp(-5.0 * (0.30 - raw_msp))))

        # Signal 5: ImageNet confidence
        garbage_entropy = float(predictive_entropy(garbage_logits)[0])
        imagenet_entropy = float(predictive_entropy(imagenet_logits)[0])
        entropy_ratio = imagenet_entropy / max(garbage_entropy, 0.01)
        score_imagenet = float(1.0 / (1.0 + np.exp(-3.0 * (raw_imagenet_msp - 0.5))))

        individual = {"mahalanobis": score_mahal, "prototype": score_proto,
                      "energy": score_energy, "msp": score_msp, "imagenet": score_imagenet}

        # Fuse — feature-space weights drop to 0 when uncalibrated.
        weights = dict(self.signal_weights)
        if self.mahalanobis is None:
            weights = {"mahalanobis": 0, "prototype": 0,
                       "energy": 0.35, "msp": 0.35, "imagenet": 0.30}
        total_w = sum(weights.values())
        fused = sum(individual[k] * weights[k] / total_w for k in weights)

        # Decision with quorum overrides — verbatim from the original.
        method = "fused"
        feat_ood_quorum = (self.mahalanobis is not None
                           and score_mahal > self.feature_space_override
                           and score_proto > self.feature_space_override)
        feat_id_quorum = (self.mahalanobis is not None
                          and score_mahal < 0.30 and score_proto < 0.30)

        if feat_id_quorum:
            is_garbage, method = True, "feature_space_id_quorum"
        elif feat_ood_quorum:
            is_garbage, method = False, "feature_space_ood_quorum"
        elif raw_msp > 0.95:
            is_garbage, method = True, "very_high_confidence"
        elif raw_imagenet_msp > 0.80 and raw_msp < 0.40:
            is_garbage, method = False, "imagenet_override"
        elif 0.48 <= fused <= 0.55 and raw_msp > 0.70:
            is_garbage, method = True, "borderline_garbage_model"
        else:
            is_garbage = fused <= self.decision_threshold

        return {
            "is_garbage": bool(is_garbage),
            "garbage_class": top_garbage[0][0] if is_garbage else None,
            "score": round(float(fused), 4),
            "confidence": round(raw_msp, 4),
            "energy": round(raw_energy, 4),
            "mahalanobis": round(raw_mahal, 2) if raw_mahal is not None else None,
            "prototype_dist": round(raw_proto, 4) if raw_proto is not None else None,
            "entropy_ratio": round(float(entropy_ratio), 4),
            "individual_scores": {k: round(v, 4) for k, v in individual.items()},
            "top_garbage": top_garbage,
            "top_imagenet": top_imagenet,
            "method": method,
            "calibrated": self.calibrated,
        }
