"""Build the out-of-distribution reference bank for the closed-set guard.

The site flags a photo as "not one of my six classes" when the image's
penultimate (pre-logits) embedding is far from every in-distribution cluster in
feature space. That test needs a bank of in-distribution embeddings; this script
precomputes it once so the running site needs only the small bank file, not the
2000+ training images.

  python build_ood_bank.py [--checkpoint PATH] [--root MODEL_REPO] [--out ood_bank.npz]

Re-run this whenever the served checkpoint changes (the embeddings are
model-specific). It prints the in-distribution distance distribution so the
threshold in app.py (FEATURE_OOD_THRESHOLD) can be sanity-checked.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=str(HERE / "checkpoints/baseline_resnet50/fold0/best.pth"))
    ap.add_argument("--root", default=str(HERE.parent / "model"),
                    help="model repo root holding src/, data/splits/folds.csv and data/raw/")
    ap.add_argument("--out", default=str(HERE / "ood_bank.npz"))
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    root = Path(args.root)
    sys.path.insert(0, str(root))
    from src.data import build_eval_transform, TrashDataset  # noqa: E402
    from src.model import build_model  # noqa: E402
    from src.utils import train_val_from_folds  # noqa: E402

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = state["cfg"]
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(state["ema"] if state.get("ema") is not None else state["model"])
    model.eval()
    tf = build_eval_transform(cfg["img_size"])

    @torch.no_grad()
    def feat(x):
        f = model.forward_head(model.forward_features(x), pre_logits=True)
        return torch.nn.functional.normalize(f, dim=1)

    # Use the whole trainval set (test stays quarantined) as the reference bank.
    folds = pd.read_csv(root / "data" / "splits" / "folds.csv")
    ds = TrashDataset(folds, tf)
    feats = []
    with torch.no_grad():
        for i in range(len(ds)):
            feats.append(feat(ds[i][0].unsqueeze(0))[0])
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(ds)} embedded", flush=True)
    F = torch.stack(feats)  # [N, D], L2-normalized

    np.savez_compressed(
        args.out,
        features=F.numpy().astype(np.float16),
        k=args.k, model=cfg["model"], img_size=cfg["img_size"],
    )
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"\nsaved {args.out}  ({F.shape[0]} x {F.shape[1]}, {size_mb:.1f} MB)")

    # in-distribution distance distribution (leave-one-out), for threshold sanity
    S = F @ F.t()
    S.fill_diagonal_(-1)
    loo = (1 - S.topk(args.k, dim=1).values.mean(1)).numpy()
    qs = [50, 90, 95, 99]
    print("in-dist kNN distance: " + "  ".join(f"p{q}={np.percentile(loo, q):.3f}" for q in qs)
          + f"  max={loo.max():.3f}")
    print("(set FEATURE_OOD_THRESHOLD in app.py safely above this max)")


if __name__ == "__main__":
    main()
