"""Near-duplicate detection → group ids for group-aware splitting.

TrashNet contains multiple photos of the same physical object at different
angles; a random split puts the same crushed can in both train and test.
Stage 1 groups near-identical shots by perceptual hash. Stage 2 (optional but
default) escalates to backbone-embedding cosine similarity to catch
same-object-different-angle pairs that phash misses.

Output: <splits_dir>/groups.csv with columns path,label,group.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .utils import CLASSES, REPO_ROOT, seed_everything

PHASH_HAMMING_MAX = 6      # <= this many differing bits of a 64-bit phash → same group
EMBED_COSINE_MIN = 0.96    # >= this cosine similarity → same group
BLOCK = 2048               # rows per similarity block; caps peak memory at ~0.1GB


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def scan_images(data_dir: Path) -> pd.DataFrame:
    rows = []
    for cls in CLASSES:
        cls_dir = data_dir / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {cls_dir}")
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                # as_posix keeps split CSVs identical across Windows/macOS/Linux
                rows.append({"path": p.relative_to(REPO_ROOT).as_posix(), "label": cls})
    return pd.DataFrame(rows)


def phash_bits(df: pd.DataFrame) -> np.ndarray:
    import imagehash

    bits = np.zeros((len(df), 64), dtype=np.uint8)
    for i, rel in enumerate(df["path"]):
        with Image.open(REPO_ROOT / rel) as im:
            bits[i] = np.asarray(imagehash.phash(im.convert("RGB")).hash).reshape(-1)
    return bits


def merge_by_phash(uf: UnionFind, bits: np.ndarray, block: int = BLOCK) -> int:
    """Pairwise hamming distance via two matmuls, one row-block at a time.

    The full (N,N) int32 distance matrix is 0.5GB at N=11k and there are two of
    them live, so the unified corpus needs blocking; the small TrashNet corpus
    runs identically in a single block.
    """
    a = bits.astype(np.int32)
    inv = 1 - a
    pairs = 0
    for start in range(0, len(a), block):
        stop = min(start + block, len(a))
        dist = a[start:stop] @ inv.T + inv[start:stop] @ a.T
        ii, jj = np.nonzero(dist <= PHASH_HAMMING_MAX)
        ii = ii + start
        upper = ii < jj  # upper triangle only: skips self-pairs and duplicates
        for i, j in zip(ii[upper], jj[upper]):
            uf.union(int(i), int(j))
        pairs += int(upper.sum())
    return pairs


def compute_embeddings(df: pd.DataFrame, batch_size: int = 64) -> np.ndarray:
    import timm
    import torch
    from timm.data import create_transform, resolve_model_data_config

    from .utils import get_device

    device = get_device()
    model = timm.create_model("resnet18.a1_in1k", pretrained=True, num_classes=0)
    model.eval().to(device)
    tf = create_transform(**resolve_model_data_config(model), is_training=False)

    feats = []
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            chunk = df["path"].iloc[start : start + batch_size]
            imgs = torch.stack(
                [tf(Image.open(REPO_ROOT / p).convert("RGB")) for p in chunk]
            ).to(device)
            f = model(imgs)
            feats.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
    return np.concatenate(feats)


def merge_by_embedding(uf: UnionFind, feats: np.ndarray, labels: np.ndarray,
                       block: int = BLOCK) -> int:
    pairs = 0
    for start in range(0, len(feats), block):
        stop = min(start + block, len(feats))
        sim = feats[start:stop] @ feats.T
        # only merge within the same class: cross-class "duplicates" are just
        # visually similar objects, and merging them would wreck stratification
        same_class = labels[start:stop, None] == labels[None, :]
        ii, jj = np.nonzero((sim >= EMBED_COSINE_MIN) & same_class)
        ii = ii + start
        upper = ii < jj
        for i, j in zip(ii[upper], jj[upper]):
            uf.union(int(i), int(j))
        pairs += int(upper.sum())
    return pairs


def group_frame(df: pd.DataFrame, use_embeddings: bool = True) -> pd.DataFrame:
    """Add a `group` column to an inventory frame (path, label) already in hand.

    Split out of build_groups so the unified corpus — which needs filtering and a
    `source` column before grouping — can reuse the identical two-stage logic.
    """
    df = df.reset_index(drop=True)
    uf = UnionFind(len(df))
    n_hash_pairs = merge_by_phash(uf, phash_bits(df))
    print(f"phash stage: {n_hash_pairs} near-identical pairs (hamming <= {PHASH_HAMMING_MAX})")

    if use_embeddings:
        feats = compute_embeddings(df)
        n_emb_pairs = merge_by_embedding(uf, feats, df["label"].to_numpy())
        print(f"embedding stage: {n_emb_pairs} same-object pairs (cosine >= {EMBED_COSINE_MIN})")

    roots = np.array([uf.find(i) for i in range(len(df))])
    _, group_ids = np.unique(roots, return_inverse=True)
    df["group"] = group_ids

    sizes = df.groupby("group").size()
    print(
        f"{df['group'].nunique()} groups for {len(df)} images "
        f"({(sizes > 1).sum()} multi-image groups, largest = {sizes.max()})"
    )
    if sizes.max() > 40:
        print("WARNING: a very large duplicate group exists — inspect it before trusting splits")
    return df


def build_groups(data_dir: Path, splits_dir: Path, use_embeddings: bool = True) -> pd.DataFrame:
    df = scan_images(data_dir)
    print(f"Scanned {len(df)} images in {data_dir}")
    df = group_frame(df, use_embeddings=use_embeddings)

    splits_dir.mkdir(parents=True, exist_ok=True)
    out = splits_dir / "groups.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--no-embeddings", action="store_true",
                    help="skip the backbone-embedding escalation stage (phash only)")
    args = ap.parse_args()
    seed_everything(42)
    build_groups(REPO_ROOT / args.data_dir, REPO_ROOT / args.splits_dir,
                 use_embeddings=not args.no_embeddings)


if __name__ == "__main__":
    main()
