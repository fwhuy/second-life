"""
Demo: Garbage vs Non-Garbage Classification
============================================
Calibrates the multi-signal OOD detector on real garbage images,
then tests on garbage vs non-garbage images.

Usage:
    python demo_nongarbage.py
    python demo_nongarbage.py --image my_photo.jpg
"""

import os
import sys
import json
import random
import urllib.request
import argparse
from pathlib import Path
from io import BytesIO

from PIL import Image
from garbage_vs_nongarbage import GarbageOrNotClassifier

CACHE_DIR = Path(__file__).parent / "test_images"


def download_picsum_images(n: int = 10) -> list:
    """Download random everyday photos as non-garbage examples."""
    CACHE_DIR.mkdir(exist_ok=True)
    paths = []
    for i in range(n):
        fname = CACHE_DIR / f"picsum_{i}.jpg"
        if not fname.exists():
            try:
                url = f"https://picsum.photos/seed/{i+200}/640/480"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = resp.read()
                img = Image.open(BytesIO(data)).convert("RGB")
                img.save(str(fname), "JPEG", quality=90)
            except Exception as e:
                print(f"    ⚠ picsum {i}: {e}")
                continue
        if fname.exists():
            paths.append({"path": str(fname), "label": "nongarbage", "name": f"picsum_{i}"})
    return paths


def find_garbage_images(data_root: str, n: int = 20) -> list:
    """Sample garbage images from class directories."""
    paths = []
    class_dirs = sorted([d for d in Path(data_root).iterdir() if d.is_dir()])
    n_per = max(1, n // max(1, len(class_dirs)))
    for cls_dir in class_dirs:
        imgs = [p for p in cls_dir.iterdir()
                if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')]
        for img_path in imgs[:n_per]:
            paths.append({
                "path": str(img_path),
                "label": "garbage",
                "name": f"{cls_dir.name}/{img_path.name}",
            })
    random.shuffle(paths)
    return paths[:n]


def main():
    parser = argparse.ArgumentParser(description="Demo: Garbage vs Non-Garbage")
    parser.add_argument("--image", nargs="*", help="Additional images to test")
    parser.add_argument("--checkpoint", default="best_swin_b.pt")
    parser.add_argument("--garbage-dir",
                        default="../archive/Garbage classification/Garbage classification")
    parser.add_argument("--n-samples", type=int, default=12)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  GARBAGE vs NON-GARBAGE — Multi-Signal OOD Demo")
    print("=" * 60)

    # ── Load classifier ──
    print("\n  Step 1: Loading models...")
    clf = GarbageOrNotClassifier(
        checkpoint_path=args.checkpoint,
        garbage_class_names=[
            "battery", "biological", "cardboard", "clothes",
            "glass", "metal", "paper", "plastic", "shoes", "trash"
        ],
    )

    # ── Calibrate on real garbage images ──
    garbage_dir = os.path.abspath(args.garbage_dir)
    if os.path.isdir(garbage_dir):
        print(f"\n  Step 2: Calibrating OOD detectors on garbage images...")
        cal = clf.calibrate_features(garbage_dir, n_samples=400)
    else:
        print(f"\n  ⚠ Garbage dir not found: {garbage_dir}")
        print(f"    Running WITHOUT calibration (less accurate)")

    # ── Collect test images ──
    print(f"\n  Step 3: Collecting test images...")
    images = []

    # Non-garbage: random photos
    nongarbage_imgs = download_picsum_images(args.n_samples)
    images.extend(nongarbage_imgs)
    print(f"    Non-garbage (picsum): {len(nongarbage_imgs)}")

    # Garbage: from dataset (use different images than calibration)
    if os.path.isdir(garbage_dir):
        garbage_imgs = find_garbage_images(garbage_dir, args.n_samples)
        # Filter out images that might have been used in calibration
        images.extend(garbage_imgs)
        print(f"    Garbage (dataset):    {len(garbage_imgs)}")

    # User images
    if args.image:
        for img_path in args.image:
            images.append({
                "path": img_path, "label": "unknown",
                "name": os.path.basename(img_path),
            })

    if not images:
        print("\n  No images to test!")
        return

    # ── Classify ──
    print(f"\n  Step 4: Classifying {len(images)} images...\n")

    results = []
    correct = 0
    n_known = 0

    for info in images:
        result = clf.classify(info["path"])
        result["name"] = info["name"]
        result["true_label"] = info["label"]
        results.append(result)

        verdict = "🗑️  GARBAGE" if result["is_garbage"] else "✅ NON-GARBAGE"
        true = result["true_label"]

        if true == "garbage":
            match = "✓" if result["is_garbage"] else "✗ MISS"
            n_known += 1
        elif true == "nongarbage":
            match = "✓" if not result["is_garbage"] else "✗ FALSE"
            n_known += 1
        else:
            match = "·"

        if match.startswith("✓"):
            correct += 1

        gc = result.get("garbage_class", "—") or "—"
        print(f"  {verdict} | {match} | score={result['score']:.3f} | "
              f"conf={result['confidence']:.3f} | {gc:<12} | {info['name']}")

    # ── Summary ──
    print(f"\n{'─' * 60}")
    print(f"  SUMMARY")
    print(f"{'─' * 60}")

    if n_known > 0:
        acc = 100 * correct / n_known
        print(f"  Overall accuracy: {correct}/{n_known} ({acc:.1f}%)")

    garbage_results = [r for r in results if r["true_label"] == "garbage"]
    nongarbage_results = [r for r in results if r["true_label"] == "nongarbage"]

    if garbage_results:
        detected = sum(1 for r in garbage_results if r["is_garbage"])
        print(f"  Garbage recall:   {detected}/{len(garbage_results)} "
              f"({100*detected/max(1,len(garbage_results)):.1f}%)")
        avg_score = sum(r["score"] for r in garbage_results) / len(garbage_results)
        print(f"  Garbage avg OOD score: {avg_score:.4f} (lower = more garbage-like)")

    if nongarbage_results:
        rejected = sum(1 for r in nongarbage_results if not r["is_garbage"])
        print(f"  Non-garbage rejected: {rejected}/{len(nongarbage_results)} "
              f"({100*rejected/max(1,len(nongarbage_results)):.1f}%)")
        avg_score = sum(r["score"] for r in nongarbage_results) / len(nongarbage_results)
        print(f"  Non-garbage avg OOD score: {avg_score:.4f} (higher = more OOD)")

    # ── Individual signal breakdown ──
    print(f"\n  PER-SIGNAL SCORES (>0.5 → non-garbage):")
    print(f"  {'Image':<25} | {'maha':>6} | {'proto':>6} | {'energy':>6} | "
          f"{'msp':>6} | {'imagenet':>6} | → {'fused':>6}")
    print(f"  {'─'*25}-+-{'─'*6}-+-{'─'*6}-+-{'─'*6}-+-{'─'*6}-+-{'─'*6}-+-{'─'*6}")
    for r in results:
        s = r["individual_scores"]
        print(f"  {r['name']:<25} | {s['mahalanobis']:>6.3f} | {s['prototype']:>6.3f} | "
              f"{s['energy']:>6.3f} | {s['msp']:>6.3f} | {s['imagenet']:>6.3f} | "
              f"→ {r['score']:>6.3f}")

    # ── Save ──
    out_path = args.json_out or str(Path(__file__).parent / "nongarbage_results.json")
    with open(out_path, "w") as f:
        json.dump([{
            "name": r["name"],
            "true_label": r["true_label"],
            "is_garbage": r["is_garbage"],
            "garbage_class": r["garbage_class"],
            "score": r["score"],
            "confidence": r["confidence"],
            "energy": r["energy"],
            "mahalanobis": r["mahalanobis"],
            "individual_scores": r["individual_scores"],
            "top_garbage": [(c, round(p,4)) for c,p in r["top_garbage"]],
            "top_imagenet": [(c, round(p,4)) for c,p in r["top_imagenet"]],
            "method": r["method"],
        } for r in results], f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    # ── How to improve ──
    print(f"\n{'=' * 60}")
    print(f"  HOW TO FURTHER IMPROVE")
    print(f"{'=' * 60}")
    print(f"  1. Add non-garbage images as an 11th class and fine-tune")
    print(f"  2. Use the --calibrate flag to adjust thresholds")
    print(f"  3. Adjust signal_weights dict in garbage_vs_nongarbage.py")
    print(f"  4. Lower decision_threshold for more aggressive rejection")
    print(f"  5. Collect more diverse garbage training examples")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
