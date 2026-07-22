"""Size batch/workers to the actual machine, and scale LR to match.

    python scripts/autotune.py --config configs/unified_convnextv2_tiny_224.yaml
    python scripts/autotune.py --config configs/... --write configs/..._tuned.yaml

Finds the largest batch size that survives a real forward+backward+step (not a VRAM
estimate — activation memory depends on the architecture), then reports the settings.

WHY THE LR CHANGES: a bigger batch means fewer, less noisy gradient steps per epoch.
Keeping the original LR at 4x the batch size systematically *underfits* — this is the
usual reason "I filled the GPU and accuracy dropped". The linear scaling rule
(Goyal et al. 2017) multiplies LR by the same factor as the batch. It is a good default,
not a law: past roughly 4x the tuned batch size the rule starts to break down, and
warmup matters more.

This does not edit your config unless you pass --write.
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import resolve_workers  # noqa: E402
from src.model import build_model  # noqa: E402
from src.utils import NUM_CLASSES, configure_performance, get_device  # noqa: E402

CANDIDATES = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
HEADROOM = 0.90  # leave room for eval batches, EMA, and allocator fragmentation


def fits(cfg: dict, batch: int, device, amp_dtype) -> bool:
    """One real training step. OOM here is the answer, not an error."""
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build_model(cfg, pretrained=False).to(device)
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        size = cfg["img_size"]
        imgs = torch.randn(batch, 3, size, size, device=device)
        if device.type == "cuda":
            imgs = imgs.to(memory_format=torch.channels_last)
        targets = torch.randint(0, NUM_CLASSES, (batch,), device=device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            loss = torch.nn.functional.cross_entropy(model(imgs), targets)
        loss.backward()
        opt.step()
        peak = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0
        total = (torch.cuda.get_device_properties(0).total_memory / 1024**3
                 if device.type == "cuda" else 0)
        del model, opt, imgs, targets, loss
        return peak <= total * HEADROOM, peak
    except torch.cuda.OutOfMemoryError:
        return False, 0.0
    finally:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--write", help="write a tuned copy of the config to this path")
    ap.add_argument("--max-scale", type=float, default=4.0,
                    help="refuse to scale batch beyond this multiple of the config's value")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = get_device()
    if device.type != "cuda":
        print(f"No CUDA device ({device.type}) — autotune only makes sense on a GPU.")
        return 1
    amp_dtype = configure_performance(cfg, device)

    base_batch, base_lr_head = cfg["batch_size"], cfg["lr_head"]
    base_lr_backbone = cfg["lr_backbone"]
    print(f"\nconfig: {Path(args.config).name}  model={cfg['model']}  img={cfg['img_size']}")
    print(f"baseline: batch={base_batch} lr_head={base_lr_head} lr_backbone={base_lr_backbone}")
    print("\nProbing batch sizes with a real forward+backward+step:")

    best, best_peak = 0, 0.0
    for batch in CANDIDATES:
        ok, peak = fits(cfg, batch, device, amp_dtype)
        print(f"  batch {batch:5d}  {'OK  ' if ok else 'OOM '} peak {peak:5.1f}GB")
        if not ok:
            break
        best, best_peak = batch, peak
    if not best:
        print("\nEven the smallest batch failed — something else is wrong.")
        return 1

    cap = int(base_batch * args.max_scale)
    chosen = min(best, cap)
    capped = chosen < best
    scale = chosen / base_batch
    workers = resolve_workers({**cfg, "num_workers": "auto"})

    print(f"\nLargest batch that fits: {best} (peak {best_peak:.1f}GB)")
    if capped:
        print(f"Capping at {chosen} = {args.max_scale:g}x the tuned baseline. Beyond that the "
              f"linear LR scaling rule degrades and you trade accuracy for speed.")
    print(f"\nRecommended:")
    print(f"  batch_size:   {chosen}   ({scale:.2g}x baseline)")
    print(f"  num_workers:  {workers}")
    print(f"  lr_head:      {base_lr_head * scale:.3g}   (linear scaling: {scale:.2g}x)")
    print(f"  lr_backbone:  {base_lr_backbone * scale:.3g}")
    print(f"  warmup_epochs: {max(cfg.get('warmup_epochs', 3), math.ceil(3 * math.sqrt(scale)))}"
          f"   (larger batches need longer warmup)")

    if args.write:
        tuned = {**cfg,
                 "run_name": f"{cfg['run_name']}_bs{chosen}",
                 "batch_size": chosen,
                 "num_workers": workers,
                 "lr_head": float(f"{base_lr_head * scale:.3g}"),
                 "lr_backbone": float(f"{base_lr_backbone * scale:.3g}"),
                 "warmup_epochs": max(cfg.get("warmup_epochs", 3),
                                      math.ceil(3 * math.sqrt(scale)))}
        header = (f"# Autotuned from {Path(args.config).name} for "
                  f"{torch.cuda.get_device_properties(0).name}.\n"
                  f"# batch {base_batch}->{chosen}, LRs scaled {scale:.2g}x "
                  f"(linear scaling rule).\n"
                  f"# Throughput change, NOT a free accuracy win — compare macro-F1 against "
                  f"the baseline run before adopting.\n")
        Path(args.write).write_text(header + yaml.safe_dump(tuned, sort_keys=False))
        print(f"\nWrote {args.write}")
        print("Treat it as an experiment: run it, compare macro-F1 to the baseline, and keep "
              "whichever wins.")
    else:
        print("\nRe-run with --write <path> to save a tuned config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
