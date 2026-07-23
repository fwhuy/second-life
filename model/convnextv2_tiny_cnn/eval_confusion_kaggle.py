# =============================================================================
# ConvNeXt V2 Tiny — validation analysis pack (KAGGLE)
# -----------------------------------------------------------------------------
# Run this in the SAME Kaggle notebook that trained the model (GPU on, Internet
# on), or add this file plus the training file and `%run` it. It re-imports the
# training file's exact corpus build / perceptual dedup / seed-42
# StratifiedGroupKFold, so the validation fold is IDENTICAL to the run that
# scored 98.30% — then it produces the full analysis pack the paper needs:
#
#   convnext_metrics.json        overall acc, macro-F1, per-class P/R/F1, loss
#   convnext_confusion.json      raw confusion counts (rows=true, cols=pred)
#   convnext_confusion.png       styled confusion matrix (counts)
#   convnext_perclass.png        per-class recall bars
#   convnext_predictions.csv     path,true,pred,confidence,correct,loss (every image)
#   convnext_misclassified.csv   only the errors, worst first
#   convnext_misclassified.png   contact sheet of the worst errors (true -> pred)
#ok 
# IMPORTANT — this is the VALIDATION fold, not an independent test set. ConvNeXt
# was trained with a single seed-42 StratifiedGroupKFold train/val split and the
# best checkpoint was selected on this fold, so 98.30% is a *validation* number.
# Label every artifact "validation" — do not call it "test accuracy". This keeps
# it consistent with the already-submitted poster (98.30% validation).
# =============================================================================
import csv
import glob
import importlib.util as ilu
import json
from pathlib import Path

import numpy as np
import timm
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

OUT = Path("/kaggle/working")

# --- 1. locate & import the training module (edit TRAIN_PY if auto-find misses) ---
CANDIDATES = (
    glob.glob("/kaggle/**/train_and_upload*.py", recursive=True)
    + glob.glob("/kaggle/**/*convnext*train*.py", recursive=True)
    + glob.glob("/kaggle/**/*train*convnext*.py", recursive=True)
)
TRAIN_PY = CANDIDATES[0] if CANDIDATES else "/kaggle/input/PATH/TO/train_and_upload.py"
print("training module:", TRAIN_PY)
spec = ilu.spec_from_file_location("trainmod", TRAIN_PY)
m = ilu.module_from_spec(spec)
spec.loader.exec_module(m)          # defines load_corpus, WasteDataset, eval_tf, MODEL, CLASSES, SEED
CLASSES = m.CLASSES
N = len(CLASSES)

# --- 2. checkpoint: same session writes /kaggle/working; otherwise find it under
#        the attached Kaggle Model / Dataset inputs (any *.pt / *.pth). ---
CKPT = "/kaggle/working/best_convnextv2.pt"
if not Path(CKPT).exists():
    print("mounted inputs:", glob.glob("/kaggle/input/*"))
    cands = (glob.glob("/kaggle/**/*.pt", recursive=True)
             + glob.glob("/kaggle/**/*.pth", recursive=True))
    print("checkpoint candidates:", cands)
    assert cands, ("No .pt/.pth found under /kaggle. The Model input isn't mounted — "
                   "re-add it via 'Add Input', then restart the session.")
    CKPT = next((c for c in cands if "convnext" in c.lower()), cands[0])
print("checkpoint:", CKPT)

# --- 3. rebuild corpus + reproduce the exact seed-42 validation fold ----------
records = m.load_corpus()
labels = np.array([r["label"] for r in records])
groups = np.array([r["group"] for r in records])
tr_idx, va_idx = next(
    StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=m.SEED).split(
        np.zeros(len(records)), labels, groups
    )
)
val_records = [records[i] for i in va_idx]
assert not (set(groups[tr_idx]) & set(groups[va_idx])), "group leak!"
print(f"reproduced val fold: {len(va_idx)} images (group-disjoint from train)")

# --- 4. load model + run inference, keeping per-image logits and paths ---------
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model(m.MODEL, pretrained=False, num_classes=N)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model = model.to(dev).eval()

loader = DataLoader(m.WasteDataset(val_records, m.eval_tf), batch_size=64, num_workers=2)
paths = [r["path"] for r in val_records]
all_logits, all_true = [], []
with torch.no_grad():
    for x, y in loader:
        all_logits.append(model(x.to(dev)).float().cpu())
        all_true.append(y)
logits = torch.cat(all_logits)                    # [n, N]
y_true = torch.cat(all_true).numpy()              # [n]
probs = torch.softmax(logits, dim=1).numpy()      # [n, N]
y_pred = probs.argmax(1)                           # [n]
conf = probs.max(1)                                # [n] winning-class probability
# per-image cross-entropy loss = -log P(true class)
sample_loss = -np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0))

# --- 5. metrics: confusion, per-class precision/recall/F1, macro-F1, accuracy --
conf_mat = np.zeros((N, N), dtype=int)             # conf_mat[true, pred]
for t, p in zip(y_true, y_pred):
    conf_mat[t, p] += 1

per_class = {}
for i, c in enumerate(CLASSES):
    tp = int(conf_mat[i, i])
    fp = int(conf_mat[:, i].sum() - tp)
    fn = int(conf_mat[i, :].sum() - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    per_class[c] = {"precision": precision, "recall": recall, "f1": f1,
                    "support": int(conf_mat[i, :].sum())}

accuracy = float(conf_mat.trace() / max(conf_mat.sum(), 1))
macro_f1 = float(np.mean([per_class[c]["f1"] for c in CLASSES]))
mean_loss = float(sample_loss.mean())

print("\n================ VALIDATION RESULTS (fold 0) ================")
print(f"overall val accuracy = {accuracy*100:.2f}%   (expected ~98.30%)")
print(f"macro-F1            = {macro_f1:.4f}    mean CE loss = {mean_loss:.4f}")
print(f"{'class':<11}{'prec':>8}{'recall':>8}{'f1':>8}{'n':>6}")
for c in CLASSES:
    pc = per_class[c]
    print(f"{c:<11}{pc['precision']*100:>7.1f}%{pc['recall']*100:>7.1f}%"
          f"{pc['f1']*100:>7.1f}%{pc['support']:>6}")
print("\nconfusion matrix (rows=true, cols=pred):")
print("           " + " ".join(f"{c[:5]:>6}" for c in CLASSES))
for i, c in enumerate(CLASSES):
    print(f"  {c:<9} " + " ".join(f"{conf_mat[i, j]:>6d}" for j in range(N)))

json.dump(
    {"split": "validation_fold0", "note": "single seed-42 StratifiedGroupKFold "
     "fold used for model selection; this is validation, not an independent test",
     "overall_accuracy": accuracy, "macro_f1": macro_f1, "mean_loss": mean_loss,
     "classes": CLASSES, "confusion": conf_mat.tolist(), "per_class": per_class},
    open(OUT / "convnext_metrics.json", "w"), indent=2,
)
json.dump(
    {"classes": CLASSES, "confusion": conf_mat.tolist(),
     "per_class_recall": {c: per_class[c]["recall"] for c in CLASSES}},
    open(OUT / "convnext_confusion.json", "w"), indent=2,
)

# --- 6. predictions + misclassified tables ------------------------------------
with open(OUT / "convnext_predictions.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["path", "true", "pred", "confidence", "correct", "loss"])
    for pth, t, p, cf, ls in zip(paths, y_true, y_pred, conf, sample_loss):
        w.writerow([pth, CLASSES[t], CLASSES[p], f"{cf:.4f}",
                    int(t == p), f"{ls:.4f}"])

# errors, worst (highest-loss) first — the actionable-analysis subset
err = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
err.sort(key=lambda i: sample_loss[i], reverse=True)
with open(OUT / "convnext_misclassified.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["path", "true", "pred", "confidence", "loss"])
    for i in err:
        w.writerow([paths[i], CLASSES[y_true[i]], CLASSES[y_pred[i]],
                    f"{conf[i]:.4f}", f"{sample_loss[i]:.4f}"])
print(f"\n{len(err)} misclassified of {len(y_true)} "
      f"({100*len(err)/max(len(y_true),1):.2f}% error). "
      "Most common confusions:")
pair_counts = {}
for i in err:
    key = (CLASSES[y_true[i]], CLASSES[y_pred[i]])
    pair_counts[key] = pair_counts.get(key, 0) + 1
for (t, p), n_ in sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]:
    print(f"  {t:>10} -> {p:<10} {n_}")

# --- 7. styled figures --------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
GREEN, INK, MUTE = "#2C5842", "#20302a", "#5A6560"
cmap = LinearSegmentedColormap.from_list("g", ["#FFFFFF", "#CFE0D6", GREEN])
labs = [c.capitalize() for c in CLASSES]

fig, ax = plt.subplots(figsize=(7.6, 6.9), dpi=200)
vmax = conf_mat.max()
ax.imshow(conf_mat, cmap=cmap, vmin=0, vmax=vmax)
ax.set_xticks(range(N)); ax.set_xticklabels(labs, fontsize=15, rotation=35, ha="right")
ax.set_yticks(range(N)); ax.set_yticklabels(labs, fontsize=15)
ax.set_xlabel("predicted", fontsize=16, color=MUTE); ax.set_ylabel("true", fontsize=16, color=MUTE)
thr = vmax * 0.5
for i in range(N):
    for j in range(N):
        v = conf_mat[i, j]
        ax.text(j, i, str(v), ha="center", va="center", fontsize=15,
                color=("white" if v > thr else ("#B9C2BC" if v == 0 else INK)),
                fontweight=("bold" if i == j else "normal"))
for s in ax.spines.values(): s.set_edgecolor("#D6DAD7")
ax.set_xticks(np.arange(-.5, N, 1), minor=True); ax.set_yticks(np.arange(-.5, N, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=3)
ax.tick_params(which="both", length=0)
ax.set_title("ConvNeXt V2-Tiny · validation fold", fontsize=13, color=MUTE, pad=12)
plt.tight_layout(); plt.savefig(OUT / "convnext_confusion.png", bbox_inches="tight", pad_inches=0.12); plt.close()

order = sorted(CLASSES, key=lambda c: per_class[c]["recall"])
fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=200)
vals = [per_class[c]["recall"] * 100 for c in order]
ax.barh(range(len(order)), vals, color=GREEN, height=0.62, zorder=3)
for yi, v in enumerate(vals):
    ax.text(v - 1.2, yi, f"{v:.1f}%", va="center", ha="right", fontsize=13,
            fontweight="bold", color="white", zorder=5)
ax.set_yticks(range(len(order))); ax.set_yticklabels([c.capitalize() for c in order], fontsize=14)
ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("Recall (%) · ConvNeXt V2 · validation fold", fontsize=12, color=MUTE)
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0); ax.set_axisbelow(True); ax.grid(axis="x", color="#E6E8E6", lw=0.9)
plt.tight_layout(); plt.savefig(OUT / "convnext_perclass.png", bbox_inches="tight", pad_inches=0.12); plt.close()

# contact sheet of the worst errors (up to 24), each captioned true -> pred
show = err[:24]
if show:
    cols = 6
    rows = (len(show) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.35), dpi=170)
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for k, i in enumerate(show):
        ax = axes[k]
        try:
            ax.imshow(Image.open(paths[i]).convert("RGB"))
        except Exception as exc:
            ax.text(0.5, 0.5, "image\nunavailable", ha="center", va="center", fontsize=8)
        ax.set_title(f"{CLASSES[y_true[i]]} → {CLASSES[y_pred[i]]}\n"
                     f"conf {conf[i]:.2f}", fontsize=9, color="#B23A2E")
    fig.suptitle("ConvNeXt V2-Tiny · worst validation errors (true → predicted)",
                 fontsize=12, color=MUTE)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUT / "convnext_misclassified.png", bbox_inches="tight", pad_inches=0.15)
    plt.close()

print("\nsaved to /kaggle/working:")
for f in ["convnext_metrics.json", "convnext_confusion.json", "convnext_confusion.png",
          "convnext_perclass.png", "convnext_predictions.csv",
          "convnext_misclassified.csv", "convnext_misclassified.png"]:
    print("  ", f)
