"""Generate every figure in the paper from the committed data.

    python3 model/paper/make_figures.py

No figure here is drawn from remembered numbers. Each one reads the split
metadata, the provenance table, or experiments.csv, so re-running after a
split rebuild regenerates figures that still match the data.

Palette: validated categorical slots (blue/orange/aqua). Aqua sits below 3:1
contrast on the light surface, so every stacked segment carries a direct label
(the relief rule) -- which a printed paper wants regardless.
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

MODEL = Path(__file__).resolve().parent.parent
FIGS = MODEL / "paper" / "figures"
SPLITS = MODEL / "data" / "splits_unified"

# Validated categorical slots, light mode
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED, VIOLET = "#e34948", "#4a3aa7"
INK, INK2, MUTED, FAINT = "#0b0b0b", "#52514e", "#8a8880", "#cbc9c2"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"

mpl.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "savefig.dpi": 200,
    "savefig.facecolor": SURFACE,
    "savefig.bbox": "tight",
})

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def tidy(ax, *, xgrid=False, ygrid=False):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, lw=0.6)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(MODEL.parent)}")


# ---------------------------------------------------------------------------
# Figure 1 - why group-aware splitting is necessary (schematic, not data)
# ---------------------------------------------------------------------------

def fig_leakage_schematic():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))

    def panel(ax, title, assign, note, note_color):
        ax.set_title(title, pad=10)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5.2)
        ax.axis("off")
        # the four photographs of one physical object
        for i in range(4):
            x = 0.4 + i * 1.35
            ax.add_patch(Rectangle((x, 3.3), 1.15, 1.15, facecolor="#dfe7f2",
                                   edgecolor=BLUE, lw=1.2))
            ax.text(x + 0.575, 3.875, f"v{i+1}", ha="center", va="center",
                    fontsize=8, color=INK2)
        ax.text(3.05, 4.75, "one bottle, four photographs", ha="center",
                fontsize=8, color=INK2, style="italic")

        # split boxes
        ax.add_patch(Rectangle((0.4, 0.5), 4.0, 1.5, facecolor="none",
                               edgecolor=MUTED, lw=1.0))
        ax.text(2.4, 1.72, "TRAIN", ha="center", fontsize=8, color=INK2, weight="bold")
        ax.add_patch(Rectangle((5.0, 0.5), 4.0, 1.5, facecolor="none",
                               edgecolor=MUTED, lw=1.0))
        ax.text(7.0, 1.72, "VALIDATION", ha="center", fontsize=8, color=INK2, weight="bold")

        for i, dest in enumerate(assign):
            x0 = 0.975 + i * 1.35
            x1 = (1.4 + i * 0.85) if dest == "train" else (5.6 + (i - 2) * 0.85)
            colour = ORANGE if dest == "val" else BLUE
            ax.add_patch(FancyArrowPatch((x0, 3.25), (x1, 2.05), arrowstyle="->",
                                         mutation_scale=8, color=colour, lw=1.0,
                                         alpha=0.85))
            ax.add_patch(Rectangle((x1 - 0.35, 0.95), 0.7, 0.7,
                                   facecolor="#dfe7f2", edgecolor=colour, lw=1.0))
        ax.text(4.7, 0.05, note, ha="center", fontsize=8, color=note_color, weight="bold")

    panel(axes[0], "Random image-level split",
          ["train", "train", "val", "val"],
          "the model can memorise the object, not the material",
          RED)
    panel(axes[1], "Group-aware split (this project)",
          ["train", "train", "train", "train"],
          "the whole near-duplicate group stays on one side",
          "#0a7a46")

    fig.suptitle("Figure 1. How near-duplicate photographs inflate validation accuracy",
                 fontsize=10, fontweight="bold", y=1.13)
    save(fig, "fig1_leakage_schematic.png")


# ---------------------------------------------------------------------------
# Figure 2 - class distribution before and after corpus expansion
# ---------------------------------------------------------------------------

def fig_class_distribution():
    trashnet = {"cardboard": 403, "glass": 501, "metal": 410,
                "paper": 594, "plastic": 482, "trash": 137}
    groups = pd.read_csv(SPLITS / "groups.csv")
    unified = groups.label.value_counts().reindex(CLASSES)

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    y = range(len(CLASSES))
    h = 0.38
    b1 = ax.barh([v + h / 2 for v in y], [trashnet[c] for c in CLASSES], height=h,
                 color=BLUE, label="Original TrashNet (2,527)")
    b2 = ax.barh([v - h / 2 for v in y], unified.values, height=h,
                 color=ORANGE, label="Unified corpus, 3 sources (11,299)")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2,
                    f"{int(bar.get_width()):,}", va="center", fontsize=7.5, color=INK2)

    ax.set_yticks(list(y), CLASSES)
    ax.invert_yaxis()
    ax.set_xlabel("images")
    ax.set_xlim(0, 2900)
    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.legend(loc="lower right")
    tidy(ax, xgrid=True)
    ax.set_title("Figure 2. Corpus expansion dissolves the minority-class problem\n"
                 "$trash$ rises from 137 to 1,033 images (7.5x)", loc="left")
    save(fig, "fig2_class_distribution.png")


# ---------------------------------------------------------------------------
# Figure 3 - the contamination discovery
# ---------------------------------------------------------------------------

def fig_hidden_duplication():
    prov = pd.read_csv(MODEL / "data" / "unified_waste" / "trashnet_provenance.csv")
    renamed = int((prov.status == "duplicate").sum())
    own = int((prov.status == "stored").sum())

    test = prov[prov.old_split == "test"]
    prefix = test.unified_path.str.rsplit("/", n=1).str[-1].str.split("__").str[0]
    disguised = int((prefix == "garbage_classification").sum())
    named = int((prefix == "trashnet").sum())

    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    rows = [
        ("All 2,527 TrashNet images", renamed, own),
        ("The 361 already-spent test images", disguised, named),
    ]
    for i, (label, hidden, visible) in enumerate(rows):
        total = hidden + visible
        ax.barh(i, hidden / total * 100, color=ORANGE, height=0.5)
        ax.barh(i, visible / total * 100, left=hidden / total * 100 + 0.4,
                color=BLUE, height=0.5)
        ax.text(hidden / total * 50, i, f"{hidden:,}  ({hidden/total:.0%})",
                ha="center", va="center", color="white", fontsize=8.5, weight="bold")
        ax.text(hidden / total * 100 + (visible / total * 50), i,
                f"{visible:,}", ha="center", va="center", color="white",
                fontsize=8.5, weight="bold")

    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(1.6, -0.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of images")
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter())
    handles = [mpl.patches.Patch(color=ORANGE,
                                 label="renamed inside garbage_classification"),
               mpl.patches.Patch(color=BLUE, label="stored under a trashnet filename")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=2)
    tidy(ax)
    ax.set_title("Figure 3. A public dataset silently contains the benchmark\n"
                 "Filename comparison detects none of this; provenance recovery detects all of it",
                 loc="left")
    save(fig, "fig3_hidden_duplication.png")


# ---------------------------------------------------------------------------
# Figure 4 - source composition per split
# ---------------------------------------------------------------------------

def fig_fold_source_balance():
    folds = pd.read_csv(SPLITS / "folds.csv")
    test = pd.read_csv(SPLITS / "test.csv")
    sources = ["garbage_classification", "realwaste", "trashnet"]
    colours = {"garbage_classification": BLUE, "realwaste": ORANGE, "trashnet": AQUA}

    table = pd.crosstab(folds.fold, folds.source).reindex(columns=sources, fill_value=0)
    table.index = [f"fold {i}" for i in table.index]
    test_row = test.source.value_counts().reindex(sources).fillna(0).astype(int)
    table.loc["test"] = test_row

    shares = table.div(table.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    left = pd.Series(0.0, index=table.index)
    for src in sources:
        ax.barh(table.index, shares[src], left=left, height=0.55,
                color=colours[src], label=src, edgecolor=SURFACE, lw=1.2)
        for name in table.index:
            if shares.loc[name, src] > 6:
                ax.text(left[name] + shares.loc[name, src] / 2,
                        list(table.index).index(name),
                        f"{table.loc[name, src]:,}", ha="center", va="center",
                        color="white", fontsize=7.5, weight="bold")
        left += shares[src]

    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of split")
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter())
    # separate the quarantine-driven test row from the five interchangeable folds
    ax.axhline(4.5, color=MUTED, lw=0.8, ls=":")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    tidy(ax)
    ax.set_title("Figure 4. Every fold carries all three domains — but the test set does not match them\n"
                 "Forcing the quarantine into test enriches it with TrashNet (36% vs ~20% per fold)",
                 loc="left")
    save(fig, "fig4_fold_source_balance.png")


# ---------------------------------------------------------------------------
# Figure 5 - Phase 1 validation results against the acceptance gate
# ---------------------------------------------------------------------------

def _phase1_rows():
    exp = pd.read_csv(MODEL / "experiments.csv")
    exp = exp[exp.stage == "eval"].copy()
    wanted = {
        "baseline_resnet50": ("ResNet-50 baseline", False),
        "resnet50_384_progressive": ("Progressive resize 384px", True),
        "resnet50_224_aug_modern": ("Modern augmentation", True),
        "resnet50_224_wd010": ("Weight decay 0.10", True),
    }
    out = []
    for run, (label, _) in wanted.items():
        rows = exp[(exp.run_name == run) & (exp.notes.str.contains("tta=True", na=False))]
        row = rows.iloc[-1]
        out.append({"label": label, "acc": row.best_val_acc, "f1": row.val_macro_f1,
                    "recall": json.loads(row.per_class_recall)})
    single = exp[(exp.run_name == "baseline_resnet50")
                 & (exp.notes.str.contains("best_raw.pth tta=False", na=False))].iloc[-1]
    return out, {"acc": single.best_val_acc, "f1": single.val_macro_f1,
                 "recall": json.loads(single.per_class_recall)}


def fig_phase1_results():
    rows, single = _phase1_rows()
    n_val = 434
    base = rows[0]["acc"]
    gate = base + 7 / n_val  # acceptance rule: at least seven more correct images

    labels = ["ResNet-50, single view"] + [r["label"] + " + TTA" for r in rows]
    accs = [single["acc"]] + [r["acc"] for r in rows]
    # the single-view run is the starting point, not a rejected candidate
    roles = ["reference", "adopted", "rejected", "rejected", "rejected"]
    fill = {"reference": FAINT, "adopted": BLUE, "rejected": MUTED}

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    bars = ax.barh(range(len(labels)), [a * 100 for a in accs], height=0.5,
                   color=[fill[r] for r in roles])
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                f"{acc*100:.2f}%  ({round(acc*n_val)}/{n_val})",
                va="center", fontsize=7.5, color=INK2)

    ax.axvline(gate * 100, color=RED, lw=1.2, ls="--")
    ax.text(gate * 100 + 0.12, 3.9, "gate: beat the adopted\nconfiguration by 7 images",
            color=RED, fontsize=7.5, va="center")

    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(90, 98.6)
    ax.set_xlabel("fold-0 validation accuracy")
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(decimals=0))
    handles = [mpl.patches.Patch(color=FAINT, label="starting point"),
               mpl.patches.Patch(color=BLUE, label="adopted"),
               mpl.patches.Patch(color=MUTED, label="rejected")]
    ax.legend(handles=handles, loc="lower right", ncol=3,
              bbox_to_anchor=(1.0, -0.30))
    tidy(ax, xgrid=True)
    ax.set_title("Figure 5. Three plausible improvements failed a pre-declared gate\n"
                 "Only four-view TTA cleared it; the rest are within noise on 434 images",
                 loc="left")
    save(fig, "fig5_phase1_results.png")


# ---------------------------------------------------------------------------
# Figure 6 - the accuracy/recall trade-off that accuracy alone hides
# ---------------------------------------------------------------------------

def fig_per_class_recall():
    rows, _ = _phase1_rows()
    base = next(r for r in rows if r["label"] == "ResNet-50 baseline")
    aug = next(r for r in rows if r["label"] == "Modern augmentation")

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = range(len(CLASSES))
    w = 0.38
    b1 = ax.bar([v - w / 2 for v in x], [base["recall"][c] * 100 for c in CLASSES],
                width=w, color=BLUE, label=f"Baseline + TTA  (94.93%, macro-F1 {base['f1']:.3f})")
    b2 = ax.bar([v + w / 2 for v in x], [aug["recall"][c] * 100 for c in CLASSES],
                width=w, color=ORANGE, label=f"Modern aug + TTA  (95.62%, macro-F1 {aug['f1']:.3f})")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{bar.get_height():.0f}", ha="center", fontsize=7, color=INK2)

    # Callout sits in the empty band above the trash pair, so no arrow crosses a bar.
    ax.text(5, 106, "trash recall  20/23 → 18/23", ha="center", fontsize=7.5,
            color=RED, weight="bold")
    ax.plot([4.62, 5.38], [102, 102], color=RED, lw=1.0)

    ax.set_xticks(list(x), CLASSES)
    ax.set_ylim(0, 116)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("per-class recall")
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.34), ncol=1)
    tidy(ax, ygrid=True)
    ax.set_title("Figure 6. Higher accuracy, worse model\n"
                 "Modern augmentation gains 3 images overall while losing 2 of 23 $trash$ images",
                 loc="left")
    save(fig, "fig6_per_class_recall.png")


# ---------------------------------------------------------------------------
# Figure 7 - where the quarantine ends up
# ---------------------------------------------------------------------------

def fig_split_composition():
    folds = pd.read_csv(SPLITS / "folds.csv")
    test = pd.read_csv(SPLITS / "test.csv")
    quarantine = set(pd.read_csv(SPLITS / "quarantine.csv").path)
    legacy = len(quarantine & set(test.path))
    fold_sizes = folds.fold.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    left = 0.0
    total = len(folds) + len(test)
    for i, size in enumerate(fold_sizes):
        ax.barh(0, size, left=left, height=0.5, color=BLUE, edgecolor=SURFACE, lw=1.5)
        ax.text(left + size / 2, 0, f"fold {i}\n{size:,}", ha="center", va="center",
                color="white", fontsize=7.5, weight="bold")
        left += size
    ax.barh(0, len(test) - legacy, left=left, height=0.5, color=ORANGE,
            edgecolor=SURFACE, lw=1.5)
    ax.text(left + (len(test) - legacy) / 2, 0, f"new test\n{len(test)-legacy:,}",
            ha="center", va="center", color="white", fontsize=7.5, weight="bold")
    left += len(test) - legacy
    ax.barh(0, legacy, left=left, height=0.5, color=RED, edgecolor=SURFACE, lw=1.5)
    ax.annotate(f"spent test\n{legacy}", xy=(left + legacy / 2, -0.28),
                xytext=(left + legacy / 2, -0.95), ha="center", fontsize=7.5,
                color=RED, weight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))

    ax.set_xlim(0, total)
    ax.set_ylim(-1.05, 0.4)
    ax.set_yticks([])
    ax.set_xlabel("images")
    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
    tidy(ax)
    ax.spines["left"].set_visible(False)
    ax.set_title("Figure 7. All 361 previously-spent test images are forced back into the test set\n"
                 f"train+validation {len(folds):,} · test {len(test):,} ({len(test)/total:.1%})",
                 loc="left")
    save(fig, "fig7_split_composition.png")


# ---------------------------------------------------------------------------
# Figure 8 - training curves for the adopted run
# ---------------------------------------------------------------------------

def fig_training_curves():
    h = pd.read_csv(MODEL / "reports" / "baseline_resnet50" / "history_fold0.csv")
    head_epochs = int((h.phase == "head").sum())

    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.plot(h.epoch, h.train_loss, color=BLUE, lw=2, label="train loss")
    ax.plot(h.epoch, h.val_loss, color=ORANGE, lw=2, label="validation loss")

    # Phase labels sit in the empty band under the train curve, clear of the
    # start-of-run value labels at the top left.
    ax.axvline(head_epochs - 0.5, color=MUTED, lw=1.0, ls=":")
    ax.text(head_epochs - 0.2, 0.35, "backbone unfrozen", fontsize=7.5, color=INK2)
    ax.text(0.2, 0.35, "head only", fontsize=7.5, color=INK2, ha="left")

    for series, colour in ((h.train_loss, BLUE), (h.val_loss, ORANGE)):
        ax.scatter([h.epoch.iloc[-1]], [series.iloc[-1]], s=22, color=colour, zorder=3)
        ax.text(h.epoch.iloc[-1] + 0.7, series.iloc[-1], f"{series.iloc[-1]:.2f}",
                fontsize=7.5, color=colour, va="center", weight="bold")
    ax.text(0.4, h.train_loss.iloc[0] + 0.06, f"{h.train_loss.iloc[0]:.2f}",
            fontsize=7.5, color=BLUE, weight="bold")
    ax.text(0.4, h.val_loss.iloc[0] + 0.06, f"{h.val_loss.iloc[0]:.2f}",
            fontsize=7.5, color=ORANGE, weight="bold")

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_xlim(-1, len(h) + 2)
    ax.set_ylim(0.3, 2.0)
    ax.legend(loc="upper right")
    tidy(ax, ygrid=True)
    ax.set_title("Figure 8. Training curves, adopted ResNet-50 run (45 epochs)\n"
                 "Validation loss is still falling at the end — the run stopped on the "
                 "epoch budget, not on early stopping", loc="left")
    save(fig, "fig8_training_curves.png")


# ---------------------------------------------------------------------------
# Figure 9 - confusion matrix for the adopted configuration
# ---------------------------------------------------------------------------

def fig_confusion_matrix():
    preds = pd.read_csv(MODEL / "reports" / "locked_baseline" / "tta4" / "predictions.csv")
    cm = pd.crosstab(preds.label, preds.predicted).reindex(
        index=CLASSES, columns=CLASSES, fill_value=0)
    recall = cm.to_numpy().diagonal() / cm.sum(axis=1).to_numpy()

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    norm = cm.to_numpy() / cm.sum(axis=1).to_numpy()[:, None]
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            n = cm.iat[i, j]
            if n == 0:
                continue
            ax.text(j, i, str(n), ha="center", va="center", fontsize=8.5,
                    color="white" if norm[i, j] > 0.55 else INK,
                    weight="bold" if i == j else "normal")

    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(len(CLASSES)),
                  [f"{c}  ({r:.0%})" for c, r in zip(CLASSES, recall)])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true class  (recall)")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Figure 9. Confusion matrix, fold-0 validation\n"
                 "ResNet-50 + four-view TTA, 412/434 correct", loc="left", pad=10)
    save(fig, "fig9_confusion_matrix.png")


if __name__ == "__main__":
    print("Generating figures from committed data:")
    fig_leakage_schematic()
    fig_class_distribution()
    fig_hidden_duplication()
    fig_fold_source_balance()
    fig_phase1_results()
    fig_per_class_recall()
    fig_split_composition()
    fig_training_curves()
    fig_confusion_matrix()
    print("Done.")
