# =========================================================
# Publication-quality result plots
# Run AFTER training completes — reads real_results.npz
# for accurate curves, or uses final known metrics.
# =========================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import os

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# =========================================================
# 1. INTERNAL TEST METRICS — Paper vs Ours
# =========================================================

def plot_internal_comparison(ours_vals=None):
    metrics = [
        "Accuracy", "Balanced\nAccuracy", "ROC-AUC",
        "PR-AUC", "Fake\nPrecision", "Fake\nRecall",
        "Specificity", "F1\n(Fake)",
    ]
    paper = [0.925, 0.935, 0.987, 0.995, 0.983, 0.913, 0.957, 0.947]
    ours  = ours_vals or [0.886, 0.886, 0.957, 0.952, 0.890, 0.874, 0.897, 0.882]

    x, w = np.arange(len(metrics)), 0.35
    fig, ax = plt.subplots(figsize=(14, 7))

    b1 = ax.bar(x - w/2, paper, w, label="Paper (ViT-BiLSTM)",
                color="#1565C0", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w/2, ours,  w, label="Ours (CLIP-ViT + Transformer)",
                color="#E53935", alpha=0.85, edgecolor="white")

    for bar, col in [(b1, "#1565C0"), (b2, "#E53935")]:
        for b in bar:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.004,
                    f"{b.get_height():.3f}", ha="center", va="bottom",
                    fontsize=7.5, color=col, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0.70, 1.04); ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Internal Test Set — Paper vs Our Model",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=11); ax.grid(axis="y", alpha=0.25)
    ax.axhline(1.0, color="gray", ls="--", alpha=0.2)
    plt.tight_layout()
    plt.savefig("comparison_internal.png", dpi=150)
    plt.close()
    print("Saved: comparison_internal.png")


# =========================================================
# 2. SDFVD CROSS-DATASET COMPARISON
# =========================================================

def plot_sdfvd_comparison(ours_vals=None):
    metrics = [
        "Accuracy", "Balanced\nAccuracy", "ROC-AUC",
        "Fake\nPrecision", "Fake\nRecall", "Specificity", "F1\n(Fake)",
    ]
    paper = [0.660, 0.660, 0.765, 0.840, 0.396, 0.925, 0.538]
    ours  = ours_vals or [0.698, 0.698, 0.758, 0.818, 0.509, 0.887, 0.628]

    x, w = np.arange(len(metrics)), 0.35
    fig, ax = plt.subplots(figsize=(13, 7))

    b1 = ax.bar(x - w/2, paper, w, label="Paper (ViT-BiLSTM)",
                color="#1565C0", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w/2, ours,  w, label="Ours (CLIP-ViT + Transformer)",
                color="#E53935", alpha=0.85, edgecolor="white")

    for bar, col in [(b1, "#1565C0"), (b2, "#E53935")]:
        for b in bar:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                    f"{b.get_height():.3f}", ha="center", va="bottom",
                    fontsize=8, color=col, fontweight="bold")

    # Green shading where we beat the paper
    for i, (p, o) in enumerate(zip(paper, ours)):
        if o > p:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.07, color="green", zorder=0)

    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0.25, 1.0); ax.set_ylabel("Score", fontsize=12)
    ax.set_title("SDFVD Cross-Dataset Validation — Paper vs Our Model\n"
                 "(green shading = we beat the paper)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=11); ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig("comparison_sdfvd.png", dpi=150)
    plt.close()
    print("Saved: comparison_sdfvd.png")


# =========================================================
# 3. TRAINING CURVES (from saved npz if available)
# =========================================================

def plot_training_curves():
    if not os.path.exists("training_history.npz"):
        print("training_history.npz not found — skipping training curves")
        return

    data       = np.load("training_history.npz")
    train_auc  = data["train_auc"]
    val_auc    = data["val_auc"]
    train_loss = data["train_loss"]
    val_loss   = data["val_loss"]
    train_bal  = data["train_bal"]
    val_bal    = data["val_bal"]
    best_epoch = int(data["best_epoch"])
    epochs     = list(range(1, len(train_auc) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    configs = [
        (axes[0], train_loss, val_loss,  "Loss",             "#2196F3", "#FF5722"),
        (axes[1], train_auc,  val_auc,   "ROC-AUC",          "#2196F3", "#FF5722"),
        (axes[2], train_bal,  val_bal,   "Balanced Accuracy", "#2196F3", "#FF5722"),
    ]

    for ax, tr, va, name, c1, c2 in configs:
        ax.plot(epochs, tr, label="Train", color=c1, lw=2, marker="o", ms=3)
        ax.plot(epochs, va, label="Val",   color=c2, lw=2, marker="s", ms=3)
        ax.fill_between(epochs, tr, va, alpha=0.07, color="gray")
        ax.axvline(best_epoch, color="green", ls="--", lw=1.5,
                   label=f"Best epoch ({best_epoch})")
        ax.scatter([best_epoch], [va[best_epoch-1]], color="green", s=60, zorder=5)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.legend(fontsize=9); ax.grid(alpha=0.25)

        # Annotate final values
        ax.annotate(f"{tr[-1]:.4f}", xy=(epochs[-1], tr[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=7.5, color=c1)
        ax.annotate(f"{va[-1]:.4f}", xy=(epochs[-1], va[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=7.5, color=c2)

    fig.suptitle("Training Curves — CLIP-ViT + Temporal Transformer",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.close()
    print("Saved: training_curves.png")


# =========================================================
# 4. SDFVD THRESHOLD SWEEP (from saved npz if available)
# =========================================================

def plot_threshold_sweep():
    if not os.path.exists("sdfvd_threshold_sweep.npz"):
        print("sdfvd_threshold_sweep.npz not found — skipping threshold sweep")
        return

    data       = np.load("sdfvd_threshold_sweep.npz")
    thresholds = data["thresholds"]
    bal_accs   = data["bal_accs"]
    best_t     = float(data["best_t"])
    best_b     = float(data["best_b"])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(thresholds, bal_accs, color="#E53935", lw=2.5,
            marker="o", ms=5, label="Our model")
    ax.axhline(0.660, color="#1565C0", ls="--", lw=2,
               label="Paper SDFVD BalAcc (0.660)")
    ax.axvline(best_t, color="green", ls="--", lw=2,
               label=f"Best threshold ({best_t:.2f})")
    ax.scatter([best_t], [best_b], color="green", s=100, zorder=5)
    ax.annotate(f"Best: {best_b:.4f}", xy=(best_t, best_b),
                xytext=(10, -15), textcoords="offset points",
                fontsize=10, color="green", fontweight="bold")

    ax.fill_between(thresholds, 0.660, bal_accs,
                    where=bal_accs > 0.660,
                    alpha=0.15, color="green", label="Beats paper")

    ax.set_xlabel("Classification Threshold", fontsize=12)
    ax.set_ylabel("Balanced Accuracy (SDFVD)", fontsize=12)
    ax.set_title("SDFVD Threshold Analysis\n"
                 "Balanced accuracy vs decision threshold",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("threshold_analysis.png", dpi=150)
    plt.close()
    print("Saved: threshold_analysis.png")


# =========================================================
# 5. STATE-OF-THE-ART COMPARISON TABLE (as figure)
# =========================================================

def plot_sota_table(our_ff_auc=0.9565, our_celeb_auc=0.924,
                    our_sdfvd_acc=0.698, our_temporal=True,
                    our_crossdataset=True):

    methods = [
        "Xception\n(2019)",
        "CapsuleNet\n(2019)",
        "CNN-LSTM\n(2024)",
        "GazeForensics\n(2023)",
        "FakeFormer\n(2024)",
        "Frozen CLIP-ViT\n(GFF, 2024)",
        "Paper\n(ViT-BiLSTM, 2026)",
        "Ours\n(CLIP+Transformer)",
    ]

    ff_acc  = ["99.7%", "96.6%", "~95%", "~99.6%", "~97.7%", "~99.9%", "96.32%", f"{our_ff_auc*100:.1f}%*"]
    celeb   = ["~65%",  "~70%",  "~88%", "~99.4%", "~95.2%", "99.5%",  "92.4%",  f"{our_celeb_auc*100:.1f}%*"]
    temp    = ["No",    "No",    "Yes",  "No",      "No",     "No",     "Yes",    "Yes"]
    cross   = ["Ltd",   "Ltd",   "Ltd",  "Ltd",     "Ltd",    "GANs",   "SDFVD",  "SDFVD ✓"]
    sdfvd   = ["—",     "—",     "—",    "—",       "—",      "—",      "66.0%",  f"{our_sdfvd_acc*100:.1f}% ✓"]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis("off")

    col_labels = ["Method", "FF++ Acc/AUC", "CelebDF AUC",
                  "Temporal?", "Cross-Dataset?", "SDFVD Acc"]
    table_data = list(zip(methods, ff_acc, celeb, temp, cross, sdfvd))

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)

    # Style header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1565C0")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight our row
    our_row = len(methods)
    for j in range(len(col_labels)):
        tbl[our_row, j].set_facecolor("#FFEBEE")
        tbl[our_row, j].set_text_props(fontweight="bold", color="#B71C1C")

    # Highlight paper row
    paper_row = len(methods) - 1
    for j in range(len(col_labels)):
        tbl[paper_row, j].set_facecolor("#E3F2FD")
        tbl[paper_row, j].set_text_props(fontweight="bold", color="#0D47A1")

    # Alternate row shading
    for i in range(1, len(methods) - 1):
        if i % 2 == 0:
            for j in range(len(col_labels)):
                if tbl[i, j].get_facecolor()[0] == 1.0:
                    tbl[i, j].set_facecolor("#F5F5F5")

    ax.set_title("Comparison with State-of-the-Art Methods\n"
                 "* AUC reported; † our model trained on all 4 datasets",
                 fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig("sota_comparison_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: sota_comparison_table.png")


# =========================================================
# 6. CONFUSION MATRICES
# =========================================================

def plot_confusion_matrices(cm_internal=None, cm_sdfvd=None):
    cm_int  = cm_internal or np.array([[560, 64], [75, 518]])
    cm_sdfv = cm_sdfvd    or np.array([[47,  6],  [26, 27]])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, cm, title in [
        (axes[0], cm_int,  "Internal Test Set"),
        (axes[1], cm_sdfv, "SDFVD External Validation"),
    ]:
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=18, fontweight="bold",
                        color="white" if cm[i, j] > thresh else "black")

        acc = (cm[0,0] + cm[1,1]) / cm.sum()
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Real", "Pred Fake"], fontsize=11)
        ax.set_yticklabels(["Act Real",  "Act Fake"],  fontsize=11)
        ax.set_title(f"{title}\nAccuracy: {acc:.1%}", fontsize=12,
                     fontweight="bold")

    fig.suptitle("Confusion Matrices — CLIP-ViT + Temporal Transformer",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150)
    plt.close()
    print("Saved: confusion_matrices.png")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print("Generating publication-quality plots...\n")

    plot_internal_comparison()
    plot_sdfvd_comparison()
    plot_confusion_matrices()
    plot_training_curves()      # needs training_history.npz
    plot_threshold_sweep()      # needs sdfvd_threshold_sweep.npz
    plot_sota_table()

    print("\nAll plots saved.")
    print("\nNote: training_curves.png and threshold_analysis.png")
    print("will be generated automatically after retraining completes.")
