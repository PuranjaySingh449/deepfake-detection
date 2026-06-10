# =========================================================
# DEEPFAKE DETECTION — CLIP-ViT + Temporal Transformer
#
# Pipeline:
#   1. Run extract_features.py first (caches CLIP features)
#   2. Run this script to train the temporal head
#
# Architecture:
#   Cached 768-dim CLIP-ViT CLS embeddings
#   → CLS token + 4-layer pre-norm Transformer (8 heads)
#   → Binary classifier
#
# Target: beat paper (ROC-AUC 0.987, BalAcc 0.935)
#         and 66%+ accuracy on SDFVD
# =========================================================

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    precision_score, recall_score,
)

# =========================================================
# REPRODUCIBILITY
# =========================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# =========================================================
# SETTINGS
# =========================================================

DATASET_PATH    = "deepfake_faces"
SDFVD_PATH      = "SDFVD/SDFVD"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"

EMBED_DIM       = 768
SEQUENCE_LENGTH = 30

BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 5e-4
WEIGHT_DECAY = 5e-4     # Stronger weight decay
PATIENCE     = 10

# Paper class weights
REAL_WEIGHT = 1.478
FAKE_WEIGHT = 0.756

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nDevice: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# =========================================================
# DATASET — loads cached CLIP .npy feature files
# =========================================================

class FeatureSequenceDataset(Dataset):

    def __init__(self, root_dir, split, seq_len=SEQUENCE_LENGTH,
                 augment=False):
        self.samples = []
        self.seq_len = seq_len
        self.augment = augment

        split_path = os.path.join(root_dir, split)

        for label_name in ["real", "fake"]:
            label      = 0 if label_name == "real" else 1
            label_path = os.path.join(split_path, label_name)
            if not os.path.exists(label_path):
                continue

            for dataset_name in os.listdir(label_path):
                dp = os.path.join(label_path, dataset_name)
                if not os.path.isdir(dp):
                    continue

                for identity in os.listdir(dp):
                    ip = os.path.join(dp, identity)
                    if not os.path.isdir(ip):
                        continue

                    for video_folder in os.listdir(ip):
                        vp       = os.path.join(ip, video_folder)
                        feat_npy = os.path.join(vp, "vit_features.npy")
                        if os.path.exists(feat_npy):
                            self.samples.append((feat_npy, label))

        real_n = sum(1 for _, l in self.samples if l == 0)
        fake_n = sum(1 for _, l in self.samples if l == 1)
        print(f"  {split.upper()}: {len(self.samples)} sequences "
              f"(real={real_n}, fake={fake_n})")

    def __len__(self):
        return len(self.samples)

    def _sample_indices(self, n):
        if n >= self.seq_len:
            step = n / self.seq_len
            if self.augment:
                jitter  = step * 0.2
                indices = [int(np.clip(
                    i * step + random.uniform(-jitter, jitter), 0, n - 1))
                    for i in range(self.seq_len)]
            else:
                indices = [int(i * step) for i in range(self.seq_len)]
        else:
            indices = list(range(n)) + [n - 1] * (self.seq_len - n)
        return indices

    def __getitem__(self, idx):
        feat_path, label = self.samples[idx]
        feats    = np.load(feat_path)           # (N_frames, 768)
        indices  = self._sample_indices(len(feats))
        selected = feats[indices]               # (seq_len, 768)

        if self.augment:
            # Small Gaussian noise
            selected = selected + np.random.randn(
                *selected.shape).astype(np.float32) * 0.02
            # Random temporal flip
            if random.random() < 0.5:
                selected = selected[::-1].copy()
            # Random feature dropout (5% of dims)
            if random.random() < 0.3:
                mask     = (np.random.rand(selected.shape[1]) > 0.05
                            ).astype(np.float32)
                selected = selected * mask
            # Random temporal crop + resize (drop up to 20% of frames)
            if random.random() < 0.3:
                n_drop  = random.randint(1, max(1, self.seq_len // 5))
                start   = random.randint(0, n_drop)
                end     = self.seq_len - (n_drop - start)
                indices = np.linspace(start, end - 1, self.seq_len).astype(int)
                indices = np.clip(indices, 0, self.seq_len - 1)
                selected = selected[indices]

        x = torch.from_numpy(selected.astype(np.float32))
        y = torch.tensor(label, dtype=torch.float32)
        return x, y


# =========================================================
# MODEL — CLS-token Temporal Transformer
# =========================================================

class TemporalTransformer(nn.Module):

    def __init__(self, embed_dim=768, num_heads=8, num_layers=4,
                 ff_dim=2048, dropout=0.2, seq_len=SEQUENCE_LENGTH):
        super().__init__()

        self.input_norm = nn.LayerNorm(embed_dim)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional encoding (seq_len + 1 for CLS)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, seq_len + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        B, T, D = x.shape
        x   = self.input_norm(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = x + self.pos_embed[:, :T + 1, :]
        x   = self.transformer(x)
        return self.head(x[:, 0, :]).squeeze(1)


# =========================================================
# LOSS
# =========================================================

class WeightedBCE(nn.Module):
    def __init__(self, real_w=REAL_WEIGHT, fake_w=FAKE_WEIGHT,
                 smoothing=0.05):
        super().__init__()
        self.real_w    = real_w
        self.fake_w    = fake_w
        self.smoothing = smoothing

    def forward(self, logits, targets):
        t_s = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        w   = targets * self.fake_w + (1 - targets) * self.real_w
        return F.binary_cross_entropy_with_logits(
            logits, t_s, weight=w, reduction="mean")


# =========================================================
# EVALUATION
# =========================================================

def evaluate(model, loader, criterion, threshold=0.5):
    model.eval()
    losses, lbls, probs = [], [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out  = model(x)
            losses.append(criterion(out, y).item())
            probs.extend(torch.sigmoid(out).cpu().numpy())
            lbls.extend(y.cpu().numpy())

    preds  = (np.array(probs) > threshold).astype(float)
    loss   = float(np.mean(losses))
    acc    = accuracy_score(lbls, preds)
    bal    = balanced_accuracy_score(lbls, preds)
    f1     = f1_score(lbls, preds, zero_division=0)
    auc    = roc_auc_score(lbls, probs)
    pr_auc = average_precision_score(lbls, probs)
    return loss, acc, bal, f1, auc, pr_auc, np.array(lbls), np.array(probs)


# =========================================================
# THRESHOLD CALIBRATION
# =========================================================

def find_threshold(labels, probs, target_specificity=0.97):
    best_spec_t = 0.5
    best_bal_t  = 0.5
    best_bal    = 0.0

    for t in np.arange(0.01, 0.99, 0.01):
        preds = (probs > t).astype(float)
        tn    = np.sum((preds == 0) & (labels == 0))
        fp    = np.sum((preds == 1) & (labels == 0))
        spec  = tn / (tn + fp + 1e-9)
        if abs(spec - target_specificity) < 0.015:
            best_spec_t = t

        bal = balanced_accuracy_score(labels, preds)
        if bal > best_bal:
            best_bal   = bal
            best_bal_t = t

    return best_spec_t, best_bal_t


# =========================================================
# SDFVD EVALUATION — uses CLIP-ViT on raw videos
# =========================================================

def evaluate_sdfvd(model, sdfvd_path, threshold=0.5):
    try:
        import cv2
        from PIL import Image
        from transformers import CLIPVisionModel, CLIPImageProcessor
    except ImportError as e:
        print(f"Skipping SDFVD: {e}")
        return

    print("\n===================================")
    print("SDFVD EXTERNAL VALIDATION")
    print("===================================")

    print(f"Loading CLIP-ViT: {CLIP_MODEL_NAME} ...")
    processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip      = CLIPVisionModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    clip.eval()
    for p in clip.parameters():
        p.requires_grad = False

    fake_dir = os.path.join(sdfvd_path, "videos_fake")
    real_dir = os.path.join(sdfvd_path, "videos_real")
    videos   = (
        [(os.path.join(fake_dir, f), 1)
         for f in sorted(os.listdir(fake_dir)) if f.endswith(".mp4")] +
        [(os.path.join(real_dir, f), 0)
         for f in sorted(os.listdir(real_dir)) if f.endswith(".mp4")]
    )
    print(f"  {len(videos)} videos")

    model.eval()
    all_labels, all_probs = [], []

    for video_path, label in tqdm(videos, desc="SDFVD"):
        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            continue

        step       = max(1, total // SEQUENCE_LENGTH)
        indices    = [min(i * step, total - 1) for i in range(SEQUENCE_LENGTH)]
        pil_frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                pil_frames.append(pil_frames[-1] if pil_frames else
                                  Image.new("RGB", (224, 224)))
                continue
            pil_frames.append(
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()

        while len(pil_frames) < SEQUENCE_LENGTH:
            pil_frames.append(pil_frames[-1])
        pil_frames = pil_frames[:SEQUENCE_LENGTH]

        inputs = processor(images=pil_frames,
                           return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            feats = clip(**inputs).pooler_output   # (T, 768)
            feats = feats.unsqueeze(0)             # (1, T, 768)
            prob  = torch.sigmoid(model(feats)).item()

        all_labels.append(label)
        all_probs.append(prob)

    del clip
    torch.cuda.empty_cache()

    if len(set(all_labels)) < 2:
        print("Only one class — skipping metrics")
        return

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # Sweep for best balanced accuracy threshold
    best_t, best_b = 0.5, 0.0
    sweep_thresholds = np.arange(0.05, 0.95, 0.05)
    sweep_bal_accs   = []
    for t in sweep_thresholds:
        b = balanced_accuracy_score(
            all_labels, (all_probs > t).astype(float))
        sweep_bal_accs.append(b)
        if b > best_b:
            best_b, best_t = b, t

    # Save for plot_results.py
    np.savez("sdfvd_threshold_sweep.npz",
             thresholds = sweep_thresholds,
             bal_accs   = np.array(sweep_bal_accs),
             best_t     = best_t,
             best_b     = best_b)

    preds = (all_probs > threshold).astype(float)
    acc   = accuracy_score(all_labels, preds)
    bal   = balanced_accuracy_score(all_labels, preds)
    auc   = roc_auc_score(all_labels, all_probs)
    f1    = f1_score(all_labels, preds, zero_division=0)
    prec  = precision_score(all_labels, preds, zero_division=0)
    rec   = recall_score(all_labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
    spec  = tn / (tn + fp + 1e-9)

    print(f"Threshold used:    {threshold:.3f}")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy: {bal:.4f}")
    print(f"ROC-AUC:           {auc:.4f}")
    print(f"Fake Precision:    {prec:.4f}")
    print(f"Fake Recall:       {rec:.4f}")
    print(f"Specificity:       {spec:.4f}")
    print(f"F1 Score (Fake):   {f1:.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(all_labels, preds)}")
    print(classification_report(all_labels, preds,
                                target_names=["Real", "Fake"]))
    print(f"\nBest SDFVD threshold: {best_t:.2f}  BalAcc={best_b:.4f}")


# =========================================================
# PLOTTING
# =========================================================

def plot_metric(train_v, val_v, name, best_epoch=None):
    epochs = list(range(1, len(train_v) + 1))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, train_v, label=f"Train {name}",
            color="#2196F3", linewidth=2, marker="o",
            markersize=4, zorder=3)
    ax.plot(epochs, val_v, label=f"Val {name}",
            color="#FF5722", linewidth=2, marker="s",
            markersize=4, zorder=3)

    # Shade the gap between train and val
    ax.fill_between(epochs, train_v, val_v,
                    alpha=0.08, color="gray", label="Train-Val gap")

    # Mark best epoch
    if best_epoch is not None:
        best_val = val_v[best_epoch - 1]
        ax.axvline(x=best_epoch, color="green", linestyle="--",
                   linewidth=1.5, alpha=0.7, label=f"Best epoch ({best_epoch})")
        ax.scatter([best_epoch], [best_val], color="green",
                   s=80, zorder=5)

    # Annotate final values
    ax.annotate(f"{train_v[-1]:.4f}",
                xy=(epochs[-1], train_v[-1]),
                xytext=(5, 0), textcoords="offset points",
                fontsize=8, color="#2196F3")
    ax.annotate(f"{val_v[-1]:.4f}",
                xy=(epochs[-1], val_v[-1]),
                xytext=(5, 0), textcoords="offset points",
                fontsize=8, color="#FF5722")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(name, fontsize=12)
    ax.set_title(f"{name} — Training Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(epochs))

    # Add best val annotation box
    if best_epoch is not None:
        best_val = val_v[best_epoch - 1]
        ax.text(0.02, 0.05,
                f"Best Val {name}: {best_val:.4f} @ epoch {best_epoch}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="lightgreen", alpha=0.5))

    plt.tight_layout()
    fname = f"{name.lower().replace(' ', '_')}_curve.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# =========================================================
# TRAINING
# =========================================================

def train():

    print("\n===================================")
    print("LOADING DATASETS")
    print("===================================")

    train_ds = FeatureSequenceDataset(
        DATASET_PATH, "train", SEQUENCE_LENGTH, augment=True)
    val_ds   = FeatureSequenceDataset(
        DATASET_PATH, "val",   SEQUENCE_LENGTH, augment=False)
    test_ds  = FeatureSequenceDataset(
        DATASET_PATH, "test",  SEQUENCE_LENGTH, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    print("\n===================================")
    print("BUILDING MODEL")
    print("===================================")

    model = TemporalTransformer(
        embed_dim=EMBED_DIM, num_heads=8, num_layers=4,
        ff_dim=2048, dropout=0.3, seq_len=SEQUENCE_LENGTH,  # Higher dropout
    ).to(DEVICE)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")

    criterion = WeightedBCE(smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Cosine annealing — smooth LR decay, avoids plateau traps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)

    hist = {k: [] for k in
            ["train_loss", "val_loss", "train_auc", "val_auc",
             "train_bal",  "val_bal"]}

    best_auc   = 0.0
    best_epoch = 1
    no_improve = 0

    for epoch in range(EPOCHS):

        print(f"\n===================================")
        print(f"EPOCH {epoch + 1}/{EPOCHS}")
        print(f"===================================")

        model.train()
        run_loss, all_lbl, all_prob = 0.0, [], []

        for x, y in tqdm(train_loader, desc="Train"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            # Mixup augmentation (50% of batches)
            if random.random() < 0.5:
                lam   = float(np.random.beta(0.4, 0.4))
                idx   = torch.randperm(x.size(0), device=DEVICE)
                x_mix = lam * x + (1 - lam) * x[idx]
                out   = model(x_mix)
                loss  = lam * criterion(out, y) + (1 - lam) * criterion(out, y[idx])
            else:
                out  = model(x)
                loss = criterion(out, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            run_loss += loss.item()
            all_lbl.extend(y.cpu().numpy())
            all_prob.extend(torch.sigmoid(out).detach().cpu().numpy())

        train_loss = run_loss / len(train_loader)
        train_auc  = roc_auc_score(all_lbl, all_prob)
        train_bal  = balanced_accuracy_score(
            all_lbl, (np.array(all_prob) > 0.5).astype(float))

        val_loss, val_acc, val_bal, val_f1, val_auc, val_pr, _, _ = evaluate(
            model, val_loader, criterion)

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["train_auc"].append(train_auc)
        hist["val_auc"].append(val_auc)
        hist["train_bal"].append(train_bal)
        hist["val_bal"].append(val_bal)

        print(f"Train | Loss:{train_loss:.4f}  AUC:{train_auc:.4f}  "
              f"BalAcc:{train_bal:.4f}")
        print(f"Val   | Loss:{val_loss:.4f}  AUC:{val_auc:.4f}  "
              f"BalAcc:{val_bal:.4f}  F1:{val_f1:.4f}  "
              f"PR-AUC:{val_pr:.4f}  LR:{lr_now:.2e}")

        if val_auc > best_auc:
            best_auc   = val_auc
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  *** BEST MODEL SAVED  AUC={best_auc:.4f} ***")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nEarly stopping (patience={PATIENCE})")
                break

    plot_metric(hist["train_loss"], hist["val_loss"], "Loss",              best_epoch)
    plot_metric(hist["train_auc"],  hist["val_auc"],  "ROC_AUC",           best_epoch)
    plot_metric(hist["train_bal"],  hist["val_bal"],  "Balanced_Accuracy", best_epoch)

    # Save history for plot_results.py
    np.savez("training_history.npz",
             train_loss = hist["train_loss"],
             val_loss   = hist["val_loss"],
             train_auc  = hist["train_auc"],
             val_auc    = hist["val_auc"],
             train_bal  = hist["train_bal"],
             val_bal    = hist["val_bal"],
             best_epoch = best_epoch)
    print("\nCurves saved.")

    # ── Final test ───────────────────────────────────────
    print("\n===================================")
    print("FINAL TEST EVALUATION")
    print("===================================")

    model.load_state_dict(torch.load("best_model.pth", weights_only=True))

    _, _, _, _, _, _, val_lbl, val_prob = evaluate(
        model, val_loader, criterion)
    t_spec, t_bal = find_threshold(val_lbl, val_prob, target_specificity=0.97)
    print(f"\nCalibrated threshold (spec≈0.97): {t_spec:.3f}")
    print(f"Calibrated threshold (best bal):  {t_bal:.3f}")

    for tname, thresh in [("spec_threshold", t_spec),
                           ("bal_threshold",  t_bal)]:

        _, acc, bal, f1, auc, pr, lbl, prob = evaluate(
            model, test_loader, criterion, thresh)

        preds = (prob > thresh).astype(float)
        tn, fp, fn, tp = confusion_matrix(lbl, preds).ravel()
        spec  = tn / (tn + fp + 1e-9)
        prec  = precision_score(lbl, preds, zero_division=0)
        rec   = recall_score(lbl, preds, zero_division=0)

        print(f"\n── {tname} = {thresh:.3f} ──")
        print(f"Test Accuracy:          {acc:.4f}")
        print(f"Test Balanced Accuracy: {bal:.4f}")
        print(f"Test ROC-AUC:           {auc:.4f}")
        print(f"Test PR-AUC:            {pr:.4f}")
        print(f"Fake Precision:         {prec:.4f}")
        print(f"Fake Recall:            {rec:.4f}")
        print(f"Specificity:            {spec:.4f}")
        print(f"F1 Score (Fake):        {f1:.4f}")
        print(f"Confusion Matrix:\n{confusion_matrix(lbl, preds)}")
        print(classification_report(lbl, preds,
                                    target_names=["Real", "Fake"]))

    if os.path.exists(SDFVD_PATH):
        # Use balanced threshold for SDFVD — spec threshold is too conservative
        evaluate_sdfvd(model, SDFVD_PATH, threshold=t_bal)
    else:
        print(f"\nSDFVD path not found: {SDFVD_PATH}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    train()
