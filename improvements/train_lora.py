"""
LoRA fine-tuning of CLIP-ViT-B/16 + Temporal Transformer.

The original train.py freezes CLIP and trains only a temporal head on cached
features. That caps cross-dataset generalization. Here we instead attach LoRA
adapters to CLIP's attention projections and train them jointly with the
temporal head on the face-crop JPGs — the documented path from ~0.80 toward
~0.90 AUC.

Engineered for a 6 GB laptop GPU (RTX 3050):
    - LoRA only (CLIP base weights frozen) -> tiny trainable param count
    - mixed precision (autocast + GradScaler)
    - gradient checkpointing on the CLIP encoder
    - small per-video batch + gradient accumulation
    - reads frames straight from deepfake_faces/<split>/.../frame_*.jpg

Outputs (in this folder):
    lora_best.pt          best adapter+head weights (by val ROC-AUC)
    lora_history.json     per-epoch metrics
Run a quick sanity check first:  python train_lora.py --smoke
"""

import os
import json
import math
import random
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from transformers import CLIPVisionModel
from peft import LoraConfig, get_peft_model

from model import TemporalTransformer

# ----------------------------------------------------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "deepfake_faces")
CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN       = 15        # frames per video (matches extracted count)
BATCH_VIDEOS  = 2         # videos per step  (2 x 15 = 30 imgs through CLIP)
ACCUM_STEPS   = 4         # effective batch = 8 videos
EPOCHS        = 6
LR_HEAD       = 5e-4
LR_LORA       = 1e-4
WEIGHT_DECAY  = 1e-4
LORA_R        = 8
LORA_ALPHA    = 16
LORA_DROPOUT  = 0.1
REAL_WEIGHT   = 1.478
FAKE_WEIGHT   = 0.756

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

# ----------------------------------------------------------------------

class FrameDataset(Dataset):
    """Loads SEQ_LEN face-crop JPGs per video as a pixel tensor (T,3,224,224)."""

    def __init__(self, split, augment=False):
        self.samples = []
        self.augment = augment
        split_path = os.path.join(DATA, split)
        for label_name in ["real", "fake"]:
            label = 0 if label_name == "real" else 1
            lp = os.path.join(split_path, label_name)
            if not os.path.isdir(lp):
                continue
            for ds in os.listdir(lp):
                dp = os.path.join(lp, ds)
                if not os.path.isdir(dp):
                    continue
                for ident in os.listdir(dp):
                    ip = os.path.join(dp, ident)
                    if not os.path.isdir(ip):
                        continue
                    for vid in os.listdir(ip):
                        vp = os.path.join(ip, vid)
                        if os.path.isdir(vp):
                            jpgs = sorted(f for f in os.listdir(vp)
                                          if f.endswith(".jpg"))
                            if len(jpgs) >= 5:
                                self.samples.append((vp, jpgs, label))

        r = sum(1 for *_, l in self.samples if l == 0)
        f = sum(1 for *_, l in self.samples if l == 1)
        print(f"  {split.upper()}: {len(self.samples)} videos (real={r}, fake={f})")

        aug = []
        if augment:
            aug = [T.RandomHorizontalFlip(),
                   T.ColorJitter(0.2, 0.2, 0.2, 0.05),
                   T.RandomApply([T.GaussianBlur(3)], p=0.2)]
        self.tf = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224), *aug,
            T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD),
        ])

    def __len__(self):
        return len(self.samples)

    def _pick(self, jpgs):
        n = len(jpgs)
        if n >= SEQ_LEN:
            step = n / SEQ_LEN
            return [jpgs[min(int(i * step), n - 1)] for i in range(SEQ_LEN)]
        return jpgs + [jpgs[-1]] * (SEQ_LEN - n)

    def __getitem__(self, idx):
        vp, jpgs, label = self.samples[idx]
        chosen = self._pick(jpgs)
        # one consistent augmentation decision per video for temporal coherence
        if self.augment and random.random() < 0.5:
            chosen = chosen[::-1]
        frames = []
        for fn in chosen:
            try:
                img = Image.open(os.path.join(vp, fn)).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224))
            frames.append(self.tf(img))
        x = torch.stack(frames)                       # (T,3,224,224)
        return x, torch.tensor(label, dtype=torch.float32)


class LoRACLIPDetector(nn.Module):
    def __init__(self):
        super().__init__()
        clip = CLIPVisionModel.from_pretrained(CLIP_MODEL_NAME)
        clip.gradient_checkpointing_enable()
        lora_cfg = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            bias="none",
        )
        self.clip = get_peft_model(clip, lora_cfg)
        self.clip.enable_input_require_grads()  # needed with grad checkpointing
        self.head = TemporalTransformer(seq_len=SEQ_LEN, dropout=0.3)

    def forward(self, x):                     # x: (B,T,3,224,224)
        B, Tn = x.shape[:2]
        flat = x.flatten(0, 1)                # (B*T,3,224,224)
        feats = self.clip(pixel_values=flat).pooler_output   # (B*T,768)
        feats = feats.view(B, Tn, -1)
        return self.head(feats)


def weighted_bce(logits, targets, smoothing=0.05):
    t = targets * (1 - smoothing) + 0.5 * smoothing
    w = targets * FAKE_WEIGHT + (1 - targets) * REAL_WEIGHT
    return F.binary_cross_entropy_with_logits(logits, t, weight=w)


@torch.no_grad()
def evaluate(model, loader, use_amp):
    # Validation runs in fp32 (no autocast). fp16 overflow in CLIP's
    # pooler_output was poisoning eval logits -> NaN probs -> NaN AUC.
    model.eval()
    probs, lbls = [], []
    n_bad = 0
    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(DEVICE, non_blocking=True)
        out = model(x).float()
        p = torch.sigmoid(out).cpu().numpy()
        n_bad += int(np.count_nonzero(~np.isfinite(p)))
        p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
        probs.extend(p)
        lbls.extend(y.numpy())
    probs, lbls = np.array(probs), np.array(lbls)
    if n_bad:
        print(f"  [warn] {n_bad} non-finite eval probs were clamped")
    preds = (probs > 0.5).astype(float)
    try:
        auc = roc_auc_score(lbls, probs)
    except ValueError:
        auc = float("nan")
    bal = balanced_accuracy_score(lbls, preds)
    f1 = f1_score(lbls, preds, zero_division=0)
    return auc, bal, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="2-step dry run to validate memory/wiring")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    use_amp = DEVICE == "cuda"
    print(f"Device: {DEVICE}  AMP: {use_amp}")
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_ds = FrameDataset("train", augment=True)
    val_ds   = FrameDataset("val",   augment=False)

    if args.smoke:
        train_ds.samples = train_ds.samples[:8]
        val_ds.samples   = val_ds.samples[:8]

    train_loader = DataLoader(train_ds, batch_size=BATCH_VIDEOS, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True, persistent_workers=args.workers > 0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_VIDEOS, shuffle=False,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0)

    model = LoRACLIPDetector().to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} "
          f"({100*trainable/total:.2f}%)")

    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "head" not in n]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "head" in n]
    optimizer = torch.optim.AdamW(
        [{"params": lora_params, "lr": LR_LORA},
         {"params": head_params, "lr": LR_HEAD}],
        weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    hist, best_auc = [], 0.0
    steps = 2 if args.smoke else len(train_loader)

    for epoch in range(args.epochs):
        model.train()
        run_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}",
                    total=steps)
        for i, (x, y) in enumerate(pbar):
            if args.smoke and i >= steps:
                break
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", enabled=use_amp):
                out = model(x)
                loss = weighted_bce(out, y) / ACCUM_STEPS
            if not torch.isfinite(loss):
                # skip the bad step rather than let NaN propagate into weights
                optimizer.zero_grad()
                continue
            scaler.scale(loss).backward()
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            run_loss += loss.item() * ACCUM_STEPS
            pbar.set_postfix(loss=f"{run_loss/(i+1):.4f}")

        scheduler.step()
        auc, bal, f1 = evaluate(model, val_loader, use_amp)
        print(f"  val ROC-AUC={auc:.4f}  BalAcc={bal:.4f}  F1={f1:.4f}")
        hist.append({"epoch": epoch + 1, "train_loss": run_loss / steps,
                     "val_auc": auc, "val_bal": bal, "val_f1": f1})
        with open(os.path.join(HERE, "lora_history.json"), "w") as fp:
            json.dump(hist, fp, indent=2)

        if auc > best_auc and not args.smoke:
            best_auc = auc
            torch.save({"clip_lora": model.clip.state_dict(),
                        "head": model.head.state_dict(),
                        "epoch": epoch + 1, "val_auc": auc},
                       os.path.join(HERE, "lora_best.pt"))
            print(f"  *** saved lora_best.pt (AUC={auc:.4f}) ***")

    print("\nDone." if not args.smoke else "\nSmoke test passed.")


if __name__ == "__main__":
    main()
