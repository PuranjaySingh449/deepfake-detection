# =========================================================
# STEP 1: Extract and cache CLIP-ViT CLS embeddings
# Uses openai/clip-vit-base-patch16 — much better
# cross-dataset generalization than ImageNet ViT.
# Saves vit_features.npy alongside each video folder.
# Run once before train.py.
# =========================================================

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import CLIPVisionModel, CLIPImageProcessor

DATASET_PATH   = "deepfake_faces"
CLIP_MODEL     = "openai/clip-vit-base-patch16"
BATCH_SIZE     = 64
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"\nLoading CLIP-ViT: {CLIP_MODEL} ...")
processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL)
model     = CLIPVisionModel.from_pretrained(CLIP_MODEL).to(DEVICE)
model.eval()
for p in model.parameters():
    p.requires_grad = False
print("CLIP-ViT loaded. Feature dim: 768")

# =========================================================
# Walk every video folder and extract features
# =========================================================

processed = skipped = errors = 0

for split in ["train", "val", "test"]:
    for label in ["real", "fake"]:
        base = os.path.join(DATASET_PATH, split, label)
        if not os.path.exists(base):
            continue

        for dataset_name in os.listdir(base):
            dp = os.path.join(base, dataset_name)
            if not os.path.isdir(dp):
                continue

            for identity in os.listdir(dp):
                ip = os.path.join(dp, identity)
                if not os.path.isdir(ip):
                    continue

                for video_folder in tqdm(
                    os.listdir(ip),
                    desc=f"{split}/{label}/{dataset_name}/{identity}",
                    leave=False
                ):
                    vp        = os.path.join(ip, video_folder)
                    feat_path = os.path.join(vp, "vit_features.npy")

                    if not os.path.isdir(vp):
                        continue

                    # Always re-extract (CLIP replaces old ViT features)
                    frame_files = sorted([
                        f for f in os.listdir(vp)
                        if f.endswith(".jpg")
                    ])

                    if len(frame_files) < 5:
                        continue

                    # Load frames as PIL images
                    pil_imgs = []
                    for fname in frame_files:
                        try:
                            img = Image.open(
                                os.path.join(vp, fname)
                            ).convert("RGB")
                            pil_imgs.append(img)
                        except Exception:
                            continue

                    if len(pil_imgs) < 5:
                        errors += 1
                        continue

                    # Extract in batches using CLIP processor
                    all_feats = []
                    for i in range(0, len(pil_imgs), BATCH_SIZE):
                        batch_imgs = pil_imgs[i:i + BATCH_SIZE]
                        inputs = processor(
                            images=batch_imgs,
                            return_tensors="pt"
                        ).to(DEVICE)

                        with torch.no_grad():
                            out = model(**inputs)

                        # CLS token = pooler_output (B, 768)
                        cls = out.pooler_output
                        all_feats.append(cls.cpu().numpy())

                    feats = np.concatenate(all_feats, axis=0)
                    np.save(feat_path, feats.astype(np.float32))
                    processed += 1

print(f"\nDone.")
print(f"  Processed : {processed}")
print(f"  Errors    : {errors}")
