"""
SDFVD cross-dataset evaluation with test-time augmentation (TTA).

Goal: improve the cross-dataset number WITHOUT retraining, using the existing
best_model.pth. Compares four inference strategies on the 106 SDFVD clips:

    1. baseline   — raw frames, single forward (reproduces the ~0.758 AUC)
    2. flip-TTA   — average prob of raw frames and horizontally-flipped frames
    3. face       — MTCNN face crops (matches training distribution)
    4. ensemble   — mean of raw + flip + face probabilities

For each strategy it reports ROC-AUC and the best balanced-accuracy over a
threshold sweep. Results are written to sdfvd_tta_results.json.
"""

import os
import json

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, accuracy_score, f1_score,
)

from model import TemporalTransformer, SEQUENCE_LENGTH

CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "..", "best_model.pth")
SDFVD_PATH = os.path.join(HERE, "..", "SDFVD", "SDFVD")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def list_videos():
    fake_dir = os.path.join(SDFVD_PATH, "videos_fake")
    real_dir = os.path.join(SDFVD_PATH, "videos_real")
    vids = [(os.path.join(fake_dir, f), 1)
            for f in sorted(os.listdir(fake_dir)) if f.endswith(".mp4")]
    vids += [(os.path.join(real_dir, f), 0)
             for f in sorted(os.listdir(real_dir)) if f.endswith(".mp4")]
    return vids


def read_frames(video_path, n=SEQUENCE_LENGTH):
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    step = max(1, total // n)
    idxs = [min(i * step, total - 1) for i in range(n)]
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            frames.append(frames[-1] if frames else Image.new("RGB", (224, 224)))
            continue
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    while len(frames) < n:
        frames.append(frames[-1])
    return frames[:n]


def sweep(labels, probs):
    labels, probs = np.array(labels), np.array(probs)
    auc = roc_auc_score(labels, probs)
    best_b, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        b = balanced_accuracy_score(labels, (probs > t).astype(float))
        if b > best_b:
            best_b, best_t = b, float(t)
    preds = (probs > best_t).astype(float)
    return {
        "roc_auc": round(float(auc), 4),
        "best_threshold": round(best_t, 2),
        "balanced_acc": round(float(best_b), 4),
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "f1_fake": round(float(f1_score(labels, preds, zero_division=0)), 4),
    }


@torch.no_grad()
def main():
    from transformers import CLIPVisionModel, CLIPImageProcessor
    from facenet_pytorch import MTCNN

    print(f"Device: {DEVICE}")
    processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip = CLIPVisionModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE).eval()
    mtcnn = MTCNN(image_size=224, margin=40, keep_all=False,
                  post_process=False, device=DEVICE)

    model = TemporalTransformer().to(DEVICE).eval()
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))

    def clip_feats(frames):
        inp = processor(images=frames, return_tensors="pt").to(DEVICE)
        f = clip(**inp).pooler_output.unsqueeze(0)        # (1, T, 768)
        return torch.sigmoid(model(f)).item()

    videos = list_videos()
    labels = []
    p_raw, p_flip, p_face = [], [], []

    for vp, label in tqdm(videos, desc="SDFVD"):
        frames = read_frames(vp)
        if frames is None:
            continue
        labels.append(label)

        # raw
        p_raw.append(clip_feats(frames))
        # horizontal flip
        flipped = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in frames]
        p_flip.append(clip_feats(flipped))
        # face crops with full-frame fallback
        faces = []
        for im in frames:
            fc = mtcnn(im)
            faces.append(im if fc is None else
                         Image.fromarray(fc.permute(1, 2, 0).byte().cpu().numpy()))
        p_face.append(clip_feats(faces))

    p_raw, p_flip, p_face = map(np.array, (p_raw, p_flip, p_face))
    strategies = {
        "1_baseline_raw":   p_raw,
        "2_flip_tta":       (p_raw + p_flip) / 2,
        "3_face":           p_face,
        "4_ensemble":       (p_raw + p_flip + p_face) / 3,
    }

    results = {name: sweep(labels, probs) for name, probs in strategies.items()}

    print("\n================ SDFVD TTA RESULTS ================")
    print(f"{'strategy':18s} {'AUC':>7} {'BalAcc':>8} {'Acc':>7} {'F1':>7} {'thr':>5}")
    for name, r in results.items():
        print(f"{name:18s} {r['roc_auc']:>7} {r['balanced_acc']:>8} "
              f"{r['accuracy']:>7} {r['f1_fake']:>7} {r['best_threshold']:>5}")

    out = os.path.join(HERE, "sdfvd_tta_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
