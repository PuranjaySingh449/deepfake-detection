"""
Single-video deepfake inference demo.

Usage:
    python predict.py path/to/video.mp4
    python predict.py path/to/video.mp4 --threshold 0.15 --no-face

Loads the trained CLS-token Temporal Transformer (best_model.pth), samples
frames from the video, runs each through frozen CLIP-ViT-B/16, and outputs a
real/fake decision with a confidence score.

By default it crops faces with MTCNN (matching the training pipeline) and falls
back to the full frame when no face is found. Pass --no-face to feed raw frames
(this is how the original SDFVD cross-dataset eval was run).
"""

import os
import sys
import argparse

import numpy as np
import torch
from PIL import Image

from model import TemporalTransformer, SEQUENCE_LENGTH

CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), "..", "best_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sample_frames(video_path, n=SEQUENCE_LENGTH):
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Could not read frames from {video_path}")
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


def crop_faces(pil_frames, image_size=224, margin=40):
    """MTCNN face crop with full-frame fallback."""
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(image_size=image_size, margin=margin, keep_all=False,
                  post_process=False, device=DEVICE)
    out = []
    for img in pil_frames:
        face = mtcnn(img)
        if face is None:
            out.append(img)  # fallback: full frame, CLIP processor will resize
        else:
            arr = face.permute(1, 2, 0).byte().cpu().numpy()
            out.append(Image.fromarray(arr))
    del mtcnn
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


@torch.no_grad()
def predict(video_path, ckpt=DEFAULT_CKPT, threshold=0.5, use_face=True):
    from transformers import CLIPVisionModel, CLIPImageProcessor

    frames = sample_frames(video_path)
    if use_face:
        frames = crop_faces(frames)

    processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip = CLIPVisionModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE).eval()
    inputs = processor(images=frames, return_tensors="pt").to(DEVICE)
    feats = clip(**inputs).pooler_output.unsqueeze(0)  # (1, T, 768)
    del clip
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    model = TemporalTransformer().to(DEVICE).eval()
    state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)

    prob = torch.sigmoid(model(feats)).item()
    label = "FAKE" if prob > threshold else "REAL"
    confidence = prob if label == "FAKE" else 1 - prob
    return label, prob, confidence


def main():
    ap = argparse.ArgumentParser(description="Single-video deepfake detector")
    ap.add_argument("video", help="path to a video file")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="model checkpoint")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="fake decision threshold (SDFVD-calibrated: 0.15)")
    ap.add_argument("--no-face", action="store_true",
                    help="skip MTCNN, feed raw frames")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        sys.exit(f"Video not found: {args.video}")

    label, prob, conf = predict(args.video, args.ckpt, args.threshold,
                                use_face=not args.no_face)

    print("\n" + "=" * 40)
    print(f"  Video:      {os.path.basename(args.video)}")
    print(f"  Prediction: {label}")
    print(f"  Fake prob:  {prob:.4f}  (threshold {args.threshold})")
    print(f"  Confidence: {conf:.1%}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
