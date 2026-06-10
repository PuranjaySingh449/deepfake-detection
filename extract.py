import os
import cv2
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from facenet_pytorch import MTCNN

# =========================================================
# SETTINGS
# =========================================================

CSV_PATH = "master_metadata.csv"

# OUTPUT LOCATION
OUTPUT_DIR = r"C:\deepfake_faces"

# TEMPORAL SETTINGS
FRAMES_PER_VIDEO = 15

# FACE SETTINGS
IMAGE_SIZE = 224
MARGIN = 40

# JPEG QUALITY
JPEG_QUALITY = 90

# VIDEO TYPES
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]

# =========================================================
# GPU CHECK
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("\n===================================")
print("GPU STATUS")
print("===================================")

print("Using device:", device)

if device != "cuda":
    raise Exception("CUDA GPU NOT detected.")

print("GPU Name:", torch.cuda.get_device_name(0))

# =========================================================
# FACE DETECTOR
# =========================================================

print("\nLoading MTCNN face detector...")

mtcnn = MTCNN(
    image_size=IMAGE_SIZE,
    margin=MARGIN,
    keep_all=False,
    post_process=False,
    device=device
)

print("Face detector loaded successfully.")

# =========================================================
# LOAD METADATA
# =========================================================

print("\nLoading metadata CSV...")

df = pd.read_csv(CSV_PATH)

print(f"Total videos in CSV: {len(df)}")

# =========================================================
# FRAME SAMPLING FUNCTION
# =========================================================

def sample_frames(total_frames, num_frames):

    if total_frames <= num_frames:
        return list(range(total_frames))

    interval = total_frames / num_frames

    return [int(i * interval) for i in range(num_frames)]

# =========================================================
# FACE EXTRACTION
# =========================================================

print("\n===================================")
print("STARTING EXTRACTION")
print("===================================\n")

processed = 0
failed = 0
skipped = 0

for idx, row in tqdm(df.iterrows(), total=len(df)):

    video_path = row["video_path"]

    label = row["label"]

    split = row["split"]

    dataset = row["dataset"]

    identity = row["identity"]

    video_name = Path(video_path).stem

    # =====================================================
    # SAVE DIRECTORY
    # =====================================================

    save_dir = Path(
        OUTPUT_DIR,
        split,
        label,
        dataset,
        identity,
        video_name
    )

    save_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # SKIP IF ALREADY DONE
    # =====================================================

    existing_frames = list(save_dir.glob("*.jpg"))

    if len(existing_frames) >= FRAMES_PER_VIDEO:
        skipped += 1
        continue

    # =====================================================
    # OPEN VIDEO
    # =====================================================

    try:

        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            failed += 1
            continue

        frame_indices = sample_frames(
            total_frames,
            FRAMES_PER_VIDEO
        )

        saved_frames = 0

        # =================================================
        # EXTRACT TEMPORAL FRAMES
        # =================================================

        for frame_idx in frame_indices:

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            success, frame = cap.read()

            if not success:
                continue

            # =============================================
            # CONVERT BGR -> RGB
            # =============================================

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # =============================================
            # FACE DETECTION
            # =============================================

            face = mtcnn(frame_rgb)

            if face is None:
                continue

            # =============================================
            # TORCH -> NUMPY
            # =============================================

            face = face.permute(1, 2, 0)

            face = face.byte().cpu().numpy()

            # =============================================
            # SAVE FRAME
            # =============================================

            output_path = save_dir / f"frame_{saved_frames:03d}.jpg"

            cv2.imwrite(
                str(output_path),
                cv2.cvtColor(face, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )

            saved_frames += 1

        cap.release()

        # =================================================
        # VALIDATION
        # =================================================

        if saved_frames < 5:

            failed += 1

            print(f"\nWARNING: Very few faces extracted")
            print(video_path)
            print(f"Saved: {saved_frames}")

        else:
            processed += 1

    except Exception as e:

        failed += 1

        print("\n===================================")
        print("ERROR PROCESSING VIDEO")
        print("===================================")

        print(video_path)
        print(e)

# =========================================================
# FINAL STATS
# =========================================================

print("\n===================================")
print("EXTRACTION COMPLETE")
print("===================================")

print(f"\nProcessed videos: {processed}")

print(f"Failed videos: {failed}")

print(f"Skipped existing: {skipped}")

print(f"\nSaved dataset at:")

print(OUTPUT_DIR)

print("\nDONE\n")