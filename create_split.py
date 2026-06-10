import random
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# =====================================================
# SETTINGS
# =====================================================

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]

RANDOM_SEED = 42

OUTPUT_CSV = "master_metadata.csv"

random.seed(RANDOM_SEED)

# =====================================================
# DATASET CONFIGURATION
# =====================================================

DATASETS = {

    # =========================
    # REAL DATASETS
    # =========================

    "FFPP_REAL": {
        "path": r"E:\FaceForencis++\original",
        "label": "real",
        "target": 1000
    },

    "CELEB_REAL": {
        "path": r"E:\celeb deepfake\Celeb-real",
        "label": "real",
        "target": 500
    },

    "YOUTUBE_REAL": {
        "path": r"E:\celeb deepfake\YouTube-real",
        "label": "real",
        "target": 300
    },

    "DFD_REAL": {
        "path": r"E:\DFD\DFD_original sequences",
        "label": "real",
        "target": 300
    },

    # =========================
    # DEEPERFORENSICS REAL
    # =========================

    "DF_SOURCE_01": {
        "path": r"E:\source_videos_part_01",
        "label": "real",
        "target": 220
    },

    "DF_SOURCE_02": {
        "path": r"E:\source_videos_part_02",
        "label": "real",
        "target": 210
    },

    "DF_SOURCE_03": {
        "path": r"E:\source_videos_part_03",
        "label": "real",
        "target": 210
    },

    "DF_SOURCE_04": {
        "path": r"E:\source_videos_part_04",
        "label": "real",
        "target": 210
    },

    "DF_SOURCE_05": {
        "path": r"E:\source_videos_part_05",
        "label": "real",
        "target": 210
    },

    "DF_SOURCE_06": {
        "path": r"E:\source_videos_part_06",
        "label": "real",
        "target": 210
    },

    "DF_SOURCE_07": {
        "path": r"E:\source_videos_part_07",
        "label": "real",
        "target": 210
    },

    "DF_SOURCE_08": {
        "path": r"E:\source_videos_part_08",
        "label": "real",
        "target": 210
    },

    "DF_SOURCE_09": {
        "path": r"E:\source_videos_part_09",
        "label": "real",
        "target": 210
    },

    # =========================
    # FF++ FAKE
    # =========================

    "DEEPFAKES": {
        "path": r"E:\FaceForencis++\Deepfakes",
        "label": "fake",
        "target": 350
    },

    "FACE2FACE": {
        "path": r"E:\FaceForencis++\Face2Face",
        "label": "fake",
        "target": 350
    },

    "FACESHIFTER": {
        "path": r"E:\FaceForencis++\FaceShifter",
        "label": "fake",
        "target": 350
    },

    "FACESWAP": {
        "path": r"E:\FaceForencis++\FaceSwap",
        "label": "fake",
        "target": 350
    },

    "NEURALTEXTURES": {
        "path": r"E:\FaceForencis++\NeuralTextures",
        "label": "fake",
        "target": 400
    },

    # =========================
    # CELEBDF FAKE
    # =========================

    "CELEB_SYNTHESIS": {
        "path": r"E:\celeb deepfake\Celeb-synthesis",
        "label": "fake",
        "target": 1000
    },

    # =========================
    # DFD FAKE
    # =========================

    "DFD_FAKE": {
        "path": r"E:\DFD\DFD_manipulated_sequences",
        "label": "fake",
        "target": 300
    },

    # =========================
    # DEEPERFORENSICS FAKE
    # =========================

    "DF_MANIPULATED_01": {
        "path": r"E:\manipulated_videos_part_01",
        "label": "fake",
        "target": 200
    },

    "DF_MANIPULATED_02": {
        "path": r"E:\manipulated_videos_part_02",
        "label": "fake",
        "target": 180
    },

    "DF_MANIPULATED_03": {
        "path": r"E:\manipulated_videos_part_03",
        "label": "fake",
        "target": 180
    },

    "DF_MANIPULATED_04": {
        "path": r"E:\manipulated_videos_part_04",
        "label": "fake",
        "target": 170
    },

    "DF_MANIPULATED_05": {
        "path": r"E:\manipulated_videos_part_05",
        "label": "fake",
        "target": 170
    },
}

# =====================================================
# VIDEO FINDER
# =====================================================

def extract_identity(video_path):

    try:

        for part in video_path.parts:

            if part.startswith("M") or part.startswith("W"):
                return part

        return video_path.stem.split("_")[0]

    except:
        return video_path.stem

# =====================================================
# LOAD DATASET
# =====================================================

all_data = []

for dataset_name, info in DATASETS.items():

    dataset_path = Path(info["path"])

    label = info["label"]

    target = info["target"]

    print(f"\nScanning: {dataset_name}")

    videos = []

    for ext in VIDEO_EXTENSIONS:

        videos.extend(
            list(dataset_path.rglob(f"*{ext}"))
        )

    print(f"Found {len(videos)} videos")

    random.shuffle(videos)

    videos = videos[:target]

    print(f"Selected {len(videos)} videos")

    for video in videos:

        identity = extract_identity(video)

        all_data.append({
            "video_path": str(video),
            "label": label,
            "identity": identity,
            "dataset": dataset_name
        })

# =====================================================
# CREATE DATAFRAME
# =====================================================

df = pd.DataFrame(all_data)

# =====================================================
# IDENTITY SPLIT
# =====================================================

identities = df["identity"].unique()

train_ids, temp_ids = train_test_split(
    identities,
    test_size=0.30,
    random_state=RANDOM_SEED
)

val_ids, test_ids = train_test_split(
    temp_ids,
    test_size=0.50,
    random_state=RANDOM_SEED
)

def assign_split(identity):

    if identity in train_ids:
        return "train"

    elif identity in val_ids:
        return "val"

    else:
        return "test"

df["split"] = df["identity"].apply(assign_split)

# =====================================================
# SAVE CSV
# =====================================================

df.to_csv(OUTPUT_CSV, index=False)

# =====================================================
# FINAL STATS
# =====================================================

print("\n===================================")
print("FINAL DATASET")
print("===================================")

print("\nSplit Counts:")
print(df["split"].value_counts())

print("\nLabel Counts:")
print(df["label"].value_counts())

print("\nDataset Counts:")
print(df["dataset"].value_counts())

print("\nSaved:", OUTPUT_CSV)