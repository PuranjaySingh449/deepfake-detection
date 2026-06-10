# Deepfake Video Detection — CLIP-ViT + Temporal Transformer

Detect deepfake videos by encoding each frame's face with a **frozen CLIP-ViT-B/16**
and modelling the temporal sequence of those embeddings with a lightweight
**Transformer**. The design is built around **cross-dataset generalization** — staying
accurate on deepfakes from sources never seen during training, which is the main
failure mode of most detectors.

---

## Results

Trained on a balanced 8,000-video corpus and evaluated on a held-out internal test set
plus **SDFVD**, an entirely separate dataset used as an unseen cross-dataset benchmark.

### Internal test set (held-out)

| Metric | Score |
|---|---|
| Accuracy | 0.886 |
| Balanced Accuracy | 0.886 |
| ROC-AUC | 0.957 |
| PR-AUC | 0.952 |
| Fake Precision | 0.890 |
| Fake Recall | 0.874 |
| Specificity | 0.897 |
| F1 (Fake) | 0.882 |

### SDFVD external validation (cross-dataset, 106 unseen clips)

| Metric | Score |
|---|---|
| Accuracy | 0.698 |
| Balanced Accuracy | 0.698 |
| ROC-AUC | 0.758 |
| Fake Precision | 0.818 |
| Fake Recall | 0.509 |
| Specificity | 0.887 |
| F1 (Fake) | 0.628 |

Optimal SDFVD decision threshold: **0.15** (balanced accuracy 0.698). The model
generalizes to unseen deepfakes while holding strong precision on real video.

Result figures are in the repo: `comparison_internal.png`, `comparison_sdfvd.png`,
`confusion_matrices.png`, `training_curves.png`, `threshold_analysis.png`,
`roc_auc_curve.png`, `sota_comparison_table.png`.

---

## Pipeline

The project runs in four stages. Stages 1–3 require the raw video datasets and a CUDA GPU;
stage 4 trains on cached features.

| Stage | Script | What it does | Output |
|------|--------|-------------|--------|
| 1. Index | `create_split.py` | Scans the source datasets, samples target counts, builds an **identity-aware** 70/15/15 split (no identity leaks across splits) | `master_metadata.csv` |
| 2. Faces | `extract.py` | Samples 15 frames/video → **MTCNN** face detection → 224×224 crops | face-crop folders |
| 3. Features | `extract_features.py` | Frozen **CLIP-ViT-B/16** → caches a 768-dim CLS embedding per frame | `vit_features.npy` per video |
| 4. Train + Eval | `train.py` | Trains the Temporal Transformer, calibrates thresholds, evaluates internally + on SDFVD | `best_model.pth`, plots, `.npz` |

`plot_results.py` renders the result figures from the saved `.npz` files.

---

## Dataset

Built from **25 source datasets** (FaceForensics++, Celeb-DF, DFD, DeeperForensics variants),
sampled and balanced:

- **8,000 videos** — 4,000 real / 4,000 fake
- **1,442 unique identities**
- Split: **train 5,370 / val 1,412 / test 1,218**

**SDFVD** (Small-scale Deepfake Forgery Video Dataset, 106 clips: 53 real / 53 fake)
is held out entirely as an external cross-dataset benchmark.

---

## Model (`train.py`)

- **Input:** 30-frame sequence of 768-dim CLIP-ViT CLS embeddings
- **Architecture:** learnable CLS token + positional embeddings → 4-layer pre-norm
  Transformer encoder (8 heads, GELU, feed-forward 2048) → LayerNorm/MLP head → 1 logit
- **Loss:** `WeightedBCE` (real 1.478 / fake 0.756) with 0.05 label smoothing
- **Optimizer:** AdamW (lr 5e-4, weight decay 5e-4) + cosine annealing, gradient clipping
- **Regularization / augmentation:** dropout 0.3, **mixup** (50% of batches), Gaussian
  noise, temporal flip, feature dropout, temporal crop
- **Early stopping** on validation ROC-AUC (patience 10)

Training ran 39 epochs, **best epoch 29** (val ROC-AUC ≈ 0.955).

---

## How to Run

```bash
# 1. Environment (Python 3.12, CUDA GPU required for stages 1-3)
python -m venv deepfake_env
deepfake_env\Scripts\activate          # Windows
pip install torch torchvision transformers facenet-pytorch \
            opencv-python pillow scikit-learn pandas numpy matplotlib tqdm

# 2. Build the dataset index (edit dataset paths in create_split.py first)
python create_split.py

# 3. Extract faces  ->  4. Cache CLIP features  ->  5. Train + evaluate
python extract.py
python extract_features.py
python train.py

# 6. Regenerate figures
python plot_results.py
```

### Paths to configure

The scripts use machine-specific absolute paths that must be edited before running:

- `create_split.py` — `DATASETS` dict points at source videos on `E:\...`
- `extract.py` — `OUTPUT_DIR = C:\deepfake_faces` (face-crop cache)
- `train.py` / `extract_features.py` — `DATASET_PATH = deepfake_faces`, `SDFVD_PATH = SDFVD/SDFVD`

---

## Not Included in the Repo

To keep the repository lightweight, the following are git-ignored:

- `deepfake_env/` — the Python virtual environment
- `SDFVD/` — the external test videos
- `deepfake_faces/` and the face-crop / `vit_features.npy` cache
- `best_model.pth` — trained weights (~87 MB); regenerate with `train.py`

---

## Repo Contents

```
create_split.py          # Stage 1: identity-aware dataset split
extract.py               # Stage 2: MTCNN face extraction
extract_features.py      # Stage 3: cache CLIP-ViT embeddings
train.py                 # Stage 4: train + evaluate the Temporal Transformer
plot_results.py          # render result figures
master_metadata.csv      # dataset index (8,000 videos)
training_history.npz      # saved training curves
sdfvd_threshold_sweep.npz # saved SDFVD threshold sweep
*.png                    # result figures
```
