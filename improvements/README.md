# Improvements — pushing cross-dataset generalization

This folder contains follow-up work on the base project (`../`). It targets the
one weakness of the original model: the gap between in-distribution performance
(internal ROC-AUC 0.957) and true cross-dataset performance on SDFVD.

## TL;DR — the headline result

The original SDFVD evaluation fed **raw video frames** into a model that was
**trained on MTCNN face crops**. That preprocessing mismatch — not the model —
was the main reason cross-dataset numbers looked weak. Cropping faces at
inference (matching training) recovers most of the gap, **with no retraining**:

| Inference strategy | ROC-AUC | Balanced Acc | Accuracy | F1 (fake) |
|---|---|---|---|---|
| Raw frames (original eval) | 0.758 | 0.698 | 0.698 | 0.628 |
| Horizontal-flip TTA | 0.754 | 0.679 | 0.679 | 0.614 |
| **MTCNN face crops** | **0.900** | **0.840** | **0.840** | **0.828** |
| Raw + flip + face ensemble | 0.874 | 0.840 | 0.840 | 0.838 |

**+0.14 AUC and +0.14 accuracy** from a correct, legitimate fix. Raw numbers in
`sdfvd_tta_results.json`. Reproduce with `python eval_sdfvd.py`.

> Takeaway: always run inference through the **same** face pipeline used in
> training. The face-crop result is the honest cross-dataset number to report.

## Contents

| File | Purpose |
|---|---|
| `model.py` | Shared `TemporalTransformer` (byte-compatible with `../train.py`, loads `best_model.pth`) |
| `predict.py` | Single-video demo → real/fake + confidence (face crops by default) |
| `eval_sdfvd.py` | SDFVD eval with face-crop / flip / ensemble TTA → `sdfvd_tta_results.json` |
| `train_lora.py` | LoRA fine-tuning of CLIP-ViT (the high-ceiling experiment) |
| `requirements.txt` | Pinned dependencies (Python 3.12, CUDA 11.8) |

## Demo

```bash
# face-crop pipeline (recommended, matches training)
python predict.py ../SDFVD/SDFVD/videos_fake/vs1.mp4

# raw frames (how the original SDFVD eval ran)
python predict.py ../SDFVD/SDFVD/videos_real/v1.mp4 --no-face --threshold 0.15
```

## LoRA fine-tuning (`train_lora.py`)

The base model **freezes** CLIP and trains only a temporal head. This experiment
instead attaches LoRA adapters to CLIP's attention projections and trains them
jointly with the head on the face-crop JPGs — the documented route from
frozen-feature (~0.80 AUC) toward fine-tuned (~0.90 AUC) territory.

Tuned for a 6 GB GPU (RTX 3050): LoRA-only (≈21% of params trainable), mixed
precision, gradient checkpointing, batch of 2 videos × 15 frames with gradient
accumulation (effective batch 8). ≈70 min/epoch.

```bash
python train_lora.py --smoke          # 2-step wiring/memory check
python train_lora.py --epochs 6        # full run
```

Outputs: `lora_best.pt` (best adapters+head by val AUC) and `lora_history.json`.

### Status
A 6-epoch run was launched overnight. Check `lora_history.json` /
`lora_train.log` for progress, and `lora_best.pt` for the best checkpoint. To
benchmark a finished LoRA model on SDFVD, load its `head` + LoRA-adapted CLIP and
run the face-crop path from `eval_sdfvd.py`.

## Next steps (not yet done)

- Self-Blended Images (SBI) style augmentation — the highest-ceiling generalization trick.
- Hybrid frequency-domain cues alongside CLIP features.
- Swap CLIP-ViT-B/16 → L/14 once a larger GPU is available.
