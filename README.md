# Automated Brain Tumour Detection

[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Model-yellow?style=flat&logo=huggingface)](https://huggingface.co/Zorrojurro/brain-tumor-cnn-vit)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Hybrid **CNN + Vision Transformer** pipeline for classifying brain MRI scans into four categories (glioma, meningioma, no tumour, pituitary), with optional radiomics fusion, self-supervised pre-training hooks, and **Grad-CAM** explanations.

---

## Live model (Hugging Face)

The trained hybrid CNN–ViT weights and model card are hosted here:

**[https://huggingface.co/Zorrojurro/brain-tumor-cnn-vit](https://huggingface.co/Zorrojurro/brain-tumor-cnn-vit)**

Use the model card to try inference in the browser (Inference API widget), read metrics, and clone or download artifacts.

---

## Highlights

| Area | Details |
|------|---------|
| Architecture | ResNet50 CNN features → patch embedding → 6-layer ViT (8 heads) + radiomics branch + fusion classifier |
| Classes | Glioma, Meningioma, No tumour, Pituitary |
| Explainability | Grad-CAM overlays for spatial attribution |
| Training | Config-driven (`config/config.yaml`), optional SSL and augmentation |

---

## Repository layout

```
Automated Brain Tumor Detection/
├── config/config.yaml       # Hyperparameters and paths
├── src/
│   ├── data/                # Loading, preprocessing, augmentation
│   ├── models/              # Hybrid CNN-ViT and helpers
│   ├── training/            # Trainer
│   ├── evaluation/          # Metrics
│   └── explainability/      # Grad-CAM
├── scripts/                 # train, evaluate, inference, CV, ablations
├── notebooks/               # Experiments and exploration
├── results/                 # Saved metrics (e.g. cross-validation JSON)
├── app.py                   # Local Gradio UI (needs checkpoints)
├── requirements.txt
└── README.md
```

---

## Results (5-fold cross-validation)

Summaries below are from `results/cross_validation_results.json` (5 folds, 30 epochs per fold).

| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | **98.95%** | ±0.27 |
| F1 (weighted) | **98.95%** | ±0.26 |
| ROC-AUC | **99.92%** | ±0.04 |

These figures reflect the experimental setup recorded in that run; your numbers may differ slightly if you change data splits or seeds.

---

## Architecture ablation — does the hybrid earn its complexity?

`scripts/ablation_study.py` compares the hybrid CNN-ViT against a plain CNN, a plain
ViT, and an EfficientNet-B0 baseline (single split, 20 epochs each, seed 42; results
in `results/ablation_study_results.json`). This is a separate, smaller run from the
5-fold CV numbers above — reported here to show what the extra architectural
complexity actually buys:

| Model | Accuracy | F1 | ROC-AUC | Params |
|---|---|---|---|---|
| CNN only (ResNet50) | 97.64% | 97.62% | 99.71% | 24.6M |
| ViT only (ViT-B/16) | 98.43% | 98.43% | 99.94% | 85.8M |
| EfficientNet-B0 baseline | 98.86% | 98.86% | 99.93% | 4.0M |
| **Hybrid CNN-ViT (ours)** | **99.13%** | **99.13%** | 99.83% | 47.0M |

Read honestly: EfficientNet-B0 is a strong, far cheaper baseline (4M params vs. 47M)
that comes within ~0.3% accuracy of the hybrid. The hybrid still edges out every
single-architecture baseline on accuracy and F1, which is the basis for using it as
the primary model — but the margin over EfficientNet-B0 is small enough that a
parameter-efficiency-constrained deployment could reasonably prefer it instead.

---

## Quick start

### 1. Environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Data

Download the [Brain Tumor MRI dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and extract it so `config/config.yaml` paths resolve (for example `data/raw/` or the layout your `raw_dir` points to). Training images are not stored in this repository.

### 3. Train

```bash
python scripts/train.py --config config/config.yaml
```

Optional pre-training flag (if enabled in config):

```bash
python scripts/train.py --pretrain --config config/config.yaml
```

### 4. Evaluate & infer

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
python scripts/inference.py --image path/to/mri.jpg --checkpoint checkpoints/best_model.pth
```

Cross-validation:

```bash
python scripts/cross_validation.py
```

### 5. Local web UI (Gradio)

After you have `checkpoints/best_model.pth` in place:

```bash
python app.py
```

---

## Hugging Face integration

This repo aligns with the public model **`Zorrojurro/brain-tumor-cnn-vit`**. The `huggingface/` directory contains helpers for packaging and uploads (e.g. `upload_model.py`, `upload_space.py`) if you maintain a Space alongside the model card.

---

## Disclaimer

This software is for **research and educational use**. It is **not** a medical device and must not replace professional diagnosis.

---

## Citation

If you use this project in research, please cite the repository and the Hugging Face model card as appropriate.

---

## License

MIT License — see [`LICENSE`](LICENSE).
