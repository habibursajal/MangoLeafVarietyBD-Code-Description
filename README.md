# MangoLeafVarietyBD — Code Repository

This repository contains the complete pipeline code for the dataset described in:

> **MangoLeafVarietyBD: A High-Resolution Leaf Image Dataset for Multi-Class Mango Cultivar Classification in Bangladesh**
> Habibur Rahman Sajal, Mahfuzur Rahman, Mohammad Monirul Islam
> *Scientific Data* (2026)

**Dataset:** https://doi.org/10.17632/hb3kvgfcvm.3

---

## Overview

The pipeline reproduces the technical validation reported in the manuscript. It trains and evaluates two convolutional networks — **VGG16** and **MobileNetV4-Small** — on a source-image-level grouped stratified partition of the MangoLeafVarietyBD dataset, free of data leakage.

| Model | Test Accuracy | Test F1 |
|-------|--------------|---------|
| VGG16 | 99.28 % | 0.9928 |
| MobileNetV4-Small | 98.07 % | 0.9807 |

---

## Pipeline Scripts

Run the scripts **in order** (Part 1 → Part 6):

| Script | Description |
|--------|-------------|
| `part1_setup.py` | Configuration, paths, random seed, progress tracking |
| `part2_split.py` | Grouped stratified split (70/15/15) at source-image level |
| `part3_augmentation.py` | Offline augmentation — training images only (6 variants each) |
| `part4_preprocessing.py` | Background removal (U²-Net via rembg), resize 224×224, JPEG 95% |
| `part5_train.py` | Train VGG16 and MobileNetV4-Small |
| `part6_evaluation.py` | Test set evaluation, confusion matrices, training curves |

---

## Requirements

```
Python          3.14
PyTorch         2.11.0+cu128
torchvision     0.26.0+cu128
timm            (latest)
albumentations  2.0.8
rembg           2.0.88
onnxruntime-gpu 1.32.0
scikit-learn    1.9.1
opencv-python   5.0.0.93
matplotlib
pandas
numpy
```

Install dependencies:

```bash
pip install torch torchvision timm albumentations rembg onnxruntime-gpu scikit-learn opencv-python matplotlib pandas numpy
```

---

## Hardware

Experiments were conducted on:
- **GPU:** NVIDIA GeForce RTX 3060 (12 GB VRAM)
- **CUDA:** 12.8

---

## Dataset Structure

Download the dataset from Mendeley Data and place it as follows:

```
MangoLeafVarietyBD/
├── Raw Images/
│   ├── Amrapali/
│   ├── Banana/
│   ├── Bandigori/
│   ├── Brunei King/
│   ├── Harivanga/
│   ├── Himsagar/
│   ├── Kacha Mitha/
│   └── Surjapuri/
├── provenance.csv
└── README.txt
```

Update the paths in `part1_setup.py` to match your local directory.

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate | 1 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| Scheduler | CosineAnnealingLR |
| Max epochs | 30 |
| Early stopping patience | 7 |
| Label smoothing | 0.1 |
| Input size | 224 × 224 |
| Mixed precision | AMP float16 |
| Random seed | 42 |

---

## Augmentation Transforms (Training Only)

Each training image is expanded into 6 deterministic variants using [Albumentations](https://albumentations.ai/):

| Variant | Transforms |
|---------|-----------|
| 1 | Rotation ±15° + Horizontal flip + Brightness/contrast |
| 2 | Rotation ±30° + Gaussian blur (σ 0.5–1.0) + Brightness/contrast |
| 3 | Rotation ±45° + Vertical flip + Brightness/contrast |
| 4 | Shear ±10–20° + Horizontal flip + Brightness/contrast |
| 5 | Rotation ±15° + Shear ±10° + Gaussian blur |
| 6 | Rotation ±30° + Brightness/contrast + Horizontal flip |

Brightness scaling: 0.6–1.4× | Contrast scaling: 0.7–1.3×

**Validation and test images are not augmented.**

---

## Important: Avoiding Data Leakage

The dataset is first split at **source-image level** before augmentation. This ensures that all augmented variants of any original photograph remain within a single partition (train, validation, or test). Random partitioning of an augmented pool would allow similar images to appear on both sides of a partition boundary, inflating reported accuracy.

---

## Citation

If you use this code or dataset, please cite:

```
Sajal, Habibur Rahman; Rahman, Mahfuzur; Islam, Mohammad Monirul (2026),
"MangoLeafVarietyBD: An Image Dataset for Multi-Class Mango Variety
Classification in Bangladesh", Mendeley Data, V3,
doi: 10.17632/hb3kvgfcvm.3
```

---

## Contact

Habibur Rahman Sajal
Department of Computer Science and Engineering
Daffodil International University, Dhaka, Bangladesh
📧 sojol15-5904@diu.edu.bd
