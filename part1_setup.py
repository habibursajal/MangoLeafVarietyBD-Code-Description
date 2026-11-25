"""
================================================================
MangoLeafVarietyBD — Part 1: Setup & Configuration
================================================================
Dataset : MangoLeafVarietyBD: An Image Dataset for
          Multi-Class Mango Variety Classification
          in Bangladesh
Authors : Habibur Rahman Sajal, Mahfuzur Rahman,
          Mohammad Monirul Islam
Journal : Scientific Data (Springer Nature)
----------------------------------------------------------------
Run this file FIRST before any other part.
Creates all output directories, verifies the renamed dataset
structure, saves config.json, and initialises progress.json.
----------------------------------------------------------------
Usage:
    python part1_setup.py
Next:
    python part2_split.py
================================================================
"""

import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# ================================================================
# PATHS
# ================================================================

# Root folder containing the renamed raw images and all outputs.
BASE = r"E:\MangoLeafVarietyBD Dataset"

# Source: renamed raw images produced by rename_dataset.py
RAW_RENAMED = os.path.join(BASE, "Raw Images Renamed")

# Destination: grouped-stratified split (created by part2_split.py)
SPLIT = os.path.join(BASE, "dataset_split")

# All experiment outputs live under this directory.
OUT = os.path.join(BASE, "results")

MODELS     = os.path.join(OUT, "models")
FIGS       = os.path.join(OUT, "figures")
LOGS       = os.path.join(OUT, "logs")
XAI        = os.path.join(OUT, "xai")

FIG_CM      = os.path.join(FIGS, "confusion_matrix")
FIG_CURVES  = os.path.join(FIGS, "training_curves")
FIG_GRADCAM = os.path.join(FIGS, "gradcam")

# Augmented training images (produced by part3_augmentation.py)
AUG_TRAIN = os.path.join(BASE, "augmented_train")

# Final preprocessed images ready for model input
PREPROCESSED = os.path.join(BASE, "preprocessed")

ALL_DIRS = [
    OUT, MODELS, FIGS, LOGS, XAI,
    FIG_CM, FIG_CURVES, FIG_GRADCAM,
]

# ================================================================
# CONFIGURATION
# ================================================================

CFG = {
    # --- Image & Training ---
    "img_size"   : 224,
    "batch_size" : 32,
    "epochs"     : 30,
    "lr"         : 1e-4,
    "patience"   : 7,

    # --- Reproducibility ---
    "seed"       : 42,

    # --- Split ratios ---
    "train_ratio": 0.70,
    "val_ratio"  : 0.15,
    "test_ratio" : 0.15,

    # --- Dataset ---
    "num_classes": 8,
    "class_names": [
        "Amrapali",
        "Banana",
        "Bandigori",
        "Brunei King",
        "Harivanga",
        "Himsagar",
        "Kacha Mitha",
        "Surjapuri",
    ],

    # --- Camera / site metadata ---
    "device_map": {
        "Amrapali"   : "Samsung Galaxy S24 Ultra",
        "Banana"     : "Samsung Galaxy S24 Ultra",
        "Bandigori"  : "Samsung Galaxy S24 Ultra",
        "Harivanga"  : "Samsung Galaxy S24 Ultra",
        "Himsagar"   : "Samsung Galaxy S24 Ultra",
        "Kacha Mitha": "Samsung Galaxy S24 Ultra",
        "Surjapuri"  : "Samsung Galaxy S24 Ultra",
        "Brunei King": "Google Pixel 8",
    },
    "site_map": {
        "Amrapali"   : "Thakurgaon",
        "Banana"     : "Thakurgaon",
        "Bandigori"  : "Thakurgaon",
        "Harivanga"  : "Thakurgaon",
        "Himsagar"   : "Thakurgaon",
        "Kacha Mitha": "Thakurgaon",
        "Surjapuri"  : "Thakurgaon",
        "Brunei King": "Savar",
    },

    # --- Hardware ---
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # --- Figure DPI ---
    "dpi_fig"   : 600,
    "dpi_xai"   : 1200,
    "dpi_cm"    : 600,

    # --- Models for technical validation ---
    "model_names": [
        "VGG16",
        "MobileNetV4_Small",
    ],

    # --- Augmentation parameters (Albumentations) ---
    "augmentation": {
        "rotation_limits"  : [15, 30, 45],
        "h_flip_p"         : 0.5,
        "v_flip_p"         : 0.1,
        "brightness_range" : [0.6, 1.4],
        "contrast_range"   : [0.7, 1.3],
        "shear_range"      : [10, 20],
        "gauss_blur_sigma" : [0.5, 1.0],
        "variants_per_img" : 6,
    },
}

# ================================================================
# PROGRESS TRACKING
# ================================================================

PROGRESS_FILE = os.path.join(OUT, "progress.json")


def load_progress() -> dict:
    """Load pipeline progress from JSON file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {
        "split_done"     : False,
        "augment_done"   : False,
        "trained_models" : {},
        "eval_done"      : {},
        "figures_done"   : False,
        "xai_done"       : False,
        "summary_done"   : False,
        "started_at"     : datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"),
    }


def save_progress(progress: dict):
    """Persist progress to JSON file."""
    progress["last_updated"] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S")
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def mark_done(key: str, value: bool = True):
    """Mark a pipeline step as complete."""
    p = load_progress()
    p[key] = value
    save_progress(p)


def is_done(key: str) -> bool:
    """Return True if a pipeline step is complete."""
    return bool(load_progress().get(key, False))


def mark_model_done(model_name: str):
    """Mark a model as trained."""
    p = load_progress()
    p.setdefault("trained_models", {})[model_name] = True
    save_progress(p)


def is_model_done(model_name: str) -> bool:
    """
    Return True if a model is already trained.
    Checks both progress.json and the .pth file on disk.
    """
    p   = load_progress()
    pth = os.path.join(MODELS, f"{model_name}.pth")
    return (
        p.get("trained_models", {}).get(model_name, False)
        and os.path.exists(pth)
    )


# ================================================================
# REPRODUCIBILITY
# ================================================================

def set_seed(seed: int):
    """
    Fix all random seeds for full reproducibility.
    Covers Python, NumPy, and PyTorch (CPU + CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ================================================================
# SETUP
# ================================================================

def setup() -> bool:
    """
    Initialise the experiment environment:
      1. Create all output directories.
      2. Detect and report hardware.
      3. Verify the renamed raw-image structure.
      4. Save config.json.
      5. Initialise progress.json.
    Returns True on success, False on any error.
    """
    print("=" * 60)
    print("MangoLeafVarietyBD — Pipeline Setup")
    print("=" * 60)

    # -- Directories --
    print("\n[1/5] Creating output directories...")
    for d in ALL_DIRS:
        os.makedirs(d, exist_ok=True)
        print(f"  OK  {d}")

    # -- Hardware --
    print(f"\n[2/5] Hardware")
    print(f"  Device : {CFG['device']}")
    if CFG["device"] == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {props.total_memory / 1e9:.1f} GB")
    else:
        print("  WARNING: CUDA not available. "
              "Training will be slow on CPU.")

    # -- Raw dataset verification --
    print(f"\n[3/5] Verifying renamed dataset at:")
    print(f"  {RAW_RENAMED}")
    if not os.path.isdir(RAW_RENAMED):
        print(f"  ERROR: Directory not found.")
        print(f"  Run rename_dataset.py first.")
        return False

    total = 0
    for cls in CFG["class_names"]:
        cls_path = os.path.join(RAW_RENAMED, cls)
        if not os.path.isdir(cls_path):
            print(f"  ERROR: Class folder missing: {cls}")
            return False
        count = len(list(Path(cls_path).glob("*.jpg")))
        print(f"  {cls:<15} : {count:>4} images  "
              f"| {CFG['device_map'][cls]:<26} "
              f"| {CFG['site_map'][cls]}")
        total += count
    print(f"  {'TOTAL':<15} : {total:>4} images")

    # -- Save config --
    print(f"\n[4/5] Saving config.json...")
    cfg_path = os.path.join(OUT, "config.json")
    cfg_save = dict(CFG)
    cfg_save["device"] = CFG["device"]
    with open(cfg_path, "w") as f:
        json.dump(cfg_save, f, indent=2)
    print(f"  Saved : {cfg_path}")

    # -- Init progress --
    print(f"\n[5/5] Initialising progress.json...")
    save_progress(load_progress())
    print(f"  Saved : {PROGRESS_FILE}")

    # -- Summary --
    print("\n" + "=" * 60)
    print("Setup complete.")
    print(f"  Models   : {CFG['model_names']}")
    print(f"  Seed     : {CFG['seed']}")
    print(f"  Split    : "
          f"{int(CFG['train_ratio']*100)} / "
          f"{int(CFG['val_ratio']*100)} / "
          f"{int(CFG['test_ratio']*100)}  "
          f"(train / val / test)")
    print("=" * 60)
    print("Next step: python part2_split.py")
    return True


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    ok = setup()
    if not ok:
        raise SystemExit(1)