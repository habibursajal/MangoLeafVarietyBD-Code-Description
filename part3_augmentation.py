"""
================================================================
MangoLeafVarietyBD — Part 3: Offline Augmentation
================================================================
Dataset : MangoLeafVarietyBD: An Image Dataset for
          Multi-Class Mango Variety Classification
          in Bangladesh
Authors : Habibur Rahman Sajal, Mahfuzur Rahman,
          Mohammad Monirul Islam
Journal : Scientific Data (Springer Nature)
----------------------------------------------------------------
Applies deterministic offline augmentation to the training
partition only.  Validation and test images are never
augmented.

Strategy
--------
Each original training image is expanded into six
deterministic variants using the Albumentations library.
The transforms follow established practice in image
augmentation and are bounded so that botanical
characteristics are not distorted.

Augmentation transforms applied
--------------------------------
  1. Rotation          : ±15°, ±30°, ±45°
  2. Horizontal flip   : p = 0.50
  3. Vertical flip     : p = 0.10  (sparingly)
  4. Brightness scale  : 0.60× – 1.40×
  5. Contrast scale    : 0.70× – 1.30×
  6. Shear             : ±10° – ±20°
  7. Gaussian blur     : σ = 0.50 – 1.00

A fixed pseudorandom seed makes the expansion fully
reproducible.  Six variants are generated per original
image; the original is also retained, giving seven copies
per source photograph in the training set.

Output
------
augmented_train/
    <Class>/
        <Cultivar>_NNN.jpg          original copy
        <Cultivar>_NNN_aug_1.jpg    variant 1
        ...
        <Cultivar>_NNN_aug_6.jpg    variant 6
----------------------------------------------------------------
Run after : python part2_split.py
Run next  : python part4_dataset.py
================================================================
"""

import os
import shutil
from pathlib import Path

import albumentations as A
import cv2
import pandas as pd
from tqdm import tqdm

from part1_setup import (
    CFG,
    SPLIT,
    BASE,
    is_done,
    mark_done,
    set_seed,
)

# ================================================================
# PATHS
# ================================================================

AUG_TRAIN = os.path.join(BASE, "augmented_train")

# ================================================================
# AUGMENTATION PIPELINE
# ================================================================

def _build_transforms() -> list:
    """
    Return a list of six deterministic Albumentations
    pipelines, one per augmented variant.

    Each pipeline applies a fixed combination of spatial
    and photometric transforms bounded so that leaf
    morphology is not distorted beyond recognition.
    """
    aug = CFG["augmentation"]
    br  = aug["brightness_range"]
    cr  = aug["contrast_range"]
    gb  = aug["gauss_blur_sigma"]

    transforms = [
        # Variant 1: mild rotation + horizontal flip
        A.Compose([
            A.Rotate(limit=aug["rotation_limits"][0],
                     p=1.0),
            A.HorizontalFlip(p=aug["h_flip_p"]),
            A.RandomBrightnessContrast(
                brightness_limit=(br[0] - 1, br[1] - 1),
                contrast_limit=(cr[0] - 1, cr[1] - 1),
                p=0.8),
        ]),
        # Variant 2: moderate rotation + gaussian blur
        A.Compose([
            A.Rotate(limit=aug["rotation_limits"][1],
                     p=1.0),
            A.GaussianBlur(
                blur_limit=(3, 7),
                sigma_limit=(gb[0], gb[1]),
                p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=(br[0] - 1, br[1] - 1),
                contrast_limit=(cr[0] - 1, cr[1] - 1),
                p=0.8),
        ]),
        # Variant 3: strong rotation + vertical flip (sparse)
        A.Compose([
            A.Rotate(limit=aug["rotation_limits"][2],
                     p=1.0),
            A.VerticalFlip(p=aug["v_flip_p"]),
            A.RandomBrightnessContrast(
                brightness_limit=(br[0] - 1, br[1] - 1),
                contrast_limit=(cr[0] - 1, cr[1] - 1),
                p=0.8),
        ]),
        # Variant 4: shear + horizontal flip
        A.Compose([
            A.Affine(
                shear=(-aug["shear_range"][1],
                        aug["shear_range"][1]),
                p=1.0),
            A.HorizontalFlip(p=aug["h_flip_p"]),
            A.RandomBrightnessContrast(
                brightness_limit=(br[0] - 1, br[1] - 1),
                contrast_limit=(cr[0] - 1, cr[1] - 1),
                p=0.8),
        ]),
        # Variant 5: mild rotation + shear + blur
        A.Compose([
            A.Rotate(limit=aug["rotation_limits"][0],
                     p=1.0),
            A.Affine(
                shear=(-aug["shear_range"][0],
                        aug["shear_range"][0]),
                p=1.0),
            A.GaussianBlur(
                blur_limit=(3, 7),
                sigma_limit=(gb[0], gb[1]),
                p=0.5),
        ]),
        # Variant 6: moderate rotation + brightness shift
        A.Compose([
            A.Rotate(limit=aug["rotation_limits"][1],
                     p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=(br[0] - 1, br[1] - 1),
                contrast_limit=(cr[0] - 1, cr[1] - 1),
                p=1.0),
            A.HorizontalFlip(p=aug["h_flip_p"]),
        ]),
    ]

    assert len(transforms) == CFG["augmentation"]["variants_per_img"], (
        "Number of transforms must equal variants_per_img in CFG."
    )
    return transforms


# ================================================================
# AUGMENTATION
# ================================================================

def run_augmentation() -> None:
    """
    Apply offline augmentation to the training partition.

    For each original training image:
      1. Copy the original into augmented_train/.
      2. Apply each of the six transform pipelines.
      3. Save each augmented variant with suffix _aug_N.

    Validation and test partitions are not touched.
    """
    if is_done("augment_done"):
        print("Augmentation already complete — skipping.")
        print("Delete results/progress.json to redo.")
        return

    set_seed(CFG["seed"])
    transforms = _build_transforms()

    train_src = os.path.join(SPLIT, "train")
    if not os.path.isdir(train_src):
        raise FileNotFoundError(
            f"Training split not found: {train_src}\n"
            "Run part2_split.py first.")

    print("=" * 60)
    print("MangoLeafVarietyBD — Offline Augmentation")
    print("=" * 60)
    print(f"  Source  : {train_src}")
    print(f"  Dest    : {AUG_TRAIN}")
    print(f"  Variants: {CFG['augmentation']['variants_per_img']}"
          " per original image")
    print(f"  Seed    : {CFG['seed']}")
    print()

    records   = []
    class_log = []

    for cls in CFG["class_names"]:
        cls_src  = os.path.join(train_src, cls)
        cls_dst  = os.path.join(AUG_TRAIN, cls)
        os.makedirs(cls_dst, exist_ok=True)

        originals = sorted(Path(cls_src).glob("*.jpg"))
        if len(originals) == 0:
            print(f"  WARNING: no training images for {cls}")
            continue

        aug_count = 0

        for img_path in tqdm(originals,
                             desc=f"  {cls:<15}",
                             leave=True):
            stem = img_path.stem

            # -- Copy original --
            dst_orig = os.path.join(cls_dst,
                                    img_path.name)
            shutil.copy2(str(img_path), dst_orig)
            records.append({
                "filename" : img_path.name,
                "cultivar" : cls,
                "type"     : "original",
                "source"   : img_path.name,
            })

            # -- Read image for augmentation --
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image,
                                  cv2.COLOR_BGR2RGB)

            # -- Apply each transform variant --
            for idx, tfm in enumerate(transforms,
                                      start=1):
                aug_name = f"{stem}_aug_{idx}.jpg"
                dst_aug  = os.path.join(cls_dst,
                                         aug_name)

                result    = tfm(image=image)
                aug_image = cv2.cvtColor(
                    result["image"],
                    cv2.COLOR_RGB2BGR)
                cv2.imwrite(dst_aug, aug_image,
                            [cv2.IMWRITE_JPEG_QUALITY,
                             95])

                records.append({
                    "filename" : aug_name,
                    "cultivar" : cls,
                    "type"     : f"aug_{idx}",
                    "source"   : img_path.name,
                })
                aug_count += 1

        total_cls = len(originals) + aug_count
        class_log.append({
            "cultivar"  : cls,
            "originals" : len(originals),
            "augmented" : aug_count,
            "total"     : total_cls,
        })

    # -- Totals --
    df_log = pd.DataFrame(class_log)
    totals = {
        "cultivar"  : "TOTAL",
        "originals" : df_log["originals"].sum(),
        "augmented" : df_log["augmented"].sum(),
        "total"     : df_log["total"].sum(),
    }

    print()
    print(f"  {'Cultivar':<15}  "
          f"{'Originals':>10}  "
          f"{'Augmented':>10}  "
          f"{'Total':>8}")
    print(f"  {'-'*15}  {'-'*10}  {'-'*10}  {'-'*8}")
    for _, row in df_log.iterrows():
        print(f"  {row['cultivar']:<15}  "
              f"{int(row['originals']):>10}  "
              f"{int(row['augmented']):>10}  "
              f"{int(row['total']):>8}")
    print(f"  {'-'*15}  {'-'*10}  {'-'*10}  {'-'*8}")
    print(f"  {'TOTAL':<15}  "
          f"{int(totals['originals']):>10}  "
          f"{int(totals['augmented']):>10}  "
          f"{int(totals['total']):>8}")

    # -- Save augmentation log --
    log_path = os.path.join(AUG_TRAIN,
                             "augmentation_log.csv")
    pd.DataFrame(records).to_csv(log_path, index=False)
    print(f"\n  Augmentation log saved : {log_path}")

    # -- Save count table --
    count_path = os.path.join(AUG_TRAIN,
                               "augmentation_counts.csv")
    df_full = pd.concat(
        [df_log, pd.DataFrame([totals])],
        ignore_index=True)
    df_full.to_csv(count_path, index=False)
    print(f"  Count table saved      : {count_path}")

    mark_done("augment_done")
    print("\nAugmentation complete.")
    print("Next step: python part4_dataset.py")
    print("=" * 60)


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    run_augmentation()