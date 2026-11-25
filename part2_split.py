"""
================================================================
MangoLeafVarietyBD — Part 2: Grouped Stratified Split
================================================================
Dataset : MangoLeafVarietyBD: An Image Dataset for
          Multi-Class Mango Variety Classification
          in Bangladesh
Authors : Habibur Rahman Sajal, Mahfuzur Rahman,
          Mohammad Monirul Islam
Journal : Scientific Data (Springer Nature)
----------------------------------------------------------------
Performs a grouped stratified split on the 2,744 original,
unaugmented images.

Split strategy
--------------
Every original image is treated as one independent unit.
The split is performed at the original-image level so that
no augmented variant of an image can appear in a different
partition (grouped split).  Within each class the images are
assigned to train / val / test in proportion (stratified
split), preserving the class distribution across all three
partitions.

Ratios  :  70 % train  |  15 % val  |  15 % test
Seed    :  42  (fixed for full reproducibility)

Output
------
dataset_split/
    train/<Class>/   original JPEG files
    val/<Class>/     original JPEG files
    test/<Class>/    original JPEG files
    split_summary.csv   one row per image with split label
    split_counts.csv    per-class count table
----------------------------------------------------------------
Run after : python part1_setup.py
Run next  : python part3_augmentation.py
================================================================
"""

import os
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from part1_setup import (
    CFG,
    RAW_RENAMED,
    SPLIT,
    is_done,
    mark_done,
    set_seed,
)

# ================================================================
# HELPERS
# ================================================================

def _collect_images(class_dir: str) -> list:
    """Return sorted list of JPEG paths inside a class folder."""
    return sorted(
        str(p) for p in Path(class_dir).glob("*.jpg")
    )


def _copy_images(paths: list, dest_dir: str) -> None:
    """Copy a list of image files into dest_dir."""
    os.makedirs(dest_dir, exist_ok=True)
    for src in paths:
        shutil.copy2(src, os.path.join(
            dest_dir, os.path.basename(src)))


# ================================================================
# SPLIT
# ================================================================

def run_split() -> None:
    """
    Execute the grouped stratified split.

    For each class:
      1. Collect all original image paths.
      2. Split into train+val and test (85 / 15).
      3. Split train+val into train and val (82.4 / 17.6,
         which yields the final 70 / 15 / 15 ratio).
      4. Copy files into the destination tree.
      5. Record split assignment in a summary CSV.

    The split is deterministic: the same seed always
    produces the same partition.
    """
    if is_done("split_done"):
        print("Split already complete — skipping.")
        print("Delete results/progress.json to redo.")
        return

    set_seed(CFG["seed"])

    # val fraction relative to the train+val subset
    # such that final val == 15 % of total
    val_frac = CFG["val_ratio"] / (
        CFG["train_ratio"] + CFG["val_ratio"])

    records   = []
    class_log = []

    print("=" * 60)
    print("MangoLeafVarietyBD — Grouped Stratified Split")
    print("=" * 60)
    print(f"  Source  : {RAW_RENAMED}")
    print(f"  Dest    : {SPLIT}")
    print(f"  Ratios  : "
          f"{int(CFG['train_ratio']*100)} / "
          f"{int(CFG['val_ratio']*100)} / "
          f"{int(CFG['test_ratio']*100)}  "
          f"(train / val / test)")
    print(f"  Seed    : {CFG['seed']}")
    print()

    for cls in CFG["class_names"]:
        cls_src = os.path.join(RAW_RENAMED, cls)
        images  = _collect_images(cls_src)

        if len(images) == 0:
            print(f"  WARNING: no images found for {cls}")
            continue

        # -- Step 1: split off the test set --
        trainval, test = train_test_split(
            images,
            test_size=CFG["test_ratio"],
            random_state=CFG["seed"],
            shuffle=True,
        )

        # -- Step 2: split train+val into train and val --
        train, val = train_test_split(
            trainval,
            test_size=val_frac,
            random_state=CFG["seed"],
            shuffle=True,
        )

        # -- Step 3: copy files --
        _copy_images(train,
                     os.path.join(SPLIT, "train", cls))
        _copy_images(val,
                     os.path.join(SPLIT, "val",   cls))
        _copy_images(test,
                     os.path.join(SPLIT, "test",  cls))

        # -- Step 4: record assignments --
        for split_name, subset in [
            ("train", train),
            ("val",   val),
            ("test",  test),
        ]:
            for path in subset:
                records.append({
                    "filename": os.path.basename(path),
                    "cultivar": cls,
                    "split"   : split_name,
                })

        class_log.append({
            "cultivar": cls,
            "total"   : len(images),
            "train"   : len(train),
            "val"     : len(val),
            "test"    : len(test),
        })

        print(f"  {cls:<15}  "
              f"total={len(images):>4}  "
              f"train={len(train):>4}  "
              f"val={len(val):>3}  "
              f"test={len(test):>3}")

    # -- Totals --
    df_log = pd.DataFrame(class_log)
    totals = {
        "cultivar": "TOTAL",
        "total"   : df_log["total"].sum(),
        "train"   : df_log["train"].sum(),
        "val"     : df_log["val"].sum(),
        "test"    : df_log["test"].sum(),
    }
    print()
    print(f"  {'TOTAL':<15}  "
          f"total={totals['total']:>4}  "
          f"train={totals['train']:>4}  "
          f"val={totals['val']:>3}  "
          f"test={totals['test']:>3}")

    # -- Save split summary CSV --
    csv_path = os.path.join(SPLIT, "split_summary.csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"\n  Split summary saved : {csv_path}")

    # -- Save per-class count table --
    table_path = os.path.join(SPLIT, "split_counts.csv")
    df_full = pd.concat(
        [df_log, pd.DataFrame([totals])],
        ignore_index=True,
    )
    df_full.to_csv(table_path, index=False)
    print(f"  Split counts saved  : {table_path}")

    mark_done("split_done")
    print("\nSplit complete.")
    print("Next step: python part3_augmentation.py")
    print("=" * 60)


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    run_split()