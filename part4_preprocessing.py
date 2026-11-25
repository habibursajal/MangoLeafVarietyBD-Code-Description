"""
================================================================
MangoLeafVarietyBD — Part 4: Preprocessing
================================================================
Dataset : MangoLeafVarietyBD: An Image Dataset for
          Multi-Class Mango Variety Classification
          in Bangladesh
Authors : Habibur Rahman Sajal, Mahfuzur Rahman,
          Mohammad Monirul Islam
Journal : Scientific Data (Springer Nature)
----------------------------------------------------------------
Applies background removal and resizing to all three
partitions.  This step produces the final model-ready images.

Processing pipeline
-------------------
  1. Background removal  : rembg (U2-Net) isolates the leaf
                           lamina from the backing sheet.
  2. White fill          : transparent regions are composited
                           onto a plain white canvas so that
                           the output is a standard RGB JPEG.
  3. Resize              : Lanczos interpolation to 224 × 224.
  4. JPEG encode         : quality 95, saved to preprocessed/.

Applied to
----------
  train/   source: augmented_train/   (13,405 images)
  val/     source: dataset_split/val/ (  414 images)
  test/    source: dataset_split/test/(  415 images)

Output
------
preprocessed/
    train/<Class>/   background-removed, resized images
    val/<Class>/     background-removed, resized images
    test/<Class>/    background-removed, resized images
----------------------------------------------------------------
Run after : python part3_augmentation.py
Run next  : python part5_train.py
================================================================
"""

import os
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove
from tqdm import tqdm

from part1_setup import (
    AUG_TRAIN,
    BASE,
    CFG,
    PREPROCESSED,
    SPLIT,
    is_done,
    mark_done,
)

# ================================================================
# PREPROCESSING
# ================================================================

def _remove_background_and_resize(
    img_path: str,
    session,
    img_size: int,
) -> np.ndarray:
    """
    Remove the background from one image using U2-Net,
    composite the result onto a white canvas, and resize
    to img_size × img_size using Lanczos interpolation.

    Parameters
    ----------
    img_path : str
        Absolute path to the source JPEG.
    session :
        rembg inference session (shared across calls).
    img_size : int
        Target spatial resolution (square).

    Returns
    -------
    np.ndarray
        BGR image of shape (img_size, img_size, 3),
        ready for cv2.imwrite.
    """
    with open(img_path, "rb") as f:
        raw = f.read()

    # Remove background — output is RGBA PNG bytes.
    out_bytes = remove(raw, session=session)

    # Decode RGBA.
    rgba = Image.open(BytesIO(out_bytes)).convert("RGBA")

    # Composite onto a white canvas.
    canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    canvas.paste(rgba, mask=rgba.split()[3])
    rgb = canvas.convert("RGB")

    # Resize with Lanczos interpolation.
    rgb_resized = rgb.resize(
        (img_size, img_size),
        Image.LANCZOS,
    )

    # Convert to BGR for OpenCV-compatible saving.
    bgr = cv2.cvtColor(
        np.array(rgb_resized),
        cv2.COLOR_RGB2BGR,
    )
    return bgr


def _process_partition(
    partition: str,
    src_root: str,
    dst_root: str,
    session,
    img_size: int,
) -> dict:
    """
    Process all images in one partition (train / val / test).

    Parameters
    ----------
    partition : str
        Partition name, used only for display.
    src_root : str
        Root folder of the source images.
    dst_root : str
        Root folder for the processed output.
    session :
        Shared rembg inference session.
    img_size : int
        Target spatial resolution.

    Returns
    -------
    dict
        Per-class counts {class_name: n_processed}.
    """
    counts = {}

    for cls in CFG["class_names"]:
        cls_src = os.path.join(src_root, cls)
        cls_dst = os.path.join(dst_root, cls)
        os.makedirs(cls_dst, exist_ok=True)

        images = sorted(Path(cls_src).glob("*.jpg"))
        if len(images) == 0:
            print(f"  [{partition}] WARNING: "
                  f"no images for {cls}")
            counts[cls] = 0
            continue

        for img_path in tqdm(
            images,
            desc=f"  [{partition}] {cls:<15}",
            leave=True,
        ):
            dst_path = os.path.join(
                cls_dst, img_path.name)

            # Skip already-processed files.
            if os.path.exists(dst_path):
                continue

            bgr = _remove_background_and_resize(
                str(img_path), session, img_size)

            cv2.imwrite(
                dst_path,
                bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

        counts[cls] = len(images)

    return counts


# ================================================================
# MAIN
# ================================================================

def run_preprocessing() -> None:
    """
    Execute background removal and resizing for all
    three partitions.

    A single rembg session is created once and reused
    across all images to avoid repeated model loading.
    """
    if is_done("preprocess_done"):
        print("Preprocessing already complete — skipping.")
        print("Delete results/progress.json to redo.")
        return

    img_size = CFG["img_size"]

    # Verify source directories exist.
    sources = {
        "train": AUG_TRAIN,
        "val"  : os.path.join(SPLIT, "val"),
        "test" : os.path.join(SPLIT, "test"),
    }
    for name, src in sources.items():
        if not os.path.isdir(src):
            raise FileNotFoundError(
                f"Source not found for '{name}': {src}\n"
                "Run earlier pipeline steps first.")

    print("=" * 60)
    print("MangoLeafVarietyBD — Preprocessing")
    print("=" * 60)
    print(f"  Output  : {PREPROCESSED}")
    print(f"  Resize  : {img_size} × {img_size} px")
    print(f"  Method  : U2-Net (rembg) + Lanczos resize")
    print(f"  Quality : JPEG 95 %")
    print()
    print("  Loading U2-Net model...")

    # Create one session for the entire run.
    session = new_session("u2net")
    print("  Model loaded.\n")

    grand_total = 0

    for partition, src_root in sources.items():
        dst_root = os.path.join(PREPROCESSED, partition)
        counts   = _process_partition(
            partition, src_root, dst_root,
            session, img_size)

        n = sum(counts.values())
        grand_total += n
        print(f"  [{partition}] {n:>6} images processed\n")

    print(f"  Grand total : {grand_total:>6} images")

    mark_done("preprocess_done")
    print("\nPreprocessing complete.")
    print("Next step: python part5_train.py")
    print("=" * 60)


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    run_preprocessing()