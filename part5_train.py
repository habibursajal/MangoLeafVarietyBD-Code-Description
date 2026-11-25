"""
================================================================
MangoLeafVarietyBD — Part 5: Model Training
================================================================
Dataset : MangoLeafVarietyBD: An Image Dataset for
          Multi-Class Mango Variety Classification
          in Bangladesh
Authors : Habibur Rahman Sajal, Mahfuzur Rahman,
          Mohammad Monirul Islam
Journal : Scientific Data (Springer Nature)
----------------------------------------------------------------
Trains two convolutional networks on the preprocessed,
augmented training partition as a technical check on the
consistency of the cultivar labels.

Models
------
  VGG16             : deep sequential CNN, ImageNet weights
  MobileNetV4-Small : compact CNN, ImageNet weights

Both models are initialised from ImageNet weights and trained
under identical settings.  The best checkpoint (lowest
validation loss) is saved for each model.

Training settings
-----------------
  Batch size    : 32
  Optimiser     : AdamW  (lr = 1e-4, weight_decay = 1e-4)
  Loss          : CrossEntropyLoss (label_smoothing = 0.1)
  Scheduler     : CosineAnnealingLR (T_max = 30)
  Max epochs    : 30
  Early stop    : patience = 7  (validation loss)
  Input size    : 224 x 224
  Precision     : AMP mixed-precision (float16 on CUDA)
  Seed          : 42

Input
-----
  preprocessed/train/   background-removed, resized images
  preprocessed/val/     background-removed, resized images
  preprocessed/test/    background-removed, resized images

Output
------
  results/models/<ModelName>.pth      best checkpoint
  results/logs/<ModelName>_log.csv    per-epoch metrics
----------------------------------------------------------------
Run after : python part4_preprocessing.py
Run next  : python part6_evaluation.py
================================================================
"""

import os
import time

import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from part1_setup import (
    CFG,
    LOGS,
    MODELS,
    PREPROCESSED,
    is_model_done,
    mark_model_done,
    set_seed,
)

# ================================================================
# IMAGENET NORMALISATION CONSTANTS
# ================================================================

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# Training images are already augmented offline (part3)
# and resized to 224x224 (part4).
# Only ToTensor + Normalize is applied here to avoid
# double-augmentation.
TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# ================================================================
# TIMM MODEL IDENTIFIERS
# ================================================================

_TIMM_NAMES = {
    "VGG16": "vgg16_bn",
    "MobileNetV4_Small": (
        "hf_hub:timm/mobilenetv4_conv_small"
        ".e2400_r224_in1k"
    ),
}

_DROPOUT = 0.5

# ================================================================
# DATA LOADERS
# ================================================================

def get_dataloaders() -> dict:
    """
    Build DataLoaders for train, val and test partitions
    from the preprocessed folder produced by part4.

    ImageFolder assigns class indices in sorted alphabetical
    order, matching the sorted CFG['class_names'] list.

    Returns
    -------
    dict with keys: train, val, test, classes
    """
    train_ds = datasets.ImageFolder(
        os.path.join(PREPROCESSED, "train"),
        transform=TRAIN_TRANSFORM,
    )
    val_ds = datasets.ImageFolder(
        os.path.join(PREPROCESSED, "val"),
        transform=EVAL_TRANSFORM,
    )
    test_ds = datasets.ImageFolder(
        os.path.join(PREPROCESSED, "test"),
        transform=EVAL_TRANSFORM,
    )

    # Windows: num_workers > 0 requires the
    # if __name__ == '__main__' guard (handled in ENTRY POINT).
    # persistent_workers avoids re-spawning workers each epoch.
    nw = 4
    loader_kwargs = dict(
        batch_size=CFG["batch_size"],
        num_workers=nw,
        pin_memory=(CFG["device"] == "cuda"),
        persistent_workers=(nw > 0),
    )

    return {
        "train"  : DataLoader(train_ds, shuffle=True,
                              **loader_kwargs),
        "val"    : DataLoader(val_ds,   shuffle=False,
                              **loader_kwargs),
        "test"   : DataLoader(test_ds,  shuffle=False,
                              **loader_kwargs),
        "classes": train_ds.classes,
    }


# ================================================================
# MODEL FACTORY
# ================================================================

def _replace_head(model: nn.Module,
                  num_classes: int) -> nn.Module:
    """
    Replace the final classification layer of a timm model
    with Dropout(0.5) + Linear(in_features, num_classes).

    Handles VGG16 (timm VGGHead with fc attribute),
    MobileNetV4 (NormMlpClassifierHead), and generic
    head/classifier/fc patterns.

    Raises ValueError if no replaceable head is found.
    """
    # -- VGG16 special case: timm VGGHead --
    # timm VGG uses model.head which is a VGGHead object
    # containing model.head.fc (a Sequential).
    # We replace model.head.fc[-1] with our Linear.
    head = getattr(model, "head", None)
    if head is not None:
        fc = getattr(head, "fc", None)
        if fc is not None and isinstance(fc, nn.Sequential):
            # Find the last Linear in the Sequential
            for i in reversed(range(len(fc))):
                if isinstance(fc[i], nn.Linear):
                    in_f = fc[i].in_features
                    # Replace from that Linear onward
                    layers = list(fc.children())[:i]
                    layers += [
                        nn.Dropout(_DROPOUT),
                        nn.Linear(in_f, num_classes),
                    ]
                    head.fc = nn.Sequential(*layers)
                    return model

        # Generic head with in_features (MobileNetV4)
        if hasattr(head, "in_features"):
            in_f = head.in_features
            model.head = nn.Sequential(
                nn.Dropout(_DROPOUT),
                nn.Linear(in_f, num_classes),
            )
            return model

    # -- Generic fallback: classifier / fc --
    for attr in ("classifier", "fc"):
        layer = getattr(model, attr, None)
        if layer is None:
            continue

        if isinstance(layer, nn.Sequential):
            last = layer[-1]
            if hasattr(last, "in_features"):
                setattr(model, attr, nn.Sequential(
                    nn.Dropout(_DROPOUT),
                    nn.Linear(last.in_features, num_classes),
                ))
                return model

        if hasattr(layer, "in_features"):
            setattr(model, attr, nn.Sequential(
                nn.Dropout(_DROPOUT),
                nn.Linear(layer.in_features, num_classes),
            ))
            return model

    raise ValueError(
        "Could not locate a replaceable classification "
        f"head on {type(model).__name__}. "
        "Add the attribute name to _replace_head."
    )


def build_model(model_name: str,
                num_classes: int) -> nn.Module:
    """
    Instantiate a timm model with ImageNet pretrained
    weights.  The classification head is resized to
    num_classes by timm's own create_model mechanism,
    which is the correct way for all timm models.
    Dropout is added only where timm exposes a dedicated
    drop_rate argument.

    Parameters
    ----------
    model_name : str
        Key from CFG['model_names'] (e.g. 'VGG16').
    num_classes : int
        Number of output classes.

    Returns
    -------
    nn.Module placed on CFG['device'].
    """
    timm_id = _TIMM_NAMES[model_name]
    model   = timm.create_model(
        timm_id,
        pretrained=True,
        num_classes=num_classes,
        drop_rate=_DROPOUT,
    )
    return model.to(CFG["device"])


# ================================================================
# TRAINING ENGINE
# ================================================================

def train_one_model(model_name: str,
                    loaders: dict) -> None:
    """
    Train one model, save the best checkpoint (lowest
    validation loss), and write a per-epoch CSV log.

    Both training and validation use AMP mixed-precision
    inference for consistent GPU utilisation.

    Parameters
    ----------
    model_name : str
        Key from CFG['model_names'].
    loaders : dict
        DataLoaders returned by get_dataloaders().
    """
    if is_model_done(model_name):
        print(f"  {model_name} already trained — skipping.")
        return

    num_classes = len(loaders["classes"])
    device      = CFG["device"]

    model     = build_model(model_name, num_classes)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CFG["epochs"],
    )
    # GradScaler manages loss scaling for AMP.
    scaler = GradScaler()

    best_val_loss = float("inf")
    patience_ctr  = 0
    history       = []

    pth_path = os.path.join(MODELS, f"{model_name}.pth")

    print(f"\n  Model      : {model_name}")
    print(f"  timm id    : {_TIMM_NAMES[model_name]}")
    print(f"  Classes    : {num_classes}")
    print(f"  Device     : {device}")
    print(f"  Train imgs : {len(loaders['train'].dataset)}")
    print(f"  Val imgs   : {len(loaders['val'].dataset)}")
    print(f"  Max epochs : {CFG['epochs']}")
    print(f"  Patience   : {CFG['patience']}")
    print(f"  LR         : {CFG['lr']}")
    print()

    for epoch in range(CFG["epochs"]):
        t0 = time.time()

        # ---- Train phase ----
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0

        for imgs, targets in tqdm(
            loaders["train"],
            desc=f"    Epoch {epoch + 1:02d} [train]",
            leave=False,
        ):
            imgs    = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(device_type=device):
                logits = model(imgs)
                loss   = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            t_loss    += loss.item()
            t_correct += (logits.argmax(1) == targets
                          ).sum().item()
            t_total   += targets.size(0)

        train_loss = t_loss / len(loaders["train"])
        train_acc  = t_correct / t_total

        # ---- Validation phase ----
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0

        with torch.no_grad():
            for imgs, targets in loaders["val"]:
                imgs    = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast(device_type=device):
                    logits = model(imgs)
                v_loss    += criterion(logits,
                                       targets).item()
                v_correct += (logits.argmax(1) == targets
                              ).sum().item()
                v_total   += targets.size(0)

        val_loss = v_loss / len(loaders["val"])
        val_acc  = v_correct / v_total
        elapsed  = time.time() - t0

        row = {
            "epoch"     : epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc" : round(train_acc,  4),
            "val_loss"  : round(val_loss,   4),
            "val_acc"   : round(val_acc,    4),
            "epoch_time": round(elapsed,    2),
        }
        history.append(row)

        print(
            f"    Epoch {epoch+1:02d} | "
            f"T-Loss: {train_loss:.4f} | "
            f"T-Acc:  {train_acc:.4f} | "
            f"V-Loss: {val_loss:.4f} | "
            f"V-Acc:  {val_acc:.4f} | "
            f"Time:   {elapsed:.1f}s"
        )

        # ---- Checkpoint: save on improvement ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), pth_path)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= CFG["patience"]:
                print(
                    f"    Early stopping triggered "
                    f"at epoch {epoch + 1}."
                )
                break

        # Scheduler steps after each epoch.
        scheduler.step()

    # ---- Save per-epoch training log ----
    log_path = os.path.join(LOGS, f"{model_name}_log.csv")
    pd.DataFrame(history).to_csv(log_path, index=False)

    print(f"\n  Best val loss : {best_val_loss:.4f}")
    print(f"  Log saved     : {log_path}")
    print(f"  Model saved   : {pth_path}")

    mark_model_done(model_name)


# ================================================================
# MAIN
# ================================================================

def run_training() -> None:
    """
    Entry point: set seed, build dataloaders, and train
    all models listed in CFG['model_names'] sequentially.
    """
    set_seed(CFG["seed"])

    print("=" * 60)
    print("MangoLeafVarietyBD — Model Training")
    print("=" * 60)

    if not os.path.isdir(PREPROCESSED):
        raise FileNotFoundError(
            f"Preprocessed data not found: {PREPROCESSED}\n"
            "Run part4_preprocessing.py first."
        )

    loaders = get_dataloaders()

    print(f"\n  Classes : {loaders['classes']}")
    print(f"  Train   : "
          f"{len(loaders['train'].dataset):>6} images")
    print(f"  Val     : "
          f"{len(loaders['val'].dataset):>6} images")
    print(f"  Test    : "
          f"{len(loaders['test'].dataset):>6} images")
    print()

    for model_name in CFG["model_names"]:
        print("-" * 60)
        train_one_model(model_name, loaders)

    print("\n" + "=" * 60)
    print("Training complete.")
    print("Next step: python part6_evaluation.py")
    print("=" * 60)


# ================================================================
# ENTRY POINT
# ================================================================
# The if __name__ guard is required on Windows when
# num_workers > 0 to prevent DataLoader worker processes
# from re-executing the top-level script on spawn.

if __name__ == "__main__":
    run_training()