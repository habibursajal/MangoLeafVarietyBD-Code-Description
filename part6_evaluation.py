"""
================================================================
MangoLeafVarietyBD — Part 6: Evaluation, Figures & Tables
================================================================
Dataset : MangoLeafVarietyBD: An Image Dataset for
          Multi-Class Mango Variety Classification
          in Bangladesh
Authors : Habibur Rahman Sajal, Mahfuzur Rahman,
          Mohammad Monirul Islam
Journal : Scientific Data (Springer Nature)
----------------------------------------------------------------
Evaluates both trained models on the held-out test partition,
produces all figures and tables required for the paper.

Figures produced (600 DPI PNG)
-------------------------------
  Fig_TrainingCurves_VGG16.png
  Fig_TrainingCurves_MobileNetV4_Small.png
  Fig_ConfusionMatrix_VGG16.png
  Fig_ConfusionMatrix_MobileNetV4_Small.png

Tables produced (CSV)
---------------------
  Table3_Overall_Performance.csv
      Accuracy, Precision, Recall, F1 for both models
  Table4_Classwise_VGG16.csv
      Per-class Precision, Recall, F1, Support for VGG16
  Table5_Classwise_MobileNetV4_Small.csv
      Per-class Precision, Recall, F1, Support for MobileNetV4

Input
-----
  results/models/<ModelName>.pth
  results/logs/<ModelName>_log.csv
  preprocessed/test/

Output
------
  results/figures/training_curves/
  results/figures/confusion_matrix/
  results/logs/Table3_Overall_Performance.csv
  results/logs/Table4_Classwise_VGG16.csv
  results/logs/Table5_Classwise_MobileNetV4_Small.csv
----------------------------------------------------------------
Run after : python part5_train.py
================================================================
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from part1_setup import (
    CFG,
    FIG_CM,
    FIG_CURVES,
    LOGS,
    MODELS,
    PREPROCESSED,
    set_seed,
)

# ================================================================
# PLOT STYLE
# ================================================================

plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "font.size"         : 14,
    "axes.titlesize"    : 16,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 14,
    "axes.labelweight"  : "bold",
    "xtick.labelsize"   : 12,
    "ytick.labelsize"   : 12,
    "legend.fontsize"   : 12,
    "figure.dpi"        : 100,
})

sns.set_style("whitegrid")

# ================================================================
# TIMM MODEL IDENTIFIERS  (must match part5)
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
# TRANSFORM
# ================================================================

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# ================================================================
# HELPERS
# ================================================================

def _get_test_loader() -> tuple:
    """
    Build the test DataLoader and return it with the
    class list.
    """
    test_ds = datasets.ImageFolder(
        os.path.join(PREPROCESSED, "test"),
        transform=EVAL_TRANSFORM,
    )
    loader = DataLoader(
        test_ds,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=(CFG["device"] == "cuda"),
        persistent_workers=True,
    )
    return loader, test_ds.classes


def _load_model(model_name: str,
                num_classes: int) -> nn.Module:
    """
    Rebuild the model architecture and load the best
    checkpoint saved by part5_train.py.
    """
    timm_id = _TIMM_NAMES[model_name]
    model   = timm.create_model(
        timm_id,
        pretrained=False,
        num_classes=num_classes,
        drop_rate=_DROPOUT,
    )
    pth = os.path.join(MODELS, f"{model_name}.pth")
    model.load_state_dict(
        torch.load(pth,
                   map_location=CFG["device"])
    )
    model.to(CFG["device"])
    model.eval()
    return model


def _run_inference(model: nn.Module,
                   loader: DataLoader) -> tuple:
    """
    Run the model over the test loader and collect
    true labels and predictions.

    Returns
    -------
    y_true, y_pred : list of int
    """
    device  = CFG["device"]
    y_true  = []
    y_pred  = []

    with torch.no_grad():
        for imgs, targets in tqdm(loader,
                                   desc="  Inference",
                                   leave=False):
            imgs    = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast(device_type=device):
                logits = model(imgs)
            preds = logits.argmax(1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(targets.cpu().numpy().tolist())

    return y_true, y_pred


# ================================================================
# FIGURE 1 & 2: TRAINING CURVES
# ================================================================

def plot_training_curves(model_name: str) -> None:
    """
    Plot loss and accuracy curves from the training log
    and save at 600 DPI.

    Saves
    -----
    results/figures/training_curves/
        Fig_TrainingCurves_<ModelName>.png
    """
    log_path = os.path.join(LOGS,
                             f"{model_name}_log.csv")
    df      = pd.read_csv(log_path)
    epochs  = df["epoch"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Training and Validation Curves — {model_name}",
        fontsize=17,
        fontweight="bold",
        y=1.02,
    )

    # -- Loss --
    ax = axes[0]
    ax.plot(epochs, df["train_loss"], "b-o",
            linewidth=2, markersize=5,
            label="Train Loss")
    ax.plot(epochs, df["val_loss"], "r-o",
            linewidth=2, markersize=5,
            label="Validation Loss")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # -- Accuracy --
    ax = axes[1]
    ax.plot(epochs, df["train_acc"], "b-o",
            linewidth=2, markersize=5,
            label="Train Accuracy")
    ax.plot(epochs, df["val_acc"], "r-o",
            linewidth=2, markersize=5,
            label="Validation Accuracy")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim([0.0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = os.path.join(
        FIG_CURVES,
        f"Fig_TrainingCurves_{model_name}.png",
    )
    plt.savefig(out_path,
                dpi=CFG["dpi_fig"],
                bbox_inches="tight",
                format="png")
    plt.close()
    print(f"  Saved : {out_path}")


# ================================================================
# FIGURE 3 & 4: CONFUSION MATRIX
# ================================================================

def plot_confusion_matrix(model_name: str,
                           cm: np.ndarray,
                           classes: list) -> None:
    """
    Plot a normalised confusion matrix and save at 600 DPI.

    Saves
    -----
    results/figures/confusion_matrix/
        Fig_ConfusionMatrix_<ModelName>.png
    """
    # Normalise row-wise (true label).
    cm_norm = cm.astype(float) / cm.sum(axis=1,
                                         keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(
        f"Confusion Matrix — {model_name}",
        fontsize=17,
        fontweight="bold",
        y=1.01,
    )

    # -- Raw counts --
    ax = axes[0]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": 11, "weight": "bold"},
    )
    ax.set_title("Raw Counts", fontsize=14,
                 fontweight="bold")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(),
                       rotation=0)

    # -- Normalised --
    ax = axes[1]
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": 11, "weight": "bold"},
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title("Normalised (Row)", fontsize=14,
                 fontweight="bold")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(),
                       rotation=0)

    plt.tight_layout()

    out_path = os.path.join(
        FIG_CM,
        f"Fig_ConfusionMatrix_{model_name}.png",
    )
    plt.savefig(out_path,
                dpi=CFG["dpi_cm"],
                bbox_inches="tight",
                format="png")
    plt.close()
    print(f"  Saved : {out_path}")


# ================================================================
# TABLES
# ================================================================

def save_classwise_table(model_name: str,
                          y_true: list,
                          y_pred: list,
                          classes: list,
                          table_num: int) -> dict:
    """
    Compute and save the per-class classification report
    as a CSV table.

    Returns
    -------
    dict with overall weighted metrics.
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        digits=4,
        output_dict=True,
    )

    rows = []
    for cls in classes:
        m = report[cls]
        rows.append({
            "Class"    : cls,
            "Precision": round(m["precision"], 4),
            "Recall"   : round(m["recall"],    4),
            "F1-score" : round(m["f1-score"],  4),
            "Support"  : int(m["support"]),
        })

    # Macro and weighted averages
    for avg_key in ("macro avg", "weighted avg"):
        m = report[avg_key]
        rows.append({
            "Class"    : avg_key.replace(" avg",
                                          " average"),
            "Precision": round(m["precision"], 4),
            "Recall"   : round(m["recall"],    4),
            "F1-score" : round(m["f1-score"],  4),
            "Support"  : int(m["support"]),
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(
        LOGS,
        f"Table{table_num}_Classwise_{model_name}.csv",
    )
    df.to_csv(csv_path, index=False)
    print(f"  Saved : {csv_path}")

    return {
        "model"    : model_name,
        "accuracy" : round(report["accuracy"], 4),
        "precision": round(
            report["weighted avg"]["precision"], 4),
        "recall"   : round(
            report["weighted avg"]["recall"], 4),
        "f1"       : round(
            report["weighted avg"]["f1-score"], 4),
        "support"  : int(
            report["weighted avg"]["support"]),
    }


def save_overall_table(overall_rows: list) -> None:
    """
    Save Table 3: overall performance comparison of
    all models on the held-out test partition.
    """
    df = pd.DataFrame(overall_rows, columns=[
        "Model", "Accuracy", "Precision",
        "Recall", "F1-score", "Test Support",
    ])
    csv_path = os.path.join(
        LOGS,
        "Table3_Overall_Performance.csv",
    )
    df.to_csv(csv_path, index=False)
    print(f"  Saved : {csv_path}")


# ================================================================
# MAIN EVALUATION LOOP
# ================================================================

def run_evaluation() -> None:
    """
    For each trained model:
      1. Load checkpoint.
      2. Run inference on the test partition.
      3. Plot training curves (Figs 1-2).
      4. Plot confusion matrix (Figs 3-4).
      5. Save per-class table (Tables 4-5).
    Then save the overall comparison table (Table 3).
    """
    set_seed(CFG["seed"])

    print("=" * 60)
    print("MangoLeafVarietyBD — Evaluation")
    print("=" * 60)

    if not os.path.isdir(PREPROCESSED):
        raise FileNotFoundError(
            f"Preprocessed data not found: {PREPROCESSED}"
        )

    test_loader, classes = _get_test_loader()
    print(f"\n  Test images : {len(test_loader.dataset)}")
    print(f"  Classes     : {classes}\n")

    overall_rows = []
    table_nums   = {
        "VGG16"            : 4,
        "MobileNetV4_Small": 5,
    }

    for model_name in CFG["model_names"]:
        print(f"\n{'=' * 60}")
        print(f"  Model : {model_name}")
        print(f"{'=' * 60}")

        # -- Training curves --
        print("\n  [1/4] Plotting training curves...")
        plot_training_curves(model_name)

        # -- Load model --
        print("\n  [2/4] Loading checkpoint...")
        model = _load_model(model_name,
                             len(classes))

        # -- Inference --
        print("\n  [3/4] Running inference on test set...")
        y_true, y_pred = _run_inference(model,
                                         test_loader)

        acc = accuracy_score(y_true, y_pred)
        print(f"\n  Test Accuracy : {acc:.4f} "
              f"({acc*100:.2f} %)")

        # -- Confusion matrix --
        print("\n  [4/4] Plotting confusion matrix...")
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(model_name, cm, classes)

        # -- Per-class table --
        print("\n  Saving per-class table...")
        overall = save_classwise_table(
            model_name,
            y_true,
            y_pred,
            classes,
            table_nums[model_name],
        )
        overall_rows.append([
            overall["model"],
            overall["accuracy"],
            overall["precision"],
            overall["recall"],
            overall["f1"],
            overall["support"],
        ])

        # -- Print classification report --
        print()
        print(classification_report(
            y_true,
            y_pred,
            target_names=classes,
            digits=4,
        ))

    # -- Overall comparison table --
    print("\n  Saving overall comparison table (Table 3)...")
    save_overall_table(overall_rows)

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    print("=" * 60)
    print("\nAll outputs saved to:")
    print(f"  Figures  : {FIG_CURVES}")
    print(f"           : {FIG_CM}")
    print(f"  Tables   : {LOGS}")


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    run_evaluation()