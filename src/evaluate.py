"""
src/evaluate.py
---------------
Measure how good a model is on data it was NOT trained on.

Accuracy alone is misleading when pneumonia is the minority class (a model
that always says "normal" scores high accuracy). For a screening tool we care
most about:
- sensitivity (recall): of the real pneumonia cases, how many did we catch?
- specificity: of the healthy cases, how many did we correctly clear?
- AUC: how well probabilities rank pneumonia above normal, at any threshold.

Run on a labelled folder:
    python -m src.evaluate \
        --csv data/rsna_subset/stage_2_train_labels.csv \
        --img_dir data/rsna_subset/train_images \
        --model saved_models/resnet50_best.pt
        --csv data/rsna_subset/stage_2_train_labels.csv \
        --img_dir data/rsna_subset/train_images \
        --model saved_models/resnet50_best.pt
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.inference import PNEUMONIA_INDEX


def compute_metrics(labels, probs, threshold=0.5) -> dict:
    """Metrics from true labels (0/1) and predicted P(pneumonia)."""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    both_classes = len(np.unique(labels)) == 2
    return {
        "accuracy": (tp + tn) / max(1, len(labels)),
        "sensitivity": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
        "auc": roc_auc_score(labels, probs) if both_classes else float("nan"),
    }


def evaluate(model, loader, device, threshold=0.5) -> dict:
    """Run the model over a DataLoader and return loss + metrics."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    all_labels, all_probs, total_loss = [], [], 0.0

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            total_loss += criterion(outputs, labels).item()
            probs = F.softmax(outputs, dim=1)[:, PNEUMONIA_INDEX]
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    metrics = compute_metrics(all_labels, all_probs, threshold)
    metrics["loss"] = total_loss / max(1, len(all_labels))
    return metrics


if __name__ == "__main__":
    from src.data_loader import get_data_loader
    from src.inference import get_device, load_model

    parser = argparse.ArgumentParser(description="Evaluate a checkpoint.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--model", default="saved_models/resnet50_best.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    dev = get_device()
    net, loaded = load_model(args.model, dev)
    if not loaded:
        raise SystemExit(f"Could not load weights from {args.model}")
    data = get_data_loader(args.csv, args.img_dir, shuffle=False)
    for name, value in evaluate(net, data, dev, args.threshold).items():
        print(f"{name:12s} {value:.4f}")
    print("\nIf AUC is well below 0.5 the class order is flipped: "
          "check PNEUMONIA_INDEX in src/inference.py.")
