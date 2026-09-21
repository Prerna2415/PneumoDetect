"""
src/train.py
------------
Training script for the PneumoDetect ResNet-50 classifier.

What it does each run:
1. Splits the labels by patient into train (80%) and validation (20%).
2. Trains with Adam + ReduceLROnPlateau, optionally with a class-balanced
   sampler (--balanced), Focal Loss (--focal) or partial fine-tuning
   (--finetune).
3. After every epoch, measures loss / AUC / sensitivity / specificity on the
   validation set. The "best" checkpoint is the one with the best validation
   AUC, NOT the best training accuracy (which just rewards memorisation).
4. Writes a CSV log and the checkpoints into --out_dir.

Run:
    python -m src.train --epochs 3 --batch_size 8 --lr 1e-3
    python -m src.train --finetune --balanced --lr 1e-4
    python -m src.train --resume saved_models/resnet50_finetuned.pt
"""

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from src.data_loader import (
    get_data_loader,
    get_balanced_loader,
    get_train_transform,
    get_default_transform,
    split_labels,
)
from src.evaluate import evaluate
from src.losses import FocalLoss
from src.model import build_resnet50_baseline, build_resnet50_finetuned


def collate_skip_none(batch):
    """Remove None samples (missing images) from minibatches."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs, labels = zip(*batch)
    return torch.stack(imgs), torch.tensor(labels)


def ensure_dataset_available(csv_path: Path) -> Path:
    """
    Ensure dataset CSV exists or create a small synthetic version for tests.
    """
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found. Creating synthetic CSV "
              "for CI/testing.")
        tmp_dir = Path("data")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / "tmp_labels.csv"
        df = pd.DataFrame(
            {
                "patientId": [f"fake_{i}" for i in range(10)],
                "Target": [0, 1] * 5,
            }
        )
        df.to_csv(tmp_path, index=False)
        return tmp_path
    return csv_path


def detect_model_from_checkpoint(ckpt_path: Path):
    """
    Detect whether to load baseline or fine-tuned model based on filename.
    Returns the appropriate model constructor.
    """
    name = ckpt_path.name.lower()
    if "finetune" in name or "fine" in name:
        print("Detected fine-tuned checkpoint.")
        return build_resnet50_finetuned
    print("Detected baseline checkpoint.")
    return build_resnet50_baseline


def _select_score(metrics: dict) -> float:
    """Higher is better. AUC is undefined if validation has one class."""
    auc = metrics["auc"]
    return metrics["accuracy"] if math.isnan(auc) else auc


def train_baseline(
    csv_path,
    img_dir,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-3,
    balanced: bool = False,
    resume: str = None,
    finetune: bool = False,
    focal: bool = False,
    val_fraction: float = 0.2,
    seed: int = 42,
    out_dir: str = "saved_models",
):
    """
    Train a baseline or fine-tuned ResNet-50 and save the best checkpoint.

    Args:
        csv_path: Path to labels CSV (columns: patientId, Target).
        img_dir: Directory with the images.
        epochs, batch_size, lr: usual training settings.
        balanced: oversample the rare class with a weighted sampler.
        resume: checkpoint to continue from (type inferred from filename).
        finetune: unfreeze layer3/layer4 (ignored when resuming).
        focal: use Focal Loss instead of cross-entropy.
        val_fraction: share of patients held out for validation.
        seed: makes the split and initialisation repeatable.
        out_dir: where checkpoints are written.
    """
    torch.manual_seed(seed)
    csv_path = ensure_dataset_available(Path(csv_path))
    out_dir = Path(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data: split by patient, augment only the training half ---
    train_df, val_df = split_labels(csv_path, val_fraction, seed)
    print(f"Train patients: {len(train_df)} | Validation: {len(val_df)}")
    make_train_loader = get_balanced_loader if balanced else get_data_loader
    train_loader = make_train_loader(
        csv_path, img_dir, get_train_transform(),
        batch_size=batch_size, df=train_df,
    )
    val_loader = get_data_loader(
        csv_path, img_dir, get_default_transform(),
        batch_size=batch_size, shuffle=False, df=val_df,
    )

    # --- Model ---
    if resume:
        ckpt_path = Path(resume)
        model = detect_model_from_checkpoint(ckpt_path)().to(device)
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        print(f"Resumed training from checkpoint: {ckpt_path.name}")
    elif finetune:
        model = build_resnet50_finetuned().to(device)
    else:
        model = build_resnet50_baseline().to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    # alpha=1.0: keep only the "focus on hard examples" part of Focal Loss.
    criterion = FocalLoss(alpha=1.0) if focal else nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    best_score = float("-inf")
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = []

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}",
                    leave=False)

        for batch in pbar:
            if batch is None:
                continue
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}",
                             acc=f"{correct / total:.3f}")

        train_loss = running_loss / max(1, len(train_loader))
        train_acc = correct / total if total > 0 else 0.0

        # Validation: data the model has never trained on.
        val = evaluate(model, val_loader, device)
        scheduler.step(val["loss"])

        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} | val_loss={val['loss']:.4f} "
            f"val_auc={val['auc']:.4f} sens={val['sensitivity']:.3f} "
            f"spec={val['specificity']:.3f}"
        )
        logs.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_auc": val["auc"],
            "val_sensitivity": val["sensitivity"],
            "val_specificity": val["specificity"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        score = _select_score(val)
        if score > best_score:
            best_score = score
            best_path = out_dir / "resnet50_best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"New best model (val score={score:.4f}) -> {best_path}")

    # --- Save logs and the final-epoch weights ---
    reports_dir = Path("reports") / "week2_metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = "training_log_balanced" if balanced else "training_log_baseline"
    log_path = reports_dir / f"{name}_{stamp}.csv"
    pd.DataFrame(logs).to_csv(log_path, index=False)
    print(f"Training log saved to: {log_path.resolve()}")

    final_path = out_dir / "resnet50_baseline.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Final-epoch model saved to: {final_path.resolve()}")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="Train PneumoDetect ResNet-50 model."
        )
        parser.add_argument("--balanced", action="store_true",
                            help="oversample the rare class")
        parser.add_argument("--finetune", action="store_true",
                            help="unfreeze layer3/layer4 (use a lower --lr)")
        parser.add_argument("--focal", action="store_true",
                            help="use Focal Loss instead of cross-entropy")
        parser.add_argument("--epochs", type=int, default=3)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--resume", type=str, default=None)
        parser.add_argument("--val_fraction", type=float, default=0.2)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--out_dir", type=str, default="saved_models",
                            help="where checkpoints are written")
        parser.add_argument(
            "--csv_path",
            type=str,
            default="data/rsna_subset/stage_2_train_labels.csv",
        )
        parser.add_argument(
            "--img_dir", type=str, default="data/rsna_subset/train_images"
        )
        args = parser.parse_args()

        train_baseline(
            csv_path=args.csv_path,
            img_dir=args.img_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            balanced=args.balanced,
            resume=args.resume,
            finetune=args.finetune,
            focal=args.focal,
            val_fraction=args.val_fraction,
            seed=args.seed,
            out_dir=args.out_dir,
        )
    except Exception as e:
        print(f"Runtime error in main: {e}", file=sys.stderr)
        sys.exit(1)
