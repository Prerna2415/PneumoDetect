"""
tests/test_train_module.py
--------------------------
Smoke test: train_baseline runs end-to-end on synthetic data and produces
the expected artifacts (log with validation metrics + checkpoints).
"""

import pandas as pd

from pathlib import Path
from src.data_loader import get_data_loader
from src.train import train_baseline


def test_loader_shapes(fake_dataset):
    csv_path, img_dir = fake_dataset
    loader = get_data_loader(csv_path, img_dir, batch_size=4)
    imgs, labels = next(iter(loader))
    assert imgs.shape == (4, 3, 224, 224)


def test_train_writes_log_and_checkpoints(fake_dataset, tmp_path):
    csv_path, img_dir = fake_dataset
    out_dir = tmp_path / "ckpts"

    train_baseline(
        csv_path, img_dir, epochs=1, batch_size=4, lr=1e-3,
        out_dir=str(out_dir),
    )

    assert (out_dir / "resnet50_best.pt").exists()
    assert (out_dir / "resnet50_baseline.pt").exists()

    logs = list(Path("reports/week2_metrics").glob("training_log_*.csv"))
    assert len(logs) == 1
    row = pd.read_csv(logs[0]).iloc[0]
    # Metrics come from held-out validation data, not the training set.
    for column in ("val_loss", "val_auc", "val_sensitivity",
                   "val_specificity"):
        assert column in row.index
