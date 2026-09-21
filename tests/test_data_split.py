"""
tests/test_data_split.py
------------------------
The validation split must be per-patient and stratified, and the balanced
sampler must line up with the rows the dataset actually keeps.
"""

import pandas as pd

from src.data_loader import (
    PneumoniaDataset,
    get_balanced_loader,
    read_labels,
    split_labels,
)


def _write_csv(path, rows):
    pd.DataFrame(rows, columns=["patientId", "Target"]).to_csv(
        path, index=False
    )
    return path


def test_duplicate_patient_rows_are_collapsed(tmp_path):
    # RSNA lists one row per bounding box: patient "a" appears twice.
    csv = _write_csv(tmp_path / "l.csv", [("a", 1), ("a", 1), ("b", 0)])
    assert len(read_labels(csv)) == 2


def test_no_patient_in_both_splits(tmp_path):
    rows = [(f"p{i}", i % 2) for i in range(40)]
    rows += [("p1", 1), ("p3", 1)]  # extra boxes for two patients
    csv = _write_csv(tmp_path / "l.csv", rows)

    train, val = split_labels(csv, val_fraction=0.25)
    assert set(train["patientId"]).isdisjoint(val["patientId"])
    assert len(train) + len(val) == 40


def test_split_is_stratified_and_repeatable(tmp_path):
    # 10% pneumonia; both halves should keep roughly that ratio.
    rows = [(f"p{i}", int(i < 20)) for i in range(200)]
    csv = _write_csv(tmp_path / "l.csv", rows)

    train, val = split_labels(csv, val_fraction=0.2, seed=1)
    assert abs(train["Target"].mean() - 0.1) < 0.02
    assert abs(val["Target"].mean() - 0.1) < 0.02
    again, _ = split_labels(csv, val_fraction=0.2, seed=1)
    assert train.equals(again)


def test_balanced_loader_weights_match_dataset_rows(tmp_path, fake_img_dir):
    """
    Regression: weights used to come from the full CSV while the dataset
    dropped rows without images, so the two lengths disagreed.
    """
    rows = [(f"fake_{i}", i % 2) for i in range(10)]
    rows += [(f"nofile_{i}", 0) for i in range(30)]  # no image on disk
    csv = _write_csv(tmp_path / "l.csv", rows)

    dataset = PneumoniaDataset(csv, fake_img_dir)
    assert len(dataset) == 10

    loader = get_balanced_loader(csv, fake_img_dir, batch_size=5)
    assert len(loader.sampler) == len(dataset) == 10
