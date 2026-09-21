"""
src/data_loader.py
------------------
Dataset, train/validation split and DataLoaders for the RSNA pneumonia data.

The CSV has one row per bounding box, so a patient with several boxes appears
several times. We keep one row per patient; otherwise the same patient could
land in both train and validation (data leakage).
"""

import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.preprocessing import (  # noqa: F401  (re-exported for callers)
    get_default_transform,
    get_train_transform,
    load_image,
)


def read_labels(csv_path) -> pd.DataFrame:
    """Read the labels CSV, one row per patient."""
    df = pd.read_csv(csv_path)
    return df.drop_duplicates("patientId").reset_index(drop=True)


def split_labels(csv_path, val_fraction=0.2, seed=42):
    """
    Stratified train/validation split by patient.

    Stratified = both halves keep the same pneumonia/normal ratio, which
    matters because pneumonia is the minority class here. If a class is too
    small to stratify (tiny synthetic test data) we fall back to a plain split.
    """
    df = read_labels(csv_path)
    counts = df["Target"].value_counts()
    stratify = df["Target"] if counts.min() >= 2 else None
    train_df, val_df = train_test_split(
        df, test_size=val_fraction, random_state=seed, stratify=stratify
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


class PneumoniaDataset(Dataset):
    """Chest X-ray images with a 0 (normal) / 1 (pneumonia) label."""

    def __init__(self, csv_path, img_dir, transform=None, df=None):
        """
        Args:
            csv_path: labels CSV (used unless `df` is given).
            img_dir: folder containing <patientId>.dcm/.png/.jpg files.
            transform: image transform; defaults to the inference transform.
            df: optional pre-split DataFrame (see split_labels).
        """
        self.img_dir = Path(img_dir)
        self.transform = transform or get_default_transform()
        self.data = read_labels(csv_path) if df is None else df

        if self.img_dir.exists():
            available = {f.stem for f in self.img_dir.glob("*")}
            matched = self.data[self.data["patientId"].isin(available)]
            if len(matched) > 0:
                self.data = matched.reset_index(drop=True)
        # Otherwise: synthetic/test mode, keep all rows (images -> zeros).

        print(f"Loaded {len(self.data)} records")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = int(row["Target"])

        for ext in (".dcm", ".png", ".jpg"):
            path = self.img_dir / f"{row['patientId']}{ext}"
            if path.exists():
                return self.transform(load_image(path)), label

        # No image on disk: only happens in synthetic/test mode.
        return torch.zeros((3, 224, 224)), label


def get_class_weights(csv_path):
    """Inverse-frequency weight per class (rarer class -> larger weight)."""
    counts = read_labels(csv_path)["Target"].value_counts().to_dict()
    weights = {cls: 1.0 / count for cls, count in counts.items()}
    print(f"Class counts: {counts} | Weights: {weights}")
    return weights


def get_balanced_loader(
    csv_path, img_dir, transform=None, batch_size=8, df=None
):
    """
    DataLoader that oversamples the rare class.

    Each sample is drawn with probability proportional to 1 / count**0.7.
    Full inverse frequency (power 1.0) would make batches exactly 50/50 but
    repeats the few pneumonia images heavily; 0.7 is a softer compromise.
    """
    from src.train import collate_skip_none

    dataset = PneumoniaDataset(csv_path, img_dir, transform, df=df)
    targets = dataset.data["Target"]
    counts = targets.value_counts().to_dict()
    weights = torch.tensor(
        [1.0 / counts[t] ** 0.7 for t in targets], dtype=torch.double
    )

    sampler = WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True
    )
    print(f"Class counts: {counts} | Balanced sampling enabled.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_skip_none,
    )


def get_data_loader(
    csv_path, img_dir, transform=None, batch_size=8, shuffle=True, df=None
):
    """Standard DataLoader (set shuffle=False for validation)."""
    from src.train import collate_skip_none

    dataset = PneumoniaDataset(csv_path, img_dir, transform, df=df)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_skip_none,
    )
