"""
src/preprocessing.py
--------------------
Everything that turns a file on disk into a model-ready tensor.

Training (data_loader.py) and serving (inference.py) both import from here,
so the model sees identically prepared images in both places. Preprocessing
that differs between training and serving is a classic silent accuracy bug
("train/serve skew").
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pydicom import dcmread
from torchvision import transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm"}


def load_image(source, suffix=None) -> Image.Image:
    """
    Load a PNG/JPG/DICOM chest X-ray as an RGB PIL image.

    Args:
        source: a file path, or an open binary file (e.g. an upload stream).
        suffix: file extension; required for file objects, inferred for paths.
    """
    if suffix is None:
        suffix = Path(source).suffix
    suffix = suffix.lower()

    if suffix in {".png", ".jpg", ".jpeg"}:
        return Image.open(source).convert("RGB")

    if suffix == ".dcm":
        ds = dcmread(source)
        pixels = ds.pixel_array.astype(np.float32)

        # Convert stored values to real units, then stretch to 0-255.
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
        pixels = pixels * slope + intercept
        pixels -= pixels.min()
        if pixels.max() > 0:
            pixels /= pixels.max()
        pixels = (pixels * 255).astype(np.uint8)

        if pixels.ndim == 2:
            pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(pixels).convert("RGB")

    raise ValueError(f"Unsupported file type: {suffix!r}")


def get_default_transform():
    """Deterministic transform used for validation and inference."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_train_transform():
    """
    Default transform plus mild augmentation (small rotation, brightness and
    contrast jitter) to reduce overfitting on a small dataset. No horizontal
    flip: heart position is anatomically meaningful on a chest X-ray.
    """
    return transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        get_default_transform(),
    ])
