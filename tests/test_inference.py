"""
tests/test_inference.py
-----------------------
Model loading, class mapping and the shared preprocessing.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from src.inference import (
    CLASS_NAMES,
    PNEUMONIA_INDEX,
    apply_threshold,
    load_model,
    predict_proba,
    preprocess,
)
from src.preprocessing import load_image


def test_class_mapping_matches_training_labels():
    # Training labels: 0 = no pneumonia, 1 = pneumonia (RSNA "Target").
    assert CLASS_NAMES[PNEUMONIA_INDEX] == "Pneumonia"
    assert PNEUMONIA_INDEX == 1


def test_load_model_missing_file_reports_not_loaded(tmp_path):
    model, loaded = load_model(tmp_path / "nope.pt")
    assert loaded is False
    assert model.fc.out_features == 2
    assert not model.training


def test_load_model_roundtrip(tmp_path):
    source, _ = load_model(tmp_path / "nope.pt")
    ckpt = tmp_path / "model.pt"
    torch.save(source.state_dict(), ckpt)

    model, loaded = load_model(ckpt)
    assert loaded is True
    assert torch.equal(model.fc.weight, source.fc.weight)


def test_load_model_rejects_incompatible_checkpoint(tmp_path):
    ckpt = tmp_path / "bad.pt"
    torch.save({"fc.weight": torch.zeros(3, 3)}, ckpt)
    _, loaded = load_model(ckpt)
    assert loaded is False


def test_preprocess_shape_and_predict_range(tmp_path):
    image = Image.new("RGB", (300, 200), (100, 100, 100))
    tensor = preprocess(image)
    assert tensor.shape == (1, 3, 224, 224)

    model, _ = load_model(tmp_path / "nope.pt")
    assert 0.0 <= predict_proba(model, tensor) <= 1.0


def test_apply_threshold():
    assert apply_threshold(0.9, 0.5) == "High Risk"
    assert apply_threshold(0.1, 0.5) == "Low Risk"


def test_load_image_from_stream_and_path(tmp_path):
    path = tmp_path / "x.png"
    Image.fromarray(np.zeros((10, 12), dtype=np.uint8)).save(path)

    from_path = load_image(path)
    with open(path, "rb") as handle:
        from_stream = load_image(handle, ".png")
    assert from_path.mode == from_stream.mode == "RGB"
    assert from_path.size == from_stream.size == (12, 10)


def test_load_image_rejects_unknown_type():
    with pytest.raises(ValueError):
        load_image("scan.bmp")


def _write_dicom(path, pixels):
    """Build a minimal single-frame grayscale DICOM file."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    meta.MediaStorageSOPInstanceUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian, ds.is_implicit_VR = True, False
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    ds.save_as(str(path))


def test_load_dicom_is_scaled_to_full_range(tmp_path):
    path = tmp_path / "scan.dcm"
    # 12-bit style values; must be stretched to 0-255 RGB.
    _write_dicom(path, np.array([[0, 1000], [2000, 4000]]))

    image = load_image(path)
    pixels = np.array(image)
    assert image.mode == "RGB"
    assert pixels.min() == 0 and pixels.max() == 255
