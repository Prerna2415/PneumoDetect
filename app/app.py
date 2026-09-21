"""
app/app.py
----------
Flask web app for PneumoDetect. This file is deliberately thin: it handles
HTTP (upload in, HTML out) and delegates everything ML-related to src/.

Design choices worth knowing:
- Uploaded X-rays are processed in memory and returned inline in the page.
  Nothing is written to disk, so patient images are never stored.
- Invalid input re-renders the form with an error message (HTTP 400).
- One model instance is shared by all requests; a lock keeps the Grad-CAM
  hooks from interfering between concurrent requests.
"""

import base64
import io
import logging
import math
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gradcam import GradCAM  # noqa: E402
from src.inference import (  # noqa: E402
    PNEUMONIA_INDEX,
    apply_threshold,
    get_device,
    load_model,
    predict_proba,
    preprocess,
)
from src.preprocessing import ALLOWED_EXTENSIONS, load_image  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pneumodetect")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB upload limit

MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", PROJECT_ROOT / "saved_models" /
                   "resnet50_best.pt")
).resolve()

device = get_device()
model, MODEL_LOADED = load_model(MODEL_PATH, device)
_model_lock = threading.Lock()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _to_data_uri(rgb: np.ndarray, max_side: int = 512) -> str:
    """Encode an RGB array as a small JPEG data URI for an <img> tag."""
    image = Image.fromarray(rgb)
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _gradcam_overlay(rgb: np.ndarray, tensor) -> np.ndarray:
    """Heatmap of the evidence for pneumonia, blended over the X-ray."""
    with GradCAM(model, target_layer_name="layer4") as cam:
        heatmap = cam.generate(
            tensor.to(device), target_class=PNEUMONIA_INDEX
        )
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = GradCAM.overlay_heatmap(bgr, heatmap)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def _error(message: str, status: int = 400):
    return render_template(
        "index.html", error=message, model_loaded=MODEL_LOADED
    ), status


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", model_loaded=MODEL_LOADED)


@app.route("/predict", methods=["POST"])
def predict():
    upload = request.files.get("file")
    if upload is None or upload.filename == "":
        return _error("Please choose a chest X-ray file to upload.")

    suffix = Path(upload.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return _error(
            f"Unsupported file type '{suffix}'. Use PNG, JPG or DICOM."
        )

    try:
        threshold = float(request.form.get("threshold", 0.5))
    except ValueError:
        threshold = float("nan")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        return _error("Threshold must be a number between 0 and 1.")

    try:
        image = load_image(upload.stream, suffix)
    except Exception:
        logger.exception("Could not read upload")
        return _error("That file could not be read as a chest X-ray image.")

    rgb = np.array(image)
    tensor = preprocess(image)
    show_cam = "show_cam" in request.form

    with _model_lock:
        start = time.perf_counter()
        pneumonia_prob = predict_proba(model, tensor)
        elapsed = time.perf_counter() - start
        overlay = _gradcam_overlay(rgb, tensor) if show_cam else None

    logger.info("P(pneumonia)=%.3f threshold=%.2f", pneumonia_prob, threshold)

    return render_template(
        "result.html",
        prediction=apply_threshold(pneumonia_prob, threshold),
        prob_pneumonia=pneumonia_prob,
        prob_normal=1.0 - pneumonia_prob,
        threshold=threshold,
        elapsed=f"{elapsed:.2f}s",
        image_data=_to_data_uri(rgb),
        overlay_data=_to_data_uri(overlay) if overlay is not None else None,
        model_loaded=MODEL_LOADED,
    )


@app.errorhandler(413)
def too_large(_):
    return _error("File is too large (limit: 20 MB).", 413)


@app.route("/health")
def health():
    return {"status": "OK", "model_loaded": MODEL_LOADED}, 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    # Bind to localhost by default: the Flask debugger must never be
    # reachable from other machines. Containers use gunicorn instead.
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=port, debug=debug)
