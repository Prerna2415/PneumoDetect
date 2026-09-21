"""
src/inference.py
----------------
Model loading and prediction, shared by the Flask app and the evaluation
script. Keeping this out of app.py means it can be tested (and reused)
without a web server.
"""

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import models

from src.preprocessing import get_default_transform

logger = logging.getLogger(__name__)

# The training labels are the RSNA "Target" column: 0 = no pneumonia,
# 1 = pneumonia. So output index 1 of the network is the pneumonia logit.
# This is the ONLY place the mapping is defined; don't hard-code 0/1.
CLASS_NAMES = ["Normal", "Pneumonia"]
PNEUMONIA_INDEX = 1


def apply_threshold(probability: float, threshold: float) -> str:
    """Triage label: "High Risk" if P(pneumonia) > threshold else "Low Risk".

    Lowering the threshold catches more pneumonia (higher sensitivity) at the
    cost of more false alarms; raising it does the opposite.
    """
    return "High Risk" if probability > threshold else "Low Risk"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path, device=None):
    """
    Build ResNet-50 with a 2-class head and load trained weights.

    Returns (model, weights_loaded). If the checkpoint is missing or does not
    fit, the model has random weights and weights_loaded is False; callers
    should surface that to the user instead of showing meaningless numbers.
    """
    device = device or get_device()
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    model_path = Path(model_path)
    weights_loaded = False
    if model_path.exists():
        try:
            # weights_only=True refuses to unpickle arbitrary code.
            state = torch.load(model_path, map_location=device,
                               weights_only=True)
            model.load_state_dict(state)
            weights_loaded = True
            logger.info("Loaded %s on %s", model_path.name, device)
        except Exception as exc:
            logger.warning("Could not load %s (%s)", model_path, exc)
    else:
        logger.warning("Model file not found: %s", model_path)

    return model.to(device).eval(), weights_loaded


def preprocess(image) -> torch.Tensor:
    """PIL image -> normalized tensor of shape (1, 3, 224, 224)."""
    return get_default_transform()(image).unsqueeze(0)


def predict_proba(model, tensor: torch.Tensor) -> float:
    """Return P(pneumonia) for a single preprocessed image tensor."""
    device = next(model.parameters()).device
    with torch.no_grad():
        probs = F.softmax(model(tensor.to(device)), dim=1)
    return probs[0, PNEUMONIA_INDEX].item()
