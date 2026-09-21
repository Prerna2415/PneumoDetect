from pathlib import Path

import cv2
import numpy as np

from src.gradcam import GradCAM
from src.inference import PNEUMONIA_INDEX
from src.preprocessing import get_default_transform, load_image


def generate_gradcam_overlay(model, img_path: str, out_dir="static/gradcam"):
    """Save a Grad-CAM overlay (evidence for pneumonia) next to the input."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = load_image(img_path)
    tensor = get_default_transform()(img).unsqueeze(0)

    with GradCAM(model, target_layer_name="layer4") as cam:
        heatmap = cam.generate(tensor, target_class=PNEUMONIA_INDEX)

    # OpenCV works in BGR, PIL gives RGB.
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    overlay = GradCAM.overlay_heatmap(img_bgr, heatmap)

    out_path = out_dir / f"cam_{Path(img_path).stem}.png"
    cv2.imwrite(str(out_path), overlay)
    return out_path
