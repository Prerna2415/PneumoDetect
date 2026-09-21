"""
tests/test_flask_routes.py
--------------------------
Integration tests for the PneumoDetect Flask app: pages load, bad input gets
a clear error, and a valid upload returns a prediction without writing files.
"""

import io

import pytest
from PIL import Image

from app.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _png_bytes(size=(64, 64), color=(120, 120, 120)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_home_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"PneumoDetect" in response.data


def test_health_reports_model_status(client):
    body = client.get("/health").get_json()
    assert body["status"] == "OK"
    assert isinstance(body["model_loaded"], bool)


def test_invalid_route_returns_404(client):
    assert client.get("/nonexistent").status_code == 404


def test_predict_without_file_shows_error(client):
    response = client.post("/predict", data={})
    assert response.status_code == 400
    assert b"choose a chest X-ray" in response.data


def test_predict_rejects_unsupported_extension(client):
    data = {"file": (io.BytesIO(b"hello"), "notes.txt")}
    response = client.post("/predict", data=data)
    assert response.status_code == 400
    assert b"Unsupported file type" in response.data


def test_predict_rejects_corrupt_image(client):
    data = {"file": (io.BytesIO(b"not really a png"), "scan.png")}
    response = client.post("/predict", data=data)
    assert response.status_code == 400
    assert b"could not be read" in response.data


@pytest.mark.parametrize("threshold", ["abc", "1.5", "-0.1", "nan"])
def test_predict_rejects_bad_threshold(client, threshold):
    data = {"file": (_png_bytes(), "scan.png"), "threshold": threshold}
    response = client.post("/predict", data=data)
    assert response.status_code == 400
    assert b"Threshold" in response.data


def test_predict_valid_image_with_gradcam(client):
    data = {
        "file": (_png_bytes(), "scan.png"),
        "threshold": "0.5",
        "show_cam": "on",
    }
    response = client.post("/predict", data=data)
    assert response.status_code == 200
    assert b"Probability (Pneumonia)" in response.data
    # Both images are inlined; nothing is stored on the server.
    assert response.data.count(b"data:image/jpeg;base64,") == 2


def test_predict_without_gradcam_has_single_image(client):
    data = {"file": (_png_bytes(), "scan.png")}
    response = client.post("/predict", data=data)
    assert response.status_code == 200
    assert response.data.count(b"data:image/jpeg;base64,") == 1


def test_repeated_requests_do_not_leak_hooks(client):
    """Regression: Grad-CAM used to add hooks to the shared model each call."""
    from app.app import model

    def hook_count():
        return sum(
            len(m._forward_hooks) + len(m._backward_hooks)
            for m in model.modules()
        )

    before = hook_count()
    for _ in range(3):
        data = {"file": (_png_bytes(), "scan.png"), "show_cam": "on"}
        assert client.post("/predict", data=data).status_code == 200
    assert hook_count() == before
