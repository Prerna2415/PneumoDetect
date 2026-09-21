"""
tests/test_threshold_logic.py
-----------------------------
Tests threshold logic and Flask routes for PneumoDetect.
"""

import pytest
from src.inference import apply_threshold
from app.app import app

# -------------------------------
# Threshold Logic Unit Tests
# -------------------------------


def test_apply_threshold_high():
    assert apply_threshold(0.86, 0.8) == "High Risk"


def test_apply_threshold_low():
    assert apply_threshold(0.3, 0.8) == "Low Risk"


# -------------------------------
# Flask Route Tests
# -------------------------------
@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home_route(client):
    """Ensure home page loads correctly."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"PneumoDetect" in response.data


def test_predict_route_rejects_missing_file(client):
    """POST without a file re-renders the form with an error."""
    response = client.post("/predict", data={})
    assert response.status_code == 400


def test_apply_threshold_boundary_is_low_risk():
    """Probability exactly at the threshold is not flagged."""
    assert apply_threshold(0.5, 0.5) == "Low Risk"
