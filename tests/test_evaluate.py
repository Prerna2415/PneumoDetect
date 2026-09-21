"""
tests/test_evaluate.py
----------------------
Metrics are hand-checkable on tiny examples.
"""

import math

import pytest
import torch

from src.evaluate import compute_metrics, evaluate


def test_perfect_predictions():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["accuracy"] == 1.0
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["auc"] == 1.0


def test_sensitivity_and_specificity_are_computed_separately():
    # 2 pneumonia cases (one caught), 2 normal cases (both cleared)
    m = compute_metrics([1, 1, 0, 0], [0.9, 0.2, 0.1, 0.3])
    assert m["sensitivity"] == pytest.approx(0.5)
    assert m["specificity"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(0.75)


def test_lower_threshold_raises_sensitivity():
    labels, probs = [1, 1, 0, 0], [0.9, 0.4, 0.1, 0.3]
    strict = compute_metrics(labels, probs, threshold=0.5)
    lenient = compute_metrics(labels, probs, threshold=0.35)
    assert lenient["sensitivity"] > strict["sensitivity"]


def test_always_normal_model_is_exposed():
    """High accuracy with zero sensitivity: why accuracy alone misleads."""
    labels = [0] * 9 + [1]
    m = compute_metrics(labels, [0.0] * 10)
    assert m["accuracy"] == pytest.approx(0.9)
    assert m["sensitivity"] == 0.0


def test_auc_is_nan_with_one_class():
    assert math.isnan(compute_metrics([0, 0], [0.1, 0.2])["auc"])


def test_evaluate_uses_pneumonia_index():
    """A model that always outputs 'pneumonia' must score sensitivity 1."""

    class AlwaysPneumonia(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[-5.0, 5.0]] * x.shape[0])

    batch = (torch.zeros(4, 3, 8, 8), torch.tensor([0, 1, 1, 0]))
    m = evaluate(AlwaysPneumonia(), [batch], torch.device("cpu"))
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 0.0
