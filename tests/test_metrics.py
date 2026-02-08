"""Tests for metrics calculation."""

from __future__ import annotations

import numpy as np
import pytest

from cleanlab_demo.metrics import classification_metrics, regression_metrics


class TestClassificationMetrics:
    """Tests for classification metrics."""

    def test_binary_classification_metrics(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])  # Perfect predictions
        y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])

        metrics = classification_metrics(y_true, y_pred, y_proba)

        assert metrics.primary == pytest.approx(1.0)  # AUC should be 1.0
        assert metrics.details["accuracy"] == pytest.approx(1.0)
        assert metrics.details["f1_weighted"] == pytest.approx(1.0)
        assert "roc_auc" in metrics.details

    def test_imperfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])  # 50% accuracy
        y_proba = np.array([[0.6, 0.4], [0.4, 0.6], [0.4, 0.6], [0.6, 0.4]])

        metrics = classification_metrics(y_true, y_pred, y_proba)

        assert metrics.details["accuracy"] == pytest.approx(0.5)

    def test_no_proba(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        metrics = classification_metrics(y_true, y_pred, None)

        # Primary should be F1 when no proba
        assert metrics.details["accuracy"] == pytest.approx(1.0)
        assert "roc_auc" not in metrics.details


class TestRegressionMetrics:
    """Tests for regression metrics."""

    def test_perfect_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])

        metrics = regression_metrics(y_true, y_pred)

        assert metrics.primary == pytest.approx(1.0)  # R² should be 1.0
        assert metrics.details["r2"] == pytest.approx(1.0)
        assert metrics.details["mae"] == pytest.approx(0.0)
        assert metrics.details["rmse"] == pytest.approx(0.0)

    def test_imperfect_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 3.5, 4.5])  # Off by 0.5

        metrics = regression_metrics(y_true, y_pred)

        assert metrics.details["mae"] == pytest.approx(0.5)
        assert metrics.details["rmse"] == pytest.approx(0.5)
        assert metrics.details["r2"] < 1.0
