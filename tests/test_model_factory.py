"""Tests for the model factory."""

from __future__ import annotations

import pytest
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge

from cleanlab_demo.config import ModelConfig, ModelName, TaskType
from cleanlab_demo.models.factory import create_estimator


class TestClassificationModels:
    """Tests for classification model creation."""

    def test_logistic_regression(self):
        config = ModelConfig(name=ModelName.logistic_regression)
        model = create_estimator(TaskType.classification, config, random_state=42)
        assert isinstance(model, LogisticRegression)
        assert model.random_state == 42

    def test_random_forest_classifier(self):
        config = ModelConfig(name=ModelName.random_forest)
        model = create_estimator(TaskType.classification, config, random_state=42)
        assert isinstance(model, RandomForestClassifier)
        assert model.random_state == 42

    def test_hist_gradient_boosting_classifier(self):
        config = ModelConfig(name=ModelName.hist_gradient_boosting)
        model = create_estimator(TaskType.classification, config, random_state=42)
        assert isinstance(model, HistGradientBoostingClassifier)
        assert model.random_state == 42

    def test_custom_params(self):
        config = ModelConfig(name=ModelName.logistic_regression, params={"max_iter": 1000})
        model = create_estimator(TaskType.classification, config, random_state=42)
        assert model.max_iter == 1000


class TestRegressionModels:
    """Tests for regression model creation."""

    def test_ridge(self):
        config = ModelConfig(name=ModelName.ridge)
        model = create_estimator(TaskType.regression, config, random_state=42)
        assert isinstance(model, Ridge)

    def test_random_forest_regressor(self):
        config = ModelConfig(name=ModelName.random_forest)
        model = create_estimator(TaskType.regression, config, random_state=42)
        assert isinstance(model, RandomForestRegressor)
        assert model.random_state == 42

    def test_hist_gradient_boosting_regressor(self):
        config = ModelConfig(name=ModelName.hist_gradient_boosting)
        model = create_estimator(TaskType.regression, config, random_state=42)
        assert isinstance(model, HistGradientBoostingRegressor)
        assert model.random_state == 42


def test_unsupported_classification_model():
    """Test that unsupported classification model raises error."""
    config = ModelConfig(name=ModelName.ridge)  # Ridge is regression only
    with pytest.raises(ValueError, match="Unsupported classification model"):
        create_estimator(TaskType.classification, config, random_state=42)


def test_unsupported_regression_model():
    """Test that unsupported regression model raises error."""
    config = ModelConfig(name=ModelName.logistic_regression)  # Logistic is classification only
    with pytest.raises(ValueError, match="Unsupported regression model"):
        create_estimator(TaskType.regression, config, random_state=42)
