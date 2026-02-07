from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from cleanlab_demo.config import ModelConfig, ModelName, TaskType


def create_estimator(task: TaskType, model_config: ModelConfig, *, random_state: int) -> Any:
    name = model_config.name
    params = dict(model_config.params)

    if task == TaskType.classification:
        if name == ModelName.logistic_regression:
            defaults = {"max_iter": 500, "solver": "lbfgs", "random_state": random_state}
            return LogisticRegression(**{**defaults, **params})
        if name == ModelName.random_forest:
            defaults = {"n_estimators": 300, "n_jobs": 1, "random_state": random_state}
            return RandomForestClassifier(**{**defaults, **params})
        if name == ModelName.hist_gradient_boosting:
            defaults = {"random_state": random_state}
            return HistGradientBoostingClassifier(**{**defaults, **params})
        raise ValueError(f"Unsupported classification model: {name}")

    if task == TaskType.regression:
        if name == ModelName.ridge:
            defaults: dict[str, Any] = {}
            return Ridge(**{**defaults, **params})
        if name == ModelName.random_forest:
            defaults = {"n_estimators": 500, "n_jobs": 1, "random_state": random_state}
            return RandomForestRegressor(**{**defaults, **params})
        if name == ModelName.hist_gradient_boosting:
            defaults = {"random_state": random_state}
            return HistGradientBoostingRegressor(**{**defaults, **params})
        raise ValueError(f"Unsupported regression model: {name}")

    raise ValueError(f"Unknown task: {task}")
