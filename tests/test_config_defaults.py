from __future__ import annotations

from cleanlab_demo.config import DatasetName, ModelName, RunConfig, TaskType


def test_defaults_for_adult_income() -> None:
    cfg = RunConfig(dataset=DatasetName.adult_income)
    assert cfg.task == TaskType.classification
    assert cfg.target_col == "income"
    assert cfg.model is not None
    assert cfg.model.name == ModelName.logistic_regression


def test_defaults_for_bike_sharing() -> None:
    cfg = RunConfig(dataset=DatasetName.bike_sharing)
    assert cfg.task == TaskType.regression
    assert cfg.target_col == "cnt"
    assert cfg.cleanlab.enabled is False
    assert cfg.model is not None
    assert cfg.model.name == ModelName.ridge

