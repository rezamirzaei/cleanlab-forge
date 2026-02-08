from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cleanlab_demo.config import (
    CleanlabConfig,
    DatasetName,
    DemoConfig,
    ModelConfig,
    ModelName,
    RunConfig,
    TaskType,
)
from cleanlab_demo.data.schemas import LoadedDataset
from cleanlab_demo.experiments.runner import ExperimentRunner


@dataclass(frozen=True)
class DummyDatasetHub:
    dataset: LoadedDataset

    def load(self, name: DatasetName) -> LoadedDataset:
        return self.dataset


def test_runner_classification_smoke() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], size=n)
    y = np.where(x > 0, ">50K", "<=50K")
    df = pd.DataFrame({"x": x, "cat": cat, "income": y})

    hub = DummyDatasetHub(
        LoadedDataset(
            name=DatasetName.adult_income,
            task=TaskType.classification,
            target_col="income",
            df=df,
        )
    )
    runner = ExperimentRunner(dataset_hub=hub)
    cfg = RunConfig(
        dataset=DatasetName.adult_income,
        model=ModelConfig(name=ModelName.logistic_regression),
        cleanlab=CleanlabConfig(enabled=False),
        demo=DemoConfig(max_rows=200),
    )
    result = runner.run(cfg)
    assert result.task == TaskType.classification
    assert result.n_train + result.n_test == n
    assert "accuracy" in result.metrics.details


def test_runner_regression_smoke() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    cat = rng.choice(["spring", "summer", "fall"], size=n)
    y = 10 * x + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"x": x, "season": cat, "cnt": y})

    hub = DummyDatasetHub(
        LoadedDataset(
            name=DatasetName.bike_sharing,
            task=TaskType.regression,
            target_col="cnt",
            df=df,
        )
    )
    runner = ExperimentRunner(dataset_hub=hub)
    cfg = RunConfig(
        dataset=DatasetName.bike_sharing,
        model=ModelConfig(name=ModelName.ridge),
        cleanlab=CleanlabConfig(enabled=False),
        demo=DemoConfig(max_rows=200),
    )
    result = runner.run(cfg)
    assert result.task == TaskType.regression
    assert result.n_train + result.n_test == n
    assert "r2" in result.metrics.details

