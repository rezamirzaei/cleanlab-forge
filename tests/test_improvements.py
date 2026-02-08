"""Tests for new improvements: timestamp, precision, sweep fault-tolerance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from cleanlab_demo.config import (
    CleanlabConfig,
    DatasetName,
    DemoConfig,
    ModelConfig,
    ModelName,
    RunConfig,
    RunResult,
    TaskType,
)
from cleanlab_demo.data.schemas import LoadedDataset
from cleanlab_demo.experiments.runner import ExperimentRunner
from cleanlab_demo.experiments.sweep import run_sweep
from cleanlab_demo.metrics import classification_metrics


@dataclass(frozen=True)
class DummyDatasetHub:
    dataset: LoadedDataset

    def load(self, name: DatasetName) -> LoadedDataset:
        return self.dataset


def _make_classification_hub(n: int = 200) -> DummyDatasetHub:
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], size=n)
    y = np.where(x > 0, ">50K", "<=50K")
    df = pd.DataFrame({"x": x, "cat": cat, "income": y})
    return DummyDatasetHub(
        LoadedDataset(
            name=DatasetName.adult_income,
            task=TaskType.classification,
            target_col="income",
            df=df,
        )
    )


class TestRunResultTimestamp:
    """Ensure RunResult always includes an ISO-8601 timestamp."""

    def test_result_has_timestamp(self) -> None:
        hub = _make_classification_hub()
        runner = ExperimentRunner(dataset_hub=hub)
        cfg = RunConfig(
            dataset=DatasetName.adult_income,
            model=ModelConfig(name=ModelName.logistic_regression),
            cleanlab=CleanlabConfig(enabled=False),
            demo=DemoConfig(max_rows=200),
        )
        result = runner.run(cfg)
        assert result.timestamp is not None
        # Must parse as a valid ISO-8601 datetime
        datetime.fromisoformat(result.timestamp)

    def test_timestamp_in_serialized_json(self) -> None:
        hub = _make_classification_hub()
        runner = ExperimentRunner(dataset_hub=hub)
        cfg = RunConfig(
            dataset=DatasetName.adult_income,
            model=ModelConfig(name=ModelName.logistic_regression),
            cleanlab=CleanlabConfig(enabled=False),
            demo=DemoConfig(max_rows=200),
        )
        result = runner.run(cfg)
        data = result.model_dump(mode="json")
        assert "timestamp" in data
        assert isinstance(data["timestamp"], str)


class TestPrecisionMetric:
    """Classification metrics should include precision_weighted."""

    def test_precision_in_details(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])

        metrics = classification_metrics(y_true, y_pred, y_proba)
        assert "precision_weighted" in metrics.details
        assert metrics.details["precision_weighted"] == pytest.approx(1.0)

    def test_precision_imperfect(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        metrics = classification_metrics(y_true, y_pred, None)
        assert "precision_weighted" in metrics.details
        assert 0.0 <= metrics.details["precision_weighted"] <= 1.0


class TestSweepFaultTolerance:
    """Sweep should skip models that fail instead of crashing entirely."""

    def test_sweep_continues_on_model_failure(self) -> None:
        hub = _make_classification_hub()
        runner = ExperimentRunner(dataset_hub=hub)

        original_run = runner.run

        call_count = 0

        def failing_run(config: RunConfig) -> RunResult:
            nonlocal call_count
            call_count += 1
            if config.model and config.model.name == ModelName.knn:
                raise RuntimeError("Simulated failure for knn")
            return original_run(config)

        with patch.object(runner, "run", side_effect=failing_run):
            rows = run_sweep(
                dataset=DatasetName.adult_income,
                models=[ModelName.logistic_regression, ModelName.knn],
                base_config=RunConfig(
                    dataset=DatasetName.adult_income,
                    cleanlab=CleanlabConfig(enabled=False),
                    demo=DemoConfig(max_rows=200),
                ),
                runner=runner,
            )

        # knn should have been called but skipped
        assert call_count == 2
        # Only logistic_regression should remain
        assert len(rows) == 1
        assert rows[0].model == ModelName.logistic_regression
