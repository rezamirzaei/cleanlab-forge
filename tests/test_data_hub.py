"""Tests for the data hub and dataset loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cleanlab_demo.config import DatasetName, TaskType
from cleanlab_demo.data.hub import DatasetHub
from cleanlab_demo.data.schemas import LoadedDataset


def test_dataset_hub_spec():
    """Test that dataset specs are registered correctly."""
    hub = DatasetHub(Path("data"))

    adult_spec = hub.spec(DatasetName.adult_income)
    assert adult_spec.task == TaskType.classification
    assert adult_spec.target_col == "income"

    bike_spec = hub.spec(DatasetName.bike_sharing)
    assert bike_spec.task == TaskType.regression
    assert bike_spec.target_col == "cnt"


def test_loaded_dataset_features_df():
    """Test that LoadedDataset.features_df excludes target column."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "target": [0, 1]})
    dataset = LoadedDataset(
        name=DatasetName.adult_income,
        task=TaskType.classification,
        target_col="target",
        df=df,
    )

    features = dataset.features_df()
    assert "target" not in features.columns
    assert list(features.columns) == ["a", "b"]


def test_loaded_dataset_target_series():
    """Test that LoadedDataset.target_series returns the target column."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "target": [0, 1]})
    dataset = LoadedDataset(
        name=DatasetName.adult_income,
        task=TaskType.classification,
        target_col="target",
        df=df,
    )

    target = dataset.target_series()
    assert target.tolist() == [0, 1]

