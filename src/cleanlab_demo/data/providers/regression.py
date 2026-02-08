"""
Regression Data Providers.

Provides data providers for regression tasks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cleanlab_demo.tasks.regression.provider import RegressionDataProvider


class TabularRegressionProvider(RegressionDataProvider):
    """
    Generic provider for tabular regression datasets.

    Can load from CSV, DataFrame, or sklearn datasets.
    """

    def __init__(
        self,
        name: str,
        load_func: Any,
        target_col: str | None = None,
        max_rows: int | None = None,
        **load_kwargs: Any,
    ) -> None:
        """
        Initialize the provider.

        Args:
            name: Dataset name for reporting
            load_func: Function that returns (X, y) or a DataFrame
            target_col: Target column name (if load_func returns DataFrame)
            max_rows: Maximum rows to sample
            **load_kwargs: Additional kwargs for load function
        """
        self._name = name
        self._load_func = load_func
        self._target_col = target_col
        self._max_rows = max_rows
        self._load_kwargs = load_kwargs

    @property
    def name(self) -> str:
        return self._name

    def load(self, seed: int, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        """Load and return features and target."""
        max_rows = kwargs.get("max_rows", self._max_rows)

        result = self._load_func(**self._load_kwargs)

        if isinstance(result, tuple):
            X, y = result
        else:
            df = result
            if max_rows is not None and len(df) > max_rows:
                df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

            target = self._target_col or df.columns[-1]
            X = df.drop(columns=[target]).to_numpy(dtype=float)
            y = df[target].to_numpy(dtype=float)
            return X, y

        if max_rows is not None and len(X) > max_rows:
            rng = np.random.default_rng(seed=seed)
            idx = rng.choice(len(X), size=max_rows, replace=False)
            X = X[idx]
            y = y[idx]

        return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


class CaliforniaHousingDataProvider(TabularRegressionProvider):
    """
    California Housing dataset provider.

    Regression for median house values.
    Source: sklearn.datasets.fetch_california_housing
    """

    def __init__(self, max_rows: int = 12_000) -> None:
        from sklearn.datasets import fetch_california_housing
        super().__init__(
            name="sklearn.fetch_california_housing",
            load_func=fetch_california_housing,
            max_rows=max_rows,
            return_X_y=True,
        )


class BikeSharingDataProvider(RegressionDataProvider):
    """
    UCI Bike Sharing dataset provider.

    Regression for bike rental counts.
    Source: UCI ML Repository
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        max_rows: int = 12_000,
    ) -> None:
        self._data_dir = data_dir
        self._max_rows = max_rows

    @property
    def name(self) -> str:
        return "bike_sharing (UCI)"

    def load(self, seed: int, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        """Load Bike Sharing data."""
        from cleanlab_demo.config import DatasetName
        from cleanlab_demo.data import DatasetHub
        from cleanlab_demo.settings import settings

        max_rows = kwargs.get("max_rows", self._max_rows)
        data_dir = self._data_dir or settings.data_dir

        settings.ensure_dirs()
        hub = DatasetHub(data_dir)
        ds = hub.load(DatasetName.bike_sharing)
        df = ds.df

        if max_rows is not None and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

        X_df = df.drop(columns=[ds.target_col])
        y = df[ds.target_col].to_numpy(dtype=float)

        try:
            X = X_df.to_numpy(dtype=float)
        except Exception as e:
            raise ValueError(
                "Bike Sharing regression expects all features to be numeric. "
                f"Got columns: {list(X_df.columns)}"
            ) from e

        return X, y
