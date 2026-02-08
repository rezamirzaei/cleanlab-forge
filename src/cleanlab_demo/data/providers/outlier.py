"""
Outlier Detection Data Providers.

Provides data providers for outlier detection tasks.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from cleanlab_demo.tasks.outlier.provider import OutlierDetectionDataProvider


class CaliforniaHousingOutlierProvider(OutlierDetectionDataProvider):
    """
    California Housing dataset provider for outlier detection.

    Uses regression task with Datalab for outlier detection.
    Source: sklearn.datasets.fetch_california_housing
    """

    def __init__(self, max_rows: int = 12_000) -> None:
        self._max_rows = max_rows

    @property
    def name(self) -> str:
        return "sklearn.fetch_california_housing"

    @property
    def label_col(self) -> str:
        return "MedHouseVal"

    @property
    def task_type(self) -> str:
        return "regression"

    def load(self, seed: int, **kwargs: Any) -> pd.DataFrame:
        """Load California Housing data as DataFrame."""
        from sklearn.datasets import fetch_california_housing

        max_rows = kwargs.get("max_rows", self._max_rows)

        data = fetch_california_housing(as_frame=True)
        df = data.frame

        if max_rows is not None and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

        return df
