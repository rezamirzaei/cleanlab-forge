"""
Multi-class Classification Data Providers.

Provides data providers for multi-class classification tasks.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from cleanlab_demo.tasks.multiclass.provider import MulticlassDataProvider


class SKLearnDatasetProvider(MulticlassDataProvider):
    """
    Generic provider for sklearn datasets.

    Can be used with any sklearn fetch_* function that returns
    a Bunch with 'data' and 'target' attributes.
    """

    def __init__(
        self,
        name: str,
        fetch_func: Callable[..., Any],
        max_rows: int | None = None,
        target_offset: int = 0,
        **fetch_kwargs: Any,
    ) -> None:
        """
        Initialize the provider.

        Args:
            name: Dataset name for reporting
            fetch_func: sklearn fetch function (e.g., fetch_covtype)
            max_rows: Maximum rows to sample (None for all)
            target_offset: Offset to subtract from targets (e.g., 1 for covtype)
            **fetch_kwargs: Additional kwargs for fetch function
        """
        self._name = name
        self._fetch_func = fetch_func
        self._max_rows = max_rows
        self._target_offset = target_offset
        self._fetch_kwargs = fetch_kwargs

    @property
    def name(self) -> str:
        return self._name

    def load(self, seed: int, **kwargs: Any) -> tuple[pd.DataFrame, np.ndarray]:
        """Load and return features and labels."""
        max_rows = kwargs.get("max_rows", self._max_rows)

        bunch = self._fetch_func(as_frame=True, **self._fetch_kwargs)
        df = bunch.frame

        # Sample if needed
        if max_rows is not None and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

        # Separate features and target
        target_col = bunch.target_names if hasattr(bunch, 'target_names') else df.columns[-1]
        if isinstance(target_col, list):
            target_col = target_col[0] if len(target_col) == 1 else df.columns[-1]

        # For datasets like covtype, the target is in a specific column
        if hasattr(bunch, 'target') and isinstance(bunch.target, pd.Series):
            target_col = bunch.target.name or df.columns[-1]

        X = df.drop(columns=[target_col] if target_col in df.columns else [df.columns[-1]])
        y = df[target_col].to_numpy(dtype=int) - self._target_offset

        return X, y


class CovtypeDataProvider(SKLearnDatasetProvider):
    """
    Forest CoverType dataset provider.

    7-class classification of forest cover types.
    Source: sklearn.datasets.fetch_covtype
    """

    def __init__(self, max_rows: int = 20_000) -> None:
        from sklearn.datasets import fetch_covtype
        super().__init__(
            name="sklearn.fetch_covtype",
            fetch_func=fetch_covtype,
            max_rows=max_rows,
            target_offset=1,  # Covtype uses 1-7, we want 0-6
        )

    def load(self, seed: int, **kwargs: Any) -> tuple[pd.DataFrame, np.ndarray]:
        """Load CoverType data."""
        from sklearn.datasets import fetch_covtype

        max_rows = kwargs.get("max_rows", self._max_rows)

        bunch = fetch_covtype(as_frame=True)
        df = bunch.frame

        if max_rows is not None and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

        X = df.drop(columns=["Cover_Type"])
        y = (df["Cover_Type"].to_numpy(dtype=int) - 1).astype(int)  # 0..6

        return X, y
