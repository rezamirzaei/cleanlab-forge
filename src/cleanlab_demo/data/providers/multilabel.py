"""
Multi-label Classification Data Providers.

Provides data providers for multi-label classification tasks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleanlab_demo.tasks.multilabel.provider import MultilabelDataProvider


class OpenMLMultilabelProvider(MultilabelDataProvider):
    """
    Generic provider for OpenML multi-label datasets.

    Handles conversion of target columns to multi-hot encoding.
    """

    def __init__(
        self,
        name: str,
        openml_name: str,
        **fetch_kwargs: Any,
    ) -> None:
        """
        Initialize the provider.

        Args:
            name: Dataset name for reporting
            openml_name: Name of the OpenML dataset
            **fetch_kwargs: Additional kwargs for fetch_openml
        """
        self._name = name
        self._openml_name = openml_name
        self._fetch_kwargs = fetch_kwargs

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def _to_multihot(target_df: pd.DataFrame) -> np.ndarray:
        """Convert target DataFrame to multi-hot encoding."""
        normalized = target_df.astype(str)
        as_bool = normalized.apply(lambda col: col.str.upper().eq("TRUE"))
        return as_bool.to_numpy(dtype=int)

    def load(self, seed: int, **kwargs: Any) -> tuple[pd.DataFrame, np.ndarray]:
        """Load and return features and multi-hot labels."""
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml(name=self._openml_name, as_frame=True, **self._fetch_kwargs)
        X = bunch.data
        y = self._to_multihot(bunch.target)

        return X, y


class EmotionsDataProvider(OpenMLMultilabelProvider):
    """
    OpenML Emotions dataset provider.

    Multi-label classification of music emotions.
    Source: OpenML (name="emotions")
    """

    def __init__(self) -> None:
        super().__init__(
            name="openml:emotions",
            openml_name="emotions",
        )
