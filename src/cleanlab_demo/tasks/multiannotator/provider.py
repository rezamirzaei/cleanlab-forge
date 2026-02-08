from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class MultiannotatorDataProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(self, seed: int, **kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return (X, labels_multiannotator).

        `labels_multiannotator` is a DataFrame of shape (N, M) containing integer class labels
        and NaN for missing labels.
        """

