from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class OutlierDetectionDataProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def label_col(self) -> str: ...

    @property
    @abstractmethod
    def task_type(self) -> str: ...

    @abstractmethod
    def load(self, seed: int, **kwargs: Any) -> pd.DataFrame:
        """Return a DataFrame that includes the label column."""

