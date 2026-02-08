from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class MulticlassDataProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(self, seed: int, **kwargs: Any) -> tuple[pd.DataFrame, np.ndarray]:
        """Return (X, y) where y contains integer class labels."""

