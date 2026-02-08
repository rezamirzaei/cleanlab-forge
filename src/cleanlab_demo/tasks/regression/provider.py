from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class RegressionDataProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(self, seed: int, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) where X is numeric features and y is continuous target."""

