from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TokenClassificationDataProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(
        self, seed: int, max_train: int, max_dev: int, **kwargs: Any
    ) -> tuple[list[list[str]], list[list[str]], list[list[str]], list[list[str]]]:
        """Return train_tokens, train_tags, dev_tokens, dev_tags."""

