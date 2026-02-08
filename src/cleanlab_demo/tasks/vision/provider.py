from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class VisionDataProvider(ABC):
    """Interface for vision datasets used in detection/segmentation tasks."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def class_mapping(self) -> dict[str, int]: ...

    @abstractmethod
    def load(
        self, data_dir: Path, max_images: int, seed: int, **kwargs: Any
    ) -> tuple[list[Any], list[np.ndarray], list[np.ndarray], list[tuple[int, int]]]:
        """
        Return images and ground-truth.

        Returns:
            images: list of image tensors
            gt_boxes: list of (N, 4) arrays in xyxy format
            gt_masks: list of (H, W) integer masks
            img_sizes: list of (width, height)
        """

    @abstractmethod
    def get_model(self) -> tuple[Any, int]:
        """Return (model, target_class_id)."""

