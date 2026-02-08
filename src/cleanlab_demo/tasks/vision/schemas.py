from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from cleanlab_demo.tasks.base import DemoConfig, DemoResult


class VisionNotes(BaseModel):
    model: str
    class_mapping: dict[str, int]
    corrupt_frac: float = Field(ge=0.0, le=1.0)


class ObjectDetectionSummary(BaseModel):
    score_threshold: float = Field(ge=0.0, le=1.0)
    n_corrupted_images: int = Field(ge=0)
    n_flagged_images: int = Field(ge=0)
    precision_vs_corruption: float = Field(ge=0.0, le=1.0)
    recall_vs_corruption: float = Field(ge=0.0, le=1.0)


class SegmentationSummary(BaseModel):
    score_threshold: float = Field(ge=0.0, le=1.0)
    n_corrupted_images: int = Field(ge=0)
    k_flagged: int = Field(ge=1)
    precision_at_k: float = Field(ge=0.0, le=1.0)
    recall_at_k: float = Field(ge=0.0, le=1.0)


class VisionDetectionSegmentationConfig(DemoConfig):
    data_dir: Path | None = Field(default=None, description="Where to cache the dataset.")
    max_images: int = Field(default=8, ge=1, le=200)
    score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    corrupt_frac: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Optional synthetic corruption fraction (0 = real-world run).",
    )


class VisionDetectionSegmentationResult(DemoResult):
    task: Literal["vision_detection_and_segmentation"] = "vision_detection_and_segmentation"
    dataset: str
    n_images: int
    notes: VisionNotes
    object_detection: ObjectDetectionSummary
    segmentation: SegmentationSummary

