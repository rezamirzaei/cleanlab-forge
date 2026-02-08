from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from cleanlab_demo.config import FeatureEngineeringConfig
from cleanlab_demo.tasks.base import DemoConfig, DemoResult


class MultiannotatorNotes(BaseModel):
    description: str
    n_classes: int = Field(ge=2)
    mean_labels_per_example: float = Field(ge=0.0)
    coverage: float = Field(ge=0.0, le=1.0, description="Fraction of (example, annotator) pairs labeled.")
    missing_consensus_classes: list[int] = Field(
        default_factory=list,
        description="Class ids present in `labels_multiannotator` but absent from consensus labels.",
    )


class MultiannotatorCleanlabSummary(BaseModel):
    consensus_method: str
    top_worst_quality_examples: list[int]
    top_active_learning_examples: list[int]


class MultiannotatorConfig(DemoConfig):
    cv_folds: int = Field(default=5, ge=2, le=20)
    top_k: int = Field(default=20, ge=1, le=100)
    features: FeatureEngineeringConfig = Field(default_factory=FeatureEngineeringConfig)
    max_iter: int = Field(default=500, ge=100, le=5000)


class MultiannotatorResult(DemoResult):
    task: Literal["multiannotator_active_learning"] = "multiannotator_active_learning"
    dataset: str
    n_examples: int
    n_annotators: int
    notes: MultiannotatorNotes
    cleanlab: MultiannotatorCleanlabSummary


def compute_multiannotator_coverage(labels_multiannotator: np.ndarray) -> tuple[float, float]:
    total = labels_multiannotator.size
    if total == 0:
        return 0.0, 0.0
    labeled = float(np.isfinite(labels_multiannotator).sum())
    per_example = np.isfinite(labels_multiannotator).sum(axis=1)
    mean_per_example = float(per_example.mean()) if len(per_example) else 0.0
    return labeled / float(total), mean_per_example
