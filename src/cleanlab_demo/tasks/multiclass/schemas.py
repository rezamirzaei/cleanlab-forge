from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from cleanlab_demo.tasks.base import DemoConfig, DemoResult


class MulticlassMetrics(BaseModel):
    accuracy: float
    macro_f1: float
    log_loss: float


class MulticlassMetricsByVariant(BaseModel):
    baseline: MulticlassMetrics
    pruned_retrain: MulticlassMetrics


class MulticlassNoiseSummary(BaseModel):
    fraction: float = Field(ge=0.0, le=0.5)
    n_flipped: int = Field(ge=0)


class MulticlassCleanlabSummary(BaseModel):
    cv_folds: int = Field(ge=2, le=20)
    n_issues_found: int = Field(ge=0)
    prune_frac: float = Field(ge=0.0, le=0.2)
    n_pruned: int = Field(ge=0)
    precision_at_prune: float = Field(ge=0.0, le=1.0)
    recall_at_prune: float = Field(ge=0.0, le=1.0)


class MulticlassClassificationConfig(DemoConfig):
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    noise_frac: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="Optional synthetic label noise (0 = real-world run).",
    )
    cv_folds: int = Field(default=5, ge=2, le=20)
    prune_frac: float = Field(default=0.02, ge=0.0, le=0.2)
    max_iter: int = Field(default=500, ge=100, le=5000)


class MulticlassClassificationResult(DemoResult):
    task: Literal["multiclass_classification"] = "multiclass_classification"
    dataset: str
    n_train: int
    n_test: int
    n_classes: int
    noise: MulticlassNoiseSummary
    cleanlab: MulticlassCleanlabSummary
    metrics: MulticlassMetricsByVariant

