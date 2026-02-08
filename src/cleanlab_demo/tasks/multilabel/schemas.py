from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from cleanlab_demo.tasks.base import DemoConfig, DemoResult


class MultilabelMetrics(BaseModel):
    micro_f1: float
    macro_f1: float
    subset_accuracy: float
    hamming_loss: float


class MultilabelMetricsByVariant(BaseModel):
    baseline: MultilabelMetrics
    pruned_retrain: MultilabelMetrics


class MultilabelNoiseSummary(BaseModel):
    fraction_examples: float = Field(ge=0.0, le=0.5)
    n_noisy_examples: int = Field(ge=0)


class MultilabelCleanlabSummary(BaseModel):
    cv_folds: int = Field(ge=2, le=20)
    n_issues_found: int = Field(ge=0)
    n_pruned: int = Field(ge=0)
    precision_at_prune: float = Field(ge=0.0, le=1.0)
    recall_at_prune: float = Field(ge=0.0, le=1.0)


class MultilabelClassificationConfig(DemoConfig):
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    noise_frac: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="Optional synthetic tag noise (0 = real-world run).",
    )
    cv_folds: int = Field(default=5, ge=2, le=20)
    prune_frac: float = Field(default=0.05, ge=0.0, le=0.5)


class MultilabelClassificationResult(DemoResult):
    task: Literal["multilabel_classification"] = "multilabel_classification"
    dataset: str
    n_train: int
    n_test: int
    n_labels: int
    noise: MultilabelNoiseSummary
    cleanlab: MultilabelCleanlabSummary
    metrics: MultilabelMetricsByVariant

