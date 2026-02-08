from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from cleanlab_demo.tasks.base import DemoConfig, DemoResult


class RegressionMetrics(BaseModel):
    r2: float
    rmse: float
    mae: float


class RegressionMetricsByVariant(BaseModel):
    baseline: RegressionMetrics
    pruned_retrain: RegressionMetrics


class RegressionNoiseSummary(BaseModel):
    fraction: float = Field(ge=0.0, le=0.5)
    n_corrupted: int = Field(ge=0)


class RegressionCleanlabSummary(BaseModel):
    cv_folds: int = Field(ge=2, le=20)
    n_issues_scored: int = Field(ge=0)
    n_pruned: int = Field(ge=0)
    precision_at_prune: float = Field(ge=0.0, le=1.0)
    recall_at_prune: float = Field(ge=0.0, le=1.0)


class RegressionCleanLearningConfig(DemoConfig):
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    noise_frac: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="Optional synthetic label noise (0 = real-world run).",
    )
    cv_folds: int = Field(default=5, ge=2, le=20)
    prune_frac: float = Field(default=0.02, ge=0.0, le=0.2)
    noise_scale: float = Field(default=5.0, ge=0.1, le=20.0)


class RegressionCleanLearningResult(DemoResult):
    task: Literal["regression_cleanlearning"] = "regression_cleanlearning"
    dataset: str
    n_train: int
    n_test: int
    noise: RegressionNoiseSummary
    cleanlab: RegressionCleanlabSummary
    metrics: RegressionMetricsByVariant

