from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from cleanlab_demo.tasks.base import DemoConfig, DemoResult


class TokenClassificationMetrics(BaseModel):
    token_accuracy: float
    macro_f1: float


class TokenClassificationMetricsByVariant(BaseModel):
    baseline: TokenClassificationMetrics
    pruned_retrain: TokenClassificationMetrics


class TokenClassificationNoiseSummary(BaseModel):
    fraction_tokens: float = Field(ge=0.0, le=0.5)
    n_corrupted_tokens: int = Field(ge=0)


class TokenClassificationCleanlabSummary(BaseModel):
    cv_folds: int = Field(ge=2, le=20)
    n_token_issues_found: int = Field(ge=0)
    n_pruned_tokens: int = Field(ge=0)
    precision_at_prune: float = Field(ge=0.0, le=1.0)
    recall_at_prune: float = Field(ge=0.0, le=1.0)


class TokenClassificationConfig(DemoConfig):
    max_train_sentences: int = Field(default=2000, ge=50, le=50_000)
    max_dev_sentences: int = Field(default=500, ge=50, le=50_000)
    noise_frac: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="Optional synthetic token noise (0 = real-world run).",
    )
    cv_folds: int = Field(default=3, ge=2, le=20)
    prune_frac: float = Field(default=0.03, ge=0.0, le=0.2)
    max_iter: int = Field(default=800, ge=100, le=5000)


class TokenClassificationResult(DemoResult):
    task: Literal["token_classification"] = "token_classification"
    dataset: str
    data_dir: Path | None = None
    n_train_sentences: int
    n_dev_sentences: int
    n_classes: int
    noise: TokenClassificationNoiseSummary
    cleanlab: TokenClassificationCleanlabSummary
    metrics: TokenClassificationMetricsByVariant

