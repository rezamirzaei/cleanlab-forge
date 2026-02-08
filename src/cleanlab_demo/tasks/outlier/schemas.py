from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from cleanlab_demo.tasks.base import DemoConfig, DemoResult


class SyntheticOutliersSummary(BaseModel):
    fraction: float = Field(ge=0.0, le=0.2)
    n_injected: int = Field(ge=0)


class OutlierRow(BaseModel):
    row: int
    outlier_score: float
    is_outlier_issue: bool


class OutlierDetectionCleanlabSummary(BaseModel):
    issue_types: list[str]
    precision_vs_injected: float = Field(ge=0.0, le=1.0)
    recall_vs_injected: float = Field(ge=0.0, le=1.0)
    issue_summary: list[dict[str, object]]
    top_outliers: list[OutlierRow]


class OutlierDetectionConfig(DemoConfig):
    outlier_frac: float = Field(
        default=0.0,
        ge=0.0,
        le=0.2,
        description="Optional synthetic outliers for evaluation (0 = real-world run).",
    )
    outlier_scale: float = Field(default=50.0, ge=1.0, le=100.0)
    top_k_outliers: int = Field(default=15, ge=1, le=100)


class OutlierDetectionResult(DemoResult):
    task: Literal["outlier_detection"] = "outlier_detection"
    dataset: str
    n_rows: int
    synthetic_outliers: SyntheticOutliersSummary
    cleanlab: OutlierDetectionCleanlabSummary

