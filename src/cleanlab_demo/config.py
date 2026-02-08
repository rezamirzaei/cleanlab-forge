from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class TaskType(str, Enum):
    """Supported ML task types."""

    classification = "classification"
    regression = "regression"


class DatasetName(str, Enum):
    """Available datasets in the demo."""

    adult_income = "adult_income"
    bike_sharing = "bike_sharing"
    california_housing = "california_housing"


@dataclass(frozen=True)
class DatasetDefaults:
    """Default configuration for a dataset."""

    task: TaskType
    target_col: str


DATASET_DEFAULTS: dict[DatasetName, DatasetDefaults] = {
    DatasetName.adult_income: DatasetDefaults(task=TaskType.classification, target_col="income"),
    DatasetName.bike_sharing: DatasetDefaults(task=TaskType.regression, target_col="cnt"),
    DatasetName.california_housing: DatasetDefaults(task=TaskType.regression, target_col="MedHouseVal"),
}


class SplitConfig(BaseModel):
    """Configuration for train/test split."""

    test_size: float = Field(default=0.2, ge=0.05, le=0.5, description="Fraction of data for testing")
    random_state: int = Field(default=42, ge=0, le=1_000_000, description="Random seed for reproducibility")
    stratify: bool = Field(default=True, description="Whether to stratify split (classification only)")


class FeatureEngineeringConfig(BaseModel):
    """Configuration for feature preprocessing."""

    impute_numeric: Literal["median", "mean"] = Field(default="median", description="Strategy for numeric imputation")
    impute_categorical: Literal["most_frequent"] = Field(default="most_frequent", description="Strategy for categorical imputation")
    scale_numeric: bool = Field(default=True, description="Whether to standardize numeric features")
    one_hot_max_categories: int = Field(default=100, ge=2, le=500, description="Max categories for one-hot encoding")


class ModelName(str, Enum):
    """Supported model types."""

    logistic_regression = "logistic_regression"
    random_forest = "random_forest"
    extra_trees = "extra_trees"
    hist_gradient_boosting = "hist_gradient_boosting"
    knn = "knn"
    ridge = "ridge"


class ModelConfig(BaseModel):
    """Configuration for the ML model."""

    name: ModelName = Field(description="Model type to use")
    params: dict[str, Any] = Field(default_factory=dict, description="Additional model hyperparameters")


class CleanlabConfig(BaseModel):
    """Configuration for Cleanlab label issue detection."""

    enabled: bool = Field(default=True, description="Enable Cleanlab analysis")
    cv_folds: int = Field(default=5, ge=2, le=20, description="Number of CV folds for out-of-sample predictions")
    use_datalab: bool = Field(default=True, description="Use Datalab for additional issue types")
    datalab_fast: bool = Field(
        default=True,
        description="Run a faster subset of Datalab checks (label/outlier/near_duplicate/non_iid).",
    )
    train_cleanlearning: bool = Field(default=False, description="Train a CleanLearning model")
    prune_and_retrain: bool = Field(
        default=True,
        description="Train an additional model after pruning the worst issues from the training set.",
    )
    prune_fraction: float = Field(
        default=0.02,
        ge=0.0,
        le=0.2,
        description="Fraction of training data to prune when retraining (0 disables pruning).",
    )
    prune_max_samples: int = Field(
        default=500,
        ge=0,
        le=100_000,
        description="Upper bound on number of samples to prune when retraining.",
    )
    issue_score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="If set, keep only issues with score <= threshold (lower is worse).",
    )
    max_issues: int = Field(default=200, ge=1, le=100_000, description="Maximum number of issues to report")


class DemoConfig(BaseModel):
    """Configuration for demo/debugging features."""

    label_noise_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="Fraction of training labels to randomly flip (classification demo).",
    )
    noise_random_state: int = Field(default=42, ge=0, le=1_000_000, description="Random seed for noise injection")
    max_rows: int | None = Field(
        default=None,
        ge=100,
        le=200_000,
        description="If set, sample at most this many rows from the dataset.",
    )


class RunConfig(BaseModel):
    """Main configuration for running an experiment."""
    dataset: DatasetName = DatasetName.adult_income
    task: TaskType | None = None
    target_col: str | None = None
    split: SplitConfig = Field(default_factory=SplitConfig)
    features: FeatureEngineeringConfig = Field(default_factory=FeatureEngineeringConfig)
    model: ModelConfig | None = None
    cleanlab: CleanlabConfig = Field(default_factory=CleanlabConfig)
    demo: DemoConfig = Field(default_factory=DemoConfig)

    @model_validator(mode="after")
    def _fill_defaults(self) -> RunConfig:
        defaults = DATASET_DEFAULTS[self.dataset]

        if self.task is None:
            self.task = defaults.task
        elif self.task != defaults.task:
            raise ValueError(f"dataset={self.dataset} requires task={defaults.task}")

        if self.target_col is None:
            self.target_col = defaults.target_col

        if self.model is None:
            default_model = (
                ModelName.logistic_regression if self.task == TaskType.classification else ModelName.ridge
            )
            self.model = ModelConfig(name=default_model)

        if self.task == TaskType.regression:
            # Stratification only applies to classification.
            if self.split.stratify:
                self.split.stratify = False
            # CleanLearning is classification-only.
            if self.cleanlab.train_cleanlearning:
                self.cleanlab.train_cleanlearning = False

        return self


class Metrics(BaseModel):
    """Container for model evaluation metrics."""

    primary: float = Field(description="Primary metric (AUC for classification, R² for regression)")
    details: dict[str, float] = Field(default_factory=dict, description="All computed metrics")


class LabelIssue(BaseModel):
    """A potential label issue detected by Cleanlab."""

    index: int = Field(description="Index in the training set")
    label: str | int | float = Field(description="Current label value")
    suggested_label: str | int | float | None = Field(default=None, description="Suggested correct label")
    score: float = Field(description="Confidence score (lower means more likely mislabeled)")


class TrainingVariant(str, Enum):
    """Different training strategies to compare."""

    baseline = "baseline"
    pruned_retrain = "pruned_retrain"
    cleanlearning = "cleanlearning"


class VariantResult(BaseModel):
    """Metrics for a particular training variant."""

    variant: TrainingVariant
    metrics: Metrics
    n_train: int
    notes: dict[str, Any] = Field(default_factory=dict)


class RunResult(BaseModel):
    """Result of a single experiment run."""

    dataset: DatasetName
    task: TaskType
    model: ModelName
    n_train: int = Field(description="Number of training samples")
    n_test: int = Field(description="Number of test samples")
    metrics: Metrics
    label_issues: list[LabelIssue] = Field(default_factory=list, description="Detected label issues")
    variants: list[VariantResult] = Field(
        default_factory=list,
        description="Comparison of baseline vs Cleanlab-enabled training variants.",
    )
    cleanlab_summary: dict[str, Any] = Field(default_factory=dict, description="Cleanlab analysis summary")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO-8601 UTC timestamp of when the experiment was run.",
    )
