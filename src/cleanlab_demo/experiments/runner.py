from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from cleanlab_demo.config import LabelIssue, RunConfig, RunResult, TaskType
from cleanlab_demo.data import DatasetHub
from cleanlab_demo.features import build_preprocessor
from cleanlab_demo.metrics import classification_metrics, regression_metrics
from cleanlab_demo.models import create_estimator
from cleanlab_demo.settings import logger, settings


@dataclass(frozen=True)
class LabelCodec:
    encoder: LabelEncoder

    @classmethod
    def fit(cls, y: pd.Series) -> "LabelCodec":
        enc = LabelEncoder()
        enc.fit(y.to_numpy())
        return cls(encoder=enc)

    def encode(self, y: pd.Series) -> np.ndarray:
        return cast(np.ndarray, self.encoder.transform(y.to_numpy()))

    def decode_one(self, label_int: int) -> str | int:
        value = self.encoder.inverse_transform([label_int])[0]
        if isinstance(value, (np.integer, int)):
            return int(value)
        return str(value)


def _inject_label_noise(y: np.ndarray, *, frac: float, random_state: int) -> np.ndarray:
    if frac <= 0:
        return y
    rng = np.random.default_rng(seed=random_state)
    y_noisy = y.copy()
    n = len(y_noisy)
    n_flip = int(round(frac * n))
    if n_flip <= 0:
        return y_noisy

    classes = np.unique(y_noisy)
    flip_indices = rng.choice(n, size=n_flip, replace=False)
    for idx in flip_indices:
        current = y_noisy[idx]
        other_classes = classes[classes != current]
        if len(other_classes) == 0:
            continue
        y_noisy[idx] = rng.choice(other_classes)
    return y_noisy


def _safe_dataframe_sample(df: pd.DataFrame, *, max_rows: int | None, random_state: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)


def _compute_pred_probs_cv(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    task: TaskType,
    cv_folds: int,
    random_state: int,
) -> np.ndarray:
    if task != TaskType.classification:
        raise ValueError("pred_probs are only computed for classification tasks")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    pred_probs = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)
    return cast(np.ndarray, pred_probs)


def _find_label_issues(
    y_train: np.ndarray,
    pred_probs: np.ndarray,
    label_codec: LabelCodec,
    *,
    max_issues: int,
    score_threshold: float | None,
) -> list[LabelIssue]:
    from cleanlab.filter import find_label_issues

    scores = pred_probs[np.arange(len(y_train)), y_train]
    issue_indices = find_label_issues(
        labels=y_train,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )

    issues: list[LabelIssue] = []
    for raw_idx in issue_indices:
        idx = int(raw_idx)
        score = float(scores[idx])
        if score_threshold is not None and score > score_threshold:
            continue
        suggested_int = int(np.argmax(pred_probs[idx]))
        issues.append(
            LabelIssue(
                index=idx,
                label=label_codec.decode_one(int(y_train[idx])),
                suggested_label=label_codec.decode_one(suggested_int),
                score=score,
            )
        )
        if len(issues) >= max_issues:
            break
    return issues


def _try_datalab_summary(
    df_train_with_labels: pd.DataFrame,
    *,
    label_col: str,
    pred_probs: np.ndarray,
    features: np.ndarray | None,
) -> dict[str, Any]:
    try:
        from cleanlab.datalab import Datalab

        datalab = Datalab(data=df_train_with_labels, label_name=label_col)
        kwargs: dict[str, Any] = {"pred_probs": pred_probs}
        if features is not None:
            kwargs["features"] = features
        datalab.find_issues(**kwargs)

        summary = datalab.get_issue_summary()
        if hasattr(summary, "to_dict"):
            # DataFrame
            issue_summary = cast(Any, summary).reset_index().to_dict(orient="records")
        else:
            issue_summary = summary

        return {"issue_summary": issue_summary}
    except Exception as e:
        return {"error": f"datalab_failed: {type(e).__name__}: {e}"}


def _try_cleanlearning_metrics(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    cv_folds: int,
) -> dict[str, Any]:
    try:
        from cleanlab.classification import CleanLearning
        import inspect

        kwargs: dict[str, Any] = {}
        sig = inspect.signature(CleanLearning)
        if "clf" in sig.parameters:
            kwargs["clf"] = clone(pipeline)
        elif "model" in sig.parameters:
            kwargs["model"] = clone(pipeline)
        else:
            kwargs["clf"] = clone(pipeline)

        if "cv_n_folds" in sig.parameters:
            kwargs["cv_n_folds"] = cv_folds
        elif "cv_folds" in sig.parameters:
            kwargs["cv_folds"] = cv_folds

        cleanlearner = CleanLearning(**kwargs)
        cleanlearner.fit(X_train, y_train)
        y_pred = cast(np.ndarray, cleanlearner.predict(X_test))
        y_proba = cast(np.ndarray, cleanlearner.predict_proba(X_test))
        metrics = classification_metrics(y_test, y_pred, y_proba)
        return {"cleanlearning_metrics": metrics.model_dump(mode="json")}
    except ImportError as e:
        return {"cleanlearning_error": f"cleanlab_not_installed: {e}"}
    except Exception as e:
        return {"cleanlearning_error": f"{type(e).__name__}: {e}"}


class ExperimentRunner:
    def __init__(
        self,
        *,
        data_dir: Path | None = None,
        artifacts_dir: Path | None = None,
        dataset_hub: DatasetHub | None = None,
    ) -> None:
        self.data_dir = data_dir or settings.data_dir
        self.artifacts_dir = artifacts_dir or settings.artifacts_dir
        self.datasets = dataset_hub or DatasetHub(self.data_dir)

    def run(self, config: RunConfig) -> RunResult:
        """Run a single experiment with the given configuration."""
        logger.info(f"Loading dataset: {config.dataset.value}")
        dataset = self.datasets.load(config.dataset)
        df = _safe_dataframe_sample(
            dataset.df, max_rows=config.demo.max_rows, random_state=config.split.random_state
        )
        logger.info(f"Dataset loaded: {len(df):,} rows")

        target_col = config.target_col or dataset.target_col
        if target_col not in df.columns:
            raise ValueError(f"target_col={target_col} missing from dataset columns")

        X = df.drop(columns=[target_col])
        y_raw = df[target_col]

        if config.task == TaskType.classification:
            label_codec = LabelCodec.fit(y_raw)
            y_all = label_codec.encode(y_raw)
            stratify = y_all if config.split.stratify else None
        else:
            label_codec = None
            y_all = y_raw.to_numpy()
            stratify = None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_all,
            test_size=config.split.test_size,
            random_state=config.split.random_state,
            stratify=stratify,
        )

        if config.task == TaskType.classification:
            y_train = _inject_label_noise(
                cast(np.ndarray, y_train),
                frac=config.demo.label_noise_fraction,
                random_state=config.demo.noise_random_state,
            )

        preprocessor = build_preprocessor(X_train, config.features)
        estimator = create_estimator(
            config.task,
            cast(Any, config.model),
            random_state=config.split.random_state,
        )
        pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])

        cleanlab_summary: dict[str, Any] = {}
        label_issues: list[LabelIssue] = []
        pred_probs_cv: np.ndarray | None = None

        if config.task == TaskType.classification and config.cleanlab.enabled:
            try:
                pred_probs_cv = _compute_pred_probs_cv(
                    pipeline,
                    X_train,
                    cast(np.ndarray, y_train),
                    task=config.task,
                    cv_folds=config.cleanlab.cv_folds,
                    random_state=config.split.random_state,
                )

                label_issues = _find_label_issues(
                    cast(np.ndarray, y_train),
                    pred_probs_cv,
                    cast(LabelCodec, label_codec),
                    max_issues=config.cleanlab.max_issues,
                    score_threshold=config.cleanlab.issue_score_threshold,
                )
                cleanlab_summary["n_label_issues"] = len(label_issues)
                examples: list[dict[str, Any]] = []
                for issue in label_issues[: min(20, len(label_issues))]:
                    try:
                        original_index = X_train.index[issue.index]
                        df_index: int | str
                        if isinstance(original_index, (np.integer, int)):
                            df_index = int(original_index)
                        else:
                            df_index = str(original_index)
                        examples.append(
                            {
                                "train_row": issue.index,
                                "df_index": df_index,
                                "score": issue.score,
                                "label": issue.label,
                                "suggested_label": issue.suggested_label,
                                "row": X_train.iloc[issue.index].to_dict(),
                            }
                        )
                    except Exception:
                        continue
                if examples:
                    cleanlab_summary["label_issue_examples"] = examples
            except ImportError as e:
                cleanlab_summary["error"] = f"cleanlab_not_installed: {e}"
            except Exception as e:
                cleanlab_summary["error"] = f"cleanlab_failed: {type(e).__name__}: {e}"

        pipeline.fit(X_train, y_train)

        if config.task == TaskType.classification:
            y_pred = cast(np.ndarray, pipeline.predict(X_test))
            y_proba = cast(np.ndarray, pipeline.predict_proba(X_test))
            metrics = classification_metrics(cast(np.ndarray, y_test), y_pred, y_proba)

            if (
                config.cleanlab.enabled
                and config.cleanlab.use_datalab
                and pred_probs_cv is not None
                and label_codec is not None
            ):
                features_for_datalab: np.ndarray | None = None
                try:
                    X_train_vec = pipeline.named_steps["preprocess"].transform(X_train)
                    # Avoid huge dense matrices: small SVD embedding for issue types that need features.
                    from sklearn.decomposition import TruncatedSVD

                    n_features = int(getattr(X_train_vec, "shape", (0, 0))[1])
                    n_components = min(50, max(2, n_features - 1))
                    if n_features >= 3:
                        features_for_datalab = TruncatedSVD(
                            n_components=n_components,
                            random_state=config.split.random_state,
                        ).fit_transform(X_train_vec)
                except Exception as e:
                    cleanlab_summary["features_failed"] = f"{type(e).__name__}: {e}"

                df_train_with_labels = X_train.copy()
                df_train_with_labels[target_col] = cast(np.ndarray, y_train)
                cleanlab_summary.update(
                    _try_datalab_summary(
                        df_train_with_labels,
                        label_col=target_col,
                        pred_probs=pred_probs_cv,
                        features=features_for_datalab,
                    )
                )

            if config.cleanlab.enabled and config.cleanlab.train_cleanlearning:
                cleanlab_summary.update(
                    _try_cleanlearning_metrics(
                        pipeline,
                        X_train,
                        cast(np.ndarray, y_train),
                        X_test,
                        cast(np.ndarray, y_test),
                        cv_folds=config.cleanlab.cv_folds,
                    )
                )
        else:
            y_pred = cast(np.ndarray, pipeline.predict(X_test))
            metrics = regression_metrics(cast(np.ndarray, y_test), y_pred)

        return RunResult(
            dataset=config.dataset,
            task=cast(TaskType, config.task),
            model=cast(Any, config.model).name,
            n_train=len(X_train),
            n_test=len(X_test),
            metrics=metrics,
            label_issues=label_issues,
            cleanlab_summary=cleanlab_summary,
        )


def run_experiment(config: RunConfig) -> RunResult:
    return ExperimentRunner().run(config)
