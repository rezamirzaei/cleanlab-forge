from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from cleanlab_demo.config import (
    LabelIssue,
    Metrics,
    RunConfig,
    RunResult,
    TaskType,
    TrainingVariant,
    VariantResult,
)
from cleanlab_demo.data import DatasetHub
from cleanlab_demo.features import build_preprocessor
from cleanlab_demo.metrics import classification_metrics, regression_metrics
from cleanlab_demo.models import create_estimator
from cleanlab_demo.settings import logger, settings


@dataclass(frozen=True)
class LabelCodec:
    encoder: LabelEncoder

    @classmethod
    def fit(cls, y: pd.Series) -> LabelCodec:
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
    n_flip = round(frac * n)
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


def _compute_datalab_features(
    pipeline: Pipeline, X_train: pd.DataFrame, *, random_state: int
) -> tuple[np.ndarray | None, str | None]:
    try:
        from scipy.sparse import issparse
        from sklearn.decomposition import TruncatedSVD

        X_vec = pipeline.named_steps["preprocess"].transform(X_train)
        n_features = int(getattr(X_vec, "shape", (0, 0))[1])
        if n_features == 0:
            return None, "no_features"

        if n_features <= 50:
            if issparse(X_vec):
                return cast(np.ndarray, X_vec.toarray()), None
            return cast(np.ndarray, np.asarray(X_vec)), None

        n_components = min(50, max(2, n_features - 1))
        features = TruncatedSVD(n_components=n_components, random_state=random_state).fit_transform(X_vec)
        return cast(np.ndarray, features), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _run_datalab(
    df_train_with_labels: pd.DataFrame,
    X_train: pd.DataFrame,
    *,
    task: TaskType,
    label_col: str,
    pred_probs: np.ndarray | None,
    features: np.ndarray | None,
    fast: bool,
    max_examples: int = 20,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    try:
        from cleanlab.datalab.datalab import Datalab

        issue_types: dict[str, Any] | None = None
        if fast:
            issue_types = {
                "label": {},
                "outlier": {},
                "near_duplicate": {},
                "non_iid": {},
            }

        datalab = Datalab(
            data=df_train_with_labels,
            task=task.value,
            label_name=label_col,
            verbosity=0,
        )
        datalab.find_issues(pred_probs=pred_probs, features=features, issue_types=issue_types)

        issue_summary_df = datalab.get_issue_summary().reset_index(drop=True)
        issues_df = datalab.get_issues().reset_index(drop=True)

        payload: dict[str, Any] = {
            "datalab_issue_summary": issue_summary_df.to_dict(orient="records"),
            "datalab_issue_columns": list(issues_df.columns),
        }

        def _examples_for(issue_name: str, score_col: str) -> list[dict[str, Any]]:
            if score_col not in issues_df.columns:
                return []
            tmp = issues_df.reset_index(names="train_row")
            flag_col = f"is_{issue_name}_issue"
            if flag_col in tmp.columns:
                tmp = tmp[tmp[flag_col].fillna(False).astype(bool)]

            tmp = tmp.sort_values(score_col, ascending=True).head(max_examples)
            out: list[dict[str, Any]] = []
            for _, r in tmp.iterrows():
                idx = int(r["train_row"])
                row = {
                    "train_row": idx,
                    score_col: float(r[score_col]),
                }
                if flag_col in tmp.columns:
                    row[flag_col] = bool(r.get(flag_col))
                if issue_name == "label" and "predicted_label" in tmp.columns:
                    pred = r.get("predicted_label")
                    if pred is not None:
                        if isinstance(pred, (np.integer, int)):
                            row["predicted_label"] = int(pred)
                        elif isinstance(pred, (np.floating, float)):
                            row["predicted_label"] = float(pred)
                        else:
                            row["predicted_label"] = str(pred)
                try:
                    label_val = df_train_with_labels.iloc[idx][label_col]
                    if isinstance(label_val, (np.integer, int)):
                        row["label"] = int(label_val)
                    elif isinstance(label_val, (np.floating, float)):
                        row["label"] = float(label_val)
                    elif label_val is not None:
                        row["label"] = str(label_val)
                except Exception:
                    pass
                with contextlib.suppress(Exception):
                    row["row"] = X_train.iloc[idx].to_dict()
                out.append(row)
            return out

        payload["datalab_examples"] = {
            "label": _examples_for("label", "label_score"),
            "outlier": _examples_for("outlier", "outlier_score"),
            "near_duplicate": _examples_for("near_duplicate", "near_duplicate_score"),
            "non_iid": _examples_for("non_iid", "non_iid_score"),
        }

        return payload, issues_df
    except ImportError as e:
        return {"datalab_error": f"datalab_not_available: {e}"}, None
    except Exception as e:
        return {"datalab_error": f"datalab_failed: {type(e).__name__}: {e}"}, None


def _try_cleanlearning_metrics(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    cv_folds: int,
) -> tuple[Metrics | None, str | None]:
    try:
        import inspect

        from cleanlab.classification import CleanLearning

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
        return metrics, None
    except ImportError as e:
        return None, f"cleanlab_not_installed: {e}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


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
        settings.ensure_dirs()
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
        variants: list[VariantResult] = []
        pred_probs_cv: np.ndarray | None = None
        datalab_issues: pd.DataFrame | None = None

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
            except ImportError as e:
                cleanlab_summary["error"] = f"cleanlab_not_installed: {e}"
            except Exception as e:
                cleanlab_summary["error"] = f"cleanlab_failed: {type(e).__name__}: {e}"

        pipeline.fit(X_train, y_train)

        if config.task == TaskType.classification:
            y_pred = cast(np.ndarray, pipeline.predict(X_test))
            y_proba = cast(np.ndarray, pipeline.predict_proba(X_test))
            metrics = classification_metrics(cast(np.ndarray, y_test), y_pred, y_proba)
        else:
            y_pred = cast(np.ndarray, pipeline.predict(X_test))
            metrics = regression_metrics(cast(np.ndarray, y_test), y_pred)

        variants.append(
            VariantResult(
                variant=TrainingVariant.baseline,
                metrics=metrics,
                n_train=len(X_train),
            )
        )

        if config.cleanlab.enabled and config.cleanlab.use_datalab:
            features_for_datalab, features_err = _compute_datalab_features(
                pipeline, X_train, random_state=config.split.random_state
            )
            if features_err:
                cleanlab_summary["datalab_features_error"] = features_err

            df_train_with_labels = X_train.copy()
            df_train_with_labels[target_col] = cast(np.ndarray, y_train)
            datalab_payload, datalab_issues = _run_datalab(
                df_train_with_labels,
                X_train,
                task=config.task,
                label_col=target_col,
                pred_probs=pred_probs_cv if config.task == TaskType.classification else None,
                features=features_for_datalab,
                fast=config.cleanlab.datalab_fast,
            )
            cleanlab_summary.update(datalab_payload)

            if datalab_issues is not None:
                try:
                    issues_path = settings.artifacts_dir / "last_datalab_issues.csv"
                    datalab_issues.to_csv(issues_path, index=False)
                    cleanlab_summary["datalab_issues_csv"] = str(issues_path)
                except Exception as e:
                    cleanlab_summary["datalab_save_failed"] = f"{type(e).__name__}: {e}"

                # If we didn't compute label issues via pred_probs (e.g., regression), populate label_issues from Datalab.
                if config.task == TaskType.regression and not label_issues and "label_score" in datalab_issues.columns:
                    tmp = datalab_issues.reset_index(names="train_row").sort_values("label_score", ascending=True)
                    for _, r in tmp.head(config.cleanlab.max_issues).iterrows():
                        idx = int(r["train_row"])
                        if bool(r.get("is_label_issue", True)) is False:
                            continue
                        label_issues.append(
                            LabelIssue(
                                index=idx,
                                label=float(cast(np.ndarray, y_train)[idx]),
                                suggested_label=float(r.get("predicted_label")) if "predicted_label" in r else None,
                                score=float(r["label_score"]),
                            )
                        )
                    cleanlab_summary["n_label_issues"] = len(label_issues)

        # Prune & retrain variant (classification: uses label_issues; regression: uses Datalab label issues if available).
        if (
            config.cleanlab.enabled
            and config.cleanlab.prune_and_retrain
            and config.cleanlab.prune_fraction > 0
            and len(X_train) > 0
        ):
            n_prune_target = min(
                config.cleanlab.prune_max_samples,
                round(config.cleanlab.prune_fraction * len(X_train)),
            )
            if n_prune_target > 0:
                prune_indices: list[int] = []
                if label_issues:
                    prune_indices = [li.index for li in label_issues[:n_prune_target]]
                elif datalab_issues is not None:
                    if {"is_label_issue", "label_score"}.issubset(datalab_issues.columns):
                        tmp = datalab_issues[datalab_issues["is_label_issue"]]
                        tmp = tmp.reset_index(names="train_row").sort_values("label_score", ascending=True)
                        prune_indices = [int(v) for v in tmp["train_row"].head(n_prune_target).tolist()]
                    if not prune_indices and {"is_outlier_issue", "outlier_score"}.issubset(datalab_issues.columns):
                        tmp = datalab_issues[datalab_issues["is_outlier_issue"]]
                        tmp = tmp.reset_index(names="train_row").sort_values("outlier_score", ascending=True)
                        prune_indices = [int(v) for v in tmp["train_row"].head(n_prune_target).tolist()]

                prune_indices = sorted({i for i in prune_indices if 0 <= i < len(X_train)})
                if prune_indices:
                    keep_mask = np.ones(len(X_train), dtype=bool)
                    keep_mask[prune_indices] = False
                    X_train_pruned = X_train.iloc[keep_mask]
                    y_train_pruned = cast(np.ndarray, y_train)[keep_mask]

                    pruned_pipeline = clone(pipeline)
                    pruned_pipeline.fit(X_train_pruned, y_train_pruned)

                    if config.task == TaskType.classification:
                        y_pred2 = cast(np.ndarray, pruned_pipeline.predict(X_test))
                        y_proba2 = cast(np.ndarray, pruned_pipeline.predict_proba(X_test))
                        pruned_metrics = classification_metrics(cast(np.ndarray, y_test), y_pred2, y_proba2)
                    else:
                        y_pred2 = cast(np.ndarray, pruned_pipeline.predict(X_test))
                        pruned_metrics = regression_metrics(cast(np.ndarray, y_test), y_pred2)

                    variants.append(
                        VariantResult(
                            variant=TrainingVariant.pruned_retrain,
                            metrics=pruned_metrics,
                            n_train=len(X_train_pruned),
                            notes={"n_pruned": len(prune_indices)},
                        )
                    )

        if config.task == TaskType.classification and config.cleanlab.enabled and config.cleanlab.train_cleanlearning:
            cl_metrics, cl_error = _try_cleanlearning_metrics(
                pipeline,
                X_train,
                cast(np.ndarray, y_train),
                X_test,
                cast(np.ndarray, y_test),
                cv_folds=config.cleanlab.cv_folds,
            )
            if cl_error:
                cleanlab_summary["cleanlearning_error"] = cl_error
            if cl_metrics is not None:
                cleanlab_summary["cleanlearning_metrics"] = cl_metrics.model_dump(mode="json")
                variants.append(
                    VariantResult(
                        variant=TrainingVariant.cleanlearning,
                        metrics=cl_metrics,
                        n_train=len(X_train),
                    )
                )

        return RunResult(
            dataset=config.dataset,
            task=cast(TaskType, config.task),
            model=cast(Any, config.model).name,
            n_train=len(X_train),
            n_test=len(X_test),
            metrics=metrics,
            label_issues=label_issues,
            variants=variants,
            cleanlab_summary=cleanlab_summary,
        )


def run_experiment(config: RunConfig) -> RunResult:
    return ExperimentRunner().run(config)
