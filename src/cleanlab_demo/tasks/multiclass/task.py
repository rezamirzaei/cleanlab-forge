from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from cleanlab_demo.tasks.multiclass.provider import MulticlassDataProvider
from cleanlab_demo.tasks.multiclass.schemas import (
    MulticlassClassificationConfig,
    MulticlassClassificationResult,
    MulticlassCleanlabSummary,
    MulticlassMetrics,
    MulticlassMetricsByVariant,
    MulticlassNoiseSummary,
)


def inject_multiclass_label_noise(
    y: np.ndarray, *, frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if frac <= 0:
        return y.copy(), np.array([], dtype=int)

    rng = np.random.default_rng(seed=seed)
    y_noisy = y.copy()
    n_flip = round(frac * len(y_noisy))
    if n_flip <= 0:
        return y_noisy, np.array([], dtype=int)

    classes = np.unique(y_noisy)
    flip_indices = rng.choice(len(y_noisy), size=n_flip, replace=False)
    for idx in flip_indices:
        current = y_noisy[int(idx)]
        other_classes = classes[classes != current]
        if len(other_classes) == 0:
            continue
        y_noisy[int(idx)] = rng.choice(other_classes)
    return y_noisy, np.sort(flip_indices.astype(int))


def build_multiclass_model(seed: int, max_iter: int = 500) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=max_iter,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


def evaluate_multiclass(model: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray) -> MulticlassMetrics:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    classes = model.named_steps["model"].classes_
    return MulticlassMetrics(
        accuracy=float(accuracy_score(y_test, y_pred)),
        macro_f1=float(f1_score(y_test, y_pred, average="macro")),
        log_loss=float(log_loss(y_test, y_proba, labels=classes)),
    )


class MulticlassClassificationTask:
    def __init__(self, data_provider: MulticlassDataProvider) -> None:
        self.data_provider = data_provider

    def run(self, config: MulticlassClassificationConfig) -> MulticlassClassificationResult:
        from cleanlab.filter import find_label_issues

        X, y = self.data_provider.load(seed=config.seed)

        if not np.issubdtype(y.dtype, np.integer):
            y = LabelEncoder().fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.seed, stratify=y
        )

        y_train_noisy, flipped_indices = inject_multiclass_label_noise(
            y_train, frac=config.noise_frac, seed=config.seed
        )
        flipped_set = set(map(int, flipped_indices.tolist()))

        baseline = build_multiclass_model(config.seed, config.max_iter)
        baseline.fit(X_train, y_train_noisy)
        baseline_metrics = evaluate_multiclass(baseline, X_test, y_test)

        cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
        pred_probs_cv = cross_val_predict(
            build_multiclass_model(config.seed, config.max_iter),
            X_train,
            y_train_noisy,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )

        issue_indices = find_label_issues(
            labels=y_train_noisy,
            pred_probs=np.asarray(pred_probs_cv, dtype=float),
            return_indices_ranked_by="self_confidence",
            n_jobs=1,
        )
        issue_indices = np.asarray(issue_indices, dtype=int)

        n_prune = min(len(issue_indices), round(config.prune_frac * len(y_train_noisy)))
        prune_set = set(map(int, issue_indices[:n_prune].tolist()))
        tp = len(prune_set & flipped_set)
        recall_at_prune = float(tp / len(flipped_set)) if flipped_set else 0.0
        precision_at_prune = float(tp / len(prune_set)) if prune_set else 0.0

        keep_mask = np.ones(len(y_train_noisy), dtype=bool)
        if prune_set:
            keep_mask[list(prune_set)] = False
        X_train_pruned = (
            X_train.iloc[keep_mask] if isinstance(X_train, pd.DataFrame) else X_train[keep_mask]
        )
        y_train_pruned = y_train_noisy[keep_mask]

        pruned = build_multiclass_model(config.seed, config.max_iter)
        pruned.fit(X_train_pruned, y_train_pruned)
        pruned_metrics = evaluate_multiclass(pruned, X_test, y_test)

        return MulticlassClassificationResult(
            dataset=self.data_provider.name,
            n_train=len(X_train),
            n_test=len(X_test),
            n_classes=len(np.unique(y)),
            noise=MulticlassNoiseSummary(
                fraction=float(config.noise_frac),
                n_flipped=len(flipped_set),
            ),
            cleanlab=MulticlassCleanlabSummary(
                cv_folds=int(config.cv_folds),
                n_issues_found=len(issue_indices),
                prune_frac=float(config.prune_frac),
                n_pruned=int(n_prune),
                precision_at_prune=float(precision_at_prune),
                recall_at_prune=float(recall_at_prune),
            ),
            metrics=MulticlassMetricsByVariant(
                baseline=baseline_metrics,
                pruned_retrain=pruned_metrics,
            ),
        )


def run_multiclass_classification(
    data_provider: MulticlassDataProvider,
    config: MulticlassClassificationConfig | None = None,
    **kwargs: Any,
) -> MulticlassClassificationResult:
    cfg = config or MulticlassClassificationConfig(**kwargs)
    return MulticlassClassificationTask(data_provider).run(cfg)
