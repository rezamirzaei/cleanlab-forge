from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cleanlab_demo.tasks.multilabel.provider import MultilabelDataProvider
from cleanlab_demo.tasks.multilabel.schemas import (
    MultilabelClassificationConfig,
    MultilabelClassificationResult,
    MultilabelCleanlabSummary,
    MultilabelMetrics,
    MultilabelMetricsByVariant,
    MultilabelNoiseSummary,
)


def inject_multilabel_noise(
    y: np.ndarray, *, frac_examples: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if frac_examples <= 0:
        return y.copy(), np.array([], dtype=int)

    rng = np.random.default_rng(seed=seed)
    y_noisy = y.copy()
    n, n_labels = y_noisy.shape
    n_noisy = round(frac_examples * n)
    if n_noisy <= 0:
        return y_noisy, np.array([], dtype=int)

    noisy_indices = rng.choice(n, size=n_noisy, replace=False).astype(int)
    for idx in noisy_indices:
        j = int(rng.integers(0, n_labels))
        y_noisy[int(idx), j] = 1 - y_noisy[int(idx), j]
    return y_noisy, np.sort(noisy_indices)


def labels_to_list_format(y: np.ndarray) -> list[list[int]]:
    return [list(np.flatnonzero(row).astype(int)) for row in y]


def build_multilabel_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("model", OneVsRestClassifier(LogisticRegression(max_iter=500, solver="lbfgs"))),
        ]
    )


def evaluate_multilabel(model: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray) -> MultilabelMetrics:
    y_pred = model.predict(X_test)
    return MultilabelMetrics(
        micro_f1=float(f1_score(y_test, y_pred, average="micro")),
        macro_f1=float(f1_score(y_test, y_pred, average="macro")),
        subset_accuracy=float(accuracy_score(y_test, y_pred)),
        hamming_loss=float(hamming_loss(y_test, y_pred)),
    )


class MultilabelClassificationTask:
    def __init__(self, data_provider: MultilabelDataProvider) -> None:
        self.data_provider = data_provider

    def run(self, config: MultilabelClassificationConfig) -> MultilabelClassificationResult:
        from cleanlab.multilabel_classification.filter import find_label_issues

        X, y = self.data_provider.load(seed=config.seed)
        n_labels = int(y.shape[1])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.seed
        )

        y_train_noisy, noisy_example_indices = inject_multilabel_noise(
            y_train, frac_examples=config.noise_frac, seed=config.seed
        )
        noisy_set = set(map(int, noisy_example_indices.tolist()))

        baseline_model = build_multilabel_model()
        baseline_model.fit(X_train, y_train_noisy)
        baseline_metrics = evaluate_multilabel(baseline_model, X_test, y_test)

        cv = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
        pred_probs_cv = cross_val_predict(
            baseline_model, X_train, y_train_noisy, cv=cv, method="predict_proba", n_jobs=1
        )
        pred_probs_cv = np.asarray(pred_probs_cv, dtype=float)
        if pred_probs_cv.shape != (len(X_train), n_labels):
            raise ValueError(f"Unexpected pred_probs shape: {pred_probs_cv.shape}")

        issue_indices = find_label_issues(
            labels=labels_to_list_format(y_train_noisy),
            pred_probs=pred_probs_cv,
            return_indices_ranked_by="self_confidence",
            n_jobs=1,
        )
        issue_indices = np.asarray(issue_indices, dtype=int)

        n_prune = min(len(issue_indices), round(config.prune_frac * len(X_train)))
        prune_set = set(map(int, issue_indices[:n_prune].tolist()))

        tp = len(prune_set & noisy_set)
        recall_at_prune = float(tp / len(noisy_set)) if noisy_set else 0.0
        precision_at_prune = float(tp / len(prune_set)) if prune_set else 0.0

        keep_mask = np.ones(len(X_train), dtype=bool)
        if prune_set:
            keep_mask[list(prune_set)] = False
        X_train_pruned = (
            X_train.iloc[keep_mask] if isinstance(X_train, pd.DataFrame) else X_train[keep_mask]
        )
        y_train_pruned = y_train_noisy[keep_mask]

        pruned_model = build_multilabel_model()
        pruned_model.fit(X_train_pruned, y_train_pruned)
        pruned_metrics = evaluate_multilabel(pruned_model, X_test, y_test)

        return MultilabelClassificationResult(
            dataset=self.data_provider.name,
            n_train=len(X_train),
            n_test=len(X_test),
            n_labels=n_labels,
            noise=MultilabelNoiseSummary(
                fraction_examples=float(config.noise_frac),
                n_noisy_examples=len(noisy_set),
            ),
            cleanlab=MultilabelCleanlabSummary(
                cv_folds=int(config.cv_folds),
                n_issues_found=len(issue_indices),
                n_pruned=int(n_prune),
                precision_at_prune=float(precision_at_prune),
                recall_at_prune=float(recall_at_prune),
            ),
            metrics=MultilabelMetricsByVariant(
                baseline=baseline_metrics,
                pruned_retrain=pruned_metrics,
            ),
        )


def run_multilabel_classification(
    data_provider: MultilabelDataProvider,
    config: MultilabelClassificationConfig | None = None,
    **kwargs: Any,
) -> MultilabelClassificationResult:
    cfg = config or MultilabelClassificationConfig(**kwargs)
    return MultilabelClassificationTask(data_provider).run(cfg)
