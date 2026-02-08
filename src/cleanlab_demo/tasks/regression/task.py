from __future__ import annotations

import inspect
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from cleanlab_demo.tasks.regression.provider import RegressionDataProvider
from cleanlab_demo.tasks.regression.schemas import (
    RegressionCleanlabSummary,
    RegressionCleanLearningConfig,
    RegressionCleanLearningResult,
    RegressionMetrics,
    RegressionMetricsByVariant,
    RegressionNoiseSummary,
)


def inject_regression_label_noise(
    y: np.ndarray, *, frac: float, seed: int, scale: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    if frac <= 0:
        return y.copy(), np.array([], dtype=int)
    rng = np.random.default_rng(seed=seed)
    y_noisy = y.copy()
    n_corrupt = round(frac * len(y_noisy))
    if n_corrupt <= 0:
        return y_noisy, np.array([], dtype=int)
    idx = rng.choice(len(y_noisy), size=n_corrupt, replace=False).astype(int)
    sigma = float(np.std(y_noisy) + 1e-12)
    y_noisy[idx] = y_noisy[idx] + rng.normal(loc=0.0, scale=scale * sigma, size=n_corrupt)
    return y_noisy, np.sort(idx)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    mse = float(mean_squared_error(y_true, y_pred))
    return RegressionMetrics(
        r2=float(r2_score(y_true, y_pred)),
        rmse=float(np.sqrt(mse)),
        mae=float(mean_absolute_error(y_true, y_pred)),
    )


def build_regression_model(*, seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(random_state=seed)


def build_cleanlearning(*, seed: int, cv_folds: int) -> Any:
    from cleanlab.regression.learn import CleanLearning

    sig = inspect.signature(CleanLearning)
    kwargs: dict[str, object] = {}

    if "model" in sig.parameters:
        kwargs["model"] = build_regression_model(seed=seed)
    elif "estimator" in sig.parameters:
        kwargs["estimator"] = build_regression_model(seed=seed)
    else:
        kwargs["model"] = build_regression_model(seed=seed)

    if "cv_n_folds" in sig.parameters:
        kwargs["cv_n_folds"] = cv_folds
    elif "cv_folds" in sig.parameters:
        kwargs["cv_folds"] = cv_folds

    if "seed" in sig.parameters:
        kwargs["seed"] = seed
    if "verbose" in sig.parameters:
        kwargs["verbose"] = False

    return CleanLearning(**kwargs)


class RegressionCleanLearningTask:
    def __init__(self, data_provider: RegressionDataProvider) -> None:
        self.data_provider = data_provider

    def run(self, config: RegressionCleanLearningConfig) -> RegressionCleanLearningResult:
        X, y = self.data_provider.load(seed=config.seed)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.seed
        )

        y_train_noisy, corrupted = inject_regression_label_noise(
            y_train, frac=config.noise_frac, seed=config.seed, scale=config.noise_scale
        )
        corrupted_set = set(map(int, corrupted.tolist()))

        base_model = build_regression_model(seed=config.seed)
        base_model.fit(X_train, y_train_noisy)
        baseline_metrics = evaluate_regression(y_test, base_model.predict(X_test))

        cleaner = build_cleanlearning(seed=config.seed, cv_folds=config.cv_folds)
        issues_df = cleaner.find_label_issues(X_train, y_train_noisy)
        if "label_quality" not in issues_df.columns:
            raise RuntimeError(
                f"Unexpected regression label issues columns: {issues_df.columns.tolist()}"
            )
        ranked = issues_df.sort_values("label_quality", ascending=True)

        n_prune = min(len(ranked), round(config.prune_frac * len(X_train)))
        prune_idx = set(map(int, ranked.head(n_prune).index.astype(int).tolist()))

        tp = len(prune_idx & corrupted_set)
        recall_at_prune = float(tp / len(corrupted_set)) if corrupted_set else 0.0
        precision_at_prune = float(tp / len(prune_idx)) if prune_idx else 0.0

        keep = np.ones(len(X_train), dtype=bool)
        if prune_idx:
            keep[list(prune_idx)] = False
        X_train_pruned = X_train[keep]
        y_train_pruned = y_train_noisy[keep]

        model2 = build_regression_model(seed=config.seed)
        model2.fit(X_train_pruned, y_train_pruned)
        pruned_metrics = evaluate_regression(y_test, model2.predict(X_test))

        return RegressionCleanLearningResult(
            dataset=self.data_provider.name,
            n_train=len(X_train),
            n_test=len(X_test),
            noise=RegressionNoiseSummary(
                fraction=float(config.noise_frac),
                n_corrupted=len(corrupted_set),
            ),
            cleanlab=RegressionCleanlabSummary(
                cv_folds=int(config.cv_folds),
                n_issues_scored=len(issues_df),
                n_pruned=int(n_prune),
                precision_at_prune=float(precision_at_prune),
                recall_at_prune=float(recall_at_prune),
            ),
            metrics=RegressionMetricsByVariant(
                baseline=baseline_metrics,
                pruned_retrain=pruned_metrics,
            ),
        )


def run_regression_cleanlearning(
    data_provider: RegressionDataProvider,
    config: RegressionCleanLearningConfig | None = None,
    **kwargs: Any,
) -> RegressionCleanLearningResult:
    cfg = config or RegressionCleanLearningConfig(**kwargs)
    return RegressionCleanLearningTask(data_provider).run(cfg)

