from __future__ import annotations

from typing import Any

from cleanlab_demo.tasks.token.cv import cv_pred_probs
from cleanlab_demo.tasks.token.featurization import flatten_for_fit
from cleanlab_demo.tasks.token.metrics import evaluate_token_model
from cleanlab_demo.tasks.token.model import build_token_model
from cleanlab_demo.tasks.token.noise import inject_token_noise
from cleanlab_demo.tasks.token.provider import TokenClassificationDataProvider
from cleanlab_demo.tasks.token.schemas import (
    TokenClassificationCleanlabSummary,
    TokenClassificationConfig,
    TokenClassificationMetricsByVariant,
    TokenClassificationNoiseSummary,
    TokenClassificationResult,
)


class TokenClassificationTask:
    def __init__(self, data_provider: TokenClassificationDataProvider) -> None:
        self.data_provider = data_provider

    def run(self, config: TokenClassificationConfig) -> TokenClassificationResult:
        from cleanlab.token_classification.filter import find_label_issues

        train_tokens, train_tags, dev_tokens, dev_tags = self.data_provider.load(
            seed=config.seed,
            max_train=config.max_train_sentences,
            max_dev=config.max_dev_sentences,
        )

        tag_set = sorted({t for sent in train_tags + dev_tags for t in sent})
        tag_to_id = {t: i for i, t in enumerate(tag_set)}
        n_classes = len(tag_set)

        train_y = [[tag_to_id[t] for t in sent] for sent in train_tags]
        dev_y = [[tag_to_id[t] for t in sent] for sent in dev_tags]

        train_y_noisy, corrupted = inject_token_noise(
            train_y, frac_tokens=config.noise_frac, seed=config.seed, n_classes=n_classes
        )

        baseline = build_token_model(config.seed, max_iter=config.max_iter)
        X_train_flat, y_train_flat = flatten_for_fit(train_tokens, train_y_noisy)
        baseline.fit(X_train_flat, y_train_flat)
        baseline_metrics = evaluate_token_model(baseline, dev_tokens, dev_y)

        pred_probs_cv = cv_pred_probs(
            train_tokens,
            train_y_noisy,
            cv_folds=config.cv_folds,
            seed=config.seed,
            max_iter=config.max_iter,
        )
        issues = find_label_issues(labels=train_y_noisy, pred_probs=pred_probs_cv, n_jobs=1)

        n_prune_target = round(config.prune_frac * len(y_train_flat))
        n_prune = min(len(issues), max(0, int(n_prune_target)))
        prune_set: set[tuple[int, int]] = (
            set(map(tuple, issues[:n_prune])) if n_prune > 0 else set()
        )

        tp = len(prune_set & corrupted)
        recall_at_prune = float(tp / len(corrupted)) if corrupted else 0.0
        precision_at_prune = float(tp / len(prune_set)) if prune_set else 0.0

        pruned = build_token_model(config.seed, max_iter=config.max_iter)
        X_train_pruned, y_train_pruned = flatten_for_fit(
            train_tokens, train_y_noisy, drop=prune_set
        )
        pruned.fit(X_train_pruned, y_train_pruned)
        pruned_metrics = evaluate_token_model(pruned, dev_tokens, dev_y)

        return TokenClassificationResult(
            dataset=self.data_provider.name,
            n_train_sentences=len(train_tokens),
            n_dev_sentences=len(dev_tokens),
            n_classes=n_classes,
            noise=TokenClassificationNoiseSummary(
                fraction_tokens=float(config.noise_frac),
                n_corrupted_tokens=len(corrupted),
            ),
            cleanlab=TokenClassificationCleanlabSummary(
                cv_folds=int(config.cv_folds),
                n_token_issues_found=len(issues),
                n_pruned_tokens=len(prune_set),
                precision_at_prune=float(precision_at_prune),
                recall_at_prune=float(recall_at_prune),
            ),
            metrics=TokenClassificationMetricsByVariant(
                baseline=baseline_metrics,
                pruned_retrain=pruned_metrics,
            ),
        )


def run_token_classification(
    data_provider: TokenClassificationDataProvider,
    config: TokenClassificationConfig | None = None,
    **kwargs: Any,
) -> TokenClassificationResult:
    cfg = config or TokenClassificationConfig(**kwargs)
    return TokenClassificationTask(data_provider).run(cfg)
