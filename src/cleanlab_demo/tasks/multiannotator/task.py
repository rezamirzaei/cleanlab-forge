from __future__ import annotations

import contextlib
import io
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from cleanlab_demo.features import build_preprocessor
from cleanlab_demo.tasks.multiannotator.provider import MultiannotatorDataProvider
from cleanlab_demo.tasks.multiannotator.schemas import (
    MultiannotatorCleanlabSummary,
    MultiannotatorConfig,
    MultiannotatorNotes,
    MultiannotatorResult,
    compute_multiannotator_coverage,
)


def _encode_labels_matrix(labels_multiannotator: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    arr = labels_multiannotator.to_numpy()
    flat = arr.ravel()
    mask = np.isfinite(flat)
    if not mask.any():
        raise ValueError("labels_multiannotator has no labels (all values are missing).")

    le = LabelEncoder()
    le.fit(flat[mask])
    n_classes = len(le.classes_)
    mapped = flat.copy()
    mapped[mask] = le.transform(flat[mask])
    mapped_arr = mapped.reshape(arr.shape).astype(float)
    return pd.DataFrame(mapped_arr, columns=list(labels_multiannotator.columns)), n_classes


class MultiannotatorTask:
    def __init__(self, data_provider: MultiannotatorDataProvider) -> None:
        self.data_provider = data_provider

    def run(self, config: MultiannotatorConfig) -> MultiannotatorResult:
        from cleanlab.multiannotator import (
            get_active_learning_scores,
            get_label_quality_multiannotator,
            get_majority_vote_label,
        )

        X, labels_ma_raw = self.data_provider.load(seed=config.seed)
        labels_ma, n_classes = _encode_labels_matrix(labels_ma_raw)

        labels_arr = labels_ma.to_numpy()
        coverage, mean_labels_per_example = compute_multiannotator_coverage(labels_arr)

        consensus = get_majority_vote_label(labels_ma, verbose=False).astype(int)
        classes_present = np.unique(consensus).astype(int)
        if len(classes_present) < 2:
            raise ValueError(
                "Consensus labels contain <2 classes. Try adjusting the dataset filtering "
                "(e.g., more movies/annotators) so consensus labels are more diverse."
            )
        missing_consensus_classes = sorted(set(range(n_classes)) - set(classes_present.tolist()))

        pre = build_preprocessor(X, config.features)
        model = Pipeline(
            steps=[
                ("preprocess", pre),
                (
                    "model",
                    LogisticRegression(
                        max_iter=config.max_iter, solver="lbfgs", random_state=config.seed
                    ),
                ),
            ]
        )
        class_counts = np.bincount(consensus, minlength=int(classes_present.max()) + 1)
        min_count = int(min(class_counts[c] for c in classes_present))
        effective_folds = min(int(config.cv_folds), min_count)
        if effective_folds < 2:
            raise ValueError(
                "Not enough examples per consensus class for cross-validation. "
                f"min_class_count={min_count}. Try reducing `cv_folds` or adjusting the dataset filtering."
            )

        cv = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=config.seed)
        pred_probs = cross_val_predict(model, X, consensus, cv=cv, method="predict_proba", n_jobs=1)
        pred_probs = np.asarray(pred_probs, dtype=float)
        if pred_probs.shape[1] < n_classes:
            probs_full = np.zeros((len(X), n_classes), dtype=float)
            probs_full[:, classes_present] = pred_probs
            pred_probs = probs_full

        results = get_label_quality_multiannotator(
            labels_ma, pred_probs, consensus_method="majority_vote", verbose=False
        )
        label_quality = results.get("label_quality")
        if label_quality is None:
            raise RuntimeError("cleanlab.multiannotator did not return label_quality")

        # Cleanlab prints a CAUTION message in `get_active_learning_scores` when some classes are
        # omitted in its internal consensus; capture stdout/stderr to keep the UI/notebooks clean.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            al_scores, _ = get_active_learning_scores(
                labels_multiannotator=labels_arr, pred_probs=pred_probs
            )
        al_scores = np.asarray(al_scores, dtype=float)
        top_al = np.argsort(al_scores)[: config.top_k].astype(int).tolist()

        worst: list[int] = []
        if isinstance(label_quality, pd.DataFrame) and "label_quality" in label_quality.columns:
            q = label_quality["label_quality"].to_numpy(dtype=float)
            worst = np.argsort(q)[: config.top_k].astype(int).tolist()

        return MultiannotatorResult(
            dataset=self.data_provider.name,
            n_examples=len(X),
            n_annotators=int(labels_ma.shape[1]),
            notes=MultiannotatorNotes(
                description="Real-world multi-annotator labels (missing labels allowed).",
                n_classes=n_classes,
                coverage=float(coverage),
                mean_labels_per_example=float(mean_labels_per_example),
                missing_consensus_classes=missing_consensus_classes,
            ),
            cleanlab=MultiannotatorCleanlabSummary(
                consensus_method="majority_vote",
                top_worst_quality_examples=worst,
                top_active_learning_examples=top_al,
            ),
        )


def run_multiannotator(
    data_provider: MultiannotatorDataProvider,
    config: MultiannotatorConfig | None = None,
    **kwargs: Any,
) -> MultiannotatorResult:
    cfg = config or MultiannotatorConfig(**kwargs)
    return MultiannotatorTask(data_provider).run(cfg)
