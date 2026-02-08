from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from cleanlab_demo.tasks.token.featurization import featurize_sentence
from cleanlab_demo.tasks.token.schemas import TokenClassificationMetrics


def evaluate_token_model(
    model: Pipeline, tokens_by_sent: list[list[str]], labels_by_sent: list[list[int]]
) -> TokenClassificationMetrics:
    y_true: list[int] = []
    y_pred: list[int] = []
    for tokens, labels in zip(tokens_by_sent, labels_by_sent, strict=True):
        feats = featurize_sentence(tokens)
        pred = model.predict(feats)
        y_true.extend(labels)
        y_pred.extend([int(v) for v in pred.tolist()])

    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    return TokenClassificationMetrics(
        token_accuracy=float(accuracy_score(y_true_arr, y_pred_arr)),
        macro_f1=float(f1_score(y_true_arr, y_pred_arr, average="macro")),
    )

