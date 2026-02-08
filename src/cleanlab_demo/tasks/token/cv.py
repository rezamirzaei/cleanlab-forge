from __future__ import annotations

from typing import cast

import numpy as np
from sklearn.model_selection import KFold

from cleanlab_demo.tasks.token.featurization import featurize_sentence, flatten_for_fit
from cleanlab_demo.tasks.token.model import build_token_model


def cv_pred_probs(
    tokens_by_sent: list[list[str]],
    labels_by_sent: list[list[int]],
    *,
    cv_folds: int,
    seed: int,
    max_iter: int = 800,
) -> list[np.ndarray]:
    n = len(tokens_by_sent)
    pred_probs_by_sent: list[np.ndarray | None] = [None] * n
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    sent_indices = np.arange(n)

    for train_idx, val_idx in kf.split(sent_indices):
        model = build_token_model(seed, max_iter=max_iter)
        X_train, y_train = flatten_for_fit(
            [tokens_by_sent[int(i)] for i in train_idx],
            [labels_by_sent[int(i)] for i in train_idx],
        )
        model.fit(X_train, y_train)

        for i in val_idx:
            feats = featurize_sentence(tokens_by_sent[int(i)])
            pred_probs_by_sent[int(i)] = model.predict_proba(feats)

    missing = [i for i, v in enumerate(pred_probs_by_sent) if v is None]
    if missing:
        raise RuntimeError(
            f"Missing CV predictions for {len(missing)} sentences, e.g. {missing[:5]}"
        )
    return [cast(np.ndarray, v) for v in pred_probs_by_sent]

