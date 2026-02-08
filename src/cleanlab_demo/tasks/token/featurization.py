from __future__ import annotations

from typing import Any

import numpy as np


def token_features(tokens: list[str], i: int) -> dict[str, Any]:
    w = tokens[i]
    w_lower = w.lower()
    prev_w = tokens[i - 1].lower() if i > 0 else "<BOS>"
    next_w = tokens[i + 1].lower() if i + 1 < len(tokens) else "<EOS>"
    return {
        "bias": 1.0,
        "w": w_lower,
        "w.isupper": w.isupper(),
        "w.istitle": w.istitle(),
        "w.isdigit": w.isdigit(),
        "p1": w_lower[:1],
        "p2": w_lower[:2],
        "p3": w_lower[:3],
        "s1": w_lower[-1:],
        "s2": w_lower[-2:],
        "s3": w_lower[-3:],
        "prev": prev_w,
        "next": next_w,
    }


def featurize_sentence(tokens: list[str]) -> list[dict[str, Any]]:
    return [token_features(tokens, i) for i in range(len(tokens))]


def flatten_for_fit(
    tokens_by_sent: list[list[str]],
    labels_by_sent: list[list[int]],
    *,
    drop: set[tuple[int, int]] | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    X_out: list[dict[str, Any]] = []
    y_out: list[int] = []
    drop = drop or set()
    for sent_i, tokens in enumerate(tokens_by_sent):
        feats = featurize_sentence(tokens)
        labels = labels_by_sent[sent_i]
        for tok_i, (x, y) in enumerate(zip(feats, labels, strict=True)):
            if (sent_i, tok_i) in drop:
                continue
            X_out.append(x)
            y_out.append(int(y))
    return X_out, np.asarray(y_out, dtype=int)

