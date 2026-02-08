from __future__ import annotations

import numpy as np


def inject_token_noise(
    labels_by_sent: list[list[int]], *, frac_tokens: float, seed: int, n_classes: int
) -> tuple[list[list[int]], set[tuple[int, int]]]:
    rng = np.random.default_rng(seed=seed)
    out = [sent.copy() for sent in labels_by_sent]
    corrupted: set[tuple[int, int]] = set()
    if frac_tokens <= 0:
        return out, corrupted

    for sent_i, sent in enumerate(out):
        for tok_i, y in enumerate(sent):
            if rng.random() >= frac_tokens:
                continue
            other = int(rng.integers(0, n_classes - 1))
            new_y = other if other < y else other + 1
            out[sent_i][tok_i] = new_y
            corrupted.add((sent_i, tok_i))
    return out, corrupted

