from __future__ import annotations

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_token_model(seed: int, *, max_iter: int = 800) -> Pipeline:
    return Pipeline(
        steps=[
            ("vec", DictVectorizer(sparse=True)),
            (
                "clf",
                LogisticRegression(
                    max_iter=max_iter,
                    solver="saga",
                    n_jobs=1,
                    random_state=seed,
                ),
            ),
        ]
    )

