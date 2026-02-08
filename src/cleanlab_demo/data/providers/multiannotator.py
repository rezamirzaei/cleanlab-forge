"""
Multi-annotator Data Providers.

These providers return a feature matrix `X` and a *label matrix*
`labels_multiannotator` of shape (N examples, M annotators), with missing labels as NaN.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from cleanlab_demo.settings import settings
from cleanlab_demo.tasks.multiannotator.provider import MultiannotatorDataProvider
from cleanlab_demo.utils.download import download_file

_MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
_ITEM_COLS = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
    "genre_unknown",
    "genre_action",
    "genre_adventure",
    "genre_animation",
    "genre_children",
    "genre_comedy",
    "genre_crime",
    "genre_documentary",
    "genre_drama",
    "genre_fantasy",
    "genre_film_noir",
    "genre_horror",
    "genre_musical",
    "genre_mystery",
    "genre_romance",
    "genre_scifi",
    "genre_thriller",
    "genre_war",
    "genre_western",
]


def _extract_movielens_year(title: pd.Series) -> pd.Series:
    year = title.astype(str).str.extract(r"\((\d{4})\)\s*$", expand=False)
    return pd.to_numeric(year, errors="coerce")


def _ensure_ml100k(root: Path) -> Path:
    """
    Ensure MovieLens 100K is downloaded/extracted and return the `ml-100k` dir.
    """
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "ml-100k.zip"
    ml_dir = root / "ml-100k"
    if (ml_dir / "u.data").exists() and (ml_dir / "u.item").exists():
        return ml_dir

    download_file(_MOVIELENS_100K_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    if not (ml_dir / "u.data").exists():
        raise RuntimeError(f"MovieLens extraction failed: missing `{ml_dir / 'u.data'}`")
    return ml_dir


class MovieLens100KProvider(MultiannotatorDataProvider):
    """
    MovieLens 100K as a *real* multi-annotator dataset.

    - Examples: movies
    - Annotators: users
    - Labels: ratings (1..5), missing = NaN
    - Features: release year + 19 genre indicator columns from `u.item`
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        *,
        max_movies: int = 800,
        max_annotators: int = 50,
        min_ratings_per_movie: int = 10,
        min_ratings_per_annotator: int = 100,
    ) -> None:
        self._data_dir = data_dir
        self._max_movies = max_movies
        self._max_annotators = max_annotators
        self._min_ratings_per_movie = min_ratings_per_movie
        self._min_ratings_per_annotator = min_ratings_per_annotator

    @property
    def name(self) -> str:
        return "MovieLens 100K (users as annotators)"

    def load(self, seed: int, **kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        data_dir = self._data_dir or (settings.data_dir / "movielens_100k")
        settings.ensure_dirs()
        ml_dir = _ensure_ml100k(data_dir)

        items = pd.read_csv(
            ml_dir / "u.item",
            sep="|",
            header=None,
            names=_ITEM_COLS,
            encoding="ISO-8859-1",
        )
        items["year"] = _extract_movielens_year(items["title"])
        feature_cols = ["year", *[c for c in items.columns if c.startswith("genre_")]]
        items_X = items.set_index("movie_id")[feature_cols].copy()

        ratings = pd.read_csv(
            ml_dir / "u.data",
            sep="\t",
            header=None,
            names=["user_id", "movie_id", "rating", "timestamp"],
        )
        ratings = ratings[["user_id", "movie_id", "rating"]].copy()

        # Pick a dense submatrix: top users, then popular movies among these users.
        user_counts = ratings["user_id"].value_counts()
        top_users = user_counts.head(self._max_annotators).index
        ratings = ratings[ratings["user_id"].isin(top_users)]

        movie_counts = ratings["movie_id"].value_counts()
        eligible_movies = movie_counts[movie_counts >= self._min_ratings_per_movie].index
        ratings = ratings[ratings["movie_id"].isin(eligible_movies)]

        movie_counts = ratings["movie_id"].value_counts()
        selected_movies = movie_counts.head(self._max_movies).index
        ratings = ratings[ratings["movie_id"].isin(selected_movies)]

        # Remove annotators with too few labels in the selected subset.
        user_counts = ratings["user_id"].value_counts()
        keep_users = user_counts[user_counts >= self._min_ratings_per_annotator].index
        ratings = ratings[ratings["user_id"].isin(keep_users)]

        labels = ratings.pivot(index="movie_id", columns="user_id", values="rating").sort_index()
        if labels.shape[1] < 2:
            raise ValueError(
                f"Need >=2 annotators; got {labels.shape[1]}. "
                "Try lowering `min_ratings_per_annotator` or increasing `max_annotators`."
            )

        # Align features to the selected movies, drop any missing metadata rows.
        movie_ids = labels.index
        available_ids = items_X.index.intersection(movie_ids)
        labels = labels.loc[available_ids]
        X = items_X.loc[available_ids]

        # Stable, sklearn-friendly columns
        labels.columns = [f"user_{int(c)}" for c in labels.columns]
        labels = labels.astype(float)
        X = X.reset_index(drop=True)
        labels = labels.reset_index(drop=True)
        return X, labels
