from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cleanlab_demo.config import FeatureEngineeringConfig


@dataclass(frozen=True)
class FeatureColumns:
    """Container for numeric and categorical column names."""

    numeric: list[str]
    categorical: list[str]


class StringDtypeConverter(BaseEstimator, TransformerMixin):
    """
    Convert pandas StringDtype columns to object dtype for sklearn compatibility.

    sklearn's SimpleImputer doesn't handle pandas StringDtype with pd.NA well,
    so we convert to regular object dtype with np.nan.
    """

    def fit(self, X: pd.DataFrame | np.ndarray, y: object = None) -> StringDtypeConverter:
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            # Convert StringDtype to object and pd.NA to np.nan
            result = X.copy()
            for col in result.columns:
                if hasattr(result[col], "dtype") and str(result[col].dtype) == "string":
                    result[col] = result[col].astype(object).replace({pd.NA: np.nan})
            return result.values
        return np.asarray(X)


def infer_feature_columns(df: pd.DataFrame) -> FeatureColumns:
    """Infer numeric and categorical columns from a DataFrame."""
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return FeatureColumns(numeric=numeric_cols, categorical=categorical_cols)


def build_preprocessor(X: pd.DataFrame, config: FeatureEngineeringConfig) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for feature preprocessing."""
    cols = infer_feature_columns(X)

    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy=config.impute_numeric)),
    ]
    if config.scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("dtype_converter", StringDtypeConverter()),
            ("imputer", SimpleImputer(strategy=config.impute_categorical)),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    max_categories=config.one_hot_max_categories,
                    sparse_output=False,  # Dense output for compatibility with all models
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, cols.numeric),
            ("cat", categorical_transformer, cols.categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

