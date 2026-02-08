"""Tests for feature preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cleanlab_demo.config import FeatureEngineeringConfig
from cleanlab_demo.features.preprocess import (
    build_preprocessor,
    infer_feature_columns,
)


def test_infer_feature_columns_numeric():
    """Test that numeric columns are correctly identified."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    cols = infer_feature_columns(df)
    assert cols.numeric == ["a", "b"]
    assert cols.categorical == []


def test_infer_feature_columns_categorical():
    """Test that categorical columns are correctly identified."""
    df = pd.DataFrame({"a": ["x", "y", "z"], "b": pd.Categorical(["a", "b", "c"])})
    cols = infer_feature_columns(df)
    assert cols.numeric == []
    assert cols.categorical == ["a", "b"]


def test_infer_feature_columns_mixed():
    """Test that mixed columns are correctly identified."""
    df = pd.DataFrame({"num": [1, 2, 3], "cat": ["x", "y", "z"]})
    cols = infer_feature_columns(df)
    assert cols.numeric == ["num"]
    assert cols.categorical == ["cat"]


def test_build_preprocessor_numeric_only():
    """Test preprocessor with only numeric features."""
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [4.0, 5.0, 6.0]})
    config = FeatureEngineeringConfig()
    preprocessor = build_preprocessor(df, config)

    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == 3
    # Check no NaN values after imputation
    assert not np.isnan(transformed).any()


def test_build_preprocessor_categorical():
    """Test preprocessor with categorical features."""
    df = pd.DataFrame({"cat": ["a", "b", "a"]})
    config = FeatureEngineeringConfig()
    preprocessor = build_preprocessor(df, config)

    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == 3
    # One-hot encoding should produce multiple columns for categories
    assert transformed.shape[1] == 2  # 2 unique categories


def test_build_preprocessor_handles_missing_categorical():
    """Test that preprocessor handles missing categorical values."""
    df = pd.DataFrame({"cat": ["a", "b", None]})
    config = FeatureEngineeringConfig()
    preprocessor = build_preprocessor(df, config)

    # Should not raise
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == 3

