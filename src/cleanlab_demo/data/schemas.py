from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cleanlab_demo.config import DatasetName, TaskType


@dataclass(frozen=True)
class LoadedDataset:
    """A dataset that has been loaded into memory."""

    name: DatasetName
    task: TaskType
    target_col: str
    df: pd.DataFrame

    def features_df(self) -> pd.DataFrame:
        """Return the feature columns (everything except the target)."""
        return self.df.drop(columns=[self.target_col])

    def target_series(self) -> pd.Series:
        """Return the target column as a Series."""
        return self.df[self.target_col]
