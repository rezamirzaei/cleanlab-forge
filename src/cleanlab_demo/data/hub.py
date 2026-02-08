from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cleanlab_demo.config import DATASET_DEFAULTS, DatasetDefaults, DatasetName, TaskType
from cleanlab_demo.data.schemas import LoadedDataset
from cleanlab_demo.settings import logger
from cleanlab_demo.utils.download import download_file

# ---------------------------------------------------------------------------
# Dataset specification: URL, loader, and metadata for each dataset.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    """Specification for how to obtain and load a single dataset."""

    url: str
    filename: str
    task: TaskType
    target_col: str
    loader: Callable[[Path], pd.DataFrame]


def _load_adult_income(path: Path) -> pd.DataFrame:
    """Load the UCI Adult Income dataset from a CSV file."""
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income",
    ]
    df = pd.read_csv(
        path,
        names=column_names,
        skipinitialspace=True,
        na_values=["?"],
    )
    # Strip whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


def _load_bike_sharing(path: Path) -> pd.DataFrame:
    """Load the UCI Bike Sharing dataset from a CSV file."""
    df = pd.read_csv(path)
    # Drop non-predictive columns if present
    drop_cols = [c for c in ("instant", "dteday", "casual", "registered") if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def _load_california_housing(path: Path) -> pd.DataFrame:
    """Load the California Housing dataset from sklearn."""
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing(as_frame=True)
    return data.frame  # type: ignore[union-attr]


_SPECS: dict[DatasetName, DatasetSpec] = {
    DatasetName.adult_income: DatasetSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        filename="adult.data",
        task=TaskType.classification,
        target_col="income",
        loader=_load_adult_income,
    ),
    DatasetName.bike_sharing: DatasetSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip",
        filename="Bike-Sharing-Dataset.zip",
        task=TaskType.regression,
        target_col="cnt",
        loader=_load_bike_sharing,
    ),
    DatasetName.california_housing: DatasetSpec(
        url="",  # Loaded from sklearn, no download needed
        filename="",
        task=TaskType.regression,
        target_col="MedHouseVal",
        loader=_load_california_housing,
    ),
}


class DatasetHub:
    """Central registry for downloading and loading datasets."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def spec(self, name: DatasetName) -> DatasetDefaults:
        """Return the default configuration for a dataset."""
        return DATASET_DEFAULTS[name]

    def load(self, name: DatasetName) -> LoadedDataset:
        """Download (if needed) and load a dataset into memory."""
        ds = _SPECS[name]
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if name == DatasetName.california_housing:
            # sklearn handles its own caching
            df = ds.loader(self.data_dir)
        elif name == DatasetName.bike_sharing:
            df = self._load_bike_sharing_zip(ds)
        else:
            dest = self.data_dir / ds.filename
            download_file(ds.url, dest)
            df = ds.loader(dest)

        logger.info(f"Loaded {name.value}: {len(df):,} rows, {len(df.columns)} columns")
        return LoadedDataset(name=name, task=ds.task, target_col=ds.target_col, df=df)

    def _load_bike_sharing_zip(self, ds: DatasetSpec) -> pd.DataFrame:
        """Handle the bike-sharing ZIP file (contains a nested CSV)."""
        import zipfile

        zip_path = self.data_dir / ds.filename
        csv_path = self.data_dir / "hour.csv"

        if not csv_path.exists():
            download_file(ds.url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Extract hour.csv (the hourly dataset)
                for member in zf.namelist():
                    if member.endswith("hour.csv"):
                        zf.extract(member, self.data_dir)
                        extracted = self.data_dir / member
                        if extracted != csv_path:
                            extracted.rename(csv_path)
                        break

        return ds.loader(csv_path)
