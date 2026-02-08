from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from cleanlab_demo.tasks.outlier.provider import OutlierDetectionDataProvider
from cleanlab_demo.tasks.outlier.schemas import (
    OutlierDetectionCleanlabSummary,
    OutlierDetectionConfig,
    OutlierDetectionResult,
    OutlierRow,
    SyntheticOutliersSummary,
)


def inject_synthetic_outliers(
    X: pd.DataFrame, *, outlier_frac: float, outlier_scale: float, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed=seed)
    n_out = round(outlier_frac * len(X))
    outlier_idx = rng.choice(len(X), size=n_out, replace=False).astype(int)

    X_noisy = X.copy()
    if n_out > 0:
        numeric_cols = X_noisy.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            X_noisy.loc[outlier_idx, col] = X_noisy.loc[outlier_idx, col] * outlier_scale

    return X_noisy, outlier_idx


class OutlierDetectionTask:
    def __init__(self, data_provider: OutlierDetectionDataProvider) -> None:
        self.data_provider = data_provider

    def run(self, config: OutlierDetectionConfig) -> OutlierDetectionResult:
        from cleanlab.datalab.datalab import Datalab

        df = self.data_provider.load(seed=config.seed)
        label_col = self.data_provider.label_col

        X = df.drop(columns=[label_col])
        y = df[label_col].to_numpy(dtype=float)

        X_noisy, outlier_idx = inject_synthetic_outliers(
            X,
            outlier_frac=config.outlier_frac,
            outlier_scale=config.outlier_scale,
            seed=config.seed,
        )
        outlier_set = set(map(int, outlier_idx.tolist()))

        numeric_cols = X_noisy.select_dtypes(include=[np.number]).columns
        features = StandardScaler().fit_transform(X_noisy[numeric_cols].to_numpy(dtype=float))
        df_with_label = pd.concat([X_noisy, pd.Series(y, name=label_col)], axis=1)

        issue_types = {"outlier": {}, "near_duplicate": {}, "non_iid": {}}
        datalab = Datalab(
            data=df_with_label,
            task=self.data_provider.task_type,
            label_name=label_col,
            verbosity=0,
        )
        datalab.find_issues(features=features, issue_types=issue_types)

        issues = datalab.get_issues().reset_index(drop=True)
        summary = datalab.get_issue_summary().reset_index(drop=True)

        if "is_outlier_issue" in issues.columns:
            is_outlier = issues["is_outlier_issue"].fillna(False).astype(bool)
        else:
            is_outlier = pd.Series(False, index=issues.index)

        flagged = issues.index[is_outlier].to_numpy(dtype=int)
        flagged_set = set(map(int, flagged.tolist()))

        tp = len(flagged_set & outlier_set)
        recall = float(tp / len(outlier_set)) if outlier_set else 0.0
        precision = float(tp / len(flagged_set)) if flagged_set else 0.0

        top_outliers: list[OutlierRow] = []
        if "outlier_score" in issues.columns:
            tmp = (
                issues.reset_index(names="row")
                .sort_values("outlier_score", ascending=True)
                .head(config.top_k_outliers)
            )
            for _, r in tmp.iterrows():
                top_outliers.append(
                    OutlierRow(
                        row=int(r["row"]),
                        outlier_score=float(r["outlier_score"]),
                        is_outlier_issue=bool(r.get("is_outlier_issue", False)),
                    )
                )

        return OutlierDetectionResult(
            dataset=self.data_provider.name,
            n_rows=len(df),
            synthetic_outliers=SyntheticOutliersSummary(
                fraction=float(config.outlier_frac),
                n_injected=len(outlier_set),
            ),
            cleanlab=OutlierDetectionCleanlabSummary(
                issue_types=list(issue_types.keys()),
                precision_vs_injected=float(precision),
                recall_vs_injected=float(recall),
                issue_summary=summary.to_dict(orient="records"),
                top_outliers=top_outliers,
            ),
        )


def run_outlier_detection(
    data_provider: OutlierDetectionDataProvider,
    config: OutlierDetectionConfig | None = None,
    **kwargs: Any,
) -> OutlierDetectionResult:
    cfg = config or OutlierDetectionConfig(**kwargs)
    return OutlierDetectionTask(data_provider).run(cfg)

