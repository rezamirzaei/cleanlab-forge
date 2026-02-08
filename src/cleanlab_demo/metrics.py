from __future__ import annotations

import logging
import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    roc_auc_score,
)

from cleanlab_demo.config import Metrics

_logger = logging.getLogger(__name__)


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None
) -> Metrics:
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="weighted"))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))

    details: dict[str, float] = {"accuracy": acc, "f1_weighted": f1, "precision_weighted": prec}
    primary = f1

    if y_proba is not None:
        try:
            if y_proba.shape[1] == 2:
                auc = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
            details["roc_auc"] = auc
            primary = auc
        except Exception:
            _logger.debug("Could not compute ROC AUC (may require >1 class in y_true)", exc_info=True)

    return Metrics(primary=primary, details=details)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    return Metrics(primary=r2, details={"r2": r2, "mae": mae, "rmse": rmse})

