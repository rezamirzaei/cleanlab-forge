from __future__ import annotations

import logging
from collections.abc import Iterable

from pydantic import BaseModel, Field

from cleanlab_demo.config import DatasetName, ModelConfig, ModelName, RunConfig, TaskType
from cleanlab_demo.experiments.runner import ExperimentRunner

_logger = logging.getLogger(__name__)


class SweepResultRow(BaseModel):
    dataset: DatasetName
    task: TaskType
    model: ModelName
    primary_metric: float
    metrics: dict[str, float] = Field(default_factory=dict)
    n_label_issues: int = 0


def run_sweep(
    *,
    dataset: DatasetName,
    models: Iterable[ModelName],
    base_config: RunConfig | None = None,
    runner: ExperimentRunner | None = None,
) -> list[SweepResultRow]:
    cfg = base_config or RunConfig(dataset=dataset)
    exp_runner = runner or ExperimentRunner()

    rows: list[SweepResultRow] = []
    for model_name in models:
        data = cfg.model_dump(mode="python")
        data["dataset"] = dataset
        data["task"] = None
        data["target_col"] = None
        data["model"] = ModelConfig(name=model_name).model_dump(mode="python")
        config = RunConfig.model_validate(data)
        try:
            result = exp_runner.run(config)
        except Exception:
            _logger.warning("Model %s failed during sweep, skipping", model_name.value, exc_info=True)
            continue
        rows.append(
            SweepResultRow(
                dataset=result.dataset,
                task=result.task,
                model=result.model,
                primary_metric=result.metrics.primary,
                metrics={k: float(v) for k, v in (result.metrics.details or {}).items()},
                n_label_issues=int(result.cleanlab_summary.get("n_label_issues", 0) or 0),
            )
        )
    return rows

