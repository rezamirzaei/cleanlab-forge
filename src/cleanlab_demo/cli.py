from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.json import JSON
from rich.table import Table

from cleanlab_demo import __version__
from cleanlab_demo.config import (
    DATASET_DEFAULTS,
    DatasetName,
    DemoConfig,
    ModelConfig,
    ModelName,
    RunConfig,
    SplitConfig,
    TaskType,
)
from cleanlab_demo.config import CleanlabConfig as CleanlabCfg
from cleanlab_demo.data import DatasetHub
from cleanlab_demo.experiments import run_sweep
from cleanlab_demo.experiments.runner import run_experiment
from cleanlab_demo.settings import logger, settings

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Cleanlab Demo CLI - Run ML experiments with automatic label issue detection.",
)
console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"cleanlab-demo version {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool = typer.Option(None, "--version", "-V", callback=version_callback, is_eager=True, help="Show version"),
) -> None:
    """Cleanlab Demo CLI."""
    pass


@app.command()
def run(
    config_path: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to a JSON config file matching RunConfig.",
    ),
    dataset: DatasetName | None = typer.Option(None, "--dataset", "-d", help="Dataset to use."),
    model: ModelName | None = typer.Option(None, "--model", "-m", help="Model to train."),
    test_size: float | None = typer.Option(None, min=0.05, max=0.5, help="Test split size."),
    max_rows: int | None = typer.Option(None, min=100, help="Max rows to sample from dataset."),
    label_noise: float | None = typer.Option(None, min=0.0, max=0.5, help="Demo label noise fraction."),
    cleanlab_enabled: bool | None = typer.Option(None, "--cleanlab/--no-cleanlab", help="Enable/disable Cleanlab analysis."),
    use_datalab: bool | None = typer.Option(None, "--datalab/--no-datalab", help="Enable/disable Datalab checks."),
    datalab_fast: bool | None = typer.Option(
        None,
        "--datalab-fast/--datalab-full",
        help="Use fast Datalab subset vs full default audit.",
    ),
    cv_folds: int | None = typer.Option(None, min=2, max=20, help="CV folds for Cleanlab."),
    train_cleanlearning: bool | None = typer.Option(None, help="Train a CleanLearning model (classification only)."),
    prune_and_retrain: bool | None = typer.Option(
        None,
        "--prune/--no-prune",
        help="Compare baseline vs pruned-retrain variant.",
    ),
    prune_fraction: float | None = typer.Option(
        None,
        min=0.0,
        max=0.2,
        help="Fraction of the training set to prune when retraining.",
    ),
    prune_max_samples: int | None = typer.Option(None, min=0, help="Max samples to prune when retraining."),
    save_json: Path | None = typer.Option(None, "--save-json", "-o", help="If set, save result JSON to this path."),
) -> None:
    """
    Run one experiment and print a JSON result.

    Examples:
        cleanlab-demo run --dataset adult_income --model logistic_regression
        cleanlab-demo run -d adult_income -m random_forest --max-rows 5000
        cleanlab-demo run --config config.json --save-json results.json
    """
    settings.ensure_dirs()
    logger.info("Starting experiment run")
    base = RunConfig() if config_path is None else RunConfig.model_validate_json(config_path.read_text())
    data = base.model_dump(mode="python")

    if dataset is not None:
        data["dataset"] = dataset
        data["task"] = None
        data["target_col"] = None
    if model is not None:
        data["model"] = ModelConfig(name=model).model_dump(mode="python")
    if test_size is not None:
        split = SplitConfig.model_validate(data.get("split", {}))
        data["split"] = split.model_copy(update={"test_size": test_size}).model_dump(mode="python")
    if max_rows is not None or label_noise is not None:
        demo = DemoConfig.model_validate(data.get("demo", {}))
        updates = {}
        if max_rows is not None:
            updates["max_rows"] = max_rows
        if label_noise is not None:
            updates["label_noise_fraction"] = label_noise
        data["demo"] = demo.model_copy(update=updates).model_dump(mode="python")
    if (
        cleanlab_enabled is not None
        or use_datalab is not None
        or datalab_fast is not None
        or cv_folds is not None
        or train_cleanlearning is not None
        or prune_and_retrain is not None
        or prune_fraction is not None
        or prune_max_samples is not None
    ):
        cfg = CleanlabCfg.model_validate(data.get("cleanlab", {}))
        updates2 = {}
        if cleanlab_enabled is not None:
            updates2["enabled"] = cleanlab_enabled
        if use_datalab is not None:
            updates2["use_datalab"] = use_datalab
        if datalab_fast is not None:
            updates2["datalab_fast"] = datalab_fast
        if cv_folds is not None:
            updates2["cv_folds"] = cv_folds
        if train_cleanlearning is not None:
            updates2["train_cleanlearning"] = train_cleanlearning
        if prune_and_retrain is not None:
            updates2["prune_and_retrain"] = prune_and_retrain
        if prune_fraction is not None:
            updates2["prune_fraction"] = prune_fraction
        if prune_max_samples is not None:
            updates2["prune_max_samples"] = prune_max_samples
        data["cleanlab"] = cfg.model_copy(update=updates2).model_dump(mode="python")

    config = RunConfig.model_validate(data)
    result = run_experiment(config)
    if save_json is not None:
        save_json.parent.mkdir(parents=True, exist_ok=True)
        save_json.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    console.print(JSON(json.dumps(result.model_dump(mode="json"), indent=2)))


@app.command()
def ai_report(
    result_path: Path | None = typer.Option(
        None,
        "--result",
        "-r",
        exists=True,
        help="Path to experiment result JSON. Defaults to artifacts/last_result.json.",
    ),
    no_ai: bool = typer.Option(False, "--no-ai", help="Skip AI and use deterministic report only."),
) -> None:
    """
    Generate a report from experiment results.

    Uses AI (pydantic-ai) when available, otherwise falls back to deterministic analysis.

    Examples:
        cleanlab-demo ai-report
        cleanlab-demo ai-report --result results.json
        cleanlab-demo ai-report --no-ai
    """
    from cleanlab_demo.ai.report import generate_ai_report

    report = generate_ai_report(result_path, use_ai=not no_ai)
    console.print(JSON(report))


@app.command()
def download_data(
    dataset: DatasetName = typer.Argument(..., help="Dataset to download."),
) -> None:
    """
    Download a dataset into the local data cache.

    Examples:
        cleanlab-demo download-data adult_income
        cleanlab-demo download-data bike_sharing
    """
    settings.ensure_dirs()
    hub = DatasetHub(settings.data_dir)
    console.print(f"[blue]Downloading {dataset.value}...[/blue]")
    ds = hub.load(dataset)
    console.print(f"[green]✓[/green] Downloaded {ds.name.value} ({len(ds.df):,} rows) to `{settings.data_dir}`")


@app.command()
def sweep(
    dataset: DatasetName = typer.Argument(..., help="Dataset to run sweep on."),
    models: list[ModelName] = typer.Option(
        [],
        "--model",
        "-m",
        help="Repeatable. If omitted, uses a sensible default list for the dataset task.",
    ),
    max_rows: int | None = typer.Option(None, min=100, help="Max rows to sample from dataset."),
    label_noise: float | None = typer.Option(None, min=0.0, max=0.5, help="Demo label noise fraction."),
    cv_folds: int = typer.Option(3, min=2, max=20, help="CV folds for Cleanlab."),
    save_csv: Path | None = typer.Option(None, "--save-csv", "-o", help="If set, save results CSV to this path."),
) -> None:
    """
    Run a model sweep comparing multiple models on the same dataset.

    Examples:
        cleanlab-demo sweep adult_income
        cleanlab-demo sweep adult_income -m logistic_regression -m random_forest
        cleanlab-demo sweep bike_sharing --max-rows 5000 --save-csv results.csv
    """
    settings.ensure_dirs()
    task = DATASET_DEFAULTS[dataset].task
    if not models:
        models = (
            [
                ModelName.logistic_regression,
                ModelName.hist_gradient_boosting,
                ModelName.random_forest,
                ModelName.extra_trees,
                ModelName.knn,
            ]
            if task == TaskType.classification
            else [
                ModelName.ridge,
                ModelName.hist_gradient_boosting,
                ModelName.random_forest,
                ModelName.extra_trees,
                ModelName.knn,
            ]
        )

    console.print(f"[blue]Running sweep on {dataset.value} with {len(models)} models...[/blue]")

    base = RunConfig(
        dataset=dataset,
        cleanlab=CleanlabCfg(enabled=(task == TaskType.classification), cv_folds=cv_folds),
        demo=DemoConfig(
            label_noise_fraction=float(label_noise or 0.0),
            max_rows=max_rows,
        ),
    )
    rows = run_sweep(dataset=dataset, models=models, base_config=base)

    # Display results as a table
    table = Table(title=f"Model Sweep Results - {dataset.value}")
    table.add_column("Model", style="cyan")
    table.add_column("Primary Metric", justify="right", style="green")
    table.add_column("Label Issues", justify="right")

    for row in rows:
        table.add_row(
            row.model.value,
            f"{row.primary_metric:.4f}",
            str(row.n_label_issues) if row.n_label_issues > 0 else "-",
        )

    console.print(table)

    payload = [r.model_dump(mode="json") for r in rows]

    if save_csv is not None:
        import pandas as pd

        save_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(payload).to_csv(save_csv, index=False)
        console.print(f"[green]✓[/green] Results saved to {save_csv}")

    console.print("\n[dim]Full JSON output:[/dim]")
    console.print(JSON(json.dumps(payload, indent=2)))
