from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from cleanlab_demo.settings import logger, settings


class AIExperimentReport(BaseModel):
    """Structured AI-generated report on experiment results."""

    headline: str = Field(description="Brief summary headline")
    summary: str = Field(description="Detailed summary of findings")
    key_metrics: dict[str, float] = Field(default_factory=dict, description="Key performance metrics")
    recommended_next_steps: list[str] = Field(default_factory=list, description="Actionable recommendations")


def _deterministic_report(result: dict[str, Any]) -> AIExperimentReport:
    """Generate a deterministic (non-LLM) report from experiment results."""
    metrics = result.get("metrics", {}).get("details", {}) or {}
    n_issues = len(result.get("label_issues", []) or [])
    dataset = result.get("dataset", "unknown")
    model = result.get("model", "unknown")
    task = result.get("task", "unknown")
    variants = result.get("variants", []) or []
    datalab_summary = (result.get("cleanlab_summary", {}) or {}).get("datalab_issue_summary", []) or []

    steps = [
        "Inspect top-ranked label issues and verify labels manually.",
        "Try training after removing/relabelling the worst issues and compare metrics.",
        "Compare at least 2 different models (linear + tree/boosting) to validate robustness.",
    ]
    if task == "regression":
        steps.insert(0, "Start with outliers and near-duplicates detected by Datalab, then retrain and compare.")

    # Add metric-specific recommendations
    if metrics.get("roc_auc", 0) < 0.7:
        steps.append("Consider feature engineering or trying more complex models - AUC is below 0.7.")
    if n_issues > 50:
        steps.append(f"High number of label issues ({n_issues}) detected - prioritize data quality review.")

    best_variant = None
    try:
        best_variant = max(variants, key=lambda v: float(v.get("metrics", {}).get("primary", float("-inf"))))
    except Exception:
        best_variant = None

    outlier_count = None
    near_dup_count = None
    try:
        for row in datalab_summary:
            if row.get("issue_type") == "outlier":
                outlier_count = int(row.get("num_issues", 0))
            if row.get("issue_type") == "near_duplicate":
                near_dup_count = int(row.get("num_issues", 0))
    except Exception:
        pass

    if outlier_count:
        steps.append(f"Review {outlier_count} potential outliers flagged by Datalab.")
    if near_dup_count:
        steps.append(f"Review {near_dup_count} potential near-duplicates flagged by Datalab.")

    return AIExperimentReport(
        headline=f"{dataset} / {model} ({task})",
        summary=(
            f"Found {n_issues} potential label issues. "
            f"Baseline primary metric: {result.get('metrics', {}).get('primary', 'N/A')}. "
            + (
                f"Best variant: {best_variant.get('variant')} "
                f"({best_variant.get('metrics', {}).get('primary')})."
                if isinstance(best_variant, dict) and best_variant.get("variant")
                else ""
            )
        ).strip(),
        key_metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        recommended_next_steps=steps,
    )


def _get_ai_model() -> str:
    """Determine which AI model to use based on available API keys."""
    if settings.openai_api_key or os.getenv("OPENAI_API_KEY"):
        return "openai:gpt-4o-mini"
    if settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic:claude-3-haiku-20240307"
    # Default to OpenAI (will fail gracefully if no key)
    return "openai:gpt-4o-mini"


def generate_ai_report(result_path: Path | None = None, *, use_ai: bool = True) -> str:
    """
    Produce a structured report using pydantic-ai when available.

    Falls back to a deterministic (non-LLM) report if pydantic-ai isn't installed/configured.

    Args:
        result_path: Path to experiment result JSON. Defaults to artifacts_dir/last_result.json
        use_ai: Whether to attempt AI-powered report generation

    Returns:
        JSON string containing the report
    """
    path = result_path or (settings.artifacts_dir / "last_result.json")
    if not path.exists():
        return json.dumps(
            {
                "error": f"Missing `{path}`. Run `cleanlab-demo run --save-json {path}` first, "
                "then rerun `cleanlab-demo ai-report`."
            },
            indent=2,
        )

    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse result JSON: {e}")
        return json.dumps({"error": f"Invalid JSON in {path}: {e}"}, indent=2)

    baseline = _deterministic_report(result)

    if not use_ai:
        return baseline.model_dump_json(indent=2)

    if not (settings.openai_api_key or settings.anthropic_api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        baseline.recommended_next_steps.insert(
            0, "Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` to enable the LLM-backed report."
        )
        return baseline.model_dump_json(indent=2)

    try:
        from pydantic_ai import Agent
    except ImportError:
        logger.info("pydantic-ai not installed, using deterministic report")
        return baseline.model_dump_json(indent=2)

    model_name = _get_ai_model()
    system_prompt = (
        "You are a senior ML engineer. You will be given access to tools that can fetch:\n"
        "- experiment metrics and model variants\n"
        "- Cleanlab label issues\n"
        "- Datalab issue summaries (outliers/near-duplicates/etc)\n"
        "\n"
        "Produce a concise report with actionable next steps.\n"
        "Return ONLY valid JSON matching the required schema."
    )

    try:
        from dataclasses import dataclass

        from pydantic_ai import RunContext

        @dataclass
        class Deps:
            result: dict[str, Any]
            baseline: AIExperimentReport

        logger.info(f"Generating AI report using {model_name}")
        agent = Agent(model=model_name, output_type=AIExperimentReport, deps_type=Deps, system_prompt=system_prompt)

        @agent.tool
        def get_result(ctx: RunContext[Deps]) -> dict[str, Any]:
            """Return the full experiment result JSON."""
            return ctx.deps.result

        @agent.tool
        def get_baseline_report(ctx: RunContext[Deps]) -> dict[str, Any]:
            """Return the deterministic (non-LLM) baseline report JSON."""
            return ctx.deps.baseline.model_dump(mode="json")

        @agent.tool
        def get_top_label_issues(ctx: RunContext[Deps], n: int = 10) -> list[dict[str, Any]]:
            """Return top-N label issues (ranked)."""
            issues = ctx.deps.result.get("label_issues") or []
            return list(issues)[: int(n)]

        @agent.tool
        def get_variant_table(ctx: RunContext[Deps]) -> list[dict[str, Any]]:
            """Return baseline vs Cleanlab variant metrics."""
            return ctx.deps.result.get("variants") or []

        @agent.tool
        def get_datalab_issue_summary(ctx: RunContext[Deps]) -> list[dict[str, Any]]:
            """Return Datalab issue summary rows."""
            summary = (ctx.deps.result.get("cleanlab_summary") or {}).get("datalab_issue_summary") or []
            return list(summary)

        deps = Deps(result=result, baseline=baseline)
        ai_result = agent.run_sync(
            "Generate the report. Prefer calling tools instead of relying on assumptions.",
            deps=deps,
        )
        return ai_result.output.model_dump_json(indent=2)
    except Exception as e:
        logger.warning(f"AI report generation failed: {e}, using deterministic fallback")
        return baseline.model_dump_json(indent=2)
