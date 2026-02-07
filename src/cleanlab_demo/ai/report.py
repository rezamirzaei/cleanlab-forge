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

    steps = [
        "Inspect top-ranked label issues and verify labels manually.",
        "Try training after removing/relabelling the worst issues and compare metrics.",
        "Compare at least 2 different models (linear + tree/boosting) to validate robustness.",
    ]
    if task == "regression":
        steps.insert(0, "Cleanlab analysis is mainly classification-focused; start with outliers and duplicates.")

    # Add metric-specific recommendations
    if metrics.get("roc_auc", 0) < 0.7:
        steps.append("Consider feature engineering or trying more complex models - AUC is below 0.7.")
    if n_issues > 50:
        steps.append(f"High number of label issues ({n_issues}) detected - prioritize data quality review.")

    return AIExperimentReport(
        headline=f"{dataset} / {model} ({task})",
        summary=f"Found {n_issues} potential label issues (if Cleanlab enabled). "
        f"Primary metric: {result.get('metrics', {}).get('primary', 'N/A')}",
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

    try:
        from pydantic_ai import Agent
    except ImportError:
        logger.info("pydantic-ai not installed, using deterministic report")
        return baseline.model_dump_json(indent=2)

    model_name = _get_ai_model()
    system_prompt = (
        "You are a senior ML engineer. Given an experiment result JSON that may include Cleanlab "
        "label issue findings, produce a concise report with actionable next steps. "
        "Focus on practical recommendations for improving model performance and data quality. "
        "Return ONLY valid JSON matching the required schema."
    )

    prompt = json.dumps(
        {
            "result": result,
            "baseline_report": baseline.model_dump(mode="json"),
        },
        indent=2,
    )

    try:
        logger.info(f"Generating AI report using {model_name}")
        agent = Agent(model=model_name, output_type=AIExperimentReport, system_prompt=system_prompt)
        ai_result = agent.run_sync(prompt)
        data = getattr(ai_result, "data", None) or getattr(ai_result, "output", None) or ai_result
        if isinstance(data, AIExperimentReport):
            return data.model_dump_json(indent=2)
        return AIExperimentReport.model_validate(data).model_dump_json(indent=2)
    except Exception as e:
        logger.warning(f"AI report generation failed: {e}, using deterministic fallback")
        return baseline.model_dump_json(indent=2)

