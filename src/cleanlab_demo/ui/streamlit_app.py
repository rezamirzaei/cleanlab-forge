from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from cleanlab_demo.config import (
    DATASET_DEFAULTS,
    CleanlabConfig,
    DatasetName,
    DemoConfig,
    ModelConfig,
    ModelName,
    RunConfig,
    SplitConfig,
    TaskType,
)
from cleanlab_demo.experiments import run_experiment
from cleanlab_demo.settings import settings

_MIN_MAX_ROWS = 100


def _available_models(task: TaskType) -> list[ModelName]:
    if task == TaskType.classification:
        return [
            ModelName.logistic_regression,
            ModelName.hist_gradient_boosting,
            ModelName.random_forest,
            ModelName.extra_trees,
            ModelName.knn,
        ]
    return [
        ModelName.ridge,
        ModelName.hist_gradient_boosting,
        ModelName.random_forest,
        ModelName.extra_trees,
        ModelName.knn,
    ]


def main() -> None:
    st.set_page_config(page_title="Cleanlab Demo", layout="wide")
    st.title("Cleanlab Demo")

    with st.sidebar:
        dataset = st.selectbox("Dataset", options=list(DatasetName), format_func=lambda d: d.value)
        task = DATASET_DEFAULTS[dataset].task
        st.caption(f"Task: {task.value}")

        model = st.selectbox(
            "Model",
            options=_available_models(task),
            format_func=lambda m: m.value,
        )

        test_size = st.slider("Test size", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
        max_rows = st.number_input(
            f"Max rows (0 = use all data, min {_MIN_MAX_ROWS} otherwise)",
            min_value=0,
            value=0,
            step=1000,
            help=f"Set to 0 to use all rows, or at least {_MIN_MAX_ROWS} to subsample.",
        )
        if 0 < max_rows < _MIN_MAX_ROWS:
            st.warning(f"Max rows must be at least {_MIN_MAX_ROWS}. Using {_MIN_MAX_ROWS}.")
            max_rows = _MIN_MAX_ROWS

        st.markdown("---")
        st.subheader("Cleanlab")

        cleanlab_enabled = st.checkbox("Enable Cleanlab analysis", value=True)
        use_datalab = st.checkbox("Use Datalab (outliers/duplicates/etc)", value=True)
        datalab_fast = st.checkbox("Fast Datalab checks", value=True)
        prune_and_retrain = st.checkbox("Compare: prune & retrain", value=True)
        prune_fraction = st.slider("Prune fraction", 0.0, 0.2, 0.02, 0.01)

        if task == TaskType.classification:
            label_noise = st.slider("Label noise (demo)", 0.0, 0.5, 0.0, 0.01)
            train_cleanlearning = st.checkbox("Train CleanLearning model", value=False)
            cv_folds = st.slider("CV folds", 2, 10, 5, 1)
        else:
            label_noise = 0.0
            train_cleanlearning = False
            cv_folds = 5

        run_clicked = st.button("Run experiment", type="primary")

    if not run_clicked:
        st.info("Pick options in the sidebar and click **Run experiment**.")
        return

    config = RunConfig(
        dataset=dataset,
        task=task,
        split=SplitConfig(test_size=float(test_size), random_state=42, stratify=True),
        model=ModelConfig(name=model),
        cleanlab=CleanlabConfig(
            enabled=bool(cleanlab_enabled),
            use_datalab=bool(use_datalab),
            datalab_fast=bool(datalab_fast),
            train_cleanlearning=bool(train_cleanlearning),
            prune_and_retrain=bool(prune_and_retrain),
            prune_fraction=float(prune_fraction),
            cv_folds=int(cv_folds),
        ),
        demo=DemoConfig(
            label_noise_fraction=float(label_noise),
            max_rows=(int(max_rows) if int(max_rows) > 0 else None),
        ),
    )

    with st.spinner("Running..."):
        result = run_experiment(config)
    try:
        settings.ensure_dirs()
        (settings.artifacts_dir / "last_result.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
    except Exception as e:
        st.warning(f"Could not save result to artifacts directory: {e}")

    st.subheader("Model comparison (with/without Cleanlab)")
    variants_df = pd.DataFrame(
        [
            {
                "variant": v.variant.value,
                "n_train": v.n_train,
                "primary": v.metrics.primary,
                **(v.metrics.details or {}),
                **(v.notes or {}),
            }
            for v in (result.variants or [])
        ]
    )
    if not variants_df.empty:
        baseline_primary = float(variants_df.loc[variants_df["variant"] == "baseline", "primary"].iloc[0])
        variants_df["delta_vs_baseline"] = variants_df["primary"].astype(float) - baseline_primary
        st.dataframe(variants_df.sort_values("primary", ascending=False), use_container_width=True)
        st.bar_chart(variants_df.set_index("variant")["primary"])
    else:
        st.write({"primary": result.metrics.primary, **result.metrics.details})

    if result.cleanlab_summary:
        st.subheader("Cleanlab")
        if "error" in result.cleanlab_summary:
            st.warning(result.cleanlab_summary["error"])
        if "datalab_error" in result.cleanlab_summary:
            st.warning(result.cleanlab_summary["datalab_error"])
        if "datalab_issue_summary" in result.cleanlab_summary:
            st.markdown("**Datalab issue summary**")
            st.dataframe(pd.DataFrame(result.cleanlab_summary["datalab_issue_summary"]), use_container_width=True)
        if "datalab_examples" in result.cleanlab_summary:
            with st.expander("Datalab examples (flagged issues)", expanded=False):
                st.json(result.cleanlab_summary["datalab_examples"])
        if "datalab_issues_csv" in result.cleanlab_summary:
            st.caption(f"Saved: `{result.cleanlab_summary['datalab_issues_csv']}`")

    with st.expander("AI report (pydantic-ai)", expanded=False):
        use_ai = st.checkbox("Use LLM (requires API key)", value=False)
        if st.button("Generate report"):
            from cleanlab_demo.ai.report import generate_ai_report

            report_json = generate_ai_report(settings.artifacts_dir / "last_result.json", use_ai=use_ai)
            try:
                st.json(json.loads(report_json))
            except Exception:
                st.code(report_json)

    st.subheader("Label issues")
    if not result.label_issues:
        st.caption("No label issues (or Cleanlab disabled).")
        return

    issues_df = pd.DataFrame([li.model_dump() for li in result.label_issues])
    st.dataframe(issues_df)


if __name__ == "__main__":
    main()
