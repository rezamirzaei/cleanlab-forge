from __future__ import annotations

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


def _available_models(task: TaskType) -> list[ModelName]:
    if task == TaskType.classification:
        return [
            ModelName.logistic_regression,
            ModelName.hist_gradient_boosting,
            ModelName.random_forest,
        ]
    return [ModelName.ridge, ModelName.hist_gradient_boosting, ModelName.random_forest]


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
        max_rows = st.number_input("Max rows (optional)", min_value=0, value=0, step=1000)

        if task == TaskType.classification:
            label_noise = st.slider("Label noise (demo)", 0.0, 0.5, 0.0, 0.01)
            cleanlab_enabled = st.checkbox("Run Cleanlab", value=True)
            use_datalab = st.checkbox("Use Datalab (more issue types)", value=True)
            train_cleanlearning = st.checkbox("Train CleanLearning model", value=False)
            cv_folds = st.slider("CV folds", 2, 10, 5, 1)
        else:
            label_noise = 0.0
            cleanlab_enabled = False
            use_datalab = False
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
            train_cleanlearning=bool(train_cleanlearning),
            cv_folds=int(cv_folds),
        ),
        demo=DemoConfig(
            label_noise_fraction=float(label_noise),
            max_rows=(int(max_rows) if int(max_rows) > 0 else None),
        ),
    )

    with st.spinner("Running..."):
        result = run_experiment(config)

    st.subheader("Metrics")
    st.write({"primary": result.metrics.primary, **result.metrics.details})

    if result.cleanlab_summary:
        st.subheader("Cleanlab")
        if "error" in result.cleanlab_summary:
            st.warning(result.cleanlab_summary["error"])
        if "issue_summary" in result.cleanlab_summary:
            st.markdown("**Datalab issue summary**")
            try:
                st.dataframe(pd.DataFrame(result.cleanlab_summary["issue_summary"]))
            except Exception:
                st.json(result.cleanlab_summary["issue_summary"])

    st.subheader("Label issues")
    if not result.label_issues:
        st.caption("No label issues (or Cleanlab disabled).")
        return

    issues_df = pd.DataFrame([li.model_dump() for li in result.label_issues])
    st.dataframe(issues_df)

    examples = result.cleanlab_summary.get("label_issue_examples", [])
    if examples:
        st.subheader("Examples (first 20)")
        st.json(examples)


if __name__ == "__main__":
    main()
