from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from pydantic import BaseModel as PydanticBaseModel

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


def _save_model_json(name: str, model: PydanticBaseModel) -> str | None:
    try:
        settings.ensure_dirs()
        path = settings.artifacts_dir / f"last_{name}.json"
        path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
        return str(path)
    except Exception as e:
        st.warning(f"Could not save result JSON: {e}")
        return None


def _render_json_expander(model: PydanticBaseModel, *, title: str = "Result JSON") -> None:
    with st.expander(title, expanded=False):
        st.json(json.loads(model.model_dump_json()))


def _render_tabular_e2e() -> None:
    st.subheader("Tabular end-to-end (CLI runner)")

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

        run_clicked = st.button("Run experiment", type="primary", key="run_tabular")

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

    saved = _save_model_json("result", result)
    if saved:
        st.caption(f"Saved: `{saved}`")

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
        baseline_primary = float(
            variants_df.loc[variants_df["variant"] == "baseline", "primary"].iloc[0]
        )
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
            st.dataframe(
                pd.DataFrame(result.cleanlab_summary["datalab_issue_summary"]),
                use_container_width=True,
            )
        if "datalab_examples" in result.cleanlab_summary:
            with st.expander("Datalab examples (flagged issues)", expanded=False):
                st.json(result.cleanlab_summary["datalab_examples"])
        if "datalab_issues_csv" in result.cleanlab_summary:
            st.caption(f"Saved: `{result.cleanlab_summary['datalab_issues_csv']}`")

    with st.expander("AI report (pydantic-ai)", expanded=False):
        use_ai = st.checkbox("Use LLM (requires API key)", value=False)
        if st.button("Generate report"):
            from cleanlab_demo.ai.report import generate_ai_report

            report_json = generate_ai_report(
                settings.artifacts_dir / "last_result.json", use_ai=use_ai
            )
            try:
                st.json(json.loads(report_json))
            except Exception:
                st.code(report_json)

    st.subheader("Label issues")
    if not result.label_issues:
        st.caption("No label issues (or Cleanlab disabled).")
        _render_json_expander(result)
        return

    issues_df = pd.DataFrame([li.model_dump() for li in result.label_issues])
    st.dataframe(issues_df)
    _render_json_expander(result)


def _render_multiclass_covtype_demo() -> None:
    st.subheader("Multi-class classification (CoverType)")
    st.caption("Optional synthetic label noise for evaluation (set to 0.0 for real-world run).")

    with st.sidebar:
        st.markdown("---")
        st.subheader("CoverType demo")
        max_rows = st.number_input(
            "Max rows", min_value=100, max_value=600_000, value=10_000, step=1000
        )
        test_size = st.slider("Test size", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
        noise_frac = st.slider("Noise fraction", min_value=0.0, max_value=0.5, value=0.0, step=0.01)
        cv_folds = st.slider("CV folds", min_value=2, max_value=20, value=3, step=1)
        prune_frac = st.slider(
            "Prune fraction", min_value=0.0, max_value=0.2, value=0.02, step=0.01
        )
        seed = st.number_input("Seed", min_value=0, max_value=1_000_000, value=42, step=1)
        run_clicked = st.button("Run demo", type="primary", key="run_covtype")

    if not run_clicked:
        st.info("Pick options in the sidebar and click **Run demo**.")
        return

    from cleanlab_demo.data.providers import CovtypeDataProvider
    from cleanlab_demo.tasks.multiclass import (
        MulticlassClassificationConfig,
        MulticlassClassificationTask,
    )

    with st.spinner("Running..."):
        task = MulticlassClassificationTask(CovtypeDataProvider(max_rows=int(max_rows)))
        config = MulticlassClassificationConfig(
            test_size=float(test_size),
            noise_frac=float(noise_frac),
            cv_folds=int(cv_folds),
            prune_frac=float(prune_frac),
            seed=int(seed),
        )
        result = task.run(config)

    saved = _save_model_json("demo_multiclass_covtype", result)
    if saved:
        st.caption(f"Saved: `{saved}`")

    metrics_df = pd.DataFrame(
        [
            {"variant": "baseline", **result.metrics.baseline.model_dump()},
            {"variant": "pruned_retrain", **result.metrics.pruned_retrain.model_dump()},
        ]
    )
    st.dataframe(metrics_df, use_container_width=True)
    st.json(result.cleanlab.model_dump())
    _render_json_expander(result)


def _render_multilabel_emotions_demo() -> None:
    st.subheader("Multi-label classification (OpenML emotions)")
    st.caption("Optional synthetic tag noise for evaluation (set to 0.0 for real-world run).")

    with st.sidebar:
        st.markdown("---")
        st.subheader("Emotions demo")
        test_size = st.slider(
            "Test size", min_value=0.05, max_value=0.5, value=0.2, step=0.05, key="emo_test"
        )
        noise_frac = st.slider(
            "Noise fraction", min_value=0.0, max_value=0.5, value=0.0, step=0.01, key="emo_noise"
        )
        cv_folds = st.slider("CV folds", min_value=2, max_value=20, value=5, step=1, key="emo_cv")
        prune_frac = st.slider(
            "Prune fraction", min_value=0.0, max_value=0.5, value=0.05, step=0.01, key="emo_prune"
        )
        seed = st.number_input(
            "Seed", min_value=0, max_value=1_000_000, value=42, step=1, key="emo_seed"
        )
        run_clicked = st.button("Run demo", type="primary", key="run_emotions")

    if not run_clicked:
        st.info("Pick options in the sidebar and click **Run demo**.")
        return

    from cleanlab_demo.data.providers import EmotionsDataProvider
    from cleanlab_demo.tasks.multilabel import (
        MultilabelClassificationConfig,
        MultilabelClassificationTask,
    )

    with st.spinner("Running..."):
        task = MultilabelClassificationTask(EmotionsDataProvider())
        config = MultilabelClassificationConfig(
            test_size=float(test_size),
            noise_frac=float(noise_frac),
            cv_folds=int(cv_folds),
            prune_frac=float(prune_frac),
            seed=int(seed),
        )
        result = task.run(config)

    saved = _save_model_json("demo_multilabel_emotions", result)
    if saved:
        st.caption(f"Saved: `{saved}`")

    metrics_df = pd.DataFrame(
        [
            {"variant": "baseline", **result.metrics.baseline.model_dump()},
            {"variant": "pruned_retrain", **result.metrics.pruned_retrain.model_dump()},
        ]
    )
    st.dataframe(metrics_df, use_container_width=True)
    st.json(result.cleanlab.model_dump())
    _render_json_expander(result)


def _render_token_udpos_demo() -> None:
    st.subheader("Token classification (UD English EWT POS)")
    st.caption(f"Downloads UD files into `{settings.data_dir / 'ud_ewt'}`.")

    with st.sidebar:
        st.markdown("---")
        st.subheader("UD POS demo")
        max_train = st.slider(
            "Max train sentences", min_value=50, max_value=10_000, value=1000, step=50
        )
        max_dev = st.slider("Max dev sentences", min_value=50, max_value=5000, value=300, step=50)
        noise_frac = st.slider(
            "Token noise fraction", min_value=0.0, max_value=0.5, value=0.0, step=0.01
        )
        cv_folds = st.slider("CV folds", min_value=2, max_value=10, value=3, step=1)
        prune_frac = st.slider(
            "Prune fraction", min_value=0.0, max_value=0.2, value=0.03, step=0.01
        )
        seed = st.number_input("Seed", min_value=0, max_value=1_000_000, value=42, step=1)
        run_clicked = st.button("Run demo", type="primary", key="run_udpos")

    if not run_clicked:
        st.info("Pick options in the sidebar and click **Run demo**.")
        return

    from cleanlab_demo.data.providers import UDEnglishEWTProvider
    from cleanlab_demo.tasks.token import TokenClassificationConfig, TokenClassificationTask

    with st.spinner("Running..."):
        task = TokenClassificationTask(UDEnglishEWTProvider())
        config = TokenClassificationConfig(
            max_train_sentences=int(max_train),
            max_dev_sentences=int(max_dev),
            noise_frac=float(noise_frac),
            cv_folds=int(cv_folds),
            prune_frac=float(prune_frac),
            seed=int(seed),
        )
        result = task.run(config)

    saved = _save_model_json("demo_token_udpos", result)
    if saved:
        st.caption(f"Saved: `{saved}`")

    metrics_df = pd.DataFrame(
        [
            {"variant": "baseline", **result.metrics.baseline.model_dump()},
            {"variant": "pruned_retrain", **result.metrics.pruned_retrain.model_dump()},
        ]
    )
    st.dataframe(metrics_df, use_container_width=True)
    st.json(result.cleanlab.model_dump())
    _render_json_expander(result)


def _render_regression_bike_cleanlearning_demo() -> None:
    st.subheader("Regression (Bike Sharing) + Cleanlab regression CleanLearning")
    st.caption("Optional synthetic label noise for evaluation (set to 0.0 for real-world run).")

    with st.sidebar:
        st.markdown("---")
        st.subheader("Bike Sharing demo")
        max_rows = st.number_input(
            "Max rows", min_value=200, max_value=200_000, value=10_000, step=1000
        )
        test_size = st.slider(
            "Test size", min_value=0.05, max_value=0.5, value=0.2, step=0.05, key="bike_test"
        )
        noise_frac = st.slider(
            "Noise fraction", min_value=0.0, max_value=0.5, value=0.0, step=0.01, key="bike_noise"
        )
        cv_folds = st.slider("CV folds", min_value=2, max_value=20, value=5, step=1, key="bike_cv")
        prune_frac = st.slider(
            "Prune fraction", min_value=0.0, max_value=0.2, value=0.02, step=0.01, key="bike_prune"
        )
        seed = st.number_input(
            "Seed", min_value=0, max_value=1_000_000, value=42, step=1, key="bike_seed"
        )
        run_clicked = st.button("Run demo", type="primary", key="run_bike_cleanlearning")

    if not run_clicked:
        st.info("Pick options in the sidebar and click **Run demo**.")
        return

    from cleanlab_demo.data.providers import BikeSharingDataProvider
    from cleanlab_demo.tasks.regression import (
        RegressionCleanLearningConfig,
        RegressionCleanLearningTask,
    )

    with st.spinner("Running..."):
        task = RegressionCleanLearningTask(BikeSharingDataProvider(max_rows=int(max_rows)))
        config = RegressionCleanLearningConfig(
            test_size=float(test_size),
            noise_frac=float(noise_frac),
            cv_folds=int(cv_folds),
            prune_frac=float(prune_frac),
            seed=int(seed),
        )
        result = task.run(config)

    saved = _save_model_json("demo_regression_bike_cleanlearning", result)
    if saved:
        st.caption(f"Saved: `{saved}`")

    metrics_df = pd.DataFrame(
        [
            {"variant": "baseline", **result.metrics.baseline.model_dump()},
            {"variant": "pruned_retrain", **result.metrics.pruned_retrain.model_dump()},
        ]
    )
    st.dataframe(metrics_df, use_container_width=True)
    st.json(result.cleanlab.model_dump())
    _render_json_expander(result)


def _render_outlier_california_housing_demo() -> None:
    st.subheader("Outlier detection (California Housing) + Cleanlab Datalab")
    st.caption("Optional synthetic feature outliers for evaluation (set to 0.0 for real-world run).")

    with st.sidebar:
        st.markdown("---")
        st.subheader("Outlier demo")
        max_rows = st.number_input(
            "Max rows", min_value=200, max_value=200_000, value=10_000, step=1000
        )
        outlier_frac = st.slider(
            "Synthetic outlier fraction", min_value=0.0, max_value=0.2, value=0.0, step=0.01
        )
        seed = st.number_input(
            "Seed", min_value=0, max_value=1_000_000, value=42, step=1, key="out_seed"
        )
        run_clicked = st.button("Run demo", type="primary", key="run_outlier")

    if not run_clicked:
        st.info("Pick options in the sidebar and click **Run demo**.")
        return

    from cleanlab_demo.data.providers import CaliforniaHousingOutlierProvider
    from cleanlab_demo.tasks.outlier import OutlierDetectionConfig, OutlierDetectionTask

    with st.spinner("Running..."):
        task = OutlierDetectionTask(CaliforniaHousingOutlierProvider(max_rows=int(max_rows)))
        config = OutlierDetectionConfig(
            outlier_frac=float(outlier_frac),
            seed=int(seed),
        )
        result = task.run(config)

    saved = _save_model_json("demo_outlier_california_housing", result)
    if saved:
        st.caption(f"Saved: `{saved}`")

    st.json(result.cleanlab.model_dump())
    _render_json_expander(result)


def _render_multiannotator_active_learning_demo() -> None:
    st.subheader("Multi-annotator classification + active learning (MovieLens 100K)")
    st.caption("Real multi-rater labels: users rate movies (1-5), with missing labels allowed.")

    with st.sidebar:
        st.markdown("---")
        st.subheader("Multi-annotator demo")
        max_movies = st.number_input("Max movies", min_value=50, max_value=5000, value=800, step=50)
        max_annotators = st.slider("Max annotators", min_value=5, max_value=200, value=50, step=5)
        min_ratings_per_movie = st.slider(
            "Min ratings per movie", min_value=5, max_value=100, value=10, step=1
        )
        min_ratings_per_annotator = st.slider(
            "Min ratings per annotator", min_value=10, max_value=500, value=100, step=10
        )
        cv_folds = st.slider("CV folds", min_value=2, max_value=20, value=5, step=1)
        seed = st.number_input(
            "Seed", min_value=0, max_value=1_000_000, value=42, step=1, key="ma_seed"
        )
        run_clicked = st.button("Run demo", type="primary", key="run_multiannotator")

    if not run_clicked:
        st.info("Pick options in the sidebar and click **Run demo**.")
        return

    from cleanlab_demo.data.providers import MovieLens100KProvider
    from cleanlab_demo.tasks.multiannotator import MultiannotatorConfig, MultiannotatorTask

    with st.spinner("Running..."):
        task = MultiannotatorTask(
            MovieLens100KProvider(
                max_movies=int(max_movies),
                max_annotators=int(max_annotators),
                min_ratings_per_movie=int(min_ratings_per_movie),
                min_ratings_per_annotator=int(min_ratings_per_annotator),
            )
        )
        config = MultiannotatorConfig(
            cv_folds=int(cv_folds),
            seed=int(seed),
        )
        result = task.run(config)

    saved = _save_model_json("demo_multiannotator_active_learning", result)
    if saved:
        st.caption(f"Saved: `{saved}`")

    st.json(result.notes.model_dump())
    st.json(result.cleanlab.model_dump())
    _render_json_expander(result)


def _render_vision_pennfudan_demo() -> None:
    st.subheader("Vision (PennFudanPed): object detection + segmentation")
    st.caption("Requires torch/torchvision installed (see README).")

    with st.sidebar:
        st.markdown("---")
        st.subheader("Vision demo")
        data_dir = st.text_input("Data dir", value=str(settings.data_dir / "pennfudan"))
        max_images = st.slider("Max images", min_value=1, max_value=50, value=8, step=1)
        score_threshold = st.slider(
            "Score threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05
        )
        corrupt_frac = st.slider(
            "Corrupt fraction", min_value=0.0, max_value=1.0, value=0.0, step=0.05
        )
        seed = st.number_input(
            "Seed", min_value=0, max_value=1_000_000, value=42, step=1, key="vis_seed"
        )
        run_clicked = st.button("Run demo", type="primary", key="run_vision")

    if not run_clicked:
        st.info("Pick options in the sidebar and click **Run demo**.")
        return

    from cleanlab_demo.data.providers import PennFudanPedProvider
    from cleanlab_demo.tasks.vision import (
        VisionDetectionSegmentationConfig,
        VisionDetectionSegmentationTask,
    )

    try:
        with st.spinner("Running... (may take a while)"):
            task = VisionDetectionSegmentationTask(PennFudanPedProvider())
            config = VisionDetectionSegmentationConfig(
                data_dir=Path(data_dir),
                max_images=int(max_images),
                score_threshold=float(score_threshold),
                corrupt_frac=float(corrupt_frac),
                seed=int(seed),
            )
            result = task.run(config)
    except Exception as e:
        st.error(str(e))
        return

    saved = _save_model_json("demo_vision_pennfudan", result)
    if saved:
        st.caption(f"Saved: `{saved}`")

    st.json(result.object_detection.model_dump())
    st.json(result.segmentation.model_dump())
    _render_json_expander(result)


def main() -> None:
    st.set_page_config(page_title="Cleanlab Demo", layout="wide")
    st.title("Cleanlab Demo")

    mode = st.sidebar.selectbox(
        "Mode",
        options=[
            "Tabular (end-to-end)",
            "Demo: multi-class (CoverType)",
            "Demo: multi-label (emotions)",
            "Demo: token classification (UD POS)",
            "Demo: regression (Bike Sharing)",
            "Demo: outlier detection (Datalab)",
            "Demo: multi-annotator + active learning",
            "Demo: vision (PennFudanPed)",
        ],
    )

    if mode == "Tabular (end-to-end)":
        _render_tabular_e2e()
    elif mode == "Demo: multi-class (CoverType)":
        _render_multiclass_covtype_demo()
    elif mode == "Demo: multi-label (emotions)":
        _render_multilabel_emotions_demo()
    elif mode == "Demo: token classification (UD POS)":
        _render_token_udpos_demo()
    elif mode == "Demo: regression (Bike Sharing)":
        _render_regression_bike_cleanlearning_demo()
    elif mode == "Demo: outlier detection (Datalab)":
        _render_outlier_california_housing_demo()
    elif mode == "Demo: multi-annotator + active learning":
        _render_multiannotator_active_learning_demo()
    elif mode == "Demo: vision (PennFudanPed)":
        _render_vision_pennfudan_demo()


if __name__ == "__main__":
    main()
