# Cleanlab Demo (end-to-end)

A comprehensive ML project demonstrating [Cleanlab](https://github.com/cleanlab/cleanlab) for automatic label issue detection. This project downloads real-world datasets, trains multiple ML models, applies Cleanlab to find mislabeled data, and provides both a CLI and web UI.

## Features

- 📊 **Multiple Datasets**: UCI Adult Income (classification), UCI Bike Sharing (regression), California Housing (regression; natural outliers)
- 🤖 **Model Support**: Logistic Regression, k-NN, Random Forest, ExtraTrees, Histogram Gradient Boosting, Ridge
- 🔍 **Cleanlab Integration**: Label issues + Datalab (outliers/near-duplicates/non-iid) + optional prune & retrain comparison
- 🧩 **More Task Types**: Multi-label, token classification, multi-annotator labeling + active learning, object detection, image segmentation (via tasks + notebooks)
- 📈 **Model Sweeps**: Compare multiple models on the same dataset
- 🎛️ **Streamlit UI**: End-to-end tabular runner + dedicated task pages
- 📓 **Jupyter Notebooks**: Step-by-step tutorials
- 🤖 **AI Reports**: Optional LLM-powered analysis reports (via pydantic-ai)

## Quickstart (Docker)

```bash
docker compose up --build
```

- **Streamlit UI**: http://localhost:8501
- **JupyterLab**: http://localhost:8888

## Local Installation (venv)

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with all optional dependencies
pip install -U pip
pip install -e ".[dev,ui,notebooks,ai]"

# Run tests
pytest -v

# Verify installation
cleanlab-demo --version
```

## Run the UI (Streamlit)

```bash
streamlit run src/cleanlab_demo/ui/streamlit_app.py
```

Use the sidebar **Mode** selector to switch between the tabular end-to-end runner and the dedicated task pages.

## CLI Usage

```bash
# Show help
cleanlab-demo --help

# Run a single experiment
cleanlab-demo run --dataset adult_income --model logistic_regression

# Run with custom parameters
cleanlab-demo run -d adult_income -m random_forest --max-rows 5000 --cleanlab --prune --prune-fraction 0.02

# Save results to file
cleanlab-demo run -d adult_income -o results/experiment.json

# Run a model sweep (compare multiple models)
cleanlab-demo sweep adult_income
cleanlab-demo sweep adult_income -m logistic_regression -m random_forest --save-csv results.csv

# Download datasets
cleanlab-demo download-data adult_income
cleanlab-demo download-data bike_sharing
cleanlab-demo download-data california_housing
```

## Task Coverage (and how to run each)

This repo includes a tabular end-to-end CLI (above) plus dedicated task implementations under `src/cleanlab_demo/tasks/`.
Each task returns a Pydantic-validated result; notebooks/UI default to real-world runs (synthetic corruption defaults to `0.0`, but can be turned on to measure precision/recall).

- **Binary classification** (tabular): `cleanlab-demo run -d adult_income --cleanlab`
- **Multi-class classification** (CoverType): `notebooks/05_multiclass_classification_covtype.ipynb`
- **Multi-label classification** (OpenML emotions): `notebooks/06_multilabel_classification_emotions.ipynb`
- **Token classification** (UD POS): `notebooks/07_token_classification_ud_pos.ipynb`
- **Regression** (with regression-specific CleanLearning): `notebooks/08_regression_cleanlearning_bike_sharing.ipynb`
- **Multi-annotator + active learning** (MovieLens 100K): `notebooks/09_multiannotator_active_learning_movielens_100k.ipynb`
- **Outlier detection** (Datalab): `notebooks/10_outlier_detection_datalab_california_housing.ipynb`
- **Object detection + image segmentation** (PennFudanPed, optional): `notebooks/11_vision_detection_segmentation_pennfudan.ipynb`

Vision dependencies (torch/torchvision) are included in the default install.

## Configuration

Environment variables (prefix: `CLEANLAB_DEMO_`):

| Variable | Description | Default |
|----------|-------------|---------|
| `CLEANLAB_DEMO_DATA_DIR` | Directory for dataset cache | `data` |
| `CLEANLAB_DEMO_ARTIFACTS_DIR` | Directory for results/artifacts | `artifacts` |
| `CLEANLAB_DEMO_LOG_LEVEL` | Logging level | `INFO` |
| `CLEANLAB_DEMO_OPENAI_API_KEY` | OpenAI key (propagated to `OPENAI_API_KEY`) | - |
| `CLEANLAB_DEMO_ANTHROPIC_API_KEY` | Anthropic key (propagated to `ANTHROPIC_API_KEY`) | - |

The project also loads a project-root `.env` file (ignored by git) if present.
Use `.env.example` as a template.

## AI Report (optional)

Generate AI-powered analysis reports using an LLM:

```bash
# Set API key
export OPENAI_API_KEY="..."
# or
export ANTHROPIC_API_KEY="..."

# or using the app prefix (also supports `.env`)
export CLEANLAB_DEMO_OPENAI_API_KEY="..."

# Run experiment and generate report
cleanlab-demo run -d adult_income -o artifacts/last_result.json
cleanlab-demo ai-report

# Use deterministic report (no LLM)
cleanlab-demo ai-report --no-ai
```

Or use the notebook: `notebooks/03_pydantic_ai_report.ipynb`

Additional notebooks:
- `notebooks/01_quickstart_cleanlab.ipynb`
- `notebooks/02_model_sweep.ipynb`
- `notebooks/04_regression_outliers.ipynb`
- `notebooks/05_multiclass_classification_covtype.ipynb`
- `notebooks/06_multilabel_classification_emotions.ipynb`
- `notebooks/07_token_classification_ud_pos.ipynb`
- `notebooks/08_regression_cleanlearning_bike_sharing.ipynb`
- `notebooks/09_multiannotator_active_learning_movielens_100k.ipynb`
- `notebooks/10_outlier_detection_datalab_california_housing.ipynb`
- `notebooks/11_vision_detection_segmentation_pennfudan.ipynb` (requires torch/torchvision)

## Project Structure

```
cleanlab_demo/
├── src/cleanlab_demo/
│   ├── ai/          # AI report generation
│   ├── data/        # Dataset loading and schemas
│   ├── tasks/       # Dedicated task implementations (multi-label, token, vision, etc.)
│   ├── experiments/ # Experiment runner and sweeps
│   ├── features/    # Feature preprocessing
│   ├── models/      # Model factory
│   ├── ui/          # Streamlit app
│   └── utils/       # Download and filesystem utilities
├── notebooks/       # Jupyter tutorials
├── tests/           # Test suite
└── docker-compose.yml
```

## Mathematical Background (high-level)

Cleanlab is model-agnostic: it only needs your dataset’s labels and *out-of-sample* model predictions (e.g., via cross-validation).

**Classification (confident learning)**

- Let noisy labels be `\tilde{y}` and unknown true labels be `y*`.
- Train a probabilistic classifier that outputs `\hat{p}(y=k|x)` and obtain **out-of-sample** predicted probabilities for each training example.
- A common label-quality score is **self-confidence**:
  - `s_i = \hat{p}(\tilde{y}_i | x_i)` (lower means the given label looks less plausible under the model).
- Confident learning estimates the (noisy label, true label) joint distribution via a **confident joint** `\hat{C}`:
  - Intuition: count examples whose predicted class is confidently `j` (above a class-dependent threshold) while their given label is `i`.
  - From `\hat{C}`, estimate noise rates like `P(\tilde{y}=i | y*=j)` and prune/rank likely label issues.

**Multi-label**

- Each example has a *set* of labels. Cleanlab scores label quality using the per-class predicted probabilities and aggregates scores over the label set (e.g., based on self-confidence).
- The demo injects missing/extra tags and shows how pruning low-quality examples can improve F1.

**Token classification**

- Treat each token as a classification decision with its own predicted probability vector.
- Cleanlab computes token-level label quality (e.g., self-confidence) and can aggregate to sentence-level scores for finding problematic sequences.

**Regression**

- Cleanlab’s regression `CleanLearning` identifies label issues using residuals (difference between observed label and model prediction) together with uncertainty estimates (via CV + bootstrapping).
- Intuition: labels that are extreme outliers relative to model predictions/uncertainty are more likely incorrect.

**Datalab (outliers, near-duplicates, non-iid)**

- Many issue types are computed from distances/similarities in feature space (or embeddings), often via kNN.
- Example: outliers have unusually large kNN distances; near-duplicates have unusually small distances.

**Object detection**

- Uses model-predicted boxes + confidences and overlaps (IoU) with annotated boxes.
- Flags images where boxes look swapped (wrong class), poorly located, or overlooked (missing) given the model’s confident predictions.

**Segmentation**

- Uses per-pixel predicted probabilities `\hat{p}(y=k|x_pixel)` and integer masks `(H,W)`.
- Finds pixels likely mislabeled and aggregates per-image scores (e.g., soft-min over pixel label quality).

**Multi-annotator + active learning**

- For label matrices `labels[example, annotator]`, Cleanlab estimates consensus labels and per-example label quality while accounting for annotator reliability.
- Active learning scores prioritize which examples are most informative to (re)label next (low confidence/disagreement).

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests with coverage
pytest --cov=cleanlab_demo --cov-report=term-missing

# Type checking
mypy src/cleanlab_demo

# Linting
ruff check src/
ruff format src/
```

## License

MIT
