# Cleanlab Demo (end-to-end)

A comprehensive ML project demonstrating [Cleanlab](https://github.com/cleanlab/cleanlab) for automatic label issue detection. This project downloads real-world datasets, trains multiple ML models, applies Cleanlab to find mislabeled data, and provides both a CLI and web UI.

## Features

- 📊 **Multiple Datasets**: UCI Adult Income (classification), UCI Bike Sharing (regression), California Housing (regression; natural outliers)
- 🤖 **Model Support**: Logistic Regression, k-NN, Random Forest, ExtraTrees, Histogram Gradient Boosting, Ridge
- 🔍 **Cleanlab Integration**: Label issues + Datalab (outliers/near-duplicates/non-iid) + optional prune & retrain comparison
- 📈 **Model Sweeps**: Compare multiple models on the same dataset
- 🎛️ **Streamlit UI**: Side-by-side comparison of baseline vs Cleanlab-enabled variants
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

## Project Structure

```
cleanlab_demo/
├── src/cleanlab_demo/
│   ├── ai/          # AI report generation
│   ├── data/        # Dataset loading and schemas
│   ├── experiments/ # Experiment runner and sweeps
│   ├── features/    # Feature preprocessing
│   ├── models/      # Model factory
│   ├── ui/          # Streamlit app
│   └── utils/       # Download and filesystem utilities
├── notebooks/       # Jupyter tutorials
├── tests/           # Test suite
└── docker-compose.yml
```

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
