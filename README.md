# MLflow Model Lifecycle Lab

A reproducible local lab for learning and demonstrating the core ML model lifecycle:

```text
Train
→ Track experiments
→ Log parameters, metrics, tags, and artifacts
→ Register versioned models
→ Compare candidate runs
→ Load a model from the registry
→ Run local inference
```

This repository is part of my public roadmap toward Data & Feature Infrastructure and ML Platform Engineering.

## What this project demonstrates

- Local MLflow Tracking Server backed by SQLite.
- Reproducible scikit-learn training runs.
- Parameter, metric, tag, and model artifact logging.
- MLflow Model Registry with versioned models and run lineage.
- Candidate model comparison in the MLflow UI.
- Registry-based model loading and inference.
- A serialized `StandardScaler → LogisticRegression` pipeline to prevent training-serving skew.
- Explicit MLflow model signatures and input examples.
- Basic automated tests for deterministic data splitting.

## Architecture

```text
                         ┌──────────────────────────┐
                         │ Breast Cancer Dataset     │
                         │ scikit-learn built-in     │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │ Deterministic Split       │
                         │ stratified, random_state  │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │ sklearn Pipeline         │
                         │ StandardScaler           │
                         │ → LogisticRegression     │
                         └────────────┬─────────────┘
                                      │
                                      ▼
             ┌────────────────────────────────────────────────┐
             │ MLflow Tracking Server — http://127.0.0.1:8080 │
             │ SQLite backend: mlflow.db                       │
             └───────┬──────────────────┬─────────────────────┘
                     │                  │
                     ▼                  ▼
       ┌──────────────────────┐   ┌──────────────────────────┐
       │ Experiment / Runs    │   │ Model Registry           │
       │ params, metrics,     │   │ engagement-classifier    │
       │ tags, artifacts      │   │ v1 → v2 → v3             │
       └──────────────────────┘   └────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌──────────────────────────┐
                                  │ Registry-based Inference │
                                  │ models:/.../3            │
                                  └──────────────────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for component responsibilities and design decisions.

## Repository structure

```text
.
├── scripts/
│   ├── train.py                    # CLI for a tracked training run
│   └── predict.py                  # Load a registry version and infer
├── src/
│   └── mlflow_lab/
│       ├── __init__.py
│       └── training.py              # Data split, training, metrics, MLflow logging
├── tests/
│   └── test_training.py             # Deterministic split tests
├── ARCHITECTURE.md
├── README.md
└── pyproject.toml
```

## Prerequisites

- Python 3.11 or later
- A virtual environment
- `pip`

## Installation

```bash
git clone [https://github.com/oster-dev/mlflow-model-lifecycle-lab.git](https://github.com/oster-dev/mlflow-model-lifecycle-lab.git)
cd mlflow-model-lifecycle-lab

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[dev]"
```

## Run MLflow locally

Start the MLflow Tracking Server in one terminal:

```bash
source .venv/bin/activate

mlflow server \
  --host 127.0.0.1 \
  --port 8080 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts
```

Open the local MLflow UI:

```text
http://127.0.0.1:8080
```

Keep this terminal running while using the training and prediction commands below.

## Train and register a model

Open a second terminal:

```bash
cd mlflow-model-lifecycle-lab
source .venv/bin/activate

export MLFLOW_TRACKING_URI=http://127.0.0.1:8080
```

Run a tracked candidate experiment:

```bash
python scripts/train.py \
  --experiment-name engagement-classifier \
  --model-name engagement-classifier \
  --c 1.0 \
  --max-iter 1000
```

The command:

1. creates or reuses the `engagement-classifier` experiment;
2. performs a deterministic stratified train/test split;
3. trains a `StandardScaler → LogisticRegression` pipeline;
4. logs parameters, metrics, tags, model signature, and input example;
5. stores the pipeline as a model artifact;
6. registers a new version under `engagement-classifier`.

## Compare candidates

In the MLflow UI:

1. Open the `engagement-classifier` experiment.
2. Select two or more runs.
3. Click **Compare**.
4. Compare `params.c`, `metrics.accuracy`, `metrics.f1`, and `metrics.roc_auc`.
5. Open **Models** or **Model registry** to inspect the version-to-run lineage.

Example local result:

| Version | Pipeline | Accuracy | F1 | ROC-AUC |
|---:|---|---:|---:|---:|
| v1 | Logistic Regression, `C=0.1` | 0.9474 | 0.9583 | 0.9937 |
| v2 | Logistic Regression, `C=1.0` | 0.9561 | 0.9655 | 0.9954 |
| v3 | `StandardScaler → LogisticRegression`, `C=1.0` | 0.9825 | 0.9861 | 0.9954 |

The exact values may vary if you change the dataset, split configuration, library versions, or model parameters.

## Load a registered model

`scripts/predict.py` loads an explicit Registry model version:

```python
model_uri = "models:/engagement-classifier/3"
```

Run local inference:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:8080
python scripts/predict.py
```

Expected output format:

```text
Model URI:   models:/engagement-classifier/3
Prediction:  0
Probability: 0.xxxx
```

The loaded artifact includes both preprocessing and classification, so inference receives raw feature inputs and does not separately scale them.

## Validation

```bash
ruff format --check .
ruff check .
pytest -q
git diff --check
```

Expected baseline:

```text
All checks passed!
2 passed
```

## Design principles

- **Reproducibility:** fixed `random_state`, deterministic stratified splitting, and tracked configuration.
- **Lineage:** every registry version is traceable to a specific MLflow run.
- **No training-serving skew:** the scaler and classifier are serialized as one sklearn pipeline.
- **Explicit contracts:** input example and model signature document the expected model input.
- **Local-first:** SQLite and local artifacts provide a low-cost, inspectable development environment before cloud deployment.

## Current scope and limits

This is intentionally a local learning and portfolio lab, not a production deployment.

- The MLflow backend uses local SQLite rather than a managed database.
- Artifacts are stored locally rather than in object storage.
- There is no remote authentication, RBAC, CI pipeline, or automated promotion policy yet.
- The dataset is a built-in scikit-learn dataset for fast and deterministic iteration.

## Next steps

- Add Metaflow orchestration around data loading, training, evaluation, and a quality gate.
- Add a quality threshold that routes candidate models to accepted or rejected workflow paths.
- Add CI for formatting, linting, tests, and reproducible workflow validation.
- Replace local storage with cloud object storage and a managed metadata backend in a future deployment profile.

## License

MIT