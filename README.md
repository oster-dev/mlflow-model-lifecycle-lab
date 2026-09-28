# MLflow Model Lifecycle Lab


A reproducible local lab for learning and demonstrating a complete ML model lifecycle with **MLflow** and **Metaflow**.


```text
Train candidate
→ Track parameters, metrics, tags, and artifacts
→ Orchestrate a reproducible workflow
→ Evaluate quality
→ Route through an accepted/rejected quality gate
→ Register only accepted model versions
→ Load a registered model for local inference
```


This repository is part of my public roadmap toward Data & Feature Infrastructure and ML Platform Engineering.


## What this project demonstrates


- Local MLflow Tracking Server backed by SQLite.
- Reproducible scikit-learn training runs.
- Parameter, metric, tag, model artifact, signature, and input-example logging.
- MLflow Model Registry with versioned models and run lineage.
- Candidate model comparison in the MLflow UI.
- Registry-based model loading and local inference.
- A serialized `StandardScaler → LogisticRegression` pipeline to prevent training-serving skew.
- Metaflow workflow orchestration with validated DAGs, step-level execution, and local artifacts.
- A quality-gated Metaflow switch that accepts or rejects a candidate before MLflow Registry promotion.
- Cross-tool lineage: accepted MLflow runs are tagged with Metaflow flow and run IDs.
- Unit-tested quality-gate policy (`decide_quality_gate`) with explicit acceptance and rejection rules.
- Basic automated tests for deterministic data splitting and quality-gate decisions.


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
                         │ sklearn Pipeline          │
                         │ StandardScaler            │
                         │ → LogisticRegression      │
                         └────────────┬─────────────┘
                                      │
                                      ▼
               ┌───────────────────────────────────────────┐
               │ Metaflow EngagementTrainingFlow            │
               │ train → evaluate → quality-gate switch     │
               └───────────────┬───────────────────┬───────┘
                               │ accepted          │ rejected
                               ▼                   ▼
             ┌─────────────────────────┐   ┌──────────────────────┐
             │ MLflow Tracking Server  │   │ No registry promotion │
             │ SQLite + local artifacts│   │ Rejection is recorded │
             └───────────┬─────────────┘   └──────────────────────┘
                         │
                         ▼
             ┌─────────────────────────┐
             │ MLflow Model Registry   │
             │ engagement-classifier   │
             │ v1 → v2 → v3 → v4       │
             └───────────┬─────────────┘
                         │
                         ▼
             ┌─────────────────────────┐
             │ Registry-based Inference│
             │ models:/.../<version>   │
             └─────────────────────────┘
```


See [ARCHITECTURE.md](ARCHITECTURE.md) for component responsibilities, workflow artifacts, decision records, and design decisions.


## MLflow and Metaflow


The two tools have complementary responsibilities:


| Tool | Primary responsibility | Evidence it preserves |
|---|---|---|
| Metaflow | Workflow orchestration | Step graph, step execution, local artifacts, quality-gate routing, workflow decision |
| MLflow | Experiment and model lifecycle tracking | Parameters, metrics, tags, model artifacts, signatures, input examples, Registry versions |


```text
Metaflow decides which path executes.
MLflow records and registers the accepted candidate.
```


A rejected candidate completes its Metaflow workflow, including a stored rejection reason, but does not create an MLflow model-registration run or a new Registry version.


## Repository structure


```text
.
├── flows/
│   ├── hello_flow.py                # Minimal Metaflow local-mode sanity flow
│   └── engagement_training_flow.py  # Orchestrated training and quality-gate workflow
├── scripts/
│   ├── train.py                     # CLI for a direct tracked MLflow training run
│   └── predict.py                   # Load a Registry version and infer
├── src/
│   └── mlflow_lab/
│       ├── __init__.py
│       └── training.py              # Data split, training, metrics, direct MLflow logging,
│                                    # and testable quality-gate policy (decide_quality_gate)
├── tests/
│   └── test_training.py             # Deterministic split tests and quality-gate policy tests
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
pip install metaflow
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


Keep this terminal running while using the training, inference, or Metaflow commands below.


## Direct MLflow training


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


1. Creates or reuses the `engagement-classifier` experiment.
2. Performs a deterministic stratified train/test split.
3. Trains a `StandardScaler → LogisticRegression` pipeline.
4. Logs parameters, metrics, tags, model signature, and input example.
5. Stores the pipeline as a model artifact.
6. Registers a new version under `engagement-classifier`.


## Compare MLflow candidates


In the MLflow UI:


1. Open the `engagement-classifier` experiment.
2. Select two or more runs.
3. Click **Compare**.
4. Compare `params.c`, `metrics.accuracy`, `metrics.f1`, and `metrics.roc_auc`.
5. Open **Models** or **Model registry** to inspect version-to-run lineage.


Example local result:


| Version | Pipeline | Accuracy | F1 | ROC-AUC | Observation |
|---:|---|---:|---:|---:|---|
| v1 | Logistic Regression, `C=0.1` | 0.9474 | 0.9583 | 0.9937 | Convergence warning observed |
| v2 | Logistic Regression, `C=1.0` | 0.9561 | 0.9655 | 0.9954 | Better metrics, warning remained |
| v3 | `StandardScaler → LogisticRegression`, `C=1.0` | 0.9825 | 0.9861 | 0.9954 | Warning resolved |
| v4 | Metaflow-orchestrated accepted candidate | 0.9825 | 0.9861 | 0.9954 | Passed quality gate and registered |


The exact values may vary if you change the dataset, split configuration, library versions, or model parameters.


## Metaflow orchestration


The Metaflow workflow adds an explicit policy boundary between model evaluation and Registry promotion:


```text
start
→ load_data
→ train_candidate
→ evaluate
→ quality_gate
├── accepted → MLflow tracking and model registration
└── rejected → rejection reason recorded; no registration
→ end
```


Inspect the validated graph before executing it:


```bash
python flows/engagement_training_flow.py show
```


### Accepted candidate


Run a candidate that should satisfy the quality policy:


```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:8080


python flows/engagement_training_flow.py run \
  --c 1.0 \
  --max-iter 1000 \
  --roc_auc_threshold 0.98
```


With the documented local configuration, the candidate produced ROC-AUC `0.9954`, passed the `0.9800` threshold, and created `engagement-classifier v4` in the MLflow Registry.


The accepted MLflow run includes tags such as:


```text
orchestrator       = metaflow
metaflow_flow      = EngagementTrainingFlow
metaflow_run_id    = <Metaflow run ID>
decision           = accepted
```


### Rejected candidate


Run the same candidate against an intentionally unreachable threshold:


```bash
python flows/engagement_training_flow.py run \
  --c 1.0 \
  --max-iter 1000 \
  --roc_auc_threshold 0.999
```


Expected behavior:


```text
ROC-AUC 0.9954 < threshold 0.9990
→ decision = rejected
→ no MLflow model-registration run
→ no new Registry version
```


This proves that the quality gate controls promotion rather than merely printing a status message.


### Metaflow local artifacts


Metaflow stores local run metadata and step artifacts in `.metaflow/`. This directory is intentionally excluded from Git. It lets you inspect local workflow runs without committing runtime state to the repository.


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
git status
```


Expected baseline after completing the quality-gate tests:


```text
All checks passed!
7 passed
nothing to commit, working tree clean
```


The test suite covers:

- Deterministic stratified train/test split behavior.
- Quality-gate policy:
  - `roc_auc == threshold` → `"accepted"`
  - `roc_auc > threshold` → `"accepted"`
  - `roc_auc < threshold` → `"rejected"`
  - Invalid inputs (outside `[0.0, 1.0]`) → `ValueError`


## Design principles


- **Reproducibility:** Fixed `random_state`, deterministic stratified splitting, tracked configuration, and versioned artifacts.
- **Workflow control:** Explicit steps and a declarative Metaflow switch make quality decisions inspectable and repeatable.
- **Lineage:** Every registered model version is traceable to a specific MLflow run; accepted MLflow runs link back to their Metaflow flow and run IDs.
- **No training-serving skew:** The scaler and classifier are serialized as one sklearn pipeline.
- **Explicit contracts:** Input example and model signature document the expected model input.
- **Local-first:** SQLite, local MLflow artifacts, and the Metaflow local datastore provide a low-cost, inspectable development environment before cloud deployment.
- **Promotion by policy:** Only candidates that meet a declared threshold are allowed to enter the Model Registry.
- **Testable policy:** The quality-gate decision is a pure, unit-testable function (`decide_quality_gate`) isolated from Metaflow and MLflow.


## Current scope and limits


This is intentionally a local learning and portfolio lab, not a production deployment.


- The MLflow backend uses local SQLite rather than a managed database.
- MLflow artifacts are stored locally rather than in object storage.
- Metaflow runs are started manually and use a local datastore rather than scheduled remote execution.
- There is no remote authentication, RBAC, CI pipeline, or automated deployment policy yet.
- The quality policy is a single ROC-AUC threshold rather than a multi-metric approval framework.
- The dataset is a built-in scikit-learn dataset for fast and deterministic iteration.


## Next steps


- Add CI for formatting, linting, tests, flow graph validation, and reproducible workflow validation.
- Add a formal candidate-promotion policy using multiple metrics and explicit approval rules.
- Replace local storage with cloud object storage and a managed metadata backend in a future deployment profile.
- Run Metaflow with remote compute, scheduled execution, and durable shared artifact storage.
- Add deployment automation, model monitoring, drift detection, and rollback policy.


## License


MIT