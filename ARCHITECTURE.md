# Architecture

## Purpose

`mlflow-model-lifecycle-lab` demonstrates a compact but complete local ML model lifecycle:

```text
deterministic data split
→ training pipeline
→ tracked experiment run
→ logged metrics and model artifact
→ registered model version
→ registry-based inference
```

The repository deliberately focuses on lifecycle traceability rather than model complexity.

## System context

```text
Developer
│
├── starts local MLflow server
│   ├── SQLite backend store
│   └── local artifact store
│
├── runs scripts/train.py
│   ├── loads deterministic sample data
│   ├── trains a sklearn pipeline
│   ├── logs run metadata to MLflow
│   └── registers a model version
│
└── runs scripts/predict.py
    ├── resolves an explicit Registry URI
    ├── loads the serialized pipeline
    └── runs inference on raw feature inputs
```

## Components

| Component | Responsibility |
|---|---|
| `scripts/train.py` | CLI boundary for experiment and model configuration |
| `src/mlflow_lab/training.py` | Deterministic split, pipeline construction, training, evaluation, and MLflow logging |
| `StandardScaler` | Learns feature scaling during training and applies the same transformation at inference |
| `LogisticRegression` | Binary classification estimator |
| `sklearn.pipeline.Pipeline` | Serializes preprocessing and classification as one artifact |
| MLflow Tracking Server | Receives experiment metadata, parameters, metrics, tags, artifacts, and model registrations |
| SQLite (`mlflow.db`) | Local metadata backend for MLflow experiments, runs, and Registry state |
| Local artifact store (`mlartifacts/`) | Stores serialized model artifacts and MLflow model metadata |
| MLflow Model Registry | Maintains `engagement-classifier` versions and links each version to its source run |
| `scripts/predict.py` | Loads a specific Registry version and executes local inference |
| `tests/test_training.py` | Verifies deterministic data split behavior |

## Training flow

```text
1. CLI arguments
   └── experiment name, model name, C, max iterations, split seed

2. Data loading
   └── sklearn breast cancer dataset

3. Deterministic split
   └── train_test_split(..., stratify=target, random_state=42)

4. Pipeline training
   └── StandardScaler → LogisticRegression

5. Evaluation
   └── accuracy, F1, ROC-AUC

6. MLflow logging
   ├── parameters
   ├── metrics
   ├── tags
   ├── input example
   ├── input/output signature
   └── serialized pipeline artifact

7. Registry registration
   └── engagement-classifier version N
```

## Inference flow

```text
1. Construct an explicit Registry URI
   └── models:/engagement-classifier/3

2. Load the registered MLflow model
   └── mlflow.sklearn.load_model(...)

3. Provide raw tabular model features
   └── no separate manual scaling

4. Pipeline applies StandardScaler internally
   └── transform input using training-time scaling statistics

5. LogisticRegression returns prediction and probability
```

## Data and model contracts

The MLflow model is logged with:

- an **input example**: a small representative DataFrame sample;
- a **model signature**: inferred input columns/types and model output schema.

Together, they document what the registered model expects at inference time. This is a lightweight local form of a serving contract.

## Design decisions

### DD-001: Local MLflow server with SQLite

**Decision:** Use a local MLflow server backed by SQLite and a local artifact root.

**Why:** The setup is free, easy to inspect, persistent across terminal sessions, and sufficient for understanding the relationship between tracking, artifacts, and registry state.

**Trade-off:** It is not appropriate for shared, highly available, multi-user production use.

### DD-002: Pipeline instead of separate preprocessing

**Decision:** Serialize `StandardScaler` and `LogisticRegression` together as one sklearn `Pipeline`.

**Why:** The same preprocessing learned during training is automatically applied during inference. This removes a common source of training-serving skew.

**Trade-off:** Individual pipeline steps are less independently deployable, which is acceptable for this compact local lab.

### DD-003: Explicit model versions

**Decision:** Load explicit Registry versions such as `models:/engagement-classifier/3`.

**Why:** Explicit versions make the selected model deterministic and auditable. A caller can connect an inference result to a specific tracked run.

**Trade-off:** Promotion aliases or environment-aware deployment stages would be more convenient later, but they are intentionally out of scope for the initial local workflow.

### DD-004: Deterministic split

**Decision:** Use a stratified train/test split with a fixed random seed.

**Why:** Re-running an unchanged configuration produces comparable training and evaluation conditions.

**Trade-off:** A real production evaluation design would include cross-validation, time-aware splits where appropriate, drift checks, and a held-out test strategy.

## Model evolution evidence

| Version | Change | Accuracy | F1 | ROC-AUC | Observation |
|---:|---|---:|---:|---:|---|
| v1 | Logistic Regression, `C=0.1` | 0.9474 | 0.9583 | 0.9937 | Convergence warning observed |
| v2 | Logistic Regression, `C=1.0` | 0.9561 | 0.9655 | 0.9954 | Better metrics, warning remained |
| v3 | `StandardScaler → LogisticRegression`, `C=1.0` | 0.9825 | 0.9861 | 0.9954 | Warning resolved; pipeline artifact created |

The metrics above are local reference evidence for the documented configuration. They should not be interpreted as a general benchmark.

## Production evolution path

```text
Local lab
→ Metaflow workflow orchestration
→ quality gate and candidate promotion policy
→ CI validation
→ remote tracking server
→ object storage for artifacts
→ managed relational backend
→ access control and deployment environment separation
→ monitoring, drift detection, and rollback policy
```

## Non-goals

The current scope does not include:

- distributed model training;
- cloud deployment;
- real-time serving endpoint;
- automated production promotion;
- role-based access control;
- monitoring or drift detection;
- a production data source.

These are intentionally deferred to preserve a focused, reproducible lifecycle foundation.