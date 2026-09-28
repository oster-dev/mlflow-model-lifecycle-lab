# MLflow Model Lifecycle Lab — Architecture


This document describes the architecture, component responsibilities, and design decisions for the MLflow Model Lifecycle Lab.


## Goals


The project has three primary goals:


1. **Learn and demonstrate MLflow** for experiment tracking, model logging, and model registry.
2. **Learn and demonstrate Metaflow** for workflow orchestration, step-level execution, and quality-gate routing.
3. **Implement a minimal but complete model lifecycle** with an explicit promotion policy:
   - Train candidate
   - Evaluate metrics
   - Apply a quality-gate decision
   - Register only accepted candidates in the MLflow Model Registry


The repository is intentionally local-first: SQLite backend, local artifacts, and Metaflow local mode. This keeps the focus on concepts and contracts rather than infrastructure complexity.


## Components


### `src/mlflow_lab/training.py`


Core training and evaluation logic:


- `TrainingResult` dataclass:
  - `run_id`, `accuracy`, `f1`, `roc_auc`
- `load_training_data(test_size, random_state)`:
  - Loads the scikit-learn breast cancer dataset.
  - Performs a stratified train/test split with a fixed `random_state`.
- `train_and_log(...)`:
  - Creates or reuses an MLflow experiment.
  - Trains a `StandardScaler → LogisticRegression` pipeline.
  - Logs parameters, metrics, tags, model signature, and input example.
  - Registers the model under a given name in the MLflow Model Registry.
- `decide_quality_gate(roc_auc, threshold)`:
  - Pure, testable function implementing the promotion policy.
  - Returns `"accepted"` if `roc_auc >= threshold`, else `"rejected"`.
  - Raises `ValueError` for inputs outside `[0.0, 1.0]`.


This module has no direct Metaflow dependency. It can be used for direct MLflow training runs (e.g. `scripts/train.py`) or as a library inside Metaflow steps.


### `flows/engagement_training_flow.py`


Metaflow workflow that orchestrates the end-to-end training and quality-gate process:


- Parameters:
  - `c`, `max_iter`: LogisticRegression hyperparameters.
  - `test_size`, `random_state`: Data split configuration.
  - `roc_auc_threshold`: Quality-gate threshold.
  - `experiment_name`, `model_name`: MLflow experiment and registry names.
- Steps:
  - `start`:
    - Reads `MLFLOW_TRACKING_URI` from the environment.
    - Logs configuration.
  - `load_data`:
    - Calls `load_training_data()` from `mlflow_lab.training`.
    - Stores train/test splits and dataset metadata as artifacts.
  - `train_candidate`:
    - Trains the `StandardScaler → LogisticRegression` pipeline.
  - `evaluate`:
    - Computes `accuracy`, `f1`, and `roc_auc`.
    - Calls `decide_quality_gate()` to determine the promotion decision.
  - `quality_gate`:
    - Routes the flow to `accepted` or `rejected` based on `quality_gate_decision`.
  - `accepted`:
    - Logs parameters, metrics, and tags to MLflow.
    - Registers the model in the MLflow Model Registry.
    - Tags the run with Metaflow flow name and run ID for cross-tool lineage.
  - `rejected`:
    - Records a rejection reason.
    - Does not create an MLflow run or register a model version.
  - `end`:
    - Summarizes the run outcome.


The flow enforces a clear separation: Metaflow controls execution and policy routing; MLflow records and registers accepted candidates.


### `scripts/train.py`


CLI for direct MLflow training runs without Metaflow:


- Accepts command-line arguments for experiment name, model name, and hyperparameters.
- Calls `train_and_log()` from `mlflow_lab.training`.
- Prints the resulting `TrainingResult`.


This script is useful for quick experiments and for comparing candidates directly in the MLflow UI.


### `scripts/predict.py`


CLI for local inference using a registered MLflow model:


- Loads a specific model version from the MLflow Model Registry, e.g.:
  ```python
  model_uri = "models:/engagement-classifier/3"
  ```
- Constructs a small input example.
- Runs inference and prints prediction and probability.


The loaded artifact includes the full sklearn pipeline (scaler + classifier), so preprocessing is applied automatically.


### `tests/test_training.py`


Test suite for deterministic behavior and quality-gate policy:


- Tests for deterministic data splitting:
  - Same `random_state` and `test_size` produce identical splits.
  - Split sizes are stable across runs.
- Tests for `decide_quality_gate`:
  - `roc_auc == threshold` → `"accepted"`
  - `roc_auc > threshold` → `"accepted"`
  - `roc_auc < threshold` → `"rejected"`
  - Invalid inputs (outside `[0.0, 1.0]`) → `ValueError`


These tests ensure that the promotion policy is explicit, testable, and stable.


## Data Flow


```text
Breast Cancer Dataset (scikit-learn)
→ Deterministic stratified split (random_state=42, test_size=0.2)
→ sklearn Pipeline (StandardScaler → LogisticRegression)
→ Metaflow EngagementTrainingFlow
   → train_candidate
   → evaluate (accuracy, f1, roc_auc)
   → quality_gate (decide_quality_gate)
      → accepted → MLflow tracking + model registration
      → rejected → rejection reason recorded; no registration
→ MLflow Model Registry (engagement-classifier v1, v2, v3, v4, ...)
→ scripts/predict.py loads models:/engagement-classifier/<version>
→ Local inference with preprocessed input
```


## MLflow Backend


- **Tracking Server:**
  - Runs locally via `mlflow server`.
  - Backend store: `sqlite:///mlflow.db`.
  - Default artifact root: `./mlartifacts`.
- **UI:**
  - Accessible at `http://127.0.0.1:8080`.
  - Shows experiments, runs, metrics, parameters, tags, and model versions.
- **Model Registry:**
  - Tracks versions of `engagement-classifier`.
  - Each version is linked to a specific MLflow run.
  - Accepted Metaflow runs include additional tags for Metaflow flow and run ID.


## Metaflow Execution


- **Mode:** Local execution with a local datastore (`.metaflow/`).
- **Graph validation:**
  - The flow graph can be validated without running:
    ```bash
    python flows/engagement_training_flow.py show
    ```
- **Artifacts:**
  - Metaflow stores step artifacts and run metadata in `.metaflow/`.
  - This directory is excluded from Git to avoid committing runtime state.


## Quality Gate Contract


The promotion decision is implemented as a pure function:


```python
def decide_quality_gate(roc_auc: float, threshold: float) -> str:
    if not 0.0 <= roc_auc <= 1.0:
        raise ValueError("roc_auc must be between 0.0 and 1.0")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")

    return "accepted" if roc_auc >= threshold else "rejected"
```


This function:

- Has no side effects.
- Does not depend on MLflow or Metaflow.
- Is fully covered by unit tests in `tests/test_training.py`.


The Metaflow flow calls this function in the `evaluate` step and routes based on its result.


## Lineage


The project maintains bidirectional lineage between Metaflow and MLflow:


- **MLflow → Metaflow:**
  - Accepted runs include tags:
    - `orchestrator = metaflow`
    - `metaflow_flow = EngagementTrainingFlow`
    - `metaflow_run_id = <run ID>`
    - `decision = accepted`
- **Metaflow → MLflow:**
  - The `accepted` step logs the MLflow run ID and model URI as artifacts.
  - The `rejected` step records the rejection reason.


This makes it possible to trace every registered model version back to:

- A specific MLflow run.
- A specific Metaflow flow execution.
- A specific code version (Git commit).


## Design Decisions


### Local-first development


- SQLite and local artifacts keep the setup simple and inspectable.
- No cloud dependencies for learning and portfolio purposes.
- Easy to extend later with cloud storage, managed databases, and remote compute.


### Single metric threshold


- The quality gate currently uses only ROC-AUC.
- This keeps the policy simple and easy to understand.
- Future extensions could add multi-metric rules (e.g. minimum F1, maximum calibration error).


### Pipeline serialization


- The sklearn pipeline (scaler + classifier) is logged as a single artifact.
- This prevents training-serving skew: inference uses the same preprocessing as training.
- The model signature and input example document the expected input format.


### Explicit promotion policy


- Model promotion is not automatic.
- The quality gate enforces an explicit rule before registry entry.
- Rejected candidates complete their workflow but do not create registry versions.


## Current Limitations


- No CI pipeline yet (formatting, linting, tests, flow validation).
- No remote authentication or RBAC for the MLflow server.
- No automated deployment or staging policy.
- No model monitoring, drift detection, or rollback logic.
- Dataset is a small built-in scikit-learn dataset for fast iteration.


These limitations are intentional for this learning lab. They define clear next steps for future iterations.


## Next Steps


Potential extensions:


- Add GitHub Actions CI for:
  - Formatting and linting.
  - Unit and integration tests.
  - Metaflow graph validation.
- Extend the promotion policy to multiple metrics and explicit approval rules.
- Replace local storage with cloud object storage and a managed metadata backend.
- Run Metaflow with remote compute and scheduled execution.
- Add deployment automation, model monitoring, drift detection, and rollback policy.