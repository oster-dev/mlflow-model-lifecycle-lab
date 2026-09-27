from __future__ import annotations

from dataclasses import dataclass

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TrainingResult:
    run_id: str
    accuracy: float
    f1: float
    roc_auc: float


def load_training_data(test_size: float, random_state: int):
    dataset = load_breast_cancer(as_frame=True)
    return train_test_split(
        dataset.data,
        dataset.target,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.target,
    )


def train_and_log(
    *,
    experiment_name: str,
    model_name: str,
    c: float,
    max_iter: int,
    test_size: float,
    random_state: int,
) -> TrainingResult:
    mlflow.set_experiment(experiment_name)

    x_train, x_test, y_train, y_test = load_training_data(
        test_size=test_size,
        random_state=random_state,
    )

    with mlflow.start_run() as run:
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=c,
                        max_iter=max_iter,
                        random_state=random_state,
                    ),
                ),
            ]
        )

        mlflow.set_tags(
            {
                "project": "mlflow-model-lifecycle-lab",
                "stage": "local-development",
                "framework": "scikit-learn",
                "pipeline": "standard_scaler_logistic_regression",
            }
        )
        mlflow.log_params(
            {
                "model_type": "standard_scaler_logistic_regression",
                "c": c,
                "max_iter": max_iter,
                "test_size": test_size,
                "random_state": random_state,
            }
        )

        pipeline.fit(x_train, y_train)

        predictions = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "f1": f1_score(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }
        mlflow.log_metrics(metrics)

        signature = infer_signature(
            x_train,
            pipeline.predict(x_train),
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name=model_name,
            input_example=x_train.head(3),
            signature=signature,
        )

        mlflow.set_tag("model_uri", model_info.model_uri)

        return TrainingResult(
            run_id=run.info.run_id,
            accuracy=metrics["accuracy"],
            f1=metrics["f1"],
            roc_auc=metrics["roc_auc"],
        )
