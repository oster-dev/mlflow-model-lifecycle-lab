from __future__ import annotations

import os

from metaflow import FlowSpec, Parameter, current, step


class EngagementTrainingFlow(FlowSpec):
    c = Parameter(
        "c",
        default=1.0,
        type=float,
        help="Inverse regularization strength for LogisticRegression.",
    )
    max_iter = Parameter(
        "max_iter",
        default=1000,
        type=int,
        help="Maximum LogisticRegression training iterations.",
    )
    test_size = Parameter(
        "test_size",
        default=0.2,
        type=float,
        help="Fraction of data reserved for evaluation.",
    )
    random_state = Parameter(
        "random_state",
        default=42,
        type=int,
        help="Random seed for the deterministic stratified split.",
    )
    roc_auc_threshold = Parameter(
        "roc_auc_threshold",
        default=0.98,
        type=float,
        help="Minimum ROC-AUC required to register a candidate model.",
    )
    experiment_name = Parameter(
        "experiment_name",
        default="engagement-classifier",
        type=str,
        help="MLflow experiment name.",
    )
    model_name = Parameter(
        "model_name",
        default="engagement-classifier",
        type=str,
        help="MLflow registered model name.",
    )

    @step
    def start(self):
        self.tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:8080",
        )

        print("=== EngagementTrainingFlow started ===")
        print(f"MLflow Tracking URI: {self.tracking_uri}")
        print(
            f"Configuration: c={self.c}, max_iter={self.max_iter}, "
            f"test_size={self.test_size}, random_state={self.random_state}, "
            f"roc_auc_threshold={self.roc_auc_threshold}"
        )

        self.next(self.load_data)

    @step
    def load_data(self):
        from mlflow_lab.training import load_training_data

        self.x_train, self.x_test, self.y_train, self.y_test = load_training_data(
            test_size=self.test_size,
            random_state=self.random_state,
        )

        self.train_rows = len(self.x_train)
        self.test_rows = len(self.x_test)
        self.feature_count = self.x_train.shape[1]

        print(
            f"Loaded deterministic split: train_rows={self.train_rows}, "
            f"test_rows={self.test_rows}, features={self.feature_count}"
        )

        self.next(self.train_candidate)

    @step
    def train_candidate(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=self.c,
                        max_iter=self.max_iter,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        self.pipeline.fit(self.x_train, self.y_train)

        print("Candidate pipeline trained: StandardScaler → LogisticRegression")

        self.next(self.evaluate)

    @step
    def evaluate(self):
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        predictions = self.pipeline.predict(self.x_test)
        probabilities = self.pipeline.predict_proba(self.x_test)[:, 1]

        self.accuracy = accuracy_score(self.y_test, predictions)
        self.f1 = f1_score(self.y_test, predictions)
        self.roc_auc = roc_auc_score(self.y_test, probabilities)

        self.passed_quality_gate = self.roc_auc >= self.roc_auc_threshold
        self.quality_gate_decision = "accepted" if self.passed_quality_gate else "rejected"
        comparison = ">=" if self.passed_quality_gate else "<"

        print(
            f"Evaluation: accuracy={self.accuracy:.4f}, "
            f"f1={self.f1:.4f}, roc_auc={self.roc_auc:.4f}"
        )
        print(
            f"Quality gate: roc_auc={self.roc_auc:.4f} "
            f"{comparison} threshold={self.roc_auc_threshold:.4f} "
            f"→ decision={self.quality_gate_decision}"
        )

        self.next(self.quality_gate)

    @step
    def quality_gate(self):
        print(f"Routing candidate through quality gate: decision={self.quality_gate_decision}")

        self.next(
            {
                "accepted": self.accepted,
                "rejected": self.rejected,
            },
            condition="quality_gate_decision",
        )

    @step
    def accepted(self):
        import mlflow
        import mlflow.sklearn
        from mlflow.models import infer_signature

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run() as run:
            mlflow.set_tags(
                {
                    "project": "mlflow-model-lifecycle-lab",
                    "orchestrator": "metaflow",
                    "pipeline": "standard_scaler_logistic_regression",
                    "decision": "accepted",
                    "metaflow_flow": current.flow_name,
                    "metaflow_run_id": current.run_id,
                }
            )
            mlflow.log_params(
                {
                    "model_type": "standard_scaler_logistic_regression",
                    "c": self.c,
                    "max_iter": self.max_iter,
                    "test_size": self.test_size,
                    "random_state": self.random_state,
                    "roc_auc_threshold": self.roc_auc_threshold,
                }
            )
            mlflow.log_metrics(
                {
                    "accuracy": self.accuracy,
                    "f1": self.f1,
                    "roc_auc": self.roc_auc,
                }
            )

            signature = infer_signature(
                self.x_train,
                self.pipeline.predict(self.x_train),
            )

            model_info = mlflow.sklearn.log_model(
                sk_model=self.pipeline,
                name="model",
                registered_model_name=self.model_name,
                input_example=self.x_train.head(3),
                signature=signature,
            )

            self.mlflow_run_id = run.info.run_id
            self.mlflow_model_uri = model_info.model_uri
            self.decision = "accepted"
            self.rejection_reason = None

        print("Candidate accepted and registered in MLflow.")
        print(f"MLflow run ID: {self.mlflow_run_id}")
        print(f"MLflow model URI: {self.mlflow_model_uri}")

        self.next(self.end)

    @step
    def rejected(self):
        self.decision = "rejected"
        self.mlflow_run_id = None
        self.mlflow_model_uri = None
        self.rejection_reason = (
            f"roc_auc={self.roc_auc:.4f} is below roc_auc_threshold={self.roc_auc_threshold:.4f}"
        )

        print(f"Candidate rejected: {self.rejection_reason}")
        print("No MLflow model version was registered.")

        self.next(self.end)

    @step
    def end(self):
        print("=== EngagementTrainingFlow finished ===")
        print(f"Decision: {self.decision}")
        print(f"Quality gate passed: {self.passed_quality_gate}")
        print(f"Training rows: {self.train_rows}")
        print(f"Evaluation rows: {self.test_rows}")
        print(f"Feature count: {self.feature_count}")
        print(f"Accuracy: {self.accuracy:.4f}")
        print(f"F1: {self.f1:.4f}")
        print(f"ROC-AUC: {self.roc_auc:.4f}")

        if self.decision == "accepted":
            print(f"MLflow run ID: {self.mlflow_run_id}")
            print(f"MLflow model URI: {self.mlflow_model_uri}")
        else:
            print(f"Rejection reason: {self.rejection_reason}")


if __name__ == "__main__":
    EngagementTrainingFlow()
