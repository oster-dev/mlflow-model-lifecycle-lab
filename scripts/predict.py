from __future__ import annotations

import mlflow.sklearn
from sklearn.datasets import load_breast_cancer


def main() -> None:
    model_uri = "models:/engagement-classifier/3"
    model = mlflow.sklearn.load_model(model_uri)

    dataset = load_breast_cancer(as_frame=True)
    sample = dataset.data.head(1)

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0, 1]

    print(f"Model URI:   {model_uri}")
    print(f"Prediction:  {prediction}")
    print(f"Probability: {probability:.4f}")


if __name__ == "__main__":
    main()
