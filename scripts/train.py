from __future__ import annotations

import argparse

from mlflow_lab.training import train_and_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="engagement-classifier")
    parser.add_argument("--model-name", default="engagement-classifier")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_and_log(
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        c=args.c,
        max_iter=args.max_iter,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"Run ID:   {result.run_id}")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"F1:       {result.f1:.4f}")
    print(f"ROC AUC:  {result.roc_auc:.4f}")


if __name__ == "__main__":
    main()
