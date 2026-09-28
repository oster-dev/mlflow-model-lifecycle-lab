from __future__ import annotations

import pytest

from mlflow_lab.training import decide_quality_gate


def test_quality_gate_accepts_metric_at_threshold() -> None:
    assert decide_quality_gate(roc_auc=0.98, threshold=0.98) == "accepted"


def test_quality_gate_accepts_metric_above_threshold() -> None:
    assert decide_quality_gate(roc_auc=0.9954, threshold=0.98) == "accepted"


def test_quality_gate_rejects_metric_below_threshold() -> None:
    assert decide_quality_gate(roc_auc=0.9954, threshold=0.999) == "rejected"


@pytest.mark.parametrize(
    ("roc_auc", "threshold"),
    [
        (-0.01, 0.98),
        (1.01, 0.98),
        (0.98, -0.01),
        (0.98, 1.01),
    ],
)
def test_quality_gate_rejects_invalid_values(
    roc_auc: float,
    threshold: float,
) -> None:
    with pytest.raises(ValueError):
        decide_quality_gate(roc_auc=roc_auc, threshold=threshold)
