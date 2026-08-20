"""Numerically safe regression and calibrated-classification metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Iterable[str],
) -> pd.DataFrame:
    true = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    names = tuple(target_names)
    if true.shape != predicted.shape or true.ndim != 2:
        raise ValueError(
            f"Expected matching 2-D arrays, got {true.shape} and {predicted.shape}."
        )
    if true.shape[1] != len(names):
        raise ValueError("target_names length does not match target dimension.")
    if not np.isfinite(true).all() or not np.isfinite(predicted).all():
        raise ValueError("Regression metrics require finite values.")

    rows = []
    for index, name in enumerate(names):
        actual = true[:, index]
        estimate = predicted[:, index]
        q05, q95 = np.quantile(actual, [0.05, 0.95])
        scale = float(q95 - q05)
        rmse = float(np.sqrt(mean_squared_error(actual, estimate)))
        denominator = np.maximum(np.abs(actual), np.finfo(np.float64).eps)
        rows.append(
            {
                "target": name,
                "MAE": float(mean_absolute_error(actual, estimate)),
                "RMSE": rmse,
                "R2": float(r2_score(actual, estimate)),
                "NRMSE_q05_q95": rmse / scale if scale > 0.0 else float("nan"),
                "MAPE_percent": float(np.mean(np.abs(estimate - actual) / denominator) * 100.0),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    labels, probabilities = _classification_arrays(y_true, probability)
    if n_bins < 2:
        raise ValueError("n_bins must be at least two.")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.minimum(np.digitize(probabilities, edges[1:-1]), n_bins - 1)
    ece = 0.0
    for bin_index in range(n_bins):
        mask = bin_indices == bin_index
        if not mask.any():
            continue
        ece += mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean())
    return float(ece)


def _classification_arrays(
    y_true: np.ndarray, probability: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probability, dtype=np.float64).reshape(-1)
    if labels.shape != probabilities.shape:
        raise ValueError("Classification labels and probabilities must have equal length.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Classification labels must be binary zero/one values.")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("Probabilities must be finite values in [0, 1].")
    return labels, probabilities


def probability_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    labels, probabilities = _classification_arrays(y_true, probability)
    has_two_classes = len(np.unique(labels)) == 2
    return {
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(labels, probabilities))
        if has_two_classes
        else float("nan"),
        "average_precision_feasible": float(
            average_precision_score(labels, probabilities)
        )
        if has_two_classes
        else float("nan"),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "ece_15_bins": expected_calibration_error(labels, probabilities, n_bins=15),
    }


def threshold_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    labels, probabilities = _classification_arrays(y_true, probability)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1].")
    predicted = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "feasible_precision": float(
            precision_score(labels, predicted, zero_division=0)
        ),
        "feasible_recall": float(recall_score(labels, predicted, zero_division=0)),
        "specificity": float(specificity),
        "false_feasible_rate_among_true_infeasible": float(1.0 - specificity),
        "true_infeasible_predicted_infeasible": int(tn),
        "false_feasible": int(fp),
        "false_infeasible": int(fn),
        "true_feasible": int(tp),
    }

