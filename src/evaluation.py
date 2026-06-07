"""Evaluation metrics and plots tuned for highly imbalanced classification.

Accuracy is meaningless when 99.8% of rows are one class, so we lead with
metrics that focus on the rare positive (fraud) class: Average Precision
(area under the precision-recall curve), F1, ROC-AUC and the confusion matrix.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(y_true, y_pred, y_score, label: str = "model") -> dict:
    """Compute the headline metrics for imbalanced fraud detection.

    Parameters
    ----------
    y_true : array-like of true labels.
    y_pred : array-like of hard predictions (0/1).
    y_score : array-like of positive-class probabilities/scores.
    label : name used in the returned record.
    """
    return {
        "model": label,
        "auc_pr": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
    }


def metrics_table(records: list[dict]) -> pd.DataFrame:
    """Turn a list of ``evaluate`` outputs into a sorted comparison table."""
    df = pd.DataFrame(records).set_index("model")
    return df.sort_values("auc_pr", ascending=False).round(4)


def plot_confusion_matrix(y_true, y_pred, label: str = "model", save_path: Path | str | None = None):
    """Plot a 2x2 confusion matrix with raw counts."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Confusion matrix - {label}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_pr_curve(y_true, y_score, label: str = "model", save_path: Path | str | None = None):
    """Plot the precision-recall curve (the key view for imbalanced data)."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"{label} (AP={ap:.3f})")
    baseline = np.mean(y_true)
    ax.axhline(baseline, ls="--", color="grey", label=f"baseline={baseline:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.legend(loc="best")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
