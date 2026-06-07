"""Model factories, cross-validation and persistence helpers.

Two model families are supported per the brief:
* Logistic Regression - the interpretable baseline.
* An ensemble (XGBoost by default, Random Forest as a dependency-free fallback).

Class imbalance is handled two ways depending on the model: SMOTE inside an
imbalanced-learn pipeline, or built-in class weighting. Both are exposed so the
modelling notebook can compare them.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from . import config


def make_logistic_regression(class_weight: str | None = "balanced") -> LogisticRegression:
    """Interpretable baseline. ``class_weight='balanced'`` counters imbalance."""
    return LogisticRegression(
        max_iter=2000,
        class_weight=class_weight,
        random_state=config.RANDOM_STATE,
    )


def make_random_forest(n_estimators: int = 300, max_depth: int | None = None):
    """Random-forest ensemble (no extra dependency beyond scikit-learn)."""
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )


def make_xgboost(scale_pos_weight: float = 1.0, n_estimators: int = 400, max_depth: int = 6):
    """Gradient-boosted trees. ``scale_pos_weight`` ~ (#neg / #pos) for imbalance.

    Falls back to a clear error if xgboost is not installed so callers can switch
    to ``make_random_forest``.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "xgboost is not installed; use make_random_forest() instead "
            "or `pip install xgboost`."
        ) from exc

    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )


def imbalance_ratio(y) -> float:
    """Return #negatives / #positives, the natural ``scale_pos_weight``."""
    y = np.asarray(y)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return float(neg / max(pos, 1))


def cross_validate_model(pipeline, X, y, k: int = 5) -> dict:
    """Stratified k-fold CV reporting mean/std of fraud-relevant metrics."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=config.RANDOM_STATE)
    scoring = {
        "auc_pr": "average_precision",
        "roc_auc": "roc_auc",
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
    }
    cv = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=1)
    summary = {}
    for metric in scoring:
        scores = cv[f"test_{metric}"]
        summary[metric] = (float(scores.mean()), float(scores.std()))
    return summary


def save_model(model, name: str) -> Path:
    """Persist a fitted model/pipeline to ``models/<name>.joblib``."""
    config.ensure_dirs()
    path = config.MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name: str):
    """Load a model previously saved with :func:`save_model`."""
    return joblib.load(config.MODELS_DIR / f"{name}.joblib")
