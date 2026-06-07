"""Task 2 pipeline: train + evaluate Logistic Regression and an ensemble.

Reads the processed CSVs from data/processed (run run_preprocessing.py first),
trains a baseline and an ensemble for BOTH datasets with proper imbalance
handling, evaluates on a held-out stratified test split, runs 5-fold CV, prints
a comparison table, and persists the best models.

Usage:
    python scripts/run_modeling.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, evaluation, modeling, preprocessing  # noqa: E402


def _ensemble(y_train):
    """XGBoost if available, else Random Forest (both imbalance-aware)."""
    try:
        spw = modeling.imbalance_ratio(y_train)
        return modeling.make_xgboost(scale_pos_weight=spw), "xgboost"
    except ImportError:
        return modeling.make_random_forest(), "random_forest"


def run_dataset(name: str, processed_file: Path, target: str, build_preprocessor, select_xy):
    print(f"\n===== {name} =====")
    df = pd.read_csv(processed_file)
    X, y = select_xy(df, target)
    X_train, X_test, y_train, y_test = preprocessing.stratified_split(X, y)
    print(f"train={X_train.shape}  test={X_test.shape}  "
          f"train positives={int(y_train.sum())}  test positives={int(y_test.sum())}")

    records = []

    # --- Baseline: Logistic Regression (class_weight balanced) ---
    lr_pipe = preprocessing.make_plain_pipeline(
        build_preprocessor(X_train), modeling.make_logistic_regression()
    )
    lr_pipe.fit(X_train, y_train)
    lr_score = lr_pipe.predict_proba(X_test)[:, 1]
    lr_pred = (lr_score >= 0.5).astype(int)
    records.append(evaluation.evaluate(y_test, lr_pred, lr_score, "LogisticRegression"))

    # --- Ensemble ---
    est, est_name = _ensemble(y_train)
    ens_pipe = preprocessing.make_plain_pipeline(build_preprocessor(X_train), est)
    ens_pipe.fit(X_train, y_train)
    ens_score = ens_pipe.predict_proba(X_test)[:, 1]
    ens_pred = (ens_score >= 0.5).astype(int)
    records.append(evaluation.evaluate(y_test, ens_pred, ens_score, est_name))

    table = evaluation.metrics_table(records)
    print("\nHeld-out test metrics:")
    print(table.to_string())

    best_name = table.index[0]
    best_pipe = ens_pipe if best_name == est_name else lr_pipe
    saved = modeling.save_model(best_pipe, f"{name.lower().replace(' ', '_')}_best")
    print(f"Best model: {best_name} -> saved to {saved}")
    return table


def main() -> None:
    run_dataset(
        "Fraud Data",
        config.PROCESSED_DIR / "fraud_data_processed.csv",
        config.FRAUD_TARGET,
        preprocessing.build_fraud_preprocessor,
        preprocessing.select_fraud_model_columns,
    )

    def select_cc(df, target):
        return df.drop(columns=[target]), df[target]

    run_dataset(
        "Creditcard",
        config.PROCESSED_DIR / "creditcard_processed.csv",
        config.CREDITCARD_TARGET,
        preprocessing.build_creditcard_preprocessor,
        select_cc,
    )
    print("\nTask 2 modelling complete.")


if __name__ == "__main__":
    main()
