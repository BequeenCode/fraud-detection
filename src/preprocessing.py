"""Preprocessing: feature selection, scaling, encoding, splitting, resampling.

The functions here turn a feature-engineered frame into model-ready arrays and
provide a reusable ``ColumnTransformer`` so that scaling/encoding is fit on the
training fold only (no leakage), and SMOTE is applied to the training set only.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

# Columns that are identifiers / raw timestamps and must NOT enter the model.
FRAUD_DROP_COLS = [
    "user_id",
    "device_id",
    "signup_time",
    "purchase_time",
    "ip_address",
    "ip_int",
]

FRAUD_CATEGORICAL = ["source", "browser", "sex", "country"]


def select_fraud_model_columns(df: pd.DataFrame, target: str = config.FRAUD_TARGET):
    """Split an engineered fraud frame into (X, y), dropping identifier columns."""
    drop = [c for c in FRAUD_DROP_COLS + [target] if c in df.columns]
    X = df.drop(columns=drop)
    y = df[target]
    return X, y


def build_fraud_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer: impute+scale numerics, impute+OHE categoricals."""
    categorical = [c for c in FRAUD_CATEGORICAL if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical]

    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )


def build_creditcard_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Impute (median) + scale all credit-card features (V1..V28, Time, Amount)."""
    numeric = list(X.columns)
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", numeric_pipe, numeric)],
        remainder="drop",
    )


def stratified_split(X, y, test_size: float = 0.2, random_state: int = config.RANDOM_STATE):
    """Stratified train/test split preserving the (rare) positive-class rate."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def make_resampling_pipeline(preprocessor, estimator, random_state: int = config.RANDOM_STATE):
    """Wrap preprocessing + SMOTE + estimator into one imblearn Pipeline.

    Using the imbalanced-learn ``Pipeline`` guarantees SMOTE runs *inside* each
    cross-validation fold and only on the training portion, which is the correct
    way to avoid synthetic-sample leakage into validation data.
    """
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    return ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=random_state)),
            ("model", estimator),
        ]
    )


def make_plain_pipeline(preprocessor, estimator):
    """Preprocessing + estimator with no resampling (for class_weight models)."""
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
