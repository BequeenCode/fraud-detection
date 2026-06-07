"""Data-cleaning routines for both transaction datasets.

Each function is pure (returns a new frame, never mutates the input) so the
steps are easy to unit-test and to chain in a notebook or pipeline.
"""
from __future__ import annotations

import pandas as pd


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column table of missing-value counts and percentages."""
    counts = df.isna().sum()
    report = pd.DataFrame(
        {
            "missing_count": counts,
            "missing_pct": (counts / len(df) * 100).round(3),
            "dtype": df.dtypes.astype(str),
        }
    )
    return report.sort_values("missing_count", ascending=False)


def clean_fraud_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the e-commerce fraud dataset.

    Steps
    -----
    * Drop exact duplicate rows.
    * Coerce timestamp columns to ``datetime`` (idempotent if already parsed).
    * Cast categorical columns to ``category`` dtype.
    * Drop rows missing values in columns that cannot be sensibly imputed
      (timestamps, IP address). Numeric/categorical gaps, if any, are imputed
      downstream by the modelling pipeline rather than here.
    """
    out = df.copy()

    out = out.drop_duplicates()

    for col in ("signup_time", "purchase_time"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    # IP address stored as float in the raw file; keep numeric for conversion.
    if "ip_address" in out.columns:
        out["ip_address"] = pd.to_numeric(out["ip_address"], errors="coerce")

    # Rows without a purchase time or IP cannot be feature-engineered.
    essential = [c for c in ("signup_time", "purchase_time", "ip_address") if c in out.columns]
    out = out.dropna(subset=essential)

    for col in ("source", "browser", "sex"):
        if col in out.columns:
            out[col] = out[col].astype("category")

    return out.reset_index(drop=True)


def clean_creditcard(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the bank credit-card dataset.

    The Kaggle credit-card file has no missing values, but it does contain a
    notable number of exact duplicate rows which should be removed before
    modelling to avoid train/test leakage of identical records.
    """
    out = df.copy()
    out = out.drop_duplicates()
    return out.reset_index(drop=True)


def class_balance(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Return the class distribution (count + percentage) of ``target``."""
    counts = df[target].value_counts().sort_index()
    pct = (counts / len(df) * 100).round(4)
    return pd.DataFrame({"count": counts, "pct": pct})
