"""Tests for the cleaning routines."""
import numpy as np
import pandas as pd

from src import cleaning


def test_clean_fraud_drops_duplicates():
    df = pd.DataFrame(
        {
            "user_id": [1, 1],
            "signup_time": ["2026-01-01", "2026-01-01"],
            "purchase_time": ["2026-01-02", "2026-01-02"],
            "ip_address": [123.0, 123.0],
            "source": ["SEO", "SEO"],
            "browser": ["Chrome", "Chrome"],
            "sex": ["M", "M"],
            "class": [0, 0],
        }
    )
    out = cleaning.clean_fraud_data(df)
    assert len(out) == 1


def test_clean_fraud_drops_rows_missing_essentials():
    df = pd.DataFrame(
        {
            "user_id": [1, 2],
            "signup_time": ["2026-01-01", "2026-01-01"],
            "purchase_time": ["2026-01-02", None],  # row 2 unusable
            "ip_address": [123.0, 456.0],
            "class": [0, 1],
        }
    )
    out = cleaning.clean_fraud_data(df)
    assert len(out) == 1
    assert out.loc[0, "user_id"] == 1


def test_clean_fraud_sets_dtypes():
    df = pd.DataFrame(
        {
            "user_id": [1],
            "signup_time": ["2026-01-01"],
            "purchase_time": ["2026-01-02"],
            "ip_address": [123.0],
            "source": ["SEO"],
            "browser": ["Chrome"],
            "sex": ["F"],
            "class": [0],
        }
    )
    out = cleaning.clean_fraud_data(df)
    assert pd.api.types.is_datetime64_any_dtype(out["purchase_time"])
    assert isinstance(out["source"].dtype, pd.CategoricalDtype)


def test_class_balance():
    df = pd.DataFrame({"class": [0, 0, 0, 1]})
    bal = cleaning.class_balance(df, "class")
    assert bal.loc[1, "count"] == 1
    assert np.isclose(bal.loc[1, "pct"], 25.0)


def test_clean_creditcard_dedup():
    df = pd.DataFrame({"Time": [1, 1, 2], "Amount": [10, 10, 20], "Class": [0, 0, 1]})
    out = cleaning.clean_creditcard(df)
    assert len(out) == 2
