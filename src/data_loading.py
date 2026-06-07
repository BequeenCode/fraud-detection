"""Dataset loading helpers.

Thin wrappers around ``pandas.read_csv`` that (a) resolve paths through
``src.config`` so callers never hard-code locations and (b) parse the
timestamp columns of the e-commerce dataset up front.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def load_fraud_data(path: Path | str = config.FRAUD_DATA_FILE) -> pd.DataFrame:
    """Load the e-commerce ``Fraud_Data.csv`` with timestamps parsed."""
    return pd.read_csv(path, parse_dates=["signup_time", "purchase_time"])


def load_ip_country(path: Path | str = config.IP_COUNTRY_FILE) -> pd.DataFrame:
    """Load the IP-range -> country lookup table."""
    return pd.read_csv(path)


def load_creditcard(path: Path | str = config.CREDITCARD_FILE) -> pd.DataFrame:
    """Load the anonymised bank ``creditcard.csv``."""
    return pd.read_csv(path)
