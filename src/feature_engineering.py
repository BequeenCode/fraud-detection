"""Feature engineering for the e-commerce fraud dataset.

Creates the behavioural and temporal signals called for in the brief:

* ``time_since_signup``  - seconds (and hours) between signup and purchase.
* ``hour_of_day`` / ``day_of_week`` - temporal context of the purchase.
* transaction-velocity features built from how often a *user* and a *device*
  transact, and how quickly purchases follow one another.

All functions return a new frame and never mutate their input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``hour_of_day``, ``day_of_week`` and ``time_since_signup``.

    ``time_since_signup_hours`` is the headline fraud signal: fraudulent
    accounts frequently transact within seconds/minutes of signing up, whereas
    legitimate users tend to wait much longer.
    """
    out = df.copy()
    purchase = pd.to_datetime(out["purchase_time"])
    signup = pd.to_datetime(out["signup_time"])

    out["hour_of_day"] = purchase.dt.hour
    out["day_of_week"] = purchase.dt.dayofweek  # Monday=0 .. Sunday=6

    delta_seconds = (purchase - signup).dt.total_seconds()
    out["time_since_signup_sec"] = delta_seconds
    out["time_since_signup_hours"] = delta_seconds / 3600.0
    return out


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add transaction-frequency / velocity features.

    Features
    --------
    user_tx_count   : total purchases made by the user.
    device_tx_count : total purchases seen on the device (shared devices are a
                      classic fraud-ring signal).
    device_shared_users : number of distinct users on the device.
    sec_since_prev_user_tx : seconds since the same user's previous purchase
                      (large value / NaN -> first purchase). Captures rapid-fire
                      "velocity" bursts.
    """
    out = df.copy()

    out["user_tx_count"] = out.groupby("user_id")["user_id"].transform("count")
    out["device_tx_count"] = out.groupby("device_id")["device_id"].transform("count")
    out["device_shared_users"] = out.groupby("device_id")["user_id"].transform("nunique")

    out = out.sort_values(["user_id", "purchase_time"])
    prev_purchase = out.groupby("user_id")["purchase_time"].shift(1)
    out["sec_since_prev_user_tx"] = (
        (pd.to_datetime(out["purchase_time"]) - pd.to_datetime(prev_purchase))
        .dt.total_seconds()
    )
    # First-ever purchase for a user: fill with a large sentinel (1 year).
    out["sec_since_prev_user_tx"] = out["sec_since_prev_user_tx"].fillna(365 * 24 * 3600)

    return out.sort_index() if out.index.is_monotonic_increasing else out.reset_index(drop=True)


def engineer_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full e-commerce feature-engineering pipeline."""
    out = add_time_features(df)
    out = add_velocity_features(out)
    return out.reset_index(drop=True)
