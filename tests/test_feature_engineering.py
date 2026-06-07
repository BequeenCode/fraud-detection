"""Tests for the e-commerce feature-engineering functions."""
import pandas as pd

from src import feature_engineering


def _base_frame():
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "device_id": ["D1", "D1", "D2"],
            "signup_time": pd.to_datetime(
                ["2026-01-01 00:00:00", "2026-01-01 00:00:00", "2026-01-02 10:00:00"]
            ),
            "purchase_time": pd.to_datetime(
                ["2026-01-01 02:00:00", "2026-01-01 05:00:00", "2026-01-02 10:30:00"]
            ),
            "class": [0, 0, 1],
        }
    )


def test_time_since_signup_hours():
    out = feature_engineering.add_time_features(_base_frame())
    assert out["time_since_signup_hours"].tolist() == [2.0, 5.0, 0.5]


def test_hour_and_dayofweek():
    out = feature_engineering.add_time_features(_base_frame())
    assert out["hour_of_day"].tolist() == [2, 5, 10]
    # 2026-01-01 is a Thursday (=3), 2026-01-02 a Friday (=4).
    assert out["day_of_week"].tolist() == [3, 3, 4]


def test_velocity_counts():
    out = feature_engineering.engineer_fraud_features(_base_frame())
    out = out.sort_values("user_id").reset_index(drop=True)
    # user 1 has 2 tx, user 2 has 1.
    assert out.loc[out["user_id"] == 1, "user_tx_count"].tolist() == [2, 2]
    assert out.loc[out["user_id"] == 2, "user_tx_count"].tolist() == [1]
    # device D1 shared by 1 user, used twice.
    assert out.loc[out["device_id"] == "D1", "device_tx_count"].tolist() == [2, 2]


def test_sec_since_prev_user_tx():
    out = feature_engineering.engineer_fraud_features(_base_frame())
    u1 = out[out["user_id"] == 1].sort_values("purchase_time")
    # second purchase is 3h (10800s) after the first.
    assert u1["sec_since_prev_user_tx"].iloc[1] == 10800.0
