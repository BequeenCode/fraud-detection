"""Generate small SYNTHETIC datasets matching the real schemas.

Purpose: let the full pipeline + tests run end-to-end *without* the real data,
so the code is verifiably correct. The numbers are fabricated and must NOT be
used for any analytical conclusions. Replace the files in ``data/raw`` with the
real Fraud_Data.csv / IpAddress_to_Country.csv / creditcard.csv for actual work.

Usage:
    python scripts/make_synthetic_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

RNG = np.random.default_rng(config.RANDOM_STATE)


def _make_fraud_data(n: int = 4000) -> pd.DataFrame:
    n_users = int(n * 0.85)  # some users repeat -> velocity signal
    user_id = RNG.integers(1, n_users, size=n)
    n_devices = int(n * 0.8)
    device_id = np.array([f"DEV{d:06d}" for d in RNG.integers(1, n_devices, size=n)])

    signup = pd.Timestamp("2026-01-01") + pd.to_timedelta(RNG.integers(0, 120 * 24 * 3600, n), unit="s")

    # Fraud: short time-since-signup; legit: long. Build target first, then time.
    is_fraud = RNG.random(n) < 0.093  # ~9% positive, like the real e-commerce set
    short = RNG.integers(1, 3600, n)            # seconds
    long_ = RNG.integers(2 * 24 * 3600, 90 * 24 * 3600, n)
    gap = np.where(is_fraud, short, long_)
    purchase = signup + pd.to_timedelta(gap, unit="s")

    df = pd.DataFrame(
        {
            "user_id": user_id,
            "signup_time": signup,
            "purchase_time": purchase,
            "purchase_value": RNG.gamma(2.0, 20.0, n).round(2),
            "device_id": device_id,
            "source": RNG.choice(["SEO", "Ads", "Direct"], n, p=[0.4, 0.45, 0.15]),
            "browser": RNG.choice(["Chrome", "Safari", "FireFox", "IE", "Opera"], n),
            "sex": RNG.choice(["M", "F"], n),
            "age": RNG.integers(18, 70, n),
            "ip_address": RNG.integers(16_000_000, 3_700_000_000, n).astype("float64"),
            "class": is_fraud.astype(int),
        }
    )
    # Inject a few duplicates and a couple of missing ages to exercise cleaning.
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    df.loc[df.sample(3, random_state=0).index, "age"] = np.nan
    return df


def _make_ip_country(n_ranges: int = 300) -> pd.DataFrame:
    edges = np.sort(RNG.integers(0, 4_000_000_000, n_ranges + 1))
    edges = np.unique(edges)
    countries = RNG.choice(
        ["United States", "Nigeria", "China", "United Kingdom", "Brazil",
         "India", "Germany", "Canada", "France", "Japan"],
        size=len(edges) - 1,
    )
    return pd.DataFrame(
        {
            "lower_bound_ip_address": edges[:-1].astype("int64"),
            "upper_bound_ip_address": (edges[1:] - 1).astype("int64"),
            "country": countries,
        }
    )


def _make_creditcard(n: int = 6000) -> pd.DataFrame:
    is_fraud = RNG.random(n) < 0.0017  # ~0.17% positive, like the real set
    data = {"Time": np.sort(RNG.integers(0, 172_800, n)).astype(float)}
    for i in range(1, 29):
        base = RNG.normal(0, 1, n)
        # Give fraud rows a slight shift on a few components so a model can learn.
        if i in (1, 3, 4, 7, 10, 14):
            base = base + is_fraud * RNG.normal(2.5, 0.5)
        data[f"V{i}"] = base.round(6)
    data["Amount"] = RNG.gamma(1.5, 60, n).round(2)
    data["Class"] = is_fraud.astype(int)
    df = pd.DataFrame(data)
    df = pd.concat([df, df.iloc[:8]], ignore_index=True)  # duplicates to clean
    return df


def main() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    _make_fraud_data().to_csv(config.FRAUD_DATA_FILE, index=False)
    _make_ip_country().to_csv(config.IP_COUNTRY_FILE, index=False)
    _make_creditcard().to_csv(config.CREDITCARD_FILE, index=False)
    print(f"Synthetic data written to {config.RAW_DIR}")
    print("  - Fraud_Data.csv")
    print("  - IpAddress_to_Country.csv")
    print("  - creditcard.csv")
    print("WARNING: synthetic numbers; for code testing only, not analysis.")


if __name__ == "__main__":
    main()
