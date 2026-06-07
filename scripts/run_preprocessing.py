"""Task 1 pipeline: clean -> geolocate -> feature-engineer -> save processed data.

Runs both datasets end-to-end and writes model-ready CSVs to data/processed.

Usage:
    python scripts/run_preprocessing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import cleaning, config, data_loading, feature_engineering, geolocation  # noqa: E402


def process_fraud() -> None:
    print("== E-commerce Fraud_Data ==")
    raw = data_loading.load_fraud_data()
    print(f"  raw shape:           {raw.shape}")

    clean = cleaning.clean_fraud_data(raw)
    print(f"  after cleaning:      {clean.shape}")
    print(cleaning.class_balance(clean, config.FRAUD_TARGET).to_string())

    ip_country = data_loading.load_ip_country()
    geo = geolocation.merge_ip_to_country(clean, ip_country)
    matched = (geo["country"] != "Unknown").mean() * 100
    print(f"  IP->country matched: {matched:.1f}% of rows")

    feats = feature_engineering.engineer_fraud_features(geo)
    print(f"  engineered shape:    {feats.shape}")
    print("  new feature columns: hour_of_day, day_of_week, time_since_signup_hours,")
    print("                       user_tx_count, device_tx_count, device_shared_users,")
    print("                       sec_since_prev_user_tx, country")

    config.ensure_dirs()
    out = config.PROCESSED_DIR / "fraud_data_processed.csv"
    feats.to_csv(out, index=False)
    print(f"  -> wrote {out}\n")


def process_creditcard() -> None:
    print("== Bank creditcard ==")
    raw = data_loading.load_creditcard()
    print(f"  raw shape:           {raw.shape}")

    clean = cleaning.clean_creditcard(raw)
    print(f"  after cleaning:      {clean.shape}")
    print(cleaning.class_balance(clean, config.CREDITCARD_TARGET).to_string())

    config.ensure_dirs()
    out = config.PROCESSED_DIR / "creditcard_processed.csv"
    clean.to_csv(out, index=False)
    print(f"  -> wrote {out}\n")


def main() -> None:
    process_fraud()
    process_creditcard()
    print("Task 1 preprocessing complete.")


if __name__ == "__main__":
    main()
