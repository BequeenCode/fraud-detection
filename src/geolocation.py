"""Geolocation enrichment: map e-commerce transactions to a country via IP.

The lookup table gives countries for *ranges* of (integer) IP addresses, so a
plain equality join will not work. We use an interval lookup:

1. Convert each transaction's IP to an integer.
2. Sort the lookup table by its lower bound.
3. ``merge_asof`` finds, for every transaction, the range whose lower bound is
   the greatest value <= the transaction IP (a vectorised binary search).
4. Validate that the matched range's *upper* bound also covers the IP; rows
   that fall in a gap between ranges are labelled ``"Unknown"``.

This is O(n log n) and handles the full dataset in well under a second.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ip_to_int(ip_series: pd.Series) -> pd.Series:
    """Convert IP addresses to unsigned 64-bit integers.

    Handles two encodings found in the wild:
    * Already-numeric values (the raw ``Fraud_Data.csv`` stores floats such as
      ``732758368.79``) -> truncated to int.
    * Dotted-quad strings such as ``"192.168.0.1"`` -> standard
      ``a*256^3 + b*256^2 + c*256 + d`` conversion.
    """
    # Numeric (float/int) path.
    if pd.api.types.is_numeric_dtype(ip_series):
        return ip_series.astype("float64").astype("int64")

    def _convert(value: object) -> int:
        s = str(value).strip()
        if "." in s and s.replace(".", "").isdigit() and s.count(".") == 3:
            a, b, c, d = (int(p) for p in s.split("."))
            return (a << 24) + (b << 16) + (c << 8) + d
        # Fallback: a numeric string like "732758368.0".
        return int(float(s))

    return ip_series.map(_convert).astype("int64")


def merge_ip_to_country(
    fraud_df: pd.DataFrame,
    ip_country_df: pd.DataFrame,
    ip_col: str = "ip_address",
) -> pd.DataFrame:
    """Attach a ``country`` column to ``fraud_df`` via range-based IP lookup.

    Parameters
    ----------
    fraud_df : DataFrame
        Transactions; must contain ``ip_col``.
    ip_country_df : DataFrame
        Lookup with ``lower_bound_ip_address``, ``upper_bound_ip_address`` and
        ``country`` columns.
    ip_col : str
        Name of the IP column in ``fraud_df``.

    Returns
    -------
    DataFrame
        Copy of ``fraud_df`` with two new columns: ``ip_int`` and ``country``.
    """
    out = fraud_df.copy()
    out["ip_int"] = ip_to_int(out[ip_col])

    lookup = ip_country_df.copy()
    lookup["lower_bound_ip_address"] = lookup["lower_bound_ip_address"].astype("float64").astype("int64")
    lookup["upper_bound_ip_address"] = lookup["upper_bound_ip_address"].astype("float64").astype("int64")
    lookup = lookup.sort_values("lower_bound_ip_address").reset_index(drop=True)

    # merge_asof requires both keys sorted; remember original order to restore.
    out = out.reset_index().rename(columns={"index": "_orig_order"})
    out_sorted = out.sort_values("ip_int")

    merged = pd.merge_asof(
        out_sorted,
        lookup[["lower_bound_ip_address", "upper_bound_ip_address", "country"]],
        left_on="ip_int",
        right_on="lower_bound_ip_address",
        direction="backward",
    )

    # Invalidate matches where the IP exceeds the matched range's upper bound.
    in_range = merged["ip_int"] <= merged["upper_bound_ip_address"]
    merged.loc[~in_range, "country"] = np.nan
    merged["country"] = merged["country"].fillna("Unknown")

    merged = merged.drop(columns=["lower_bound_ip_address", "upper_bound_ip_address"])
    merged = merged.sort_values("_orig_order").drop(columns="_orig_order").reset_index(drop=True)
    return merged


def fraud_rate_by_country(df: pd.DataFrame, target: str = "class", top_n: int = 15) -> pd.DataFrame:
    """Summarise transaction volume and fraud rate per country.

    Returns countries with at least a handful of transactions, ranked by
    fraud rate, so genuinely risky geographies surface ahead of low-volume noise.
    """
    grouped = (
        df.groupby("country")[target]
        .agg(transactions="count", frauds="sum")
        .assign(fraud_rate=lambda x: (x["frauds"] / x["transactions"] * 100).round(3))
    )
    grouped = grouped[grouped["transactions"] >= 50]
    return grouped.sort_values("fraud_rate", ascending=False).head(top_n)
