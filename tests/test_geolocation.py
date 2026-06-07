"""Tests for IP conversion and range-based country lookup."""
import pandas as pd

from src import geolocation


def test_ip_to_int_dotted_quad():
    s = pd.Series(["0.0.0.0", "0.0.1.0", "1.0.0.0", "255.255.255.255"])
    out = geolocation.ip_to_int(s)
    assert out.tolist() == [0, 256, 16_777_216, 4_294_967_295]


def test_ip_to_int_numeric_passthrough():
    s = pd.Series([732758368.79, 350311387.0])
    out = geolocation.ip_to_int(s)
    assert out.tolist() == [732758368, 350311387]


def test_merge_assigns_correct_country(tiny_ip_country):
    fraud = pd.DataFrame({"ip_address": [50, 150, 220, 350], "class": [0, 1, 0, 1]})
    merged = geolocation.merge_ip_to_country(fraud, tiny_ip_country)
    assert merged["country"].tolist() == ["A", "B", "C", "D"]


def test_merge_gap_is_unknown(tiny_ip_country):
    # 275 falls in the 250-299 gap -> Unknown.
    fraud = pd.DataFrame({"ip_address": [275], "class": [0]})
    merged = geolocation.merge_ip_to_country(fraud, tiny_ip_country)
    assert merged.loc[0, "country"] == "Unknown"


def test_merge_preserves_row_order(tiny_ip_country):
    # Out-of-order IPs must come back aligned to the original rows.
    fraud = pd.DataFrame({"ip_address": [350, 50, 150], "class": [1, 0, 1]})
    merged = geolocation.merge_ip_to_country(fraud, tiny_ip_country)
    assert merged["country"].tolist() == ["D", "A", "B"]
    assert merged["class"].tolist() == [1, 0, 1]
