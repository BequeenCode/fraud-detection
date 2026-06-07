"""Centralised path and constant configuration for the fraud-detection project.

Keeping every filesystem location in one place means notebooks, scripts and
tests all agree on where data lives, and the project relocates cleanly.
"""
from __future__ import annotations

from pathlib import Path

# Project root = parent of the `src` package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Raw input files (expected to be downloaded into data/raw).
FRAUD_DATA_FILE = RAW_DIR / "Fraud_Data.csv"
IP_COUNTRY_FILE = RAW_DIR / "IpAddress_to_Country.csv"
CREDITCARD_FILE = RAW_DIR / "creditcard.csv"

# Target column names differ between the two datasets.
FRAUD_TARGET = "class"
CREDITCARD_TARGET = "Class"

# Global random seed for reproducibility.
RANDOM_STATE = 42


def ensure_dirs() -> None:
    """Create the output directories if they do not already exist."""
    for directory in (PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
