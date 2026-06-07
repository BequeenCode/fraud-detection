"""Shared pytest fixtures and path setup."""
import sys
from pathlib import Path

import pytest

# Make the project root importable so `import src...` works under pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def tiny_ip_country():
    import pandas as pd

    return pd.DataFrame(
        {
            "lower_bound_ip_address": [0, 100, 200, 300],
            "upper_bound_ip_address": [99, 199, 249, 399],  # gap: 250-299
            "country": ["A", "B", "C", "D"],
        }
    )
