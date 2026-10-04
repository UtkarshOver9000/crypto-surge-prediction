from pathlib import Path

import pandas as pd
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "binance_sample.csv"


@pytest.fixture(scope="session")
def prices() -> pd.DataFrame:
    """200 real daily Binance candles for 6 coins (2026-03-18 to 2026-10-03)."""
    return pd.read_csv(FIXTURE, parse_dates=["date"])
