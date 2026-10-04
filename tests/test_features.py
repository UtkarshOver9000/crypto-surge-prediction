import numpy as np
import pandas as pd

from cryptosurge.features import FEATURE_COLS, LABEL_COL, NEXT_RETURN_COL, SURGE_HORIZON_DAYS, build_features


def test_features_and_labels_are_complete(prices):
    df = build_features(prices)
    assert set(FEATURE_COLS) <= set(df.columns)
    assert df[FEATURE_COLS].notna().all().all()
    assert set(df[LABEL_COL].unique()) <= {0, 1}
    # The 90-day SMA needs 89 earlier candles; the label needs 7 later ones.
    assert df["date"].min() >= prices["date"].min() + pd.Timedelta(days=89)
    assert df["date"].max() <= prices["date"].max() - pd.Timedelta(days=SURGE_HORIZON_DAYS)


def test_label_matches_the_real_forward_return(prices):
    df = build_features(prices)
    row = df[df["symbol"] == "BTCUSDT"].iloc[10]
    btc = prices[prices["symbol"] == "BTCUSDT"].set_index("date")["close"]
    expected = btc[row["date"] + pd.Timedelta(days=SURGE_HORIZON_DAYS)] / btc[row["date"]] - 1
    assert np.isclose(row[NEXT_RETURN_COL], expected)
    assert row[LABEL_COL] == int(expected >= 0.15)


def test_features_do_not_use_future_candles(prices):
    full = build_features(prices, include_labels=False)
    day = np.sort(full["date"].unique())[30]
    truncated = build_features(prices[prices["date"] <= day], include_labels=False)
    a = full[full["date"] == day].set_index("symbol")[FEATURE_COLS].sort_index()
    b = truncated[truncated["date"] == day].set_index("symbol")[FEATURE_COLS].sort_index()
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)


def test_inference_keeps_the_latest_day(prices):
    df = build_features(prices, include_labels=False)
    assert df["date"].max() == prices["date"].max()
    assert NEXT_RETURN_COL not in df.columns
