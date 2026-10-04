import numpy as np
import pandas as pd
import pytest

from cryptosurge.backtest import run_backtest
from cryptosurge.features import NEXT_RETURN_COL, build_features
from cryptosurge.train import EMBARGO_DAYS, best_f1_threshold, chronological_split, classification_metrics


def test_chronological_split_with_embargo(prices):
    df = build_features(prices)
    parts = chronological_split(df, val_frac=0.2, test_frac=0.2)
    gap = pd.Timedelta(days=EMBARGO_DAYS)
    assert parts["val"]["date"].min() - parts["train"]["date"].max() > gap
    assert parts["test"]["date"].min() - parts["val"]["date"].max() > gap
    assert sum(len(p) for p in parts.values()) < len(df)  # embargo rows are dropped


def test_classification_metrics_known_values():
    y = np.array([1, 0, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.4, 0.6, 0.1])
    m = classification_metrics(y, p, 0.5)
    assert m["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 1}
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["accuracy"] == 0.6
    assert 0 < best_f1_threshold(y, p) <= 0.9


def test_backtest_arithmetic_on_real_returns(prices):
    df = build_features(prices)[["date", "symbol", NEXT_RETURN_COL]].copy()
    df["surge_probability"] = np.where(df["symbol"] == "SOLUSDT", 0.9, 0.1)
    bt = run_backtest(df, threshold=0.5, max_positions=5, fee=0.001, random_draws=10)
    rebalance = np.sort(df["date"].unique())[::7]
    sol = df[(df["symbol"] == "SOLUSDT") & df["date"].isin(rebalance)].sort_values("date")[NEXT_RETURN_COL]
    expected = np.prod((1 + sol.to_numpy()) * 0.999**2) - 1
    assert bt["strategy"]["total_return"] == pytest.approx(expected, abs=1e-4)
    assert bt["positions_taken"] == len(sol) == bt["weeks_invested"]
    assert bt["strategy"]["max_drawdown"] <= 0


def test_backtest_stays_in_cash_without_signals(prices):
    df = build_features(prices)[["date", "symbol", NEXT_RETURN_COL]].copy()
    df["surge_probability"] = 0.0
    bt = run_backtest(df, threshold=0.5, random_draws=5)
    assert bt["strategy"]["total_return"] == 0.0 and bt["positions_taken"] == 0
