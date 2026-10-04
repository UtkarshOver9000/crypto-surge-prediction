"""
Features and surge labels from daily OHLCV candles.

Every feature for day t uses only candles up to and including day t's close.
The label looks forward: a "surge" means the close 7 days later is at least
15% above day t's close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import bollinger, ema, macd, rsi

SURGE_THRESHOLD = 0.15
SURGE_HORIZON_DAYS = 7
LABEL_COL = "surge_label"
NEXT_RETURN_COL = "next_return"
BTC = "BTCUSDT"

FEATURE_COLS = [
    "ret_1d",
    "ret_3d",
    "ret_7d",
    "ret_14d",
    "ret_30d",
    "vol_7d",
    "vol_14d",
    "vol_30d",
    "price_sma_7",
    "price_sma_14",
    "price_sma_30",
    "price_sma_90",
    "price_z_30",
    "ema_12_gap",
    "ema_26_gap",
    "rsi_14",
    "macd_norm",
    "macd_hist_norm",
    "bb_width_20",
    "bb_z_20",
    "range_pct",
    "range_z_30",
    "log_volume_chg_1d",
    "volume_z_30",
    "trades_z_30",
    "taker_buy_ratio",
    "log_days_listed",
    "market_ret_1d",
    "market_ret_7d",
    "market_vol_7d",
    "market_breadth_7d",
    "btc_ret_1d",
    "btc_ret_7d",
    "btc_vol_7d",
    "btc_volume_share",
    "volume_rank_pct",
    "rel_ret_1d",
    "rel_ret_7d",
]
RAW_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "quote_volume", "trades", "taker_buy_quote"]


def _zscore(s: pd.Series, window: int) -> pd.Series:
    return (s - s.rolling(window).mean()) / s.rolling(window).std()


def _coin_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    close, volume = g["close"], g["quote_volume"]
    log_ret = np.log(close).diff()

    for n in (1, 3, 7, 14, 30):
        g[f"ret_{n}d"] = close.pct_change(n)
    for n in (7, 14, 30):
        g[f"vol_{n}d"] = log_ret.rolling(n).std()
    for n in (7, 14, 30, 90):
        g[f"price_sma_{n}"] = close / close.rolling(n).mean() - 1
    g["price_z_30"] = _zscore(close, 30)

    g["ema_12_gap"] = close / ema(close, 12) - 1
    g["ema_26_gap"] = close / ema(close, 26) - 1
    g["rsi_14"] = rsi(close, 14)
    macd_line, _, macd_hist = macd(close, 12, 26, 9)
    g["macd_norm"] = macd_line / close
    g["macd_hist_norm"] = macd_hist / close
    mid, upper, lower, std = bollinger(close, 20, 2.0)
    g["bb_width_20"] = (upper - lower) / mid
    g["bb_z_20"] = (close - mid) / std

    g["range_pct"] = (g["high"] - g["low"]) / close
    g["range_z_30"] = _zscore(g["range_pct"], 30)
    g["log_volume_chg_1d"] = np.log1p(volume).diff()
    g["volume_z_30"] = _zscore(volume, 30)
    g["trades_z_30"] = _zscore(g["trades"].astype(float), 30)
    g["taker_buy_ratio"] = g["taker_buy_quote"] / volume.replace(0, np.nan)
    g["log_days_listed"] = np.log1p(np.arange(len(g), dtype=float))
    g["volume_30d_avg"] = volume.rolling(30).mean()
    return g


def build_features(
    df: pd.DataFrame,
    horizon: int = SURGE_HORIZON_DAYS,
    surge_threshold: float = SURGE_THRESHOLD,
    include_labels: bool = True,
) -> pd.DataFrame:
    df = df[RAW_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"])
    df = pd.concat([_coin_features(g) for _, g in df.groupby("symbol", sort=False)], ignore_index=True)

    by_date = df.groupby("date")
    df["market_ret_1d"] = by_date["ret_1d"].transform("mean")
    df["market_ret_7d"] = by_date["ret_7d"].transform("mean")
    df["market_vol_7d"] = by_date["vol_7d"].transform("mean")
    df["market_breadth_7d"] = by_date["ret_7d"].transform(lambda s: (s > 0).mean())
    df["volume_rank_pct"] = by_date["volume_30d_avg"].rank(pct=True)
    total_volume = by_date["quote_volume"].transform("sum")

    btc = df[df["symbol"] == BTC][["date", "ret_1d", "ret_7d", "vol_7d", "quote_volume"]].rename(
        columns={"ret_1d": "btc_ret_1d", "ret_7d": "btc_ret_7d", "vol_7d": "btc_vol_7d", "quote_volume": "btc_qv"}
    )
    df = df.merge(btc, on="date", how="left")
    df["btc_volume_share"] = df["btc_qv"] / total_volume
    df["rel_ret_1d"] = df["ret_1d"] - df["market_ret_1d"]
    df["rel_ret_7d"] = df["ret_7d"] - df["market_ret_7d"]
    df = df.replace([np.inf, -np.inf], np.nan)

    required = list(FEATURE_COLS)
    if include_labels:
        future = df.groupby("symbol")["close"].shift(-horizon)
        df[NEXT_RETURN_COL] = future / df["close"] - 1
        df[LABEL_COL] = (df[NEXT_RETURN_COL] >= surge_threshold).astype(int)
        required.append(NEXT_RETURN_COL)
    return df.dropna(subset=required).reset_index(drop=True)
