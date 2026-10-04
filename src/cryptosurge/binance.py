"""
Binance public market-data client (no API key needed).

Endpoints used (Binance Spot REST API, https://developers.binance.com/docs/binance-spot-api-docs):

* ``GET /api/v3/exchangeInfo``: which USDT pairs exist and are trading
* ``GET /api/v3/ticker/24hr``: 24-hour quote volume, used to rank pairs
* ``GET /api/v3/klines``: daily OHLCV candles, up to 1,000 per request

``data-api.binance.vision`` is Binance's market-data-only host; ``api.binance.com``
is the fallback.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd
import requests

BASE_URLS = ("https://data-api.binance.vision", "https://api.binance.com")
STABLE_BASES = {
    "USDC",
    "FDUSD",
    "TUSD",
    "BUSD",
    "DAI",
    "USDP",
    "PAX",
    "USDS",
    "USDE",
    "PYUSD",
    "AEUR",
    "EUR",
    "EURI",
    "GBP",
    "TRY",
    "BRL",
    "AUD",
    "RUB",
    "UST",
    "USTC",
    "SUSD",
    "XUSD",
    "USD1",
    "BFUSD",
    "RLUSD",
    "U",
    "PAXG",
}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]
DAY_MS = 86_400_000


def _get(path: str, params: dict | None = None, session: requests.Session | None = None):
    http = session or requests
    last_error: Exception | None = None
    for base in BASE_URLS:
        for attempt in range(3):
            try:
                resp = http.get(base + path, params=params, timeout=30)
                if resp.status_code == 429:
                    time.sleep(2 + attempt * 2)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(0.5)
    raise RuntimeError(f"Binance request failed for {path}: {last_error}")


def eligible_usdt_symbols(session: requests.Session | None = None) -> list[str]:
    """USDT spot pairs that are trading, excluding stablecoins and leveraged tokens."""
    info = _get("/api/v3/exchangeInfo", {"permissions": "SPOT"}, session)
    out = []
    for s in info["symbols"]:
        base = s["baseAsset"]
        if (
            s["quoteAsset"] == "USDT"
            and s["status"] == "TRADING"
            and s.get("isSpotTradingAllowed", True)
            and base.isascii()
            and base.isalnum()
            and base not in STABLE_BASES
            and not base.endswith(LEVERAGED_SUFFIXES)
        ):
            out.append(s["symbol"])
    return sorted(out)


def rank_by_quote_volume(symbols: list[str], session: requests.Session | None = None) -> list[str]:
    tickers = _get("/api/v3/ticker/24hr", None, session)
    volume = {t["symbol"]: float(t["quoteVolume"]) for t in tickers}
    return sorted((s for s in symbols if s in volume), key=lambda s: -volume[s])


def daily_klines(symbol: str, start_ms: int = 0, session: requests.Session | None = None) -> pd.DataFrame:
    """All completed daily candles for one symbol from ``start_ms`` (default: listing day)."""
    rows, cursor = [], start_ms
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    while True:
        batch = _get(
            "/api/v3/klines",
            {"symbol": symbol, "interval": "1d", "startTime": cursor, "limit": 1000},
            session,
        )
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        cursor = batch[-1][0] + DAY_MS
    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    df = df[df["close_time"] < now_ms]  # drop today's unfinished candle
    df = df.drop_duplicates("open_time")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize(),
            "symbol": symbol,
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": df["close"].astype(float),
            "quote_volume": df["quote_volume"].astype(float),
            "trades": df["trades"].astype(int),
            "taker_buy_quote": df["taker_buy_quote"].astype(float),
        }
    )
    return out.reset_index(drop=True)


def many_daily_klines(symbols: list[str], start_ms: int = 0, workers: int = 8) -> pd.DataFrame:
    with requests.Session() as session, ThreadPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(lambda s: daily_klines(s, start_ms, session), symbols))
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
