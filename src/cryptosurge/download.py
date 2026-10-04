"""
Download the full daily price history of the coin universe from Binance.

    python -m cryptosurge.download --out data

Universe: the ``--top`` USDT spot pairs by current 24h quote volume that have at
least ``--min-days`` of daily history (stablecoins and leveraged tokens excluded).
Every completed daily candle since each pair's listing is downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from .binance import BASE_URLS, eligible_usdt_symbols, many_daily_klines, rank_by_quote_volume

PRICES_FILE = "binance_daily.csv.gz"
META_FILE = "binance_daily_meta.json"


def build_universe(top: int, min_days: int) -> tuple[list[str], pd.DataFrame]:
    with requests.Session() as session:
        ranked = rank_by_quote_volume(eligible_usdt_symbols(session), session)
    candidates = ranked[: top * 3]  # fetch extra so the history filter still leaves `top`
    prices = many_daily_klines(candidates)
    days = prices.groupby("symbol")["date"].nunique()
    keep = [s for s in candidates if days.get(s, 0) >= min_days][:top]
    return keep, prices[prices["symbol"].isin(keep)].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Binance daily candles for the coin universe")
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--min-days", type=int, default=730)
    args = parser.parse_args()

    fetched_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    symbols, prices = build_universe(args.top, args.min_days)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / PRICES_FILE
    prices.sort_values(["symbol", "date"]).to_csv(path, index=False, compression="gzip")
    meta = {
        "source": BASE_URLS[0] + "/api/v3/klines (interval=1d)",
        "fetched_at": fetched_at,
        "universe_rule": f"top {args.top} USDT spot pairs by 24h quote volume with >= {args.min_days} daily candles",
        "symbols": symbols,
        "rows": int(len(prices)),
        "first_date": str(prices["date"].min().date()),
        "last_date": str(prices["date"].max().date()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (args.out / META_FILE).write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "symbols"}, indent=2))


if __name__ == "__main__":
    main()
