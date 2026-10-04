import pandas as pd

from cryptosurge import binance


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise binance.requests.HTTPError(str(self.status_code))


def _kline(i):
    day = binance.DAY_MS
    # Field order of a /api/v3/klines row: open time, open, high, low, close, volume, close time,
    # quote volume, number of trades, taker buy base volume, taker buy quote volume, ignore.
    return [i * day, "1.0", "2.0", "0.5", "1.5", "10", i * day + day - 1, "15.0", 42, "4", "6.0", "0"]


def test_daily_klines_parses_binance_rows(monkeypatch):
    rows = [_kline(i) for i in range(3)]
    monkeypatch.setattr(binance.requests, "get", lambda url, params=None, timeout=None: FakeResponse(rows))
    df = binance.daily_klines("ABCUSDT")
    assert list(df.columns) == [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "quote_volume",
        "trades",
        "taker_buy_quote",
    ]
    assert len(df) == 3 and df["date"].iloc[0] == pd.Timestamp("1970-01-01")
    assert df["close"].iloc[0] == 1.5 and df["trades"].iloc[0] == 42 and df["quote_volume"].iloc[0] == 15.0


def test_eligible_symbols_filters_stablecoins_and_leveraged(monkeypatch):
    info = {
        "symbols": [
            {"symbol": "BTCUSDT", "baseAsset": "BTC", "quoteAsset": "USDT", "status": "TRADING"},
            {"symbol": "USDCUSDT", "baseAsset": "USDC", "quoteAsset": "USDT", "status": "TRADING"},
            {"symbol": "ETHUPUSDT", "baseAsset": "ETHUP", "quoteAsset": "USDT", "status": "TRADING"},
            {"symbol": "ETHBTC", "baseAsset": "ETH", "quoteAsset": "BTC", "status": "TRADING"},
            {"symbol": "OLDUSDT", "baseAsset": "OLD", "quoteAsset": "USDT", "status": "BREAK"},
        ]
    }
    monkeypatch.setattr(binance.requests, "get", lambda url, params=None, timeout=None: FakeResponse(info))
    assert binance.eligible_usdt_symbols() == ["BTCUSDT"]


def test_falls_back_to_second_host(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        return FakeResponse({}, 451) if "binance.vision" in url else FakeResponse({"ok": True})

    monkeypatch.setattr(binance.requests, "get", fake_get)
    monkeypatch.setattr(binance.time, "sleep", lambda s: None)
    assert binance._get("/api/v3/ping") == {"ok": True}
    assert any("api.binance.com" in c for c in calls)
