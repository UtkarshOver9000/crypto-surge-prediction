from fastapi.testclient import TestClient

from cryptosurge import api

client = TestClient(api.app)


def test_health_and_dashboard():
    assert client.get("/v1/health").json() == {"status": "ok"}
    assert "Crypto Surge Scanner" in client.get("/").text


def test_stats_come_from_model_card():
    s = client.get("/v1/stats").json()
    assert s["coins"] == 100 and s["horizon_days"] == 7
    for key in ("roc_auc", "pr_auc", "precision", "recall", "accuracy", "log_loss"):
        assert key in s["test_metrics"]
    assert "strategy" in s["backtest"] and "btc_buy_and_hold" in s["backtest"]


def test_scan_scores_latest_real_candles(monkeypatch, prices):
    monkeypatch.setattr(api, "many_daily_klines", lambda symbols, start_ms=0: prices)
    res = client.get("/v1/scan", params={"top": 3, "refresh": True})
    assert res.status_code == 200
    data = res.json()
    assert data["coins_scored"] == prices["symbol"].nunique()
    assert len(data["results"]) == 3
    probs = [r["surge_probability"] for r in data["results"]]
    assert probs == sorted(probs, reverse=True) and all(0 <= p <= 1 for p in probs)
    assert data["results"][0]["as_of"] == str(prices["date"].max().date())


def test_scan_reports_upstream_failure(monkeypatch):
    def boom(symbols, start_ms=0):
        raise RuntimeError("451 restricted location")

    monkeypatch.setattr(api, "many_daily_klines", boom)
    assert client.get("/v1/scan", params={"refresh": True}).status_code == 502
