"""
FastAPI service: live surge scan of the coin universe.

The models are trained offline (``python -m cryptosurge.train``) and loaded from
``src/cryptosurge/artifacts``. ``/v1/scan`` downloads the last 150 daily
candles of every coin from Binance's public API, rebuilds the same features
used in training, and scores the most recent completed day.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from . import __version__
from .binance import DAY_MS, many_daily_klines
from .features import FEATURE_COLS, build_features

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DASHBOARD = Path(__file__).resolve().parent / "dashboard" / "index.html"
LOOKBACK_DAYS = 150
CACHE_SECONDS = int(os.getenv("SCAN_CACHE_SECONDS", "1800"))

app = FastAPI(
    title="Crypto Surge Scanner",
    description=(
        "Probability that each of 100 Binance USDT pairs closes at least 15% higher in 7 days, from a model "
        "trained on the full daily history of those coins. Research tool, not financial advice."
    ),
    version=__version__,
    license_info={"name": "MIT"},
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

_scan_cache: dict[str, Any] = {"at": 0.0, "payload": None}


@lru_cache(maxsize=1)
def load_artifacts() -> tuple[dict, dict]:
    models = joblib.load(ARTIFACT_DIR / "models.joblib")
    card = json.loads((ARTIFACT_DIR / "model_card.json").read_text())
    return models, card


def score_latest(prices) -> list[dict]:
    """Score the most recent date in ``prices`` (raw daily candles for the universe)."""
    models, card = load_artifacts()
    feats = build_features(prices, include_labels=False)
    latest = feats[feats["date"] == feats["date"].max()].copy()
    x = latest[FEATURE_COLS]
    latest["surge_probability"] = models["classifier"].predict_proba(x)[:, 1]
    latest["predicted_7d_return"] = models["regressor"].predict(x)
    latest = latest.sort_values("surge_probability", ascending=False)
    return [
        {
            "symbol": r.symbol,
            "as_of": str(r.date.date()),
            "close": float(r.close),
            "surge_probability": round(float(r.surge_probability), 4),
            "signal": bool(r.surge_probability >= card["threshold"]),
            "predicted_7d_return": round(float(r.predicted_7d_return), 4),
            "return_last_7d": round(float(r.ret_7d), 4),
        }
        for r in latest.itertuples()
    ]


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard():
    if DASHBOARD.exists():
        return FileResponse(str(DASHBOARD), media_type="text/html")
    return HTMLResponse("<h1>Crypto Surge Scanner</h1><p>See <a href='/docs'>/docs</a></p>")


@app.get("/v1/health", include_in_schema=False)
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/stats", tags=["Model"])
def stats() -> dict[str, Any]:
    """Held-out test metrics and backtest of the deployed model."""
    _, card = load_artifacts()
    return {k: v for k, v in card.items() if k not in ("symbols", "features")} | {"coins": len(card["symbols"])}


@app.get("/v1/scan", tags=["Live"])
def scan(top: int = Query(20, ge=1, le=100), refresh: bool = False) -> dict[str, Any]:
    """Live scan: every coin's surge probability for the latest completed daily candle."""
    now = time.time()
    if refresh or _scan_cache["payload"] is None or now - _scan_cache["at"] > CACHE_SECONDS:
        _, card = load_artifacts()
        start_ms = int((now * 1000) - LOOKBACK_DAYS * DAY_MS)
        try:
            prices = many_daily_klines(card["symbols"], start_ms=start_ms)
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=f"Binance data unavailable: {exc}") from exc
        if prices.empty:
            raise HTTPException(status_code=502, detail="Binance returned no candles")
        results = score_latest(prices)
        _scan_cache.update(
            at=now,
            payload={
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "threshold": card["threshold"],
                "coins_scored": len(results),
                "results": results,
            },
        )
    payload = dict(_scan_cache["payload"])
    payload["results"] = payload["results"][:top]
    return payload
