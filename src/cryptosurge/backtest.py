"""
Weekly trading backtest of the surge signal on the held-out test period.

Every 7 days the strategy buys, in equal weights, the coins (at most
``max_positions``) whose surge probability is at or above the threshold, holds
them for exactly 7 days, then sells. Each buy and sell pays ``fee`` (Binance
spot standard fee: 0.1%). Weeks without a signal stay in cash.

Benchmarks over the same weeks:
* BTC buy-and-hold,
* an equal-weight basket of every coin, rebalanced weekly with no fees,
* random picks: the same number of coins as the strategy each week, chosen at
  random (mean of many draws), with the same fees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import BTC, NEXT_RETURN_COL, SURGE_HORIZON_DAYS

PERIODS_PER_YEAR = 365 / SURGE_HORIZON_DAYS


def _summary(weekly: pd.Series) -> dict:
    weekly = weekly.fillna(0.0)
    equity = (1 + weekly).cumprod()
    years = len(weekly) / PERIODS_PER_YEAR
    drawdown = equity / equity.cummax() - 1
    std = weekly.std(ddof=1)
    return {
        "total_return": round(float(equity.iloc[-1] - 1), 4),
        "annualized_return": round(float(equity.iloc[-1] ** (1 / years) - 1), 4) if years > 0 else None,
        "annualized_volatility": round(float(std * np.sqrt(PERIODS_PER_YEAR)), 4),
        "sharpe": round(float(weekly.mean() / std * np.sqrt(PERIODS_PER_YEAR)), 3) if std > 0 else None,
        "max_drawdown": round(float(drawdown.min()), 4),
        "weeks": int(len(weekly)),
    }


def run_backtest(
    scored: pd.DataFrame,
    threshold: float,
    max_positions: int = 5,
    fee: float = 0.001,
    random_draws: int = 500,
    seed: int = 7,
) -> dict:
    """``scored`` needs date, symbol, surge_probability and next_return columns."""
    dates = np.sort(scored["date"].unique())
    rebalance = dates[::SURGE_HORIZON_DAYS]
    rng = np.random.default_rng(seed)
    round_trip = (1 - fee) ** 2

    strat, btc, basket, random_mean, picks_log = [], [], [], [], []
    for day in rebalance:
        today = scored[scored["date"] == day]
        chosen = today[today["surge_probability"] >= threshold].nlargest(max_positions, "surge_probability")
        n = len(chosen)
        strat.append(((1 + chosen[NEXT_RETURN_COL]).mean() * round_trip - 1) if n else 0.0)
        btc_row = today[today["symbol"] == BTC]
        btc.append(float(btc_row[NEXT_RETURN_COL].iloc[0]) if len(btc_row) else 0.0)
        basket.append(float(today[NEXT_RETURN_COL].mean()))
        if n:
            draws = [
                (1 + rng.choice(today[NEXT_RETURN_COL].to_numpy(), size=n, replace=False)).mean() * round_trip - 1
                for _ in range(random_draws)
            ]
            random_mean.append(float(np.mean(draws)))
            picks_log.extend(chosen[NEXT_RETURN_COL].tolist())
        else:
            random_mean.append(0.0)

    picks = np.array(picks_log)
    index = pd.to_datetime(rebalance)
    return {
        "rule": f"every {SURGE_HORIZON_DAYS} days buy up to {max_positions} coins with p >= {threshold:.3f}, "
        f"hold {SURGE_HORIZON_DAYS} days, fee {fee:.1%} per trade",
        "period": [str(index[0].date()), str(index[-1].date())],
        "strategy": _summary(pd.Series(strat, index=index)),
        "btc_buy_and_hold": _summary(pd.Series(btc, index=index)),
        "equal_weight_all_coins_no_fees": _summary(pd.Series(basket, index=index)),
        "random_picks_same_count": _summary(pd.Series(random_mean, index=index)),
        "weeks_invested": int(sum(1 for r in strat if r != 0.0)),
        "positions_taken": int(len(picks)),
        "pick_hit_rate_surge": round(float((picks >= 0.15).mean()), 4) if len(picks) else None,
        "pick_win_rate_positive": round(float((picks > 0).mean()), 4) if len(picks) else None,
        "pick_mean_7d_return_before_fees": round(float(picks.mean()), 4) if len(picks) else None,
        "pick_median_7d_return_before_fees": round(float(np.median(picks)), 4) if len(picks) else None,
        "equity_curves": {
            "dates": [str(d.date()) for d in index],
            "strategy": [round(float(v), 4) for v in np.cumprod(1 + np.array(strat))],
            "btc_buy_and_hold": [round(float(v), 4) for v in np.cumprod(1 + np.array(btc))],
            "equal_weight_all_coins_no_fees": [round(float(v), 4) for v in np.cumprod(1 + np.array(basket))],
        },
    }
