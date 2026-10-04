"""
Train and evaluate the surge models on the full Binance daily history.

    python -m cryptosurge.train --data-dir data

Chronological split by date, with a 7-day embargo after train and after
validation so no label window crosses into the next period:

    train  |embargo|  validation (15%)  |embargo|  test (last 20% of dates)

Validation picks the boosting round, the MLP epoch and the decision threshold.
The test period is used once, for the reported metrics and the trading backtest.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .backtest import run_backtest
from .download import META_FILE, PRICES_FILE
from .features import FEATURE_COLS, LABEL_COL, NEXT_RETURN_COL, SURGE_HORIZON_DAYS, SURGE_THRESHOLD, build_features

PACKAGE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PACKAGE_DIR / "artifacts"
EMBARGO_DAYS = SURGE_HORIZON_DAYS


def chronological_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.20) -> dict:
    dates = np.sort(df["date"].unique())
    n = len(dates)
    test_start = dates[int(n * (1 - test_frac))]
    val_start = dates[int(n * (1 - test_frac - val_frac))]
    embargo = pd.Timedelta(days=EMBARGO_DAYS)
    parts = {
        "train": df[df["date"] < val_start - embargo],
        "val": df[(df["date"] >= val_start) & (df["date"] < test_start - embargo)],
        "test": df[df["date"] >= test_start],
    }
    return {k: v.reset_index(drop=True) for k, v in parts.items()}


def _clip(p):
    return np.clip(p, 1e-7, 1 - 1e-7)


def classification_metrics(y, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "rows": int(len(y)),
        "base_rate": round(float(np.mean(y)), 4),
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "pr_auc": round(average_precision_score(y, proba), 4),
        "log_loss": round(log_loss(y, _clip(proba), labels=[0, 1]), 4),
        "brier": round(brier_score_loss(y, proba), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_f1_threshold(y, proba) -> float:
    grid = np.unique(np.quantile(proba, np.linspace(0.5, 0.999, 300)))
    scores = [f1_score(y, (proba >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def train_boosting(xt, yt, xv, yv, max_rounds: int = 800, seed: int = 7):
    params = dict(
        learning_rate=0.03,
        max_leaf_nodes=31,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )
    full = HistGradientBoostingClassifier(max_iter=max_rounds, **params).fit(xt, yt)
    train_curve = [log_loss(yt, _clip(p[:, 1]), labels=[0, 1]) for p in full.staged_predict_proba(xt)]
    val_curve = [log_loss(yv, _clip(p[:, 1]), labels=[0, 1]) for p in full.staged_predict_proba(xv)]
    best = int(np.argmin(val_curve)) + 1
    model = HistGradientBoostingClassifier(max_iter=best, **params).fit(xt, yt)
    return model, {
        "rounds_trained": max_rounds,
        "best_round": best,
        "train_log_loss_per_round": [round(v, 5) for v in train_curve],
        "val_log_loss_per_round": [round(v, 5) for v in val_curve],
    }


def train_mlp(xt, yt, xv, yv, max_epochs: int = 60, patience: int = 6, seed: int = 7):
    scaler = StandardScaler().fit(xt)
    st, sv = scaler.transform(xt), scaler.transform(xv)
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32), alpha=1e-3, batch_size=1024, learning_rate_init=1e-3, random_state=seed
    )
    train_curve, val_curve, best, best_state, bad = [], [], np.inf, None, 0
    for _ in range(max_epochs):
        clf.partial_fit(st, yt, classes=[0, 1])
        train_curve.append(log_loss(yt, _clip(clf.predict_proba(st)[:, 1]), labels=[0, 1]))
        val_curve.append(log_loss(yv, _clip(clf.predict_proba(sv)[:, 1]), labels=[0, 1]))
        if val_curve[-1] < best - 1e-5:
            best, bad = val_curve[-1], 0
            best_state = ([w.copy() for w in clf.coefs_], [b.copy() for b in clf.intercepts_])
        else:
            bad += 1
            if bad >= patience:
                break
    clf.coefs_, clf.intercepts_ = best_state
    return Pipeline([("scale", scaler), ("clf", clf)]), {
        "epochs_run": len(val_curve),
        "best_epoch": int(np.argmin(val_curve)) + 1,
        "train_log_loss_per_epoch": [round(v, 5) for v in train_curve],
        "val_log_loss_per_epoch": [round(v, 5) for v in val_curve],
    }


def train_regressor(xt, rt, xv, rv, max_rounds: int = 600, seed: int = 7):
    params = dict(
        loss="absolute_error",
        learning_rate=0.03,
        max_leaf_nodes=31,
        min_samples_leaf=200,
        early_stopping=False,
        random_state=seed,
    )
    full = HistGradientBoostingRegressor(max_iter=max_rounds, **params).fit(xt, rt)
    val_curve = [mean_absolute_error(rv, p) for p in full.staged_predict(xv)]
    train_curve = [mean_absolute_error(rt, p) for p in full.staged_predict(xt)]
    best = int(np.argmin(val_curve)) + 1
    model = HistGradientBoostingRegressor(max_iter=best, **params).fit(xt, rt)
    return model, {
        "best_round": best,
        "train_mae_per_round": [round(v, 5) for v in train_curve],
        "val_mae_per_round": [round(v, 5) for v in val_curve],
    }


def regression_metrics(df: pd.DataFrame, pred: np.ndarray) -> dict:
    y = df[NEXT_RETURN_COL].to_numpy()
    zero = np.zeros_like(y)
    ic = (
        pd.DataFrame({"date": df["date"], "y": y, "p": pred})
        .groupby("date")
        .apply(lambda g: g["y"].rank().corr(g["p"].rank()) if len(g) > 4 else np.nan, include_groups=False)
        .dropna()
    )
    return {
        "mae": round(float(mean_absolute_error(y, pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y, pred))), 4),
        "mae_predicting_zero": round(float(mean_absolute_error(y, zero)), 4),
        "direction_accuracy": round(float((np.sign(pred) == np.sign(y)).mean()), 4),
        "mean_daily_rank_ic": round(float(ic.mean()), 4),
        "share_of_days_with_positive_ic": round(float((ic > 0).mean()), 4),
    }


def save_figures(hist: dict, backtest: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    b = hist["gradient_boosting"]
    axes[0].plot(b["train_log_loss_per_round"], label="train")
    axes[0].plot(b["val_log_loss_per_round"], label="validation")
    axes[0].axvline(b["best_round"] - 1, color="grey", ls="--", label=f"kept: round {b['best_round']}")
    axes[0].set(title="Surge classifier (boosting): log loss per round", xlabel="round", ylabel="log loss")
    axes[0].legend()
    m = hist["mlp"]
    epochs = np.arange(1, len(m["val_log_loss_per_epoch"]) + 1)
    axes[1].plot(epochs, m["train_log_loss_per_epoch"], marker="o", label="train")
    axes[1].plot(epochs, m["val_log_loss_per_epoch"], marker="o", label="validation")
    axes[1].axvline(m["best_epoch"], color="grey", ls="--", label=f"kept: epoch {m['best_epoch']}")
    axes[1].set(title="Surge classifier (MLP): log loss per epoch", xlabel="epoch", ylabel="log loss")
    axes[1].legend()
    r = hist["return_regressor"]
    axes[2].plot(r["train_mae_per_round"], label="train")
    axes[2].plot(r["val_mae_per_round"], label="validation")
    axes[2].axvline(r["best_round"] - 1, color="grey", ls="--", label=f"kept: round {r['best_round']}")
    axes[2].set(title="7-day return regressor: MAE per round", xlabel="round", ylabel="MAE")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=130)
    plt.close(fig)

    curves = backtest["equity_curves"]
    dates = pd.to_datetime(curves["dates"])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, curves["strategy"], label="surge strategy (after 0.1% fees)")
    ax.plot(dates, curves["btc_buy_and_hold"], label="BTC buy and hold")
    ax.plot(dates, curves["equal_weight_all_coins_no_fees"], label="equal-weight 100 coins (no fees)")
    ax.set_yscale("log")
    ax.set(title="Held-out test period: growth of $1 (log scale)", ylabel="value of $1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "backtest_equity.png", dpi=130)
    plt.close(fig)


def run(data_dir: Path, reports_dir: Path, seed: int = 7) -> dict:
    t0 = time.time()
    raw = pd.read_csv(Path(data_dir) / PRICES_FILE, parse_dates=["date"])
    meta = json.loads((Path(data_dir) / META_FILE).read_text())
    df = build_features(raw)
    split = chronological_split(df)
    x = {k: v[FEATURE_COLS] for k, v in split.items()}
    y = {k: v[LABEL_COL].to_numpy() for k, v in split.items()}

    models, hist = {}, {}
    lr = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]).fit(x["train"], y["train"])
    models["logistic_regression"] = lr
    hist["logistic_regression"] = {"solver_iterations": int(lr.named_steps["clf"].n_iter_[0])}
    models["gradient_boosting"], hist["gradient_boosting"] = train_boosting(
        x["train"], y["train"], x["val"], y["val"], seed=seed
    )
    models["mlp"], hist["mlp"] = train_mlp(x["train"], y["train"], x["val"], y["val"], seed=seed)

    results, thresholds = {}, {}
    for name, model in models.items():
        p_val = model.predict_proba(x["val"])[:, 1]
        thresholds[name] = best_f1_threshold(y["val"], p_val)
        results[name] = {
            "test": classification_metrics(y["test"], model.predict_proba(x["test"])[:, 1], thresholds[name]),
            "validation": classification_metrics(y["val"], p_val, thresholds[name]),
            "train": classification_metrics(y["train"], model.predict_proba(x["train"])[:, 1], thresholds[name]),
        }
    prior = float(np.mean(y["train"]))
    results["baseline_always_no_surge"] = {
        "test": classification_metrics(y["test"], np.full(len(y["test"]), prior), 0.5),
    }

    # Same features, different question: will the coin move >= 15% in EITHER direction within 7 days?
    big_y = {k: (v[NEXT_RETURN_COL].abs() >= SURGE_THRESHOLD).astype(int).to_numpy() for k, v in split.items()}
    big_model, big_hist = train_boosting(x["train"], big_y["train"], x["val"], big_y["val"], seed=seed)
    big_threshold = best_f1_threshold(big_y["val"], big_model.predict_proba(x["val"])[:, 1])
    big_move = {
        "question": f"|close in {SURGE_HORIZON_DAYS} days / close today - 1| >= {SURGE_THRESHOLD:.0%}",
        "best_round": big_hist["best_round"],
        "test": classification_metrics(big_y["test"], big_model.predict_proba(x["test"])[:, 1], big_threshold),
    }

    reg, hist["return_regressor"] = train_regressor(
        x["train"], split["train"][NEXT_RETURN_COL], x["val"], split["val"][NEXT_RETURN_COL], seed=seed
    )
    regression = regression_metrics(split["test"], reg.predict(x["test"]))

    shipped = "gradient_boosting"
    scored = split["test"][["date", "symbol", NEXT_RETURN_COL]].copy()
    scored["surge_probability"] = models[shipped].predict_proba(x["test"])[:, 1]
    backtest = run_backtest(scored, threshold=thresholds[shipped], seed=seed)

    report = {
        "data": {
            **{k: v for k, v in meta.items() if k != "symbols"},
            "coins": len(meta["symbols"]),
            "labelled_rows": int(len(df)),
            "label": f"close {SURGE_HORIZON_DAYS} days later >= +{SURGE_THRESHOLD:.0%} vs today's close",
        },
        "split": {
            k: {
                "rows": int(len(v)),
                "from": str(v["date"].min().date()),
                "to": str(v["date"].max().date()),
                "surge_rate": round(float(v[LABEL_COL].mean()), 4),
            }
            for k, v in split.items()
        },
        "embargo_days": EMBARGO_DAYS,
        "features": FEATURE_COLS,
        "training": hist,
        "classification": results,
        "shipped_model": shipped,
        "shipped_threshold": round(thresholds[shipped], 4),
        "return_regression_test": regression,
        "big_move_either_direction": big_move,
        "backtest_test_period": backtest,
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
    }
    save_figures(hist, backtest, reports_dir / "figures")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": models[shipped], "regressor": reg}, ARTIFACT_DIR / "models.joblib", compress=3)
    card = {
        "classifier": shipped,
        "threshold": report["shipped_threshold"],
        "features": FEATURE_COLS,
        "symbols": meta["symbols"],
        "horizon_days": SURGE_HORIZON_DAYS,
        "surge_threshold": SURGE_THRESHOLD,
        "trained_through": report["split"]["train"]["to"],
        "test_period": [report["split"]["test"]["from"], report["split"]["test"]["to"]],
        "test_metrics": results[shipped]["test"],
        "return_regression_test": regression,
        "big_move_either_direction_test": big_move["test"],
        "backtest": {k: v for k, v in backtest.items() if k != "equity_curves"},
        "data_fetched_at": meta["fetched_at"],
        "scikit_learn": sklearn.__version__,
    }
    (ARTIFACT_DIR / "model_card.json").write_text(json.dumps(card, indent=2))

    report["runtime_seconds"] = round(time.time() - t0, 1)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train surge models on Binance daily history")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    report = run(args.data_dir, args.reports_dir)
    print(json.dumps({k: v["test"] for k, v in report["classification"].items()}, indent=1))
    print(json.dumps(report["return_regression_test"], indent=1))
    print(json.dumps(report["big_move_either_direction"], indent=1))
    print(json.dumps({k: v for k, v in report["backtest_test_period"].items() if k != "equity_curves"}, indent=1))
    print(json.dumps(report["split"], indent=1))


if __name__ == "__main__":
    main()
