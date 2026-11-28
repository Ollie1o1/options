"""Offline, aggregate-only test specified in docs/EMA_BREAKOUT_STUDY.md."""
from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = {5: (5, 10), 10: (20, 50), 20: (50, 100)}
TRAIN_END = "2022-12-31"
HOLDOUT_START = "2024-01-01"
DATA_END = "2026-11-30"


def observation_frame(close: pd.Series, anchor: np.ndarray, horizon: int) -> pd.DataFrame:
    """Use a fixed trailing level and require complete forecast horizons."""
    level = close.shift().rolling(20).max()
    distance = level / close - 1
    vol = close.pct_change(fill_method=None).rolling(20).std(ddof=0)
    eligible = anchor & (close.rolling(100).count() == 100) & (distance >= 0)
    future = pd.concat([close.shift(-i) for i in range(1, horizon + 1)], axis=1)
    observed = future.notna().all(axis=1)
    bucket = 3 * np.searchsorted([.02, .05], distance, side="left") + np.searchsorted([.01, .02], vol)
    frame = pd.DataFrame({
        "bucket": bucket,
        "observed": observed,
        "hit": future.max(axis=1) >= level * 1.01,
        **{str(span): close > close.ewm(span=span, adjust=False, ignore_na=True).mean() for span in ARMS[horizon]},
    }, index=close.index)
    return frame.loc[eligible]


def _result(frame: pd.DataFrame, sessions: pd.Index, horizon: int) -> dict:
    maturity = pd.Series(sessions, index=sessions).shift(-horizon).fillna("9999-12-31").reindex(frame.index)
    train = frame.loc[(frame.index >= "2010-01-01") & (maturity.to_numpy() <= TRAIN_END) & frame.observed]
    holdout = frame.loc[frame.index >= HOLDOUT_START]
    test = holdout.loc[holdout.observed]
    if train.empty or test.empty:
        raise ValueError(f"Insufficient data for the {horizon}-session horizon")
    bins = train.bucket.to_numpy()
    counts = np.bincount(bins, minlength=9)
    hits = np.bincount(bins, weights=train.hit.to_numpy(), minlength=9)
    baseline = (1 + hits) / (2 + counts)
    baseline[counts == 0] = (1 + train.hit.sum()) / (2 + len(train))
    errors = (test.hit.to_numpy() - baseline[test.bucket.to_numpy()]) ** 2
    baseline_brier = float(errors.mean())

    arms = []
    for span in ARMS[horizon]:
        keys = 2 * bins + train[str(span)].to_numpy()
        n = np.bincount(keys, minlength=18)
        k = np.bincount(keys, weights=train.hit.to_numpy(), minlength=18)
        rates = np.where(n >= 100, (k + 1) / (n + 2), np.repeat(baseline, 2))
        estimates = rates[2 * test.bucket.to_numpy() + test[str(span)].to_numpy()]
        gain = errors - (test.hit.to_numpy() - estimates) ** 2
        weeks = pd.Series(gain, index=test.index).groupby(level=0).agg(["sum", "size"])
        # Keep the session grid intact even if a week has no eligible outcomes.
        anchors = sessions[::5]
        anchors = anchors[(anchors >= weeks.index[0]) & (anchors <= weeks.index[-1])]
        blocks = weeks.reindex(anchors, fill_value=0).to_numpy()
        rng = np.random.default_rng(2846)
        starts = rng.integers(len(blocks), size=(3000, (len(blocks) + 3) // 4, 1))
        draw = ((starts + np.arange(4)) % len(blocks)).reshape(3000, -1)[:, :len(blocks)]
        totals = blocks[draw].sum(axis=1)
        lower, upper = np.quantile(totals[totals[:, 1] > 0, 0] / totals[totals[:, 1] > 0, 1], [1/240, 239/240])
        skill = float(gain.mean() / baseline_brier)
        split = test.index < "2025-01-01"
        early, late = float(gain[split].mean()), float(gain[~split].mean())
        arms.append({
            "ema_sessions": span, "brier_skill": skill, "brier_gain_interval": [float(lower), float(upper)],
            "gain_2024": early, "gain_2025_26": late,
            "follow_up": bool(lower > 0 and skill >= .01 and early > 0 and late > 0),
        })
    return {"horizon_sessions": horizon, "holdout_observations": len(test), "training_observations": len(train),
            "unresolved_observations": len(holdout) - len(test), "holdout_breakout_rate": float(test.hit.mean()),
            "baseline_brier": baseline_brier, "comparisons": arms}


def study(db_path: Path) -> dict:
    uri = db_path.resolve().as_uri() + "?mode=ro"
    params = ("2009-01-01", DATA_END)
    frames: dict[int, list[pd.DataFrame]] = {h: [] for h in ARMS}
    symbols = 0
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        sessions = pd.Index([d for d, in conn.execute(
            "SELECT date FROM ohlcv WHERE ticker='SPY' AND date BETWEEN ? AND ? ORDER BY date", params
        )])
        if sessions.empty:
            raise ValueError("Study calendar unavailable")
        anchor = np.arange(len(sessions)) % 5 == 0
        rows = conn.execute("SELECT ticker,date,close FROM ohlcv WHERE date BETWEEN ? AND ? ORDER BY ticker,date", params)
        for _, group in groupby(rows, key=lambda row: row[0]):
            raw = pd.Series({day: price for _, day, price in group}, dtype=float)
            prices = raw.where(np.isfinite(raw) & (raw > 0)).reindex(sessions)
            if prices.count() < 100:
                continue
            symbols += 1
            for horizon in frames:
                frames[horizon].append(observation_frame(prices, anchor, horizon))
    return {"last_cached_session": sessions[-1], "symbols_with_history": symbols,
            "horizons": [_result(pd.concat(frames[h]), sessions, h) for h in frames]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/equity_ohlcv.db"))
    parser.add_argument("--out", type=Path, help="Write the historical aggregate report to JSON")
    options = parser.parse_args()
    report = json.dumps(study(options.db), indent=2, allow_nan=False) + "\n"
    if options.out:
        options.out.write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
