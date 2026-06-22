"""Forward-only news/qualitative overlay. The historical backtest NEVER uses
this — there is no multi-year point-in-time news archive, so the overlay's
accuracy can only be earned forward. Each live prediction is logged with its
baseline and news-adjusted probability; outcomes are filled in as horizons
elapse; accuracy is reported over resolved rows only."""
from __future__ import annotations
import sqlite3
from typing import Callable, Optional
from src.breakout import metrics as M
import numpy as np

K_TILT = 0.10
_SCHEMA = """
CREATE TABLE IF NOT EXISTS breakout_fwd (
  date TEXT, ticker TEXT, horizon TEXT,
  baseline_prob REAL, adjusted_prob REAL, realized_up REAL,
  PRIMARY KEY (date, ticker, horizon)
)"""


def _conn(db_path: str):
    c = sqlite3.connect(db_path)
    c.execute(_SCHEMA)
    return c


def tilt(baseline_prob: float, sentiment: float) -> float:
    return float(min(1.0, max(0.0, baseline_prob + K_TILT * sentiment)))


def log_prediction(db_path, date, ticker, horizon, baseline_prob, adjusted_prob) -> None:
    with _conn(db_path) as c:
        c.execute("INSERT OR IGNORE INTO breakout_fwd"
                  "(date,ticker,horizon,baseline_prob,adjusted_prob,realized_up)"
                  " VALUES (?,?,?,?,?,NULL)",
                  (date, ticker, horizon, baseline_prob, adjusted_prob))


def resolve_outcomes(db_path, as_of_date: str,
                     realized_lookup: Callable[[str, str, str], Optional[float]]) -> int:
    n = 0
    with _conn(db_path) as c:
        rows = c.execute("SELECT date,ticker,horizon FROM breakout_fwd "
                         "WHERE realized_up IS NULL").fetchall()
        for date, ticker, horizon in rows:
            r = realized_lookup(ticker, date, horizon)
            if r is not None:
                c.execute("UPDATE breakout_fwd SET realized_up=? WHERE "
                          "date=? AND ticker=? AND horizon=?", (r, date, ticker, horizon))
                n += 1
    return n


def forward_accuracy(db_path) -> dict:
    with _conn(db_path) as c:
        rows = c.execute("SELECT baseline_prob,adjusted_prob,realized_up FROM "
                         "breakout_fwd WHERE realized_up IS NOT NULL").fetchall()
    if not rows:
        return {"n": 0, "brier_adjusted": None, "brier_baseline": None, "skill": None}
    base = np.array([r[0] for r in rows]); adj = np.array([r[1] for r in rows])
    y = np.array([r[2] for r in rows])
    bb, ba = M.brier_score(base, y), M.brier_score(adj, y)
    return {"n": len(rows), "brier_adjusted": ba, "brier_baseline": bb,
            "skill": M.skill_score(ba, bb)}
