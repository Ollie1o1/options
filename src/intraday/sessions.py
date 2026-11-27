"""Per-session prices at fixed clock times from 1-minute bars.

Definitions follow docs/PREREG_INTRADAY_LATE_DAY_AND_OPEN_20261127.md:

* bar timestamps are minute STARTS;
* the price at clock time HH:MM is the close of the bar starting HH:MM-1, or, if
  that bar is missing, of the latest bar starting in the 5 minutes before HH:MM;
  otherwise NaN;
* `open` is the open of the bar starting 09:30 (NaN if absent);
* a date is a full session iff the reference symbol (SPY) has a bar starting in
  [15:55, 15:59];
* the previous close is P(16:00) of the previous trading date, and only if that
  date was itself a full session.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

ET = "America/New_York"
REGULAR_OPEN_MIN = 9 * 60 + 30
FALLBACK_MINUTES = 5


def hhmm(text: str) -> int:
    """'15:30' -> minutes after midnight."""
    h, m = text.split(":")
    return int(h) * 60 + int(m)


def with_clock(bars: pd.DataFrame) -> pd.DataFrame:
    """Add ET `date` and minute-of-day `mod` columns (bars sorted by t)."""
    local = pd.to_datetime(bars["t"], utc=True).dt.tz_convert(ET)
    out = bars.assign(date=local.dt.date, mod=(local.dt.hour * 60 + local.dt.minute).astype(int))
    return out.sort_values("t", kind="stable")


def price_at(clocked: pd.DataFrame, minute: int) -> pd.Series:
    """Per-date price at clock `minute` (see module docstring)."""
    win = clocked[(clocked["mod"] >= minute - FALLBACK_MINUTES) & (clocked["mod"] <= minute - 1)]
    return win.groupby("date")["c"].last()


def session_open(clocked: pd.DataFrame) -> pd.Series:
    return clocked[clocked["mod"] == REGULAR_OPEN_MIN].groupby("date")["o"].first()


def full_sessions(ref_clocked: pd.DataFrame, calendar: Optional[pd.DataFrame] = None) -> pd.Series:
    """Boolean per trading date: True iff a full session (amendment A1 of the prereg).

    With `calendar` (columns date, close 'HH:MM'): trading dates are the calendar's
    dates up to the last date with data, and a date is full iff its close is 16:00
    AND the reference symbol has a bar starting in [15:55, 15:59]. Without it
    (tests / legacy): trading dates are dates with a regular-session bar and only
    the bar condition applies.
    """
    regular = ref_clocked[ref_clocked["mod"] >= REGULAR_OPEN_MIN]
    late = set(regular[(regular["mod"] >= hhmm("15:55")) & (regular["mod"] <= hhmm("15:59"))]["date"])
    if calendar is None:
        dates = pd.Index(sorted(regular["date"].unique()))
        return pd.Series([d in late for d in dates], index=dates, dtype=bool)
    last = max(regular["date"]) if len(regular) else None
    cal = calendar[calendar["date"] <= last] if last is not None else calendar.iloc[:0]
    cal = cal.sort_values("date")
    first = min(regular["date"]) if len(regular) else None
    if first is not None:
        cal = cal[cal["date"] >= first]
    dates = pd.Index(list(cal["date"]))
    full = [(c == "16:00") and (d in late) for d, c in zip(cal["date"], cal["close"])]
    return pd.Series(full, index=dates, dtype=bool)


def session_table(clocked: pd.DataFrame, targets: Mapping[str, int],
                  sessions: pd.Series) -> pd.DataFrame:
    """One row per trading date in `sessions` (full or not).

    Columns: each target name, `open`, `full`, and `prev_close` (NaN unless the
    previous trading date was full and had a 16:00 price).
    """
    idx = sessions.index
    cols: Dict[str, pd.Series] = {name: price_at(clocked, m).reindex(idx) for name, m in targets.items()}
    cols["open"] = session_open(clocked).reindex(idx)
    table = pd.DataFrame(cols, index=idx)
    table["full"] = sessions.values
    close16 = price_at(clocked, hhmm("16:00")).reindex(idx)
    prev = close16.shift(1)
    prev_full = sessions.shift(1, fill_value=False).astype(bool)
    table["prev_close"] = prev.where(prev_full.values)
    return table


def build_tables(bars_by_symbol: Mapping[str, pd.DataFrame], targets: Mapping[str, int],
                 reference: str = "SPY",
                 calendar: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    """Session tables for every symbol, restricted to full sessions of `reference`."""
    clocked = {s: with_clock(b) for s, b in bars_by_symbol.items()}
    sessions = full_sessions(clocked[reference], calendar)
    out = {}
    for s, c in clocked.items():
        t = session_table(c, targets, sessions)
        out[s] = t[t["full"]].drop(columns="full")
    return out


def stack(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Long frame (date, symbol, ...) from per-symbol tables."""
    parts = []
    for s, t in tables.items():
        parts.append(t.assign(symbol=s).rename_axis("date").reset_index())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def years(dates: Iterable) -> np.ndarray:
    return np.array([d.year for d in dates])
