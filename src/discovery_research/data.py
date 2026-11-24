"""Read-only inputs and strictly past-looking features for channel research."""
from __future__ import annotations

import csv
import hashlib
import math
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def readonly(path: str) -> sqlite3.Connection:
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def load_prices(path: str, symbols: list[str], start: str) -> tuple[pd.DataFrame, dict]:
    """Use the adjusted daily-price cache; never repair large jumps by guessing.

    Windows touching missing/invalid prices or a >45% daily move are excluded
    later. This protects the study from unverified splits without rewriting
    genuine crashes into winning observations.
    """
    with closing(readonly(path)) as con:
        marks = ",".join("?" for _ in symbols)
        rows = con.execute(
            f"SELECT ticker,date,close FROM ohlcv WHERE ticker IN ({marks}) "
            "AND date>=? ORDER BY date,ticker", [*symbols, start]).fetchall()
    frame = pd.DataFrame(rows, columns=["symbol", "date", "close"])
    if frame.empty:
        return pd.DataFrame(), {"source": path, "symbols": 0, "error": "no prices"}
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    prices = frame.pivot(index="date", columns="symbol", values="close").sort_index()
    if "SPY" not in prices:
        raise ValueError("SPY is required to define the trading calendar and benchmark")
    prices = prices.loc[prices["SPY"].notna()]
    coverage = {"source": path, "first": str(prices.index.min()), "last": str(prices.index.max()),
                "sessions": len(prices), "symbols": len(prices.columns),
                "missing_symbols": sorted(set(symbols) - set(prices.columns)),
                "adjustment_policy": "cached adjusted closes; exclude windows with daily changes exceeding 45% in magnitude"}
    return prices, coverage


def load_cboe_csv(path: str) -> tuple[pd.Series, dict]:
    file = Path(path)
    if not file.exists():
        return pd.Series(dtype=float), {"source": path, "error": "not downloaded"}
    values: dict[str, float] = {}
    from datetime import datetime
    with file.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                day = datetime.strptime(row["DATE"], "%m/%d/%Y").date().isoformat()
                value = float(row.get("CLOSE") or row.get("VVIX") or row.get("SKEW") or "nan")
            except (ValueError, KeyError):
                continue
            if math.isfinite(value) and value > 0:
                values[day] = value
    series = pd.Series(values, dtype=float).sort_index()
    return series, {"source": path, "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                    "url": ("https://cdn.cboe.com/api/global/us_indices/daily_prices/" + file.name)
                    if file.name in ("VIX_History.csv", "VIX3M_History.csv", "VVIX_History.csv", "SKEW_History.csv") else None,
                    "observations": len(series), "first": str(series.index.min()) if len(series) else None,
                    "last": str(series.index.max()) if len(series) else None}


def load_news(path: str, symbols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Availability is first archival, not the potentially revised publication date."""
    if not Path(path).exists():
        return pd.DataFrame(), {"source": path, "error": "not present"}
    with closing(readonly(path)) as con:
        marks = ",".join("?" for _ in symbols)
        rows = con.execute(
            "SELECT symbol,archived_at,sentiment,relevance,headline FROM news_archive "
            f"WHERE symbol IN ({marks}) AND archived_at IS NOT NULL ORDER BY archived_at", symbols).fetchall()
    frame = pd.DataFrame(rows, columns=["symbol", "first_seen", "sentiment", "relevance", "headline"])
    if frame.empty:
        return frame, {"source": path, "rows": 0}
    # A repeated/syndicated headline is one observation for that symbol/day.
    frame["date"] = frame["first_seen"].str.slice(0, 10)
    frame["normalized"] = frame["headline"].fillna("").str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    frame = frame.drop_duplicates(["symbol", "date", "normalized"])
    return frame, {"source": path, "rows": len(frame), "dates": int(frame["date"].nunique()),
                   "first": str(frame["date"].min()), "last": str(frame["date"].max()),
                   "availability": "first archived_at; publication dates not used"}


def load_analysts(path: str, symbols: list[str]) -> tuple[pd.DataFrame, dict]:
    if not Path(path).exists():
        return pd.DataFrame(), {"source": path, "error": "not present"}
    with closing(readonly(path)) as con:
        marks = ",".join("?" for _ in symbols)
        rows = con.execute(
            "SELECT symbol,first_seen,action FROM analyst_actions "
            f"WHERE symbol IN ({marks}) AND first_seen IS NOT NULL ORDER BY first_seen", symbols).fetchall()
    frame = pd.DataFrame(rows, columns=["symbol", "first_seen", "action"])
    if frame.empty:
        return frame, {"source": path, "rows": 0}
    frame["date"] = frame["first_seen"].str.slice(0, 10)
    return frame, {"source": path, "rows": len(frame), "dates": int(frame["date"].nunique()),
                   "first": str(frame["date"].min()), "last": str(frame["date"].max()),
                   "availability": "first_seen; grade_date not used"}


def _valid_window(values: np.ndarray) -> bool:
    if len(values) < 2 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        return False
    return bool(np.all(np.abs(values[1:] / values[:-1] - 1) <= .45))


def _event_features(news: pd.DataFrame, analysts: pd.DataFrame, calendar: list[str]) -> dict:
    """Weekend arrivals become next-session observations, with next-close entry.

    Absence of an archived story does not prove that ticker was fetched: it is
    missing coverage, never a synthetic zero-news sample.
    """
    out: dict[tuple[str, str], dict[str, float]] = {}
    if not news.empty:
        frame = news.copy()
        frame["session"] = frame["date"].map(lambda d: _session_on_or_after(calendar, str(d)))
        if "normalized" in frame:
            # Multiple weekend archive dates can map to one feature session.
            frame = frame.drop_duplicates(["session", "symbol", "normalized"])
        frame["sentiment"] = pd.to_numeric(frame["sentiment"], errors="coerce")
        frame["relevance"] = pd.to_numeric(frame["relevance"], errors="coerce").fillna(0).clip(lower=0)
        for (session, sym), group in frame.groupby(["session", "symbol"]):
            valid = group.loc[np.isfinite(group["sentiment"])]
            if valid.empty:
                continue
            weights = valid["relevance"].to_numpy(dtype=float)
            values = valid["sentiment"].to_numpy(dtype=float)
            tone = float(np.average(values, weights=weights)) if weights.sum() > 0 else float(values.mean())
            out[(str(session), str(sym))] = {"news_tone": tone, "news_activity": float(np.log1p(len(valid))),
                                            "news_disagreement": float(values.std(ddof=0))}
    if not analysts.empty:
        frame = analysts.copy()
        frame["session"] = frame["date"].map(lambda d: _session_on_or_after(calendar, str(d)))
        for (session, sym), group in frame.groupby(["session", "symbol"]):
            actions = group["action"].astype(str).str.lower()
            net = float((actions == "upgrade").sum() - (actions == "downgrade").sum())
            out.setdefault((str(session), str(sym)), {})["analyst_net"] = net
    return out


def _session_on_or_after(calendar: list[str], day: str) -> Any:
    import bisect
    i = bisect.bisect_left(calendar, day)
    return calendar[i] if i < len(calendar) else None


def make_features(prices: pd.DataFrame, vix: pd.Series, news: pd.DataFrame,
                  analysts: pd.DataFrame, groups: dict) -> pd.DataFrame:
    calendar = [str(d) for d in prices.index]
    events = _event_features(news, analysts, calendar)
    rows: list[dict[str, Any]] = []
    market = prices["SPY"].to_numpy(dtype=float)
    vix_values = vix.reindex(prices.index).to_numpy(dtype=float)
    symbols = {sym: prices[sym].to_numpy(dtype=float) for sym in prices.columns}
    group_for = {sym: members for members in groups.values() for sym in members}
    for sym, values in symbols.items():
        for i in range(20, len(calendar)):
            window, market_window = values[i - 20:i + 1], market[i - 20:i + 1]
            if not _valid_window(window) or not _valid_window(market_window):
                continue
            lr = np.diff(np.log(window))
            rv20, rv5 = float(lr.std(ddof=1) * math.sqrt(252)), float(lr[-5:].std(ddof=1) * math.sqrt(252))
            rec: dict[str, Any] = {"date": calendar[i], "symbol": sym, "index": i,
                                   "ret5": values[i] / values[i - 5] - 1,
                                   "ret20": values[i] / values[i - 20] - 1,
                                   "market_ret5": market[i] / market[i - 5] - 1,
                                   "log_rv20": math.log(max(rv20, 1e-8)), "log_rv5": math.log(max(rv5, 1e-8))}
            if np.isfinite(vix_values[i]) and vix_values[i] > 0:
                rec["log_vix"] = math.log(vix_values[i] / 100)
                # VIX spikes are genuine observations, not equity splits.
                vix_window = vix_values[i - 5:i + 1]
                if np.all(np.isfinite(vix_window)) and np.all(vix_window > 0):
                    rec["vix_change5"] = vix_values[i] / vix_values[i - 5] - 1
            peers = [symbols[p] for p in group_for.get(sym, []) if p != sym and p in symbols]
            peer_returns = [p[i] / p[i - 5] - 1 for p in peers if _valid_window(p[i - 5:i + 1])]
            if peers and len(peer_returns) == len(peers):
                rec["peer_excess5"] = float(np.mean(peer_returns)) - rec["market_ret5"]
            rec.update(events.get((calendar[i], sym), {}))
            rows.append(rec)
    return pd.DataFrame(rows)


def make_dataset(prices: pd.DataFrame, features: pd.DataFrame, trial: dict,
                 groups: dict, entry_lag: int = 1) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    population = trial["population"]
    eligible = ({"SPY"} if population == "SPY" else
                {s for members in groups.values() for s in members} if population == "peer_groups" else set(prices.columns))
    required = trial["baseline"] + trial["candidate"]
    if any(c not in features for c in required):
        return pd.DataFrame()
    sample = features.loc[features["symbol"].isin(eligible)].dropna(subset=required).copy()
    rows: list[dict[str, Any]] = []
    calendar = [str(d) for d in prices.index]
    market = prices["SPY"].to_numpy(dtype=float)
    arrays = {s: prices[s].to_numpy(dtype=float) for s in eligible if s in prices}
    horizon = int(trial["horizon"])
    for rec in sample.to_dict("records"):
        entry = int(rec["index"]) + entry_lag
        end = entry + horizon
        values = arrays[rec["symbol"]]
        if end >= len(calendar) or not _valid_window(values[entry:end + 1]):
            continue
        if not _valid_window(market[entry:end + 1]):
            continue
        if trial["target"] == "log_future_realized_vol":
            realized = float(np.diff(np.log(values[entry:end + 1])).std(ddof=1) * math.sqrt(252))
            target = math.log(max(realized, 1e-8))
        else:
            target = values[end] / values[entry] - 1
            if trial["target"] == "forward_market_excess_return":
                target -= market[end] / market[entry] - 1
        rec.update({"target": target, "entry_date": calendar[entry], "end_date": calendar[end]})
        rows.append(rec)
    return pd.DataFrame(rows)


def relationship_snapshot(prices: pd.DataFrame, groups: dict, window: int = 252) -> list[dict]:
    """Descriptive co-movement map, controlling for the common SPY component.

    This is deliberately separate from the peer FORECAST test. A strong
    same-day correlation is useful exposure context and says nothing about
    which stock leads the next move. Both halves are printed to show instability.
    """
    import itertools
    returns = prices.pct_change(fill_method=None).tail(window)
    rows = []
    for group, members in groups.items():
        available = [s for s in members if s in returns]
        for a, b in itertools.combinations(available, 2):
            frame = returns[[a, b, "SPY"]].replace([np.inf, -np.inf], np.nan).dropna()
            frame = frame.loc[(frame.abs() <= .45).all(axis=1)]
            if len(frame) < 60:
                continue
            x = np.column_stack([np.ones(len(frame)), frame["SPY"].to_numpy(dtype=float)])
            residuals = {}
            for s in (a, b):
                y = frame[s].to_numpy(dtype=float)
                residuals[s] = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
            ra, rb = residuals[a], residuals[b]
            if min(float(ra.std()), float(rb.std())) <= 1e-12:
                continue
            half = len(frame) // 2
            def corr(left, right):
                return float(np.corrcoef(left, right)[0, 1]) if min(float(left.std()), float(right.std())) > 1e-12 else None
            rows.append({"group": group, "a": a, "b": b, "observations": len(frame),
                         "first": str(frame.index[0]), "last": str(frame.index[-1]),
                         "raw_correlation": float(frame[a].corr(frame[b])),
                         "market_adjusted_correlation": corr(ra, rb),
                         "first_half": corr(ra[:half], rb[:half]), "second_half": corr(ra[half:], rb[half:]),
                         "interpretation": "Contemporaneous return relationship; not causation, lead-lag or forecast evidence."})
    return rows
