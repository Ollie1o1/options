"""Recompute the EV band for trades that closed before it was persisted.

WHY THIS EXISTS
---------------
`worth.assess` grades on margins it cannot yet validate. Schema 21 began
persisting `entry_ev_net` and `entry_ev_noise` so that "did CLEAR beat THIN"
would accumulate an answer from 2026-08-10 — but on 2026-08-18 that is 22 rows,
20 of them still open. n=2 closed. The question cannot be asked.

The INPUTS to those numbers are on the historical book: `entry_vega`, `strike`,
`date`, `expiration`, `entry_price`, `strategy_name`. Given daily bars sliced as
of the entry date, the band is recomputable for trades that have already closed
and carry a realised `pnl_pct`. That turns n=2 into several hundred.

WHAT IT IS NOT
--------------
Not a record of what the screener showed. It is today's model applied to
historical inputs, so it answers "does this grade discriminate" and NOT "did the
grade as displayed discriminate". Those are different claims and only the first
one is available retrospectively.

Two deliberate limits, both of which produce None rather than a number:

  MULTI-LEG STRUCTURES ARE REFUSED. A spread's EV is not its short leg's — that
  cost this repo a board where condors averaged an EV of 4. Reconstructing a
  spread needs both legs' quotes at entry, and the ledger stores one entry
  price for the structure. Bull Put / Bear Call / Iron Condor therefore get no
  reconstruction rather than a wrong one.

  THE SPREAD COST IS NOT RECONSTRUCTABLE. `spread_pct` was never persisted, and
  round-trip friction is the term that inverts rankings, so guessing it would
  poison the exact comparison this exists to enable. What is computed is
  `ev_gross` — the edge BEFORE crossing costs — which is an upper bound on net
  EV, and `sigma_gross` with it. Read it as a bound, not as production's sigma.

NULL DISCIPLINE
---------------
Every failure path returns None. A reconstruction that could not be computed is
not a reconstruction worth 0.0, and this codebase already carries the scars of a
column where zero and not-recorded were the same value.

The output goes to its OWN database (`data/ev_reconstruction.db`), never into
`entry_ev_*`. A derived number sitting in a scan-time column is indistinguishable
from a recorded one the moment anybody reads it back.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import pandas as pd

from .trade_analysis import VOL_FORECAST_RELATIVE_ERROR, vol_basis_of
from .utils import bs_price, is_short_position, safe_float

# `long_window_volatility` returns None below 120 bars rather than a noisy
# short window wearing a long-window name. Matching it here means a trade is
# refused for the same reason production would refuse it.
MIN_BARS = 120

# Structures whose entry price is one leg's. Anything else is refused.
SINGLE_LEG = {"Long Call", "Long Put", "Short Put", "Short Call", "Long Straddle"}

DEFAULT_RFR = 0.04
DB_PATH = "data/ev_reconstruction.db"


@dataclass(frozen=True)
class Recon:
    """One reconstructed band. Every field measured or derived, none guessed."""
    entry_id: int
    ticker: str
    strategy_name: str
    as_of: str
    spot: float
    vol_basis: float
    n_bars: int
    ev_gross: float          # per contract, BEFORE crossing costs
    ev_noise: float          # half-width of the band, production's formula
    sigma_gross: float       # ev_gross / ev_noise — an UPPER BOUND on sigma
    pnl_pct: Optional[float]


def _as_date(v: Any) -> Optional[dt.date]:
    try:
        return dt.date.fromisoformat(str(v)[:10])
    except (TypeError, ValueError):
        return None


def _hv_blend(hist: pd.DataFrame) -> Optional[float]:
    """The EV's vol basis, computed exactly the way `data_fetching` builds it.

    Imported rather than reimplemented: two copies of a vol rule is how this
    codebase produced a board that ranked on one number and a table on another.
    """
    from .data_fetching import (calculate_ewma_volatility,
                                calculate_historical_volatility,
                                calculate_parkinson_volatility,
                                long_window_volatility)
    hv_252d = long_window_volatility(hist)
    roll = calculate_historical_volatility(hist, period=30)
    ewma = calculate_ewma_volatility(hist, span=20)
    park = calculate_parkinson_volatility(hist, period=30)
    if roll and ewma and park:
        hv_30d: Optional[float] = 0.34 * roll + 0.33 * ewma + 0.33 * park
    elif roll and ewma:
        hv_30d = 0.5 * roll + 0.5 * ewma
    else:
        hv_30d = roll or ewma or park
    return vol_basis_of(hv_252d, hv_30d)


def reconstruct_one(trade: Mapping[str, Any],
                    hist: pd.DataFrame,
                    *,
                    rfr: float = DEFAULT_RFR) -> Optional[Recon]:
    """One trade's EV band, or None if it cannot be computed honestly.

    `hist` is a daily-bar frame indexed by date. It is truncated here at the
    entry date, so passing a frame that runs past the entry is safe and cannot
    leak the future into the estimate.
    """
    strategy = str(trade.get("strategy_name") or "")
    if strategy not in SINGLE_LEG:
        return None

    entry_date = _as_date(trade.get("date"))
    expiry = _as_date(trade.get("expiration"))
    if entry_date is None or expiry is None:
        return None
    dte = (expiry - entry_date).days
    if dte <= 0:
        return None

    premium = safe_float(trade.get("entry_price"))
    strike = safe_float(trade.get("strike"))
    vega = safe_float(trade.get("entry_vega"))
    if not premium or not strike or vega is None:
        return None

    try:
        past = hist[hist.index <= pd.Timestamp(entry_date)]
    except TypeError:
        return None
    if len(past) < MIN_BARS:
        return None

    vol = _hv_blend(past)
    if not vol or vol <= 0:
        return None

    try:
        spot = float(past["Close"].iloc[-1])
    except (KeyError, IndexError, ValueError):
        return None
    if spot <= 0:
        return None

    opt_type = str(trade.get("type") or "call").lower()
    opt_type = "put" if opt_type.startswith("p") else "call"
    fair = safe_float(bs_price(opt_type, spot, strike, dte / 365.0, rfr, vol))
    if fair is None:
        return None

    # The SELLER's edge on a seller's position. Pricing every row as the buyer
    # is a defect already on this repo's record.
    edge_per_share = (premium - fair) if is_short_position(strategy) else (fair - premium)
    ev_gross = edge_per_share * 100.0

    # `vega_dollar` is per ONE IV point: abs(vega) * 100, as options_screener
    # builds it. The band is that times the forecast's own relative error
    # expressed in points.
    vega_dollar = abs(vega) * 100.0
    ev_noise = vega_dollar * VOL_FORECAST_RELATIVE_ERROR * vol * 100.0
    if ev_noise <= 0:
        return None

    return Recon(
        entry_id=int(trade.get("entry_id") or 0),
        ticker=str(trade.get("ticker") or ""),
        strategy_name=strategy,
        as_of=entry_date.isoformat(),
        spot=spot,
        vol_basis=vol,
        n_bars=len(past),
        ev_gross=ev_gross,
        ev_noise=ev_noise,
        sigma_gross=ev_gross / ev_noise,
        pnl_pct=safe_float(trade.get("pnl_pct")),
    )


SCHEMA = """
CREATE TABLE IF NOT EXISTS ev_reconstruction (
    entry_id      INTEGER PRIMARY KEY,
    ticker        TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    as_of         TEXT NOT NULL,
    spot          REAL NOT NULL,
    vol_basis     REAL NOT NULL,
    n_bars        INTEGER NOT NULL,
    ev_gross      REAL NOT NULL,
    ev_noise      REAL NOT NULL,
    sigma_gross   REAL NOT NULL,
    pnl_pct       REAL,
    built_at      TEXT NOT NULL
)
"""


def write(recons, db_path: str = DB_PATH) -> int:
    """Persist reconstructions to their own database. Returns rows written."""
    con = sqlite3.connect(db_path)
    try:
        con.execute(SCHEMA)
        now = dt.datetime.now().isoformat(timespec="seconds")
        rows = [(r.entry_id, r.ticker, r.strategy_name, r.as_of, r.spot,
                 r.vol_basis, r.n_bars, r.ev_gross, r.ev_noise, r.sigma_gross,
                 r.pnl_pct, now) for r in recons]
        con.executemany(
            "INSERT OR REPLACE INTO ev_reconstruction VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            rows)
        con.commit()
        return len(rows)
    finally:
        con.close()


LEDGER_PATH = "paper_trades.db"
# 400 calendar days of runway before the earliest entry, so the first trade
# still has the 120+ bars `long_window_volatility` needs.
HISTORY_RUNWAY_DAYS = 400


def load_closed_single_leg(ledger_path: str = LEDGER_PATH):
    """Closed trades that carry a realised outcome and the inputs to a band.

    Read-only URI: a scan already learned the hard way that opening the ledger
    writable to look at it rewrites the file.
    """
    con = sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" * len(SINGLE_LEG))
        return [dict(r) for r in con.execute(
            f"""SELECT entry_id, ticker, strategy_name, type, strike, date,
                       expiration, entry_price, entry_vega, pnl_pct
                  FROM trades
                 WHERE status = 'CLOSED'
                   AND pnl_pct IS NOT NULL
                   AND strategy_name IN ({placeholders})
                   AND entry_vega IS NOT NULL
                   AND entry_price IS NOT NULL
                   AND strike IS NOT NULL""",
            tuple(sorted(SINGLE_LEG)))]
    finally:
        con.close()


def fetch_history(tickers, start: str, end: str):
    """Daily bars per ticker. Missing tickers are absent, never empty frames."""
    import yfinance as yf
    out = {}
    uniq = sorted({str(t).upper() for t in tickers if t})
    for i in range(0, len(uniq), 25):
        batch = uniq[i:i + 25]
        try:
            data = yf.download(batch, start=start, end=end, progress=False,
                               auto_adjust=False, group_by="ticker",
                               threads=True)
        except Exception:
            continue
        for sym in batch:
            try:
                frame = data[sym] if len(batch) > 1 else data
                frame = frame.dropna(subset=["Close"])
                if len(frame) >= MIN_BARS:
                    out[sym] = frame
            except Exception:
                continue
    return out


def build(ledger_path: str = LEDGER_PATH, db_path: str = DB_PATH):
    """Reconstruct every eligible closed trade. Returns (recons, skipped)."""
    trades = load_closed_single_leg(ledger_path)
    if not trades:
        return [], {"no eligible trades": 0}
    dates = sorted(str(t["date"])[:10] for t in trades)
    start = (dt.date.fromisoformat(dates[0])
             - dt.timedelta(days=HISTORY_RUNWAY_DAYS)).isoformat()
    end = (dt.date.fromisoformat(dates[-1]) + dt.timedelta(days=2)).isoformat()
    hist = fetch_history({t["ticker"] for t in trades}, start, end)

    recons, skipped = [], {}
    for t in trades:
        frame = hist.get(str(t["ticker"]).upper())
        if frame is None:
            skipped["no price history"] = skipped.get("no price history", 0) + 1
            continue
        r = reconstruct_one(t, frame)
        if r is None:
            skipped["not reconstructable"] = skipped.get("not reconstructable", 0) + 1
            continue
        recons.append(r)
    if recons:
        write(recons, db_path)
    return recons, skipped


def report(db_path: str = DB_PATH) -> str:
    """Does the reconstructed band discriminate? Spearman, overall and within
    strategy — the family effect dominates this book, so the pooled number
    alone would restate it rather than test the band."""
    import statistics as st

    from scipy import stats
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in con.execute(
            "SELECT * FROM ev_reconstruction WHERE pnl_pct IS NOT NULL")]
    finally:
        con.close()
    if len(rows) < 20:
        return f"  n={len(rows)} — too few to report"

    out = [f"  RECONSTRUCTED EV BAND vs REALISED OUTCOME   n={len(rows)}", ""]
    xs = [r["sigma_gross"] for r in rows]
    ys = [r["pnl_pct"] for r in rows]
    rho, p = stats.spearmanr(xs, ys)
    out.append(f"  sigma_gross vs pnl_pct        rho {rho:+.3f}   p {p:.4f}")

    by: dict = {}
    for r in rows:
        by.setdefault(r["strategy_name"], []).append(r)
    dx, dy = [], []
    for _, v in by.items():
        if len(v) < 20:
            continue
        mx = st.mean(r["sigma_gross"] for r in v)
        my = st.mean(r["pnl_pct"] for r in v)
        dx += [r["sigma_gross"] - mx for r in v]
        dy += [r["pnl_pct"] - my for r in v]
    if len(dx) >= 20:
        r2, p2 = stats.spearmanr(dx, dy)
        out.append(f"  demeaned within strategy      rho {r2:+.3f}   p {p2:.4f}   n {len(dx)}")

    out += ["", "  by sigma quintile (gross edge in error bars):",
            f"    {'quintile':<10}{'n':>5}{'mean sigma':>12}{'win%':>8}{'mean pnl':>10}"]
    ranked = sorted(rows, key=lambda r: r["sigma_gross"])
    q = max(1, len(ranked) // 5)
    for i in range(5):
        chunk = ranked[i * q:(i + 1) * q] if i < 4 else ranked[4 * q:]
        if not chunk:
            continue
        w = sum(1 for r in chunk if r["pnl_pct"] > 0) / len(chunk) * 100
        out.append(f"    Q{i+1:<9}{len(chunk):>5}"
                   f"{st.mean(r['sigma_gross'] for r in chunk):>12.2f}"
                   f"{w:>7.1f}%{st.mean(r['pnl_pct'] for r in chunk)*100:>+9.1f}%")
    out += ["",
            "  sigma_gross excludes crossing cost, which was never persisted and",
            "  is the term that inverts rankings. Read it as an upper bound."]
    return "\n".join(out)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true", help="fetch history and reconstruct")
    ap.add_argument("--report", action="store_true", help="measure the reconstructed band")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--ledger", default=LEDGER_PATH)
    a = ap.parse_args(argv)
    if a.build:
        recons, skipped = build(a.ledger, a.db)
        print(f"  reconstructed {len(recons)} trades -> {a.db}")
        for k, v in sorted(skipped.items()):
            print(f"    skipped {v}: {k}")
    if a.report or not a.build:
        print(report(a.db))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
