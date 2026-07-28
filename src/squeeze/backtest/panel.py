"""Assemble the point-in-time panel: one row per (settlement date, symbol).

Each row carries the grader's inputs as they stood on the settlement date, the
grade returned by the *production* ``assess_squeeze`` (imported, never
reimplemented — a copy would drift and quietly stop testing the real thing), and
the forward outcome.

Two deliberate choices about the outcome:

* The measure is the **path maximum**, not the endpoint. A squeeze that spikes
  40% and gives it all back is a winning trade for a call holder and an invisible
  one to an endpoint return.
* It is scaled by each name's own trailing volatility. A 20% move in a 100%-vol
  microcap is unremarkable; the same move in a 30%-vol name is not. Without that
  normalisation the test would mostly rediscover that heavily-shorted stocks are
  volatile.

``iv_skew`` is passed as None throughout: no options history exists for these
names, so that one point of the grader's score cannot be reconstructed and is
reported as a stated limitation rather than silently faked.
"""
from __future__ import annotations

import bisect
import datetime as _dt
import math
import sqlite3
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.squeeze.backtest import DEFAULT_DB, PRICES_DB, SHARES_DB
from src.squeeze.detector import assess_squeeze

TREND_TOL = 0.05          # matches src/short_interest.py
VOL_WINDOW = 60           # trading days for trailing realised vol
RVOL_WINDOW = 30          # matches data_fetching.calculate_rvol
HORIZONS = (10, 21, 42)   # trading days forward
MIN_PRICE = 1.0           # sub-$1 names: quotes are noise, options rarely exist

# FINRA disseminates a settlement date's short interest roughly eight business
# days later. Entering on the settlement date itself would be trading on data
# nobody could see yet — and that gap is precisely when a squeeze can begin, so
# the look-ahead would flatter the result exactly where it matters. Entry is
# therefore delayed by this many trading days, and every price feature is
# measured at the entry bar, not the settlement bar.
PUB_LAG_DAYS = 8
# Eight trading days is ~12 calendar days; anything past this means the name was
# halted or barely trading, so the "entry" would not correspond to publication.
MAX_ENTRY_GAP_DAYS = 30


class PriceBook:
    """In-memory daily closes/volumes keyed by symbol, with as-of lookups."""

    def __init__(self, db_path: str, symbols: Optional[Iterable[str]] = None):
        conn = sqlite3.connect(db_path, timeout=120)
        keep = set(symbols) if symbols is not None else None
        cur = conn.execute("SELECT symbol, date, close, volume FROM px ORDER BY symbol, date")
        self._dates: Dict[str, List[str]] = {}
        closes: Dict[str, List[float]] = {}
        vols: Dict[str, List[float]] = {}
        # Every symbol repeats the same ~2,200 trading-day strings. Interning
        # collapses tens of millions of duplicate str objects down to one each,
        # which is the difference between this fitting in memory and not.
        intern: Dict[str, str] = {}
        for sym, date, close, vol in cur:
            if keep is not None and sym not in keep:
                continue
            date = intern.setdefault(date, date)
            self._dates.setdefault(sym, []).append(date)
            closes.setdefault(sym, []).append(close)
            vols.setdefault(sym, []).append(vol if vol is not None else 0.0)
        conn.close()
        self._close = {s: np.asarray(v, dtype=float) for s, v in closes.items()}
        self._vol = {s: np.asarray(v, dtype=float) for s, v in vols.items()}

    def __contains__(self, sym: str) -> bool:
        return sym in self._dates

    def __len__(self) -> int:
        return len(self._dates)

    def _idx_asof(self, sym: str, date: str) -> int:
        """Index of the last bar on or before *date*; -1 if none."""
        dates = self._dates.get(sym)
        if not dates:
            return -1
        return bisect.bisect_right(dates, date) - 1

    def features(self, sym: str, date: str, lag: int = 0) -> Optional[dict]:
        """Price features at *lag* trading days after the bar as of *date*.

        The lag models FINRA's publication delay: the features a trader could
        actually observe when the short-interest figure became public.
        """
        i0 = self._idx_asof(sym, date)
        if i0 < 0:
            return None
        i = i0 + lag
        close = self._close[sym]
        if i >= close.size:
            return None
        vol = self._vol[sym]
        if i < VOL_WINDOW or i < RVOL_WINDOW:
            return None
        # "Eight bars later" is only eight *trading* days if the name kept
        # trading. For a halted or barely-traded name the eighth subsequent bar
        # can sit months out, which would silently turn the publication lag into
        # an arbitrary delay, so entries that drift too far are dropped.
        if lag:
            entry_date = self._dates[sym][i]
            if entry_date < date:
                return None
            gap = (_dt.date.fromisoformat(entry_date) - _dt.date.fromisoformat(date)).days
            if gap > MAX_ENTRY_GAP_DAYS:
                return None
        spot = float(close[i])
        if not math.isfinite(spot) or spot < MIN_PRICE:
            return None

        ret_5d = float(close[i] / close[i - 5] - 1.0) if i >= 5 and close[i - 5] > 0 else None

        window = vol[i - RVOL_WINDOW + 1: i + 1]
        avg_vol = float(window.mean()) if window.size else 0.0
        rvol = float(vol[i] / avg_vol) if avg_vol > 0 else None

        seg = close[i - VOL_WINDOW + 1: i + 1]
        rets = np.diff(np.log(seg))
        rets = rets[np.isfinite(rets)]
        sigma_d = float(rets.std(ddof=1)) if rets.size > 5 else None
        if not sigma_d or sigma_d <= 0:
            return None

        return {"spot": spot, "ret_5d": ret_5d, "rvol": rvol, "sigma_d": sigma_d, "_i": i}

    def forward(self, sym: str, i: int, horizon: int) -> Optional[Tuple[float, float, float]]:
        """(max up-move, end return, min move) over the next *horizon* bars."""
        close = self._close[sym]
        if i + horizon >= close.size:
            return None
        base = float(close[i])
        if base <= 0:
            return None
        path = close[i + 1: i + horizon + 1]
        if path.size < horizon:
            return None
        return (float(path.max() / base - 1.0),
                float(path[-1] / base - 1.0),
                float(path.min() / base - 1.0))


def _trend(shares_short: Optional[float], shares_prior: Optional[float]) -> Optional[str]:
    if not shares_short or not shares_prior or shares_prior <= 0:
        return None
    ratio = shares_short / shares_prior
    if ratio > 1 + TREND_TOL:
        return "rising"
    if ratio < 1 - TREND_TOL:
        return "falling"
    return "flat"


def build(db_path: str = DEFAULT_DB, prices_db: str = PRICES_DB,
          horizons: Sequence[int] = HORIZONS, keep_ungraded: bool = True,
          shares_db: str = SHARES_DB, pub_lag: int = PUB_LAG_DAYS,
          verbose: bool = True) -> List[dict]:
    """Build the ungraded panel: point-in-time inputs plus forward outcomes.

    Grading is deliberately a separate step (``grade``) so the gate calibration
    and the ``ret_5d`` scale can be varied without repeating this pass.

    ``keep_ungraded`` retains rows that have prices but no EDGAR shares
    outstanding (``si_ratio`` None). They cannot be graded, but their forward
    outcomes are still observable — which is what makes it possible to test
    whether dropping them biases the result, instead of merely hoping it does
    not. See ``coverage_bias``.
    """
    from src.squeeze.backtest.shares import SharesLookup
    from src.squeeze.backtest.universe import study_symbols

    syms = study_symbols(db_path)
    if verbose:
        print(f"  universe: {len(syms):,} symbols", flush=True)
    book = PriceBook(prices_db, syms)
    if verbose:
        print(f"  price book: {len(book):,} symbols", flush=True)
    shares = SharesLookup(shares_db)
    if verbose:
        print(f"  shares-out: {len(shares):,} symbols", flush=True)

    conn = sqlite3.connect(db_path, timeout=120)
    rows = conn.execute(
        "SELECT settlement_date, symbol, shares_short, shares_prior, dtc "
        "FROM si WHERE shares_short >= 250000 ORDER BY settlement_date").fetchall()
    conn.close()

    out: List[dict] = []
    for date, sym, ss, prior, dtc in rows:
        if sym not in book:
            continue
        # Shares outstanding is taken as of the settlement date, pairing it with
        # the shares-short figure from the same date. Slightly stale by entry,
        # never look-ahead.
        so = shares.get(sym, date)
        si_ratio = (ss / so) if (so and ss) else None
        # >150% of shares outstanding is a data error, not a famous squeeze.
        if si_ratio is not None and not (0 < si_ratio < 1.5):
            continue
        if si_ratio is None and not keep_ungraded:
            continue
        feats = book.features(sym, date, lag=pub_lag)
        if feats is None:
            continue
        fwd = {}
        for h in horizons:
            f = book.forward(sym, feats["_i"], h)
            if f is None:
                continue
            fwd[h] = f
        if not fwd:
            continue

        rec = {
            "date": date, "symbol": sym, "pub_lag": pub_lag,
            "si_ratio": si_ratio, "dtc": dtc, "trend": _trend(ss, prior),
            "spot": feats["spot"], "sigma_d": feats["sigma_d"],
            "ret_5d": feats["ret_5d"], "rvol": feats["rvol"],
        }
        for h, (mx, end, mn) in fwd.items():
            sigma_h = feats["sigma_d"] * math.sqrt(h)
            rec[f"max_{h}"] = mx
            rec[f"end_{h}"] = end
            rec[f"min_{h}"] = mn
            rec[f"z_{h}"] = mx / sigma_h
            rec[f"zdn_{h}"] = mn / sigma_h
        out.append(rec)

    if verbose:
        print(f"  panel rows: {len(out):,}", flush=True)
    return out


def grade(panel: List[dict], si_scale: float = 1.0,
          ret5d_scale: str = "as_written") -> List[dict]:
    """Apply the production grader to a built panel, in place, returning it.

    ``si_scale`` converts shares-outstanding into an assumed float: EDGAR reports
    shares outstanding, which exceeds float by whatever insiders and restricted
    holders own, so the raw ratio understates short interest as a share of float.
    1.0 treats float as all shares outstanding (most conservative — fewest
    SETUPs); 1.25 assumes float is 80% of shares out; 1.5 assumes 67%. Sweeping
    it shows whether any conclusion depends on where that line is drawn.

    ``ret5d_scale`` is ``as_written`` to reproduce live behaviour (the pipeline
    stores ret_5d as a fraction while the rule compares it to -10.0, so the rule
    cannot fire), or ``as_intended`` to pass percent and let it fire.
    """
    for rec in panel:
        if rec.get("si_ratio") is None:
            rec["grade"] = None            # no shares-out: ungradeable, not NONE
            rec["points"] = None
            continue
        r5 = rec.get("ret_5d")
        r5_arg = None if r5 is None else (r5 * 100.0 if ret5d_scale == "as_intended" else r5)
        setup = assess_squeeze({
            "short_interest": rec["si_ratio"] * si_scale,
            "short_interest_dtc": rec.get("dtc"),
            "short_interest_trend": rec.get("trend"),
            "iv_skew": None,                  # no options history for these names
            "ret_5d": r5_arg,
            "rvol": rec.get("rvol"),
            "gex_flip_price": None,
            "spot": rec.get("spot"),
        })
        rec["grade"] = setup.grade
        rec["points"] = setup.points
    return panel


def to_arrays(panel: List[dict]) -> dict:
    """Column-oriented view for the statistics layer."""
    if not panel:
        return {}
    keys = set()
    for r in panel:
        keys.update(r.keys())
    cols = {}
    for k in keys:
        vals = [r.get(k) for r in panel]
        if k in ("date", "symbol", "grade", "trend"):
            cols[k] = np.array([v if v is not None else "" for v in vals], dtype=object)
        else:
            cols[k] = np.array([np.nan if v is None else float(v) for v in vals], dtype=float)
    return cols
